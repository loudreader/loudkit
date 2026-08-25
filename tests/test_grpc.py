"""The gRPC transport, and the one claim that matters about it.

A transport is only correct if it returns what the others return. `render_bytes`
is the single place this library makes audio, and HTTP, MCP and gRPC all call
it — so the test worth writing is not "does gRPC respond" but "does gRPC respond
with the same bytes". A second synthesis path would pass a smoke test and fail
this one.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import threading
import time

import pytest

from loudkit.synthesis import VoiceLibrary, render_bytes

from .test_server import _engine, _voice

grpc = pytest.importorskip("grpc", reason="needs the [grpc] extra")

REPO = pathlib.Path(__file__).resolve().parent.parent

TIMED = os.environ.get("LOUDKIT_TIMED", "").lower() in {"1", "true", "yes"}
"""Whether wall-clock bounds are assertions or observations.

Same switch shape as ``LOUDKIT_REQUIRE_ASSETS`` in ``tests/assets.py``, for the
same reason: a machine that cannot support a claim must not pretend to. The
claim here is a latency — ``ListVoices`` answers *immediately* while synthesis
is saturated — and on the twelve-way hosted matrix a runner sharing a core with
three other jobs can miss a two-second bound while the transport is behaving
exactly as designed. That failure says something about the runner and nothing
about this library, and a suite that goes red for reasons unrelated to the code
is a suite people rerun until it is green.

The mechanism is asserted unconditionally either way: that callers past the
depth bound are *refused* rather than parked on a thread is what makes the
latency true, and it is checkable without a clock. Set ``LOUDKIT_TIMED=1`` on a
machine that is not contended — a developer box, the self-hosted runner — to
pin the number as well."""


@pytest.fixture
def library(tmp_path) -> VoiceLibrary:  # type: ignore[no-untyped-def]
    _voice().save(tmp_path / "fake.safetensors")
    return VoiceLibrary(tmp_path)


def _work(seconds: float, should_cancel) -> bool:  # type: ignore[no-untyped-def]
    """Spend ``seconds`` the way a real render spends them, and say if it finished.

    A render is not one uninterruptible call: `Engine.stream` polls
    ``should_cancel`` on **every decode step**, so a four-second chunk is a few
    hundred chances to stop. A test that stood in for it with `time.sleep` would
    be measuring a render nothing can stop, which is not the render this
    transport serves — and it would let a transport that cancels correctly look
    identical to one that does not.

    Returns:
        True if the work ran to its end, False if the flag stopped it.
    """
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        if should_cancel is not None and should_cancel():
            return False
        time.sleep(0.01)
    return True


def _error_code(exc) -> str | None:  # type: ignore[no-untyped-def]
    """The `loudkit-error-code` a failed call carried, or None."""
    for key, value in exc.trailing_metadata() or ():
        if key == "loudkit-error-code":
            return str(value)
    return None


def _channel(engine, library):  # type: ignore[no-untyped-def]
    """A started in-process server and a channel to it."""
    from loudkit.transports.grpc import build_server

    server = build_server(engine, library)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, grpc.insecure_channel(f"127.0.0.1:{port}")


def test_grpc_returns_the_same_bytes_as_the_library(library) -> None:  # type: ignore[no-untyped-def]
    """The claim the transport exists to keep.

    Same text, same voice, same seed: gRPC must hand back exactly what
    `render_bytes` produces, byte for byte. Anything else means the transport
    grew a synthesis path of its own, which is the failure this library is
    organised against — and it would be invisible to a test that only checked
    the reply parsed.
    """
    from loudkit.proto import loudkit_pb2, loudkit_pb2_grpc

    engine = _engine()
    direct = render_bytes(engine, "Hello there.", library.load("fake"), seed=7)

    server, channel = _channel(engine, library)
    try:
        stub = loudkit_pb2_grpc.SpeechStub(channel)
        reply = stub.Synthesize(
            loudkit_pb2.SynthesizeRequest(text="Hello there.", voice="fake", seed=7)
        )
    finally:
        channel.close()
        server.stop(None)

    assert reply.audio == direct.data
    assert reply.media_type == direct.media_type
    assert reply.token_count == direct.n_tokens
    assert reply.fingerprint == engine.algorithm.fingerprint()


def test_the_stream_concatenates_to_the_unary_answer(library) -> None:  # type: ignore[no-untyped-def]
    """Both RPCs read the same passage, so both must produce the same tokens.

    The chunk audio does not concatenate to the unary bytes — each chunk is its
    own encoded container with its own header — so the comparison that means
    something is the token count and the continuation tail.
    """
    from loudkit.proto import loudkit_pb2, loudkit_pb2_grpc

    engine = _engine()
    server, channel = _channel(engine, library)
    try:
        stub = loudkit_pb2_grpc.SpeechStub(channel)
        request = loudkit_pb2.SynthesizeRequest(text="Hello there.", voice="fake", seed=7)
        unary = stub.Synthesize(request)
        chunks = list(stub.SynthesizeStream(request))
    finally:
        channel.close()
        server.stop(None)

    assert chunks, "the stream produced nothing"
    assert sum(c.token_count for c in chunks) == unary.token_count
    assert list(chunks[-1].continuation) == list(unary.continuation)


def test_a_bad_request_is_a_status_not_a_traceback(library) -> None:  # type: ignore[no-untyped-def]
    """A caller asking wrongly gets a code it can act on.

    `INVALID_ARGUMENT` for an empty text and `NOT_FOUND` for a voice that is not
    there — the same split `/v1/synthesize` makes with 400 and 404. A transport
    that answered `UNKNOWN` to both would leave a client unable to tell a
    fixable request from a server defect.
    """
    from loudkit.proto import loudkit_pb2, loudkit_pb2_grpc

    server, channel = _channel(_engine(), library)
    try:
        stub = loudkit_pb2_grpc.SpeechStub(channel)
        with pytest.raises(grpc.RpcError) as empty:
            stub.Synthesize(loudkit_pb2.SynthesizeRequest(text="   ", voice="fake"))
        with pytest.raises(grpc.RpcError) as missing:
            stub.Synthesize(loudkit_pb2.SynthesizeRequest(text="hi", voice="nope"))
        with pytest.raises(grpc.RpcError) as path:
            stub.Synthesize(loudkit_pb2.SynthesizeRequest(text="hi", voice="../secret"))
    finally:
        channel.close()
        server.stop(None)

    assert empty.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert missing.value.code() == grpc.StatusCode.NOT_FOUND
    assert path.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_a_public_bind_is_refused_because_there_is_no_auth() -> None:
    """`serve` requires a bearer token for a public bind; this has none.

    So it refuses the bind rather than opening an unauthenticated port. A
    half-guarded public surface is worse than an absent one: it looks like a
    decision was made.
    """
    from loudkit.transports.grpc import serve

    with pytest.raises(ValueError, match="no authentication"):
        serve("unused.safetensors", host="0.0.0.0")


def test_regenerating_the_stubs_is_reproducible(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The committed stubs are what `proto/loudkit.proto` produces.

    Generated code that is committed without this check drifts from its source
    the first time someone edits the schema and forgets to regenerate — and a
    stub that disagrees with the `.proto` is a wire format nobody declared. Same
    rule the respelling lexicon follows.
    """
    pytest.importorskip("grpc_tools", reason="needs the [grpc] extra")

    committed = REPO / "python" / "loudkit" / "proto"
    # Read before regenerating: regeneration is in place, so once it has run
    # there is nothing left to compare the committed bytes against. Without
    # this snapshot the test proves the generator is deterministic and says
    # nothing about whether what is in the tree came from the current .proto.
    committed_bytes = {p.name: p.read_bytes() for p in sorted(committed.glob("loudkit_pb2*"))}
    result = subprocess.run(
        [sys.executable, str(REPO / "tools" / "gen_proto.py")],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO,
    )
    assert result.returncode == 0, result.stderr
    first = {p.name: p.read_bytes() for p in sorted(committed.glob("loudkit_pb2*"))}
    assert first == committed_bytes, (
        "the committed stubs do not match proto/loudkit.proto; "
        "run tools/gen_proto.py and commit the result"
    )
    subprocess.run(
        [sys.executable, str(REPO / "tools" / "gen_proto.py")],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO,
    )
    second = {p.name: p.read_bytes() for p in sorted(committed.glob("loudkit_pb2*"))}
    assert first == second, "gen_proto.py is not deterministic"
    assert "loudkit_pb2.py" in first


class TestTheEngineIsNeverHeldForever:
    """One lock serialises the whole transport, so the wait for it is bounded.

    The failure these pin is not "a synthesis is slow" — it is that a slow
    synthesis used to take the *server* down with it. Every caller waiting for
    the lock holds a thread, the pool is eight threads, and past that gRPC has
    nowhere to run anything at all, including the methods that never touch an
    engine.
    """

    @staticmethod
    def _slow(monkeypatch, seconds: float):  # type: ignore[no-untyped-def]
        """Make one synthesis take `seconds`, so the pool can be observed full."""
        import time as _time

        import loudkit.transports.grpc as mod

        real = mod.render_bytes

        def slow_bytes(*a, **k):  # type: ignore[no-untyped-def]
            _time.sleep(seconds)
            return real(*a, **k)

        monkeypatch.setattr(mod, "render_bytes", slow_bytes)

    def test_listing_voices_answers_while_the_engine_is_besieged(
        self, library, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        """The measurement that made this a transport bug rather than a slow call.

        Ten callers, an eight-thread pool, one 8 s synthesis. With an unbounded
        queue this exact setup answered `ListVoices` with DEADLINE_EXCEEDED
        after four seconds — a method that reads a list of filenames, defeated
        by a queue it has no part in. With the depth bound in place the losers
        are turned away, their threads come back, and the answer is immediate.

        Note which bound this pins. Raising `_MAX_WAIT_S` does not make it pass;
        only refusing past a depth *below the pool size* does, because here a
        waiting caller costs a thread. A 120 s wait is a wedged server too.
        """
        import threading
        import time

        from loudkit.proto import loudkit_pb2, loudkit_pb2_grpc

        self._slow(monkeypatch, seconds=8.0)
        server, channel = _channel(_engine(), library)
        try:
            stub = loudkit_pb2_grpc.SpeechStub(channel)

            refusals: list[str] = []
            tally = threading.Lock()
            # Five of the ten, and the arithmetic is the transport's: one caller
            # holds the engine and `max_queued = max_workers // 2 = 4` may wait
            # behind it, so at least five are turned away. Waiting on that count
            # rather than on a `sleep` is the point — it is the state the test
            # describes ("the engine is besieged"), reached when it is reached
            # rather than when a number someone guessed has elapsed.
            besieged = threading.Event()

            def hog() -> None:
                try:
                    stub.Synthesize(
                        loudkit_pb2.SynthesizeRequest(text="hi", voice="fake"), timeout=60
                    )
                except grpc.RpcError as exc:
                    # The expected outcome for most of them, and the mechanism
                    # under test: refused rather than parked on a thread.
                    with tally:
                        refusals.append(exc.details() or "")
                        if len(refusals) >= 5:
                            besieged.set()

            # More callers than the pool has threads: eight is `max_workers`.
            hogs = [threading.Thread(target=hog, daemon=True) for _ in range(10)]
            for t in hogs:
                t.start()
            assert besieged.wait(timeout=30.0), (
                f"only {len(refusals)} callers were refused; the depth bound is not engaged"
            )

            began = time.monotonic()
            names = stub.ListVoices(loudkit_pb2.ListVoicesRequest(), timeout=4)
            waited = time.monotonic() - began
            assert list(names.voices) == ["fake"]
            assert any("already waiting" in r for r in refusals)
            # The latency claim, and the only line here a loaded runner can move.
            if TIMED:
                assert waited < 2.0, f"ListVoices took {waited:.2f}s while synthesis ran"
            for t in hogs:
                t.join(timeout=60)
        finally:
            channel.close()
            server.stop(0)

    def test_waiting_for_the_engine_is_refused_rather_than_queued(
        self, library, monkeypatch
    ) -> None:
        """`RESOURCE_EXHAUSTED`, not a worker held until the client gives up.

        The status is the contract, and it is the reason the test above passes:
        a caller told to come back releases its thread, and a caller queued
        behind a wedge does not.
        """
        import loudkit.transports.grpc as mod
        from loudkit.proto import loudkit_pb2, loudkit_pb2_grpc

        self._slow(monkeypatch, seconds=5.0)
        monkeypatch.setattr(mod, "_MAX_WAIT_S", 0.05)
        server, channel = _channel(_engine(), library)
        try:
            stub = loudkit_pb2_grpc.SpeechStub(channel)
            import threading

            busy = threading.Thread(
                target=lambda: stub.Synthesize(
                    loudkit_pb2.SynthesizeRequest(text="hi", voice="fake"), timeout=60
                ),
                daemon=True,
            )
            busy.start()
            import time

            time.sleep(1.0)  # long enough for `busy` to be holding the engine

            with pytest.raises(grpc.RpcError) as caught:
                stub.Synthesize(
                    loudkit_pb2.SynthesizeRequest(text="hi", voice="fake"), timeout=10
                )
            assert caught.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
            assert "never got it" in caught.value.details()
            busy.join(timeout=60)
        finally:
            channel.close()
            server.stop(0)

    def test_a_cancelled_stream_stops_the_decode_it_is_inside(
        self, library, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        """A client that leaves stops costing the server a forward pass at a time.

        Two claims, and the second is the one that used to be false. grpc closes
        the response generator of an abandoned RPC, so no *further* chunk is
        started — that has always held. But the chunk already in flight ran to
        its natural end, because nothing handed the engine a cancel flag:
        `Engine.stream` polls `should_cancel` on every decode step and this
        transport passed none, so a cancel arriving one token into a ten-second
        chunk bought nothing until that chunk was over.

        Here the render in flight is four seconds long and polls the flag the
        way the engine does. The cancel has to land inside it.
        """
        import loudkit.transports.grpc as mod
        from loudkit.proto import loudkit_pb2, loudkit_pb2_grpc

        produced: list[int] = []
        stopped = threading.Event()
        real = mod.render_stream_chunks

        def counting(*a, should_cancel=None, **k):  # type: ignore[no-untyped-def]
            for chunk in real(*a, should_cancel=should_cancel, **k):
                produced.append(1)
                if not _work(4.0, should_cancel):
                    stopped.set()
                    return
                yield chunk

        monkeypatch.setattr(mod, "render_stream_chunks", counting)
        server, channel = _channel(_engine(), library)
        try:
            stub = loudkit_pb2_grpc.SpeechStub(channel)
            call = stub.SynthesizeStream(
                loudkit_pb2.SynthesizeRequest(text="one two three four five", voice="fake")
            )
            next(iter(call))
            at_cancel = len(produced)
            began = time.monotonic()
            call.cancel()
            # Inside the render, not after it: the chunk in flight has ~4 s left
            # to run and the flag is polled every step, so a second is a long
            # time to allow and still far short of letting it finish.
            assert stopped.wait(timeout=1.0), (
                "the render in flight ignored the cancel; the engine was never "
                "given a cancel flag"
            )
            assert time.monotonic() - began < 4.0
            assert len(produced) - at_cancel <= 2, (
                f"{len(produced) - at_cancel} chunks rendered after the cancel; "
                "the server is working for a client that left"
            )
        finally:
            channel.close()
            server.stop(0)

    def test_an_engine_that_cannot_be_reclaimed_is_not_handed_to_the_next_caller(
        self, library, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        """Admission needs the previous holder to be provably gone, not merely returned.

        Closing an abandoned stream runs `Engine.stream`'s teardown, which
        cancels the renders nobody will read and joins the producer thread. When
        that teardown *fails*, the stages cannot be shown to be free — and the
        engine is not reentrant, so admitting the next caller is a wrong answer
        rather than a slow one. The transport used to leave the close to frame
        teardown, where a raising teardown is an ignored exception and the lock
        goes to the next caller regardless: nothing looked wrong until two
        callers were inside the stages.

        Now the reclaim is explicit and its failure is a verdict: the engine is
        marked unusable, and every later call says so. The HTTP route reaches
        the same verdict by the same route.
        """
        import loudkit.transports.grpc as mod
        from loudkit.proto import loudkit_pb2, loudkit_pb2_grpc

        real = mod.render_stream_chunks

        def unreclaimable(*a, should_cancel=None, **k):  # type: ignore[no-untyped-def]
            try:
                for chunk in real(*a, should_cancel=should_cancel, **k):
                    _work(4.0, should_cancel)
                    yield chunk
            finally:
                # What a producer thread that outlives its join looks like from
                # here: the teardown cannot say the stages are idle.
                raise RuntimeError("the producer thread did not stop")

        monkeypatch.setattr(mod, "render_stream_chunks", unreclaimable)
        server, channel = _channel(_engine(), library)
        try:
            stub = loudkit_pb2_grpc.SpeechStub(channel)
            call = stub.SynthesizeStream(
                loudkit_pb2.SynthesizeRequest(text="one two three four five", voice="fake")
            )
            next(iter(call))
            call.cancel()

            with pytest.raises(grpc.RpcError) as caught:
                stub.Synthesize(
                    loudkit_pb2.SynthesizeRequest(text="hi", voice="fake"), timeout=30
                )
            assert caught.value.code() == grpc.StatusCode.INTERNAL, (
                "the next caller was admitted to an engine nothing could show was idle"
            )
            assert _error_code(caught.value) == "server_fault"
        finally:
            channel.close()
            server.stop(0)


def test_describe_reports_a_held_engine_without_waiting_for_it(library) -> None:  # type: ignore[no-untyped-def]
    """The wedge is not closed; it is no longer invisible.

    A client that stops reading a stream blocks the server inside `yield` and
    holds the engine indefinitely — the elapsed-time check between chunks
    cannot fire while the yield itself is blocked, and closing that needs a
    bounded hand-off between the thread holding the engine and the thread
    writing the socket. That case stays open and is documented in the module.

    What was also true, and did not have to be, is that a gRPC deployment could
    not tell a held engine from a slow one: the HTTP server reports exactly
    this on `/health` as "stuck" and gRPC reported nothing. `Describe` now
    answers with how long the engine has been held, and — the part that makes
    it useful — answers *while* it is held, because it does not take the lock.
    """
    from loudkit.proto import loudkit_pb2, loudkit_pb2_grpc

    server, channel = _channel(_engine(), library)
    try:
        stub = loudkit_pb2_grpc.SpeechStub(channel)
        assert stub.Describe(loudkit_pb2.DescribeRequest()).engine_held_seconds == 0.0

        # Hold the engine from another thread and ask again. The assertion that
        # matters is that the call *returns* — a report that waits for the
        # engine reports nothing at the only moment anyone wants it.
        holder = threading.Thread(
            target=lambda: stub.Synthesize(
                loudkit_pb2.SynthesizeRequest(text="Hold the engine.", voice="fake", seed=1)
            ),
            daemon=True,
        )
        holder.start()
        try:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                held = stub.Describe(loudkit_pb2.DescribeRequest()).engine_held_seconds
                if held > 0.0:
                    break
            else:
                pytest.skip("the render finished before Describe could observe the hold")
            assert held > 0.0
        finally:
            holder.join(timeout=30)

        # ...and it goes back to zero once the engine is free.
        assert stub.Describe(loudkit_pb2.DescribeRequest()).engine_held_seconds == 0.0
    finally:
        channel.close()
        server.stop(None)


def test_the_stream_cap_ends_the_stream_at_the_cap_and_frees_the_engine(
    library, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    """What `_MAX_STREAM_S` bounds, measured rather than assumed.

    The cap used to be read between chunks, which bounded a stream that kept
    producing and bounded nothing that stopped: with the cap at 0.5 s and a
    render blocking 2 s before its second chunk, the stream ended at 2.02 s —
    the client had its DEADLINE_EXCEEDED long before, and the server rendered on,
    holding the engine. The old form of this test asserted exactly that, and said
    a transport that bounded a blocked render would make the module docstring
    true. This is that transport.

    The cap now sets the flag `Engine.stream` polls every decode step, so it
    fires inside the render rather than after it. Two assertions: the stream ends
    near the cap, and the engine is free when it does — a bound that ends the RPC
    while the server keeps rendering would be the same defect wearing a status
    code.
    """
    import loudkit.transports.grpc as mod
    from loudkit.proto import loudkit_pb2, loudkit_pb2_grpc

    block = 4.0
    cap = 0.5
    monkeypatch.setattr(mod, "_MAX_STREAM_S", cap)
    real = mod.render_stream_chunks
    finished = threading.Event()

    def blocking(*a, should_cancel=None, **k):  # type: ignore[no-untyped-def]
        for i, chunk in enumerate(real(*a, should_cancel=should_cancel, **k)):
            if i == 1 and not _work(block, should_cancel):
                return
            yield chunk
        finished.set()

    monkeypatch.setattr(mod, "render_stream_chunks", blocking)
    server, channel = _channel(_engine(), library)
    try:
        stub = loudkit_pb2_grpc.SpeechStub(channel)
        started = time.monotonic()
        call = stub.SynthesizeStream(
            loudkit_pb2.SynthesizeRequest(text="one two three four five", voice="fake")
        )
        with pytest.raises(grpc.RpcError) as caught:
            for _ in call:
                pass
        elapsed = time.monotonic() - started
        assert caught.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
        assert _error_code(caught.value) == "timeout"
        assert not finished.is_set(), "the blocked render ran to its end"
        # Well inside the block rather than after it. Generous against the cap
        # (0.5 s) and unambiguous against the render it stopped (4 s): a
        # transport that still waits for the render cannot pass this by being
        # slightly slow.
        assert elapsed < block / 2, (
            f"the stream ended after {elapsed:.2f}s with a {cap}s cap and a "
            f"{block:.0f}s render: the cap is still being read between chunks"
        )
        # And the engine is back, which is the half of the bound that matters to
        # everyone who is not this caller.
        assert stub.Describe(loudkit_pb2.DescribeRequest()).engine_held_seconds == 0.0
        assert stub.Synthesize(
            loudkit_pb2.SynthesizeRequest(text="hi", voice="fake"), timeout=10
        ).audio
    finally:
        channel.close()
        server.stop(0)


def test_an_expired_client_deadline_stops_the_render(library, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The caller's own deadline is a cancel, and it reaches the engine.

    This is the reviewer's measurement: a 0.5 s deadline against a render that
    takes 4 s ended the RPC on time and left the server rendering for four
    seconds more, holding the engine the whole way. The status a client sees and
    the work a server does had come apart, which is the expensive kind of wrong —
    the deployment looks bounded and is not.

    grpc terminates the call at the deadline, the servicer's termination
    callback sets the same flag the decode loop polls, and the render stops.
    """
    import loudkit.transports.grpc as mod
    from loudkit.proto import loudkit_pb2, loudkit_pb2_grpc

    real = mod.render_stream_chunks
    stopped = threading.Event()

    def blocking(*a, should_cancel=None, **k):  # type: ignore[no-untyped-def]
        for chunk in real(*a, should_cancel=should_cancel, **k):
            if not _work(4.0, should_cancel):
                stopped.set()
                return
            yield chunk

    monkeypatch.setattr(mod, "render_stream_chunks", blocking)
    server, channel = _channel(_engine(), library)
    try:
        stub = loudkit_pb2_grpc.SpeechStub(channel)
        began = time.monotonic()
        with pytest.raises(grpc.RpcError) as caught:
            for _ in stub.SynthesizeStream(
                loudkit_pb2.SynthesizeRequest(text="one two three four five", voice="fake"),
                timeout=0.5,
            ):
                pass
        assert caught.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
        assert stopped.wait(timeout=1.5), (
            "the server rendered on past the client's deadline; the deadline "
            "never reached the engine"
        )
        assert time.monotonic() - began < 4.0
        # Free for the next caller, not four seconds from now.
        assert stub.Synthesize(
            loudkit_pb2.SynthesizeRequest(text="hi", voice="fake"), timeout=10
        ).audio
    finally:
        channel.close()
        server.stop(0)


def test_a_reply_no_client_could_receive_is_refused_before_the_render(
    library, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    """The caller learns before the wait, not after the work.

    grpc's default receive limit is 4 MiB in every language, and a unary reply
    past it is refused by the *client*. Rendered first: the server spent the
    whole synthesis, held the single-flight engine for it, and handed back
    something the caller then dropped on the floor. The text cap alone does not
    prevent it — 10 000 characters is well over an hour of audio.

    So the size is bounded from the request, out of numbers the engine already
    publishes, and the refusal names the RPC that has no such ceiling.
    """
    import loudkit.transports.grpc as mod
    from loudkit.proto import loudkit_pb2, loudkit_pb2_grpc

    rendered: list[int] = []
    real = mod.render_bytes

    def counting(*a, **k):  # type: ignore[no-untyped-def]
        rendered.append(1)
        return real(*a, **k)

    monkeypatch.setattr(mod, "render_bytes", counting)
    server, channel = _channel(_engine(), library)
    try:
        stub = loudkit_pb2_grpc.SpeechStub(channel)
        with pytest.raises(grpc.RpcError) as caught:
            stub.Synthesize(
                loudkit_pb2.SynthesizeRequest(text="one two three. " * 400, voice="fake")
            )
        # A passage that fits is untouched: the bound refuses, it does not cap.
        assert stub.Synthesize(
            loudkit_pb2.SynthesizeRequest(text="Hello there.", voice="fake")
        ).audio
    finally:
        channel.close()
        server.stop(None)

    assert caught.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert _error_code(caught.value) == "payload_too_large"
    assert "SynthesizeStream" in (caught.value.details() or "")
    assert rendered == [1], (
        f"{len(rendered)} renders ran: the oversized request was refused after "
        "the engine had already produced audio nobody can receive"
    )


def test_a_bad_speed_or_format_is_invalid_argument_with_a_code(library, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Two values the engine refuses, named at the boundary instead of escaping.

    An out-of-range `speed` raised `ValueError` from the engine and an unknown
    `audio_format` raised `KeyError` from the encoder. Neither was caught here,
    so grpc reported UNKNOWN with no `loudkit-error-code` at all — a client was
    told neither what was wrong nor whose fault it was, and was told it after
    waiting for the engine and rendering.

    `INVALID_ARGUMENT` with the frozen catalog's `invalid_request`, refused
    before the engine is taken. The HTTP server refuses both in its request
    model, and the vocabulary is the same across transports on purpose.
    """
    import loudkit.transports.grpc as mod
    from loudkit.proto import loudkit_pb2, loudkit_pb2_grpc

    rendered: list[int] = []

    def never_reached(*_a, **_k):  # type: ignore[no-untyped-def]
        rendered.append(1)

    monkeypatch.setattr(mod, "render_bytes", never_reached)
    server, channel = _channel(_engine(), library)
    try:
        stub = loudkit_pb2_grpc.SpeechStub(channel)
        with pytest.raises(grpc.RpcError) as speed:
            stub.Synthesize(loudkit_pb2.SynthesizeRequest(text="hi", voice="fake", speed=5.0))
        with pytest.raises(grpc.RpcError) as fmt:
            stub.Synthesize(
                loudkit_pb2.SynthesizeRequest(text="hi", voice="fake", audio_format="mp3")
            )
        with pytest.raises(grpc.RpcError) as streamed:
            for _ in stub.SynthesizeStream(
                loudkit_pb2.SynthesizeRequest(text="hi", voice="fake", speed=0.1)
            ):
                pass
    finally:
        channel.close()
        server.stop(None)

    for caught in (speed, fmt, streamed):
        assert caught.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert _error_code(caught.value) == "invalid_request"
    assert "0.5" in (speed.value.details() or "")
    assert "flac" in (fmt.value.details() or ""), "the refusal does not say what would work"
    assert rendered == [], "the engine was entered for a request that was never valid"


def test_an_expired_waiter_never_takes_the_engine(library, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A deadline that passes in the queue means the engine is never taken.

    The reviewer's reproduction: a request with a short deadline expires while
    waiting for the lock, grpc answers it DEADLINE_EXCEEDED on time — and the
    server then acquires the engine anyway and renders a full reply for an RPC
    that no longer exists. The wait must be capped at the caller's own
    `time_remaining()`, and an acquisition re-checked against `is_active()`,
    so the render below never runs.
    """
    import loudkit.transports.grpc as mod
    from loudkit.proto import loudkit_pb2, loudkit_pb2_grpc

    renders: list[int] = []
    real = mod.render_bytes

    def slow(*a, **k):  # type: ignore[no-untyped-def]
        renders.append(1)
        time.sleep(2.0)
        return real(*a, **k)

    monkeypatch.setattr(mod, "render_bytes", slow)
    server, channel = _channel(_engine(), library)
    try:
        stub = loudkit_pb2_grpc.SpeechStub(channel)
        holder = threading.Thread(
            target=lambda: stub.Synthesize(
                loudkit_pb2.SynthesizeRequest(text="hi", voice="fake"), timeout=30
            ),
            daemon=True,
        )
        holder.start()
        time.sleep(0.5)  # long enough for `holder` to be inside the render
        assert renders == [1], "the holder never took the engine; nothing is queued behind it"

        with pytest.raises(grpc.RpcError) as caught:
            stub.Synthesize(loudkit_pb2.SynthesizeRequest(text="hi", voice="fake"), timeout=0.4)
        # At the boundary the two refusals race: the client's own clock fires
        # DEADLINE_EXCEEDED, and a server whose wait ran out a few milliseconds
        # before the RPC read as inactive answers RESOURCE_EXHAUSTED first.
        # Both are truthful "you did not get the engine" answers; the claim
        # this test pins is the render count below, not which refusal wins the
        # race.
        assert caught.value.code() in (
            grpc.StatusCode.DEADLINE_EXCEEDED,
            grpc.StatusCode.RESOURCE_EXHAUSTED,
        )
        holder.join(timeout=30)

        # The waiter's chance to (wrongly) render begins when the holder
        # releases, so the window to watch is after the join: with an uncapped
        # wait it acquires within milliseconds and the second render appears
        # here. Polled rather than slept-and-checked, so the failure is caught
        # the moment it happens.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            assert renders == [1], (
                "a caller whose deadline expired in the queue took the engine "
                "and rendered for an RPC grpc had already closed"
            )
            time.sleep(0.05)
    finally:
        channel.close()
        server.stop(0)


def test_a_cancelled_waiter_without_a_deadline_frees_its_slot(library, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A cancel is the ending no timeout can see, and it must free the queue.

    The deadline-bounded wait caps how long an *expired* waiter can sit on a
    worker thread; a caller with no deadline that hangs up got no such cap and
    held its thread, and its queue slot, for the full ``_MAX_WAIT_S``. The
    wait is sliced so a dead RPC gives both back within a poll interval.

    Observed through the depth bound: with the engine held, cancelled
    no-deadline waiters fill ``max_queued``, and a fresh probe is then either
    depth-refused ("already waiting", the pre-fix answer, because the dead
    waiters still count) or waits on its own merits. The claim pinned here is
    that dead waiters stop counting.
    """
    import loudkit.transports.grpc as mod
    from loudkit.proto import loudkit_pb2, loudkit_pb2_grpc
    from loudkit.transports.grpc import build_server

    release = threading.Event()
    started = threading.Event()
    real = mod.render_bytes

    def held(*a, **k):  # type: ignore[no-untyped-def]
        started.set()
        release.wait(30.0)
        return real(*a, **k)

    monkeypatch.setattr(mod, "render_bytes", held)
    # max_workers=4 makes max_queued 2: small enough to fill with two dead
    # waiters, large enough that the probe still gets a worker thread.
    server = build_server(_engine(), library, max_workers=4)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    try:
        stub = loudkit_pb2_grpc.SpeechStub(channel)
        request = loudkit_pb2.SynthesizeRequest(text="hi", voice="fake")
        holder = threading.Thread(
            target=lambda: stub.Synthesize(request, timeout=30), daemon=True
        )
        holder.start()
        assert started.wait(10.0), "the holder never took the engine"

        # Two waiters with no deadline, cancelled once they are queued. Their
        # RPCs are gone; only their server threads know it.
        waiters = [stub.Synthesize.future(request) for _ in range(2)]
        time.sleep(0.3)
        for w in waiters:
            w.cancel()
        # Post-fix each dead waiter notices within ``_QUEUE_POLL_S``; this is
        # two orders of magnitude of margin, not a tuned number.
        time.sleep(1.0)

        with pytest.raises(grpc.RpcError) as caught:
            stub.Synthesize(request, timeout=0.5)
        # The probe must wait on its own merits and expire, not be turned
        # away by a queue full of RPCs that no longer exist.
        details = caught.value.details() or ""
        assert "already waiting" not in details, (
            f"cancelled no-deadline waiters still held their queue slots: {details}"
        )
    finally:
        release.set()
        holder.join(timeout=30)
        channel.close()
        server.stop(0)


def test_an_expired_unary_deadline_stops_the_render_it_is_inside(library) -> None:  # type: ignore[no-untyped-def]
    """`Synthesize` gets the same cancellation the stream has.

    The stream wires the RPC's termination callback to the engine's cancel
    flag; the unary path passed nothing, so a client whose deadline expired
    got DEADLINE_EXCEEDED on time while the server computed the entire reply —
    token generation, mel, vocoder — for nobody. Here the token phase is four
    seconds long and polls the flag the way the real generator does; the
    expired deadline has to stop it from inside.
    """
    from dataclasses import replace

    from loudkit.config import AlgorithmConfig
    from loudkit.engine import Engine
    from loudkit.proto import loudkit_pb2, loudkit_pb2_grpc

    from .test_server import _FakeGenerator, _FakeMelDecoder, _FakeVocoder, _SplitFrontend

    stopped = threading.Event()

    class _SlowGenerator(_FakeGenerator):
        def generate(  # type: ignore[no-untyped-def]
            self,
            text_tokens,
            voice,
            *,
            sampler,
            max_new_tokens=None,
            prefix=(),
            should_cancel=None,
        ):
            if not _work(4.0, should_cancel):
                stopped.set()
                return [self.config.stop_speech_token]
            return super().generate(
                text_tokens,
                voice,
                sampler=sampler,
                max_new_tokens=max_new_tokens,
                prefix=prefix,
                should_cancel=should_cancel,
            )

    chunking = replace(AlgorithmConfig().chunking, max_tokens=2, prefix_tokens=0)
    algo = AlgorithmConfig().with_(chunking=chunking)
    engine = Engine(
        frontend=_SplitFrontend(),
        token_generator=_SlowGenerator(algo),
        mel_decoder=_FakeMelDecoder(algo),
        vocoder=_FakeVocoder(algo),
        algorithm=algo,
    )
    server, channel = _channel(engine, library)
    try:
        stub = loudkit_pb2_grpc.SpeechStub(channel)
        began = time.monotonic()
        with pytest.raises(grpc.RpcError) as caught:
            stub.Synthesize(loudkit_pb2.SynthesizeRequest(text="hi", voice="fake"), timeout=0.5)
        assert caught.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
        assert stopped.wait(timeout=1.5), (
            "the server computed on past the client's deadline; the unary path "
            "never handed the engine a cancel flag"
        )
        assert time.monotonic() - began < 4.0
        # And the engine came back with the render, not four seconds later.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if stub.Describe(loudkit_pb2.DescribeRequest()).engine_held_seconds == 0.0:
                break
            time.sleep(0.05)
        else:
            pytest.fail("the engine is still held after the cancelled render stopped")
    finally:
        channel.close()
        server.stop(0)


def test_the_final_chunks_continuation_is_the_passages_tail(library) -> None:  # type: ignore[no-untyped-def]
    """A last chunk shorter than the prefix still hands back a full tail.

    A chunk's own tokens are all it can contribute, and a two-token closing
    sentence has two — while `Synthesize` for the same request hands back the
    engine's full `prefix_tokens` worth, reaching into the chunk before. The
    proto promises the two are equal; a client chaining from the stream's last
    chunk otherwise restarts most of its prosodic context exactly when the
    passage ends on a short sentence.
    """
    from dataclasses import replace

    from loudkit.config import AlgorithmConfig
    from loudkit.engine import Engine
    from loudkit.proto import loudkit_pb2, loudkit_pb2_grpc

    from .test_server import _FakeGenerator, _FakeMelDecoder, _FakeVocoder, _SplitFrontend

    # A real prefix, unlike the shared fake's zero: the defect only exists
    # when there is a tail to get wrong.
    chunking = replace(AlgorithmConfig().chunking, max_tokens=5, prefix_tokens=4)
    algo = AlgorithmConfig().with_(chunking=chunking)
    engine = Engine(
        frontend=_SplitFrontend(),
        token_generator=_FakeGenerator(algo),
        mel_decoder=_FakeMelDecoder(algo),
        vocoder=_FakeVocoder(algo),
        algorithm=algo,
    )
    server, channel = _channel(engine, library)
    try:
        stub = loudkit_pb2_grpc.SpeechStub(channel)
        # The closing sentence produces fewer tokens than the prefix length.
        request = loudkit_pb2.SynthesizeRequest(
            text="One two three four five. Go.", voice="fake", seed=7
        )
        unary = stub.Synthesize(request)
        chunks = list(stub.SynthesizeStream(request))
    finally:
        channel.close()
        server.stop(None)

    assert len(chunks) >= 2, "the passage did not split; the short-chunk case is not exercised"
    assert chunks[-1].token_count < 4, "the last chunk is not shorter than the prefix"
    assert len(unary.continuation) == 4
    assert list(chunks[-1].continuation) == list(unary.continuation), (
        "the stream's final continuation is the last chunk's own tail, not the "
        "passage's; a chaining client loses the context the engine conditions on"
    )


def test_the_preflight_measures_the_text_the_engine_will_speak(library, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The 4 MiB preflight runs on the normalised text, not the raw characters.

    The funnel expands: a thousand characters of digits normalise to about five
    thousand characters of number words, which is minutes of audio. Estimated
    from the raw length this request promises ~80 s and slips under the ~87 s
    the client's limit allows, so the server rendered a reply several times
    over the limit and the *client* then refused it — the exact failure the
    preflight exists to prevent, reachable straight through it.
    """
    import loudkit.transports.grpc as mod
    from loudkit.proto import loudkit_pb2, loudkit_pb2_grpc

    rendered: list[int] = []
    real = mod.render_bytes

    def counting(*a, **k):  # type: ignore[no-untyped-def]
        rendered.append(1)
        return real(*a, **k)

    monkeypatch.setattr(mod, "render_bytes", counting)
    server, channel = _channel(_engine(), library)
    try:
        stub = loudkit_pb2_grpc.SpeechStub(channel)
        with pytest.raises(grpc.RpcError) as caught:
            stub.Synthesize(
                loudkit_pb2.SynthesizeRequest(text="9" * 1000, voice="fake"), timeout=60
            )
    finally:
        channel.close()
        server.stop(None)

    assert caught.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert _error_code(caught.value) == "payload_too_large"
    assert "SynthesizeStream" in (caught.value.details() or "")
    assert rendered == [], (
        "the oversized request was rendered: the preflight measured the raw "
        "characters and the funnel's expansion walked straight past it"
    )


def test_a_peer_that_stops_reading_stalls_the_queue_and_not_the_engine(
    library, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    """A stalled read blocks delivery; the engine still comes back at the cap.

    A connected client that stops reading blocks the server's write. When the
    render and the write share a thread, the engine lock is held mid-`yield`
    for as long as the peer cares to stall — a resource contract broken by one
    silent client. Production is separated from the write by a bounded queue:
    the queue fills, the producer parks on a `put` that polls the cancel flag,
    and at `_MAX_STREAM_S` the render stops, the engine is reclaimed and the
    lock is released while the write stays stuck. The worker thread is the
    price; the engine is not.
    """
    import loudkit.transports.grpc as mod
    from loudkit.proto import loudkit_pb2, loudkit_pb2_grpc
    from loudkit.synthesis import Rendered

    cap = 1.0
    monkeypatch.setattr(mod, "_MAX_STREAM_S", cap)
    # Chunks far larger than an HTTP/2 flow-control window, so a peer that
    # stops reading stops the server's send within one chunk; endless, so only
    # the cancel flag can end the render.
    big = b"\x00" * (512 * 1024)

    def gushing(*a, should_cancel=None, **k):  # type: ignore[no-untyped-def]
        while not (should_cancel is not None and should_cancel()):
            yield Rendered(
                data=big,
                duration=1.0,
                n_tokens=1,
                hit_token_cap=False,
                media_type="application/octet-stream",
            )

    monkeypatch.setattr(mod, "render_stream_chunks", gushing)
    server, channel = _channel(_engine(), library)
    try:
        stub = loudkit_pb2_grpc.SpeechStub(channel)
        call = stub.SynthesizeStream(loudkit_pb2.SynthesizeRequest(text="hi", voice="fake"))
        next(iter(call))  # the stream is live and the peer now stops reading
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            if stub.Describe(loudkit_pb2.DescribeRequest()).engine_held_seconds == 0.0:
                break
            time.sleep(0.1)
        else:
            pytest.fail(
                f"the engine is still held {8.0:.0f}s into a {cap:.0f}s cap: a "
                "peer that stopped reading is holding the lock through the write"
            )
        # Free means free: the next caller synthesises now, not when the peer
        # deigns to read.
        assert stub.Synthesize(
            loudkit_pb2.SynthesizeRequest(text="hi", voice="fake"), timeout=10
        ).audio
        call.cancel()
    finally:
        channel.close()
        server.stop(0)


def test_every_reply_names_its_sample_rate(library) -> None:  # type: ignore[no-untyped-def]
    """Raw pcm16 frames carry no header, so the message must say the rate.

    wav, flac and ogg record it in their own headers; `pcm16` is header-less
    by design, and a caller feeding frames to a device had nowhere at all to
    read the rate from — the HTTP transport sends X-Loudkit-Sample-Rate for
    exactly this. On every chunk as well as the unary reply, so a client that
    joins a stream mid-way still knows what it is holding.
    """
    from loudkit.proto import loudkit_pb2, loudkit_pb2_grpc

    engine = _engine()
    server, channel = _channel(engine, library)
    try:
        stub = loudkit_pb2_grpc.SpeechStub(channel)
        request = loudkit_pb2.SynthesizeRequest(
            text="Hello there.", voice="fake", audio_format="pcm16"
        )
        unary = stub.Synthesize(request)
        chunks = list(stub.SynthesizeStream(request))
    finally:
        channel.close()
        server.stop(None)

    assert unary.sample_rate == engine.algorithm.sample_rate
    assert chunks
    assert all(c.sample_rate == engine.algorithm.sample_rate for c in chunks)
