"""The long-form pipeline: generate window k+1 while window k renders.

The chunk chain is sequential only through the token phase, so rendering moves
to a worker and the audio is byte-identical to the serial path. Backpressure,
cancellation and the wedge are in ``docs/design/engine-pipeline.md``.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from .errors import CancelledError, NothingToSpeakError
from .result import Result, StageTimings
from .timing import ChunkSpan, timeline
from .window import carry_pair_aligned, provenance_for

if TYPE_CHECKING:
    from .engine import Engine
    from .voice import VoiceProfile

_LOG = logging.getLogger("loudkit.stream")

_PIPELINE_DEPTH = 2
"""Rendered-or-rendering windows held in flight ahead of the consumer."""

_PRODUCER_JOIN_TIMEOUT = 60.0
"""How long a closing stream waits for the render in flight. Past it the engine is wedged."""


def joined(
    engine: Engine,
    text: str,
    voice: VoiceProfile,
    *,
    seed: int,
    language: str | None,
    speed: float,
    previous_tokens: Sequence[int] | None,
    should_cancel: Callable[[], bool] | None,
) -> Result:
    """:meth:`Engine.stream`, drained and concatenated into one :class:`Result`.

    Through the raising form of the stream: a cancelled passage has no answer,
    and joining the chunks that finished would be silent truncation.
    """
    parts = list(
        engine._stream(
            text,
            voice,
            seed=seed,
            language=language,
            speed=speed,
            previous_tokens=previous_tokens,
            should_cancel=should_cancel,
            latency_mode=False,
        )
    )
    if not parts:
        raise NothingToSpeakError("nothing to speak")
    if len(parts) == 1:
        return replace(parts[0], seed=seed)

    audio = np.concatenate([p.audio for p in parts])
    mel = np.concatenate([p.mel for p in parts], axis=1)
    algorithm = engine.algorithm
    return Result(
        audio=audio,
        tokens=[t for p in parts for t in p.tokens],
        mel=mel,
        seed=seed,
        sample_rate=algorithm.sample_rate,
        timings=StageTimings(
            sum(p.timings.tokens for p in parts),
            sum(p.timings.mel for p in parts),
            sum(p.timings.audio for p in parts),
        ),
        hit_token_cap=any(p.hit_token_cap for p in parts),
        inspections=tuple(i for p in parts for i in p.inspections),
        speed=speed,
        chunks=timeline(
            [
                ChunkSpan(
                    text=p.chunks[0].text if p.chunks else "",
                    samples=len(p.audio),
                    tokens=len(p.tokens),
                )
                for p in parts
            ],
            sample_rate=algorithm.sample_rate,
        ),
        provenance=provenance_for(engine, voice, parts[0].provenance.language),
    )


def pipelined(  # noqa: PLR0915 - one producer, one consumer, one teardown
    engine: Engine,
    chunks: Sequence[str],
    voice: VoiceProfile,
    *,
    seed: int,
    language: str,
    speed: float,
    carry: list[int],
    prefix_len: int,
    should_cancel: Callable[[], bool] | None,
    latency_mode: bool,
) -> Generator[Result, None, None]:
    """Yield one :class:`Result` per window, rendering on a worker while the
    producer generates the next.

    Raises :class:`CancelledError` where the caller's ``should_cancel``
    stopped it; the chunks yielded before are the partial.

    ``latency_mode`` holds generation of window 1 until window 0's render is
    out when both stages share a device, which protects time to first audio
    at the cost of one render's overlap.
    """
    stop = threading.Event()
    protect_first_render = latency_mode and (
        engine.execution.resolved_generator_device()
        == engine.execution.resolved_renderer_device()
    )

    def cancelled() -> bool:
        return stop.is_set() or (should_cancel is not None and should_cancel())

    def refuse_if_cancelled() -> None:
        if should_cancel is not None and should_cancel():
            raise CancelledError("cancelled: should_cancel returned true")

    out: queue.Queue[tuple[str, Future[Result] | BaseException | None]] = queue.Queue(
        maxsize=_PIPELINE_DEPTH
    )
    # Every render handed to the pool and not finished, so teardown can cancel
    # the ones nobody will read. The lock pairs the stop check with the submit.
    submitted: list[Future[Result]] = []
    submitted_lock = threading.Lock()

    def produce() -> None:
        try:
            with ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="loudkit-render"
            ) as renderer:
                prefix = carry
                for index, chunk in enumerate(chunks):
                    if stop.is_set():
                        break
                    refuse_if_cancelled()
                    windows = engine._windows_for_chunk(
                        chunk,
                        voice,
                        index=index,
                        seed=seed,
                        language=language,
                        prefix=prefix,
                        is_terminal=index == len(chunks) - 1,
                        should_cancel=cancelled,
                    )
                    # The carry comes off the last window of a halved chunk.
                    if prefix_len:
                        prefix = carry_pair_aligned(
                            engine.algorithm, windows[-1].speech, prefix_len
                        )
                    swept = False
                    for offset, window in enumerate(windows):
                        with submitted_lock:
                            if stop.is_set():
                                swept = True
                                break
                            future = renderer.submit(
                                engine._render_window,
                                window,
                                voice,
                                speed=speed,
                                should_cancel=cancelled,
                            )
                            submitted[:] = [f for f in submitted if not f.done()]
                            submitted.append(future)
                        # Blocks when the consumer is _PIPELINE_DEPTH behind.
                        out.put(("item", future))
                        if index == 0 and offset == 0 and protect_first_render:
                            wait([future])
                    if swept:
                        break
            out.put(("done", None))
        except BaseException as exc:  # ferried to the consumer
            out.put(("error", exc))

    producer = threading.Thread(target=produce, name="loudkit-generate", daemon=True)
    producer.start()
    try:
        while True:
            kind, payload = out.get()
            if kind == "done":
                return
            if kind == "error":
                assert isinstance(payload, BaseException)
                raise payload
            # Once the caller cancels nothing more is yielded, including a
            # window already rendered ahead.
            refuse_if_cancelled()
            assert isinstance(payload, Future)
            yield payload.result()
    finally:
        stop.set()
        # Take back every render that has not started; the one running cannot
        # be preempted and is the only one the join waits for.
        with submitted_lock:
            doomed = list(submitted)
            submitted.clear()
        for future in doomed:
            future.cancel()
        try:
            while True:
                out.get_nowait()
        except queue.Empty:
            pass
        producer.join(timeout=_PRODUCER_JOIN_TIMEOUT)
        if producer.is_alive():
            engine._wedge(
                f"{producer.name} was still running "
                f"{_PRODUCER_JOIN_TIMEOUT:.0f}s after a stream closed, and it "
                "still holds the token generator and the renderer"
            )
            _LOG.error(
                "%s is still running %.0fs after the stream closed; this "
                "engine refuses every call from here",
                producer.name,
                _PRODUCER_JOIN_TIMEOUT,
            )
