"""An MCP server, so any MCP-aware agent can speak in a cloned voice.

The same philosophy as :mod:`loudkit.server`, carried to a second protocol:
the server holds **no synthesis path of its own**. Its tools build nothing and
decide nothing — they resolve a voice and call :func:`~loudkit.server.render_bytes`
and the engine's methods, so a request cannot reach code the library tests do
not cover.

That shared implementation is deliberate and worth keeping strict. A second
synthesis path is a second thing to keep in agreement, and this library exists
because two paths drifted. The MCP server and the HTTP server are two transports
for one engine, not two engines.

Tools exposed: ``list_voices``, ``synthesize`` (text, voice, seed, language,
speed, previous_tokens, format → audio), ``describe`` (the resolved
algorithm/execution, i.e. the line every run should be able to answer about
itself). ``synthesize`` returns the audio as base64 — WAV by default, or any
other :data:`~loudkit.synthesis.AudioFormat`; ``flac`` carries the same
samples at about a quarter the size, which matters here more than over HTTP
because the reply lands in a model's context — plus duration, token count and
the ``continuation`` tail to chain the next call from: the same facts the HTTP
route puts in its headers.

Install with ``pip install "loudkit[mcp]"``. Run with
``loudkit mcp --checkpoint <path> --voices <dir>``.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, cast, get_args

from ..errors import UnsupportedLanguageError, VoiceNotFoundError
from ..synthesis import (
    _MAX_PREVIOUS_TOKENS,
    _MAX_TEXT_LEN,
    _MAX_WAIT_S,
    AudioFormat,
    VoiceLibrary,
)

_AUDIO_FORMATS: frozenset[str] = frozenset(get_args(AudioFormat))
"""What ``synthesize`` accepts as ``format`` — derived from the one
:data:`~loudkit.synthesis.AudioFormat` rather than repeated, so this transport
cannot offer an encoding the synthesis surface does not have."""

__all__ = ["build_server", "run_stdio"]

_MISSING_EXTRA = 'the MCP server needs the "mcp" package.\n  pip install "loudkit[mcp]"'


def _over_cap(text: str, previous_tokens: list[int] | None) -> dict[str, Any] | None:
    """A refusal for input past the HTTP surface's caps, or ``None``.

    ``/v1/synthesize`` bounds both of these through its pydantic model, and this
    entry point had no schema doing it — so the same single-flight engine was
    reachable over stdio with an unbounded prompt and an unbounded conditioning
    history, from a host that hands the tool whatever a model emitted. The
    constants are imported rather than repeated: a caller finding one door
    stricter than the other is how the looser one gets used.
    """
    if not text.strip():
        # `SpeakRequest.text` is `Field(min_length=1)`, so HTTP refuses this at
        # the schema with a message naming the field. Over stdio it went to the
        # engine, which raises "nothing to speak" from three frames deeper — the
        # same refusal, worse addressed, from the transport whose caller is a
        # model reading the error rather than a person.
        return {"error": "text is empty", "error_kind": "bad_request"}
    if len(text) > _MAX_TEXT_LEN:
        return {
            "error": f"text is {len(text)} characters; the cap is {_MAX_TEXT_LEN}",
            "error_kind": "bad_request",
        }
    if previous_tokens is not None and len(previous_tokens) > _MAX_PREVIOUS_TOKENS:
        return {
            "error": (
                f"previous_tokens has {len(previous_tokens)} entries; "
                f"the cap is {_MAX_PREVIOUS_TOKENS}"
            ),
            "error_kind": "bad_request",
        }
    return None


def _load_mcp() -> Any:
    """Import the MCP SDK, with a name that says what to install."""
    try:
        from mcp.server.mcpserver import MCPServer
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on extras
        raise ModuleNotFoundError(_MISSING_EXTRA) from exc
    return MCPServer


def build_server(
    checkpoint: str,
    voices: str | Path | None = None,
    *,
    device: str | None = None,
    engine: Any | None = None,
) -> Any:
    """Build an MCP server backed by a warm engine and a voice library.

    Args:
        checkpoint: the synthesis ``.safetensors``, or a repo id.
        voices: directory of voice profiles. Defaults to ``voices/`` beside the
            checkpoint, matching :func:`~loudkit.server.serve`.
        engine: an already-loaded :class:`~loudkit.engine.Engine`. When given,
            ``checkpoint`` is used only to locate the default voices directory;
            this is how tests inject a fake engine without a 747 MB load. Give
            ``voices`` as well and ``checkpoint`` is not read at all, so a repo
            id there costs nothing and needs no ``hub`` extra installed.

    Returns:
        A configured :class:`mcp.server.mcpserver.MCPServer`.
    """
    mcp_server_cls = _load_mcp()

    from .. import __version__, load

    if engine is not None and voices is not None:
        # Both questions the checkpoint answers have been answered by the
        # caller: the engine is loaded and the voice directory was named. It is
        # the only argument left, and nothing reads it. Resolving it anyway
        # would fetch the release -- 747 MB, and the `hub` extra -- to compute a
        # path that is then discarded, which is the cost the `engine` argument
        # exists to avoid.
        library = VoiceLibrary(Path(voices))
    else:
        from ..hub import backend_for_device, resolve_checkpoint

        # A repo id resolves to the checkpoint *inside the snapshot the hub
        # returned*, so the default voice directory below is the snapshot's own
        # `voices/`. Handing the raw id to `Path` instead computed
        # `Path("org/repo").parent / "voices"` — a directory that does not exist
        # — and the server started with the release's voices silently absent.
        # The same seam `serve` uses in the HTTP and gRPC transports.
        #
        # Normalised whatever shape it has, and by the backend the device needs.
        # Two things went wrong when this ran only for a repo id and always
        # fetched torch: a local *directory* kept its own name, so the voices
        # below were looked for one level above the release; and `device="onnx"`
        # fetched a snapshot holding no graphs, which the backend then could not
        # run. `resolve_checkpoint` answers all three shapes: a file is itself, a
        # directory yields the checkpoint inside it, a repo id fetches the set.
        ckpt = resolve_checkpoint(str(checkpoint), backend=backend_for_device(device))
        library = VoiceLibrary(Path(voices) if voices else ckpt.parent / "voices")
    if engine is None:
        # `device` reaches `load` here. The CLI parsed `--device` for every
        # subcommand and this one dropped it, so `loudkit mcp --device cuda:3`
        # was accepted and ignored — a different device, a different memory
        # profile and a different speed from the one the operator asked for,
        # with no warning at all.
        engine = load(str(ckpt), device=device)
        names = library.names()
        if names:
            # First-use costs belong to startup, not the first tool call —
            # see Engine.warm.
            engine.warm(library.load(names[0]))

    # The engine is single-flight (same as the HTTP server): a CUDA graph
    # capture is not reentrant, and torch modules are not thread-safe. MCP tool
    # calls can be dispatched concurrently by the host, so synthesis must
    # serialise behind a lock.
    import threading

    # Same bound as the HTTP server's queue wait: a wedged render must not hold
    # every tool call open indefinitely.
    _synth_lock = threading.Lock()
    _synth_lock_timeout_s = _MAX_WAIT_S

    server = mcp_server_cls(
        name="loudkit",
        title="loudkit TTS",
        description="Local text-to-speech in any voice the engine can clone.",
        version=__version__,
        instructions=(
            "Same text, voice and seed give the same audio every time. "
            "Voices are files in the library directory, resolved by name."
        ),
    )

    @server.tool(  # type: ignore[untyped-decorator]
        title="List voices",
        description="Names of every voice profile the server can speak in.",
    )
    def list_voices() -> list[str]:
        return library.names()

    @server.tool(  # type: ignore[untyped-decorator]
        title="Synthesize speech",
        description=(
            "Turn text into speech in a named voice. Returns the audio as "
            "base64 plus the audio duration and token count. `format` is "
            '"wav" by default; "flac" is the same samples, losslessly, at '
            "about a quarter the size — worth asking for when the reply is "
            "saved to a file rather than played. Same text, voice and seed "
            "give the same bytes. Omit `language` to read the text in the "
            "voice's own language; pass one only to read text in a language the "
            "voice was not enrolled in. `speed` is playback speed in [0.5, 2.0] "
            "with the pitch preserved — 1.0, the default, is an exact bypass. "
            "To read a long text as several calls without an audible restart at "
            "each join, pass the previous reply's `continuation` list back as "
            "`previous_tokens`. "
            "Check `truncated`: when true the "
            "utterance hit the token cap and the speech is cut off mid-sentence. "
            "A refusal comes back as `error` with `error_kind` "
            '"bad_request" — something about this call to fix, and `supported` '
            "or `available` listing what would have worked."
        ),
    )
    def synthesize(  # noqa: PLR0911 - each error kind is its own answer
        text: str,
        voice: str,
        seed: int = 0,
        language: str | None = None,
        speed: float = 1.0,
        previous_tokens: list[int] | None = None,
        format: str = "wav",  # noqa: A002 - the HTTP surface's name for the same choice
    ) -> dict[str, Any]:
        refusal = _over_cap(text, previous_tokens)
        if refusal is not None:
            return refusal
        if format not in _AUDIO_FORMATS:
            return {
                "error": f"unknown format {format!r}",
                "error_kind": "bad_request",
                "supported": sorted(_AUDIO_FORMATS),
            }
        try:
            profile = library.load(voice)
        except VoiceNotFoundError as exc:
            return {"error": str(exc), "error_kind": "bad_request", "available": exc.available}
        except ValueError as exc:  # not a voice *name* — a path, an empty string
            return {"error": str(exc), "error_kind": "bad_request"}
        try:
            if not _synth_lock.acquire(timeout=_synth_lock_timeout_s):
                return {
                    "error": (
                        "engine busy: another synthesis is holding the lock "
                        f"(waited {_synth_lock_timeout_s:.0f}s)"
                    ),
                    "error_kind": "busy",
                }
            try:
                rendered = _render(
                    engine,
                    profile,
                    text,
                    seed=seed,
                    language=language,
                    speed=speed,
                    previous_tokens=previous_tokens,
                    audio_format=cast(AudioFormat, format),
                )
            finally:
                _synth_lock.release()
        except UnsupportedLanguageError as exc:
            # UnsupportedLanguageError, not the builtin NotImplementedError.
            # Both are answers rather than transport failures — an agent that
            # gets a protocol error learns nothing, while the message names the
            # twelve languages that do work. But they are different answers:
            # "you asked for Chinese" versus "a backend method is a stub", and
            # an agent that cannot tell them apart will helpfully retry the
            # request that was never the problem.
            return {
                "error": str(exc),
                "error_kind": "bad_request",
                "supported": list(exc.supported),
            }
        except ValueError as exc:  # over-window without chunking, bad config
            return {"error": str(exc), "error_kind": "bad_request"}
        # Deliberately not caught: a bare NotImplementedError is a defect here,
        # not a question about the call. Swallowing it into the same
        # {"error": ...} shape told the agent its request was wrong and hid a
        # broken build from whoever could fix it. It escapes to the MCP
        # framework's own failure path, which is what a server fault looks like.
        return {
            "audio": base64.b64encode(rendered.data).decode(),
            # Beside the bytes, as everywhere else: a field of base64 does not
            # say what it decodes to, and the agent writing it to a file needs
            # the extension and the player needs the type.
            "format": format,
            "media_type": rendered.media_type,
            "duration": round(rendered.duration, 4),
            "tokens": rendered.n_tokens,
            "sample_rate": engine.algorithm.sample_rate,
            "fingerprint": engine.algorithm.fingerprint(),
            # Not an error: the audio is real, it is just incomplete. An agent
            # that cannot see this reads a cut-off sentence as a finished one.
            "truncated": rendered.hit_token_cap,
            # The tail to send back as `previous_tokens` next call, so a
            # multi-part reading does not restart its pitch contour at every
            # join. The tail rather than every token id: it is all the engine
            # uses, and a few hundred integers in a tool result is context an
            # agent pays for and cannot act on.
            "continuation": list(rendered.continuation),
        }

    @server.tool(  # type: ignore[untyped-decorator]
        title="Describe the engine",
        description=(
            "The resolved algorithm and execution configuration. Log this "
            "whenever a synthesis surprises you: it is the line that tells you "
            "which mode was active."
        ),
    )
    def describe() -> dict[str, str]:
        return {
            "algorithm": engine.algorithm.describe(),
            "execution": engine.execution.describe(),
            "fingerprint": engine.algorithm.fingerprint(),
            "device": engine.execution.device,
        }

    return server


def _render(
    engine: Any,
    profile: Any,
    text: str,
    *,
    seed: int,
    language: str | None,
    speed: float = 1.0,
    previous_tokens: list[int] | None = None,
    audio_format: AudioFormat = "wav",
) -> Any:
    """The one place audio is made, shared with the HTTP server.

    Imported here rather than at module top so a missing ``soundfile`` (the
    ``audio`` extra) fails at tool-call time with its own message, not at
    import of a server the caller may only be inspecting.
    """
    from ..synthesis import render_bytes

    return render_bytes(
        engine,
        text,
        profile,
        seed=seed,
        language=language,
        speed=speed,
        previous_tokens=previous_tokens,
        audio_format=audio_format,
    )


def run_stdio(
    checkpoint: str, voices: str | Path | None = None, *, device: str | None = None
) -> None:
    """Run the MCP server over stdio, for stdio-only clients."""
    server = build_server(checkpoint, voices, device=device)
    server.run(transport="stdio")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``loudkit mcp``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="loudkit mcp", description="Serve loudkit over the Model Context Protocol."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="the synthesis .safetensors, or a repo id",
    )
    parser.add_argument("--voices", type=Path, help="directory of voice profiles")
    parser.add_argument("--device", help="backend: cpu, cuda, cuda:<index>, mps, coreml, onnx")
    args = parser.parse_args(argv)

    run_stdio(args.checkpoint, args.voices, device=args.device)
    return 0


# Deliberately not in `[project.scripts]`, and kept only for `python -m`.
#
# `loudkit mcp` is the entry point: it shares `--checkpoint`, `--voices` and
# `--device` with every other subcommand, and its `--help` is the one a reader
# finds. This parser accepted a different spelling with a different help text
# for the same program — two CLIs for one thing, which is how a flag gets fixed
# in one of them.
if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
