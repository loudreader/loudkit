"""The HTTP wire shapes: request models, reply headers, stream events, error envelopes.

Pydantic models are declared at module level because FastAPI's route
introspection cannot see a model defined inside ``build_app`` under lazy
annotations; the route then silently stops accepting a body. The helpers
beside them build every reply's headers and events from a :class:`Rendered`
and keep every error answer JSON-safe and named from the one catalog in
:mod:`loudkit.errors`. See ``docs/design/transports.md``.
"""

from __future__ import annotations

import base64
import json
import math
from typing import Any

from ..errors import LoudkitError, error_code
from ..models.timestretch import MAX_SPEED, MIN_SPEED
from ..synthesis import AudioFormat, Rendered, _first_exception
from .limits import _MAX_PREVIOUS_TOKENS, _MAX_TEXT_LEN

_MISSING_EXTRA = (
    'the server needs fastapi, uvicorn and soundfile.\n  pip install "loudkit[server]"'
)
"""The HTTP transport's install line, defined here because this is the first
extras guard it crosses.

``http`` imports this module at its own module scope, above its ``fastapi``
guard, so the failure a reader meets first is the one below and the message has
to exist by then. Defining it in ``http`` and importing it back would be a
cycle. Moving it to a module both can import is a 0.1.2 item."""

try:
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except ModuleNotFoundError as exc:  # pragma: no cover - depends on extras
    raise ModuleNotFoundError(_MISSING_EXTRA) from exc


class SpeakRequest(BaseModel):
    """One synthesis request, shared by the one-shot and streaming routes."""

    text: str = Field(min_length=1, max_length=_MAX_TEXT_LEN)
    voice: str
    seed: int = 0
    language: str | None = None
    """Omitted means "the voice's language". Passed through as ``None``: the
    engine owns the argument-then-voice-then-``"en"`` chain."""

    long_form: bool = True
    """True splits at sentence boundaries and joins; false asks for one window,
    which refuses text that does not fit. The wire's name for the engine's
    ``single_window``."""

    speed: float = Field(default=1.0, ge=MIN_SPEED, le=MAX_SPEED)
    """Playback speed, pitch preserved. Bounded here as well as in the engine
    so an out-of-range value is refused before the request queues."""

    format: AudioFormat = "wav"
    """A ``Literal``, so an unknown name is a 422 listing the six that work."""

    previous_tokens: list[int] | None = Field(default=None, max_length=_MAX_PREVIOUS_TOKENS)
    """Speech tokens this request continues from: the ``X-Loudkit-Continuation``
    of the previous reply, or the ``continuation`` of its ``done`` event."""


class OpenAISpeechRequest(BaseModel):
    """OpenAI's ``/v1/audio/speech`` body, so anything that speaks it can use this."""

    model: str = ""
    """Accepted and ignored: this server has one engine, and ``/health`` says which."""

    input: str = Field(min_length=1, max_length=_MAX_TEXT_LEN)
    """Their name for the text."""

    voice: str
    """A voice name in this server's library; ``/v1/voices`` lists them."""

    response_format: str = "wav"
    """Defaults to ``wav`` where OpenAI defaults to ``mp3``: lossless, and a
    format every client decodes."""

    speed: float = 1.0
    """OpenAI allows 0.25 to 4.0 and this engine 0.5 to 2.0. Out of range is
    refused rather than clamped, so a caller is never handed audio it did not
    ask for."""


_OPENAI_FORMATS: dict[str, AudioFormat] = {
    # Their spelling on the left, ours on the right; ours accepted too.
    "wav": "wav",
    "pcm": "pcm16",
    "flac": "flac",
    "pcm16": "pcm16",
    "ogg": "ogg",
    "mp3": "mp3",
    # Ogg Opus, not the Vorbis `ogg` above: the two share a container and no
    # bitstream, so each name is answered with its own codec.
    "opus": "opus",
}
"""OpenAI ``response_format`` values this server can honour, mapped to ours."""

_OPENAI_UNSUPPORTED = ("aac",)
"""Refused by name: libsndfile has no AAC encoder, so this server has none."""

_STREAMABLE: frozenset[str] = frozenset({"wav", "pcm16", "flac"})
"""Formats a chunk can be delivered in on its own. An Ogg bitstream's state
spans the whole stream, Vorbis and Opus alike, and each separately encoded MP3
chunk carries its own encoder delay into the join, so a per-chunk container
would lie about what the bytes are."""

_CODE_BY_STATUS = {
    400: "invalid_request",
    404: "invalid_request",
    # Starlette's router raises this one itself, for a path that exists under
    # another method. A request error, not a server fault: telling the caller
    # to retry would be telling them to retry a request that cannot work.
    405: "invalid_request",
    413: "payload_too_large",
    422: "invalid_request",
    429: "rate_limited",
    503: "busy",
}
"""What an error answer says when the exception that caused it named nothing."""


def _continuation_header(continuation: tuple[int, ...]) -> dict[str, str]:
    """``X-Loudkit-Continuation``, or no header at all when there is nothing to
    carry: an empty header is one every client would have to special-case."""
    if not continuation:
        return {}
    return {"X-Loudkit-Continuation": ",".join(str(t) for t in continuation)}


def _audio_headers(rendered: Rendered, *, sample_rate: int, fingerprint: str) -> dict[str, str]:
    """The one-shot reply's headers: the facts about the audio, since the body is the audio."""
    return {
        "X-Loudkit-Duration": f"{rendered.duration:.3f}",
        "X-Loudkit-Tokens": str(rendered.n_tokens),
        # Always: raw frames carry no header to read the rate from.
        "X-Loudkit-Sample-Rate": str(sample_rate),
        "X-Loudkit-Fingerprint": fingerprint,
        # A truncated utterance is still a 200; the header is the only
        # way a client can tell.
        "X-Loudkit-Truncated": "true" if rendered.hit_token_cap else "false",
        # The claim-only manifest out of band as well as in the WAV
        # trailer: the other encodings cannot carry the box.
        **(
            {"X-Loudkit-Provenance": json.dumps(rendered.provenance, sort_keys=True)}
            if rendered.provenance is not None
            else {}
        ),
        **_continuation_header(rendered.continuation),
    }


def _event(payload: dict[str, object]) -> str:
    """One Server-Sent Event carrying one JSON object."""
    return f"data: {json.dumps(payload)}\n\n"


def _chunk_event(rendered: Rendered, *, sample_rate: int, fingerprint: str) -> str:
    """One streamed chunk: a complete, playable payload in base64.

    The rate and the fingerprint ride on every chunk, as they do on gRPC's
    ``SynthesizeChunk``. The response headers carry both too, but a client that
    is handed events rather than the response object -- a proxy that re-frames
    the stream, a queue, a log -- sees only what is inside the event, and a
    ``pcm16`` payload with no rate beside it is unplayable.
    """
    return _event(
        {
            "audio": base64.b64encode(rendered.data).decode(),
            "media_type": rendered.media_type,
            "duration": rendered.duration,
            "tokens": rendered.n_tokens,
            "truncated": rendered.hit_token_cap,
            "sample_rate": sample_rate,
            "fingerprint": fingerprint,
        }
    )


def _done_event(
    fingerprint: str,
    truncated: bool,
    continuation: tuple[int, ...] = (),
    *,
    error: str | None = None,
    kind: str | None = None,
    code: str | None = None,
) -> str:
    """The one event every client waits for, success or not."""
    done: dict[str, object] = {
        "done": True,
        "fingerprint": fingerprint,
        "truncated": truncated,
        "continuation": list(continuation),
    }
    if error is not None:
        done["error"] = error
        done["error_kind"] = kind
        done["error_code"] = code
    return _event(done)


def _openai_error(status: int, message: str, kind: str) -> JSONResponse:
    """An error in OpenAI's envelope: their clients read ``error.message``."""
    return JSONResponse(
        {"error": {"message": message, "type": kind, "param": None, "code": None}},
        status_code=status,
    )


def _json_safe(value: Any) -> Any:
    """``value`` with every non-finite float replaced by its repr.

    ``json.dumps`` refuses ``nan`` and ``inf``, and pydantic's error list
    carries the input it rejected, so a validation error about such a value
    is the one payload the response renderer cannot write.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _first_message(exc: BaseException) -> str:
    """What the stream's terminal event says: the root exception's class and message.

    The class is carried because ``error_kind`` names only the side of the
    line the exception fell on, and a caller reading the terminal event has no
    status code to tell it which condition refused.
    """
    root = _first_exception(exc)
    return f"{type(root).__name__}: {root}" if str(root) else type(root).__name__


def _error_kind(exc: BaseException) -> str:
    """``bad_request`` or ``server_fault``, for a failure that missed the status line.

    :class:`~loudkit.errors.LoudkitError` is the line: everything loudkit
    refuses on purpose is under it, and everything else is a defect. An agent
    that cannot tell the two apart retries the request that was never wrong.
    """
    return "bad_request" if isinstance(_first_exception(exc), LoudkitError) else "server_fault"


def _error_code(exc: BaseException) -> str:
    """The catalog code for a failure, from its root cause; ``server_fault``
    for anything that is not a deliberate refusal."""
    root = _first_exception(exc)
    return error_code(root) if isinstance(root, LoudkitError) else "server_fault"
