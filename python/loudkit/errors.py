"""The errors this library raises on purpose, and what a caller can do with them.

See ``docs/design/runtime-notes.md``.
"""

from __future__ import annotations

__all__ = [
    "AudioNotFoundError",
    "CancelledError",
    "InvalidTokensError",
    "LoudkitError",
    "NothingToSpeakError",
    "NumberGrammarError",
    "ProvenanceError",
    "UnsupportedFormatError",
    "UnsupportedLanguageError",
    "VoiceNotFoundError",
    "WindowOverflowError",
    "error_code",
]


def _rebuild(
    cls: type[BaseException], args: tuple[object, ...], state: dict[str, object]
) -> BaseException:
    """Reconstruct an exception without going through its ``__init__``.

    Module level because pickle has to be able to name it.
    """
    obj = cls.__new__(cls)
    BaseException.__init__(obj, *args)
    obj.__dict__.update(state)
    return obj


class LoudkitError(Exception):
    """Base for every error loudkit raises deliberately.

    See ``docs/design/runtime-notes.md``.
    """

    code: str = "invalid_request"
    """This condition's name in the frozen error-code catalog, which is written
    out in ``docs/reference/errors.md`` and frozen by
    ``docs/reference/COMPATIBILITY.md``. Class-level and stable: transports send
    it, callers in any language branch on it."""

    def __reduce__(self) -> tuple[object, ...]:
        """Survive ``pickle`` and ``copy``, despite required keyword arguments.

        See ``docs/design/runtime-notes.md``.
        """
        return (_rebuild, (type(self), self.args, self.__dict__.copy()))


class NumberGrammarError(LoudkitError, ValueError):
    """A number could not be said in the requested language.

    See ``docs/design/runtime-notes.md``.
    """

    code = "number_grammar"


class UnsupportedLanguageError(LoudkitError, NotImplementedError):
    """A language this build's text frontend cannot preprocess.

    See ``docs/design/runtime-notes.md``.
    """

    code = "unsupported_language"

    def __init__(self, message: str, *, language: str, supported: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.language = language
        self.supported = supported


_CLOSE_ENOUGH = 0.6
"""How similar a name must be to be offered as the one you meant.

``difflib``'s own default, kept rather than tuned: below it the suggestions stop
being suggestions, every two-voice library would name one of them for any typo
at all, including a name that shares nothing with it. A wrong guess is worse
than none here, because the caller is already confused about which voices exist.
"""


class VoiceNotFoundError(LoudkitError, FileNotFoundError):
    """No voice by that name or path.

    See ``docs/design/runtime-notes.md``.
    """

    code = "voice_not_found"

    def __init__(self, message: str, *, ref: str, available: tuple[str, ...] = ()) -> None:
        super().__init__(message + _did_you_mean(ref, available))
        self.ref = ref
        self.available = available


def _did_you_mean(ref: str, available: tuple[str, ...]) -> str:
    """``, did you mean 'x'?`` for the nearest name, or nothing.

    One suggestion, not three: a caller who mistyped one name is choosing
    between it and the correct one, and a list of maybes is the same work as
    reading ``available`` themselves.
    """
    import difflib

    # `ref` may be a path when the caller passed one; the stem is what would
    # have been a name, and comparing the whole path against bare names finds
    # nothing.
    from pathlib import Path

    stem = Path(ref).stem or ref
    close = difflib.get_close_matches(stem, available, n=1, cutoff=_CLOSE_ENOUGH)
    return f"; did you mean {close[0]!r}?" if close else ""


class InvalidTokensError(LoudkitError, ValueError):
    """A speech token sequence the caller supplied that the engine cannot use.

    See ``docs/design/runtime-notes.md``.
    """

    code = "invalid_tokens"

    def __init__(self, message: str, *, token: int, limit: int) -> None:
        super().__init__(message)
        self.token = token
        self.limit = limit


class AudioNotFoundError(LoudkitError, FileNotFoundError):
    """A recording :func:`~loudkit.enroll` was asked to read is not there.

    See ``docs/design/runtime-notes.md``.
    """

    code = "audio_not_found"


class NothingToSpeakError(LoudkitError, ValueError):
    """The text funnel removed every character of the request.

    Emoji, bare symbols and invisible marks are legal input at a transport and
    gone by the frontend. Every entry point refuses identically, generating
    against an empty prompt yields near-silence with no error, the one failure
    a caller is least likely to notice.
    """


class UnsupportedFormatError(LoudkitError, ValueError):
    """An audio format the loaded libsndfile was built without.

    ``mp3`` and ``opus`` come from libsndfile's MPEG and Opus codecs, which the
    soundfile wheels bundle and a system build may omit. A refusal rather than a
    defect: the request names a real format this server cannot write, and it is
    refused before the engine runs.
    """


class WindowOverflowError(LoudkitError, ValueError):
    """More speech tokens than the renderer's window holds.

    See ``docs/design/runtime-notes.md``.
    """

    code = "window_overflow"

    def __init__(self, message: str, *, n_tokens: int, window: int) -> None:
        super().__init__(message)
        self.n_tokens = n_tokens
        self.window = window


class CancelledError(LoudkitError):
    """``should_cancel`` returned true and :meth:`~loudkit.engine.Engine.synthesize`
    produced nothing.

    Raised rather than returning the chunks that finished, because a short
    result and an interrupted one would be the same object. ``stream`` does
    not raise it: the chunks already yielded are the partial, and the caller
    who flipped the flag knows why the stream ended. Not the ``asyncio`` class
    of the same name; this one is a :class:`LoudkitError`, so a transport can
    map it by ``code``.
    """

    code = "cancelled"


class ProvenanceError(LoudkitError, ValueError):
    """A provenance box is present and cannot be read.

    Distinct from "no provenance", which :func:`~loudkit.provenance.read_provenance` still
    reports as ``None``. The difference matters to the only caller who asks: an
    auditor establishing whether a file is labelled cannot act on one answer
    that covers both "this file was never marked" and "this file was marked and
    the marking is damaged", and the second is what tampering looks like.
    """

    code = "provenance_invalid"


def error_code(exc: BaseException) -> str:
    """The catalog code for ``exc``, the one mapping every transport uses.

    See ``docs/design/runtime-notes.md``.
    """
    if isinstance(exc, LoudkitError):
        return exc.code
    return "invalid_request"
