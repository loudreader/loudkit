"""The errors this library raises on purpose, and what a caller can do with them.

Until now loudkit defined exactly one exception of its own and otherwise raised
bare ``ValueError``, ``RuntimeError``, ``NotImplementedError`` and
``FileNotFoundError``. That is fine inside a function and expensive at a
boundary, because the boundary has to classify: the HTTP server maps exception
*types* to status codes, and with builtins the only thing it could map was
"something raised NotImplementedError" — which is true of an unsupported
language, and equally true of a backend with a stub method. One is the caller's
question and answers 400; the other is a server defect and must answer 500. The
server could not tell them apart, so every backend bug arrived at the client as
"your fault".

**Every class here inherits from the builtin it replaces.** ``except
ValueError`` still catches a :class:`WindowOverflowError`, ``except
FileNotFoundError`` still catches a :class:`VoiceNotFoundError`. Nothing written
against the bare-exception shape breaks; what is added is the ability to be *specific* —
``except loudkit.LoudkitError`` for "loudkit refused this", or a named class for
one refusal.

The hierarchy is deliberately small. A class earns its place by being something
a caller would branch on, and each one carries the values a caller would
otherwise have to parse back out of the message.

**Error codes.** Each class also names its condition as a short stable string,
``code`` — the vocabulary the transports speak. An HTTP error body carries it
as ``"code"``, the SSE ``done`` event as ``"error_code"``, gRPC as
``loudkit-error-code`` trailing metadata, so the same refusal has the same name
whether it crossed a process boundary or not. The catalog is frozen: codes are
never renamed or reused, only added. Conditions no class names yet fall back to
``invalid_request`` (the caller's question) or ``server_fault`` (a defect
here); the transports add ``unauthorized`` and ``busy`` for conditions that
exist only at a boundary. :func:`error_code` is the one mapping.
"""

from __future__ import annotations

__all__ = [
    "InvalidTokensError",
    "LoudkitError",
    "NumberGrammarError",
    "UnsupportedLanguageError",
    "VoiceNotFoundError",
    "ProvenanceError",
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

    Raised: never directly. It exists so a caller embedding the library can
    write ``except loudkit.LoudkitError`` and catch the refusals loudkit means,
    without also catching the ``ValueError`` that came out of numpy.

    Carries: nothing of its own. The subclasses carry the diagnostics.

    An error that is *not* a ``LoudkitError`` coming out of loudkit is either a
    bug here or a failure in a dependency — in both cases something to report,
    not something to handle.
    """

    code: str = "invalid_request"
    """This condition's name in the frozen error-code catalog (see the module
    docstring). Class-level and stable: transports send it, callers in any
    language branch on it."""

    def __reduce__(self) -> tuple[object, ...]:
        """Survive ``pickle`` and ``copy``, despite required keyword arguments.

        ``BaseException.__reduce__`` returns ``(type(self), self.args)``, so
        unpickling calls ``cls(*args)`` — and every subclass below takes its
        diagnostics as *required keyword-only* arguments, which that call does
        not supply. Default pickling therefore raises ``TypeError`` and masks
        the original error wherever exceptions cross a process boundary, such
        as errors ferried back from a ``ProcessPoolExecutor`` worker.

        Rebuilt through ``__new__`` and ``BaseException.__init__`` rather than
        by giving the keywords defaults, because the diagnostics are the whole
        reason these classes exist and a default would make them optional at
        every raise site.
        """
        return (_rebuild, (type(self), self.args, self.__dict__.copy()))


class NumberGrammarError(LoudkitError, ValueError):
    """A number could not be said in the requested language.

    Raised: by :mod:`loudkit.numbers` when a language has no grammar, or a
    value is larger than the grammar's largest scale. Raised rather than
    returning the digits: a caller who gets ``"1000000000"`` back has no way to
    tell it apart from a number the grammar handled, and silently reading
    digits aloud is the failure that module exists to remove.

    Carries: nothing beyond its message.

    Defined here rather than in :mod:`loudkit.numbers`, where it started and
    where it is still exported from, only because that module imports this one:
    the class has to live below the base it now inherits.
    ``loudkit.frontend.numbers.NumberGrammarError`` remains the same object.
    """

    code = "number_grammar"


class UnsupportedLanguageError(LoudkitError, NotImplementedError):
    """A language this build's text frontend cannot preprocess.

    Raised: by :class:`~loudkit.frontend.text.GraphemeTextFrontend` for the
    languages whose upstream pipeline needs model-based preprocessing this
    frontend does not carry — Cangjie codes, kana conversion, diacritisation,
    jamo decomposition, stress marks. Refused rather than silently skipped: a
    grapheme read of Chinese is the wrong sounds in the right order, and no
    error downstream would say why.

    Carries:
        language: the id that was refused, lowercased.
        supported: the language ids this build *does* accept, sorted. Read from
            the tokenizer's own vocabulary rather than hardcoded, so it cannot
            drift from the tokenizer that ships.

    Still a ``NotImplementedError``, so existing ``except NotImplementedError``
    handlers keep working — but the server now catches *this* rather than the
    builtin, which is how a genuine ``NotImplementedError`` from a half-written
    backend stopped being reported to the client as a bad request.
    """

    code = "unsupported_language"

    def __init__(self, message: str, *, language: str, supported: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.language = language
        self.supported = supported


_CLOSE_ENOUGH = 0.6
"""How similar a name must be to be offered as the one you meant.

``difflib``'s own default, kept rather than tuned: below it the suggestions stop
being suggestions — every two-voice library would name one of them for any typo
at all, including a name that shares nothing with it. A wrong guess is worse
than none here, because the caller is already confused about which voices exist.
"""


class VoiceNotFoundError(LoudkitError, FileNotFoundError):
    """No voice by that name or path.

    Raised: by :class:`~loudkit.transports.http.VoiceLibrary` when a request names a
    voice the library does not hold, and by
    :func:`~loudkit.hub.resolve_voice` when a reference is neither a file nor
    resolvable against a repo.

    Carries:
        ref: what was asked for — a name over HTTP, a path or name in-process.
        available: the names that *were* found, when listing them is cheap
            (a local directory). Empty when it is not — resolving a name
            against a remote repo would mean a network call to answer an error,
            and an error that goes slower than the thing that failed is its own
            problem.

    When ``available`` holds a close match, the message ends with ``did you mean
    '<name>'?``. Done here rather than at each raise site so that every site
    which can afford to list alternatives gets the suggestion automatically —
    the message and the list are the same fact, and letting them be assembled
    separately is how one of them ends up stale. Voice names are short,
    lowercase donor or character names (``kathleen``, ``gosia``, ``thorsten``),
    which is the exact shape a caller retypes wrong and a reader scans past in
    a list of twenty.

    Still a ``FileNotFoundError``, so a caller catching that keeps catching
    this, and the CLI's "not found:" path is unchanged.
    """

    code = "voice_not_found"

    def __init__(self, message: str, *, ref: str, available: tuple[str, ...] = ()) -> None:
        super().__init__(message + _did_you_mean(ref, available))
        self.ref = ref
        self.available = available


def _did_you_mean(ref: str, available: tuple[str, ...]) -> str:
    """`` — did you mean 'x'?`` for the nearest name, or nothing.

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
    return f" — did you mean {close[0]!r}?" if close else ""


class InvalidTokensError(LoudkitError, ValueError):
    """A speech token sequence the caller supplied that the engine cannot use.

    Raised: by :meth:`~loudkit.engine.Engine.synthesize` and friends when
    ``previous_tokens`` holds an id outside the acoustic codebook — a control
    token, a negative, or something past the vocabulary. Those ids index an
    embedding table, so the alternative to refusing is an index error three
    stages away from the argument that caused it.

    Carries:
        token: the first offending id, so a caller filtering a long sequence
            knows which entry to look at rather than which list.
        limit: the exclusive upper bound. Ids must satisfy ``0 <= id < limit``.

    It earns a class rather than a bare ``ValueError`` because a boundary has to
    classify it, and getting that wrong was visible: the streaming route maps
    exception *types* to ``bad_request`` or ``server_fault``, everything outside
    this hierarchy is a server fault by definition, and so a client sending one
    bad integer was told the server had broken. That is the one verdict a client
    cannot act on, and it is exactly backwards — the request is the thing to fix.
    The one-shot route was already correct, which is how the two disagreed.

    Still a ``ValueError``, which is what the HTTP server maps to 422 and what an
    existing ``except ValueError`` around a synthesis call already catches.
    """

    code = "invalid_tokens"

    def __init__(self, message: str, *, token: int, limit: int) -> None:
        super().__init__(message)
        self.token = token
        self.limit = limit


class NothingToSpeakError(LoudkitError, ValueError):
    """The text funnel removed every character of the request.

    Emoji, bare symbols and invisible marks are legal input at a transport and
    gone by the frontend. Every entry point refuses identically — generating
    against an empty prompt yields near-silence with no error, the one failure
    a caller is least likely to notice.
    """


class WindowOverflowError(LoudkitError, ValueError):
    """More speech tokens than the renderer's window holds.

    Raised: by :class:`~loudkit.engine.Engine` when a single window's generation
    exceeds ``window.max_speech_tokens``. Silent truncation is not an option:
    text would go missing while the audio still sounds fine, and only a
    listener who knows the passage would notice.
    :meth:`~loudkit.engine.Engine.synthesize_long` is the remedy for long
    input.

    Carries:
        n_tokens: how many speech tokens were produced.
        window: how many the window holds. The overflow is the difference, and
            the message states it in seconds of speech as well as in tokens.

    Still a ``ValueError``, which is what the HTTP server maps to 422 and what
    every existing test expects.
    """

    code = "window_overflow"

    def __init__(self, message: str, *, n_tokens: int, window: int) -> None:
        super().__init__(message)
        self.n_tokens = n_tokens
        self.window = window


class ProvenanceError(LoudkitError, ValueError):
    """A provenance box is present and cannot be read.

    Distinct from "no provenance", which :func:`~loudkit.provenance.read` still
    reports as ``None``. The difference matters to the only caller who asks: an
    auditor establishing whether a file is labelled cannot act on one answer
    that covers both "this file was never marked" and "this file was marked and
    the marking is damaged" — and the second is what tampering looks like.
    """

    code = "provenance_invalid"


def error_code(exc: BaseException) -> str:
    """The catalog code for ``exc`` — the one mapping every transport uses.

    A :class:`LoudkitError` names its own condition. Anything else that a
    boundary chose to report as a refusal (a bare ``ValueError`` from a layer
    that has not earned a class yet) is ``invalid_request``; an exception the
    boundary did *not* choose — a stub method, a numpy failure, a bug — is
    ``server_fault``, decided by the caller passing it here only for errors it
    classified as refusals. This function does not guess: it reads the class.
    """
    if isinstance(exc, LoudkitError):
        return exc.code
    return "invalid_request"
