"""Text to text-tokens, exactly as the shipped engine does it.

See ``docs/design/text-funnel.md``.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from tokenizers import Tokenizer

from ..errors import UnsupportedLanguageError
from .numbers import supported_languages

__all__ = ["GraphemeTextFrontend"]

_SPACE = "[SPACE]"

_NEEDS_MODEL_PREPROCESSING = frozenset({"zh", "ja", "he", "ko", "ru"})
"""Refused languages whose refusal has a *specific* reason worth stating.

Their upstream pipeline needs model-based preprocessing this frontend does not
carry. They are a subset of "not in the roster", kept only so the message can
say why rather than just no."""


class GraphemeTextFrontend:
    """The multilingual grapheme frontend (``TextFrontend`` implementation).

    See ``docs/design/text-funnel.md``.
    """

    def __init__(self, tokenizer: str | Path | bytes) -> None:
        if isinstance(tokenizer, bytes):
            # Loaded from memory, because a checkpoint may carry its tokenizer
            # inside it (`Checkpoint.asset`). Spilling to a temp file to satisfy
            # `from_file` would reintroduce exactly the loose artefact that
            # packing removes.
            self._tokenizer = Tokenizer.from_buffer(tokenizer)
            # What to call it in the one message below. A `Path` was standing
            # in for this, so a phrase that is not a filename was being asked
            # for its `.name`.
            source = "<packed in the checkpoint>"
        else:
            path = Path(tokenizer)
            if not path.exists():
                raise FileNotFoundError(
                    f"text tokenizer not found: {path}: it ships beside the checkpoint"
                )
            self._tokenizer = Tokenizer.from_file(str(path))
            source = path.name
        vocab = self._tokenizer.get_vocab()
        for required in ("[START]", "[STOP]", _SPACE):
            if required not in vocab:
                raise ValueError(f"{source}: vocabulary is missing {required!r}")

    def encode(self, text: str, language: str = "en") -> NDArray[np.int64]:
        """Normalise and tokenise, the recipe in ``docs/design/text-funnel.md``.

        Raises:
            UnsupportedLanguageError: ``language`` is not one of the twelve in
                :func:`loudkit.frontend.numbers.supported_languages`.
        """
        lang = language.lower()
        roster = supported_languages()
        if lang not in roster:
            why = (
                "needs model-based text preprocessing "
                "(Cangjie/kana/diacritics/jamo/stress) that this frontend does not carry"
                if lang in _NEEDS_MODEL_PREPROCESSING
                # The tokenizer holds tags for 31 languages and would happily
                # emit ids for any of them, which is exactly the danger: the
                # model reads those ids as sounds, and a language it was not
                # trained on comes out as confident nonsense rather than as an
                # error.
                else "is not one of the languages this build's text layer is written for"
            )
            raise UnsupportedLanguageError(
                f"language {lang!r} {why}. Supported: {', '.join(roster)}",
                language=lang,
                supported=roster,
            )
        normalised = unicodedata.normalize("NFKD", text.lower())
        # Square brackets never reach the tokenizer from user text.
        normalised = normalised.replace("[", " ").replace("]", " ")
        tagged = f"[{lang}]{normalised}".replace(" ", _SPACE)
        ids = self._tokenizer.encode(tagged).ids
        return np.asarray(ids, dtype=np.int64)

    def __repr__(self) -> str:
        return f"GraphemeTextFrontend(vocab={self._tokenizer.get_vocab_size()})"
