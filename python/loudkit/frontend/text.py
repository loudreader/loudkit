"""Text to text-tokens, exactly as the shipped engine does it.

The pipeline is deliberately thin — lowercase, NFKD, a language tag, spaces to
``[SPACE]``, then plain BPE over Unicode scalars — because that is what the
production engine runs (ChatterboxTokenizer.swift is a bit-parity port of the
same recipe, tested against the Python reference). Anything richer, like the
upstream ``punc_norm`` that rewrites ellipses and appends full stops, was a
research-harness convenience that the shipped app never applied; adding it
here would make loudkit read text differently from the engine it is replacing.

The speech funnel that scrubs raw text before tokenising — invisible
characters, symbols, footnote markers, punctuation, and (for Polish) English
respelling — lives in :mod:`loudkit.frontend.polish` and is applied by the
engine, mirroring the Swift backend's call to ``SpeechText.prepared``. This
module is the tokenizer and only the tokenizer, so the conformance fixture's
frontend vectors keep testing it in isolation.

Language handling is an **allowlist**: the twelve ids
:func:`loudkit.numbers.supported_languages` reports, which is the roster in
``models/data/numbers.json`` that every port already loads. Anything else is
refused.

It was a blacklist of zh/ja/he/ko/ru, and the difference matters because the
tokenizer's vocabulary carries tags for 31 languages. A blacklist accepted the
other 26 and the tag went through, so ``encode(text, "bg")`` NFKD-mangled
Cyrillic into ids the model reads as sounds it was never trained to make — no
error, plausible-sounding audio, wrong language. Worse once
:class:`~loudkit.errors.UnsupportedLanguageError` began advertising what *would*
have worked: a client refused for ``zh`` would read the list and retry with a
language the kit cannot actually speak.

The five model-based ones are still named separately, because *why* they are
refused is real information — Cangjie codes, kanji→hiragana, diacritisation,
jamo decomposition, stress marks, all of them optional heavyweight models this
frontend does not carry. A Polish-style grapheme read of Chinese is the wrong
sounds in the right order, and no error downstream would say why.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..errors import UnsupportedLanguageError
from .numbers import supported_languages

__all__ = ["GraphemeTextFrontend"]

_SPACE = "[SPACE]"

_NEEDS_MODEL_PREPROCESSING = frozenset({"zh", "ja", "he", "ko", "ru"})
"""Refused languages whose refusal has a *specific* reason worth stating.

Their upstream pipeline needs model-based preprocessing this frontend does not
carry. They are a subset of "not in the roster" — kept only so the message can
say why rather than just no."""


class GraphemeTextFrontend:
    """The multilingual grapheme frontend (``TextFrontend`` implementation).

    Deterministic and model-free: the same text and language always produce the
    same ids. Start/stop text tokens are *not* added here — they belong to the
    token generator, which owns its own sequence framing (mirroring the shipped
    split, where the tokenizer emits bare ids and the T3 runner frames them).

    Args:
        tokenizer_path: the ``grapheme_mtl_merged_expanded_v1`` tokenizer JSON
            (HF ``tokenizers`` format), normally shipped beside the checkpoint.
    """

    def __init__(self, tokenizer: str | Path | bytes) -> None:
        from tokenizers import Tokenizer

        if isinstance(tokenizer, bytes):
            # Loaded from memory, because a checkpoint may carry its tokenizer
            # inside it (`Checkpoint.asset`). Spilling to a temp file to satisfy
            # `from_file` would reintroduce exactly the loose artefact that
            # packing removes.
            self._tokenizer = Tokenizer.from_buffer(tokenizer)
            path = Path("<packed in the checkpoint>")
        else:
            path = Path(tokenizer)
            if not path.exists():
                raise FileNotFoundError(
                    f"text tokenizer not found: {path} — it ships beside the checkpoint"
                )
            self._tokenizer = Tokenizer.from_file(str(path))
        vocab = self._tokenizer.get_vocab()
        for required in ("[START]", "[STOP]", _SPACE):
            if required not in vocab:
                raise ValueError(f"{path.name}: vocabulary is missing {required!r}")

    def encode(self, text: str, language: str = "en") -> NDArray[np.int64]:
        """Normalise and tokenise. See the module docstring for the recipe.

        Raises:
            UnsupportedLanguageError: ``language`` is not one of the twelve in
                :func:`loudkit.numbers.supported_languages`.
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
        #
        # The vocabulary holds 117 bracket tokens — the language tags, and
        # paralinguistic events like [sigh], [gasp], [UH] trained into the base
        # model — and the tokenizer matches them greedily, so "he [sigh]ed"
        # would emit control token 611 and the model would sigh. The funnel
        # already destroys brackets on the engine path; this makes the
        # guarantee structural for anyone calling encode directly. The one tag
        # that belongs here is the language tag, and it is added after.
        normalised = normalised.replace("[", " ").replace("]", " ")
        tagged = f"[{lang}]{normalised}".replace(" ", _SPACE)
        ids = self._tokenizer.encode(tagged).ids
        return np.asarray(ids, dtype=np.int64)

    def __repr__(self) -> str:
        return f"GraphemeTextFrontend(vocab={self._tokenizer.get_vocab_size()})"
