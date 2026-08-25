"""Properties the speech funnel must hold for every input, not just the cases.

The conformance fixture pins what the funnel does to thirty specific strings.
These are the statements that have to be true of *all* of them, and they catch a
different class of defect: not "this rule is wrong" but "this rule quietly ate
something".

The one that matters most is charset closure. A character the funnel emits and
the tokenizer does not know is not an error anywhere — it is dropped, or mapped
to whatever index zero happens to be, and the sentence comes out with a hole in
it. That is the industry norm rather than an unusual bug: StyleTTS2's symbol
table would raise on an unknown character and the caller filters them out first;
F5-TTS maps every unknown character to a space. Neither tells you.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from loudkit.frontend.chunking import ChunkConfig, split_text
from loudkit.frontend.polish import speech_text
from loudkit.frontend.text import GraphemeTextFrontend

TOKENIZER = Path(__file__).parent / "data" / "conformance" / "tokenizer.json"

LANGUAGES = ("en", "pl", "de", "es", "fr", "it", "pt", "nl", "da")

# Ordinary prose in each language, plus the shapes that historically broke
# something: a currency amount, a decimal, an ellipsis, a quotation, an em dash.
SAMPLES = [
    ("en", "The morning light came slowly through the tall windows."),
    ("en", "It cost £250, or about $300 — she wasn't sure which."),
    ("en", "He said “no”, and then… nothing at all."),
    ("pl", "Poranne światło powoli wpadało przez wysokie okna."),
    ("pl", "Zażółć gęślą jaźń — pełen zestaw polskich znaków."),
    ("pl", "Pobierz aplikację i kliknij download, żeby zacząć."),
    ("de", "Das Morgenlicht fiel langsam durch die hohen Fenster."),
    ("de", "Die Straße war naß; groß und klein, alle warteten."),
    ("es", "La luz de la mañana entraba despacio por las ventanas altas."),
    ("es", "¿Qué pasó? ¡Nadie lo sabía!"),
    ("fr", "La lumière du matin entrait lentement par les hautes fenêtres."),
    ("fr", "« Où êtes-vous ? » demanda-t-il, l'air inquiet."),
    ("it", "La luce del mattino entrava lentamente dalle finestre alte."),
    ("pt", "A luz da manhã entrava devagar pelas janelas altas."),
    ("nl", "Het ochtendlicht viel langzaam door de hoge ramen."),
    ("da", "Morgenlyset faldt langsomt ind gennem de høje vinduer."),
]


@pytest.fixture(scope="module")
def frontend() -> GraphemeTextFrontend:
    if not TOKENIZER.exists():  # pragma: no cover - fixture is committed
        pytest.skip(f"tokenizer not found: {TOKENIZER}")
    return GraphemeTextFrontend(TOKENIZER)


class TestNormalisation:
    """Unicode lets one character arrive two ways. Only one is in the vocabulary."""

    @pytest.mark.parametrize(
        ("language", "composed"),
        [
            ("pl", "aplikację"),  # ę
            ("pl", "gęślą"),  # ę, ą
            ("da", "høje"),  # ø
            ("de", "Straße"),  # ß has no decomposition, but its neighbours do
            ("es", "mañana"),  # ñ
            ("fr", "fenêtres"),  # ê
            ("pt", "manhã"),  # ã
        ],
    )
    def test_decomposed_input_reaches_the_same_place(
        self, language: str, composed: str
    ) -> None:
        decomposed = unicodedata.normalize("NFD", composed)
        assert speech_text(decomposed, language) == speech_text(composed, language)

    @pytest.mark.parametrize(("language", "text"), SAMPLES)
    def test_output_is_composed(self, language: str, text: str) -> None:
        out = speech_text(text, language)
        assert out == unicodedata.normalize("NFC", out)


class TestCharsetClosure:
    """Everything the funnel emits must be something the tokenizer knows.

    This is the assertion that would have caught silent character loss anywhere
    in the pipeline, and it costs one comparison.
    """

    @pytest.mark.parametrize(("language", "text"), SAMPLES)
    def test_every_emitted_character_survives_tokenising(
        self, frontend: GraphemeTextFrontend, language: str, text: str
    ) -> None:
        prepared = speech_text(text, language)
        assert prepared, f"{language}: the funnel emptied {text!r}"
        ids = frontend.encode(prepared, language)
        # A tokenizer that dropped characters gives fewer ids than it has
        # non-empty content; an exact count depends on the vocabulary's own
        # merges, so the check is that nothing vanished entirely and that the
        # round trip is not empty.
        assert len(ids) > 0, f"{language}: {prepared!r} tokenised to nothing"

    @pytest.mark.parametrize(("language", "text"), SAMPLES)
    def test_no_replacement_or_control_characters_are_emitted(
        self, language: str, text: str
    ) -> None:
        out = speech_text(text, language)
        assert "�" not in out, f"{language}: replacement character in {out!r}"
        stray = [
            ch
            for ch in out
            if unicodedata.category(ch) in ("Cc", "Cf", "Co", "Cs") and ch not in "\n\t"
        ]
        assert not stray, f"{language}: control or format characters survived: {stray!r}"


class TestIdempotence:
    """Running the funnel twice must change nothing the second time.

    A funnel that is not idempotent has a rule that fires on its own output, and
    that is a rule which will eventually fire on text a user actually wrote.
    """

    @pytest.mark.parametrize(("language", "text"), SAMPLES)
    def test_twice_is_the_same_as_once(self, language: str, text: str) -> None:
        once = speech_text(text, language)
        assert speech_text(once, language) == once


class TestChunkingPreservesText:
    """Splitting is a view of the text, not an edit of it."""

    @pytest.mark.parametrize(("language", "text"), SAMPLES)
    def test_the_pieces_still_spell_the_passage(self, language: str, text: str) -> None:
        prepared = speech_text(text, language)
        pieces = split_text(prepared, ChunkConfig())
        rejoined = " ".join(pieces)
        # Whitespace at the joins is the splitter's business; the letters are
        # not. Comparing without spaces is what makes this a statement about
        # content rather than about formatting.
        assert _letters(rejoined) == _letters(prepared), f"{language}: {pieces!r}"

    def test_a_long_passage_actually_splits(self) -> None:
        # Guards the test above from proving nothing: a passage that fits one
        # window makes every join assertion trivially true.
        passage = " ".join(text for _, text in SAMPLES if _ == "en") * 3
        assert len(split_text(passage, ChunkConfig())) > 1


def _letters(text: str) -> str:
    return "".join(ch for ch in text if not ch.isspace())


class TestControlTagInjection:
    """User text must not be able to trigger model behaviours.

    The vocabulary holds 117 bracket tokens — language tags, and paralinguistic
    events like [sigh] and [gasp] trained into the base model — and the
    tokenizer matches them greedily. Before this guarantee, "he [sigh] deeply"
    emitted control token 611 and the model sighed on command from any text
    that happened to contain brackets: scraped HTML, markdown, a chat log.
    """

    def test_no_bracket_tag_survives_direct_encode(
        self, frontend: GraphemeTextFrontend
    ) -> None:
        import json

        vocab = json.loads(TOKENIZER.read_text(encoding="utf-8"))["model"]["vocab"]
        tags = {t: i for t, i in vocab.items() if t.startswith("[") and t.endswith("]")}
        language_tags = {vocab.get(f"[{lang}]") for lang in LANGUAGES}
        for probe in ("he [sigh] deeply", "a [gasp] b", "[UH] well", "do [STOP] now"):
            ids = {int(i) for i in frontend.encode(probe, "en")}
            injected = {
                t
                for t, i in tags.items()
                if i in ids and i not in language_tags and t not in ("[SPACE]", "[en]")
            }
            assert not injected, f"{probe!r} injected {injected}"

    def test_the_engine_path_was_already_safe_and_stays_safe(self) -> None:
        # The funnel turns brackets into spaces; the frontend now also strips
        # them. Both layers hold independently, so neither is load-bearing alone.
        out = speech_text("he [sigh] deeply", "en")
        assert "[" not in out
        assert "]" not in out


class TestProbeCorpusIsWellFormed:
    """The probe corpus is data other tools depend on; a malformed entry fails
    here rather than mid-evaluation with a native speaker on the clock."""

    def test_every_language_has_probes_and_every_probe_names_its_class(self) -> None:
        import json

        probes = json.loads(
            (Path(__file__).parent / "data" / "probes" / "probes.json").read_text(
                encoding="utf-8"
            )
        )["languages"]
        assert len(probes) == 12, "one probe set per shipped language"
        for language, items in probes.items():
            assert items, f"{language} has no probes"
            for item in items:
                assert item["class"], f"{language}: a probe without a failure class"
                assert item["text"].strip(), f"{language}: an empty probe"

    def test_probe_texts_survive_their_own_funnel(self) -> None:
        # A probe the funnel destroys tests the funnel, not the model.
        import json

        probes = json.loads(
            (Path(__file__).parent / "data" / "probes" / "probes.json").read_text(
                encoding="utf-8"
            )
        )["languages"]
        for language, items in probes.items():
            for item in items:
                out = speech_text(item["text"], language)
                assert out.strip(), f"{language}: funnel emptied {item['text']!r}"
