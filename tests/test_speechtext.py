"""The Polish speech funnel — a bit-parity port of the Swift engine's
SpeechText + LexicalRespelling.

The shipped Swift engine reads Polish text through ``SpeechText.prepared``
before tokenising; this is the Python half of that contract, so the two
engines read the same Polish text identically. The expected values below are
the ones the Swift code's own comments document and the ear test approved —
a drift here means the Python reader and the Swift reader disagree on Polish.
"""

from __future__ import annotations

import re

import pytest

from loudkit.frontend.speechtext import (
    _drop_footnote_markers,
    _punctuation_for_speech,
    _speak_symbols,
    _strip_invisibles,
    lexical_respelling,
    speech_text,
)

from .conftest import tool


class TestFunnel:
    def test_invisibles_are_stripped(self) -> None:
        assert _strip_invisibles("he\u200bllo") == "hello"
        assert _strip_invisibles("\ufefflead") == "lead"
        assert _strip_invisibles("plain") == "plain"

    def test_symbols_become_words(self) -> None:
        assert _speak_symbols("15%", "pl") == "15 procent "
        # `5 €` is a currency *suffix* now, so it leaves this pass spaced the
        # way `$5` does rather than with the generic symbol loop's padding —
        # which the whitespace pass collapsed to the same thing either way. The
        # reason it moved is `0.49¢`: an amount followed by a currency mark is a
        # price, and reaching the clock reader with its dot intact made German
        # say "null Uhr neunundvierzig Cent".
        assert _speak_symbols("5 €", "pl") == "5 euro"
        assert _speak_symbols("a → b", "pl") == "a ,  b"

    def test_currency_prefix_becomes_suffix(self) -> None:
        # "$5" reads "5 dollars", not "dollars 5"
        assert _speak_symbols("$5", "en") == "5 dollars"
        # seven of nine used to hear English; now they hear their own
        assert _speak_symbols("$5", "de") == "5 Dollar"
        assert _speak_symbols("€10", "es") == "10 euros"

    def test_footnote_markers_are_dropped(self) -> None:
        assert _drop_footnote_markers("text[12]") == "text"
        assert _drop_footnote_markers("a[3, 4]b[1-5]") == "ab"
        # a real bracketed phrase survives (bounded at 20 chars)
        assert "[a real phrase]" in _drop_footnote_markers("[a real phrase]")

    def test_non_prosodic_punctuation_becomes_space(self) -> None:
        out = _punctuation_for_speech("hello#world")
        assert "#" not in out
        assert "hello" in out
        assert "world" in out

    def test_prosodic_punctuation_survives(self) -> None:
        assert _punctuation_for_speech("Hello, world!") == "Hello, world!"

    def test_number_separators_survive_between_digits(self) -> None:
        assert _punctuation_for_speech("2.5") == "2.5"
        assert _punctuation_for_speech("3/4") == "3/4"
        # a lone period is prosodic and survives
        assert _punctuation_for_speech("hello. world") == "hello. world"

    def test_in_word_hyphen_survives(self) -> None:
        assert _punctuation_for_speech("well-known") == "well-known"


class TestTheGeneratedLexiconLoadsOnce:
    """One load, one cache, and the class as the door onto it.

    The lexicon is read once through an `lru_cache` over a frozen record, with
    every JSON value checked rather than cast. `PolishLexicon` is in
    `__all__`, so this pins that its five classmethods keep answering.
    """

    def test_the_four_values_are_loaded_once_and_shared(self) -> None:
        from loudkit.frontend.speechtext import PolishLexicon

        assert PolishLexicon.generated() is PolishLexicon.generated()
        assert PolishLexicon.words() is PolishLexicon.words()
        assert PolishLexicon.respell_all() is PolishLexicon.respell_all()
        assert PolishLexicon.polish() is PolishLexicon.polish()
        assert PolishLexicon.payload() is PolishLexicon.payload()

        payload = PolishLexicon.payload()
        assert payload["respell"] is PolishLexicon.generated()
        assert payload["respellAll"] is PolishLexicon.respell_all()

    def test_a_malformed_file_names_the_key_it_tripped_on(self) -> None:
        """A truncated or half-written file gets a container shape wrong, and
        the message has to say which one rather than failing later inside a
        lookup on a value that is not the type it was cast to."""
        from loudkit.frontend import speechtext

        cases = (
            (speechtext._respell_object, {"respell": []}, "respell"),
            (speechtext._respell_object, {"respellAll": 3}, "respellAll"),
            (speechtext._respell_object, {}, "respell"),
            (speechtext._respell_array, {"words": {}}, "words"),
            (speechtext._respell_array, {"polish": "x"}, "polish"),
            (speechtext._respell_array, {}, "polish"),
        )
        for read, payload, key in cases:
            with pytest.raises(ValueError, match=re.escape(repr(key))):
                read(payload, key)

        # A well-formed file passes both readers untouched.
        good = {"respell": {"a": "b"}, "words": ["a"]}
        assert speechtext._respell_object(good, "respell") == {"a": "b"}
        assert speechtext._respell_array(good, "words") == ["a"]


class TestRespelling:
    def test_curated_lexicon(self) -> None:
        for word, want in [
            ("download", "dałnloud"),
            ("deadline", "dedlajn"),
            ("feedback", "fidbek"),
            ("weekend", "łikend"),
            ("workflow", "łorkfloł"),
            ("github", "githab"),
            ("release", "rilis"),
        ]:
            assert lexical_respelling(word, "pl") == want, word

    def test_case_is_preserved(self) -> None:
        assert lexical_respelling("GitHub", "pl") == "Githab"
        assert lexical_respelling("Download", "pl") == "Dałnloud"

    def test_phrases_respell_as_a_unit(self) -> None:
        assert lexical_respelling("release notes", "pl") == "rilis nołc"
        assert lexical_respelling("pull request", "pl") == "pul rekłest"
        assert lexical_respelling("code review", "pl") == "koud riwju"

    def test_only_polish_is_respelled(self) -> None:
        assert lexical_respelling("download", "en") == "download"

    def test_numbers_become_cardinals(self) -> None:
        assert lexical_respelling("0", "pl") == "zero"
        assert lexical_respelling("1", "pl") == "jeden"
        assert lexical_respelling("15", "pl") == "piętnaście"
        assert lexical_respelling("101", "pl") == "sto jeden"
        assert lexical_respelling("1234", "pl") == "tysiąc dwieście trzydzieści cztery"

    def test_decimals_read_whole_comma_fraction(self) -> None:
        assert lexical_respelling("2.5", "pl") == "dwa przecinek pięć"

    def test_acronyms_are_spelled_earlier_in_the_funnel_now(self) -> None:
        """The respeller no longer owns this decision.

        It saw one word at a time, so it could not tell an initialism from a
        shout and spelled "TO JEST WAŻNE" letter by letter.
        `loudkit.frontend.letters.spell_acronyms` decides for all twelve languages while
        the surrounding capitals are still visible; the respeller now sees the
        already-spelled lowercase form and leaves it alone.
        """
        from loudkit.frontend.letters import spell_acronyms
        from loudkit.frontend.speechtext import speech_text

        assert spell_acronyms("GPT", "pl") == "gie-pe-te"
        assert spell_acronyms("USB", "pl") == "u-es-be"
        # word-acronyms keep their word form
        assert spell_acronyms("NASA", "pl") == "nasa"
        assert spell_acronyms("PIN", "pl") == "pin"
        # and the whole funnel still produces the Polish letter names
        assert "gie-pe-te" in speech_text("Model GPT jest dobry.", "pl")

    def test_english_word_alone_stays_polish(self) -> None:
        # gated out of the lexicon by the frequency gate
        assert lexical_respelling("brown", "pl") == "brown"

    def test_english_span_transliterates(self) -> None:
        # inside a 4+ word span the gate is ignored
        assert lexical_respelling("the quick brown fox", "pl") == "da kłyk brałn faks"

    def test_inflection_via_stem(self) -> None:
        assert lexical_respelling("update", "pl") == "apdejt"
        assert lexical_respelling("updates", "pl") == "apdejc"
        # apostrophe form: the ending survives the respelling, vowel-folded
        # ("dedlajn" ends in a consonant, so "u" just appends)
        assert lexical_respelling("deadline'u", "pl") == "dedlajnu"

    def test_polish_words_are_untouched(self) -> None:
        assert lexical_respelling("temperatura", "pl") == "temperatura"
        assert lexical_respelling("piątku", "pl") == "piątku"


class TestEndToEnd:
    def test_polish_sentence_with_anglicism(self) -> None:
        assert speech_text("Pobierz download i zrób code review.", "pl") == (
            "Pobierz dałnloud i zrób koud riwju."
        )

    def test_polish_sentence_with_percent(self) -> None:
        assert speech_text("Rabat 15% na weekend!", "pl") == (
            "Rabat piętnaście procent na łikend!"
        )

    def test_english_sentence_passes_through_funnel(self) -> None:
        # the funnel is a no-op on clean prose except collapsing runs of spaces
        out = speech_text("The quick brown fox jumps over the lazy dog.", "en")
        assert "The quick brown fox jumps over the lazy dog." in out

    def test_the_symbol_respelling_pass_is_unreachable_through_the_funnel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`_respell_symbols` claims to say `przez`, `równa się` and `mniejsze
        niż`. Through `speech_text` it never runs on anything: `_speak_symbols`
        has taken `%` and `°`, `expand_numbers` has turned the digits its
        `(?<=\\d)` guards need into words, and `_punctuation_for_speech` has
        made a space of the rest. This pins the gap so that moving the pass in
        0.1.2 fails here and the docstring gets retracted with it.
        """
        from loudkit.frontend import speechtext

        aimed = [
            "5% ludzi",
            "20 °C dzisiaj",
            "3 / 4 szklanki",
            "2 * 3 razy",
            "2 ^ 8 bitów",
            "2 + 2 = 4",
            "x < y",
            "x > y",
            "5 - 3",
        ]
        # Direct: every rule fires.
        assert all(speechtext._respell_symbols(case) != case for case in aimed)

        # Through the funnel: none of them does.
        real = speechtext._respell_symbols
        fired: list[str] = []
        reached: list[str] = []

        def spy(text: str) -> str:
            reached.append(text)
            out = real(text)
            if out != text:
                fired.append(text)
            return out

        monkeypatch.setattr(speechtext, "_respell_symbols", spy)
        for case in aimed:
            speech_text(case, "pl")
        monkeypatch.undo()
        # Not vacuous: the pass runs on every case, it just has nothing left.
        assert len(reached) == len(aimed)
        assert not fired, f"reached _respell_symbols with symbols left: {fired}"

        # What the gap costs, as the render says it.
        assert speech_text("2 + 2 = 4", "pl") == "dwa dwa cztery"

    def test_polish_needs_the_lexicon(self) -> None:
        # loading the 6.5 MB lexicon lazily is what non-Polish runs avoid
        pytest.importorskip("importlib.resources")
        from loudkit.frontend.speechtext import PolishLexicon

        assert "download" in PolishLexicon.generated()


class TestSharedFunnelFixture:
    """The funnel against the fixture every implementation checks.

    Hand-written cases in five languages are five tests of five different
    things. One fixture, run through all five funnels, is one test of one
    thing: that they agree.
    """

    @staticmethod
    def _fixture() -> dict:
        import json
        from pathlib import Path

        path = Path(__file__).resolve().parent / "data" / "conformance" / "speechtext.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_every_case_matches(self) -> None:
        cases = self._fixture()["cases"]
        assert cases, "the fixture is empty; nothing was compared"
        for case in cases:
            got = speech_text(case["text"], case["language"])
            assert got == case["expected"], (
                f"{case['text']!r} / {case['language']!r}: "
                f"expected {case['expected']!r}, got {got!r}"
            )

    def test_disputed_cases_still_differ_as_recorded(self) -> None:
        """The known Swift/Python disagreements, pinned so they cannot drift.

        These are judgements about how a Polish reader says a loanword, not
        bugs with an obvious side — so they are recorded rather than resolved.
        Pinning the Python half means a silent change to either implementation
        shows up here instead of quietly making the record wrong.
        """
        for case in self._fixture()["disputed"]:
            assert speech_text(case["text"], case["language"]) == case["python"], (
                f"the recorded Python output for {case['text']!r} is stale"
            )

    def test_unicode_digits_are_never_dropped(self) -> None:
        """Losing text is not an available outcome.

        The Swift funnel returned an empty string for a Polish passage of
        Arabic-Indic numerals: `Int("١٢٣")` is nil there, and the digit-by-digit
        fallback mapped through an ASCII-keyed table with `compactMap`, which
        drops what it cannot map. Pinned on this side too, because a port
        rewriting the number path is exactly when it would come back.

        The English half used to assert the opposite — that the digits arrived
        at the model as written — and that was the divergence, not the pin. The
        number pass is ASCII by design, so eleven languages passed these through
        untouched while Polish read them, because the respeller's own digit test
        is `str.isdigit()` and that is true of every Unicode decimal digit. One
        funnel, one fingerprint, two answers. They are folded to ASCII beside
        NFC now, so all twelve say the same number.
        """
        assert speech_text("١٢٣", "pl") == speech_text("123", "pl")
        assert speech_text("١٢٣", "en") == speech_text("123", "en")
        # The separator that travels with them, and the reason the fold has to
        # know the language: U+066B is a decimal point, and in the eleven
        # comma-decimal languages folding it to a dot would make "٣٫١٤" the
        # written form of a clock time.
        assert speech_text("٣٫١٤", "en") == speech_text("3.14", "en")
        assert speech_text("٣٫١٤", "pl") == speech_text("3,14", "pl")
        assert speech_text("٣٫١٤", "de") == speech_text("3,14", "de")


class TestTheNumeralTableIsTheOnlyAuthority:
    """The fold must not move with the interpreter's Unicode version.

    `numerals.json` fixed *what* a numeral reads as and left *whether* a
    character is a numeral to `unicodedata.category`, which is 15.0.0 on Python
    3.12 and 16.0.0 on 3.14 — both in this project's CI matrix. `Add \U00010d41 cups`
    therefore read "Add cups" on one and left the digit written on the other,
    under one `algorithm_fingerprint`, and neither said "one".
    """

    def test_the_shipped_table_is_the_pinned_ucd(self) -> None:
        import json

        make_numerals = tool("make_numerals")
        provenance = json.loads(make_numerals.PROVENANCE.read_text(encoding="utf-8"))
        assert provenance["unicode_version"] == make_numerals.EXPECTED_UNICODE

    def test_the_hashed_data_files_carry_no_prose(self) -> None:
        """A sentence about the data must not be able to move the fingerprint.

        `grammar_digest` hashes these files as raw bytes, so a description
        carried inside one of them makes correcting a sentence move the
        grammar digest and the `algorithm_fingerprint` under it, for audio
        that renders exactly as before. That is the opposite of what
        `COMPATIBILITY.md` promises the fingerprint means.

        The prose lives beside the data instead: `numbers.about.md` and
        `numerals.provenance.json`, neither hashed nor copied to the ports.
        This test is what keeps it there.
        """
        import json

        from loudkit.frontend.textconfig import GRAMMAR_PATH, NUMERALS_PATH, RESPELL_PATH

        allowed = {
            GRAMMAR_PATH.name: {"version", "languages"},
            NUMERALS_PATH.name: {"decimal_zeros", "spelled"},
        }
        for path, keys in (
            (GRAMMAR_PATH, allowed[GRAMMAR_PATH.name]),
            (NUMERALS_PATH, allowed[NUMERALS_PATH.name]),
        ):
            loaded = json.loads(path.read_text(encoding="utf-8"))
            assert set(loaded) == keys, (
                f"{path.name} carries {sorted(set(loaded) - keys)} on top of the data "
                f"the funnel reads. Everything in a hashed file moves the "
                f"fingerprint; put descriptions in the sidecar beside it."
            )
        # The lexicon is a mapping of words and has never carried a header;
        # asserted so that adding one is a red test rather than a re-pin.
        lexicon = json.loads(RESPELL_PATH.read_text(encoding="utf-8"))
        assert not any(
            isinstance(value, str) and len(value) > 200 for value in lexicon.values()
        ), "pl_en_respell.json has grown a prose member"

    def test_regenerating_on_the_pinned_ucd_reproduces_the_shipped_bytes(
        self, tmp_path
    ) -> None:
        """The pin only means something if the command reproduces the file.

        Checking the declared `unicode_version` checks a string in the file
        against a string in the generator, and both are written by the same
        run: a table edited by hand, or cut by an older generator, passes that
        and this. So this one regenerates into a temp directory and compares
        the bytes -- which is also what makes the digest, the grammar hash and
        the fingerprint auditable rather than asserted.

        Runs on the interpreter carrying the pinned UCD and skips on the
        others; CI's matrix has both. The generator refuses to write under a
        different one, so a skip here is that refusal, not a gap.
        """
        import unicodedata

        make_numerals = tool("make_numerals")
        if unicodedata.unidata_version != make_numerals.EXPECTED_UNICODE:
            pytest.skip(
                f"this interpreter carries Unicode {unicodedata.unidata_version}; "
                f"the table is cut from {make_numerals.EXPECTED_UNICODE}"
            )
        shipped, shipped_provenance = make_numerals.OUT, make_numerals.PROVENANCE
        regenerated = tmp_path / "numerals.json"
        try:
            make_numerals.OUT = regenerated
            make_numerals.PROVENANCE = tmp_path / "numerals.provenance.json"
            make_numerals.main()
            fresh_provenance = make_numerals.PROVENANCE.read_bytes()
        finally:
            make_numerals.OUT, make_numerals.PROVENANCE = shipped, shipped_provenance
        assert fresh_provenance == shipped_provenance.read_bytes(), (
            "the shipped numerals.provenance.json is not what the generator writes"
        )
        assert regenerated.read_bytes() == shipped.read_bytes(), (
            "the shipped numerals.json is not what tools/make_numerals.py "
            "writes on the pinned UCD — regenerate it, and expect the grammar "
            "digest and the algorithm fingerprint to move"
        )

    def test_the_generator_refuses_an_interpreter_carrying_another_ucd(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pin is only a pin if regenerating on the wrong interpreter fails.

        Otherwise the next person to run `make_numerals.py` writes a different
        table from the same command, and the diff looks like a data update
        rather than a Unicode upgrade.
        """
        import unicodedata

        make_numerals = tool("make_numerals")
        monkeypatch.setattr(unicodedata, "unidata_version", "9.9.9")
        with pytest.raises(SystemExit, match="numerals.json is cut from"):
            make_numerals.main()

    def test_a_numeral_this_interpreter_has_never_heard_of_still_reads(self) -> None:
        """The pinned table is 16.0.0; this suite also runs on interpreters
        carrying 15.0.0, where `unicodedata` calls U+10D41 an unassigned code
        point. It reads as one either way, because nothing asks."""
        assert speech_text("Add \U00010d41 cups", "en") == "Add one cups"

    def test_a_fraction_with_no_nfkc_form_is_exact(self) -> None:
        """`unicodedata.numeric` hands back a float, and the float was printed:
        GREEK TWO THIRDS SIGN read as "zero point six six six six six seven",
        which is not the value of the symbol. It reads as the two thirds it is."""
        assert speech_text("\U00010177 of it", "en") == "two divided by three of it"

    def test_an_ideographic_numeral_is_still_a_letter(self) -> None:
        """The table names no `Lo` character, which is what keeps a language's
        own numerals out of the fold — the failure a numeric-type test causes."""
        assert speech_text("\u4e00\u4e8c\u4e09", "en") == "\u4e00\u4e8c\u4e09"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("\u2474\u00b9", "(one)one"),
            ("\u2474\u3007", "(one)zero"),
            ("\u2474\u00bd", "(one)one divided by two"),
            # \u2160 is Nl and folds to the letter it is, not to a word.
            ("\u2474\u2160", "(one)I"),
            ("\u2474\u2460", "(one)one"),
        ],
    )
    def test_a_numeral_does_not_separate_itself_from_a_numeral(
        self, text: str, expected: str
    ) -> None:
        r"""The spacing guard is ``\p{L}`` or ``\p{Nd}``, not ``str.isalnum()``.

        ``isalnum()`` is also true for Nl and No -- Roman numerals, circled
        digits, superscripts, vulgar fractions -- which are exactly the
        characters about to be replaced, so it asked whether a numeral needs
        separating from a numeral and answered yes. The four ports test the
        word class and answer no, and this guard was the only thing the Python
        and Go funnels disagreed about: over a 51766-row dump across all twelve
        languages, 6190 disagreements before and none after.
        """
        assert speech_text(text, "en") == expected

    def test_a_letter_or_a_decimal_digit_still_separates(self) -> None:
        """The other half of the guard, so the cases above can fail."""
        assert speech_text("\u2474a", "en") == "(one) a"
        assert speech_text("a\u2474", "en") == "a (one)"
        assert speech_text("\u24747", "en") == "(one) seven"


class TestThePolishRulesAreShippedData:
    """The curated Polish tables must reach the funnel from the packaged file.

    They moved out of `speechtext.py` and into `pl_respell_rules.json`, and a
    data file that fails to ship degrades Polish silently rather than loudly:
    every curated lookup misses, every anglicism falls back to the generated
    long tail or to Polish grapheme rules, and nothing raises. So the file is
    asserted present, whole, and actually consulted.
    """

    def test_the_file_ships_beside_the_other_funnel_data(self) -> None:
        import json

        from loudkit.frontend.textconfig import PL_RULES_PATH

        assert PL_RULES_PATH.is_file(), (
            f"{PL_RULES_PATH.name} is missing, and Polish respelling degrades "
            f"without raising when it is"
        )
        loaded = json.loads(PL_RULES_PATH.read_text(encoding="utf-8"))
        assert set(loaded) == {
            "version",
            "phrases",
            "lexicon",
            "keep_polish",
            "function_words",
            "endings",
        }

    def test_every_table_reaches_the_funnel(self) -> None:
        """One input per member, each reading differently if its table is
        empty. A shape check alone would pass a file nothing reads."""
        # phrases: respelled whole, before the word pass.
        assert lexical_respelling("release notes", "pl") == "rilis no\u0142c"
        # lexicon: the curated form.
        assert lexical_respelling("youtube", "pl") == "jutjub"
        # keep_polish: an English word that is also an everyday Polish one.
        assert lexical_respelling("notes", "pl") == "notes"
        # endings: the stem is found under a Polish case ending.
        assert lexical_respelling("deadline'u", "pl") == "dedlajnu"
        # function_words: never a member of an English span.
        assert lexical_respelling("i", "pl") == "i"

    def test_an_edit_here_cannot_pass_unannounced(self) -> None:
        """The guard the one unhashed funnel file has until it joins the digest.

        `grammar_digest` hashes the other three, so editing any of them moves
        the fingerprint and the ports re-pin. This file sits outside that hash:
        rewriting one curated entry changes the spoken word while the digest
        stays exactly where it was, which is the failure the digest exists to
        prevent. Adding the path to `grammar_digest` is the real fix and it
        moves the fingerprint, so it waits for a release allowed to move it.

        Meanwhile the bump is something a person has to remember, and this is
        what remembers for them.
        """
        import hashlib

        from loudkit.frontend.textconfig import PL_RULES_PATH

        pinned = "eeaf3912c9be4810b2f3abb5817cb5f82cc62687ec7d433459ed8e881e1c1a2c"
        digest = hashlib.sha256(PL_RULES_PATH.read_bytes()).hexdigest()
        assert digest == pinned, (
            f"{PL_RULES_PATH.name} changed and the fingerprint did not follow, "
            f"because this file is outside `grammar_digest`. Either bump "
            f"`FUNNEL_PORTED` and re-pin this hash to {digest}, or close the gap "
            f"by hashing `PL_RULES_PATH` too and letting the fingerprint move."
        )


class TestWhichLanguageSpeaksTheSymbols:
    """Which language a symbol is read in is one question, asked once.

    The choice used to be made by probing the grammar for a percent sign,
    which answers a different question: it is true of every language today
    only because every grammar happens to carry that one row, and a language
    added without it would have been read in English under its own name.
    """

    def test_an_unknown_language_falls_back_to_english(self) -> None:
        assert "degrees" in _speak_symbols("21°", "xx")
        assert "degrees" in _speak_symbols("21°", None)

    def test_a_supported_language_is_used(self) -> None:
        assert "Grad" in _speak_symbols("21°", "de")
        assert "stopni" in _speak_symbols("21°", "pl")

    def test_the_choice_does_not_hang_on_one_row(self, monkeypatch) -> None:
        """With the percent row gone, German is still German."""
        from loudkit.frontend import speechtext

        real = speechtext.unit_word
        monkeypatch.setattr(
            speechtext,
            "unit_word",
            lambda symbol, language: None if symbol == "%" else real(symbol, language),
        )
        assert "Grad" in _speak_symbols("21°", "de")

    def test_a_symbol_the_grammar_does_not_cover_is_left_written(self, monkeypatch) -> None:
        """No English fallback per symbol: the row is this language's or it is none.

        The four ports read the English row when their own language had no
        wording, so a grammar with a hole in it spoke the wrong language rather
        than saying nothing. Every language covers every symbol today, which is
        what makes this the pin: nothing else reaches the branch.
        """
        from loudkit.frontend import speechtext

        real = speechtext.unit_word
        monkeypatch.setattr(
            speechtext,
            "unit_word",
            lambda symbol, language: None if symbol == "≈" else real(symbol, language),
        )
        assert _speak_symbols("alfa ≈ beta", "de") == "alfa ≈ beta"
        # A mark is not a grammar row, so cutting the table does not reach it.
        # The blank runs this pass leaves are the funnel's to collapse later.
        assert _speak_symbols("alfa • beta", "de") == "alfa ,  beta"


class TestTheReadingsWrittenByHand:
    """Expectations written from what each language says, then checked here.

    Not captured from this implementation's output. A fixture that records what
    the code does proves only that the code is deterministic, and the funnel
    fixture's own generator writes ``speech_text(text, language)`` into every
    expectation, so a defect it holds is pinned as the contract five ports
    assert against. These readings are the answer first; the assertion is the
    check that the code arrived at it.
    """

    def test_the_ascii_comparison_operators_are_spoken(self) -> None:
        # Deleted, silently, in ordinary technical prose: "if latency two
        # hundred ms" states the opposite condition as readily as the one
        # written. The Unicode spellings were given words and the ASCII ones
        # people type were not.
        said = speech_text("if latency > 200 ms", "en")
        assert said == "if latency greater than two hundred ms"
        assert speech_text("x <= 10 and y >= 3", "en") == "x at most ten and y at least three"
        assert speech_text("a < b", "en") == "a less than b"
        assert speech_text("a != b", "en") == "a not equal to b"
        assert speech_text("a == b", "en") == "a equals b"
        # A language that is not the fallback, reading its own words. The lone
        # `x` is the Polish respeller's, and is what Polish calls that letter.
        assert speech_text("x > 3", "pl") == "eks większe niż trzy"

    def test_the_ascii_spellings_agree_with_the_unicode_ones(self) -> None:
        for ascii_form, unicode_form in (("<=", "≤"), (">=", "≥"), ("!=", "≠")):
            for language in ("en", "de", "pl", "fi"):
                assert speech_text(f"x {ascii_form} 10", language) == speech_text(
                    f"x {unicode_form} 10", language
                )

    def test_a_mark_without_space_around_it_stays_written(self) -> None:
        """The spacing is the evidence. Without it the mark is markup, an
        emoticon or a glued token, and prose that reads correctly today would
        gain a word it never had."""
        assert speech_text("i <3 u", "en") == "i three u"
        assert speech_text("a<b", "en") == "a b"

    def test_a_markup_tag_is_not_read_aloud(self) -> None:
        # The tag name reached the model as a word, and a comment left its `!`
        # behind as a sentence-final exclamation.
        assert speech_text("<p>Hello</p>", "en") == "Hello"
        assert speech_text('<div class="x">Hi</div>', "en") == "Hi"
        assert speech_text("<!-- note --> text", "en") == "text"
        # A space and not nothing: two block tags meeting are two paragraphs.
        assert speech_text("<p>Hello</p><p>World</p>", "en") == "Hello World"

    def test_a_date_does_not_repeat_an_article_the_sentence_has(self) -> None:
        # "See the the third of April minutes" is a stammer: the noun phrase's
        # determiner is already written.
        assert speech_text("See the 3 April minutes", "en") == "See the third of April minutes"
        assert speech_text("The 3 April deadline", "en") == "The third of April deadline"
        # ...and the article is still supplied where the sentence has none.
        assert speech_text("on 3 April", "en") == "on the third of April"

    def test_a_roman_numeral_is_a_number(self) -> None:
        # "Chapter eye-vee" inside a book reader.
        assert speech_text("Chapter IV", "en") == "Chapter four"
        assert speech_text("World War II", "en") == "World War two"
        assert speech_text("Act III, Scene II", "en") == "Act three, Scene two"
        assert speech_text("Rozdział XIV", "pl") == "Rozdział czternaście"
        # The Unicode numeral folds to the letters and reads the same.
        assert speech_text("Chapter Ⅳ", "en") == "Chapter four"

    def test_the_initialisms_that_are_also_numerals_are_left_alone(self) -> None:
        """L, C, D and M are outside the alphabet this pass reads, and that is
        the whole rule: every two-letter initialism that is also a valid Roman
        numeral needs one of them, and so does MIX."""
        assert speech_text("Buy a CD", "en") == "Buy a see-dee"
        assert speech_text("send my CV", "en") == "send my see-vee"
        assert speech_text("the MC said", "en") == "the em-see said"
        assert speech_text("size XL", "en") == "size ex-el"
        assert speech_text("Washington DC", "en") == "Washington dee-see"
        assert speech_text("MIX tape", "en") == "em-eye-ex tape"
        # A spelling no numeral has is not a numeral, and what happens to it
        # next is the acronym pass's decision, not this one's.
        assert speech_text("XIIX", "en") == "ex-eye-eye-ex"
        # The English pronoun is one letter, and one letter is not a numeral.
        assert speech_text("I am here", "en") == "I am here"

    def test_a_clock_time_carries_its_seconds(self) -> None:
        # The colon reached the acoustic model as a colon.
        assert speech_text("Split at 10:30:45 exactly.", "en") == (
            "Split at ten thirty forty-five exactly."
        )
        # A zero minute is spoken where seconds follow it, or the seconds move
        # into the minutes' place.
        assert speech_text("10:00:45", "en") == "ten zero forty-five"
        # ...and a zero seconds field says nothing the minute has not said.
        assert speech_text("10:30:00", "en") == speech_text("10:30", "en")
        # A dotted run is a version string as readily as a timestamp.
        assert speech_text("1.2.3", "en") == "1.2.3"
        assert speech_text("10.30.45", "sv") == "10.30.45"

    def test_a_vulgar_fraction_is_a_fraction(self) -> None:
        # `½` read as "one two", which is two numbers and not a value.
        said = speech_text("Add ½ cup of flour.", "en")
        assert said == "Add one divided by two cup of flour."
        assert speech_text("¾", "de") == "drei geteilt durch vier"

    def test_a_typed_slash_is_left_alone(self) -> None:
        """The character asserts the fraction; a typed slash does not. `24/7`
        is ordinary English prose, `4/7` is a date in half the world, and both
        read correctly today."""
        assert speech_text("open 24/7", "en") == "open twenty-four seven"
        assert speech_text("and/or", "en") == "and or"
        assert speech_text("3/12/2026", "en") == "three twelve two thousand and twenty-six"

    def test_a_price_says_its_magnitude_before_its_currency(self) -> None:
        # "two point five dollarsM": the suffix was left inside the word.
        assert speech_text("$2.5M", "en") == "two point five million dollars"
        assert speech_text("$5m", "en") == "five million dollars"
        assert speech_text("$5bn", "en") == "five billion dollars"
        assert speech_text("$20k", "en") == "twenty thousand dollars"
        assert speech_text("£3m", "en") == "three million pounds"
        # The written noun is the same defect wearing a longer word.
        assert speech_text("$5 million in funding", "en") == "five million dollars in funding"
        assert speech_text("€2 milliarder", "no") == "to milliarder euro"

    def test_a_letter_that_is_not_a_magnitude_leaves_the_price_as_it_was(self) -> None:
        assert speech_text("$5kg", "en") == "five dollarskg"
        # No currency mark, no magnitude: a bare `5m` is five metres as readily.
        assert speech_text("2.5M users", "en") == "2.5M users"

    def test_an_iso_datetime_does_not_keep_its_separator(self) -> None:
        # "twenty twenty-sixTten": the field separator glued to the year.
        assert speech_text("2026-03-04T10:00", "en") == "fourth March twenty twenty-six ten"
        assert speech_text("2026-03-04T10:30:45", "en") == (
            "fourth March twenty twenty-six ten thirty forty-five"
        )
        # A date with nothing after it is untouched.
        assert speech_text("2026-03-04", "en") == "fourth March twenty twenty-six"

    def test_a_date_written_against_a_letter_is_not_a_date(self) -> None:
        """A run that continues into a letter is an identifier. Both ends, and
        all three numeric forms: `25/03/2026x` is no more a date than
        `x25/03/2026` is, and the guard that named only the digits and the
        separators admitted the letter at either end."""
        # "â¦twenty twenty-sixx": a year welded to the letter behind it.
        assert speech_text("2026-03-04x", "en") == "2026-03-04x"
        assert speech_text("x2026-03-04", "en") == "x2026-03-04"
        assert speech_text("x04.03.2026", "de") == "x04.03.2026"
        # The slashed form keeps no date reading either. Its digits are still
        # read as the numbers they are: a slash is not glue, which is what
        # leaves `24/7` and `and/or` alone.
        assert "März" not in speech_text("25/03/2026x", "de")
        # ...and a date with nothing glued to it is still a date.
        assert speech_text("2026-03-04", "en") == "fourth March twenty twenty-six"
        assert speech_text("04.03.2026", "de") == "vierte März zweitausendsechsundzwanzig"
        assert (
            speech_text("25/03/2026", "de")
            == "fünfundzwanzigste März zweitausendsechsundzwanzig"
        )

    def test_a_telephone_number_written_against_a_letter_is_not_one(self) -> None:
        """The same rule, and the same reason: `+12345678abc` is an identifier,
        and half-expanding it said eight digits and then left the letters
        glued to the last of them."""
        assert speech_text("+12345678abc", "en") == "12345678abc"
        assert speech_text("abc+12345678", "en") == "abc+12345678"
        # A number with nothing glued to it is still read digit by digit.
        assert speech_text("+12345678", "en") == "one two three four five six seven eight"
        assert speech_text("+48 123 456 789", "pl") == (
            "cztery osiem jeden dwa trzy cztery pięć sześć siedem osiem dziewięć"
        )

    def test_a_bare_year_reads_as_a_cardinal_in_all_twelve(self) -> None:
        """No language here has a bare-year rule, and English is not an
        exception to one. `1776` inside a written date reads as a year in the
        six languages whose grammar says so; standing alone in a sentence it is
        a number in all twelve, because nothing in the string says otherwise.
        """
        from loudkit.frontend.numbers import cardinal, supported_languages

        for language in supported_languages():
            assert cardinal(1776, language) in speech_text("In 1776 he wrote it.", language)
