"""Generate the shared speech-funnel fixture, so five ports can be compared.

``SpeechText.prepared`` is the funnel every implementation runs before
tokenising, and the whole Polish claim rests on the ports agreeing with it
character for character. Until now that agreement was asserted by hand-written
cases in each language, twenty-odd in Python, a few in Go and Rust, and
**none at all** in the Swift funnel, which is the implementation the others
are described as ports *of*.

Hand-written cases in five languages are five different tests of five different
things. One fixture is one test of one thing, and disagreement names itself.

The cases below deliberately cover the parts of the funnel that are easy to
port slightly wrong: Unicode digits (Rust used ``is_ascii_digit`` where the
others accept ``Nd``), curly apostrophes and multi-byte suffixes (Go sliced one
*byte*), invisible characters, footnote markers, currency prefixes that become
suffixes in speech, and inflected anglicisms.

Usage:
  .venv/bin/python tools/make_speechtext_fixture.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loudkit.config import ChunkConfig
from loudkit.frontend.chunking import split_text
from loudkit.frontend.speechtext import speech_text

OUT = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "data"
    / "conformance"
    / "speechtext.json"
)

# Cases where the shipped Swift funnel and the Python port disagree and the
# right answer is a judgement about how a Polish reader says the word, not
# something to settle by picking whichever implementation was edited last.
# Recorded here so the disagreement is visible and testable rather than lost;
# see the conventions audit.
DISPUTED: list[tuple[str, str | None, str]] = [
    (
        "Mam 21 maili i 3 deadline'y.",
        "pl",
        "Python leaves 'maili'; Swift respells it to 'mejli'. Both are written "
        "in the wild; 'mejl' is the commoner Polish spelling of the loanword, "
        "which argues for Swift — but the curated lexicon is approved by ear "
        "and this needs the same treatment.",
    ),
    (
        "apostrophe",
        "pl",
        "Python's generated CMUdict tail respells 'apostrophe' to 'apastrafi'; "
        "the Swift payload does not carry the word and leaves it. A generated "
        "lexicon that differs between ports is a parity gap in the data, not "
        "in the code.",
    ),
]

CASES: list[tuple[str, str | None, str | None]] = [
    # --- the funnel, language-independent -------------------------------
    ("Hello there.", "en", None),
    ("Zero\u200bwidth\u200cand\ufeffsoft\u00adhyphen", "en", None),
    ("A footnote[12] and a range[3–5] marker.", "en", None),
    (
        "temperature 21°, 50% done, ≈ 3 items",
        "en",
        "digits are said as words now: the number verbalizer is wired into all "
        "five funnels in one commit",
    ),
    (
        "$5 and £250, and €10 too",
        "en",
        "digits are said as words now: the number verbalizer is wired into all "
        "five funnels in one commit",
    ),
    ("a → b • c ✓ d ✗", "en", None),
    ("well-known in-word hyphen", "en", None),
    (
        "ranges 1-5 and fractions 3/4 and times 10:30",
        "en",
        "digits are said as words now: the number verbalizer is wired into all "
        "five funnels in one commit",
    ),
    (
        "decimal 0.49 and thousands 1,000",
        "en",
        "digits are said as words now: the number verbalizer is wired into all "
        "five funnels in one commit",
    ),
    (
        "keep? these! marks; and: those…",
        "en",
        "a run of clause marks collapses to one; the pair rule this fixture "
        'recorded left ".." on one pass and "." on the next',
    ),
    ("drop|these\\slashes*and+plus=signs", "en", None),
    ("  collapse   the    spaces  ", "en", None),
    # `H.mm` is a clock time in the eleven languages that write decimals with a
    # comma, and a decimal in the one that does not. All of these read as the
    # clock before, in all five implementations, and "decimal 0.49" above had
    # pinned the wrong answer as truth, which is how five ports agreed on it.
    ("Pi equals 3.14 exactly.", "en", None),
    ("It costs $0.49 today.", "en", None),
    ("Termin um 14.30 Uhr.", "de", None),
    # A combining mark that cannot compose into its base character. NFC leaves
    # it standing, so it reaches the punctuation pass, where Foundation's
    # `CharacterSet.letters` is documented as L* *and M\u002A* and answered
    # "letter", while `str.isalpha()` and the other three answered "not". Swift
    # kept it glued to the word and the rest turned it into a space: one funnel,
    # one fingerprint, two readings. Nothing in the fixture had a mark like this.
    ("Az\u032cb w\u032c x.", "en", None),
    ("Az\u032cb w\u032c x.", "pl", None),
    # The half-expansion family the right-hand guard did not close. The
    # fraction group `(?:[.,][0-9]+)*` may match zero times, and the regex will
    # shrink it to zero so the guard lands on a dot rather than a letter:
    # "1.5e3" matched just the "1" and read "one.5e3", "3.14abc" read
    # "three.14abc". And an exponent's sign hid the letter from the backward
    # walk, so "1e-3" read "1e-three", then, once the number pass declined it,
    # punctuation took the token apart into "1e 3".
    ("The value is 1.5e3.", "en", None),
    ("The value is 1e-3.", "en", None),
    ("It is 3.14abc here.", "en", None),
    # ...without disturbing the three that must still read.
    ("Pi equals 3.14.", "en", None),
    ("It is -5 degrees.", "en", None),
    ("A well-known fact.", "en", None),
    # The Polish respeller is a second, independent reader of digits, and it
    # has a rule for dotted runs, "three or more is a version, an address or a
    # date, and is left exactly as written". The rule only started on a
    # pure-digit group, so "v1.2.3" collected as ["v1", "2", "3"], never became
    # a run, and came out "fał jeden.dwa przecinek trzy". A version is a version
    # whether or not its first group happens to be all digits.
    ("Wersja v1.2.3 juz jest.", "pl", None),
    ("Wersja 1.2.3 juz jest.", "pl", None),
    ("Mam R2 tutaj.", "pl", None),
    ("Cena 2.5 tutaj.", "pl", None),
    # A *grouped* digit run glued to a letter. `x200 000` binds as one match,
    # so a non-backtracking engine lets the lookbehind refuse the whole run
    # and leaves the token written; a backtracking one can fall back to the
    # standalone `000` and read "x200 zero zero zero". Half a token spoken is
    # the class the right-hand guard exists to stop, so all five leave the
    # token written.
    ("x200 000 y", "en", None),
    ("a1 000 000 b", "en", None),
    ("200 000x here", "en", None),
    # ...and the grouped runs that must still read, which a careless version of
    # this would take with it.
    ("Sold 200 000 units.", "en", None),
    ("In 2024 200 people came.", "en", None),
    # An exponent's plus, which the number pass declines and punctuation then
    # took apart into "1e 3", only the hyphen was kept between alphanumerics.
    ("The value is 1e+3.", "en", None),
    ("Value 2.5E+1 here.", "en", None),
    # Currency behind the amount, which the prefix rule could not see. `2.50 €`
    # and `0.49¢` are prices by exactly the evidence `€2.50` is, and reached the
    # time pass with the dot intact: German answered "zwei Uhr fünfzig Euro".
    ("It costs 0.49¢.", "de", None),
    ("It costs 0.49¢.", "en", None),
    ("Es kostet 2.50 €.", "de", None),
    # Ordinary text that aborts a port whose ragged-run branch moves the
    # cursor past its own match: the next match arrives from before it, and
    # slicing backwards panics.
    #
    # Only the shape all five agree on. Two neighbouring ones,
    # "200 000.200 000!" and "121 euros 234 567 5 000", are read differently
    # by Python and the RE2 ports. That family is open, and a fixture case is
    # a claim that five implementations agree, so those two are pinned as
    # crash regressions in `rust/tests/speechtext.rs` instead.
    ("1 234 567 12.", "fr", None),
    # The code speller is all or nothing. Deleting a character it has no name
    # for reads `Müller123` as *em el el e er jeden dwa* with the `ü` simply
    # gone, and truncating at the length cap loses the last digits of
    # `żelazny2024`. A token half-spelled is worse than one left written,
    # because the listener cannot tell anything was dropped.
    ("Mam Müller123 tutaj.", "pl", None),
    ("Mam żelazny2024 tutaj.", "pl", None),
    ("Mam R2 tutaj.", "pl", None),
    ("Mam iOS18 tutaj.", "pl", None),
    # `R$` is the Brazilian real and this table has a wording for no
    # multi-character mark, so matching the `$` alone said "Dollar", the wrong
    # currency, confidently, with the orphaned `R` still in front of it. The
    # mark is dropped and the amount reads as a decimal now, which is a smaller
    # lie than naming the wrong money.
    ("Kosztuje R$3,14.", "de", None),
    ("Es kostet $2.50.", "de", None),
    # A number followed by an unrelated token across a space. The forward walk
    # crossed a thousands space whenever a digit followed, so it walked out of
    # `1000`, found the `e` of an exponent two tokens later, and refused the
    # whole thing. Four of the fuzzer's Go divergences were this, with Go right.
    ("Son 1000 5.1e+3 aqui.", "es", None),
    ("Cena 2,50 1e6 tutaj.", "pl", None),
    ("Value 3.14 200 000 here.", "en", None),
    # The four shapes that pinned down where a backward walk may cross a
    # thousands space. Each of the first three versions of that rule read one of
    # them wrong: `R2 5` is not a grouped number, `a1 000 000`'s first group is
    # legitimately one digit, and `Sold 200 000` reached "Sold" when the walk was
    # allowed to cross space after space.
    ("R2 5 iOS.", "de", None),
    ("a1 000 000 b", "en", None),
    ("Sold 200 000 units.", "en", None),
    ("x1 000 000 y", "en", None),
    ("Spotkanie o 14.30 dzisiaj.", "pl", None),
    ("Meeting at 14:30 today.", "en", None),
    # A plus in front of digits is E.164 and never a grouped thousand. Read as a
    # cardinal, "+48 123 456 789" is forty-eight billion; the ragged
    # "1 202 555 0199" was worse, matching only as far as it fit and leaving a
    # bare "9" behind. The two below it must keep reading as quantities.
    ("Call +48 123 456 789 now.", "en", None),
    ("Dial 1 202 555 0199 now.", "en", None),
    ("Gained +1 000 000 users.", "en", None),
    ("Sold 200 000 units.", "en", None),
    # U+2212, the typographic minus, which unfolded vanishes into a space, taking
    # the sign of the number with it.
    ("Minus sign \u22125 here.", "en", None),
    ("Temperatura \u22125 stopni.", "pl", None),
    # A digit run glued to a letter on *either* side is part of that token.
    # Without a mirror for the lookbehind, a run touching a word on the right
    # is expanded up to the letter and then abandoned: "5x3" says *fivex3* and
    # "1e6" says *onee6*, a word welded to a digit. And a one-character
    # lookbehind cannot see past a dot between a letter and its digits, so
    # "v1.2.3" reads "v1.two point three".
    ("Value 1e6 here.", "en", None),
    ("It is 5x3 grid.", "en", None),
    ("Version v1.2.3 out.", "en", None),
    ("See SVN r123 now.", "en", None),
    # ISO 8601 writes end-of-day as 24:00. The hour was outside the pattern,
    # so both halves read as unrelated numbers and the colon stood between
    # them: "twenty-four:zero zero" reaching the model as written.
    ("At 24:00 sharp.", "en", None),
    ("At 23:59 sharp.", "en", None),
    # The clock pass wrote its words against the letters behind the digits, so
    # `3:45pm` read *three forty-fivepm*, one word to a listener, while `3:45
    # pm` read correctly. Nothing here had a meridiem in it, in any language,
    # which is how the commonest way an English caller writes a time survived
    # five implementations. The spaced form is beside each glued one because
    # the claim is that they read the same.
    ("Call at 3:45pm.", "en", None),
    ("Call at 3:45 pm.", "en", None),
    (
        "It is 9:05am now.",
        "en",
        "a minute under ten reads as a plain cardinal, *nine five*, in all "
        "twelve. English says *nine oh five* and the word for that zero is not "
        "a cardinal, so it is a per-language grammar entry rather than a "
        "boundary rule, and this case pins what the funnel says today",
    ),
    (
        "Meet at 11:30AM sharp.",
        "en",
        "the acronym speller runs before the clock pass and sees a meridiem "
        "only where a space already stood: `11:30 AM` becomes *ay-em* and "
        "`11:30AM` stays `AM`. Pass order, not the boundary, and the pair below "
        "keeps the difference visible instead of asserting one of them alone",
    ),
    ("Meet at 11:30 AM sharp.", "en", None),
    (
        "Due 12:00p.m. today.",
        "en",
        "the dotted spelling is letters like any other, and the speller "
        "declines a dotted token, so both forms of this one read alike",
    ),
    # ...and the same boundary on the one grammar with a written infix, where a
    # missing space made the reading say the infix twice.
    ("Termin um 14.30Uhr.", "de", None),
    ("Termin um 14.30 Uhr.", "de", None),
    ("Es ist 14:30Uhr genau.", "de", None),
    # A meridiem in a language that does not write one is still two words.
    ("Spotkanie o 14:30am.", "pl", None),
    # The negatives the rule must not reach: a seconds field, and a letter in
    # front of the time rather than behind it.
    ("Split at 10:30:45 exactly.", "en", None),
    ("Meet at a14:30.", "en", None),
    # Spanish scales stop at `billón`, 10^12, which is the right Spanish word
    # and raises the ceiling enough for `mil millones` to compose on its own.
    # Modelling 10^9 as its own scale word instead gives "dos mil millones
    # quinientos millones" for 2 500 000 000. Portuguese looks like the same
    # shape and is not: the CLDR corpus says `um mil milhões`, so its 10^9
    # scale word stays and this case pins the difference.
    ("Son 2 500 000 000 personas.", "es", None),
    ("São 2 500 000 000 pessoas.", "pt", None),
    # --- Unicode digits ---------------------------------------------------
    # Rust used `is_ascii_digit` here, and the Swift funnel deleted these
    # outright: `Int("١٢٣")` is nil, and the digit-by-digit fallback mapped
    # through an ASCII-keyed table with `compactMap`, which drops what it
    # cannot map. Kept as bare digits so the case tests digit handling and
    # not the surrounding lexicon.
    (
        "١٢٣",
        "en",
        "digits are said as words now: the number verbalizer is wired into all "
        "five funnels in one commit",
    ),
    ("١٢٣", "pl", None),
    # The decimal separator that goes with those digits. U+066B is not in the
    # `[.,]` the number pass looks for, so it was dropped and "٣٫١٤" read as
    # two numbers, *trzy czternaście*, the same change of meaning as reading a
    # decimal as a clock time, arriving through a character set. It folds to
    # whichever mark the language actually writes, which is why folding it to a
    # dot everywhere would have put German back on the clock.
    ("٣٫١٤", "en", None),
    ("٣٫١٤", "pl", None),
    ("٣٫١٤", "de", None),
    ("٥٪", "en", None),
    ("٥٪", "pl", None),
    # A currency mark in front of foreign digits. The five currency patterns
    # spell their digit class differently, Python and Rust `\\d` is Unicode,
    # Go and JS `\\d` is ASCII, so "$٥" was five dollars in two ports and a
    # dollar sign in front of an unread numeral in the others. Folding the
    # digits before any pattern sees them settles it without touching five
    # regexes, which is why these cases live here rather than in a note.
    ("It costs $٥ today.", "en", None),
    ("Kosztuje $٥ dzisiaj.", "pl", None),
    ("Cost £٢٥٠ now.", "en", None),
    # A price is the one dotted pair that is never a clock time. The currency
    # symbol becomes a trailing word before the time pass runs, so "$0.49"
    # reaches it looking exactly like "14.30", which in the eleven
    # comma-decimal languages is how a time is written: unguarded, German
    # answers "null Uhr neunundvierzig Dollar", a price read as an hour.
    ("It costs $0.49.", "de", None),
    ("It costs $0.49.", "pl", None),
    ("It costs $0.49.", "en", None),
    ("Es kostet $2.50.", "de", None),
    # ...and the dotted time it must not swallow along with it.
    ("Termin um 14.30 Uhr.", "de", None),
    ("123", "pl", None),
    ("0", "pl", None),
    (
        "0042",
        "pl",
        "digits are said as words now: the number verbalizer is wired into all "
        "five funnels in one commit",
    ),
    # --- Polish: the respelling half ------------------------------------
    ("download", "pl", None),
    ("Download", "pl", None),
    ("DOWNLOAD", "pl", None),
    ("download", "en", None),
    ("Zrobiłem deadline'u wczoraj.", "pl", None),
    ("Wysłałem release notes i pull request.", "pl", None),
    ("To jest 0,49 procenta.", "pl", None),
    # A run of three or more digit groups is a version, an address or a
    # date, and the respeller has to leave it alone. Python measures the
    # whole run before reading any of it; Swift, Go, Rust and JS looked only
    # at the next pair, read "192.168" as a decimal and left ".0.1" trailing
    # behind it. Absent from this fixture, that divergence sat under a green
    # "exact" row: the four ports agreed with Python on every case anyone
    # had written down, and disagreed on every IP address in the world.
    ("Adres 192.168.0.1 dzisiaj.", "pl", None),
    ("Wersja 1.2.3 juz jest.", "pl", None),
    ("Maska 10.0.0.255 tutaj.", "pl", None),
    # The two-group case the rule above must not break.
    ("To 2.5 metra.", "pl", None),
    ("Sprawdź API i HTTP oraz JSON.", "pl", None),
    (
        "Kod ABC-123 do sprawdzenia.",
        "pl",
        "digits are said as words now: the number verbalizer is wired into all "
        "five funnels in one commit",
    ),
    ("Polskie słowa zostają nietknięte.", "pl", None),
    ("deadline’u", "pl", None),  # curly apostrophe: a byte-wise slice cuts it in half
    # --- language id handling -------------------------------------------
    ("download", "PL", None),
    ("download", None, None),
    (
        "Es kostet $5 und 50%.",
        "de",
        "digits are said as words now: the number verbalizer is wired into all "
        "five funnels in one commit",
    ),
    (
        "Cuesta €10, un 3%.",
        "es",
        "digits are said as words now: the number verbalizer is wired into all "
        "five funnels in one commit",
    ),
    (
        "Il fait 30° dehors.",
        "fr",
        "digits are said as words now: the number verbalizer is wired into all "
        "five funnels in one commit",
    ),
    (
        "Det koster €10.",
        "da",
        "digits are said as words now: the number verbalizer is wired into all "
        "five funnels in one commit",
    ),
    (
        "Costa £250 al mese.",
        "it",
        "digits are said as words now: the number verbalizer is wired into all "
        "five funnels in one commit",
    ),
    (
        "Kost €5 per maand.",
        "nl",
        "digits are said as words now: the number verbalizer is wired into all "
        "five funnels in one commit",
    ),
    (
        "Custa $100 por mês.",
        "pt",
        "digits are said as words now: the number verbalizer is wired into all "
        "five funnels in one commit",
    ),
    (
        "Meeting at 14:30, e.g. tomorrow.",
        "en",
        "clock times and authority-listed abbreviations are spoken now",
    ),
    (
        "Es ist 14.30, z.B. morgen, usw.",
        "de",
        "clock times and authority-listed abbreviations are spoken now",
    ),
    (
        "Klockan 16.31, t.ex. fr.o.m. måndag.",
        "sv",
        "clock times and authority-listed abbreviations are spoken now",
    ),
    (
        "Kello 9.05, esim. huomenna.",
        "fi",
        "clock times and authority-listed abbreviations are spoken now",
    ),
    (
        "Spotkanie o 14:30, np. jutro, itd.",
        "pl",
        "clock times and authority-listed abbreviations are spoken now",
    ),
    (
        "Kl. 9.05, f.eks. i morgen.",
        "da",
        "clock times and authority-listed abbreviations are spoken now",
    ),
    (
        "Zaźółć gęślą jaźń.",
        "pl",
        "NFD input. The funnel's first pass composes it; without NFC the output "
        "differs from every other implementation's by combining marks alone, which "
        "no eye catches in a diff.",
    ),
    (
        "Månsken er små.",
        "da",
        "NFD input. The funnel's first pass composes it; without NFC the output "
        "differs from every other implementation's by combining marks alone, which "
        "no eye catches in a diff.",
    ),
    (
        "The CIA said so.",
        "en",
        "acronyms are spelled in the render language in all twelve, by all five "
        "implementations",
    ),
    (
        "Die CIA sagte es.",
        "de",
        "acronyms are spelled in the render language in all twelve, by all five "
        "implementations",
    ),
    (
        "NASA and NATO.",
        "en",
        "acronyms are spelled in the render language in all twelve, by all five "
        "implementations",
    ),
    (
        "El FBI y la CIA.",
        "es",
        "acronyms are spelled in the render language in all twelve, by all five "
        "implementations",
    ),
    (
        "Spotkanie 12.03.2026.",
        "pl",
        "dates and ordinals run in all five implementations",
    ),
    (
        "Termin 12.03.2026.",
        "de",
        "dates and ordinals run in all five implementations",
    ),
    (
        "The 1st of May.",
        "en",
        "dates and ordinals run in all five implementations",
    ),
    (
        "iOS18 is out.",
        "en",
        "the four number-regex shapes every funnel must agree on",
    ),
    (
        "I have 1 000 things.",
        "en",
        "the four number-regex shapes every funnel must agree on",
    ),
    (
        "It is -5 degrees.",
        "en",
        "the four number-regex shapes every funnel must agree on",
    ),
    (
        "Meet at a14:30.",
        "en",
        "the four number-regex shapes every funnel must agree on",
    ),
    (
        "Price 1 234 567 exact.",
        "en",
        "the four number-regex shapes every funnel must agree on",
    ),
    # --- Norwegian ------------------------------------------------------
    #
    # A block rather than a row in each family above, because until now `no`
    # had *zero* cases here: the roster ships Norwegian voices and the fuzzer
    # generates Norwegian, but the one gating parity contract exercised the
    # grammar nowhere. These five are the same five shapes the other languages
    # are held to, currency in a sentence, a clock time beside an
    # abbreviation, a date, grouping and the decimal comma, acronyms, so a
    # Norwegian divergence now fails in the same place a Danish one does.
    (
        "Det koster 250 kroner.",
        "no",
        "digits are said as words now: the number verbalizer is wired into all "
        "five funnels in one commit",
    ),
    (
        "Kl. 9.05, f.eks. i morgen.",
        "no",
        "clock times and authority-listed abbreviations are spoken now",
    ),
    (
        "Møtet er 12.03.2026.",
        "no",
        "dates and ordinals run in all five implementations",
    ),
    (
        "Prisen er 1 234 567 kroner, altså 2,5 millioner.",
        "no",
        "space grouping and the decimal comma, which is how Norwegian writes "
        "both — the pair a port that assumes the English conventions reads as "
        "seven separate numbers and a date",
    ),
    (
        "NATO og FBI sa det.",
        "no",
        "acronyms are spelled in the render language in all twelve, by all five "
        "implementations",
    ),
    # Found by `tools/fuzz_parity.py`, all five in the same family: a port
    # reading Unicode through a word class its regex engine made ASCII, or made
    # too wide. Each is here because the fuzzer found it and nothing in the
    # fixture had the shape.
    (
        "a̬123",
        "no",
        "a combining mark is not a word character. ICU's `\\w` — which is what "
        "NSRegularExpression gives Swift — is documented as including `\\p{M}`, "
        "so the mark behind the digits looked like the tail of a word and the "
        "number stayed written while four ports read it",
    ),
    (
        "a̬CIA.",
        "it",
        "the same mark, one pass earlier: Python splits `(\\W+)` on code points "
        "and Swift was walking `Character`, which is a grapheme cluster, so "
        "`a̬CIA` was one mixed-case token rather than a lone acronym beside "
        "a letter, and never reached the speller",
    ),
    (
        "zł€ 000 000",
        "nl",
        "a letter in front of a currency mark refuses the whole amount — `R$` is "
        "the Brazilian real and this table cannot name it. The regex crate's "
        "`[:alpha:]` is ASCII even in Unicode mode, so Rust alone read `ł` as "
        "a non-letter, took the `€` for a bare euro sign and moved the first "
        "group behind it: `zł000 euro nul nul nul`, the rest stranded",
    ),
    (
        "CIA CIA",
        "pl",
        "a run of capitals is emphasis, not initialisms, and `spell_acronyms` "
        "decides that where the neighbours are still visible. The Polish "
        "respeller had its own acronym branch in Go, Rust, JS and Swift — "
        "deleted in Python, kept in four — which spelled one word at a time "
        "with no view of the run: `ce-i-a ce-i-a`",
    ),
    (
        "Preis٣٫١٤ iOS.",
        "de",
        "the mixed-script class this fuzzer was written for, now closed: foreign "
        "digits fold to ASCII, the token still has a letter glued to it, and all "
        "five leave it written. It read `Preis3,vierzehn` — half a token spoken, "
        "which is exactly what the refusal rule exists to stop",
    ),
    # --- five-way divergences, each found by differential probing ---------
    #
    # None of these had a case here, which is why each survived. Every one was
    # a real difference in what a listener hears, in a language the roster
    # ships two voices for.
    (
        "\u00bfComo estas hoy? \u00a1Que bien!",
        "es",
        "the inverted marks are prosody: they are the earliest cue a Spanish "
        "reader has that a question is coming. The reference dropped them "
        "while Go, Rust, JS and Swift kept them, so every Spanish question "
        "read differently in the implementation that defines the others",
    ),
    (
        "Bonjour\u00a0! Comment allez-vous\u00a0? Voici\u00a0: midi\u00a0;",
        "fr",
        "French typography puts a non-breaking space before ! ? : ; and Word, "
        "LibreOffice and French CMSs insert it automatically. RE2's \\s is "
        "ASCII, so Go alone kept it, and the frontend folds NBSP to a space -- "
        "79 token ids against 75 in the other four on this one sentence",
    ),
    (
        "Cena to \U0001d7e3\U0001d7e4 zlotych.",
        "pl",
        "mathematical sans-serif digits, the kind pasted out of social bios. "
        "Four Nd blocks sit back to back in this range, so walking down to "
        "find the block's zero crossed the boundary: Rust aborted the process "
        "on the unwrap, Go and JS read 12 as 232",
    ),
    (
        "one \u6f22\u5b57 two three four",
        "pl",
        "a script without case is still letters. JS tested for letters by "
        "comparing upper to lower case, which is false for CJK, Hebrew, "
        "Arabic, Hangul, Thai and Devanagari, so the word took the "
        "digits-only branch and was deleted from the sentence",
    ),
    (
        "cena to \uff10\uff11\uff12\uff13 euro",
        "pl",
        "fullwidth digits. The leading-zero refusal reads the original token, "
        "where a fullwidth zero is not an ASCII one; Swift normalised first "
        "and then refused its own normalisation, spelling four glyphs one by "
        "one where the other four said a hundred and twenty three",
    ),
    (
        "Alpha beta\u0085gamma delta",
        "en",
        "U+0085 is what CP1252 byte 0x85 becomes when text is decoded as "
        "Latin-1, which is ordinary in scraped and epub sources. It is "
        "whitespace to Python, Go, Rust and Swift and not to ECMAScript, so "
        "JS alone turned it into a space",
    ),
    # --- three places the reference deviated from all four ports ---------
    (
        "the end of record\u001ethe next begins",
        "en",
        "U+001C-U+001F: `str.isspace()` calls them whitespace and Unicode's "
        "White_Space does not, so Python kept the separator where Rust, Go, JS "
        "and Swift all replaced it with a space. Measured over 0..0x2FFFF, "
        "these four are the only disagreement between the two predicates. The "
        "tokenizer saw [UNK] where the ports saw [SPACE], and the splitter "
        "found no word boundary where they did — different tokens and a "
        "different chunk split under one fingerprint",
    ),
    (
        "tak \u2019 nie",
        "pl",
        "a token with no letters and no digits was mapped through the "
        "ASCII-keyed digit table and filtered to nothing, so a spaced "
        "apostrophe was erased from the utterance — and it is in the prosodic "
        "set, which is to say the funnel is meant to keep it. Swift had "
        "already fixed this and written down why",
    ),
    (
        "\u00b29",
        "en",
        "a superscript written against a digit left a BARE DIGIT in the "
        'output. Every "is this a word character" test in the five ports is '
        "\\w, \\p{N} or str.isdigit(), and all three admit No — so this was one "
        "token, the number matcher's boundary guard declined it, and the "
        "symbol pass then deleted the superscript and left a 9 nothing "
        "verbalised. A bare digit is what this layer exists to prevent, and a "
        "grapheme model reads one badly with no way to hear that it happened",
    ),
    (
        "9\u00b2",
        "en",
        "the same, on the other side: the character has to become a space "
        "rather than be deleted, or the digit joins whatever followed",
    ),
    (
        "\u00bd7",
        "en",
        "a vulgar fraction is the same category (No) and was the same defect",
    ),
    (
        "The area is 12 m\u00b2 and the price is $9.",
        "en",
        "the shape this reaches in real prose. Both numbers must be read, and "
        "the currency mark must still find its amount",
    ),
    (
        "Pok\u00f3j ma 12 m\u00b2 i kosztuje 2000 z\u0142.",
        "pl",
        "the same in Polish, where the respelling pass runs as well",
    ),
    (
        "5\u09e93",
        "en",
        "a Bengali digit inside a Latin number. Unfolded, a non-ASCII Nd "
        "passes the whole funnel untouched, so the ASCII digits around it "
        "reach the model bare, and a funnel whose boundary test is ASCII "
        "reads them aloud instead: one string, a reading per implementation",
    ),
    (
        "\uff11\uff12\uff13",
        "en",
        "fullwidth digits, which are ordinary in CJK text and were not read at all",
    ),
    (
        "Add \u00bd cup of flour.",
        "en",
        "a vulgar fraction in a recipe. Unfolded it is deleted -- 'add cup of "
        "flour' -- which is the failure this layer exists to prevent, arriving "
        "as fluent audio that says something else. It reads as its ASCII "
        "spelling reads, which is the rule for every numeral this pass folds",
    ),
    (
        "Chapter \u2166.",
        "en",
        "a Roman numeral, deleted the same way. NFKC gives the letters and the "
        "acronym pass spells them, exactly as it would for a written IV",
    ),
    (
        "x\u24d0y",
        "en",
        "a circled letter is So, not a letter. Rust and Swift tested the "
        "Alphabetic property here and kept it; the other three spaced it",
    ),
    (
        "\u24d0-1",
        "en",
        "the same class in front of a signed number, and the sharper case: "
        "Swift's Alphabetic test made the boundary fail and left a BARE DIGIT "
        "in front of the model, which is what the numeral fold exists to stop",
    ),
    (
        "1 000.\U00017000",
        "no",
        "a grouped number, a dot, and a TANGUT IDEOGRAPH. Swift's walks read one "
        "UTF-16 unit and turned half a surrogate pair into a SPACE, so the "
        "number looked unglued and Swift alone read it aloud where the other "
        "four left it written",
    ),
    (
        "\U00017000.1",
        "no",
        "the same pair, the other way round: the letter is BEHIND the digit, "
        "and the backward walk stepped into the middle of the pair",
    ),
    (
        "24:00.\u4e00\u4e8c\u4e09",
        "no",
        "a clock, a dot, and CJK numerals. `Character.isNumber` is Numeric_Type, "
        "so \u4e00 (category Lo, value 1) counted as a digit behind the dot, the "
        "time looked like part of a longer number, and Swift alone read "
        "'tjuefire:00' -- half the clock spoken and half left written",
    ),
    (
        "1 000 \u4e00\u4e8c\u4e09",
        "no",
        "the same class at the other end: the CJK numerals made the grouped "
        "thousand look glued to a number rather than followed by a word",
    ),
    (
        "\u4e00\u4e8c\u4e09.2024",
        "pl",
        "the respelling pass normalised digits with `Character.isNumber`, which "
        "is Numeric_Type: the three CJK ideographs have values 1, 2 and 3, so "
        "Swift alone read the word as 'sto dwadziescia trzy' -- a number said "
        "aloud where the other four leave a word written",
    ),
    (
        "12.03.2026\u24d0",
        "pt",
        "a date followed by a CIRCLED LATIN SMALL LETTER A. ICU's \\b counts "
        "Other_Alphabetic as a word character and Python's does not, so the "
        "date found no boundary here and Swift alone left it written",
    ),
    (
        "Add \U00010d41 cups",
        "en",
        "GARAY DIGIT ONE, added in Unicode 16.0. Detection driven by the "
        "runtime's own tables, with the fold table coming from a file, reads "
        "'Add cups' on an interpreter carrying Unicode 15 and leaves it "
        "written on one carrying Unicode 16: two readings under one "
        "fingerprint, from two interpreters the CI matrix both runs. The "
        "table decides membership so that neither can",
    ),
    (
        "\U00010177 of it",
        "en",
        "GREEK TWO THIRDS SIGN has no NFKC form, so the fold reads its numeric "
        "value -- which arrived from `unicodedata.numeric` as the float "
        "0.6666666666666666 and was written out as 0.666667, read aloud in full "
        "as 'zero point six six six six six seven'. The table carries the exact "
        "fraction now",
    ),
    (
        "a\u0345b",
        "en",
        "COMBINING GREEK YPOGEGRAMMENI is Other_Alphabetic, so Rust's Alphabetic "
        "test kept a mark the other four removed",
    ),
    (
        "text[1,\u00a02] more",
        "en",
        "a footnote marker whose separator is a non-breaking space -- ordinary "
        "French and German typography. RE2 reads \\s and \\d as ASCII, so Go kept "
        "the marker and then READ IT ALOUD; the other four dropped it",
    ),
    (
        "a\u0085.",
        "en",
        "NEL before a clause mark. ECMAScript \\s excludes U+0085, so JS alone "
        "left it standing; it is ordinary in scraped and epub sources",
    ),
    (
        "a\u000b.",
        "en",
        "VERTICAL TAB, the mirror image: Go's hand-expanded whitespace class "
        "omitted it and Go alone left it standing",
    ),
    (
        "12.03.2026\u4e00\u4e8c\u4e09",
        "de",
        "a date glued to a CJK ideograph. Go's byte-wise ASCII boundary and "
        "ECMAScript's ASCII `\\b` both read that as a word boundary and spoke "
        "the date; Python, Rust and Swift read `Lo` as a word character and "
        "left it written. One string, two readings, one fingerprint",
    ),
    (
        "1\u00e9",
        "en",
        "the mirror: a digit glued to a non-ASCII letter, which the same three "
        "boundaries answered differently",
    ),
    (
        "Add \u1372 cups",
        "en",
        "ETHIOPIC NUMBER TEN has no compatibility decomposition, so an NFKC-only "
        "fold left it unchanged and the punctuation pass then DELETED it: 'add "
        "cups'. The Aegean numbers, the Kaktovik digits and the Meroitic "
        "numerals were the same. The table carries a numeric value where NFKC "
        "has nothing",
    ),
    (
        "Cena to \U0001d7e3\U0001d7e4 zlotych.",
        "pl",
        "MATHEMATICAL SANS-SERIF ONE and TWO. Decimal blocks are contiguous "
        "with each other, so walking down to 'the previous character is not a "
        "digit' walked out of the block and read these as ninety-nine. This "
        "fixture pinned that wrong answer",
    ),
    (
        "Add \U0001d7d8 cups",
        "en",
        "the sharpest case of the same bug: MATHEMATICAL DOUBLE-STRUCK DIGIT "
        "ZERO sits immediately after MATHEMATICAL SANS-SERIF DIGIT NINE, so a "
        "walk read a zero as a nine",
    ),
    (
        "\u4e00\u4e8c\u4e09",
        "en",
        "the guard on the fix. Ideographic numerals are Numeric_Type but "
        "category Lo — letters — so a pass keyed on str.isnumeric() or Swift's "
        "numericType would have replaced a language's numerals with spaces. "
        "The pass reads the general category, and these survive untouched",
    ),
    # --- what the funnel says, and what it must go on saying -------------
    # Every expectation below was written from what the language says and then
    # checked against the code, not read off it. `tests/test_speechtext.py`
    # holds the same readings as assertions, which is where they can be read
    # without a generator standing between them and the reader.
    (
        "if latency > 200 ms",
        "en",
        "the comparison operators were deleted, silently, in ordinary technical "
        "prose: the Unicode spellings had words and the ASCII ones people type "
        "did not",
    ),
    ("x <= 10 and y >= 3", "en", None),
    ("a < b, a != b, a == b", "en", None),
    (
        "x > 3 i x <= 9",
        "pl",
        "the operator words come from the grammar, so a render speaks them in its own language",
    ),
    (
        "i <3 u and a<b",
        "en",
        "the guard: whitespace on both sides is the evidence that a mark is an "
        "operator, and without it the mark stays written",
    ),
    (
        '<p>Hello</p><div class="x">Hi</div>',
        "en",
        "a tag name reached the model as a word. A tag becomes a space and not "
        "nothing, because two block tags meeting are two paragraphs",
    ),
    ("<!-- note --> text", "en", "a comment left its ! behind as a sentence-final stop"),
    (
        "See the 3 April minutes",
        "en",
        "the article the date supplies is one the sentence already had",
    ),
    ("The 3 April deadline, and on 3 April.", "en", None),
    (
        "Chapter IV of World War II",
        "en",
        "a Roman numeral was spelled letter by letter: Chapter eye-vee, inside a book reader",
    ),
    ("Act III, Scene II", "en", None),
    ("Rozdział XIV", "pl", None),
    (
        "Buy a CD, send my CV, size XL, and a MIX tape",
        "en",
        "the guard: L, C, D and M are outside the alphabet the numeral pass "
        "reads, which is what keeps every initialism that is also a numeral",
    ),
    (
        "Split at 10:30:45 exactly.",
        "en",
        "a clock time with seconds kept its colons, so a literal colon reached "
        "the acoustic model",
    ),
    ("10:00:45 and 10:30:00", "en", None),
    ("version 10.30.45 shipped", "sv", "a dotted run is a version as readily as a time"),
    (
        "Add ½ cup and ¾ more",
        "en",
        "a vulgar fraction read as two numbers. The character asserts the "
        "fraction; a typed slash does not",
    ),
    ("¾", "de", None),
    (
        "open 24/7 and/or 3/12/2026",
        "en",
        "the guard on the fraction rule: ordinary prose that reads correctly",
    ),
    (
        "$2.5M and $5m and $5bn and $20k and £3m",
        "en",
        "the scale suffix was left inside the currency word: two point five dollarsM",
    ),
    ("$5 million in funding", "en", "the written noun is the same defect in a longer word"),
    ("€2 milliarder", "no", None),
    ("$5kg costs 2.5M users", "en", "a letter that is not a magnitude leaves the price alone"),
    (
        "2026-03-04T10:00 and 2026-03-04T10:30:45",
        "en",
        "the ISO field separator glued to the last word of the date",
    ),
    (
        "2026-03-04x and x2026-03-04",
        "en",
        "a date written against a letter is part of an identifier. The guard "
        "named the digits and the separators and admitted the letter at either "
        "end, so the year welded to it: twenty twenty-sixx",
    ),
    ("x04.03.2026 and 25/03/2026x", "de", None),
    ("04.03.2026 and 25/03/2026", "de", "...and a date with nothing glued to it is one"),
    (
        "+12345678abc",
        "en",
        "the same guard on the telephone rule: eight digits were said and the "
        "letters left glued to the last of them",
    ),
    ("+48 123 456 789 and +12345678", "pl", None),
]


# Long-form splitting: where the reader breathes. Every port must cut in the
# same places, because a different split is a different set of joins and a
# different reading.
CHUNK_CASES: list[str] = [
    "One. Two. Three.",
    "A single sentence that is comfortably shorter than one window.",
    " ".join(f"Sentence number {i} runs on for a while." for i in range(1, 12)),
    "No punctuation at all just a very long run of words that has to break "
    "somewhere and the only available boundary is a space between two of them "
    "so that is where it goes even though nobody enjoys it",
    "Clauses, separated only by commas, keep going and going and going, and "
    "the splitter should prefer the latest comma inside the budget, not the "
    "first one it happens to find while scanning, which would make chunks "
    "shorter than they need to be.",
    # A character is a code point, not a UTF-16 unit and not a byte. JS
    # indexed UTF-16 units, so it charged every one of these two characters
    # and cut surrogate pairs in half, a lone surrogate went straight to
    # frontend.encode(). Astral text is not exotic: emoji, CJK Extension B and
    # mathematical alphanumerics all live up there.
    "😀" * 40 + ". A tail sentence after the emoji run.",
    "𠀋𠀌𠀍" * 20 + ", and a clause after the ideographs, and one more.",
    # Whitespace is Unicode. Go trimmed a four-character cutset where Python
    # uses lstrip(), so an NBSP was charged against the next chunk's budget
    # and then removed from the chunk, every split after it drifted, usually
    # leaving a one-character chunk that becomes its own utterance with its
    # own derived seed. NBSP is ordinary in typeset prose.
    "x" * 20 + "\u00a0" + "y" * 40,
    "Ten\u2009tysięcy\u00a0złotych, a potem\u2028nowa linia, i koniec zdania.",
    # A period that closes a title is not a full stop. Under the old law the
    # only `. ` inside the first window was the one after "Mr", so the passage
    # opened with the seven-character chunk "But Mr.", its own utterance, its
    # own derived seed, and a token ceiling proportional to seven characters.
    # It is the only chunk in a 9920-row rendered census that hit that ceiling.
    "But Mr. Smith went home to the house on the hill where he had lived for "
    "forty years without ever once complaining about any of it at all.",
    # Two held candidates in a row, so the search has to walk back through the
    # same separator more than once before it settles on a real sentence end.
    "Alpha ends here. Mrs. Watson and Mr. Holmes then agreed on the one point "
    "that had ever really mattered to either of them at all, and said so.",
    # Held all the way down: every sentence end in the window closes a title,
    # so the split falls back to the latest comma instead.
    "Mr. Norrell, who had been waiting in the hall for the better part of an "
    "hour, said nothing at all to either of them about what he had seen there.",
    # The guard on the character in front of the match. "NASA" ends in "A",
    # and "A" is a listed initial; without the guard every word whose tail
    # spells an entry would be held.
    "The rocket that carried them up there was built by NASA. And the rest of "
    "the afternoon went by without anybody saying much about it to anyone.",
    # The half of the law that no list could do. `speech_text` maps an ellipsis
    # to `...` and then folds a run of [.,;:] to one mark, so a mid-sentence
    # ellipsis reaches the splitter as a period. It is the dominant cause in
    # Polish, which has no abbreviation cuts at all.
    "Grzeja sie i swieca. ciepłem ktore pamietaja z lata i z kazdej innej "
    "pory roku na swiecie, a potem gasna powoli i nikt juz nie pamieta.",
    # `Dr` and `St` are in the list on a second measurement rather than the
    # ten-language survey's, whose English sample contained neither. Both
    # appear here so the five implementations are held to them by behaviour
    # and not only by the list's copy in each of their sources.
    "Dr. Watson walked the length of St. James Street twice over before he "
    "found the one door he had been looking for all that long grey afternoon.",
    # A comma is followed by a lower-case word almost every time it is
    # written, so a rule that did not gate on the period would veto every
    # comma in the language. This case is byte-identical under both laws.
    "Alpha beta gamma delta, epsilon zeta eta theta, iota kappa lambda mu, "
    "nu xi omicron pi rho, sigma tau upsilon phi chi psi omega at the end.",
]

DIVERGENT_WHY = (
    "What one implementation still does differently. A family leaves this "
    "block for `cases` above as soon as every implementation agrees on it, "
    "which is where all five are held to it."
)
"""Kept as a field rather than deleted: an empty `divergent` is a claim, that
nothing is known to differ, and a reader can only tell an empty block from a
forgotten one if the block is still there."""


CHUNK_CONFIGS: list[tuple[str, ChunkConfig]] = [
    ("shipping", ChunkConfig()),
    ("tiny", ChunkConfig(max_tokens=40, prefix_tokens=6)),
    # The law before `mid_sentence_period` existed, kept in the fixture rather
    # than only in the config: a port that reads the field but ignores it
    # passes every "shipping" case, because holding is what its hardcoded
    # search already does not do. Only a case that asks for "break" separates
    # a port that implements the field from one that defaults it.
    ("break", ChunkConfig(mid_sentence_period="break")),
]


def build_payload() -> dict[str, object]:
    """The fixture, as a dict, without touching the filesystem.

    Separate from :func:`main` so the test suite can assert that the committed
    file is what this generator produces. Without that assertion the generator
    is documentation: it had silently fallen 23 cases and 35 explanations
    behind the file it claims to write, and running it as documented would have
    deleted them.
    """
    return {
        "version": 1,
        "generated_by": "tools/make_speechtext_fixture.py",
        "note": (
            "Expected output of SpeechText.prepared / speech_text for each "
            "(text, language). Every port must reproduce these exactly; a "
            "difference is a divergence, not a dialect."
        ),
        "cases": [
            {
                "text": text,
                "language": language,
                "expected": speech_text(text, language),
                # Only when there is one: `why` names the divergence a case was
                # added for, and inventing a sentence for the cases that are
                # simply coverage would bury the ones that carry a warning.
                **({"why": why} if why else {}),
            }
            for text, language, why in CASES
        ],
        "chunking": [
            {
                "config": name,
                "max_tokens": cfg.max_tokens,
                "prefix_tokens": cfg.prefix_tokens,
                "split_on": list(cfg.split_on),
                "abbreviations": list(cfg.abbreviations),
                "mid_sentence_period": cfg.mid_sentence_period,
                "text": text,
                "chunks": split_text(text, cfg),
            }
            for name, cfg in CHUNK_CONFIGS
            for text in CHUNK_CASES
        ],
        # Not asserted: recorded so a known disagreement stays visible instead
        # of being quietly removed from the suite.
        "disputed": [
            {
                "text": text,
                "language": language,
                "python": speech_text(text, language),
                "why": why,
            }
            for text, language, why in DISPUTED
        ],
        # Emitted even though it is empty, and emitted from here rather than
        # hand-written into the JSON: it was hand-written, and running this
        # generator as documented deleted it along with 23 cases and 35 of
        # their explanations. A fixture that its own generator cannot
        # reproduce is a fixture nobody dares regenerate.
        "divergent": {"why": DIVERGENT_WHY},
    }


def rendered(payload: dict[str, object]) -> str:
    """The exact bytes `main` writes, so a comparison can be byte-for-byte."""
    return json.dumps(payload, ensure_ascii=False, indent=1) + "\n"


def main() -> None:
    # `newline="\n"`: five implementations read these bytes and one of them
    # regenerating under a translating text mode would rewrite every line.
    OUT.write_text(rendered(build_payload()), encoding="utf-8", newline="\n")
    print(
        f"wrote {OUT} ({len(CASES)} funnel + "
        f"{len(CHUNK_CASES) * len(CHUNK_CONFIGS)} chunking cases)"
    )


if __name__ == "__main__":
    # In the entry point, not in `main`, which the suite calls as a function.
    # The cases are in this file and the destination is fixed, so any argument
    # is a misreading; parsing is what makes `--help` print help rather than
    # rewrite the fixture five ports are compared against.
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()
    main()
