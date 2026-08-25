"""Generate the shared speech-funnel fixture, so five ports can be compared.

``SpeechText.prepared`` is the funnel every implementation runs before
tokenising, and the whole Polish claim rests on the ports agreeing with it
character for character. Until now that agreement was asserted by hand-written
cases in each language — twenty-odd in Python, a few in Go and Rust, and
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

import json
from pathlib import Path

from loudkit.config import ChunkConfig
from loudkit.frontend.chunking import split_text
from loudkit.frontend.polish import speech_text

OUT = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "data"
    / "conformance"
    / "speechtext.json"
)

# Cases where the shipped Swift funnel and the Python port disagree and the
# right answer is a judgement about how a Polish reader says the word — not
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
    # clock before, in all five implementations — and "decimal 0.49" above had
    # pinned the wrong answer as truth, which is how five ports agreed on it.
    ("Pi equals 3.14 exactly.", "en", None),
    ("It costs $0.49 today.", "en", None),
    ("Termin um 14.30 Uhr.", "de", None),
    # A combining mark that cannot compose into its base character. NFC leaves
    # it standing, so it reaches the punctuation pass — where Foundation's
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
    # walk, so "1e-3" read "1e-three" — then, once the number pass declined it,
    # punctuation took the token apart into "1e 3".
    ("The value is 1.5e3.", "en", None),
    ("The value is 1e-3.", "en", None),
    ("It is 3.14abc here.", "en", None),
    # ...without disturbing the three that must still read.
    ("Pi equals 3.14.", "en", None),
    ("It is -5 degrees.", "en", None),
    ("A well-known fact.", "en", None),
    # The Polish respeller is a second, independent reader of digits, and it
    # has a rule for dotted runs — "three or more is a version, an address or a
    # date, and is left exactly as written". The rule only started on a
    # pure-digit group, so "v1.2.3" collected as ["v1", "2", "3"], never became
    # a run, and came out "fał jeden.dwa przecinek trzy". A version is a version
    # whether or not its first group happens to be all digits.
    ("Wersja v1.2.3 juz jest.", "pl", None),
    ("Wersja 1.2.3 juz jest.", "pl", None),
    ("Mam R2 tutaj.", "pl", None),
    ("Cena 2.5 tutaj.", "pl", None),
    # The first real parity break in four rounds of review, and the fixture's
    # blind spot is why: it had no case of a *grouped* run glued to a letter.
    # `x200 000` binds as one match in Go and Rust, whose engines do not
    # backtrack, so the lookbehind refused the whole run and the token was left
    # written — while Python, JS and Swift backtracked, matched the standalone
    # `000`, and read "x200 zero zero zero". Half a token spoken, which is the
    # class the right-hand guard exists to stop, so the three converged on the
    # two rather than the other way round.
    ("x200 000 y", "en", None),
    ("a1 000 000 b", "en", None),
    ("200 000x here", "en", None),
    # ...and the grouped runs that must still read, which a careless version of
    # this would take with it.
    ("Sold 200 000 units.", "en", None),
    ("In 2024 200 people came.", "en", None),
    # An exponent's plus, which the number pass declines and punctuation then
    # took apart into "1e 3" — only the hyphen was kept between alphanumerics.
    ("The value is 1e+3.", "en", None),
    ("Value 2.5E+1 here.", "en", None),
    # Currency behind the amount, which the prefix rule could not see. `2.50 €`
    # and `0.49¢` are prices by exactly the evidence `€2.50` is, and reached the
    # time pass with the dot intact: German answered "zwei Uhr fünfzig Euro".
    ("It costs 0.49¢.", "de", None),
    ("It costs 0.49¢.", "en", None),
    ("Es kostet 2.50 €.", "de", None),
    # Ordinary text that aborted the Go and Rust processes: the ragged-run
    # branch moves the cursor past its own match, the next match arrives from
    # before it, and slicing backwards panics. Go was fixed first and the guard
    # did not reach Rust — the second time this week a fix landed in four
    # implementations of five, and the fuzzer found both.
    # Only the one all five agree on. The other two the fuzzer found —
    # "200 000.200 000!" and "121 euros 234 567 5 000" — also used to abort the
    # process and no longer do, but Python and the RE2 ports still read them
    # differently: that is the open family, and a fixture case is a claim that
    # five implementations agree. They are pinned as crash regressions in
    # `rust/tests/speechtext.rs` instead, which is what they are.
    ("1 234 567 12.", "fr", None),
    # The code speller deleted characters it had no name for and truncated at
    # eight: `Müller123` read *em el el e er jeden dwa* with the `ü` simply
    # gone, and `żelazny2024` lost its last digits. All or nothing now — a
    # token half-spelled is worse than one left written, because the listener
    # cannot tell anything was dropped.
    ("Mam Müller123 tutaj.", "pl", None),
    ("Mam żelazny2024 tutaj.", "pl", None),
    ("Mam R2 tutaj.", "pl", None),
    ("Mam iOS18 tutaj.", "pl", None),
    # `R$` is the Brazilian real and this table has a wording for no
    # multi-character mark, so matching the `$` alone said "Dollar" — the wrong
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
    # U+2212, the typographic minus, which used to vanish into a space and take
    # the sign of the number with it.
    ("Minus sign \u22125 here.", "en", None),
    ("Temperatura \u22125 stopni.", "pl", None),
    # A digit run glued to a letter on *either* side is part of that token.
    # The lookbehind had no mirror, so a run touching a word on the right was
    # expanded up to the letter and then abandoned: "5x3" said *fivex3* and
    # "1e6" said *onee6*, a word welded to a digit. And an identifier that puts
    # a dot between its letter and its digits slipped past the one-character
    # lookbehind entirely, so "v1.2.3" read "v1.two point three".
    ("Value 1e6 here.", "en", None),
    ("It is 5x3 grid.", "en", None),
    ("Version v1.2.3 out.", "en", None),
    ("See SVN r123 now.", "en", None),
    # ISO 8601 writes end-of-day as 24:00. The hour was outside the pattern,
    # so both halves read as unrelated numbers and the colon stood between them
    # — "twenty-four:zero zero" reaching the model as written.
    ("At 24:00 sharp.", "en", None),
    ("At 23:59 sharp.", "en", None),
    # Spanish had no scale above a million, so anything from a billion up fell
    # back to digit-by-digit. A `billón` at 10^12 is the right Spanish word and
    # raises the ceiling enough for `mil millones` to compose on its own;
    # modelling 10^9 as its own scale word instead gave "dos mil millones
    # quinientos millones" for 2 500 000 000. Portuguese looks like the same
    # defect and is not: the CLDR corpus says `um mil milhões`, so its 10^9
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
    # two numbers — *trzy czternaście*, the same change of meaning as reading a
    # decimal as a clock time, arriving through a character set. It folds to
    # whichever mark the language actually writes, which is why folding it to a
    # dot everywhere would have put German back on the clock.
    ("٣٫١٤", "en", None),
    ("٣٫١٤", "pl", None),
    ("٣٫١٤", "de", None),
    ("٥٪", "en", None),
    ("٥٪", "pl", None),
    # A currency mark in front of foreign digits. The five currency patterns
    # spell their digit class differently — Python and Rust `\\d` is Unicode,
    # Go and JS `\\d` is ASCII — so "$٥" was five dollars in two ports and a
    # dollar sign in front of an unread numeral in the others. Folding the
    # digits before any pattern sees them settles it without touching five
    # regexes, which is why these cases live here rather than in a note.
    ("It costs $٥ today.", "en", None),
    ("Kosztuje $٥ dzisiaj.", "pl", None),
    ("Cost £٢٥٠ now.", "en", None),
    # A price is the one dotted pair that is never a clock time, and the funnel
    # used to forget it: the currency symbol becomes a trailing word before the
    # time pass runs, so "$0.49" reached it looking exactly like "14.30" — which
    # in the eleven comma-decimal languages is how a time is written. German
    # answered "null Uhr neunundvierzig Dollar", a price read as an hour.
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
    ("deadline’u", "pl", None),  # curly apostrophe; Go used to slice one *byte* here
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
        "the four number-regex shapes Swift used to read differently",
    ),
    (
        "I have 1 000 things.",
        "en",
        "the four number-regex shapes Swift used to read differently",
    ),
    (
        "It is -5 degrees.",
        "en",
        "the four number-regex shapes Swift used to read differently",
    ),
    (
        "Meet at a14:30.",
        "en",
        "the four number-regex shapes Swift used to read differently",
    ),
    (
        "Price 1 234 567 exact.",
        "en",
        "the four number-regex shapes Swift used to read differently",
    ),
    # --- Norwegian ------------------------------------------------------
    #
    # A block rather than a row in each family above, because until now `no`
    # had *zero* cases here: the roster ships Norwegian voices and the fuzzer
    # generates Norwegian, but the one gating parity contract exercised the
    # grammar nowhere. These five are the same five shapes the other languages
    # are held to — currency in a sentence, a clock time beside an
    # abbreviation, a date, grouping and the decimal comma, acronyms — so a
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
]


# Long-form splitting: where the reader breathes. Every port must cut in the
# same places, because a different split is a different set of joins and a
# different reading — and until now JS, Go and Rust had no long-form path at
# all while the documentation called them supported.
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
    # and cut surrogate pairs in half — a lone surrogate went straight to
    # frontend.encode(). Astral text is not exotic: emoji, CJK Extension B and
    # mathematical alphanumerics all live up there.
    "😀" * 40 + ". A tail sentence after the emoji run.",
    "𠀋𠀌𠀍" * 20 + ", and a clause after the ideographs, and one more.",
    # Whitespace is Unicode. Go trimmed a four-character cutset where Python
    # uses lstrip(), so an NBSP was charged against the next chunk's budget
    # and then removed from the chunk — every split after it drifted, usually
    # leaving a one-character chunk that becomes its own utterance with its
    # own derived seed. NBSP is ordinary in typeset prose.
    "x" * 20 + "\u00a0" + "y" * 40,
    "Ten\u2009tysięcy\u00a0złotych, a potem\u2028nowa linia, i koniec zdania.",
]

DIVERGENT_WHY = (
    "What one implementation still does differently. Dates, ordinals, acronyms "
    "and NFC used to live here; all four ports have them now and their cases "
    "moved into `cases` above, where every implementation is held to them."
)
"""Kept as a field rather than deleted: an empty `divergent` is a claim — that
nothing is known to differ — and a reader can only tell an empty block from a
forgotten one if the block is still there."""


CHUNK_CONFIGS: list[tuple[str, ChunkConfig]] = [
    ("shipping", ChunkConfig()),
    ("tiny", ChunkConfig(max_tokens=40, prefix_tokens=6)),
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
    OUT.write_text(rendered(build_payload()), encoding="utf-8")
    print(
        f"wrote {OUT} ({len(CASES)} funnel + "
        f"{len(CHUNK_CASES) * len(CHUNK_CONFIGS)} chunking cases)"
    )


if __name__ == "__main__":
    main()
