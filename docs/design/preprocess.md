# Preparing text for narration

The model reads graphemes. Before synthesis, the text frontend turns written
forms such as prices, clock times and dates into words in the voice's language.
The engine prepares the whole passage, then plans the chunks it can generate.
Postprocessing runs on generated speech tokens and can trim or retry a bad
window. To inspect the words going into narration, use `loudkit text`.

```sh
loudkit text --language en 'At 9:05, Prof. Jones paid $1.'
# At nine oh five, Professor Jones paid one dollar.
```

English is the first language to check when changing narration rules. Changes
to shared text handling also need examples in the other shipped languages.
Compare the prepared words across SDKs and use recordings to evaluate timing
and pronunciation. [Evaluation protocol](evaluation.md)

| Input | Language | Prepared words |
| --- | --- | --- |
| `He bought pens, etc. She followed.` | en | He bought pens, etcetera. She followed. |
| `1.00€, $1.01, and $1 million.` | en | one euro, one point zero one dollars, and one million dollars. |
| `1 234,56 €` | fr | mille deux cent trente-quatre virgule cinq six euros |
| `12 m²` | pl | dwanaście metrów kwadratowych |

In the French example the thousands separator is a narrow no-break space.
Recognizing the full group before moving the currency mark preserves the
amount in typeset text.

## The stages, in order, and why that order

```
NFC → fold foreign digits → remove invisible characters and markup tags
    → recognize grouped numbers and Polish quantity phrases → fold numeral symbols
    → expand symbols → drop footnotes → expand Roman numerals and acronyms
    → dates → ordinals → abbreviations → clock times → numbers
    → filter punctuation → language-specific respelling → tidy spacing and clause marks
```

The order preserves clues that would otherwise disappear. For instance,
a currency mark still identifies the amount in `£250` when the frontend
moves *pounds* behind it. Dropping `[12]` before general number expansion
keeps a footnote out of the narration. Parsing the date before the clock
keeps a German date such as `12.03.2026` together. Sentence punctuation then
informs both the chunk planner and the model.

---

## Numbers

All twelve text languages have grammar tables for integers and decimals.
The voice roster covers ten of them. Gender, plural forms, and word order
depend on the language and the construction. A successful check of isolated
cardinal numbers tells us little about how a unit or a price reads in prose.

### The grammar is data; only the interpreter is code

One JSON file, `models/data/numbers.json`, holds the grammar for all twelve
languages, and five small interpreters read it. A rule lives in one place, and
the shared fixture checks that the five interpreters agree. The format follows
the shape these systems have, a regular generative core plus a listed set of
irregular forms. That is how their own reference works describe them, and a
listed form can be checked by eye.

Three fields exist because of specific language phenomena:

| field | the phenomenon | why it is a field |
|---|---|---|
| combining forms | German says *eins* standing alone and *ein* before a scale: `einhunderteins` (101) has both in one word | The form depends on position. Treated as gender, it would make callers pass a gender that plays no part. |
| round-hundreds joiner | Portuguese *mil **e** oitocentos* (1800) but *mil oitocentos e noventa e dois* (1892) | The joiner depends on the shape of the remainder, not on its magnitude. |
| agreement outranks a listed form | Spanish lists *veintiuno*; the feminine is *veintiuna* | If the listed citation form won, a feminine count would come out masculine. |

### Past the largest scale it refuses

The recursion would stack scales into "a million milliards", which no language
here uses for a number. A value that large in running text is almost always an
identifier. `cardinal` raises `NumberGrammarError` and lets the caller decide.
`expand` reads the value digit by digit, so the sentence is still spoken.

### What the fixture is

145 cardinals and 14 agreement cases in
`tests/data/conformance/numbers.json`, **hand-written from each language's own
description, not captured from this implementation's output.** A fixture
captured from the code proves only that the code is deterministic. Written by
hand, it exposes defects: Danish *entusind* for 1000, and the Spanish
precedence bug above.

### How it is wired

Numbers, abbreviations and clock times run in every implementation. This
funnel is **algorithm-layer**: five implementations run the same shared
fixture. A Python-only pass would make them produce different text while the
fingerprint declares they agree, which is the defect class the fingerprint
exists to catch.

**All four are ported now.** `TextConfig.recipe` is `FUNNEL_PORTED`
(`"funnel-10"`, shared by text preparation and chunk planning),
unconditionally, in every implementation. There is no opt-in and
no `FUNNEL_EXTENDED`. The `divergent` block in
`tests/data/conformance/speechtext.json` is empty of cases and says so. What
used to sit there moved into `cases`, where every implementation is held to it.

---

## Which language this layer runs as

Every rule above is per-language, so the language is a preprocess question
first. The engine resolves it with the same chain of three in all five
implementations:

1. **the `language` argument**, if the caller gave one;
2. **`voice.language`**, the language the profile was enrolled from, if it is
   not empty;
3. **`"en"`**.

A profile records its own language, so the engine consults it. Without step 2,
`engine.synthesize("Cześć", polish_voice)` would read Polish text through the
English frontend (English number words, English abbreviation expansion, no
Polish respelling), and nothing would report it: a wrong-language read still
sounds plausible.

Pass `language` explicitly to request **cross-lingual** synthesis, such as an
English voice reading Polish text. The argument always wins over the profile.

| implementation | absent means |
|---|---|
| Python | `language=None` (the default) |
| Swift | `language: nil` (the default) |
| Rust | `language: None` |
| TypeScript | omit the argument |
| Go | the empty string |

Go cannot tell an omitted argument from an explicit `""`, so in Go an explicit
`""` falls through to the voice's language.

**Profile files without a language key load as English.** Every loader reads a *missing*
`language` header key as `"en"`. A non-English profile written without the key
therefore loads as English. Pass an explicit `language` argument, or set the
profile's language and save it again. Step 3 is reached only by a profile
built in memory without a language, or a header hand-edited to `""`.

### Which languages resolve at all

The chain's output is checked against an **allowlist**: the twelve ids
`loudkit.frontend.numbers.supported_languages()` reports, which is the roster in
`models/data/numbers.json` that all five implementations already load. Anything
else is refused, and the refusal names the twelve.
Use a base language ID such as `en` or `fr`. Regional tags such as `fr-FR`
are rejected by the CLI and the tokenizer.

It was a blacklist of `zh`, `ja`, `he`, `ko`, `ru`, the five whose upstream
pipeline needs model-based preprocessing this frontend does not carry. But the
tokenizer's vocabulary carries tags for 31 languages, so a blacklist accepted
the other 26. `encode(text, "bg")` NFKD-mangled Cyrillic into ids the model
reads as sounds it was never trained to make: no error, plausible audio, wrong
language. That is the failure this layer exists to prevent, arriving through the
language argument instead of through the text. The five are still named
separately in the message, because *why* they are refused is information the
caller can act on.

---

## Properties as well as cases

The shared conformance fixture checks concrete text expansions. Property
tests exercise additional inputs to catch character loss or repeated changes
to text that has already passed through the frontend.

| property | what it prevents |
|---|---|
| **charset closure** | a character the funnel emits and the tokenizer does not know is dropped or mapped to index zero, and nothing reports it |
| **output is NFC** | a decomposed character reaching the rules, and the vocabulary, as a base letter plus a mark |
| **idempotence** (`f(f(x)) == f(x)`) | a rule that fires on its own output will eventually fire on text a user wrote |
| **chunk-join invariance** | splitting must not add, drop or change characters |

The tests check these properties on their samples. They are not guarantees for
every string. Two fixture inputs are known not to be idempotent:
`Meet at 11:30AM sharp.` gives *eleven thirty AM*, and a second pass spells
*ay-em*; in Polish, `one 漢字 two three four` gives *tri*, and a second pass
gives *traj*.

---

## What this layer refuses to read

These rules guard against partially rewriting an identifier or assigning
the wrong currency to a number. “Left written” describes the prepared text.
The model can still pronounce that text poorly or skip part of it.

This was checked by ear, because a refusal hands the reading to the model, and
for some inputs the half-expansion might sound better. Both directions were
rendered at the same seed and voice: `x200 000` against "x200 zero zero zero",
and `Müller123` against "em el el e er jeden dwa". The refusals were clearly
better.

If you revisit a refusal, listen for whether a listener can tell that
something was dropped, not for which reading sounds nicer. "em el el e er
jeden dwa" sounds like a complete, correctly read name. It is a different
name, and the audio gives no sign that the `ü` and the `3` are missing.

| Input | What happens | Why |
|---|---|---|
| `iOS18`, `r123`, `5x3`, `1e6` | left written | A digit run touching a word on either side is part of that word. *iOSeighteen* is not a reading of anything, and expanding one side only gave *fivex3* and *onee6*: a word welded to a digit. |
| `v1.2.3`, `Ver.2` | left written | The lookbehind sees one character, and an identifier can put a dot between its letters and its digits. Answered by walking back over word characters, dots and commas until a letter appears. |
| `1.2.3`, `192.168.0.1`, `18.08.2026` (en) | left written | A dotted run no convention resolves. English field order is unrecoverable (`3/12` is March twelfth to half the English-speaking world), and a wrong month is worse than heard digits. |
| `1e+3`, `2.5E+1`, `1e-3` | left written | The exponent's sign is part of the token, so the whole run is refused rather than read up to it. |
| `x200 000`, `200 000x` | left written | A grouped run needs a boundary at both ends. Reading the part that fits left `200` spoken and `000` behind it, or the reverse. |
| `+1 202 555 0199` | digit by digit | A run of unequal groups is not a grouped number. Read as one it came out "one billion … nineteen" with a bare "9" trailing. |
| `Müller123`, `żelazny2024` | left written | The Polish code speller spells all of a token or none: a character with no letter name was dropped in silence, so `ü` vanished and the name changed. Also refused past eight characters. Spelling `R2` is how a reader says it; spelling an eleven-character identifier is a wall nobody follows. |
| `R$3,14`, `HK$5` | mark dropped, amount read as a decimal | A currency mark with a letter in front is one this table cannot name. `R$` is the Brazilian real; matching the `$` alone said "Dollar". Losing a symbol is a smaller lie than naming the wrong money. |
| `14.30` in en | a decimal | `H.mm` is a clock in the eleven languages whose decimal separator is a comma, and a decimal in English. The grammar file already knows which is which, and reading `$0.49` as "zero forty-nine" was the cost of not asking it. |
| `24:30` | In English, *twenty-four:thirty* | The clock rule only accepts hour 24 with minute 00. Ordinary number expansion can still read the individual values. |
| `[12]`, `💩`, `©®™` | Empty prepared text | Synthesis raises `NothingToSpeakError` when these are the entire input. |

Two of these are *not* refusals. Arabic-Indic digits (`١٢٣`, `٣٫١٤`) are
**read**, folded to ASCII beside NFC, because they are digits, and eleven of the
twelve languages used to pass them through untouched while Polish read them.
`1 234 567` grouped correctly is read as a cardinal. The recognized group
separators include regular space, no-break space (U+00A0), thin space
(U+2009) and narrow no-break space (U+202F).

### Where a run of digits ends, and what is still not settled

Go and Rust use RE2, which does not backtrack, so where Python's engine retries
an alternative they take the longest prefix that fits. Everything below is that
difference or a consequence of it.

**The target rule, stated once and applied five times: a maximal run of digits
and separators that does not reduce to a single readable number is left
written.** It is a decision about where the token *ends*, not a guard bolted
onto either implementation's current behaviour. Where the token ends is the
question the two engine families answer differently.

**Where a token ends, in one rule with one asymmetry.** A space is part of a
number where it groups thousands: a digit immediately before it and three digits
after. Anywhere else it ends a token. Each token is then decided on its own: all
of it read, or all of it left written.

The asymmetry is the fourth digit. The walk that runs *forward* out of a match
crosses a space in front of `0023` as readily as one in front of `000`. A ragged
group is precisely why the pattern refused to bind the run, and the forward walk
is there to finish what the pattern would not. The walk that runs *backward*
does not cross it, because there the group is the match itself and its width is
already fixed. In the reference these are `_starts_a_group` and
`_continues_a_group` in `python/loudkit/frontend/numbers.py`.

**Boundary behavior.** `1 0023R` matched
the `1`, read *en*, and left `0023R` written. That is half a run spoken with the
rest welded to a letter, the class the right-hand guard exists to stop. The
engines that do not backtrack never had it: they bind `1 002` greedily, find the
`R` behind it, and refuse the whole run. A backtracking engine reaches the same
answer by asking the looser question forwards. Same family:
`1 234 5672.5E+1`, which used to say *en* and *tohundrede og fireogtredive*
before giving up.

The other direction is why the clause stays there. Dropping it in both walks was
measured over 4800 generated sentences: 60 readings change and **56 are
losses**, because the walk then steps out of one token and into the next.
`e3 1000` stops saying *ettusen*: `1000` is four digits with no thousands group,
and the walk crosses an ordinary space to find the `e` of an exponent it shares
nothing with. Asked forwards only, 20 readings change: four are numbers that had
gone unsaid, sixteen are ragged runs no longer read half way.

**Closed with it: the width check, which the reference had wrong and three ports
had right.** `text[i:i+3].isdigit()` is True of a one-character slice, so a lone
digit passed for a thousands group: in `R2 2` and `R2 12` the backward walk
crossed the space, reached the `R`, and refused a number nothing was glued to.

**And the walks now ask about ASCII digits.** `str.isdigit` is true of `²` and
of every Unicode decimal digit, none of which the digit-run pattern can match,
so in `R² 200` the `²` counted as the digit behind a grouping space and
swallowed the number after it. The other three ports test ASCII.

A forward walk had the same rule as the backward one and did not use it. It
crossed a thousands space whenever *a digit* followed, which walked out of one
number and into the next. `1000 5.1e+3` therefore refused the `1000`: the walk
found the `e` of an exponent two tokens away and called the whole thing one
glued token. Four of the fuzzer's Go divergences were exactly that, with Go
right and this side wrong. Three digits and not fewer is what stops it, and that
half of the rule is unchanged.

**Closed with it: runs whose separators are dots.** `200 0003.14` and
`1 00012.03.2026` reach a fraction or a date rather than a letter. The ports
used to read different halves of them: one said the whole thing digit by digit,
another stopped at the decimal point and wrote the rest. All five now agree on
both, in all twelve languages.

**Language-specific word classes.** The divergences that
outlived the space fix were not about digits at all. Each was a port reading
Unicode through whatever word class its regex engine handed it:

- ICU's `\w` counts a combining mark as a word character, so Swift alone refused
  `a̬123`.
- The `regex` crate's `[:alpha:]` is ASCII even in Unicode mode, so Rust alone
  took the `€` in `zł€ 000 000` for a bare euro sign.
- Swift's acronym splitter walked `Character`, which is a grapheme cluster,
  where Python's `\W+` splits on code points.

The fourth was not a class but a rule the reference had **deleted** and four
ports had kept. `spell_acronyms` owns the initialism decision. The Polish
respeller in Go, Rust, JS and Swift was still spelling acronyms a second time
with no view of the surrounding capitals, so `CIA CIA`, a run and therefore
emphasis, came out *ce-i-a ce-i-a*.

`tools/fuzz_parity.py` is green on twenty seeds and eight thousand generated
sentences across all five implementations, so the fuzz-parity job in `ci.yml`
gates rather than reports. Swift is not in the Linux command -- `LoudKitText`
imports CryptoKit and OSLog and `LoudKit` imports CoreML, none of which exist on
that runner -- so it is fuzzed by the macOS `swift` job instead. It used to be
fuzzed "before merge", meaning by hand, and this page said so as though that
were a gate; four of the first seven seeds were red at the time.

**One question the rule does not answer yet.** It is a question about the rule
rather than a divergence. `200 0003.14` reads *two hundred zero zero zero three
point one four* in all five. The maximal run there does not reduce to a single
readable number, so the rule as stated above argues it should be left written
the way `200 0001e-3` is. The ports read it digit by digit instead. That is
consistent across the five and so invisible to the fuzzer, which compares them
against each other. Deciding it means deciding whether digit-by-digit is a
*reading* or a *refusal*, and that choice belongs with the ear tests rather than
with a parity gate.

## The periods this layer leaves written

An abbreviation expander can turn `e.g.` into *for example*. For `St.`,
there are at least two ordinary spoken readings, *Saint* and *Street*.
The planner can keep an abbreviation with its following word even when the
frontend has insufficient context to expand it.

### What the splitter tests

For the standard punctuation policy, a period can be held when the preceding
word is in `ChunkConfig.abbreviations`, when an initial precedes a capitalized
word, or when the next word starts with an ASCII lowercase letter. The latter
also covers ellipses collapsed to a single period. Final punctuation can
include a closing quote or bracket. `ChunkConfig.mid_sentence_period="break"`
allows a cut at an otherwise held period.

The planner also considers how much of the sentence would remain after the
cut. This matters when a conservative text-length estimate places the final
word of a sentence in a separate render. [Full chunk selection and recovery
rules](text-funnel.md#loudkitfrontendchunkingpy)

### Evidence behind the original abbreviation list

**The thirteen single-letter entries pay for themselves, and the arithmetic is
worth stating because their cost is easy to find and their benefit is not.**
They hold a period after any lone capital, which is what an initial looks like,
and a review reasonably objected that this is the same shape as the
"short capitalised token" rule the survey rejected for wrongly holding 57
genuine sentence ends. Measured on five English Gutenberg books, 1087
paragraph-sized passages: dropping all thirteen takes chunks that end on a
dangling initial from 5 to 47, and chunks that end on a comma from 3019 to
3009. Forty-two prevented against ten caused. Per letter, only `D` (+36
dangling initials if dropped), `S` (+5) and `J` (+1) do anything on this
corpus; the other ten are inert here, so they cost nothing and may earn their
place on prose this corpus does not contain. A comma cut matters more than its
count suggests -- it is the seam class `cap_resplit` exists to avoid -- which is
why the trade is written out rather than assumed.

### Two traps for anyone amending the rule

**Check what the number pass emits.** `speech_text("Der 2. Weltkrieg", "de")` gives
`"Der zwei. Weltkrieg"`. The splitter never sees the digit, so a condition keyed
on a digit in front of the period is dead code in this pipeline.

**A word expansion can affect the sentence boundary.** English `etc.` keeps
its final period at the end of a passage or before a capitalized next sentence.
For other abbreviations, check what `loudkit text` actually hands the splitter.

### Why the tests are ASCII

A rule that asks whether a character is a lowercase letter gets different
answers in five languages: `islower`, `unicode.IsLower`, `char::is_lowercase`
and `Character.isLowercase` are not the same predicate. Measured on the
surveyed corpus: of 2253 periods in ten languages, four are followed by a word
starting with a non-ASCII lowercase letter, and reading the whole Unicode
`Lowercase` property instead moves one passage in 1200. Two ASCII comparisons
give one answer in all five implementations; those four periods are the cost.

### What it is worth

Surveyed over 1200 passages in ten languages, 4973 chunks:

| | cuts on a period inside a sentence | chunks of 20 characters or fewer | word-boundary breaks |
|---|---|---|---|
| `"break"` | 59 | 98 | 144 |
| `"hold"` | 1 | 93 | 150 |

In that survey, falling back to a held period saved six word breaks but
introduced seven breaks at a period, five of which left a title at the end of
a chunk. The list was extended with `Dr` and `St` for English prose. In
the corresponding 24-book comparison, chunks ending on `Mr.`, `Mrs.`,
`Dr.` or `St.` fell from 4,115 to 29. These counts describe that earlier
policy and corpus.

## Out of scope

These were considered and left out.

**A mandatory phonemizer.** Measured evidence says a *mismatched* one is worse
than none: a controlled study found a phoneme front-end underperforming
characters at nearly every training budget, because the phonemizer's dialect
did not match the corpus's. The standard open phonemizer's own error rate
against a pronunciation-dictionary gold standard is in the mid-teens for
English and German. It is also GPL-3.0. The common alternative is an optional
inline pronunciation override.

**Learned prosody prediction.** Chunk planning uses the prepared punctuation
and a word-boundary fallback. The model determines the timing and intonation
within each rendered window.

**POS tagging as a stage.** It pays off mainly as a homograph feature, and the
measured end-to-end contribution of homograph handling in a full front-end is
about a third of a percentage point.

**Neural or LLM normalization.** In-domain it beats rules; on ambiguous input it
fails badly. Production text-normalisation systems stay largely rule-based
because they must avoid unrecoverable errors.

---

## One class question, asked the same way five times

Every pass in this funnel asks some version of "is this a letter, a digit, or a
space". Each runtime's own classes differ, and a difference changes what the
model hears. The five implementations therefore share one written-down answer
for each class.

### Numerals

A number character with no ASCII spelling (a superscript, a vulgar fraction, a
circled or Roman numeral, a digit from any script but Latin) is folded before
the symbol table. `\w`, `\p{N}` and `str.isdigit()` all admit `No`. Without the
fold, `²9` would be one token, the number matcher would decline it, the
punctuation pass would delete the `²`, and a bare `9` would reach the model.

`fold_numerals` turns each of them into text a reader says aloud:

| input | output, in all five |
|---|---|
| `²9` | `two nine` |
| `Add ½ cup` | `Add one divided by two cup` |
| `Chapter Ⅶ.` | `Chapter seven.` |
| `5৩3` | `five hundred and thirty-three` |
| `１２３` | `one hundred and twenty-three` |
| `12 m² and $9` | `twelve m two and nine dollars` |

**What each character becomes is a table**, `models/data/numerals.json`,
generated by `tools/make_numerals.py` and copied to every port. A decimal digit
of any script becomes the ASCII digit of the same value and **joins the run it
was in**, so a number written across two scripts is still one number. Every
other number character becomes the text the table names (`²`→`2`, `½`→`1/2`,
`③`→`3`, `Ⅳ`→`IV`, `፲`→`10`), separated from an adjacent letter or decimal
digit, so `²9` does not fold into `29` and read as twenty-nine. Inside a
spelled numeral the fold writes `/` as `÷`, so `½` reaches the symbol pass as
`1÷2`.

The table text is ASCII for all but 27 of its 1,151 entries. Those 27 map to
CJK ideographs: `〸` → `十`, and ten in parenthesised form, such as `㈠` →
`(一)`. They leave the fold as ideographs, which later passes treat as letters.
No entry maps to Hangul.

**Whether a character is a numeral comes from the same table.** `unicodedata`,
`unicode.IsNumber`, ICU and V8 each carry the Unicode version their runtime was
built with. CI runs Python 3.10 to 3.14, whose `unicodedata` covers Unicode
13.0.0 to 16.0.0. With detection from the runtime, `Add 𐵁 cups` (GARAY DIGIT
ONE, added in 16.0) reads differently on different interpreters under the same
`algorithm_fingerprint`. The table is cut from one pinned UCD
(`EXPECTED_UNICODE` in the generator, which refuses an interpreter that carries
another version), so every runtime reads it the same way, including runtimes
older than the table. `Add 𐵁 cups` reads *Add one cups*.

The fold needs a table because computing it from Unicode data goes wrong in two
ways:

- **Walking down to a decimal block's start** leaves the block where two blocks
  touch. `MATHEMATICAL DOUBLE-STRUCK DIGIT ZERO` follows
  `MATHEMATICAL SANS-SERIF DIGIT NINE` with nothing between them, so the walk
  reads that zero as a nine. The table lists every block's zero instead.
- **NFKC does not reach every numeral.** `ETHIOPIC NUMBER TEN`, the Aegean
  numbers, the Kaktovik digits and the Meroitic numerals have no compatibility
  decomposition. The table names them: `Add ፲ cups` reads *Add ten cups*.

The table is hashed into `TextConfig.grammar` beside the grammar and the
lexicon. That also pins the *Unicode version* the fold uses: two builds that
hash the same file fold the same characters the same way, whatever Unicode
their runtime carries.

For a fraction glyph, the fold preserves its mathematical meaning by passing
the numerator, division symbol and denominator to the language rules. A typed
slash can also denote a date or a path. See the English examples under
[Known difficulties](#known-difficulties).

The fold leaves category `Lo` alone. Ideographic numerals such as `一` and `十`
are letters, and a pass keyed on numeric type would delete a language's
numerals. `一二三` is a fixture case for that reason. Keeping the characters
does not add Chinese support.

### Letters

`\p{L}`, everywhere. The **Alphabetic** property is wider by the circled
letters (`So`), the Other_Alphabetic marks and the letter numbers. Fixture
cases pin the difference: `xⓐy` reads *x y*, and `ⓐ-1` reads *minus one*.
Swift reads the general category, not `CharacterSet.letters`, which Foundation
answers wrongly above the BMP. Measured: it omits all 6,145 Tangut ideographs,
and `subtracting(.nonBaseCharacters)` fails for astral scalars, which lets
1,162 combining marks through.

Walks step over code points, not UTF-16 units. In JavaScript an astral letter
would otherwise arrive as a lone surrogate that `\p{L}` does not match, and the
walk would step past it. `0.𗀀` stays written in all five.

### Whitespace

`WHITE_SPACE` is Unicode White_Space written out (25 code points), and every
pattern in the funnel uses it. The runtimes' `\s` classes are four different
sets: ECMAScript's excludes U+0085 NEL, RE2's is ASCII and omits U+000B, and
CPython's also admits U+001C–U+001F.

Without the shared set, Go reads a footnote marker aloud when its separator is
a non-breaking or thin space, which is ordinary French and German typography,
and misses a currency mark separated by a vertical tab. JavaScript splits
chunks in different places, because a NEL left on the front of the remainder
is charged against the next chunk's budget (measured: 51 of 500 fuzz cases).

### Boundaries

`(?![\p{L}\p{Nd}_])`, written out instead of `\b`. ECMAScript's `\b` is ASCII,
and so is a byte-wise test in Go. With either, a date glued to a CJK ideograph
would be read in those two implementations and left written in the other
three.

### What still reaches the model unread

A digit glued to a letter stays written. For example, `iOS18` stays
`iOS18` in the prepared text, and
`12.03.2026一二三` keeps its digits. That is deliberate and long-standing — a
run glued to a word is part of that word — and it is now at least the same
deliberate answer in all five.

## Known difficulties

**Recipe fractions and metric units.** In English, `1/2 cup` prepares as
*one two cup*, while `½ cup` prepares as *one divided by two cup*. Write
*half a cup* when that is the intended reading. English `2 kg` leaves
`kg` in the prepared text, and `12 m²` becomes *twelve m two*. French
similarly yields *deux kg* and *douze m deux*. Supplying expanded units is
the reliable option for those cases. Polish has dedicated quantity rules for
these units. Check the actual language and construction with `loudkit text`.

**Polish grammatical case.** Gendered number forms and selected case-sensitive
constructions have dedicated rules. Broader case selection depends on syntax. `w 2026 roku` needs
the locative *w dwa tysiące dwudziestym szóstym roku*, and 5-and-above governs
the genitive plural while making the whole phrase neuter singular. **Selecting
the case requires knowing the syntactic role of the number in its sentence**,
which means a morphological tagger. That is a different component with a
different failure mode, and it cannot be a lookup table. The published recipe is
a tagger plus a transducer plus an inflected lexicon. The seam for it exists:
the `gender` argument is the same shape a `case` argument would take.

**Years.** `1892` is *eighteen ninety-two* as a year and *one thousand eight
hundred and ninety-two* as a quantity, and **nothing in the string says which**.
Deciding needs context the funnel does not have. A wrong guess is an
unrecoverable error in the literature's sense, because it conveys a different
number. The number is currently read as a quantity, which is at least never
absurd.

**Ordinals.** German `3.` is *dritte*, and its ending is selected by the
preceding preposition or article: `am 5. Mai` is *fünften*, `der 5. Mai` is
*fünfte*. The best open WFST grammar for German emits all five endings as
alternatives and defers the choice to a language model. Same problem as Polish
case, same missing component.

**Dates and times.** The rules are known and per-language. English and German
take ordinal days. Spanish, Portuguese, French and Italian take cardinal ones
(`5 de mayo` is *cinco*, not *quinto*). Dutch `half drie` is 2:30, not 3:30.
What blocks them is not the rules but **the ambiguity of the input**: `3/14` is
a date in one convention, a fraction in another, and a ratio in a third.
Resolving that needs a locale the funnel is not currently told.

**Native-speaker verification.** The Danish, Dutch and Portuguese grammars were written
from reference descriptions and are internally consistent, but they have not
been verified by native speakers, and the literature covers Danish and Dutch
particularly thinly. The Danish numerals are vigesimal (*tres* for 60,
*halvfjerds* for 70), and at least one widely-used library ships *treds* for 60,
which is wrong in every number containing it. Native-speaker review complements
shared text fixtures in every language.

**Portuguese variant.** European and Brazilian Portuguese diverge in ways
spelling does not mark, and the number grammar here is European (*dezasseis*,
not *dezesseis*). State the variant wherever the voice is described: the ASR
round-trip does not separate the two variants (see
[evaluation.md](evaluation.md)).
