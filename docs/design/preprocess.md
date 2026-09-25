# Preprocess: turning text into something the model has seen

The first stage of `preprocess → tts → postprocess`. It narrows the gap between
what a person writes and what the checkpoint was trained on.

The model reads **graphemes**. It has no phoneme layer and no lexicon. A
character the training transcripts did not contain arrives as an embedding the
model barely trained, or does not arrive at all.

---

## The failure it exists for

The typical failure is a number that goes missing, not one that is
mispronounced. A checkpoint trained on normalised transcripts has never seen
`45`. Peer systems show the effect: StyleTTS2's symbol table contains no
digits, and F5-TTS maps every unknown character to a space. Neither reports it,
and the sentence comes out with a hole where the number was.

Number expansion is therefore the first job of this layer, ahead of anything to
do with pronunciation quality.

---

## The stages, in order, and why that order

```
NFC  →  foreign digits  →  invisibles  →  markup tags  →  numerals
     →  symbols  →  footnotes  →  Roman numerals  →  acronyms  →  dates
     →  ordinals  →  abbreviations  →  times  →  numbers  →  punctuation
     →  respelling  →  collapse
```

Every stage runs in all five implementations. The reference is `speech_text`
in `python/loudkit/frontend/speechtext.py`; the per-symbol notes are in
[text-funnel.md](text-funnel.md).

**NFC first, before any pass inspects a character.** Unicode lets the same
character arrive two ways: Polish `ą` as U+0105 or as `a` + U+0328, Danish `å`
as U+00E5 or `a` + U+030A. The regexes, lexicon lookups and character classes
below are written against the composed form. Without NFC, a decomposed spelling
reaches them as a base letter followed by a combining mark, and they miss it.
The tokenizer applies its own NFKD later; NFC here is for the rules.

**Foreign digits beside NFC.** Arabic-Indic and Extended Arabic-Indic digits,
their decimal and grouping separators, and the Arabic percent sign fold to the
language's own spelling, so every later pass sees one form. `٣٫١٤` reads
*three point one four* in English and *drei Komma eins vier* in German.

**Invisibles next.** Zero-width joiners, format characters and the soft hyphen
are not whitespace by Unicode's rules, so no later pass removes them. A
grapheme model would read them as characters inside the word, which makes a
word no training text contains.

**Markup tags before symbols.** The symbol pass would read a tag's angle
brackets as comparison operators and its attributes as text. `<p>Hello</p>`
reads *Hello*.

**Numerals before symbols.** `fold_numerals` turns superscripts, vulgar
fractions, circled and Roman numerals and non-Latin digits into text the later
passes read. See [Numerals](#numerals).

**Symbols while the digits are still digits.** `£250` reads as *two hundred and
fifty pounds*. The mark is written in front and spoken behind, so the amount
must still sit next to its symbol.

**Footnotes, then Roman numerals, then acronyms.** Footnote markers go before
any number pass, because a dropped `[12]` must not become words first. Roman
numerals run before the acronym pass, which would spell `IV` letter by letter.

**Dates, ordinals, abbreviations, times, then numbers.** Dates run first
because the clock pattern and the digit-run pattern each match part of
`12.03.2026`. Ordinals run before numbers, which would expand `1st` and leave
the suffix behind. All of them run before punctuation, which turns a minus sign
after a space into a space: `-5` would lose its sign.

**Punctuation last of the rewriting passes.** Prosodic marks stay: `. , ! ? ; :`,
dashes, the ellipsis, quotation marks, parentheses, apostrophes, `¿` and `¡`.
They are the model's main cue for a question contour or a clause break. Every
other symbol becomes a space, except a numeric separator between digits and a
`-` or `+` inside a token.

**Respelling.** Polish only: English words in Polish text are rewritten the way
a Polish reader says them.

**Collapse runs, not pairs.** A run of clause marks folds to one mark, so `...`
becomes `.` in one pass. A pair rule would not be idempotent: regex
substitution does not overlap its matches, so `...` would become `..` on one
pass and `.` on the next.

---

## Numbers

`loudkit.frontend.numbers` verbalizes integers and decimals in all twelve
languages it has grammars for. That set differs from the ten languages the
voices ship for. The grammars are written in this repository instead of taken
from a library, for two reasons.

The widely used Python library for number words is LGPL-2.1, which this kit
cannot carry into every embedding it is meant for. It also has **no case or
gender machinery for Polish at all**: one nominative form per numeral, while
the same release ships six-case declension for Russian. Polish needs agreement
most, so the dependency would have to be replaced for that language anyway.

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

`TextConfig.recipe` is `FUNNEL_PORTED` (`"funnel-6"`) in every implementation,
unconditionally. There is no opt-in. The `divergent` block in
`tests/data/conformance/speechtext.json` holds no cases: every case sits in
`cases`, where all five implementations are held to it.

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
`models/data/numbers.json` that all five implementations load. Anything else is
refused, and the refusal names the twelve.

The tokenizer's vocabulary carries tags for 31 languages, so a tag alone proves
nothing. `encode(text, "bg")` would turn Cyrillic into ids the model reads as
sounds it was never trained to make, with no error and plausible audio in the
wrong language. `zh`, `ja`, `he`, `ko` and `ru` are named separately in the
message, because their upstream pipeline needs model-based preprocessing this
frontend does not carry, and the caller can act on that.

---

## Properties as well as cases

The conformance fixture pins what the funnel does to 217 specific strings.
`tests/test_funnel_properties.py` checks four properties on sample prose in
every language. They catch a different class of defect: a rule that quietly
drops something, where a case catches a rule that is wrong.

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

The rule underneath this section: **a token is read whole or left written,
never half.** A half-expanded number, a name with a letter missing, or the
wrong currency is a confident wrong answer, and nothing in the audio says so.
A token left written goes to the model as written.

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
| `iOS18`, `r123`, `5x3`, `1e6` | left written | A digit run touching a word on either side is part of that word. Expanding the digits alone would give *iOSeighteen*, *fivex3* or *onee6*: a word welded to a number. |
| `v1.2.3`, `Ver.2` | left written | An identifier can put a dot between its letters and its digits, so the glue test walks back over word characters, dots and commas until it finds a letter. |
| `1.2.3`, `192.168.0.1`, `18.08.2026` (en) | left written | A dotted run with more than one separator is not a quantity. English marks dotted numeric dates as ambiguous (US and UK order day and month differently), so the date pass leaves them all, and a wrong month is worse than heard digits. |
| `1e+3`, `2.5E+1`, `1e-3` | left written | The exponent's sign is part of the token, so the whole run is refused. |
| `x200 000`, `200 000x` | left written | A grouped run needs a boundary at both ends. Reading only the part that fits would leave `200` spoken and `000` glued to the letter, or the reverse. |
| `+1 202 555 0199` | digit by digit | A plus followed by eight or more digits is a telephone number. |
| `Müller123`, `żelazny2024` (pl) | left written | The Polish code speller spells all of a token or none of it. `ü` has no letter name in its table, so spelling the rest would drop it and change the name. Tokens over eight characters are also left written. `R2` is spelled (*er dwa*). |
| `R$3,14`, `HK$5` | `$` dropped, letters kept, amount read (*R three point one four*, *aitch-kay five*) | A currency mark with a letter in front is one the symbol table cannot name. `R$` is the Brazilian real; matching the `$` alone would say "dollars". |
| `14.30` in en | a decimal (*fourteen point three zero*) | `H.mm` is a clock time only in languages whose decimal separator is a comma, which is the other eleven. The grammar file records the separator, so English `$0.49` stays a price. |
| `24:30` | not a time: each side read as a number, colon kept (*twenty-four:thirty*) | 24 is an hour only with a zero minute (ISO 8601 writes end of day as `24:00`). |
| `[12]`, `💩`, `©®™` | removed; the engine raises `NothingToSpeakError` (a `ValueError`) | The funnel removes all three and leaves empty text. Empty text must not reach the tokenizer, which would still produce audio. |

Two inputs are read, not refused. Arabic-Indic digits (`١٢٣`, `٣٫١٤`) fold to
ASCII beside NFC and read as numbers in all twelve languages. A correctly
grouped `1 234 567` reads as a cardinal. The refusals above are about runs that
only *look* grouped.

### Where a run of digits ends

The five implementations run two kinds of regex engine. Python's, JavaScript's
and Swift's backtrack. Go's `regexp` (RE2) and Rust's `regex` crate do not:
where Python retries an alternative, they take the longest prefix that fits.
The rules below make both kinds answer the same.

**The target rule: a maximal run of digits and separators that does not reduce
to a single readable number is left written.** It decides where the token
*ends*, which is the question the two engine kinds answer differently. The
exceptions still open are listed under [Open questions](#open-questions).

**Where a token ends.** A space is part of a number where it groups thousands:
an ASCII digit immediately before it and three ASCII digits after. Anywhere
else it ends a token. Each token is then decided on its own: all of it read, or
all of it left written.

**The fourth digit.** The walk that runs *forward* out of a match
(`_starts_a_group`) crosses a space in front of `0023` as readily as one in
front of `000`: it asks only whether three digits start there. A ragged group
is the reason the pattern refused to bind the run, and the forward walk
finishes that run. So `1 0023R` is one token, glued to the `R`, and is left
written. The walk that runs *backward* (`_continues_a_group`) asks for exactly
three digits, with no fourth behind them, because there the group is the match
itself and its width is already fixed. So in `e3 1000` it does not cross the
space, and `1000` reads (*ettusen* in Swedish). Both helpers are in
`python/loudkit/frontend/numbers.py`. `1 234 5672.5E+1` is left written the
same way.

Measured over 4800 generated sentences: allowing a fourth digit in both walks
changes 60 readings, and 56 of them are losses, because the backward walk then
steps out of one token into the next. Allowing it forwards only changes 20:
four numbers that went unsaid are read, and sixteen ragged runs are left
written instead of read half way.

**Width.** `_starts_a_group` requires three characters as well as three digits.
`text[i:i+3].isdigit()` alone is true of a one-character slice, so a lone digit
would pass for a group: in `R2 2` the backward walk would cross the space,
reach the `R`, and refuse a number nothing is glued to. `R2 2` reads
*R2 two*.

**ASCII digits.** The walks test ASCII digits, the class the digit-run pattern
matches. `str.isdigit` is also true of `²` and of every Unicode decimal digit.

**Three digits forwards.** The forward walk crosses a thousands space only when
three digits follow. So in `1000 5.1e+3` it does not walk into the exponent
two tokens away: `1000` reads *one thousand* and `5.1e+3` stays written.

**Dotted separators.** `200 0003.14` and `1 00012.03.2026` reach a fraction or
a date, not a letter. All five implementations read them the same way in all
twelve languages; the readings are listed under
[Open questions](#open-questions).

**Word classes.** The five implementations use one definition for each
character class (see [One class question](#one-class-question-asked-the-same-way-five-times)).
Fixture cases pin the points where runtime classes differ:

- A combining mark is not a word character. ICU's `\w`, which Swift's
  `NSRegularExpression` uses, includes combining marks. `a̬123` reads its
  number in all five.
- A letter is `\p{L}`. The Rust `regex` crate's `[:alpha:]` is ASCII even in
  Unicode mode. In `zł€ 000 000` the `ł` is a letter, so the `€` does not open
  an amount.
- The acronym splitter splits on code points, as Python's `\W+` does, not on
  grapheme clusters, which Swift's `Character` walks.

`spell_acronyms` makes the initialism decision while the neighbouring words are
still visible. The Polish respeller does not spell acronyms a second time. So
`CIA CIA`, a run of capitals and therefore read as emphasis, stays *CIA CIA*.

**Fuzzing.** `tools/fuzz_parity.py` compares the five funnels on generated
sentences. Measured: twenty seeds, 8000 sentences, no divergence. In CI the
`fuzz-parity` job runs seeds 1 and 2 (300 cases each) for Go, Rust and JS, and
a divergence fails the build. Swift is not in that Linux job: `LoudKitText`
imports CryptoKit and OSLog, and `LoudKit` imports CoreML, none of which exist
on Linux. The macOS `swift` job fuzzes Swift with the same two seeds.

### Open questions

These readings are the same in all five implementations, so the fuzzer, which
compares the implementations with each other, cannot flag them:

- `200 0003.14` reads *two hundred zero zero zero three point one four*. The
  maximal run does not reduce to a single readable number, so the target rule
  says it should stay written, as `200 0001e-3` does.
- `1 00012.03.2026` reads *one 00012.03.2026*: half the run is read.
- `R² 200` reads *R two thousand two hundred*. The numeral fold turns `²` into a
  separate `2`, and `2 200` is a valid thousands group.

Deciding the first means deciding whether digit-by-digit is a *reading* or a
*refusal*. That needs a listening test; a parity gate cannot answer it.

## The periods this layer leaves written

The abbreviation stage expands only the unambiguous entries: a wrong expansion
is worse than a spelled abbreviation. `e.g.` becomes *for example*, and its
period goes with it. `St.` is Saint or Street, so it stays written, and its
period reaches the next stage.

That next stage is the splitter, which breaks long text at `. ` because a full
stop is the least audible place to break. A period that closes a title is not
a full stop. Without a rule, `"But Mr. Smith went home"` splits off the chunk
`"But Mr."`: seven characters, its own utterance, its own derived seed, and a
token ceiling proportional to seven characters. Over a 9920-row rendered census
it is the only chunk that hit that ceiling.

**The period rule lives in the splitter, not in this layer.** The funnel could
hold these phrases together in two ways, and both are worse:

* **Expand them.** That is the stage above, and it refuses these entries by
  name. Adding `St.` to it to fix a chunk boundary would trade an audible
  boundary for a wrong word.
* **Mark them.** A marker character survives into the string the tokenizer
  reads, so it either reaches the model or needs stripping in five
  implementations. The funnel's output is read aloud, so it must stay plain
  text.

Where a window ends depends on the token budget, and no other stage needs to
know it. So the splitter decides.

### What the splitter tests

Two conditions, and one of them is a list. Both apply only to a period: the
splitter treats `!` and `?` as sentence ends and `;` and `,` as clause marks,
so the question is only about the one mark written for two jobs.

1. **The next character is an ASCII lowercase letter.** A new sentence rarely
   starts in lower case. This condition catches what this layer produces:
   `speech_text` maps an ellipsis to `...` and then folds a run of `[.,;:]` to
   one mark, so `"grzeja sie... cieplem"` reaches the splitter as
   `"grzeja sie. cieplem"`. It is the main cause in Polish, which has no
   abbreviation cuts at all, and no abbreviation list reaches it.
2. **The text up to the period ends with a listed abbreviation**, at a word
   boundary: the character in front of the match is absent, or is not an ASCII
   letter or digit. `NASA.` ends in `A`, and `A` is a listed initial, so the
   guard is what keeps that one breaking.

The list is `ChunkConfig.abbreviations`. `ChunkConfig.mid_sentence_period`
selects `"hold"` (the two conditions above, the default) or `"break"` (every
period is a boundary), so a pack can name the rule it was measured under. Both
fields are hashed into the algorithm fingerprint.

**The thirteen single-letter entries earn their place.** They hold a period
after any lone capital, which is what an initial looks like. That is the same
shape as the "short capitalised token" rule the survey rejected for wrongly
holding 57 genuine sentence ends, so it was measured. On five English
Gutenberg books (1087 paragraph-sized passages), dropping all thirteen raises
chunks that end on a dangling initial from 5 to 47 and lowers chunks that end
on a comma from 3019 to 3009: 42 dangling initials prevented against 10 comma
cuts caused. Per letter, only `D` (+36 dangling initials if dropped), `S` (+5)
and `J` (+1) change anything on this corpus. The other ten are inert on it and
may matter on prose it does not contain. A comma cut counts for more than its
number: a chunk that ends on a comma invites an overlong pause, which is why
`split_in_half` does not seek commas.

### Two traps for anyone amending the rule

**Numbers are already expanded.** `speech_text("Der 2. Weltkrieg", "de")` gives
`"Der zwei. Weltkrieg"`. The splitter never sees the digit, so a condition
keyed on a digit in front of the period is dead code in this pipeline. (The
number pass reads the cardinal *zwei*; see [Ordinals](#known-difficulties).)

**`expand_abbreviations` is case sensitive.** A sentence-initial `Bijv.`,
`F.eks.`, `T.ex.` or `Etc.` is not expanded and does reach the splitter.

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

The six added word breaks are the cost. The alternative, taking a held period
when a window has no other punctuation, saves those six and costs seven period
breaks, five of which leave a title dangling at the end of a chunk. A word
break is joined with a carried prefix, so the sentence continues across the
join. A false full stop is read as a full stop, with the closing fall and the
pause.

The list is the surveyed union plus `Dr` and `St`, which that corpus did not
contain and English prose does. On 24 books of it, chunks ending on `Mr.`,
`Mrs.`, `Dr.` or `St.` go from 4115 to 29, and chunks of 12 characters or fewer
from 2273 to 1946. The 29 that remain are the word-boundary fallback, which is a
separate decision.

## Out of scope

These were considered and left out.

**A mandatory phonemizer.** Measured evidence says a *mismatched* one is worse
than none: a controlled study found a phoneme front-end underperforming
characters at nearly every training budget, because the phonemizer's dialect
did not match the corpus's. The standard open phonemizer's own error rate
against a pronunciation-dictionary gold standard is in the mid-teens for
English and German. It is also GPL-3.0. The common alternative is an optional
inline pronunciation override.

**Prosody and break prediction.** Published phrase-break prediction stays under
60% phrase-level F1. The funnel keeps the prosodic punctuation instead, which
carries much of this signal.

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

**A folded numeral goes through the later passes as its ASCII text.** `Ⅳ`
becomes `IV`, and the Roman-numeral pass reads *four*. Fractions differ from
typed text: the fold writes `½` as `1÷2`, which reads *one divided by two*,
while a typed `1/2` has no fraction bar and reads *one two*.

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

A digit glued to a letter stays written: `iOS18` stays `iOS18`, and
`12.03.2026一二三` keeps its digits. A run glued to a word is part of that word,
in all five implementations, and the model reads it as written.

## Known difficulties

Each of these needs something this repository does not have, and no guess
ships in its place.

**Polish grammatical case.** The grammar has Polish gender forms
(*dwa / dwie / dwaj*), and the fixture tests them. It does not select case.
`w 2026 roku` needs the locative *w dwa tysiące dwudziestym szóstym roku*; the
funnel reads *w dwa tysiące dwadzieścia sześć roku*. From 5 up, a numeral
governs the genitive plural while it makes the whole phrase neuter singular.
**Selecting the case requires the syntactic role of the number in its
sentence**, which means a morphological tagger: a different component with a
different failure mode. The published recipe is a tagger plus a transducer
plus an inflected lexicon. The API has a `gender` argument and no `case`
argument; a `case` argument would take the same shape. Dates are the exception,
because the date rule fixes the case: `5 maja 2026` reads *piątego maja dwa
tysiące dwudziestego szóstego*.

**Years.** `1892` is *eighteen ninety-two* as a year and *one thousand eight
hundred and ninety-two* as a quantity, and **nothing in a bare number says
which**. The funnel reads a bare number as a quantity. Inside a recognised date
the year follows the language's year rule: `May 5, 1892` reads *May fifth
eighteen ninety-two*.

**Ordinals.** German `3.` is *dritte*, and its ending depends on the preceding
preposition or article: `am 5. Mai` is *fünften*, `der 5. Mai` is *fünfte*.
Inside a date the funnel selects the ending from the word in front
(`oblique_triggers` in `numbers.json`: `am`, `vom`, `zum`, `seit`, `bis`,
`den`, `ab`). Outside a date it does not: `zum 3. Mal` reads *zum drei. Mal*.
Open WFST grammars for German emit all five endings as alternatives and leave
the choice to a language model. The missing component is the same as for
Polish case.

**Ambiguous dates and colloquial times.** The date rules are per language and
implemented: English, German, Danish, Polish, Finnish, Norwegian and Swedish
take ordinal days; Dutch, Spanish and Portuguese take cardinal ones (`5 de
mayo` is *cinco de mayo*); French and Italian take a cardinal except for the
first. What stays open is **ambiguous input**: `3/14` is a date in one
convention, a fraction in another and a ratio in a third. Resolving it needs a
locale the funnel is not told, so `3/14` reads *three fourteen*. Clock times
read as hour and minute; the funnel does not produce colloquial forms such as
Dutch `half drie` (2:30).

**Danish, Dutch and Portuguese verification.** These three grammars were
written from reference descriptions and pass the fixture, but no native speaker
has checked them, and the literature covers Danish and Dutch thinly. Danish
numerals are vigesimal (*tres* for 60, *halvfjerds* for 70), and a widely used
library writes *treds* for 60, which is wrong in every number that contains it.
**Have a native speaker check these three before any public claim about
them.** An hour each is enough.

**Portuguese variant.** European and Brazilian Portuguese diverge in ways
spelling does not mark, and the number grammar here is European (*dezasseis*,
not *dezesseis*). State the variant wherever the voice is described: the ASR
round-trip does not separate the two variants (see
[evaluation.md](evaluation.md)).
