# Preprocess: turning text into something the model has seen

The first stage of `preprocess → tts → postprocess`. It removes the gap between
what a person writes and what the checkpoint was trained on.

The model reads **graphemes**. It has no phoneme layer, no lexicon, and no way
to ask what a character means. Anything the training transcripts did not contain
arrives as a dead embedding, or does not arrive at all.

---

## The failure it exists for

A digit is not mispronounced. It is **missing**.

A checkpoint trained on normalised transcripts has never seen `45`. Peer systems
show the shape of this. StyleTTS2's symbol table contains no digits. F5-TTS maps
every unknown character to a space. Neither reports it. The sentence comes out
with a hole where the number was.

Number expansion is therefore the first thing this layer owes, ahead of anything
to do with pronunciation quality.

---

## The stages, in order, and why that order

```
NFC  →  invisibles  →  symbols  →  footnotes  →  acronyms  →  dates
     →  ordinals  →  abbreviations  →  times  →  numbers  →  punctuation
     →  respelling  →  collapse
```

Every stage runs in all five implementations, including dates and ordinals.
See "How it is wired".

**NFC first, before anything inspects a character.** Unicode lets the same
character arrive two ways: Polish `ą` as U+0105 or as `a` + U+0328, Danish `å`
as U+00E5 or `a` + U+030A. The tokenizer's vocabulary holds one of them. Without
normalisation the decomposed spelling arrives as a base letter followed by an
unknown combining mark, and every rule below matches against a string its author
never pictured. NFC must also precede the invisible-strip: normalisation can
compose a sequence into a single character, and a later pass would leave that
composition unexamined.

**Invisibles second.** Zero-width joiners, format characters and the soft hyphen
are not whitespace by Unicode's rules, so nothing later catches them. A grapheme
model reads them as letters, which makes a word that exists in no training text.

**Symbols third, while the digits are still digits.** `£250` reads as *two
hundred and fifty pounds*. The mark is written in front and spoken behind. That
move needs the amount still adjacent to its symbol, so this pass cannot wait.

**Abbreviations, then times, then numbers: fourth.** They run after footnote
markers have gone, because a dropped `[12]` must not become words first. They
run before punctuation, which would turn a decimal separator into a space and
leave `3.5` as two unrelated numbers. All three run in every one of the five
implementations. See below.

**Punctuation last of the destructive passes.** Prosodic marks stay exactly
where they are. They are the model's only route to a question contour or a
clause break. Everything else becomes a space.

**Collapse, and why it is a run and not a pair.** Repeated clause marks fold to
one. Written as a pair it was *not idempotent*: regex substitution does not
overlap its matches, so `...` became `..` on one pass and `.` on the next. A
funnel whose output depends on how many times it was called is a funnel nobody
can predict. The shared fixture had recorded the broken value, an expectation
captured from the implementation rather than written from intent.

---

## Numbers

`loudkit.frontend.numbers` verbalizes integers and decimals in all twelve languages it
has grammars for. That is not the ten that voices ship for, which is a
different set for a different reason. The grammars are **written from first
principles rather than depended on**, for two reasons.

The obvious library is LGPL-2.1, which this kit cannot carry into every
embedding it is meant for. It also has **no case or gender machinery for Polish
at all**, one nominative form per numeral, while shipping six-case declension
for Russian in the same release. Polish is where this matters most, so the
dependency would have to be replaced for the hardest language anyway.

### The grammar is data; only the interpreter is code

Twelve languages × five implementations is sixty chances for a rule to drift.
One JSON file read by five small interpreters is one chance, and the fixture
catches it. The format follows the shape these systems actually have, a regular
generative core plus a listed set of irregular forms, because that is how their
own reference works describe them and a listed form can be checked by eye.

Three things the format had to grow, each a real phenomenon:

| | what it is | why it is a field |
|---|---|---|
| **combining forms** | German says *eins* to "how many" and *ein* in every compound: `einhunderteins` carries both in one word | position, not gender. Folding it into gender would make callers pass a gender for something gender has no part in |
| **round-hundreds joiner** | Portuguese *mil **e** oitocentos* (1800) but *mil oitocentos e noventa e dois* (1892) | not derivable from magnitude; the rule is about the shape of the remainder |
| **agreement outranks a listed form** | Spanish lists *veintiuno*; the feminine is *veintiuna* | a citation form that wins over agreement silently un-inflects the number |

### Past the largest scale it refuses

The recursion would stack scales into "a million milliards", which is not what
any of these languages calls the number. A value that large in running text is
an identifier, not a quantity. `cardinal` raises and lets the caller decide.
`expand` reads it digit by digit and never raises, because its user's sentence
must still be spoken. A library call has a caller; a text funnel has a user.

### What the fixture is

100 cardinals and 9 agreement cases, **hand-written from each language's own
description rather than captured from this implementation's output.** Writing
them that way caught two defects while the file was still being written: Danish
*entusind* for 1000, and the Spanish precedence bug above. A fixture that
records what the code does proves only that the code is deterministic.

### How it is wired

Numbers, abbreviations and clock times are connected, in every implementation.
They were held back until the four ports had them, because **this funnel is
algorithm-layer**. Five implementations run the same shared fixture. A
Python-only expansion makes them produce different text while the fingerprint
goes on declaring they agree, which is the defect class the fingerprint exists
to catch.

That discipline was not kept once. NFC, acronyms, dates and ordinals landed in
Python alone and stayed there while all five went on reporting `funnel-1`: one
fingerprint over four different funnels.

**All four are ported now.** `TextConfig.recipe` is `FUNNEL_PORTED`
(`"funnel-6"` since the marks that carry meaning stopped being dropped),
unconditionally, in every implementation. There is no opt-in and
no `FUNNEL_EXTENDED`. The `divergent` block in
`tests/data/conformance/speechtext.json` is empty of cases and says so. What
used to sit there moved into `cases`, where every implementation is held to it.

---

## Which language this layer runs as

Every rule above is per-language, so "which language" is a preprocess question
before it is anything else. The answer is a chain of three, and it is the same
chain in all five implementations:

1. **the `language` argument**, if the caller gave one;
2. **`voice.language`**, the language the profile was enrolled from, if it is
   not empty;
3. **`"en"`**.

The default used to be `"en"` outright.
`engine.synthesize("Cześć", polish_voice)` then read Polish text through the
English frontend: English number words, English abbreviation expansion, no
Polish respelling. Nothing said so, because a wrong-language read is a
plausible-sounding read. A profile has
always recorded its own language for provenance, so consulting it costs nothing.

Pass `language` explicitly to request **cross-lingual** synthesis, such as an
English voice reading Polish text. The argument always wins over the profile.

| implementation | absent means |
|---|---|
| Python | `language=None` (the default) |
| Swift | `language: nil` (the default) |
| Rust | `language: None` |
| TypeScript | omit the argument |
| Go | the empty string |

Go cannot tell an omitted argument from an explicit `""`. An explicit `""` there
reaches the voice's language rather than tagging the text `[]`, the better of
the two behaviours available to it.

**This does not retrofit existing profile files.** Every loader defaults a
*missing* `language` header key to `"en"`, so a profile written before its port
read the field back loads as English rather than as blank and inheriting
nothing. A non-English voice
from an older writer needs an explicit `language` argument, or a re-save. Step 3
is only reached by a profile built in memory without a language, or a header
hand-edited to `""`.

### Which languages resolve at all

The chain's output is checked against an **allowlist**: the twelve ids
`loudkit.frontend.numbers.supported_languages()` reports, which is the roster in
`models/data/numbers.json` that all five implementations already load. Anything
else is refused, and the refusal names the twelve.

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

## Properties, not just cases

The conformance fixture pins what the funnel does to thirty specific strings.
The properties below hold for *all* strings, and they catch a different class of
defect: not "this rule is wrong" but "this rule quietly ate something".

| property | what it prevents |
|---|---|
| **charset closure** | a character the funnel emits and the tokenizer does not know is dropped or mapped to index zero, and nothing reports it |
| **output is NFC** | the same, one layer up: a decomposed character reaching the vocabulary as an unknown mark |
| **idempotence** (`f(f(x)) == f(x)`) | a rule that fires on its own output will eventually fire on text a user wrote |
| **chunk-join invariance** | splitting is a view of the text, not an edit of it |

---

## What this layer refuses to read

The rule underneath all of these: **a token read half way is worse than a token
left written.** Digits that reach the model are digits a listener hears as
digits. A number half-expanded, a name with a letter missing, or the wrong
currency is a confident wrong answer, and nothing in the audio says so.

**Checked by ear, not only argued.** Refusing hands the reading to a model
nobody controls, and there are inputs where the old half-expansion would have
sounded better. So both directions were rendered at the same seed and voice:
`x200 000` against "x200 zero zero zero", and `Müller123` against "em el el e er
jeden dwa". The refusals won clearly.

If this is ever revisited, listen for whether a listener can tell that something
was dropped, not for which reading is prettier. "em el el e er jeden dwa" sounds
like a complete, correctly read name. It is a different name, and the audio
contains no evidence of the `ü` and the `3` that went missing.

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
| `24:30` | left written | 24 is an hour only with a zero minute (ISO 8601 writes end-of-day as `24:00`). |
| `[12]`, `💩`, `©®™` | `ValueError` | The funnel removes all three, and an empty string used to reach the tokeniser and come back as a `Result` with audio in it. |

Two of these are *not* refusals. Arabic-Indic digits (`١٢٣`, `٣٫١٤`) are
**read**, folded to ASCII beside NFC, because they are digits, and eleven of the
twelve languages used to pass them through untouched while Polish read them.
`1 234 567` grouped correctly is read as a cardinal. The refusals above are
about runs that only *look* grouped.

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

The abbreviation stage expands only the unambiguous entries, and says so: *a
wrong expansion is worse than a spelled abbreviation*. `e.g.` becomes *for
example* and its period is gone. `St.` is Saint or Street, so it is left
written, and its period reaches the next stage down the pipe.

That next stage is the splitter, which breaks long text at `. ` because a break
at a full stop is inaudible. A period that closes a title is not a full stop.
`"But Mr. Smith went home"` used to produce the chunk `"But Mr."`: seven
characters, its own utterance, its own derived seed, and a token ceiling
proportional to seven characters. Over a 9920-row rendered census it is the only
chunk that hit that ceiling.

**The fix is in the splitter, not here, and that placement is the load-bearing
decision.** The funnel could hold these phrases together in two ways, and both
are worse:

* **Expand them.** That is the stage above, and it already refuses these
  entries by name. Adding `St.` to it to fix a chunk boundary would trade an
  audible boundary for a wrong word.
* **Mark them.** A marker character survives into the string the tokenizer
  reads, so it either reaches the model or needs stripping in five
  implementations. The funnel must not mangle text that is later read aloud.

The boundary is not a property of the text. It is a decision about where a
window ends, it depends on the token budget, and nothing else in the pipeline
needs to know it. It belongs where it is made.

### What the splitter tests

Two conditions, and only one of them is a list. Both are gated on the period:
`!` and `?` are unmistakable sentence ends and `;` and `,` are not sentence ends
at all, so the whole question is about the one mark written for two jobs.

1. **The next character is an ASCII lowercase letter.** A sentence does not
   resume in lower case. This is the condition that catches what this layer
   manufactures: `speech_text` maps an ellipsis to `...` and then folds a run of
   `[.,;:]` to one mark, so `"grzeja sie... cieplem"` reaches the splitter as
   `"grzeja sie. cieplem"`. It is the dominant cause in Polish, which has no
   abbreviation cuts at all, and no list of any size would reach it.
2. **The text up to the period ends with a listed abbreviation**, at a word
   boundary: the character in front of the match is absent, or is not an ASCII
   letter or digit. `NASA.` ends in `A`, and `A` is a listed initial, so the
   guard is what keeps that one breaking.

The list is `ChunkConfig.abbreviations` and the law above it is
`ChunkConfig.mid_sentence_period`, both hashed into the algorithm fingerprint.
`"break"` is the law as it stood before this field existed, kept namable so a
pack can say which one it was measured under.

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

**Ordinals are already expanded.** `speech_text("Der 2. Weltkrieg", "de")` gives
`"Der zwei. Weltkrieg"`. The splitter never sees the digit, so a condition keyed
on a digit in front of the period is dead code in this pipeline.

**`expand_abbreviations` is case sensitive.** A sentence-initial `Bijv.`,
`F.eks.`, `T.ex.` or `Etc.` is not expanded and does reach the splitter.

### Why the tests are ASCII

A rule that asks whether a character is a lowercase letter has five answers in
five languages, and `islower`, `unicode.IsLower`, `char::is_lowercase` and
`Character.isLowercase` are not the same predicate. Measured on the surveyed
corpus: of 2253 periods in ten languages, four are followed by a word starting
with a non-ASCII lowercase letter, and reading the whole Unicode `Lowercase`
property instead moves one passage in 1200. Two comparisons that five
implementations cannot disagree about are worth more than those four.

### What it is worth

Surveyed over 1200 passages in ten languages, 4973 chunks:

| | cuts on a period inside a sentence | chunks of 20 characters or fewer | word-boundary breaks |
|---|---|---|---|
| before | 59 | 98 | 144 |
| after | 1 | 93 | 150 |

The six added word breaks are the cost. Taking the held period after all when a
window has no other punctuation was measured and rejected: it saved those six
and cost seven period breaks, five of which left a title dangling at the end of
a chunk. A word break is joined with a carried prefix and keeps its contour. A
false full stop is read as one, with the closing fall and the pause.

The list is the surveyed union plus `Dr` and `St`, which that corpus did not
contain and English prose does. On 24 books of it, chunks ending on `Mr.`,
`Mrs.`, `Dr.` or `St.` go from 4115 to 29, and chunks of 12 characters or fewer
from 2273 to 1946. The 29 that remain are the word-boundary fallback, which is a
different decision.

## Out of scope

Each of these was considered against evidence and left out on purpose.

**A mandatory phonemizer.** Measured evidence says a *mismatched* one is worse
than none: a controlled study found a phoneme front-end underperforming
characters at nearly every training budget, because the phonemizer's dialect did
not match the corpus's. The standard open phonemizer's own error rate against a
pronunciation-dictionary gold standard is in the mid-teens for English and
German. It is also GPL-3.0. The field's convergent 2026 answer is an *optional
inline override*, not a pipeline stage.

**Prosody and break prediction.** State-of-the-art phrase-level F1 is under 60%.
Punctuation already carries this, which is why punctuation survives the funnel
untouched.

**POS tagging as a stage.** It pays off only as a homograph feature, and the
measured end-to-end contribution of homograph handling in a full front-end is
about a third of a percentage point.

**Neural or LLM normalization.** In-domain it beats rules; on ambiguous input it
collapses. Low tolerance for unrecoverable errors is why production inverse-text
systems are still largely rule-based.

---

## One class question, asked the same way five times

Every pass in this funnel asks some version of "is this a letter, a digit, or a
space". The five implementations spelled those three questions eleven different
ways, and each difference was audible. They are one written-down answer now.

### Numerals

A number character with no ASCII spelling — a superscript, a vulgar fraction, a
circled or Roman numeral, a digit from any script but Latin — used to be
**deleted** by the punctuation pass, and while it was still there it sat inside
the word, because `\w`, `\p{N}` and `str.isdigit()` all admit `No`. So `²9`
was one token, the number matcher declined it, and a bare `9` reached the
model.

`fold_numerals` runs before the symbol table and turns each of them into
something a reader says aloud:

| input | before | now, in all five |
|---|---|---|
| `²9` | `9` | `two nine` |
| `Add ½ cup` | `Add cup` | `Add one two cup` |
| `Chapter Ⅶ.` | `Chapter.` | `Chapter vee-eye-eye.` |
| `5৩3` | `5৩3` | `five hundred and thirty-three` |
| `１２３` | `１２３` | `one hundred and twenty-three` |
| `12 m² and $9` | `twelve m and 9 dollars` | `twelve m two and nine dollars` |

**What each character becomes is a table**, `models/data/numerals.json`,
generated by `tools/make_numerals.py` and copied to every port. A decimal digit
of any script becomes the ASCII digit of the same value and **joins the run it
was in**, so a number written across two scripts is still one number; every
other number character becomes the text the table names — `²`→`2`, `½`→`1/2`,
`③`→`3`, `Ⅳ`→`IV`, `፲`→`10` — separated from an adjacent alphanumeric so `²9`
does not fold into `29` and read as twenty-nine.

That text is ASCII for all but 27 of the 1,151 entries, and those 27 are named
here rather than rounded off: every one decomposes to CJK ideographs — `〸` →
`十`, and ten in parenthesised form, `㈠` → `(一)`. There is nothing more ASCII
to say about them, so they leave the fold as the characters they mean, which
the passes downstream treat as letters. No entry decomposes to Hangul; an
earlier revision of this page said so and was wrong.

**Whether a character is a numeral comes from the same table.** That is not a
detail of the implementation: `unicodedata`, `unicode.IsNumber`, ICU and V8 each
carry the Unicode version their runtime was built with, and this project's CI
runs Python 3.10 through 3.14 — 15.0.0 and 16.0.0. While detection came from the
runtime, `Add 𐵁 cups` (GARAY DIGIT ONE, added in 16.0) read *add cups* on 3.12
and stayed written on 3.14, both reporting the same `algorithm_fingerprint`, and
neither said *one*. The table is cut from one pinned UCD — `EXPECTED_UNICODE` in
the generator, refused if the interpreter carries another — so every runtime
reads it the same way, including runtimes older than the table.

A table rather than each runtime's own Unicode, because both ways of computing
it were wrong and each was wrong in silence:

- **Walking down to a decimal block's start** leaves its own block where two
  blocks touch. `MATHEMATICAL DOUBLE-STRUCK DIGIT ZERO` follows
  `MATHEMATICAL SANS-SERIF DIGIT NINE` with nothing between them, so a zero
  read as a nine, in all five ports, and the shared fixture pinned the wrong
  answer.
- **NFKC does not reach every numeral.** `ETHIOPIC NUMBER TEN`, the Aegean
  numbers, the Kaktovik digits and the Meroitic numerals have no compatibility
  decomposition, so they came through unchanged and the punctuation pass then
  deleted them — `Add ፲ cups` read as *add cups*.

The table is hashed into `TextConfig.grammar` beside the grammar and the
lexicon, which also pins the *Unicode version* the fold uses: two builds that
hash the same file fold the same way whatever their ICU says.

**A folded numeral reads as its ASCII spelling reads.** `½` becomes `1/2` and is
read the way `1/2` is read; `Ⅳ` becomes `IV` and is read the way `IV` is read.
That is the rule, and it is why this pass needs no verbaliser of its own in five
languages.

`Lo` is untouched. Ideographic numerals — `一`, `十` — are letters, and a pass
keyed on numeric type rather than on the general category would delete a
language's numerals. `一二三` is a fixture case for that reason.

### Letters

`\p{L}`, everywhere. Rust and Swift tested the **Alphabetic** property, which
is wider by the circled letters (`So`), the Other_Alphabetic marks and the
letter numbers: Rust kept `ⓐ` where three ports spaced it, and Swift's copy
made a boundary fail and left a **bare digit** in front of the model for
`ⓐ-1`. Swift also read `CharacterSet.letters`, which Foundation answers wrongly
above the BMP — it omits all 6,145 Tangut ideographs, so that text was deleted,
and `subtracting(.nonBaseCharacters)` silently fails for astral scalars, so
1,162 combining marks leaked through. Both now read the general category.

JavaScript's walks indexed UTF-16 units, so an astral letter arrived as a lone
surrogate that `\p{L}` does not match and the walk stepped straight past it:
`0.𗀀` read as *zero* here and stayed written everywhere else. They read code
points now.

### Whitespace

`WHITE_SPACE` is Unicode White_Space written out, and every pattern in the
funnel uses it. `\s` was four different sets: ECMAScript's excludes U+0085
NEL, RE2's is ASCII and omits U+000B, CPython's uniquely admits U+001C–U+001F.

Go **read a footnote marker aloud** when its separator was a non-breaking or
thin space, which is ordinary French and German typography, and missed a
currency mark separated by a vertical tab. JavaScript split chunks in different
places than the other four, because a NEL left on the front of the remainder is
charged against the next chunk's budget — measured, 51 of 500 fuzz cases.

### Boundaries

`(?![\p{L}\p{Nd}_])`, written out rather than `\b`. ECMAScript's `\b` is
ASCII and Go's byte-wise test was too, so a date glued to a CJK ideograph was
spoken by those two and left written by the other three.

### What still reaches the model unread

A digit glued to a letter stays written: `iOS18` reads as *iOSeighteen*, and
`12.03.2026一二三` keeps its digits. That is deliberate and long-standing — a
run glued to a word is part of that word — and it is now at least the same
deliberate answer in all five.

## Known difficulties

Each of these needs something this repository does not currently have. Shipping
a guess would be worse than shipping nothing.

**Polish grammatical case.** Gender is done: *dwa / dwie / dwaj* are in the
grammar and tested. Case is not, and it is the larger half. `w 2026 roku` needs
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

**Danish, Dutch and Portuguese verification.** The grammars here were written
from reference descriptions and are internally consistent, but they have not
been verified by native speakers, and the literature covers Danish and Dutch
particularly thinly. The Danish numerals are vigesimal (*tres* for 60,
*halvfjerds* for 70), and at least one widely-used library ships *treds* for 60,
which is wrong in every number containing it. **These three want a native
speaker's eye before the claim is made in public.** An hour each with a native
speaker is the highest-value verification available.

**Portuguese variant.** European and Brazilian Portuguese diverge in ways
spelling does not mark, and the number grammar here is European (*dezasseis*,
not *dezesseis*). That is a decision, not a default. State it wherever the voice
is described, because no automatic metric can detect the mismatch.
