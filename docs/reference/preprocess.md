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

`loudkit.numbers` verbalizes integers and decimals in all twelve languages it
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
(`"funnel-2"`), unconditionally, in every implementation. There is no opt-in and
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

`tools/fuzz_parity.py` is green on twenty seeds and roughly nine thousand
generated sentences across all five implementations, so the fuzz-parity job in
`ci.yml` gates rather than reports. Swift is not in the CI command: `LoudKitText`
imports CryptoKit and OSLog and `LoudKit` imports CoreML, none of which exist on
the hosted Linux runner. Swift is fuzzed on macOS before merge.

**One question the rule does not answer yet.** It is a question about the rule
rather than a divergence. `200 0003.14` reads *two hundred zero zero zero three
point one four* in all five. The maximal run there does not reduce to a single
readable number, so the rule as stated above argues it should be left written
the way `200 0001e-3` is. The ports read it digit by digit instead. That is
consistent across the five and so invisible to the fuzzer, which compares them
against each other. Deciding it means deciding whether digit-by-digit is a
*reading* or a *refusal*, and that choice belongs with the ear tests rather than
with a parity gate.

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
