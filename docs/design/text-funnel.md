# The text funnel

Maintainer notes for `python/loudkit/frontend/`: the reasoning and the
measurements behind each symbol. Each heading names a module and a symbol, and
the runtime docstrings point here. These notes are not user documentation.
[preprocess.md](preprocess.md) describes the layer as a whole.


## `loudkit/frontend/__init__.py`


### `module`

`TextConfig.recipe` and the grammar digest put what this package does into the
algorithm fingerprint. What the digest covers, and what it does not, is under
`textconfig.py` below.

Import the passes from their submodules (`loudkit.frontend.numbers`,
`loudkit.frontend.speechtext`, ...). The package `__init__` imports nothing,
because `loudkit.config` and the chunking pass import each other, and an eager
`__init__` would turn that into a circular import.


## `loudkit/frontend/chunking.py`


### `module`

A window carries about 255 speech tokens, roughly ten seconds. Longer text is
split, generated in pieces and joined. Where the splits fall is audible, so the
split rule is part of the algorithm.

The splitter breaks at the strongest punctuation available, as late as
possible. A full stop is the least audible place to break; a break mid-clause
is heard. A single sentence too long for one window breaks at its best comma,
and failing that at a word boundary. A word-boundary break is audible, but no
text is lost.

A period is the one ambiguous separator: the splitter treats `!` and `?` as
sentence ends and `;` and `,` as clause marks. In `"But Mr. Smith went home"`
the period ends a title. Breaking there hands the renderer the chunk
`"But Mr."`: seven characters, its own utterance, its own derived seed, and a
length-proportional token ceiling that leaves no room for a closing pause. In a
9920-row rendered census it is the only chunk that hit that ceiling.

`ChunkConfig.mid_sentence_period` says what to do at such a period. Two
conditions find one. The first is the `ChunkConfig.abbreviations` list
(*Mr.*, *Hr.*). The second is a next word that starts in lower case, which
catches what the funnel produces: an ellipsis folded to a single period. That
is the main cause in Polish, and no list reaches it. The conditions and their
measurements are in [preprocess.md](preprocess.md#what-the-splitter-tests).

When every candidate in a window is held, no usable punctuation is left, and
the splitter breaks at a word boundary. Taking the held period instead was
measured and rejected: over 1200 passages it saves six word breaks and costs
seven period breaks, five of which leave a title dangling at the end of a chunk
("...forte Hr." then "Theodor Franz..."). A word break is joined with a carried
prefix. A false full stop is read as a full stop: closing fall, pause, and a
new sentence begun in the middle of the old one.

Splitting counts characters, not tokens, because it runs before the
tokenizer, so the token budget is an estimate. The estimate is conservative. A
chunk that overflows the window stops at the cap mid-word and loses its tail
(see `split_in_half` and `CHARS_PER_TOKEN`). A chunk slightly too short costs
one more join.


### `_holds`

`look` is the search window plus one character, because the test reads the
character *after* the separator, and the latest candidate can end the window
exactly.

The test applies only to a period. The splitter treats `! ` and `? ` as
sentence ends and `; ` and `, ` as clause marks, so the question is only about
the one mark written for two jobs.

The lowercase test is ASCII only. Measured over 2253 periods in ten languages:
four are followed by a word that starts with a non-ASCII lowercase letter, and
reading the whole Unicode Lowercase property instead moves one passage in 1200.
An ASCII test gives the same answer in all five implementations; `islower` and
its counterparts in the ports do not.


### `split_text`

Arguments:

- `text`: the passage to read.
- `config`: a `ChunkConfig`, the chunking policy from `AlgorithmConfig`.

Returns the chunks in order, each stripped of surrounding whitespace, together
covering the input. The list is empty only when the input is empty or all
whitespace.

```python
>>> from loudkit.config import ChunkConfig
>>> split_text("One. Two. Three.", ChunkConfig(max_tokens=12, prefix_tokens=0))
['One.', 'Two.', 'Three.']
```

A chunk that fits no separator breaks at the last word boundary. The break is
audible, but the text is spoken. The boundary set is `WORD_BOUNDARIES`, not
U+0020 alone: NBSP survives the funnel and is ordinary in real prose, and text
whose every space is non-breaking (HTML full of `&nbsp;`) would otherwise find
no boundary and be cut mid-word.


### `split_in_half`

`split_text`'s estimate is conservative but not a guarantee. It budgets
characters against a constant, and a speaker slower than that constant fills
the window before the text runs out. The generator then stops at the cap
mid-word, and the words that did not fit are **lost**, because chunk texts are
fixed before any of them is rendered. With `ChunkConfig.cap_resplit = "word"`
(the default), such a chunk is halved here, and both halves are generated
under the same chunk index.

Measured across ten languages and 9920 chunks: 54 reached a token cap and 30
were still speaking when it closed, over five voices. Counting only the window
cap, the figures are 51 and 27, over two voices.

The boundary is the nearest *word* break, and punctuation is not sought. A
comma is an instruction to pause, this model has no pause-duration prior, and
given a comma at a chunk end it overshoots the pause. No valid measurement
compares punctuation with word boundaries here: the only one used a harness
that seeded the second half from the next chunk's stream, and its figures are
withdrawn. The current measurement compares this rule with no rule. The
nearest word break can still fall after a comma.

Returns `None` when the chunk has no interior word boundary (a single unbroken
run of characters), or when stripping would leave an empty half. Splitting
cannot help such a chunk, and cutting inside a word would mangle it, so the
truncation stands.


### `WORD_BOUNDARIES`

The characters `split_in_half` may cut on, written out: space, tab, LF, CR,
NBSP (U+00A0), figure space (U+2007) and narrow no-break space (U+202F). A
predicate would be shorter, but the runtimes disagree on one. Measured:
Python's `str.isspace()` treats U+001C–U+001F as whitespace where the other
four do not, and JS alone keeps U+0085 where the other four strip it. Swift
alone strips U+200B and JS alone strips U+FEFF; the funnel removes both before
the splitter sees them, but NEL (U+0085) survives it. A disagreement there is a
different split point, which is different audio for the same text and seed. A
written-out tuple gives every implementation the same set.

The funnel does not remove NBSP. It is ordinary in real prose ("10 000", French
punctuation, typeset copy), and `split_text` treats it as whitespace when it
strips a chunk, so a capped chunk whose only boundaries are NBSP still splits.


### `CHARS_PER_TOKEN`

0.5 characters of prepared text per speech token. Measured on the reference
voice across English, Polish (after the respelling funnel) and German: 0.53 to
0.64 characters per speech token, consistent with about 25 speech tokens/s at
14 to 16 characters/s of narration. The constant sits below the measured
minimum (0.5 < 0.53) for margin. The margin costs slightly more, slightly
shorter chunks.

**The constant is a budget, not a guarantee.** Measured over 9920 rendered
chunks in ten languages, 54 reached a token cap anyway, 51 of them the window
itself. Only one roster voice reads below the constant (soren, 0.481). Of the
54, only 17 carry enough characters to need more than 255 tokens at their own
voice's median pace. The rest should have fitted: the model did not emit a stop
token in time, and a per-voice average cannot bound that variance. An overflow
does not raise a `ValueError`: the generator stops at the cap mid-word and the
rest is never spoken. `ChunkConfig.cap_resplit` and `split_in_half` exist for
that case.


## `loudkit/frontend/dates.py`


### `module`

Each language here writes and says dates its own way:

- The day is an ordinal in English, German, Danish, Polish, Finnish, Norwegian
  and Swedish, and a cardinal in Dutch, Spanish and Portuguese. French and
  Italian use a cardinal for every day *except* the first.
- The month is nominative in most, genitive in Polish (`marca`, never
  `marzec`) and partitive in Finnish (`maaliskuuta`).
- Spanish and Portuguese speak a preposition between every part.
- The year splits into halves in English and Norwegian, and groups in hundreds
  in German, Dutch, Swedish and Danish (Danish in its long form, *nitten
  hundrede og fireogfirs*). Polish reads it as an ordinal in the genitive, and
  Spanish, French, Italian, Portuguese and Finnish as one plain cardinal.

These forms are written out, not derived. The day words are in `numbers.json`
per language, from the national authority: the five English irregulars,
German's `siebte`/`achte`, Danish `ellevte` (Retskrivningsordbogen withdrew
`elvte`, which some word lists still carry), Italian `ventotto` (never
*ventiotto*), and Finnish's ordinal suffix repeated inside every part of a
compound. `tests/test_dates.py` pins these forms and names the authority where
a plausible-looking form is wrong.

**Why dates run before times and numbers.** `12.03.2026` is the ordinary
written date of German, Polish, Danish, Finnish and Norwegian, and both later
passes match part of it: the clock pattern matches `12.03`, and the digit-run
pattern the whole run. The date pass runs first, so the date is read as a
date.

**What it refuses.** A version number, an address and a score can all look
like a date to a permissive matcher, and reading one as a date is worse than
leaving it: `1.2.3` must never become *the first of February, three*. Every
candidate is bounds-checked: the day against the month's maximum (February
counts 29, because this is a plausibility bound, not a calendar), the month
against twelve, and a four-digit year against 1000–2999. In English a dotted
numeric date is always left as written, and a slashed one is left when its
first field is 12 or less, because US and UK writing put day and month in
opposite order: `3/12/2026` could be either. A confident wrong month cannot be
recovered; heard digits can. Swedish dotted dates are not read either
(`no_dotted_dates`): Swedish marks an ordinal with a colon (`1:a`), never a
trailing period, so `12.` there is a list number or a sentence end.


### Numeric dates need a year

`12.03.2026` is a date because the year makes it one. The yearless `12.3.` that
German, Danish, Finnish and Norwegian also write is not matched: its closing
period looks like a sentence end, so `Die Zahl ist 3.5.` would read as *dritte
Mai* in ten of the twelve languages. With no evidence in the string to separate
the two readings, the date pass leaves it. A yearless date written with a month
name still reads, because the name is the evidence: `12. März` reads
*zwölfte März*.


### `ordinal_day`

`ordinal_day(day, language, *, oblique=False)`:

- `day`: 1 to 31.
- `language`: one of `supported_languages()`.
- `oblique`: German only, the `-en` ending that `am`, `den` and the other
  `oblique_triggers` select (`am 5. Mai` reads *am fünften Mai*). Other
  languages ignore it: no other grammar here has an oblique day form.


### `ordinal`

Composed past ninety-nine: the hundreds and above stay cardinal and only the
last two digits become an ordinal, so *101st* is "one hundred first". A value
whose last two digits are 00 (`100th`, `1000th`) has no ordinal and is left as
written. The irregulars a suffix rule gets wrong (fifth, eighth, ninth,
twelfth, twentieth) are all inside the two-digit tables and are written out
there.


### `expand_ordinals`

Only English has ordinal suffixes in the grammar (`st`, `nd`, `rd`, `th`). For
the other eleven languages the suffix list is empty and the pass does nothing,
so French `1er` stays written. The pass runs before the number pass, which
would expand the digits and leave the suffix stuck to them (*onest*,
*twenty-twond*).

A value the tables cannot say is left as written, suffix included.


## `loudkit/frontend/letters.py`


### `module`

`CIA` is *see-eye-ay* in an English render and *ce-i-a* in a Polish one: each
is what that language says. The engine is grapheme-based with a single language
tag per utterance, so the letter name is written in the target language's own
orthography. English `see` reads as /siː/ under English letter-to-sound rules,
and Polish `ce` reads as /t͡sɛ/ under Polish ones. Either spelling inside the
other language's render produces the wrong sounds.

The tables are per language, because letter names are orthography-specific
(Polish `FBI` is *ef-be-i*). A render with no table would reach the model with
raw capitals, which a grapheme engine reads as a word, not as letters. The
tables are data, one per language in `numbers.json`, read by every
implementation.

**What is not spelled.** An acronym that is a word in its language stays a
word, from the `word_acronyms` list of each language:

- `NASA` in every language;
- `NATO` everywhere except Spanish, French and Portuguese, which say `OTAN`;
- `SIDA` and `OVNI` in Spanish, French and Portuguese;
- `PESEL`, `ZUS` and `LOT` in Polish;
- `TUTKA` in Finnish.

The lists are per language because the facts differ by language: Polish reads `LOT`, the
airline, as a word, and English spells it.


### `spell_acronyms`

A run of capitals is emphasis, and it is left alone. The rule that tells a
shout from an initialism reads context, not the word itself. An initialism
usually stands as a single capitalised island in ordinary text ("the CIA
said"); emphasis comes in runs. The word alone cannot separate them: `IT` is a
word, an initialism or a shout depending only on what sits beside it. So a
capitalised word is spelled only when neither neighbour is also capitalised.

A text that is entirely capitals, such as a pasted headline, passes through
whole. A text that is one capitalised token, `synthesize("GPT")`, is spelled:
there is no run to read emphasis from, and the call has exactly the shape of
"say this acronym".


### `spell_acronym`

Consults the word table before the length cap (two to five letters). The cap
limits how long a run of capitals may be before spelling it is worse than
leaving it; it says nothing about a listed word. With the cap first, every
entry over five letters would never reach the table: UNESCO, UNICEF and
INTERPOL.


## `loudkit/frontend/numbers.py`


### `module`

#### Why this exists

The frontend tokenises **graphemes**, and a checkpoint trained on normalised
transcripts has never seen `45`. A digit that reaches the model is at best an
embedding it barely trained, and at worst dropped. The result is silence or
garbage where a number should be, and no acoustic quality repairs it. The
funnel still leaves some digits written on purpose; see `expand`.

#### Why it is written here

The widely used Python library for number words is LGPL-2.1, which this kit
cannot carry into every embedding it is meant for.

That library also has **no case or gender machinery for Polish at all**: one
nominative form per numeral, while the same release ships six-case declension
for Russian. Polish needs agreement most (a published measurement puts a
morphology-blind normaliser at about 30% against about 91% for a
morphology-aware one), so the dependency would have to be replaced for that
language anyway.

#### How it is organised

**The grammar is data; only the interpreter is code.** One JSON file holds the
grammar for all twelve languages, and five small interpreters read it. Each
rule lives in one place, and the fixture checks that the interpreters agree.

The data format follows the shape these systems have: a regular generative
core, plus a listed set of irregular forms. Every number grammar in this set is
units, teens, tens, scales and a composition rule, and the grammars differ in
how the pieces join:

- **order**: German, Dutch and Danish say the unit first (*einundzwanzig*,
  literally "one-and-twenty"); the rest say the ten first.
- **joiners**: Spanish *treinta y uno*, Portuguese *vinte e um*, French
  *soixante-et-onze* but *quatre-vingt-un*, German's bare *und*. French follows
  the 1990 rectified spelling, with hyphens throughout.
- **elision**: Italian *ventuno* and *ventotto* drop the ten's final vowel
  before a vowel-initial unit.
- **agreement**: Polish, Spanish, Portuguese, Danish, Norwegian and Swedish
  have forms that agree with the gender of what is counted. Polish also
  inflects for case, which the grammar does not model.

Irregular *values* are listed outright, not modelled as rules. That is how the
grammars' own reference works describe them, it keeps the interpreter small
enough to port exactly, and a listed form can be checked by eye against a
dictionary.

#### What it does not do

Nothing here reads context. A numeral's case in Polish, or its gender in
Spanish, is a property of the *sentence*, not of the number. `cardinal` and
`expand` take a `gender` argument, where a caller with that knowledge supplies
it; there is no `case` argument. Choosing either automatically needs a
morphological tagger, which is a different component with a different failure
mode.


### `cardinal`

`cardinal(value, language, *, gender=None)`:

- `value`: the integer to say. A negative value is read with the language's own
  minus word; every language in this set has one.
- `language`: one of `supported_languages()`.
- `gender`: the grammatical gender of the counted noun, for the languages whose
  grammar has gendered forms (Polish, Spanish, Portuguese, Danish, Norwegian,
  Swedish). `None` gives the citation form, which is what a bare number in a
  list wants.

Raises `NumberGrammarError` if the language is unknown, or if the value is at or
above 1000 times the grammar's largest scale (10^15 in English). It raises
instead of falling back to digits; `expand` makes the other choice.

```python
cardinal(21, "en")                  # 'twenty-one'
cardinal(21, "de")                  # 'einundzwanzig'
cardinal(71, "fr")                  # 'soixante-et-onze'
cardinal(2, "pl", gender="f")       # 'dwie'
```


### `expand`

Runs *after* the symbol pass, so a currency amount has already become "250
pounds" and only the `250` is left to say. Runs *before* the punctuation pass,
which would turn a leading minus into a space.

It does not raise on an out-of-range value, and it never *half*-reads a number:
a value past the largest scale the grammar has a word for is read digit by
digit. Such a value is almost always an identifier, a code or a serial.
`cardinal` refuses the same value, because a library caller can decide; the
funnel must still speak the sentence.

It leaves some digits written. A run glued to a word is part of that word and
stays written: `iOS18`, `r123`, `v1.2.3`, `5x3`. Expanding only the digits
would give *iOSeighteen*. A dotted run with more than one separator stays too:
`1.2.3` is a version, `192.168.0.1` an address, and `18.08.2026` in English a
date the date pass declines. Digits that reach the model get whatever reading
the model gives them; a confident wrong number cannot be undone.

A separator between digits is a decimal mark only when it is *the* language's
decimal mark. English `3.5` is three point five. English `3,500` is a grouped
thousand, and its comma is dropped, as a reader drops it.


### `fold_foreign_digits`

Applied beside NFC, not in the number pass, because it is a normalisation:
every later pass should see one spelling. It also has to run before the symbol
table, which turns the folded Arabic percent sign (U+066A → `%`) into a word.

The separators depend on the language. U+066B is a *decimal* separator, so it
folds to the language's decimal mark: `٣٫١٤` becomes `3,14` in German and reads
*drei Komma eins vier*. Folded to a dot everywhere, it would read as the clock
time *drei Uhr vierzehn* in the eleven languages that write decimals with a
comma. U+066C, the thousands separator, folds to the other mark.


### `_FOREIGN_DIGITS`

The Arabic-Indic and Extended Arabic-Indic digits, and the Arabic percent sign.
They fold so all twelve languages give one answer. `_DIGIT_RUN` is ASCII, so
without the fold `١٢٣` would reach the model as written in eleven languages,
while Polish, whose respeller tests digits with `str.isdigit()`, would read it.
With the fold it reads *one hundred and twenty-three* in English and *sto
dwadzieścia trzy* in Polish.

The separators matter more than the digits. U+066B is not in the `[.,]` the
number pass looks for, so without the fold `٣٫١٤` would lose its separator and
read as two numbers.


### `_UNICODE_MINUS`

U+2212 MINUS SIGN and U+2010 HYPHEN, when a digit follows, fold to ASCII `-`.
Everything downstream reads the sign as `-`. Without the fold, the punctuation
pass turns the mark into a space, and `−5` reads as *five*: the opposite
temperature.

Only these two, and only before a digit. U+2013 EN DASH writes a *range*
("1979–1983"), and U+2014 EM DASH is punctuation; folding either into a minus
would invent a sign where the text has none. The digit lookahead keeps the fold
off a hyphenated word.


### `_PHONE_RUN`

Read digit by digit, and matched before `_DIGIT_RUN`, because it is the one
shape that pattern cannot decline on its own. `+48 123 456 789` is a valid
one-to-three-then-threes grouping and would read as *forty-eight billion one
hundred and twenty-three million…*. `+1 202 555 0199` would read as a cardinal
with a stray *nine*, because its last group has four digits and only three
fit.

The plus is the evidence: an international telephone number is written with a
leading plus, and a grouped quantity normally is not. This is not a
phone-number *detector*: it does not guess at national formats, area codes or
separators.

`_MIN_E164_DIGITS` (8) keeps it off a signed number. `+5 degrees` and `+250
points` are deltas, and `+1 000 000 users` (seven digits) reads *one million
users*. E.164 allows up to fifteen digits.


### `_TIME_RUN`

`H:mm` and `H:mm:ss`, and nothing that merely contains them. A colon between an
hour and two minute digits is read as a clock time in every language here.

A digit, a dot, a comma or a colon on either side disqualifies the match, so
the pattern cannot read the front of a longer run as a time and leave the rest
as a separate number. A letter may follow (`3:45pm`). `14:30.` at the end of a
sentence still matches, because no digit follows the dot.


### `_DOTTED_TIME_RUN`

`14.30` is how German, Danish, Finnish, Norwegian and Swedish write a time;
`3.14` is how English writes pi. The shapes are identical, and no lookaround
separates them.

The grammar file does: where the dot is not the decimal separator, `H.mm` is a
time. German writes `14.30 Uhr` and `2,50 €`; English writes `2:30` and
`$2.50`. So this pattern applies only where `decimal_separator` is not `.`,
which is every language here except English. In English it would read
two-digit decimals as clock times (`3.14` as *three fourteen*). A price is
handled before this pass, by `_priced`.

The dotted form takes no seconds: `10.30.45` is a version string as readily as
a timestamp.


### `_time_patterns`

German writes the time *with* the word the spoken form also carries:
`um 14.30 Uhr`. The reading puts the infix between hour and minutes, *vierzehn
Uhr dreißig*, so the written `Uhr` is that same spoken token. When the infix
follows the time, with or without spaces between (`14:30Uhr`), the match takes
it, and the reading supplies the one copy. Otherwise the output would say
*vierzehn Uhr dreißig Uhr*.

The pieces are spelled out so five implementations match identically:

- the whitespace between time and infix is ASCII space and tab (regex engines
  disagree on what `\s` covers);
- the guard after the infix refuses an ASCII letter or digit, so *Uhrzeit*
  keeps its word whole;
- the match is case sensitive, because the grammar data says `Uhr`.


### `expand_times`

Runs before `expand`, which would otherwise read the two sides as unrelated
numbers and leave the colon behind. The hour is 0 to 24 and the minutes exactly
two digits, so `3.5` (a decimal) and `1.000` (a grouped thousand) never match.
The dotted form applies only in the languages whose decimal separator is not
the dot; see `_DOTTED_TIME_RUN`.

The reading is hour and minute as cardinals with the language's own infix:
*vierzehn Uhr dreißig*, *neljätoista kolmekymmentä*, *fourteen thirty*.

- A zero minute says the hour alone (`14:00`). Before non-zero seconds it is
  kept, so the seconds stay in their place (`10:00:05` reads *ten zero five*).
- A zero seconds field is dropped: `10:30:00` reads as `10:30` does.
- Hour 24 is a time only with zero minutes and seconds (ISO 8601 writes end of
  day as `24:00`). `24:30` is left to the number pass and reads
  *twenty-four:thirty*.
- A reading followed directly by an ASCII letter gets a space, so `3:45pm` and
  `3:45 pm` both read *three forty-five pm*.
- A written infix directly after the time (German `um 14.30 Uhr`) is taken, not
  doubled; see `_time_patterns`.

The reading is the plain clock, not the colloquial one. *Half three* means 2:30
in German, Dutch, Danish, Norwegian, Swedish and Finnish, and 3:30 in British
English. A time read an hour off is a severe error, so the plain reading stays
until the colloquial one is measured.


### The digit-run pattern (`_DIGIT_RUN`)

- A leading minus is part of the number only where it cannot be a hyphen or a
  range: at a boundary, with a digit right behind it. Anywhere else the
  punctuation pass turns it into a space.
- A space groups thousands only when every group after the first is exactly
  three digits and the first is one to three, so "in 2024 200 people" stays two
  numbers.
- A grouped run must reach a boundary; a partial one is not a grouped number,
  and the pattern declines the whole join, so each group reads as its own
  number.
- `+48 123 456 789` is a valid 1-3-plus-threes grouping that no boundary rule
  saves, so `_PHONE_RUN` takes it first.
- A digit run touching a word at either end is part of the word and is left
  written for the model, as `iOS18` is. The lookahead `(?![\w])` mirrors the
  backward walk, and `(?! ?[0-9])` makes the grouping rule exact. Reading half
  a token (`5x3` as *fivex3*, `1e6` as *onee6*) is worse than either whole
  answer.

Digits of other scripts are folded to ASCII before this pass, by
`fold_foreign_digits` and `fold_numerals`.


### `_starts_a_group`

Whether three ASCII digits start at `i`: the shape `_DIGIT_RUN` binds as a group
after the first, and so the shape a space in front of them may be grouping.

The slice must *have* three characters. `"2"[0:3].isdigit()` is true, so
without the length check a lone digit would pass for a group, and in `R2 2` the
backward walk would cross the space, reach the `R`, and refuse a number nothing
is glued to. `R2 2` reads *R2 two*.

The walks test ASCII digits, the class `_DIGIT_RUN` matches. `str.isdigit` is
also true of `²` and of every Unicode decimal digit.


### `_continues_a_group`

Three ASCII digits with no fourth behind them: a group the pattern could have
*bound*, not a ragged run that only looks like one. The first group may be one
to three digits and says nothing about whether the space groups, so this is
asked of the half whose width `_DIGIT_RUN` fixes.

The forward walk asks `_starts_a_group` and the backward walk asks this. See
`_glued_forward` for why the two directions differ.


### `_glued_to_a_word`

Whether the digit run at `start` sits inside a token that contains a letter.
The walk goes backwards over word characters, dots, commas, `-`, `+` and
grouping spaces, and answers yes on the first letter. Any other character ends
the walk, and the answer is no.

- Dots: an identifier can put a dot between its letters and its digits. In
  `v1.2.3` the scan starts at the `2`, and one character of lookbehind cannot
  see the `v`.
- Commas: `x3,14` is refused whole, not read as "x3,vierzehn".
- Signs: an exponent puts one between the letter and the digits. In `1e-3` the
  scan starts at the `3`, walks back over `-` to `e`, and stops calling it a
  number. A bare `-5` reaches a space or the start and finds no letter.
- Grouping spaces: `x200 000` binds as a single match in Go and Rust, whose
  engines do not backtrack, while Python, JS and Swift backtrack and match the
  standalone `000`. Crossing the grouping space gives all five the same answer:
  the whole token stays written.

A space is a grouping space only when the group behind it continues: one to
three digits behind it, a digit ahead of it. The walk asks
`_continues_a_group`, not `_starts_a_group`, because admitting a fourth digit
here reaches the `e` of `e3 1000` and welds two tokens into one. Simpler rules
fail on real text:

- "a digit on each side" crosses `R2 5`;
- "exactly three digits behind the space" breaks `a1 000 000`, whose first
  group is legitimately one digit;
- crossing without the digit-ahead rule walks `Sold 200 000` from `000` to
  `200` and into "Sold".

The forward direction has its own walk, `_glued_forward`.


### `_glued_forward`

The mirror of `_glued_to_a_word`. `200 000x` must be refused whole. A
backtracking engine would match `200` alone once the grouped alternative
reached the `x` and the right-hand guard refused it, and read "two hundred
000x". Go and Rust, which do not backtrack, leave the whole token written. This
walk makes the backtracking engines give the same answer.

A grouping space is crossed, so `200 000x` is one token. The ordinary space in
`2024 200 people` is not, because what follows *it* is a word, not a digit
group, and those two numbers stay two numbers.

Forwards the group may be *ragged* (three digits and a fourth); backwards it
may not. This walk finishes the run the pattern refused to bind, and a ragged
group is exactly why the pattern refused: `1 0023R` must not read "one 0023R",
half a run spoken and the rest glued to a letter. Backwards the group *is* the
match, whose width the pattern already fixed, and the same looseness there
swallows the `1000` of `e3 1000`, a four-digit number across an ordinary space,
unrelated to the exponent in front of it.

Measured over 4800 fuzzer sentences: the loose test in both directions changes
60 readings, and 56 of them are losses. Forwards only, 20 change: four numbers
that went unsaid are read, and sixteen ragged runs are left written instead of
read half way.


### `_truncated_by_a_fraction`

The fraction group is `(?:[.,][0-9]+)*`, which can match zero times. The regex
can therefore shrink it to zero and let the trailing `(?![\w])` land on the dot
instead of on a letter: `1.5e3` would match only the `1` and read "one.5e3",
and `3.14abc` would read "three.14abc". This check refuses such a match, and
both stay written.

A number that really ends at the match has nothing of the sort behind it:
`3.14.` at the end of a sentence is followed by a dot and then a space, and
`1,000` by a space. Only a separator *with a digit after it* means the match
stopped early.


### `_is_number`

`1.2.3`, `192.168.0.1` and `12.03.2026` are a version, an address and a date.
The partition splits on one separator at a time, so the leftovers reach `int()`
here. Treating the remainder as a quantity would raise `ValueError` on ordinary
text or, with a comma decimal mark, where segments concatenate, speak
`192.168.0.1` as *nineteen million two hundred sixteen thousand eight hundred
one*.

A run is a quantity when it has at most one separator, or when its separators
group: every segment after the first exactly three digits, and the first one to
three. Anything else is left as written. A valid date in that shape has already
been read by the date pass, which runs earlier.


### `gendered`

`position` names where in the number this value sits: `"standalone"` (it is the
whole number), `"tail"` (it ends a larger number) or `"tens_pair"` (inside the
solid units-and-tens compound). The grammar's per-value scope decides whether
agreement reaches it there; see `gender_scopes`.


### `gender_scopes`

Three scopes exist in this language set. The CLDR differential test found them:

- `"standalone"`: only when the value is the entire number. Polish: *jedna
  kobieta*, but *sto jeden kobiet* and *dwadzieścia jeden*, while Polish 2
  agrees everywhere (*dwadzieścia dwie*).
- `"outside_tens"`: everywhere except inside the solid tens compound. Danish:
  *hundrede og et* (agrees), but *enogtyve* (does not).
- the default: everywhere. Spanish: *treinta y una*.


## `loudkit/frontend/speechtext.py`


### `module`

`speech_text` is the funnel for **all twelve supported languages**: the engine
calls it on every synthesis, whatever the language tag. Only
`lexical_respelling` and the lexicons behind it are Polish. The four ports
give their copies the same name: `speechtext.rs`, `speechtext.go`,
`speechText.ts` and `SpeechText.swift`.

`speech_text` prepares the raw text in every language; the full pass order is
under `speech_text` below. For Polish, `lexical_respelling` then rewrites
English words embedded in Polish the way a Polish reader says them ("download"
→ "dałnloud", "deadline'u" → "dedlajnu"), and spells mixed letter-digit tokens
such as `R2` with Polish letter names (*er dwa*).

The engine is grapheme-based with one language tag per utterance, so a Polish
render reads "download" with Polish letter-to-sound rules and mangles it. In a
listening comparison, respelling beat inline `[en]` tag switching: "dałnloud"
is how the word sounds in a Polish sentence, accent included. All five
implementations must read the same text identically, and this module is the
Python reference.

Respelling uses dictionaries only. A rule-based English G2P here would misfire
on real Polish words, and a false positive (mangled native text) costs far more
than a miss. Two files feed it: `pl_en_respell.json` (110k generated entries)
and `pl_respell_rules.json` (hand-written phrases, a curated lexicon and word
lists). A curated entry wins over a generated one: the generated file holds
*dałnlołd* for "download", and the curated lexicon *dałnloud*. Poles decline
these words ("maila", "deadline'u"), so matching is stem plus a known Polish
ending, with apostrophe forms handled.

The inverted marks `¿` and `¡` stay in the text. Spanish opens a question or an
exclamation with them, and they are the reader's earliest cue that one is
coming. The four ports keep them too.


### `speech_text`

The passes, in order, in all five implementations:

1. NFC.
2. Foreign digits (`fold_foreign_digits`).
3. Invisible characters.
4. Markup tags.
5. Numeral folding (`fold_numerals`).
6. Symbols (`_speak_symbols`).
7. Footnote markers.
8. Roman numerals.
9. Acronyms (`spell_acronyms`).
10. Dates.
11. Ordinals.
12. Abbreviations.
13. Clock times.
14. Numbers.
15. Punctuation (`_punctuation_for_speech`).
16. Polish respelling (`lexical_respelling`).
17. Collapse: runs of spaces and tabs, a space in front of a clause mark, and a
    run of clause marks.

The order and its reasons, briefly:

- **NFC first.** Unicode lets the same character arrive two ways: Polish `ą` as
  U+0105 or as `a` + U+0328, Danish `å` as U+00E5 or `a` + U+030A. Every rule
  below (regexes, lexicon lookups, character classes) is written against the
  composed form, so a decomposed spelling would reach it as a base letter
  followed by a combining mark. The tokenizer applies its own NFKD later.
- **Digits to ASCII before the symbol table.** Arabic-Indic digits fold in
  `fold_foreign_digits`, every other decimal digit in `fold_numerals`. The
  ports ask "is this a digit" in different ways: Python's `\d` matches a
  fullwidth digit, and RE2's and ECMAScript's do not. Without the fold, `€１`
  would read *un euros* in Python and *euros un* in JavaScript. Folding early is
  safe: no key in the symbol or abbreviation tables contains a digit.
- **Markup before symbols**, which would read a tag's angle brackets as
  comparison operators and its attributes as text.
- **Symbols while the digits are still digits.** A currency mark moves behind
  its amount, so "£250" arrives at the number pass as "250 pounds", with only
  the digits left to say.
- **Footnotes before numbers**, because a dropped `[12]` must not become words
  first.
- **Roman numerals after the numeral fold**, which turns `Ⅳ` into `IV`, and
  before the acronym pass, which would spell `IV` letter by letter.
- **Acronyms while the capitals are still capitals.** Later passes rewrite
  text, and a spelled acronym has to be decided while the only evidence (the
  word stands alone in capitals) still exists. The pass spells acronyms in the
  render language in all twelve languages, in all five implementations.
- **Dates before times and numbers.** `12.03.2026` is the ordinary written
  date of five of these languages. The clock pattern matches `12.03` and the
  digit run matches the lot, so a date recognised any later has already been
  read as a time with a stray year, or as one eight-digit number.
- **Ordinals before numbers**, which would expand the digits and leave the
  suffix stuck to them.
- **Numbers before punctuation**, which turns a minus sign after a space into a
  space.
- **Punctuation.** Prosodic marks stay; every other symbol becomes a space,
  except a numeric separator between digits and a sign inside a token.
- **Collapse runs, not pairs.** `re.sub` does not overlap its matches, so a
  pair rule would turn "..." into ".." on one pass and "." on the next. A run
  rule folds it to "." in one pass. An ellipsis reaches the collapse as three
  periods from the symbol map.

The language id is compared case-insensitively. `GraphemeTextFrontend`
lowercases its own tag, so a case-sensitive compare here would give `"PL"`
Polish *tokens* without the Polish respelling: the same utterance read half one
way and half the other.


### `_priced`

A currency amount is where a dot between digits is known not to be a clock
time. `$0.49` is a price, but by the time the funnel reaches `expand_times` the
symbol has become a trailing word, and the dot looks the same as the one in
`14.30`, which in the eleven comma-decimal languages *is* how a time is
written. Without this function, German would read *null Uhr neunundvierzig
Dollar*: zero o'clock forty-nine dollars.

Rewriting the separator here, while the currency symbol is still in hand,
removes the ambiguity. In a comma-decimal language, a lone dot with a plain
fraction becomes the language's decimal mark, so `$0.49` reads *null Komma vier
neun Dollar*. Only that shape is touched: `$1,234.56` carries a grouping mark
this cannot safely reinterpret, and its amount stays written.


### `_speak_symbols`

This pass decides *where* a word goes (a currency mark is written before its
amount and spoken after it) and asks `loudkit.frontend.numbers` *which* word,
because the word is a per-language fact. The wording is a row per language in
`unit_words`: `≈ 5` reads *ungefähr fünf* in German and *noin viisi* in
Finnish. The eight marks that are punctuation in every language
(`→ ← ⇒ • · ▪ ◦ …`) stay a rule in code.

Two fallbacks answer different questions. A language with no wording table at
all is read with the English table, so an unknown tag is spoken, not dropped. A
symbol the named language has no row for gets no English word: an English word
inside another language's render is the defect the per-language table removes.
The symbol is left for the later passes. The five implementations agree on
this.

A price with a scale reads amount, scale, currency: `$2.5M` reads *two point
five million dollars*, and `$5 million` reads *five million dollars*. The
abbreviating letter is read only beside a currency mark, which is what makes it
unambiguous: a bare `5m` is five metres as readily, and stays written.

The ASCII comparison operators `<`, `>`, `<=`, `>=`, `!=` and `==` are read only
with whitespace on both sides: `if latency > 200 ms` reads *if latency greater
than two hundred ms*. The spacing separates an operator from markup, an
emoticon or a glued token. `-`, `/`, `.` and `+` stay silent, because each is a
hyphen, a path, a decimal and a sign far more often than an operator.


### `_speak_symbols` and a multi-character currency mark

`R$`, `HK$` and `NT$` have no row in the table. Matching the `$` alone would
read `R$3,14` as "R3,14 Dollar": the wrong currency, with an orphaned `R` in
front of it. The symbol pass leaves such a mark written (see `_letter_before`),
and the punctuation pass then removes the `$`: `R$3,14` reads *R drei Komma
eins vier* in German.


### `_speak_symbols` reads a suffixed price

`2.50 €` and `0.49¢` are prices by the same evidence as `€2.50`, and are read
before the time pass can see the dot: `2.50 €` reads *zwei Komma fünf null Euro*
in German, not a clock time.

Currency written as a word (`5.50 zł`, `12.30 kr`) is not covered. Those are
ordinary words to every pass here, and telling them from a unit or a name needs
a per-language lexicon, not a symbol table. `5.50 zł` in Polish reads as the
clock time *pięć pięćdziesiąt zł*.


### `_letter_before`

The prefix-currency rule fires only where a mark opens an amount, so it skips a
mark glued to the end of a word. The test is
`unicodedata.category(...).startswith("L")`, which is `\p{L}`: the class Go's
`unicode.IsLetter` and JavaScript's `(?<!\p{L})` test. A regex lookbehind such
as `(?<![^\W\d_])` does not work here: CPython's `\w` for `str` also admits Nl
and No (`½ ¼ ² ③ Ⅳ`), none of which are letters. (`str.isalpha()` tests the
same L categories as `unicodedata`.)


### `fold_numerals`

A number character this layer has no ASCII spelling for (a superscript, a
vulgar fraction, a circled or Roman numeral, a digit from any script but Latin)
is folded before the symbol table. Every "is this a word character" test in the
five ports (`\w`, `\p{N}`, `str.isdigit()`) admits `No`. Without the fold, `²9`
would be one token, the number matcher would decline it, the punctuation pass
would delete the `²`, and a bare `9` would reach the model. In the same way,
`Add ½ cup` would read *add cup*, `Chapter Ⅶ` *chapter*, and `5०3` a pair of
bare digits.

Two mechanisms, and both read `models/data/numerals.json`, which says *whether*
a character is a numeral as well as what it reads as:

* **A decimal digit of any script** (`Nd`) becomes the ASCII digit of the same
  value, so `５9` is fifty-nine and `5०3` is five hundred and three. The value
  is the distance from the digit's block zero, and the zeros are in the table
  (`decimal_zeros`).
* **Every other number character** (`No`, `Nl`) becomes the text the table
  names: `²` → `2`, `½` → `1/2`, `③` → `3`, `Ⅳ` → `IV`. Inside a spelled numeral
  the fold writes `/` as `÷`, so `½` reaches the symbol pass as `1÷2`. The text
  is ASCII for all but 27 of the 1,151 entries: those decompose to CJK
  ideographs, ten of them parenthesised (`〸` → `十`, `㈠` → `(一)`), and leave
  the fold as the ideographs they mean.

Asking the table, not the runtime, is what makes the fingerprint hold.
`unicodedata.category` is whatever Unicode the interpreter bundles. With
runtime detection, `Add \U00010D41 cups` would read *add cups* on Python 3.12
(Unicode 15.0) and stay written on Python 3.14 (Unicode 16.0), under the same
digest, and neither would say *one*. CI runs both. With the table it reads *Add
one cups* on every runtime.

A `No`/`Nl` expansion is separated by a space from an adjacent letter or
decimal digit (`\p{L}`, `\p{Nd}`), and from nothing else. Without the space,
`²9` would fold to `29` and read as twenty-nine, a number that is not in the
text. A folded `Nd` is never separated: it is a digit replacing a digit and
belongs to the run it was already in.

The later passes then read the expansions. `Ⅳ` becomes `IV`, which the
Roman-numeral pass reads as *four*. A fraction reads as a division: `½` reads
*one divided by two*. A typed `1/2` has no fraction bar the character asserts,
so it reads *one two*, and `24/7` reads *twenty-four seven*.

`Lo` is untouched. Ideographic numerals (`一`, `十`) are letters, and the table
has no entry for them, so this pass cannot replace a language's numerals with
spaces.


### `_folded_numeral`

`None` means the table does not name this character, and it stays as written:
a letter, an ideographic numeral, or an ASCII digit that is already what it
folds to.

Both answers (whether to fold, and to what) come from
`models/data/numerals.json`. Computing either goes wrong:

- Walking down to a block start reads a zero as a nine. Every `Nd` block is ten
  contiguous code points, but blocks can be contiguous too, so a walk that
  stops at "the previous character is not a digit" leaves its own block:
  `MATHEMATICAL DOUBLE-STRUCK DIGIT ZERO` follows
  `MATHEMATICAL SANS-SERIF DIGIT NINE` with nothing in between.
- NFKC alone does not reach every numeral. `ETHIOPIC NUMBER TEN`, the Aegean
  numbers, the Kaktovik digits and the Meroitic numerals have no compatibility
  decomposition. The table names them: `Add ፲ cups` reads *Add ten cups*.
- Asking `unicodedata.category` *whether* to fold puts the runtime's Unicode
  version back into the answer; see `fold_numerals`.

`tools/make_numerals.py` cuts the table from one pinned UCD, and the table is
hashed into `TextConfig.grammar`, so five ports on five Unicode versions fold
one way.


### `_punctuation_for_speech` keeps a sign between alphanumerics

A `-` or `+` with a letter or digit on both sides stays, so `1e-3` and `1e+3`,
which the number pass left as tokens with a letter in them, reach the model
whole, not as "1e 3". A sign with a space on either side becomes a space.


### `_respelled` returns an unknown token unchanged

A word this function cannot improve is returned as written. The word collector
keeps `'` and `’` inside a word, so a spaced apostrophe arrives as a token of
its own. Filtered to the characters the ASCII-keyed table knows, it would
vanish from the utterance, and it is in `_PROSODIC`, which the funnel keeps.


### `_respell_words` uses `isdecimal`

`str.isdigit()` is true of `No` characters (`²`, `③`), so `²9` would be one
token that `int()` refuses. `isdecimal` is `Nd`: what `int()` accepts, and the
digit class the four ports test.


### `_respell_words` measures a dotted run whole

The respeller decides for the whole run, not for its first pair. Two groups
read as a decimal: "dwa przecinek pięć". Three or more (a version, an address or
a date) are left as written; reading the first pair would give "jeden przecinek
dwa" with a stray ".trzy" behind it. The same holds when the run starts with a
token that has a letter in it: `v1.2.3` is a version whether or not its first
group is all digits. `numbers.expand` declines these runs by the same rule, and
this second reader of digits agrees with it.


### `_spelled_code_token`

All or nothing. A character with no letter name refuses the whole token instead
of being skipped: a dropped `ü` would change *Müller123* from a name into a
different name. The token must also fit whole (eight characters at most),
because truncation would drop digits from `żelazny2024` without a trace.

A listener cannot tell that anything was dropped from a half-read token. So if
every character has a name and the token is short enough, it is spelled;
otherwise the model gets it as written.


### `WHITE_SPACE`

Unicode White_Space, written out: 25 code points. Every pattern in the funnel
uses it. `\s` is a different set in each runtime: ECMAScript's `\s` excludes
U+0085 NEL, RE2's (Go) is ASCII and omits U+000B, and CPython's also admits
U+001C–U+001F. NEL is ordinary in scraped and EPUB text, and a separator that
survives in one port changes where that port's chunks break.

`_is_space` answers the same question for a single character; this is the
spelling a pattern needs.


### `_NOT_SPACE_IN_THE_PORTS`

FILE, GROUP, RECORD and UNIT SEPARATOR (U+001C–U+001F). CPython's
`str.isspace()` includes them. Rust's `char::is_whitespace`, Go's
`unicode.IsSpace`, JavaScript's `/\s/` and Swift's `whitespacesAndNewlines` do
not. Measured over every code point, these four are the **only** difference
between `str.isspace()` and White_Space, so subtracting them gives exactly
White_Space.

`str.isspace()` answers for every other character, where
`frontend/chunking.py` writes out `WORD_BOUNDARIES`: a cut point lands on a
handful of characters, while this predicate has to keep every `Zs` there is.

Without the subtraction, Python would keep these separators where the four
ports turn them into spaces. The tokenizer would see `[UNK]` where the ports
see `[SPACE]`, and `split_text` would find no word boundary where the ports
find one: different tokens *and* a different chunk split for one text and one
seed, under an `algorithm_fingerprint` that says the five engines agree.


## `loudkit/frontend/text.py`


### `module`

The tokenizer pipeline is thin: lowercase, NFKD, a language tag, spaces to
`[SPACE]`, then plain BPE over Unicode scalars. `swift/LoudKit/TextFrontend.swift`
implements the same recipe, and the conformance fixture's frontend vectors test
it against this reference. Richer preprocessing, such as the upstream
`punc_norm` that rewrites ellipses and appends full stops, is not applied here:
it would make loudkit read text differently from the implementations that share
this recipe.

The speech funnel (invisible characters, symbols, footnote markers,
punctuation and, for Polish, English respelling) lives in
`loudkit.frontend.speechtext`. The engine applies it before this module, as the
Swift engine calls `SpeechText.prepared`. This module is only the tokenizer, so
the conformance fixture's frontend vectors test it in isolation.

Language handling is an **allowlist**: the twelve ids
`loudkit.frontend.numbers.supported_languages()` reports, which is the roster in
`models/data/numbers.json` that every port loads. Anything else is refused with
`UnsupportedLanguageError`.

The tokenizer's vocabulary carries tags for 31 languages, so a known tag is not
enough. `encode(text, "bg")` would turn Cyrillic, through NFKD, into ids the
model reads as sounds it was never trained to make: no error, plausible-sounding
audio, wrong language. `UnsupportedLanguageError` lists the languages that
work, so a client refused for one language can retry with another, and that
list must hold only languages the kit can speak.

`zh`, `ja`, `he`, `ko` and `ru` are named separately, because the reason they
are refused is useful to the caller: their upstream pipelines need Cangjie
codes, kanji-to-hiragana conversion, diacritisation, jamo decomposition or
stress marks, all from optional heavyweight models this frontend does not
carry.


### `GraphemeTextFrontend`

Deterministic and model-free: with one tokenizer file, the same text and
language always give the same ids. Start and stop text tokens are *not* added
here. They belong to the token generator, which frames its own sequence.

Argument: `tokenizer`, a path to `tokenizer.json` (HF `tokenizers` format,
shipped beside the checkpoint), or its bytes, because a checkpoint can carry
its tokenizer inside it.


### `encode` strips bracket tokens

The vocabulary holds 117 bracket tokens: the 31 language tags, and
paralinguistic events such as `[sigh]`, `[gasp]` and `[UH]` from the base
model's training. The tokenizer matches them greedily, so "he [sigh]ed" would
emit control token 611 (`[sigh]`) and invite a sigh. The funnel already turns
brackets into spaces on the engine path. `encode` replaces `[` and `]` with
spaces too, so a direct caller gets the same protection. The one tag that
belongs, the language tag, is added after.


## `loudkit/frontend/textconfig.py`


### `module`

The funnel decides what string the model is handed, so it decides what the
model says. It is therefore part of the identity contract: two builds that read
the same text into different words must not report the same fingerprint.

Two fields, because the funnel changes in two ways and only one of them can be
detected automatically:

* `recipe` names the funnel's *code*: the pass order, the classification
  rules, the realisation logic. A maintainer bumps it by hand, as
  `recipe_version` is bumped for the sampling law.
* `grammar` is the digest of the funnel's hashed data files: `numbers.json`,
  `pl_en_respell.json` and `numerals.json`, in that order. It moves when the
  data moves, with no manual bump.

The digest makes a data edit visible without anyone declaring it. Each of the
five implementations hashes *its own copy* of the files, so a port whose copy
has drifted computes a different fingerprint. The conformance vectors, which
pin the fingerprint, then fail, and a graph backend (ONNX or CoreML) refuses
an export recorded under another fingerprint.

`pl_respell_rules.json` (hand-written phrases, the curated lexicon and word
lists) is a funnel input but is not in the digest. An edit to it changes spoken
words under the same fingerprint unless `FUNNEL_PORTED` is bumped by hand.
Adding it to the digest moves the digest and re-pins five ports, so it waits
for a release that is allowed to move them.


### `grammar_digest`

Hashed as raw bytes, not as parsed JSON. Two files that differ only in
whitespace produce the same speech, but they are not the same file, and
"the ports ship byte-identical data" is a simpler contract to check than "the
ports ship semantically equivalent data".

Raw bytes work only while the files hold nothing but data. A prose note inside
a hashed file would move this digest, and `algorithm_fingerprint` with it, on
an edit that changes no audio, and would re-pin five ports. The prose lives
beside the data (`numbers.about.md`, `numerals.provenance.json`), unhashed and
not copied to the ports, and `test_the_hashed_data_files_carry_no_prose` keeps
it there.


### `FUNNEL_PORTED`

The current value is `"funnel-6"`. It keeps two builds that read the same text
into different words from reporting the same fingerprint. Bump it when a pass
changes what the funnel emits for text it already handled. A new language or a
new table moves `grammar` on its own and needs no bump; an edit to
`pl_respell_rules.json` does need one. There is no `funnel-3` in any released
build. [CHANGELOG.md](../../CHANGELOG.md) lists what each value changed.

What each pass covers under `funnel-6`, in order, in all five implementations:

1. NFC: one composed spelling for every later rule.
2. Foreign digits: Arabic-Indic and Extended Arabic-Indic digits, their
   separators (per language) and the Arabic percent sign, to ASCII.
3. Invisibles: zero-width and format characters and the soft hyphen, removed.
4. Markup tags: a tag, comment or declaration becomes a space (`<p>Hello</p>`
   reads *Hello*).
5. Numeral folding: superscripts, vulgar fractions (as division), circled and
   Roman numeral characters, and decimal digits of every script, from
   `numerals.json`.
6. Symbols: currency before or after an amount, with a scale between amount
   and currency word; the comparison operators `<` `>` `<=` `>=` `!=` `==` with
   whitespace on both sides; other symbols from `unit_words` in the render's
   language.
7. Footnote markers: removed.
8. Roman numerals: I, V and X only, 2 to 39, the range of chapter, act,
   volume, war and regnal numbers (`Chapter IV`, `World War II`). Every
   two-letter initialism that is also a valid numeral needs L, C, D or M
   (`CD`, `CV`, `DC`, `MC`, `MD`, `XL`, `CM`), and so does `MIX`.
9. Acronyms: two to five capitals spelled in the render language; runs of
   capitals left as emphasis.
10. Dates: ISO, dotted, slashed and month-name forms, bounded by a word boundary
    at both ends (`2026-03-04x` and `x25/03/2026` are not dates). The ISO `T`
    between a date and a time becomes a space (`2026-03-04T10:00`). The English
    day-first reading does not repeat an article already in the sentence (`See
    the 3 April minutes` reads *See the third of April minutes*).
11. Ordinals: English digit suffixes (`1st`).
12. Abbreviations: the unambiguous entries per language, case sensitive.
13. Clock times: `H:mm`, `H:mm:ss` and, outside English, `H.mm`, with the
    language's infix; a letter after the time is a separate word (`3:45pm`).
14. Numbers: cardinals and decimals, grouped thousands, telephone numbers
    (a plus and at least eight digits, bounded at both ends) digit by digit,
    identifiers left written.
15. Punctuation: prosodic marks kept, other symbols to spaces.
16. Respelling: Polish only.
17. Collapse: a run of clause marks to one.

A bare year does not read as a year: `In 1776 he wrote it.` reads *one thousand
seven hundred and seventy-six* in English, and the other languages read it as a
cardinal too. The year reading belongs to a written date, where the month says
the digits are a year.


### `recipe`

Every implementation runs the same passes under one recipe value. A build
whose passes differ while it reports the same recipe would give divergent
readings of `12.03.2026`, `1st`, `CIA` or a decomposed `ą` under one
fingerprint. The recipe is a marker a maintainer moves by hand, so the shared
conformance fixture is what checks that the passes agree.
