# The text funnel

Maintainer notes moved out of the runtime's docstrings: the reasoning and
the measurements behind each symbol. Each heading names the module and
the symbol the note belongs to. Not user documentation.


## `loudkit/frontend/chunking.py`


### `module`

A window carries about 255 speech tokens, roughly ten seconds. Anything longer
has to be split, generated in pieces and joined, and *where* the splits fall is
audible, so it is an algorithm-layer decision rather than a caller's convenience.

The rule is simple: break at the strongest punctuation available, as late as
possible. A break at a full stop is inaudible; a break mid-clause is not. When a
single sentence is too long for a window on its own, it gets broken at the best
available comma, and if it has none, at a word boundary, a bad break, but a
break the caller can hear and complain about, which is better than text that
silently disappears.

Not every period-space is a full stop, and this is the one separator that is
ambiguous: `!` and `?` are unmistakable sentence ends and `;` and `,` are not
sentence ends at all. `"But Mr. Smith went home"` carries a period that ends a
title, and the splitter used to break there and hand the renderer the chunk
`"But Mr."`, seven characters, its own utterance, its own derived seed, and a
length-proportional token ceiling that leaves no room for a closing pause. It
is the only chunk in a 9920-row rendered census that hit that ceiling.

`ChunkConfig.mid_sentence_period` says what to do at such a period. Two
conditions find them and only one is a list: `ChunkConfig.abbreviations` for
*Mr.* and *Hr.*, and a next word starting in lower case for what the funnel
manufactures, an ellipsis folded to a single period, which is the dominant
cause in Polish and which no list would touch.

When every candidate in a window is held there is no usable punctuation left,
and the splitter does what it already does in that situation: it breaks at a
word boundary. Taking the held period after all was measured and rejected -
over 1200 passages it saved six word breaks and cost seven period breaks, five
of which left a title dangling at the end of a chunk ("...forte Hr." then
"Theodor Franz..."). A word break is joined with a carried prefix and keeps its
contour. A false full stop is read as a full stop: closing fall, pause, and a
new sentence begun in the middle of the old one.

Splitting is done on characters rather than tokens because it has to happen
before the tokenizer runs, so the token budget is estimated. The estimate is
deliberately conservative: producing a chunk that overflows the window costs a
hard failure, while producing one slightly too short costs nothing but a join.


### `split_in_half`

``split_text``'s estimate is conservative but not a guarantee: it budgets
characters against a constant, and a speaker slower than that constant
fills the window before the text runs out. The generator then stops at the
cap mid-word, and the words that did not fit are **lost** rather than
deferred, because chunk texts are fixed before any of them is rendered.
Measured across ten languages, 54 of 9920 chunks reached a cap and 30
were still speaking when it closed, over five voices; gating on the
window leaves 51 and 27, on two.

The boundary is the nearest *word* break, and punctuation is not sought.
The reason is mechanical rather than comparative: a comma is an instruction
to pause, this model has no pause-duration prior, and fed one it overshoots
-- the same mechanism as every other defect in this campaign, arriving
through the chunker. The arm that measured punctuation against word
boundaries was retracted with the harness that produced it, which seeded
the second half from the next chunk's stream; the re-measurement compared
only this law against no law, so the figures that once stood here have no
un-retracted source. Not sought is not avoided: the nearest word break can
follow a comma.

Returns ``None`` when there is no word boundary to use, which is a single
unbroken run of characters. Such a chunk cannot be helped by splitting it,
and mangling a word to fit is worse than the truncation.


## `loudkit/frontend/dates.py`


### `module`

A written date is the one construction where every language in this kit
disagrees with every other about something load-bearing. The day is an ordinal
in English, German, Danish, Polish, Finnish, Norwegian and Swedish, and a
cardinal in Dutch, Spanish and Portuguese; French and Italian use a cardinal for
every day *except* the first. The month is nominative in most, genitive in
Polish (``marca``, never ``marzec``) and partitive in Finnish (``maaliskuuta``).
Spanish and Portuguese speak a preposition between every part. The year splits
into halves in English and Norwegian, groups in hundreds in German, Dutch and
Swedish, and is one plain cardinal in the six others.

None of that is derivable, so none of it is derived: the day words are written
out in ``numbers.json`` per language, sourced from the national authority, the
five English irregulars, German's ``siebte``/``achte``, Danish ``ellevte`` (the
``elvte`` form was added to Retskrivningsordbogen and then withdrawn, and the
crowd-sourced lists still carry it), Italian ``ventotto`` (never *ventiotto*),
Finnish's ordinal suffix repeating inside every part of a compound. ``docs``
records which authority said what.

**Why this runs first.** ``12.03.2026`` is the ordinary written date of German,
Polish, Danish, Finnish and Norwegian, and both later passes want a piece of it:
the clock pattern matches ``12.03`` and the digit-run pattern matches the whole
thing. Without this pass running first, the clock pattern reads ``12.03`` as twelve
o'clock three with the year trailing behind, or the digit-run pattern reads the
whole thing as a single eight-digit number. Recognising dates before either
pass runs is the only ordering that leaves nothing to argue over.

**What it refuses.** A version number, an address and a score all look like a
date to a permissive matcher, and reading one aloud as a date is worse than
leaving it alone: ``1.2.3`` must never become *the first of February, three*.
Every candidate is bounds-checked, the day against the month's real length, the
month against twelve, and a four-digit year against a plausible range, and an
all-numeric ``dd/mm`` in English is left untouched entirely, because ``3/12`` is
March twelfth to half the English-speaking world and the third of December to
the other half, and a confident wrong reading is unrecoverable where a literal
one is not.


## `loudkit/frontend/letters.py`


### `module`

``CIA`` is *see-eye-ay* in an English render and *ce-i-a* in a Polish one, and
those are not two spellings of one thing, they are what the two languages
actually say. The engine is grapheme-based with a single language tag per
utterance, so the letter name has to be written in the target language's own
orthography: English ``see`` reads as /siː/ under English letter-to-sound rules,
Polish ``ce`` reads as /t͡sɛ/ under Polish ones, and putting either into the
other's render produces a word in no one's language.

The table is per language: letter names are orthography-specific (Polish
``FBI`` is *ef-be-i*), so a render without its own table reaches the model with
raw graphemes, which a grapheme engine reads as a word-shaped thing rather
than as letters. That is the same
failure the currency wording had before it moved into the shared grammar file,
and it is fixed the same way: the tables are data, one per language, read by
every implementation.

**What is not spelled.** An acronym that is a word in its language stays a word:
``NASA`` and ``NATO`` everywhere, ``SIDA`` and ``OVNI`` in the Romance three,
``PESEL`` and ``ZUS`` in Polish, ``TUTKA`` in Finnish. Those lists are per
language because the fact is: ``LOT`` is an airline in Poland and a common noun
in English, and only one of them should be spelled out.


## `loudkit/frontend/numbers.py`


### `module`

Why this exists at all
----------------------

Digits do not reach the model. The frontend tokenises **graphemes**, and a
checkpoint trained on normalised transcripts has never seen ``45``, so a digit
is at best a dead embedding and at worst dropped. The failure is not a
mispronunciation, it is silence or garbage where a number should be, and no
amount of acoustic quality repairs it.

Why it is written here rather than taken from a library
-------------------------------------------------------

Two reasons, and the second is the one that matters.

The licence: the obvious dependency is LGPL-2.1, which is not a licence this kit
can carry into every embedding it is meant for.

The grammar: that library has **no case or gender machinery for Polish at all**
- one nominative form per numeral, while shipping full six-case declension for
Russian in the same release. Polish is the language where getting this right
matters most (a published measurement puts a morphology-blind normaliser at
~30% against ~91% for a morphology-aware one), so the dependency would have to
be replaced for the hardest language anyway. There is nothing to inherit.

How it is organised, and why
----------------------------

**The grammar is data; only the interpreter is code.** Nine languages × five
implementations is forty-five chances for a rule to drift. One JSON file read by
five small interpreters is one chance, and the fixture catches it.

The data format follows the shape these systems actually have: a regular
generative core, plus a listed set of irregular forms. Every European number
grammar in this set is "units, teens, tens, scales, and a composition rule",
with the interesting variation in *how* the pieces join:

- **order**, German, Dutch and Danish say the unit first (*einundzwanzig*,
  literally "one-and-twenty"); the rest say the ten first.
- **joiners**, Spanish *treinta y uno*, Portuguese *vinte e um*, French
  *soixante et onze* but *quatre-vingt-un*, German's bare *und*.
- **elision**, Italian *ventuno* and *ventotto* drop the ten's final vowel
  before a vowel-initial unit.
- **agreement**, Polish and Spanish inflect the numeral for the gender of what
  is counted; Polish additionally for case.

Rather than model each of those as a rule engine, irregular *values* are listed
outright. That is not a shortcut: it is how the grammars are described in their
own reference works, it keeps the interpreter small enough to port honestly, and
a listed form can be checked by eye against a dictionary.

What it deliberately does not do
--------------------------------

Nothing here reads context. A numeral's case in Polish, or its gender in
Spanish, is a property of the *sentence*, not of the number, see
:func:`cardinal` and the ``grammar`` argument for the seam where a caller with
that knowledge supplies it. Choosing it automatically needs a morphological
tagger, and that is a different component with a different failure mode.


### `cardinal`

Args:
    value: the integer to say. Negatives are read with the language's own
        minus word; there is no locale in this set that omits it.
    language: one of :func:`supported_languages`.
    gender: the grammatical gender of the counted noun, for the languages
        that agree with it, Polish and Spanish inflect *one* and *two*,
        German inflects *one*. ``None`` gives the citation form, which is
        what a bare number in a list wants and what every other language
        here uses unconditionally.

Raises:
    NumberGrammarError: if the language is unknown, or the value is larger
        than the grammar's largest scale. Raising rather than falling back
        to digits is deliberate, see :class:`NumberGrammarError`.

Examples::

    cardinal(21, "en")                  # 'twenty-one'
    cardinal(21, "de")                  # 'einundzwanzig'
    cardinal(71, "fr")                  # 'soixante et onze'
    cardinal(2, "pl", gender="f")       # 'dwie'


### `expand`

The seam between this module and the speech funnel. It runs *after* the
symbol pass, so a currency amount has already become "250 pounds" and only
the ``250`` is left to say, and *before* the punctuation pass, which turns
a decimal separator into a space and would leave two unrelated numbers.

**It never raises**, and it never *half*-reads a number: a value past the
largest scale the grammar has a word for is read digit by digit, which is
what such a number almost always is, an identifier, a code, a serial. That
is a deliberate difference from :func:`cardinal`, which refuses: a library
call has a caller who can decide, and a text funnel has a user whose
sentence must still be spoken.

It does leave digits behind, deliberately: a run glued to a word is part
of that word and stays written -
`iOS18`, `r123`, `v1.2.3`, `5x3`, because *iOSeighteen* is not a reading of
anything. So is a dotted run that no convention resolves: `1.2.3` is a
version, `192.168.0.1` an address, and `18.08.2026` in English a date whose
field order half the English-speaking world reads the other way round.
Digits reaching the model are a reading the model can make; a confident
wrong number is one that cannot be undone.

A separator between digits is a decimal mark only when it is *the*
language's decimal mark. English ``3.5`` is three point five; English
``3,500`` is a grouped thousand, and reading its comma aloud would be
absurd, so the other mark is simply dropped, which is what a reader does
with it.


### `_glued_forward`

The mirror of :func:`_glued_to_a_word`, and needed for the same reason it
is: `200 000x` matched `200` alone, because the grouped alternative reached
the `x` and the right-hand guard refused it, so the regex backtracked to
the first group and read "two hundred 000x". Go and Rust, which do not
backtrack, left the whole token written. Half a token spoken again, and the
opposite half from `x200 000`.

A grouping space is crossed, so `200 000x` is one token; the ordinary space
in `2024 200 people` is not, because what follows *it* is a word rather than
a digit group, and those two numbers stay two numbers.

Forwards the group may be *ragged*, three digits and a fourth, where
backwards it may not, and the asymmetry is the measurement rather than an
oversight. This walk finishes the run the pattern refused to bind, and a
ragged group is exactly why it refused: `1 0023R` matched the `1` alone and
read "en 0023R", half a run spoken with the rest welded to a letter, which
is the class the right-hand guard exists to stop. Backwards the group *is*
the match, whose width the pattern already fixed, and the same looseness
there swallows the `1000` of `e3 1000`, a four-digit number across an
ordinary space, unrelated to the exponent behind it. Over 4800 fuzzer
sentences the loose question in both directions changes 60 readings and 56
of them are losses; asked forwards only, 20 change, four numbers that had
gone unsaid, sixteen ragged runs no longer read half way.


### `_time_patterns`

German writes the time *with* the word the spoken form also carries:
``um 14.30 Uhr``. The reading puts the infix where it belongs, between
hour and minutes, *vierzehn Uhr dreißig*, so the written ``Uhr`` is that
same spoken token, not an additional one, and leaving it standing said it
twice: *vierzehn Uhr dreißig Uhr*. When the source carries the infix
immediately after the time, the match swallows it and the normal reading
supplies the one copy.

Every piece is spelled out because five implementations must match
identically: the whitespace run is ASCII space and tab (regex engines
disagree on what ``\s`` covers), the guard refuses an ASCII letter or
digit so *Uhrzeit* keeps its word whole, and case matters, the grammar
data says ``Uhr`` and this rule does not reach past that.


### `expand_times`

Runs before :func:`expand`, which would otherwise read the separator's two
sides as unrelated numbers and leave the colon behind. An hour of 0–23 and
exactly two minute digits, so ``3.5`` (a decimal) and ``1.000`` (a grouped
thousand) never match, and, for the dotted form, only in the languages
whose decimal separator is not the dot. See ``_DOTTED_TIME_RUN``: the same
``H.mm`` is a time in German and a price in English, and the grammar file
already knows which is which.

The reading is hour and minute as cardinals with the language's own infix:
*vierzehn Uhr dreißig*, *neljätoista kolmekymmentä*, *fourteen thirty*. A
minute of zero says the hour alone. Deliberately not the colloquial clock
(*half three* is 2:30 in six of these languages and 3:30 in none of them -
emitting it wrong by an hour is the highest-severity mistake a time reader
can make, so the plain reading wins until the colloquial one is measured).

A written infix directly after the time, German ``um 14.30 Uhr``, is
consumed rather than duplicated; see :func:`_time_patterns`.


## `loudkit/frontend/speechtext.py`


### `module`

:func:`speech_text` is the funnel for **all twelve supported languages**: the
engine calls it on every synthesis, whatever the language tag. Only
:func:`lexical_respelling` and the dictionary behind it are Polish.

This module was called `polish.py` until 0.1.1, which is where it started and
not what it does. The name outlived the scope by eleven languages, and a
reader looking for the funnel had no reason to open a file named after one of
them. `speechtext` is what the four ports already call their copies -
`speechtext.rs`, `speechtext.go`, `speechText.ts`, `SpeechText.swift`, so the
five now agree on the name as well as on the bytes.

The shipped engine reads text through a two-stage funnel before tokenising:
:func:`speech_text` scrubs the raw text (invisible characters, symbols,
footnote markers, punctuation) in every language, then, for Polish,
:func:`lexical_respelling` rewrites English words embedded in Polish the way a
Polish reader says them ("download" → "dałnloud", "deadline'u" → "dedlajnu"),
numbers to Polish cardinals, and acronyms/code tokens to spelled-out Polish
letter names.

The engine is grapheme-based with ONE language tag per utterance, so a Polish
render reads "download" with Polish letter-to-sound rules and mangles it.
Respelling won the ear test over inline ``[en]`` tag switching: "dałnloud" is
not a hack, it is how the word actually sounds in a Polish sentence, accent
included. This is the same promise as everywhere else in this library, the
Swift and Python engines must read the same text identically, and this module
is the Python half of that contract.

Dictionary-first and ONLY dictionary: a rule-based English G2P bolted on here
would misfire on real Polish words, and the cost of a false positive (mangling
native text) is far higher than the cost of a miss. Inflections ride as
suffixes: Poles decline these words ("maila", "deadline'u"), so matching is
stem + known Polish ending, with apostrophe forms handled.


### `_priced`

The one place a dot between digits is *known* not to be a clock time, and
the last place that knows it. `$0.49` reads as a price in every language on
earth, but by the time the funnel reaches `expand_times` the symbol has
already become a trailing word and the dot is indistinguishable from the one
in `14.30`, which in the eleven comma-decimal languages *is* how a time is
written. So German answered "null Uhr neunundvierzig Dollar": zero o'clock
forty-nine dollars.

Rewriting the separator here, where the currency symbol is still in hand, is
what removes the ambiguity rather than adding a rule that tries to guess it
back later. Only a lone dot with a plain fraction is touched, `$1,234.56`
carries a grouping mark this cannot safely reinterpret, and leaving it is the
same refusal the rest of this module makes when evidence runs out.


### `_letter_before`

The prefix-currency rule only fires where a mark opens an amount, so it has
to skip one glued to the end of a word. That was a regex lookbehind,
`(?<![^\W\d_])`, whose comment said "not preceded by a letter" and whose
behaviour was not that: CPython's `\w` for `str` is `Py_UNICODE_ISALNUM`,
which admits Nl and No, `½ ¼ ² ③ Ⅳ`, none of which are letters. So the
rule was suppressed after them, and because `$` is not in `_SPOKEN_SYMBOLS`
it then fell through to `_punctuation_for_speech` and became a space: `²$9`
read as "nine", losing the amount's currency entirely.

`unicodedata.category` rather than `str.isalpha()`: `isalpha` is the
Alphabetic property, which is wider than `\p{L}` by some 7,000 code points
(combining marks, mostly). Go's `unicode.IsLetter` and JS's `(?<!\p{L})`
read `\p{L}`, so this reads `\p{L}` too.


### `fold_numerals`

One rule for one class. A number character this layer has no ASCII
spelling for, a superscript, a vulgar fraction, a circled or Roman
numeral, a digit from any script but Latin, used to be *deleted* by the
punctuation pass, and while it was still there it sat inside the word:
every "is this a word character" test in the five ports is ``\w``,
``\p{N}`` or ``str.isdigit()``, and all three admit ``No``. So ``²9`` was
one token, the number matcher declined it, and a bare ``9`` reached the
model.

Deleting was the earlier answer and it was the wrong one. Text that
silently does not arrive is the failure this layer exists to prevent:
``Add ½ cup`` read as *add cup*, ``Chapter Ⅶ`` as *chapter*, ``5०3`` as a
pair of bare digits. Nothing is dropped now.

Two mechanisms, because Unicode gives two, and **both are read out of
`models/data/numerals.json`**, which character is a numeral as much as
what it reads as:

* **A decimal digit of any script** (``Nd``) becomes the ASCII digit of the
  same value, so ``５9`` is fifty-nine and ``5०3`` is five hundred and
  three. The value is the distance from the digit's block zero, and the
  zeros are in the table.
* **Every other number character** (``No``, ``Nl``) becomes the text the
  table names: ``²`` → ``2``, ``½`` → ``1/2``, ``③`` → ``3``, ``Ⅳ`` →
  ``IV``. ASCII for all but 27 of the 1,151 entries: those decompose to CJK
  ideographs, ten of them parenthesised (``〸`` → ``十``, ``㈠`` → ``(一)``),
  and leave the fold as the characters they mean, because there is nothing
  more ASCII to say about them.

Asking the table rather than the runtime is what makes the fingerprint mean
something. `unicodedata.category` is whatever Unicode the interpreter
bundles: on 15.0 (Python 3.12) ``Add \U00010D41 cups`` deleted the digit and
read *add cups*, and on 16.0 (Python 3.14) the same build, reporting the
same digest, left it written. Neither said *one*, the two disagreed, and
the CI matrix runs both. A funnel that reads its Unicode from the process
it happens to be in cannot promise a reading at all.

A ``No``/``Nl`` expansion is separated from an adjacent **alphanumeric**
and from nothing else. Without that, ``²9`` folds to ``29`` and reads as
twenty-nine, a number that is not in the text; with a space everywhere
instead, ``Chapter Ⅶ.`` becomes ``Chapter VII .`` and the sentence loses
its end. A folded ``Nd`` is never separated, it is a digit replacing a
digit and belongs to the run it was already in.

What the expansions then *read* as is the number pass's business, and they
read exactly as their ASCII spelling does: ``½`` becomes ``1/2`` and is
read the way ``1/2`` is read, ``Ⅳ`` becomes ``IV`` and is read the way
``IV`` is read. That is the rule, a numeral is spoken as its ASCII form
would be, and it is the reason this pass needs no verbaliser of its own
in five languages.

``Lo`` is untouched: ideographic numerals, ``一``, ``十``, are letters,
and the table carries no entry for them, so a pass keyed on numeric type
rather than on the general category cannot replace a language's numerals
with spaces here.


### `_folded_numeral`

``None`` means the table does not name this character, and the character is
left exactly as written, which is the answer for a letter, for an
ideographic numeral, and for an ASCII digit that is already what it folds
to.

Both halves come from `models/data/numerals.json`, and that is the whole
design. Computing either half was wrong twice over and silently:

Walking down to a block start read a zero as a nine: every `Nd` block is
ten contiguous code points, but the *blocks* are contiguous too, so a walk
that stops at "the previous character is not a digit" walks out of its own
block. `MATHEMATICAL DOUBLE-STRUCK DIGIT ZERO` follows
`MATHEMATICAL SANS-SERIF DIGIT NINE` with nothing in between.

NFKC alone did not reach every numeral: `ETHIOPIC NUMBER TEN`, the Aegean
numbers, the Kaktovik digits and the Meroitic numerals have no
compatibility decomposition, so they came through unchanged and were then
deleted by the punctuation pass.

And asking `unicodedata.category` *whether* to fold put the runtime's
Unicode version back into the answer after the table had taken it out -
see :func:`fold_numerals`. The table is cut from one pinned UCD by
`tools/make_numerals.py` and hashed into `TextConfig.grammar`, so five
ports on five Unicode versions fold one way.


### `speech_text`

Every pass here runs in all five implementations. Dates, ordinals, acronyms
and NFC were Python-only for a while, gated behind an explicit ``recipe`` so
the fingerprint could not claim a parity that did not exist; the four ports
have them now, so the gate is gone and so is the divergence it described.

Order matters and is deliberate: invisible characters first (they are not
whitespace by Unicode's rules), symbols that carry meaning become words
while the digits around them are still intact, footnote markers before
punctuation rules, punctuation last (prosodic marks stay exactly where
they are, everything else becomes a space).

The language id is compared case-insensitively. ``GraphemeTextFrontend``
lowercases its own tag, so ``"PL"`` produced Polish *tokens* while silently
skipping the Polish respelling here, the same utterance read half one way
and half the other, with nothing to indicate it.


## `loudkit/frontend/text.py`


### `module`

The pipeline is deliberately thin, lowercase, NFKD, a language tag, spaces to
``[SPACE]``, then plain BPE over Unicode scalars, because that is what the
production engine runs (ChatterboxTokenizer.swift is a bit-parity port of the
same recipe, tested against the Python reference). Anything richer, like the
upstream ``punc_norm`` that rewrites ellipses and appends full stops, was a
research-harness convenience that the shipped app never applied; adding it
here would make loudkit read text differently from the engine it is replacing.

The speech funnel that scrubs raw text before tokenising, invisible
characters, symbols, footnote markers, punctuation, and (for Polish) English
respelling, lives in :mod:`loudkit.frontend.speechtext` and is applied by the
engine, mirroring the Swift backend's call to ``SpeechText.prepared``. This
module is the tokenizer and only the tokenizer, so the conformance fixture's
frontend vectors keep testing it in isolation.

Language handling is an **allowlist**: the twelve ids
:func:`loudkit.frontend.numbers.supported_languages` reports, which is the roster in
``models/data/numbers.json`` that every port already loads. Anything else is
refused.

It was a blacklist of zh/ja/he/ko/ru, and the difference matters because the
tokenizer's vocabulary carries tags for 31 languages. A blacklist accepted the
other 26 and the tag went through, so ``encode(text, "bg")`` NFKD-mangled
Cyrillic into ids the model reads as sounds it was never trained to make, no
error, plausible-sounding audio, wrong language. Worse once
:class:`~loudkit.errors.UnsupportedLanguageError` began advertising what *would*
have worked: a client refused for ``zh`` would read the list and retry with a
language the kit cannot actually speak.

The five model-based ones are still named separately, because *why* they are
refused is real information, Cangjie codes, kanji→hiragana, diacritisation,
jamo decomposition, stress marks, all of them optional heavyweight models this
frontend does not carry. A Polish-style grapheme read of Chinese is the wrong
sounds in the right order, and no error downstream would say why.


## `loudkit/frontend/textconfig.py`


### `module`

The funnel decides what string the model is handed, so it decides what the
model says. It is therefore part of the identity contract: two builds reporting
the same sixteen hex digits must not speak differently.

Two fields, because the funnel changes in two ways and only one of them can be
detected automatically:

* ``recipe`` names the funnel's *code*, the pass order, the classification
  rules, the realisation logic. A human bumps it, the same way
  ``recipe_version`` is bumped for the sampling law.
* ``grammar`` is the digest of the shared data file every implementation reads.
  It moves when the data moves and requires no manual bump.

The digest is what makes a data edit visible without anyone remembering to
declare it, and because each of the five implementations hashes *its own copy*
of that file, a port whose copy has drifted computes a different fingerprint
and the engine refuses to start.


### `grammar_digest`

Hashed as raw bytes rather than as parsed JSON: two files that differ only
in whitespace produce the same speech, but they are also not the same file,
and "the ports must ship byte-identical data" is a cheaper contract to keep
than "the ports must ship semantically equivalent data".

Raw bytes is only honest while the files hold nothing but data. Both JSON
files used to carry an ``about`` member, and correcting one sentence in it
moved this digest, moved ``algorithm_fingerprint`` under it, and re-pinned
five ports while every sample rendered identically, a fingerprint that
moves without the audio moving is exactly the promise
``docs/reference/COMPATIBILITY.md`` makes it not make. The prose lives
beside the data now (``numbers.about.md``, ``numerals.provenance.json``),
unhashed and not copied to the ports, and
``test_the_hashed_data_files_carry_no_prose`` keeps it there.


## Notes moved from the runtime docstrings


## `loudkit/frontend/__init__.py`


### `module`

One recipe, five implementations. ``TextConfig.recipe`` and the grammar digest
pin what this package does into the algorithm fingerprint, so two builds that
report the same sixteen hex digits hand the model the same string.

Import the passes from their submodules directly (``loudkit.frontend.numbers``,
``loudkit.frontend.speechtext``, ...). The package ``__init__`` stays import-free on
purpose: ``loudkit.config`` and this package's chunking pass are mutually
dependent by design, and an eager init would turn that into a circular import.


## `loudkit/frontend/chunking.py`


### `_holds`

``look`` is the search window plus one character, because the test below
reads the character *after* the separator and the latest candidate can end
the window exactly.

Gated on the period. `! ` and `? ` end sentences and `; ` and `, ` do not
end them at all, so neither is ever in doubt; the whole question is about
the one mark that is written for two jobs.


### `split_text`

Args:
    text: the passage to read.
    config: the chunking policy, from :class:`~loudkit.config.AlgorithmConfig`.

Returns:
    Chunks in order, each stripped of surrounding whitespace, together
    covering the input. Never empty for non-empty input.

Example:
    >>> from loudkit.config import ChunkConfig
    >>> split_text("One. Two. Three.", ChunkConfig(max_tokens=12, prefix_tokens=0))
    ['One.', 'Two.', 'Three.']


## `loudkit/frontend/dates.py`


### `ordinal_day`

Args:
    day: 1–31.
    language: one of :func:`supported_languages`.
    oblique: German only, the ``-en`` ending that ``am``/``den`` select.
        Ignored everywhere else, because no other language here inflects the
        day by its frame.


### `ordinal`

Composed rather than enumerated past ninety-nine: the hundreds and above
stay cardinal and only the last two digits become an ordinal, so *101st* is
"one hundred and first". The irregulars a suffix rule gets wrong, fifth,
eighth, ninth, twelfth, twentieth, are all inside the two-digit tables and
are written out there.


### `expand_ordinals`

English is the only one of the twelve that writes an ordinal as digits plus
a letter suffix, so for every other language this is a no-op, the suffix
list is empty and nothing matches. It runs before the number pass, which
would otherwise expand the digits and leave the suffix stuck to them:
*onest*, *fiveth place*, *twenty-twond*.

A value the tables cannot say is left exactly as written, suffix included,
rather than half-said.


## `loudkit/frontend/letters.py`


### `spell_acronyms`

**Shouting is left alone**, and the rule for telling it from an initialism
is context rather than anything inside the word. An initialism appears as a
single capitalised island in ordinary text, "the CIA said", while emphasis
comes in runs. That distinction is not available from the word itself: `IT`
is a word, an initialism and a shout depending only on what sits beside it,
and no table can separate those. So a capitalised word spells out only when
neither neighbour is also capitalised, and a text that is *entirely*
capitals is passed through whole, because someone pasted a headline and
spelling all of it would be the loudest possible wrong answer.


## `loudkit/frontend/numbers.py`


### `fold_foreign_digits`

Applied by the funnel next to NFC rather than in the number pass, and for
the same reason NFC is: it is a normalisation, and every pass after it
should see one spelling. In the number pass it was also too late for the
percent sign, since the table that turns U+066A into a word runs earlier.

Language-dependent for the separators, and this is not a detail. U+066B is a
*decimal* separator, so folding it to a dot everywhere turned ``٣٫١٤`` into
``3.14``, which in the eleven languages that write decimals with a comma is
the written form of a clock time, and read out as *drei Uhr vierzehn*. The
same wrong reading this module had just been fixed for, arriving through a
character set instead of a pattern. It folds to whatever mark the language
actually uses.


### `_starts_a_group`

The shape :data:`_DIGIT_RUN` binds as a group after the first, and so the
shape a space in front of them may be grouping.

The slice has to *have* three, which is where the reference was wrong and
the three ports checking a width were right: ``"2"[0:3].isdigit()`` is True,
so a lone digit passed for a group and `R2 2` had the backward walk cross
the space, reach the `R` and refuse a number nothing was glued to.


### `_continues_a_group`

A group the pattern could have *bound*, rather than a ragged run that only
looks like one, the first group may be one to three digits and says nothing
about whether the space groups, so this is asked of the half whose width
:data:`_DIGIT_RUN` fixes.

Which of the two questions to ask differs by direction, and the asymmetry is
the measurement rather than an oversight; see :func:`_glued_forward`.


### `_glued_to_a_word`

Walks back over word characters and dots, the two things an identifier puts
between its letters and its digits, and answers yes on the first letter. A
space, a comma or any other separator ends the walk and the answer is no.

Only backwards: the forward direction is `(?![\w])` in the pattern, which
the regex can express. Backwards it cannot, because the lookbehind sees one
character and the letter may be several away.


### `_truncated_by_a_fraction`

The fraction group is `(?:[.,][0-9]+)*`, which can match zero times, and the
regex will happily shrink it to zero so that the trailing `(?![\w])` lands
on the dot instead of on a letter. `1.5e3` matched just the `1` that way and
read "one.5e3"; `3.14abc` read "three.14abc". A word welded to digits, which
is what the right-hand guard was added to stop, arriving through the one
part of the pattern that is allowed to disappear.

A number that really ends here has nothing of the sort behind it: `3.14.` at
the end of a sentence is followed by a dot and then a space, and `1,000` by
a space. Only a separator *with a digit after it* means the match stopped
early.


### `_is_number`

``1.2.3``, ``192.168.0.1`` and ``12.03.2026`` are a version, an address and a
date. None is a number. Partition splits on one separator at a time, so
the leftovers reach ``int()`` here: treating the remainder as a quantity
either raises ``ValueError`` on ordinary text or, with a comma decimal
mark, where segments concatenate, speaks ``192.168.0.1`` as *nineteen
million two hundred sixteen thousand eight hundred one*.

A run is a quantity when it has at most one separator, or when its
separators actually group, every segment after the first exactly three
digits, and the first one to three. Anything else is left exactly as it was
written, for a later pass (a date) or for the reader to deal with.


### `gendered`

``position`` names where in the number this value sits, "standalone"
(it is the whole number), "tail" (it ends a larger number), or
"tens_pair" (inside the solid units-and-tens compound), and the
grammar's per-value scope decides whether agreement reaches it there.
See ``gender_scopes``.


## `loudkit/frontend/speechtext.py`


### `_speak_symbols`

This pass owns *where* a word goes, a currency mark is written before its
amount and spoken after it, and asks ``loudkit.frontend.numbers`` *which* word,
because which word is a per-language fact. A two-language table would
render "$5" in German as "5 dollars" and read `≈` as *about* in German and in
Finnish: the wording must cover every shipped language, so it is a row per
language in `unit_words`. The eight marks that are punctuation in every
language stay a rule in code.

Two fallbacks, and they answer different questions. A caller naming a
language with no wording table at all is read in English, so an unknown tag
is spoken rather than dropped. A *symbol* the named language has no row for
is left written: falling back to the English row per symbol is the defect the
per-language table exists to remove, and the five implementations agree on
leaving it, which is what this funnel does everywhere the evidence runs out.


### `_spelled_code_token`

All or nothing: a character with no letter name refuses the token rather
than skipping silently (a dropped `ü` changes *Müller123* from a name into
a different name), and the token must fit whole, truncation would drop
digits from `żelazny2024` without a trace.

Both are the failure this funnel spent the week closing, one pass further
on: a token half-read is worse than a token left written, because the
listener cannot tell that anything was dropped. If every character has a
name and the token is short enough, it is spelled; otherwise the model gets
it as written and reads it however it reads it.


## `loudkit/frontend/text.py`


### `GraphemeTextFrontend`

Deterministic and model-free: the same text and language always produce the
same ids. Start/stop text tokens are *not* added here, they belong to the
token generator, which owns its own sequence framing (mirroring the shipped
split, where the tokenizer emits bare ids and the T3 runner frames them).

Args:
    tokenizer_path: the ``grapheme_mtl_merged_expanded_v1`` tokenizer JSON
        (HF ``tokenizers`` format), normally shipped beside the checkpoint.


## Notes moved from attribute docstrings and comments


### `loudkit/frontend/chunking.py`


#### `WORD_BOUNDARIES`

A predicate would be shorter and the five ports do not agree on one: measured,
Python's `str.isspace()` treats U+001C-U+001F as whitespace where the other
four do not, and JS alone KEEPS U+0085 where the other four strip it. Swift
alone strips U+200B, JS alone strips U+FEFF; the funnel removes both before
the splitter sees them, but U+0085 and U+001C-U+001F survive it. A
disagreement there is a different split point, which is different audio for
the same text and seed. A hand-written tuple cannot drift.

The funnel does not remove these. NBSP in particular is ordinary in real prose
-- "10 000", French punctuation, typeset copy -- and `split_text` says so where
it strips leading whitespace, so a capped chunk whose only boundaries are NBSP
still splits.


#### `CHARS_PER_TOKEN`

Measured on the reference voice across English, Polish (after the respelling
funnel) and German: 0.53-0.64 characters of prepared text per speech token,
consistent with ~25 speech tokens/s at ~14-16 characters/s of narration. The
old constant (3.2) was the inverse unit mistake, it let a chunk carry ~816
characters, which the generator turned into ~1300 tokens against a 255-token
window, silently dropping the tail of every over-long chunk.

The constant is the **low end with margin** (0.5 < the 0.53 measured minimum),
not the middle of the range, and being conservative costs nothing but slightly
more, slightly shorter chunks.

**It is a budget, not a guarantee.**
Measured over 9920 rendered chunks in ten languages, 54 overran the window
anyway. Only one voice on the roster reads below the constant (soren, 0.481);
of the 54, just 17 carry enough characters to need more than 255 tokens at
their own voice's median pace. The rest are chunks that should have fitted and
did not, because the model did not emit a stop token in time, a mean cannot
bound a variance. Nor does overflowing cost a `ValueError`: the generator stops
at the cap mid-word and the remainder is simply never spoken, which is why
`ChunkConfig.cap_resplit` exists and why `split_in_half` is in this module.


#### `_holds` reads ASCII lowercase only

A period followed by a lowercase word is mid-sentence, which is what catches
the ellipsis the funnel folds to a single period and no abbreviation list
reaches. ASCII only, measured rather than assumed: over 2253 periods in ten
languages, four are followed by a word starting with a non-ASCII lowercase
letter, and reading the whole Unicode Lowercase property instead moves one
passage in 1200. Not worth four ports disagreeing about what `islower` means.


#### `split_text` breaks at a word boundary

A chunk that fits no separator breaks at the last word boundary; it will be
heard, and that is the point. `WORD_BOUNDARIES`, not U+0020: NBSP survives
the funnel and is ordinary in real prose, and text whose every space is
non-breaking (HTML where `&nbsp;` won) would otherwise find no boundary and be
cut mid-word.


### `loudkit/frontend/dates.py`


#### `module` reads only dates with a year

`12.03.2026` is a date because the year makes it one. The yearless `12.3.`
that German, Danish, Finnish and Norwegian also write is deliberately not
matched: its closing period is indistinguishable from a sentence's, so `Die
Zahl ist 3.5.` would read as *dritte Mai* in ten of the twelve languages. With
no evidence in the string to separate the two readings, the text is left
alone. A yearless date written with a month name still reads, because the
name is the evidence.


### `loudkit/frontend/letters.py`


#### `spell_acronym` consults the table before the length cap

The cap is about how long a thing may be before spelling it becomes worse than
leaving it, and it has nothing to say about a whole word. Capped first, every
entry over five letters would be dead: UNESCO, UNICEF and INTERPOL would never
reach the table.


#### `spell_acronyms` leaves a shout alone and reads a lone acronym

A run of capitalised words is emphasis, and spelling all of it would be the
loudest possible wrong answer. A text that is a single capitalised token,
`synthesize("GPT")`, is an acronym on its own: there is no run to read
emphasis from, and refusing it would refuse the one call shaped exactly like
"say this acronym".


### `loudkit/frontend/numbers.py`


#### `_FOREIGN_DIGITS`

Folded because the alternative was twelve different answers to the same string.
`_DIGIT_RUN` is ASCII by design, so ``١٢٣`` reached the model as written in
eleven languages, and in Polish it did not, because the respeller's own digit
test is `str.isdigit()`, which is true of every Unicode decimal digit. The same
input read as *sto dwadzieścia trzy* in one language and as three raw code
points in the rest, from one funnel reporting one fingerprint.

The separators matter more than the digits. U+066B is a decimal point, and it
is not in the `[.,]` this module looks for, so ``٣٫١٤`` lost its separator
entirely and was read as two numbers: *trzy czternaście*. That is the same
change of meaning as reading a decimal as a clock time, arriving through a
character set instead of a pattern.


#### `_UNICODE_MINUS`

Everything downstream reads the sign as `-`, so a typographically correct minus
was not a sign at all -- it fell through to the punctuation pass, which turns a
symbol between a space and a digit into a space, and "−5" was read as *five*.
The same loss of meaning the ASCII case was fixed for, and the temperature is
the opposite of what was written either way.

Only these two, and only before a digit. U+2013 EN DASH is how a *range* is
written ("1979–1983"), and U+2014 EM DASH is punctuation; folding either into a
minus would invent a sign where the text has none. The digit lookahead is what
keeps this from touching a hyphenated word.


#### `_PHONE_RUN`

Read digit by digit, and taken before `_DIGIT_RUN` because it is the one shape
that pattern cannot decline on its own. "+48 123 456 789" is a perfectly valid
one-to-three-then-threes grouping, so it was read as *forty-eight billion one
hundred and twenty-three million…* in eight of the twelve languages, and
"+1 202 555 0199" came out as a ten-digit cardinal with a bare "9" dangling
behind it, because the last group has four digits and only three of them fit.

The plus is the evidence. E.164 requires one and a grouped thousand never
carries one, so this is not a phone-number *detector* -- it does not guess at
national formats, area codes or separators -- it is the one written form that
says outright it is not a quantity.

`_MIN_E164_DIGITS` keeps it away from a signed number. "+5 degrees" and
"+250 points" are deltas, not numbers to spell out, and "+1 000 000 users" is a
million however it is punctuated. E.164 allows fifteen digits and a number
short enough to be a delta is not one.


#### `_TIME_RUN`

A colon between an hour and two minute digits is a clock time in every language
here, which is why this is the pattern that needs no help deciding.

The lookarounds are the whole point. ``\b`` on its own let this match *inside* a
longer run and read the front of it as a time, with the rest left behind as a
separate number. A time is a time only when nothing else is attached to either
end, so a digit, a dot, a comma or a colon on either side disqualifies it.
``14:30.`` at the end of a sentence still matches, because what follows the dot
there is not a digit.


#### `_DOTTED_TIME_RUN`

The dot is the whole difficulty. ``14.30`` is how German, Danish, Finnish,
Norwegian and Swedish write half past two in the afternoon; ``3.14`` is how
English writes pi. The shapes are identical and no lookaround separates them.

What separates them is already in the grammar file: **a language that writes
clock times with a dot does not use the dot as its decimal separator.** German
writes ``14.30 Uhr`` and ``2,50 €``; English writes ``2:30`` and ``$2.50``. So
this pattern applies exactly where ``decimal_separator`` is not ``.``, which
today means everywhere except English.

Applying the dotted pattern in a dot-decimal language reads every two-digit
decimal as a clock: ``$0.49`` as *zero forty-nine dollars*, ``3.14`` as *three
fourteen*. The pattern is therefore gated on the grammar's decimal separator,
which is what keeps those decimals intact.


#### `gender_scopes`

Three scopes exist in this language set, and they were found by the CLDR
differential rather than guessed:

- ``"standalone"``, only when the value is the entire number. Polish:
*jedna kobieta*, but *sto jeden kobiet* and *dwadzieścia jeden*, while
Polish 2 agrees everywhere (*dwadzieścia dwie*).
- ``"outside_tens"``, everywhere except inside the solid tens compound.
Danish: *hundrede og et* (agrees), but *enogtyve* (does not).
- the default, everywhere. Spanish: *treinta y una*.


#### `module`: the digit-run pattern

Eastern-Arabic digits take their own, five-way-conformant path through the
Polish respeller. A leading minus is part of the number, but only where it
cannot be a hyphen or a range: at a boundary, with a digit right behind it.
Anywhere else the punctuation pass turns it into a space, which is how "-5
degrees" would lose its sign and read as the opposite temperature. A space
groups thousands only when every group after the first is exactly three
digits and the first is one to three, so "in 2024 200 people" stays two
numbers. A grouped run must reach a boundary; a partial one is not a grouped
number. Without that rule the regex takes the longest prefix that fits, and
"+1 202 555 0199", a phone number, read as one billion and something in eight
of the twelve languages with a bare "9" behind it; refusing the whole join
drops each group back to being its own number. "+48 123 456 789" is a valid
1-3-plus-threes grouping that no boundary rule saves, so `_PHONE_RUN` takes it
first. A digit run touching a word at either end is part of the word and left
as written for the model, exactly as `iOS18` always was; the lookahead
`(?![\w])` mirrors the lookbehind, and `(?! ?[0-9])` makes the grouping rule
exact. Half-reading a token ("5x3" as *fivex3*, "1e6" as *onee6*) is worse
than either whole answer.


#### `say` walks back over word characters and dots

The question the lookbehind asks is whether the digit run is part of a token
that has a letter in it, and one character cannot answer it: an identifier can
put a dot between the letter and the digits, so in "v1.2.3" the scan starts at
the `2`. Walking back over word characters and dots answers it. The
alternative was teaching every later function about two more spellings of a
number.


#### `_glued_to_a_word`

The backward walk crosses commas, `-`, `+` and grouping spaces. The comma, so
`x3,14` is refused whole rather than read as "x3,vierzehn". The signs, because
an exponent puts one between the letter and the digits: in `1e-3` the scan
starts at the `3`, walks back over `-` to `e`, and stops calling it a number;
a bare `-5` reaches a space or the start and finds no letter. The grouping
space, because `x200 000` binds as a single match in Go and Rust, whose
engines have no backtracking, while Python, JS and Swift backtrack and match
the standalone `000`: five implementations, two answers, until the walk
crossed it too.

A space is a grouping space only when the group behind it is being continued:
one to three digits behind it, a digit ahead of it. `_continues_a_group`
rather than `_starts_a_group`, which is the forward walk's question, because
admitting a fourth digit here reaches the `e` of `e3 1000` and welds two
tokens into one. The simpler rules fail on real text: "a digit on each side"
crosses `R2 5`, "exactly three digits behind the space" breaks `a1 000 000`,
whose first group is legitimately one digit, and crossing without the
digit-ahead rule walks `Sold 200 000` from `000` to `200` into "Sold".


### `loudkit/frontend/speechtext.py`


#### `WHITE_SPACE`

`\s` is four different sets across the five implementations and the difference
is audible. Measured: ECMAScript `\s` excludes U+0085 NEL, so JS left it in a
clause where the other four collapsed it; Go's hand-expanded class omitted
U+000B; CPython's `\s` uniquely admits U+001C-U+001F. NEL in particular is
ordinary in scraped and epub text, and a separator that survives in one port
changes where that port's reader breathes.

Twenty-five code points is short enough to write down, and a written-down set
cannot drift the way a shorthand can. `_is_space` below answers the same
question for a single character; this is the spelling a pattern needs.


#### `_NOT_SPACE_IN_THE_PORTS`

FILE, GROUP, RECORD and UNIT SEPARATOR. Rust's `char::is_whitespace`, Go's
`unicode.IsSpace`, JS's `/\s/` and Swift's `whitespacesAndNewlines` all read
the Unicode White_Space property, which excludes them; CPython's `str.isspace()`
includes them. Measured over code points 0..0x2FFFF, these four are the **only**
disagreement between the two, so subtracting them is exactly White_Space and
nothing else moves.

Left in `str.isspace()`'s hands otherwise, rather than hand-writing the set the
way `frontend/chunking.py` does for `WORD_BOUNDARIES`: that tuple is short
because a cut point only ever lands on a handful of characters, while this
predicate has to keep every `Zs` there is.

The consequence of the difference was audible. Python kept the separator, so
the tokenizer saw `[UNK]` where the four ports saw `[SPACE]`, and `split_text`
found no word boundary there where the ports did, different tokens *and* a
different chunk split, for one text and one seed, under an
`algorithm_fingerprint` that says the five engines agree. `chunking.py` already
records the Unicode fact and hand-wrote its own table to dodge it; the funnel,
one layer earlier, was never fixed, and its docstring notes that these
characters "survive it" as though that were the design.


#### `module` keeps `¿` and `¡`

The inverted marks stay in the text. Spanish opens a question with them and
they are the earliest cue a reader has that one is coming; two shipped voices
read Spanish, and the four ports keep them.


#### `_speak_symbols` leaves a multi-character currency mark written

`R$`, `HK$` and `NT$` have no wording in the table, and matching the `$` alone
reads `R$3,14` as "R3,14 Dollar": the wrong currency, said confidently, with
the orphaned `R` in front of it. A mark this module cannot name is left as
written, the same answer it gives everywhere else when the evidence runs out.


#### `_speak_symbols` reads a suffixed price

`2.50 €` and `0.49¢` are prices by exactly the evidence `€2.50` is, and are
read before the time pass can see the dot, which read "zwei Uhr fünfzig Euro"
in German. Currency written as a word, `5.50 zł`, `12.30 kr`, is not covered:
those are ordinary words to every pass here, and telling them from a unit or
a name needs a per-language lexicon rather than a symbol table.


#### `_punctuation_for_speech` keeps a sign between alphanumerics

A hyphen or plus with an alphanumeric on either end stays, so `1e-3` and
`1e+3`, which the number pass declined as tokens with a letter in them, reach
the model whole rather than as "1e 3". A sign between spaces is still a space
in both readings.


#### `speech_text` normalises to NFC first

Unicode lets the same character arrive two ways: Polish ą as U+0105 or as a +
U+0328, Danish å as U+00E5 or a + U+030A, and the tokenizer's vocabulary
holds one of them. Without normalisation the decomposed spelling reaches every
rule below, every regex, lexicon lookup and character class, as a base letter
followed by an unknown combining mark. It runs before `_strip_invisibles`,
which removes format characters, because normalisation can compose a sequence
into a single character that would otherwise go unexamined.


#### `speech_text` folds foreign digits first

Every pass downstream asks "is this a digit", and the five ports spell that
question four ways: Python's `\d` matches a fullwidth digit, RE2's and
ECMAScript's do not. Folding first means every one of them sees ASCII digits
and the difference cannot arise (`\u20ac\uff11` read as *un euros* in
Python and *euros un* in JS). Safe to run first: no key in any table here is
a numeral.


#### `speech_text` pass order

Acronyms first, while the capitals are still capitals: every later pass
lowercases or rewrites, and a spelled acronym has to be decided while the
only evidence, that the word stands alone in caps, still exists. Not gated by
language: Swift and Go spell acronyms inside their Polish respelling modules
and the shared fixture pins the Polish result, so gating it here broke a case
all five implementations agreed on. The reach is the remaining divergence:
this pass spells acronyms in every language, the ports' respeller only in
Polish, so `CIA` in English text is recorded in the fixture's `divergent`
block.

Dates before times and numbers: `12.03.2026` is the ordinary written date of
five of these languages, the clock pattern matches `12.03` and the digit run
matches the lot, so a date recognised any later has already been read as a
time with a stray year or as one eight-digit number. The symbol pass has
already moved a currency mark behind its amount, so "£250" arrives at the
number pass as "250 pounds" with only the digits left to say; the expansion is
wired in every port, because a Python-only one would make five funnels produce
different text under one fingerprint.


#### `speech_text` folds a run of periods, not a pair

`re.sub` does not overlap its matches, so a pair rule turns "..." into ".." on
one pass and "." on the next: a funnel that is not idempotent, whose output
depends on the pass count. A run rule is idempotent. An ellipsis reaches here
as three periods from the symbol map.


#### `_respelled` returns an unknown token unchanged

A word this function cannot improve is returned as written, because losing
text is not an available outcome. The word collector keeps `'` and `’` inside
a word, so a spaced apostrophe arrives as a token of its own; filtered to the
characters the ASCII-keyed table knows, it would vanish from the utterance,
and it is in `_PROSODIC`, which the funnel is supposed to keep.


#### `_respell_words` uses `isdecimal`

`str.isdigit()` is true of `No` characters (`²`, `③`), so `²9` would be one
token that `int()` refuses and the symbol pass then reduces to a bare `9`
nothing verbalises. A bare digit reaching a grapheme model is the outcome this
layer exists to prevent, and it is not audible as a failure. Nd is what
`int()` accepts and what `unicode.IsDigit`, `char::is_numeric` with Nd and
`\p{Nd}` mean in the four ports, so it is the definition they can all agree
on.


#### `_respell_words` measures a dotted run whole

The decision belongs to the run, not to its first pair. Two groups is a
decimal: "dwa przecinek pięć". Three or more is a version, an address or a
date, and is left exactly as written; reading the first pair gives "jeden
przecinek dwa" with a stray ".trzy" behind it. The same holds when the run
starts with a token that has a letter in it: `v1.2.3` is a version whether or
not its first group is all digits. `numbers.expand` declines these whole by
the same rule, and this second reader of digits has to agree with it.


### `loudkit/frontend/text.py`


#### `encode` refuses bracket tokens

The vocabulary holds 117 bracket tokens: the language tags, and paralinguistic
events like [sigh], [gasp], [UH] trained into the base model. The tokenizer
matches them greedily, so "he [sigh]ed" would emit control token 611 and the
model would sigh. The funnel destroys brackets on the engine path; `encode`
refuses them too, so the guarantee is structural for anyone calling it
directly. The one tag that belongs is the language tag, added after.


### `loudkit/frontend/textconfig.py`


#### `FUNNEL_PORTED`

Not a feature name, a **contract marker**. Two builds that read the same text
into different words must not report the same sixteen hex digits, and this is
the field that keeps them apart. It moves when a pass changes what the funnel
emits for text it already handled, and only then.

The passes, in order, in all five implementations: NFC, invisibles, **markup
tags**, **numeral folding**, symbols, footnote markers, **Roman numerals**,
acronyms, dates, ordinals, abbreviations, clock times, numbers, punctuation,
respelling.

What each bump was:

`funnel-1` -> `funnel-2`
NFC, acronym spelling, dates and ordinals had landed in Python alone and
stayed there while all five reported `funnel-1`: one fingerprint over four
different funnels. `funnel-2` is the funnel with all of them, in all five.

`funnel-2` -> `funnel-4`
One family, closed in one move: **every character class the five ports
disagreed about**. They had drifted independently and each disagreement was
audible.

* *Numerals.* A superscript, a vulgar fraction, a circled or Roman numeral,
a digit from any script but Latin, all were deleted by the punctuation
pass, and while they were still there they sat *inside* the word, because
`\w`, `\p{N}` and `str.isdigit()` all admit `No`. So `²9` was one token,
the number matcher declined it, and a bare `9` reached the model.
:func:`~loudkit.frontend.speechtext.fold_numerals` now turns each of them
into something a reader says aloud, and nothing is dropped.
* *Letters.* Rust and Swift tested the **Alphabetic** property where the
other three tested `\p{L}`, wider by the circled letters, the
Other_Alphabetic marks and the letter numbers. Swift's copy put a bare
digit in front of the model for `ⓐ-1`; Rust's kept characters the other
four removed. Swift also read `CharacterSet.letters`, which Foundation
answers wrongly above the BMP: it omits all 6,145 Tangut ideographs, so
that text was deleted, and `subtracting(.nonBaseCharacters)` silently
fails for astral scalars, so 1,162 combining marks leaked through.
* *Whitespace.* `\s` was four different sets. ECMAScript's excludes U+0085
NEL, Go's hand-expanded class omitted U+000B, CPython's uniquely admits
U+001C-U+001F. Go read a footnote marker *aloud* when its separator was a
non-breaking or thin space, which is ordinary French and German
typography; JS split chunks in different places, because a NEL left on
the front of the remainder is charged against the next chunk's budget.
:data:`~loudkit.frontend.speechtext.WHITE_SPACE` is the one written-down
set every pattern now uses.

There is no `funnel-3` in any released build. It named the first third of
this family, the `No`/`Nl` half of the numeral fix, and was superseded
inside the same release, before a tag existed. Numbering a partial fix was
the mistake; a marker should move once per audible change to the funnel,
and this was one change.

`funnel-4` -> `funnel-5`
The clock pass wrote its words against the letters that followed the
digits. `3:45pm` read *three forty-fivepm*, one word to a listener, where
`3:45 pm` read correctly: a space in the source, which the reading does not
otherwise depend on, was deciding whether the meridiem was a word. Two
rules close it, and both are about the same boundary. The reading is
followed by a space when an ASCII letter stands where the written time
ended, so the glued and spaced forms produce the same output. And the
written infix is taken whether or not a space precedes it, so German
`14:30Uhr` reads *vierzehn Uhr dreißig* rather than saying the infix twice.

Text with a digit or a separator after the time is untouched: the guards
that tell `14:30` from the `12.03` inside a date are the same guards. So is
a letter *before* the time, which `Meet at a14:30.` pins.

`funnel-5` -> `funnel-6`
One family, and it is the family every caller meets: **meaning carried by a
mark the funnel had no rule for.** Each of these deleted or corrupted
something the writer wrote, in prose nobody would call unusual.

* *Comparison operators.* `≤` and `≥` had words and `<=` and `>=` did not,
so `if latency > 200 ms` read *if latency two hundred ms* -- a condition
with its relation removed, which states the opposite as readily as the one
written. Six ASCII spellings are read now, `<` `>` `<=` `>=` `!=` `==`, and
only with whitespace on both sides: the spacing is the evidence that the
mark is an operator rather than markup, an emoticon or a glued token. `-`,
`/`, `.` and `+` stay silent, because each is a hyphen, a path, a decimal
and a sign far more often than an operator.
* *Markup.* A tag's name reached the model as a word, `<p>Hello</p>` read
*p Hello p*, and an HTML comment left its `!` behind as a sentence-final
exclamation. A tag becomes a space rather than nothing, so two block tags
meeting are two paragraphs rather than one glued word.
* *Roman numerals.* Spelled letter by letter: `Chapter IV` read *Chapter
eye-vee* and `World War II` read *World War eye-eye*, inside a book reader.
The pass reads I, V and X only, which is 2 to 39 -- chapter, act, volume,
war and regnal numbers -- and that boundary is the whole rule rather than a
narrowing of it: every two-letter initialism that is also a valid numeral
needs L, C, D or M (`CD`, `CV`, `DC`, `MC`, `MD`, `XL`, `CM`), and so does
`MIX`.
* *Fractions.* `½` read *one two*, two numbers instead of a value. A slash
inside a **spelled numeral** is a fraction bar the character itself
asserts, so it becomes `÷` and takes the word the grammar already has. A
typed `1/2` is left alone: `24/7` and `and/or` are ordinary prose that
reads correctly, and `4/7` is a date to half the world.
* *Prices.* The magnitude sat on the wrong side of the currency word.
`$2.5M` read *two point five dollarsM* and `$5 million` read *five dollars
million*. Both shapes are one defect -- a scale written after the amount
and before the mark that moved -- and both read `amount, scale, currency`
now. The abbreviating letter is only read beside a currency mark, which is
what makes it unambiguous: a bare `5m` is five metres as readily.
* *Dates.* The English day-first reading supplies an article the sentence
may already carry, so `See the 3 April minutes` read *See the the third of
April minutes*. The prefix now reads what stands before it, which is the
same question `am`/`den`/`vom` already ask for German.
* *Clock times.* `10:30:45` kept its colons, so a literal colon reached the
acoustic model. The colon form takes seconds; the dotted form does not,
because `10.30.45` is a version string as readily as a timestamp. A zero
minute is spoken where seconds follow it, or the seconds move into the
minutes' place; a zero seconds field is dropped, saying nothing the minute
has not said.
* *ISO datetimes.* `2026-03-04T10:00` read *…twenty twenty-sixTten*: the
field separator is not a letter and glued itself to the last word of the
date. It is taken with the date and becomes the space that keeps the two
readings apart.
* *Where a date and a telephone number end.* A run that continues into a
letter is an identifier, and every guard on the three numeric date forms
named the digits and the separators and admitted the letter: `2026-03-04x`
read *…twenty twenty-sixx* and `x25/03/2026` invented a date out of the
middle of a token. The E.164 rule had no guard at all, so `+12345678abc`
said eight digits and left the letters glued to the last of them. All four
are bounded by a word boundary now, at both ends, which is the guard the
dotted form already carried behind it and the one the digit run answers to.

A bare year is **not** in this family and did not move. `In 1776 he wrote
it.` reads one thousand seven hundred and seventy-six in all twelve
languages, none of them excepted: the year reading belongs to a written
date, where the month says a year is what the digits are, and nothing in a
bare four-digit run says so.


#### `recipe`

Every implementation runs the same passes under one recipe value; that is
the state in which this field's promise holds. A build whose passes differ
while reporting the same recipe would cover divergent readings of
``12.03.2026``, ``1st``, ``CIA`` or a decomposed ``ą`` under one
fingerprint, the failure :data:`FUNNEL_PORTED` exists to prevent.
