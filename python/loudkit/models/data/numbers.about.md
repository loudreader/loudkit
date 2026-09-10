# `numbers.json`

Moved out of the data file itself: `numbers.json` is hashed as raw bytes
into the grammar digest and through it into `algorithm_fingerprint`, whose
one promise is that it moves when the audio moves. Prose inside a hashed
file breaks that promise -- editing a sentence re-fingerprints the engine
and re-pins five ports while every sample renders identically.

Number grammars for the nine languages the kit speaks, written from the
languages' own reference descriptions rather than derived from any library.
This file is the single source of truth: five implementations read it, so a
rule lives once and the conformance fixture catches a port that drifts.

Format is documented on `loudkit.numbers.Grammar`. In short: a regular
generative core (ones, teens, tens, hundreds, scales) plus a listed set of
irregular forms, because that is how these grammars are actually described.

`unit_tens_joiner` carries its own spacing, so one concatenation serves a
hyphenating language (en), a spacing one (es) and a compounding one (de).

Scale `forms` is one word where the language does not inflect the scale noun
and three where it does (Polish: singular / few / many).
