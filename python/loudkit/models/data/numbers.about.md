# `numbers.json`

This note is kept outside `numbers.json`. The raw bytes of that file are hashed
into the grammar digest, and through it into `algorithm_fingerprint`, which
moves when the audio changes. A sentence inside the data file would make a
prose edit change the fingerprint in all five implementations, while every
sample renders the same.

The file holds number grammars for the twelve languages of the text layer,
written from reference descriptions of each language. All five
implementations read this file, so each rule exists once, and the conformance
fixture catches a port that differs.

The format is documented on `loudkit.frontend.numbers.Grammar`. It has a
regular core (ones, teens, tens, hundreds, scales) and a list of irregular
forms.

`unit_tens_joiner` carries its own spacing, so one concatenation serves a
hyphenating language (en), a spacing one (es) and a compounding one (de).

Scale `forms` has one word when the language does not inflect the scale noun.
Otherwise it has one word per grammatical number: two for singular and plural
(German *Million / Millionen*), or three for singular, few and many (Polish
*tysiąc / tysiące / tysięcy*).
