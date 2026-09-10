# `pl_respell_rules.json`

The hand-written half of Polish respelling. The other half,
`pl_en_respell.json`, is generated from CMUdict by `tools/gen_pl_respell.py`;
this file is the part chosen by ear, and the curated `lexicon` here always wins
over the generated long tail.

Prose lives here rather than in the data file, following `numbers.about.md`:
this file is not in `grammar_digest` today, and when it joins the digest a
sentence inside it would move the fingerprint while every sample rendered
identically.

Until it joins, an edit to the data file changes spoken Polish without moving
the fingerprint. `tests/test_speechtext.py` pins its bytes so that the edit
fails a test instead of shipping quietly; the failure says what to bump.

Five members, all Polish:

- `phrases` is ordered and applied in order, before the word pass. A multi-word
  anglicism is respelled whole because word by word is wrong: "release notes"
  would read `notes` as the Polish homograph, the notebook.
- `lexicon` maps a common anglicism to its Polish phonetic respelling. An entry
  earns its place by the grapheme reading audibly failing, so words Poles
  already read correctly by Polish rules (laptop, internet, blog, film) are
  deliberately absent. The order is the curated one: computing first, everyday
  code-switching after.
- `keep_polish` is English words that are also everyday Polish words. Two
  families: Polish homographs, and loanwords Poles read orthographically ("bug"
  is [bug] in Polish mouths, never [bag]). Only a phrase may respell one.
- `function_words` is the Polish words that happen to spell English ones ("i",
  "to", "on"). They never join an English span, or the span eats the Polish
  conjunction after it.
- `endings` is the Polish case and derivation endings these loanwords take,
  which is how `deadline'u`, `maila` and `updatem` find their stems.

Read by `loudkit.frontend.speechtext`, and by Swift's `LexicalRespelling` from
the copy `tools/sync_grammar.py` puts in `swift/LoudKitText/Resources/`. Go,
Rust and JS still carry the same tables as literals; each joins the sync list
when it reads the file instead.

## Why the generated lexicon is tracked four times

`pl_en_respell.json` is 6.6 MB and four copies of it are tracked, under
`python/loudkit/models/data/`, `go/speechtext/`, `rust/src/` and
`swift/LoudKitText/Resources/`. Together they are about a third of the
repository, and they are deliberate.

Go embeds the file with `go:embed`, Rust with `include_str!`, and a Swift
package resolves its resources from the checkout. All three publish the git
tree itself, so a copy the tree does not carry is a package that does not
build. Only npm publishes a build artefact rather than a checkout, which is why
`js/data/` is gitignored and `prebuild` copies the file into it.

`tests/test_respell_data.py` hashes the three copies against the Python one, so
a drifted copy fails instead of teaching one language a different Polish.
