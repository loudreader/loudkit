# `pl_respell_rules.json`

The hand-written rules of Polish respelling. The generated part is
`pl_en_respell.json`, which `tools/gen_pl_respell.py` builds from CMUdict. The
rules in this file were chosen by ear. An entry in the curated `lexicon` here
takes precedence over a generated entry.

`pl_respell_rules.json` is not in `grammar_digest`, so an edit to it changes
spoken Polish without moving the fingerprint. `tests/test_speechtext.py` pins
the bytes of the file, so an edit fails that test, and the failure says what to
bump. This note is kept outside the data file, like `numbers.about.md`, so that
a prose edit does not change the bytes of the data file.

The file has five members, all Polish:

- `phrases` is ordered and applied in order, before the word pass. A
  multi-word anglicism is respelled as a whole, because a word-by-word
  respelling reads it wrongly: in "release notes", `notes` would be read as the
  Polish homograph that means notebooks.
- `lexicon` maps a common anglicism to its Polish phonetic respelling. An entry
  is added only when the Polish reading of the spelling is audibly wrong, so
  words that Polish rules already read correctly (laptop, internet, blog, film)
  are not in it. The order is curated: computing terms first, then everyday
  code-switching.
- `keep_polish` lists English words that are also everyday Polish words, in two
  groups: Polish homographs, and loanwords that Polish speakers read by Polish
  spelling rules ("bug" is [bug] in Polish, not [bag]). Only a phrase entry can
  respell one of them.
- `function_words` lists Polish words that are spelled like English words
  ("i", "to", "on"). They never join an English span. Otherwise the span would
  take in the Polish conjunction that follows it.
- `endings` lists the Polish case and derivation endings that these loanwords
  take. They let `deadline'u`, `maila` and `updatem` match their stems.

Python reads this file through `loudkit.frontend.speechtext`. Swift's
`LexicalRespelling` reads the copy that `tools/sync_grammar.py` puts in
`swift/LoudKitText/Resources/`. Go, Rust and JS carry the same tables as
literals.

## Why the generated lexicon is tracked four times

`pl_en_respell.json` is 6.6 MB, and four copies of it are tracked, under
`python/loudkit/models/data/`, `go/speechtext/`, `rust/src/` and
`swift/LoudKitText/Resources/`. Together they are about a third of the
repository.

Go embeds the file with `go:embed`, Rust with `include_str!`, and a Swift
package resolves its resources from the checkout. Go and Swift packages are
fetched from the git tree, and the Rust crate is packaged from the files under
`rust/`, so each of the three needs its own tracked copy to build. npm
publishes a build artefact instead: `js/data/` is gitignored, and `prebuild`
copies the file into it.

`tests/test_respell_data.py` compares the hashes of the three port copies with
the Python copy, and fails when a copy differs.
