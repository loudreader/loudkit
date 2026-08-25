# Releasing loudkit

The release workflow (`.github/workflows/release.yml`) builds and verifies the
artefacts, attests their provenance, coordinates the first npm and crates.io
bootstraps and, after a reviewer approves the `release` environment, publishes
PyPI before creating the GitHub Release. The manual commands in §7 are the two
first-package bootstraps, the fallback and the record of what the jobs do. This
file is the whole procedure, in order.

Read §0 before anything else. The public history of this project begins at §5,
in a single commit, and §5 cannot be redone.

## 0. Preconditions

### 0.1 An empty `loudreader/loudkit` exists, with a separate public remote

Everything downstream assumes that path. The Go module path is
`github.com/loudreader/loudkit/go`, the crate, the wheel and the npm package
all carry `repository` URLs under that org, the docs site builds for
`loudreader.github.io/loudkit`, and the README Colab badge points into
`loudreader/loudkit`.

**This is a fresh repository, not a transfer.** The public history starts as
one squashed commit pushed from this working tree at §5. What that costs, and
what it buys:

- **No redirect from the private staging repository.** A transfer would have
  preserved its full history. Nothing in the release relies on that URL.
- **No issues, pull requests, stars or watchers carry over.** They stay on the
  private repository, which remains private.
- **No old commits.** This is the security upside, and it is the reason for
  the choice: nothing can leak from a history that does not exist. No stray
  key, no scraped page, no working note in an old tree, no author email in a
  commit nobody re-reads. The §4.9 audit only has to hold for one tree.

Do now:

- [ ] Create `loudreader/loudkit` in the org, **empty**: no README, no
      `.gitignore`, no licence template. Any initial commit GitHub makes for
      you has to be discarded before §5 and is easiest never to create.
- [ ] Make it public.
- [ ] Keep the private staging remote separate and add the public repository:
      `git remote add public https://github.com/loudreader/loudkit.git`.
      If `public` already exists, verify it with `git remote get-url public`.
- [ ] Confirm anonymously, from a logged-out session:
      `curl -s -o /dev/null -w '%{http_code}\n' https://api.github.com/repos/loudreader/loudkit`
      must print `200`.

**This step comes first and cannot be reordered.** The Go module proxy, the
Colab badge and every `repository` link in the three published manifests
resolve against this path. Publishing a crate or a wheel whose `repository`
URL 404s is not reversible by editing the field later: crates.io metadata is
frozen per version, and a PyPI version number is spent once uploaded.

### 0.2 The checkpoint is published and reachable without authentication

The target repository `loudreader/loudr-1` exists, is public and is ungated.
That is all this section asserts. What it holds is §5.5's business, and nothing
before §5.5 updates it.

Everything a reader is told to run depends on the contents landing there: the
README block, the Colab badge, all five quickstarts.

**This is a precondition of §7, not of §0.** The bundle cannot be built here,
because the model card is a hashed member of it and the card embeds audio the
public repository has to serve -- so the repository has to be published first
(§5), and only then can the card be finalised, the bundle built and uploaded
(§5.5). Reading this document top to bottom does the right thing; the check
that this precondition holds is §5.5's closing checklist.

### 0.3 Registry accounts and names

- [ ] Accounts exist with two-factor authentication enabled on PyPI, npm and
      crates.io.
- [ ] The name `loudkit` is free on all three. Verified 2026-08-23: PyPI, the
      npm registry and crates.io each answer 404 for it. Re-check on the day.
      A name taken between now and then changes the release, not just a field.

## 1. Version sync

Four files carry a version and must agree before tagging. Go and Swift take
their version from the git tag and have nothing to edit.

| file | field | pre-release value | release value |
| --- | --- | --- | --- |
| `pyproject.toml` | `version` | `0.1.0.dev0` | `0.1.0` |
| `js/package.json` | `version` | `0.1.0-dev0` | `0.1.0` |
| `rust/Cargo.toml` | `version` | `0.1.0-dev.0` | `0.1.0` |
| `python/loudkit/__init__.py` | `__version__` | `0.1.0.dev0` | `0.1.0` |

The release workflow refuses a tag that does not match `pyproject.toml`. A
missed edit fails the build instead of shipping a mislabeled wheel. The other
three are held to `pyproject.toml` by
`tests/test_release.py::test_every_published_manifest_carries_the_same_version`,
and the table above is held to all four by
`test_the_release_table_names_the_versions_the_files_carry`. Edit the table in
the same commit as the files, or the suite goes red.

`tests/test_release.py` is the contract. Run it after the edit and before the
tag:

```bash
pytest tests/test_release.py -q
```

## 2. The release commit

One commit, containing all of the following. They are grouped because each of
them is a claim that stops being true at the moment of publication, and a
release that ships half of them documents a state that does not exist.

- [ ] The four version fields from §1, and the §1 table itself.
- [ ] `CHANGELOG.md`: `## [0.1.0] — XXXX-XX-XX` becomes `## [0.1.0] — <date>`.
- [ ] The two audio players go into the "Listen" section of
      `docs/MODEL_CARD.md`. They point at `samples/` in the Hugging Face model
      repository; the strict builder copies those bytes from the voice roster
      and includes them in both manifests. This is the only place the card can
      be written, because §5.2 freezes the tree and the card is a hashed member
      of the bundle §5.5 builds. §5.5 confirms they answer.

          <audio controls src="https://huggingface.co/loudreader/loudr-1/resolve/main/samples/joe.opus"></audio>
          <audio controls src="https://huggingface.co/loudreader/loudr-1/resolve/main/samples/kathleen.opus"></audio>

      Both read the same sentence. Their seeds differ and the card does not
      pretend the voice is the only changed variable.
      Nothing from `docs/voices/roster/audio/refs/`: those are compressed
      previews of the human enrollment recordings, not model output, and they
      are deliberately absent from the model repository. A player beside the
      model name reads as a sample of the model.
- [ ] Every banner in §9 comes off. `tests/test_release_coherence.py` refuses a
      stable version while any of them stands, so this is the section that
      unblocks the tag in §6.
- [ ] Decide whether this release is a patch, a minor or a major against the
      promises in
      [docs/reference/COMPATIBILITY.md](docs/reference/COMPATIBILITY.md).

## 3. Local gates

All of these must pass on the release commit, from the repo root:

```bash
pytest -m "not slow" -q
mypy python/loudkit && mypy --config-file tools/mypy.ini tools/
ruff check python tests tools integrations/speech-dispatcher
ruff format --check python tests tools integrations/speech-dispatcher
(cd rust && cargo clippy --all-targets -- -D warnings && cargo test)
(cd go && go vet ./... && go test ./...)
(cd js && npm test)
swift test
```

## 4. Manual acceptance pass: use the library as a stranger would

Do this **before** tagging, on the tree you intend to tag. Each item is a
role-play: no repo checkout knowledge, no cached state, only what a new user
has. A failure here is a release blocker.

The release operator can execute every command in §§4.1–4.6 and attach the
output. The release owner does not need to retype those commands. These are the
five decisions that still require a person's judgment rather than a green
process exit:

- [ ] **GitHub first minute.** Open the rendered README in a private window.
      Without diving into the technical proof, can you say what loudkit is,
      who it is for, how to make the first WAV and where to go for your
      runtime? Reject it if the page feels like an internal specification.
- [ ] **Hugging Face first minute.** Both native players render and play. The
      card answers in this order: how it sounds, how to try it, what gets
      downloaded, and what the quality boundary is. Reject it if internals
      dominate before the first successful synthesis.
- [ ] **Listening.** Hear all twenty roster samples from start to finish, then
      a fresh English, Polish and Spanish render. Reject dropouts, repetitions,
      bad tails, wrong-language number reading or a sample you would be
      uncomfortable presenting as the model's first impression.
- [ ] **Cloning.** Enroll one recording for which you have explicit permission,
      listen to at least two sentences, and decide whether identity and
      intelligibility are good enough for the v0.1 claim. A passing cosine or
      successful API call cannot make that decision.
- [ ] **Claims and responsibility.** Read the top half of the model card,
      `VOICES.md` and `RESPONSIBLE_USE.md` as the person whose name is attached
      to the release. Confirm that the English-only quality boundary, consent
      basis and C2PA trust boundary are described clearly and accurately.

Everything below is reproducible operator evidence. It still has to pass, but
it may be delegated.

### 4.1 The Python user

- [ ] Build locally and install the wheel into a fresh venv, not `pip -e .`.
      The glob has to be expanded before pip sees it, so resolve it into a
      variable rather than quoting a pattern:

      ```bash
      python -m build
      python -m venv /tmp/lk
      whl=$(ls dist/loudkit-0.1.0-*.whl)
      /tmp/lk/bin/pip install "${whl}[torch,audio,hub]"
      ```

- [ ] Run the README "Make a WAV" block **verbatim, copy-pasted**: the download
      path, the cache, `engine.synthesize(...)`, `.save("hello.wav")`.
      Listen to `hello.wav`.
- [ ] Repeat the synthesize line in `pl` and `es`. Listen: numbers, currency
      and times in a sentence like „Pociąg o 14:30 kosztuje 2,5 mln zł" must
      come out in the right language, not English.
- [ ] Enrollment: `loudkit.enroll(...)` on a 10-second clip, synthesize with
      the result, listen for identity.
- [ ] Break it on purpose and read the errors as a stranger:
      wrong language code produces a 400-class message naming the supported
      set; no network on first load produces an error that says what to
      download, not a traceback from inside `huggingface_hub`.
- [ ] `python -c "import loudkit; print(loudkit.__version__)"` matches the tag.
- [ ] `twine check dist/*` passes on both artefacts.

### 4.2 The Colab user

- [ ] Open the README Colab badge **logged out of GitHub**. It points at
      `loudreader/loudkit`, opens without authentication and does not fall back
      to a contributor's fork.
- [ ] The pre-release banner is off both the README and the notebook (§2).
- [ ] Runtime → Run all, no edits. Every cell green, audio plays inline.

### 4.3 The npm user

- [ ] `cd js && npm pack --dry-run`. The `prepack` guard
      (`js/scripts/check-pack.mjs`) runs `npm run build`, copies the data files
      and refuses a tarball missing `dist/`, either data file, `LICENSE`,
      `NOTICE` or `DISCLOSURE`. Confirm the listing carries those files,
      `README.md`, `data/numbers.json`, `data/pl_en_respell.json` (about
      6.6 MB) and `dist/`, and that no `dist/test/` entry appears. Confirm
      `package.json` declares `contentPolicy.class` as `dual-use`.
- [ ] In a scratch dir: `npm install /path/to/loudkit-0.1.0.tgz`, then run the
      example from `js/README.md` against downloaded weights.

### 4.4 The Rust user

- [ ] `cargo new /tmp/lk-rs && cd /tmp/lk-rs`, add the crate as a path
      dependency, paste the example from `rust/README.md`,
      `cargo run`. (Post-publish this repeats with the registry version.)
- [ ] `cargo publish --dry-run` from `rust/` succeeds. It reads the index and
      uploads nothing. It needs no token.
- [ ] `cargo package --list` still shows `src/numbers.json` and
      `src/pl_en_respell.json`. The crate is unusable without them.
- [ ] `cargo package --list | grep -c '^tests/'` prints `0`. Several of those
      tests panic without the monorepo's fixtures, so shipping them makes
      `cargo test` on the published crate report failures that say nothing
      about the crate. The `include` list in `rust/Cargo.toml` keeps them out.
      An `include = ["tests/**"]` undoes it silently.

### 4.5 The Go user

- [ ] Pre-push stand-in: `go build ./...` and `go test ./...` from `go/`.
- [ ] Post-push, in a scratch module:

      ```bash
      go mod init tmp
      go get github.com/loudreader/loudkit/go@v0.1.0
      ```

      **The version query is `@v0.1.0`, not `@go/v0.1.0`.** The `/go` suffix on
      the module path is what tells the proxy the tag it wants is prefixed
      `go/`. Writing the tag name into the query fails to resolve.
      Paste the example from `go/README.md`, then `go run .`.

### 4.6 The Swift user

- [ ] Pre-push stand-in: `swift build && swift test` at the repo root.
- [ ] Post-push, a scratch package depending on
      `.package(url: "https://github.com/loudreader/loudkit", from: "0.1.0")`,
      built and run against the example from the Swift section of the README on
      macOS. SwiftPM strips the leading `v` from tags, so `from: "0.1.0"`
      resolves the `v0.1.0` tag. No separate Swift tag is needed:
      `Package.swift` is at the repo root precisely because SwiftPM cannot read
      a manifest from a subdirectory.

### 4.7 The listener

- [ ] Play every sample under `docs/voices/roster/audio/` start to finish. Any
      dropout, repetition, tail artifact or wrong-language reading disqualifies
      the sample. Re-render it, put the new sha256 in
      `docs/voices/roster/provenance.json`, and ship the new bytes. The hashes
      record what is published. They do not reproduce on another machine.

### 4.8 The reader

- [ ] On the GitHub rendering of the public repo, click **every** link in
      README.md and VOICES.md: relative links, docs/, the model card, the
      benchmarks page. No 404s, no links into files that no longer exist.
- [ ] Read `docs/MODEL_CARD.md` for anything a reader would trip on. §2
      finalises it before it becomes a hashed member of the model bundle, so
      this is the last pass where a wording fix is free.
- [ ] The repo has a description, topics, and issues enabled.

### 4.9 The auditor

- [ ] The published tree contains nothing that is not the project's to ship:
      working notes, scraped pages, probe outputs, `.env`, credentials.
      Verified 2026-08-23: `release-dir/`, `space-dir/`, `out/`, `dist/`,
      `jobs/`, `progress.md`, `.opencode/` and `.agents/` are all untracked
      and gitignored.
- [ ] `LICENSE` (Apache-2.0) and `NOTICE` are at the root and render on GitHub.
- [ ] `go/`, `rust/` and `js/` each carry a byte-identical copy of both, held
      there by
      `test_every_published_package_carries_the_licence_and_the_notice`.

## 5. Settle the tree, squash, push to `main`

This is the step that creates the public history. It happens once.

### 5.1 Confirm the tree has settled

- [ ] `git status --porcelain` prints nothing.
- [ ] Nobody else is mid-edit. During the 2026-08-23 audit the working tree
      grew from 6 modified files to 12 while the gates were running: a parallel
      agent owned `site/**` and `docs/benchmarks.md` for a visual-review pass,
      and `tools/build_voices_md.py`, `VOICES.md` and `docs/MODEL_CARD.md` were
      being corrected at the same time. None of it came from another machine,
      and all of it was legitimate, which is the point: a tree can be moving
      under you while every gate reports green, because the gates ran against
      an earlier tree than the one you are about to squash.
- [ ] Re-run §3 **after** the tree stops moving, not before. A green run over a
      superseded tree proves nothing about the commit you push.

### 5.2 Create the public root and push

- [ ] Produce **one parentless commit** containing the whole settled tree and
      point a local `public-main` branch at it. Everything from §2 is inside
      it; none of the private staging history is an ancestor.
- [ ] `git rev-list --parents -n 1 public-main` prints exactly one hash.
- [ ] `git diff --exit-code HEAD public-main` prints nothing.
- [ ] `git push -u public public-main:main`

**`main` must be correct on the first push.** A squashed history has no earlier
commit to fix it from and no previous release to roll back to. A follow-up
commit is a visible correction on a repository whose entire history is two
commits, and any secret in the squashed tree is public the moment it lands.

### 5.3 Verify before tagging

- [ ] The repository is public: the anonymous 200 from §0.1 still holds.
- [ ] The pushed tree matches local, exactly:
      `git fetch public && git diff --stat public/main public-main` prints
      nothing.
- [ ] Spot-check the rendered `README.md` and `VOICES.md` on GitHub. The
      pre-release banner is gone (§9) and no link 404s.
- [ ] The docs site has deployed from `main` and carries no banner.

Tagging is the next step and it is what the Go proxy caches. Do not tag until
all four boxes are ticked.

## 5.5 Build the model bundle and upload it

Two things have to be true before the build reads anything.

**The checkpoint is split.** A release ships two halves, and the builder looks
for the enrollment one *beside* the synthesis one under its canonical name --
there is no flag for it. Both have to sit in one directory, which is not the
checkpoint's own: the tool refuses that, because a half written beside the
packed original is a different artefact wearing the same directory.

```bash
.venv/bin/python tools/split_checkpoint.py \
    --checkpoint assets/loudr-1.safetensors \
    --out-dir    dist/split
```

**Only if the pair is not there yet.** With both halves present the tool says
so and stops, which is correct and needs no answer -- do not reach for a force
flag. The pair does not need re-cutting to be trusted: each half carries the
digest of the packed original, and the build refuses two halves that came from
different runs.

**The model card is already final**, from §2. Nothing here edits a tracked file:
§5.2 pushed the last commit and §6 tags it, so a card that changed after that
point would ship to the Hub without existing in `main`, and the tag would not
cover it. Read it, do not write it.

- [ ] The two audio players §2 added name the model repository's own
      `samples/joe.opus` and `samples/kathleen.opus`, not GitHub raw content.
      They cannot answer until the new bundle is merged; the post-merge check
      below is the listening gate.
- [ ] The card's claims still match the artefacts: voice count, language list,
      artefact sizes, the download table.
- [ ] Every link in the card resolves against the site §5 published.

Then build the bundle with the strict profile. It is the default, and it is the
only profile a release is cut from:

The inputs are settled, so the command is the command rather than a shape to
fill in. `--checkpoint` names the *synthesis half*, and the builder finds the
enrollment one beside it:

```bash
.venv/bin/python tools/build_release.py \
    --checkpoint    dist/split/loudr-1.safetensors \
    --voice-encoder assets/ve.safetensors \
    --voices        assets/voices \
    --onnx          assets/onnx \
    --coreml        assets/coreml \
    --out           dist/loudr-1 \
    --profile       full-0.1
```

`--profile full-0.1` requires the checkpoint under its canonical name
`loudr-1.safetensors`, its manifest, the tokenizer, `ve.safetensors`, all 20
voices of the roster by name, 9 ONNX graphs, 6 CoreML packages, the four
documents, the LoudKit wordmark, the two listening samples, `release.json`,
and a `SHA256SUMS` covering every file. The roster
is `docs/voices/roster/provenance.json`: twenty voices, two per language
across ten languages. A missing voice is a refusal, and so is a voice that is
not on the roster.

The tool checks the sources first and refuses with a list of what is absent
and which tool exports it. It then assembles into a staging directory beside
`--out` and renames it into place only after every check has passed, so a
failed run leaves no directory that looks publishable. A run that is killed
outright cannot clean up after itself, so the staging directory carries the
pid of the build that owns it and the next run reclaims the trees whose
process is gone, naming each one as it goes. A build that is still running is
left alone, so two builds of one target do not eat each other. It copies the
graphs and the packages **by name**, then audits the assembled bundle against the
profile's allowlist: a file the profile does not name is an error, not a
passenger.

The closing gate loads what the bundle ships. It speaks on the torch path,
loads all 20 voices, clones a voice with the shipped `ve.safetensors`, speaks
on the ONNX path, speaks on the CoreML path, and runs one enrollment through
the three enrollment graphs on both graph paths, against the enrollment
fixture.

**Cut the release on macOS.** Six of the shipped artefacts are CoreML
packages, and they only open on an Apple platform. `full-0.1` refuses on any
other platform rather than skipping that half of the gate, because a bundle
whose packages nothing ever opened is the defect this profile exists to
prevent.

`release.json` records the profile that built it and `"verified": true` when
the closing gate ran and passed, so a consumer or a CI job can tell a
releasable bundle from a development one without counting files. It is written
before `SHA256SUMS` and covered by it, so the file that says a bundle is
trustworthy carries a checksum like every other file.

The gate imports the bundle's own code and runs it, so after the gate the
builder judges the bundle again from disk alone: every file is re-hashed
against the manifests, the inventory is matched both ways, and the allowlist
is re-audited. The manifests carry the digests taken before the gate, so a
gate that mutated a byte or added a file ends in a refusal, not in a bundle
stamped `verified: true` about bytes nothing verified.

The same audit runs in place, without a build:

```bash
.venv/bin/python tools/build_release.py --verify-only dist/loudr-1
```

It is one function (`check_bundle`) shared by both callers, so the pre-upload
check and the post-build check cannot drift apart. It does not load the model;
it says whether the bytes on disk are the bytes the manifests vouch for, and
whether the bundle matches the profile it claims.

`full-0.1` refuses `--skip-verify`. A bundle that names the profile is a
bundle that passed the gate.

`--profile lenient` builds a partial bundle for development. It writes
`"profile": "lenient"` and `"verified": false`. Do not upload one.

Before a byte leaves the machine:

- [ ] `.venv/bin/python tools/build_release.py --verify-only dist/loudr-1`
      passes. Run it immediately before the upload; it catches anything that
      touched the bundle after the build.
- [ ] `shasum -a 256 -c SHA256SUMS` passes from the release root, and
      `SHA256SUMS` has **one fewer line than the bundle has files** — the count
      difference below is exactly one. `SHA256SUMS` cannot contain its own
      digest; every other file, including `release.json`, every graph, package
      leaf and sample, has a line.

      ```bash
      (
        cd dist/loudr-1
        shasum -a 256 -c SHA256SUMS
        echo $(( $(find . -type f | wc -l) - $(wc -l < SHA256SUMS) ))   # 1
      )
      ```

- [ ] `release.json` says `"profile": "full-0.1"` and `"verified": true`.
      A lenient or unverified bundle is not a release.
- [ ] `diff docs/MODEL_CARD.md dist/loudr-1/README.md` is empty. The card is a
      hashed bundle member; editing it on the Hub afterwards makes the remote
      tree disagree with its own checksums.
- [ ] The layout has `voices/*.safetensors` at one level and exactly three
      safetensors at the root: the synthesis half, enrollment half and
      `ve.safetensors`. `_only_checkpoint_in` selects the synthesis role, so
      that ordinary three-file layout is not an ambiguity.
- [ ] Re-read
      [`docs/PROVENANCE-voice-encoder.md`](docs/PROVENANCE-voice-encoder.md).
      It records the encoder's hash, tensors, upstream repository and licence;
      publishing the weight is not reversible.

Open a new replacement pull request. Hub PR 2 already delivered the first full
bundle and is merged, so it cannot be reused. `--delete "*"` is part of the
command: without it, files that disappeared from the new release survive beside
the new inventory and the remote repository is no longer the bundle the builder
verified.

```bash
(
  cd dist/loudr-1
  hf upload loudreader/loudr-1 . . \
      --repo-type model \
      --create-pr \
      --delete "*" \
      --commit-message "loudr-1: complete v0.1 bundle"
)
```

- [ ] Review the Hub PR before merging. Its final tree, not only its additions,
      matches `dist/loudr-1`; the old packed checkpoint is deleted, both split
      halves are present, and `samples/` contains exactly the two players.
- [ ] Merge it in the Hugging Face interface, then record the immutable commit
      SHA. The default branch name is not a release identifier:

      ```bash
      curl -s https://huggingface.co/api/models/loudreader/loudr-1 \
        | python3 -c "import json,sys; print(json.load(sys.stdin)['sha'])"
      ```

- [ ] Open the rendered model card in a private window. The LoudKit wordmark
      is visible. Both players fetch from `samples/` on the model repository
      and play to the end. This is the first point in the sequence where those
      claims can be tested against the public bytes.

What is on the Hub is what the bundle contains, so the list below counts the
bundle.

- [ ] `loudreader/loudr-1` on Hugging Face is public and ungated, carrying
      `loudr-1.safetensors`, `loudr-1-enrollment.safetensors`,
      `ve.safetensors`, `tokenizer.json`,
      `manifest.json`, `release.json`, `SHA256SUMS`, the nine ONNX graphs
      (six synthesis, three enrollment), the six CoreML packages, the twenty
      voice profiles under `voices/`, `README.md`, `logo.png`, `LICENSE`,
      `NOTICE` and `RESPONSIBLE_USE.md`, plus `samples/joe.opus` and
      `samples/kathleen.opus` used by the card's native players.
      `SUPPORTED.md` declares voice enrollment in five ports. Enrollment
      needs `s3_tokenizer`, `camp` and `voice_encoder`, on both graph paths,
      so a bundle without those six pieces is not the release.
- [ ] **The first-mile test passes.** `lk.load("loudreader/loudr-1")`,
      `lk.voice(...)` and a synthesize, run against the public repo from a
      machine with no local cache, pinned to the SHA recorded above. A public
      checkpoint that does not load is the same blocker as a private one.

      ```bash
      cd "$(git rev-parse --show-toplevel)"
      CACHE_DIR=$(mktemp -d)
      RELEASE_SHA="PASTE_SHA_HERE"
      HF_HOME="$CACHE_DIR" RELEASE_SHA="$RELEASE_SHA" .venv/bin/python -c '
      import os
      import loudkit as lk

      repo, sha = "loudreader/loudr-1", os.environ["RELEASE_SHA"]
      engine = lk.load(repo, revision=sha)
      voice = lk.voice("joe", repo=repo, revision=sha)
      result = engine.synthesize("The split release works.", voice, seed=1234)
      result.save(os.path.join(os.environ["HF_HOME"], "smoke.wav"))
      print("OK", round(result.duration, 2), "seconds")
      '
      ```

## 6. Tag and let CI build

Push one tag now, and only one:

```bash
git tag v0.1.0 public-main
git push public v0.1.0
```

The explicit `public-main` target is a security boundary, not decoration. The
maintainer checkout may be on a private development branch whose history must
never become reachable from the public repository. Before pushing, both of
these must print the same commit:

```bash
git rev-parse v0.1.0^{}
git ls-remote public refs/heads/main | cut -f1
```

**Do not create or push `go/v0.1.0` yet.** That tag is the Go release, and it
is irrevocable in a way `v0.1.0` is not: once `proxy.golang.org` has fetched
it, that version is cached immutably and forever. Retagging does not change
what the proxy serves, and there is no yank. A `go/` tag pushed before the
gate has run publishes to Go whatever the gate later refuses; the only remedy
is `go/v0.1.1`. So the Go tag comes last, in §7.4, after the workflow is
green, the reviewer has approved the `release` environment, and every publish
job has finished.

`release.yml` triggers on `v*` only, so the later `go/v0.1.0` push starts no
second build. That tag exists purely so the module proxy can find the
subdirectory module.

**First release only:** when the run reaches the `release` environment wall,
complete the interactive npm bootstrap in §7.1 before approving it. npm cannot
stage or configure trust for a package that does not exist, and its dual-use
policy requires 2FA for the first direct publish. After approval, npm and PyPI
publish, then crates.io stops safely and asks for its own first-package
bootstrap in §7.3. No long-lived registry token is stored in GitHub.

Wait for `release.yml` to go green. It does the following:

- refuses the tag unless its commit is on `main` **and** that exact SHA has a
  green `ci` run whose `parity` job succeeded (the `tag-gate` job);
- checks the tag against `pyproject.toml`;
- builds sdist and wheel;
- verifies the lexicon and `py.typed` are inside the wheel;
- smoke-tests **both** artefacts in clean venvs;
- runs the no-weights suite;
- generates an SBOM scoped to the wheel, not the runner;
- attests the Python distributions and npm tarball in isolated jobs that run
  no fetched tooling, so jobs that mint OIDC identities never execute the
  build's downloaded dependencies;
- **creates the GitHub Release for the tag and uploads `dist/*` to it**,
  including the SBOM, with auto-generated notes.

The Release is made by CI, not by hand. What is left for a person is to replace
the auto-generated notes with the `CHANGELOG.md` section for this version.

## 7. Publish, in this order

`release.yml` does this on a version tag, and nothing is published until every
registry has agreed that it can be:

```
tag-gate → build → attest            ┐
                 → check-pypi        ├→ [approve `release`] →
                 → check-npm → attest-npm
                 → check-crates      ┘   publish-npm → publish-pypi →
                                         publish-crates → github-release
```

The three checks run each ecosystem's acceptance gate against the exact bytes
the publish jobs will send: `twine check --strict` on the built distributions;
`npm pack` followed by installing the packed tarball into a clean directory and
importing it; `cargo package` followed by building and testing the unpacked
`.crate`, plus a fresh consumer crate that depends on it and calls the API with
no lockfile and no monorepo paths. They gate the first publish, so a packaging
fault in one ecosystem surfaces while the other two are still untouched. That
ordering is the point: a version number spent on crates.io is never freed, and
a filename spent on PyPI is never reused, so a half-published release cannot be
repaired, only worked around.

The GitHub Release is created last, after the publishes, rather than before the
reviewer sees the request.

The manual commands below stay as the fallback and as the record of what the
jobs do.

### 7.0 Registry setup, once, by a human

None of this can be done from CI, and until it is done the publish jobs fail
at the authentication step rather than publishing something wrong.

- [ ] **PyPI** — add a Trusted Publisher at
      <https://pypi.org/manage/account/publishing/>. Create a pending publisher
      with project name `loudkit`, owner `loudreader`, repository `loudkit`,
      workflow `release.yml` and environment `release`. PyPI turns it into a
      normal publisher on first use; no token is stored. A pending publisher
      does not reserve the name, so do this immediately before tagging.
- [ ] **npm** — no repository secret. `loudkit` declares voice enrollment as
      dual-use, so npm requires proof of presence: direct automated publishing
      is forbidden. The first tarball is published interactively with 2FA from
      the exact attested Actions artifact. Once the package exists, configure
      its trusted publisher for **`npm stage publish` only**: owner
      `loudreader`, repository `loudkit`, workflow `release.yml`, environment
      `release`. Also set Publishing access to "Require two-factor
      authentication and disallow tokens". Later workflows stage over OIDC and
      wait while a maintainer reviews and approves the stage with 2FA. §7.1 is
      the exact sequence. These requirements come from npm's
      [Dual-Use Content Policy](https://docs.npmjs.com/policies/dual-use/) and
      [staged publishing](https://docs.npmjs.com/staged-publishing/) contract.
- [ ] **crates.io** — no repository secret. The first crate must exist before
      crates.io lets it trust a workflow, so §7.3 uses a short-lived token with
      only the `publish-new` endpoint scope, logs out and revokes it. Then add
      a GitHub Actions Trusted Publisher for owner `loudreader`, repository
      `loudkit`, workflow `release.yml` and environment `release`, and require
      Trusted Publishing for later versions. The workflow obtains a temporary
      token through `rust-lang/crates-io-auth-action` and that action revokes it
      when the job ends.
- [ ] **GitHub** — create the `release` environment under Settings →
      Environments and add at least one required reviewer. Without the
      reviewer the wall is not there, and a mistaken tag publishes to three
      registries with no human in between.

Upload **the artefacts CI built and attested**, never a local rebuild. The
provenance attestation is for those exact bytes. Fetch them first:

```bash
gh run download --repo loudreader/loudkit --name <artifact> --dir dist/
```

The order below runs from most reversible to least, and each step is gated on
the previous one installing cleanly from the public registry. A packaging
defect that surfaces on npm costs an unpublish. The same defect found after
crates.io costs the version number in every ecosystem.

### 7.1 npm: the first-release bootstrap

npm has no `loudkit` package yet, so there is nothing for a trusted publisher
to be attached to, and npm also refuses to stage a brand-new package. LoudKit's
voice enrollment is declared as dual-use, so its first direct publish must be
interactive and protected by 2FA. A CI token that bypasses 2FA and a direct
OIDC publish are both disallowed by npm's policy.

The unavoidable bootstrap exception is native npm provenance: npm can only
mint it from CI, while CI cannot create this first dual-use package. The exact
tarball is still covered by the separate GitHub build attestation created by
`attest-npm`. Every later version is staged through npm OIDC and receives npm's
own provenance as well.

The first-release sequence is:

1. Push the tag (§6) and wait until `attest`, `attest-npm` and all three
   `check-*` jobs are green. `publish-npm` then waits at the GitHub `release`
   environment. Do not approve it yet.
2. Download the exact npm artifact from that run and verify its attestation.
   Use a fresh directory so no local `npm pack` output can be selected:

   ```bash
   RUN_ID=PASTE_RELEASE_RUN_ID
   NPM_OUT=$(mktemp -d)
   gh run download "$RUN_ID" --repo loudreader/loudkit \
     --name npm-tarball --dir "$NPM_OUT"
   gh attestation verify "$NPM_OUT"/*.tgz --repo loudreader/loudkit
   shasum -a 256 "$NPM_OUT"/*.tgz
   tar -tzf "$NPM_OUT"/*.tgz
   ```

   The listing must include `DISCLOSURE`, `LICENSE`, `NOTICE`, `dist/` and both
   files under `data/`. Never run `npm pack` here; that would create different,
   unattested bytes.
3. Log in to npm with the owner account and publish that tarball interactively.
   Complete the 2FA challenge npm presents:

   ```bash
   npm login --auth-type=web
   npm publish --access public "$NPM_OUT"/*.tgz
   ```

4. On the new `loudkit` package, add its GitHub Actions trusted publisher:
   repository `loudreader/loudkit`, workflow `release.yml`, environment
   `release`, and allow **`npm stage publish` only**. Then set Publishing access
   to "Require two-factor authentication and disallow tokens".
5. Approve the GitHub `release` environment. `publish-npm` compares npm's live
   `dist.shasum` with the checked tarball and only then unblocks PyPI and
   crates.io. A mismatch stops the chain.
6. In a scratch directory, install `loudkit@0.1.0` and import it. Also repeat
   `gh attestation verify` on the downloaded tarball as the provenance gate for
   this bootstrap version.

From the second release on, `publish-npm` stages the checked tarball through
the stage-only trusted publisher and waits for up to 30 minutes. Open npm's
Staged Packages page, inspect the version, approve it with 2FA, and leave the
workflow running. Only after the live registry shasum matches does PyPI start.
Those later versions carry both GitHub's tarball attestation and npm's native
OIDC provenance.

Re-running a tag whose publish already succeeded does not publish twice.
`publish-npm` finds the version on the registry, compares the registry's
`dist.shasum` against the tarball this run packed, and passes only if they are
the same bytes. A mismatch fails the job on purpose: it means the registry
holds bytes this workflow never checked.

Rollback: `npm unpublish loudkit@0.1.0` works for **72 hours** and only while
nothing depends on it. After that the version is permanent and `npm deprecate`
is the whole remedy.

Gate: in a scratch dir, `npm install loudkit@0.1.0` and import it. Do not
continue until this passes.

### 7.2 PyPI

```bash
twine check dist/loudkit-0.1.0*
twine upload dist/loudkit-0.1.0*.whl dist/loudkit-0.1.0*.tar.gz
```

Rollback: none worth the name. Deleting the release removes the files, so
nobody installs a broken artefact, but **the version number is burned**. PyPI
never allows `0.1.0` to be uploaded again, under any content. A mistake here
costs `0.1.1`.

Gate: `pip install loudkit==0.1.0` in a fresh venv, run the README block.

### 7.3 crates.io

crates.io also cannot attach a Trusted Publisher before the package exists.
Unlike npm, it has no staged first publish, so the workflow deliberately stops
here after npm and PyPI are live. The failed job prints the same procedure.

1. Create an API token at <https://crates.io/settings/tokens/new> with a short
   expiry and only the `publish-new` endpoint scope. Do not add it to GitHub.
2. Check out the exact public tag in a clean worktree. Never publish from the
   private development branch:

   ```bash
   REPO=$(git rev-parse --show-toplevel)
   CRATE_TREE=$(mktemp -d)
   git -C "$REPO" worktree add --detach "$CRATE_TREE" v0.1.0
   cd "$CRATE_TREE/rust"
   cargo login
   cargo publish --locked
   cargo logout
   ```

   Paste the `publish-new` token only into `cargo login`'s prompt. It does not
   enter shell history. Whether the publish succeeds or fails, run
   `cargo logout` and revoke the token on crates.io immediately afterwards.
3. On the new `loudkit` crate, configure GitHub Actions Trusted Publishing:
   owner `loudreader`, repository `loudkit`, workflow `release.yml`,
   environment `release`. Enable the setting that requires Trusted Publishing
   for new versions.
4. Rerun the failed `publish-crates` job. It compares crates.io's immutable
   checksum with the `.crate` artifact that passed both consumer gates before
   anything was published, and creates the GitHub Release only if those exact
   bytes match:

   ```bash
   gh run rerun PASTE_RELEASE_RUN_ID --repo loudreader/loudkit --failed
   ```

From `0.1.1` onward, the job obtains a short-lived crates.io token through
OIDC and runs `cargo publish --locked` itself. No manual token is involved.

Rollback: none. `cargo yank --version 0.1.0` stops **new** dependents from
selecting it; it does not delete anything, existing lockfiles keep resolving
it, and the version can never be reused or replaced. crates.io is last because
it is the only registry where a bad upload cannot be withdrawn at all.

### 7.4 Go and Swift

Nothing to upload, but Go has one act left: the tag §6 deliberately held back.
Push it only now, with the workflow green, the approval given and the three
registries published:

```bash
git tag go/v0.1.0 public-main
git push public go/v0.1.0
```

The pushed tags are the release. First `go get` against a fresh module can
take a few minutes while `proxy.golang.org` indexes it. Swift needs no tag of
its own: SwiftPM resolves `v0.1.0`.

## 8. Post-publish smoke, from the public internet only

Run these on a machine that has never seen this repository.

The Python half is scripted, so it is a gate rather than a memory of having
tried it:

```bash
python tools/acceptance.py --from-pypi --extras torch,audio,hub --speak
```

It builds a venv outside any checkout, installs the published distribution,
and **refuses to continue unless `loudkit` imported from that venv**. That
refusal is the point: run the same venv's interpreter with the working
directory inside `python/` and it imports the checkout instead, so without the
check a wheel missing half its data files passes every step. `--wheel <path>`
runs the same gate against a locally built wheel before anything is published.

- [ ] `python tools/acceptance.py --from-pypi --extras torch,audio,hub --speak`
- [ ] `pip install loudkit` in a fresh venv; run the README block.
- [ ] `npm install loudkit` in a scratch dir; import it.
- [ ] `cargo add loudkit` in a scratch crate; build.
- [ ] `go get github.com/loudreader/loudkit/go@latest` resolves via
      `proxy.golang.org`.
- [ ] `swift package resolve` picks up `0.1.0` from the repository URL.
- [ ] The Colab badge, clicked from the public README, runs end to end.
- [ ] The docs site at `https://loudreader.github.io/loudkit/` has deployed
      from `main` and carries no pre-release banner.

Then run each guide as a stranger would, from the published packages rather
than from a checkout. A guide that was true against this working copy can
still be false against a registry: the wheel ships a subset of the tree, the
npm tarball another, and the crate a third.

- [ ] Guide 1: install, `loudkit download`, the two-line synthesis, and the
      five one-liners in English, Spanish, French, German and Italian.
- [ ] Guide 2: `stream` and `synthesize_long`, and `previous_tokens` across
      two calls.
- [ ] Guide 3: `lk.enroll` from a ten-second recording, then synthesise with
      the profile it wrote.
- [ ] Guide 4: `loudkit serve`, one `/v1/synthesize` call, one stream, the
      OpenAI route, `loudkit mcp` and `loudkit grpc`.
- [ ] Guide 5: `loudkit bench` produces a row on this machine.
- [ ] Guides 7 to 10: the TypeScript, Go, Rust and Swift quickstarts from
      scratch consumers, using the packed local artefacts described in
      §§4.3–4.6 against exported graphs. The registry versions do not exist on
      a first release yet; §8 repeats the same pass from the public registries.

Record what broke. A guide that needs a fix is a patch release, not an
edit to the tag.

## 9. Pre-release wording, and when each piece comes off

Two rounds, because two things publish at different times. The weights went to
the Hub first, so the wording that said *they* had not is already gone. What is
left says the packages are not on the registries yet.

All of it comes off in the **release commit (§2)**, not at §7 where the
packages actually publish: `tests/test_release_coherence.py` refuses a stable
version while any of these stands, and the tag is cut in §6. So the wording
goes one section before the gate that would otherwise stop the release.

This table and `PRERELEASE_BANNERS` in that test are one list written twice,
and a test asserts they agree. Grep for the marker rather than trusting a line
number: this table has rotted twice, once by quoting strings that had already
been replaced, and once by a rewrite that dropped two rows and asserted in
prose that those files were clean while the gate was still watching them.

| file | grep for | what it is |
| --- | --- | --- |
| `README.md` | `packages are not yet` | the README pre-release note |
| `notebooks/loudkit_quickstart.ipynb` | `Pre-release.` | the Colab pre-release note |
| `docs/reference/troubleshooting.md` | `lands on PyPI with the 0.1.0 release` | the not-on-PyPI-yet paragraph |
| `site/scripts/sync-docs.mjs` | `banner:` | the site-wide banner written into every generated page. Already absent; the row stays so it cannot come back unnoticed |
| `site/src/handwritten/index.mdx` | `lk-banner` | the landing-page banner block |
| `site/src/handwritten/demo.mdx` | `banner:` | the demo page banner front matter |
| `site/src/handwritten/index.mdx` | `git = "https://github.com/loudreader/loudkit"` | the landing page's Rust tab. Becomes `loudkit = "0.1"` |
| `site/src/handwritten/index.mdx` | `branch: "main"` | the landing page's Swift tab. Becomes `from: "0.1.0"` |
| `docs/guides/10-swift.md` | `branch: "main"` | guide 10's Swift dependency. Becomes `from: "0.1.0"`, which the sentence under it already tells the reader to use |

The last three are not banners, and that is why they went unnoticed: an install
line naming a branch or a git URL is a pre-release instruction wearing ordinary
syntax. A reader who copies one after 0.1.0 ships builds from `main` — the
moving target the release exists to replace.

`site/src/content/docs/` is generated and gitignored. Do not edit the copies
there; they are rewritten from `docs/` and `site/src/handwritten/` on every
build.

The README sentence "the teacher's training data is Resemble AI's and is not
published" is a permanent statement about the upstream teacher, not a
pre-release note. It stays.
