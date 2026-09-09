# Releasing loudkit

The release workflow (`.github/workflows/release.yml`) builds and verifies the
artefacts, attests their provenance, coordinates the first npm and crates.io
bootstraps and, after a reviewer approves the `release` environment, publishes
PyPI before creating the GitHub Release. The manual commands in §7 are the two
first-package bootstraps, the fallback and the record of what the jobs do. This
file is the whole procedure, in order.

This procedure updates an existing public release. Preserve public history and
existing registry packages; the one-time repository bootstrap is finished.

## 0. Preconditions

- Verify that the existing `public` remote is `loudreader/loudkit` and fetch it.
- Preserve its current `main` history. Never create another parentless root,
  overwrite published tags, or force-push a release update.
- Verify ownership and Trusted Publishing for the existing PyPI, npm and
  crates.io packages. The names are already registered by this project.
- Publish both verified model bundles and create their HF `v0.1.1` revisions
  before the code tag. Record both immutable HF commit IDs in CI.
- Obtain the release owner's listening and provenance acceptance (§4).

## 1. Version sync

Four files carry a version and must agree before tagging. Go and Swift take
their version from the git tag and have nothing to edit.

| file | field | pre-release value | release value |
| --- | --- | --- | --- |
| `pyproject.toml` | `version` | `0.1.1` | `0.1.1` |
| `js/package.json` | `version` | `0.1.1` | `0.1.1` |
| `rust/Cargo.toml` | `version` | `0.1.1` | `0.1.1` |
| `python/loudkit/_version.py` | `__version__` | `0.1.1` | `0.1.1` |

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
- [ ] Confirm that the `0.1.1` section in `CHANGELOG.md` carries the release date.
- [ ] The two audio players go into the "Listen" section of
      `docs/MODEL_CARD.md`. They point at `samples/` in the Hugging Face model
      repository; the strict builder copies those bytes from the voice roster
      and includes them in both manifests. This is the only place the card can
      be written, because §5.5 freezes the tree and the card is a hashed member
      of the bundle §5.1 builds. §5.1 confirms they answer.

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
- [ ] The Hub commits of both bundles pinned in `.github/workflows/ci.yml`
      (`LOUDKIT_HF_REVISION`, `LOUDKIT_TURBO_HF_REVISION`). They exist only
      after §5.1 and §5.2, so the release commit is cut after the uploads;
      §5.3 is that commit.
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

`cargo fmt --check` and `eslint` are not in that list only because
`cargo clippy` and `npm test` do not cover them; CI runs both, so run them too:

```bash
(cd rust && cargo fmt --check)
(cd js && npm run lint)
```

Every one of these is a CI gate. Run them locally with the pinned versions: a
release branch that reaches this page red has already spent a review cycle on
formatting.

**The list above is not all of CI.** Five
more jobs gate a merge, and each has caught something no test above would:

```bash
# The funnels fuzzed against each other. Gating in CI, and the cheapest of
# the five to run: no weights, no network.
python tools/fuzz_parity.py --cases 300 --seed 1 --ports go,rust,js
python tools/fuzz_parity.py --cases 300 --seed 2 --ports go,rust,js

# The crate builds at its declared MSRV, and with the coreml feature on.
(cd rust && cargo build --locked)
(cd rust && cargo clippy --all-targets --features coreml -- -D warnings && \
            cargo test --features coreml)   # macOS only

# The wheel imports from a directory that is not the repo, and the sdist
# carries only what the allowlist names.
python -m build --sdist --wheel --outdir /tmp/lk-dist
```

The MSRV job needs a 1.88 toolchain and the coreml job needs macOS, so on the
wrong host they are CI's to run rather than yours, but know which ones you are
leaving to CI. `dependency audit` (pip-audit, `npm audit`, `govulncheck`,
`cargo audit`) and `parity` (hosted macOS against the pinned public release)
are CI's by design: the first wants a clean resolver environment, the second
downloads the release.

## 3.5 Re-measure the speed table if the engine got faster

`docs/benchmarks.md` is the one page the published real-time factors are
measured on; the README, the model card and the website quote it, and
`tests/test_docs.py` holds them to it. What the test cannot know is whether the
page is *current*, because the two machines that hold the record, an RTX 3090
and a Jetson Orin Nano, are not in CI.

So: if this release changed the speed of the engine, take the table again
before tagging. Both boxes, `--cuda-graphs`, the same voice and seed the page
declares:

```bash
python tools/bench.py --checkpoint <checkpoint> --voice <voice> \
  --device cuda --cuda-graphs --seed 1234 --json out/bench.json
```

Then update `docs/benchmarks.md`, re-quote the four figures the test checks,
and move the `measured on 0.1.0` markers to this release. If the numbers are
*not* re-taken, leave those markers where they are: a stale figure that says
which release it is remains true, and one that does not is a claim about today.

Historical measurements are retained this way. On 0.1.1 every user-facing figure was
re-measured on 2026-09-06 for both models; the one table still marked 0.1.0 is the
ONNX-provider comparison on the RTX 3090 in `docs/benchmarks.md`.

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
- [ ] **Listening.** Hear all 28 roster samples from start to finish, then
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
      whl=$(ls dist/loudkit-0.1.1-*.whl)
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
      and refuses a tarball missing `dist/`, any of the three data files,
      `LICENSE`, `NOTICE` or `DISCLOSURE`. Confirm the listing carries those
      files, `README.md`, `data/numbers.json`, `data/numerals.json`,
      `data/pl_en_respell.json` (about 6.6 MB) and `dist/`, and that no
      `dist/test/` entry appears. Confirm `package.json` declares
      `contentPolicy.class` as `dual-use`.

      `js/data/` is gitignored and copied at build time, so a file the copier
      does not name is missing from a fresh clone and present on every machine
      that has ever built the package. That is how `numerals.json` shipped in
      nobody's tarball while the suite was green; `js/src/test/pack.test.ts`
      packs, unpacks and folds a numeral out of the result.
- [ ] In a scratch dir: `npm install /path/to/loudkit-0.1.1.tgz`, then run the
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
      go get github.com/loudreader/loudkit/go@v0.1.1
      ```

      **The version query is `@v0.1.1`, not `@go/v0.1.1`.** The `/go` suffix on
      the module path is what tells the proxy the tag it wants is prefixed
      `go/`. Writing the tag name into the query fails to resolve.
      Paste the example from `go/README.md`, then `go run .`.

### 4.6 The Swift user

- [ ] Pre-push stand-in: `swift build && swift test` at the repo root.
- [ ] Post-push, a scratch package depending on
      `.package(url: "https://github.com/loudreader/loudkit", from: "0.1.1")`,
      built and run against the example from the Swift section of the README on
      macOS. SwiftPM strips the leading `v` from tags, so `from: "0.1.1"`
      resolves the `v0.1.1` tag. No separate Swift tag is needed:
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

## 5. Bundles, pins, then public main

The order matters: the two bundles are uploaded first (§5.1, §5.2), their Hub
commits are pinned in CI (§5.3), and only then is the tree pushed to public
`main` (§5.5), so the commit that carries the tag can pass its own parity job.

### 5.1 Build the loudr-1 bundle and upload it

Two things have to be true before the build reads anything.

**The checkpoint is split.** A release ships two halves, and the builder looks
for the enrollment one *beside* the synthesis one under its canonical name --
there is no flag for it. Both have to sit in one directory, which is not the
checkpoint's own: the tool refuses that, because a half written beside the
packed original is a different artefact wearing the same directory.

```bash
.venv/bin/python tools/split_checkpoint.py \
    --checkpoint assets/loudr-1.safetensors \
    --out-dir    dist/release-0.1.1/split/loudr-1
```

**Only if the pair is not there yet.** With both halves present the tool says
so and stops, which is correct and needs no answer -- do not reach for a force
flag. The pair does not need re-cutting to be trusted: each half carries the
digest of the packed original, and the build refuses two halves that came from
different runs.

**Then stamp the fade into the synthesis half, and nothing else.** The packed
original predates the 20 ms ramp, so the halves cut from it carry no
`edge_fade_seconds`. The amender writes only that key when told to, and it has
to be told: its full list also rewrites `chunking` over the shorter block the
manifest carries, and the tool refuses to do that over a manifest that already
says something. The loader fills the same defaults either way, so the
fingerprint is the same; what `--only` keeps still is the file digest.

```bash
.venv/bin/python tools/amend_manifest.py \
    --checkpoint dist/release-0.1.1/split/loudr-1/loudr-1.safetensors \
    --only edge_fade_seconds
shasum -a 256 dist/release-0.1.1/split/loudr-1/loudr-1.safetensors
```

The digest is `73e69a78d58176a3…`: the one `docs/voices/roster/provenance.json`
records for every sample and the one the export records bind to. A different
digest here is a different input, not a different day. The same step on the
turbo half gives `590dcf9e1dcec31c…` (§5.2).

**Export from the split files, after splitting.** Copy `tokenizer.json` beside
`dist/release-0.1.1/split/loudr-1/loudr-1.safetensors`, then run the synthesis
exporters with that synthesis half as `--checkpoint`. Export enrollment from
`dist/release-0.1.1/split/loudr-1/loudr-1-enrollment.safetensors` and the shared
`ve.safetensors`. Every path below is under `dist/release-0.1.1/`: `split/<model>`
for the halves, `exports/<model>/{onnx,coreml}` for graphs the exporters wrote
(`assets/onnx` and `assets/coreml` hold loudr-1's in this tree), `samples/<model>`
for the two rendered players, and `<model>` for the bundle itself.

```bash
S=dist/release-0.1.1/split/loudr-1; E=dist/release-0.1.1/exports/loudr-1
.venv/bin/python tools/export_onnx.py          --checkpoint $S/loudr-1.safetensors --out $E/onnx
.venv/bin/python tools/export_enroll_onnx.py   --checkpoint $S/loudr-1-enrollment.safetensors --voice-encoder assets/ve.safetensors --out $E/onnx
.venv/bin/python tools/export_coreml.py        --checkpoint $S/loudr-1.safetensors --out $E/coreml
.venv/bin/python tools/export_enroll_coreml.py --checkpoint $S/loudr-1-enrollment.safetensors --voice-encoder assets/ve.safetensors --out $E/coreml
```

Every exporter gates its graphs against torch before it writes, and the two
synthesis records name `73e69a78…` and `7cd75498ad4e7531`. Run them from the
checkout that holds `assets/`; a worktree has only the tracked logos there.
Do not reuse synthesis graphs exported from the original unsplit file, which is
what the tree's `assets/onnx` and `assets/coreml` are: their checkpoint digest
is different, and the runtime refuses the mixed set.
Keep each model's output directories separate. Run CoreML export and inference
with access to the macOS compute services; a sandboxed execution can select a
different compute path and does not reproduce the published measurements.

**The model card is already final**, from §2. The bundle is built before the
public push (§5.5) and the tag (§6), so a card that changed after this point
would ship to the Hub without existing in `main`. The one tracked file the
rest of §5 edits is `.github/workflows/ci.yml`, in the pin commit (§5.3).
Read the card, do not write it.

- [ ] The two audio players §2 added name the model repository's own
      `samples/joe.opus` and `samples/kathleen.opus`, not GitHub raw content.
      They cannot answer until the new bundle is merged; the post-merge check
      below is the listening gate.
- [ ] The card's claims still match the artefacts: voice count, language list,
      artefact sizes, the download table.
- [ ] Every link in the card resolves; the docs site is live, and §5.6
      re-checks it after the push.

Then build the bundle with the strict profile. It is the default, and it is the
only profile a release is cut from:

The inputs are settled, so the command is the command rather than a shape to
fill in. `--checkpoint` names the *synthesis half*, and the builder finds the
enrollment one beside it:

```bash
.venv/bin/python -m tools.release --model loudr-1 \
    --checkpoint    dist/release-0.1.1/split/loudr-1/loudr-1.safetensors \
    --voice-encoder assets/ve.safetensors \
    --voices        assets/voices \
    --onnx          dist/release-0.1.1/exports/loudr-1/onnx \
    --coreml        dist/release-0.1.1/exports/loudr-1/coreml \
    --out           dist/release-0.1.1/loudr-1 \
    --profile       full-0.1
```

`--profile full-0.1` requires the checkpoint under its canonical name
`loudr-1.safetensors`, its manifest, the tokenizer, `ve.safetensors`, all 28
voices of the roster by name, the complete ONNX and CoreML synthesis and
enrollment graph families, the four
documents, the LoudKit wordmark, the two listening samples, `release.json`,
and a `SHA256SUMS` covering every file. The roster
is `docs/voices/roster/provenance.json`: 28 voices: ten English and two per other language
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
loads all 28 voices, clones a voice with the shipped `ve.safetensors`, speaks
on the ONNX path, speaks on the CoreML path, and runs one enrollment through
the three enrollment graphs on both graph paths, against the enrollment
fixture.

**Cut the release on macOS.** The shipped CoreML artefacts are platform-specific
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
.venv/bin/python -m tools.release --verify-only dist/release-0.1.1/loudr-1
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

- [ ] `.venv/bin/python -m tools.release --verify-only dist/release-0.1.1/loudr-1`
      passes. Run it immediately before the upload; it catches anything that
      touched the bundle after the build.
- [ ] `shasum -a 256 -c SHA256SUMS` passes from the release root, and
      `SHA256SUMS` has **one fewer line than the bundle has files**: the count
      difference below is exactly one. `SHA256SUMS` cannot contain its own
      digest; every other file, including `release.json`, every graph, package
      leaf and sample, has a line.

      ```bash
      (
        cd dist/release-0.1.1/loudr-1
        shasum -a 256 -c SHA256SUMS
        echo $(( $(find . -type f | wc -l) - $(wc -l < SHA256SUMS) ))   # 1
      )
      ```

- [ ] `release.json` says `"profile": "full-0.1"` and `"verified": true`.
      A lenient or unverified bundle is not a release.
- [ ] `diff docs/MODEL_CARD.md dist/release-0.1.1/loudr-1/README.md` is empty. The card is a
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
  cd dist/release-0.1.1/loudr-1
  hf upload loudreader/loudr-1 . . \
      --repo-type model \
      --create-pr \
      --delete "*" \
      --commit-message "loudr-1: complete v0.1 bundle"
)
```

- [ ] Review the Hub PR before merging. Its final tree, not only its additions,
      matches `dist/release-0.1.1/loudr-1`; the old packed checkpoint is deleted, both split
      halves are present, and `samples/` contains exactly the two players.
- [ ] Merge it in the Hugging Face interface, then record the immutable commit
      SHA. The default branch name is not a release identifier:

      ```bash
      SHA=$(curl -s https://huggingface.co/api/models/loudreader/loudr-1 \
        | python3 -c "import json,sys; print(json.load(sys.stdin)['sha'])")
      echo "$SHA"
      ```

- [ ] Tag that commit on the Hub as `v0.1.1`. The guides and the port READMEs
      pin `revision: "v0.1.1"`, and the Hub has no tags until this is run:

      ```bash
      .venv/bin/hf repos tag create loudreader/loudr-1 v0.1.1 --revision "$SHA"
      HF_HOME=$(mktemp -d) .venv/bin/python -c '
      import loudkit as lk
      print(lk.load("loudreader/loudr-1", revision="v0.1.1").describe())
      '
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
      (six synthesis, three enrollment), the six CoreML packages, the 28
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

### 5.2 The second model, `loudreader/loudr-1-turbo`

Turbo is its own repository, published before or after `loudr-1` without
touching it. It is not optional: the README, the guides and the site name
`loudreader/loudr-1-turbo` as a repo id anyone can load. Publish it before §6,
or cut those mentions.

Both models ship ONNX and CoreML graphs and the same portable voice profiles.
Split the fusion checkpoint with `tools/split_checkpoint.py`. Its bundle
carries the synthesis half and its matching enrollment half, `ve.safetensors`, and the same enrollment
graphs as loudr-1. A caller clones once and can use that profile with either
model. Downloads for synthesis alone do not need the enrollment assets.

Export synthesis from the verified built checkpoint
`dist/release-0.1.1/split/turbo/loudr-1-turbo.safetensors`, into
`dist/release-0.1.1/exports/turbo/onnx` and `coreml`. Copy the canonical
base enrollment graphs into those directories. Do not rewrite the synthesis
export records or substitute base renderer graphs: the decoder and the Euler
step count come from this checkpoint. Render `joe.opus` and `kathleen.opus`
with this model into `dist/release-0.1.1/samples/turbo`; copying the base samples
would mislabel the audio.

```bash
S=dist/release-0.1.1/split/turbo; E=dist/release-0.1.1/exports/turbo; B=dist/release-0.1.1/exports/loudr-1
cp assets/tokenizer.json $S/tokenizer.json
.venv/bin/python tools/amend_manifest.py --checkpoint $S/loudr-1-turbo.safetensors --only edge_fade_seconds
shasum -a 256 $S/loudr-1-turbo.safetensors            # 590dcf9e1dcec31c…
.venv/bin/python tools/export_onnx.py   --checkpoint $S/loudr-1-turbo.safetensors --out $E/onnx
.venv/bin/python tools/export_coreml.py --checkpoint $S/loudr-1-turbo.safetensors --out $E/coreml
for g in s3_tokenizer camp voice_encoder; do cp $B/onnx/$g.onnx $E/onnx/; cp -R $B/coreml/$g.mlpackage $E/coreml/; done
```

The synthesis records name `590dcf9e…` and `e5303ba243087222`, one Euler step,
the `fusion_mtp2` family. The samples are the roster's two texts at seed 7,
spoken on the CPU torch path and encoded the way the roster files are:

```bash
.venv/bin/python - <<'PY'
import json, subprocess, loudkit as lk
from pathlib import Path
R = Path("dist/release-0.1.1"); out = R / "samples/turbo"; out.mkdir(parents=True, exist_ok=True)
engine = lk.load(str(R / "split/turbo/loudr-1-turbo.safetensors"), device="cpu")
roster = {v["name"]: v["sample"] for v in json.load(open("docs/voices/roster/provenance.json"))}
for name in ("joe", "kathleen"):
    s = roster[name]
    wav, opus = out / f"{name}.wav", out / f"{name}.opus"
    engine.synthesize(s["text"], lk.voice(f"assets/voices/{name}.safetensors"), seed=s["seed"]).save(str(wav))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-c:a", "libopus", "-b:a", "64k", "-vbr", "on", str(opus)], check=True)
    wav.unlink()
PY
```

```bash
.venv/bin/python -m tools.release --model turbo \
    --checkpoint    dist/release-0.1.1/split/turbo/loudr-1-turbo.safetensors \
    --tokenizer     dist/release-0.1.1/split/turbo/tokenizer.json \
    --voice-encoder dist/release-0.1.1/loudr-1/ve.safetensors \
    --enrollment    dist/release-0.1.1/split/turbo/loudr-1-enrollment.safetensors \
    --voices        dist/release-0.1.1/loudr-1/voices \
    --onnx          dist/release-0.1.1/exports/turbo/onnx \
    --coreml        dist/release-0.1.1/exports/turbo/coreml \
    --samples       dist/release-0.1.1/samples/turbo \
    --out           dist/release-0.1.1/loudr-1-turbo
```

The builder refuses missing graphs, enrollment or samples before copying.
It then speaks on torch, ONNX and CoreML, verifies enrollment and the roster,
and checks that the checkpoint declares the paired decoder. Only then does
it stamp `profile: turbo-0.1, verified: true`. `--skip-verify` remains refused.

- [ ] `docs/MODEL_CARD-turbo.md` is final and its links resolve. It ships as
      the repository's `README.md`, and like §2's card it is read here, never
      written.
- [ ] The bundle passes its own audit in place:
      `.venv/bin/python -m tools.release --verify-only dist/release-0.1.1/loudr-1-turbo`
- [ ] Every port passes its asset-backed conformance against the two built
      bundles. The vectors were made with the reference voice, so
      `LOUDKIT_VOICE` is `tests/data/reference/testvoice.voice.safetensors`,
      never a shipped one; the ONNX Runtime library each port wants is in its
      guide.

      ```bash
      D=$PWD/dist/release-0.1.1; V=$PWD/tests/data/reference/testvoice.voice.safetensors
      LOUDKIT_REQUIRE_ASSETS=1 LOUDKIT_ASSET_ROOT=$D/loudr-1 LOUDKIT_TURBO_CHECKPOINT=$D/loudr-1-turbo/loudr-1-turbo.safetensors \
        .venv/bin/python -m pytest tests/test_conformance.py -q
      (cd go && LOUDKIT_REQUIRE_ASSETS=1 LOUDKIT_CKPT=$D/loudr-1/loudr-1.safetensors LOUDKIT_ONNX_DIR=$D/loudr-1/onnx \
        LOUDKIT_TOKENIZER=$D/loudr-1/tokenizer.json LOUDKIT_VOICE=$V \
        LOUDKIT_SECOND_CKPT=$D/loudr-1-turbo/loudr-1-turbo.safetensors LOUDKIT_SECOND_ONNX_DIR=$D/loudr-1-turbo/onnx \
        LOUDKIT_ONNXRUNTIME_LIB=/path/to/libonnxruntime.dylib go test ./conformance/)
      (cd rust && LOUDKIT_REQUIRE_ASSETS=1 LOUDKIT_CKPT=$D/loudr-1/loudr-1.safetensors LOUDKIT_ONNX_DIR=$D/loudr-1/onnx \
        LOUDKIT_TOKENIZER=$D/loudr-1/tokenizer.json LOUDKIT_VOICE=$V \
        LOUDKIT_FUSION_CKPT=$D/loudr-1-turbo/loudr-1-turbo.safetensors LOUDKIT_FUSION_ONNX_DIR=$D/loudr-1-turbo/onnx \
        ORT_DYLIB_PATH=/path/to/libonnxruntime.1.27.0.dylib cargo test --release -- --ignored)
      (cd js && LOUDKIT_REQUIRE_ASSETS=1 LOUDKIT_CKPT=$D/loudr-1/loudr-1.safetensors LOUDKIT_ONNX_DIR=$D/loudr-1/onnx \
        LOUDKIT_TOKENIZER=$D/loudr-1/tokenizer.json LOUDKIT_VOICE=$V \
        LOUDKIT_SECOND_CKPT=$D/loudr-1-turbo/loudr-1-turbo.safetensors \
        LOUDKIT_SECOND_ONNX_DIR=$D/loudr-1-turbo/onnx npm test)
      LOUDKIT_REQUIRE_ASSETS=1 LOUDKIT_ASSET_ROOT=$D/loudr-1 LOUDKIT_COREML_ASSETS=$D/loudr-1/coreml \
        LOUDKIT_FUSION_CHECKPOINT=$D/loudr-1-turbo/loudr-1-turbo.safetensors LOUDKIT_FUSION_COREML_ASSETS=$D/loudr-1-turbo/coreml \
        swift test
      ```

The repository `loudreader/loudr-1-turbo` is public and ungated, which the
parity job needs because it downloads anonymously. Confirm that in its
settings, then upload the directory's contents at its root, exactly as §5.1
does for `loudr-1`: the same `hf upload ... --create-pr --delete "*"`, the
same review of the PR's final tree, the merge, the recorded commit SHA, and
the `v0.1.1` tag on that commit. Then, from a machine with no cache:

```bash
HF_HOME=$(mktemp -d) .venv/bin/python -c '
import loudkit as lk
engine = lk.load("loudreader/loudr-1-turbo")
voice = engine.voice("joe")
print("OK", round(engine.synthesize("Turbo works.", voice, seed=7).duration, 2))
'
```

- [ ] Nothing to switch on in the tests: `tests/assets.py` already registers
      the turbo assets, so `LOUDKIT_REQUIRE_ASSETS=1` insists on them. The
      `parity` job starts passing once the repository is public and §5.3 pins
      its commit.
- [x] `docs/benchmarks.md` carries 0.1.1 figures for both models on every
      machine, measured 2026-09-06, and the README, the landing page and the
      guides quote them. The ONNX-provider table on the RTX 3090 is the one
      table still marked 0.1.0.

### 5.3 The pin commit

The `parity` job downloads both repositories anonymously and exits before
its first step while `LOUDKIT_TURBO_HF_REVISION` is empty, and `release.yml`
refuses a tag without a green parity run for that commit. So the last commit
before the public push pins what §5.1 and §5.2 merged:

- [ ] `LOUDKIT_HF_REVISION` and `LOUDKIT_TURBO_HF_REVISION` in
      `.github/workflows/ci.yml` carry the two 40-hex commits recorded above,
      and the comment beside them describes those bundles.
- [ ] `PYTHONPATH=$PWD/python .venv/bin/python -m pytest tests/test_release.py -q`
      passes with the new pins.
- [ ] This is the release commit of §2, with every other §2 item already in
      it. Nothing after it edits a tracked file.

### 5.4 Confirm the tree has settled

- All changes committed, worktree clean, all §3 checks run on that commit.
- Merge the release work into the private main branch after review.
- Fetch `public/main` before constructing the public update.

### 5.5 Publish an update preserving public history

Create the next public commit with `public/main` as its parent and the reviewed
release tree as its contents. A private staging history must not accidentally
be published. Review the resulting tree and diff, then push without force.
The public branch must remain a descendant of the previous public release.
Wait for the complete public CI run, including both models, before tagging.

### 5.6 Verify before tagging

- [ ] The repository is public: the anonymous 200 from §0 still holds.
- [ ] The pushed tree matches local, exactly:
      `git fetch public && git diff --stat public/main public-main` prints
      nothing.
- [ ] Spot-check the rendered `README.md` and `VOICES.md` on GitHub. The
      pre-release banner is gone (§9) and no link 404s.
- [ ] The docs site has deployed from `main` and carries no banner.

Tagging is the next step and it is what the Go proxy caches. Do not tag until
all four boxes are ticked.

## 6. Tag and let CI build

Push one tag now, and only one:

```bash
git tag v0.1.1 public-main
git push public v0.1.1
```

The explicit `public-main` target is a security boundary, not decoration. The
maintainer checkout may be on a private development branch whose history must
never become reachable from the public repository. Before pushing, both of
these must print the same commit:

```bash
git rev-parse v0.1.1^{}
git ls-remote public refs/heads/main | cut -f1
```

**Do not create or push `go/v0.1.1` yet.** That tag is the Go release, and it
is irrevocable in a way `v0.1.1` is not: once `proxy.golang.org` has fetched
it, that version is cached immutably and forever. Retagging does not change
what the proxy serves, and there is no yank. A `go/` tag pushed before the
gate has run publishes to Go whatever the gate later refuses; the only remedy
is a new version such as `go/v0.1.2`. So the Go tag comes last, in §7.4, after the workflow is
green, the reviewer has approved the `release` environment, and every publish
job has finished.

`release.yml` triggers on `v*` only, so the later `go/v0.1.1` push starts no
second build. That tag exists purely so the module proxy can find the
subdirectory module.

**The registries already hold 0.1.0**, so this is an update release:
`publish-npm` stages the tarball through the stage-only trusted publisher and
waits for a 2FA approval on npm's Staged Packages page (§7.1); PyPI and
crates.io publish through their trusted publishers (§7.0). Confirm those
publishers exist before approving the wall: a missing one fails the job at
authentication, and §7.1 or §7.3 says what to do by hand. No long-lived
registry token is stored in GitHub.

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

- [ ] **PyPI**: add a Trusted Publisher at
      <https://pypi.org/manage/account/publishing/>. Create a pending publisher
      with project name `loudkit`, owner `loudreader`, repository `loudkit`,
      workflow `release.yml` and environment `release`. PyPI turns it into a
      normal publisher on first use; no token is stored. A pending publisher
      does not reserve the name, so do this immediately before tagging.
- [ ] **npm**: no repository secret. `loudkit` declares voice enrollment as
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
- [ ] **crates.io**: no repository secret. The first crate must exist before
      crates.io lets it trust a workflow, so §7.3 uses a short-lived token with
      only the `publish-new` endpoint scope, logs out and revokes it. Then add
      a GitHub Actions Trusted Publisher for owner `loudreader`, repository
      `loudkit`, workflow `release.yml` and environment `release`, and require
      Trusted Publishing for later versions. The workflow obtains a temporary
      token through `rust-lang/crates-io-auth-action` and that action revokes it
      when the job ends.
- [ ] **GitHub**: create the `release` environment under Settings →
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

### 7.1 npm: the staged path, and the bootstrap if trust is not configured

`loudkit@0.1.0` is on npm. If the package carries the stage-only trusted
publisher and 2FA-required publishing (§7.0), the staged path further down
applies and the bootstrap below is history. If it does not, do the bootstrap
once. npm refuses to stage a package without a trusted publisher, and LoudKit's
voice enrollment is declared as dual-use, so a direct publish must be
interactive and protected by 2FA; a CI token that bypasses 2FA and a direct
OIDC publish are both disallowed by npm's policy.

The unavoidable bootstrap exception is native npm provenance: npm can only
mint it from CI, while CI cannot create this first dual-use package. The exact
tarball is still covered by the separate GitHub build attestation created by
`attest-npm`. Every later version is staged through npm OIDC and receives npm's
own provenance as well.

The bootstrap sequence, for a package without a trusted publisher, is:

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
6. In a scratch directory, install `loudkit@0.1.1` and import it. Also repeat
   `gh attestation verify` on the downloaded tarball as the provenance gate for
   this bootstrap version.

With the trusted publisher in place, `publish-npm` stages the checked tarball through
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

Rollback: `npm unpublish loudkit@0.1.1` works for **72 hours** and only while
nothing depends on it. After that the version is permanent and `npm deprecate`
is the whole remedy.

Gate: in a scratch dir, `npm install loudkit@0.1.1` and import it. Do not
continue until this passes.

### 7.2 PyPI

```bash
twine check dist/loudkit-0.1.1*
twine upload dist/loudkit-0.1.1*.whl dist/loudkit-0.1.1*.tar.gz
```

Rollback: none worth the name. Deleting the release removes the files, so
nobody installs a broken artefact, but **the version number is burned**. PyPI
never allows `0.1.1` to be uploaded again, under any content. A mistake here
costs `0.1.1`.

Gate: `pip install loudkit==0.1.1` in a fresh venv, run the README block.

### 7.3 crates.io

`loudkit@0.1.0` is on crates.io. With its GitHub Actions Trusted Publisher
configured (§7.0), `publish-crates` publishes through OIDC and nothing below
is needed. Without it the workflow stops here, after npm and PyPI are live,
and the failed job prints this procedure:

1. Create an API token at <https://crates.io/settings/tokens/new> with a short
   expiry and only the `publish-new` endpoint scope. Do not add it to GitHub.
2. Check out the exact public tag in a clean worktree. Never publish from the
   private development branch:

   ```bash
   REPO=$(git rev-parse --show-toplevel)
   CRATE_TREE=$(mktemp -d)
   git -C "$REPO" worktree add --detach "$CRATE_TREE" v0.1.1
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

Rollback: none. `cargo yank --version 0.1.1` stops **new** dependents from
selecting it; it does not delete anything, existing lockfiles keep resolving
it, and the version can never be reused or replaced. crates.io is last because
it is the only registry where a bad upload cannot be withdrawn at all.

### 7.4 Go and Swift

Nothing to upload, but Go has one act left: the tag §6 deliberately held back.
Push it only now, with the workflow green, the approval given and the three
registries published:

```bash
git tag go/v0.1.1 public-main
git push public go/v0.1.1
```

The pushed tags are the release. First `go get` against a fresh module can
take a few minutes while `proxy.golang.org` indexes it. Swift needs no tag of
its own: SwiftPM resolves `v0.1.1`.

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
- [ ] `swift package resolve` picks up `0.1.1` from the repository URL.
- [ ] The Colab badge, clicked from the public README, runs end to end.
- [ ] The docs site at `https://loudreader.github.io/loudkit/` has deployed
      from `main` and carries no pre-release banner.
- [ ] The Space `jer3mi/loudkit` runs the release: bump `requirements.txt`
      in `space-dir/` to `loudkit[torch,audio,enroll,hub]==0.1.1`, push it,
      and confirm the Space restarts and speaks.

Then run each guide as a stranger would, from the published packages rather
than from a checkout. A guide that was true against this working copy can
still be false against a registry: the wheel ships a subset of the tree, the
npm tarball another, and the crate a third.

- [ ] Guide 1: install, `loudkit download`, the two-line synthesis, and the
      five one-liners in English, Spanish, French, German and Italian.
- [ ] Guide 2: `stream` and `synthesize`, and `previous_tokens` across
      two calls.
- [ ] Guide 3: `lk.enroll` from a ten-second recording, then synthesise with
      the profile it wrote.
- [ ] Guide 4: `loudkit serve`, one `/v1/synthesize` call, one stream, the
      OpenAI route, `loudkit serve --mcp` and `loudkit serve --grpc`.
- [ ] `python tools/bench.py` produces a row on this machine (guide: docs/design/benchmarking.md).
- [ ] Guides 7 to 10: the TypeScript, Go, Rust and Swift quickstarts from
      scratch consumers, against the published packages and the public Hub
      release.

Record what broke. A guide that needs a fix is a patch release, not an
edit to the tag.

## 9. Pre-release wording, and when each piece comes off

Two rounds, because two things publish at different times. The weights went to
the Hub first, and the wording that said they had not went with them. The
package wording below is gone too; every row stays, so that a marker cannot
come back unnoticed.

Anything that reappears comes off in the **release commit (§2)**, not at §7
where the packages actually publish: `tests/test_release_coherence.py` refuses a stable
version while any of these stands, and the tag is cut in §6. So the wording
goes one section before the gate that would otherwise stop the release.

This table and `PRERELEASE_BANNERS` in that test are one list written twice,
and a test asserts they agree. Grep for the marker rather than trusting a line
number.

| file | grep for | what it is |
| --- | --- | --- |
| `README.md` | `packages are not yet` | the README pre-release note |
| `notebooks/loudkit_quickstart.ipynb` | `Pre-release.` | the Colab pre-release note |
| `docs/reference/troubleshooting.md` | `lands on PyPI with the 0.1.0 release` | the not-on-PyPI-yet paragraph |
| `site/scripts/sync-docs.mjs` | `banner:` | the site-wide banner written into every generated page. Already absent; the row stays so it cannot come back unnoticed |
| `site/src/handwritten/index.mdx` | `lk-banner` | the landing-page banner block |
| `site/src/handwritten/demo.mdx` | `banner:` | the demo page banner front matter |
| `site/src/handwritten/index.mdx` | `git = "https://github.com/loudreader/loudkit"` | the landing page's Rust tab. Becomes `loudkit = "0.1"` |
| `site/src/handwritten/index.mdx` | `branch: "main"` | the landing page's Swift tab. Becomes `from: "0.1.1"` |
| `docs/guides/10-swift.md` | `branch: "main"` | guide 10's Swift dependency. Becomes `from: "0.1.1"`, which the sentence under it already tells the reader to use |

The last three are not banners, and that is why they went unnoticed: an install
line naming a branch or a git URL is a pre-release instruction wearing ordinary
syntax. A reader who copies one after 0.1.1 ships builds from `main`, the
moving target the release exists to replace.

`site/src/content/docs/` is generated and gitignored. Do not edit the copies
there; they are rewritten from `docs/` and `site/src/handwritten/` on every
build.

The README sentence "the teacher's training data is Resemble AI's and is not
published" is a permanent statement about the upstream teacher, not a
pre-release note. It stays.

### Preparing the complete voice directory

Both release profiles require all 28 names in the canonical roster. The eight
additional English profiles are versioned in this repository. When preparing
`assets/voices` from an older 20-voice release, include their corrected files
before running either builder:

```bash
cp docs/voices/preview/profiles/*.safetensors assets/voices/
```

The `preview` directory is a retained source path, not a separate download
requirement for model users. Both finished bundles include these files under
`voices/`, and every SDK loads them by name.
