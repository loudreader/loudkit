# Releasing loudkit

`.github/workflows/release.yml` builds and checks the packages, and attests the
Python distributions and the npm tarball. After a reviewer approves the
`release` environment, it publishes to npm, PyPI and crates.io, then creates
the GitHub Release. §7 gives the manual fallback for each registry. Follow the
sections in order.

Each release adds to the existing public history and to the existing registry
packages. In this file, `X.Y.Z` is the version you release.

## 0. Preconditions

- Verify that the `public` remote is `loudreader/loudkit`, and fetch it.
- Keep its `main` history. Do not create another parentless root commit,
  overwrite a published tag or force-push.
- Confirm that this project owns `loudkit` on PyPI, npm and crates.io, and
  that the trust settings in §7.0 are in place.
- Confirm that the public repository answers without authentication:

  ```bash
  curl -s -o /dev/null -w '%{http_code}\n' https://api.github.com/repos/loudreader/loudkit   # 200
  ```

- Publish both verified model bundles and create their `vX.Y.Z` Hub tags
  before the code tag (§5). Record both immutable Hub commit IDs in CI (§5.3).
- Get the release owner's listening and provenance acceptance (§4).

## 1. Version sync

Four files carry a version and must agree before tagging. Go and Swift take
their version from the git tag and have nothing to edit.

| file | field | value on `main` |
| --- | --- | --- |
| `pyproject.toml` | `version` | `0.1.1` |
| `js/package.json` | `version` | `0.1.1` |
| `rust/Cargo.toml` | `version` | `0.1.1` |
| `python/loudkit/_version.py` | `__version__` | `0.1.1` |

Set all four to `X.Y.Z`, and change the last column of this table in the same
commit. `js/package-lock.json` carries the version twice; update it too.

The release workflow refuses a tag that does not match `pyproject.toml`.
`tests/test_release.py::test_every_published_manifest_carries_the_same_version`
holds the other three files to `pyproject.toml`, and
`test_the_release_table_names_the_versions_the_files_carry` holds this table to
the files. `tests/test_release_coherence.py` also checks the lockfile. Run the
tests after the edit and before the tag:

```bash
pytest tests/test_release.py tests/test_release_coherence.py -q
```

## 2. The release commit

Put all of the following in one commit.

- [ ] The four version fields from §1, and the §1 table.
- [ ] The `X.Y.Z` section of `CHANGELOG.md`, with the release date
      (`YYYY-MM-DD`) on its heading.
- [ ] The "Listen" section of `docs/MODEL_CARD.md` carries the two audio
      players below. They point at `samples/` in the Hugging Face model
      repository. The strict builder copies those files from the voice roster
      and lists them in both manifests. Edit the card in this commit: it is a
      hashed member of the bundle §5.1 builds, and §5.5 freezes the tree. §5.1
      confirms that the players play.

          <audio controls src="https://huggingface.co/loudreader/loudr-1/resolve/main/samples/joe.opus"></audio>
          <audio controls src="https://huggingface.co/loudreader/loudr-1/resolve/main/samples/kathleen.opus"></audio>

      Both read the same sentence at seed 7. Do not use files from
      `docs/voices/roster/audio/refs/`. Those are compressed previews of the
      human enrollment recordings, not model output, and the model repository
      does not ship them.
- [ ] No pre-release wording from §9 remains. `tests/test_release_coherence.py`
      refuses a stable version while any §9 marker remains, so the tag in §6
      depends on this item.
- [ ] The Hub commits of both bundles are pinned in
      `.github/workflows/ci.yml` (`LOUDKIT_HF_REVISION`,
      `LOUDKIT_TURBO_HF_REVISION`). They exist only after §5.1 and §5.2, so
      cut the release commit after the uploads. §5.3 is that commit.
- [ ] Decide whether this release is a patch, a minor or a major against
      [docs/reference/COMPATIBILITY.md](docs/reference/COMPATIBILITY.md).

## 3. Local gates

Run these on the release commit, from the repo root. They are the CI gates.
Use the tool versions CI pins.

```bash
just check    # ruff check, ruff format --check and mypy, as CI's check job runs them
just test     # pytest -m "not slow"
(cd rust && cargo fmt --check \
  && cargo clippy --all-targets -- -D warnings \
  && cargo clippy --all-targets --no-default-features -- -D warnings \
  && cargo test --no-default-features && cargo test)
(cd go && test -z "$(gofmt -l .)" && go vet ./... && go test ./...)
(cd js && npm ci && npm run lint && npm test)
swiftlint lint --quiet
swift build && swift test
```

CI also runs the jobs below. Run the ones your host supports:

```bash
# Funnel fuzzing across the ports. Needs no weights and no network.
python tools/fuzz_parity.py --cases 300 --seed 1 --ports go,rust,js
python tools/fuzz_parity.py --cases 300 --seed 2 --ports go,rust,js
python tools/fuzz_parity.py --cases 300 --seed 1 --ports swift   # macOS
python tools/fuzz_parity.py --cases 300 --seed 2 --ports swift   # macOS

# The crate at its MSRV, and with the coreml feature on (macOS only).
(cd rust && cargo +1.88 build --locked)
(cd rust && cargo clippy --all-targets --features coreml -- -D warnings && \
            cargo test --features coreml)

# The wheel and the sdist. This only builds them. CI's clean-install and
# packaging jobs also import the wheel outside the repo and check the sdist
# against its allowlist.
python -m build --sdist --wheel --outdir /tmp/lk-dist
```

The MSRV check needs a Rust 1.88 toolchain and the coreml check needs macOS. On
another host, leave them to CI and check their results there. Two more CI jobs
run only in CI: `dependency audit` (`pip-audit`, `npm audit`, `govulncheck`,
`cargo audit`) needs a clean resolver environment, and `parity` (hosted macOS)
downloads the pinned public release. Confirm both in CI.

## 3.5 Re-measure the speed tables if engine speed changed

`docs/benchmarks.md` holds the published real-time factors. The README, the
port guides and the website quote it, and
`tests/test_docs.py::test_the_headline_speed_figures_are_quoted_from_one_page`
checks that the quotes agree with it. The test checks agreement, not whether
the figures are current. The machines behind the page, an M3 Pro and six
NVIDIA parts, are not in CI.

If this release changes engine speed, measure the affected tables again before
tagging. Measure both models. Use the voice and seed the page declares (`joe`,
seed 7). On NVIDIA, measure eager and with `--cuda-graphs`:

```bash
python tools/bench.py --checkpoint <checkpoint> --voice joe --seed 7 \
  --device cuda --cuda-graphs --json out/bench.json
```

The port table comes from `tools/bench_ports` and the batch table from
`research/bench_batch.py`.

Then update `docs/benchmarks.md` and the quoted figures the test checks (six
rows, both models). Set the release named under "Measured on which release"
and in each table you measured again. Keep the release label of every table
you did not measure again.

## 4. Manual acceptance

Do this **before** tagging, on the tree you intend to tag. Work as a new user:
a fresh environment, no cache and no knowledge from the checkout. A failure
here blocks the release.

The release operator can run every command in §§4.1–4.9 and attach the output.
The release owner does not need to repeat those commands. These five decisions
need the release owner:

- [ ] README on GitHub. Open the rendered README in a private window. From the
      README alone, confirm that a reader can tell what loudkit is, who it is
      for, how to make the first WAV and which page covers their runtime.
      Reject it if it reads as an internal specification.
- [ ] Model card on Hugging Face. Both native players render and play. The
      card gives, in this order: how the model sounds, how to try it, what
      gets downloaded, and the quality limits. Reject it if implementation
      details come before the first synthesis.
- [ ] Listening. Play all 28 roster samples to the end, then a fresh English,
      Polish and Spanish render. Reject dropouts, repetitions, bad tails,
      numbers read in the wrong language, and any sample you would not publish
      as a first example of the model.
- [ ] Cloning. Enroll one recording that you have explicit permission to use,
      and listen to at least two sentences. Decide by listening whether
      identity and intelligibility meet the v0.1 claim.
- [ ] Claims. Read the top half of the model card, `VOICES.md` and
      `RESPONSIBLE_USE.md` as the person named on the release. Confirm that
      they state the English-only quality evaluation, the consent basis and
      the provenance trust boundary clearly and correctly.

The checks below give evidence that anyone can reproduce. They must pass, and
the operator can run them.

### 4.1 Python

- [ ] Build locally and install the wheel into a fresh venv, not with
      `pip -e .`. Put the wheel path in a variable so the shell expands the
      glob:

      ```bash
      python -m build
      python -m venv /tmp/lk
      whl=$(ls dist/loudkit-X.Y.Z-*.whl)
      /tmp/lk/bin/pip install "${whl}[torch,audio,enroll,hub]"
      ```

- [ ] Run the Python example in the README (section "Python") unchanged. It
      covers the download, the cache, `engine.synthesize(...)` and
      `.save("hello.wav")`. Listen to `hello.wav`.
- [ ] Repeat the synthesize line with a `pl` voice and an `es` voice. Listen:
      numbers, currency and times in a sentence such as
      „Pociąg o 14:30 kosztuje 2,5 mln zł" must be read in the voice's
      language, not in English.
- [ ] Enrollment: run `lk.enroll(...)` on a 10-second clip, synthesize with the
      result and listen for identity.
- [ ] Errors: an unsupported language code raises `UnsupportedLanguageError`,
      and its message names the supported languages. A first load with no
      network gives an error that names what to download.
- [ ] `python -c "import loudkit; print(loudkit.__version__)"` prints `X.Y.Z`.
- [ ] `twine check dist/*` passes on both artefacts.

### 4.2 Colab

- [ ] Open the README Colab badge while **logged out of GitHub**. It points at
      `loudreader/loudkit`, opens without authentication and does not fall
      back to a contributor's fork.
- [ ] The README and the notebook carry no pre-release note (§9).
- [ ] Runtime → Run all, with no edits. Every cell passes and the audio plays
      inline.

### 4.3 npm

- [ ] Run `cd js && npm pack --dry-run`. The `prepack` guard
      (`js/scripts/check-pack.mjs`) runs `npm run build`, copies the data files
      and refuses a tarball that lacks `dist/`, any of the three data files,
      `LICENSE`, `NOTICE` or `DISCLOSURE`. Confirm that the listing contains
      those files, `README.md`, `data/numbers.json`, `data/numerals.json`,
      `data/pl_en_respell.json` (about 6.6 MB) and `dist/`, and no
      `dist/test/` entry. Confirm that `package.json` declares
      `contentPolicy.class` as `dual-use`.

      `js/data/` is gitignored and generated at build time. Run this check
      from a fresh clone: a machine that built the package before can hold a
      data file that the copier does not name. `js/src/test/pack.test.ts`
      packs the package, unpacks it and folds a numeral from the result.
- [ ] In a scratch directory, run `npm install /path/to/loudkit-X.Y.Z.tgz`,
      then run the example from `js/README.md` against downloaded weights.

### 4.4 Rust

- [ ] Run `cargo new /tmp/lk-rs && cd /tmp/lk-rs`, add the crate as a path
      dependency, paste the example from `rust/README.md` and run
      `cargo run`. §8 repeats this with the registry version.
- [ ] `cargo publish --dry-run` from `rust/` succeeds. It reads the index,
      uploads nothing and needs no token.
- [ ] `cargo package --list` shows `src/numbers.json`, `src/numerals.json`,
      `src/pl_en_respell.json` and the files under `src/enroll_data/`. The
      crate compiles them in and does not build without them.
- [ ] `cargo package --list | grep -c '^tests/'` prints `0`. Some of those
      tests need the monorepo's fixtures and fail in the published crate. The
      `include` list in `rust/Cargo.toml` keeps `tests/` out. Do not add
      `tests/**` to it.

### 4.5 Go

- [ ] Before the push: `go build ./...` and `go test ./...` from `go/`.
- [ ] After the Go tag (§7.4), in a scratch module:

      ```bash
      go mod init tmp
      go get github.com/loudreader/loudkit/go@vX.Y.Z
      ```

      Use `@vX.Y.Z`, not `@go/vX.Y.Z`. The `/go` suffix of the module path
      tells the proxy that the tag has the `go/` prefix, and a query that
      names the tag does not resolve. Paste the example from `go/README.md`,
      then run `go run .`.

### 4.6 Swift

- [ ] Before the push: `swift build && swift test` at the repo root.
- [ ] After the push: on macOS, build and run a scratch package that depends
      on `.package(url: "https://github.com/loudreader/loudkit", from: "X.Y.Z")`,
      with the example from the Swift section of the README. SwiftPM strips
      the leading `v` from tags, so `from: "X.Y.Z"` resolves the `vX.Y.Z` tag.
      Swift needs no tag of its own: `Package.swift` is at the repo root, and
      SwiftPM cannot read a manifest from a subdirectory.

### 4.7 Roster samples

- [ ] Play every sample under `docs/voices/roster/audio/` to the end. A
      dropout, repetition, tail artifact or wrong-language reading
      disqualifies the sample. Render it again, put the new sha256 in
      `docs/voices/roster/provenance.json` and ship the new file. The hashes
      record the published bytes. They do not reproduce on another machine.

### 4.8 Links and model card

- [ ] On the GitHub rendering of the public repo, click every link in
      `README.md` and `VOICES.md`: relative links, `docs/`, the model card and
      the benchmarks page. No link returns 404 or points to a file that does
      not exist.
- [ ] Read `docs/MODEL_CARD.md` for errors a reader would notice. This is the
      last check before §5.1 hashes the card into the bundle. A later change
      means a rebuild.
- [ ] The repo has a description and topics, and issues are enabled.

### 4.9 Published tree

- [ ] The release commit tracks nothing that is not the project's to ship:
      working notes, scraped pages, probe outputs, `.env` files, credentials,
      or build output such as `out/` and `dist/`. Check `git ls-files`, not
      only `.gitignore`.
- [ ] `LICENSE` (Apache-2.0) and `NOTICE` are at the root and render on
      GitHub.
- [ ] `go/`, `rust/` and `js/` each carry a byte-identical copy of both.
      `test_every_published_package_carries_the_licence_and_the_notice`
      checks this.

## 5. Bundles, pins, then public main

Upload the two bundles first (§5.1, §5.2). Then pin their Hub commits in CI
(§5.3). Then push the tree to public `main` (§5.5). In this order, the commit
that carries the tag can pass its own parity job.

### 5.1 Build the loudr-1 bundle and upload it

Split the checkpoint into a new directory. A release ships two halves. The
builder finds the enrollment half beside the synthesis half by its canonical
name; there is no flag for it. Both halves must be in one directory, and that
directory must not be the packed checkpoint's own. The tool refuses to write a
half beside the packed original.

```bash
.venv/bin/python tools/split_checkpoint.py \
    --checkpoint assets/loudr-1.safetensors \
    --out-dir    dist/release-X.Y.Z/split/loudr-1
```

Split only if the pair is missing. When both halves exist, the tool reports
that and stops. Do not use a force flag. Each half records the digest of the
packed original, and the build refuses two halves from different runs.

Add `edge_fade_seconds` to the synthesis half, and nothing else. A half cut
from a packed original that lacks the key does not carry it either. Without
`--only`, the amender also writes `chunking`, and it refuses a manifest that
already carries a different value. The loader fills the same defaults either
way, so the fingerprint does not change; `--only` keeps the file digest
stable. If the half already carries the key, the tool reports that there is
nothing to do.

```bash
.venv/bin/python tools/amend_manifest.py \
    --checkpoint dist/release-X.Y.Z/split/loudr-1/loudr-1.safetensors \
    --only edge_fade_seconds
shasum -a 256 dist/release-X.Y.Z/split/loudr-1/loudr-1.safetensors
```

For the loudr-1 weights that ship now, the digest is `73e69a78d58176a3…`.
`docs/voices/roster/provenance.json` records it for every sample, and the
export records bind to it. Investigate any other digest before you export. The
same step on the turbo half gives `590dcf9e1dcec31c…` (§5.2).

`assets/voices` must hold all 28 roster voices. Eight of the English profiles
are versioned in this repository. Before you build either release, copy them
in, also over files that already exist. The release audit does not compare
these profiles with the versioned copies, so an old copy would ship:

```bash
cp docs/voices/preview/profiles/*.safetensors assets/voices/
```

Model users do not download from `docs/voices/preview/`: both bundles ship
these profiles under `voices/`, and every SDK loads them by name.

Export from the split files. Copy `tokenizer.json` beside
`dist/release-X.Y.Z/split/loudr-1/loudr-1.safetensors`, then run the synthesis
exporters with that synthesis half as `--checkpoint`. Export enrollment from
`dist/release-X.Y.Z/split/loudr-1/loudr-1-enrollment.safetensors` and the
shared `ve.safetensors`. Every path below is under `dist/release-X.Y.Z/`:

- `split/<model>`: the two halves
- `exports/<model>/{onnx,coreml}`: the graphs the exporters write
- `samples/<model>`: the two rendered players
- `<model>`: the bundle

```bash
S=dist/release-X.Y.Z/split/loudr-1; E=dist/release-X.Y.Z/exports/loudr-1
.venv/bin/python tools/export_onnx.py          --checkpoint $S/loudr-1.safetensors --out $E/onnx
.venv/bin/python tools/export_enroll_onnx.py   --checkpoint $S/loudr-1-enrollment.safetensors --voice-encoder assets/ve.safetensors --out $E/onnx
.venv/bin/python tools/export_coreml.py        --checkpoint $S/loudr-1.safetensors --out $E/coreml
.venv/bin/python tools/export_enroll_coreml.py --checkpoint $S/loudr-1-enrollment.safetensors --voice-encoder assets/ve.safetensors --out $E/coreml
```

Every exporter checks its graphs against torch before it writes. The two
synthesis export records name the checkpoint digest `73e69a78…` and the
fingerprint `7cd75498ad4e7531`. The exporters need the untracked release
assets in `assets/`: the packed checkpoint, `ve.safetensors`, `tokenizer.json`
and `voices/`. Git tracks only the logos there, so a fresh worktree does not
have them. Do not reuse synthesis graphs exported from the unsplit checkpoint:
their checkpoint digest is different, and the runtime refuses a mixed set.
Keep each model's output directories separate. Run CoreML export and inference
with access to the macOS compute services. A sandboxed process can select a
different compute path and does not reproduce the published measurements.

Do not edit the model card in §5; §2 finalizes it. The bundle is built before
the public push (§5.5) and the tag (§6). A card changed after this point would
ship to the Hub without being in `main`. The only tracked file the rest of §5
edits is `.github/workflows/ci.yml`, in the pin commit (§5.3).

- [ ] The two audio players from §2 name the model repository's own
      `samples/joe.opus` and `samples/kathleen.opus`, not GitHub raw content.
      They play only after the new bundle is merged. The check after the merge
      below is the listening gate.
- [ ] The card's claims match the artefacts: voice count, language list,
      artefact sizes and the download table.
- [ ] Every link in the card resolves, and the docs site is live. §5.6 checks
      the site again after the push.

Build the bundle with the strict profile, `full-0.1`. It is the default, and a
release is cut only from it. `--checkpoint` names the synthesis half, and the
builder finds the enrollment half beside it:

```bash
.venv/bin/python -m tools.release --model loudr-1 \
    --checkpoint    dist/release-X.Y.Z/split/loudr-1/loudr-1.safetensors \
    --voice-encoder assets/ve.safetensors \
    --voices        assets/voices \
    --onnx          dist/release-X.Y.Z/exports/loudr-1/onnx \
    --coreml        dist/release-X.Y.Z/exports/loudr-1/coreml \
    --out           dist/release-X.Y.Z/loudr-1 \
    --profile       full-0.1
```

`--profile full-0.1` requires:

- the checkpoint under its canonical name `loudr-1.safetensors`, and its
  manifest
- the tokenizer and `ve.safetensors`
- all 28 roster voices, by name
- the complete ONNX and CoreML synthesis and enrollment graph families
- the four documents and the loudkit wordmark
- the two listening samples
- `release.json`, and a `SHA256SUMS` that covers every other file

The roster is `docs/voices/roster/provenance.json`: 28 voices in ten
languages, ten English and two for each other language. The build refuses a
missing voice and a voice that is not on the roster.

The tool checks the sources first. If a source is missing, it refuses and
lists each missing file with the tool that exports it. It assembles the bundle
in a staging directory beside `--out` and renames it into place only after
every check passes, so a failed run leaves no bundle that looks publishable. A
killed run cannot clean up, so the staging directory records the pid of the
build that owns it. The next run removes staging directories whose process is
gone and names each one. It leaves the directory of a running build alone. The
tool copies graphs and packages by name. Then it audits the bundle against the
profile's allowlist and refuses any file the profile does not name.

The closing gate loads what the bundle ships. It synthesizes on the torch
path, loads all 28 voices and clones a voice with the shipped
`ve.safetensors`. It synthesizes on the ONNX and CoreML paths. It runs one
enrollment through the three enrollment graphs on both graph paths and checks
it against the enrollment fixture.

Build on macOS. The CoreML packages open only on an Apple platform.
`full-0.1` refuses to run on another platform; it does not skip the CoreML
checks.

`release.json` records the profile that built the bundle, and
`"verified": true` when the closing gate ran and passed. A consumer or a CI job
reads it to tell a release bundle from a development bundle. The builder writes
it before `SHA256SUMS`, so `SHA256SUMS` covers it.

The gate imports and runs the bundle's own code. After the gate, the builder
checks the bundle again from disk: it hashes every file against the
manifests, matches the inventory in both directions and audits the allowlist
again. The manifests carry the digests taken before the gate, so a gate that
changed or added a file causes a refusal.

The same audit runs in place, without a build:

```bash
.venv/bin/python -m tools.release --verify-only dist/release-X.Y.Z/loudr-1
```

The build and `--verify-only` use the same function, `check_bundle`. It does
not load the model. It reports whether the files on disk match the manifests,
and whether the bundle matches the profile it names.

`full-0.1` refuses `--skip-verify`.

`--profile lenient` builds a partial bundle for development. It writes
`"profile": "lenient"` and `"verified": false`. Do not upload one.

Before the upload:

- [ ] `.venv/bin/python -m tools.release --verify-only dist/release-X.Y.Z/loudr-1`
      passes. Run it immediately before the upload. It detects any change to
      the bundle after the build.
- [ ] `shasum -a 256 -c SHA256SUMS` passes from the release root, and
      `SHA256SUMS` has **one fewer line than the bundle has files**: the count
      difference below is exactly one. `SHA256SUMS` cannot contain its own
      digest. Every other file has a line, including `release.json`, every
      graph, every package file and every sample.

      ```bash
      (
        cd dist/release-X.Y.Z/loudr-1
        shasum -a 256 -c SHA256SUMS
        echo $(( $(find . -type f | wc -l) - $(wc -l < SHA256SUMS) ))   # 1
      )
      ```

- [ ] `release.json` says `"profile": "full-0.1"` and `"verified": true`. Do
      not upload a lenient or unverified bundle.
- [ ] `diff docs/MODEL_CARD.md dist/release-X.Y.Z/loudr-1/README.md` prints
      nothing. The card is a hashed bundle member. An edit on the Hub later
      makes the remote tree disagree with its own checksums.
- [ ] The layout has `voices/*.safetensors` one level down and exactly three
      safetensors files at the root: the synthesis half, the enrollment half
      and `ve.safetensors`. The loader selects the synthesis half by its role,
      so these three files are not ambiguous.
- [ ] Read
      [`docs/PROVENANCE-voice-encoder.md`](docs/PROVENANCE-voice-encoder.md)
      again. It records the encoder's hash, tensors, upstream repository and
      licence. Publishing the weight cannot be reversed.

Open a new pull request on the Hub for the bundle. `--delete "*"` is part of
the command. Without it, files that the new release does not ship stay beside
the new inventory, and the remote repository no longer matches the verified
bundle.

```bash
(
  cd dist/release-X.Y.Z/loudr-1
  hf upload loudreader/loudr-1 . . \
      --repo-type model \
      --create-pr \
      --delete "*" \
      --commit-message "loudr-1: X.Y.Z bundle"
)
```

- [ ] Review the Hub pull request before merging. Its final tree, not only its
      additions, matches `dist/release-X.Y.Z/loudr-1`: both split halves are
      present, no file outside the bundle remains, and `samples/` holds
      exactly the two players.
- [ ] Merge it in the Hugging Face interface, then record the immutable commit
      SHA. The default branch name does not identify a release:

      ```bash
      SHA=$(curl -s https://huggingface.co/api/models/loudreader/loudr-1 \
        | python3 -c "import json,sys; print(json.load(sys.stdin)['sha'])")
      echo "$SHA"
      ```

- [ ] Tag that commit on the Hub as `vX.Y.Z`. The guides and the port READMEs
      pin a Hub tag as their `revision`. If the tag already exists, check its
      target and do not move it:

      ```bash
      .venv/bin/hf repos tag create loudreader/loudr-1 vX.Y.Z --revision "$SHA"
      HF_HOME=$(mktemp -d) .venv/bin/python -c '
      import loudkit as lk
      print(lk.load("loudreader/loudr-1", revision="vX.Y.Z").describe())
      '
      ```

- [ ] Open the rendered model card in a private window. The loudkit wordmark
      is visible. Both players fetch from `samples/` in the model repository
      and play to the end.

Check the Hub repository against the bundle:

- [ ] `loudreader/loudr-1` on Hugging Face is public and ungated, and carries:
      - `loudr-1.safetensors`, `loudr-1-enrollment.safetensors`,
        `ve.safetensors` and `tokenizer.json`
      - `manifest.json`, `release.json` and `SHA256SUMS`
      - nine ONNX graphs and nine CoreML packages (in each format, six for
        synthesis and three for enrollment), with `export.json` in `onnx/`
        and in `coreml/`
      - the 28 voice profiles under `voices/`
      - `README.md`, `logo.png`, `LICENSE`, `NOTICE` and `RESPONSIBLE_USE.md`
      - `samples/joe.opus` and `samples/kathleen.opus`, for the card's
        players

      `SUPPORTED.md` declares voice enrollment in all five implementations.
      Enrollment needs `s3_tokenizer`, `camp` and `voice_encoder` on both
      graph paths. A bundle without those six files is not a release.
- [ ] The first-mile test passes. Run `lk.load("loudreader/loudr-1")`,
      `lk.voice(...)` and a synthesis against the public repo, from a machine
      with no local cache, pinned to the SHA recorded above. A failure blocks
      the release.

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

Turbo has its own repository. Publish it before or after `loudr-1`; the two
uploads are independent. The README, the guides and the site load
`loudreader/loudr-1-turbo` by repo id, so publish it before §6, or remove those
mentions.

Both models ship ONNX and CoreML graphs and the same portable voice profiles.
Split the fusion checkpoint with `tools/split_checkpoint.py`. The turbo bundle
carries its synthesis half, its matching enrollment half, `ve.safetensors` and
the same enrollment graphs as loudr-1. A voice profile cloned once works with
either model. A synthesis-only download does not need the enrollment assets.

Export synthesis from the verified split checkpoint
`dist/release-X.Y.Z/split/turbo/loudr-1-turbo.safetensors`, into
`dist/release-X.Y.Z/exports/turbo/onnx` and `coreml`. Copy the base model's
enrollment graphs into those directories. Do not rewrite the synthesis export
records or substitute base renderer graphs: the decoder and the Euler step
count come from this checkpoint. Render `joe.opus` and `kathleen.opus` with
this model into `dist/release-X.Y.Z/samples/turbo`. Copies of the base samples
would carry the other model's audio.

```bash
S=dist/release-X.Y.Z/split/turbo; E=dist/release-X.Y.Z/exports/turbo; B=dist/release-X.Y.Z/exports/loudr-1
cp assets/tokenizer.json $S/tokenizer.json
.venv/bin/python tools/amend_manifest.py --checkpoint $S/loudr-1-turbo.safetensors --only edge_fade_seconds
shasum -a 256 $S/loudr-1-turbo.safetensors            # 590dcf9e1dcec31c…
.venv/bin/python tools/export_onnx.py   --checkpoint $S/loudr-1-turbo.safetensors --out $E/onnx
.venv/bin/python tools/export_coreml.py --checkpoint $S/loudr-1-turbo.safetensors --out $E/coreml
for g in s3_tokenizer camp voice_encoder; do cp $B/onnx/$g.onnx $E/onnx/; cp -R $B/coreml/$g.mlpackage $E/coreml/; done
```

The synthesis export records name `590dcf9e…` and the fingerprint
`e5303ba243087222`, one Euler step and the `fusion_mtp2` family. The samples
read the roster's sample text at seed 7, once per voice, on the CPU torch path.
They are encoded the same way as the roster files:

```bash
.venv/bin/python - <<'PY'
import json, subprocess, loudkit as lk
from pathlib import Path
R = Path("dist/release-X.Y.Z"); out = R / "samples/turbo"; out.mkdir(parents=True, exist_ok=True)
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
    --checkpoint    dist/release-X.Y.Z/split/turbo/loudr-1-turbo.safetensors \
    --tokenizer     dist/release-X.Y.Z/split/turbo/tokenizer.json \
    --voice-encoder dist/release-X.Y.Z/loudr-1/ve.safetensors \
    --enrollment    dist/release-X.Y.Z/split/turbo/loudr-1-enrollment.safetensors \
    --voices        dist/release-X.Y.Z/loudr-1/voices \
    --onnx          dist/release-X.Y.Z/exports/turbo/onnx \
    --coreml        dist/release-X.Y.Z/exports/turbo/coreml \
    --samples       dist/release-X.Y.Z/samples/turbo \
    --out           dist/release-X.Y.Z/loudr-1-turbo
```

The builder refuses missing graphs, enrollment files or samples before it
copies anything. It then synthesizes on torch, ONNX and CoreML, verifies
enrollment and the roster, and checks that the checkpoint declares the paired
decoder. Only then does it write `profile: turbo-0.1, verified: true`. It
refuses `--skip-verify`.

- [ ] `docs/MODEL_CARD-turbo.md` is final and its links resolve. It ships as
      the repository's `README.md`. Do not edit it in this step.
- [ ] The bundle passes its own audit in place:
      `.venv/bin/python -m tools.release --verify-only dist/release-X.Y.Z/loudr-1-turbo`
- [ ] Every port passes its asset-backed conformance against the two built
      bundles, with the passes CI's `parity` job runs. The vectors were made
      with the reference voice, so `LOUDKIT_VOICE` is
      `tests/data/reference/testvoice.voice.safetensors`, not a shipped voice.
      Each port's guide names the ONNX Runtime library it needs.

      ```bash
      D=$PWD/dist/release-X.Y.Z; T=$D/loudr-1-turbo
      F=$PWD/tests/data/conformance/vectors_fusion_mtp2.json
      export LOUDKIT_REQUIRE_ASSETS=1 LOUDKIT_TOKENIZER=$D/loudr-1/tokenizer.json \
        LOUDKIT_VOICE=$PWD/tests/data/reference/testvoice.voice.safetensors
      # Python: loudr-1 and the fused turbo case in one run.
      LOUDKIT_ASSET_ROOT=$D/loudr-1 LOUDKIT_TURBO_CHECKPOINT=$T/loudr-1-turbo.safetensors \
        .venv/bin/python -m pytest tests/test_conformance.py -q
      # Go, Rust and JS against loudr-1. Rust also runs its turbo test here.
      (cd go && LOUDKIT_CKPT=$D/loudr-1/loudr-1.safetensors LOUDKIT_ONNX_DIR=$D/loudr-1/onnx \
        LOUDKIT_SECOND_CKPT=$T/loudr-1-turbo.safetensors LOUDKIT_SECOND_ONNX_DIR=$T/onnx \
        LOUDKIT_ONNXRUNTIME_LIB=/path/to/libonnxruntime.dylib go test ./...)
      (cd rust && LOUDKIT_CKPT=$D/loudr-1/loudr-1.safetensors LOUDKIT_ONNX_DIR=$D/loudr-1/onnx \
        LOUDKIT_FUSION_CKPT=$T/loudr-1-turbo.safetensors LOUDKIT_FUSION_ONNX_DIR=$T/onnx \
        ORT_DYLIB_PATH=/path/to/libonnxruntime.1.27.0.dylib cargo test --release -- --ignored)
      (cd js && LOUDKIT_CKPT=$D/loudr-1/loudr-1.safetensors LOUDKIT_ONNX_DIR=$D/loudr-1/onnx \
        LOUDKIT_SECOND_CKPT=$T/loudr-1-turbo.safetensors LOUDKIT_SECOND_ONNX_DIR=$T/onnx \
        npm run test:all)
      # Go and JS against turbo and its own fixture, as CI's turbo step runs them.
      (cd go && LOUDKIT_CKPT=$T/loudr-1-turbo.safetensors LOUDKIT_FUSION_CKPT=$T/loudr-1-turbo.safetensors \
        LOUDKIT_ONNX_DIR=$T/onnx LOUDKIT_FIXTURE=$F \
        LOUDKIT_ONNXRUNTIME_LIB=/path/to/libonnxruntime.dylib \
        go test ./conformance -run TestEngineConformance -count=1)
      (cd js && LOUDKIT_CKPT=$T/loudr-1-turbo.safetensors LOUDKIT_FUSION_CKPT=$T/loudr-1-turbo.safetensors \
        LOUDKIT_ONNX_DIR=$T/onnx LOUDKIT_FIXTURE=$F npm run test:fixture)
      # Swift: both models.
      LOUDKIT_ASSET_ROOT=$D/loudr-1 LOUDKIT_COREML_ASSETS=$D/loudr-1/coreml \
        LOUDKIT_FUSION_CHECKPOINT=$T/loudr-1-turbo.safetensors LOUDKIT_FUSION_COREML_ASSETS=$T/coreml \
        swift test
      ```

The repository `loudreader/loudr-1-turbo` is public and ungated. The parity
job needs this because it downloads anonymously. Confirm it in the repository
settings. Then upload the directory's contents at its root, as §5.1 does for
`loudr-1`: the same `hf upload ... --create-pr --delete "*"`, the same review
of the pull request's final tree, the merge, the recorded commit SHA and the
`vX.Y.Z` tag on that commit. Then, from a machine with no cache:

```bash
HF_HOME=$(mktemp -d) .venv/bin/python -c '
import loudkit as lk
engine = lk.load("loudreader/loudr-1-turbo")
voice = engine.voice("joe")
print("OK", round(engine.synthesize("Turbo works.", voice, seed=7).duration, 2))
'
```

- [ ] `tests/assets.py` registers the turbo assets, so
      `LOUDKIT_REQUIRE_ASSETS=1` requires them. The `parity` job passes only
      when the repository is public and §5.3 pins its commit.
- [ ] `docs/benchmarks.md` names the release each table was measured on
      (§3.5).

### 5.3 The pin commit

The `parity` job downloads both repositories anonymously. Its first step fails
unless `LOUDKIT_TURBO_HF_REVISION` is a 40-hex commit, and `release.yml`
refuses a tag without a successful `parity` run for that commit. So the last
commit before the public push pins what §5.1 and §5.2 merged:

- [ ] `LOUDKIT_HF_REVISION` and `LOUDKIT_TURBO_HF_REVISION` in
      `.github/workflows/ci.yml` carry the two 40-hex commits recorded above,
      and the comment beside them describes those bundles.
- [ ] `PYTHONPATH=$PWD/python .venv/bin/python -m pytest tests/test_release.py -q`
      passes with the new pins.
- [ ] This is the release commit of §2, and it contains every other §2 item.
      No later step edits a tracked file.

### 5.4 Check the release commit

- All changes are committed, the worktree is clean, and every §3 check ran on
  that commit.
- Merge the release work into the private main branch after review.
- Fetch `public/main` before you make the public commit.

### 5.5 Push the public commit

Make the next public commit on a local branch named `public-main`. Its parent
is `public/main` and its tree is the reviewed release tree, so no private
history becomes public:

```bash
git fetch public
RELEASE=PASTE_RELEASE_COMMIT
git branch -f public-main "$(git commit-tree "$RELEASE^{tree}" -p public/main -m "loudkit X.Y.Z")"
git diff --stat "$RELEASE" public-main   # prints nothing
git log --oneline -2 public-main         # the second line is public/main
```

Review the resulting tree and diff, then push without force:

```bash
git push public public-main:main
```

The public branch must stay a descendant of the previous public release. Wait
for the complete public CI run, both models included, before tagging.

### 5.6 Check before tagging

- [ ] The repository is public: the anonymous check from §0 still prints
      `200`.
- [ ] The pushed tree matches the local one:
      `git fetch public && git diff --stat public/main public-main` prints
      nothing.
- [ ] Spot-check the rendered `README.md` and `VOICES.md` on GitHub. No
      pre-release note remains (§9) and no link returns 404.
- [ ] The docs site has deployed from `main` and carries no banner.

Do not tag until all four items pass. The tag starts the publish workflow.

## 6. Tag and let CI build

Push one tag now:

```bash
git tag vX.Y.Z public-main
git push public vX.Y.Z
```

Tag `public-main` explicitly. The maintainer checkout can be on a private
development branch, and its history must not become reachable from the public
repository. Before the push, both of these must print the same commit:

```bash
git rev-parse vX.Y.Z^{}
git ls-remote public refs/heads/main | cut -f1
```

**Do not create or push `go/vX.Y.Z` yet.** That tag is the Go release. Once
`proxy.golang.org` fetches a version, it serves the cached copy permanently.
Moving the tag does not change what the proxy serves, and a later version can
only `retract` it. A `go/` tag pushed before the workflow passes publishes to
Go whatever the workflow later refuses, and the only remedy is a new version.
Push the Go tag last, in §7.4: after the workflow passes, the reviewer approves
the `release` environment and every publish job finishes.

`release.yml` runs on `v*` tags only, so the `go/vX.Y.Z` push starts no second
build. The Go tag lets the proxy find the module in the `go/` subdirectory.

npm, PyPI and crates.io already hold earlier versions, so the release adds a
version to each. `publish-npm` stages the tarball through the stage-only
trusted publisher and waits for a 2FA approval on npm's Staged Packages page.
That staged path currently fails, and the manual fallback in §7.1 is the
working path for npm. PyPI and crates.io publish through their trusted publishers. Before
you approve the `release` environment, confirm the settings in §7.0. A missing
trusted publisher fails its job at authentication, and §7.1 to §7.3 give the
manual fallback. GitHub stores no long-lived registry token.

Wait for `release.yml` to pass. It does the following:

- refuses the tag unless its commit is on `main` **and** that exact SHA has a
  successful `ci` run whose `parity` job succeeded (the `tag-gate` job);
- checks the tag against `pyproject.toml`;
- builds the sdist and the wheel;
- checks that the wheel contains the lexicon and `py.typed`;
- installs both artefacts in clean venvs, imports them and checks the
  respelling data;
- runs the no-weights suite;
- generates an SBOM for the wheel's dependency set, not the runner's;
- attests the Python distributions and the npm tarball in separate jobs that
  run no fetched tooling, so no job that holds an OIDC identity runs the
  build's downloaded dependencies;
- checks and publishes the three packages (§7);
- after the publish jobs, creates the GitHub Release for the tag, uploads
  `dist/*` and the SBOM to it, and generates release notes.

CI creates the Release. Replace its generated notes with the `CHANGELOG.md`
section for this version.

## 7. Publish, in this order

`release.yml` runs this on a version tag. No publish job starts until all
three package checks pass:

```
tag-gate → build → attest            ┐
                 → check-pypi        ├→ [approve `release`] →
                 → check-npm → attest-npm
                 → check-crates      ┘   publish-npm → publish-pypi →
                                         publish-crates → github-release
```

Each check runs on the exact bytes its publish job sends:

- `check-pypi`: `twine check --strict` on the built distributions.
- `check-npm`: `npm pack`, then an install of the packed tarball into a clean
  directory and an import.
- `check-crates`: `cargo package`, then a build and test of the unpacked
  `.crate`, and a new consumer crate that depends on it and calls the API with
  no lockfile and no monorepo paths.

The checks finish before the first publish, so a packaging fault in one
ecosystem shows while the other two are untouched. crates.io never frees a
version number and PyPI never reuses a filename, so a half-published release
cannot be repaired; the next version replaces it. Registry authentication and
upload can still fail in the publish jobs.

The GitHub Release is created last, after the publish jobs.

The manual commands below are the fallback, and they show what the jobs do.

### 7.0 Registry trust settings

Confirm these settings before you approve the `release` environment. CI cannot
set them. Without them, the publish jobs fail at authentication.

- [ ] PyPI: the `loudkit` project has a Trusted Publisher with owner
      `loudreader`, repository `loudkit`, workflow `release.yml` and
      environment `release` (pypi.org → Your projects → `loudkit` → Manage →
      Publishing). No token is stored.
- [ ] npm: the `loudkit` package has a GitHub Actions trusted publisher for
      owner `loudreader`, repository `loudkit`, workflow `release.yml` and
      environment `release`, allowed for **`npm stage publish` only**.
      Publishing access is "Require two-factor authentication and disallow
      tokens". No repository secret is stored. `loudkit` declares voice
      enrollment as dual-use. npm then requires proof of presence, so direct
      automated publishing is not allowed: each staged version waits until a
      maintainer approves it with 2FA. These requirements come from npm's
      [Dual-Use Content Policy](https://docs.npmjs.com/policies/dual-use/) and
      [staged publishing](https://docs.npmjs.com/staged-publishing/) rules.
- [ ] crates.io: the `loudkit` crate has a GitHub Actions Trusted Publisher
      for owner `loudreader`, repository `loudkit`, workflow `release.yml` and
      environment `release`, and the setting that requires Trusted Publishing
      for new versions is on. No repository secret is stored. The workflow gets
      a temporary token through `rust-lang/crates-io-auth-action`, which
      revokes it when the job ends.
- [ ] GitHub: the `release` environment (Settings → Environments) has at
      least one required reviewer. Without one, a tag publishes to three
      registries with no approval.

For a manual step, upload only the artefacts CI built and attested. The
attestation covers those exact bytes. Download them from the release run (the
artifacts are `dist`, `npm-tarball` and `crate-tarball`):

```bash
gh run download PASTE_RELEASE_RUN_ID --repo loudreader/loudkit --name <artifact> --dir dist/
```

The order runs from most to least reversible: npm, then PyPI, then crates.io.
Each publish job runs only after the one before it succeeds. A defect found on
npm can still be unpublished; a defect found after crates.io costs the version
number in every registry.

### 7.1 npm

Until `release.yml` is fixed, the staged path fails. `publish-npm` passes
`npm-tarball/loudkit-X.Y.Z.tgz` to `npm stage publish` without a leading `./`,
and npm reads that argument as a GitHub repository, not as a file. Use the
manual fallback below: it is the working path for npm until the workflow
passes `./npm-tarball/...`.

When it works, `publish-npm` stages the checked tarball through the stage-only
trusted publisher and waits up to 30 minutes. Open npm's Staged Packages page,
inspect the version and approve it with 2FA. Leave the workflow running. PyPI
starts only after the live registry shasum matches the checked tarball. A
staged version carries GitHub's tarball attestation and npm's own OIDC
provenance.

A rerun of a tag whose publish already succeeded does not publish twice.
`publish-npm` finds the version on the registry, compares the registry's
`dist.shasum` with the tarball this run packed, and passes only if they
match. A mismatch fails the job, because the registry then holds bytes this
workflow did not check.

Manual fallback, when `publish-npm` fails at `npm stage publish` (the argument
defect above, or a missing trusted publisher):

1. Download the exact npm artifact from the release run and verify its
   attestation. Use a fresh directory so no local `npm pack` output can be
   selected:

   ```bash
   RUN_ID=PASTE_RELEASE_RUN_ID
   NPM_OUT=$(mktemp -d)
   gh run download "$RUN_ID" --repo loudreader/loudkit \
     --name npm-tarball --dir "$NPM_OUT"
   gh attestation verify "$NPM_OUT"/*.tgz --repo loudreader/loudkit
   shasum -a 256 "$NPM_OUT"/*.tgz
   tar -tzf "$NPM_OUT"/*.tgz
   ```

   The listing must include `DISCLOSURE`, `LICENSE`, `NOTICE`, `dist/` and the
   three files under `data/`: `numbers.json`, `numerals.json` and
   `pl_en_respell.json`. Do not run `npm pack` here. It creates different,
   unattested bytes.
2. Log in to npm with the owner account and publish that tarball
   interactively. Complete the 2FA challenge npm presents:

   ```bash
   npm login --auth-type=web
   npm publish --access public "$NPM_OUT"/*.tgz
   ```

3. Rerun the failed jobs. `publish-npm` compares npm's live `dist.shasum` with
   the checked tarball, and only then lets PyPI and crates.io continue. A
   mismatch stops the chain:

   ```bash
   gh run rerun "$RUN_ID" --repo loudreader/loudkit --failed
   ```

4. In a scratch directory, install `loudkit@X.Y.Z` and import it. Run
   `gh attestation verify` on the downloaded tarball again. A hand-published
   version has the GitHub attestation but no npm provenance.
5. If a trust setting from §7.0 was missing, restore it before the next
   release.

Rollback: `npm unpublish loudkit@X.Y.Z` works within 72 hours if no other
package depends on it. After 72 hours, npm allows it only if nothing depends
on the package, it had fewer than 300 downloads in the last week and it has a
single owner (npm's [unpublish policy](https://docs.npmjs.com/policies/unpublish/)).
npm never accepts an unpublished version number again. Otherwise,
`npm deprecate` marks the version.

Gate: in a scratch directory, `npm install loudkit@X.Y.Z` and import it. Do
not continue until this passes.

### 7.2 PyPI

`publish-pypi` uploads through the Trusted Publisher. Manual fallback, with the
`dist` artifact from the release run:

```bash
twine check dist/loudkit-X.Y.Z*
twine upload dist/loudkit-X.Y.Z*.whl dist/loudkit-X.Y.Z*.tar.gz
```

Rollback: deleting the release removes the files, so nobody installs a broken
artefact. PyPI never accepts the same filename again, so the fix ships as a
new version.

Gate: `pip install "loudkit[torch,audio,hub]==X.Y.Z"` in a fresh venv, then
run the README's Python example.

### 7.3 crates.io

With the Trusted Publisher from §7.0, `publish-crates` gets a short-lived
crates.io token through OIDC and runs `cargo publish --locked`. Nothing below
is needed. Without it, the job fails at the authentication step, after npm and
PyPI are live. Then:

1. Create an API token at <https://crates.io/settings/tokens/new> with a short
   expiry, limited to the `loudkit` crate and the `publish-update` endpoint
   scope. Do not add it to GitHub.
2. Check out the exact public tag in a clean worktree. Do not publish from the
   private development branch:

   ```bash
   REPO=$(git rev-parse --show-toplevel)
   CRATE_TREE=$(mktemp -d)
   git -C "$REPO" worktree add --detach "$CRATE_TREE" vX.Y.Z
   cd "$CRATE_TREE/rust"
   cargo login
   cargo publish --locked
   cargo logout
   ```

   Paste the token only into the prompt of `cargo login`, so it does not enter
   shell history. Whether the publish succeeds or fails, run `cargo logout`
   and revoke the token on crates.io immediately afterwards.
3. Restore the crates.io Trusted Publisher from §7.0.
4. Rerun the failed `publish-crates` job. It compares crates.io's immutable
   checksum with the `.crate` artifact that passed both consumer checks before
   anything was published. It creates the GitHub Release only if those bytes
   match:

   ```bash
   gh run rerun PASTE_RELEASE_RUN_ID --repo loudreader/loudkit --failed
   ```

Rollback: none. `cargo yank --version X.Y.Z` stops **new** dependents from
selecting the version. It deletes nothing, existing lockfiles keep resolving
it, and the version can never be reused or replaced. crates.io is last because
an upload there cannot be withdrawn.

### 7.4 Go and Swift

Go and Swift upload nothing. Push the Go tag now, after the workflow passes,
the approval is given and all three registries are published:

```bash
git tag go/vX.Y.Z public-main
git push public go/vX.Y.Z
```

The pushed tags are the release. The first `go get` against a fresh module can
take a few minutes while `proxy.golang.org` indexes the version. Swift needs no
tag of its own: SwiftPM resolves `vX.Y.Z`.

## 8. Post-publish checks, from the public internet only

Run these outside any checkout of this repository, with empty caches.

`tools/acceptance.py` runs the Python checks:

```bash
python tools/acceptance.py --from-pypi --extras torch,audio,hub --speak
```

It builds a venv outside any checkout, installs the published distribution
and stops unless `loudkit` imports from that venv. With a working directory
inside `python/`, the same interpreter imports the checkout instead, and a
wheel that lacks half its data files would pass every step. `--wheel <path>`
runs the same checks against a locally built wheel before anything is
published.

- [ ] `python tools/acceptance.py --from-pypi --extras torch,audio,hub --speak`
- [ ] `pip install "loudkit[torch,audio,hub]"` in a fresh venv; run the
      README's Python example.
- [ ] `npm install loudkit` in a scratch directory; import it.
- [ ] `cargo add loudkit` in a scratch crate; build it.
- [ ] `go get github.com/loudreader/loudkit/go@latest` resolves through
      `proxy.golang.org`.
- [ ] `swift package resolve` picks up `X.Y.Z` from the repository URL.
- [ ] The Colab badge on the public README runs end to end.
- [ ] The docs site at `https://loudreader.github.io/loudkit/` has deployed
      from `main` and carries no pre-release banner.
- [ ] The Space `jer3mi/loudkit` runs the release. In the Space's repository,
      set `requirements.txt` to `loudkit[torch,audio,enroll,hub]==X.Y.Z` and
      push it. Confirm that the Space restarts and speaks.

Then run each guide against the published packages and the public Hub
release, not against a checkout. The wheel, the npm tarball and the crate each
ship a different subset of the tree.

- [ ] Guide 1: install, `loudkit download`, the two-line synthesis, and the
      three one-liners in English, Spanish and French.
- [ ] Guide 2: `stream` and `synthesize`, and `previous_tokens` across
      two calls.
- [ ] Guide 3: `lk.enroll` from a ten-second recording, then synthesize with
      the profile it wrote.
- [ ] Guide 4: `loudkit serve`, one `/v1/synthesize` call, one stream, the
      OpenAI route, `loudkit serve --mcp` and `loudkit serve --grpc`.
- [ ] `python tools/bench.py --checkpoint loudreader/loudr-1 --voice joe`
      prints a row on this machine (see
      [docs/design/benchmarking.md](docs/design/benchmarking.md)).
- [ ] Guides 7 to 10: the TypeScript, Go, Rust and Swift quickstarts, from
      scratch projects, against the published packages and the public Hub
      release.

Record what failed. Fix a failure in its source, and do not move a published
tag. A fix to a distributed artefact needs a new version.

## 9. Pre-release wording

`tests/test_release_coherence.py` refuses a stable version while any marker
below remains in its file. `PRERELEASE_BANNERS` in that test and this table
are the same list, and a test checks that they agree. Remove a marker that
comes back in the release commit (§2), before the tag in §6. Line numbers
change, so search for the marker.

| file | grep for | what it is |
| --- | --- | --- |
| `README.md` | `packages are not yet` | the README pre-release note |
| `notebooks/loudkit_quickstart.ipynb` | `Pre-release.` | the Colab pre-release note |
| `docs/reference/troubleshooting.md` | `lands on PyPI with the 0.1.0 release` | the not-on-PyPI-yet paragraph |
| `site/scripts/sync-docs.mjs` | `banner:` | the site-wide banner written into every generated page |
| `site/src/handwritten/index.mdx` | `lk-banner` | the landing-page banner block |
| `site/src/handwritten/demo.mdx` | `banner:` | the demo page banner front matter |
| `site/src/handwritten/index.mdx` | `git = "https://github.com/loudreader/loudkit"` | the landing page's Rust tab; the release line is `loudkit = "0.1"` |
| `site/src/handwritten/index.mdx` | `branch: "main"` | the landing page's Swift tab; the release line is a `from:` version |
| `docs/guides/10-swift.md` | `branch: "main"` | guide 10's Swift dependency; the release line is a `from:` version |

The last three rows are install lines, not banners. An install line that names
a branch or a git URL installs from `main`, not from the release.

`site/src/content/docs/` is generated and gitignored. Do not edit the copies
there. Each build rewrites them from `docs/` and `site/src/handwritten/`.
