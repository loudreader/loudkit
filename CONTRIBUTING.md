# Contributing

Thanks for wanting to work on loudkit. Read `README.md` and
[`docs/reference/IDENTITY-CONTRACT.md`](docs/reference/IDENTITY-CONTRACT.md)
first. The contract is the one document the rest of the code answers to. The
open issues are the authority on what still needs doing.

The Python environment is managed with [uv](https://docs.astral.sh/uv/), which
`just setup` calls: install it before the first run.

## The ground rules

The identity contract
([`docs/reference/IDENTITY-CONTRACT.md`](docs/reference/IDENTITY-CONTRACT.md))
is what makes cross-backend determinism possible, and tests enforce it.

* **Never weaken a determinism or parity gate.** `mel corr >= 0.999`,
  bit-identical output per build/device, and exact free-run tokens against the
  conformance fixture. A change that moves a golden is a re-baselining event
  under a bumped contract version.
* **The conformance fixture is law.** Python, Swift, JS, Go and Rust must all
  pass `tests/data/conformance/vectors.json` weight-free, and match the
  end-to-end fixture when weights are present. If you change the sampler, the
  RNG, the frontend, or windowing, change the fixture too. Update every
  language's port in the same change.
* **AlgorithmConfig vs ExecutionConfig.** What is computed is fixed and shared.
  How fast it runs is per-backend. Do not smuggle a tuning knob into the
  algorithm layer.
* **int8 stays blocked.** The tested int8 path did not pass the release quality
  gate. Reconsidering it requires a full measurement and listening pass, not a
  configuration change.
* **No publishing from a PR.** Weights, PyPI and HuggingFace releases are a
  manual act by the maintainers.

## Before you start

1. `just setup` installs the dev extras into `.venv`.
2. `just check` runs ruff, mypy strict and formatting.
3. `just test` runs the weight-free suite.

A change is done when `just check && just test` pass, and when every language
touched by the change runs its own suite:

* Python: `just check && just test`
* JS/TS: `cd js && npm test`
* Go: `cd go && test -z "$(gofmt -l .)" && go vet ./... && go test ./...`
  (`gofmt -l` prints what is unformatted and exits 0, so it stops nothing on
  its own; `test -z` is what CI runs and what actually fails)
* Rust: `cd rust && cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test`
* Swift: `swift build && swift test` (the whole suite, as CI runs it: the
  conformance filter leaves five of its six tests skipped without weights, so
  it goes green having run one)

## Adding a backend

Backends register themselves rather than being wired into `build_engine`. Add
one by implementing the backend protocol and registering it. The existing
backends in `python/loudkit/backends/` show the shape. Open an issue first if
the backend needs a change to the protocol itself.

## Where things live

The package is layered, dependencies point downwards only, and
`tests/test_import_graph.py` fails on an undeclared edge. The layer diagram is
in [`docs/design/ARCHITECTURE.md`](docs/design/ARCHITECTURE.md), which
also maps every concept to its file in each of the five implementations.

There used to be a second, shorter copy of that diagram here. It named eight
layers where the real graph has nine, omitted `checkpoint`, `postprocess`,
`hub`, `voice` and `provenance` entirely, and drifted for as long as nothing
compared the two — which is the failure mode this project spends most of its
review budget on. One diagram, one place.

If your change adds an import edge, add that edge to the graph test's allowlist
in the same commit.

## Comments

The comments in this repository are unusual and deliberately so. Match them.

* **Say why, not what.** The code says what. A comment that restates it is
  noise that later becomes a lie.
* **Name the failure.** Most non-obvious lines here exist because something
  went wrong once. Say what went wrong, and where the reader can see it: a
  measurement, a device trace, a fixture case, an issue.
* **Carry the number.** "Measured 0.53-0.64 across three languages" is
  reviewable; "measured to be safe" is not. A constant without its measurement
  is a constant nobody can change safely.
* **Retract in place.** When a claim stops being true, correct the comment in
  the same commit as the code, in every port that carries it. A comment that
  outlives its code is worse than no comment: it is read as current.
* **Write it once.** A fact stated in two files is a fact that will disagree
  with itself. Put it where it belongs and point at it from the other place.

## Provenance

Every shipped voice is an enrollment of a recording made or released for
speech-technology use, with donor or source, licence and consent basis recorded
in `docs/voices/roster/provenance.json`. Do not add voices or assets whose
provenance is not clean and documented. `NOTICE` must name every third-party
component.

## Tests

Weight-free tests run without a checkpoint and are the PR gate. Weighted tests
(`pytest -m slow`, the engine-conformance targets, `swift test`) need the local
packed checkpoint and are the parity job's business.

## Licensing

By contributing you agree that your work is Apache-2.0 (see `LICENSE`), and that
it may be redistributed under the same terms.

## Local runtime checks

Use the project virtual environment and `PYTHONPATH=python` when testing source;
an installed wheel elsewhere can otherwise hide your edits. A clean wheel
smoke must instead run outside the checkout without that override.

Always pair graphs with the exact checkpoint from one verified bundle. Packed
and synthesis-only files have different file digests even when synthesis tensors
match. Swift weighted tests are `EndToEndConformanceTests` and `EnrollmentTests`;
use release configuration and explicitly set both base and fusion asset paths.
