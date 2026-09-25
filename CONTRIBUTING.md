# Contributing

Thanks for working on loudkit. Read `README.md` and
[`docs/reference/IDENTITY-CONTRACT.md`](docs/reference/IDENTITY-CONTRACT.md)
first. All code must conform to the identity contract. Check the open issues
for planned work and existing reports.

`just setup` uses [uv](https://docs.astral.sh/uv/) to manage the Python
environment. Install uv before the first run.

## Ground rules

Tests enforce the identity contract
([`docs/reference/IDENTITY-CONTRACT.md`](docs/reference/IDENTITY-CONTRACT.md)).

* Never weaken a determinism or parity gate: `mel corr >= 0.999`,
  bit-identical output per build and device, and exact free-run tokens against
  the conformance fixture. A change that moves a golden output is a
  re-baselining event under a bumped contract version.
* Every implementation must pass the conformance fixture. Python, Swift, JS, Go
  and Rust must all pass `tests/data/conformance/vectors.json` weight-free, and
  match the end-to-end fixture when weights are present. If you change the
  sampler, the RNG, the frontend or windowing, change the fixture too, and
  update every port in the same change.
* `AlgorithmConfig` holds what is computed, and every backend shares it.
  `ExecutionConfig` holds how fast it runs, per backend. Put execution tuning in
  `ExecutionConfig`, never in the algorithm layer.
* int8 stays blocked: the tested int8 path fails the release quality gate.
  Enabling it requires a full measurement and listening pass.
* Pull requests never publish. Maintainers publish weights and packages through
  the process in [RELEASING.md](RELEASING.md).

## Before you start

1. `just setup` installs the dev extras into `.venv`.
2. `just check` runs ruff, mypy strict and formatting.
3. `just test` runs the weight-free suite.

A change is done when `just check && just test` pass, and when every language
touched by the change runs its own suite, as CI does:

* Python: `just check && just test`
* JS/TS: `cd js && npm run lint && npm test`
* Go: `cd go && test -z "$(gofmt -l .)" && go vet ./... && go test ./...`
  (`gofmt -l` only lists unformatted files; `test -z` makes that list fail the
  check)
* Rust: `cd rust && cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test`
* Swift: `swiftlint lint --quiet && swift build && swift test` (the whole suite;
  a `--filter` leaves tests out)

## Adding a backend

Each backend registers itself with `register_backend`, so `build_engine` needs
no change. To add one, implement the backend protocol and register it. The
backends in `python/loudkit/backends/` are examples. Open an issue first if the
backend needs a change to the protocol itself.

## Where things live

The package is layered, dependencies point downwards only, and
`tests/test_import_graph.py` fails on an undeclared edge. The layer diagram is
in [`docs/design/ARCHITECTURE.md`](docs/design/ARCHITECTURE.md), which also
maps the main concepts to their files in each of the five implementations.

If your change adds an import edge, add that edge to the graph test's allowlist
in the same commit.

## Comments

Write comments in this style:

* Explain why. The code already says what it does, so do not restate it.
* Name the failure that a non-obvious line prevents, and where the reader can
  see it: a measurement, a device trace, a fixture case or an issue.
* Give each constant the measurement it comes from. "Measured 0.53-0.64 across
  three languages" can be reviewed; "measured to be safe" cannot.
* When a claim stops being true, correct the comment in the same commit as the
  code, in every port that carries it.
* State each fact in one file, and link to it from the others.

## Provenance

Every shipped voice is an enrollment of a recording made or released for
speech-technology use. `docs/voices/roster/provenance.json` records its donor
or source, licence and consent basis. Add a voice or asset only when its
source, licence and consent basis are documented. `NOTICE` must name every
third-party component.

## Tests

Weight-free tests run without a checkpoint and gate every pull request.
Weighted tests (`pytest -m slow`, the engine-conformance targets, and the
weighted part of `swift test`) need the release assets. They run in the CI
parity job. Locally they skip when the assets are missing, unless
`LOUDKIT_REQUIRE_ASSETS=1` is set.

## Licensing

By contributing you agree that your work is Apache-2.0 (see `LICENSE`), and that
it may be redistributed under the same terms.

## Local runtime checks

Test source with the project virtual environment and `PYTHONPATH=python`.
Without it, an installed wheel elsewhere can hide your edits. Run a clean-wheel
check outside the checkout, without that override.

Always pair graphs with the exact checkpoint from one verified bundle. Packed
and synthesis-only files have different file digests even when synthesis tensors
match. Swift tests that need weights, such as `EndToEndConformanceTests` and
`EnrollmentTests`, skip without them. To run them, use the release
configuration and set `LOUDKIT_ASSET_ROOT` (or `LOUDKIT_CHECKPOINT`),
`LOUDKIT_COREML_ASSETS`, `LOUDKIT_FUSION_CHECKPOINT` and
`LOUDKIT_FUSION_COREML_ASSETS`, the variables that name the loudr-1 and
loudr-1-turbo assets.
