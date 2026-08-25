# Contributing

Thanks for wanting to work on loudkit. Read `README.md` and
[`docs/reference/IDENTITY-CONTRACT.md`](docs/reference/IDENTITY-CONTRACT.md)
first. The contract is the one document the rest of the code answers to. The
open issues are the authority on what still needs doing.

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
* Go: `cd go && gofmt -l . && go test ./...`
* Rust: `cd rust && cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test`
* Swift: `swift build && swift test --filter Conformance`

## Adding a backend

Backends register themselves rather than being wired into `build_engine`. Add
one by implementing the backend protocol and registering it. The existing
backends in `python/loudkit/backends/` show the shape. Open an issue first if
the backend needs a change to the protocol itself.

## Where things live

The package is layered. Dependencies point downwards only, and
`tests/test_import_graph.py` fails on an undeclared edge:

```
errors · rng · timing · contracts   foundation
frontend/                           the text funnel (its own package)
config · models/                    configuration; signal & network modules
sampler · backends/                 sampling; execution backends
engine                              orchestration: one synthesis path
synthesis                           render_bytes: where audio is made
transports/                         http · mcp · grpc: peers, never layered
cli                                 argument parsing over all of it
```

`docs/reference/ARCHITECTURE.md` maps every concept to its file in each of
the five implementations. If your change adds an import edge, add that edge to
the graph test's allowlist in the same commit.

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
