**What this changes, and its contract class.** Describe the change and what a
user sees. If it changes numerical output, name its identity-contract class:
`bit-exact` (same arithmetic, same order), `equivalent` (deterministic,
different reduction order) or `changes-maths` (different numerics). If a golden
waveform or a sampled token moves, the change is a re-baseline under a bumped
contract version. Say so.

**Conformance.** If this touches the algorithm layer (sampler, RNG, frontend,
windowing, the Polish pipeline), update every port in the same change. The
weight-free conformance vectors must pass in every language.

**Verification.**
- [ ] `just check` (ruff, mypy strict, formatting)
- [ ] `just test` (Python, weight-free)
- [ ] JS: `cd js && npm run lint && npm test`
- [ ] Go: `cd go && test -z "$(gofmt -l .)" && go vet ./... && go test ./...`
- [ ] Rust: `cd rust && cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test`
- [ ] Swift: `swiftlint lint --quiet && swift build && swift test`
- [ ] Parity (if weights are available): `LOUDKIT_REQUIRE_ASSETS=1 pytest tests/test_parity.py tests/test_conformance.py`

**Not in this PR.** A pull request does not publish weights or packages.
Maintainers publish releases through the process in `RELEASING.md`.
