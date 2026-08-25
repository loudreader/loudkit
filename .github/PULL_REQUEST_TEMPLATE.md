---
name: Pull request
about: A change to loudkit
title: ""
labels: ""
assignees: ""
---

**What this changes, and the contract class.** Per the identity contract,
every change is one of: `bit-exact` (same arithmetic, same order), `equivalent`
(deterministic, different reduction order), or `changes-maths` (different
numerics). If a golden waveform or a sampled token moves, that is a re-baseline
event under a bumped contract version — say so.

**Conformance.** If this touches the algorithm layer (sampler, RNG, frontend,
windowing, the Polish pipeline), the shared fixture is the law: every language
port must be updated in the same change, and the weight-free vectors must pass
everywhere.

**Verification.**
- [ ] `just check` (ruff, mypy strict, formatting)
- [ ] `just test` (Python, weight-free)
- [ ] JS: `cd js && npm test`
- [ ] Go: `cd go && gofmt -l . && go test ./...`
- [ ] Rust: `cd rust && cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test`
- [ ] Swift: `swift build && swift test --filter Conformance`
- [ ] Parity (if weights are available): `LOUDKIT_REQUIRE_ASSETS=1 pytest tests/test_parity.py tests/test_conformance.py`

**Not in this PR.** Weights, PyPI and HuggingFace releases are deliberate,
manual acts by the maintainers; no release automation runs from a PR.
