---
name: Feature request
about: An idea for loudkit
title: ""
labels: enhancement
assignees: ""
---

**What it does.** Describe the user need and the behaviour you want. Include an
API proposal if it helps. If you know, say whether the request changes what is
computed (`AlgorithmConfig`) or only how fast it runs (`ExecutionConfig`).

**Which backends does it need to reach?** A change to the algorithm layer
(sampler, RNG, frontend, windowing) must land in Python, Swift, JS, Go and Rust
together, with the conformance fixture updated. An execution-only change can
target one backend.

**What is the measurable acceptance test?** For example: "RTF on a 3090 with
CUDA graphs", "mel corr ≥ 0.999 vs the eager path", or "token-identical to the
eager path for the benchmark texts".

**Trade-offs you accept.** Say whether the result must stay bit-identical, or
may differ within the identity contract's `equivalent` class (deterministic,
different reduction order).
