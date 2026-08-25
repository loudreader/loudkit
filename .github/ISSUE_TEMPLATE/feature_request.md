---
name: Feature request
about: An idea for loudkit
title: ""
labels: enhancement
assignees: ""
---

**What it does, not how.** Describe the behaviour a user gets, not the API you
imagine for it. The project separates *what is computed* (`AlgorithmConfig`)
from *how fast it runs* (`ExecutionConfig`) — say which side your request is on.

**Which backends does it need to reach?** If it touches the algorithm layer
(sampler, RNG, frontend, windowing), it must land in Python, Swift, JS, Go and
Rust together, because the conformance fixture is the law. If it is execution
only, one backend is enough.

**What is the measurable acceptance test?** E.g. "RTF on a 3090 with
`--cuda-graphs`", "mel corr ≥ 0.999 vs the eager path", "token-identical to
the eager path for the benchmark texts". An idea without a gate is a wish.

**Trade-offs you accept.** Faster often means a different reduction order
(the identity contract's `equivalent` class). Say whether that is acceptable
or whether this must stay bit-identical.
