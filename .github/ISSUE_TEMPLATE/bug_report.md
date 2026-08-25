---
name: Bug report
about: Something in loudkit does not behave as promised
title: ""
labels: bug
assignees: ""
---

**What I expected and what happened.** The promise is "same text, same voice,
same seed → same speech tokens on every backend, bit-identical waveform per
build". If your report is about determinism, name the two runs that differed.

**Reproduction.** The smallest text + voice + seed that triggers it, and the
exact command:

```bash
loudkit speak --checkpoint … --voice … --seed 7 "…"
```

**Environment.**
- loudkit version (or commit):
- device: `cpu` / `cuda` / `mps` / `coreml` / `onnx` (with index if multi-GPU)
- execution flags, if any (`--cuda-graphs`, precision overrides, …)
- OS, Python/Swift/Node/Go/Rust versions — whichever apply to the port
  you're using

**Which contract does this touch?** Determinism (bit-identical per build),
the sampling law (same seed → same tokens), conformance (the shared fixture),
or provenance (voices)? If you are not sure, say so — that is a useful answer.

**Is the conformance fixture affected?** If this changes tokens, mel, or
waveforms for a given seed, it re-bases goldens and needs the full parity gate,
not just a local fix.
