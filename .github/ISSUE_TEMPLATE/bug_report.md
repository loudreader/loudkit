---
name: Bug report
about: Something in loudkit does not behave as documented
title: ""
labels: bug
assignees: ""
---

**What I expected and what happened.** Describe both. For a determinism
report, name the two runs that differed. The
[identity contract](https://github.com/loudreader/loudkit/blob/main/docs/reference/IDENTITY-CONTRACT.md)
states which runs must match: a waveform is bit-identical only on the same
build, device, backend and execution settings, and speech tokens can differ
across backends.

**Reproduction.** The smallest text, voice and seed that trigger it, and the
exact command or API call:

```bash
loudkit speak --checkpoint … --voice … --seed 7 "…"
```

**Environment.**
- loudkit version (or commit):
- device: `cpu` / `cuda` / `mps` / `coreml` / `onnx` (with index if multi-GPU)
- execution settings, if any (CUDA graphs, precision overrides, ...), and how
  you set them
- OS, and the Python, Swift, Node, Go or Rust version for the SDK you use

**Which contract does this touch?** If you know, name it: determinism
(bit-identical per build), sampling (the same logits and seed give the same
tokens), conformance (the shared fixture), or voice provenance. If you are not
sure, say so.

**Evidence.** If you have it, include the failing conformance case, or the
token, mel or waveform difference you observed. The maintainers decide whether
a fix restores the existing golden files or needs a documented re-baseline.
