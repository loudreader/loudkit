# Compare the five decode loops

The library has five implementations of the same decode: `python/loudkit/models/generator.py`
(and `backends/onnx_backend.py`), `go/engine/engine.go`, `rust/src/engine.rs`,
`js/src/engine.ts` and `swift/LoudKit/TokenGenerator.swift`. They are read often
and run side by side rarely, because the ONNX Runtime each binding wants is not
the same build. This harness runs them on one input and compares the stages in
the order a divergence has to be localised in.

Each probe writes one JSON record and the raw float32 arrays beside it:

1. `text_tokens` -- what the funnel handed the generator
2. `prefill` / `cond` -- the embedding rows (Python and JS; unexported elsewhere)
3. `speech_tokens_raw` and `speech_tokens` -- the decode, before and after the
   stop marker is stripped
4. `mel` -- the flow decoder's frames
5. `audio` -- the vocoder's samples
6. `longform` -- the whole `synthesize` path: chunking, the prefix carry, the
   retry ladder

`compare.py` stops at the first stage that disagrees, because everything after
it is a consequence rather than a second finding.

## Running

```sh
tools/decode_probe/all.sh TEXT SEED LANGUAGE VOICE OUTDIR
```

`env.sh` holds the two ONNX Runtime paths this laptop needs. They are not
interchangeable: the Rust `ort` build loads only 1.27.x and the Go binding wants
API 28, which is 1.29. JS ships its own. Swift runs CoreML and reads `coreml/`
out of the same bundle.

Every probe prints the module or binary it actually loaded before it measures.

## What it is not

The four ONNX ports here run **three different ONNX Runtime builds**, so a
sample-level difference between them is not attributable to port code. Tokens
are the comparison that survives that; the identity contract
(`docs/reference/IDENTITY-CONTRACT.md`) declines to promise waveform identity
across backends anyway.

## The other two probes

- `behaviour.py` -- determinism, the two-thread split in `python/loudkit/stream.py`,
  and the cancellation poll, none of which a single decode exercises.
- `cancel_py.py` and `cancel.go` -- the same two callback shapes in two ports,
  for measuring what each does with a partial row.
