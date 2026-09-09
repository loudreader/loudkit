# Choosing a model

Start with **loudr-1** for reference quality. Try **loudr-1-turbo** when you
want audio sooner. Compare them on your own text and voice before deciding:
they produce different readings, even with the same seed.

Switch by changing the name. Keep the rest of your code and your voice profile:

```python
import loudkit as lk

engine = lk.load("loudreader/loudr-1")
# Or: engine = lk.load("loudreader/loudr-1-turbo")
voice = engine.voice("joe")
engine.synthesize("Hello from loudkit.", voice, seed=7).save("hello.wav")
```

## Where they run

Both models are supported by the 0.1.1 code:

| Your environment | Runtime | loudr-1 | loudr-1-turbo |
|---|---|---|---|
| Python on CPU or GPU | PyTorch | yes | yes |
| Python without PyTorch | ONNX Runtime | yes | yes |
| Python on a Mac | CoreML | yes | yes |
| Swift on macOS or iOS | native generator + CoreML renderer | yes | yes |
| Go, Rust, JavaScript / TypeScript | ONNX Runtime | yes | yes |

Use a release containing the graphs for your runtime. See
[troubleshooting](../reference/troubleshooting.md) if a download or model
load fails.

## What changes

Turbo generates pairs of speech tokens and uses one renderer estimation step.
It does less model work, but the speedup depends on the device and passage.
[Benchmarks](../benchmarks.md) distinguishes measured results from expectations.

Listen for naturalness, pauses and pronunciation on the material you plan to
read. A faster result is useful only if you like listening to it.

## What stays the same

The API, available voices, text preparation, streaming and voice-profile format
are shared. Clone a voice once and reuse its file with either model. Loading
an existing profile does not load the enrollment models.

A seed makes a reading repeatable within the same model and execution setup;
it does not make the two models produce identical audio.

## Next

[Getting started](01-getting-started.md) · [Clone a voice](03-cloning-a-voice.md) ·
[loudr-1 model card](../MODEL_CARD.md) · [Turbo model card](../MODEL_CARD-turbo.md)
