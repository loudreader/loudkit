# 6. Embedding loudkit

loudkit has three integration paths: a Python API, a CLI, and a server/MCP.
This guide covers the API path, loudkit inside your own program.

## Load once, keep it

The synthesis checkpoint is 747 MB and takes a few seconds to load. `lk.load` returns a
warm `Engine`. Keep it for the life of the process and synthesise against it.
Do not build an engine per utterance.

```python
import loudkit as lk

engine = lk.load("loudr-1.safetensors")
```

`load` picks the best device for this machine: CUDA, then Apple silicon, then
CPU. On Apple silicon it builds a *split* engine, token generator on the CPU
and renderer on the GPU, because the two stages want different hardware.

## The five calls you will use

```python
voice = lk.VoiceProfile.load("voices/joe.safetensors")

# one window
engine.synthesize("Hello.", voice, seed=7).save("a.wav")

# a passage, one waveform; peak memory holds the whole render
engine.synthesize_long(passage, voice, seed=7).save("book.wav")

# a stream: first audio before the passage finishes
for result in engine.stream(passage, voice, seed=7):
    play(result.audio)  # first sentence immediately

# render tokens you already have (inspect intermediate tokens)
engine.synthesize_tokens(tokens, voice, seed=7)

# the line to log on every run
engine.describe()
```

Every call takes an optional `seed`. The three that take text also take an
optional `language`; `synthesize_tokens` has no text to read, and `describe`
takes neither. Same text, voice and seed give the same audio on this build.

**Omit `language` and the voice decides.** A profile records the language it
was enrolled from, so a Polish voice reads Polish with no argument. The chain
is the argument, then `voice.language`, then `"en"`. See
[which language this layer runs as](../reference/preprocess.md#which-language-this-layer-runs-as).
Pass `language` only for **cross-lingual** synthesis, such as an English voice
reading Polish text:

```python
engine.synthesize("Cześć.", polish_voice, seed=7)  # Polish
engine.synthesize("Cześć.", english_voice, seed=7, language="pl")  # also Polish
```

## What it raises, and what that means

Every error loudkit raises is a `loudkit.LoudkitError` **and** the builtin its
raise site used before. So `except ValueError` still catches an over-window
refusal, `except FileNotFoundError` still catches a missing voice, and
`except loudkit.LoudkitError` catches only what loudkit itself refused.

| class | builtin base | means |
|---|---|---|
| `UnsupportedLanguageError` | `NotImplementedError` | a language off the twelve-id roster; carries `.language` and `.supported` (the roster itself, so a caller can retry into something that works) |
| `VoiceNotFoundError` | `FileNotFoundError` | no voice by that name or path; carries `.ref` and, where listing is cheap, `.available` |
| `WindowOverflowError` | `ValueError` | one window's speech exceeded the render window; carries `.n_tokens` and `.window`. Use `synthesize_long` |
| `NumberGrammarError` | `ValueError` | a number could not be said in that language |
| `InvalidTokensError` | `ValueError` | a speech token id outside the codebook, or an empty sequence; carries `.token` and `.limit` |
| `NothingToSpeakError` | `ValueError` | the text funnel removed every character of the request |
| `ProvenanceError` | `ValueError` | a C2PA manifest is present and cannot be read |

The split matters at a boundary that has to classify. The HTTP server answers
`400` for `UnsupportedLanguageError` and `500` for any other
`NotImplementedError`. The first is a fault in the request; the second is a
stub in a backend. While both were the same builtin, every backend defect
reached the client labelled as the caller's mistake.

An exception out of loudkit that is **not** a `LoudkitError` is a bug here or a
failure in a dependency. Report it rather than handle it.

## Picking devices yourself

Override the split when your hardware disagrees:

```python
from loudkit.config import ExecutionOverrides

execution = ExecutionOverrides(
    device="cuda",
    generator_device="cuda",  # an autoregressive step likes a fast kernel
    renderer_device="cuda",
)
engine = lk.load(ckpt, execution=execution)
```

On Apple silicon the defaults are the measured optimum. The knobs exist for
hardware where they are not.

`ExecutionOverrides` is a **patch**. The fields you name win; everything else
keeps the checkpoint manifest's shipping values, including its fp16 precision
map, which is what the published benchmarks were measured in. `precision`
merges per module, so `ExecutionOverrides(precision={"vocoder": "fp32"})`
changes that one module and leaves the others alone.

Pass a full `ExecutionConfig` when you mean exactly this and nothing inherited,
for example to replay a recorded configuration. The two are separate types
because "I did not say" and "I said the default value" are different requests.
With one type, a run that asked for an all-fp32 map was indistinguishable from
one that asked for nothing, and quietly measured the manifest's fp16.

**No torch at all.** `device="onnx"` runs every stage as fp32 ONNX graphs on
onnxruntime, so a `loudkit[onnx]` deployment synthesises with no torch in the
process. The execution provider defaults to `auto`, which takes CUDA where the
machine offers it and CPU otherwise. Name `onnx_provider` on an
`ExecutionOverrides` to pin one. `tools/export_onnx.py` exports the graphs once. Each
graph is gated against the torch module it came from, and the whole generator
is held to teacher-forced top-1 >= 99.5% and median KL < 1e-3, measured at
1.00000 and ~1e-7. fp16 is not exported (measured: not worth a second
artifact). int8 stays blocked.

```python
engine = lk.load(ckpt, device="onnx")
```

## Embedding in a larger torch program

**Building a loudkit engine on torch mutates process-global torch state.**
Engine construction pins determinism by setting `cudnn.deterministic`,
`benchmark` and the TF32 flags, and a second engine does not restore what the
first changed. An engine running under flags it did not set cannot honour the
identity contract. A host that does its own torch work must account for this.
See `pin_determinism` in `backends/torch_backend.py`.

## The other embeddings

- **A CLI** for batch and scripting: `loudkit speak --checkpoint ... --voice ... "text"`.
- **A server** for anything that is not Python: `POST /v1/synthesize` for WAV,
  or `POST /v1/synthesize/stream` for SSE (guide 4).
- **MCP** for any MCP-aware agent: `synthesize(text, voice, seed)`.
- **Swift** for Apple targets: the `LoudKit` package at the repo root drives the
  same engine through CoreML; see `docs/platforms/apple.md`.

All of them call the same engine. There is no second synthesis path.
