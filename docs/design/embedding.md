# Embedding loudkit

loudkit has three integration paths: a Python API, a CLI, and a server/MCP.
This page covers the API path, loudkit inside your own program, past what
[getting started](../guides/01-getting-started.md) shows.

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
engine.synthesize(passage, voice, seed=7).save("book.wav")

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
[which language this layer runs as](preprocess.md#which-language-this-layer-runs-as).
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
| `WindowOverflowError` | `ValueError` | one window's speech exceeded the render window; carries `.n_tokens` and `.window`. Only `synthesize(single_window=True)` and `synthesize_tokens` raise it |
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
from loudkit.execution import ExecutionConfig

execution = ExecutionConfig(
    device="cuda",
    generator_device="cuda",  # an autoregressive step likes a fast kernel
    renderer_device="cuda",
)
engine = lk.load(ckpt, execution=execution)
```

On Apple silicon the defaults are the measured optimum. The knobs exist for
hardware where they are not.

A field left `None` inherits the checkpoint manifest's shipping value,
including its precision map, which is what the published benchmarks were
measured in. Name only what you mean to change.

**No torch at all.** `device="onnx"` runs every stage as fp32 ONNX graphs on
onnxruntime, so a `loudkit[onnx]` deployment synthesises with no torch in the
process. The execution provider defaults to `auto`, which takes CUDA where the
machine offers it and CPU otherwise. Name `onnx_provider` on an
`ExecutionConfig` to pin one. `tools/export_onnx.py` exports the graphs once. Each
graph is gated against the torch module it came from, and the whole generator
is held to teacher-forced top-1 >= 99.5% and median KL < 1e-3, measured at
1.00000 and ~1e-7. fp16 is not exported (measured: not worth a second
artifact). int8 stays blocked.

```python
engine = lk.load(ckpt, device="onnx")
```

## One engine, one call at a time

**An engine is not a thread pool.** Keeping it for the life of the process is
right; calling it from several threads at once is not. Under `cuda_graphs` or
`compile_model` the KV cache, the decode positions and the sampler's noise are
per-generator buffers that a captured graph holds *by address*, so a second
concurrent `generate` writes into the first one's state and both callers get
fluent speech that is not their text. That case is refused outright, with a
message saying so, rather than served.

The eager path allocates per call and does not have that failure, but the
stages are still shared and there is no benefit in racing them: one forward
pass already saturates the device.

Every shipped transport serialises. The HTTP server holds a single engine slot
across both routes, and gRPC does the same. If you embed the library, do one of
the two things they do:

```python
import threading

engine = lk.load("loudr-1.safetensors")
lock = threading.Lock()

def speak(text, voice, seed):
    with lock:
        return engine.synthesize(text, voice, seed=seed)
```

or build one engine per worker thread, which costs a second copy of the
weights.

**A long call can be stopped.** `synthesize` and `stream` both take
`should_cancel`, polled on every decode step; return `True` from it and the
call stops within one step rather than at the next chunk boundary:
`synthesize` raises `CancelledError`, `stream` ends after the chunks it
already yielded. That is what a request whose client has gone away should do,
and what the transports do with it.

## Embedding in a larger torch program

**Building a loudkit engine on torch mutates process-global torch state.**
Engine construction pins determinism by setting `cudnn.deterministic`,
`benchmark`, the TF32 flags and, when it is set, `num_threads`; a second engine
does not restore what the first changed, and building one with different flags
warns rather than changing them in silence. An engine running under flags it
did not set cannot honour the identity contract. A host that does its own torch
work must account for this. See `pin_determinism` in
`backends/torch_backend.py`.

## The two stages

The engine has two stages. A **token generator** writes discrete speech tokens
at 25 Hz, autoregressively. A **renderer** turns those tokens into a waveform in
one parallel pass. Their shapes are opposite, so on Apple silicon they do not
share a device: the generator is faster on the CPU, the renderer on the GPU.
`Engine.describe()` prints what was active, one line, and it is the line to log
on every run. The first token of it is the algorithm fingerprint, a hash of
every audible decision; it moves whenever any of them moves, so a value copied
into prose is wrong by the next release. `/health` and `describe()` report the
same value.

## The other embeddings

- **A CLI** for batch and scripting: `loudkit speak --checkpoint ... --voice ... "text"`.
- **A server** for anything that is not Python: `POST /v1/synthesize` for WAV,
  or `POST /v1/synthesize/stream` for SSE (guide 4).
- **MCP** for any MCP-aware agent: `synthesize(text, voice, seed)`.
- **Swift** for Apple targets: the `LoudKit` package at the repo root drives the
  same engine through CoreML; see `docs/platforms/apple.md`.

All of them call the same engine. There is no second synthesis path.

## Prompt-ending measurement

Append silence after the last usable pause in the ten-second prompt.

Prefer the last pause of at least 60 ms after the first three seconds.
A pause near ten seconds leaves only part of the 0.4-second pad inside
the prompt; keep it rather than cutting an earlier word to fit the pad.
Without a pause, truncate at 9.6 seconds if needed to leave room for silence.
`enroll(..., end_in_silence=True)` asks for this cut; the default leaves the
recording as it is.

In the enrollment experiment recorded at 92baeed, ending the prompt in
silence moved speech onset from 0 to 50 ms and the first 60 ms from
-32 to -51 dBFS on one cloned voice. The shipped voices measured 70-120 ms
of initial silence. These are observations, not a guarantee for other
recordings or evidence that 0.4 seconds is better than a shorter pause.
