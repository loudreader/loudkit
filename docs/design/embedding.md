# Embedding loudkit

loudkit has three integration paths: a Python API, a CLI, and a server or MCP.
The notes below cover the Python API inside your own program, past what
[getting started](../guides/01-getting-started.md) shows.

## Load once, keep it

The synthesis checkpoint is 747 MB and takes a few seconds to load. `lk.load`
returns an `Engine` that is ready to synthesize; it does not run a warm-up
render, so the first call can take longer than later ones. Keep the engine for
the life of the process and synthesize with it. Do not build an engine per
utterance.

```python
import loudkit as lk

engine = lk.load("loudr-1.safetensors")
```

`load` picks the best device for this machine: CUDA, then Apple silicon, then
CPU. On Apple silicon it builds a *split* engine: the token generator on the
CPU and the renderer on the GPU (see [the two stages](#the-two-stages)).

## The calls you will use

```python
voice = lk.VoiceProfile.load("voices/joe.safetensors")

# one window
engine.synthesize("Hello.", voice, seed=7).save("a.wav")

# a passage, one waveform; peak memory holds the whole render
engine.synthesize(passage, voice, seed=7).save("book.wav")

# a stream: first audio before the passage finishes
for result in engine.stream(passage, voice, seed=7):
    play(result.audio)  # play each chunk as it arrives

# render tokens you already have (inspect intermediate tokens)
engine.synthesize_tokens(tokens, voice, seed=7)

# the line to log on every run
engine.describe()
```

`synthesize`, `stream` and `synthesize_tokens` take an optional `seed`. The
two that take text also take an optional `language`; `synthesize_tokens` has
no text to read, and `describe` takes neither. The same text, voice and seed
give the same audio on the same build, device and execution config.

**Language resolution.** The engine uses the `language` argument, then
`voice.language`, then `"en"`. A profile records the language it was enrolled
from, so a Polish voice reads Polish with no argument. Pass `language` to
override the profile, for example for **cross-lingual** synthesis with an
English voice reading Polish text. See
[which language this layer runs as](preprocess.md#which-language-this-layer-runs-as).

```python
engine.synthesize("Cześć.", polish_voice, seed=7)  # Polish
engine.synthesize("Cześć.", english_voice, seed=7, language="pl")  # also Polish
```

## What it raises, and what that means

Every class in the table subclasses `loudkit.LoudkitError`. All of them except
`CancelledError` also subclass a builtin. So `except ValueError` catches an
over-window refusal, `except FileNotFoundError` catches a missing voice, and
`except loudkit.LoudkitError` catches only what loudkit refused.

| class | builtin base | means |
|---|---|---|
| `UnsupportedLanguageError` | `NotImplementedError` | a language off the twelve-id roster; carries `.language` and `.supported`, the roster a caller can retry with |
| `VoiceNotFoundError` | `FileNotFoundError` | no voice by that name or path; carries `.ref` and, where listing is cheap, `.available` |
| `AudioNotFoundError` | `FileNotFoundError` | the recording `lk.enroll` was asked to read does not exist |
| `WindowOverflowError` | `ValueError` | one window's speech exceeded the render window; carries `.n_tokens` and `.window`. Only `synthesize(single_window=True)` and `synthesize_tokens` raise it |
| `NumberGrammarError` | `ValueError` | the number grammar cannot read a number in that language |
| `InvalidTokensError` | `ValueError` | a speech token id outside the codebook, or an empty sequence; carries `.token` and `.limit` |
| `NothingToSpeakError` | `ValueError` | the text funnel removed every character of the request |
| `UnsupportedFormatError` | `ValueError` | an audio format the loaded libsndfile was built without, such as `mp3` or `opus` on some builds; refused before the engine runs |
| `ProvenanceError` | `ValueError` | a provenance manifest is present and cannot be read |
| `CancelledError` | none | `should_cancel` returned true during `synthesize`; `stream` does not raise it |

Argument checks outside this table raise plain builtins. An out-of-range
`speed` or an unknown `ExecutionConfig` value raises `ValueError`.

The builtin bases matter at a boundary that has to classify. The HTTP server
answers `400` for `UnsupportedLanguageError` and `500` for any other
`NotImplementedError`: the first is a fault in the request, the second is a
stub in a backend.

Any other exception is a bug in loudkit or a failure in a dependency. Report
it with the `engine.describe()` line.

## Picking devices yourself

Override the split when a measurement on your hardware favours another
placement:

```python
from loudkit.execution import ExecutionConfig

execution = ExecutionConfig(
    device="cuda",
    generator_device="cuda",  # token generator
    renderer_device="cuda",
)
engine = lk.load(ckpt, execution=execution)
```

On Apple silicon the defaults are the fastest placement measured for the
published benchmarks.

A field left `None` takes the backend's default. On torch that is the
checkpoint manifest's shipping value, including its precision map, which is
what the published benchmarks were measured with. Name only what you mean to
change.

**Without torch.** `device="onnx"` runs every stage as fp32 ONNX graphs on
onnxruntime, so a `loudkit[onnx]` deployment synthesizes with no torch in the
process. The execution provider defaults to `auto`, which takes CUDA where
onnxruntime offers it and CPU otherwise. Set `onnx_provider` on an
`ExecutionConfig` to pin one; an explicit provider that is not available
raises and does not fall back.

The release ships the graphs. `tools/export_onnx.py` builds them from a
checkpoint, and it needs `torch` and the `onnx` package as well as
onnxruntime. Each graph is checked against the torch module it came from. The
generator graphs are kept only when they reproduce the torch generator's
tokens exactly on the conformance vectors. `tests/test_onnx.py` also holds the
generator to teacher-forced top-1 >= 99.5% and median KL < 1e-3, measured at
1.00000 and about 1e-7. fp16 graphs are not exported, because measured, they
do not justify a second artifact. int8 is not exported.

```python
engine = lk.load(ckpt, device="onnx")
```

## One engine, one call at a time

Call an engine from one thread at a time. Under `cuda_graphs` or
`compile_model`, the KV cache, the decode positions and the sampler's noise
are per-generator buffers that a captured graph holds *by address*. A second
concurrent `generate` would write into the first one's state, and both
callers would get speech that does not match their text. loudkit refuses that
case with an error that says so.

The eager path allocates its decode state per call and does not have that
failure. The stages are still shared, so serialize calls there too.

Every transport serializes synthesis: the HTTP server holds one engine slot
across both routes, and gRPC and MCP each hold a lock. If you embed the
library, do the same:

```python
import threading

engine = lk.load("loudr-1.safetensors")
lock = threading.Lock()

def speak(text, voice, seed):
    with lock:
        return engine.synthesize(text, voice, seed=seed)
```

Or build one engine per worker thread, which holds a second copy of the
weights.

**A long call can be stopped.** `synthesize` and `stream` both take
`should_cancel`. The decode loop polls it on every step, so the call stops at
the next decode step, not at the next chunk boundary. A render stage that is
already running finishes first. `synthesize` raises `CancelledError`, and
`stream` ends after the chunks it already yielded. The transports use this
when a client goes away (see [barge-in](barge-in.md)).

## Embedding in a larger torch program

**Building a loudkit engine on torch changes process-global torch state.**
Engine construction pins determinism by setting `cudnn.deterministic`,
`benchmark`, the TF32 flags and, when it is set, `num_threads`. A second
engine built with different flags re-pins them for both engines and emits a
`RuntimeWarning`; the first engine's `describe()` then no longer matches what
it runs. The identity contract does not hold for an engine that runs under
flags it did not set. A host that does its own torch work must account for
this. See `pin_determinism` in `backends/torch_backend.py`.

## The two stages

The engine has two stages. A **token generator** writes discrete speech tokens
at 25 Hz, autoregressively, one small step at a time. A **renderer** turns
those tokens into a waveform with a mel encoder, the flow estimator's Euler
steps and a vocoder, each over the whole window at once. Because the shapes
differ, the Apple silicon default puts the generator on the CPU and the
renderer on the GPU.

`Engine.describe()` prints what was active on one line; log it on every run.
Its first token is the algorithm fingerprint, a hash of every audible
decision. It moves when an audible decision changes
(`docs/reference/COMPATIBILITY.md` lists the exceptions), so do not copy a
value into prose. `/health` and `describe()` report the same value.

## The other embeddings

- **A CLI** for batch and scripting: `loudkit speak --checkpoint ... --voice ... "text"`.
- **A server** for anything that is not Python: `POST /v1/synthesize` for WAV,
  or `POST /v1/synthesize/stream` for SSE (guide 4).
- **MCP** for any MCP-aware agent: `synthesize(text, voice, seed)`.
- **Swift** for Apple targets: the `LoudKit` package at the repo root; see
  `docs/platforms/apple.md`.

The CLI, the server and MCP call the same Python engine. The Swift package is
a separate implementation of the same algorithm, held to the same conformance
fixtures, with CoreML for the renderer.

## Ending the prompt in silence

`enroll(..., end_in_silence=True)` cuts the reference clip so that the
ten-second prompt ends in silence. The library default leaves the recording as
it is. The cut:

1. Take the last pause of at least 60 ms after the first three seconds.
2. Append 0.4 s of silence after it. A pause near ten seconds leaves only part
   of the pad inside the prompt. Keep that pause; do not cut an earlier word to
   fit the pad.
3. Without a pause, truncate at 9.6 seconds if needed, to leave room for the
   silence.

In an enrollment experiment (2026-09-04), ending the prompt in silence moved
speech onset from 0 to 50 ms, and the first 60 ms from -32 to -51 dBFS, on one
cloned voice. The released voices measured 70-120 ms of initial silence. These
are observations on one voice. They do not guarantee the same result for
other recordings, and they do not show that 0.4 seconds is better than a
shorter pause.
