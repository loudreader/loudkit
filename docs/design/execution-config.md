# Execution configuration

Maintainer notes for `python/loudkit/execution.py` and the backends. The
docstrings say what each field does. These notes give the measurement behind
each default and what each backend does with each field. Several fields change
the computed samples as well as the speed; the identity contract classifies
those changes.

## One class, `None` means "the checkpoint's default"

`ExecutionConfig` has twelve fields, and each defaults to `None`, which means
the backend's default for the device. A field set explicitly wins, even when
its value equals the default. `resolved(defaults)` fills the unset fields.
`precision` merges per module: `{"vocoder": "fp32"}` sets that one module, and
a map that names all four modules replaces the whole map.

`None` is the only spelling of "unset", so one class keeps an explicit value
apart from a default. A design that detects "unset" by comparing each field
with the dataclass default reads an explicit value equal to the default as
unset. Under that design an explicit all-fp32 map over a manifest's fp16 map is
dropped, and a conformance run measures a precision it did not ask for.

`build_engine` resolves the caller's config against `_default_execution`: the
manifest's storage dtype map on the requested device, and the shipping
fallbacks (`_FALLBACK` in `execution.py`) for every other field. The graph
backends do not take the storage dtypes. The ONNX default is all fp32. The
CoreML default is fp32 except the loudr-1 (`single` decode) flow estimator,
which is fp16.

## Precision

Measurement settles three of the four modules:

- The token generator tolerates fp16: median KL 1.3e-06 and top-1 agreement
  99.9% against fp32. A sampling decision absorbs logit error below its
  margin, but fp16 can still change a token sampled near a decision boundary.
- The flow encoder does not: fp16 gives mel correlation 0.619 and +22 dB of
  high-frequency energy.
- The vocoder does not: fp16 produces an audible tone at Nyquist from its
  cumulative phase accumulator.

The torch backend applies `precision` per module and refuses fp16 on the
encoder and the vocoder (`_check_precision`).

The ONNX backend runs exported fp32 graphs and refuses any other value in the
precision map (`build_onnx_engine`). The manifest stores the token generator
and the flow estimator in fp16. Upcasting fp16 storage to fp32 is exact, so the
comparison against the torch fp32 reference measures the export alone. The fp32
ONNX export matches torch at a max abs logit delta of 1.9e-05. An fp16 ONNX
export was measured and is not worth a second artifact. int8 measured 2.15x
faster, has no quality measurement, and stays blocked.

The CoreML backend reads the generator's precision. Native CoreML generation
requires fp32; `generator_device="cpu"` runs the PyTorch generator instead,
which also accepts fp16. The renderer packages carry the precision they were
exported with: the flow encoder and the vocoder are fp32 on the CPU, and the
flow estimator runs on the CPU and the Neural Engine, in fp16 for loudr-1 and
in fp32 for loudr-1-turbo. The precision map records the package's precision;
the backend does not convert packages.

The turbo estimator is fp32 because of a measurement. An fp16 export gave mel
correlation above 0.99999 and waveform correlation around 0.989. Feeding
identical mel into the ONNX and the CoreML vocoders isolated the error to the
estimator. The fp32 export restores the reference waveform band without
changing a token. The loudr-1 estimator keeps its fp16 export.

## Device placement

On MPS the default puts the generator on the CPU and the renderer on the GPU.
In the streaming pipeline the renderer renders window k on the GPU while the
generator computes window k+1 on the CPU. Measured on an Apple Silicon laptop,
a six-window passage ran at RTF 3.37 with this split and at 2.15 with both
stages on the GPU, where they share one device. A single window has no overlap
to gain, and all-GPU is slightly faster there. The default favours
multi-window synthesis: long text, streaming and servers.

`device=` and `execution.device` name the same choice. When both are set they
must agree, and a disagreement is refused (`_agreed_device`). When neither is
set, `loudkit.load` uses `best_device()` and `build_engine` uses `cpu`, so the
fallback belongs to the caller.

`build_coreml_engine` records the device each stage runs on:
`generator_device` is `coreml` for native generation and `cpu` for the PyTorch
generator. `describe()` then names both placements, so a benchmark row records
the placement that ran.

## CUDA graphs and the static cache

`cuda_graphs` and `compile_model` are off by default on every device, CUDA
included. No checkpoint default sets them, so `resolved` takes `False` from
`_FALLBACK`. Without either flag the eager decode loop runs. A benchmark taken
with `--cuda-graphs` measures the opt-in path, not the default one.

Both flags capture the decode step as one CUDA graph over a static KV cache:
preallocated buffers at fixed addresses, written in place. Eager decode issues
about 1442 kernel launches per token, and launch overhead is about half of the
step. Measured on a desktop NVIDIA GPU, the token generator runs about 5x
faster, and end-to-end RTF goes from 2.3x to 8.7x. The capture path of
`torch.compile` hits an Inductor mask-alignment defect on this model, so both
flags use the same manual capture.

Three caveats:

- CUDA graphs need a Volta-class GPU or newer. On an older GPU the flag runs
  the static cache eagerly, which is slower than eager decode.
- The static cache is in the identity contract's `equivalent` class. Attention
  reduces over a padded buffer, which switches the cuBLAS kernel at large
  widths. Logits drift about 2e-4 per layer at a 750-token prefill, so a
  255-token utterance diverges from eager decode somewhere between token 26
  and token 130. The path is deterministic but not token-identical to eager
  decode, and `build_engine` warns.
- Capture has a fixed cost: three warm-up steps, a synchronisation and the
  capture itself, about as long as decoding a 200-token window. The generator
  caches each captured graph by KV length, rounded up to `_GRAPH_BUCKET` (64),
  and by sampling law, and later calls in the same bucket reuse it. The first
  call in a new bucket pays the cost, which can exceed the saving on a
  20-token utterance.

## The ragged vocoder

`vocoder_ragged` vocodes the frames a chunk has plus `VOCODER_RIGHT_CONTEXT`
(32) frames of right context, rounded up to `VOCODER_LENGTH_BUCKET` (64) frames
and capped at the static window. Without it, every mel is padded to the static
window (twice `max_speech_tokens` in frames) and truncated after rendering. A
fixed-shape backend needs that padding. On a free-shape backend the padding
makes a two-second chunk cost as much as a ten-second window, because the
excitation noise is drawn per output sample, and that noise was most of the
stage's time.

Measured against the full-window output, the right-context padding leaves a
difference of 1e-6, which is fp32 noise from cuDNN choosing different
algorithms for different widths. With no padding the difference is 1e-1. A
difference of 1e-6 on a waveform in [-1, 1] is 120 dB down. The ragged path is
in the `equivalent` class, like the static cache: turn it off for byte
agreement with another build. Measured by chunk length on a desktop GPU, it is
4.3x faster at 2 s, 2.0x at 5 s, 1.3x at 8 s and 1.0x at a full window, and 9x
to 13x faster end to end over a multi-chunk read.

The field is a request. `resolved_vocoder_ragged()` returns what runs: true only
when the torch backend renders on CUDA and the field is not `False`. The
measurement is a CUDA one, and the CPU path is the reference that the
conformance fixture is taken on. The graph backends have no torch vocoder, so
the field has no effect there. `describe()` prints the resolved value, because
benchmark rows record `describe()`.

`tools/bench.py` has two flags, `--vocoder-ragged` and `--no-vocoder-ragged`. A
single `store_true` flag over a default of `True` could only request the
default. With neither flag the tool sends `None`, and the build's default
applies. Every knob the tool sends is `None` unless its flag is passed.

## Attention

`auto` resolves to `eager` when either stage runs on MPS. There the fused
scaled-dot-product path can abort the interpreter (`LLVM ERROR: Failed to infer
result type(s)` from `mps_matmul`, with no Python traceback) or fail to compile
its shader. Either stage counts, because the renderer's flow estimator runs
attention too, so the default MPS split (generator on the CPU, renderer on MPS)
resolves to `eager`. `auto` also resolves to `eager` when the generator runs on
an NVIDIA GPU older than Ampere (compute capability below 8), where the fused
path raises "FlashAttention only supports Ampere or newer" at the first step.
Everywhere else it resolves to `sdpa`.

The capability probe catches five exception types (`ImportError`,
`RuntimeError`, `AssertionError`, `ValueError` and `IndexError`), warns, and
answers `sdpa`. Any other exception propagates, so an unexpected failure is
reported where it happens and not at the first forward pass.

The exported CoreML generator settles one attention choice at export time. For
single-query decoding the exporter writes the attention-value product as
multiply-and-reduce operations. The weighted-sum form let a CPU-only CoreML run
select an inaccurate vector-matrix kernel at KV length 113, despite fp32 graph
precision. The multiply-and-reduce form keeps the reference tokens and changes
neither the weights nor the sampling.
`tests/test_export_generator.py::test_coreml_decode_attention_across_kernel_boundary`
checks KV lengths 112, 113, 114, 129 and 256, and the generator export gate
also checks free-running sentences with and without a prefix.

## The ONNX provider

`onnx_provider` selects the onnxruntime execution provider that runs the
graphs. It takes the five spellings every port accepts: `auto`, `cpu`, `cuda`,
`coreml` and `directml`. `auto` takes the first provider in `AUTO_ORDER` (cuda,
then cpu) that the installed runtime offers; `coreml` and `directml` run only
when named. The backend writes the resolved provider back into the config, so
`describe()` names what ran. An explicit provider that the build lacks is an
error (`resolve_provider`) and never falls back to the CPU, so a figure
labelled with a provider was measured on that provider. A GPU provider may
change the sampled tokens; that is a per-provider measurement.

`AUTO_ORDER` lists a provider only where a measurement shows it is faster, and
CUDA leads on that basis. DirectML has not been measured by this project, so it
is selectable but never a default. CoreML is faster for the renderer and is
still not a default, because of its compile cost (see below).

### CoreML placement under the ONNX backend

With `onnx_provider="coreml"`, `_session_providers` applies CoreML to the three
renderer graphs (`RENDERER_GRAPHS`) only. Every generator graph stays on the
CPU provider. `t3_step` runs once per speech token, so its speed sets the
synthesis time: 9.8 ms on the CPU against 17.6 ms for the best CoreML
configuration found. `t3_prefill` and `t3_step` also fail to compile under
MLProgram: session creation fails in `model.mil` with error -7. Measured on an
M3 Pro on 2026-08-23 (before 0.1.0; not re-measured on 0.1.1), over three
synthesis repeats, this placement runs at RTF 1.35 to 1.70, against 0.85 to
1.02 for all-CPU.

Because the generator stays on the CPU provider, the token stream is identical
to the CPU run, index for index. The waveform is not bit-identical, which the
identity contract already allows for a renderer that runs on other hardware.

`_COREML_OPTIONS` pins `ModelFormat=MLProgram`. The CoreML default,
NeuralNetwork, splits the renderer graphs into many partitions (flow_estimator
342, flow_encoder 47, vocoder 51), and each boundary is a copy between CoreML
and the CPU. Under MLProgram the same graphs take 2, 1 and 25 partitions, and
CoreML beats the CPU instead of losing to it. NeuralNetwork also changes the
output: the sum of a NeuralNetwork vocoder's output is 217.70, against 211.15
on the CPU and 211.149 under MLProgram. MLProgram is both the faster setting
and the one that matches the CPU.

Every provider list ends with the CPU provider. That entry is onnxruntime's
per-operator placement fallback, not a provider fallback: `resolve_provider`
has already refused a provider the build lacks. Without the CPU entry, a graph
with one operator the accelerator cannot place fails to load.

### CoreML compile cost

Compiling the renderer graphs for CoreML takes about 146 s the first time on a
machine, and the compiled cache takes about 1.6 GB. With a cache directory,
later loads take about 25 s, against 2.5 s for the CPU provider. Without one,
every session pays the full compile. These load figures were measured on an M3
Pro on 2026-08-23, before 0.1.0; the 0.1.1 measurements have no load figure
for the ONNX CoreML provider.
`$LOUDKIT_COREML_CACHE` overrides `_coreml_cache_dir`. The default is
`~/Library/Caches/loudkit/coreml`, the macOS convention, because CoreML runs
only on Apple platforms.

A default must not spend two minutes and 1.6 GB of disk without a request, and
a first call that seems to hang is harder to diagnose than a slower one that
returns. CoreML therefore runs only when named.

## Global torch flags

`allow_tf32` is set explicitly and never inherited. PyTorch ships
`cudnn.allow_tf32` on and `cuda.matmul.allow_tf32` off, so its default "fp32" is
neither plain fp32 nor bit-reproducible against plain fp32. Measured, a
baseline that inherited the setting made TF32 look worth 1.05x when it is worth
1.17x, and was not bit-exact with itself. The default is off, and `describe()`
prints the value either way.

`deterministic` pins cuDNN algorithm selection and turns `cudnn.benchmark` off.
It costs about 5% end to end. With it, the same seed on the same build and
device gives a bit-identical waveform. Without it, the vocoder's convolutions
choose algorithms freely and two runs differ by about 5e-06. `num_threads` is
pinned with them because `torch.set_num_threads` is process-global like the
other two.

`pin_determinism` in `backends/torch_backend.py` applies all three settings on
every host, with or without CUDA. They are process-global and harmless without
CUDA, and pinning them unconditionally keeps the recorded configuration equal
to the running one.

**These settings change process-global torch state.** Unrelated code in the
same process inherits `cudnn.deterministic` and the TF32 settings. Building a
second engine does not restore what the first one changed, and loudkit never
restores them. loudkit accepts this side effect because an engine that runs
under another caller's flags cannot honour the identity contract. An
application that embeds loudkit in a larger torch program must coordinate
these flags: pinning determinism for a render pins it for the rest of the
process.

A second engine built with different flags applies them and raises a
`RuntimeWarning`. `_PINNED` holds the flags of the most recent pin, so each new
engine is compared with the last one. The read, the comparison and the write
run under `_PIN_LOCK`, so two threads that build engines at once cannot both
read `None` and miss the conflict. The warning exists because the older
engine's `describe()` no longer matches what it runs.

The thread count is in that set because it changes the audio, not only the
wall time. Measured on the CPU, two processes running the same
`engine.synthesize`, one with one thread and one with five, differ by up to
4.7e-2 per sample (correlation 0.99925, -28.2 dB error to signal). The mel
estimator is fp16 there, and its intra-op reductions split by thread, so the
split changes the rounding. At a fixed thread count the path is deterministic
and byte-identical across processes; see `docs/design/models-notes.md`. Because
the setting is process-global, two engines in one process cannot keep
different thread counts: the second one re-pins the first engine's threads
while the first one still reports its own. loudkit reports that conflict and
does not resolve it.

## Backends

There are three, in `python/loudkit/backends/`: `torch_backend.py`,
`onnx_backend.py` and `coreml_backend.py`.

A backend turns `(checkpoint, ExecutionConfig, AlgorithmConfig)` into an
`Engine`. It chooses execution settings and takes every algorithm value from
the `AlgorithmConfig`; a backend never decides an algorithm value. A new
backend registers a builder with `register_backend` and needs no change to
`Engine.from_checkpoint`. The torch backend registers only when a torch device
is requested, so an ONNX-only install never imports torch.

For a checkpoint whose manifest omits them, `production_algorithm` fills three
blocks with the shipped values:

- the static window recipe (`PRODUCTION_WINDOW`);
- the EOS floor, `max(10, 1.2 x text tokens)`;
- the `postprocess` block, at its defaults, keeping the render-id censuses read
  from the top level of the manifest.

`check_export_record` is one check for both graph backends. A folder of graphs
or packages can look like a set without being one. The exporters take a
`--stages` list, so a run that exports only the renderer leaves the other
graphs as they were, and a mixed folder would speak one checkpoint's tokens
through another checkpoint's renderer. `export.json` beside the set records,
per member, the checkpoint digest, the fingerprint, the Euler step count and
the estimator digest. Every member must agree with the others and with the
checkpoint being loaded. The estimator digest is in the set because
re-exporting only the estimator, from a different estimator with the same step
count, leaves every other field identical.

The member list follows the decode mode. A `single` set has six ONNX graphs:
`t3_cond`, `t3_prefill`, `t3_step` and the three renderer graphs. A
`fusion_mtp2` set has seven, with `t3_pair_step` and `t3_head2` in place of
`t3_step`. CoreML uses the matching `.mlpackage` directories, or only the three
renderer packages when the PyTorch generator runs. A missing record is a
warning, because older exports carry none; a record that disagrees is a
refusal. `tools/build_release.py` requires the record.

Every port runs the same check: `Checkpoint.VerifyExport` in Go,
`export::check_export_record` in Rust, `checkExportRecord` in TypeScript and
`verifyCoreMLExport` in Swift. Each checks the set its own loader is about to
open, before the first session. The three enrollment graphs are outside the
check in all five implementations, because the enrollment exporters write no
record for them.

### The torch backend

cpu, cuda and mps are one code path with three execution profiles. The model
modules are identical, and the differences are declared through
`ExecutionConfig`: the attention implementation (`eager` on MPS, for the
interpreter abort above) and the determinism and TF32 pinning, which matters on
CUDA. Precision is applied per module and validated against what measurement
allows.

`build_torch_enroller` builds enrollment separately from `build_torch_engine`.
It loads the two tensor groups that synthesis never touches, the speech
tokenizer and the speaker encoder, and it needs torchaudio's Kaldi fbank and
librosa's mel filters, dependencies that synthesis does not have. Its `path`
takes the enrollment checkpoint, or a pre-split checkpoint that still carries
those tensors, and `device` takes the torch device for the enrollment models.
The 256-d utterance voice encoder is in neither checkpoint: pass
`ve.safetensors` as `voice_encoder_weights` to enroll the token generator's
speaker embedding. Without it, enrollment fails with an error that names that
parameter.

`build_torch_frontend_and_generator` builds the first two stages only, for the
CoreML backend's PyTorch generator path. Building a whole torch engine there
would load the mel decoder and the vocoder and then discard them: several
hundred megabytes of weights read, materialised and freed, with the startup
time and peak memory that go with them.

### Architecture validation before allocation

Three numbers in the manifest (`vocab_size`, `hidden_size` and
`num_hidden_layers`) decide how much memory the model constructor requests,
before any tensor is read. The manifest is data from outside the process.
Without a check, a twenty-kilobyte file that carries a manifest and almost
nothing else can declare an architecture large enough to exhaust the machine,
and `load_state_dict` rejects it only after the allocation.
`_check_architecture_against_weights` runs before the constructor.

The check compares the manifest with the weights, not with fixed bounds. A
ceiling is either low enough to reject a legitimate future checkpoint or high
enough to still allow a costly allocation. The weights are in the same file,
their shapes are in the header, and they cannot be inflated without inflating
the file. A manifest that claims 100 000 layers beside 16 layers of tensors is
refused because the two disagree.

The shapes come from the safetensors header. `gate_proj` is
`(intermediate_size, hidden_size)`, `q_proj` is
`(num_attention_heads x head_dim, hidden_size)`, and `k_proj` is the same with
the key-value head count. Layer zero is enough, because the layers are
uniform, and a file whose layers disagree fails `load_state_dict` without
having allocated more than the header already declared. Each field is checked
by name, so the error names the manifest field and the value that was written.
A check through a product of fields would make `head_dim` zero by integer
division and report a mismatch of `0`.

Every field that sizes an allocation is checked, including fields that a state
dict appears to cover later. The model constructor builds the MLP before any
state dict loads, so `intermediate_size` is allocated as soon as it is read.
Without the check, a manifest declaring `intermediate_size = 16_000_000` beside
a 2100-row `gate_proj` would request 196 GB of `gate/up/down` weights.

`_shape` requires each layer-zero projection to be present and compares both of
its dimensions:

- A missing `gate_proj` would skip the `intermediate_size` check. A
  twenty-six-kilobyte checkpoint that claims `intermediate_size = 16_000_000`
  and omits `gate_proj` would ask the constructor for 197 GB. A transformer
  layer without a gate or query projection is not a layer this engine can
  build, so a missing projection is refused.
- A `(16_000_000, 0)` tensor weighs almost nothing on disk and matches an
  `intermediate_size` of sixteen million by its first dimension alone. A
  projection's second dimension is the hidden size in every case here, so
  both dimensions are compared.

`tests/test_checkpoint.py` holds both cases.

### The ONNX backend

`tools/export_onnx.py` exports the graphs from the packed checkpoint. They run
on the execution provider described above.

The token generator runs on the graphs. `t3_cond` builds the 34-slot
conditioning row (speaker projection, perceiver, emotion). `t3_prefill` runs
one causal forward over the whole framed sequence and returns the logits at
every position, for teacher forcing, and the KV cache for the loop. For
`single` decoding, `t3_step` runs one decode step against the cache, with an
embedding row and a position as input. For `fusion_mtp2`, `t3_pair_step` takes
the pair's token ids and positions and returns logits and the hidden state, and
`t3_head2` takes that hidden state and the first token id and returns the
second token's logits. The logic around the graphs (framing, embeddings,
positions, the pair fusion for `fusion_mtp2`, the sampler loop and the EOS
floor) is NumPy, a mirror of the torch generator's loop. The export gate
compares teacher-forced logits in fp32 and requires freely sampled tokens to
match the torch reference exactly on the conformance fixture's end-to-end
sentences, with and without a prefix. Token agreement beyond those sentences is
measured per language in the identity contract.

The module imports no torch module. The helpers it needs (the window framing,
the Euler grid and the Philox stream ids) are in `loudkit.models.windowing`,
which is torch-free, so a `loudkit[onnx]` install synthesises without torch in
the process.

The renderer is `flow_encoder` plus `flow_estimator` behind the `MelDecoder`
protocol, and `vocoder` (HiFT, conv STFT/iSTFT) behind `Vocoder`. The CoreML
backend subclasses these renderer classes and replaces only the session calls.

`_Session` reads its input and output names from the loaded model, so the
exporter and this module do not keep a list of names in step. Values cross the
graph boundary as NumPy arrays, typed `Any` because onnxruntime ships no type
information (see the pyproject override), and the call sites fix their dtypes
with `np.asarray(..., dtype=...)`.

### The CoreML backend

This backend runs the CoreML packages exported by `tools/export_coreml.py`
behind the loudkit `MelDecoder` and `Vocoder` protocols:

- `flow_encoder`: CPU, fp32;
- `flow_estimator`: CPU and Neural Engine, fp16 for loudr-1 and fp32 for
  loudr-1-turbo;
- `vocoder`: CPU, fp32.

The graph geometry is query 255 and prompt 238 tokens, a 986-frame mel
condition, and a 510-frame HiFT window. The packages are exported from the
packed checkpoint, so that one file is their provenance.

A complete bundle also carries fp32 generator packages with explicit KV inputs,
and runs without PyTorch. A renderer-only loudr-1 bundle uses the CPU PyTorch
generator. A partial native generator set is refused.

`_assets_dir` looks for the packages in `LOUDKIT_COREML_ASSETS` first, then in
a `coreml/` directory beside the checkpoint. A candidate directory is accepted
only when it holds every required package, so an incomplete directory named by
the environment variable cannot hide a complete one beside the checkpoint.
When no candidate is complete, the error lists what each candidate is missing
and does not fall back to another engine. The ONNX backend applies the same
rule to its graphs.

#### Pinned inputs

Every predict call goes through `_PinnedInputs`, which keeps each input array
alive for the model's lifetime. Without it, the host process can crash about a
second after a successful synthesis.

coremltools wraps each input array without copying: `PybindCompatibleArray`
(`coremlpython/CoreMLPythonArray.mm`) builds the `MLMultiArray` over the
caller's NumPy buffer and keeps the `py::array` as an Objective-C ivar. CoreML
does not drop that reference when `predict` returns. The MLE5 execution stream
stays alive and resets itself about a second later on
`com.apple.coreml.MLE5ExecutionStream.resetQueue`, and that is where
`-[MLFeatureValue dealloc]` runs. The compiler-generated `.cxx_destruct` then
releases the `py::array` on a dispatch thread that holds no GIL and has no
thread state. If that was the last reference, the release reaches
`_PyObject_Free` and corrupts pymalloc's arenas, and the process dies inside
whatever it does next. The failure is reported upstream as
[apple/coremltools#2827](https://github.com/apple/coremltools/issues/2827); it
was open against coremltools 9.0 on 2026-08-23.

`_PinnedInputs` therefore keeps its own reference to every array it passes to
CoreML, for the model's lifetime, and copies each input into that storage
before `predict`. CoreML's release then only lowers the reference count, and the
free happens on a thread that holds the GIL. Storage is kept per input name and
grows geometrically (to the next power of two) when an input grows, which is
what a growing KV cache does. The array views handed to CoreML stay referenced
too, so retained memory stays linear in the largest input. Reusing storage is
safe because `predict` is synchronous: the lingering stream holds a reference,
not a pending read. Measured, the waveform is bit-identical to the unpinned
path, and the copy does not show above run-to-run noise (2.09 s against 2.06 s,
median of four runs of one 4.96 s sentence on an M3 Pro). A lock serialises the
copy and the predict, because `Engine.stream` runs the renderer on its own
thread, and two callers would otherwise interleave a copy with a predict.

### What a graph backend refuses

A graph is an algorithm fixed at export time. When the config asks for a
different one, both graph backends refuse and ask for a re-export; they never
substitute.

**A window the graphs were not exported for** (`_require_static_window`). The
exported graphs are static at query 255 and prompt 238, and both graph
backends refuse any other framing when the engine is built. A different
framing would make the mel decoder read a prompt boundary that the framing did
not place, and the mel would come out subtly wrong with no error. A framing
mismatch measures mel correlation 0.975 to 0.993 (see
`docs/design/models-notes.md`), which is why the window recipe is part of the
configuration. A different window is a different algorithm: re-export the
graphs.

**A decode mode with no graph set.** The graph backends dispatch on the
manifest's `decode.mode`: `t3_step` for `single`, `t3_pair_step` and
`t3_head2` for `fusion_mtp2`. They refuse only an unknown mode
(`_require_known_decode`), before any graph loads, so the error names the mode
and does not read as a missing file.

**Guidance the exported estimator would apply twice.** Both graph mel decoders
refuse `cfg_dual_path`. The exported estimator is guidance-distilled, and
running classifier-free guidance on it applies guidance twice. A component
that ran different arithmetic under the config it carries would still pass
`_assert_one_algorithm`, because the fingerprints would agree.
