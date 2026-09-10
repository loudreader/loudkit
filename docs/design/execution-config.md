# Execution config: how fast, never what

Maintainer notes for `python/loudkit/execution.py` and the backends. The
docstrings say what each field does; this page says what measurement settled
it, and what each backend does with the answer.

## One class, `None` means "the checkpoint's default"

`ExecutionConfig` has twelve fields and every one defaults to `None`, which
means the backend's default for the device. A named field wins even when its
value equals what the default would have been. `resolved(defaults)` fills the
rest; `precision` merges per module, so `{"vocoder": "fp32"}` means that one
module and naming all four replaces the map.

The alternative is two classes with the same twelve names, one for a complete
configuration and one for a patch. It fails on the merge: comparing each field
against the dataclass default reads equality as "not specified", so an explicit
all-fp32 map over a manifest's fp16 map is dropped and a conformance run
measures a precision it did not ask for. `None` as the one spelling of "unset"
draws the same distinction with one class.

`build_engine` resolves the caller's config against `_default_execution`,
which is the manifest's shipping dtype map on the requested device with the
production fallbacks for everything else. The graph backends run their own
exported graphs at fp32, so the ONNX default is all-fp32.

## Precision

Three of the four modules are settled by measurement. The token generator
tolerates fp16 easily (median KL 1.3e-06, top-1 99.9%) because a sampling
decision boundary annihilates sub-threshold error. The flow encoder does not
(fp16 gives mel correlation 0.619 and +22 dB of high-frequency energy). The
vocoder does not (fp16 produces an audible tone at Nyquist from a cumulative
phase accumulator). The torch backend applies `precision` per module and
refuses fp16 on the encoder and the vocoder outright.

The graph backends do not read the map at all; they run what was exported,
and what was exported is fp32. That is the gate rather than a default. The
manifest stores the generator and the flow estimator in fp16, and fp32 ONNX
measured as the only version worth a second artifact (EXP-015): parity at max
abs delta-logit 1.9e-05, against int8 at 2.15x with quality nobody has
measured, which is why int8 stays blocked (EXP-017). The backend's own refusal
quotes both. Upcasting fp16 storage to fp32 is exact, so
nothing is lost and the comparison against the torch fp32 reference is a pure
export question.

The turbo CoreML estimator is exported fp32 for a reason found the hard way.
An fp16 export gave mel correlation above 0.99999 and waveform correlation
around 0.989; feeding identical mel into the ONNX and the CoreML vocoders
isolated the error to the estimator. The fp32 export restores the reference
waveform band without changing a token. Base loudr-1 keeps its existing fp16
estimator recipe.

## Device placement

The generator and the renderer want different hardware, and with the
streaming pipeline the split's value is parallelism: the renderer renders
window k on the GPU while the generator computes window k+1 on the CPU. On
an Apple laptop a six-window passage ran at RTF 3.37 split against 2.15 with
both stages on the GPU, where they contend for one device. For a single
window there is no pipeline and all-GPU is mildly faster; the default
favours the multi-window paths, which are the ones a reader or a server runs
hot. So the MPS default puts the generator on the CPU.

`device=` and `execution.device` are two spellings of one decision. They were
once resolved independently, so naming both and disagreeing built an engine
on one device and placed it on the other; a disagreement is refused. When
neither names one, `loudkit.load` takes `best_device()` and `build_engine`
takes `cpu`, which is why the fallback lives in the caller.

A split placement has to be reported as one. `build_coreml_engine` hands
`describe()` the device each half runs on, because under CoreML the token
loop and the renderer are placed separately, and a line naming one device for
a build that runs on two records the wrong placement in every benchmark row
that quotes it.

## CUDA graphs and the static cache

Both default to **off**, on CUDA as much as anywhere: no checkpoint default
names them, so `resolved` takes them from the shipping fallback, which is
`False`. A CUDA box that names neither runs the eager decode loop, and a
benchmark taken with `--cuda-graphs` measures the opt-in path, not the
shipped one.

`cuda_graphs` and `compile_model` both capture the decode step as one CUDA
graph over a static KV cache (preallocated fixed-address buffers written in
place), instead of about 1442 kernel launches per token, of which about half
the step was launch overhead. Measured on a desktop NVIDIA GPU: the token
generator about 5x faster, end-to-end RTF 2.3x to 8.7x. `torch.compile`'s
own capture path hits an inductor mask-alignment defect on this model, so
both flags share the manual capture.

Three caveats. CUDA graphs need a Volta-class or newer GPU; on an older card
the flag falls back to the static cache running eagerly, which is slower
than eager. The static cache is the identity contract's `equivalent` class:
the attention reduces over a padded buffer, which switches the cuBLAS kernel
at large widths and drifts logits about 2e-4 per layer at a 750-token
prefill, so a 255-token utterance diverges from eager somewhere around token
26 to 130; deterministic, not token-identical, and `build_engine` warns. And
the graph is captured per synthesis, not per engine, because the static
buffers are sized to the utterance's prefill; every `generate()` pays three
warm-up steps plus a synchronise plus the capture, which amortises over a
long window and can cost more than it saves on a 20-token agent utterance.

## The ragged vocoder

`vocoder_ragged` vocodes the frames a chunk has plus the receptive field,
instead of padding every mel out to the static window (twice the window in
frames) and truncating afterwards. On a fixed-shape backend the padding is
the point; on a free-shape backend a two-second chunk paid for a ten-second
one twice over, because the excitation noise is drawn for the padded length
too, and that noise was most of the stage's time. Padding by
`VOCODER_RIGHT_CONTEXT` frames is enough for the convolutions and the iSTFT
to see what they would have seen: against the full-window output the
difference is 1e-6, fp32 noise from cuDNN picking different algorithms for
different widths, against 1e-1 with no padding at all. 1e-6 on a waveform in
[-1, 1] is 120 dB down. It is the same `equivalent` class as the static
cache: turn it off for byte agreement with another build. Measured by chunk
length on a desktop GPU: 4.3x at 2 s, 2.0x at 5 s, 1.3x at 8 s, 1.0x at a
full window; 9x to 13x end to end over a multi-chunk read.

It is requested, not resolved: the measurement is a CUDA one and the CPU
path is the reference the conformance fixture is taken on, so the torch
backend honours it only where the renderer is CUDA, and the graph backends
have no torch vocoder to honour it at all. `resolved_vocoder_ragged()` is
what runs, and `describe()` prints only that, because `describe()` is what a
benchmark records and a reader checks when a number looks wrong.

The flag pair `--vocoder-ragged` / `--no-vocoder-ragged` on the measuring
tools exists because a `store_true` flag over a default of True could only
ever ask for the default, and a benchmark campaign once measured the padded
vocoder while reporting the ragged one because `--cuda-graphs` sent an
explicit `False` for a knob nobody had named. Every knob a tool sends is
`None` unless its flag was passed.

## Attention

`auto` resolves to `eager` when **either half** runs on MPS, where the fused
scaled-dot-product path aborts the whole interpreter (`LLVM ERROR: Failed to
infer result type(s)` from `mps_matmul`, no Python traceback) or fails to
compile its shader. Either half, because the renderer's flow estimator runs
attention as well as the generator: the split that puts the generator on the
CPU and the renderer on MPS asked only the generator, answered `sdpa`, and
handed Metal the kernel this rule exists to keep away from it. It also
resolves to `eager` on NVIDIA GPUs older than
Ampere, where the fused path raises "FlashAttention only supports Ampere or
newer" at the first step. Elsewhere `sdpa`. A capability probe that fails is
reported as a warning and answered `sdpa`, and only the five exceptions a
probe raises are caught, because `except Exception` once swallowed a torch
that failed to import for a real reason and made a broken install look like
a working one until the first forward pass.

The exported CoreML generator has an attention question of its own, decided
at export rather than at runtime. Representing the attention value product
as a weighted sum for single-query decoding let a local CPU-only CoreML run
select an inaccurate vector-matrix kernel at KV length 113, despite fp32
graph precision. The equivalent multiply-and-reduce form preserves the
reference tokens without changing model weights or sampling, so that is what
the exporter emits. The export regression checks lengths 112, 113, 114, 129
and 256, and the full generator gate also checks free-running sentences with
and without a prefix, in
`tests/test_export_generator.py::test_coreml_decode_attention_across_kernel_boundary`.

## The ONNX provider

`onnx_provider` names which onnxruntime execution provider runs the graphs,
in the five spellings every port accepts. `auto` takes the best the installed
runtime offers (`AUTO_ORDER`, cuda then cpu; coreml and directml are selected
by name only, for the reasons below), and the backend
writes the provider it chose back into the config so `describe()` names what
ran. An explicit provider the build does not offer is an error, never a quiet
downgrade: every session was once pinned to the CPU provider, which is how
published figures came to describe the torch path only. A GPU provider may
change the sampled tokens; that is a per-provider measurement.

`auto` prefers a provider only where a measurement says it is faster. CUDA
leads on that basis and drops out the same way if it loses. DirectML has
never been run by this project: it stays selectable and is not a default.
CoreML is the interesting case, because it is faster and still not a default.

### CoreML is a placement, not a provider

`_session_providers` does not apply CoreML to all six graphs, because it
loses on three of them. `t3_step` runs one decode step and is called once per speech token, so it
decides the whole synthesis: 9.8 ms on CPU against 17.6 ms for the best
CoreML configuration found. `t3_prefill` and `t3_step` also fail to compile
under MLProgram outright, with session creation dying in `model.mil` with
error -7, and MLProgram is the only setting worth having. So loudkit's
`coreml` is a placement: the generator on CPU, the renderer on CoreML.
Measured on an M3 Pro over three synthesis repeats that is RTF 1.35 to 1.70
against 0.85 to 1.02 for all-CPU.

The generator never touching CoreML is what keeps the token stream identical
to the CPU run, index for index. The waveform is not bit-identical, which is
what the identity contract already says about running the renderer somewhere
else.

MLProgram, pinned in `_COREML_OPTIONS`, is the only setting worth having
because the default,
`NeuralNetwork`, shatters the renderer graphs into hundreds of partitions:
flow_estimator 342, flow_encoder 47, vocoder 51, and each boundary is a copy
between CoreML and CPU. Under MLProgram the same graphs take 2, 1 and 25,
which is the difference between losing to CPU and beating it. The default
also *changes the numbers*: a NeuralNetwork vocoder sums 217.70 where CPU
sums 211.15, while MLProgram sums 211.149. So this is not a speed knob with
a quality cost; the fast setting is also the faithful one.

Every provider list ends in CPU, and that tail is onnxruntime's
*per-operator* placement fallback, not a provider fallback: `resolve_provider`
has already refused a provider this build lacks. Without CPU in the list a
graph holding one op the accelerator cannot place fails to load at all, which
is how every non-trivial CUDA and CoreML session behaves.

### Why the faster provider is not the default

Compiling the renderer graphs costs about 146 s the first time on a machine.
With a cache directory that is paid once and later loads cost about 25 s;
without one it is paid on *every* session, which no interactive use can
absorb. The cache runs to roughly 1.6 GB. `$LOUDKIT_COREML_CACHE` overrides
`_coreml_cache_dir`; the default follows the platform convention, and CoreML
exists on exactly one platform.

A default may not spend two minutes and 1.6 GB without being asked, and a
first call that appears to hang is a worse first impression than a slower one
that returns. Ask for CoreML by name.

## Global torch flags

`allow_tf32` is declared rather than inherited because PyTorch ships
`cudnn.allow_tf32` on and `cuda.matmul.allow_tf32` off, so "plain fp32" is
by default neither fp32 nor bit-reproducible against it. A baseline that
inherited it silently made TF32 look worth 1.05x when it is worth 1.17x, and
was not bit-exact with itself. Off by default, and printed either way.

`deterministic` pins cuDNN algorithm selection and turns `benchmark` off,
costing about 5% end to end and buying "same seed, same build, bit-identical
waveform"; without it the vocoder's convolutions pick algorithms freely and
two runs differ by about 5e-06. `num_threads` is pinned beside them because
`torch.set_num_threads` is process-global like the other two.

`pin_determinism` applies all three, on non-CUDA hosts as well: they are
process-global, harmless there, and pinning unconditionally means the
recorded configuration is the running one.

**This mutates process-global torch state.** These are torch's own global
switches. Unrelated code in the same process inherits `cudnn.deterministic`
and the TF32 setting, building a second engine does not restore what the
first one changed, and nothing puts them back. That is a deliberate trade,
because an engine running under someone else's flags could not honour the
identity contract at all. It is still a side effect, and callers who embed
loudkit in a larger torch program should know that an application which pins
determinism for a render has pinned it for whatever else it does afterwards.

A second engine built with contradictory flags is reported as a
`RuntimeWarning` rather than applied in silence, because a recorded
configuration that is not the running one is the failure this library was
built to end. `_PINNED` holds what the first engine asked for so the
contradiction has something to be measured against, and the read, compare and
write are under a lock: two threads building engines at once could otherwise
both read `None`, both pin, and the second would never be reported as the
contradiction it is.

The thread count is in that set for a reason stronger than tidiness: it moves
the audio, not only the wall time. Measured on CPU, two separate processes
running the same `engine.synthesize`, one at one thread and one at five, come
back 4.7e-2 apart at the sample, correlation 0.99925, -28.2 dB error to
signal. The mel estimator is fp16 and its intra-op reductions split by
thread, so the split changes the rounding. At a fixed thread count the path
is deterministic and byte-identical across processes; it is only across
counts that it moves. See `docs/design/models-notes.md`. A second engine
asking for a different count would otherwise re-pin the first one's threads
while the first kept reporting its own. loudkit cannot make two contradictory
engines both correct in one process; it refuses to let the contradiction go
unreported.

## Backends

There are three, in `python/loudkit/backends/`: `torch_backend.py`,
`onnx_backend.py` and `coreml_backend.py`.

A backend turns `(checkpoint, ExecutionConfig, AlgorithmConfig)` into an
`Engine`. It declares execution choices and inherits every algorithm value;
a backend that decides an algorithm value is the bug class this library
exists to end. The registry exists so a new backend is an entry rather than
a fork of `Engine.from_checkpoint`. The torch backend registers lazily and
only when asked for, so an onnx-only install never imports torch.
`production_algorithm` fills, for a checkpoint packed before its manifest
carried them, exactly the static window recipe and the EOS floor with the
shipped constants, and nothing else.

`check_export_record` is one check for both graph backends. Six `.onnx`
files or three `.mlpackage` directories in one folder look like a set and
need not be one: the exporters take a `--stages` list, so a run that names
the renderer leaves the other graphs as they were, and a mixed folder loads
and speaks one checkpoint's tokens through another checkpoint's renderer.
`export.json` beside the set records, per member, the checkpoint digest, the
fingerprint, the step count and the estimator digest; every member must
agree with the others and with the checkpoint being loaded. The estimator
digest is in the agreement set because re-exporting only the estimator stage
from a different estimator with the same step count left every other field
identical. Absent is a warning, because sets exported before the record
exist and stranding them buys nothing a sentence cannot say; present and
wrong is a refusal. `tools/build_release.py` requires it outright.

Every port runs it: `checkpoint.VerifyExport` in Go,
`export::check_export_record` in Rust, `checkExportRecord` in
TypeScript and `verifyCoreMLExport` in Swift, each on the set its own
loader is about to open and each before the first session. The three
enrollment graphs are outside it in all five, because the enrollment
exporters write no record for them to be held to.

### The torch backend

cpu, cuda and mps are one code path with three execution profiles. The model
modules are identical, and the differences are declared through
`ExecutionConfig` rather than decided here: the attention implementation
(`eager` on MPS, for the interpreter abort above) and the determinism and
TF32 pinning on CUDA. Precision is applied per module and validated against
what measurement allows.

Enrollment is built separately from `build_torch_engine`.
`build_torch_enroller` loads the two tensor
groups, the speech tokenizer and the speaker encoder, that synthesis never
touches, and carries a dependency surface (torchaudio's Kaldi fbank,
librosa's mel filters) that should not tax anyone who only synthesises. Its
`path` takes the enrollment checkpoint or a pre-split one carrying those
tensors, and `device` the torch device for the enrollment models. The 256-d
utterance voice encoder is in neither checkpoint: pass its `ve.safetensors`
as `voice_encoder_weights` to enroll the token generator's speaker
embedding, or enrollment fails with an error naming that parameter.

`build_torch_frontend_and_generator` is split out for the CoreML backend,
which supplies its own renderer. The alternative is to build a *whole* torch
engine and discard the mel decoder and vocoder it has just loaded. On the
device where CoreML is the lightweight option, that is several hundred
megabytes of weights read from disk, materialised and freed, startup time
and peak RSS spent on modules that never run, and a plausible OOM on a
phone.

### A manifest is not evidence

The manifest is data from outside the process, and three of its numbers
(`vocab_size`, `hidden_size`, `num_hidden_layers`) decide how much memory
the model constructor asks for, before a single tensor is read. Unchecked, a
twenty-kilobyte file that carries a manifest and almost nothing else can
name an architecture large enough to exhaust the machine; `load_state_dict`
would reject it a moment later, which is a moment too late.
`_check_architecture_against_weights` runs before the constructor.

Bounds would be a weaker answer. Any ceiling is either low enough to reject
a legitimate future checkpoint or high enough to still be worth an
attacker's while. The weights are the honest limit: they are in the same
file, their shapes are in the header, and they cannot be inflated without
inflating the file. A manifest claiming 100 000 layers beside 16 layers'
worth of tensors is refused for the reason that makes it wrong.

The corroboration comes from that header. `gate_proj` is
`(intermediate_size, hidden_size)`, `q_proj` is
`(num_attention_heads x head_dim, hidden_size)` and `k_proj` the same with
the key-value count. Layer zero is enough, because the layers are uniform
and a file whose layers disagree fails `load_state_dict` without having
allocated anything the header did not already promise. Each field is checked
by name rather than through the product of fields, because integer division
would make `head_dim` zero and report a mismatch of `0`, naming neither the
field the caller got wrong nor the value they wrote.

Every field that drives allocation is checked, including the ones a state
dict would appear to cover later: the *model constructor* builds the MLP
before any state dict is loaded, so `intermediate_size` is spent the moment
it is believed, and "checked by `load_state_dict`" is too late for a field
that sizes an allocation. A manifest naming 16 000 000 beside 2100 layers'
worth of tensors asked for 196 GB of `gate/up/down` weights and got past
this function without a word.

`_shape` is where the check either holds or leaks, and it has leaked twice.
Making the comparison conditional on the tensor being present leaves a door
open: a crafted file that *omits* `gate_proj` skips the `intermediate_size`
check, so a twenty-six kilobyte checkpoint claiming
`intermediate_size = 16_000_000` walks past preflight and asks the
constructor for 197 GB. The bypass was one deleted tensor, in the function
written to stop exactly this. There is nothing to be compatible with: a
transformer layer without a gate projection or a query projection is not a
layer this engine can build, and `load_state_dict` would say so afterwards,
after the allocation this function exists to prevent. Both dimensions are
compared, because comparing only the first was a hole of its own: a
`(16_000_000, 0)` tensor weighs almost nothing on disk and satisfied an
`intermediate_size` of sixteen million, after which the constructor asked
for 197 GB. A projection's columns are the hidden size in every case here,
so there is no reason not to check them.

### The ONNX backend

The graphs are exported by `tools/export_onnx.py` from the packed checkpoint
and run on the execution provider named above.

The token generator runs entirely on the graphs. `t3_cond` builds the
34-slot conditioning row (speaker projection, perceiver, emotion),
`t3_prefill` does one causal forward over the whole framed sequence,
returning every-position logits for teacher forcing *and* the KV cache for
the loop, and `t3_step` does one decode step against the cache. The
surrounding logic, framing, embeddings, RoPE positions, the sampler loop and
the EOS floor, is replicated here in numpy and matched bit-for-bit against
the torch generator; the graph only ever sees embedding rows and position
ids.

This module imports no torch module. The helpers it needs, window framing,
the Euler grid and the Philox stream ids, live in
`loudkit.models.windowing`, which is torch-free by design, so a
`loudkit[onnx]` install can synthesise without torch in the process at all.

The renderer mirrors the CoreML backend exactly: `flow_encoder` plus
`flow_estimator` behind the `MelDecoder` protocol and `vocoder` (HiFT, conv
STFT/iSTFT) behind `Vocoder`.

`_Session` pulls its input and output names from the model object rather
than asserting them from a constant the exporter and this module would have
to keep in sync, because onnxruntime ships no type information (see the
pyproject override). The values entering and leaving the graph are numpy
arrays, typed `Any` at the boundary and pinned down with
`np.asarray(..., dtype=...)` at the call sites.

### The CoreML backend

This backend runs the CoreML stage packages exported by
`tools/export_coreml.py`, `flow_encoder` (CPU, fp32), `flow_estimator`
(CPU plus Neural Engine, fp16 pipeline) and `vocoder` (CPU, fp32), behind
the loudkit `MelDecoder` and `Vocoder` protocols. The graph geometry is
exactly the one the iOS app ships (query 255 / prompt 238, T986 mel,
510-frame HiFT); the weights are re-exported from the packed checkpoint so
provenance is one file, not archaeology. It exists so that "matches the
shipped engine" is a table produced by this repo rather than a belief.

New CoreML bundles include validated fp32 generator graphs with explicit KV
inputs, and run without PyTorch. Renderer-only loudr-1 bundles retain the
CPU PyTorch generator for compatibility; a partially present native
generator set is refused. This is separate from the app's stateful
multi-function T3.

`_assets_dir` resolves assets from a `coreml/` directory beside the
checkpoint, or from `LOUDKIT_COREML_ASSETS`. Missing assets fail with the
expected filenames, not with a fallback to a different engine, and a
candidate directory is accepted only when the whole set is there. Testing
only for the estimator
accepted a partial export and stopped looking, so a stale or half-written
directory named by the environment variable shadowed a complete one beside
the checkpoint, and the failure surfaced later as a missing encoder or
vocoder, naming a file rather than the directory choice that caused it. The
ONNX backend has always required the full set; this matches it, and reports
what each candidate was missing.

#### Pinned inputs, or the host dies a second later

Every predict goes through `_PinnedInputs`, which is what keeps this backend
from killing its host a second after it succeeds.

coremltools wraps each input array without copying: `PybindCompatibleArray`
(`coremlpython/CoreMLPythonArray.mm`) builds the `MLMultiArray` over the
caller's numpy buffer and keeps the `py::array` as an Objective-C ivar.
CoreML does not drop that reference when `predict` returns. The MLE5
execution stream lingers and resets itself about a second later on
`com.apple.coreml.MLE5ExecutionStream.resetQueue`, and *that* is where
`-[MLFeatureValue dealloc]` runs. The compiler-generated `.cxx_destruct`
then releases the `py::array` on a dispatch thread that holds no GIL and has
no thread state; if the interpreter's reference was the last one, the
release reaches `_PyObject_Free` and corrupts pymalloc's arenas. The host
process dies about a second after a perfectly successful synthesis, inside
whatever it does next (upstream: apple/coremltools#2827, open and unfixed
through 9.0).

So the interpreter keeps a reference of its own, for the model's lifetime,
and every predict copies into that same buffer. CoreML's release then only
ever takes the count from two to one, and the actual free happens here, on a
thread that holds the GIL. The buffers are bounded rather than leaked
because the exported graphs are static: one buffer per input, reallocated
never. Reusing them is safe because `predict` is synchronous, so the
lingering stream holds the reference and not the data. Measured: waveform
bit-identical to the unpinned path, and the copy does not show above
run-to-run noise (2.09 s against 2.06 s median of four, one 4.96 s sentence
on an M3 Pro). The lock is what makes reuse safe under `Engine.stream`,
whose renderer runs on its own thread; two callers sharing one buffer would
otherwise interleave a fill with a predict.

### What a graph backend refuses

A graph is an algorithm that has already been decided. Where the config asks
for a different one, the honest answer is a refusal and a re-export, not a
substitution, and both graph backends give it.

**A window the graphs were not exported for** (`_require_static_window`).
The CoreML backend has always
checked this; ONNX did not, and the two are exported from the same recipe. A
config framing anything other than 255/238 reached `decode`, where
`static_prompt_tokens or row.shape[1] // 2` silently picked a *different*
prompt split from the one `frame_windows` used, so the graph read a prompt
boundary the framing never put there and the mel came out subtly wrong, with
no error on any layer. It is the same defect class, mel correlation 0.975 to
0.993, that moved the window recipe into configuration in the first place. A
different window is a different algorithm. Re-export the graphs.

**A decode mode the graphs do not implement**
(`_require_single_token_decode`).
`SUPPORTED_FORMAT_VERSIONS` is a statement about the *package*, and only the
torch backend earned the `2` in it: `t3_step` is a one-token step and
`ONNXTokenGenerator` is the numpy mirror of the one-token loop. Handed
fusion weights it would read half the context the model expects and speak
fluently about something else, which is the failure the version gate exists
to make loud, reached from inside the one engine that declares it reads
version 2. The Rust, Go, TypeScript and Swift engines answer this by
refusing the file, and their comments say refusing is the correct answer
from an engine that has not implemented the loop. This is that answer, on
the backend that has not implemented it here.

**Guidance the exported estimator would apply twice.** The CoreML mel
decoder refuses `cfg_dual_path` rather than substituting another form of
guidance. A component that ran different maths under the config object it
carries would pass `_assert_one_algorithm`: the fingerprints agree while the
arithmetic does not.
