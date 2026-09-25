# Model modules

Maintainer notes for `python/loudkit/models/`: the reasoning and the
measurements behind each module and symbol. Each heading names the module and
the symbol a note belongs to. User documentation is under `docs/guides/` and
`docs/reference/`.


## `loudkit/models/__init__.py`


### `module`

One module per pipeline stage, mirroring `loudkit.contracts`, plus the helpers
the stages share:

- `.generator`: the Llama-architecture token generator (T3);
- `.flow`: conditional flow matching, tokens to mel;
- `.vocoder`: HiFT (HiFiGAN with an NSF source), mel to waveform;
- `.enroll`: reference audio to a `loudkit.voice.VoiceProfile`;
- `.enrollment_audio`: the reference-audio contract and the pause cut, with no
  model and no torch;
- `.resample`: the enrollment resampler, one law in every implementation;
- `.timestretch`: WSOLA time stretching for `speed`;
- `.windowing`: the torch-free window framing, Euler grid and noise-stream ids;
- `.noise`: render randomness as data, addressed by the Philox counter (see
  `docs/design/sampler-and-noise.md`).

The text frontend is `loudkit.frontend.text`.

The torch modules mirror the packed checkpoint's tensor names (`t3.*` and
`s3gen.*` with the prefix stripped), so weights load with `strict=True`. A
naming mismatch fails at load time, before any audio is rendered.


## `loudkit/models/enroll.py`


### `module`

Enrollment is separate from synthesis. It is slow, it needs models that
rendering never touches (shipped as the 523 MB enrollment checkpoint and the
5.7 MB `ve.safetensors`), and its output is a small file that can be produced
once on a fast machine and shipped. Four models cooperate:

- the **S3 speech tokenizer** (a Whisper-style FSMN encoder with an FSQ
  quantiser, checkpoint namespace `s3gen.tokenizer`): reference audio to the
  25 Hz speech tokens that become the mel decoder's prompt and, truncated to
  150, the token generator's conditioning prompt;
- the **CAM++ x-vector encoder** (`s3gen.speaker_encoder`): the 192-d speaker
  vector the flow conditions on;
- a **mel extractor** (the Matcha recipe: 24 kHz, n_fft 1920, hop 480, Slaney
  mels, log-compressed): the prompt mel;
- the **utterance voice encoder** (a 3-layer LSTM over 40-mel partials): the
  256-d vector the token generator was trained against. Its weights are in
  neither checkpoint. They ship as `ve.safetensors` and are passed as the
  `voice_encoder_weights` constructor argument; without them, enrollment fails
  with an error that names that argument.

The architecture code is adapted from CosyVoice, S3Tokenizer and 3D-Speaker
(Apache-2.0), and from FunASR and Real-Time-Voice-Cloning (MIT), reduced to the
inference path. NOTICE lists each of the five projects with its own copyright
holder. Module names mirror the checkpoint, so weights load with `strict=True`.

Enrollment has no sampling step, so it is deterministic for a fixed build,
device and execution configuration. Across implementations, the enrollment
fixture in `tests/data/enrollment/` measures agreement.


### `enroll`

The input contract is `validate_reference_audio`'s: 5 to 10 seconds is the
recommended length, and a clip over 30 seconds is refused. The clip is used at
two rates: 24 kHz for the prompt mel, and 16 kHz for tokenisation and both
speaker encoders. The prompt uses at most the first ten seconds, because the
static prompt window holds about 9.5 s and anything past it would be enrolled
and then cut on every render. The CAM++ x-vector reads the same ten seconds;
the 256-d utterance embedding reads the whole clip.

The 24 kHz to 16 kHz downsample uses one resampler, `loudkit.models.resample`,
for both paths. Every port can reproduce its Hann-windowed-sinc law bit for
bit. A C library such as librosa's `soxr_hq` has a float accumulation order no
port can reproduce, so it is not used. Tokens from this resampler differ from
those of the reference pipeline, which split the downsample between
torchaudio and `soxr_hq`, and the shipped voices are enrolled with this
resampler.

Raises `ValueError` for input outside the contract: a non-positive sample rate,
a non-mono shape, NaN or infinite samples, silence, a clip shorter than
`_MIN_ENROLL_SECONDS` (1 s) or longer than 30 seconds. The message describes an
acceptable input.


## `loudkit/models/enrollment_audio.py`


### `validate_reference_audio`

The one statement of the reference-audio contract, used by every caller: the
CLI's `clone`, `loudkit.enroll`, and the ports, which check their own input the
same way. The clip must be mono, have finite samples, last between
`_MIN_ENROLL_SECONDS` and `_MAX_ENROLL_SECONDS`, and not be silence. The
recommended input is 5 to 10 seconds. The prompt is built from the first 10
seconds and the utterance embedding reads the whole clip, so a long recording
is refused: enrolling it would produce a voice the documentation does not
describe.

Raises `ValueError` with a message that describes an acceptable input.


## `loudkit/models/flow.py`


### `module`

Architecture (checkpoint namespace `s3gen.flow`, names mirrored so weights load
strict):

- a token embedding;
- an upsampling conformer encoder (6 blocks at 25 Hz, a nearest-neighbour x2
  upsample, 4 more blocks at 50 Hz) that produces the mean field `mu`;
- a 1-D U-Net estimator (1 down, 12 mid and 1 up stages of causal resnet and
  transformer blocks) that predicts the flow velocity.

Euler steps integrate noise into a mel: two for loudr-1 and one for
loudr-1-turbo, as the manifest's `n_cfm_timesteps` declares. The loudr-1
estimator was step-distilled from a six-step teacher. Guidance is distilled
into the shipped estimators' weights, so `single_path` is the shipping mode and
`cfg_dual_path` exists only to drive an undistilled teacher.

Every algorithm value comes from `AlgorithmConfig`. Three of them matter most:

- **Guidance mode and rate come from the config.** The upstream class
  hard-codes `inference_cfg_rate = 0.7`. Applied to a guidance-distilled
  estimator, that default applies guidance twice, and the mel correlation
  against the correct output falls to 0.979, a defect no output check catches.
- **The default time grid is cosine**, `t_i = 1 − cos(i/K · π/2)`, where `K` is
  the step count. It is the schedule the estimators were trained against. An
  explicit `euler_grid` in the config takes precedence. The upstream `meanflow`
  branch integrates a linear grid, which does not match these estimators.
- **The window recipe is the static one** when `WindowConfig` says so: the
  query padded to 255 tokens and the prompt framed to exactly 238 tokens with
  the silence unit, the mel condition zero-padded to 986 frames, and no masks.
  The recipe is configuration, shared by every backend (see
  `loudkit/models/windowing.py` below for the measured effect of framing).

The flow prior is Philox-addressed data (`.noise`), not device RNG state, so the
same seed draws the same prior on every device and backend. Measured, two
renders of identical tokens with different prior noise correlate at 0.109, so
the prior's seed is part of the output.


### `_Estimator`

Stages are stored as nested `ModuleList`s, `[resnet, transformers, tail-conv]`,
which are the containers the checkpoint's parameter names follow, so the
weights load without any remapping.

With a single 256-channel level, the "down" tail conv has stride 1, so nothing
is downsampled, and the skip connection concatenates features of the same
length. The stage names stay, the shapes do not change, and every conv is
causal.


### `TorchMelDecoder`

- `config`: the algorithm, including the guidance mode, the Euler grid and the
  window recipe.
- `estimator_dtype`: the compute dtype for the estimator only. fp16 is measured
  safe there (mel correlation 0.999999). The encoder has no dtype parameter and
  runs in fp32; the torch backend refuses any other precision for
  `mel_decoder.encoder`, where fp16 measured mel correlation 0.619 with +22 dB
  of high-frequency energy. Precision is therefore per module here, as
  `ExecutionConfig.precision` declares it.


### `_EspnetRelPositionalEncoding._extend`

The positional-encoding table is a cache that grows during inference. Two
threads calling `synthesize` on one engine can both find the table too short
and both rebuild it. `_extend` rebuilds the table under a lock and returns the
reference it holds inside the lock, and the caller slices that reference
instead of reading `self._pe` again. Another thread can replace `self._pe`
between the rebuild and the slice, and a slice of a table built for another
length is in range for both and silently wrong.

The table grows on demand. It is `(1, 2·length - 1, d_model)`, and `length` is
a passage's token count, so precomputing it for the worst case would allocate
for a passage no caller asked for. On the single-flight path that the server
and the CLI use, the lock is uncontended and costs one atomic operation per
call.


## `loudkit/models/generator.py`


### `module`

In loudr-1 the token generator (the T3 student) has 16 decoder layers, hidden
size 1024, 16 query heads of width 64 with 4 KV heads (GQA), SwiGLU MLPs, RoPE
with the Llama 3 long-context scaling, and a speech head over an 8194-token
vocabulary. The architecture is read from the manifest's `llama_config`.
loudr-1-turbo uses a smaller token generator, with a second head (`head2`) and a
fusion MLP for two-token decoding.

The generator is implemented directly in torch, without `transformers`, for
three reasons: the arithmetic is visible, the attention implementation is
selectable (the fused path can abort the interpreter on MPS with only an
`LLVM ERROR` from `mps_matmul`), and the dependency surface stays small. Module
attribute names mirror the packed checkpoint (`t3.*`), so weights load with
`strict=True`.

Sequence layout:

```text
[ cond (34) | [START] text [STOP] | speech: START, s0, s1, ... ]
```

The conditioning row is a speaker projection (1 slot), a perceiver-resampled
speech prompt (32) and the emotion slot (1). Text and speech carry learned
positional embeddings on top of their token embeddings, each starting at zero.
RoPE handles relative order inside the transformer, and the learned tables tell
it which segment a position is in.

Two rules live in this module:

- **The generator applies the EOS floor** before the sampler sees the logits:
  the stop token is masked to −inf until the configured minimum length (in
  production, `max(10, 1.2 x text tokens)`). The sampler's exemption rules
  cannot forbid one token until a step count, and the floor is an EOS policy,
  which belongs to the algorithm layer.
- `teacher_forced_logits` runs one causal forward over the whole forced
  sequence and does not replay the decode loop. The mathematics is the same
  and the reduction shapes differ, an `equivalent`-class difference, so the
  comparisons that use it apply that class's tolerance.


### `DecodeGeometry`

`DecodeGeometry` is public because a benchmark sizes KV buffers from it:
`research/bench_batch.py` reads it through `decode_geometry()`. Without it, a
tool would read `gen.tfmr.layers[0].self_attn.n_kv_heads` from the module's
internals and break on the first decoder refactor.

- `n_layers`: decoder layers, the first axis of a KV cache.
- `n_kv_heads`: key/value heads per layer. Under grouped-query attention there
  are fewer than query heads, so this cannot be derived from the head count.
- `head_dim`: the width of one head.
- `device`: where the weights are.
- `dtype`: their storage dtype.


### `forward_static`

`k_buf` and `v_buf` are `[1, n_kv_heads, max_len, head_dim]`. The new key and
value are written into column `pos` in place with `index_put_` and a
preallocated index grid, `grid`, so the KV write allocates nothing, every
address is fixed, and the step can be captured as a CUDA graph (the
`cuda_graphs` execution flag). `grid` is `(batch_idx, kv_idx, head_idx)`, each
`[n_kv_heads * head_dim]`, built once per generator. `pos` is a 0-dim device
buffer updated between replays.

The query attends over the whole padded buffer with a causal mask, so positions
beyond `pos` contribute exactly zero. Attending over the padding changes the
reduction order relative to the dynamic `torch.cat` path. That is an
`equivalent`-class difference: deterministic, with a small logit drift (about
1e-6), and tokens can diverge from eager decode on long sequences (see
`docs/design/execution-config.md`). The path is opt-in
through `cuda_graphs` or `compile_model`; the default path stays bit-identical.

The mask depends on `pos` and the buffer length, not on the layer, so
`LlamaDecoder.forward_static` builds it once and passes it to every layer. The
mask is what keeps the buffer's unwritten tail out of the result.


### `generate`

The sampler implements the sampling law. The generation loop applies the EOS
floor and keeps the `seen` bookkeeping. A stop token that is sampled naturally
is included in the returned sequence, so the engine can tell a natural ending
from a cap hit.

`prefix` holds speech tokens from the preceding chunk. They are fed through the
model to build context and are not part of the returned sequence. They also
seed the repetition-penalty state, because a token repeated across a chunk join
is as repeated as one within a chunk.

Under `fusion_mtp2`, pairing starts at the prefix's first token, so an
odd-length prefix drops its newest token, not its oldest. Dropping from the
front would shift every pair by one and fuse each token with the wrong partner,
the defect `window.carry_pair_aligned` prevents. Prefixes that come through the
engine are already even-length and lose nothing.

With `cuda_graphs` or `compile_model` set, decoding uses the static KV cache
described under `forward_static`.


### `check_manifest_sizes`

`speech_vocab_size` and `start_speech_token` come from the manifest on both the
torch and the ONNX path. The torch tables have fixed dimensions (8194 speech
ids, start token 6561), and `check_manifest_sizes` refuses a manifest that
declares other values when the generator is built, before the weights load.
Without the check, such a manifest would load on both backends under one
fingerprint and behave differently on each.

A mismatch is refused. A checkpoint with other dimensions needs other weights,
and guessing which side is right could make one backend speak wrongly with
nothing to report it.


### `TorchTokenGenerator`

- `config`: the algorithm. The engine compares its fingerprint with every other
  component's.
- `llama_config`: the architecture dict from the checkpoint manifest.
- `attention`: `"eager"` or `"sdpa"`, from
  `ExecutionConfig.resolved_attention()`. On MPS it must be eager, because the
  fused kernel can abort the interpreter with no traceback.


### `__init__`

Every parameter is initialised before the checkpoint overwrites it, so
`torch.manual_seed(0)` makes a build reproducible within one environment. An
uninitialised parameter reads whatever the allocator last wrote there. With
NaNs left on the heap, a model built that way produced NaN logits in about one
run in three, which looked like numerical trouble in the static-cache path.


### `self._cond_lock` and `_prefill_embeds`

The conditioning cache is a dict keyed by `VoiceProfile.cond_key`, capped at
eight entries. A lost entry costs only a recomputation, but a lookup, the LRU
promotion, an eviction and an insert are several operations. Two threads in
`_prefill_embeds` could both find a key, and the second thread's `pop` for the
LRU promotion would raise `KeyError` after the first had already evicted it.
`_cond_lock` serialises that sequence. The `_decoding` lock does not cover it,
because it is taken after the prefill and only on the static path.

Two threads that miss on the same key both compute the row. The second check,
under the lock, keeps the first insert and discards the other. Without it, both
threads would evict to make room, and a burst of N threads on one cold voice
could shrink the cache to almost nothing. The row is a pure function of the
key, so discarding the duplicate loses nothing but the computation.


### `self._graphs`

Captured decode graphs are cached by kind, KV buffer length and sampling law.
Capture costs about as long as decoding a 200-token window, and a graph depends
only on the shapes, so it outlives the utterance that built it.

Lengths are rounded up to `_GRAPH_BUCKET` (64) so that chunks of similar size
share one entry. The rounding has a cost: the query attends over the whole
padded buffer, so a longer buffer costs a little on every step (1.41 ms per
pair at 300 columns against 1.64 ms at 1200). The bucket is small so that the
padding stays small.


### `self._decoding`

Keeping graphs across calls makes the decode buffers per generator, not per
call: the KV cache, the position scalars, the pair register and the device
sampler's noise block are all addresses a captured graph holds. Two `generate`
calls running at once on one engine would write each other's tokens into each
other's slots. Nothing would crash, and both callers would get fluent speech
that is not their text.

`_decoding` refuses the second concurrent static-cache call. It does not queue
it, because a queue would hide that both callers share one engine's throughput.
The eager path allocates per call and is re-entrant, so the guard is taken only
where the buffers are shared. The bundled transports already serialise each
whole request behind one lock.


### `_static_decode`

There are three static loops: single-token, fused with host sampling, and
fused with on-device sampling. The decode mode (a property of the weights) and
whether the sampler has a device form (a property of the sampler) select one;
the caller does not. The selection is split out of `generate` so the whole
static family runs inside one acquisition of `_decoding`, which covers every
return path.


### `_pair_slot_embed`

The slot embedding is `mean(e_a, e_b) + fuse([e_a ; e_b])`, then the pair's
position embedding. The fusion MLP was initialised as a zero delta from the
mean, so the slot starts as the average of the two embeddings and the MLP
learns the correction. A single-token loop would carry only `e_b`, and that
makes an unadapted model decode a halved context and blur its phonemes.


### `_generate_fused`

The fused loop keeps every rule of the single-token loop: the EOS floor, the
`seen` bookkeeping, token-level cancellation, and a natural stop token returned
inside the sequence. Three things differ, all because a pair shares one slot:

- The sampler's `step` counts **tokens**, not forwards, so its counter-based
  RNG draws the same numbers it would for a sequence of the same length
  decoded one token at a time.
- Either head can emit the stop token. When the pair's first token ends the
  sequence, its second token is never sampled.
- Positions advance once per **pair**, because one pair is one KV slot.


### `_generate_static_fused`

The captured step runs the slot fusion, the forward and the first head, and
copies the hidden state out. The second head runs outside the graph, because
its input is the token sampled from the first head, and that dependency crosses
back through the host sampler anyway. It is one small matmul against a forward
about two orders of magnitude larger. The graph exists for the forward's
roughly 1442 kernel launches, and this path issues half as many forwards per
token as the single-token path.


### `_generate_static_fused_ondevice`

`_generate_static_fused` makes two device-to-host copies of a vocabulary of
logits per pair, because the sampler runs in NumPy and the second head needs
the token the first draw produced. Each copy forces a synchronisation. Measured
on an RTX 3090, the pair's replay takes 0.70 ms per token against 0.64 ms of
sampler and 0.80 ms of copy and Python: two thirds of the loop is outside the
model.

With token selection on the device, one replay covers the fusion, the forward,
`speech_head`, the first draw, `head2` on that draw's embedding, the second
draw, and the slot the pair leaves behind. Only the two token ids cross back to
the host, which still decides when to stop.

The sampling law does not change. `DeviceSamplerV1` in
`docs/design/sampler-and-noise.md` lists what is bit-identical: everything that
selects a token, including the Gumbel noise, which the host generates and
uploads. The one quantity that differs is the divisor of the EOS observation, a
parallel sum on the device and an ordered sum on the host. Two things change on
this path: cancellation is polled once per pair, not once per token, and the
path needs a sampler with a device form, so any other sampler falls back to
`_generate_static_fused`.


### `_fused_device_slot`

A captured slot is built once per length bucket and sampling law, then reused.
Capture costs more than a short window's whole decode. The graph's structure
and buffer addresses do not depend on the utterance; between calls only the
buffer contents change (the prefill's KV, the seed's noise, the positions), and
each is written through a tensor the graph already holds.


### `_capture_runner`

`cuda_graphs` and `compile_model` both select the same thing: the per-token
decode as one captured graph instead of about 1442 kernel launches, over the
same static KV cache. Both use the manual `torch.cuda.CUDAGraph` capture here,
because the capture path of `torch.compile` hits an Inductor mask-alignment
defect on this model. With neither flag, `step` itself is the runner. A graph
needs its buffers at fixed addresses and its input values written with `.fill_`
before each replay, which the static loops do.

Capture uses thread-local mode. In the default global mode, CUDA refuses
potentially unsafe API calls from every thread while a capture is open
("operation not permitted when stream is capturing"). Under the streaming
pipeline the render thread issues CUDA work for the previous window while this
thread captures, so the restriction is scoped to the capturing thread. In both
modes only the capturing stream's work is recorded into the graph.


### `_teacher_forced_fused`

Teacher forcing under `fusion_mtp2` is still one causal forward with one logit
row per forced token, so two backends compare row for row as they do under
`single`. Only the source of a row changes: a pair's first token is read from
`speech_head` at the preceding slot, and its second from `head2` at that same
slot, with the first token's embedding appended. That is the pair the decode
loop samples.


### Module casts

Where a module's own `forward` returns a Tensor, the call is wrapped in
`cast(Tensor, ...)`. The cast states torch's return type for the type checker.
See `docs/design/typing.md`.


## `loudkit/models/resample.py`


### `module`

Enrollment downsamples the reference clip from 24 kHz to 16 kHz with one
resampler, this one, and every port reimplements the same law. It is the
algorithm of torchaudio's `sinc_interp_hann` (a Hann-windowed sinc,
band-limited interpolation) with an explicit contract, so the five
implementations stay bit-identical:

- The kernel is computed in float64 from the formula in `sinc_hann_kernel`,
  then rounded to float32 once. The float32 values are the contract, not the
  float64 intermediates.
- The FIR accumulates **left to right in float32**, with one multiply and one
  add per tap and never a fused multiply-add. An FMA rounds differently, and
  the divergence would be silent.

After GCD reduction (24k/16k = 3/2) the kernel has 2 phases of 23 taps, so the
resampler is a 23-tap strided FIR. The float32 kernel is small enough to ship
as data, but it is computed here so that the definition is self-contained.

A compiler can remove the no-FMA rule without warning. Go contracts a multiply
into the following add on arm64 and not on amd64, so the same source computes
different last bits per architecture. The Go port writes the rounding out at
the accumulation, so the compiler cannot make that choice. Measured against the
enrollment fixture, contraction moves 29% of the 16 kHz samples by up to one
float32 ulp, and the difference grows through the filterbanks to 3.5e-04 on the
speaker embedding, a value a voice profile stores.
`TestResampleIsBitExactAgainstTheFixture` holds the FIR to byte equality, not to
a tolerance, because the difference it guards is one ulp wide.

Two accumulations elsewhere in the ports look like the same hazard and are not.
An L2 norm over an embedding squares a float32 widened to float64, and that
product is exact: 24 significand bits times 24 is at most 48, and float64
carries 53, so no rounding exists for a fused operation to skip. Those are left
as written.


### `resample`

The loop is vectorised across output samples, never across taps: each output's
taps accumulate in ascending order, one float32 addition at a time, as the
docstring specifies. `np.dot` would sum in whatever order BLAS prefers and stop
matching the four ports. Measured, the result is bit-identical with a naive
triple loop (max difference 0.0 on one second of 24 kHz noise), and the naive
loop costs 0.18 s of Python per enrollment.


## `loudkit/models/timestretch.py`


### `module`

`speed` means what it means on a media player: 1.5x is the same voice,
finished sooner. Resampling would change the pitch with the tempo. This module
stretches time and leaves the pitch alone.

**WSOLA and the phase vocoder.** A phase vocoder does well on sustained,
harmonic material such as held notes and chords. Speech is mostly transients
(plosives, the attack of every syllable) on a pitch that moves continuously. A
phase vocoder resynthesises from magnitudes and unwrapped phases, and on that
material its typical failure is transient smearing (a /t/ that arrives as a
soft thud) and "phasiness" on voiced segments, the parts intelligibility
depends on. WSOLA stays in the time domain: it copies waveform segments and
chooses only where to copy them from, so it avoids the phase vocoder's
transient smearing. The overlap-add and a poor alignment can still add
artefacts, mostly at large speed changes (see below).

**The algorithm.** The input is cut into overlapping frames of about 25 ms.
Frames are written out at a hop fixed by the output rate (50% overlap) and read
in at a hop scaled by `speed`. The read position is then moved by up to ±10 ms
to the offset that best matches what the previously written frame would
naturally be followed by. That search is the "waveform similarity" in the name.
It keeps successive frames in phase, so the overlap-add adds up instead of
cancelling. A plain OLA is the same code with the search window set to zero,
and it produces a periodic warble at the frame rate.

The algorithm uses no random numbers. The constants are derived from the sample
rate, not written as sample counts, so the same code is correct at 16 kHz or
48 kHz, and the five implementations derive them the same way.

**What it costs.** By ear, 1.25x is hard to tell from a natural reading. At 2x,
or at 0.5x, the result is audibly processed: the alignment search does not
always find a match, and the artefact is a faint roughness or a doubled
consonant. `MIN_SPEED` and `MAX_SPEED` (0.5 and 2.0) mark where the result
stops being usable. The arithmetic itself runs outside that range.


### `time_stretch`

- `audio`: mono samples.
- `sample_rate`: the samples' rate. The frame, hop and search window are
  derived from it.
- `speed`: above 1 shortens, below 1 lengthens. At `1.0` the function returns
  the input array itself, not a copy. The engine's default must bypass this DSP
  path entirely, and returning the same object shows that no arithmetic ran.

Returns `floor(len(audio) / speed + 0.5)` samples.

A fragment no longer than one frame (600 samples at 24 kHz, a fortieth of a
second, below anything the engine renders) is cut or zero-padded to the target
length, not stretched. A zero hop takes the same branch, so it cannot hang the
loop; a zero hop needs a sample rate below 60 Hz.


### `_best_match`

Candidates are scored by cross-correlation normalised by the candidate's
energy only. The target's energy is the same for every candidate and does not
change the ranking. Without the normalisation the search prefers the loudest
candidate over the best fit, which at a syllable onset is the wrong choice.

A tie goes to the lower offset, so the result does not depend on iteration
order, and the five implementations agree.


## `loudkit/models/vocoder.py`


### `module`

Checkpoint namespace `s3gen.mel2wav`, names mirrored. Weight norm is folded at
pack time, so the convolutions are plain convolutions that carry the exact
tensors the parametrised forward computes.

Signal path: mel -> f0 (a small conv net) -> harmonic sine excitation at
24 kHz (nine harmonics, cumulative-phase NSF source) -> STFT of the excitation,
fused into the HiFiGAN upsampling stack at every scale -> predicted magnitude
and phase -> iSTFT -> waveform.

**fp32 only.** The NSF source accumulates phase with a running `cumsum` that
reaches roughly 1400 cycles over a ten-second render. At that magnitude fp16's
resolution is coarser than the per-sample phase increment, the excitation
degenerates, and the result is an audible tone at Nyquist, measured as a
~12 kHz whine in 80% of frames. This is a property of the algorithm, not of a
backend, so the refusal is in the module: `TorchVocoder.half()` raises, and
`TorchVocoder.to()` refuses fp16 and bf16. The torch backend also refuses any
`precision["vocoder"]` other than fp32.

The harmonic phase offsets and the excitation noise are Philox-addressed data
from `.noise`, so the same seed produces the same excitation on every device.
The noise is drawn fresh for every sample, and the second Box–Muller output is
never cached for reuse. A cached-spare variant measured +5.3 dB at Nyquist; see
`docs/design/sampler-and-noise.md` for the noise design.


### `TorchVocoder`

With the full static padding, the mel is zero-padded to `2 x max_speech_tokens`
frames before rendering, and the waveform is trimmed back to the real region
afterwards, the same framing as the exported HiFT graph. The padding at the
tail reaches a few frames back into the kept audio through the conv stack, so
it is part of the algorithm, not an export artefact.

With `vocoder_ragged`, the mel is padded only to its length plus
`VOCODER_RIGHT_CONTEXT` frames, rounded up to `VOCODER_LENGTH_BUCKET` and capped
at the static window (see `docs/design/execution-config.md`).


### `VOCODER_LENGTH_BUCKET`

Ragged mel lengths are rounded up to 64 frames. A shape the convolution library
has not seen is one it either runs with a poor algorithm or stops to autotune,
and either costs more than the frames the rounding wastes. Measured on a
desktop GPU over a multi-chunk read, ragged lengths with autotuning ran at 6.85x
against 12.49x for the same lengths without it, because the autotuner re-ran for
each distinct shape. The rounding reduces "one shape per chunk" to a handful of
shapes.


### `VOCODER_RIGHT_CONTEXT`

The value is 32 mel frames, set by measurement, because the stack ends in an
iSTFT and a derivation through it is easy to get wrong. Padding by k frames
and comparing with the full-window output, the difference falls to 1e-6 at
k = 16 and stops improving after that; 1e-6 is fp32 noise from cuDNN choosing
different algorithms for different widths. The constant is double the measured
need, because 16 extra frames of mel cost little and a shortfall would sound
like a click at every chunk join.


## `loudkit/models/windowing.py`


### `module`

The parts of the flow and vocoder modules that a renderer backend needs and
that are not torch modules: the window framing recipe, the Euler time grid, and
the Philox substream ids that address the render randomness. The ONNX and
CoreML backends import this file and never import a torch module.

The window recipe is data here, shared by every renderer, not derived again per
backend. Framing is where implementations drift: in the measured CoreML (Neural
Engine) against torch comparison, a framing mismatch accounted for the whole
mel deviation, with correlation 0.975 to 0.993.


### `frame_windows`

Returns `(token_row (1, P+Q), cond (1, 80, 2·(P+Q)), prompt_frames, n)`, where
`n` is the count of real speech tokens and `prompt_frames` is the mel region to
cut after integration. In static mode the prompt is framed to exactly
`static_prompt_tokens` (the start kept when long, silence-padded when short)
and the query to `static_length`, the production recipe.

A sequence longer than the window is refused with `WindowOverflowError`, not
truncated. Truncation would turn 300 tokens in into 255 tokens of audio with
nothing to say the rest was lost, and only a listener who knows the text would
notice. A direct `decode` caller must split longer sequences first; the engine
already splits text into chunks.


## Renderer batching

`TorchMelDecoder.decode_batch` and `TorchVocoder.synthesize_batch` are
prototypes beside the single-utterance methods, and the engine does not call
them. The measurements below were taken on 2026-09-03 with
`research/bench_render.py`, which renders one real utterance's tokens and mel N
times, serially and in one batched call. In summary:

- batching the renderer shows no stable gain on MPS;
- it gains 1.11x to 1.24x on CPU, where synthesis is latency-bound;
- a batched row does not reproduce its own single call;
- the engine and the transports are single-flight, so batching needs a
  scheduler first.

No serial-against-batched measurement on CUDA is recorded here.


### The `N vs 1` timing check

`N vs 1` is the wall time of N serial single calls divided by one single call.
On a quiet machine it is close to N whatever the device's utilisation, because
each serial call does its own work and pays its own overhead. It is a check on
the timing, not a measure of saturation or of what batching gains: a row well
above its own batch size was interrupted or throttled. The serial-against-batched
speedup in the next section is the measure of batching.

Batch 4, warm, median of several rounds:

| model | device | vocoder | mel decoder |
|---|---|---|---|
| loudr-1 | Jetson Orin (CUDA) | 4.00x | 4.00x |
| loudr-1 | RTX 3090 (CUDA) | 3.30x | 3.91x |
| loudr-1 | MPS | 3.98x | 3.68x |
| loudr-1-turbo | MPS | 3.70x | 4.10x |
| loudr-1 | M-series CPU | see below | see below |

Repeated runs of loudr-1 on MPS put the mel between 3.68x and 4.11x and the
vocoder between 3.98x and 4.10x, so a single figure here varies by about 10%.
Machine load affects both the single-call baseline and the N-call timing. The
CPU runs were taken while other jobs ran, so the CPU figures are given only in
the speedup table below.

For comparison, the token generator does gain from batching. On the same RTX
3090, `research/bench_batch.py` measures 2.8x the aggregate throughput of batch
1 at batch 8 for loudr-1 (see `docs/benchmarks.md`). That benchmark is a
forced-token decoder measurement of batched throughput and says nothing about
the renderer.


### Batched against serial

Speedup is the serial loop's wall time divided by the batched call's wall time.
Above 1.00x, the batched call is faster.

| model | device | stage | batch 2 | batch 4 | batch 8 |
|---|---|---|---|---|---|
| loudr-1-turbo | MPS | mel | 1.02x | 1.05x | 1.04x |
| loudr-1-turbo | MPS | vocoder | 0.94x | 0.98x | 0.68x |
| loudr-1 | MPS | mel | 1.12x | 1.07x | 1.24x |
| loudr-1 | MPS | vocoder | 0.50x | 1.00x | 1.13x |
| loudr-1 | CPU | mel | 1.13x | | |
| loudr-1-turbo | CPU | mel | 1.15x | 1.22x | |
| loudr-1-turbo | CPU | vocoder | 1.11x | 1.24x | |

Repeated MPS runs disagree by more than most effects in the table: the loudr-1
vocoder at batch 2 measured 0.95x once and 0.50x another time, and the mel at
batch 8 measured 0.75x and 1.24x. These runs show no stable gain from batching
the renderer on MPS. More controlled repetitions would be needed to estimate
any small effect.

Batching trades per-request throughput for aggregate throughput. Turbo's mel
runs at about 36x real time batched or not. At batch 8 the aggregate throughput
stays about the same, while per-request throughput falls from 36x to 4.6x,
because every request in a batch waits for the whole batch.

CPU is the only device where a gain reproduces: 1.11x to 1.24x. On the measured
machine the loudr-1 mel takes 15.0 s for 5.0 s of audio, 0.34x real time for
one stage before the vocoder or the token generator runs, so a CPU deployment
is latency-bound end to end. A likely source of the gain is the CPU estimator's
many small operations, whose dispatch and thread-synchronisation overhead a
wider tensor amortises; this has not been profiled.

The loudr-1 CPU row covers batch 2 only, and both models' CPU rows were taken
while other jobs ran. `research/bench_render.py` records the load average at
both ends of a run, so a suspect row can be identified. Re-take the CPU rows on
a quiet machine before relying on them.


### A batched row does not reproduce its own single call

A batch of one is bit-identical to the single call, on CPU and on MPS, for both
stages. From two rows up, the result depends on the shape, the device and the
thread count, and in the configurations tested no single setting made every
case byte-equal:

| stage | device | shape | batched row against its own single call |
|---|---|---|---|
| mel | MPS | production | identical |
| mel | CPU | production | 1.6e-2 max, 2.9e-3 rms, mel corr 0.9999994 |
| mel | CPU | production, `set_num_threads(1)` | identical |
| mel | CPU | 96-frame test window | 2.6e-6 max |
| vocoder | CPU | production | identical |
| vocoder | CPU | 96-frame test window | 1.8e-7 max |
| vocoder | MPS | production | 1.5e-6 max, 8.4e-8 rms |

"Production" is the benchmark utterance at the production window, a 252-frame
mel. The two test-window rows use the shapes `tests/test_models.py` runs, and
they are why that file bounds correlation and not bytes: the deviation there is
about a thousand times smaller than at the production shape, so agreement at
96 frames does not establish the same bound at 252 frames.

Two experiments point to the intra-op reduction order as the cause. Two
identical rows inside one batched CPU call differ from each other by the same
1.6e-2, and `torch.set_num_threads(1)` removes the difference at those shapes,
which points to parallel reduction order and not to row addressing. The MPS
vocoder behaves differently: its rows agree with each other inside a batch, and
the whole batch differs from the batch-1 call, which is consistent with Metal
choosing a different kernel for a wider tensor. Neither has been profiled.

The mel estimator runs in fp16 here (`TorchMelDecoder.estimator_dtype`). Near
this mel's peak value of 11.9, the fp16 spacing is 7.8e-3, so the worst cell
moved by about two fp16 steps at that magnitude. 44% of cells moved, by an
average of 7.8 fp16 steps.

Through the vocoder, the difference is larger than the mel figure suggests:

| through the vocoder | max sample | corr | error to signal |
|---|---|---|---|
| batched mel against its own single call | 0.105 | 0.99871 | -25.9 dB |
| single call, 5 threads against 1 thread | 0.042 | 0.99979 | -33.7 dB |

The second row puts the first in proportion: the CPU renderer does not
reproduce itself across thread counts before any batching. The full engine
shows the same effect. Two processes running `engine.synthesize`, one with one
thread and one with five, differ by 4.7e-2 at -28.2 dB. Batching widens an
existing gap and does not open a new one.
`loudkit.backends.torch_backend.pin_determinism` applies
`ExecutionConfig.num_threads` for this reason. At a fixed thread count, every
path measured here is deterministic, batched or not.

Making a batched renderer the production path would change the rendered bytes,
so every golden and conformance vector taken on the single path would have to
be re-taken, and the new goldens would still depend on the thread count.


### The engine takes one utterance

`MelDecoder.decode(tokens, voice, seed)` and `Vocoder.synthesize(mel, voice,
seed)` in `contracts.py` take one utterance each, and `Engine.synthesize` is one
call. The HTTP, gRPC and MCP transports each hold a single-flight lock over the
engine, so two requests never reach the renderer at the same time. The chunks
of one passage are generated in order, because each chunk's token generation
is conditioned on the previous chunk's tokens (the prefix carry), and each
chunk is rendered as soon as its tokens exist.

A batched renderer would need a scheduler:

- a queue that gathers requests across callers, in place of the single-flight
  lock;
- a grouping policy, because both batched methods refuse rows of different
  padded lengths. Padding a short row to a longer neighbour's length changes
  the short row's own output: the conformer attends over the whole unmasked
  row, and the vocoder's excitation and iSTFT tail depend on the padded length;
- a latency budget for how long a request may wait to be gathered.

On the measurements above, that scheduler gains nothing stable on MPS and at
most a quarter on a CPU that is latency-bound anyway, at several times the
per-request latency and with bytes that no longer match the goldens.


### When to reopen this

Reopen renderer batching when a controlled serial-against-batched measurement
on the target device shows a throughput gain that meets the deployment's
per-request latency and numerical requirements. `research/bench_render.py` runs
on CUDA unmodified.

The scheduler comes before the batch. The token generator is the better first
target for a scheduler: it is launch-bound, and batch 8 already gives 2.8x the
aggregate throughput of batch 1 on the RTX 3090. The renderer can join that
scheduler later if its own measurement supports it.
