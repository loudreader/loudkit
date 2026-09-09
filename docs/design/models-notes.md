# Model modules

Maintainer notes moved out of the runtime's docstrings: the reasoning and
the measurements behind each symbol. Each heading names the module and
the symbol the note belongs to. Not user documentation.


## `loudkit/models/__init__.py`


### `module`

One module per pipeline stage, mirroring :mod:`loudkit.contracts`:

- :mod:`.text`, text normalisation + the multilingual grapheme tokenizer
- :mod:`.generator`, the Llama-architecture token generator (T3 student)
- :mod:`.flow`, conditional flow matching, tokens to mel
- :mod:`.vocoder`, HiFT (HiFiGAN + NSF source), mel to waveform
- :mod:`.enroll`, reference audio to a :class:`~loudkit.voice.VoiceProfile`
- :mod:`.noise`, render randomness as *data*, addressed by the Philox counter

The torch modules in here mirror the packed checkpoint's tensor names exactly
(``t3.*`` / ``s3gen.*`` with the prefix stripped), so weights load with
``strict=True`` and a naming drift fails at load time instead of rendering
plausible-but-wrong audio.


## `loudkit/models/enroll.py`


### `module`

Deliberately separate from synthesis. It is slow, it needs ~40% of the
checkpoint that rendering never touches, and its output is a small file that
can be produced once on a fast machine and shipped. Four models cooperate:

- the **S3 speech tokenizer** (whisper-style FSMN encoder + FSQ quantiser,
  checkpoint namespace ``s3gen.tokenizer``): reference audio to the 25 Hz
  speech tokens that become the mel decoder's prompt and, truncated to 150,
  the token generator's conditioning prompt;
- the **CAM++ x-vector encoder** (``s3gen.speaker_encoder``): the 192-d
  speaker vector the flow conditions on;
- a **mel extractor** (matcha recipe: 24 kHz, n_fft 1920, hop 480, Slaney
  mels, log-compressed): the prompt mel;
- the **utterance voice encoder** (a 3-layer LSTM over 40-mel partials): the
  256-d vector the token generator was trained against. Its weights are *not*
  in the packed checkpoint, they were never part of the s3gen/T3 artifacts -
  so it is an optional constructor argument and enrollment without it fails
  with an error that says exactly what to pass.

The architecture ports here are adapted from CosyVoice, S3Tokenizer and
3D-Speaker (Apache-2.0), FunASR and Real-Time-Voice-Cloning (MIT), stripped to
the inference path; module names mirror the checkpoint so weights load strict.
Those are five separate projects under two licences, not one name, see NOTICE,
which carries each with its own holder.

Determinism note: enrollment is deterministic (no sampling anywhere), so the
same clip always yields the same profile, which is why profiles can be
compared across machines at all.


### `enroll`

The input contract is :func:`validate_reference_audio`'s: 5 to 10
seconds is right, 30 seconds is the refusal line. The clip is used at
two rates: 24 kHz for the prompt mel, 16 kHz for tokenisation and both
speaker encoders. Input past ten seconds is truncated for the prompt -
the static prompt window holds ~9.5 s, and everything past it would be
enrolled and then cut on every render, while the speaker embedding
reads the whole clip.

The 24->16 kHz downsample uses **one** resampler (see
:mod:`loudkit.models.resample`), not two: the reference pipeline once
split this between torchaudio and librosa's ``soxr_hq``, but the latter
is a C library no port can reproduce bit for bit, so enrollment is
unified on a single, portable Hann-windowed-sinc law. That means the
tokens differ from the historical two-resampler enrollment, and the
reference voices are re-enrolled against this law, there is nothing to
be faithful to that five languages could not all reach.

Raises:
    ValueError: for input outside the contract, non-positive rate,
        non-mono shape, NaN or Inf samples, silence, too short or over
        30 seconds. The message says what a good input looks like.


## `loudkit/models/flow.py`


### `module`

Architecture (checkpoint namespace ``s3gen.flow``, names mirrored so weights
load strict): a token embedding, an upsampling conformer encoder (6 blocks at
25 Hz, nearest-neighbour x2 upsample, 4 more blocks at 50 Hz) that produces
the mean field ``mu``, and a 1-D U-Net estimator (1 down / 12 mid / 1 up
stages of causal resnet + transformer blocks) that predicts the flow velocity.
Two Euler steps integrate noise into a mel, the estimator was step-distilled
from a six-step teacher, and its guidance was distilled *into* the weights,
which is why ``single_path`` is the shipping mode and ``cfg_dual_path`` exists
only to drive an undistilled teacher.

Everything algorithm-shaped is read from :class:`AlgorithmConfig` and nowhere
else. Three of those decisions deserve names:

* **Guidance mode comes from the config** (EXP-016: the upstream class carried
  ``inference_cfg_rate = 0.7`` as a buried default and every torch bench ran
  guidance-on-guidance for a whole campaign, a 0.979 mel-correlation defect
  that no output check caught).
* **The time grid is cosine**, ``t_i = 1 − cos(i/K · π/2)``, the schedule the
  students were distilled against and the one the shipped engine runs. The
  upstream ``meanflow`` branch integrates a *linear* grid, which is one more
  way the torch path deviated from what ships.
* **The window recipe is the shipped static one** when ``WindowConfig`` says
  so: query padded to 255 and prompt framed to exactly 238 tokens with the
  silence unit, mel condition zero-padded to 986 frames, no masks. The recipe
  is the entire measured ANE-vs-torch mel deviation (corr 0.975–0.993), so it
  is configuration, not backend folklore.

The flow prior is Philox-addressed data (:mod:`.noise`), not device RNG state:
the same seed draws the same prior on every device and backend. Unseeded, two
renders of identical tokens correlate at 0.109.


## `loudkit/models/generator.py`


### `module`

This is the T3 student, 16 decoder layers, hidden 1024, 16 query heads of 64
with 4 KV heads (GQA), SwiGLU MLPs, RoPE with the llama3 long-context scaling,
and a speech head over an 8194-token vocabulary. Implemented directly in torch
rather than through ``transformers`` so that the arithmetic is visible, the
attention implementation is selectable (the fused path aborts the interpreter
on MPS, no traceback, just ``LLVM ERROR`` from ``mps_matmul``), and the
dependency surface stays small. Module attribute names mirror the packed
checkpoint (``t3.*``), so weights load with ``strict=True``.

Sequence layout, identical to the shipped engine's runner::

    [ cond (34) | [START] text [STOP] | speech: START, s0, s1, ... ]

where cond = speaker projection (1) + perceiver-resampled speech prompt (32) +
emotion (1). Text and speech carry *learned* positional embeddings on top of
their token embeddings, each restarting at zero, RoPE handles relative order
inside the transformer, the learned tables tell it which segment it is in.

Two behaviours are deliberate and worth naming:

* **The EOS floor is applied here**, before the sampler sees the logits: the
  stop token is masked to −inf until the configured minimum length (production:
  ``max(10, 1.2 x text tokens)``). Sampler exemptions cannot express "forbid
  one token for a while", and the floor is an EOS *policy*, which the algorithm
  layer owns.
* ``teacher_forced_logits`` runs one causal forward over the whole forced
  sequence rather than replaying the decode loop. Same mathematics, different
  reduction shapes, an *equivalent*-class difference, which is exactly the
  tolerance the comparisons that use it are designed to absorb.


### `DecodeGeometry`

Published because the batch benchmark needed it and was reading it out of
``gen.tfmr.layers[0].self_attn.n_kv_heads``, a tool reaching three levels
into a module's internals, which then breaks the first time the decoder is
refactored and says nothing about why. Benchmarking a new target is a real
use, so the numbers a benchmark needs are part of the surface.

Attributes:
    n_layers: decoder layers, and so the first axis of a KV cache.
    n_kv_heads: key/value heads per layer, fewer than query heads under
        grouped-query attention, which is why this cannot be derived from
        the head count.
    head_dim: width of one head.
    device: where the weights live.
    dtype: what they are stored as.


### `forward_static`

``k_buf``/``v_buf`` are ``[1, n_kv_heads, max_len, head_dim]`` and the
new key/value is written into column ``pos`` in place via
``index_put_`` with a **preallocated** index grid ``grid``, so no
tensor is created on the critical path, every address is fixed, and
the whole step can be captured by a CUDA graph (the ``cuda_graphs``
execution flag). ``grid`` is ``(batch_idx, kv_idx, head_idx)``, all
``[n_kv_heads * head_dim]``, built once per generator; ``pos`` is a
0-dim device buffer updated between replays.

The query attends over the whole padded buffer with a causal mask, so
positions beyond ``pos`` contribute exactly zero. Attending over the
padding changes the reduction order relative to the dynamic
``torch.cat`` path, an *equivalent*-class difference (deterministic,
logit drift ~1e-6, same tokens in practice), which is the identity
contract's own classification for a static KV cache. It is opt-in via
``ExecutionConfig.cuda_graphs``; the default path stays bit-identical.

The mask is a function of ``pos`` and the buffer length, not of the
layer, so ``LlamaDecoder.forward_static`` builds one and hands it to
every layer. It is the only thing keeping the buffer's unwritten tail
out of the answer.


### `generate`

The sampler owns the law; this loop owns only the EOS floor and the
``seen`` bookkeeping. The stop token, when it fires naturally, is
*included* in the returned sequence so the engine can distinguish a
natural ending from a cap hit.

``prefix`` holds speech tokens from the preceding chunk. They are fed
through the model to build context and then dropped from the result:
the caller asked for this chunk, not the previous one. They also seed
the repetition-penalty state, since a token repeated across a join is
just as repeated as one repeated within a chunk.

With ``cuda_graphs`` or ``compile_model`` set, the per-token decode
step runs over a **static KV cache**: preallocated buffers written in
place, so the step has fixed addresses and can be captured as a CUDA
graph (or compiled). That changes the attention reduction order, an
*equivalent*-class difference (deterministic, logit drift ~1e-6, same
tokens in practice), exactly what the identity contract sanctions for a
static cache. The default path below is unchanged and bit-identical.


### `_generate_static_fused_ondevice`

:meth:`_generate_static_fused` pays two device-to-host copies of a
vocabulary of logits per pair, because the sampler runs in NumPy and the
second head needs the token the first draw produced. Each copy is a
forced synchronisation, and measured on a 3090 the pair's replay is
0.70 ms per token against 0.64 ms of sampler and 0.80 ms of copy and
Python, two thirds of the loop spent outside the model.

With selection on the device the data dependency never leaves the card.
One replay covers the fusion, the forward, ``speech_head``, the first
draw, ``head2`` on that draw's embedding, the second draw, and the slot
the pair leaves behind; what crosses back is two integers, because the
host still decides when to stop.

The law does not change, see :class:`~loudkit.sampler.DeviceSamplerV1`
for what is bit-identical (everything that selects a token, including
the Gumbel noise, which is generated by the host and uploaded) and the
one quantity that is not (the EOS observation's divisor, a parallel sum
where the host's is ordered). Two things do change and are the price:
cancellation is polled per pair rather than per token, and this path
needs a sampler that can describe itself on a device, so anything else
falls back to :meth:`_generate_static_fused`.


## `loudkit/models/resample.py`


### `module`

Enrollment downsamples the reference clip from 24 kHz to 16 kHz, and it used
to do so through **two different** resamplers: torchaudio's polyphase
``sinc_interp_hann`` on the flow side and librosa's ``soxr_hq`` on the
token-generator side. That split is an accident of history, not a feature -
the two are both anti-aliased and differ only by a hair, and it is fatal to
cross-language parity, because ``soxr_hq`` is a C library whose float
accumulation order no port can reproduce bit for bit.

So enrollment uses **one** resampler, this one, and every port reimplements
this exact law. It is the same algorithm as torchaudio's ``sinc_interp_hann``
(a Hann-windowed sinc, band-limited interpolation) restated with an explicit
contract so the five ports stay bit-identical:

* the kernel is computed in float64 from the formula below, then rounded to
  float32 once, the float32 values are the contract, not the float64
  intermediates;
* the FIR accumulates **left to right in float32**, one multiply and one add
  per tap, never a fused multiply-add (an FMA would round differently and the
  divergence would be silent, exactly the failure this library exists to end).

The kernel is 2 phases by 23 taps after GCD reduction (24k/16k = 3/2), so the
whole thing is a 23-tap strided FIR, small enough that the float32 kernel can
be shipped as data, but computed here so the definition is self-contained.

The no-FMA clause is the one a compiler can take away without saying so. Go
contracts a multiply into the following add on arm64 and not on amd64, so the
same source computes different last bits per architecture; the Go port writes
the rounding out at the accumulation to keep the choice from the compiler.
Measured against the enrollment fixture, contracting moves 29% of the 16 kHz
samples by up to one float32 ulp, and the drift does not stay there: it grows
through the filterbanks to 3.5e-04 on the speaker embedding, which is a number
a voice profile stores. `TestResampleIsBitExactAgainstTheFixture` holds the FIR
to byte equality rather than to a tolerance, because the gap it guards is one
ulp wide and any tolerance loose enough to be useful admits it.

Two accumulations elsewhere in the ports look like the same hazard and are not.
An L2 norm over an embedding squares a float32 widened to float64, and that
product is exact: 24 significand bits times 24 is at most 48, and float64
carries 53, so there is no rounding for a fusion to skip. Those are left as
written.


## `loudkit/models/timestretch.py`


### `module`

"Speed" in a reading app means what it means on a video player: 1.5x is the same
voice, sooner. Resampling gives you a chipmunk; what is wanted is *time*
stretched while *pitch* is left alone.

**Why WSOLA and not a phase vocoder.** The phase vocoder is the other standard
answer and is better on sustained, harmonic material, held notes, chords. Speech
is the opposite kind of signal: it is mostly transients (plosives, the attack of
every syllable) sitting on a pitch that moves continuously. A phase vocoder
resynthesises from magnitudes and unwrapped phases, and its characteristic
failure on that material is transient smearing, a /t/ arriving as a soft thud,
"phasiness" on voiced segments, which is precisely the part of speech
intelligibility rests on. WSOLA never leaves the time domain: it copies real
waveform segments and only chooses *where* to copy them from, so a plosive is
either included whole or not at all. It cannot smear what it never transforms.

**The algorithm.** Cut the input into overlapping ~25 ms frames. Write them back
out at a hop that is fixed by the output rate (50 % overlap), and read them in at
a hop scaled by ``speed``. The read position is not used as computed: it is moved
by up to ±10 ms to whichever offset best matches what the previously written
frame *would* naturally have been followed by. That search is the "waveform
similarity" in the name, and it is the whole trick, it keeps successive frames
in phase with each other, so the overlap-add reinforces rather than cancels. A
plain OLA without the search is the same code with the search window set to zero,
and it sounds like it: periodic warble at the frame rate.

Everything here is deterministic, no RNG, no adaptivity, no libraries. The
constants are derived from the sample rate rather than written as sample counts,
so the same code is correct at 16 kHz or 48 kHz, and the five implementations
derive them the same way.

**What it costs.** At 1.25x this is hard to tell from a native reading. At 2x, or
at 0.5x, it is audibly processed: the alignment search cannot always find a
match, and the artefact is a faint roughness or a doubled consonant. That is the
honest range, and the bounds below are set where the result stops being worth
offering rather than where the arithmetic stops working.


## `loudkit/models/vocoder.py`


### `module`

Checkpoint namespace ``s3gen.mel2wav``, names mirrored, weight-norm already
folded at pack time, the convolutions here are plain convolutions carrying
the exact tensors the parametrised forward would have computed.

Signal path: mel -> f0 (a small conv net) -> harmonic sine excitation at
24 kHz (nine harmonics, cumulative-phase NSF source) -> STFT of the excitation
fused into the HiFiGAN upsampling stack at every scale -> predicted magnitude
and phase -> iSTFT -> waveform.

**fp32 only, enforced at construction.** The NSF source accumulates phase with
a running ``cumsum`` that reaches roughly 1400 cycles over a ten-second
render; at that magnitude fp16's resolution is coarser than the per-sample
phase increment, the excitation degenerates, and the result is an audible tone
at Nyquist (EXP-003 measured it as a ~12 kHz whine present in 80% of frames).
This is a property of the algorithm, not of any one backend, so the refusal
lives here rather than in a backend checklist.

Randomness (the harmonic phase offsets and the excitation noise) is
Philox-addressed data from :mod:`.noise`, the same seed produces the same
excitation on every device, and the noise is drawn fresh per sample (no
Box–Muller spare caching; the cached-spare variant put +5.3 dB at Nyquist).


## Notes moved from the runtime docstrings


## `loudkit/models/enrollment_audio.py`


### `validate_reference_audio`

The contract, stated once and enforced here for every caller, the CLI's
``clone``, :func:`loudkit.enroll`, and a port checking its own input the
same way: mono, finite samples, between :data:`_MIN_ENROLL_SECONDS` and
:data:`_MAX_ENROLL_SECONDS`, and not silence. Recommended input is 5 to
10 seconds; the prompt is built from the first 10 and the speaker
embedding reads the whole clip, which is why a long recording is refused
instead of quietly enrolling something the docs do not describe.

Raises:
    ValueError: with a message that says what a good input looks like.


## `loudkit/models/flow.py`


### `_Estimator`

Stages are stored as nested ``ModuleList``s, ``[resnet, transformers,
tail-conv]``, exactly the containers the original used, so parameter
names match the checkpoint without any remapping.

Geometry note: with a single 256-channel level the "down" tail conv is
stride-1, so nothing is actually downsampled; the skip connection
concatenates same-length features. The names stay, the shapes never
change, and the causality of every conv is real.


### `TorchMelDecoder`

Args:
    config: the algorithm, guidance mode, Euler grid, window recipe.
    estimator_dtype: compute dtype for the estimator only (fp16 is
        measured safe there: mel corr 0.999999). The encoder half of this
        module refuses fp16 outright, measured mel corr 0.619 with
        +22 dB of high-frequency energy, so precision is per-module here,
        exactly as ``ExecutionConfig.precision`` declares it.


### `_extend`

Mutates ``self._pe`` during inference, which is a latent data race: two
threads calling ``synthesize`` on one engine can both find the buffer
short and both rebuild it, and the reader of the smaller one indexes a
tensor that has been replaced underneath it.

Serialised rather than precomputed: the table is
``(1, 2·length - 1, d_model)`` and ``length`` is a passage's token
count, so sizing it for the worst case would allocate for a passage
no caller asked for. The lock is uncontended on the single-flight path the
server and the CLI both use, it costs an atomic per call there and
makes the public API safe for the caller who does not serialise.


## `loudkit/models/generator.py`


### `check_manifest_sizes`

The ONNX backend reads ``speech_vocab_size`` and ``start_speech_token`` from
the manifest and shapes its inputs accordingly; the torch path has them as
constants, because they are the dimensions of *these* tables. A manifest
carrying different values therefore loaded on both backends under one
fingerprint and behaved differently on each, the divergence class this
library exists to prevent, arriving through data rather than code.

Refused rather than reconciled: whichever side is wrong, a checkpoint with
other dimensions needs other weights, and guessing which to believe is how
one of them starts speaking wrongly with nothing flagging it.


### `TorchTokenGenerator`

Args:
    config: the algorithm. Never copied, never defaulted, the engine
        checks its fingerprint against every other component's.
    llama_config: architecture dict from the checkpoint manifest.
    attention: ``"eager"`` or ``"sdpa"``, from
        ``ExecutionConfig.resolved_attention()``. On MPS this must be
        eager; the fused kernel kills the interpreter with no traceback.


### `_pair_slot_embed`

``mean(e_a, e_b) + fuse([e_a ; e_b])``, then the pair's position. The
mean is the identity the fusion MLP was initialised as a zero-delta
from, so the slot starts as the average of the two embeddings and the
MLP learns how far to move it; carrying only ``e_b``, the shape a
single-token loop would produce, is what makes an unadapted model
decode a halved context and blur its phonemes.


### `_static_decode`

Three of them exist and the choice is not the caller's: the decode mode
is a property of the weights, and whether the sampler can describe
itself on a device is a property of the sampler. Split out of
``generate`` so the whole static family sits inside one acquisition of
:attr:`_decoding`, a guard taken in the caller and released three
returns later is a guard somebody eventually leaks.


### `_generate_fused`

Everything the single-token loop owns is owned here identically, the
EOS floor, the ``seen`` bookkeeping, token-level cancellation, and
returning a natural stop token inside the sequence. Three things differ,
and all three are consequences of a pair sharing one slot:

* the sampler's ``step`` counts **tokens**, not forwards, so its
  counter-based RNG draws the same numbers it would for a sequence of
  the same length decoded one at a time;
* either head can emit the stop token, and the pair's first ending the
  sequence means its second is never sampled;
* positions advance once per **pair**, because one pair is one KV slot.


### `_generate_static_fused`

The captured step does the slot fusion, the forward and the first head,
and also copies the hidden state out. The second head stays outside the
graph because its input is the token sampled from the first, a data
dependency that has to cross back through the CPU anyway, and it is
one small matmul against a forward that is two orders of magnitude
larger. What the graph is for is the forward's ~1442 launches, and this
path issues half as many of those per token as the single path does.


### `_fused_device_slot`

Built once per (length bucket, sampling law) and then reused: capture
costs more than a short window's whole decode, and nothing in the graph
depends on the utterance. What changes between calls is the *contents*
of the buffers, the prefill's KV, the seed's noise, the positions -
and every one of those is written through a tensor the graph already
holds.


### `_capture_runner`

``cuda_graphs`` and ``compile_model`` both want the same thing, the
per-token decode as one captured graph instead of ~1442 kernel launches
- and both run over the same static KV cache, so they share the manual
``torch.cuda.CUDAGraph`` capture here (``torch.compile``'s own capture
hits an inductor mask-alignment bug on this model, and its intent is
identical). With neither flag, ``step`` itself is the runner. A graph
needs the buffers to stay at fixed addresses and the input values to be
pushed through ``.fill_`` before each replay, which ``_generate_static``
does.


### `_teacher_forced_fused`

Still one causal forward and still one row per forced token, so two
backends compare exactly as before. Only where a row comes from
changes: a pair's first token is read from ``speech_head`` at the
preceding slot, its second from ``head2`` at that same slot with the
first token's embedding appended, which is the pair the decode loop
samples, held still.


## `loudkit/models/timestretch.py`


### `time_stretch`

Args:
    audio: mono samples.
    sample_rate: theirs. The frame, hop and search window are derived from
        it, so this is not decorative.
    speed: >1 shortens, <1 lengthens. ``1.0`` returns the input unchanged -
        the *same array*, not a copy that happens to be equal, because the
        engine's default must be a bypass and "bit-identical" is easier to
        trust when there is no arithmetic to be identical about.

Returns:
    ``floor(len(audio) / speed + 0.5)`` samples.


### `_best_match`

Scored by cross-correlation normalised by the *candidate's* energy only -
the target's is the same for every candidate and cancels out of the ranking.
Without that normalisation the search prefers whichever candidate is loudest
rather than whichever fits, which at a syllable onset is exactly the wrong
one.

Ties go to the lower offset, so the choice does not depend on iteration
order and the five ports agree.


## `loudkit/models/vocoder.py`


### `TorchVocoder`

Static geometry: when the window recipe is static, the mel is zero-padded
to ``2 x max_speech_tokens`` frames before rendering and the waveform is
trimmed back to the real region afterwards, the same framing the shipped
HiFT graph runs, whose tail-padding effects bleed a few frames back into
the kept audio through the conv stack and are therefore part of the
algorithm, not an export artefact.


## `loudkit/models/windowing.py`


### `module`

Everything a renderer backend needs from the flow and vocoder modules that is
*not* a torch module: the window framing recipe, the Euler time grid, and the
Philox sub-stream ids that address the render randomness. A runtime-only
backend (ONNX, CoreML) imports this file and never touches a torch module,
which is the checkpoint module's promise ("a future runtime-only backend can
load the same file without dragging torch in") kept for the parts that are
pure geometry and bookkeeping.

The window recipe is the entire measured ANE-vs-torch mel deviation (corr
0.975–0.993) when implementations disagree, so it lives here as data, shared
by every renderer, not re-derived per backend.


### `frame_windows`

Returns ``(token_row (1, P+Q), cond (1, 80, 2·(P+Q)), prompt_frames, n)``
where ``n`` is the count of real speech tokens and ``prompt_frames`` the
mel region to cut after integration. In static mode the prompt is framed
to exactly ``static_prompt_tokens`` (truncate long, silence-pad short) and
the query to ``static_length``, the production recipe, which is the
entire measured ANE-vs-torch mel deviation when implementations disagree.


## Notes moved from attribute docstrings and comments


### `loudkit/models/flow.py`


#### `_extend` holds the reference

The positional-encoding buffer is rebuilt under a lock, and the caller slices
the reference it took rather than re-reading the attribute. A second thread
can replace the buffer between the rebuild and the slice, and a slice into a
tensor sized for another length is in range for both and silently wrong. A
reference already held cannot be reached by a later reassignment.


### `loudkit/models/generator.py`


#### `self._cond_lock`

The dict is only a cache, so a lost entry costs a recomputation and
nothing else, but the sequence is not one operation. Two threads
reaching `_prefill_embeds` together could both find a key, and the
second's `pop` for the LRU promotion would raise `KeyError` after the
first had already evicted it: a crash out of a memoisation. The
`_decoding` lock does not cover this, it is taken after the prefill,
and only on the static path.


#### `self._graphs`

Capture is not cheap (about as long as decoding a 200-token window), and
the graph only depends on the shapes, so it outlives the utterance that
built it.

Lengths are rounded up to :data:`_GRAPH_BUCKET` so that chunks of
similar size share one entry. The rounding is not free: the query
attends over the whole padded buffer, so a longer buffer costs a little
on every step (1.41 ms per pair at 300 columns against 1.64 at 1200),
which is why the bucket is small rather than "size it to the window and
be done".


#### `self._decoding`

Keeping graphs across calls moved the decode buffers from per-call to
per-*generator*: the KV cache, the position scalars, the pair register
and the device sampler's noise block are all addresses a captured graph
holds, so two `generate` calls running at once on one engine write each
other's tokens into each other's slots. Nothing crashes. Both callers
get fluent speech, and neither gets their text, the failure this
project treats as worse than a crash.

Not a queue: waiting would hide the fact that a caller thought they had
two engines' worth of throughput and had one. Refusing names it. The
eager path allocates per call and is genuinely re-entrant, so the guard
is taken only where the buffers are shared, and the shipped transports
already serialise a whole request behind one lock.


#### `module` casts

Where a module's own `forward` provably returns a Tensor, the call is wrapped
in `cast(Tensor, ...)`: an assertion about torch's contract, not a guess. See
docs/design/typing.md.


#### `__init__` builds deterministically

Every parameter is initialised before the checkpoint overwrites it, so
`torch.manual_seed(0)` pins a build. An uninitialised parameter reads whatever
the heap held, and a heap that recently held NaNs produces a model whose
logits are NaN roughly one run in three, in a way that reads as numerical
trouble in the static-cache path rather than as an uninitialised read.


#### `TorchTokenGenerator` reads the manifest's vocabulary

`speech_vocab_size` and `start_speech_token` come from the manifest on the
torch path as they do on the ONNX path, and a manifest that disagrees with the
tables in the weights is refused at load. A checkpoint with other dimensions
needs other weights, not another constant, and two backends honouring
different values under one fingerprint would behave differently.


#### `_prefill_embeds` checks the cache twice

Two threads missing on the same key both compute the row. The second check
under the lock keeps the first insert and discards the other, because both
evicting to make room for a key the other has already added lets a burst of N
threads on one cold voice shrink an 8-entry cache to almost nothing. The row
is a pure function of the key, so discarding is free.


#### `generate` drops the newest token of an odd prefix

Under `fusion_mtp2` pairing runs from the prefix's first token, so an
odd-length prefix loses its newest token, not its oldest. Dropping from the
front shifts every pair by one and fuses the right tokens with the wrong
partners, the defect `carry_pair_aligned` exists to prevent. Callers that came
through the engine are already even and lose nothing.


#### `_capture_runner` captures in thread-local mode

Global capture mode forbids any concurrent CUDA call process-wide ("operation
not permitted when stream is capturing"). Thread-local scopes the restriction
to the capturing thread, and only that thread's stream is recorded into the
graph either way.


### `loudkit/models/resample.py`


#### `resample` is vectorised across outputs, never across taps

The loop is vectorised across output samples and accumulates each output's
taps in ascending order, one addition at a time, in float32: the docstring's
"walks each output sample's taps left to right" is a specification. `np.dot`
would sum in whatever order BLAS prefers and stop matching torchaudio and the
four ports. The result is bit-identical with the naive triple loop it replaces
(checked on a 24 kHz second of noise, max difference 0.0), at a fraction of
the 0.18 s of Python that loop cost per enrolment.


### `loudkit/models/timestretch.py`


#### `time_stretch` short fragments

A fragment shorter than one frame (600 samples at 24 kHz, a fortieth of a
second, below anything the engine renders) is cut or zero-padded to the target
length rather than stretched: wrong in the way silence is wrong rather than in
the way a pitch shift is. A zero hop takes the same branch, which turns a hang
no traceback explains into the short-fragment path; it takes a sample rate
under 60 Hz to reach.


### `loudkit/models/vocoder.py`


#### `VOCODER_LENGTH_BUCKET`

Not tidiness: a shape the convolution library has not seen is a shape it either
runs with a poor algorithm or stops to autotune, and either answer costs more
than the frames the rounding wastes. Measured on a desktop GPU over a multi-chunk
read, ragged lengths *with* autotuning ran at 6.85x against 12.49x for the same
lengths without it, the autotuner re-running per distinct shape. Rounding
turns "every chunk is its own shape" into a handful.


#### `VOCODER_RIGHT_CONTEXT`

Measured, not derived: the stack ends in an iSTFT and an arithmetic derivation
through that is the kind of reasoning that is right until it is not. Padding by
k frames and comparing against the full-window output, the difference falls to
1e-6 at k=16 and stops improving after, 1e-6 being fp32 noise from cuDNN
picking different algorithms for different widths, not arithmetic. This is
double the measured need, because 16 frames of mel cost nothing and being wrong
here sounds like a click at every chunk join.


### `loudkit/models/windowing.py`


#### `frame_windows` refuses a sequence the window cannot hold

A `.decode` called directly with more tokens than the window holds is refused
rather than truncated. Truncation gave 300 tokens in and 255 tokens of audio
out with nothing to say the rest had gone; in a reading tool that is the end
of a passage not existing while the audio sounds fine, and only a listener who
knows the text notices.


## Is the renderer worth batching?

Measured in wave O3 with `research/bench_render.py`, against the prototype
`TorchMelDecoder.decode_batch` and `TorchVocoder.synthesize_batch` that sit
beside the single-utterance methods. Nothing calls them. This section is why.

### The device is already full at batch 1

The token generator is launch-latency-bound: at batch 1 a decode step is a few
hundred tiny serial dispatches and the device idles, which is what makes
`research/bench_batch.py` show aggregate throughput growing with N. The renderer
is the opposite shape, one large parallel pass per utterance.

Four calls against one call, warm, median of several rounds, the `N vs 1` column
of `research/bench_render.py` at batch 4. Near 4.0 means four calls cost four
calls' arithmetic and there is no idle device for a batch to fill; near 1.5
would mean one call leaves most of the device waiting:

| model | device | vocoder | mel decoder |
|---|---|---|---|
| loudr-1 | Jetson Orin (CUDA) | 4.00x | 4.00x |
| loudr-1 | RTX 3090 (CUDA) | 3.30x | 3.91x |
| loudr-1 | MPS | 3.98x | 3.68x |
| loudr-1-turbo | MPS | 3.70x | 4.10x |
| loudr-1 | M-series CPU | see below | see below |

The two CUDA rows were run by the integrator on his own boxes; the rest were run
here. The 3090 is the loosest of the GPUs and it still says 3.30x, an upper
bound of about 17% on the vocoder and 2% on the mel. Compare the generator on
the same 3090, where batch 8 gives 2.8x the aggregate of batch 1: that is what a
stage with an idle device between launches looks like, and the renderer is not
it.

**How much these ratios can be trusted.** Repeated runs of loudr-1 on MPS put
the mel between 3.68x and 4.11x and the vocoder between 3.98x and 4.10x, so a
single figure here is worth about plus or minus 10%. That is far too coarse to
argue 3.7 against 4.0 and far too fine to matter: the reading that would change
the decision is 1.5, and nothing measured is near it.

Contention biases this ratio *upward*, because a longer measurement window
offers more chances to be preempted, so a busy machine makes a stage look more
compute-bound than it is. A ratio above its own batch size is the signature, and
it is why the CPU rows are quoted from the next section instead: the machine
here could not be kept quiet long enough to trust a fifteen-second call.

### Batched against serial, which is the actual trade

Speedup is the serial loop's wall time over the batched call's, so above 1.00x
batching wins:

| model | device | stage | batch 2 | batch 4 | batch 8 |
|---|---|---|---|---|---|
| loudr-1-turbo | MPS | mel | 1.02x | 1.05x | 1.04x |
| loudr-1-turbo | MPS | vocoder | 0.94x | 0.98x | 0.68x |
| loudr-1 | MPS | mel | 1.12x | 1.07x | 1.24x |
| loudr-1 | MPS | vocoder | 0.50x | 1.00x | 1.13x |
| loudr-1 | CPU | mel | 1.13x | | |
| loudr-1-turbo | CPU | mel | 1.15x | 1.22x | |
| loudr-1-turbo | CPU | vocoder | 1.11x | 1.24x | |

**Nothing here is a win, and the scatter is the point.** Two runs of the same
loudr-1 MPS cell disagree by more than any effect in the table: the vocoder at
batch 2 came back 0.95x once and 0.50x another time, the mel at batch 8 0.75x
and 1.24x. Every MPS cell sits inside the noise around 1.00x. The honest reading
is that batching the renderer on MPS changes nothing measurable, not that some
particular cell buys 9% or loses 25%.

What is stable is the trade underneath. Turbo's mel runs at about 36x real time
whether it is batched or not, and batching eight of them leaves aggregate
throughput where it was while per-request throughput falls from 36x to 4.6x.
Everyone in a batch waits for the whole batch, and here they wait for nothing.

CPU is the only place a gain reproduces, 1.11x to 1.24x, and it is the least
useful place to have it. The loudr-1 mel is a **15.0 s call** for 5.0 s of audio
on this machine, 0.34x real time for one stage before the vocoder or the token
generator has run, so a CPU deployment is latency-bound end to end and has no
spare throughput to trade. The gain is the CPU estimator's many small ops
carrying dispatch and thread-sync overhead that a wider tensor amortises, which
is why the four-calls probe misses it: that probe asks whether one call
saturates the device, not whether wider ops are cheaper per element.

The CPU rows are the ones to re-take on a quiet machine if this is ever
reopened. The `loudr-1` CPU row above is batch 2 only, and both models' rows
were taken while other jobs were running; `research/bench_render.py` records the
load average at both ends of a run so a suspect row can be identified rather
than argued about.

### A batched row does not reproduce its own single call

**Above one row the batched renderer is not a drop-in for anything that has to
reproduce a render.** A batch of one is bit-identical to the single call, on CPU
and on MPS, for both stages. From two rows up it depends on the shape, the
device and the thread count, and there is no setting that makes it byte-equal
everywhere:

| stage | device | shape | batched row against its own single call |
|---|---|---|---|
| mel | MPS | production | identical |
| mel | CPU | production | 1.6e-2 max, 2.9e-3 rms, mel corr 0.9999994 |
| mel | CPU | production, `set_num_threads(1)` | identical |
| mel | CPU | 96-frame test window | 2.6e-6 max |
| vocoder | CPU | production | identical |
| vocoder | CPU | 96-frame test window | 1.8e-7 max |
| vocoder | MPS | production | 1.5e-6 max, 8.4e-8 rms |

The two test-window rows are the shapes `tests/test_models.py` runs, and they
are why that file bounds correlation rather than bytes: the deviation there is a
thousand times smaller than at production shapes, so a test that passed on bytes
at 96 frames would say nothing about the 252-frame case that ships.

The cause is the intra-op reduction order, not the addressing, which two
experiments settle. Two *identical* rows inside one batched CPU call disagree
with each other by the same 1.6e-2, and no row-addressing bug can produce that.
`torch.set_num_threads(1)` removes the difference at those shapes. The MPS
vocoder is the other shape of the same thing: its rows agree with each other
inside a batch and the whole batch differs from the batch-1 call, which is Metal
choosing a different kernel for a wider tensor rather than threads splitting a
reduction.

**Read 1.6e-2 against the arithmetic it lives in.** The mel estimator runs in
fp16 (`TorchMelDecoder.estimator_dtype`), and at this mel's peak of 11.9 the
fp16 grid step is 7.8e-3, so the worst cell has moved two of the smallest steps
its own dtype can express. 44% of cells move at all, by 7.8 steps on average.

What that costs downstream is bigger than the mel number suggests, and it is the
number to quote:

| through the vocoder | max sample | corr | error to signal |
|---|---|---|---|
| batched mel against its own single call | 0.105 | 0.99871 | -25.9 dB |
| single call, 5 threads against 1 thread | 0.042 | 0.99979 | -33.7 dB |

The second row is the one that puts the first in proportion. **The CPU renderer
does not reproduce itself across thread counts before any batching is
involved**: two separate processes running plain `engine.synthesize`, one at one
thread and one at five, come back 4.7e-2 apart at -28.2 dB. Batching widens an
existing gap rather than opening a new one, and `ExecutionConfig.pin_determinism`
already pins `num_threads` for exactly this reason. At a fixed thread count every
path here is deterministic, batched or not.

So adopting a batched renderer would mean re-taking every golden and every
conformance vector, and it would not close the question, because the goldens
would still be a function of the thread count.

### And the engine takes one utterance

`MelDecoder.decode(tokens, voice, seed)` and `Vocoder.synthesize(mel, voice,
seed)` in `contracts.py` take one utterance each. `Engine.synthesize` is one
call; the HTTP, gRPC and MCP transports each hold a single-flight lock over the
engine, so two requests never reach the renderer at the same time; and the
chunks of one passage are rendered in order because the postprocessor reads the
previous chunk's tail. Wiring a batch in means undoing the single-flight lock,
a queue that gathers requests across callers, a grouping policy (both batched
methods refuse rows of different window lengths, because padding a short row up
to a long one changes the long one), and a latency budget for how long a request
waits to be gathered. That is a scheduler, and it buys nothing measurable on any
GPU tried and at most a quarter on a CPU that is latency-bound anyway, for
several times the per-request latency and bytes that no longer match the
goldens.

Nothing is left for a bigger card to overturn: the 3090 was the case for "a
larger GPU has idle SMs at batch 1", and it answered 3.30x on the vocoder and
3.91x on the mel. `research/bench_render.py` runs on CUDA unmodified if the
question is ever reopened.

### What would have to change first

Not a faster device. The measurement that would reopen this is a renderer stage
whose four-calls ratio comes back near 1.5 rather than near 4, which means a
device that idles between launches where none of the five measured here does.

Even then the batch is the second thing to build, not the first. It needs a
scheduler above a single-flight engine: a queue that gathers work across
callers, a policy that groups by window length, and a latency budget saying how
long a request may wait to be gathered. Nothing in loudkit has that shape today,
and the honest order is to build the scheduler for the token generator, which is
launch-bound and where batch 8 already gives 2.8x, and only then ask whether the
renderer wants to ride along behind it.
