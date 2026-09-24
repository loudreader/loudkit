# ONNX graphs: the signatures and the call order

The `loudreader/loudr-1` release carries nine ONNX graphs: six for synthesis
and three for enrollment (`s3_tokenizer`, `camp`, `voice_encoder`). The six
synthesis graphs are the whole synthesis model. Sections 1 to 7 give what each
graph takes, what it returns, and the order to call them in, so any runtime
that loads ONNX can run the weights. `loudreader/loudr-1-turbo` decodes two
tokens per step with a different set of generator graphs; section 8 lists the
differences.

The shapes below are what `onnxruntime.InferenceSession(path).get_inputs()`
and `.get_outputs()` report for the published loudr-1 graphs. Use the same
calls to check the graphs of a pinned release.

Opset 17. All tensors are `float32` or `int64`. The graphs are batch 1.

---

## 1. What ships

| File | Stage | What it does |
|---|---|---|
| `onnx/t3_cond.onnx` | generator | Builds the 34-row conditioning block from a voice profile |
| `onnx/t3_prefill.onnx` | generator | One causal forward over the whole prompt. Returns all logits and the KV cache |
| `onnx/t3_step.onnx` | generator | One decode step against the cache |
| `onnx/flow_encoder.onnx` | renderer | Speech tokens to the mel mean `mu` |
| `onnx/flow_estimator.onnx` | renderer | One Euler step of the conditional flow |
| `onnx/vocoder.onnx` | renderer | Mel to waveform at 24 kHz |

Three more release files are required.

| File | Why you need it |
|---|---|
| `tokenizer.json` | The text tokenizer vocabulary. See [text normalization](preprocess.md) |
| `loudr-1.safetensors` | Six tensors the graphs do not contain. See below |
| `voices/*.safetensors` | A voice profile. See section 7 |

The six tensors are the four embedding tables the generator needs to build its
own input rows, and the weight and bias of the affine layer that turns a flow
embedding into the speaker vector the estimator takes.

| Tensor | Shape | Dtype |
|---|---|---|
| `t3.text_emb.weight` | `[2454, 1024]` | float16 |
| `t3.speech_emb.weight` | `[8194, 1024]` | float16 |
| `t3.text_pos_emb.emb.weight` | `[2050, 1024]` | float16 |
| `t3.speech_pos_emb.emb.weight` | `[4100, 1024]` | float16 |
| `s3gen.flow.spk_embed_affine_layer.weight` | `[80, 192]` | float32 |
| `s3gen.flow.spk_embed_affine_layer.bias` | `[80]` | float32 |

Upcast the float16 tables to float32. The graphs hold float32 weights, and
float16 to float32 is exact.

`SHA256SUMS` in the release covers every file in the bundle, `onnx/` included.
The one exception is `SHA256SUMS` itself, which cannot hold its own digest. A
bundle of N files therefore carries N-1 checksum lines.

---

## 2. The pipeline

Text becomes text tokens in the frontend, which is code and not a graph. The
generator turns those text tokens plus a voice profile into speech tokens, at 25
tokens per second: `t3_cond` once per voice, `t3_prefill` once per utterance,
`t3_step` once per token. The renderer turns the speech tokens into audio:
`flow_encoder` produces a mel mean, `flow_estimator` integrates the flow to a
mel over a two-step Euler grid, and `vocoder` turns the mel into a 24 kHz
waveform. One speech token is two mel frames. One mel frame is 480 samples.

---

## 3. The graphs

### `t3_cond.onnx`

Run once per voice and cache the result. The output depends only on the profile.

| Direction | Name | Dtype | Shape | Meaning |
|---|---|---|---|---|
| in | `speaker_emb` | float32 | `[1, 256]` | `speaker_embedding` from the voice profile |
| in | `prompt_tokens` | int64 | `[1, seq]` | `cond_prompt_tokens` from the voice profile |
| in | `emotion` | float32 | `[1, 1]` | Emotion scalar. Pass `0.5` |
| out | `t3_cond_out` | float32 | `[1, 34, 1024]` | The conditioning rows: speaker projection, perceiver output, emotion |

The emotion input has no measured effect on this checkpoint. loudkit passes
the neutral value, `0.5`; pass the same.

### `t3_prefill.onnx`

| Direction | Name | Dtype | Shape | Meaning |
|---|---|---|---|---|
| in | `embeds` | float32 | `[1, seq, 1024]` | The full input rows, built by you |
| in | `positions` | int64 | `[seq]` | RoPE position ids, `0 .. seq-1` |
| out | `logits` | float32 | `[1, seq, 8194]` | Speech logits at every position |
| out | `kv_k_0 .. kv_k_15` | float32 | `[1, 4, seq, 64]` | Key cache, one per layer |
| out | `kv_v_0 .. kv_v_15` | float32 | `[1, 4, seq, 64]` | Value cache, one per layer |

16 layers, 4 key/value heads, head dim 64. The 33 outputs come in the order
`logits, kv_k_0, kv_v_0, kv_k_1, kv_v_1, ...`.

Read the cache layout from the shape, `[1, 4, past, 64]`: axis 1 is the head
count and axis 2 is the sequence length. Do not rely on the symbolic axis
labels. The published loudr-1 graphs name axis 1 `seq` and axis 2 `kv`.
`tools/export_generator.py` labels only axis 2 (`seq` in prefill, `past` and
`present` in the step graphs), so a graph exported with it carries different
labels.

Build `embeds` by concatenating, in this order:

1. the 34 conditioning rows from `t3_cond`,
2. `text_emb[framed] + text_pos_emb[0 .. len-1]`, where `framed` is the text
   tokens wrapped as `[255] + text_tokens + [0]`,
3. one row for the speech start token: `speech_emb[6561] + speech_pos_emb[0]`,
4. optionally, a prefix of speech tokens carried from the previous chunk:
   `speech_emb[prefix] + speech_pos_emb[1 .. len(prefix)]`.

### `t3_step.onnx`

| Direction | Name | Dtype | Shape | Meaning |
|---|---|---|---|---|
| in | `embeds` | float32 | `[1, 1, 1024]` | The row for the token chosen in the previous step |
| in | `position` | int64 | `[1]` | RoPE position id for that row |
| in | `past_k_0 .. past_k_15` | float32 | `[1, 4, past, 64]` | Key cache from the previous call |
| in | `past_v_0 .. past_v_15` | float32 | `[1, 4, past, 64]` | Value cache from the previous call |
| out | `logits` | float32 | `[1, 8194]` | Speech logits for the next token |
| out | `present_k_0 .. present_k_15` | float32 | `[1, 4, past+1, 64]` | Key cache, grown by one |
| out | `present_v_0 .. present_v_15` | float32 | `[1, 4, past+1, 64]` | Value cache, grown by one |

The 34 inputs are `embeds, position, past_k_0, past_v_0, past_k_1, ...`. The 33
outputs match. Feed each call's `present_*` straight into the next call's
`past_*`, in order.

`t3_prefill` returns logits with a sequence axis; `t3_step` does not. Take
`logits[0, -1]` from prefill and `logits[0]` from step.

### `flow_encoder.onnx`

Fixed shapes. The graph is exported for one window and refuses any other.

| Direction | Name | Dtype | Shape | Meaning |
|---|---|---|---|---|
| in | `prompt_token` | int64 | `[1, 238]` | The voice profile's `prompt_tokens`, truncated or padded to 238 |
| in | `speech_tokens` | int64 | `[1, 255]` | The generated speech tokens, padded to 255 |
| out | `flow_encoder_out` | float32 | `[1, 80, 986]` | `mu`, the mel mean over prompt and query, 80 bins |

986 is `2 * (238 + 255)`. Pad both token rows with token id 4254. Padding with 0
adds about 3 dB of high-band energy to the tail.

### `flow_estimator.onnx`

| Direction | Name | Dtype | Shape | Meaning |
|---|---|---|---|---|
| in | `x` | float32 | `[1, 80, 986]` | Current state. Starts as Gaussian noise |
| in | `mu` | float32 | `[1, 80, 986]` | The encoder output |
| in | `t` | float32 | `[1]` | Time on the Euler grid |
| in | `spks` | float32 | `[1, 80]` | Speaker vector, `affine(flow_embedding / ‖flow_embedding‖)` |
| in | `cond` | float32 | `[1, 80, 986]` | The profile's `prompt_mel` in the first 476 frames, zeros after |
| out | `flow_estimator_out` | float32 | `[1, 80, 986]` | The velocity field at `t` |

Call this estimator once per step. Do not form a classifier-free guidance
combination: running it twice and mixing the results is a different
algorithm.

### `vocoder.onnx`

Fixed shapes.

| Direction | Name | Dtype | Shape | Meaning |
|---|---|---|---|---|
| in | `mel` | float32 | `[1, 80, 510]` | The mel, zero padded to 510 frames |
| in | `phase` | float32 | `[1, 9, 1]` | Harmonic phase offsets. Row 0 is 0 |
| in | `noise` | float32 | `[1, 9, 244800]` | Excitation noise, one row per harmonic |
| out | `vocoder_out` | float32 | `[1, 244800]` | Waveform at 24 kHz |

510 frames is `2 * 255` speech tokens. 244800 samples is `510 * 480`. Take row
0 of the output and trim it to `frames * 480` samples, where `frames` is the
real mel length.

---

## 4. The call sequence for one utterance

```text
# once per voice
cond = t3_cond(speaker_emb, cond_prompt_tokens, [[0.5]])   # [1, 34, 1024]

# once per utterance
embeds    = concat(cond, text_rows, start_row, prefix_rows)
positions = arange(embeds.shape[1])
logits_all, *kv = t3_prefill(embeds, positions)
logits = logits_all[0, -1]

floor = max(10, int(len(text_tokens) * 1.2))
seen = set(prefix)                   # for the repetition penalty
tokens = []
for step in range(255):
    if len(tokens) < floor:
        logits[6562] = -inf          # mask the stop token below the EOS floor
    token = sample(logits, step, seen)   # your LR-SAMPLER-v1 implementation
    if token == 6562:
        break                        # the stop token is never rendered
    tokens.append(token)
    seen.add(token)
    row = speech_emb[token] + speech_pos_emb[len(prefix) + step + 1]
    logits_all, *kv = t3_step(row[None, None], [embeds.shape[1] + step], *kv)
    logits = logits_all[0]

# renderer
speech = [t for t in tokens if t < 6561]               # drop special tokens
prompt, query, cond_mel = frame(speech, profile)       # 238 and 255, pad id 4254
mu   = flow_encoder(prompt, query)
spks = affine(flow_embedding / norm(flow_embedding))
x    = gaussian_noise([1, 80, 986])
grid = [0.0, 0.2928932188134524, 1.0]                  # 1 - cos(i/2 * pi/2)
for t0, t1 in pairs(grid):
    x = x + (t1 - t0) * flow_estimator(x, mu, [t0], spks, cond_mel)
mel  = x[0, :, 476 : 476 + 2 * len(speech)]
wav  = vocoder(pad(mel, 510), phase, noise)[0, : len(speech) * 2 * 480]
```

The stop token is 6562. The speech start token is 6561. The speech vocabulary is
8194. The acoustic codebook holds ids 0 to 6560, so ids from 6561 up are
special tokens and are dropped before the renderer. The cap is 255 new tokens,
the same 255 the window allows, so a row that never emits the stop token still
fits the renderer.

The reference driver for all of this is
[`python/loudkit/backends/onnx_backend.py`](../../python/loudkit/backends/onnx_backend.py).

---

## 5. What the graphs do not contain

Four layers run outside the graphs. Each one changes what a listener hears,
and each one has a shared conformance fixture to test an implementation
against.

| Layer | Where it runs | Specification |
|---|---|---|
| Text frontend | Before the generator | [Text normalization](preprocess.md) |
| Sampler, LR-SAMPLER-v1 | Between every pair of generator calls | [Identity contract](../reference/IDENTITY-CONTRACT.md#the-sampler-lr-sampler-v1) |
| Chunking and joins | Around the whole pipeline | [Streaming and long-form](../guides/02-streaming-and-long-form.md#where-the-splits-fall) |
| Postprocess detectors | After the generator, before the renderer | [Postprocess](postprocess.md) |

Three more parts are code, not graphs:

- The flow prior, the vocoder phase and the vocoder noise are counter-based
  Philox draws, addressed by seed and sub-stream. See
  [`python/loudkit/rng.py`](../../python/loudkit/rng.py). A library RNG gives
  different audio from the same seed.
- Enrollment makes a new voice profile with the three enrollment graphs,
  `s3_tokenizer.onnx`, `camp.onnx` and `voice_encoder.onnx`, which ship beside
  the six above. Their signatures are not documented here. The released
  profiles in `voices/` cover synthesis without them.
- The window recipe truncates the prompt to 238, pads the query to 255 with
  token 4254, and cuts the first 476 mel frames after integration. See
  `frame_windows` in
  [`python/loudkit/models/windowing.py`](../../python/loudkit/models/windowing.py).

---

## 6. What a direct integration must check

loudkit computes an algorithm fingerprint over the settings and the recipe
version, and refuses a run whose parts disagree. A direct integration of the
graphs has no such check, so it must validate the framing, the sampler
settings and the mel offsets itself. A window framed to the wrong length, a
sampler with the wrong temperature or a mel cut at the wrong offset raises no
error and produces wrong audio.

Read the recipe from the manifest. The released checkpoint declares
`recipe_version: loudkit-1` and carries the `postprocess`, `window` and
`eos_floor` blocks that set the detectors, the static window and the EOS
floor. Those blocks are what loudkit runs; no other resolved recipe exists.

`loudkit-1` is the only recipe version. All five implementations refuse a
manifest that declares any other `recipe_version`, so a checkpoint either
states the recipe described here or does not load. `Engine.describe()` prints
the declared version.

`production_algorithm` in
[`backends/__init__.py`](https://github.com/loudreader/loudkit/blob/main/python/loudkit/backends/__init__.py)
supplies the window, the detectors and the EOS floor only for a checkpoint
whose manifest omits those blocks. The released checkpoint carries all three,
so it fills nothing.

The [identity contract](../reference/IDENTITY-CONTRACT.md) covers loudkit
builds only. A reimplementation that passes the shared fixtures agrees with
loudkit on the cases the fixtures cover; other inputs and runtimes need their
own validation.

---

## 7. The voice profile

A voice profile is a safetensors file of five tensors. The shapes below are
those of `voices/kathleen.safetensors` (165 KB). Prompt lengths vary between
profiles; the window recipe frames them to the fixed render window.

| Tensor | Dtype | Shape | Used by |
|---|---|---|---|
| `speaker_embedding` | float32 | `[256]` | `t3_cond`, as `speaker_emb` |
| `cond_prompt_tokens` | int64 | `[150]` | `t3_cond`, as `prompt_tokens` |
| `prompt_tokens` | int64 | `[250]` | `flow_encoder`, truncated to 238 |
| `prompt_mel` | float32 | `[80, 500]` | `flow_estimator`, as the first 476 frames of `cond` |
| `flow_embedding` | float32 | `[192]` | `flow_estimator`, normalised then affine projected to `spks` |

The file's safetensors metadata carries a `voice` key holding the profile name,
the language, the source sample rate and the enrollment recipe.

---

## 8. loudr-1-turbo: two-token decoding

`loudreader/loudr-1-turbo` declares `decode.mode: fusion_mtp2` under
`format_version 2`. It has a smaller token generator, and each decode step
produces two speech tokens. Its release carries seven synthesis graphs:
`t3_cond`, `t3_prefill`, `t3_pair_step`, `t3_head2`, `flow_encoder`,
`flow_estimator` and `vocoder`. There is no `t3_step`. The enrollment graphs
are the same three.

`t3_cond`, `flow_encoder`, `flow_estimator` and `vocoder` have the signatures
in section 3. The renderer runs one Euler step, grid `[0.0, 1.0]`, as the
manifest declares.

`t3_prefill` has the inputs in section 3 and one more output, `hidden`,
between `logits` and the cache: float32 `[1, 1024]`, the transformer state at
the last position. The number of cache tensors is the graph's own; read it
from the output list.

`t3_head2.onnx` predicts the second token of a pair:

| Direction | Name | Dtype | Shape | Meaning |
|---|---|---|---|---|
| in | `hidden` | float32 | `[1, 1024]` | The `hidden` output of the last prefill or pair step |
| in | `first_id` | int64 | `[1]` | The pair's first token, sampled from the current logits |
| out | `logits` | float32 | `[1, 8194]` | Speech logits for the second token |

`t3_pair_step.onnx` feeds one pair back into the transformer:

| Direction | Name | Dtype | Shape | Meaning |
|---|---|---|---|---|
| in | `pair_ids` | int64 | `[1, 2]` | The two tokens of the pair |
| in | `speech_position` | int64 | `[1]` | Speech position of the pair: one position per pair |
| in | `position` | int64 | `[1]` | RoPE position id of the pair's row |
| in | `past_k_i`, `past_v_i` | float32 | `[1, heads, past, head_dim]` | Cache from the previous call, interleaved as in `t3_step` |
| out | `logits` | float32 | `[1, 8194]` | Speech logits for the next pair's first token |
| out | `hidden` | float32 | `[1, 1024]` | The state `t3_head2` reads |
| out | `present_k_i`, `present_v_i` | float32 | `[1, heads, past+1, head_dim]` | Cache, grown by one row per pair |

The graph builds the pair's row itself: `0.5 * (ea + eb) + fuse([ea ; eb])`,
where `ea` and `eb` are the speech embeddings of the two tokens. A carried
prefix in the prefill rows needs the same row outside the graph, so
`loudr-1-turbo.safetensors` also carries the `fuse` weights
(`t3.fuse.0.weight`, `t3.fuse.0.bias`, `t3.fuse.2.weight`, `t3.fuse.2.bias`,
with an exact GELU between the two layers). Prefix rows are
`pair_row(prefix pair) + speech_pos_emb[1 .. len(prefix) / 2]`. loudkit drops
the last token of an odd-length prefix.

```text
logits_all, hidden, *kv = t3_prefill(embeds, positions)
logits = logits_all[0, -1]
seen = set(prefix)
tokens = []
while len(tokens) < 255:
    first = sample(logits, len(tokens), seen)      # EOS floor masking as above
    if first == 6562:
        break
    tokens.append(first)
    seen.add(first)
    if len(tokens) == 255:
        break
    logits = t3_head2(hidden, [first])[0]
    second = sample(logits, len(tokens), seen)     # the floor applies here too
    if second == 6562:
        break
    tokens.append(second)
    seen.add(second)
    pair = len(tokens) // 2 - 1
    logits_all, hidden, *kv = t3_pair_step(
        [[first, second]],
        [len(prefix) // 2 + pair + 1],
        [embeds.shape[1] + pair],
        *kv,
    )
    logits = logits_all[0]
```

Either head emitting the stop token ends the sequence. The reference driver is
`ONNXTokenGenerator._generate_pairs` in
[`python/loudkit/backends/onnx_backend.py`](../../python/loudkit/backends/onnx_backend.py),
and [two-token decode](two-token-decode.md) gives the design.
