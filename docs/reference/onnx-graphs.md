# ONNX graphs: the signatures and the call order

The `loudreader/loudr-1` release carries nine ONNX graphs: six for synthesis
and three for enrollment (`s3_tokenizer`, `camp`, `voice_encoder`). The six
synthesis graphs are the whole synthesis model, and they are this page's
subject: what each graph takes, what it returns, and the order to call them
in, so you can run the weights from any runtime that loads ONNX.

Every shape below was read off the published graph files with
`onnxruntime.InferenceSession(path).get_inputs()` and `.get_outputs()`, and
confirmed by running the graphs.

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

Three more release files are not optional.

| File | Why you need it |
|---|---|
| `tokenizer.json` | The text tokenizer vocabulary. See [text normalization](preprocess.md) |
| `loudr-1.safetensors` | Six tensors the graphs do not contain. See below |
| `voices/*.safetensors` | A voice profile. See section 7 |

The six tensors are the four embedding tables the generator needs to build its
own input rows, and the weight and bias of the affine layer that turns a flow
embedding into the speaker vector the estimator wants.

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
mel over a two step Euler grid, and `vocoder` turns the mel into a 24 kHz
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

The emotion input has no measured effect on this checkpoint. loudkit passes the
neutral value and so should you.

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

The symbolic axis labels on the cache tensors are wrong. The graph names axis 1
`seq` and axis 2 `kv`. Axis 1 is the head count and axis 2 is the sequence
length. The label comes from `_kv_axes` in
[`tools/export_onnx.py`](../../tools/export_onnx.py), which applies one naming
to both. Trust the observed shape `[1, 4, past, 64]`, not the label.

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
| in | `embeds` | float32 | `[1, 1, 1024]` | The row for the token just chosen |
| in | `position` | int64 | `[1]` | RoPE position id for that row |
| in | `past_k_0 .. past_k_15` | float32 | `[1, 4, past, 64]` | Key cache from the previous call |
| in | `past_v_0 .. past_v_15` | float32 | `[1, 4, past, 64]` | Value cache from the previous call |
| out | `logits` | float32 | `[1, 8194]` | Speech logits for the next token |
| out | `present_k_0 .. present_k_15` | float32 | `[1, 4, past+1, 64]` | Key cache, grown by one |
| out | `present_v_0 .. present_v_15` | float32 | `[1, 4, past+1, 64]` | Value cache, grown by one |

The 34 inputs are `embeds, position, past_k_0, past_v_0, past_k_1, ...`. The 33
outputs match. Feed each call's `present_*` straight into the next call's
`past_*`, in order.

Note that `t3_prefill` returns logits with a sequence axis and `t3_step` does
not. Take `logits[0, -1]` from prefill and `logits[0]` from step.

### `flow_encoder.onnx`

Fixed shapes. The graph is exported for one window and refuses any other.

| Direction | Name | Dtype | Shape | Meaning |
|---|---|---|---|---|
| in | `prompt_token` | int64 | `[1, 238]` | The voice profile's `prompt_tokens`, truncated or padded to 238 |
| in | `speech_tokens` | int64 | `[1, 255]` | The generated speech tokens, padded to 255 |
| out | `flow_encoder_out` | float32 | `[1, 80, 986]` | `mu`, the mel mean over prompt and query, 80 bins |

986 is `2 * (238 + 255)`. Pad both token rows with token id 4254. Padding with 0
adds about 3 dB of high band energy to the tail.

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
combination. Running it twice and mixing is a different algorithm.

### `vocoder.onnx`

Fixed shapes.

| Direction | Name | Dtype | Shape | Meaning |
|---|---|---|---|---|
| in | `mel` | float32 | `[1, 80, 510]` | The mel, zero padded to 510 frames |
| in | `phase` | float32 | `[1, 9, 1]` | Harmonic phase offsets. Row 0 is 0 |
| in | `noise` | float32 | `[1, 9, 244800]` | Excitation noise, one row per harmonic |
| out | `vocoder_out` | float32 | `[1, 244800]` | Waveform at 24 kHz |

510 frames is `2 * 255` speech tokens. 244800 samples is `510 * 480`. Trim the
output to `frames * 480` samples, where `frames` is the real mel length.

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
tokens = []
for step in range(255):
    if len(tokens) < floor:
        logits[6562] = -inf          # mask the stop token below the EOS floor
    token = sample(logits, step)     # your LR-SAMPLER-v1 implementation
    tokens.append(token)
    if token == 6562:
        break
    row = speech_emb[token] + speech_pos_emb[len(prefix) + step + 1]
    logits_all, *kv = t3_step(row[None, None], [embeds.shape[1] + step], *kv)
    logits = logits_all[0]

# renderer
prompt, query, cond_mel = frame(tokens, profile)       # 238 and 255, pad id 4254
mu   = flow_encoder(prompt, query)
spks = affine(flow_embedding / norm(flow_embedding))
x    = gaussian_noise([1, 80, 986])
grid = [0.0, 0.2928932188134524, 1.0]                  # 1 - cos(i/2 * pi/2)
for t0, t1 in pairs(grid):
    x = x + (t1 - t0) * flow_estimator(x, mu, [t0], spks, cond_mel)
mel  = x[0, :, 476 : 476 + 2 * len(tokens)]
wav  = vocoder(pad(mel, 510), phase, noise)[: len(tokens) * 2 * 480]
```

The stop token is 6562. The speech start token is 6561. The speech vocabulary is
8194. The cap is 255 new tokens, which is the same 255 the window allows, so a
row that never emits the stop token still fits the renderer.

The reference driver for all of this is
[`python/loudkit/backends/onnx_backend.py`](../../python/loudkit/backends/onnx_backend.py).

---

## 5. What the graphs do not contain

Four layers sit outside the graphs. Each one changes what a listener hears, and
each one is specified with a shared fixture that your port can be held to.

| Layer | Where it runs | Specification |
|---|---|---|
| Text frontend | Before the generator | [Text normalization](preprocess.md) |
| Sampler, LR-SAMPLER-v1 | Between every pair of generator calls | [Identity contract](IDENTITY-CONTRACT.md#the-sampler-lr-sampler-v1) |
| Chunking and joins | Around the whole pipeline | [Streaming and long-form](../guides/02-streaming-and-long-form.md#where-the-splits-fall) |
| Postprocess detectors | After the generator, before the renderer | [Postprocess](postprocess.md) |

Three more things are code, not graphs.

- **Randomness.** The flow prior, the vocoder phase and the vocoder noise are
  counter based Philox draws, addressed by seed and sub stream. See
  [`python/loudkit/rng.py`](../../python/loudkit/rng.py). Feeding a library RNG
  instead gives different audio from the same seed.
- **Enrollment.** Making a new voice profile needs the three enrollment
  graphs. The release ships them beside the six above: `s3_tokenizer.onnx`,
  `camp.onnx` and `voice_encoder.onnx`. Their signatures are not documented
  on this page; the shipped profiles in `voices/` cover synthesis without
  them.
- **The window recipe.** Truncating the prompt to 238, padding the query to 255
  with token 4254, and cutting the first 476 mel frames after integration. See
  `frame_windows` in
  [`python/loudkit/models/windowing.py`](../../python/loudkit/models/windowing.py).

---

## 6. What you give up

loudkit computes an algorithm fingerprint over the settings and the recipe
version, and refuses a run whose parts disagree. Running the graphs directly
gives you no such check. Nothing catches a window framed to the wrong length, a
sampler with the wrong temperature, or a mel cut at the wrong offset. Those
failures are silent and audible: the audio plays, and it is wrong.

**Read the manifest, and take it at face value.** The packed checkpoint
declares `recipe_version: loudkit-1` and carries the `postprocess`, `window`
and `eos_floor` blocks that name the detectors, the static window and the EOS
floor. Read those blocks and you have what loudkit runs. There is no second,
resolved recipe held somewhere else.

`loudkit-1` is the only recipe. All five ports refuse a manifest that declares
any other `recipe_version` rather than reconciling it, so a checkpoint either
states the recipe this page describes or does not load. `Engine.describe()`
prints the declared version.

`production_algorithm` in
[`backends/__init__.py`](https://github.com/loudreader/loudkit/blob/main/python/loudkit/backends/__init__.py)
supplies the window, the detectors and the EOS floor only for a checkpoint
whose manifest omits those blocks. The released checkpoint carries all three,
so it fills nothing.

The [identity contract](IDENTITY-CONTRACT.md) also stops at the loudkit
boundary. Its promises hold for loudkit builds, not for yours. A reimplementation
that matches the shared fixtures earns the same behaviour on the cases the
fixtures cover, and nothing beyond them.

---

## 7. The voice profile

A voice profile is a safetensors file of five tensors. `voices/kathleen.safetensors`
is 165 KB.

| Tensor | Dtype | Shape | Used by |
|---|---|---|---|
| `speaker_embedding` | float32 | `[256]` | `t3_cond`, as `speaker_emb` |
| `cond_prompt_tokens` | int64 | `[150]` | `t3_cond`, as `prompt_tokens` |
| `prompt_tokens` | int64 | `[250]` | `flow_encoder`, truncated to 238 |
| `prompt_mel` | float32 | `[80, 500]` | `flow_estimator`, as the first 476 frames of `cond` |
| `flow_embedding` | float32 | `[192]` | `flow_estimator`, normalised then affine projected to `spks` |

The file's safetensors metadata carries a `voice` key holding the profile name,
the language, the source sample rate and the enrolment recipe.
