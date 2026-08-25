/**
 * The engine: all three stages over the exported ONNX graphs, fp32, no torch.
 *
 * Bit-parity port of `loudkit.backends.onnx_backend`. The generator runs
 * entirely on the graphs — `t3_cond` (34-slot conditioning), `t3_prefill`
 * (one causal forward: every-position logits + KV cache), `t3_step` (one
 * decode step against the cache) — with the framing, embeddings, positions and
 * the sampler loop here in JS. The renderer is `flow_encoder` +
 * `flow_estimator` integrated with Euler steps, then the HiFT `vocoder`.
 *
 * The conformance fixture pins the whole pipeline: same text, voice and seed
 * give the same tokens and the same mel/waveform band as the Python engine.
 */

import { Checkpoint } from "./checkpoint.js";
import { splitText } from "./chunking.js";
import { GraphemeTextFrontend } from "./frontend.js";
import { fingerprint } from "./fingerprint.js";
import { ort, type OrtTensor } from "./ort.js";

/**
 * Chunk seeds start here, clear of the per-stage streams (1 = flow, 2 =
 * vocoder). Mirrors `_STREAM_CHUNK` in loudkit.engine.
 */
const CHUNK_STREAM_BASE = 16;

/**
 * What a synthesis reads as when neither the caller nor the voice says.
 *
 * Reached less often than it looks: `loadVoice` defaults a *missing* header key
 * to `"en"`, and Python writes the key, so an empty string only
 * arrives from a profile built in memory or a header hand-edited to `""`. A
 * profile file with no language field inherits nothing — it loads as `"en"`.
 */
const FALLBACK_LANGUAGE = "en";

/**
 * The language chain: the argument, then the voice's recorded language, then
 * English.
 *
 * Without the voice link, `engine.synthesize("Cześć", polishVoice, 7)` runs
 * Polish text through the English frontend — English number words, English
 * abbreviation expansion, no Polish respelling — and says so nowhere. A profile
 * records the language of the audio it was enrolled from, so the voice is the
 * better answer than a constant.
 *
 * Passing `language` is how cross-lingual synthesis is requested: an English
 * voice reading Polish text is `"pl"`, and the argument always wins over the
 * profile.
 *
 * `||` rather than `??` for the profile: an empty language id is not a
 * language, it would tag the text `[]`, and `loadVoice` only defaults the field
 * when the header key is absent — a header that says `"language": ""` still
 * arrives here empty.
 *
 * Mirrors `loudkit.engine._resolve_language`.
 */
export function resolveLanguage(
  language: string | undefined,
  voice: Pick<VoiceProfile, "language">
): string {
  if (language !== undefined) return language;
  return voice.language || FALLBACK_LANGUAGE;
}

/**
 * Execution inputs that are neither the text, the voice nor the seed.
 *
 * A trailing options object rather than two more positional parameters. The
 * three synthesis methods already end in `language?, shouldCancel?`, and
 * `synthesize(text, voice, seed, undefined, undefined, 1.5, prev)` is a call
 * nobody can read and everybody eventually mis-orders — Python names these two
 * keyword-only for the same reason. Every existing call site keeps working
 * unchanged because the object is optional and so is every field in it.
 *
 * Both are execution inputs, like the seed: they are not part of
 * `AlgorithmConfig` and they do not move the fingerprint. Two engines that
 * disagree about them are still computing the same thing.
 */
export interface SynthesisOptions {
  /**
   * Playback speed in `[0.5, 2.0]`; greater than one is faster and pitch is
   * preserved. `1.0` — the default — is an exact bypass: the waveform is the
   * vocoder's own array, untouched. See `timestretch.ts`.
   */
  speed?: number;

  /**
   * Speech tokens this utterance continues from — the `tokens` of the call
   * before it. The first window is then conditioned on their tail exactly as an
   * interior chunk is conditioned on its predecessor, which is what stops a
   * second request from restarting the pitch contour like a fresh sentence.
   */
  previousTokens?: number[];
}

/**
 * The conditioning context a call inherits from the one before it.
 *
 * The same slice the streaming loop takes between two chunks — last
 * `chunking.prefixTokens` — applied to tokens that came from a different call.
 * There is deliberately no second mechanism: a request boundary and a chunk
 * boundary are the same join, and the reason chunk joins do not stutter is the
 * reason request joins should not either.
 *
 * Any length is accepted because only the tail is used, so
 * `previousTokens: result.tokens` is the intended call and a caller should never
 * have to know the prefix length to make it.
 *
 * A free function taking its two configuration numbers explicitly, rather than a
 * method reading them off `this.config`, so that it can be tested without a
 * loaded engine — the unit under test here is the slice-and-check, and every other
 * path to it needs the ONNX graphs and the checkpoint.
 *
 * Throws for an id outside the acoustic codebook. The whole input is checked
 * rather than only the slice that will be used: an id out of range means the
 * sequence was built wrong, and reporting that only when it lands in
 * the last six tokens would make the failure depend on the length of the
 * caller's text. A non-integer or a NaN fails the same comparison, which is the
 * answer that costs nothing to be right about.
 */
export function carryFrom(
  previousTokens: number[] | undefined,
  prefixTokens: number,
  startSpeechToken: number
): number[] {
  if (previousTokens === undefined || previousTokens.length === 0) return [];
  for (const token of previousTokens) {
    if (!(token >= 0 && token < startSpeechToken)) {
      throw new RangeError(
        `previousTokens contains ${token}, which is not an acoustic speech ` +
          `token (expected 0 <= id < ${startSpeechToken}). Pass the \`tokens\` ` +
          "of an earlier result; the generator's own control tokens are already " +
          "stripped from it."
      );
    }
  }
  // Not `slice(-prefixTokens)`: a zero there is the whole list rather than
  // nothing, which would condition on the entire previous utterance at exactly
  // the setting that means "chunks are independent".
  if (prefixTokens <= 0) return [];
  return previousTokens.slice(Math.max(0, previousTokens.length - prefixTokens));
}

/**
 * Concatenate row-major `[MEL_BINS, frames]` mels along the TIME axis.
 *
 * Appending the flat buffers end to end — the obvious thing, and what this did
 * — is not concatenation: after the first chunk the next chunk's bin 0 lands
 * after the previous chunk's bin 79, so every row but the first is wrong. The
 * audio is unaffected (each chunk is vocoded on its own) but the mel is the
 * diagnostic people reach for when two backends disagree, and a mis-shaped one
 * sends them looking in the wrong place.
 */
export function concatMelAlongTime(mels: Float32Array[]): Float32Array {
  if (mels.length === 0) return new Float32Array(0);
  if (mels.length === 1) return mels[0];
  const frameCounts = mels.map((m) => m.length / MEL_BINS);
  const total = frameCounts.reduce((n, f) => n + f, 0);
  const out = new Float32Array(MEL_BINS * total);
  for (let bin = 0; bin < MEL_BINS; bin++) {
    let at = bin * total;
    for (let i = 0; i < mels.length; i++) {
      const frames = frameCounts[i];
      out.set(mels[i].subarray(bin * frames, (bin + 1) * frames), at);
      at += frames;
    }
  }
  return out;
}
import { gaussianField, symmetricUniforms } from "./noise.js";
import { ceilingFor, inspect, type Inspection } from "./postprocess.js";
import { normalizeSeed } from "./rng.js";
import { LRSamplerV1 } from "./sampler.js";
import {
  type ExecutionOptions,
  type ResolvedONNXProvider,
  describeExecution,
} from "./execution.js";
import { Session, openSessions } from "./session.js";
import { speechText } from "./speechText.js";
import { timeStretch, validateSpeed } from "./timestretch.js";
import { timeline, type ChunkSpan, type ChunkTiming } from "./timing.js";
import { AlgorithmConfig, VoiceProfile } from "./types.js";
import { EMOTION_NEUTRAL, loadVoice } from "./voice.js";
import {
  FLOW_NOISE_STREAM,
  START_TEXT_TOKEN,
  STOP_TEXT_TOKEN,
  VOCODER_NOISE_STREAM,
  VOCODER_PHASE_STREAM,
  eosFloor,
  frameWindows,
  timeGrid,
} from "./windowing.js";

const MEL_BINS = 80;
const N_HARMONICS = 9;
const UPSAMPLE_PER_FRAME = 480;
const N_LAYERS = 16;
const KV_HEADS = 4;
const HEAD_DIM = 64;

export class Engine {
  readonly config: AlgorithmConfig;

  /**
   * This engine's algorithm fingerprint, comparable with the Python, Swift, Go
   * and Rust ones. Two engines whose fingerprints differ are computing
   * different things, whatever their audio sounds like.
   */
  fingerprint(): string {
    return fingerprint(this.config);
  }

  /**
   * The execution provider these six graphs were opened on — never `"auto"`,
   * always the one that ran.
   */
  readonly onnxProvider: ResolvedONNXProvider;

  /**
   * One line for logs, benchmark rows and bug reports, in the `exec[...]`
   * shape Python and Swift print.
   *
   * The algorithm half is the fingerprint and the recipe name, not Python's
   * full knob list: the fingerprint already hashes those knobs, and printing
   * the same floats through two languages' number formatters — Python's `1.0`
   * against JS's `1` — would make two identical engines read as different.
   */
  describe(): string {
    return `algo[${this.fingerprint()}] ${this.config.recipeVersion} | ${describeExecution(this.onnxProvider)}`;
  }

  private frontend: GraphemeTextFrontend;
  private tables: { textEmb: Float32Array; speechEmb: Float32Array; textPos: Float32Array; speechPos: Float32Array };
  private spkAffine: { weight: Float32Array; bias: Float32Array };
  private cond: Session;
  private prefill: Session;
  private step: Session;
  private encoder: Session;
  private estimator: Session;
  private vocoder: Session;

  private constructor(
    config: AlgorithmConfig,
    frontend: GraphemeTextFrontend,
    tables: Engine["tables"],
    spkAffine: Engine["spkAffine"],
    provider: ResolvedONNXProvider,
    sessions: Record<string, Session>
  ) {
    this.config = config;
    this.frontend = frontend;
    this.tables = tables;
    this.spkAffine = spkAffine;
    this.onnxProvider = provider;
    this.cond = sessions.cond;
    this.prefill = sessions.prefill;
    this.step = sessions.step;
    this.encoder = sessions.encoder;
    this.estimator = sessions.estimator;
    this.vocoder = sessions.vocoder;
  }

  /**
   * Release the six native graph sessions.
   *
   * The counterpart of Go's `Engine.Close`; Rust and Swift get this from their
   * ownership rules and Python from the runtime's finaliser, so this was the
   * one implementation of the five where a caller had no way to hand back a
   * checkpoint's worth of native memory. One long-lived engine — what the
   * README recommends — never needed it; a second one did.
   *
   * Safe to call twice, and the engine must not be used afterwards.
   */
  async close(): Promise<void> {
    await Promise.all(
      [this.cond, this.prefill, this.step, this.encoder, this.estimator, this.vocoder].map(
        (s) => s.close()
      )
    );
  }

  /**
   * Build an engine from a packed checkpoint plus a directory of ONNX graphs.
   *
   * `execution.onnxProvider` selects the execution provider; omitted, it is
   * `"auto"` and the best provider this build and machine offer wins. The
   * answer is on `engine.onnxProvider` and in `engine.describe()`.
   */
  static async load(
    checkpointPath: string,
    onnxDir: string,
    tokenizerPath: string,
    execution: ExecutionOptions = {}
  ): Promise<Engine> {
    const ckpt = Checkpoint.open(checkpointPath);
    const graphs: Array<[string, string]> = [
      ["cond", "t3_cond.onnx"],
      ["prefill", "t3_prefill.onnx"],
      ["step", "t3_step.onnx"],
      ["encoder", "flow_encoder.onnx"],
      ["estimator", "flow_estimator.onnx"],
      ["vocoder", "vocoder.onnx"],
    ];
    const { provider, sessions } = await openSessions(
      graphs.map(([name, file]) => [name, `${onnxDir}/${file}`] as const),
      execution.onnxProvider
    );
    return new Engine(
      ckpt.algorithm(),
      new GraphemeTextFrontend(tokenizerPath),
      ckpt.generatorTables(),
      ckpt.speakerAffine(),
      provider,
      sessions
    );
  }

  /**
   * The release layout: `tokenizer.json` **beside** the checkpoint.
   *
   * Expects the release layout: `tools/build_release.py` writes a sibling
   * `tokenizer.json`; looking for `<stem>.tokenizer.json` names a file no
   * release contains, so the convenience constructor cannot open an official
   * release without an explicit override.
   */
  static async loadWithDefaults(
    checkpointPath: string,
    onnxDir: string,
    execution: ExecutionOptions = {}
  ): Promise<Engine> {
    const dir = checkpointPath.replace(/[^/\\]*$/, "");
    return Engine.load(checkpointPath, onnxDir, `${dir}tokenizer.json`, execution);
  }

  encode(text: string, language = "en"): number[] {
    // The speech funnel the Python/Swift engines run before tokenising
    // (SpeechText.prepared): scrub invisibles/symbols/footnotes/punctuation,
    // then Polish English-respelling; see speechText.ts and respell.ts.
    return this.frontend.encode(speechText(text, language), language);
  }

  // ------------------------------------------------------------ generator

  private async condRow(voice: VoiceProfile): Promise<Float32Array> {
    const speaker = new ort.Tensor(
      "float32",
      new Float32Array(voice.speakerEmbedding),
      [1, 256]
    );
    const prompt = new ort.Tensor(
      "int64",
      new BigInt64Array(voice.condPromptTokens),
      [1, voice.condPromptTokens.length]
    );
    // The emotion conditioning slot is dead on these weights (distillation
    // collapsed the axis); it is fed the training constant, same as every port.
    const emotion = new ort.Tensor("float32", new Float32Array([EMOTION_NEUTRAL]), [1, 1]);
    const out = await this.cond.run({
      speaker_emb: speaker,
      prompt_tokens: prompt,
      emotion: emotion,
    });
    return new Float32Array(out[this.cond.outNames[0]].data as Float32Array); // [1, 34, 1024]
  }

  private textRow(textTokens: number[]): Float32Array {
    const framed = [START_TEXT_TOKEN, ...textTokens, STOP_TEXT_TOKEN];
    const out = new Float32Array(framed.length * 1024);
    const rows = 1024;
    for (let i = 0; i < framed.length; i++) {
      const id = framed[i];
      const base = id * rows;
      for (let j = 0; j < rows; j++) out[i * rows + j] = this.tables.textEmb[base + j];
      const pbase = i * rows;
      for (let j = 0; j < rows; j++) out[i * rows + j] += this.tables.textPos[pbase + j];
    }
    return out;
  }

  private speechRow(token: number, position: number): Float32Array {
    const out = new Float32Array(1024);
    const sbase = token * 1024;
    const pbase = position * 1024;
    for (let j = 0; j < 1024; j++) out[j] = this.tables.speechEmb[sbase + j] + this.tables.speechPos[pbase + j];
    return out;
  }

  private async prefillEmbeds(
    textTokens: number[],
    voice: VoiceProfile,
    prefix: number[]
  ): Promise<{ embeds: Float32Array; length: number }> {
    const cond = await this.condRow(voice); // [34, 1024]
    const text = this.textRow(textTokens); // [M+2, 1024]
    const bos = this.speechRow(this.config.startSpeechToken, 0); // [1024]
    const rows: Float32Array[] = [cond, text, bos];
    let prefixLen: number;
    if (prefix.length) {
      prefixLen = prefix.length;
      const pe = new Float32Array(prefixLen * 1024);
      for (let i = 0; i < prefixLen; i++) {
        const sbase = prefix[i] * 1024;
        const pbase = (i + 1) * 1024;
        for (let j = 0; j < 1024; j++) pe[i * 1024 + j] = this.tables.speechEmb[sbase + j] + this.tables.speechPos[pbase + j];
      }
      rows.push(pe);
    }
    const total = rows.reduce((a, r) => a + r.length, 0);
    const embeds = new Float32Array(total);
    let off = 0;
    for (const r of rows) {
      embeds.set(r, off);
      off += r.length;
    }
    return { embeds, length: off / 1024 };
  }

  /**
   * Autoregressive decode to the stop token or the cap. Port of the Python
   * `generate`: the sampler owns the law, this loop owns only the EOS floor
   * and the `seen` bookkeeping.
   */
  async generate(
    textTokens: number[],
    voice: VoiceProfile,
    sampler: LRSamplerV1,
    maxNewTokens?: number,
    shouldCancel?: () => boolean,
    prefix: number[] = []
  ): Promise<number[]> {
    const cap = maxNewTokens ?? this.config.sampling.maxNewTokens;
    const floor = eosFloor(textTokens.length, this.config);
    const stop = this.config.stopSpeechToken;

    // `prefix` holds speech tokens from the preceding chunk: fed in as context
    // and NOT returned. `prefillEmbeds` accepts it, and a caller that passes
    // `[]` restarts its
    // pitch contour at every chunk boundary — the audible stutter the prefix
    // exists to remove. They also seed the repetition-penalty state, since a
    // token repeated across a join is as repeated as one within a chunk.
    const { embeds, length: prefillLen } = await this.prefillEmbeds(textTokens, voice, prefix);
    const positions = new BigInt64Array(prefillLen);
    for (let i = 0; i < prefillLen; i++) positions[i] = BigInt(i);

    const prefillOut = await this.prefill.run({
      embeds: new ort.Tensor("float32", embeds, [1, prefillLen, 1024]),
      positions: new ort.Tensor("int64", positions, [prefillLen]),
    });
    const allLogits = prefillOut.logits.data as Float32Array; // [1, T, 8194]
    let logitsLast = new Float32Array(
      allLogits.subarray(
        (prefillLen - 1) * this.config.speechVocabSize,
        prefillLen * this.config.speechVocabSize
      )
    );

    let kv = this.collectKV(prefillOut);

    const seen = new Uint8Array(this.config.speechVocabSize);
    for (const t of prefix) seen[t] = 1;
    const out: number[] = [];
    for (let step = 0; step < cap; step++) {
      if (shouldCancel?.()) break; // token-level barge-in, mirroring Python
      const row = new Float32Array(logitsLast);
      if (out.length < floor) row[stop] = -Infinity;
      const token = sampler.call(row, step, seen);
      out.push(token);
      if (token === stop) break;
      seen[token] = 1;

      // `prefix.length + step + 1`, not `step + 1`: the prefill above put the
      // prefix at speech positions 1..P, so the first generated token is P+1.
      // `step + 1` re-uses a row already written for a carried token and never
      // reaches P+1 — wrong on every chunk that carries a prefix, identical on
      // one that does not. Python's onnx_backend.py:353 and Swift's
      // TokenGenerator.swift:586 index the same way.
      const emb = this.speechRow(token, prefix.length + step + 1); // [1024]
      const pos = new BigInt64Array([BigInt(prefillLen + step)]);
      const stepFeeds: Record<string, OrtTensor> = {
        embeds: new ort.Tensor("float32", emb, [1, 1, 1024]),
        position: new ort.Tensor("int64", pos, [1]),
      };
      for (let i = 0; i < N_LAYERS; i++) {
        stepFeeds[`past_k_${i}`] = new ort.Tensor("float32", kv.k[i], [1, KV_HEADS, kv.k[i].length / (KV_HEADS * HEAD_DIM), HEAD_DIM]);
        stepFeeds[`past_v_${i}`] = new ort.Tensor("float32", kv.v[i], [1, KV_HEADS, kv.v[i].length / (KV_HEADS * HEAD_DIM), HEAD_DIM]);
      }
      const stepOut = await this.step.run(stepFeeds);
      const stepLogits = stepOut.logits.data as Float32Array;
      logitsLast = new Float32Array(stepLogits);
      kv = this.collectKVFromStep(stepOut);
    }
    return out;
  }

  private collectKV(out: Record<string, OrtTensor>): { k: Float32Array[]; v: Float32Array[] } {
    const k: Float32Array[] = [];
    const v: Float32Array[] = [];
    for (let i = 0; i < N_LAYERS; i++) {
      k.push(new Float32Array(out[`kv_k_${i}`].data as Float32Array));
      v.push(new Float32Array(out[`kv_v_${i}`].data as Float32Array));
    }
    return { k, v };
  }

  private collectKVFromStep(out: Record<string, OrtTensor>): { k: Float32Array[]; v: Float32Array[] } {
    const k: Float32Array[] = [];
    const v: Float32Array[] = [];
    for (let i = 0; i < N_LAYERS; i++) {
      k.push(new Float32Array(out[`present_k_${i}`].data as Float32Array));
      v.push(new Float32Array(out[`present_v_${i}`].data as Float32Array));
    }
    return { k, v };
  }

  // -------------------------------------------------------------- renderer

  /**
   * Tokens -> mel via the exported encoder + estimator, Euler-integrated on
   * the host exactly like the Python backend.
   */
  async decodeMel(tokens: number[], voice: VoiceProfile, seed: bigint): Promise<Float32Array> {
    const framed = frameWindows(this.config, tokens, voice);
    const pLen = this.config.window.staticPromptTokens ?? framed.row.length - framed.n;
    const prompt = framed.row.slice(0, pLen);
    const query = framed.row.slice(pLen);
    const tMel = 2 * framed.row.length;

    const muOut = await this.encoder.run({
      prompt_token: new ort.Tensor("int64", prompt, [1, pLen]),
      speech_tokens: new ort.Tensor("int64", query, [1, query.length]),
    });
    const mu = muOut[this.encoder.outNames[0]].data as Float32Array; // [1,80,986]

    // speaker affine: normalize flow_embedding, then W@emb + b
    const emb = voice.flowEmbedding;
    let norm = 0;
    for (let i = 0; i < emb.length; i++) norm += emb[i] * emb[i];
    norm = Math.sqrt(norm);
    const spks = new Float32Array(MEL_BINS);
    for (let i = 0; i < MEL_BINS; i++) {
      let acc = this.spkAffine.bias[i];
      for (let j = 0; j < emb.length; j++) acc += this.spkAffine.weight[i * emb.length + j] * (emb[j] / norm);
      spks[i] = acc;
    }

    const grid = timeGrid(this.config);
    let x = gaussianField(seed, FLOW_NOISE_STREAM, MEL_BINS, tMel); // [1,80,986]
    const cond = framed.cond;
    for (let i = 0; i < grid.length - 1; i++) {
      const t0 = grid[i];
      const dt = grid[i + 1] - t0;
      const vOut = await this.estimator.run({
        x: new ort.Tensor("float32", x, [1, MEL_BINS, tMel]),
        mu: new ort.Tensor("float32", new Float32Array(mu), [1, MEL_BINS, tMel]),
        t: new ort.Tensor("float32", new Float32Array([t0]), [1]),
        spks: new ort.Tensor("float32", spks, [1, MEL_BINS]),
        cond: new ort.Tensor("float32", cond, [1, MEL_BINS, tMel]),
      });
      const v = vOut[this.estimator.outNames[0]].data as Float32Array;
      const next = new Float32Array(x.length);
      for (let j = 0; j < x.length; j++) next[j] = x[j] + dt * v[j];
      x = next;
    }

    // cut to the real speech region: [promptFrames, promptFrames + 2n)
    const n = framed.n;
    const promptFrames = framed.promptFrames;
    const outLen = 2 * n;
    const mel = new Float32Array(MEL_BINS * outLen);
    for (let b = 0; b < MEL_BINS; b++) {
      for (let f = 0; f < outLen; f++) {
        mel[b * outLen + f] = x[b * tMel + (promptFrames + f)];
      }
    }
    return mel;
  }

  /**
   * Mel -> waveform via the exported HiFT graph. Port of the Python vocoder
   * backend: pad to the static frame count, inject Philox randomness.
   */
  async vocode(mel: Float32Array, seed: bigint): Promise<Float32Array> {
    const frames = 2 * this.config.window.maxSpeechTokens;
    const melFrames = mel.length / MEL_BINS;
    const nFrames = Math.min(melFrames, frames);
    const padded = new Float32Array(MEL_BINS * frames);
    for (let b = 0; b < MEL_BINS; b++) {
      for (let f = 0; f < nFrames; f++) padded[b * frames + f] = mel[b * melFrames + f];
    }
    const nSamples = frames * UPSAMPLE_PER_FRAME;
    const phase = new Float32Array(N_HARMONICS);
    const phaseOffsets = symmetricUniforms(seed, VOCODER_PHASE_STREAM, N_HARMONICS - 1, Math.PI);
    for (let i = 0; i < N_HARMONICS - 1; i++) phase[i + 1] = phaseOffsets[i];
    const noise = gaussianField(seed, VOCODER_NOISE_STREAM, N_HARMONICS, nSamples);

    const wavOut = await this.vocoder.run({
      mel: new ort.Tensor("float32", padded, [1, MEL_BINS, frames]),
      phase: new ort.Tensor("float32", phase, [1, N_HARMONICS, 1]),
      noise: new ort.Tensor("float32", noise, [1, N_HARMONICS, nSamples]),
    });
    const wav = wavOut[this.vocoder.outNames[0]].data as Float32Array;
    return wav.slice(0, nFrames * UPSAMPLE_PER_FRAME);
  }

  /**
   * One complete synthesis: text -> tokens -> mel -> audio. `shouldCancel` is
   * polled at every decode step, same as {@link generate}; omit it for no
   * cancellation.
   */
  /**
   * The one path that produces speech tokens.
   *
   * Single-shot and streaming both go through it so they cannot drift: the
   * generation ceiling, the stop-token observation and the artifact detectors
   * are applied once, here, rather than twice and eventually differently.
   *
   * `isTerminal` says whether this chunk ends the passage. A continuation chunk
   * has no sentence end, so its stop peak means nothing and its trailing pause
   * is the sentence's rhythm rather than dead air — the detectors that cut a
   * tail are told so and hold off.
   */
  private async generateInspected(
    textIds: number[],
    voice: VoiceProfile,
    seed: number | bigint,
    prefix: number[],
    isTerminal: boolean,
    shouldCancel?: () => boolean
  ): Promise<{ tokens: number[]; inspection: Inspection; hitCap: boolean }> {
    const pp = this.config.postprocess;
    const floor = eosFloor(textIds.length, this.config);
    let cap = this.config.sampling.maxNewTokens;
    if (pp.mode !== "off") {
      // Applied during generation, not after it: the tokens past the ceiling
      // cost real time on a device and are certain to be discarded. It only
      // ever stops a row that was going to run away.
      cap = Math.min(cap, ceilingFor(textIds.length, pp, this.config.window.maxSpeechTokens));
    }

    // Selective re-roll: a window whose verdict is unfixable — dropout
    // (content missing) or suspect (certainly wrong, nowhere to cut) — is
    // regenerated from a derived seed, up to retryMaxAttempts times. Only
    // condemned windows pay; the ladder is a pure function of the caller's
    // seed, so the same seed still gives the same audio, retries included.
    let gen: number[];
    let inspection: Inspection;
    // True when the row stopped at the ceiling rather than at a stop token:
    // the utterance is cut off mid-sentence. Computed here — where `ended`
    // and the effective cap are both in hand — and carried out, because a
    // caller cannot recompute it after the specials are stripped and the cap
    // is forgotten.
    let hitCap: boolean;
    for (let attempt = 0; ; attempt++) {
      // Retry attempts draw derive(seed, 8 + attempt): clear of the stage
      // streams (1, 2) and below the chunk streams at 16.
      const attemptSeed = attempt === 0 ? seed : this.deriveSeed(seed, 8 + attempt);
      const sampler = new LRSamplerV1(this.config.sampling, attemptSeed);
      if (pp.mode !== "off") sampler.observeEos(this.config.stopSpeechToken, floor);

      const raw = await this.generate(textIds, voice, sampler, cap, shouldCancel, prefix);

      // `gen` is what the shipped engine calls a row: every token the model
      // committed to, with the stop marker itself excluded. Indices into it
      // are decode-step indices, which is what makes the observed peak
      // comparable against it — so the detectors run here, before the specials
      // are stripped and free to renumber anything.
      gen = raw.slice();
      const ended = gen.length > 0 && gen[gen.length - 1] === this.config.stopSpeechToken;
      if (ended) gen.pop();

      const [peakAt, peakProb] = sampler.eosPeak;
      hitCap = !ended && gen.length >= cap;
      inspection = inspect(
        gen,
        {
          textTokenCount: textIds.length,
          minTokens: floor,
          eosPeakAt: peakAt,
          eosPeakProb: peakProb,
          ended,
          isTerminal,
          hitCeiling: hitCap,
        },
        this.config.sampling.silenceTokenIds,
        pp
      );
      const condemned = inspection.reason === "dropout" || inspection.suspect;
      if (!condemned || pp.mode === "off" || attempt >= pp.retryMaxAttempts) break;
    }
    if (pp.mode === "trim" && inspection.keep < gen.length) gen = gen.slice(0, inspection.keep);

    return { tokens: gen.filter((t) => t < this.config.startSpeechToken), inspection, hitCap };
  }

  /**
   * Speak `text` in `voice`.
   *
   * Omit `language` for "the voice's own language" — see
   * {@link resolveLanguage}. Pass one to read text in a language the voice was
   * not enrolled in; that is what cross-lingual synthesis is, and the argument
   * always wins.
   *
   * `options.speed` and `options.previousTokens` are the two execution inputs
   * that are neither text, voice nor seed; see {@link SynthesisOptions}. Both
   * are refused here, before the seconds of generation they would otherwise be
   * discovered after.
   */
  async synthesize(
    text: string,
    voice: VoiceProfile,
    seed: number | bigint,
    language?: string,
    shouldCancel?: () => boolean,
    options?: SynthesisOptions
  ): Promise<{
    audio: Float32Array;
    tokens: number[];
    mel: Float32Array;
    sampleRate: number;
    inspection: Inspection;
    /**
     * True when generation stopped at the token cap rather than at a stop
     * token, so this reading is probably truncated. Truncation is not an
     * error — the audio is real, it is just incomplete — so it travels as a
     * field rather than a rejection.
     */
    hitCap: boolean;
    /** The time-stretch this render was asked for; 1.0 means none was applied. */
    speed: number;
    /**
     * Where this render lands in `audio` — one entry, covering all of it, since
     * a single window is a single chunk. Its `words` are an estimate; see
     * `timing.ts` before building anything that depends on them.
     */
    chunks: ChunkTiming[];
  }> {
    const speed = options?.speed ?? 1.0;
    validateSpeed(speed);
    const prefix = carryFrom(
      options?.previousTokens,
      this.config.chunking.prefixTokens,
      this.config.startSpeechToken
    );
    // The funnel is spelled out rather than left inside `encode`, because the
    // text the timings describe is the text that was tokenised: "I have 3
    // apples." is spoken, and therefore timed, as "I have three apples.".
    const lang = resolveLanguage(language, voice);
    const prepared = speechText(text, lang);
    const textIds = this.frontend.encode(prepared, lang);
    // A single window is the whole passage, so it is terminal.
    const { tokens, inspection, hitCap } = await this.generateInspected(
      textIds,
      voice,
      seed,
      prefix,
      true,
      shouldCancel
    );
    const mel = await this.decodeMel(tokens, voice, this.deriveSeed(seed, 1));
    const rendered = await this.vocode(mel, this.deriveSeed(seed, 2));
    // Last, and after the detectors above rather than before them: they judge
    // pacing by duration per token, and stretching first would move every number
    // they compare against. `speed = 1.0` returns the vocoder's array itself, so
    // the default costs nothing and changes no byte.
    const audio = timeStretch(rendered, this.config.sampleRate, speed);
    return {
      audio,
      tokens,
      mel,
      sampleRate: this.config.sampleRate,
      inspection,
      hitCap,
      // Recorded rather than left to be inferred, because it cannot be: a
      // stretched reading and a naturally faster one are the same numbers
      // afterwards, and the duration alone cannot say which this is.
      speed,
      // Measured on the stretched waveform — the one the caller is holding — so
      // there is no `1/speed` correction to apply anywhere.
      chunks: timeline(
        [{ text: prepared, samples: audio.length, tokens: tokens.length }],
        this.config.sampleRate
      ),
    };
  }

  /**
   * Speak text of any length, splitting it across windows.
   *
   * This binding had no long-form path: `synthesize` refused anything over one
   * window while the documentation called the port supported and
   * conformance-verified. Two things make the joins match Python's rather than
   * merely existing:
   *
   * * **Per-chunk seeds.** Each chunk draws from `derive(seed, 16 + index)`,
   *   so a chunk's audio does not depend on how many chunks came before it and
   *   stopping early cannot change what was already produced.
   * * **Prefix carry.** The last `chunking.prefixTokens` speech tokens of a
   *   chunk are fed into the next one as context and dropped from its output.
   *   Without it every chunk restarts its pitch contour like a fresh sentence;
   *   measured on the reference voice, the restart is ~74 Hz at the join
   *   against ~7 Hz with a 6-token prefix.
   */
  /**
   * Speak `text` chunk by chunk, yielding each as it becomes ready.
   *
   * The difference from {@link Engine.synthesizeLong} is delivery, not
   * synthesis: time to first audio is set by the first chunk rather than by the
   * whole passage, which is what lets a reading app start playing a sentence
   * while the rest is still being made.
   *
   * `shouldCancel` is polled on **every decode step**, so an interrupt is
   * honoured within one forward pass rather than at the next chunk boundary.
   * The partial chunk is discarded without being rendered: those tokens are
   * speech the listener has already stopped wanting.
   *
   * Cancelling does not un-deliver audio. A chunk already yielded from this
   * generator is the caller's, and it will play unless the caller drops it.
   *
   * Omit `language` for "the voice's own language" — see
   * {@link resolveLanguage}. Resolved once here, before splitting, so every
   * chunk of a passage is read in the same language.
   *
   * `options.speed` stretches each chunk independently, which is the same
   * independence the seeds and the prefix already have: a chunk's audio must not
   * depend on how many came before it, or a listener who stops early would have
   * heard something different from one who did not.
   * `options.previousTokens` seeds the carry, so the first chunk of *this* call
   * is conditioned on the tail of a previous one — the same conditioning the
   * joins inside a passage already use, with the carry variable below simply
   * starting non-empty.
   *
   * Both are validated inside the generator body, which an async generator does
   * not run until the first `next()`: a refused speed therefore surfaces on the
   * first iteration rather than at the call. That is where the first byte of
   * work would have happened anyway, and {@link Engine.synthesize} — the path
   * that renders one window — refuses immediately.
   */
  async *stream(
    text: string,
    voice: VoiceProfile,
    seed: number | bigint,
    language?: string,
    shouldCancel?: () => boolean,
    options?: SynthesisOptions
  ): AsyncGenerator<{
    index: number;
    audio: Float32Array;
    tokens: number[];
    mel: Float32Array;
    /**
     * What the artifact detectors concluded about this chunk. Per chunk rather
     * than aggregated because chunks fail independently: one hallucinated tail
     * among six clean ones is the case worth seeing.
     */
    inspection: Inspection;
    /**
     * True when generation stopped at the token cap rather than at a stop
     * token, so this chunk is cut off mid-sentence. Per chunk, for the same
     * reason the inspection is: chunks truncate independently.
     */
    hitCap: boolean;
    /**
     * What this chunk was asked to say, after the speech funnel — the text that
     * was tokenised, which is not always the caller's substring (numbers become
     * words, Polish respells embedded English).
     */
    text: string;
    /**
     * Where this chunk lands **in its own audio**, starting at zero: a streamed
     * chunk is its own result and cannot know what preceded it, so anything else
     * would be a guess about the caller's playback. A caller stitching the
     * stream adds the offsets, which is exactly what
     * {@link Engine.synthesizeLong} does — in samples rather than seconds.
     */
    timing: ChunkTiming;
  }> {
    const speed = options?.speed ?? 1.0;
    validateSpeed(speed);
    const lang = resolveLanguage(language, voice);
    // The funnel runs on the whole text BEFORE splitting: Polish respelling
    // changes the length ("download" -> "dałnloud"), so a budget computed
    // first would be a budget for text the engine never speaks.
    const prepared = speechText(text, lang);
    const chunks = splitText(prepared, this.config.chunking);
    if (chunks.length === 0) throw new Error("nothing to speak");

    const prefixLen = this.config.chunking.prefixTokens;
    let carry: number[] = carryFrom(
      options?.previousTokens,
      prefixLen,
      this.config.startSpeechToken
    );

    for (let index = 0; index < chunks.length; index++) {
      if (shouldCancel?.()) break;
      // Each chunk gets its own derived seed, so the same passage sounds
      // identical whether or not the caller stops early: a chunk's audio does
      // not depend on how many came before it.
      const chunkSeed = this.deriveSeed(seed, CHUNK_STREAM_BASE + index);
      const ids = this.frontend.encode(chunks[index], lang);
      // Only the last chunk ends the passage.
      const { tokens: chunkTokens, inspection, hitCap } = await this.generateInspected(
        ids,
        voice,
        chunkSeed,
        carry,
        index === chunks.length - 1,
        shouldCancel
      );
      if (shouldCancel?.()) break;
      const mel = await this.decodeMel(chunkTokens, voice, this.deriveSeed(chunkSeed, 1));
      const rendered = await this.vocode(mel, this.deriveSeed(chunkSeed, 2));
      // The stretch is the last stage, after the detectors have judged this
      // chunk's pacing against its text; see {@link Engine.synthesize}.
      const wav = timeStretch(rendered, this.config.sampleRate, speed);
      carry = prefixLen ? chunkTokens.slice(-prefixLen) : [];
      yield {
        index,
        audio: wav,
        tokens: chunkTokens,
        mel,
        inspection,
        hitCap,
        text: chunks[index],
        timing: timeline(
          [{ text: chunks[index], samples: wav.length, tokens: chunkTokens.length }],
          this.config.sampleRate
        )[0],
      };
    }
  }

  /**
   * Speak text of any length as one waveform.
   *
   * Exactly {@link Engine.stream} with the chunks concatenated — one loop, so
   * the streaming and whole-passage paths cannot drift apart. Use `stream` when
   * you want to start playing before the passage is finished.
   *
   * Omit `language` for "the voice's own language"; left unresolved here so
   * {@link Engine.stream} resolves it once, on the one path that renders.
   *
   * `options` is passed straight through: `speed` is applied per chunk, exactly
   * as {@link Engine.stream} applies it, so the two paths still produce the same
   * waveform, and `previousTokens` conditions the *first* chunk — every chunk
   * after it is conditioned on the one before, as always.
   */
  async synthesizeLong(
    text: string,
    voice: VoiceProfile,
    seed: number | bigint,
    language?: string,
    shouldCancel?: () => boolean,
    options?: SynthesisOptions
  ): Promise<{
    audio: Float32Array;
    tokens: number[];
    mel: Float32Array;
    sampleRate: number;
    /** The time-stretch this render was asked for; 1.0 means none was applied. */
    speed: number;
    /**
     * True when any chunk stopped at the token cap rather than at a stop
     * token, so the passage is probably truncated — one truncated chunk
     * truncates the whole. Same field `synthesize` returns.
     */
    hitCap: boolean;
    /**
     * Where every chunk lands in `audio`, in order and adjacent: chunk *k*'s
     * `end` is the same float as chunk *k+1*'s `start`, and the last `end` is
     * the whole duration.
     */
    chunks: ChunkTiming[];
  }> {
    const audio: Float32Array[] = [];
    const mels: Float32Array[] = [];
    const tokens: number[] = [];
    const spans: ChunkSpan[] = [];
    let hitCap = false;

    for await (const chunk of this.stream(text, voice, seed, language, shouldCancel, options)) {
      audio.push(chunk.audio);
      mels.push(chunk.mel);
      tokens.push(...chunk.tokens);
      hitCap = hitCap || chunk.hitCap;
      spans.push({ text: chunk.text, samples: chunk.audio.length, tokens: chunk.tokens.length });
    }

    const total = audio.reduce((n, a) => n + a.length, 0);
    const joined = new Float32Array(total);
    let at = 0;
    for (const a of audio) {
      joined.set(a, at);
      at += a.length;
    }
    return {
      audio: joined,
      tokens,
      mel: concatMelAlongTime(mels),
      sampleRate: this.config.sampleRate,
      speed: options?.speed ?? 1.0,
      hitCap,
      // Rebuilt from the parts rather than shifting each part's own timing by a
      // running float: `timeline` accumulates sample offsets as integers, so the
      // joins are exact and every chunk's `end` is the next one's `start` down
      // to the last bit.
      chunks: timeline(spans, this.config.sampleRate),
    };
  }

  private deriveSeed(seed: number | bigint, stream: number): bigint {
    // Mirrors engine._derive: (seed * PHI + stream * PSI) & 2^64. BigInt keeps
    // the 64-bit product exact, which JS numbers cannot — and normalizeSeed
    // rejects a `number` seed that already lost precision before arriving.
    const PHI = 0x9e3779b97f4a7c15n;
    const PSI = 0xbf58476d1ce4e5b9n;
    const MASK = (1n << 64n) - 1n;
    return (normalizeSeed(seed) * PHI + BigInt(stream) * PSI) & MASK;
  }
}

export { loadVoice, LRSamplerV1 };
