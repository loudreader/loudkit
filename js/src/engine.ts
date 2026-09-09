/**
 * The engine: all three stages over the exported ONNX graphs, fp32, no torch.
 *
 * Bit-parity port of `loudkit.backends.onnx_backend`. The generator runs
 * entirely on the graphs (`t3_cond`, 34-slot conditioning; `t3_prefill`, one
 * causal forward giving every-position logits plus a KV cache; `t3_step`, one
 * decode step against the cache) with the framing, embeddings, positions and
 * the sampler loop here in JS. The renderer is `flow_encoder` +
 * `flow_estimator` integrated with Euler steps, then the HiFT `vocoder`.
 *
 * The conformance fixture pins the whole pipeline: same text, voice and seed
 * give the same tokens and the same mel/waveform band as the Python engine.
 */

import { existsSync } from "node:fs";
import { basename, join } from "node:path";

import { Checkpoint } from "./checkpoint.js";
import { splitInHalf, splitText } from "./chunking.js";
import { Enroller, asSamples, profileFrom } from "./enroll.js";
import {
  InvalidTokensError,
  LoudkitError,
  NothingToSpeakError,
  VoiceNotFoundError,
  WindowOverflowError,
} from "./errors.js";
import {
  type ExecutionOptions,
  type ResolvedONNXProvider,
  describeExecution,
} from "./execution.js";
import { fingerprint } from "./fingerprint.js";
import { GraphemeTextFrontend } from "./frontend.js";
import {
  VOICE_DIR,
  VOICE_SUFFIX,
  checkExportRecord,
  ensureBundle,
  ensureCloning,
  listVoices,
  resolveBundle,
} from "./hub.js";
import { gaussianField, symmetricUniforms } from "./noise.js";
import { ort, type OrtTensor } from "./ort.js";
import {
  RETRY_LADDER_HEADROOM,
  ceilingFor,
  inspect,
  type Inspection,
} from "./postprocess.js";
import { deriveSeed, normalizeSeed } from "./rng.js";
import { LRSamplerV1 } from "./sampler.js";
import { Session, openSessions } from "./session.js";
import { speechText } from "./speechText.js";
import { timeStretch, fadeEdges, validateSpeed, EDGE_FADE_SECONDS } from "./timestretch.js";
import { timeline, type ChunkSpan, type ChunkTiming } from "./timing.js";
import { AlgorithmConfig, VoiceProfile } from "./types.js";
import { EMOTION_NEUTRAL, loadVoice } from "./voice.js";
import { withWav, type WavOutput } from "./wavFile.js";
import {
  FLOW_NOISE_STREAM,
  START_TEXT_TOKEN,
  STOP_TEXT_TOKEN,
  VOCODER_NOISE_STREAM,
  VOCODER_PHASE_STREAM,
  eosFloor,
  frameForEncoder,
  frameWindows,
  requireStaticWindow,
  timeGrid,
} from "./windowing.js";

/**
 * Chunk seeds start here, clear of the per-stage streams (1 = flow, 2 =
 * vocoder). Mirrors `_STREAM_CHUNK` in loudkit.engine.
 */
const CHUNK_STREAM_BASE = 16;

/**
 * Where the retry ladder's streams start. `RETRY_LADDER_HEADROOM` attempts fit
 * between here and `CHUNK_STREAM_BASE`, which is what bounds
 * `retry_max_attempts`.
 */
const RETRY_STREAM_BASE = CHUNK_STREAM_BASE - RETRY_LADDER_HEADROOM;

/**
 * Mirrors `_STREAM_RESPLIT` in `loudkit.engine`: the second half of a re-split
 * chunk draws from its own stream off the chunk's seed. Chunk streams run from
 * `CHUNK_STREAM_BASE` upwards with no ceiling, so there is no room above them
 * to claim; deriving off the chunk seed leaves only the values already drawn
 * from it to avoid, which are the flow at 1, the vocoder at 2, and the retry
 * ladder from 8 up.
 */
const RESPLIT_STREAM = 4096;

/**
 * What a synthesis reads as when neither the caller nor the voice says.
 *
 * Reached less often than it looks: `loadVoice` defaults a *missing* header key
 * to `"en"`, and Python writes the key, so an empty string only
 * arrives from a profile built in memory or a header hand-edited to `""`. A
 * profile file with no language field inherits nothing: it loads as `"en"`.
 */
const FALLBACK_LANGUAGE = "en";

/**
 * The language chain: the argument, then the voice's recorded language, then
 * English.
 *
 * Without the voice link, `engine.synthesize("Cześć", polishVoice, 7)` runs
 * Polish text through the English frontend (English number words, English
 * abbreviation expansion, no Polish respelling) and says so nowhere. A profile
 * records the language of the audio it was enrolled from, so the voice is the
 * better answer than a constant.
 *
 * Passing `language` is how cross-lingual synthesis is requested: an English
 * voice reading Polish text is `"pl"`, and the argument always wins over the
 * profile.
 *
 * `||` rather than `??` for the profile: an empty language id is not a
 * language, it would tag the text `[]`, and `loadVoice` only defaults the field
 * when the header key is absent, so a header that says `"language": ""` still
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
 * What varies about one synthesis besides the text and the voice. Every field
 * has a default: seed 0, the voice's own language, normal speed, no previous
 * tokens, no cancellation. None of them moves the fingerprint.
 */
export interface SynthesisOptions {
  /** The seed. `0` when omitted. */
  seed?: number | bigint;

  /**
   * Language of the text. Omitted means the voice's own; name one only to
   * read text in a language the voice was not enrolled in.
   */
  language?: string;

  /**
   * Playback speed in `[0.5, 2.0]`; greater than one is faster and pitch is
   * preserved. `1.0` is an exact bypass: the vocoder's own array, untouched.
   */
  speed?: number;

  /**
   * Speech tokens this utterance continues from: the `tokens` of the call
   * before it. Any length; only the tail is used.
   */
  previousTokens?: number[];

  /**
   * Polled on every decode step. The chunk being generated is discarded, not
   * rendered: `synthesize` then throws {@link CancelledError}, and `stream`
   * ends after the chunks it already yielded.
   */
  shouldCancel?: () => boolean;
}

/**
 * Thrown by `synthesize` and `synthesizeWindow` when `options.shouldCancel`
 * returned true: nothing was produced. `stream` does not throw it; the chunks
 * already yielded are the partial, and the caller flipped the flag.
 */
export class CancelledError extends LoudkitError {
  override get code(): string {
    return "cancelled";
  }

  constructor() {
    super("cancelled: shouldCancel returned true");
    this.name = "CancelledError";
  }
}

/** One chunk of {@link Engine.stream}, handed over as soon as it is rendered. */
export interface StreamChunk {
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
  hitTokenCap: boolean;
  /**
   * What this chunk was asked to say, after the speech funnel: the text that
   * was tokenised, which is not always the caller's substring (numbers become
   * words, Polish respells embedded English).
   */
  text: string;
  /**
   * Where this chunk lands **in its own audio**, starting at zero: a streamed
   * chunk is its own result and cannot know what preceded it, so anything else
   * would be a guess about the caller's playback. A caller stitching the
   * stream adds the offsets, which is exactly what
   * {@link Engine.synthesize} does, in samples rather than seconds.
   */
  timing: ChunkTiming;
}

/**
 * The conditioning context a call inherits from the one before it.
 *
 * The same slice the streaming loop takes between two chunks, the last
 * `chunking.prefixTokens`, applied to tokens that came from a different call.
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
 * loaded engine: the unit under test here is the slice-and-check, and every other
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
  startSpeechToken: number,
  decode: "single" | "fusion_mtp2" = "single"
): number[] {
  if (previousTokens === undefined || previousTokens.length === 0) return [];
  for (const token of previousTokens) {
    if (!Number.isInteger(token) || !(token >= 0 && token < startSpeechToken)) {
      throw new InvalidTokensError(
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
  let end = previousTokens.length;
  if (decode === "fusion_mtp2") end -= end % 2;
  let start = end - prefixTokens;
  if (decode === "fusion_mtp2") start -= ((start % 2) + 2) % 2;
  return previousTokens.slice(Math.max(0, start), end);
}

/**
 * Concatenate row-major `[MEL_BINS, frames]` mels along the TIME axis.
 *
 * Appending the flat buffers end to end, which is the obvious thing, is not
 * concatenation: after the first chunk the next chunk's bin 0 lands
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
const MEL_BINS = 80;
const N_HARMONICS = 9;
const UPSAMPLE_PER_FRAME = 480;
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
   * The execution provider these six graphs were opened on, never `"auto"`,
   * always the one that ran.
   */
  readonly onnxProvider: ResolvedONNXProvider;

  /**
   * One line for logs, benchmark rows and bug reports, in the `exec[...]`
   * shape Python and Swift print.
   *
   * The algorithm half is the fingerprint and the recipe name, not Python's
   * full knob list: the fingerprint already hashes those knobs, and printing
   * the same floats through two languages' number formatters (Python's `1.0`
   * against JS's `1`) would make two identical engines read as different.
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
  private head2?: Session;
  private fusion?: ReturnType<Checkpoint["fusionWeights"]>;
  private encoder: Session;
  private estimator: Session;
  private vocoder: Session;
  /** The release directory this engine was loaded from, when it was one. */
  private readonly dir?: string;
  /**
   * The repo id the release was fetched by, so `enroll` can fetch the rest of
   * it into `dir`. Undefined for a directory of your own.
   */
  private readonly repo?: string;
  private readonly onnxDir: string;
  /**
   * The open enrollment graphs, held as the promise rather than its result.
   *
   * `this.enroller ??= await load()` reads the field before the await, so two
   * concurrent first calls to `enroll` both open the three graphs and the
   * second assignment drops the first with no `close()`: three native sessions
   * leaked, the class `Session` exists to prevent. Cleared when a load fails,
   * so the next call opens the graphs rather than replaying the rejection.
   */
  private enrolling?: Promise<Enroller>;

  private constructor(
    config: AlgorithmConfig,
    frontend: GraphemeTextFrontend,
    tables: Engine["tables"],
    spkAffine: Engine["spkAffine"],
    provider: ResolvedONNXProvider,
    sessions: Record<string, Session>,
    onnxDir: string,
    dir?: string,
    repo?: string,
    fusion?: ReturnType<Checkpoint["fusionWeights"]>
  ) {
    this.config = config;
    this.frontend = frontend;
    this.tables = tables;
    this.spkAffine = spkAffine;
    this.onnxProvider = provider;
    this.cond = sessions.cond;
    this.prefill = sessions.prefill;
    this.step = sessions.step;
    this.head2 = sessions.head2;
    this.fusion = fusion;
    this.encoder = sessions.encoder;
    this.estimator = sessions.estimator;
    this.vocoder = sessions.vocoder;
    this.onnxDir = onnxDir;
    this.dir = dir;
    this.repo = repo;
  }

  /**
   * Release the native graph sessions: a checkpoint's worth of memory, handed
   * back when a second engine is built. Safe to call twice; the engine must
   * not be used afterwards.
   */
  async close(): Promise<void> {
    await Promise.all(
      [this.cond, this.prefill, this.step, this.encoder, this.estimator, this.vocoder].map(
        (s) => s.close()
      )
    );
    await this.head2?.close();
    const enrolling = this.enrolling;
    this.enrolling = undefined;
    // A load that never resolved has no sessions to hand back, and `close`
    // stays callable twice.
    if (enrolling !== undefined) await enrolling.then((e) => e.close(), () => undefined);
  }

  /**
   * Build an engine from a release directory, or from a repo id.
   *
   * A repo id (`"loudreader/loudr-1"`) is fetched into the cache the first
   * time and read from there afterwards; a path that exists on disk is always
   * read as a path. `execution.onnxProvider` names the execution provider;
   * omitted, `"auto"` takes the best one this build offers.
   */
  static async load(ref: string, execution: ExecutionOptions = {}): Promise<Engine> {
    const dir = await ensureBundle(ref);
    // `ensureBundle` answers for a directory or a repo id and nothing else.
    const repo = dir === ref ? undefined : ref;
    const bundle = resolveBundle(dir);
    return Engine.open(bundle.checkpoint, bundle.onnxDir, bundle.tokenizer, execution, dir, repo);
  }

  /**
   * Build an engine from a layout of your own: the checkpoint, the directory
   * of ONNX graphs and `tokenizer.json`, named one by one. `voices`, `voice`
   * and `enroll` need a release directory and are not available on an engine
   * opened this way.
   */
  static async loadPaths(
    checkpointPath: string,
    onnxDir: string,
    tokenizerPath: string,
    execution: ExecutionOptions = {}
  ): Promise<Engine> {
    return Engine.open(checkpointPath, onnxDir, tokenizerPath, execution);
  }

  private static async open(
    checkpointPath: string,
    onnxDir: string,
    tokenizerPath: string,
    execution: ExecutionOptions,
    dir?: string,
    repo?: string
  ): Promise<Engine> {
    const ckpt = Checkpoint.open(checkpointPath);
    const config = ckpt.algorithm();
    const fusion = ckpt.fusionWeights();
    const graphs: Array<[string, string]> = [
      ["cond", "t3_cond.onnx"],
      ["prefill", "t3_prefill.onnx"],
      ["step", config.decode === "fusion_mtp2" ? "t3_pair_step.onnx" : "t3_step.onnx"],
      ["encoder", "flow_encoder.onnx"],
      ["estimator", "flow_estimator.onnx"],
      ["vocoder", "vocoder.onnx"],
    ];
    if (config.decode === "fusion_mtp2") graphs.push(["head2", "t3_head2.onnx"]);
    // Validate file-backed inputs before acquiring native sessions.
    const frontend = new GraphemeTextFrontend(tokenizerPath);
    const tables = ckpt.generatorTables();
    const affine = ckpt.speakerAffine();
    // For the same reason: a manifest that frames a window the exported graphs
    // cannot run is named here rather than as a tensor shape inside onnxruntime.
    requireStaticWindow(config);
    // And for the same reason again: graphs from one export of one checkpoint
    // or they are not a set, and a mixed set speaks this checkpoint's tokens
    // through another one's renderer.
    await checkExportRecord(
      onnxDir,
      checkpointPath,
      ckpt.manifest,
      fingerprint(config),
      config.eulerSteps,
      graphs.map(([, file]) => file)
    );
    const { provider, sessions } = await openSessions(
      graphs.map(([name, file]) => [name, `${onnxDir}/${file}`] as const),
      execution.onnxProvider
    );
    return new Engine(
      config,
      frontend,
      tables,
      affine,
      provider,
      sessions,
      onnxDir,
      dir,
      repo,
      fusion
    );
  }

  private release(what: string): string {
    if (this.dir === undefined) {
      throw new Error(
        `this engine was opened with Engine.loadPaths, which has no release to take ${what} from`
      );
    }
    return this.dir;
  }

  /** The names `voice` will accept, sorted. */
  voices(): string[] {
    return listVoices(this.release("voices"));
  }

  /** One shipped voice by name, such as `"joe"`. */
  voice(name: string): VoiceProfile {
    const dir = this.release("voices");
    if (name === "" || name.startsWith(".") || name.includes("/") || name.includes("\\")) {
      throw new Error(`${name}: a voice is named, not addressed. Pass a bare name such as "joe"`);
    }
    const path = join(dir, VOICE_DIR, `${name}${VOICE_SUFFIX}`);
    if (existsSync(path)) return loadVoice(path);
    const available = listVoices(dir);
    throw new VoiceNotFoundError(
      available.length > 0
        ? `${dir} has no voice named ${name}. It has: ${available.join(", ")}.`
        : `${dir} holds no ${VOICE_DIR}/ directory, so it ships no voices`
    );
  }

  /**
   * Clone a voice from a recording: a WAV path, WAV bytes, or samples with
   * their rate. `language` is what the voice reads in by default; `"en"`
   * when omitted.
   *
   * The three enrollment graphs are not in a plain fetch. An engine loaded
   * by repo id fetches them into its own cache directory the first time; one
   * loaded from a directory needs a fetch made with `{ cloning: true }`.
   */
  async enroll(
    source: Float32Array | Uint8Array | string,
    options: { name?: string; language?: string; sampleRate?: number } = {}
  ): Promise<VoiceProfile> {
    const dir = this.release("the enrollment graphs");
    await ensureCloning(dir, this.repo);
    // Assigned before the await, so a second call that arrives while the first
    // is still loading waits on the same graphs rather than opening its own.
    this.enrolling ??= Enroller.load(this.onnxDir, { onnxProvider: this.onnxProvider });
    let enroller: Enroller;
    try {
      enroller = await this.enrolling;
    } catch (err) {
      this.enrolling = undefined;
      throw err;
    }
    const { audio, sampleRate } = asSamples(source, options.sampleRate);
    const enrolled = await enroller.enroll(audio, sampleRate);
    const name =
      options.name ?? (typeof source === "string" ? basename(source, ".wav") : "voice");
    return profileFrom(enrolled, { name, language: options.language ?? "en", sampleRate });
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

  private pairRow(first: number, second: number, position: number): Float32Array {
    const w = this.fusion;
    // Only the fusion decode reaches here, and it is opened with both the
    // head2 graph and these weights. Named rather than asserted: without them
    // a pair row is arithmetic on undefined, which reads as a bad voice.
    if (!w) {
      throw new Error("fusion_mtp2 decode was opened without the fused embedding weights");
    }
    const a = this.tables.speechEmb.subarray(first * 1024, (first + 1) * 1024);
    const b = this.tables.speechEmb.subarray(second * 1024, (second + 1) * 1024);
    const hidden = new Float32Array(1024);
    for (let i = 0; i < 1024; i++) {
      let sum = 0;
      for (let j = 0; j < 1024; j++) sum += a[j] * w.first[i * 2048 + j] + b[j] * w.first[i * 2048 + 1024 + j];
      const x = Math.fround(Math.fround(sum) + w.firstBias[i]);
      hidden[i] = Math.fround(Math.fround(0.5 * x) * Math.fround(1 + Math.fround(erf(Math.fround(x / Math.SQRT2)))));
    }
    const out = new Float32Array(1024);
    for (let i = 0; i < 1024; i++) {
      let sum = 0;
      for (let j = 0; j < 1024; j++) sum += hidden[j] * w.second[i * 1024 + j];
      const fused = Math.fround(Math.fround(sum) + w.secondBias[i]);
      out[i] = Math.fround(Math.fround(0.5 * Math.fround(a[i] + b[i])) + fused) + this.tables.speechPos[position * 1024 + i];
    }
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
      prefixLen = this.config.decode === "fusion_mtp2" ? prefix.length / 2 : prefix.length;
      const pe = new Float32Array(prefixLen * 1024);
      for (let i = 0; i < prefixLen; i++) {
        if (this.config.decode === "fusion_mtp2") {
          pe.set(this.pairRow(prefix[2 * i], prefix[2 * i + 1], i + 1), i * 1024);
          continue;
        }
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
    if (this.config.decode === "fusion_mtp2") prefix = prefix.slice(0, prefix.length - prefix.length % 2);
    const cap = maxNewTokens ?? this.config.sampling.maxNewTokens;
    const floor = eosFloor(textTokens.length, this.config);
    const stop = this.config.stopSpeechToken;

    // `prefix` holds speech tokens from the preceding chunk: fed in as context
    // and NOT returned. `prefillEmbeds` accepts it, and a caller that passes
    // `[]` restarts its
    // pitch contour at every chunk boundary, the audible stutter the prefix
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

    let kv = this.collectKV(prefillOut, "kv");
    let hidden = prefillOut.hidden;

    const seen = new Uint8Array(this.config.speechVocabSize);
    for (const t of prefix) seen[t] = 1;
    const out: number[] = [];
    for (let step = 0; step < cap; step++) {
      // Token-level barge-in: the partial row is discarded, not returned.
      if (shouldCancel?.()) throw new CancelledError();
      const row = new Float32Array(logitsLast);
      if (out.length < floor) row[stop] = -Infinity;
      const token = sampler.call(row, step, seen);
      out.push(token);
      if (token === stop) break;
      seen[token] = 1;

      if (this.head2) {
        if (out.length >= cap) break;
        if (shouldCancel?.()) throw new CancelledError();
        const secondOut = await this.head2.run({
          hidden,
          first_id: new ort.Tensor("int64", new BigInt64Array([BigInt(token)]), [1]),
        });
        const secondLogits = new Float32Array(secondOut.logits.data as Float32Array);
        if (out.length < floor) secondLogits[stop] = -Infinity;
        const second = sampler.call(secondLogits, out.length, seen);
        out.push(second);
        step++;
        if (second === stop || out.length >= cap) break;
        seen[second] = 1;
        const pair = out.length / 2 - 1;
        const feeds: Record<string, OrtTensor> = {
          pair_ids: new ort.Tensor("int64", new BigInt64Array([BigInt(token), BigInt(second)]), [1, 2]),
          speech_position: new ort.Tensor("int64", new BigInt64Array([BigInt(prefix.length / 2 + pair + 1)]), [1]),
          position: new ort.Tensor("int64", new BigInt64Array([BigInt(prefillLen + pair)]), [1]),
        };
        this.feedKV(feeds, kv);
        const next = await this.step.run(feeds);
        logitsLast = new Float32Array(next.logits.data as Float32Array);
        hidden = next.hidden;
        kv = this.collectKV(next, "present");
        continue;
      }

      // `prefix.length + step + 1`, not `step + 1`: the prefill above put the
      // prefix at speech positions 1..P, so the first generated token is P+1.
      // `step + 1` re-uses a row already written for a carried token and never
      // reaches P+1: wrong on every chunk that carries a prefix, identical on
      // one that does not. The decode loops in `onnx_backend.generate` and in
      // Swift's `TokenGenerator` index the same way.
      const emb = this.speechRow(token, prefix.length + step + 1); // [1024]
      const pos = new BigInt64Array([BigInt(prefillLen + step)]);
      const stepFeeds: Record<string, OrtTensor> = {
        embeds: new ort.Tensor("float32", emb, [1, 1, 1024]),
        position: new ort.Tensor("int64", pos, [1]),
      };
      this.feedKV(stepFeeds, kv);
      const stepOut = await this.step.run(stepFeeds);
      const stepLogits = stepOut.logits.data as Float32Array;
      logitsLast = new Float32Array(stepLogits);
      kv = this.collectKV(stepOut, "present");
    }
    return out;
  }

  private feedKV(feeds: Record<string, OrtTensor>, kv: { k: Float32Array[]; v: Float32Array[] }): void {
    for (let i = 0; i < kv.k.length; i++) {
      feeds[`past_k_${i}`] = new ort.Tensor("float32", kv.k[i], [1, KV_HEADS, kv.k[i].length / (KV_HEADS * HEAD_DIM), HEAD_DIM]);
      feeds[`past_v_${i}`] = new ort.Tensor("float32", kv.v[i], [1, KV_HEADS, kv.v[i].length / (KV_HEADS * HEAD_DIM), HEAD_DIM]);
    }
  }

  private collectKV(
    out: Record<string, OrtTensor>, prefix: "kv" | "present"
  ): { k: Float32Array[]; v: Float32Array[] } {
    const k: Float32Array[] = [];
    const v: Float32Array[] = [];
    for (let i = 0; `${prefix}_k_${i}` in out; i++) {
      k.push(new Float32Array(out[`${prefix}_k_${i}`].data as Float32Array));
      v.push(new Float32Array(out[`${prefix}_v_${i}`].data as Float32Array));
    }
    if (!k.length) throw new Error("generator returned no KV cache");
    return { k, v };
  }

  // -------------------------------------------------------------- renderer

  /**
   * Tokens -> mel via the exported encoder + estimator, Euler-integrated on
   * the host exactly like the Python backend.
   */
  async decodeMel(tokens: number[], voice: VoiceProfile, seed: bigint): Promise<Float32Array> {
    const framed = frameWindows(this.config, tokens, voice);
    const { prompt, query } = frameForEncoder(framed, this.config);
    const tMel = 2 * framed.row.length;

    const muOut = await this.encoder.run({
      prompt_token: new ort.Tensor("int64", prompt, [1, prompt.length]),
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
   * The one path that produces speech tokens.
   *
   * Single-shot and streaming both go through it so they cannot drift: the
   * generation ceiling, the stop-token observation and the artifact detectors
   * are applied once, here, rather than twice and eventually differently.
   *
   * `isTerminal` says whether this chunk ends the passage. A continuation chunk
   * has no sentence end, so its stop peak means nothing and its trailing pause
   * is the sentence's rhythm rather than dead air, so the detectors that cut a
   * tail are told so and hold off.
   */
  private async generateInspected(
    textIds: number[],
    voice: VoiceProfile,
    seed: number | bigint,
    prefix: number[],
    isTerminal: boolean,
    shouldCancel?: () => boolean
  ): Promise<{
    tokens: number[];
    inspection: Inspection;
    hitTokenCap: boolean;
    hitWindow: boolean;
  }> {
    const pp = this.config.postprocess;
    const floor = eosFloor(textIds.length, this.config);
    let cap = this.config.sampling.maxNewTokens;
    if (pp.mode !== "off") {
      // Applied during generation, not after it: the tokens past the ceiling
      // cost real time on a device and are certain to be discarded. It only
      // ever stops a row that was going to run away.
      cap = Math.min(cap, ceilingFor(textIds.length, pp, this.config.window.maxSpeechTokens));
    }

    // Selective re-roll: a window whose verdict is unfixable (dropout,
    // content missing; or suspect, certainly wrong with nowhere to cut) is
    // regenerated from a derived seed, up to retryMaxAttempts times. Only
    // condemned windows pay; the ladder is a pure function of the caller's
    // seed, so the same seed still gives the same audio, retries included.
    let gen: number[];
    let inspection: Inspection;
    // True when the row stopped at the ceiling rather than at a stop token:
    // the utterance is cut off mid-sentence. Computed here, where `ended`
    // and the effective cap are both in hand, and carried out, because a
    // caller cannot recompute it after the specials are stripped and the cap
    // is forgotten.
    let hitTokenCap: boolean;
    let hitWindow: boolean;
    // When the ladder exhausts with every attempt condemned, the attempt that
    // ships is the *best* seen, not the last: fewest tokens in the
    // true-silence set, integer and portable, like the detectors. Measured:
    // on the worst voices 30% of condemned fires exhaust the ladder, and
    // keeping the last attempt shipped rows worse than the first. The render
    // census gates the count where the checkpoint carries one; the configured
    // silence list is the fallback.
    const deadAir = new Set(
      pp.silenceRenderIds.length > 0
        ? pp.silenceRenderIds
        : this.config.sampling.silenceTokenIds
    );
    let best: {
      count: number;
      gen: number[];
      inspection: Inspection;
      hitTokenCap: boolean;
      hitWindow: boolean;
    } | null =
      null;
    for (let attempt = 0; ; attempt++) {
      // Retry attempts draw derive(seed, RETRY_STREAM_BASE + attempt): clear
      // of the stage streams (1, 2) and below the chunk streams at 16.
      const attemptSeed =
        attempt === 0 ? seed : deriveSeed(seed, RETRY_STREAM_BASE + attempt);
      const sampler = new LRSamplerV1(this.config.sampling, attemptSeed);
      if (pp.mode !== "off") sampler.observeEos(this.config.stopSpeechToken, floor);

      const raw = await this.generate(textIds, voice, sampler, cap, shouldCancel, prefix);

      // `gen` is what the shipped engine calls a row: every token the model
      // committed to, with the stop marker itself excluded. Indices into it
      // are decode-step indices, which is what makes the observed peak
      // comparable against it, so the detectors run here, before the specials
      // are stripped and free to renumber anything.
      gen = raw.slice();
      const ended = gen.length > 0 && gen[gen.length - 1] === this.config.stopSpeechToken;
      if (ended) gen.pop();

      const [peakAt, peakProb] = sampler.eosPeak;
      hitTokenCap = !ended && gen.length >= cap;
      // The window, asked separately and asked here, where `gen` is still what
      // the model produced. `cap` is `min(maxNewTokens, ceilingFor(...))`, so
      // `hitTokenCap` cannot tell a filled window from a runaway short text; and the
      // trim below can cut a filled window down to a few tokens, which is how a
      // caller measuring the returned array saw room to spare.
      hitWindow = !ended && gen.length >= this.config.window.maxSpeechTokens;
      inspection = inspect(
        gen,
        {
          textTokenCount: textIds.length,
          minTokens: floor,
          eosPeakAt: peakAt,
          eosPeakProb: peakProb,
          ended,
          isTerminal,
          hitCeiling: hitTokenCap,
        },
        this.config.sampling.silenceTokenIds,
        pp
      );
      const condemned = inspection.reason === "dropout" || inspection.suspect;
      if (!condemned || pp.mode === "off") break;
      const silenceCount = gen.filter((t) => deadAir.has(t)).length;
      if (best === null || silenceCount < best.count) {
        // Strict `<`: on a tie the earlier attempt stands, so the ladder stays
        // a pure function of the caller's seed with no dependence on iteration
        // order.
        best = { count: silenceCount, gen, inspection, hitTokenCap, hitWindow };
      }
      if (attempt >= pp.retryMaxAttempts) {
        ({ gen, inspection, hitTokenCap, hitWindow } = best);
        break;
      }
    }
    if (pp.mode === "trim" && inspection.keep < gen.length) gen = gen.slice(0, inspection.keep);

    return {
      tokens: gen.filter((t) => t < this.config.startSpeechToken),
      inspection,
      hitTokenCap,
      hitWindow,
    };
  }

  /**
   * Render text that fits one model window, and refuse text that does not.
   *
   * {@link Engine.synthesize} is the call for any length; this one is for the
   * conformance harness and for a caller who wants the refusal.
   */
  async synthesizeWindow(
    text: string,
    voice: VoiceProfile,
    options: SynthesisOptions = {}
  ): Promise<{
    audio: Float32Array;
    tokens: number[];
    mel: Float32Array;
    sampleRate: number;
    inspection: Inspection;
    /**
     * True when generation stopped at the token cap rather than at a stop
     * token, so this reading is probably truncated. Truncation is not an
     * error, since the audio is real and merely incomplete, so it travels as a
     * field rather than a rejection.
     */
    hitTokenCap: boolean;
    /** The time-stretch this render was asked for; 1.0 means none was applied. */
    speed: number;
    /**
     * Where this render lands in `audio`: one entry, covering all of it, since
     * a single window is a single chunk. Its `words` are an estimate; see
     * `timing.ts` before building anything that depends on them.
     */
    chunks: ChunkTiming[];
  } & WavOutput> {
    const { seed = 0, language, shouldCancel } = options;
    const speed = options.speed ?? 1.0;
    validateSpeed(speed);
    const prefix = carryFrom(
      options.previousTokens,
      this.config.chunking.prefixTokens,
      this.config.startSpeechToken,
      this.config.decode
    );
    // The funnel is spelled out rather than left inside `encode`, because the
    // text the timings describe is the text that was tokenised: "I have 3
    // apples." is spoken, and therefore timed, as "I have three apples.".
    const lang = resolveLanguage(language, voice);
    const prepared = speechText(text, lang);
    const textIds = this.frontend.encode(prepared, lang);
    // A single window is the whole passage, so it is terminal.
    const { tokens, inspection, hitTokenCap, hitWindow } = await this.generateInspected(
      textIds,
      voice,
      seed,
      prefix,
      true,
      shouldCancel
    );
    // Refused rather than truncated. Shipping the window's worth of audio with
    // the rest of the text never spoken is silent data loss: what a caller
    // hears is fluent, complete-sounding and short. This package's README has
    // promised the error since before 0.1.1.
    //
    // The *window*, not the cap: `hitTokenCap` is also set by the postprocess length
    // ceiling, which stops a short text that ran away. That text fitted and
    // there is nothing to split.
    //
    // Read from the generation rather than from the returned array: postprocess
    // trims, so a window that filled and was then cut back measured short here
    // and the refusal never fired. The overflow was gone from the evidence, not
    // from the audio.
    if (hitWindow) {
      throw new WindowOverflowError(
        `the text did not fit one ${this.config.window.maxSpeechTokens}-token ` +
          "window and its tail was not spoken. Use synthesize, which splits " +
          "at sentence boundaries and joins the audio."
      );
    }
    // Discarded, not rendered: every port polls here, between the token
    // phase and the render.
    if (shouldCancel?.()) throw new CancelledError();
    const mel = await this.decodeMel(tokens, voice, deriveSeed(seed, 1));
    if (shouldCancel?.()) throw new CancelledError();
    const rendered = await this.vocode(mel, deriveSeed(seed, 2));
    // Last, and after the detectors above rather than before them: they judge
    // pacing by duration per token, and stretching first would move every number
    // they compare against. `speed = 1.0` returns the vocoder's array itself, so
    // the default costs nothing and changes no byte.
    const audio = fadeEdges(
      timeStretch(rendered, this.config.sampleRate, speed), this.config.sampleRate,
      this.config.edgeFadeSeconds ?? EDGE_FADE_SECONDS);
    if (shouldCancel?.()) throw new CancelledError();
    return withWav({
      audio,
      tokens,
      mel,
      sampleRate: this.config.sampleRate,
      inspection,
      hitTokenCap,
      // Recorded rather than left to be inferred, because it cannot be: a
      // stretched reading and a naturally faster one are the same numbers
      // afterwards, and the duration alone cannot say which this is.
      speed,
      // Measured on the stretched waveform, the one the caller is holding, so
      // there is no `1/speed` correction to apply anywhere.
      chunks: timeline(
        [{ text: prepared, samples: audio.length, tokens: tokens.length }],
        this.config.sampleRate
      ),
    });
  }

  /**
   * Speak `text` chunk by chunk, yielding each as it becomes ready.
   *
   * The same synthesis as {@link Engine.synthesize}, delivered as it is made:
   * time to first audio is set by the first chunk. Chunk 0 draws the caller's
   * seed and every later chunk `derive(seed, 16 + index)`, so a chunk's audio
   * does not depend on how many came before it; the last
   * `chunking.prefixTokens` tokens of each chunk condition the next, and
   * `options.previousTokens` seed that carry for the first.
   *
   * `options.shouldCancel` is polled on every decode step. When it returns
   * true the stream ends: the chunks already yielded are the partial, the one
   * in flight is discarded, and nothing is thrown, since the caller flipped
   * the flag. Validation happens on the first `next()`, which is when an
   * async generator runs.
   */
  async *stream(
    text: string,
    voice: VoiceProfile,
    options: SynthesisOptions = {}
  ): AsyncGenerator<StreamChunk> {
    try {
      yield* this.chunks(text, voice, options);
    } catch (err) {
      if (err instanceof CancelledError) return;
      throw err;
    }
  }

  /** {@link Engine.stream}, throwing {@link CancelledError} where the flag stopped it. */
  private async *chunks(
    text: string,
    voice: VoiceProfile,
    options: SynthesisOptions = {}
  ): AsyncGenerator<StreamChunk> {
    const { seed = 0, language, shouldCancel } = options;
    const speed = options.speed ?? 1.0;
    validateSpeed(speed);
    const lang = resolveLanguage(language, voice);
    // The funnel runs on the whole text BEFORE splitting: Polish respelling
    // changes the length ("download" -> "dałnloud"), so a budget computed
    // first would be a budget for text the engine never speaks.
    const prepared = speechText(text, lang);
    const chunks = splitText(prepared, this.config.chunking);
    if (chunks.length === 0) throw new NothingToSpeakError("nothing to speak");

    const prefixLen = this.config.chunking.prefixTokens;
    let carry: number[] = carryFrom(
      options.previousTokens,
      prefixLen,
      this.config.startSpeechToken,
      this.config.decode
    );

    // A work queue rather than a walk over `chunks`: a chunk the window could
    // not hold is replaced, in place, by its two halves. The decision needs a
    // generated window, because the overrun is a fact about this voice and
    // this text together that nothing before generation knows, so a queue is
    // the only
    // structure that lets one entry become two after the fact.
    //
    // Both halves keep the ORIGINAL chunk's index, so a repair cannot move the
    // seed of any later chunk. Seeds only: chunk k+1 is conditioned on the tail
    // of chunk k, which after a repair comes from the second half, so later
    // audio in THIS passage does move. What the index buys is that the change
    // stops at this passage.
    interface Part {
      text: string;
      index: number;
      seed: bigint;
      terminal: boolean;
      // False on a half, so a half that still overruns ships as it is. One did,
      // measured through the engine over the 51 passages carrying a cap hit, and
      // its audio ended on a 0.42 s tail: it finished its clause rather than
      // being cut. An unbounded split is a new way to fail.
      splittable: boolean;
    }
    const queue: Part[] = chunks.map((text, i) => ({
      text,
      index: i,
      // Every chunk after the first gets its own derived seed, so the same
      // passage sounds identical whether or not the caller stops early: a
      // chunk's audio does not depend on how many came before it. Chunk 0
      // draws the caller's seed itself, so a text that fits one window
      // renders the same here and through `synthesize`.
      seed: i === 0 ? normalizeSeed(seed) : deriveSeed(seed, CHUNK_STREAM_BASE + i),
      terminal: i === chunks.length - 1,
      splittable: true,
    }));

    for (let qi = 0; qi < queue.length; qi++) {
      const part = queue[qi];
      const index = part.index;
      if (shouldCancel?.()) throw new CancelledError();
      const chunkSeed = part.seed;
      const ids = this.frontend.encode(part.text, lang);
      // Only the last chunk ends the passage.
      const {
        tokens: chunkTokens,
        inspection,
        hitTokenCap,
        hitWindow: chunkFilledWindow,
      } = await this.generateInspected(
        ids,
        voice,
        chunkSeed,
        carry,
        part.terminal,
        shouldCancel
      );
      // The window has to be what stopped it, not the length-proportional
      // ceiling: generateInspected caps at min(maxNewTokens, ceilingFor(...)),
      // and the second fires when a short text runs away. Halving a runaway
      // gives two runaways with smaller ceilings each.
      // Measured before the trim, for the reason `synthesize` states.
      if (chunkFilledWindow && part.splittable &&
          this.config.chunking.capResplit === "word") {
        const halves = splitInHalf(part.text);
        if (halves !== null) {
          // Discard this window and do the two halves instead. The second draws
          // from a stream of its own off the chunk seed: the flow takes 1, the
          // vocoder 2, and the retry ladder 8 up.
          queue.splice(qi, 1, {
            text: halves[0],
            index,
            seed: chunkSeed,
            terminal: false,
            splittable: false,
          }, {
            text: halves[1],
            index,
            seed: deriveSeed(chunkSeed, RESPLIT_STREAM),
            terminal: part.terminal,
            splittable: false,
          });
          qi--;
          continue;
        }
      }
      // Discarded, not rendered: every port polls here, between the token
      // phase and the render.
      if (shouldCancel?.()) throw new CancelledError();
      const mel = await this.decodeMel(chunkTokens, voice, deriveSeed(chunkSeed, 1));
      const rendered = await this.vocode(mel, deriveSeed(chunkSeed, 2));
      // The stretch is the last stage, after the detectors have judged this
      // chunk's pacing against its text; see {@link Engine.synthesize}.
      const wav = fadeEdges(
        timeStretch(rendered, this.config.sampleRate, speed), this.config.sampleRate,
        this.config.edgeFadeSeconds ?? EDGE_FADE_SECONDS);
      if (shouldCancel?.()) throw new CancelledError();
      carry = carryFrom(chunkTokens, prefixLen, this.config.startSpeechToken, this.config.decode);
      yield {
        index,
        audio: wav,
        tokens: chunkTokens,
        mel,
        inspection,
        hitTokenCap,
        text: part.text,
        timing: timeline(
          [{ text: part.text, samples: wav.length, tokens: chunkTokens.length }],
          this.config.sampleRate
        )[0],
      };
    }
  }

  /**
   * Speak text of any length as one waveform.
   *
   * Exactly {@link Engine.stream} with the chunks concatenated, one loop, so
   * the two paths cannot drift. `hitTokenCap` is ORed across chunks: one truncated
   * chunk truncates the passage. Throws {@link CancelledError} when
   * `options.shouldCancel` returned true: a passage cut short is never
   * handed back as a result.
   */
  async synthesize(
    text: string,
    voice: VoiceProfile,
    options: SynthesisOptions = {}
  ): Promise<{
    audio: Float32Array;
    tokens: number[];
    mel: Float32Array;
    sampleRate: number;
    /** The time-stretch this render was asked for; 1.0 means none was applied. */
    speed: number;
    /**
     * True when any chunk stopped at the token cap rather than at a stop
     * token, so the passage is probably truncated: one truncated chunk
     * truncates the whole. Same field `synthesize` returns.
     */
    hitTokenCap: boolean;
    /**
     * Where every chunk lands in `audio`, in order and adjacent: chunk *k*'s
     * `end` is the same float as chunk *k+1*'s `start`, and the last `end` is
     * the whole duration.
     */
    chunks: ChunkTiming[];
  } & WavOutput> {
    const audio: Float32Array[] = [];
    const mels: Float32Array[] = [];
    const tokens: number[] = [];
    const spans: ChunkSpan[] = [];
    let hitTokenCap = false;

    for await (const chunk of this.chunks(text, voice, options)) {
      audio.push(chunk.audio);
      mels.push(chunk.mel);
      tokens.push(...chunk.tokens);
      hitTokenCap = hitTokenCap || chunk.hitTokenCap;
      spans.push({ text: chunk.text, samples: chunk.audio.length, tokens: chunk.tokens.length });
    }

    const total = audio.reduce((n, a) => n + a.length, 0);
    const joined = new Float32Array(total);
    let at = 0;
    for (const a of audio) {
      joined.set(a, at);
      at += a.length;
    }
    return withWav({
      audio: joined,
      tokens,
      mel: concatMelAlongTime(mels),
      sampleRate: this.config.sampleRate,
      speed: options.speed ?? 1.0,
      hitTokenCap,
      // Rebuilt from the parts rather than shifting each part's own timing by a
      // running float: `timeline` accumulates sample offsets as integers, so the
      // joins are exact and every chunk's `end` is the next one's `start` down
      // to the last bit.
      chunks: timeline(spans, this.config.sampleRate),
    });
  }

}


function erf(value: number): number {
  const x = Math.abs(value);
  if (x > 6) return Math.sign(value);
  let term = x;
  let sum = x;
  for (let n = 1; n < 256; n++) {
    term *= 2 * x * x / (2 * n + 1);
    const next = sum + term;
    if (next === sum) break;
    sum = next;
  }
  return Math.sign(value) * 2 / Math.sqrt(Math.PI) * Math.exp(-x * x) * sum;
}
