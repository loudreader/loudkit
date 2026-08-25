/**
 * Shared configuration types — a mirror of `loudkit.config`.
 *
 * Values here are the *algorithm*: identical on every backend. The JS engine
 * reads its windowing recipe, EOS floor, sampling law and token ids from
 * these, never from its own guess.
 */

import { CHARS_PER_TOKEN } from "./chunking.js";

import {
  POSTPROCESS_MODES,
  PRODUCTION_POSTPROCESS,
  type PostprocessConfig,
  type PostprocessMode,
} from "./postprocess.js";
import { TEXT_RECIPE, grammarDigest } from "./numbers.js";

export interface SamplingConfig {
  temperature: number;
  repetitionPenalty: number;
  minP: number;
  maxNewTokens: number;
  silenceTokenIds: number[];
  minTokensFloor: number;
  minTokensTextRatio: number;
}

export interface WindowConfig {
  maxSpeechTokens: number;
  staticLength: number | null;
  padTokenId: number | null;
  staticPromptTokens: number | null;
}

export interface ChunkConfig {
  enabled: boolean;
  maxTokens: number;
  prefixTokens: number;
  splitOn: string[];
}

export interface AlgorithmConfig {
  recipeVersion: string;
  guidance: "single_path" | "cfg_dual_path";
  guidanceRate: number;
  eulerSteps: number;
  eulerGrid: number[] | null;
  sampling: SamplingConfig;
  window: WindowConfig;
  chunking: ChunkConfig;
  /**
   * The artifact detectors. They remove tokens, so they change the audio
   * and are read from the manifest for the same reason the joins are: a
   * backend that re-guesses where a chunk ended cuts somewhere else, and
   * the difference is a hallucinated word that either does or does not
   * reach a listener.
   */
  postprocess: PostprocessConfig;
  /**
   * The funnel's identity — its code version and the digest of the grammar
   * file this port reads. In the fingerprint because the funnel decides what
   * string the model is handed, and therefore what it says.
   */
  text: TextConfig;
  sampleRate: number;
  tokenRateHz: number;
  speechVocabSize: number;
  startSpeechToken: number;
  stopSpeechToken: number;
}

/** The shipped production window recipe (ChatterboxMelSynthesizer.swift). */
export function productionWindow(): WindowConfig {
  return {
    maxSpeechTokens: 255,
    staticLength: 255,
    padTokenId: 4254,
    staticPromptTokens: 238,
  };
}

/** The shipped EOS floor. */
export const PRODUCTION_EOS_FLOOR = 10;
export const PRODUCTION_EOS_TEXT_RATIO = 1.2;

/**
 * Recipe version for a checkpoint manifest: `loudkit-1` when it carries none,
 * and nothing else accepted. One recipe means one value: a foreign tag
 * believed here would ride into every fingerprint this port reports, so it is
 * refused with the declared value named. A manifest that omits the key left a
 * shipping default unstated.
 */
function recipeVersionFromManifest(manifest: Record<string, unknown>): string {
  const raw = manifest.recipe_version;
  if (raw === undefined) return "loudkit-1";
  if (raw !== "loudkit-1") {
    throw new Error(
      `manifest declares recipe_version ${JSON.stringify(raw)}; ` +
        `the only recipe is "loudkit-1"`
    );
  }
  return raw;
}

/**
 * Read the postprocess block, or fall back to the shipping detectors.
 *
 * An unknown mode is refused rather than defaulted: it would trim where the
 * manifest said not to, under a matching `recipe_version`.
 */
function postprocessFromManifest(manifest: Record<string, unknown>): PostprocessConfig {
  const block = manifest.postprocess as Record<string, unknown> | undefined;
  const cfg: PostprocessConfig = { ...PRODUCTION_POSTPROCESS };
  if (block === undefined) return cfg;

  const mode = block.mode;
  if (mode !== undefined) {
    if (typeof mode !== "string" || !(POSTPROCESS_MODES as string[]).includes(mode)) {
      throw new Error(
        `manifest declares unknown postprocess mode ${JSON.stringify(mode)}; ` +
          `expected one of ${POSTPROCESS_MODES.join(", ")}`
      );
    }
    cfg.mode = mode as PostprocessMode;
  }
  // The manifest spells these in snake_case; this port holds them in camelCase.
  // Listed as pairs rather than derived, so a rename on either side is a
  // compile error here instead of a value silently keeping its default.
  const numeric: [keyof PostprocessConfig, string][] = [
    ["ceilingSpeechPerTextToken", "ceiling_speech_per_text_token"],
    ["ceilingSlackTokens", "ceiling_slack_tokens"],
    ["trailingFillerThreshold", "trailing_filler_threshold"],
    ["trailingSilenceRunTokens", "trailing_silence_run_tokens"],
    ["desperationBandRatio", "desperation_band_ratio"],
    ["desperationBandFloor", "desperation_band_floor"],
    ["fillerMinEosProbability", "filler_min_eos_probability"],
    ["fillerMaxSpeechAfterRun", "filler_max_speech_after_run"],
    ["desperationSpeechPerTextToken", "desperation_speech_per_text_token"],
    ["desperationMinTextTokens", "desperation_min_text_tokens"],
    ["endedTailSilenceRun", "ended_tail_silence_run"],
    ["endedTailBlipMax", "ended_tail_blip_max"],
    ["endedTailWordMax", "ended_tail_word_max"],
    ["endedTailKeep", "ended_tail_keep"],
    ["echoStrongEosProbability", "echo_strong_eos_probability"],
    ["echoStrongMaxTail", "echo_strong_max_tail"],
    ["echoStrongMinPositionPct", "echo_strong_min_position_pct"],
    ["echoWeakEosProbability", "echo_weak_eos_probability"],
    ["echoWeakMaxTail", "echo_weak_max_tail"],
    ["echoWeakMinPositionPct", "echo_weak_min_position_pct"],
    // The six this list was missing. Python reads its fields off the dataclass
    // precisely so a new constant cannot be left out of a hand-written list;
    // the four ports write the list by hand, and every one of them had drifted
    // the same six fields behind. Defaults matched, so nothing sounded wrong —
    // until a checkpoint sets one, at which point the manifest declares one
    // recipe and four engines run another.
    ["dropoutMinTokens", "dropout_min_tokens"],
    ["retryMaxAttempts", "retry_max_attempts"],
    ["pacingTolerance", "pacing_tolerance"],
    ["repetitionMaxPeriod", "repetition_max_period"],
    ["repetitionMinCycles", "repetition_min_cycles"],
    ["repetitionMinSpan", "repetition_min_span"],
  ];
  for (const [field, key] of numeric) {
    const raw = block[key];
    if (raw === undefined) continue;
    if (typeof raw !== "number" || !Number.isFinite(raw)) {
      throw new Error(
        `manifest['postprocess']['${key}'] must be a number, got ${JSON.stringify(raw)}`
      );
    }
    (cfg[field] as number) = raw;
  }
  return cfg;
}

const GUIDANCE_MODES = ["single_path", "cfg_dual_path"] as const;
type GuidanceMode = (typeof GUIDANCE_MODES)[number];

/**
 * Validate the manifest's guidance mode instead of asserting it.
 *
 * Python raises here ("a teacher checkpoint loading silently as single_path is
 * the founding defect with its arrow reversed") and Swift throws; a cast would
 * let an unknown or misspelled mode through as whatever string it was, which
 * this port then never reads — rendering single-path audio for a checkpoint
 * that asked for something else, with no complaint.
 *
 * `cfg_dual_path` is refused outright rather than silently downgraded: this
 * binding implements only the single path.
 */
function isGuidanceMode(value: unknown): value is GuidanceMode {
  return typeof value === "string" && (GUIDANCE_MODES as readonly string[]).includes(value);
}

function guidanceFromManifest(manifest: Record<string, unknown>): GuidanceMode {
  const raw = manifest.guidance ?? "single_path";
  if (!isGuidanceMode(raw)) {
    throw new Error(
      `manifest declares unknown guidance mode ${JSON.stringify(raw)}; ` +
        `expected one of ${GUIDANCE_MODES.join(", ")}`
    );
  }
  if (raw === "cfg_dual_path") {
    throw new Error(
      "manifest declares guidance mode cfg_dual_path, which this binding does not " +
        "implement — it would render single-path audio and silently disagree with " +
        "the Python engine"
    );
  }
  return raw;
}

/** The shipping chunking recipe: where the reader breathes. */
export const PRODUCTION_CHUNKING: ChunkConfig = {
  enabled: true,
  maxTokens: 255,
  prefixTokens: 6,
  splitOn: [". ", "! ", "? ", "; ", ", "],
};

/**
 * Read the chunking block instead of hard-coding the shipping recipe.
 *
 * A checkpoint can declare its own boundaries and prefix carry, and a runtime
 * that silently uses different ones agrees on `recipe_version` while disagreeing
 * on the reading — which is the drift the fingerprint exists to prevent.
 */
function chunkingFromManifest(manifest: Record<string, unknown>): ChunkConfig {
  const block = manifest.chunking as Record<string, unknown> | undefined;
  if (!block || typeof block !== "object") return { ...PRODUCTION_CHUNKING };
  const splitOn = Array.isArray(block.split_on)
    ? (block.split_on as unknown[]).filter((s): s is string => typeof s === "string")
    : [];
  return {
    enabled: typeof block.enabled === "boolean" ? block.enabled : PRODUCTION_CHUNKING.enabled,
    maxTokens:
      typeof block.max_tokens === "number" ? block.max_tokens : PRODUCTION_CHUNKING.maxTokens,
    prefixTokens:
      typeof block.prefix_tokens === "number"
        ? block.prefix_tokens
        : PRODUCTION_CHUNKING.prefixTokens,
    splitOn: splitOn.length > 0 ? splitOn : [...PRODUCTION_CHUNKING.splitOn],
  };
}

/** The explicit Euler time grid, or null for the cosine schedule. */
function eulerGridFromManifest(manifest: Record<string, unknown>): number[] | null {
  const raw = manifest.euler_grid;
  if (raw === undefined || raw === null) return null;
  if (!Array.isArray(raw)) {
    throw new Error(
      `manifest['euler_grid'] must be a list of floats or null, got ${typeof raw}`
    );
  }
  return raw.map(Number);
}

/**
 * Validate the chunking recipe, the way `loudkit.config.ChunkConfig` does.
 *
 * Python refuses four configurations here; a plain interface that
 * reads `max_tokens` straight from the manifest accepts all of them. The
 * second refusal is the one that matters: a `maxTokens` small
 * enough that `Math.floor(maxTokens * CHARS_PER_TOKEN)` is zero makes the
 * splitter cut nothing and loop forever.
 */
export function validateChunkConfig(c: ChunkConfig): void {
  if (c.maxTokens <= 0) {
    throw new Error(`chunking.max_tokens must be positive: ${c.maxTokens}`);
  }
  if (Math.floor(c.maxTokens * CHARS_PER_TOKEN) < 1) {
    throw new Error(
      `chunking.max_tokens=${c.maxTokens} leaves no character budget to split on ` +
        `(floor(${c.maxTokens} * ${CHARS_PER_TOKEN}) == 0); ` +
        `needs at least ${Math.ceil(1 / CHARS_PER_TOKEN)}`
    );
  }
  if (c.prefixTokens < 0 || c.prefixTokens >= c.maxTokens) {
    throw new Error(`chunking.prefix_tokens must be in [0, max_tokens): ${c.prefixTokens}`);
  }
  if (c.splitOn.length === 0) {
    throw new Error("chunking.split_on cannot be empty: there would be nowhere to break");
  }
}

/** Read the algorithm values out of a packed checkpoint's manifest. */
/**
 * Python refuses a manifest with a non-positive `sample_rate` and the other four
 * took it: every duration this engine reports is `samples / sample_rate`, so a
 * zero divides by zero and a negative reports negative seconds. A rate is the one
 * manifest field whose wrongness is not caught by any shape.
 */
function requirePositiveRate(value: number): number {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`sample_rate must be > 0: ${value}`);
  }
  return value;
}

/**
 * Python and Swift refuse a non-positive cap; the other three took it and
 * decoded nothing, which reaches a caller as silence they have to diagnose
 * rather than an error they can read. A cap of zero is not a configuration,
 * it is a typo in a manifest.
 */
function requirePositiveCap(value: number): number {
  if (value <= 0) throw new Error(`max_new_tokens must be positive: ${value}`);
  return value;
}

export function algorithmFromManifest(manifest: Record<string, unknown>): AlgorithmConfig {
  const samplingDefaults = (manifest.sampling_defaults ?? {}) as Record<string, number>;
  // `as number[]` on a JSON value is a claim, not a check: a manifest carrying
  // `"123"` reached `.includes()` as a string and matched substrings. Python
  // now refuses a string for this key by name; a manifest that one port
  // misreads while another defaults is the divergence class this library
  // exists to prevent.
  const silRaw = manifest.silence_token_ids ?? [];
  if (!Array.isArray(silRaw)) {
    throw new Error(
      `manifest key 'silence_token_ids' must be a list, got ${typeof silRaw} — ` +
        "a string is a sequence of characters, which is not what this field means"
    );
  }
  const sil = silRaw as number[];
  const speech = (manifest.speech_tokens ?? {}) as Record<string, number>;
  const window = (manifest.window ?? {}) as Record<string, unknown>;
  const eos = (manifest.eos_floor ?? {}) as Record<string, number>;

  const win: WindowConfig = {
    maxSpeechTokens: (window.max_speech_tokens as number) ?? 255,
    staticLength: window.static_length === null ? null : ((window.static_length as number) ?? 255),
    padTokenId: window.pad_token_id === null ? null : ((window.pad_token_id as number) ?? 4254),
    staticPromptTokens:
      window.static_prompt_tokens === null
        ? null
        : ((window.static_prompt_tokens as number) ?? 238),
  };

  const chunking = chunkingFromManifest(manifest);
  // Checked once, at the door, rather than per utterance. A chunking recipe
  // with no character budget makes `splitText` cut nothing and loop forever;
  // Python has refused it since d8742aa and this port reads the same key.
  validateChunkConfig(chunking);

  return {
    recipeVersion: recipeVersionFromManifest(manifest),
    guidance: guidanceFromManifest(manifest),
    guidanceRate: (manifest.guidance_rate as number) ?? 0.0,
    eulerSteps: (manifest.n_cfm_timesteps as number) ?? 2,
    // Read, not hard-coded to null. `timeGrid` honours `eulerGrid`, so this
    // port looked like the one that supported an explicit grid — while the
    // manifest parser threw it away before `timeGrid` ever saw it.
    // `config.py:296`: an explicit grid is preferred for anything that must
    // match across implementations, because "cosine" is a formula two
    // codebases can write two ways.
    eulerGrid: eulerGridFromManifest(manifest),
    sampling: (() => {
      // Range checks mirror Python's `SamplingConfig.__post_init__`: a
      // manifest the reference refuses must be refused here too, or two
      // implementations render different audio under one fingerprint.
      const temperature = samplingDefaults.temperature ?? 0.8;
      if (temperature <= 0 || temperature > 4)
        throw new Error(`temperature out of range: ${temperature}`);
      const repetitionPenalty = samplingDefaults.repetition_penalty ?? 1.2;
      if (repetitionPenalty < 1.0)
        throw new Error(`repetition_penalty out of range: ${repetitionPenalty}`);
      const minP = samplingDefaults.min_p ?? 0.05;
      if (minP < 0 || minP >= 1)
        throw new Error(`min_p out of range: ${minP}`);
      const minTokensFloor = eos.min_tokens_floor ?? PRODUCTION_EOS_FLOOR;
      if (minTokensFloor < 0)
        throw new Error(`min_tokens_floor must be >= 0: ${minTokensFloor}`);
      const minTokensTextRatio =
        eos.min_tokens_text_ratio ?? PRODUCTION_EOS_TEXT_RATIO;
      if (minTokensTextRatio < 0)
        throw new Error(
          `min_tokens_text_ratio must be >= 0: ${minTokensTextRatio}`,
        );
      return {
      temperature,
      repetitionPenalty,
      minP,
      maxNewTokens: requirePositiveCap(samplingDefaults.max_new_tokens ?? 255),
      silenceTokenIds: [...sil],
      minTokensFloor,
      minTokensTextRatio,
      };
    })(),
    window: win,
    chunking,
    postprocess: postprocessFromManifest(manifest),
    text: { recipe: TEXT_RECIPE, grammar: grammarDigest() },
    sampleRate: requirePositiveRate((manifest.sample_rate as number) ?? 24_000),
    // Read, not hard-coded — the same shape of bug `eulerGrid` had. It is
    // hashed into the fingerprint, so a checkpoint at a different token rate
    // must move the fingerprint rather than be silently overridden here.
    tokenRateHz: (manifest.token_rate_hz as number) ?? 25.0,
    speechVocabSize: (manifest.speech_vocab_size as number) ?? 8194,
    startSpeechToken: speech.start ?? 6561,
    stopSpeechToken: speech.stop ?? 6562,
  };
}

/** A loaded voice profile (mirror of `loudkit.voice.VoiceProfile`). */
export interface VoiceProfile {
  name: string;
  speakerEmbedding: Float32Array;
  flowEmbedding: Float32Array;
  promptTokens: BigInt64Array;
  promptMel: Float32Array;
  condPromptTokens: BigInt64Array;
  sourceSampleRate: number;
  language: string;
}


/**
 * Identifies the text funnel: what its code does, and what data it reads.
 *
 * The digest is of *this* package's copy of `numbers.json`, so a copy that has
 * drifted from the reference produces a different fingerprint and the engine
 * refuses to start rather than silently speaking something else.
 */
export interface TextConfig {
  /**
   * Bumped when the funnel's passes change what they emit for text they already
   * handled. A new language or table moves `grammar` on its own.
   */
  recipe: string;
  grammar: string;
}
