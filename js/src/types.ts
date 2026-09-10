/**
 * Shared configuration types: a mirror of `loudkit.config`.
 *
 * Values here are the *algorithm*: identical on every backend. The JS engine
 * reads its windowing recipe, EOS floor, sampling law and token ids from
 * these, never from its own guess.
 */

import { CHARS_PER_TOKEN } from "./chunking.js";

import {
  POSTPROCESS_MODES,
  PRODUCTION_POSTPROCESS,
  REPETITION_RESUME_VALUES,
  REPETITION_SILENCE_VALUES,
  RETRY_LADDER_HEADROOM,
  type PostprocessConfig,
  type PostprocessMode,
  type RepetitionResume,
  type RepetitionSilence,
  validatePostprocessRanges,
} from "./postprocess.js";
import { TEXT_RECIPE, grammarDigest } from "./numbers.js";
// The float spelling the canonical form hashes, so a value reads the same in a
// refusal as it does in the fingerprint. `fingerprint.js` imports this module
// for types only, so nothing is imported in a circle at run time.
import { reprFloat } from "./fingerprint.js";

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

/**
 * The two spellings of `ChunkConfig.midSentencePeriod`.
 *
 * A period that does not end a sentence is not a boundary. Under `"break"`,
 * `"But Mr. Smith went home"` splits after the title and hands the renderer a
 * seven-character chunk with its own derived seed and a token ceiling
 * proportional to seven characters. `"hold"` is the law; `"break"` stays
 * namable so a pack can say what it was measured under.
 */
export type MidSentencePeriod = "hold" | "break";

export const MID_SENTENCE_PERIODS: readonly MidSentencePeriod[] = ["hold", "break"];

/**
 * The two spellings of `ChunkConfig.capResplit`.
 *
 * `splitText` budgets characters against a constant, and a speaker slower than
 * it fills the window before the text runs out; the generator then stops at the
 * cap mid-word and the remainder is lost, because chunk texts are fixed before
 * any of them renders. `"word"` halves such a chunk and generates both halves
 * in its place. `"off"` ships the truncated window, which is what every
 * checkpoint built before this field did.
 */
export type CapResplit = "word" | "off";

export const CAP_RESPLITS: readonly CapResplit[] = ["word", "off"];

export interface ChunkConfig {
  enabled: boolean;
  maxTokens: number;
  prefixTokens: number;
  splitOn: string[];
  /**
   * Written forms whose following period does not end a sentence, given
   * without that period: the period is the separator's.
   *
   * One union list for every language. Surveyed over 1200 passages in ten, a
   * language-blind union re-chunks the corpus identically to ten per-language
   * lists. Data, not code: replacing the array is the whole of adding a
   * language. Not the funnel's list, which maps a written abbreviation to
   * spoken words and carries only the unambiguous ones; what reaches here is
   * the residue the funnel refuses to touch.
   */
  abbreviations: string[];
  midSentencePeriod: MidSentencePeriod;
  capResplit: CapResplit;
}

export interface AlgorithmConfig {
  decode?: "single" | "fusion_mtp2";
  /**
   * The raised-cosine ramp on both edges of every rendered window, in seconds.
   * Absent means 0.02. Only the historical 0.005 is omitted from the
   * canonical form, so the fingerprint moves only when the ramp does.
   */
  edgeFadeSeconds?: number;
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
   * The funnel's identity: its code version and the digest of the grammar
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

/**
 * The window the released manifest declares, spelled once.
 *
 * Not a parser default: `algorithmFromManifest` fills the absent keys the way
 * `manifest.py` does, with nulls, so a manifest that omits them fingerprints
 * identically in both. This is what the shipped one carries.
 */
export function productionWindow(): WindowConfig {
  return {
    maxSpeechTokens: 255,
    staticLength: 255,
    padTokenId: 4254,
    staticPromptTokens: 238,
  };
}

/** The EOS floor the released manifest declares, for the same reason. */
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
 * `block[key]` as a number, or `fallback` when the key is absent.
 *
 * `as number` on a JSON value is a claim, not a check: a manifest carrying
 * `"255"` for `max_speech_tokens` reached `new Array(pLen)` as a string and
 * `2 * "255"` as string concatenation, with no error naming the key. Python's
 * `manifest._number` refuses the same values by name, and a manifest one port
 * reads while another refuses is the divergence class this library exists to
 * prevent.
 *
 * `where` names the enclosing block in the message; the top level passes none.
 */
function numberAt(
  block: Record<string, unknown>,
  key: string,
  fallback: number,
  where = "manifest"
): number {
  const value = block[key];
  if (value === undefined) return fallback;
  if (typeof value !== "number") {
    throw new Error(`${where}['${key}'] should be a number, got ${JSON.stringify(value)}`);
  }
  return value;
}

/**
 * The same check where absent and `null` both mean "no value": the window's
 * three optional keys, which Python reads as `None` rather than filling in.
 */
function optionalNumberAt(
  block: Record<string, unknown>,
  key: string,
  where: string
): number | null {
  const value = block[key];
  if (value === undefined || value === null) return null;
  if (typeof value !== "number") {
    throw new Error(`${where}['${key}'] should be a number or null, got ${JSON.stringify(value)}`);
  }
  return value;
}

/**
 * A manifest number read as the whole number a count field holds.
 *
 * The reference reads every count through `int()`. Read as a plain JSON number
 * instead, `n_cfm_timesteps: 2.7` fingerprints differently from the same
 * manifest read by the reference and breaks the schedule with it: `timeGrid`
 * answers `[0, 0.1645, 0.6039]` where two steps answer `[0, 0.2929, 1.0]`, so
 * the flow ODE stops at t~0.60 and the audio is rendered from a half-integrated
 * state. No error, plausible audio, wrong.
 */
function intAt(
  block: Record<string, unknown>,
  key: string,
  fallback: number,
  where = "manifest"
): number {
  return wholeNumber(numberAt(block, key, fallback, where), `${where}['${key}']`);
}

/** {@link optionalNumberAt} held as the whole number {@link intAt} holds. */
function optionalIntAt(
  block: Record<string, unknown>,
  key: string,
  where: string
): number | null {
  const value = optionalNumberAt(block, key, where);
  return value === null ? null : wholeNumber(value, `${where}['${key}']`);
}

/**
 * A count as the reference reads it: whole, and exact in this runtime.
 *
 * Whole first. A key that counts things takes a whole number, and 2.7 in one
 * is a value a packer computed wrong, so truncating it to 2 hides that
 * arithmetic behind a chunker that breathes in a different place, under a
 * `recipe_version` saying the five implementations agree. `manifest._int`
 * refuses it in this sentence. The rate, threshold and probability fields go
 * through {@link numberAt} and keep their fractions, because a temperature of
 * 0.75 is a value.
 *
 * Then exact. The reference's `int()` is arbitrary precision, so
 * `n_cfm_timesteps: 1e30` is a definite 31-digit integer there. A JS number is
 * a double: it holds `1e30` as an approximation, spells it `1e+30`, and cannot
 * represent the integer next to it at all. Every whole number up to 2**53 - 1
 * is exact in both, and past that the two languages are no longer reading the
 * same value.
 *
 * So this port refuses rather than continuing on the approximation. The
 * alternative is to saturate at some ceiling, which decides silently that a
 * manifest meant something it did not say: the resulting count is then hashed
 * into the fingerprint, and a fingerprint that agrees while the two engines
 * hold different numbers is the one failure this library exists to prevent.
 * A manifest wanting a count this large is malformed in any case; no bound in
 * the algorithm is within nine quadrillion of it. Go and Rust cast instead, and
 * land on `MaxInt64` and `usize::MAX`.
 *
 * `path` is the whole key as a refusal quotes it back, so a census element says
 * `[0]` rather than `['0']`, which is how `manifest._where` spells it.
 */
function wholeNumber(value: number, path: string): number {
  if (!Number.isInteger(value)) {
    throw new Error(`${path} must be a whole number, got ${reprFloat(value)}`);
  }
  if (!Number.isSafeInteger(value)) {
    throw new Error(
      `${path} must be a whole number this runtime holds exactly, got ` +
        `${reprFloat(value)}: past ${Number.MAX_SAFE_INTEGER} a JS number is an ` +
        "approximation, so every bound checked against it and the canonical form " +
        "it is hashed into would be about a different number than the reference reads"
    );
  }
  return value;
}

/**
 * `manifest[key]` checked to be an object, or `{}` when the key is absent.
 *
 * A check, not a cast: only an absent key takes the shipping defaults. An
 * explicit `null` is a declared value rather than an absence, so
 * `"chunking": null`, `"sampling_defaults": null` and `"eos_floor": null` are
 * refused here as the reference refuses the file; and a list or a scalar is
 * refused rather than read as a block, so `"window": [1, 2, 3]` cannot quietly
 * render the ragged window, which `go/config/config.go`,
 * `rust/src/checkpoint.rs` and `swift/LoudKit/ManifestReader.swift` each say
 * in their own words must never happen. A block named in a manifest and read
 * as something else is a law the file declared and this port would not run,
 * and the canonical form would record the default it fell back to, so the
 * fingerprint would agree with the misreading instead of reporting it.
 *
 * `window` and `decode` read `null` as a value rather than an absence and so
 * call {@link objectOrNull} instead.
 */
function blockAt(manifest: Record<string, unknown>, key: string): Record<string, unknown> {
  const value = manifest[key];
  if (value === undefined) return {};
  const block = asObject(value);
  if (block === null) {
    throw new Error(`manifest key '${key}' must be an object, got ${describe(value)}`);
  }
  return block;
}

/**
 * The same check where `null` is one of the readings: `window: null` is the
 * ragged window said out loud (`manifest.py:_window_from`) and `decode: null`
 * is the single loop said out loud (`manifest.py:decode_from`). Anything else
 * that is not an object is still refused.
 */
function objectOrNull(
  manifest: Record<string, unknown>,
  key: string,
  nullMeans: string
): Record<string, unknown> | null {
  const value = manifest[key];
  if (value === undefined || value === null) return null;
  const block = asObject(value);
  if (block === null) {
    throw new Error(
      `manifest['${key}'] must be an object or null (${nullMeans}), got ${describe(value)}`
    );
  }
  return block;
}

/** A JSON object, or null for every other JSON value. Arrays are not objects. */
function asObject(value: unknown): Record<string, unknown> | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return null;
  return value as Record<string, unknown>;
}

/** What a refusal calls the value it was handed. */
function describe(value: unknown): string {
  if (value === null) return "null";
  if (Array.isArray(value)) return "a list";
  return typeof value;
}

/**
 * A token-id census: the id lists that are properties of the weights.
 *
 * Absent is the empty census; `null` is not, because the reference refuses a
 * key it cannot read rather than defaulting it. A string is refused by name,
 * being a sequence of characters, which is not what an id list means. Each
 * entry is held the way every other count in this reader is held.
 */
function idsAt(manifest: Record<string, unknown>, key: string): number[] {
  const raw = manifest[key];
  if (raw === undefined) return [];
  if (!Array.isArray(raw)) {
    throw new Error(
      `manifest key '${key}' must be a list, got ${describe(raw)}: ` +
        "a string is a sequence of characters, which is not what this field means"
    );
  }
  return raw.map((id, i) => {
    if (typeof id !== "number") {
      throw new Error(
        `manifest['${key}'][${i}] must be a number, got ${JSON.stringify(id)}`
      );
    }
    return wholeNumber(id, `manifest['${key}'][${i}]`);
  });
}

/**
 * A JSON boolean under `block`, or the fallback when the key is absent.
 *
 * Not a truthiness test. JavaScript calls 0 false and 1 true and JSON does
 * not, so a manifest carrying `"enabled": 0` was written by a tool that meant
 * `false` and emitted a number. Reading it as false hides that mistake behind
 * audio joined differently rather than audio missing, and this is the one key
 * whose misreading changes every join in a passage at once. All five ports
 * refuse it in this sentence.
 */
function flagAt(block: Record<string, unknown>, key: string, fallback: boolean, where: string): boolean {
  if (!(key in block)) return fallback;
  const raw = block[key];
  if (typeof raw !== "boolean") {
    throw new Error(`${where}['${key}'] must be JSON true or false, got ${JSON.stringify(raw)}`);
  }
  return raw;
}

/**
 * A list of strings, or the fallback when the key is absent.
 *
 * Presence, not length, and not `Array.isArray` either. A fallback keyed to
 * anything but absence makes three refusals unreachable from a manifest:
 * `split_on: []`, which `validateChunkConfig` refuses because there would be
 * nowhere to break, and a `split_on` or `abbreviations` given as a bare
 * string, which the reference refuses by name because a string is a sequence
 * of characters.
 */
function stringListAt(
  block: Record<string, unknown>,
  key: string,
  fallback: readonly string[],
  where: string
): string[] {
  const raw = block[key];
  if (raw === undefined) return [...fallback];
  if (typeof raw === "string") {
    throw new Error(`${where}['${key}'] must be a list of strings, got a string`);
  }
  if (!Array.isArray(raw)) {
    throw new Error(`${where}['${key}'] must be a list of strings, got ${describe(raw)}`);
  }
  return raw.map((s, i) => {
    if (typeof s !== "string") {
      throw new Error(`${where}['${key}'][${i}] must be a string, got ${JSON.stringify(s)}`);
    }
    return s;
  });
}

/**
 * Read the postprocess block, or fall back to the shipping detectors.
 *
 * An unknown mode is refused rather than defaulted: it would trim where the
 * manifest said not to, under a matching `recipe_version`.
 */
function postprocessFromManifest(manifest: Record<string, unknown>): PostprocessConfig {
  // `manifest.postprocess as Record | undefined` read `null` as a block and
  // then indexed it, so `"postprocess": null` threw a raw TypeError naming
  // nothing instead of the by-name refusal the other four readers give.
  const block = blockAt(manifest, "postprocess");
  const cfg: PostprocessConfig = { ...PRODUCTION_POSTPROCESS };
  // The render censuses: which ids actually render as digital silence
  // (`silence_render_ids`) and which as contextually-quiet breath/decay
  // (`quiet_render_ids`). Optional, because a checkpoint packed before the census has
  // neither, and the stall detector then falls back to `silence_token_ids`,
  // degraded but safe. Read from the manifest top level, before the block
  // check, because they are properties of the weights, like
  // `silence_token_ids`, not detector constants someone tuned; a manifest
  // with no postprocess block still carries them.
  cfg.silenceRenderIds = idsAt(manifest, "silence_render_ids");
  cfg.quietRenderIds = idsAt(manifest, "quiet_render_ids");

  // The censuses live at the manifest top level, where they are read above.
  // Accepting them here too would give one value two homes in one file;
  // Python's block reader refuses them the same way.
  for (const key of ["silence_render_ids", "quiet_render_ids"]) {
    if (block[key] !== undefined) {
      throw new Error(
        `manifest['postprocess']['${key}'] belongs at the manifest top level, ` +
          "beside 'silence_token_ids'"
      );
    }
  }

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
  // A string field like mode, and refused like mode: a law this port does
  // not implement must not fall back to a default, or the resolver would cut
  // where the manifest said to condemn.
  const repetitionResume = block.repetition_resume;
  if (repetitionResume !== undefined) {
    if (
      typeof repetitionResume !== "string" ||
      !(REPETITION_RESUME_VALUES as string[]).includes(repetitionResume)
    ) {
      throw new Error(
        `manifest declares unknown repetition_resume ${JSON.stringify(repetitionResume)}; ` +
          `expected one of ${REPETITION_RESUME_VALUES.join(", ")}`
      );
    }
    cfg.repetitionResume = repetitionResume as RepetitionResume;
  }
  // A string field like mode, and refused like mode: a family this port does
  // not implement must not fall back to a default, or the loop exemption would
  // read one silence list under a manifest declaring another.
  const repetitionSilence = block.repetition_silence;
  if (repetitionSilence !== undefined) {
    if (
      typeof repetitionSilence !== "string" ||
      !(REPETITION_SILENCE_VALUES as string[]).includes(repetitionSilence)
    ) {
      throw new Error(
        `manifest declares unknown repetition_silence ${JSON.stringify(repetitionSilence)}; ` +
          `expected one of ${REPETITION_SILENCE_VALUES.join(", ")}`
      );
    }
    cfg.repetitionSilence = repetitionSilence as RepetitionSilence;
  }
  // The manifest spells these in snake_case; this port holds them in camelCase.
  // Listed as triples rather than derived, so a rename on either side is a
  // compile error here instead of a value silently keeping its default. The
  // third member is the field's type on `PostprocessConfig` in the reference,
  // which reads it off the dataclass: an `int` field refuses a fractional
  // number rather than truncating it, because a threshold written 2.7 is a
  // typo and the reference says so by name.
  const numeric: [keyof PostprocessConfig, string, "int" | "float"][] = [
    ["ceilingSpeechPerTextToken", "ceiling_speech_per_text_token", "float"],
    ["ceilingSlackTokens", "ceiling_slack_tokens", "int"],
    ["trailingFillerThreshold", "trailing_filler_threshold", "float"],
    ["trailingSilenceRunTokens", "trailing_silence_run_tokens", "int"],
    ["desperationBandRatio", "desperation_band_ratio", "float"],
    ["desperationBandFloor", "desperation_band_floor", "int"],
    ["fillerMinEosProbability", "filler_min_eos_probability", "float"],
    ["fillerMaxSpeechAfterRun", "filler_max_speech_after_run", "int"],
    ["desperationSpeechPerTextToken", "desperation_speech_per_text_token", "float"],
    ["desperationMinTextTokens", "desperation_min_text_tokens", "int"],
    ["desperationMinKeepPerTextToken", "desperation_min_keep_per_text_token", "float"],
    ["endedTailSilenceRun", "ended_tail_silence_run", "int"],
    ["endedTailBlipMax", "ended_tail_blip_max", "int"],
    ["endedTailWordMax", "ended_tail_word_max", "int"],
    ["endedTailKeep", "ended_tail_keep", "int"],
    ["echoStrongEosProbability", "echo_strong_eos_probability", "float"],
    ["echoStrongMaxTail", "echo_strong_max_tail", "int"],
    ["echoStrongMinPositionPct", "echo_strong_min_position_pct", "int"],
    ["echoWeakEosProbability", "echo_weak_eos_probability", "float"],
    ["echoWeakMaxTail", "echo_weak_max_tail", "int"],
    ["echoWeakMinPositionPct", "echo_weak_min_position_pct", "int"],
    // Every postprocess parameter, because a hand-written list that misses
    // one is a manifest key this port does not read. Python takes its fields
    // off the dataclass precisely so a new constant cannot be left out; the
    // four ports write the list by hand, so the list has to be complete.
    // Defaults matching hides the gap until a checkpoint sets one of them, at
    // which point the manifest declares one recipe and the engine runs
    // another.
    ["dropoutMinTokens", "dropout_min_tokens", "int"],
    ["retryMaxAttempts", "retry_max_attempts", "int"],
    ["pacingTolerance", "pacing_tolerance", "float"],
    ["repetitionMaxPeriod", "repetition_max_period", "int"],
    ["repetitionMinCycles", "repetition_min_cycles", "int"],
    ["repetitionMinSpan", "repetition_min_span", "int"],
    ["stallRunTokens", "stall_run_tokens", "int"],
  ];
  for (const [field, key, kind] of numeric) {
    const raw = block[key];
    if (raw === undefined) continue;
    if (typeof raw !== "number" || !Number.isFinite(raw)) {
      throw new Error(
        `manifest['postprocess']['${key}'] must be a number, got ${JSON.stringify(raw)}`
      );
    }
    // Through the same door the counts outside this block go through, so a
    // detector count answers to one rule rather than to a second copy of half
    // of it, which would check the fraction and not the exactness.
    (cfg[field] as number) =
      kind === "int" ? wholeNumber(raw, `manifest['postprocess']['${key}']`) : raw;
  }
  // The two range checks `PostprocessConfig._validate_ranges` carries. Finite
  // is not enough: a value the reference refuses and this port accepts is one
  // manifest rendering two ways under one fingerprint.
  if (cfg.retryMaxAttempts < 0 || cfg.retryMaxAttempts >= RETRY_LADDER_HEADROOM) {
    throw new Error(
      `retry_max_attempts must be in [0, ${RETRY_LADDER_HEADROOM}): ` +
        `${cfg.retryMaxAttempts}. Above that the ladder's derived seeds run ` +
        "into the streams the chunk seeds use."
    );
  }
  if (cfg.repetitionMinCycles < 2) {
    // One cycle is not a repetition and two is the definition of one; a
    // threshold below two would cut every row that says a word twice.
    throw new Error(`repetition_min_cycles must be at least 2: ${cfg.repetitionMinCycles}`);
  }
  // The reference validates its ranges in `__post_init__`, so a manifest
  // naming an impossible constant never becomes a config there. This port read
  // every number and asked nothing of it: `ceiling_slack_tokens: -1` loaded and
  // widened the ceiling, `stall_run_tokens: 0` loaded and called every row with
  // one leading silence token a stall. Checked after the whole block is read
  // because three of the rules compare two fields.
  validatePostprocessRanges(cfg);
  return cfg;
}

const GUIDANCE_MODES = ["single_path", "cfg_dual_path"] as const;
type GuidanceMode = (typeof GUIDANCE_MODES)[number];

/**
 * Validate the manifest's guidance mode instead of asserting it.
 *
 * Python raises here and Swift throws. A cast would let an unknown or
 * misspelled mode through as whatever string it was, which this port then
 * never reads, rendering single-path audio for a checkpoint that asked for
 * something else, with no complaint.
 *
 * `cfg_dual_path` is refused outright rather than silently downgraded: this
 * binding implements only the single path.
 */
function isGuidanceMode(value: unknown): value is GuidanceMode {
  return typeof value === "string" && (GUIDANCE_MODES as readonly string[]).includes(value);
}

function guidanceFromManifest(manifest: Record<string, unknown>): GuidanceMode {
  // Presence, not `??`: an explicit `null` is a declared value this port
  // cannot read, and reading it as an absent key would render single-path
  // audio for a manifest that named something else. The reference renders it
  // (`str(None)`) and refuses what it rendered.
  const raw = "guidance" in manifest ? manifest.guidance : "single_path";
  if (!isGuidanceMode(raw)) {
    throw new Error(
      `manifest declares unknown guidance mode ${JSON.stringify(raw)}; ` +
        `expected one of ${GUIDANCE_MODES.join(", ")}`
    );
  }
  if (raw === "cfg_dual_path") {
    throw new Error(
      "manifest declares guidance mode cfg_dual_path, which this binding does not " +
        "implement: it would render single-path audio and silently disagree with " +
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
  // The surveyed union, sorted. It must equal
  // `loudkit.config.ChunkConfig.abbreviations`.
  // prettier-ignore
  abbreviations: [
    "A", "B", "Cpn", "D", "Dr", "F", "H", "Hr", "I", "J", "K", "M", "Mr",
    "Mrs", "R", "S", "St", "T", "V", "Vors", "dr", "mrs", "prof", "\u015bw",
  ],
  midSentencePeriod: "hold",
  capResplit: "word",
};

/**
 * Read the chunking block instead of hard-coding the shipping recipe.
 *
 * A checkpoint can declare its own boundaries and prefix carry, and a runtime
 * that silently uses different ones agrees on `recipe_version` while disagreeing
 * on the reading, which is the drift the fingerprint exists to prevent.
 */
function chunkingFromManifest(manifest: Record<string, unknown>): ChunkConfig {
  const where = "manifest['chunking']";
  const block = blockAt(manifest, "chunking");
  // Refused by name rather than ignored. `ChunkConfig` here has no such field,
  // so a manifest that sets it would split the first chunk one way in Python
  // and another way here, under one fingerprint.
  if ("first_chunk_max_tokens" in block) {
    throw new Error(
      "manifest['chunking']['first_chunk_max_tokens'] is not implemented by this port"
    );
  }
  return {
    enabled: flagAt(block, "enabled", PRODUCTION_CHUNKING.enabled, where),
    maxTokens: intAt(block, "max_tokens", PRODUCTION_CHUNKING.maxTokens, where),
    prefixTokens: intAt(block, "prefix_tokens", PRODUCTION_CHUNKING.prefixTokens, where),
    // Presence, not length, for both lists. An empty `split_on` is refused by
    // `validateChunkConfig`, and it could never reach it while an empty list
    // fell back to the shipping five; an empty `abbreviations` is meaningful,
    // being the old law spelled as data, and must not fall back at all.
    splitOn: stringListAt(block, "split_on", PRODUCTION_CHUNKING.splitOn, where),
    abbreviations: stringListAt(
      block,
      "abbreviations",
      PRODUCTION_CHUNKING.abbreviations,
      where
    ),
    // Type-checked rather than rendered, as `go/config.stringKey` and
    // `rust/src/checkpoint.rs`'s text reader are. Rendering a present value of
    // the wrong type is safe in a language where rendering a list gives
    // something no closed set contains; in JavaScript `String(["word"])` is
    // "word", so a `cap_resplit` given as a one-element list passed the
    // membership check and the recipe ran under a spelling the manifest never
    // wrote. The reference refuses that manifest too, in its own sentence.
    midSentencePeriod: stringAt(
      block,
      "mid_sentence_period",
      PRODUCTION_CHUNKING.midSentencePeriod,
      where
    ) as MidSentencePeriod,
    capResplit: stringAt(
      block,
      "cap_resplit",
      PRODUCTION_CHUNKING.capResplit,
      where
    ) as CapResplit,
  };
}

/**
 * A string under `block`, or the fallback when the key is absent.
 *
 * Presence, then type. The value it returns is still checked against the
 * closed set it belongs to; this only refuses the values no rendering of which
 * means anything.
 */
function stringAt(
  block: Record<string, unknown>,
  key: string,
  fallback: string,
  where: string
): string {
  if (!(key in block)) return fallback;
  const raw = block[key];
  if (typeof raw !== "string") {
    throw new Error(`${where}['${key}'] must be a string, got ${JSON.stringify(raw)}`);
  }
  return raw;
}

/**
 * The explicit Euler time grid, or null for the cosine schedule.
 *
 * A point with no numeric reading is refused rather than coerced, which is
 * what `go/config.eulerGrid` and `rust/src/checkpoint.rs:euler_grid_from` do
 * and the sentence they say. `Number` reads `null` as 0 and `[]` as 0, so a
 * grid with a hole in it would integrate the ODE on a schedule that starts
 * where the manifest does not, with the canonical form recording that
 * schedule.
 */
function eulerGridFromManifest(manifest: Record<string, unknown>): number[] | null {
  const raw = manifest.euler_grid;
  if (raw === undefined || raw === null) return null;
  if (!Array.isArray(raw)) {
    throw new Error(
      `manifest['euler_grid'] must be a list of floats or null, got ${typeof raw}`
    );
  }
  return raw.map((point, i) => {
    if (typeof point !== "number") {
      throw new Error(
        `manifest['euler_grid'][${i}] should be a number, got ${JSON.stringify(point)}`
      );
    }
    return point;
  });
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
        `(int(${c.maxTokens} * ${CHARS_PER_TOKEN}) == 0); ` +
        `needs at least ${Math.ceil(1 / CHARS_PER_TOKEN)}`
    );
  }
  if (c.prefixTokens < 0 || c.prefixTokens >= c.maxTokens) {
    throw new Error(`chunking.prefix_tokens must be in [0, max_tokens): ${c.prefixTokens}`);
  }
  if (c.splitOn.length === 0) {
    throw new Error("chunking.split_on cannot be empty: there would be nowhere to break");
  }
  if (!CAP_RESPLITS.includes(c.capResplit)) {
    throw new Error(
      `unknown cap_resplit ${JSON.stringify(c.capResplit)}: ` +
        `expected ${CAP_RESPLITS.map((s) => JSON.stringify(s)).join(" or ")}`,
    );
  }
  if (!MID_SENTENCE_PERIODS.includes(c.midSentencePeriod)) {
    throw new Error(
      `unknown mid_sentence_period ${JSON.stringify(c.midSentencePeriod)}: ` +
        `expected ${MID_SENTENCE_PERIODS.map((s) => JSON.stringify(s)).join(" or ")}`
    );
  }
  // An empty entry is a suffix of everything, so it would hold every candidate
  // and drive every split down to a word boundary. Silent, and audible on
  // every long passage.
  if (c.abbreviations.some((a) => a.length === 0)) {
    throw new Error("chunking.abbreviations cannot contain an empty string");
  }
}

/**
 * Validate the framing recipe, with the sentences of
 * `loudkit.config.WindowConfig.__post_init__`.
 *
 * A static query buffer shorter than the window it frames does not fail: it
 * truncates. A manifest declaring `max_speech_tokens` 300 with `static_length`
 * 255 drops 45 speech tokens, 1.8 seconds of the passage, and returns audio
 * that sounds finished to everyone who does not know the text.
 */
export function validateWindowConfig(w: WindowConfig): void {
  if (w.staticLength !== null && w.staticLength < w.maxSpeechTokens) {
    throw new Error(
      `static_length ${w.staticLength} cannot be shorter than ` +
        `max_speech_tokens ${w.maxSpeechTokens}`
    );
  }
  if (w.staticPromptTokens !== null && w.staticPromptTokens <= 0) {
    throw new Error(`static_prompt_tokens must be positive: ${w.staticPromptTokens}`);
  }
}

/**
 * The rules that need more than one field: `AlgorithmConfig.__post_init__`, in
 * the reference's order and with its sentences.
 *
 * Without them a manifest the reference refuses loads here and computes a
 * fingerprint for an algorithm that cannot run: `n_cfm_timesteps: 0` makes the
 * flow loop run zero times and renders the prior noise as audio,
 * `token_rate_hz: 0` divides every duration by zero, `speech_vocab_size: 0`
 * leaves no vocabulary, a start token past the embedding table indexes off the
 * end, a start equal to the stop never stops, and a grid of the wrong length
 * or the wrong direction integrates the ODE on a schedule that is not one.
 *
 * Called last, after every key has been read, and after the blocks have
 * validated themselves, so a manifest with two faults names the same one here
 * as in `go/config.AlgorithmConfig.Validate` and `rust/src/engine.rs`.
 */
export function validateAlgorithmConfig(cfg: AlgorithmConfig): void {
  if (cfg.guidance === "single_path" && cfg.guidanceRate !== 0) {
    throw new Error("guidance_rate must be 0.0 in single_path mode");
  }
  // The `cfg_dual_path` half of the reference's pair has no call site here:
  // `guidanceFromManifest` refuses that mode outright, because this binding
  // would render single-path audio under a manifest that says otherwise.
  if (cfg.eulerSteps < 1) {
    throw new Error(`euler_steps must be >= 1: ${cfg.eulerSteps}`);
  }
  validateNumericCore(cfg);
  validateEulerGrid(cfg);
  // The three token budgets have to agree, or a chunk overruns the render
  // window mid-stream, after earlier chunks have already played.
  const window = cfg.window.maxSpeechTokens;
  if (cfg.chunking.enabled && cfg.chunking.maxTokens > window) {
    throw new Error(
      `chunking.max_tokens ${cfg.chunking.maxTokens} exceeds the render window ` +
        `(${window}): every chunk would be sized past what the renderer accepts, ` +
        "and the refusal would land mid-stream, after audio had already been delivered"
    );
  }
  if (cfg.sampling.maxNewTokens > window) {
    throw new Error(
      `sampling.max_new_tokens ${cfg.sampling.maxNewTokens} exceeds the render ` +
        `window (${window}): generation is allowed to produce more speech than the ` +
        "renderer will accept, so a long utterance fails after it has been " +
        "generated rather than before"
    );
  }
}

/**
 * `AlgorithmConfig._validate_numeric_core`: the values every duration, every
 * index and every ramp is computed from.
 *
 * The reference exempts the one value its reader turns into the unset
 * sentinel, today's 20 ms, and that value is inside the ramp's range: so the
 * rule is the range, on whatever the ramp actually is. An absent key is the
 * legacy 5 ms, which is inside it too.
 */
function validateNumericCore(cfg: AlgorithmConfig): void {
  if (
    cfg.edgeFadeSeconds !== undefined &&
    !(cfg.edgeFadeSeconds >= 0.001 && cfg.edgeFadeSeconds <= 0.05)
  ) {
    throw new Error(
      `edge_fade_seconds must be in [0.001, 0.05]: ${reprFloat(cfg.edgeFadeSeconds)}`
    );
  }
  // Every duration this engine reports is `samples / sample_rate`, so a zero
  // divides by zero and a negative reports negative seconds. A rate is the one
  // manifest field whose wrongness is not caught by any shape.
  if (cfg.sampleRate <= 0) {
    throw new Error(`sample_rate must be > 0: ${cfg.sampleRate}`);
  }
  if (cfg.tokenRateHz <= 0) {
    throw new Error(`token_rate_hz must be > 0: ${reprFloat(cfg.tokenRateHz)}`);
  }
  if (cfg.speechVocabSize < 1) {
    throw new Error(`speech_vocab_size must be >= 1: ${cfg.speechVocabSize}`);
  }
  for (const [name, value] of [
    ["start_speech_token", cfg.startSpeechToken],
    ["stop_speech_token", cfg.stopSpeechToken],
  ] as const) {
    if (value < 0 || value >= cfg.speechVocabSize) {
      throw new Error(`${name} must be in [0, ${cfg.speechVocabSize}): ${value}`);
    }
  }
  if (cfg.startSpeechToken === cfg.stopSpeechToken) {
    throw new Error(
      "start_speech_token and stop_speech_token must differ: both are " +
        `${cfg.startSpeechToken}`
    );
  }
}

/**
 * The explicit time grid against the step count it has to schedule.
 *
 * A grid of the wrong length integrates a different number of steps than the
 * manifest declared; one that does not run from 0 to 1 stops the flow ODE
 * somewhere the checkpoint was never trained for.
 */
function validateEulerGrid(cfg: AlgorithmConfig): void {
  const grid = cfg.eulerGrid;
  if (grid === null) return;
  const want = cfg.eulerSteps + 1;
  if (grid.length !== want) {
    throw new Error(`euler_grid has ${grid.length} points, expected ${want}`);
  }
  for (let i = 1; i < grid.length; i++) {
    if (!(grid[i] > grid[i - 1])) {
      throw new Error("euler_grid must be strictly increasing");
    }
  }
  // `grid` is non-empty here: a zero-length grid cannot have `eulerSteps + 1`
  // points, and `eulerSteps` is at least one.
  if (Math.abs(grid[0]) > 1e-6 || Math.abs(grid[grid.length - 1] - 1.0) > 1e-6) {
    throw new Error("euler_grid must run from 0.0 to 1.0");
  }
}

/**
 * A cap of zero decodes nothing, which reaches a caller as silence they have
 * to diagnose rather than an error they can read. A cap of zero is not a
 * configuration, it is a typo in a manifest, and every port refuses it.
 */
function requirePositiveCap(value: number): number {
  if (value <= 0) throw new Error(`max_new_tokens must be positive: ${value}`);
  return value;
}

export function decodeFromManifest(manifest: Record<string, unknown>): "single" | "fusion_mtp2" {
  // A check, not a cast: a list or a scalar named `decode` is refused rather
  // than read as a block that answers `undefined` for `.mode` and decodes the
  // single loop under a manifest naming another one. `decode: null` is the
  // single loop said out loud, which is what the reference reads it as.
  const block = objectOrNull(manifest, "decode", "single");
  const mode = block?.mode ?? "single";
  if (mode !== "single" && mode !== "fusion_mtp2") {
    throw new Error(`unsupported decode mode: ${JSON.stringify(mode)}`);
  }
  return mode;
}

/** Read the algorithm values out of a packed checkpoint's manifest. */
export function algorithmFromManifest(manifest: Record<string, unknown>): AlgorithmConfig {
  const samplingDefaults = blockAt(manifest, "sampling_defaults");
  const sil = idsAt(manifest, "silence_token_ids");
  const speech = blockAt(manifest, "speech_tokens");
  // `window: null` is the ragged window said out loud, the reference's own
  // spelling of it; an absent block is the same window said by saying nothing.
  // Anything else that is not an object is refused rather than ignored, which
  // `(manifest.window ?? {})` cannot do: it renders the ragged window for
  // `window: [1, 2, 3]` and then records that window in the canonical form, so
  // the fingerprint agrees with the misreading.
  const window = objectOrNull(manifest, "window", "ragged") ?? {};
  const eos = blockAt(manifest, "eos_floor");

  // An absent optional key is a ragged window, not the production one:
  // `manifest.py:_window_from` reads exactly these three as `None` when they
  // are missing, and Swift does the same. Filling 255/4254/238 here would make
  // the same manifest fingerprint differently in this port, which is the
  // divergence class this library exists to prevent. `productionWindow()`
  // is where the shipped numbers live.
  const winAt = "manifest['window']";
  const win: WindowConfig = {
    maxSpeechTokens: intAt(window, "max_speech_tokens", 255, winAt),
    staticLength: optionalIntAt(window, "static_length", winAt),
    padTokenId: optionalIntAt(window, "pad_token_id", winAt),
    staticPromptTokens: optionalIntAt(window, "static_prompt_tokens", winAt),
  };

  // Read before the chunking recipe because the reference builds it there, and
  // a manifest with a fault in both blocks must name the same one in both.
  const postprocess = postprocessFromManifest(manifest);
  const chunking = chunkingFromManifest(manifest);
  // Checked once, at the door, rather than per utterance. A chunking recipe
  // with no character budget makes `splitText` cut nothing and loop forever;
  // Python refuses it too, and this port reads the same key.
  validateChunkConfig(chunking);

  const cfg: AlgorithmConfig = {
    decode: decodeFromManifest(manifest),
    // Absent and an explicit null are one reading here, and it is the legacy
    // 5 ms: a manifest that never names the key predates the field, and one
    // that names it null says the same thing out loud.
    edgeFadeSeconds: numberAt(
      manifest.edge_fade_seconds === null ? {} : manifest,
      "edge_fade_seconds",
      0.005
    ),
    recipeVersion: recipeVersionFromManifest(manifest),
    guidance: guidanceFromManifest(manifest),
    guidanceRate: numberAt(manifest, "guidance_rate", 0.0),
    eulerSteps: intAt(manifest, "n_cfm_timesteps", 2),
    // Read, not hard-coded to null. `timeGrid` honours `eulerGrid`, so this
    // port looked like the one that supported an explicit grid while the
    // manifest parser threw it away before `timeGrid` ever saw it. An explicit
    // grid is preferred for anything that must match across implementations,
    // because "cosine" is a formula two codebases can write two ways;
    // `AlgorithmConfig.euler_grid` in `config.py` is where the reference says so.
    eulerGrid: eulerGridFromManifest(manifest),
    sampling: (() => {
      // Range checks mirror Python's `SamplingConfig.__post_init__`: a
      // manifest the reference refuses must be refused here too, or two
      // implementations render different audio under one fingerprint.
      const sampleAt = "manifest['sampling_defaults']";
      const temperature = numberAt(samplingDefaults, "temperature", 0.8, sampleAt);
      if (temperature <= 0 || temperature > 4)
        throw new Error(`temperature out of range: ${reprFloat(temperature)}`);
      const repetitionPenalty = numberAt(samplingDefaults, "repetition_penalty", 1.2, sampleAt);
      if (repetitionPenalty < 1.0)
        throw new Error(
          `repetition_penalty below 1.0 rewards repetition: ${reprFloat(repetitionPenalty)}`
        );
      const minP = numberAt(samplingDefaults, "min_p", 0.05, sampleAt);
      if (minP < 0 || minP >= 1)
        throw new Error(`min_p out of range: ${reprFloat(minP)}`);
      // Zero when absent, as `SamplingConfig` declares it: the floor the
      // release ships is in its manifest, and filling it in here made a
      // manifest without an `eos_floor` block fingerprint differently from
      // the reference reading the same bytes.
      const eosAt = "manifest['eos_floor']";
      const minTokensFloor = intAt(eos, "min_tokens_floor", 0, eosAt);
      if (minTokensFloor < 0)
        throw new Error(`min_tokens_floor must be >= 0: ${minTokensFloor}`);
      const minTokensTextRatio = numberAt(eos, "min_tokens_text_ratio", 0.0, eosAt);
      if (minTokensTextRatio < 0)
        throw new Error(
          `min_tokens_text_ratio must be >= 0: ${reprFloat(minTokensTextRatio)}`,
        );
      return {
      temperature,
      repetitionPenalty,
      minP,
      maxNewTokens: requirePositiveCap(intAt(samplingDefaults, "max_new_tokens", 255, sampleAt)),
      silenceTokenIds: sil,
      minTokensFloor,
      minTokensTextRatio,
      };
    })(),
    window: win,
    chunking,
    postprocess,
    text: { recipe: TEXT_RECIPE, grammar: grammarDigest() },
    sampleRate: intAt(manifest, "sample_rate", 24_000),
    // Read, not hard-coded: the same shape of bug `eulerGrid` had. It is
    // hashed into the fingerprint, so a checkpoint at a different token rate
    // must move the fingerprint rather than be silently overridden here.
    tokenRateHz: numberAt(manifest, "token_rate_hz", 25.0),
    speechVocabSize: intAt(manifest, "speech_vocab_size", 8194),
    startSpeechToken: intAt(speech, "start", 6561, "manifest['speech_tokens']"),
    stopSpeechToken: intAt(speech, "stop", 6562, "manifest['speech_tokens']"),
  };

  // Last, once every key has been read, which is where the reference runs the
  // rules that need more than one of them.
  validateWindowConfig(cfg.window);
  validateAlgorithmConfig(cfg);
  return cfg;
}

/** A loaded voice profile (mirror of `loudkit.voice.VoiceProfile`). */
export interface VoiceProfile {
  /** Prompt preparation strategy; older profiles default to first-10s. */
  enrolment?: string;
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
