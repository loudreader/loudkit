/**
 * The algorithm fingerprint — one string that says whether two engines agree.
 *
 * Every other cross-language check in this project compares a behaviour
 * somebody thought to compare: the speech funnel because there are 30 fixture
 * cases for it, the splitter because there are 18. This compares the *whole*
 * algorithm configuration in one string, so a field nobody wrote a test for
 * still cannot drift silently.
 *
 * The failure mode is concrete: an `euler_grid` hard-coded to `null` by one
 * port's manifest parser while `timeGrid` sits there honouring it;
 * a `silence_token_ids` cast rather than checked, so a JSON string reaches
 * `.includes()` and matches substrings. Each is invisible to behaviour
 * comparison alone. This finds all of them at once, and the next one
 * for free.
 *
 * The canonical form is specified rather than incidental — see
 * `AlgorithmConfig.canonical_form` in `loudkit/config.py`. Built by hand rather
 * than with `JSON.stringify`: the byte-for-byte output is the contract, and a
 * serialiser is free to change how it renders a number or orders a key.
 */

import { createHash } from "node:crypto";

import type { AlgorithmConfig } from "./types.js";

/**
 * Bumped only when the *set* of hashed fields changes, never when a value does.
 * Adding a field with a default must not re-fingerprint an algorithm that did
 * not change — a check that cries wolf on every upgrade is one people learn to
 * override.
 */
export const FINGERPRINT_SCHEMA = 1;

/**
 * A float the way Python's `repr()` renders it, as a JSON *string*.
 *
 * Quoted on purpose: it keeps every JSON parser in every language from
 * re-rendering the number with its own idea of precision. JS `String(n)`
 * already gives the shortest round-tripping decimal — but renders `25` where
 * Python renders `25.0`, and that one character is the difference between a
 * matching fingerprint and a mysterious one.
 */
function reprFloat(value: number): string {
  const s = String(value);
  return Number.isFinite(value) && !/[.eE]/.test(s) ? `${s}.0` : s;
}

/** A JSON string literal, escaped the way `json.dumps` escapes. */
function jsonStr(s: string): string {
  let out = '"';
  for (const ch of s) {
    const code = ch.codePointAt(0)!;
    if (ch === '"') out += '\\"';
    else if (ch === "\\") out += "\\\\";
    else if (ch === "\n") out += "\\n";
    else if (ch === "\t") out += "\\t";
    else if (ch === "\r") out += "\\r";
    else if (code < 0x20) out += `\\u${code.toString(16).padStart(4, "0")}`;
    else out += ch;
  }
  return out + '"';
}

const num = (v: number): string => jsonStr(reprFloat(v));
const optInt = (v: number | null): string => (v === null ? "null" : String(v));

/** The exact string that gets hashed. */
export function canonicalForm(cfg: AlgorithmConfig): string {
  const splitOn = cfg.chunking.splitOn.map(jsonStr).join(",");
  const chunking =
    `{"enabled":${cfg.chunking.enabled},"max_tokens":${cfg.chunking.maxTokens},` +
    `"prefix_tokens":${cfg.chunking.prefixTokens},"split_on":[${splitOn}]}`;

  // Sorted, because the manifest's order is whatever the packer wrote and the
  // hash must not depend on it.
  const silence = [...cfg.sampling.silenceTokenIds].sort((a, b) => a - b).join(",");
  const sampling =
    `{"max_new_tokens":${cfg.sampling.maxNewTokens},"min_p":${num(cfg.sampling.minP)},` +
    `"min_tokens_floor":${cfg.sampling.minTokensFloor},` +
    `"min_tokens_text_ratio":${num(cfg.sampling.minTokensTextRatio)},` +
    `"repetition_penalty":${num(cfg.sampling.repetitionPenalty)},` +
    `"silence_token_ids":[${silence}],"temperature":${num(cfg.sampling.temperature)}}`;

  const window =
    `{"max_speech_tokens":${cfg.window.maxSpeechTokens},` +
    `"pad_token_id":${optInt(cfg.window.padTokenId)},` +
    `"static_length":${optInt(cfg.window.staticLength)},` +
    `"static_prompt_tokens":${optInt(cfg.window.staticPromptTokens)}}`;

  // Keys sorted, as everywhere in this form. The detectors remove tokens, so a
  // port using a different threshold produces different audio — exactly the
  // silent drift a whole-config hash exists to catch.
  const pp = cfg.postprocess;
  const postprocess =
    `{"ceiling_slack_tokens":${pp.ceilingSlackTokens},` +
    `"ceiling_speech_per_text_token":${num(pp.ceilingSpeechPerTextToken)},` +
    `"desperation_band_floor":${pp.desperationBandFloor},` +
    `"desperation_band_ratio":${num(pp.desperationBandRatio)},` +
    `"desperation_min_text_tokens":${pp.desperationMinTextTokens},` +
    `"desperation_speech_per_text_token":${num(pp.desperationSpeechPerTextToken)},` +
    `"dropout_min_tokens":${pp.dropoutMinTokens},` +
    `"echo_strong_eos_probability":${num(pp.echoStrongEosProbability)},` +
    `"echo_strong_max_tail":${pp.echoStrongMaxTail},` +
    `"echo_strong_min_position_pct":${pp.echoStrongMinPositionPct},` +
    `"echo_weak_eos_probability":${num(pp.echoWeakEosProbability)},` +
    `"echo_weak_max_tail":${pp.echoWeakMaxTail},` +
    `"echo_weak_min_position_pct":${pp.echoWeakMinPositionPct},` +
    `"ended_tail_blip_max":${pp.endedTailBlipMax},` +
    `"ended_tail_keep":${pp.endedTailKeep},` +
    `"ended_tail_silence_run":${pp.endedTailSilenceRun},` +
    `"ended_tail_word_max":${pp.endedTailWordMax},` +
    `"filler_max_speech_after_run":${pp.fillerMaxSpeechAfterRun},` +
    `"filler_min_eos_probability":${num(pp.fillerMinEosProbability)},` +
    `"mode":${jsonStr(pp.mode)},` +
    `"pacing_tolerance":${num(pp.pacingTolerance)},` +
    `"repetition_max_period":${pp.repetitionMaxPeriod},` +
    `"repetition_min_cycles":${pp.repetitionMinCycles},` +
    `"repetition_min_span":${pp.repetitionMinSpan},` +
    `"retry_max_attempts":${pp.retryMaxAttempts},` +
    `"trailing_filler_threshold":${num(pp.trailingFillerThreshold)},` +
    `"trailing_silence_run_tokens":${pp.trailingSilenceRunTokens}}`;

  const eulerGrid = cfg.eulerGrid ? `[${cfg.eulerGrid.map(num).join(",")}]` : "null";

  // The funnel's identity travels in the fingerprint: its code version, and the
  // digest of the grammar file this port reads. Each implementation hashes its own
  // copy, so a port whose data has drifted computes a different fingerprint and the
  // engine refuses to start — which is how drift is caught, rather than by someone
  // eventually hearing it.
  const text = `{"grammar":${jsonStr(cfg.text.grammar)},"recipe":${jsonStr(cfg.text.recipe)}}`;

  const body =
    `{"chunking":${chunking},"euler_grid":${eulerGrid},"euler_steps":${cfg.eulerSteps},` +
    `"guidance":${jsonStr(cfg.guidance)},"guidance_rate":${num(cfg.guidanceRate)},` +
    `"postprocess":${postprocess},` +
    `"recipe_version":${jsonStr(cfg.recipeVersion)},"sample_rate":${cfg.sampleRate},` +
    `"sampling":${sampling},"speech_vocab_size":${cfg.speechVocabSize},` +
    `"start_speech_token":${cfg.startSpeechToken},` +
    `"stop_speech_token":${cfg.stopSpeechToken},` +
    `"text":${text},` +
    `"token_rate_hz":${num(cfg.tokenRateHz)},"window":${window}}`;

  return `{"algorithm":${body},"schema":${FINGERPRINT_SCHEMA}}`;
}

/**
 * First 16 hex characters of SHA-256 over {@link canonicalForm}.
 *
 * Two engines whose fingerprints differ are computing different things,
 * whatever their outputs happen to sound like — which is the point: the
 * guidance defect this project was built around produced plausible audio on
 * both sides of the mismatch, so no listening test could have found it.
 */
export function fingerprint(cfg: AlgorithmConfig): string {
  return createHash("sha256").update(canonicalForm(cfg), "utf8").digest("hex").slice(0, 16);
}
