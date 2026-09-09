/**
 * The algorithm fingerprint: one string that says whether two engines agree.
 *
 * Every other cross-language check in this project compares a behaviour
 * somebody thought to compare: the speech funnel because there are 30 fixture
 * cases for it, the splitter because there are 48. This compares the *whole*
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
 * The canonical form is specified rather than incidental; see
 * `AlgorithmConfig.canonical_form` in `loudkit/config.py`. Built by hand rather
 * than with `JSON.stringify`: the byte-for-byte output is the contract, and a
 * serialiser is free to change how it renders a number or orders a key.
 */

import { createHash } from "node:crypto";
import { EDGE_FADE_SECONDS } from "./timestretch.js";

import type { AlgorithmConfig } from "./types.js";

/**
 * Bumped only when the *set* of hashed fields changes, never when a value does.
 * Adding a field with a default must not re-fingerprint an algorithm that did
 * not change, because a check that cries wolf on every upgrade is one people
 * learn to
 * override.
 */
export const FINGERPRINT_SCHEMA = 1;

/**
 * A float the way Python's `repr()` renders it.
 *
 * The canonical form quotes it as a JSON *string*, which keeps every JSON
 * parser in every language from re-rendering the number with its own idea of
 * precision; the refusal sentences in `types.ts` render the same way, so a
 * manifest reads the same in a message as it hashes. `go/internal/pyfmt.Float`
 * is this function in Go.
 *
 * Both languages give the shortest decimal that round-trips, so the digits
 * agree. Where they part is the layout around those digits, on three counts,
 * all three of which move a fingerprint:
 *
 * - Python spells a whole float `25.0`; `String(25)` gives `25`.
 * - Python switches to an exponent when the decimal point sits past 16 digits
 *   or at or before the fourth leading zero (`1e+16`, `1e-05`); JS switches at
 *   1e21 and 1e-7, so it spells those `10000000000000000` and `0.00001`.
 * - Python pads an exponent to two digits (`1e-09`); JS writes `1e-9`.
 *
 * Held to `decpt <= -4 || decpt > 16` because that is the test CPython's
 * `format_float_short` makes, where `decpt` is the position of the decimal
 * point in the digit string.
 */
export function reprFloat(value: number): string {
  if (Number.isNaN(value)) return "nan";
  if (!Number.isFinite(value)) return value > 0 ? "inf" : "-inf";
  const sign = value < 0 || Object.is(value, -0) ? "-" : "";
  const abs = Math.abs(value);
  if (abs === 0) return `${sign}0.0`;
  const [mantissa, exponent] = abs.toExponential().split("e");
  const exp = Number(exponent);
  const digits = mantissa.replace(".", "");
  const decpt = exp + 1;
  if (decpt <= -4 || decpt > 16) {
    const head = digits.length > 1 ? `${digits[0]}.${digits.slice(1)}` : digits;
    const width = String(Math.abs(exp)).padStart(2, "0");
    return `${sign}${head}e${exp < 0 ? "-" : "+"}${width}`;
  }
  if (decpt <= 0) return `${sign}0.${"0".repeat(-decpt)}${digits}`;
  if (decpt >= digits.length) {
    return `${sign}${digits}${"0".repeat(decpt - digits.length)}.0`;
  }
  return `${sign}${digits.slice(0, decpt)}.${digits.slice(decpt)}`;
}

/** A JSON string literal, escaped the way `json.dumps` escapes. */
function jsonStr(s: string): string {
  let out = '"';
  for (const ch of s) {
    const code = ch.codePointAt(0) ?? 0;
    if (ch === '"') out += '\\"';
    else if (ch === "\\") out += "\\\\";
    else if (ch === "\n") out += "\\n";
    else if (ch === "\t") out += "\\t";
    else if (ch === "\r") out += "\\r";
    // Python's `json.dumps` defaults to `ensure_ascii=True`, so every
    // non-ASCII character in the canonical form is a \uXXXX escape and an
    // astral one is a surrogate pair. No hashed string had a non-ASCII
    // character in it until `chunking.abbreviations` carried "\u015bw", and
    // the raw UTF-8 spelling hashed differently from the reference in every
    // port at once.
    else if (code < 0x20 || code >= 0x7f) {
      if (code > 0xffff) {
        const v = code - 0x10000;
        out += `\\u${(0xd800 + (v >> 10)).toString(16).padStart(4, "0")}`;
        out += `\\u${(0xdc00 + (v & 0x3ff)).toString(16).padStart(4, "0")}`;
      } else {
        out += `\\u${code.toString(16).padStart(4, "0")}`;
      }
    } else out += ch;
  }
  return out + '"';
}

const num = (v: number): string => jsonStr(reprFloat(v));
const optInt = (v: number | null): string => (v === null ? "null" : String(v));
/**
 * A token-id census in the order the manifest declared it.
 *
 * Sorted here once, on the argument that the packer's order was arbitrary and
 * the hash should not depend on it. It is the reference that decides that, and
 * the reference does not sort: `AlgorithmConfig.canonical_form` sorts the
 * *keys* of every mapping and renders every list as it stands, and Rust and
 * Swift do the same. So a manifest declaring `[9, 1, 5]` hashed here to the
 * number Python answers for `[1, 5, 9]`, and the one mechanism whose whole job
 * is to notice that two engines are computing different things was quietly
 * agreeing with a misreading.
 *
 * Nothing sorts on the way in either, so the order a census is declared in is
 * the order the sampler and the detectors read it in. The shipped 0.1.1
 * manifest happens to declare all three ascending, which is why the
 * fingerprint never moved and nothing caught this.
 */
const idList = (ids: readonly number[]): string => ids.join(",");

/** The exact string that gets hashed. */
export function canonicalForm(cfg: AlgorithmConfig): string {
  const splitOn = cfg.chunking.splitOn.map(jsonStr).join(",");
  // Keys sorted, as everywhere in this form: "abbreviations" before "enabled",
  // and "mid_sentence_period" between "max_tokens" and "prefix_tokens". Five
  // canonical forms are hand-written and a new field's position in that order
  // is part of the contract.
  const abbreviations = cfg.chunking.abbreviations.map(jsonStr).join(",");
  const chunking =
    `{"abbreviations":[${abbreviations}],` +
    `"cap_resplit":${jsonStr(cfg.chunking.capResplit)},` +
    `"enabled":${cfg.chunking.enabled},` +
    `"max_tokens":${cfg.chunking.maxTokens},` +
    `"mid_sentence_period":${jsonStr(cfg.chunking.midSentencePeriod)},` +
    `"prefix_tokens":${cfg.chunking.prefixTokens},"split_on":[${splitOn}]}`;

  const silence = idList(cfg.sampling.silenceTokenIds);
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
  // port using a different threshold produces different audio, which is exactly
  // the
  // silent drift a whole-config hash exists to catch.
  const pp = cfg.postprocess;
  const postprocess =
    `{"ceiling_slack_tokens":${pp.ceilingSlackTokens},` +
    `"ceiling_speech_per_text_token":${num(pp.ceilingSpeechPerTextToken)},` +
    `"desperation_band_floor":${pp.desperationBandFloor},` +
    `"desperation_band_ratio":${num(pp.desperationBandRatio)},` +
    `"desperation_min_keep_per_text_token":${num(pp.desperationMinKeepPerTextToken)},` +
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
    `"quiet_render_ids":[${idList(pp.quietRenderIds)}],` +
    `"repetition_max_period":${pp.repetitionMaxPeriod},` +
    `"repetition_min_cycles":${pp.repetitionMinCycles},` +
    `"repetition_min_span":${pp.repetitionMinSpan},` +
    `"repetition_resume":${jsonStr(pp.repetitionResume)},` +
    `"repetition_silence":${jsonStr(pp.repetitionSilence)},` +
    `"retry_max_attempts":${pp.retryMaxAttempts},` +
    `"silence_render_ids":[${idList(pp.silenceRenderIds)}],` +
    `"stall_run_tokens":${pp.stallRunTokens},` +
    `"trailing_filler_threshold":${num(pp.trailingFillerThreshold)},` +
    `"trailing_silence_run_tokens":${pp.trailingSilenceRunTokens}}`;

  const eulerGrid = cfg.eulerGrid ? `[${cfg.eulerGrid.map(num).join(",")}]` : "null";

  // The funnel's identity travels in the fingerprint: its code version, and the
  // digest of the grammar file this port reads. Each implementation hashes its own
  // copy, so a port whose data has drifted computes a different fingerprint and the
  // engine refuses to start, which is how drift is caught rather than by someone
  // eventually hearing it.
  const text = `{"grammar":${jsonStr(cfg.text.grammar)},"recipe":${jsonStr(cfg.text.recipe)}}`;

  const body =
    `{"chunking":${chunking},` +
    (cfg.decode === "fusion_mtp2" ? `"decode_mode":"fusion_mtp2",` : "") +
    ((cfg.edgeFadeSeconds ?? EDGE_FADE_SECONDS) !== 0.005
      ? `"edge_fade_seconds":${num(cfg.edgeFadeSeconds ?? EDGE_FADE_SECONDS)},`
      : "") +
    `"euler_grid":${eulerGrid},"euler_steps":${cfg.eulerSteps},` +
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
 * whatever their outputs happen to sound like, which is the point: the
 * guidance defect this project was built around produced plausible audio on
 * both sides of the mismatch, so no listening test could have found it.
 */
export function fingerprint(cfg: AlgorithmConfig): string {
  return createHash("sha256").update(canonicalForm(cfg), "utf8").digest("hex").slice(0, 16);
}
