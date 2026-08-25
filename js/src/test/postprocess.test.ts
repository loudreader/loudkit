/**
 * The postprocess layer, against the shared conformance fixture.
 *
 * Every case in `tests/data/conformance/postprocess.json` is a regression from
 * the shipped reader or a named device trace, and every port runs the same file.
 * A rule that drifts in one language fails in one language.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import {
  PRODUCTION_POSTPROCESS,
  ceilingFor,
  desperationCut,
  endedTailTrim,
  inspect,
  isTrailingFiller,
  repetitionCut,
  isDropout,
  pacingOutliers,
  terminalEchoCut,
  type PostprocessConfig,
  type PostprocessMode,
} from "../postprocess.js";
import { LRSamplerV1 } from "../sampler.js";
import { uniforms } from "../rng.js";

// `fileURLToPath`, not `.pathname`: a file: URL's pathname keeps the leading
// slash that a POSIX path wants and a Windows one does not, so on Windows this
// resolved to `/F:/…/postprocess.json`, which `readFileSync` then read as
// `F:\F:\…` — every case in this file failed with ENOENT on a checkout that
// had the fixture sitting right there.
const FIXTURE =
  process.env.LOUDKIT_POSTPROCESS_FIXTURE ??
  fileURLToPath(new URL("../../../tests/data/conformance/postprocess.json", import.meta.url));

 
type Json = any;

function fixture(): Json {
  return JSON.parse(readFileSync(FIXTURE, "utf8"));
}

/** The fixture's token-shape builder, spelled out in its header. */
function build(shape: [string, number, number?][]): number[] {
  const out: number[] = [];
  for (const [kind, count, repeats] of shape) {
    if (kind === "speech") {
      for (let i = 0; i < count; i += 1) out.push(20 + (i % 60));
    } else if (kind === "quiet") {
      for (let i = 0; i < count; i += 1) out.push(i % 8);
    } else if (kind === "cycle") {
      // `count` is the period here; the third element the repeat count.
      const cycle = Array.from({ length: count }, (_, i) => 20 + (i % 60));
      for (let r = 0; r < (repeats ?? 0); r += 1) out.push(...cycle);
    } else if (kind === "cycle_mixed") {
      // Second half silence: the word-then-pause stutter.
      const half = Math.floor(count / 2);
      const cycle = [
        ...Array.from({ length: count - half }, (_, i) => 20 + i),
        ...Array.from({ length: half }, (_, i) => i % 8),
      ];
      for (let r = 0; r < (repeats ?? 0); r += 1) out.push(...cycle);
    } else throw new Error(`unknown segment kind ${kind}`);
  }
  return out;
}

/**
 * Build the detector config out of the fixture, so the numbers the tests run on
 * are the ones the fixture declares rather than this port's own defaults —
 * which is the whole point of a shared file.
 */
function configFrom(fx: Json, mode?: string): PostprocessConfig {
  const c = fx.config;
  const need = (key: string): number => {
    const v = c[key];
    if (typeof v !== "number") throw new Error(`fixture config missing ${key}`);
    return v;
  };
  // The band keys predate the fixture; absent means the shipping value,
  // exactly as the manifest parsers treat absence.
  const opt = (key: string, fallback: number): number =>
    typeof c[key] === "number" ? c[key] : fallback;
  return {
    mode: (mode ?? c.mode) as PostprocessMode,
    ceilingSpeechPerTextToken: need("ceiling_speech_per_text_token"),
    ceilingSlackTokens: need("ceiling_slack_tokens"),
    trailingFillerThreshold: need("trailing_filler_threshold"),
    trailingSilenceRunTokens: need("trailing_silence_run_tokens"),
    desperationBandRatio: opt(
      "desperation_band_ratio",
      PRODUCTION_POSTPROCESS.desperationBandRatio
    ),
    desperationBandFloor: opt(
      "desperation_band_floor",
      PRODUCTION_POSTPROCESS.desperationBandFloor
    ),
    fillerMinEosProbability: need("filler_min_eos_probability"),
    fillerMaxSpeechAfterRun: need("filler_max_speech_after_run"),
    desperationSpeechPerTextToken: need("desperation_speech_per_text_token"),
    desperationMinTextTokens: need("desperation_min_text_tokens"),
    endedTailSilenceRun: need("ended_tail_silence_run"),
    endedTailBlipMax: need("ended_tail_blip_max"),
    endedTailWordMax: need("ended_tail_word_max"),
    endedTailKeep: need("ended_tail_keep"),
    echoStrongEosProbability: need("echo_strong_eos_probability"),
    echoStrongMaxTail: need("echo_strong_max_tail"),
    echoStrongMinPositionPct: need("echo_strong_min_position_pct"),
    echoWeakEosProbability: need("echo_weak_eos_probability"),
    echoWeakMaxTail: need("echo_weak_max_tail"),
    echoWeakMinPositionPct: need("echo_weak_min_position_pct"),
    repetitionMaxPeriod: need("repetition_max_period"),
    repetitionMinCycles: need("repetition_min_cycles"),
    repetitionMinSpan: need("repetition_min_span"),
    dropoutMinTokens: need("dropout_min_tokens"),
    retryMaxAttempts: need("retry_max_attempts"),
    pacingTolerance: need("pacing_tolerance"),
  };
}

test("the shipping constants are the fixture's", () => {
  // Otherwise the cases below prove nothing about what actually runs.
  assert.deepEqual(PRODUCTION_POSTPROCESS, configFrom(fixture()));
});

test("the generation ceiling matches the fixture", () => {
  const fx = fixture();
  const cfg = configFrom(fx);
  for (const c of fx.ceiling) {
    assert.equal(ceilingFor(c.text_tokens, cfg, c.window), c.expect, `${c.name}: ${c.why}`);
  }
});

test("trailing filler matches the fixture", () => {
  const fx = fixture();
  const cfg = configFrom(fx);
  for (const c of fx.trailing_filler) {
    const got = isTrailingFiller(build(c.shape), c.from, fx.silence_token_ids, cfg);
    assert.equal(got, c.expect, `${c.name}: ${c.why}`);
  }
});

test("the desperation cut matches the fixture", () => {
  const fx = fixture();
  const cfg = configFrom(fx);
  for (const c of fx.desperation) {
    const got = desperationCut(
      build(c.shape),
      c.text_tokens,
      c.min_tokens,
      c.eos_peak_at,
      fx.silence_token_ids,
      cfg,
      c.peak_allowed
    );
    assert.equal(got, c.expect, `${c.name}: ${c.why}`);
  }
});

test("the ended-tail trim matches the fixture", () => {
  const fx = fixture();
  const cfg = configFrom(fx);
  for (const c of fx.ended_tail) {
    const got = endedTailTrim(build(c.shape), fx.silence_token_ids, cfg, c.is_terminal);
    assert.equal(got, c.expect, `${c.name}: ${c.why}`);
  }
});

test("the terminal echo cut matches the fixture", () => {
  const fx = fixture();
  const cfg = configFrom(fx);
  for (const c of fx.terminal_echo) {
    const got = terminalEchoCut(
      c.token_count,
      c.eos_peak_at,
      c.eos_peak_prob,
      c.min_tokens,
      c.is_terminal,
      c.hit_ceiling,
      cfg
    );
    assert.equal(got, c.expect, `${c.name}: ${c.why}`);
  }
});

test("the precedence matches the fixture", () => {
  // The part a caller cannot get right by itself.
  const fx = fixture();
  for (const c of fx.resolve) {
    const cfg = configFrom(fx, c.mode);
    const got = inspect(
      build(c.shape),
      {
        textTokenCount: c.text_tokens,
        minTokens: c.min_tokens,
        eosPeakAt: c.eos_peak_at,
        eosPeakProb: c.eos_peak_prob,
        ended: c.ended,
        isTerminal: c.is_terminal,
        hitCeiling: c.hit_ceiling,
      },
      fx.silence_token_ids,
      cfg
    );
    assert.deepEqual(
      { keep: got.keep, reason: got.reason, suspect: got.suspect },
      c.expect,
      `${c.name}: ${c.why}`
    );
  }
});

/**
 * The stop-token observation the postprocess layer reads.
 *
 * Pinned across languages because it is hand-written in five of them and it is
 * *audible*: two of the detector rules compare it against a threshold, so a port
 * that computes it differently cuts a chunk somewhere else. The quantity has two
 * subtleties either of which a reimplementation gets wrong silently — the
 * numerator is the stop token's weight taken BEFORE the min_p cutoff, and the
 * peak is recorded only PAST the floor.
 */
test("the EOS peak matches the shared fixture", () => {
  const vectors: Json = JSON.parse(
    readFileSync(
      process.env.LOUDKIT_FIXTURE ??
        fileURLToPath(new URL("../../../tests/data/conformance/vectors.json", import.meta.url)),
      "utf8"
    )
  );
  const section = vectors.eos_peak;
  assert.ok(section?.cases?.length, "the fixture has no eos_peak cases; nothing was compared");

  for (const c of section.cases) {
    const sampler = new LRSamplerV1({
      temperature: c.config.temperature,
      repetitionPenalty: c.config.repetition_penalty,
      minP: c.config.min_p,
      maxNewTokens: 255,
      silenceTokenIds: c.config.silence_token_ids,
      minTokensFloor: 0,
      minTokensTextRatio: 0,
    }, c.seed);
    sampler.observeEos(c.stop_token, c.eos_floor);

    const r = c.logits_recipe;
    const seen = new Uint8Array(r.vocab);
    for (let step = 0; step < r.steps; step++) {
      const u = uniforms(BigInt(r.seed), r.stream, step, 1, r.vocab);
      const row = new Float32Array(r.vocab);
      for (let i = 0; i < r.vocab; i++) row[i] = u[i] * r.scale + r.offset;
      seen[sampler.call(row, step, seen)] = 1;
    }
    const [at, prob] = sampler.eosPeak;
    assert.equal(at, c.expected_at, c.name);
    assert.ok(
      Math.abs(prob - c.expected_prob) <= section.prob_rtol * Math.abs(c.expected_prob),
      `${c.name}: peak prob ${prob}, want ${c.expected_prob}`
    );
  }
});

/**
 * The ceiling was settled on English traces; nine languages ship.
 *
 * Speech tokens per *text* token is a property of the orthography, so a constant
 * tuned on one language is an assumption everywhere else — and the expensive
 * direction of that assumption is a guard that truncates correct speech in a
 * language nobody measured. Measured with one voice held constant across nine
 * language tags, because the voice-to-voice spread on a single sentence is
 * larger than the language-to-language spread.
 */
test("the language guard matches the fixture", () => {
  const fx = fixture();
  const cfg = configFrom(fx);
  const cases = fx.language_guard?.cases;
  assert.ok(cases?.length, "the fixture has no language_guard cases; nothing was compared");

  const stopped: string[] = [];
  for (const c of cases) {
    const ceiling = ceilingFor(c.text_tokens, cfg, c.window);
    assert.equal(ceiling, c.expect, `${c.name}: ${c.why}`);
    const hit = c.measured_speech_tokens >= ceiling;
    assert.equal(
      hit,
      c.expect_stopped_by_ceiling,
      `${c.name} changed side of the ceiling: ${c.why}`
    );
    if (hit) stopped.push(c.name);
  }
  // One row belongs here and it is not a false positive: a Spanish three-word
  // phrase whose decoder never emitted a stop token. The guard caught a runaway;
  // it did not cut a legitimate read.
  assert.deepEqual(
    stopped,
    ["es_short"],
    "a new entry is a language being truncated by an English-tuned constant"
  );
});

/**
 * The loop the tail rules cannot see, because it happens mid-row.
 *
 * Every other rule reads the end of the chunk. A stuck decoder repeats inside
 * it, and the literature puts that failure first or second in every ranking of
 * what goes wrong with autoregressive speech models.
 */
test("repetition matches the fixture", () => {
  const fx = fixture();
  const cfg = configFrom(fx);
  const sil = new Set<number>(fx.silence_token_ids);
  const cases = fx.repetition;
  assert.ok(cases?.length, "the fixture has no repetition cases; nothing was compared");

  let negatives = 0;
  for (const c of cases) {
    if (c.expect === null) negatives += 1;
    const got = repetitionCut(build(c.shape), sil, cfg);
    assert.equal(got, c.expect, `${c.name}: ${c.why}`);
  }
  // A mid-sequence cut is the most destructive thing this layer can do, so the
  // cases that must NOT fire carry more weight than the ones that must.
  assert.ok(negatives >= 6, `only ${negatives} negative cases; too few`);
});

/**
 * Early truncation — the failure a listener cannot hear.
 *
 * Every other rule says the end of the row is wrong. This one says the row is
 * incomplete, which is why it reports rather than cuts.
 */
test("dropout matches the fixture", () => {
  const fx = fixture();
  const cfg = configFrom(fx);
  const cases = fx.dropout?.cases;
  assert.ok(cases?.length, "the fixture has no dropout cases; nothing was compared");
  for (const c of cases) {
    assert.equal(isDropout(c.tokens, c.text_tokens, cfg), c.expect, `${c.name}: ${c.why}`);
  }
});

test("pacing matches the fixture", () => {
  const fx = fixture();
  const cfg = configFrom(fx);
  const cases = fx.pacing?.cases;
  assert.ok(cases?.length, "the fixture has no pacing cases");
  for (const c of cases) {
    assert.deepEqual(pacingOutliers(c.ratios, cfg), c.expect, `${c.name}: ${c.why}`);
  }
});
