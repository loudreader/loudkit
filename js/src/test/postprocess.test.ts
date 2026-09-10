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
  isStalled,
  isTrailingFiller,
  repetitionCut,
  isDropout,
  pacingOutliers,
  terminalEchoCut,
  type PostprocessConfig,
  type PostprocessMode,
  type RepetitionResume,
  type RepetitionSilence,
} from "../postprocess.js";
import { LRSamplerV1 } from "../sampler.js";
import { uniforms } from "../rng.js";

// `fileURLToPath`, not `.pathname`: a file: URL's pathname keeps the leading
// slash that a POSIX path wants and a Windows one does not, so on Windows this
// resolved to `/F:/…/postprocess.json`, which `readFileSync` then read as
// `F:\F:\…`, so every case in this file failed with ENOENT on a checkout that
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
    } else if (kind === "sil") {
      // True digital silence in the stall section's two-class scheme.
      for (let i = 0; i < count; i += 1) out.push(i % 4);
    } else if (kind === "breath") {
      // The contextually-quiet family: extends a dead-air run without
      // counting toward its gate.
      for (let i = 0; i < count; i += 1) out.push(4 + (i % 4));
    } else if (kind === "dead") {
      // True silence only the render census knows (the 6405 class): outside
      // the fixture's sampler list, inside its silence census.
      for (let i = 0; i < count; i += 1) out.push(12);
    } else if (kind === "sigh") {
      // Breath only the quiet census knows.
      for (let i = 0; i < count; i += 1) out.push(13);
    } else if (kind === "cycle_dead") {
      // The stutter with its pause on a census-only id.
      const half = Math.floor(count / 2);
      const cycle = [
        ...Array.from({ length: count - half }, (_, i) => 20 + i),
        ...Array.from({ length: half }, () => 12),
      ];
      for (let r = 0; r < (repeats ?? 0); r += 1) out.push(...cycle);
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
 * are the ones the fixture declares rather than this port's own defaults,
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
    desperationMinKeepPerTextToken: opt(
      "desperation_min_keep_per_text_token",
      PRODUCTION_POSTPROCESS.desperationMinKeepPerTextToken
    ),
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
    // String fields, absent from the fixture's config block like the band
    // keys, so the shipping values apply.
    repetitionResume:
      typeof c.repetition_resume === "string"
        ? (c.repetition_resume as RepetitionResume)
        : PRODUCTION_POSTPROCESS.repetitionResume,
    repetitionSilence:
      typeof c.repetition_silence === "string"
        ? (c.repetition_silence as RepetitionSilence)
        : PRODUCTION_POSTPROCESS.repetitionSilence,
    // Like the band keys: absent from the fixture's config block, so the
    // shipping value applies; Python builds its config the same way.
    stallRunTokens: opt("stall_run_tokens", PRODUCTION_POSTPROCESS.stallRunTokens),
    silenceRenderIds: [],
    quietRenderIds: [],
    dropoutMinTokens: need("dropout_min_tokens"),
    retryMaxAttempts: need("retry_max_attempts"),
    pacingTolerance: need("pacing_tolerance"),
  };
}

/**
 * `configFrom` plus the repetition_silence section's render-id censuses: the
 * ids only the censuses know, which the loop exemption must union in under
 * the acoustic family.
 */
function repSilenceConfig(fx: Json): PostprocessConfig {
  const cfg = configFrom(fx);
  cfg.silenceRenderIds = [...fx.repetition_silence.silence_render_ids];
  cfg.quietRenderIds = [...fx.repetition_silence.quiet_render_ids];
  return cfg;
}

/**
 * `configFrom` plus the stall section's render-id censuses. The fallback arm
 * (`renderIds` false) runs without them, as a checkpoint packed before the
 * census, where only the run trigger fires.
 */
function stallConfig(fx: Json, renderIds: boolean): PostprocessConfig {
  const cfg = configFrom(fx);
  if (renderIds) {
    cfg.silenceRenderIds = [...fx.stall.silence_render_ids];
    cfg.quietRenderIds = [...fx.stall.quiet_render_ids];
  }
  return cfg;
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
 * subtleties either of which a reimplementation gets wrong silently: the
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
 * tuned on one language is an assumption everywhere else, and the expensive
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
 * The decoder trapped in silence: the failure no tail rule can see.
 *
 * Every mute chunk and every mid-row hole in the interior-stall study shipped
 * as clean, because all six other rules anchor on the tail. The stall rule
 * condemns instead of cutting: the failure is a hole, and the fix is the retry
 * ladder. Detection is two-class (a true-silence gate, a quiet-family
 * continuation), which the fixture pins because single-set counting was
 * measured broken.
 */
test("the stall rule matches the fixture", () => {
  const fx = fixture();
  const sil = fx.silence_token_ids;
  const cases = fx.stall?.cases;
  assert.ok(cases?.length, "the fixture has no stall cases; nothing was compared");
  for (const c of cases) {
    const got = isStalled(build(c.shape), c.hit_ceiling, sil, stallConfig(fx, c.render_ids));
    assert.equal(got, c.stalled, `${c.name}: ${c.why}`);
  }
});

test("the stall resolver matches the fixture", () => {
  // The wiring is part of the contract: after repetition, before every tail
  // rescue, condemned like dropout.
  const fx = fixture();
  for (const c of fx.stall.cases) {
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
      stallConfig(fx, c.render_ids)
    );
    assert.deepEqual(
      { keep: got.keep, reason: got.reason, suspect: got.suspect },
      c.expect,
      `${c.name}: ${c.why}`
    );
  }
});

/**
 * A cap-hit desperation cut that keeps less than any full read.
 *
 * The one row that survived the stall fix: soren/da0028 chunk 4 burned 132
 * tokens to the ceiling and the seam cut kept 36, 1.44 s in which 33 of the
 * 36 kept tokens render near-silent through ids outside both manifest
 * censuses, invisible to every set-membership rule. The keep's *length* is
 * the only evidence there is: a keep under `desperationMinKeepPerTextToken`
 * per text token cannot hold a full read, so the verdict is condemned into
 * the retry ladder. The cut stands as the keep: an exhausted ladder ships
 * the trim, flagged, rather than the untrimmed babble. No censuses configured
 * here on purpose: the trigger is a length test and must fire identically on
 * a checkpoint packed before them.
 */
test("the starved rescue matches the fixture", () => {
  const fx = fixture();
  const cfg = configFrom(fx);
  const cases = fx.starved_rescue?.cases;
  assert.ok(cases?.length, "the fixture has no starved_rescue cases; nothing was compared");
  for (const c of cases) {
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

test("the starved-rescue floor is exclusive", () => {
  // keep == floor ships: `<`, not `<=`, so the pinned law has no ambiguity at
  // the boundary for a port to resolve differently. Text 20 puts the floor at
  // exactly 34.0.
  const fx = fixture();
  const cfg = configFrom(fx);
  const row = build([
    ["speech", 34],
    ["sil", 12],
    ["speech", 90],
  ]);
  const got = inspect(
    row,
    {
      textTokenCount: 20,
      minTokens: 24,
      eosPeakAt: -1,
      eosPeakProb: 0.0,
      ended: false,
      isTerminal: true,
      hitCeiling: true,
    },
    fx.silence_token_ids,
    cfg
  );
  assert.equal(got.reason, "desperation");
  assert.equal(got.keep, 34);
  assert.ok(!got.suspect, "a keep exactly at the floor is not under it");
});

test("a zero starved-rescue floor disables the trigger", () => {
  const fx = fixture();
  const cfg = { ...configFrom(fx), desperationMinKeepPerTextToken: 0.0 };
  const c = fx.starved_rescue.cases[0];
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
  assert.equal(got.reason, "desperation");
  assert.ok(!got.suspect, "zero must disable the trigger");
});

test("a stalled row is condemned whole, never cut", () => {
  const fx = fixture();
  const cfg = stallConfig(fx, true);
  const row = build([
    ["speech", 30],
    ["sil", 30],
    ["speech", 30],
  ]);
  const got = inspect(
    row,
    {
      textTokenCount: 40,
      minTokens: 48,
      eosPeakAt: -1,
      eosPeakProb: 0.0,
      ended: true,
      isTerminal: true,
      hitCeiling: false,
    },
    fx.silence_token_ids,
    cfg
  );
  assert.equal(got.reason, "stall");
  assert.equal(
    got.keep,
    row.length,
    "a stalled row must be handed back whole; the hole is mid-row and no cut can remove it"
  );
  assert.ok(got.suspect, "the caller has to be told, since nothing was changed");
});

/**
 * The loop exemption keys on acoustic silence, not the sampler list.
 *
 * The specimen: kathleen/en0023 seed 1234 parked a mid-chunk pause on ids
 * 6486 (x7) then 6405 (x24). Both render true silence, both are in the
 * manifest's `silence_render_ids`, neither in the sampler's list the
 * exemption used to read. The period-1 run fired as a loop and the cut
 * deleted the pause plus two whole sentences of correctly-read speech behind
 * it, verdict repetition, not suspect, audibly fluent. Six of the
 * checkpoint's eight truly-silent ids sit outside the sampler list, so the
 * exemption was blind on most real pauses. The law now unions the sampler
 * list with both render censuses for this one rule; the tail rules keep the
 * list they were calibrated against.
 */
test("the repetition-silence family matches the fixture", () => {
  const fx = fixture();
  const cfg = repSilenceConfig(fx);
  const cases = fx.repetition_silence?.cases;
  assert.ok(cases?.length, "the fixture has no repetition_silence cases; nothing was compared");
  for (const c of cases) {
    const got = repetitionCut(build(c.shape), fx.silence_token_ids, cfg);
    assert.equal(got, c.loop, `${c.name}: ${c.why}`);
  }
});

test("the repetition-silence resolver matches the fixture", () => {
  // The cascade is part of the contract: a declined loop falls through to
  // stall, which condemns the specimen's pause into the retry ladder instead
  // of shipping the cut.
  const fx = fixture();
  const cfg = repSilenceConfig(fx);
  for (const c of fx.repetition_silence.cases) {
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

test("sampling names the old law", () => {
  // The pre-amendment behaviour stays nameable, so a checkpoint measured
  // under it can declare what it measured, and this is what it did: cut at
  // the pause and delete everything behind it.
  const fx = fixture();
  const cfg = { ...repSilenceConfig(fx), repetitionSilence: "sampling" as const };
  const c = fx.repetition_silence.cases[0];
  const got = repetitionCut(build(c.shape), fx.silence_token_ids, cfg);
  assert.equal(got, 31, "the old law cut one token past the pause's start");
});

test("without censuses the union is the sampler list", () => {
  // A checkpoint packed before the censuses changes nothing: nothing on such
  // a build knows id 12 is silent, so the run still reads as a loop there,
  // under either field value.
  const fx = fixture();
  const c = fx.repetition_silence.cases[0];
  const got = repetitionCut(build(c.shape), fx.silence_token_ids, configFrom(fx));
  assert.equal(got, 31);
});

function resumeRequest(c: Json) {
  return {
    textTokenCount: c.text_tokens,
    minTokens: c.min_tokens,
    eosPeakAt: c.eos_peak_at,
    eosPeakProb: c.eos_peak_prob,
    ended: c.ended,
    isTerminal: c.is_terminal,
    hitCeiling: c.hit_ceiling,
  };
}

/**
 * A loop the decoder resumed from is condemned, never cut.
 *
 * The census fix (`repetitionSilence`) needs a manifest that names the silent
 * ids, and the published pack has none: on it the en0023 pause fired again as
 * a period-1 loop and the cut kept 52 of 206 tokens, deleting two sentences
 * of correctly-read speech, verdict repetition, no retry. The guard here
 * needs no silence knowledge at all: a genuine lock-up runs its cycle to the
 * end of the row (a ceiling truncates at most one incomplete copy,
 * `period - 1` tokens), so a qualifying loop followed by a full period or
 * more of other content is a decoder that resumed, and a decoder that
 * resumed was never locked. Such a row is handed back whole, suspect, into
 * the retry ladder. The section configures no censuses on purpose: it is the
 * arm `repetitionSilence` cannot reach.
 */
test("the repetition-resume rule matches the fixture", () => {
  // The bare rule still reports the loop; the law lives in the resolver.
  const fx = fixture();
  const cfg = configFrom(fx);
  const cases = fx.repetition_resume?.cases;
  assert.ok(cases?.length, "the fixture has no repetition_resume cases; nothing was compared");
  for (const c of cases) {
    const got = repetitionCut(build(c.shape), fx.silence_token_ids, cfg);
    assert.equal(got, c.loop, `${c.name}: ${c.why}`);
  }
});

test("the repetition-resume resolver matches the fixture", () => {
  const fx = fixture();
  const cfg = configFrom(fx);
  for (const c of fx.repetition_resume.cases) {
    const got = inspect(build(c.shape), resumeRequest(c), fx.silence_token_ids, cfg);
    assert.deepEqual(
      { keep: got.keep, reason: got.reason, suspect: got.suspect },
      c.expect,
      `${c.name}: ${c.why}`
    );
  }
});

test("cut names the old law", () => {
  // The pre-amendment behaviour stays nameable, so a checkpoint measured
  // under it can declare what it measured, and this is what it did: cut at
  // the pause and delete everything behind it.
  const fx = fixture();
  const cfg = { ...configFrom(fx), repetitionResume: "cut" as const };
  const c = fx.repetition_resume.cases[0];
  const got = inspect(build(c.shape), resumeRequest(c), fx.silence_token_ids, cfg);
  assert.equal(got.reason, "repetition");
  assert.equal(got.keep, c.loop, "the old law shipped the specimen's cut");
  assert.ok(!got.suspect);
});

test("the census arm is untouched", () => {
  // With the censuses configured the specimen's pause is exempt from the
  // loop rule entirely and stall condemns it: the repetition_silence
  // contract, byte for byte, guard or no guard.
  const fx = fixture();
  const cfg = repSilenceConfig(fx);
  const c = fx.repetition_silence.cases[0];
  const got = inspect(build(c.shape), resumeRequest(c), fx.silence_token_ids, cfg);
  assert.equal(got.reason, "stall");
  assert.equal(got.keep, c.expect.keep);
});

test("repetition outranks the stall condemnation", () => {
  // A row that both loops and stalls answers to the loop: an exactly
  // repeated cycle pins where the failure began. Here the decoder resumed
  // after the region, so the loop condemns rather than cuts, but it still
  // outranks the stall's condemnation, and the verdict names the anchor
  // that was found.
  const fx = fixture();
  const cfg = stallConfig(fx, true);
  const row = build([
    ["cycle", 4, 8],
    ["sil", 30],
    ["speech", 20],
  ]);
  const got = inspect(
    row,
    {
      textTokenCount: 40,
      minTokens: 48,
      eosPeakAt: -1,
      eosPeakProb: 0.0,
      ended: true,
      isTerminal: true,
      hitCeiling: false,
    },
    fx.silence_token_ids,
    cfg
  );
  assert.equal(got.reason, "repetition", "the exact anchor outranks the condemnation");
});

/**
 * Early truncation: the failure a listener cannot hear.
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
