/**
 * The five refusals `validate_reference_audio` makes on the Python side, run
 * here without weights. Each message is the reference's, word for word,
 * because a caller reads it and a port is held to it.
 */

import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { validateReferenceAudio } from "../enroll.js";

const SR = 24_000;

const GOOD_INPUT =
  "A good input is 5 to 10 seconds of one person speaking, clean, " +
  "without music or a second voice.";

const NAN_REFUSAL =
  "the recording contains NaN or Inf samples, so no voice can be derived " +
  `from it. Re-export the file. ${GOOD_INPUT}`;

/** A deterministic clip loud enough to pass the silence floor. */
function speechLike(n: number): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.fround(0.5 * Math.sin((2 * Math.PI * 220 * i) / SR));
  return out;
}

function refusal(audio: Float32Array, sampleRate: number): string {
  try {
    validateReferenceAudio(audio, sampleRate);
  } catch (error) {
    return (error as Error).message;
  }
  throw new Error("the recording was accepted");
}

test("a non-positive rate is refused", () => {
  const audio = speechLike(SR);
  assert.equal(refusal(audio, 0), "sample rate must be positive, got 0");
  assert.equal(refusal(audio, -24_000), "sample rate must be positive, got -24000");
  // `number` also carries the three the other ports cannot express.
  assert.equal(refusal(audio, NaN), "sample rate must be positive, got NaN");
  assert.equal(refusal(audio, Infinity), "sample rate must be positive, got Infinity");
});

test("NaN and Inf samples are refused", () => {
  for (const [at, value] of [
    [0, NaN],
    [2 * SR - 1, NaN],
    [0, Infinity],
    [2 * SR - 1, -Infinity],
  ] as [number, number][]) {
    const audio = speechLike(2 * SR);
    audio[at] = value;
    assert.equal(refusal(audio, SR), NAN_REFUSAL, `sample ${at} = ${value}`);
  }
});

/**
 * 720 samples is one short of the reflect padding `matchaMel` reads. Without
 * this guard it reads past the end of the typed array, where an out-of-range
 * index is `undefined`, and the whole prompt mel comes back NaN.
 */
test("a clip shorter than a second is refused before the reflect padding", () => {
  for (const [samples, seconds] of [
    [720, "0.03"],
    [0, "0.00"],
    // Half a second: no NaN, but the utterance encoder pads it out to its
    // 1.6 s first partial and enrolls mostly padding.
    [SR / 2, "0.50"],
    // One sample under the minimum. The reported seconds round to 1.00 and the
    // refusal still stands: the comparison is on the exact length.
    [SR - 1, "1.00"],
  ] as [number, string][]) {
    assert.equal(
      refusal(speechLike(samples), SR),
      `the recording is ${seconds} s: too short to enroll a speaker from ` +
        `(minimum 1 s). ${GOOD_INPUT}`,
      `${samples} samples`
    );
  }
});

test("a recording longer than thirty seconds is refused", () => {
  assert.equal(
    refusal(speechLike(31 * SR), SR),
    "the recording is 31.0 s. Only the first 10 s become the voice prompt, and " +
      "the whole clip shapes the speaker embedding, so a long recording enrolls " +
      "something the prompt does not carry. Trim it to the best 5 to 10 seconds " +
      `(at most 30 s). ${GOOD_INPUT}`
  );
});

test("a silent recording is refused", () => {
  for (const [value, peak] of [
    [0, "0.0e+00"],
    [5e-5, "5.0e-05"],
    // The floor itself. Float32Array rounds 1e-4 down, so the loudest sample a
    // float32 clip can hold at this level is still under the float64 floor the
    // reference compares against.
    [1e-4, "1.0e-04"],
  ] as [number, string][]) {
    assert.equal(
      refusal(new Float32Array(2 * SR).fill(value), SR),
      `the recording is silent (peak ${peak}); there is no voice in it to ` +
        `enroll. ${GOOD_INPUT}`,
      `peak ${value}`
    );
  }
});

/**
 * Finiteness is checked before anything arithmetic, because one NaN poisons
 * every statistic below it. A clip that breaks two rules names the first.
 */
test("the order of the checks is the reference order", () => {
  const shortAndNaN = speechLike(720);
  shortAndNaN[0] = NaN;
  assert.equal(refusal(shortAndNaN, SR), NAN_REFUSAL);

  const silentAndNaN = new Float32Array(2 * SR);
  silentAndNaN[7] = NaN;
  assert.equal(refusal(silentAndNaN, SR), NAN_REFUSAL);

  // Length before loudness, as the reference orders them.
  assert.ok(refusal(new Float32Array(31 * SR), SR).startsWith("the recording is 31.0 s"));
});

/**
 * The clip every port's enrollment conformance runs on. It needs no graphs to
 * be judged, so the guard is held to real reference audio and not only to
 * synthetic tones.
 */
test("the fixture clip still enrolls", (t) => {
  let dir = process.env.LOUDKIT_ENROLL_FIXTURE;
  if (!dir) {
    let up = dirname(fileURLToPath(import.meta.url));
    for (;;) {
      const candidate = join(up, "tests", "data", "enrollment");
      if (existsSync(join(candidate, "ref_audio.f32"))) {
        dir = candidate;
        break;
      }
      const parent = dirname(up);
      if (parent === up) break;
      up = parent;
    }
  }
  if (!dir || !existsSync(join(dir, "ref_audio.f32"))) {
    t.skip("enrollment fixture not found");
    return;
  }
  const buf = readFileSync(join(dir, "ref_audio.f32"));
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const audio = new Float32Array(buf.byteLength / 4);
  for (let i = 0; i < audio.length; i++) audio[i] = view.getFloat32(i * 4, true);
  validateReferenceAudio(audio, SR);
});

/**
 * The bounds are inclusive at both ends and the fixture clip sits inside them,
 * so tightening the guard cannot start refusing what already enrolls.
 */
test("the band itself is accepted", () => {
  for (const samples of [SR, SR + 1, 15 * SR, 30 * SR]) {
    // One sample just over the float32 floor, and silence everywhere else.
    const audio = new Float32Array(samples);
    audio[0] = 1.01e-4;
    validateReferenceAudio(audio, SR);
  }
});
