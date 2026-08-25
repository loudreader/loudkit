/**
 * WSOLA time-stretch: properties, deliberately not bytes.
 *
 * There is **no shared byte-level fixture** for this feature, in any of the five
 * ports, and that is a decision rather than an omission. The alignment search
 * ranks candidate offsets by a cross-correlation whose last bit depends on the
 * order a language sums floats in; one offset chosen differently moves every
 * sample after it, so a golden file would fail for a reason that is not a
 * defect, and the usual response to a fixture that fails for no reason is to
 * regenerate it — which switches the check off. The four properties asserted
 * here fail only when the behaviour is actually wrong: the output length is
 * exact, the pitch does not move (a resampler would move it by exactly `speed`),
 * the loudness survives, and `speed = 1.0` is a bypass rather than a stretch by
 * one.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { MAX_SPEED, MIN_SPEED, stretchedLength, timeStretch } from "../timestretch.js";

const RATE = 24_000;
const SPEEDS = [0.5, 0.8, 1.25, 1.5, 2.0];

/**
 * A voiced-sounding test signal: a fundamental plus its second harmonic, so the
 * alignment search has real periodic structure to lock onto rather than the
 * single lobe of a pure tone.
 */
function tone(f0: number, samples: number): Float32Array {
  const x = new Float32Array(samples);
  for (let i = 0; i < samples; i++) {
    const t = i / RATE;
    x[i] = Math.sin(2 * Math.PI * f0 * t) + 0.5 * Math.sin(4 * Math.PI * f0 * t);
  }
  return x;
}

/**
 * Fundamental by autocorrelation, over the lag range of a human voice.
 *
 * Unnormalised on purpose: the biased sum decays slowly with lag, which breaks
 * the tie between a period and twice a period in favour of the period — the
 * octave error every naive pitch tracker makes. The decay is far too gentle to
 * move the peak within its own lobe.
 */
function fundamental(x: Float32Array): number {
  const minLag = Math.floor(RATE / 500);
  const maxLag = Math.floor(RATE / 80);
  let bestLag = minLag;
  let best = -Infinity;
  for (let lag = minLag; lag <= maxLag; lag++) {
    let sum = 0;
    for (let i = 0; i + lag < x.length; i++) sum += x[i] * x[i + lag];
    if (sum > best) {
      best = sum;
      bestLag = lag;
    }
  }
  return RATE / bestLag;
}

function rms(x: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < x.length; i++) sum += x[i] * x[i];
  return Math.sqrt(sum / Math.max(1, x.length));
}

test("speed 1.0 gives back the very same array", () => {
  // Identity, not equality. The default must not enter the DSP at all: every
  // conformance vector and every existing caller has to be unaffected by this
  // feature existing, and "no arithmetic happened" is easier to trust than "the
  // arithmetic was the identity".
  const x = tone(220, 4_000);
  assert.equal(timeStretch(x, RATE, 1.0), x);
});

test("the output length is exactly floor(n / speed + 0.5)", () => {
  const n = 9_973; // odd, and not a multiple of the frame or the hop
  const x = tone(200, n);
  for (const speed of SPEEDS) {
    const out = timeStretch(x, RATE, speed);
    assert.equal(out.length, stretchedLength(n, speed), `speed ${speed}`);
    assert.equal(out.length, Math.floor(n / speed + 0.5), `speed ${speed}`);
  }
});

test("stretchedLength rounds halves up, the way the other four ports do", () => {
  // 5 / 2 is exactly 2.5. `round()` in Python answers 2 (halves to even) and
  // `Math.round` answers 3, so the literal `floor(x + 0.5)` is what keeps the
  // ports agreeing rather than agreeing-except-on-exact-halves.
  assert.equal(stretchedLength(5, 2.0), 3);
  assert.equal(stretchedLength(3, 2.0), 2);
  assert.equal(stretchedLength(24_000, 1.5), 16_000);
  assert.equal(stretchedLength(0, 1.5), 0);
});

test("pitch does not move and loudness survives", () => {
  const f0 = 220;
  const x = tone(f0, RATE / 2);
  const before = fundamental(x);
  for (const speed of [0.5, 1.5, 2.0]) {
    const out = timeStretch(x, RATE, speed);
    const after = fundamental(out);
    // A resampler would land at f0 * speed here — 110 Hz or 440 Hz — so 3 % is
    // a wide margin against the failure this is guarding, and a tight one
    // against a stretch that has started smearing.
    assert.ok(
      Math.abs(after - before) / before < 0.03,
      `speed ${speed}: pitch moved ${before.toFixed(1)} Hz -> ${after.toFixed(1)} Hz`
    );
    const ratio = rms(out) / rms(x);
    assert.ok(
      Math.abs(ratio - 1) < 0.15,
      `speed ${speed}: RMS ratio ${ratio.toFixed(3)} is outside 15 %`
    );
    assert.ok(
      out.every((v) => Number.isFinite(v)),
      `speed ${speed}: the output is not finite everywhere`
    );
  }
});

test("a speed outside the offered range is refused, not clamped", () => {
  const x = tone(220, 4_000);
  for (const bad of [0.49, 0, -1, 2.01, 3.0]) {
    assert.throws(
      () => timeStretch(x, RATE, bad),
      new RegExp(`outside \\[${MIN_SPEED}, ${MAX_SPEED}\\]`),
      `speed ${bad} was not refused`
    );
  }
  // Non-finite is its own message: NaN is neither inside nor outside a range,
  // and reporting it as "outside [0.5, 2.0]" would send the caller looking at
  // their bounds rather than at where their NaN came from.
  for (const bad of [NaN, Infinity, -Infinity]) {
    assert.throws(() => timeStretch(x, RATE, bad), /finite/, `speed ${bad} was not refused`);
  }
  // The bounds themselves are inside the range.
  assert.equal(timeStretch(x, RATE, MIN_SPEED).length, stretchedLength(x.length, MIN_SPEED));
  assert.equal(timeStretch(x, RATE, MAX_SPEED).length, stretchedLength(x.length, MAX_SPEED));
});

test("a fragment shorter than one frame is cut or padded, not stretched", () => {
  // Below one frame there is no second frame to align against, so the correct
  // answer is the right *length* filled with what there is — wrong in the way
  // silence is wrong rather than in the way a pitch shift is.
  const x = tone(220, 100);
  const faster = timeStretch(x, RATE, 2.0);
  assert.equal(faster.length, 50);
  assert.deepEqual(Array.from(faster), Array.from(x.subarray(0, 50)));

  const slower = timeStretch(x, RATE, 0.5);
  assert.equal(slower.length, 200);
  assert.deepEqual(Array.from(slower.subarray(0, 100)), Array.from(x));
  assert.ok(
    slower.subarray(100).every((v) => v === 0),
    "the pad is not silence"
  );

  // An empty input stays empty rather than throwing on a zero-length window.
  assert.equal(timeStretch(new Float32Array(0), RATE, 1.5).length, 0);
});

test("the frame is derived from the sample rate, not hardcoded", () => {
  // The same fragment is above one frame at 8 kHz (200 samples) and below it at
  // 48 kHz (1200), so a port that hardcodes 600 samples takes the wrong branch
  // at one of these two rates and the difference is a whole feature silently not
  // running.
  //
  // The signal has to be non-stationary for the two branches to be
  // distinguishable at all: a perfectly steady tone compressed 1.5x *is* its own
  // prefix, sample for sample, because nothing in it happens at a particular
  // time. An amplitude ramp gives the stretch something to move, and makes the
  // difference between the branches legible — a compressed ramp reaches its top,
  // a truncated one stops two thirds of the way up.
  const n = 1_200;
  const x = tone(220, n);
  for (let i = 0; i < n; i++) x[i] *= i / n;
  const peak = (a: Float32Array) => a.reduce((m, v) => Math.max(m, Math.abs(v)), 0);

  const at8k = timeStretch(x, 8_000, 1.5);
  const at48k = timeStretch(x, 48_000, 1.5);
  assert.equal(at8k.length, stretchedLength(n, 1.5));
  assert.equal(at48k.length, stretchedLength(n, 1.5));

  // 48 kHz: one frame is 1200 samples, so this fragment is not longer than a
  // frame and the output is the input's head, cut.
  assert.deepEqual(Array.from(at48k), Array.from(x.subarray(0, at48k.length)));

  // 8 kHz: one frame is 200 samples, so the overlap-add ran instead. Same input,
  // same speed, same output length, different samples — and the ramp climbs
  // further, because a compressed ramp keeps rising where a truncated one simply
  // stops.
  assert.notDeepEqual(Array.from(at8k), Array.from(at48k));
  assert.ok(peak(at8k) > peak(at48k), "the overlap-add path only truncated the ramp");
});

test("a sample rate too low to have a hop does not hang", () => {
  // The guard that no implementation tested, which is how two of the five —
  // this one included — shipped without it.
  //
  // Below ~60 Hz the derived frame is one sample, so the hop (frame / 2) is
  // zero, and `writeAt += hop` never advances. This port computed the hop
  // *after* the degenerate-shape guard, so `timeStretch(new Float32Array(64),
  // 40, 1.5)` ran forever while Python returned 43 samples in microseconds.
  // Nothing was red, because nothing asked.
  //
  // Bounded by the clock as well as by the length: a regression here is a hang,
  // and a hang wedges CI rather than failing it.
  const started = Date.now();
  const got = timeStretch(new Float32Array(64), 40, 1.5);
  assert.ok(Date.now() - started < 5_000, "the overlap-add loop did not terminate");
  assert.equal(got.length, stretchedLength(64, 1.5));
  assert.equal(got.length, 43);
});

test("two calls agree bit for bit", () => {
  // No RNG, no adaptivity, no wall clock. Same in, same out, forever — which is
  // what lets the engine promise the same bytes for the same seed *and* the
  // same speed.
  const x = tone(220, RATE);
  const first = timeStretch(x, RATE, 1.4);
  const second = timeStretch(x, RATE, 1.4);
  assert.deepEqual(Array.from(first), Array.from(second));
});

test("silence stays silent", () => {
  // The one case that exercises the `energy <= 0` branch of the alignment
  // search: every candidate frame has zero energy, so the normalised
  // cross-correlation would divide by nothing. A missing guard there shows up
  // as NaN across the whole output rather than as a wrong sample.
  const got = timeStretch(new Float32Array(RATE), RATE, 1.5);
  assert.equal(got.length, stretchedLength(RATE, 1.5));
  for (let i = 0; i < got.length; i++) {
    assert.equal(got[i], 0, `silence came back as ${got[i]} at sample ${i}`);
  }
});

test("nothing clips or goes non-finite", () => {
  // Overlap-add divides by an accumulated window; a denominator that reaches
  // zero where the numerator does not is an infinity, and one infinity in a
  // waveform is a click loud enough to hurt.
  const x = tone(220, RATE);
  for (const speed of SPEEDS) {
    const got = timeStretch(x, RATE, speed);
    for (let i = 0; i < got.length; i++) {
      assert.ok(Number.isFinite(got[i]), `speed ${speed} produced ${got[i]} at sample ${i}`);
      assert.ok(
        Math.abs(got[i]) <= 1.2 * 1.5,
        `speed ${speed} produced ${got[i]} at sample ${i}, well past the input's range`
      );
    }
  }
});
