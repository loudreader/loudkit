/**
 * Playing faster without talking higher — WSOLA, from first principles.
 *
 * Port of `loudkit.models.timestretch`. "Speed" in a reading app means what it
 * means on a video player: 1.5x is the same voice, sooner. Resampling gives you
 * a chipmunk; what is wanted is *time* stretched while *pitch* is left alone.
 *
 * **Why WSOLA and not a phase vocoder.** The phase vocoder is the other standard
 * answer and is better on sustained, harmonic material — held notes, chords.
 * Speech is the opposite kind of signal: it is mostly transients (plosives, the
 * attack of every syllable) sitting on a pitch that moves continuously. A phase
 * vocoder resynthesises from magnitudes and unwrapped phases, and its
 * characteristic failure on that material is transient smearing — a /t/ arriving
 * as a soft thud, "phasiness" on voiced segments — which is precisely the part
 * of speech intelligibility rests on. WSOLA never leaves the time domain: it
 * copies real waveform segments and only chooses *where* to copy them from, so a
 * plosive is either included whole or not at all. It cannot smear what it never
 * transforms.
 *
 * **The algorithm.** Cut the input into overlapping ~25 ms frames. Write them
 * back out at a hop that is fixed by the output rate (50 % overlap), and read
 * them in at a hop scaled by `speed`. The read position is not used as computed:
 * it is moved by up to ±10 ms to whichever offset best matches what the
 * previously written frame *would* naturally have been followed by. That search
 * is the "waveform similarity" in the name, and it keeps successive frames in
 * phase with each other, so the overlap-add reinforces
 * rather than cancels. A plain OLA without the search is the same code with the
 * search window set to zero, and it sounds like it: periodic warble at the frame
 * rate.
 *
 * Everything here is deterministic — no RNG, no adaptivity, no dependencies. The
 * constants are derived from the sample rate rather than written as sample
 * counts, so the same code is correct at 16 kHz or 48 kHz, and the five
 * implementations derive them the same way.
 *
 * **What it costs.** At 1.25x this is hard to tell from a native reading. At 2x,
 * or at 0.5x, it is audibly processed: the alignment search cannot always find a
 * match, and the artefact is a faint roughness or a doubled consonant. That is
 * the practical range, and the bounds below are set where the result stops being
 * worth offering rather than where the arithmetic stops working.
 */

/**
 * The range worth offering, not the range that runs.
 *
 * Outside it the alignment search stops finding matches often enough — the
 * required shift exceeds the ±10 ms it may look over — and the output is
 * recognisably processed rather than merely faster. Refused rather than clamped:
 * a caller who asked for 3x and silently got 2x has a bug that only a stopwatch
 * finds.
 */
export const MIN_SPEED = 0.5;
export const MAX_SPEED = 2.0;

/**
 * Analysis/synthesis frame, in milliseconds. Long enough to hold two periods of
 * the lowest voiced pitch this is used on (~80 Hz), short enough that a frame is
 * inside one phone.
 */
const FRAME_MS = 25.0;

/**
 * How far the read position may move to find a better join — a bit under one
 * pitch period at the low end of the voiced range, which is what the search is
 * looking for.
 */
const SEARCH_MS = 10.0;

/**
 * Frames overlap by half. A periodic Hann window at hop = frame/2 sums to
 * exactly one, so the overlap-add needs no normalisation of its own — the
 * denominator below only ever corrects the ends and the places the alignment
 * search moved a frame off the grid.
 */
const HANN_COLA_HOP = 2;

/**
 * Throw unless `speed` is usable, with a message that names the range.
 *
 * Kept here rather than in the engine so that every entry point — three engine
 * methods and whatever a caller builds on top of them — refuses the same values
 * with the same words, and a new entry point cannot forget to.
 *
 * A `RangeError` rather than a plain `Error` because that is what the language
 * calls this failure; the message is the part the other four ports share.
 */
export function validateSpeed(speed: number): void {
  if (!Number.isFinite(speed)) {
    throw new RangeError(`speed must be a finite number, not ${speed}`);
  }
  if (!(speed >= MIN_SPEED && speed <= MAX_SPEED)) {
    throw new RangeError(
      `speed ${speed} is outside [${MIN_SPEED}, ${MAX_SPEED}]. Beyond that ` +
        "range the time-stretch is audibly processed rather than merely " +
        "faster or slower, so it is refused rather than clamped."
    );
  }
}

/**
 * How long `n` samples become at `speed`.
 *
 * Written as `floor(n / speed + 0.5)` rather than `Math.round` on purpose:
 * Python rounds halves to even, Go, Rust, Swift and JavaScript do not, and a
 * one-sample disagreement between ports on an exact half is the kind of thing
 * that is found six months later in a conformance run. Spelling the literal in
 * all five keeps the arithmetic identical instead of merely usually equal.
 */
export function stretchedLength(n: number, speed: number): number {
  return Math.floor(n / speed + 0.5);
}

/**
 * `audio` played at `speed`, same pitch.
 *
 * `speed` greater than one shortens, less than one lengthens. `1.0` returns the
 * input unchanged — the *same* `Float32Array`, not a copy that happens to be
 * equal, because the engine's default must be a bypass and "bit-identical" is
 * easier to trust when there is no arithmetic to be identical about.
 *
 * `sampleRate` is not decorative: the frame, the hop and the search window are
 * all derived from it.
 *
 * Returns exactly `stretchedLength(audio.length, speed)` samples.
 */
export function timeStretch(audio: Float32Array, sampleRate: number, speed: number): Float32Array {
  validateSpeed(speed);
  if (speed === 1.0) return audio;

  const n = audio.length;
  const outLen = stretchedLength(n, speed);
  const frame = Math.floor((sampleRate * FRAME_MS) / 1000.0 + 0.5);
  const hop = Math.floor(frame / HANN_COLA_HOP);
  if (n <= frame || outLen <= 0 || hop <= 0) {
    // Nothing to overlap-add: a fragment shorter than one frame has no second
    // frame to align against. Cut or zero-padded to the right length instead,
    // which is wrong in the way silence is wrong rather than in the way a pitch
    // shift is. At 24 kHz a frame is 600 samples — a fortieth of a second, below
    // anything the engine renders.
    //
    // A zero hop joins that branch rather than looping forever — `writeAt += hop`
    // would never advance. It takes a sample rate under 60 Hz to reach, so it is
    // not a behaviour difference in any case a caller can hit; it turns a hang,
    // which no stack trace explains, into the short-fragment path. Python, Go and
    // Rust all guard it, and this port did not: the hop was computed *below* the
    // guard, so there was nothing to test.
    const out = new Float32Array(Math.max(outLen, 0));
    const keep = Math.min(Math.max(outLen, 0), n);
    out.set(audio.subarray(0, keep));
    return out;
  }

  const search = Math.floor((sampleRate * SEARCH_MS) / 1000.0 + 0.5);
  // Periodic Hann, i.e. 2*pi*i/frame and not /(frame-1). The periodic form is
  // the one that sums to exactly one at 50 % overlap; the symmetric form is off
  // by a hair at every frame boundary, which reads as a low-level buzz at the
  // frame rate — 40 Hz here, right in the range a listener notices.
  const window = new Float64Array(frame);
  for (let i = 0; i < frame; i++) {
    window[i] = 0.5 - 0.5 * Math.cos((2.0 * Math.PI * i) / frame);
  }

  // Everything below is float64. Float32Array accumulators would round each
  // overlap-add, and the five ports would then differ by more than the search
  // already makes them differ by.
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = audio[i];
  // Room for the last frame to be written whole; trimmed at the end.
  const acc = new Float64Array(outLen + frame);
  const weight = new Float64Array(outLen + frame);

  const target = new Float64Array(frame);
  let lastFrameAt = 0;
  let writeAt = 0;
  let k = 0;
  while (writeAt < outLen) {
    const ideal = Math.floor(k * hop * speed + 0.5);
    let readAt: number;
    if (k === 0) {
      readAt = 0;
    } else {
      // What the previous frame would naturally have been followed by. The
      // search asks which nearby segment continues *this*, not which one the
      // arithmetic pointed at.
      const from = lastFrameAt + hop;
      const available = Math.max(0, Math.min(frame, n - from));
      for (let i = 0; i < available; i++) target[i] = x[from + i];
      for (let i = available; i < frame; i++) target[i] = 0.0;
      readAt = bestMatch(x, target, available, ideal, search, frame);
    }
    readAt = Math.min(Math.max(readAt, 0), n - frame);

    for (let i = 0; i < frame; i++) {
      acc[writeAt + i] += window[i] * x[readAt + i];
      weight[writeAt + i] += window[i];
    }

    lastFrameAt = n >= frame + hop ? Math.min(readAt, n - frame - hop) : readAt;
    writeAt += hop;
    k += 1;
  }

  // The Hann pair sums to one in the interior, so this division is the identity
  // almost everywhere; it earns its place at the two ends, where only one frame
  // contributes and the raw sum would fade in and out.
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    out[i] = weight[i] > 1e-12 ? acc[i] / weight[i] : 0;
  }
  return out;
}

/**
 * The offset within ±`search` of `ideal` whose frame best continues `target`.
 *
 * Scored by cross-correlation normalised by the *candidate's* energy only — the
 * target's is the same for every candidate and cancels out of the ranking.
 * Without that normalisation the search prefers whichever candidate is loudest
 * rather than whichever fits, which at a syllable onset is exactly the wrong
 * one.
 *
 * Ties go to the lower offset (a strict `>`), so the choice does not depend on
 * iteration order and the five ports agree.
 *
 * `targetLength` is how much of `target` is real signal rather than the padding
 * a fixed-size buffer carries: a target truncated by the end of the input is
 * short, and a short target is no basis for a search, exactly as in the other
 * ports where the slice simply comes out smaller.
 */
function bestMatch(
  x: Float64Array,
  target: Float64Array,
  targetLength: number,
  ideal: number,
  search: number,
  frame: number
): number {
  const n = x.length;
  const lo = Math.max(0, ideal - search);
  const hi = Math.min(n - frame, ideal + search);
  if (hi < lo || targetLength < frame) {
    return Math.min(Math.max(ideal, 0), n - frame);
  }

  let bestAt = lo;
  let bestScore = -Infinity;
  for (let at = lo; at <= hi; at++) {
    let energy = 0.0;
    let dot = 0.0;
    for (let i = 0; i < frame; i++) {
      const c = x[at + i];
      energy += c * c;
      dot += c * target[i];
    }
    // A silent candidate scores zero rather than dividing by nothing.
    const score = energy <= 0.0 ? 0.0 : dot / Math.sqrt(energy);
    if (score > bestScore) {
      bestScore = score;
      bestAt = at;
    }
  }
  return bestAt;
}
