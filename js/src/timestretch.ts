/**
 * Playing faster without talking higher: WSOLA, from first principles.
 *
 * Port of `loudkit.models.timestretch`. "Speed" in a reading app means what it
 * means on a video player: 1.5x is the same voice, sooner. Resampling gives you
 * a chipmunk; what is wanted is *time* stretched while *pitch* is left alone.
 *
 * **Why WSOLA and not a phase vocoder.** The phase vocoder is the other standard
 * answer and is better on sustained, harmonic material such as held notes and
 * chords.
 * Speech is the opposite kind of signal: it is mostly transients (plosives, the
 * attack of every syllable) sitting on a pitch that moves continuously. A phase
 * vocoder resynthesises from magnitudes and unwrapped phases, and its
 * characteristic failure on that material is transient smearing (a /t/ arriving
 * as a soft thud, "phasiness" on voiced segments), which is precisely the part
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
 * Everything here is deterministic: no RNG, no adaptivity, no dependencies. The
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
 * Outside it the alignment search stops finding matches often enough, because
 * the required shift exceeds the ±10 ms it may look over, and the output is
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
 * How far the read position may move to find a better join: a bit under one
 * pitch period at the low end of the voiced range, which is what the search is
 * looking for.
 */
const SEARCH_MS = 10.0;

/**
 * Frames overlap by half. A periodic Hann window at hop = frame/2 sums to
 * exactly one, so the overlap-add needs no normalisation of its own: the
 * denominator below only ever corrects the ends and the places the alignment
 * search moved a frame off the grid.
 */
const HANN_COLA_HOP = 2;

/**
 * Throw unless `speed` is usable, with a message that names the range.
 *
 * Kept here rather than in the engine so that every entry point (three engine
 * methods and whatever a caller builds on top of them) refuses the same values
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
 * input unchanged: the *same* `Float32Array`, not a copy that happens to be
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
    // shift is. At 24 kHz a frame is 600 samples, a fortieth of a second, below
    // anything the engine renders.
    //
    // A zero hop joins that branch rather than looping forever, since `writeAt += hop`
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
  // frame rate, 40 Hz here, right in the range a listener notices.
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
 * Scored by cross-correlation normalised by the *candidate's* energy only: the
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


/** The raised-cosine ramp {@link fadeEdges} puts on both ends of a rendered window, in seconds. */
export const EDGE_FADE_SECONDS = 0.02;

// The Python float32 ramps at 24 kHz, pinned by tests/data/conformance/edge_fade.json.
// The historical 5 ms ramp, 120 samples.
const EDGE_RAMP_24K_5MS = new Float32Array(new Uint32Array([
  0x0, 0x3936b000, 0x3a36aa00, 0x3acd6f00, 0x3b368880, 0x3b8e8780, 0x3bcd1c80, 0x3c0b7c00,
  0x3c360660, 0x3c6625e0, 0x3c8de8e0, 0x3cab8040, 0x3ccbd3e0, 0x3ceedde0, 0x3d0a4c00, 0x3d1e7dc8,
  0x3d3400b0, 0x3d4ad0d8, 0x3d62ea30, 0x3d7c4878, 0x3d8b738c, 0x3d9960b0, 0x3da7e924, 0x3db70a54,
  0x3dc6c190, 0x3dd70c08, 0x3de7e6d0, 0x3df94ef0, 0x3e05a0a0, 0x3e0edd4a, 0x3e185bd0, 0x3e221a80,
  0x3e2c179e, 0x3e36515e, 0x3e40c5f2, 0x3e4b7378, 0x3e56580e, 0x3e6171bc, 0x3e6cbe8a, 0x3e783c76,
  0x3e81f4b6, 0x3e87e1b0, 0x3e8de418, 0x3e93fadb, 0x3e9a24e3, 0x3ea06118, 0x3ea6ae5d, 0x3ead0b90,
  0x3eb37790, 0x3eb9f139, 0x3ec07761, 0x3ec708de, 0x3ecda485, 0x3ed44929, 0x3edaf59a, 0x3ee1a8a4,
  0x3ee8611a, 0x3eef1dc6, 0x3ef5dd76, 0x3efc9ef3, 0x3f01b086, 0x3f051146, 0x3f08711c, 0x3f0bcf73,
  0x3f0f2bae, 0x3f128534, 0x3f15db6b, 0x3f192dbd, 0x3f1c7b91, 0x3f1fc450, 0x3f230763, 0x3f264438,
  0x3f297a38, 0x3f2ca8d2, 0x3f2fcf74, 0x3f32ed8e, 0x3f360293, 0x3f390df4, 0x3f3c0f29, 0x3f3f05a5,
  0x3f41f0e2, 0x3f44d05e, 0x3f47a391, 0x3f4a69fc, 0x3f4d2322, 0x3f4fce84, 0x3f526ba9, 0x3f54fa19,
  0x3f577960, 0x3f59e90c, 0x3f5c48ae, 0x3f5e97d8, 0x3f60d622, 0x3f630326, 0x3f651e7e, 0x3f6727ce,
  0x3f691eb5, 0x3f6b02dc, 0x3f6cd3ea, 0x3f6e918e, 0x3f703b79, 0x3f71d15d, 0x3f7352f2, 0x3f74bff6,
  0x3f761824, 0x3f775b40, 0x3f788911, 0x3f79a161, 0x3f7aa3fe, 0x3f7b90b9, 0x3f7c6768, 0x3f7d27e6,
  0x3f7dd210, 0x3f7e65c7, 0x3f7ee2f1, 0x3f7f4978, 0x3f7f9948, 0x3f7fd256, 0x3f7ff495, 0x3f800000,
]).buffer);

// The shipped 20 ms ramp, 480 samples.
const EDGE_RAMP_24K_20MS = new Float32Array(new Uint32Array([
  0x0, 0x37348000, 0x38346000, 0x38caf000, 0x39346800, 0x398cf000, 0x39caf400, 0x3a0a1c00,
  0x3a346200, 0x3a644800, 0x3a8ce700, 0x3aaa7b00, 0x3acadf00, 0x3aee1200, 0x3b0a0a00, 0x3b1e7200,
  0x3b344180, 0x3b4b7800, 0x3b641480, 0x3b7e1800, 0x3b8cc080, 0x3b9b2840, 0x3baa4240, 0x3bba0f00,
  0x3bca8e40, 0x3bdbbf80, 0x3beda300, 0x3c001c00, 0x3c09bf60, 0x3c13bb60, 0x3c1e1020, 0x3c28bd20,
  0x3c33c2a0, 0x3c3f2060, 0x3c4ad620, 0x3c56e3e0, 0x3c634980, 0x3c7006e0, 0x3c7d1bc0, 0x3c854410,
  0x3c8c25d0, 0x3c933330, 0x3c9a6c10, 0x3ca1d050, 0x3ca95fe0, 0x3cb11ab0, 0x3cb900a0, 0x3cc111b0,
  0x3cc94db0, 0x3cd1b4a0, 0x3cda4650, 0x3ce302b0, 0x3cebe9b0, 0x3cf4fb40, 0x3cfe3720, 0x3d03ceb0,
  0x3d0896e8, 0x3d0d7428, 0x3d126668, 0x3d176da0, 0x3d1c89b0, 0x3d21ba98, 0x3d270048, 0x3d2c5ab0,
  0x3d31c9c0, 0x3d374d60, 0x3d3ce590, 0x3d429238, 0x3d485348, 0x3d4e28b0, 0x3d541268, 0x3d5a1050,
  0x3d602260, 0x3d664880, 0x3d6c82b0, 0x3d72d0c8, 0x3d7932c8, 0x3d7fa890, 0x3d83190c, 0x3d8667a8,
  0x3d89c00c, 0x3d8d2234, 0x3d908e14, 0x3d9403a4, 0x3d9782dc, 0x3d9b0bac, 0x3d9e9e0c, 0x3da239f8,
  0x3da5df60, 0x3da98e3c, 0x3dad4680, 0x3db10820, 0x3db4d314, 0x3db8a754, 0x3dbc84d0, 0x3dc06b80,
  0x3dc45b58, 0x3dc85450, 0x3dcc5658, 0x3dd06164, 0x3dd4756c, 0x3dd89268, 0x3ddcb844, 0x3de0e6fc,
  0x3de51e7c, 0x3de95ec0, 0x3deda7bc, 0x3df1f958, 0x3df65398, 0x3dfab664, 0x3dff21b8, 0x3e01cac0,
  0x3e0408dc, 0x3e064b24, 0x3e089196, 0x3e0adc2a, 0x3e0d2ada, 0x3e0f7da0, 0x3e11d476, 0x3e142f54,
  0x3e168e32, 0x3e18f10a, 0x3e1b57d8, 0x3e1dc294, 0x3e203136, 0x3e22a3b8, 0x3e251a12, 0x3e27943e,
  0x3e2a1236, 0x3e2c93f0, 0x3e2f1968, 0x3e31a296, 0x3e342f74, 0x3e36bff8, 0x3e39541c, 0x3e3bebd8,
  0x3e3e8726, 0x3e4125fe, 0x3e43c85a, 0x3e466e32, 0x3e49177e, 0x3e4bc436, 0x3e4e7452, 0x3e5127ca,
  0x3e53de9a, 0x3e5698b8, 0x3e59561c, 0x3e5c16c0, 0x3e5eda9a, 0x3e61a1a2, 0x3e646bd2, 0x3e673922,
  0x3e6a0988, 0x3e6cdd02, 0x3e6fb37c, 0x3e728cfa, 0x3e756970, 0x3e7848d6, 0x3e7b2b22, 0x3e7e104e,
  0x3e807c29, 0x3e81f192, 0x3e83685e, 0x3e84e08a, 0x3e865a10, 0x3e87d4ee, 0x3e89511e, 0x3e8ace9e,
  0x3e8c4d67, 0x3e8dcd74, 0x3e8f4ec6, 0x3e90d156, 0x3e92551e, 0x3e93da1c, 0x3e95604a, 0x3e96e7a6,
  0x3e98702a, 0x3e99f9d1, 0x3e9b8498, 0x3e9d107a, 0x3e9e9d74, 0x3ea02b80, 0x3ea1ba9a, 0x3ea34abb,
  0x3ea4dbe4, 0x3ea66e0e, 0x3ea80135, 0x3ea99554, 0x3eab2a66, 0x3eacc067, 0x3eae5753, 0x3eafef25,
  0x3eb187d9, 0x3eb3216a, 0x3eb4bbd4, 0x3eb65712, 0x3eb7f320, 0x3eb98ff8, 0x3ebb2d96, 0x3ebccbf8,
  0x3ebe6b16, 0x3ec00aee, 0x3ec1ab7a, 0x3ec34cb6, 0x3ec4ee9d, 0x3ec6912a, 0x3ec8345a, 0x3ec9d826,
  0x3ecb7c8b, 0x3ecd2185, 0x3ecec70d, 0x3ed06d21, 0x3ed213ba, 0x3ed3bad4, 0x3ed5626b, 0x3ed70a7b,
  0x3ed8b2fe, 0x3eda5bf1, 0x3edc054d, 0x3eddaf0f, 0x3edf5932, 0x3ee103b0, 0x3ee2ae86, 0x3ee459ae,
  0x3ee60525, 0x3ee7b0e5, 0x3ee95ce9, 0x3eeb092b, 0x3eecb5aa, 0x3eee6260, 0x3ef00f47, 0x3ef1bc5b,
  0x3ef36998, 0x3ef516f7, 0x3ef6c476, 0x3ef8720f, 0x3efa1fbd, 0x3efbcd7b, 0x3efd7b45, 0x3eff2917,
  0x3f006b75, 0x3f01425e, 0x3f021942, 0x3f02f021, 0x3f03c6f8, 0x3f049dc5, 0x3f057484, 0x3f064b34,
  0x3f0721d2, 0x3f07f85c, 0x3f08ced0, 0x3f09a52b, 0x3f0a7b6a, 0x3f0b518c, 0x3f0c278e, 0x3f0cfd6e,
  0x3f0dd32a, 0x3f0ea8bd, 0x3f0f7e28, 0x3f105367, 0x3f112878, 0x3f11fd59, 0x3f12d207, 0x3f13a680,
  0x3f147ac2, 0x3f154eca, 0x3f162296, 0x3f16f624, 0x3f17c970, 0x3f189c7a, 0x3f196f3e, 0x3f1a41ba,
  0x3f1b13ed, 0x3f1be5d3, 0x3f1cb76b, 0x3f1d88b1, 0x3f1e59a5, 0x3f1f2a43, 0x3f1ffa88, 0x3f20ca74,
  0x3f219a04, 0x3f226934, 0x3f233804, 0x3f240670, 0x3f24d478, 0x3f25a216, 0x3f266f4a, 0x3f273c13,
  0x3f28086d, 0x3f28d456, 0x3f299fcc, 0x3f2a6acd, 0x3f2b3556, 0x3f2bff65, 0x3f2cc8f8, 0x3f2d920e,
  0x3f2e5aa2, 0x3f2f22b4, 0x3f2fea41, 0x3f30b147, 0x3f3177c2, 0x3f323db4, 0x3f330317, 0x3f33c7eb,
  0x3f348c2c, 0x3f354fdb, 0x3f3612f2, 0x3f36d572, 0x3f379754, 0x3f38589c, 0x3f391944, 0x3f39d94c,
  0x3f3a98b1, 0x3f3b5770, 0x3f3c1588, 0x3f3cd2f8, 0x3f3d8fbb, 0x3f3e4bd1, 0x3f3f0737, 0x3f3fc1ec,
  0x3f407bec, 0x3f413538, 0x3f41edca, 0x3f42a5a4, 0x3f435cc2, 0x3f441321, 0x3f44c8c0, 0x3f457d9e,
  0x3f4631b8, 0x3f46e50c, 0x3f479798, 0x3f48495a, 0x3f48fa51, 0x3f49aa7a, 0x3f4a59d3, 0x3f4b085a,
  0x3f4bb60e, 0x3f4c62eb, 0x3f4d0ef2, 0x3f4dba20, 0x3f4e6473, 0x3f4f0de9, 0x3f4fb680, 0x3f505e36,
  0x3f51050a, 0x3f51aaf9, 0x3f525002, 0x3f52f423, 0x3f53975a, 0x3f5439a6, 0x3f54db04, 0x3f557b73,
  0x3f561af1, 0x3f56b97c, 0x3f575712, 0x3f57f3b3, 0x3f588f5c, 0x3f592a0a, 0x3f59c3be, 0x3f5a5c74,
  0x3f5af42c, 0x3f5b8ae3, 0x3f5c2098, 0x3f5cb54a, 0x3f5d48f6, 0x3f5ddb9b, 0x3f5e6d36, 0x3f5efdc8,
  0x3f5f8d4f, 0x3f601bc8, 0x3f60a932, 0x3f61358c, 0x3f61c0d4, 0x3f624b08, 0x3f62d428, 0x3f635c30,
  0x3f63e320, 0x3f6468f7, 0x3f64edb3, 0x3f657152, 0x3f65f3d4, 0x3f667535, 0x3f66f576, 0x3f677495,
  0x3f67f290, 0x3f686f66, 0x3f68eb16, 0x3f69659e, 0x3f69defc, 0x3f6a5730, 0x3f6ace39, 0x3f6b4414,
  0x3f6bb8c1, 0x3f6c2c3e, 0x3f6c9e8b, 0x3f6d0fa5, 0x3f6d7f8b, 0x3f6dee3c, 0x3f6e5bb9, 0x3f6ec7fe,
  0x3f6f330a, 0x3f6f9cde, 0x3f700576, 0x3f706cd4, 0x3f70d2f4, 0x3f7137d5, 0x3f719b78, 0x3f71fdda,
  0x3f725efb, 0x3f72beda, 0x3f731d75, 0x3f737acc, 0x3f73d6dc, 0x3f7431a7, 0x3f748b2a, 0x3f74e364,
  0x3f753a55, 0x3f758ffc, 0x3f75e456, 0x3f763765, 0x3f768926, 0x3f76d99a, 0x3f7728be, 0x3f777692,
  0x3f77c316, 0x3f780e46, 0x3f785826, 0x3f78a0b2, 0x3f78e7ea, 0x3f792dce, 0x3f79725b, 0x3f79b592,
  0x3f79f772, 0x3f7a37fa, 0x3f7a772a, 0x3f7ab501, 0x3f7af17e, 0x3f7b2ca0, 0x3f7b6666, 0x3f7b9ed2,
  0x3f7bd5e0, 0x3f7c0b91, 0x3f7c3fe4, 0x3f7c72da, 0x3f7ca470, 0x3f7cd4a8, 0x3f7d037e, 0x3f7d30f6,
  0x3f7d5d0c, 0x3f7d87c0, 0x3f7db112, 0x3f7dd902, 0x3f7dff90, 0x3f7e24ba, 0x3f7e4880, 0x3f7e6ae4,
  0x3f7e8be2, 0x3f7eab7b, 0x3f7ec9b0, 0x3f7ee67e, 0x3f7f01e8, 0x3f7f1beb, 0x3f7f3488, 0x3f7f4bbe,
  0x3f7f618e, 0x3f7f75f6, 0x3f7f88f7, 0x3f7f9a90, 0x3f7faac2, 0x3f7fb98c, 0x3f7fc6ee, 0x3f7fd2e8,
  0x3f7fdd79, 0x3f7fe6a2, 0x3f7fee62, 0x3f7ff4ba, 0x3f7ff9a8, 0x3f7ffd2e, 0x3f7fff4c, 0x3f800000,
]).buffer);

/** The pinned ramp of `n` samples, or null when no table covers that length. */
function edgeRampTable(n: number): Float32Array | null {
  if (n === EDGE_RAMP_24K_5MS.length) return EDGE_RAMP_24K_5MS;
  if (n === EDGE_RAMP_24K_20MS.length) return EDGE_RAMP_24K_20MS;
  return null;
}

/** Taper both waveform edges; see docs/design/postprocess.md. */
export function fadeEdges(
  audio: Float32Array, sampleRate: number, seconds: number = EDGE_FADE_SECONDS
): Float32Array {
  const n = Math.floor(seconds * sampleRate);
  if (n <= 0 || audio.length < 2 * n) return audio;
  const out = new Float32Array(audio);
  const last = out.length - 1;
  // A length no table covers is computed in double and narrowed. That ramp
  // tracks the float32 reference to within two units in the last place, under
  // 1.2e-07 at full scale, about -138 dBFS: the identity contract's equivalent
  // class, not its bit-exact one.
  const table = edgeRampTable(n);
  for (let i = 0; i < n; i++) {
    const w = table !== null ? table[i] : Math.fround(0.5 - 0.5 * Math.cos((Math.PI * i) / (n - 1)));
    out[i] = Math.fround(out[i] * w);
    out[last - i] = Math.fround(out[last - i] * w);
  }
  return out;
}
