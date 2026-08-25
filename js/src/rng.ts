/**
 * Philox-4x32-10 — counter-based RNG, a bit-parity port of `loudkit.rng`.
 *
 * Every operation is a 32-bit multiply, xor or add on integers, so a correct
 * implementation in Python, Swift or TypeScript produces identical bits by
 * construction. The n-th random number is a pure function of
 * `(seed, stream, step, index)`, which is what lets every backend generate in
 * any order and still agree.
 *
 * Seeds are full 64-bit values (the engine derives per-stage seeds by mixing a
 * user seed with a stage constant, both 64-bit), which do not fit in a JS
 * `number` exactly — so every seed-taking function accepts `bigint` and splits
 * it into its two 32-bit halves up front. The counters (index, step, stream)
 * are < 2^32 and stay plain `number`s, masked after every op so no value above
 * 2^53 ever survives an operation.
 */

/** 32x32 -> (high, low), exact via 16-bit splitting.
 *
 * a*b = p2*2^32 + p1*2^16 + p0, where p0=aLo*bLo (<= 2^32),
 * p1=aLo*bHi+aHi*bLo (< 2^33), p2=aHi*bHi (< 2^32). Each partial is far
 * below 2^53, so float arithmetic is exact.
 */
function mulhilo(a: number, b: number): [number, number] {
  const aHi = Math.floor(a / 65536);
  const aLo = a % 65536;
  const bHi = Math.floor(b / 65536);
  const bLo = b % 65536;
  const p0 = aLo * bLo;
  const p1 = aLo * bHi + aHi * bLo;
  const p2 = aHi * bHi;
  // p0 lands in bits 0-31, p1's low 16 in bits 16-31, p1's high and p2 in the
  // high word, plus a possible carry out of the low-word sum. Multiply by
  // 65536, never `<< 16`: the shift operator works on signed 32-bit ints and
  // silently wraps to negative once bit 31 is set.
  const loSum = p0 + (p1 & 0xffff) * 65536;
  const lo = loSum & 0xffffffff;
  const carry = Math.floor(loSum / 4294967296);
  // Math.floor, not `>> 16`: p1 can exceed 2^31 and the shift is signed.
  const hi = (p2 + Math.floor(p1 / 65536) + carry) & 0xffffffff;
  return [hi >>> 0, lo >>> 0];
}

const M0 = 0xd2511f53;
const M1 = 0xcd9e8d57;
const W0 = 0x9e3779b9;
const W1 = 0xbb67ae85;
const ROUNDS = 10;
const MASK32 = 0xffffffff;

/** Largest seed the 64-bit counter can address, exclusive. */
const SEED_LIMIT = 1n << 64n;

/**
 * Normalise a caller-supplied seed to the 64-bit value the Philox counter uses.
 *
 * Accepts `bigint` as well as `number` because a JS `number` is a double: every
 * other binding takes a full 64-bit seed (`int` / `UInt64` / `uint64` / `u64`),
 * and any seed above 2^53 silently rounds on the way in. That would break the
 * library's one promise — same seed, same tokens on every binding — with no
 * error and no way for the caller to notice. So a `number` seed must be a safe
 * integer, and anything larger has to be passed as a `bigint`.
 *
 * @throws if the seed is not an integer, is negative, or does not fit in 64 bits.
 */
export function normalizeSeed(seed: number | bigint): bigint {
  if (typeof seed === "number") {
    if (!Number.isInteger(seed)) {
      throw new RangeError(`seed must be an integer, got ${seed}`);
    }
    if (!Number.isSafeInteger(seed)) {
      throw new RangeError(
        `seed ${seed} exceeds Number.MAX_SAFE_INTEGER and would lose precision; ` +
          "pass it as a bigint so this binding matches the others"
      );
    }
  }
  const value = BigInt(seed);
  if (value < 0n || value >= SEED_LIMIT) {
    throw new RangeError(`seed must be in [0, 2^64), got ${value}`);
  }
  return value;
}

/** Ten rounds of Philox-4x32 over one counter quad. */
export function philox4x32(
  c0: number,
  c1: number,
  c2: number,
  c3: number,
  k0: number,
  k1: number
): [number, number, number, number] {
  let x0 = c0 >>> 0;
  let x1 = c1 >>> 0;
  let x2 = c2 >>> 0;
  let x3 = c3 >>> 0;
  let key0 = k0 >>> 0;
  let key1 = k1 >>> 0;
  for (let r = 0; r < ROUNDS; r++) {
    const [hi0, lo0] = mulhilo(x0, M0);
    const [hi1, lo1] = mulhilo(x2, M1);
    x0 = ((hi1 ^ x1 ^ key0) & MASK32) >>> 0;
    x1 = lo1 >>> 0;
    x2 = ((hi0 ^ x3 ^ key1) & MASK32) >>> 0;
    x3 = lo0 >>> 0;
    key0 = ((key0 + W0) & MASK32) >>> 0;
    key1 = ((key1 + W1) & MASK32) >>> 0;
  }
  return [x0, x1, x2, x3];
}

/** Split a 64-bit seed into two 32-bit key halves, exactly. */
function seedKeys(seed: bigint): [number, number] {
  return [Number(seed & 0xffffffffn), Number(seed >> 32n)];
}

/**
 * `(n_steps, width)` uniforms in the open interval (0, 1).
 *
 * Open at both ends deliberately: the Gumbel transform takes two logarithms,
 * and a zero would produce an infinity that poisons an argmax.
 */
export function uniforms(
  seed: bigint,
  stream: number,
  step0: number,
  nSteps: number,
  width: number
): Float64Array {
  const out = new Float64Array(nSteps * width);
  const quads = Math.ceil(width / 4);
  const [k0, k1] = seedKeys(seed);
  const c2 = stream >>> 0;
  for (let s = 0; s < nSteps; s++) {
    const step = (s + step0) >>> 0;
    for (let q = 0; q < quads; q++) {
      const [r0, r1, r2, r3] = philox4x32(q, step, c2, 0, k0, k1);
      const vals = [r0, r1, r2, r3];
      for (let i = 0; i < 4; i++) {
        const idx = s * width + q * 4 + i;
        if (idx < out.length) {
          out[idx] = (vals[i] + 0.5) / 4294967296.0;
        }
      }
    }
  }
  return out;
}

/** `-log(-log(u))`, the additive form of a categorical draw. */
export function gumbelNoise(
  seed: bigint,
  stream: number,
  step0: number,
  nSteps: number,
  width: number
): Float64Array {
  const u = uniforms(seed, stream, step0, nSteps, width);
  const out = new Float64Array(u.length);
  for (let i = 0; i < u.length; i++) {
    out[i] = -Math.log(-Math.log(u[i]));
  }
  return out;
}
