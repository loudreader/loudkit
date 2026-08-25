/**
 * Render randomness as data — a port of `loudkit.models.noise`.
 *
 * The flow prior and the vocoder excitation are *inputs* that happen to be
 * random. They come from the Philox counter, so the same seed produces the
 * same bytes on every backend, and a cross-backend render comparison measures
 * arithmetic rather than RNG plumbing.
 */

import { uniforms } from "./rng.js";

export function gaussianField(
  seed: bigint,
  stream: number,
  rows: number,
  cols: number
): Float32Array {
  const u1 = uniforms(seed, stream, 0, rows, cols);
  const u2 = uniforms(seed, stream + 1, 0, rows, cols);
  const out = new Float32Array(rows * cols);
  for (let i = 0; i < rows * cols; i++) {
    out[i] = Math.sqrt(-2.0 * Math.log(u1[i])) * Math.cos(2.0 * Math.PI * u2[i]);
  }
  return out;
}

export function symmetricUniforms(
  seed: bigint,
  stream: number,
  n: number,
  halfWidth: number
): Float32Array {
  const u = uniforms(seed, stream, 0, 1, n);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = (u[i] * 2.0 - 1.0) * halfWidth;
  }
  return out;
}
