/**
 * LR-SAMPLER-v1: one sampling law, specified tightly enough to reimplement.
 *
 * Bit-parity port of `loudkit.sampler.LRSamplerV1`. Three decisions make it
 * portable:
 *
 * - the RNG is counter-based Philox, so a token's randomness depends on
 *   `(seed, step, index)` alone, never on how many tokens came before;
 * - `min_p` is evaluated in **logit space** (`z/T >= max(z/T) + ln(min_p)`),
 *   identical to the probability form but with no softmax and therefore no
 *   order-dependent reduction;
 * - selection is **Gumbel-argmax**, an order-independent categorical draw with
 *   ties broken by lowest index.
 *
 * The conformance vectors pin this bit-for-bit; a port that disagrees on any
 * vector is not "close", it is broken.
 */

import { gumbelNoise as gumbelBlock, normalizeSeed } from "./rng.js";

export interface SamplingConfig {
  temperature: number;
  repetitionPenalty: number;
  minP: number;
  maxNewTokens: number;
  silenceTokenIds: number[];
  minTokensFloor: number;
  minTokensTextRatio: number;
}

const SAMPLING_STREAM = 0;

/**
 * Stateless with respect to *which* numbers it draws, since a token's
 * randomness is a pure function of `(seed, step)`, but caches a block of Gumbel
 * noise, because generating ten Philox rounds per token costs more than
 * running the entire model.
 */
export class LRSamplerV1 {
  readonly config: SamplingConfig;
  /** The 64-bit seed actually addressed, normalised from the constructor's. */
  readonly seed: bigint;
  private block: number;
  private noise: Float64Array | null = null;
  private base = 0;
  private silence: number[];

  /**
   * Observation of how close each step came to stopping. Never feeds back into
   * the draw; read by the postprocess detectors after generation. `null`
   * disables it, and with it its cost: one exponential and one sum over the
   * vocabulary per step.
   */
  private stopToken: number | null = null;
  private eosFloorStep = 0;
  private peakAt = -1;
  private peakProb = 0;

  constructor(config: SamplingConfig, seed: number | bigint, block = 256) {
    this.config = config;
    this.seed = normalizeSeed(seed);
    this.block = block;
    this.silence = config.silenceTokenIds;
  }

  /**
   * Enable the stop-token observation the postprocess layer reads.
   *
   * Done here, in the sampler, rather than by changing the generator: every
   * backend already calls the sampler on every step, and it owns the RNG
   * stream, so a backend that skipped it would produce different tokens, which
   * means
   * the observation reaches every generation path without a new seam.
   *
   * `eosFloor` is the floor this generation runs under. The peak is only
   * recorded past it, matching the shipped engine: below the floor the
   * generator masks the stop token, so its probability there describes the mask
   * rather than the model.
   */
  observeEos(stopToken: number, eosFloor: number): void {
    this.stopToken = stopToken;
    this.eosFloorStep = eosFloor;
    this.peakAt = -1;
    this.peakProb = 0;
  }

  /**
   * Where the model came closest to stopping, as `[step, probability]`.
   *
   * `[-1, 0]` when the stop token was never plausible, or when
   * {@link observeEos} was not called. **If the model never stops, that peak is
   * where the sentence really ended**, which is what makes it worth carrying.
   */
  get eosPeak(): [number, number] {
    return [this.peakAt, this.peakProb];
  }

  /**
   * Record how close this step came to stopping. Never changes the draw.
   *
   * The quantity is the shipped engine's, reproduced exactly: the stop token's
   * softmax weight over the sum of the weights that survived `min_p`. The
   * numerator is taken **before** the cutoff is applied, so a step where the
   * stop token was itself filtered out still reports how near it came: the
   * number answers "how close was this to being the end", not "what was the
   * chance of stopping", and the first question is the one the detectors need,
   * because the rows they exist to rescue are precisely the ones where stopping
   * never won.
   *
   * The floor is `>` and not `>=`: at exactly the floor step the generator has
   * only just unmasked the stop token, and the shipped engine records from the
   * step after.
   */
  private observe(
    s: Float64Array,
    maxS: number,
    threshold: number,
    silenceSet: Set<number>,
    step: number
  ): void {
    const stop = this.stopToken;
    if (stop === null || step <= this.eosFloorStep || stop >= s.length) return;
    const hasMinP = this.config.minP !== 0;
    let total = 0;
    for (let i = 0; i < s.length; i++) {
      if (!hasMinP || s[i] >= threshold || silenceSet.has(i)) total += Math.exp(s[i] - maxS);
    }
    if (total <= 0) return;
    const prob = Math.exp(s[stop] - maxS) / total;
    if (prob > this.peakProb) {
      this.peakProb = prob;
      this.peakAt = step;
    }
  }

  private noiseFor(step: number, width: number): Float64Array {
    let row = this.noise;
    if (
      row === null ||
      step < this.base ||
      step >= this.base + this.block ||
      row.length !== this.block * width
    ) {
      this.base = Math.floor(step / this.block) * this.block;
      row = gumbelBlock(this.seed, SAMPLING_STREAM, this.base, this.block, width);
      this.noise = row;
    }
    const start = (step - this.base) * width;
    return row.subarray(start, start + width);
  }

  /** Choose the next token from raw, unnormalised logits. */
  call(logits: Float32Array, step: number, seen: Uint8Array): number {
    const cfg = this.config;
    const n = logits.length;
    const z = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      z[i] = logits[i];
    }

    if (cfg.repetitionPenalty !== 1.0) {
      // The penalty applies to every seen token, silence included: exempt
      // from it, and re-admitted below the min_p floor by the exemption
      // further down, a silence run has no exit. Penalising seen silence is
      // what lets the decoder leave a pause it has entered. The min_p
      // exemption is a separate rule. The measurements are in
      // `docs/design/sampler-and-noise.md`.
      for (let i = 0; i < n; i++) {
        if (seen[i]) {
          z[i] = z[i] > 0 ? z[i] / cfg.repetitionPenalty : z[i] * cfg.repetitionPenalty;
        }
      }
    }

    const s = new Float64Array(n);
    let maxS = -Infinity;
    for (let i = 0; i < n; i++) {
      s[i] = z[i] / cfg.temperature;
      if (s[i] > maxS) maxS = s[i];
    }

    // min_p in logit space: keep i iff s[i] >= max(s) + ln(min_p).
    const threshold = cfg.minP > 0 ? maxS + Math.log(cfg.minP) : -Infinity;
    const silenceSet = cfg.minP > 0 ? new Set(this.silence) : new Set<number>();

    if (this.stopToken !== null) {
      // A separate set from the min_p one above, which is empty when min_p is
      // 0: silence exemption still applies to the sum, exactly as in Python.
      this.observe(s, maxS, threshold, new Set(this.silence), step);
    }

    const g = this.noiseFor(step, n);
    let best = -Infinity;
    let bestIdx = -1;
    for (let i = 0; i < n; i++) {
      // Silence stays available even when min_p would drop it: a pause token
      // is what makes a reader pause, and a filter that removes the only way
      // to pause is a filter that removes prosody. This is the one exemption
      // silence keeps: removing it was measured catastrophic (median
      // long-form gap 2.46 s -> 4.64 s). The repetition penalty above now
      // applies to silence like everything else, so a pause that overstays
      // decays instead of never ending.
      const keep = cfg.minP === 0 || s[i] >= threshold || silenceSet.has(i);
      if (!keep) continue;
      const v = s[i] + g[i];
      if (v > best) {
        best = v;
        bestIdx = i;
      }
    }
    // argmax already breaks ties toward the lowest index; a missing bestIdx
    // (all kept values -inf) falls back to 0 like the reference's argmax.
    return bestIdx === -1 ? 0 : bestIdx;
  }
}
