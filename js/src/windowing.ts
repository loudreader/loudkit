/**
 * The renderer's geometry and randomness addressing — a port of
 * `loudkit.models.windowing`. Everything here is pure data geometry shared by
 * every backend; the window recipe in particular is the entire measured
 * ANE-vs-torch mel deviation when implementations disagree, so it is pinned by
 * the conformance fixture rather than re-derived.
 */

import { AlgorithmConfig, VoiceProfile } from "./types.js";

export const FLOW_NOISE_STREAM = 0;
export const VOCODER_PHASE_STREAM = 0;
export const VOCODER_NOISE_STREAM = 1;

export const START_TEXT_TOKEN = 255;
export const STOP_TEXT_TOKEN = 0;

const TOKEN_MEL_RATIO = 2;
const MEL_BINS = 80;

/** Minimum speech tokens before the stop token becomes sampleable. */
export function eosFloor(nTextTokens: number, config: AlgorithmConfig): number {
  const s = config.sampling;
  return Math.max(s.minTokensFloor, Math.floor(nTextTokens * s.minTokensTextRatio));
}

/** The Euler time grid: the explicit one if configured, else cosine. */
export function timeGrid(config: AlgorithmConfig): number[] {
  if (config.eulerGrid) return [...config.eulerGrid];
  const k = config.eulerSteps;
  const grid: number[] = [];
  for (let i = 0; i <= k; i++) {
    grid.push(1.0 - Math.cos((i / k) * Math.PI / 2.0));
  }
  return grid;
}

/** The token that fills unused static-window slots. */
export function padTokenId(config: AlgorithmConfig): number {
  if (config.window.padTokenId !== null) return config.window.padTokenId;
  if (config.sampling.silenceTokenIds.length) return config.sampling.silenceTokenIds[0];
  throw new Error(
    "static window needs a pad token: set window.padTokenId or provide silence tokens"
  );
}

export interface Framed {
  row: BigInt64Array; // (P+Q,) token row
  cond: Float32Array; // (1, 80, 2*(P+Q)) mel condition
  promptFrames: number;
  n: number;
}

/**
 * Apply the window recipe. In static mode the prompt is framed to exactly
 * `staticPromptTokens` and the query to `staticLength`, with the silence unit
 * padding — the production recipe.
 */
export function frameWindows(
  config: AlgorithmConfig,
  tokens: Iterable<number>,
  voice: VoiceProfile
): Framed {
  const w = config.window;
  const toks = Array.from(tokens);
  // Refused, not trimmed. Truncating at maxSpeechTokens hides the end of a
  // long passage behind audio that still sounds fine —
  // the only listener who notices is one who already knows the text. The
  // Python engine refuses this loudly and says how much speech
  // would be lost.
  if (toks.length > w.maxSpeechTokens) {
    throw new Error(
      `${toks.length} speech tokens exceed the ${w.maxSpeechTokens}-token window ` +
        `by ${toks.length - w.maxSpeechTokens}; split the text first`
    );
  }
  const n = toks.length;
  const promptTokens = Array.from(voice.promptTokens).map(Number);
  const promptMel = voice.promptMel;
  const promptFramesTarget = TOKEN_MEL_RATIO * (promptTokens.length);

  let prompt: number[];
  let query: number[];
  let condWidth: number;
  let promptFrames: number;

  if (w.staticLength !== null) {
    const pad = padTokenId(config);
    const pLen = w.staticPromptTokens ?? promptTokens.length;
    prompt = new Array<number>(pLen).fill(pad);
    for (let i = 0; i < Math.min(promptTokens.length, pLen); i++) prompt[i] = promptTokens[i];
    query = new Array<number>(w.staticLength).fill(pad);
    for (let i = 0; i < n; i++) query[i] = toks[i];
    condWidth = TOKEN_MEL_RATIO * (pLen + w.staticLength);
    promptFrames = TOKEN_MEL_RATIO * pLen;
  } else {
    prompt = promptTokens;
    query = toks;
    condWidth = TOKEN_MEL_RATIO * (promptTokens.length + n);
    promptFrames = promptFramesTarget;
  }

  const row = new BigInt64Array(prompt.length + query.length);
  for (let i = 0; i < prompt.length; i++) row[i] = BigInt(prompt[i]);
  for (let i = 0; i < query.length; i++) row[prompt.length + i] = BigInt(query[i]);

  // cond is [bin, time] = [80, 2*(P+Q)], laid out row-major; only the first
  // `keepF` frames of each bin carry the prompt mel, the rest stay zero.
  const promptMelFrames = promptMel.length / MEL_BINS;
  const cond = new Float32Array(condWidth * MEL_BINS);
  const keepF = Math.min(promptMelFrames, promptFrames);
  for (let b = 0; b < MEL_BINS; b++) {
    for (let f = 0; f < keepF; f++) {
      cond[b * condWidth + f] = promptMel[b * promptMelFrames + f];
    }
  }

  return { row, cond, promptFrames, n };
}

/** Row and cond split for the encoder graph: prompt | query, mel cond. */
export function frameForEncoder(
  framed: Framed,
  config: AlgorithmConfig
): { prompt: BigInt64Array; query: BigInt64Array; cond: Float32Array; promptFrames: number; n: number } {
  const pLen = config.window.staticPromptTokens ?? framed.row.length - framed.n;
  const prompt = framed.row.slice(0, pLen);
  const query = framed.row.slice(pLen);
  return {
    prompt,
    query,
    cond: framed.cond,
    promptFrames: framed.promptFrames,
    n: framed.n,
  };
}
