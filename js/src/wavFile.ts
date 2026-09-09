/**
 * The file half of `wav.ts`.
 *
 * Separate module, not a separate branch: `wav.ts` stays free of `node:fs`
 * because everything that reaches for it is here.
 */

import { readFileSync, writeFileSync } from "node:fs";

import { decodeWav, encodeWav, type Wave } from "./wav.js";

/** Read a WAV file as mono float32. */
export function readWav(path: string): Wave {
  return decodeWav(readFileSync(path));
}

/** Write `audio` to `path` as a mono 16-bit PCM WAV. */
export function saveWav(path: string, audio: Float32Array, sampleRate: number): void {
  writeFileSync(path, encodeWav(audio, sampleRate));
}

/** What `withWav` adds to a render. */
export interface WavOutput {
  /** This render as WAV bytes: mono, 16-bit PCM, at `sampleRate`. */
  toWav(): Uint8Array;
  /** This render written to `path` as a WAV. */
  saveWav(path: string): void;
}

/**
 * Give a render the two ways to leave the process.
 *
 * `result.audio` is a `Float32Array`, which is the right thing for a caller
 * who has an audio device and the wrong thing for the far more common caller
 * who wants a file. Hanging the encoder off the result rather than exporting a
 * free function means the sample rate cannot be passed wrong.
 */
export function withWav<T extends { audio: Float32Array; sampleRate: number }>(
  result: T
): T & WavOutput {
  return Object.assign(result, {
    toWav: () => encodeWav(result.audio, result.sampleRate),
    saveWav: (path: string) => saveWav(path, result.audio, result.sampleRate),
  });
}
