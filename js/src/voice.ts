/**
 * Voice profile loading — a port of `loudkit.voice.VoiceProfile.load`.
 */

import { statSync } from "node:fs";
import { basename } from "node:path";

import { SafetensorsFile } from "./safetensors.js";
import { VoiceProfile } from "./types.js";

export const VOICE_FORMAT_VERSION = 1;

/** The constant fed to the generator's emotion conditioning slot.
 *
 * The checkpoint reserves one of its 34 conditioning slots for an emotion
 * scalar. On these weights the axis is dead (distillation collapsed it), so
 * the slot is not a control and not part of the profile format — but it must
 * be fed the value the model was distilled with. Every port uses this. */
export const EMOTION_NEUTRAL = 0.5;

/** The two speaker encoders' widths and the mel bin count, as Python validates them. */
const SPEAKER_DIM = 256;
const FLOW_DIM = 192;
const MEL_BINS = 80;

/**
 * Smallest speaker-vector norm a profile may carry.
 *
 * Below this the renderers stop agreeing: this port and CoreML divide by the
 * raw norm and yield NaN, torch's `F.normalize` carries an epsilon and yields a
 * finite — but arbitrary — direction. Enrolled vectors are order-1; anything
 * this small is a corrupt or synthetic file, not a quiet voice.
 */
const MIN_EMBEDDING_NORM = 1e-6;

/**
 * Reject an embedding the renderers would disagree about.
 *
 * A profile is a file that gets copied, mailed and downloaded, so these checks
 * belong at the boundary rather than in each backend. Python has validated them
 * since the degenerate-profile fix; the ports accepted anything shaped like
 * floats and blew up deeper in inference, where the error names a matrix rather
 * than a file.
 */
function checkEmbedding(name: string, values: Float32Array, expected: number): void {
  if (values.length !== expected) {
    throw new Error(`${name} must be ${expected}-d, got ${values.length}`);
  }
  let sum = 0;
  for (const v of values) {
    if (!Number.isFinite(v)) throw new Error(`${name} contains NaN or infinity`);
    sum += v * v;
  }
  const norm = Math.sqrt(sum);
  if (norm < MIN_EMBEDDING_NORM) {
    throw new Error(
      `${name} has norm ${norm}, below ${MIN_EMBEDDING_NORM}: a zero or near-zero speaker ` +
        `vector normalises to NaN here and to a finite arbitrary direction on torch, so the ` +
        `same file would speak differently per backend`
    );
  }
}

/**
 * The shipped model's dimensions, the same two Python reads out of its
 * `AlgorithmConfig`.
 *
 * Both ends, not just the floor: Python has bounded these since its loader was
 * written and the other four never did, so `prompt_tokens = [9000]` loaded
 * cleanly and then indexed past the end of the embedding table. The ceilings are
 * the shipped model's — prompt tokens index the speech codebook below the
 * start-of-speech marker, conditioning tokens the whole speech vocabulary.
 */
const START_SPEECH_TOKEN = 6561n;
const SPEECH_VOCAB_SIZE = 8194n;

/**
 * Matches Python's `MAX_VOICE_BYTES`, which the other four readers never had.
 *
 * A voice profile is a handful of small tensors, and a safetensors file
 * claiming otherwise is not one. The cap is on the file, before it is opened,
 * because the shape checks that follow only run once a header has been parsed.
 */
export const MAX_VOICE_BYTES = 8 * 1024 * 1024;

export function loadVoice(path: string): VoiceProfile {
  const size = statSync(path).size;
  if (size > MAX_VOICE_BYTES) {
    throw new Error(
      `${path}: ${size} bytes, over the ${MAX_VOICE_BYTES} byte limit for a voice`,
    );
  }
  const f = new SafetensorsFile(path);
  const headerRaw = f.metadata ? (f.metadata.voice as string | undefined) : undefined;
  const header = headerRaw ? JSON.parse(headerRaw) : {};
  const version = header.format_version ?? 0;
  if (version !== VOICE_FORMAT_VERSION) {
    throw new Error(
      `${path}: voice format version ${version}, this build reads ${VOICE_FORMAT_VERSION}`
    );
  }
  const promptTokens = f.i64("prompt_tokens");
  const condTokens = f.i64("cond_prompt_tokens");
  const speakerEmbedding = f.f32("speaker_embedding");
  const flowEmbedding = f.f32("flow_embedding");
  const promptMel = f.f32("prompt_mel");

  checkEmbedding("speaker_embedding", speakerEmbedding, SPEAKER_DIM);
  checkEmbedding("flow_embedding", flowEmbedding, FLOW_DIM);
  for (const v of promptMel) {
    if (!Number.isFinite(v)) throw new Error("prompt_mel contains NaN or infinity");
  }
  if (promptMel.length % MEL_BINS !== 0) {
    throw new Error(`prompt_mel must be (${MEL_BINS}, frames), got ${promptMel.length} values`);
  }
  for (const [name, tokens, ceiling] of [
    ["prompt_tokens", promptTokens, START_SPEECH_TOKEN],
    ["cond_prompt_tokens", condTokens, SPEECH_VOCAB_SIZE],
  ] as const) {
    for (const t of tokens) {
      // Negative ids index an embedding table from the end — silently.
      if (t < 0n) throw new Error(`${name} contains a negative id: ${t}`);
      if (t >= ceiling) {
        throw new Error(`${name} contains id ${t}, at or past the ${ceiling} the model has`);
      }
    }
  }
  return {
    // basename, not split("/").pop(): a Windows path ("C:\\voices\\james.safetensors")
    // has no "/" and the old form returned the whole path as the name. The
    // extension argument strips exactly a trailing ".safetensors".
    name: header.name ?? (basename(path, ".safetensors") || "voice"),
    speakerEmbedding,
    flowEmbedding,
    promptTokens,
    promptMel,
    condPromptTokens: condTokens,
    sourceSampleRate: header.source_sample_rate ?? 24_000,
    language: header.language ?? "en",
  };
}
