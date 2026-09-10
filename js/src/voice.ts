/**
 * Voice profile loading: a port of `loudkit.voice.VoiceProfile.load`.
 */

import { statSync } from "node:fs";
import { basename } from "node:path";

import { listVoices } from "./hub.js";
import { describeJson, pyRepr } from "./errors.js";
import { SafetensorsFile, writeSafetensors } from "./safetensors.js";
import { VoiceProfile } from "./types.js";

export const VOICE_FORMAT_VERSION = 1;

/** The constant fed to the generator's emotion conditioning slot.
 *
 * The checkpoint reserves one of its 34 conditioning slots for an emotion
 * scalar. On these weights the axis is dead (distillation collapsed it), so
 * the slot is not a control and not part of the profile format, but it must
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
 * finite but arbitrary direction. Enrolled vectors are order-1; anything
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
 * The strategies this build can honour, and the one a profile that names none
 * was made by.
 *
 * A profile whose header names a strategy this build does not implement was
 * cut from its recording a different way, so loading it here would speak in a
 * different voice under the same name. Refused rather than ignored.
 */
export const KNOWN_ENROLMENTS = ["first-10s", "first-10s-pause"] as const;
const ENROLMENT_FIRST_WINDOW = "first-10s";

/**
 * Longest `name` a profile's header carries, as the reference truncates it.
 *
 * The name is metadata a caller supplies at enrolment and a server echoes back;
 * nothing downstream shortens it, so a megabyte of it in a 165 KB file is legal
 * safetensors and pointless otherwise.
 */
const MAX_NAME_CHARS = 200;

/**
 * The shipped model's dimensions, the same two Python reads out of its
 * `AlgorithmConfig`.
 *
 * Both ends, not just the floor: without a ceiling `prompt_tokens = [9000]`
 * loads cleanly and then indexes past the end of the embedding table. The
 * ceilings are the shipped model's: prompt tokens index the speech codebook
 * below the start-of-speech marker, conditioning tokens the whole speech
 * vocabulary.
 */
const START_SPEECH_TOKEN = 6561n;
const SPEECH_VOCAB_SIZE = 8194n;

/**
 * Matches Python's `MAX_VOICE_BYTES`.
 *
 * A voice profile is a handful of small tensors, and a safetensors file
 * claiming otherwise is not one. The cap is on the file, before it is opened,
 * because the shape checks that follow only run once a header has been parsed.
 */
export const MAX_VOICE_BYTES = 8 * 1024 * 1024;

export function loadVoice(path: string): VoiceProfile {
  // The file's name, not the path it was found at, which is what the reference
  // and the other three ports put in front of a voice refusal.
  const file = basename(path);
  const size = statSync(path).size;
  if (size > MAX_VOICE_BYTES) {
    throw new Error(
      `${file}: ${size} bytes, over the ${MAX_VOICE_BYTES} byte ` +
        "limit for a voice: see MAX_VOICE_BYTES",
    );
  }
  const f = new SafetensorsFile(path);
  const headerRaw = f.metadata ? (f.metadata.voice as string | undefined) : undefined;
  const parsed: unknown = headerRaw ? JSON.parse(headerRaw) : {};
  // Checked to be an object, in the reference's sentence and Swift's. A header
  // holding a JSON list, number or string answered `undefined` for every key
  // and reported "voice format version 0", which blames a version the file
  // does not carry; one holding `null` reached a bare TypeError naming no
  // file at all. A profile is a thing that gets copied, mailed and
  // downloaded, so its refusals have to name what is wrong with the file.
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error(
      `${file}: voice header is ${describeJson(parsed)}, expected a JSON object`
    );
  }
  const header = parsed as Record<string, unknown>;
  // A field the header carries is read by its JSON type and refused by name
  // when it is another, rather than stringified: `language: 5` would become
  // the language id "5", and language selects the text funnel. An absent key
  // takes the default, which is the branch every profile written before a
  // field existed goes down.
  const text = (key: string, fallback: string): string => {
    const value = header[key];
    if (value === undefined) return fallback;
    if (typeof value !== "string") {
      throw new Error(`${file}: voice header '${key}' must be a string, got ${pyRepr(value)}`);
    }
    return value;
  };
  // A whole number, and nothing else. Both keys it reads are counts, and the
  // reference reads them through a check that a JSON number is whole:
  // `source_sample_rate: 48000.7` divides every duration a profile reports by
  // a rate no reader would agree on, and `format_version: 1.9` truncated to
  // the version this build implements, so a profile written by a build that
  // does something else loaded as one of ours.
  const number = (key: string, fallback: number): number => {
    const value = header[key];
    if (value === undefined) return fallback;
    if (typeof value !== "number") {
      throw new Error(`${file}: voice header '${key}' must be a number, got ${pyRepr(value)}`);
    }
    if (!Number.isInteger(value)) {
      throw new Error(
        `${file}: voice header '${key}' must be a whole number, got ${pyRepr(value)}`
      );
    }
    return value;
  };
  const version = number("format_version", 0);
  if (version !== VOICE_FORMAT_VERSION) {
    throw new Error(
      `${file}: voice format version ${version}, this build reads ${VOICE_FORMAT_VERSION}`
    );
  }
  const promptTokens = f.i64("prompt_tokens");
  const condTokens = f.i64("cond_prompt_tokens");
  const speakerEmbedding = f.f32("speaker_embedding");
  const flowEmbedding = f.f32("flow_embedding");
  const promptMel = f.f32("prompt_mel");

  // The declared shape, not the element count. `prompt_mel` declared [2, 80]
  // holds the same 160 floats as [80, 2] and means the transpose of them, so
  // a count alone loads the profile and reads the voice on its side. A vector
  // declared [16, 16] or [1, 256] is 256 floats too, and the reference
  // refuses all of these by rank.
  requireRank("speaker_embedding", f.shape("speaker_embedding"), 1);
  requireRank("flow_embedding", f.shape("flow_embedding"), 1);
  requireRank("prompt_tokens", f.shape("prompt_tokens"), 1);
  requireRank("cond_prompt_tokens", f.shape("cond_prompt_tokens"), 1);
  const melShape = f.shape("prompt_mel");
  if (melShape.length !== 2 || melShape[0] !== MEL_BINS) {
    throw new Error(
      `prompt_mel must be (${MEL_BINS}, frames), got (${melShape.join(", ")})`
    );
  }

  checkEmbedding("speaker_embedding", speakerEmbedding, SPEAKER_DIM);
  checkEmbedding("flow_embedding", flowEmbedding, FLOW_DIM);
  for (const v of promptMel) {
    if (!Number.isFinite(v)) throw new Error("prompt_mel contains NaN or infinity");
  }
  for (const [name, tokens, ceiling] of [
    ["prompt_tokens", promptTokens, START_SPEECH_TOKEN],
    ["cond_prompt_tokens", condTokens, SPEECH_VOCAB_SIZE],
  ] as const) {
    for (const t of tokens) {
      // Negative ids index an embedding table from the end, silently.
      if (t < 0n) throw new Error(`${name} contains a negative id: ${t}`);
      if (t >= ceiling) {
        throw new Error(`${name} contains id ${t}, at or past the ${ceiling} the model has`);
      }
    }
  }
  // A rate is metadata, and a negative one is not a slower recording: every
  // duration derived from a profile is samples over this number. Python and Go
  // refuse it; this port kept -48000 and Rust silently rewrote it to 24000.
  const sourceSampleRate = number("source_sample_rate", 24_000);
  if (sourceSampleRate <= 0) {
    throw new Error(`source_sample_rate must be positive: ${sourceSampleRate}`);
  }

  // basename, not split("/").pop(): a Windows path ("C:\\voices\\james.safetensors")
  // has no "/", so splitting on one names the voice after the whole path. The
  // extension argument strips exactly a trailing ".safetensors".
  const name = text("name", basename(path, ".safetensors")).slice(0, MAX_NAME_CHARS);

  // The law the prompt was cut under, checked rather than carried. A profile
  // naming a strategy this build does not implement was cut differently, so
  // loading it here speaks in a different voice under the same name. Absent
  // means the profile predates the field, and every one of those was the first
  // ten seconds.
  const enrolment = text("enrolment", ENROLMENT_FIRST_WINDOW);
  if (!(KNOWN_ENROLMENTS as readonly string[]).includes(enrolment)) {
    throw new Error(
      `${name || "voice"}: enrolment strategy ${pyRepr(enrolment)} is not ` +
        `one this build implements (${KNOWN_ENROLMENTS.join(", ")}). The profile was ` +
        "made by a build that cuts its prompt differently, so loading it here would " +
        "speak in a different voice under the same name."
    );
  }

  return {
    name,
    speakerEmbedding,
    flowEmbedding,
    promptTokens,
    promptMel,
    condPromptTokens: condTokens,
    sourceSampleRate,
    language: text("language", "en"),
    enrolment,
  };
}

/**
 * Refuse a tensor whose declared shape has the wrong rank.
 *
 * A rank the reference refuses is a file two implementations read as two
 * different voices, which is the divergence class a profile format exists to
 * prevent: a profile is a file that gets copied, mailed and downloaded.
 */
function requireRank(name: string, shape: number[], rank: number): void {
  if (shape.length !== rank) {
    throw new Error(
      `${name} must be ${rank}-D, got shape (${shape.join(", ")})`
    );
  }
}

/**
 * Write a profile as safetensors, with the header `loudkit.voice.VoiceProfile.save`
 * writes, so every implementation reads it back. Owner-only permissions: a
 * profile derives from a recording of a person.
 */
export function saveVoice(profile: VoiceProfile, path: string): void {
  if (profile.promptMel.length % MEL_BINS !== 0) {
    throw new Error(`prompt_mel must be (${MEL_BINS}, frames), got ${profile.promptMel.length} values`);
  }
  // Checked on the way out as well as on the way in. `VoiceProfile` is an
  // interface here, not the frozen dataclass the reference validates on
  // construction, so this is the only place a written profile can be stopped
  // from carrying a law no reader will accept.
  const enrolment = profile.enrolment ?? ENROLMENT_FIRST_WINDOW;
  if (!(KNOWN_ENROLMENTS as readonly string[]).includes(enrolment)) {
    throw new Error(
      `${profile.name || "voice"}: enrolment strategy ${JSON.stringify(enrolment)} is ` +
        `not one this build implements (${KNOWN_ENROLMENTS.join(", ")})`
    );
  }
  if (profile.sourceSampleRate <= 0) {
    throw new Error(`source_sample_rate must be positive: ${profile.sourceSampleRate}`);
  }
  const header = {
    format_version: VOICE_FORMAT_VERSION,
    name: profile.name,
    source_sample_rate: profile.sourceSampleRate,
    language: profile.language,
    enrolment,
  };
  writeSafetensors(
    path,
    [
      { name: "speaker_embedding", dtype: "F32", shape: [profile.speakerEmbedding.length], data: bytesOf(profile.speakerEmbedding) },
      { name: "flow_embedding", dtype: "F32", shape: [profile.flowEmbedding.length], data: bytesOf(profile.flowEmbedding) },
      { name: "prompt_tokens", dtype: "I64", shape: [profile.promptTokens.length], data: bytesOf(profile.promptTokens) },
      { name: "prompt_mel", dtype: "F32", shape: [MEL_BINS, profile.promptMel.length / MEL_BINS], data: bytesOf(profile.promptMel) },
      { name: "cond_prompt_tokens", dtype: "I64", shape: [profile.condPromptTokens.length], data: bytesOf(profile.condPromptTokens) },
    ],
    { voice: JSON.stringify(header) }
  );
}

function bytesOf(values: Float32Array | BigInt64Array): Uint8Array {
  // A copy, so a typed array over a larger buffer (a subarray) does not leak
  // its neighbours into the file.
  const copy = values.slice();
  return new Uint8Array(copy.buffer, copy.byteOffset, copy.byteLength);
}

export { listVoices };
