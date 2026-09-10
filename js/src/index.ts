/**
 * loudkit: text to speech over onnxruntime-node, the same engine as the
 * Python package, same sampling law, no torch.
 */

export { Engine, CancelledError } from "./engine.js";
export type { SynthesisOptions, StreamChunk } from "./engine.js";
// The refusals this library makes on purpose, each carrying the catalog `code`
// every port and transport sends. Additive: they extend the built-ins these
// sites already threw, so a `catch` written against `Error` or `RangeError`
// still catches them and reads the same message.
export {
  LoudkitError,
  LoudkitRangeError,
  InvalidTokensError,
  NothingToSpeakError,
  NumberGrammarError,
  UnsupportedLanguageError,
  VoiceNotFoundError,
  WindowOverflowError,
  errorCode,
} from "./errors.js";
// The execution provider is a public knob and its vocabulary is the same five
// words in every port, so both the type and the list are exported.
export { ONNX_PROVIDERS, availableProviders } from "./execution.js";
export type { ExecutionOptions, ONNXProvider, ResolvedONNXProvider } from "./execution.js";
export { LRSamplerV1 } from "./sampler.js";
export { listVoices, loadVoice, saveVoice } from "./voice.js";
export { download } from "./hub.js";
export type { DownloadOptions } from "./hub.js";
// WAV both ways, with no file in sight: these two are pure and import nothing
// from Node, so they can move to a browser bundle when the package exports a
// subpath for them. `result.saveWav(path)` writes a render and `engine.enroll`
// reads a WAV path, so neither needs a free function.
export { decodeWav, encodeWav } from "./wav.js";
export type { Wave } from "./wav.js";
export type { WavOutput } from "./wavFile.js";
export type { SamplingConfig, WindowConfig, AlgorithmConfig, VoiceProfile } from "./types.js";
export { algorithmFromManifest, productionWindow } from "./types.js";
export { canonicalForm, fingerprint, FINGERPRINT_SCHEMA } from "./fingerprint.js";
export { philox4x32, uniforms, gumbelNoise } from "./rng.js";
export { gaussianField, symmetricUniforms } from "./noise.js";
export { timeGrid, eosFloor, frameWindows } from "./windowing.js";
export { splitText, estimateTokens, CHARS_PER_TOKEN } from "./chunking.js";
export { speechText } from "./speechText.js";
export { lexicalRespelling } from "./respell.js";
export { Enroller, profileFrom, resample, validateReferenceAudio } from "./enroll.js";
export type { Enrolled } from "./enroll.js";
// `ChunkTiming` and `WordTiming` appear on what `synthesize` returns; the two
// speed bounds are for a UI drawing a slider.
export type { ChunkTiming, WordTiming } from "./timing.js";
export { MIN_SPEED, MAX_SPEED } from "./timestretch.js";
