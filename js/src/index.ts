/**
 * loudkit — text to speech over onnxruntime-node, the same engine as the
 * Python package, same sampling law, no torch.
 */

export { Engine } from "./engine.js";
export type { SynthesisOptions } from "./engine.js";
// The execution provider is a public knob and its vocabulary is the same five
// words in every port, so both the type and the list are exported: a caller
// building a `--provider` flag needs the list to validate against, and
// re-deriving it from a string union is not something JS callers can do.
export { ONNX_PROVIDERS, availableProviders } from "./execution.js";
export type { ExecutionOptions, ONNXProvider, ResolvedONNXProvider } from "./execution.js";
export { LRSamplerV1 } from "./sampler.js";
export { loadVoice } from "./voice.js";
export type { SamplingConfig, WindowConfig, AlgorithmConfig, VoiceProfile } from "./types.js";
export { algorithmFromManifest, productionWindow } from "./types.js";
export { canonicalForm, fingerprint, FINGERPRINT_SCHEMA } from "./fingerprint.js";
export { philox4x32, uniforms, gumbelNoise } from "./rng.js";
export { gaussianField, symmetricUniforms } from "./noise.js";
export { timeGrid, eosFloor, frameWindows } from "./windowing.js";
export { splitText, estimateTokens, CHARS_PER_TOKEN } from "./chunking.js";
export { speechText } from "./speechText.js";
export { lexicalRespelling } from "./respell.js";
export { Enroller, resample } from "./enroll.js";
export type { Enrolled } from "./enroll.js";
// The public surface here matches Python's, deliberately: `ChunkTiming` and
// `WordTiming` because they appear on what `synthesize` returns and a consumer
// must be able to name them, and the two speed bounds because a UI drawing a
// speed slider needs them. `timeline`, `estimateWords`, `timeStretch`,
// `stretchedLength`, `validateSpeed`, `ChunkSpan` and `carryFrom` are how the
// engine renders rather than things a caller composes with, and Python keeps
// its equivalents in `loudkit.timing` / `loudkit.models.timestretch` /
// `Engine._carry_from`. Five implementations of one API means the same answer
// to "what may I import", not five. The tests reach them by module path, which
// is what a port's own tests are allowed to do.
export type { ChunkTiming, WordTiming } from "./timing.js";
export { MIN_SPEED, MAX_SPEED } from "./timestretch.js";
