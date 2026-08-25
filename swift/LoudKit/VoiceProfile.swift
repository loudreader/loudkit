import Foundation

/// A voice, as data — the same safetensors files `loudkit.voice.VoiceProfile`
/// writes, read without modification. A few hundred kilobytes, no weights.
public struct VoiceProfile {
    public let name: String
    /// 256-d utterance vector for the token generator's conditioning encoder.
    public let speakerEmbedding: [Float]
    /// 192-d x-vector for the mel decoder. Not interchangeable with the above.
    public let flowEmbedding: [Float]
    /// Reference speech tokens the mel decoder continues from (natural length;
    /// the window recipe decides framing).
    public let promptTokens: [Int]
    /// `(80, frames)` mel of the reference, row-major.
    public let promptMel: [Float]
    public let promptMelFrames: Int
    /// The token generator's own conditioning prompt.
    public let condPromptTokens: [Int]
    /// Language of the reference audio, for provenance — and, since the engine
    /// consults it, the language a synthesis reads as when the caller names
    /// none. Written by `loudkit.voice.VoiceProfile.save` into the same header
    /// as `name`; dropping the key makes a Polish voice read Polish text as
    /// English.
    public let language: String

    /// The constant fed to the generator's emotion conditioning slot.
    ///
    /// The checkpoint reserves one of its 34 conditioning slots for an emotion
    /// scalar. On these weights the axis is dead (distillation collapsed it),
    /// so the slot is not a control and not part of the profile format — but
    /// it must be fed the value the model was distilled with. Every port uses
    /// this constant.
    public static let emotionNeutral: Float = 0.5

    /// Matches Python's `MAX_VOICE_BYTES`, which the other four readers never
    /// had. A voice profile is a handful of small tensors, and a safetensors
    /// file claiming otherwise is not one. The cap is on the file, before it is
    /// opened, because the shape checks that follow only run once a header has
    /// been parsed.
    /// The shipped model's dimensions, the same two Python reads out of
    /// its `AlgorithmConfig`.
    static let startSpeechToken = 6561
    static let speechVocabSize = 8194

    public static let maxVoiceBytes = 8 * 1024 * 1024

    public static func load(url: URL) throws -> VoiceProfile {
        if let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize,
           size > maxVoiceBytes {
            throw LoudKitError.asset(
                "\(url.lastPathComponent): \(size) bytes, over the "
                    + "\(maxVoiceBytes) byte limit for a voice")
        }
        let store = try Safetensors(url: url)
        var name = url.deletingPathExtension().lastPathComponent
        var language = "en"
        if let headerJSON = store.metadata["voice"],
           let header = try JSONSerialization.jsonObject(
            with: Data(headerJSON.utf8)) as? [String: Any] {
            let version = (header["format_version"] as? NSNumber)?.intValue ?? 0
            guard version == 1 else {
                throw LoudKitError.asset("\(url.lastPathComponent): voice format version \(version), this build reads 1")
            }
            name = header["name"] as? String ?? name
            // Profiles written before 0.1 also carry an "emotion" key; the
            // axis is dead on these weights, so the key is ignored and the
            // conditioning slot is fed `VoiceProfile.emotionNeutral`.
            language = header["language"] as? String ?? language
        }
        let melShape = try store.shape("prompt_mel")
        guard melShape.count == 2, melShape[0] == 80 else {
            throw LoudKitError.shape("prompt_mel must be (80, frames), got \(melShape)")
        }

        let speaker = try store.floats("speaker_embedding")
        let flow = try store.floats("flow_embedding")
        let promptTokens = try store.ints("prompt_tokens")
        let promptMel = try store.floats("prompt_mel")
        let condPromptTokens = try store.ints("cond_prompt_tokens")

        try check(speaker, name: "speaker_embedding", expected: 256)
        try check(flow, name: "flow_embedding", expected: 192)
        guard promptMel.allSatisfy({ $0.isFinite }) else {
            throw LoudKitError.shape("prompt_mel contains NaN or infinity")
        }
        // Negative ids index an embedding table from the end in torch and are
        // an out-of-bounds read everywhere else, so they are refused at the
        // file rather than diagnosed at a matrix.
        if let bad = (promptTokens + condPromptTokens).first(where: { $0 < 0 }) {
            throw LoudKitError.shape("prompt token ids must be non-negative, got \(bad)")
        }
        // Both ends, not just the floor: without the ceiling `prompt_tokens = [9000]`
        // loads cleanly and then indexes past the end of the embedding table. The ceilings are
        // the shipped model's — prompt tokens index the speech codebook below the
        // start-of-speech marker, conditioning tokens the whole speech vocabulary.
        for (name, tokens, ceiling) in [
            ("prompt_tokens", promptTokens, startSpeechToken),
            ("cond_prompt_tokens", condPromptTokens, speechVocabSize),
        ] {
            if let bad = tokens.first(where: { $0 >= ceiling }) {
                throw LoudKitError.shape(
                    "\(name) contains id \(bad), at or past the \(ceiling) the model has")
            }
        }
        return VoiceProfile(
            name: name,
            speakerEmbedding: speaker,
            flowEmbedding: flow,
            promptTokens: promptTokens,
            promptMel: promptMel,
            promptMelFrames: melShape[1],
            condPromptTokens: condPromptTokens,
            language: language)
    }

    /// Smallest speaker-vector norm a profile may carry.
    ///
    /// Below this the renderers stop agreeing: this one and ONNX divide by the
    /// raw norm and yield NaN, torch's `F.normalize` carries an epsilon and
    /// yields a finite but arbitrary direction — the same file speaking
    /// differently per backend. Enrolled vectors are order-1; anything this
    /// small is a corrupt or synthetic file, not a quiet voice.
    static let minEmbeddingNorm: Float = 1e-6

    /// The check Python, Rust, Go and JS have all had since the
    /// degenerate-profile fix, and Swift had not.
    ///
    /// This module's own docstring calls profiles safe to load from an
    /// untrusted source; it checked `format_version` and the mel's row count
    /// and nothing else. A hand-built profile with zero 8-d embeddings,
    /// negative prompt ids and a NaN mel loaded cleanly here and was refused by
    /// all four other ports — and downstream `Renderer` divides by that zero
    /// norm and indexes `spkWeight[r * k + c]` with `k = emb.count`, which for
    /// a wrong-width embedding runs past the array and traps.
    private static func check(_ values: [Float], name: String, expected: Int) throws {
        guard values.count == expected else {
            throw LoudKitError.shape("\(name) must be \(expected)-d, got \(values.count)")
        }
        guard values.allSatisfy({ $0.isFinite }) else {
            throw LoudKitError.shape("\(name) contains NaN or infinity")
        }
        let norm = values.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
        guard norm >= minEmbeddingNorm else {
            throw LoudKitError.shape(
                "\(name) has norm \(norm), below \(minEmbeddingNorm): a zero or near-zero "
                    + "speaker vector normalises to NaN here and to a finite arbitrary "
                    + "direction on torch, so the same file would speak differently per backend")
        }
    }
}
