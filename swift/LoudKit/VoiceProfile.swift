import Foundation

/// A voice, as data, the same safetensors files `loudkit.voice.VoiceProfile`
/// writes, read without modification. A few hundred kilobytes, no weights.
public struct VoiceProfile: Sendable {
    /// What this voice is called. The file's stem when the header omits it.
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
    /// Frames in `promptMel`, since `promptMel` is flat. Two per prompt token.
    public let promptMelFrames: Int
    /// The token generator's own conditioning prompt.
    public let condPromptTokens: [Int]
    /// Language of the reference audio, and the language a synthesis reads
    /// as when the caller names none.
    public let language: String
    /// The recording's own sample rate, for provenance.
    public var sourceSampleRate: Int = 24_000
    /// How the conditioning prompt was prepared.
    public var enrolment: String = "first-10s"

    /// The prompt cuts this build implements, matching `KNOWN_ENROLMENTS`.
    ///
    /// A profile naming any other strategy is refused rather than cut this
    /// build's way: the prompt decides the timbre, so the same file would
    /// speak differently here than where it was made, under one name.
    static let knownEnrolments: Set<String> = ["first-10s", "first-10s-pause"]

    /// The constant fed to the generator's emotion conditioning slot.
    ///
    /// The checkpoint reserves one of its 34 conditioning slots for an emotion
    /// scalar. On these weights the axis is dead (distillation collapsed it),
    /// so the slot is not a control and not part of the profile format, but
    /// it must be fed the value the model was distilled with. Every port uses
    /// this constant.
    public static let emotionNeutral: Float = 0.5

    /// The shipped model's dimensions, the same two Python reads out of
    /// its `AlgorithmConfig`.
    static let startSpeechToken = 6561
    static let speechVocabSize = 8194

    /// Largest file this reader will open, matching `MAX_VOICE_BYTES`. A voice
    /// profile is a handful of small tensors, and a safetensors file claiming
    /// otherwise is not one. The cap is on the file, before it is opened,
    /// because the shape checks that follow only run once a header has been
    /// parsed.
    public static let maxVoiceBytes = 8 * 1024 * 1024

    /// A JSON string under `key`, or `fallback` when the key is absent.
    ///
    /// A present value of another type is refused, not defaulted: a header
    /// value of the wrong type was written by something that did not mean it,
    /// and language selects the text funnel, so defaulting reads the voice
    /// through a funnel nobody chose. `VoiceProfile._header_str` refuses the
    /// same shape in the same sentence.
    ///
    /// - Throws: `LoudKitError.asset` when the key holds anything but a string.
    private static func headerString(
        _ header: [String: Any], _ file: String, _ key: String, _ fallback: String
    ) throws -> String {
        guard let raw = header[key] else { return fallback }
        guard let value = raw as? String else {
            throw LoudKitError.asset(
                "\(file): voice header '\(key)' must be a string, "
                + "got \(ManifestReader.pyRepr(raw))")
        }
        return value
    }

    /// A whole JSON number under `key`, or `fallback` when the key is absent.
    ///
    /// `as? NSNumber` is true of a JSON boolean, and its `intValue` is 1, so a
    /// header saying `format_version: true` read as a version-1 voice.
    /// `ManifestReader.asNumber` is the one reading that tells the two apart.
    ///
    /// - Throws: `LoudKitError.asset` when the key holds anything but a whole
    ///   number.
    private static func headerInt(
        _ header: [String: Any], _ file: String, _ key: String, _ fallback: Int
    ) throws -> Int {
        guard let raw = header[key] else { return fallback }
        guard let value = ManifestReader.asNumber(raw) else {
            throw LoudKitError.asset(
                "\(file): voice header '\(key)' must be a number, "
                + "got \(ManifestReader.pyRepr(raw))")
        }
        // A magnitude no `Int` can hold is refused rather than converted: the
        // conversion traps, and a trap is a crash report where the header key
        // that caused it does not appear.
        guard value.isFinite, value.magnitude < 9.007199254740992e15,
              value.rounded(.towardZero) == value else {
            throw LoudKitError.asset(
                "\(file): voice header '\(key)' must be a whole number, "
                + "got \(ManifestReader.pyRepr(raw))")
        }
        return Int(value)
    }

    /// Read a profile written by this package or by `loudkit.voice`.
    ///
    /// - Throws: `LoudKitError.asset` when the file is larger than
    ///   `maxVoiceBytes`, is not a safetensors store, or is missing a tensor;
    ///   `LoudKitError.shape` when a tensor is the wrong size for its role.
    public static func load(url: URL) throws -> VoiceProfile {
        if let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize,
           size > maxVoiceBytes {
            throw LoudKitError.asset(
                "\(url.lastPathComponent): \(size) bytes, over the "
                    + "\(maxVoiceBytes) byte limit for a voice")
        }
        let store = try Safetensors(url: url)
        let file = url.lastPathComponent
        // The version guard sits OUTSIDE the header read, and that is the whole
        // point of the shape below.
        //
        // A guard inside `if let header = ...` lets a profile with no `voice`
        // metadata at all, or with `voice` holding a JSON list or string, skip
        // the block entirely and load with every default: name from the
        // filename, English, 24 kHz, first-10s. The reference reads
        // `json.loads(meta.get("voice", "{}"))` and then
        // `int(header.get("format_version", 0))`, so an absent header is
        // version 0 and refused, and a non-object header is refused by name.
        // Go, Rust and JS all refuse both.
        let headerJSON = store.metadata["voice"] ?? "{}"
        let raw = try ManifestReader.json(Data(headerJSON.utf8), "\(file): bad voice header JSON")
        guard let header = raw as? [String: Any] else {
            throw LoudKitError.asset(
                "\(file): voice header is "
                + "\(ManifestReader.jsonTypePhrase(raw)), expected a JSON object")
        }
        let version = try Self.headerInt(header, file, "format_version", 0)
        guard version == 1 else {
            throw LoudKitError.asset(
                "\(file): voice format version \(version), this build reads 1")
        }
        let name = try Self.headerString(
            header, file, "name", url.deletingPathExtension().lastPathComponent)
        // An "emotion" key in older profiles is ignored: the axis is dead
        // on these weights and the slot is fed `emotionNeutral`.
        let language = try Self.headerString(header, file, "language", "en")
        let enrolment = try Self.headerString(header, file, "enrolment", "first-10s")
        let sourceSampleRate = try Self.headerInt(header, file, "source_sample_rate", 24_000)
        // How the prompt was cut is not provenance: a profile made by a build
        // that cuts differently would speak in a different voice under the
        // name the caller asked for. `VoiceProfile._validate_enrolment` and
        // `go/voice` both refuse it in these words.
        guard Self.knownEnrolments.contains(enrolment) else {
            throw LoudKitError.asset(
                "\(name.isEmpty ? "voice" : name): enrolment strategy "
                + "\(ManifestReader.pyRepr(enrolment)) is "
                + "not one this build implements "
                + "(\(Self.knownEnrolments.sorted().joined(separator: ", "))). "
                + "The profile was made by a build that cuts its prompt differently, "
                + "so loading it here would speak in a different voice under the same "
                + "name.")
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
        // the shipped model's, prompt tokens index the speech codebook below the
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
            language: language,
            sourceSampleRate: sourceSampleRate,
            enrolment: enrolment)
    }

    /// Write the profile as safetensors, with the header
    /// `loudkit.voice.VoiceProfile.save` writes, so every implementation
    /// reads it back. Owner-only permissions: a profile derives from a
    /// recording of a person.
    public func save(to url: URL) throws {
        guard promptMel.count == 80 * promptMelFrames else {
            throw LoudKitError.shape(
                "prompt_mel must be (80, \(promptMelFrames)), got \(promptMel.count) values")
        }
        let header: [String: Any] = [
            "format_version": 1,
            "name": name,
            "source_sample_rate": sourceSampleRate,
            "language": language,
            "enrolment": enrolment
        ]
        let headerData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        guard let headerJSON = String(bytes: headerData, encoding: .utf8) else {
            throw LoudKitError.shape("\(name): the voice header is not UTF-8")
        }
        func floats(_ values: [Float]) -> Data {
            values.withUnsafeBufferPointer { Data(buffer: $0) }
        }
        func ints(_ values: [Int]) -> Data {
            values.map { Int64($0).littleEndian }.withUnsafeBufferPointer { Data(buffer: $0) }
        }
        try Safetensors.write(
            [
                .init(name: "speaker_embedding", dtype: "F32", shape: [speakerEmbedding.count],
                      data: floats(speakerEmbedding)),
                .init(name: "flow_embedding", dtype: "F32", shape: [flowEmbedding.count],
                      data: floats(flowEmbedding)),
                .init(name: "prompt_tokens", dtype: "I64", shape: [promptTokens.count],
                      data: ints(promptTokens)),
                .init(name: "prompt_mel", dtype: "F32", shape: [80, promptMelFrames],
                      data: floats(promptMel)),
                .init(name: "cond_prompt_tokens", dtype: "I64", shape: [condPromptTokens.count],
                      data: ints(condPromptTokens))
            ],
            metadata: ["voice": headerJSON], to: url)
    }

    /// `save(to:)` with a path.
    public func save(_ path: String) throws {
        try save(to: URL(fileURLWithPath: path))
    }

    /// Smallest speaker-vector norm a profile may carry.
    ///
    /// Below this the renderers stop agreeing: this one and ONNX divide by the
    /// raw norm and yield NaN, torch's `F.normalize` carries an epsilon and
    /// yields a finite but arbitrary direction, the same file speaking
    /// differently per backend. Enrolled vectors are order-1; anything this
    /// small is a corrupt or synthetic file, not a quiet voice.
    static let minEmbeddingNorm: Float = 1e-6

    /// Refuse a vector no renderer can use: the width its role fixes, finite
    /// values, and a norm at or above `minEmbeddingNorm`.
    ///
    /// This module's own documentation calls profiles safe to load from an
    /// untrusted source, so every field a renderer indexes or divides by is
    /// checked at the file: these three invariants here, and in `load` the
    /// mel's shape and finiteness and the prompt token range. `Renderer`
    /// divides by the speaker norm and indexes `spkWeight[r * k + c]` with
    /// `k = emb.count`, so a zero norm reaches it as NaN and a wrong-width
    /// embedding runs past the array and traps. The other four ports enforce
    /// the same invariants on the same file.
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
