import CryptoKit
import Foundation
import LoudKitText

/// What the engine computes and how fast it gets there, kept apart exactly
/// as in `loudkit/config.py`. `AlgorithmConfig` is identical on every backend
/// *and in every language*: the fingerprint here is computed over the same
/// canonical JSON the Python side hashes, so a Swift engine and a Python
/// engine can refuse to be compared before they disagree about what they are
/// computing, which is how the founding guidance defect (mel corr 0.979
/// between two "identical" pipelines) becomes a loud error instead of a
/// campaign-long mystery.

/// How the flow estimator is called at each Euler step.
public enum GuidanceMode: String, Sendable {
    /// One estimator call per Euler step. Correct for the guidance-distilled
    /// student, and the only mode the packed checkpoint supports.
    case singlePath = "single_path"
    /// Teacher-only CFG. Running it on the distilled student applies guidance
    /// twice, so this port does not implement it and `fromManifest` refuses a
    /// pack that asks for it, before any fingerprint is reported.
    case cfgDualPath = "cfg_dual_path"
}

/// The sampling law, LR-SAMPLER-v1 (`Sampler.swift`). Identical on every
/// backend, so it is hashed into the fingerprint rather than left to a caller.
public struct SamplingConfig: Sendable, Equatable {
    /// Logit scale before the min_p floor. Zero divides by zero; above 4 the
    /// distribution is flat enough to sample noise.
    public var temperature: Double = 0.8
    /// Multiplicative penalty on ids already seen in this window. Below 1.0 it
    /// rewards repetition instead of discouraging it.
    public var repetitionPenalty: Double = 1.2
    /// Candidates below this share of the peak probability are dropped. At 1.0
    /// the candidate set is empty.
    public var minP: Double = 0.05
    /// Longest run of speech tokens one window may generate. Cannot exceed the
    /// render window, which refuses anything past `maxSpeechTokens`.
    public var maxNewTokens: Int = 255
    /// Tokens exempt from the min_p floor, and from nothing else.
    ///
    /// A pause token is the only way to pause, and a min_p cutoff that
    /// removes it removes prosody: dropping this exemption was measured
    /// catastrophic (median long-form gap 2.46 s -> 4.64 s). These ids were
    /// also exempt from the repetition penalty. The pair of exemptions makes a
    /// silence run absorbing, so the penalty applies to silence like every
    /// other token; see `LRSamplerV1` for the measurement.
    public var silenceTokenIds: [Int] = []
    /// EOS floor (len-prior gate): the stop token is masked until
    /// `max(minTokensFloor, floor(nTextTokens * minTokensTextRatio))`.
    public var minTokensFloor: Int = 0
    /// The proportional half of the same floor, in speech tokens per text
    /// token. Zero leaves only the absolute floor.
    public var minTokensTextRatio: Double = 0.0

    public init() {}
}

/// How text longer than one window is split. Algorithm layer: where the splits
/// fall and what each piece is conditioned on decide where the reader breathes,
/// so they are configuration shared with Python, not caller whim.
///
/// Where `Chunking.splitText` decides to cut, and how a chunk is conditioned on
/// the previous one. The values are load-bearing: the fingerprint hashes them,
/// and `Engine.stream` and `synthesize` apply them (see `Chunking.swift`).
public struct ChunkConfig: Sendable, Equatable {
    /// Whether a passage is split at all. Off, a passage longer than one window
    /// is refused rather than joined.
    public var enabled: Bool = true
    /// Longest run of speech tokens one chunk may produce; matches the window.
    public var maxTokens: Int = 255
    /// Speech tokens from the previous chunk fed back as context. Zero means
    /// independent chunks, which stutters at joins (the F0 contour restarts
    /// ~74 Hz at the join on the reference voice); 6 tokens of context carry
    /// it across (~7 Hz, a natural phrase boundary).
    public var prefixTokens: Int = 6
    /// Split candidates, strongest first.
    public var splitOn: [String] = [". ", "! ", "? ", "; ", ", "]

    /// Written forms whose following period does not end a sentence, given
    /// without that period: the period is the separator's.
    ///
    /// One union list for every language, surveyed over 1200 passages in ten,
    /// a language-blind union re-chunks the corpus identically to ten
    /// per-language lists. Data, not code: replacing the array is the whole of
    /// adding a language. Not the funnel's list, which maps a written
    /// abbreviation to spoken words and carries only the unambiguous ones; what
    /// reaches here is the residue the funnel refuses to touch.
    ///
    /// It must equal `loudkit.config.ChunkConfig.abbreviations`.
    public var abbreviations: [String] = [
        "A", "B", "Cpn", "D", "Dr", "F", "H", "Hr", "I", "J", "K", "M", "Mr",
        "Mrs", "R", "S", "St", "T", "V", "Vors", "dr", "mrs", "prof", "\u{15B}w",
    ]

    /// What a period that does not end a sentence does to the split.
    ///
    /// A period that does not end a sentence is not a boundary. Breaking "But
    /// Mr. Smith went home" after the title hands the renderer a
    /// seven-character chunk with its own derived seed and a token ceiling
    /// proportional to seven characters. `.hold` is the law; `.break` is named
    /// so a pack can say which of the two it was measured under.
    public var midSentencePeriod: MidSentencePeriod = .hold

    /// What happens to a chunk the generator could not finish inside its window.
    public var capResplit: CapResplit = .word

    public init() {}

    /// Refuse a split recipe `Chunking.splitText` cannot run, with the
    /// sentences of Python's `ChunkConfig.__post_init__`.
    func validate() throws {
        guard maxTokens > 0 else {
            throw LoudKitError.manifest("max_tokens must be positive: \(maxTokens)")
        }
        // A character budget of zero makes splitText loop forever.
        let budget = Int(Double(maxTokens) * Chunking.charsPerToken)
        guard budget >= 1 else {
            let least = Int((1.0 / Chunking.charsPerToken).rounded(.up))
            throw LoudKitError.manifest(
                "max_tokens=\(maxTokens) leaves no character budget to split on "
                + "(int(\(maxTokens) * \(Chunking.charsPerToken)) == 0); "
                + "needs at least \(least)")
        }
        guard prefixTokens >= 0, prefixTokens < maxTokens else {
            throw LoudKitError.manifest(
                "prefix_tokens must be in [0, max_tokens): \(prefixTokens)")
        }
        guard !splitOn.isEmpty else {
            throw LoudKitError.manifest(
                "split_on cannot be empty: there would be nowhere to break")
        }
        guard !abbreviations.contains(where: \.isEmpty) else {
            // An empty entry is a suffix of everything and holds every split.
            throw LoudKitError.manifest("abbreviations cannot contain an empty string")
        }
    }
}

/// The two spellings of `ChunkConfig.midSentencePeriod`.
public enum MidSentencePeriod: String, Sendable, Equatable {
    case hold
    case brk = "break"
}

/// The two spellings of `ChunkConfig.capResplit`.
///
/// `Chunking.splitText` budgets characters against a constant, and a speaker
/// slower than it fills the window before the text runs out; the generator then
/// stops at the cap mid-word and the remainder is lost, because chunk texts are
/// fixed before any of them renders. `.word` halves such a chunk and generates
/// both halves in its place. `.off` ships the truncated window, which is what
/// every checkpoint built before this field did.
public enum CapResplit: String, Sendable, Equatable {
    /// Split the chunk at its middle word and generate both halves.
    case word
    /// Ship the truncated window.
    case off
}

/// How token sequences are framed for the mel decoder.
///
/// Algorithm layer, because the pad-and-truncate recipe was the whole measured
/// deviation between two backends' renders.
public struct WindowConfig: Sendable, Equatable {
    /// Longest token sequence one window carries, about 10.2 s at 25 Hz.
    public var maxSpeechTokens: Int = 255
    /// Pad every window to this many tokens, or `nil` for ragged.
    public var staticLength: Int?
    /// Pad token for unused static slots. The shipped engine pads with
    /// silence unit 4254; padding with token 0 bleeds +3 dB of high-band mel
    /// energy into the tail through the encoder's attention.
    public var padTokenId: Int?
    /// Fixed length of the reference-prompt window, or `nil` for ragged.
    public var staticPromptTokens: Int?

    public init() {}

    /// Refuse a framing recipe the renderer cannot pad to, with the sentences
    /// of Python's `WindowConfig.__post_init__`.
    func validate() throws {
        if let length = staticLength, length < maxSpeechTokens {
            throw LoudKitError.manifest(
                "static_length \(length) cannot be shorter than max_speech_tokens "
                + "\(maxSpeechTokens)")
        }
        if let prompt = staticPromptTokens, prompt <= 0 {
            throw LoudKitError.manifest("static_prompt_tokens must be positive: \(prompt)")
        }
    }
}

/// Everything that determines what the engine produces. Hashable, and printed
/// on every run.
public struct AlgorithmConfig: Sendable, Equatable {
    /// Names the parts of the algorithm that are code: the sampling law, the
    /// grid formula, the framing recipe. One accepted value per build.
    public var recipeVersion: String = "loudkit-1"
    /// How the flow estimator is called at each Euler step.
    public var guidance: GuidanceMode = .singlePath
    /// Read only in `cfgDualPath`, where it must be above zero.
    public var guidanceRate: Double = 0.0
    /// Euler steps the flow decoder takes. Below one there is no grid to walk.
    public var eulerSteps: Int = 2
    /// Explicit time grid of `eulerSteps + 1` values in [0, 1], or `nil` for
    /// the cosine schedule.
    public var eulerGrid: [Double]?
    /// The sampling law's constants.
    public var sampling = SamplingConfig()
    /// The framing recipe the mel decoder pads and truncates to.
    public var window = WindowConfig()
    /// Where a passage longer than one window is cut.
    public var chunking = ChunkConfig()
    /// The artifact detectors. They remove tokens, so they change the audio
    /// and are read from the manifest for the same reason the joins are: a
    /// backend that re-guesses where a chunk ended cuts somewhere else, and
    /// the difference is a hallucinated word that either does or does not
    /// reach a listener.
    public var postprocess = PostprocessConfig()

    /// The funnel's identity: its code version and the digest of the data
    /// files this port reads. In the fingerprint because the funnel decides
    /// what string the model is handed, and therefore what it says.
    public var text = TextConfig()
    /// Which decoder loop the token generator runs, `single` or `fusion_mtp2`.
    /// `Checkpoint.supportedDecodeModes` is the accepted set.
    public var decode: String = "single"
    /// The raised-cosine ramp on both edges of every rendered window, in
    /// seconds, bounded to [0.001, 0.05]. The shipped 20 ms is in the canonical
    /// form; only the historical 5 ms is omitted, so packs made before the key
    /// existed keep their identity.
    public var edgeFadeSeconds: Double = TimeStretch.edgeFadeSeconds
    /// Sample rate of the rendered waveform. Every duration this engine reports
    /// is `samples / sampleRate`.
    public var sampleRate: Int = 24_000
    /// Speech tokens per second, the rate a token count is read as a duration
    /// at. Hashed, and not otherwise read here.
    public var tokenRateHz: Double = 25.0
    /// Size of the acoustic codebook the generator samples from.
    public var speechVocabSize: Int = 8194
    /// Id that opens a generated window. Also the ceiling on an acoustic id:
    /// anything at or above it is a control token.
    public var startSpeechToken: Int = 6561
    /// Id the generator emits to stop. Must differ from `startSpeechToken`.
    public var stopSpeechToken: Int = 6562

    public init() {}

    /// The Euler time grid: explicit if configured, else the cosine schedule
    /// `t_i = 1 - cos(i/K * pi/2)`. One formula, shared with
    /// `loudkit.models.windowing.time_grid`; the fixture pins the values.
    public func timeGrid() -> [Double] {
        if let grid = eulerGrid { return grid }
        let k = Double(eulerSteps)
        return (0...eulerSteps).map { 1.0 - Foundation.cos(Double($0) / k * Double.pi / 2.0) }
    }

    /// The token count below which the stop token stays masked, the larger of
    /// the absolute floor and the proportional one.
    public func eosFloor(nTextTokens: Int) -> Int {
        max(sampling.minTokensFloor, Int(Double(nTextTokens) * sampling.minTokensTextRatio))
    }

    /// Refuse an algorithm no engine can run, in the order and with the
    /// sentences of Python's `AlgorithmConfig.__post_init__`.
    ///
    /// Python builds its dataclasses to read a manifest, so every range check
    /// runs at load; this port parses the JSON by hand and so has to say them
    /// again. Three of the gaps ended in a trap rather than a refusal: zero
    /// Euler steps makes `timeGrid()` return `[nan]` and the renderer emit
    /// noise, a negative count traps in `0...eulerSteps`, and the postprocess
    /// preset carries two more (see `PostprocessConfig.validate`).
    func validate() throws {
        if guidance == .singlePath, guidanceRate != 0.0 {
            throw LoudKitError.manifest("guidance_rate must be 0.0 in single_path mode")
        }
        if guidance == .cfgDualPath, guidanceRate <= 0.0 {
            throw LoudKitError.manifest(
                "cfg_dual_path with a zero rate does twice the work for nothing")
        }
        guard eulerSteps >= 1 else {
            throw LoudKitError.manifest("euler_steps must be >= 1: \(eulerSteps)")
        }
        try validateNumericCore()
        if let grid = eulerGrid {
            guard grid.count == eulerSteps + 1 else {
                throw LoudKitError.manifest(
                    "euler_grid has \(grid.count) points, expected \(eulerSteps + 1)")
            }
            guard zip(grid, grid.dropFirst()).allSatisfy({ $1 > $0 }) else {
                throw LoudKitError.manifest("euler_grid must be strictly increasing")
            }
            guard let first = grid.first, let last = grid.last,
                  Swift.abs(first) <= 1e-6, Swift.abs(last - 1.0) <= 1e-6 else {
                throw LoudKitError.manifest("euler_grid must run from 0.0 to 1.0")
            }
        }
        try chunking.validate()
        try window.validate()
        try postprocess.validate()
        // The three token budgets have to agree, or a chunk overruns the render
        // window mid-stream, after earlier chunks have already played.
        let win = window.maxSpeechTokens
        if chunking.enabled, chunking.maxTokens > win {
            throw LoudKitError.manifest(
                "chunking.max_tokens \(chunking.maxTokens) exceeds the render window "
                + "(\(win)): every chunk would be sized past what the renderer "
                + "accepts, and the refusal would land mid-stream, after audio had "
                + "already been delivered")
        }
        if sampling.maxNewTokens > win {
            throw LoudKitError.manifest(
                "sampling.max_new_tokens \(sampling.maxNewTokens) exceeds the render "
                + "window (\(win)): generation is allowed to produce more speech than "
                + "the renderer will accept, so a long utterance fails after it has "
                + "been generated rather than before")
        }
    }

    private func validateNumericCore() throws {
        guard edgeFadeSeconds >= 0.001, edgeFadeSeconds <= 0.05 else {
            throw LoudKitError.manifest(
                "edge_fade_seconds must be in [0.001, 0.05]: \(edgeFadeSeconds)")
        }
        guard sampleRate > 0 else {
            throw LoudKitError.manifest("sample_rate must be > 0: \(sampleRate)")
        }
        guard tokenRateHz > 0 else {
            throw LoudKitError.manifest("token_rate_hz must be > 0: \(tokenRateHz)")
        }
        guard speechVocabSize >= 1 else {
            throw LoudKitError.manifest("speech_vocab_size must be >= 1: \(speechVocabSize)")
        }
        for (name, value) in [("start_speech_token", startSpeechToken),
                              ("stop_speech_token", stopSpeechToken)]
        where !(0 <= value && value < speechVocabSize) {
            throw LoudKitError.manifest(
                "\(name) must be in [0, \(speechVocabSize)): \(value)")
        }
        guard startSpeechToken != stopSpeechToken else {
            throw LoudKitError.manifest(
                "start_speech_token and stop_speech_token must differ: both are "
                + "\(startSpeechToken)")
        }
    }

    // MARK: identity

    /// The fingerprint schema version. Bump it in lockstep with Python's
    /// `loudkit.config.FINGERPRINT_SCHEMA`.
    public static let fingerprintSchema = 1

    /// The exact string that gets hashed: `AlgorithmConfig.canonical_form()`
    /// on the Python side, implemented here against the same three rules
    /// rather than copied: **floats are `repr` strings** (shortest
    /// round-trip, which Swift's `Double` description also produces, so both
    /// languages emit the same digits from the same IEEE-754 value); **keys
    /// are sorted**; and the body travels in an explicit
    /// `{"algorithm": ..., "schema": N}` envelope. The conformance suite
    /// compares the two languages' canonical forms directly, so this is a
    /// second computation of the definition, not a stored constant.
    public func canonicalForm() -> String {
        func num(_ d: Double) -> String { "\"\(d)\"" }  // repr, as a JSON string
        func str(_ s: String) -> String {
            var out = "\""
            for ch in s.unicodeScalars {
                switch ch {
                case "\"": out += "\\\""
                case "\\": out += "\\\\"
                case "\n": out += "\\n"
                case "\t": out += "\\t"
                case "\r": out += "\\r"
                default:
                    // Python's `json.dumps` defaults to `ensure_ascii=True`, so
                    // every non-ASCII character in the canonical form is a
                    // \uXXXX escape and an astral one is a surrogate pair. No
                    // hashed string had a non-ASCII character in it until
                    // `chunking.abbreviations` carried "\u015bw", and the raw
                    // UTF-8 spelling hashed differently from the reference in
                    // every port at once.
                    if ch.value < 0x20 || ch.value >= 0x7f {
                        if ch.value > 0xFFFF {
                            let v = ch.value - 0x10000
                            out += String(format: "\\u%04x\\u%04x",
                                          0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF))
                        } else {
                            out += String(format: "\\u%04x", ch.value)
                        }
                    } else {
                        out.unicodeScalars.append(ch)
                    }
                }
            }
            return out + "\""
        }
        func opt(_ v: Int?) -> String { v.map(String.init) ?? "null" }
        func ids(_ v: [Int]) -> String { "[" + v.map(String.init).joined(separator: ",") + "]" }

        let grid = eulerGrid.map { "[" + $0.map(num).joined(separator: ",") + "]" } ?? "null"
        let sil = ids(sampling.silenceTokenIds)
        let splitOn = "[" + chunking.splitOn.map(str).joined(separator: ",") + "]"
        // Keys sorted, as everywhere in this form: "abbreviations" before
        // "cap_resplit" before "enabled", and "mid_sentence_period" between
        // "max_tokens" and "prefix_tokens". Five canonical forms are hand-written and a new
        // field's position in that order is part of the contract.
        let abbreviations = "[" + chunking.abbreviations.map(str).joined(separator: ",") + "]"
        let chunkingJSON = "{"
            + "\"abbreviations\":\(abbreviations),"
            + "\"cap_resplit\":\(str(chunking.capResplit.rawValue)),"
            + "\"enabled\":\(chunking.enabled),"
            + "\"max_tokens\":\(chunking.maxTokens),"
            + "\"mid_sentence_period\":\(str(chunking.midSentencePeriod.rawValue)),"
            + "\"prefix_tokens\":\(chunking.prefixTokens),"
            + "\"split_on\":\(splitOn)"
            + "}"
        let samplingJSON = "{"
            + "\"max_new_tokens\":\(sampling.maxNewTokens),"
            + "\"min_p\":\(num(sampling.minP)),"
            + "\"min_tokens_floor\":\(sampling.minTokensFloor),"
            + "\"min_tokens_text_ratio\":\(num(sampling.minTokensTextRatio)),"
            + "\"repetition_penalty\":\(num(sampling.repetitionPenalty)),"
            + "\"silence_token_ids\":\(sil),"
            + "\"temperature\":\(num(sampling.temperature))"
            + "}"
        let windowJSON = "{"
            + "\"max_speech_tokens\":\(window.maxSpeechTokens),"
            + "\"pad_token_id\":\(opt(window.padTokenId)),"
            + "\"static_length\":\(opt(window.staticLength)),"
            + "\"static_prompt_tokens\":\(opt(window.staticPromptTokens))"
            + "}"
        // Keys sorted, as everywhere in this form. The detectors remove
        // tokens, so a port using a different threshold produces different
        // audio, exactly the silent drift a whole-config hash exists to catch.
        let pp = postprocess
        let postprocessJSON = "{"
            + "\"ceiling_slack_tokens\":\(pp.ceilingSlackTokens),"
            + "\"ceiling_speech_per_text_token\":\(num(pp.ceilingSpeechPerTextToken)),"
            + "\"desperation_band_floor\":\(pp.desperationBandFloor),"
            + "\"desperation_band_ratio\":\(num(pp.desperationBandRatio)),"
            + "\"desperation_min_keep_per_text_token\":\(num(pp.desperationMinKeepPerTextToken)),"
            + "\"desperation_min_text_tokens\":\(pp.desperationMinTextTokens),"
            + "\"desperation_speech_per_text_token\":\(num(pp.desperationSpeechPerTextToken)),"
            + "\"dropout_min_tokens\":\(pp.dropoutMinTokens),"
            + "\"echo_strong_eos_probability\":\(num(pp.echoStrongEosProbability)),"
            + "\"echo_strong_max_tail\":\(pp.echoStrongMaxTail),"
            + "\"echo_strong_min_position_pct\":\(pp.echoStrongMinPositionPct),"
            + "\"echo_weak_eos_probability\":\(num(pp.echoWeakEosProbability)),"
            + "\"echo_weak_max_tail\":\(pp.echoWeakMaxTail),"
            + "\"echo_weak_min_position_pct\":\(pp.echoWeakMinPositionPct),"
            + "\"ended_tail_blip_max\":\(pp.endedTailBlipMax),"
            + "\"ended_tail_keep\":\(pp.endedTailKeep),"
            + "\"ended_tail_silence_run\":\(pp.endedTailSilenceRun),"
            + "\"ended_tail_word_max\":\(pp.endedTailWordMax),"
            + "\"filler_max_speech_after_run\":\(pp.fillerMaxSpeechAfterRun),"
            + "\"filler_min_eos_probability\":\(num(pp.fillerMinEosProbability)),"
            + "\"mode\":\(str(pp.mode.rawValue)),"
            + "\"pacing_tolerance\":\(num(pp.pacingTolerance)),"
            + "\"quiet_render_ids\":\(ids(pp.quietRenderIds)),"
            + "\"repetition_max_period\":\(pp.repetitionMaxPeriod),"
            + "\"repetition_min_cycles\":\(pp.repetitionMinCycles),"
            + "\"repetition_min_span\":\(pp.repetitionMinSpan),"
            + "\"repetition_resume\":\(str(pp.repetitionResume.rawValue)),"
            + "\"repetition_silence\":\(str(pp.repetitionSilence.rawValue)),"
            + "\"retry_max_attempts\":\(pp.retryMaxAttempts),"
            + "\"silence_render_ids\":\(ids(pp.silenceRenderIds)),"
            + "\"stall_run_tokens\":\(pp.stallRunTokens),"
            + "\"trailing_filler_threshold\":\(num(pp.trailingFillerThreshold)),"
            + "\"trailing_silence_run_tokens\":\(pp.trailingSilenceRunTokens)"
            + "}"
        let body = "{"
            + "\"chunking\":\(chunkingJSON),"
            + (decode == "single" ? "" : "\"decode_mode\":\(str(decode)),")
            + (edgeFadeSeconds == 0.005 ? "" : "\"edge_fade_seconds\":\(num(edgeFadeSeconds)),")
            + "\"euler_grid\":\(grid),"
            + "\"euler_steps\":\(eulerSteps),"
            + "\"guidance\":\(str(guidance.rawValue)),"
            + "\"guidance_rate\":\(num(guidanceRate)),"
            + "\"postprocess\":\(postprocessJSON),"
            + "\"recipe_version\":\(str(recipeVersion)),"
            + "\"sample_rate\":\(sampleRate),"
            + "\"sampling\":\(samplingJSON),"
            + "\"speech_vocab_size\":\(speechVocabSize),"
            + "\"start_speech_token\":\(startSpeechToken),"
            + "\"stop_speech_token\":\(stopSpeechToken),"
            + "\"text\":{\"grammar\":\(str(text.grammar)),"
            + "\"recipe\":\(str(text.recipe))},"
            + "\"token_rate_hz\":\(num(tokenRateHz)),"
            + "\"window\":\(windowJSON)"
            + "}"
        return "{\"algorithm\":\(body),\"schema\":\(Self.fingerprintSchema)}"
    }

    /// First 16 hex chars of SHA-256 over the canonical form. Comparable
    /// with `AlgorithmConfig.fingerprint()` on the Python side.
    public func fingerprint() -> String {
        let digest = SHA256.hash(data: Data(canonicalForm().utf8))
        return digest.map { String(format: "%02x", $0) }.joined().prefix(16).description
    }

    /// One line for a log: the fingerprint and the values a reader checks first.
    public func describe() -> String {
        let g = guidance == .singlePath ? "single_path" : "cfg@\(guidanceRate)"
        let grid = eulerGrid == nil ? "cosine" : "explicit"
        let win = window.staticLength.map(String.init) ?? "ragged"
        return "algo[\(fingerprint())] \(recipeVersion) \(g) euler=\(eulerSteps)(\(grid)) "
            + "temp=\(sampling.temperature) rep=\(sampling.repetitionPenalty) "
            + "min_p=\(sampling.minP) sil=\(sampling.silenceTokenIds.count) win=\(win)"
    }

    // MARK: manifest

    /// Build from a checkpoint manifest. Swift requires an *amended*
    /// checkpoint (`tools/amend_manifest.py`): the window recipe and EOS
    /// floor must be manifest-borne, because this implementation deliberately
    /// carries no fallback constants to re-guess them from.
    public static func fromManifest(_ manifest: [String: Any]) throws -> AlgorithmConfig {
        var config = AlgorithmConfig()
        if manifest["decode"] != nil {
            let block = try ManifestReader.block(manifest, "decode")
            let mode = try ManifestReader.text(block, "manifest['decode']", "mode") ?? "single"
            guard Checkpoint.supportedDecodeModes.contains(mode) else {
                throw LoudKitError.manifest(
                    "manifest['decode']['mode'] is \(mode); this build decodes "
                    + Checkpoint.supportedDecodeModes.joined(separator: " or "))
            }
            config.decode = mode
        }

        // One recipe means one accepted value: a foreign tag believed here
        // would ride into every fingerprint, and a foreign tag defaulted would
        // claim this recipe for a checkpoint that named another. Absence is
        // not a tag; it is the shipping default left unstated. A non-string
        // is refused rather than defaulted, so all five ports read one way.
        if let raw = manifest["recipe_version"] {
            guard let declared = raw as? String, declared == "loudkit-1" else {
                throw LoudKitError.manifest(
                    "manifest declares recipe_version \(raw); "
                        + "the only recipe is \"loudkit-1\"")
            }
            config.recipeVersion = declared
        }
        // The block's *values*, not just whether the key is present. This read
        // only `manifest["chunking"] == nil` and ignored what was inside the
        // block, so a manifest declaring
        // `prefix_tokens: 0` or `split_on: ["; "]` ran with Swift's defaults
        // while Python (`config.py`) parsed and honoured them, two engines,
        // one manifest, different chunk boundaries and therefore different
        // audio.
        let chunking = try ManifestReader.block(manifest, "chunking")
        if !chunking.isEmpty {
            // A chunking key this splitter does not honour is refused by name
            // rather than ignored. `ChunkConfig` here carries no first-chunk
            // cap, so the canonical form has no slot for one: a manifest that
            // sets `first_chunk_max_tokens` and a manifest that omits it hash
            // identically, while Python cuts the first chunk short for time to
            // first audio and this splitter does not. That is one reading in
            // Python and another here under a matching `recipe_version`, with
            // nothing to report it. Go, Rust and JS refuse it by this name too.
            // Presence, not value: `null` is still the key being set, and
            // Python reads it as "no cap".
            for key in ["first_chunk_max_tokens"] where chunking[key] != nil {
                throw LoudKitError.manifest(
                    "manifest['chunking']['\(key)'] is not implemented by this port, "
                        + "which would split the text differently from the Python "
                        + "engine under a matching recipe_version")
            }
            let path = "manifest['chunking']"
            if let enabled = try ManifestReader.flag(chunking, path, "enabled") {
                config.chunking.enabled = enabled
            }
            if chunking["max_tokens"] != nil {
                let maxTokens = try ManifestReader.count(
                    chunking, path, "max_tokens", config.chunking.maxTokens)
                // Refused, like the other four. Python, Go, Rust and JS all
                // reject a non-positive `max_tokens`; Swift took it and
                // degraded to one character per chunk through `max(cut, 1)` in
                // `Chunking`, which is a passage split into single letters
                // rather than an error anyone can act on.
                guard maxTokens > 0 else {
                    throw LoudKitError.manifest(
                        "chunking.max_tokens must be positive: \(maxTokens)")
                }
                config.chunking.maxTokens = maxTokens
            }
            config.chunking.prefixTokens = try ManifestReader.count(
                chunking, path, "prefix_tokens", config.chunking.prefixTokens)
            // An array of separators. A bare string is eight separators of one
            // character each once Swift iterates it, which is not what any
            // manifest means, refuse rather than reinterpret.
            if let list = try ManifestReader.strings(chunking, path, "split_on") {
                config.chunking.splitOn = list
            }
            // Presence, not emptiness: an empty list is meaningful here, being
            // one law spelled as data, so it must not fall back to the
            // shipping set.
            //
            // The same trap as split_on, and worse: a bare "Mr" iterates to
            // ("M", "r"), and the entry "r" is the suffix of a great many
            // ordinary words, so their sentence ends would all be held.
            if let list = try ManifestReader.strings(chunking, path, "abbreviations") {
                guard !list.contains(where: \.isEmpty) else {
                    throw LoudKitError.manifest(
                        "chunking.abbreviations cannot contain an empty string")
                }
                config.chunking.abbreviations = list
            }
            if let raw = try ManifestReader.text(chunking, path, "mid_sentence_period") {
                guard let law = MidSentencePeriod(rawValue: raw) else {
                    throw LoudKitError.manifest(
                        "manifest declares unknown chunking.mid_sentence_period \(raw)")
                }
                config.chunking.midSentencePeriod = law
            }
            if let raw = try ManifestReader.text(chunking, path, "cap_resplit") {
                guard let law = CapResplit(rawValue: raw) else {
                    throw LoudKitError.manifest(
                        "manifest declares unknown chunking.cap_resplit \(raw)")
                }
                config.chunking.capResplit = law
            }
        }
        config.postprocess = try Self.postprocess(from: manifest)
        let guidanceRaw = try ManifestReader.text(manifest, "manifest", "guidance")
            ?? "single_path"
        guard let guidance = GuidanceMode(rawValue: guidanceRaw) else {
            throw LoudKitError.manifest("unknown guidance mode \(guidanceRaw)")
        }
        // Refused here, at the door, and not at the first Euler step: a pack
        // this port cannot run must not load, because a loaded pack publishes
        // a fingerprint for its algorithm and the caller then has every reason
        // to believe it is good. Go and JS refuse in their manifest readers,
        // in these words, and Rust in `guidance_from`; this is the same
        // sentence in the same place. `MelDecoder.init` refuses too, as the
        // second line of defence.
        guard guidance != .cfgDualPath else {
            throw LoudKitError.manifest(
                "manifest declares guidance mode cfg_dual_path, which this binding does "
                + "not implement: it would render single-path audio and silently "
                + "disagree with the Python engine")
        }
        config.guidance = guidance
        config.guidanceRate = try ManifestReader.number(manifest, "manifest", "guidance_rate", 0.0)
        // An explicit null is the one place absence is spelled out: a manifest
        // older than the field and one that writes `edge_fade_seconds: null`
        // both mean the 5 ms those releases shipped. Anything else with no
        // numeric reading is refused, as `edge_fade_from` refuses it.
        config.edgeFadeSeconds = manifest["edge_fade_seconds"] is NSNull
            ? 0.005
            : try ManifestReader.number(manifest, "manifest", "edge_fade_seconds", 0.005)
        config.eulerSteps = try ManifestReader.count(manifest, "manifest", "n_cfm_timesteps", 2)
        // Read, not ignored: a manifest declaring a grid or a token rate that
        // this port left at its default runs one algorithm and fingerprints
        // another, which is the drift the whole-config hash exists to catch.
        config.eulerGrid = try ManifestReader.floats(manifest, "euler_grid")
        config.tokenRateHz = try ManifestReader.number(manifest, "manifest", "token_rate_hz", 25.0)
        config.sampleRate = try ManifestReader.count(manifest, "manifest", "sample_rate", 24_000)
        // Python refuses a manifest with a non-positive `sample_rate` and the other four
        // took it: every duration this engine reports is `samples / sample_rate`, so a
        // zero divides by zero and a negative reports negative seconds. A rate is the one
        // manifest field whose wrongness is not caught by any shape.
        guard config.sampleRate > 0 else {
            throw LoudKitError.manifest("sample_rate must be > 0: \(config.sampleRate)")
        }
        config.speechVocabSize = try ManifestReader.count(
            manifest, "manifest", "speech_vocab_size", 8194)
        let speech = try ManifestReader.block(manifest, "speech_tokens")
        config.startSpeechToken = try ManifestReader.count(
            speech, "manifest['speech_tokens']", "start", 6561)
        config.stopSpeechToken = try ManifestReader.count(
            speech, "manifest['speech_tokens']", "stop", 6562)

        var sampling = SamplingConfig()
        let defaults = try ManifestReader.block(manifest, "sampling_defaults")
        if !defaults.isEmpty {
            let path = "manifest['sampling_defaults']"
            sampling.temperature = try ManifestReader.number(defaults, path, "temperature", 0.8)
            sampling.repetitionPenalty = try ManifestReader.number(
                defaults, path, "repetition_penalty", 1.2)
            sampling.minP = try ManifestReader.number(defaults, path, "min_p", 0.05)
            // Range checks mirror Python's `SamplingConfig.__post_init__`: a
            // manifest the reference refuses must be refused here too, or two
            // implementations render different audio under one fingerprint.
            guard sampling.temperature > 0, sampling.temperature <= 4 else {
                throw LoudKitError.manifest("temperature out of range: \(sampling.temperature)")
            }
            guard sampling.repetitionPenalty >= 1.0 else {
                throw LoudKitError.manifest(
                    "repetition_penalty out of range: \(sampling.repetitionPenalty)")
            }
            guard sampling.minP >= 0, sampling.minP < 1 else {
                throw LoudKitError.manifest("min_p out of range: \(sampling.minP)")
            }
            sampling.maxNewTokens = try ManifestReader.count(
                defaults, path, "max_new_tokens", 255)
            // The manifest's `max_new_tokens` gets the range check `Engine`
            // gives a caller's `maxNewTokens`: zero generates an empty row and
            // returns it as a result, and a negative indexes `0..<cap` in the
            // token generator. Python, Go, Rust and JS refuse the manifest
            // here too, so a pack loads on all five ports or on none.
            guard sampling.maxNewTokens >= 1 else {
                throw LoudKitError.manifest(
                    "max_new_tokens must be positive: \(sampling.maxNewTokens)")
            }
        }
        sampling.silenceTokenIds = try ManifestReader.ids(manifest, "silence_token_ids")

        let eos = try ManifestReader.block(manifest, "eos_floor")
        // `window: null` is the ragged window said out loud, and an absent
        // `window` or `eos_floor` is the same ragged default: `_window_from`
        // returns `WindowConfig()` for an absent key and for an explicit null,
        // and Go, Rust and JS default the same way, so a manifest that carries
        // neither block loads on every port. Anything else non-null is
        // refused, by `ManifestReader.block`, exactly as `_window_from`
        // refuses it.
        let win = manifest["window"] is NSNull
            ? [:] : try ManifestReader.block(manifest, "window")
        sampling.minTokensFloor = try ManifestReader.count(
            eos, "manifest['eos_floor']", "min_tokens_floor", 0)
        sampling.minTokensTextRatio = try ManifestReader.number(
            eos, "manifest['eos_floor']", "min_tokens_text_ratio", 0.0)
        guard sampling.minTokensFloor >= 0 else {
            throw LoudKitError.manifest(
                "min_tokens_floor must be >= 0: \(sampling.minTokensFloor)")
        }
        guard sampling.minTokensTextRatio >= 0 else {
            throw LoudKitError.manifest(
                "min_tokens_text_ratio must be >= 0: \(sampling.minTokensTextRatio)")
        }
        config.sampling = sampling

        // Only the three lengths have a null reading, and it is the ragged
        // one: `_window_from` reads them through an `opt` that spells an absent
        // key and an explicit null the same way, and reads `max_speech_tokens`
        // with a bare `int()` that refuses a null like any other non-number.
        var window = WindowConfig()
        let winPath = "manifest['window']"
        window.maxSpeechTokens = try ManifestReader.count(win, winPath, "max_speech_tokens", 255)
        window.staticLength = try ManifestReader.optCount(win, winPath, "static_length")
        window.padTokenId = try ManifestReader.optCount(win, winPath, "pad_token_id")
        window.staticPromptTokens = try ManifestReader.optCount(win, winPath,
                                                               "static_prompt_tokens")
        config.window = window
        // The whole table at the end, where every field has its manifest value:
        // Python runs these in the dataclasses' initialisers, so a manifest it
        // refuses has to be refused here too.
        try config.validate()
        return config
    }

    /// The `postprocess` block, or the shipping detectors when it is absent.
    ///
    /// An unknown mode is refused rather than defaulted: it would trim where
    /// the manifest said not to, under a matching `recipe_version`.
    private static func postprocess(from manifest: [String: Any]) throws -> PostprocessConfig {
        var cfg = PostprocessConfig()
        // The render censuses: which ids actually render as digital silence
        // (`silence_render_ids`) and which as contextually-quiet breath/decay
        // (`quiet_render_ids`). Optional: a checkpoint packed before the
        // census has neither, and the stall detector then falls back to
        // `silence_token_ids`, degraded but safe. Read from the manifest top
        // level, before the block check, because they are properties of the
        // weights, like `silence_token_ids`, not detector constants someone
        // tuned, a manifest with no postprocess block still carries them.
        cfg.silenceRenderIds = try ManifestReader.ids(manifest, "silence_render_ids")
        cfg.quietRenderIds = try ManifestReader.ids(manifest, "quiet_render_ids")
        let block = try ManifestReader.block(manifest, "postprocess")
        if block.isEmpty { return cfg }
        let path = "manifest['postprocess']"

        // The censuses live at the manifest top level, where they are read
        // above. Accepting them here too would give one value two homes in
        // one file; Python's block reader refuses them the same way.
        for key in ["silence_render_ids", "quiet_render_ids"] where block[key] != nil {
            throw LoudKitError.manifest(
                "manifest['postprocess']['\(key)'] belongs at the manifest "
                + "top level, beside 'silence_token_ids'")
        }

        if let raw = try ManifestReader.text(block, path, "mode") {
            guard let mode = Postprocess.Mode(rawValue: raw) else {
                throw LoudKitError.manifest(
                    "unknown postprocess mode \(raw); expected one of "
                    + Postprocess.Mode.allCases.map(\.rawValue).joined(separator: ", "))
            }
            cfg.mode = mode
        }
        func d(_ key: String, _ current: Double) throws -> Double {
            try ManifestReader.number(block, path, key, current)
        }
        func i(_ key: String, _ current: Int) throws -> Int {
            try ManifestReader.count(block, path, key, current)
        }
        cfg.ceilingSpeechPerTextToken =
            try d("ceiling_speech_per_text_token", cfg.ceilingSpeechPerTextToken)
        cfg.ceilingSlackTokens = try i("ceiling_slack_tokens", cfg.ceilingSlackTokens)
        cfg.trailingFillerThreshold = try d("trailing_filler_threshold", cfg.trailingFillerThreshold)
        cfg.trailingSilenceRunTokens =
            try i("trailing_silence_run_tokens", cfg.trailingSilenceRunTokens)
        cfg.desperationBandRatio = try d("desperation_band_ratio", cfg.desperationBandRatio)
        cfg.desperationBandFloor = try i("desperation_band_floor", cfg.desperationBandFloor)
        cfg.fillerMinEosProbability =
            try d("filler_min_eos_probability", cfg.fillerMinEosProbability)
        cfg.fillerMaxSpeechAfterRun =
            try i("filler_max_speech_after_run", cfg.fillerMaxSpeechAfterRun)
        cfg.desperationSpeechPerTextToken =
            try d("desperation_speech_per_text_token", cfg.desperationSpeechPerTextToken)
        cfg.desperationMinTextTokens =
            try i("desperation_min_text_tokens", cfg.desperationMinTextTokens)
        cfg.desperationMinKeepPerTextToken =
            try d("desperation_min_keep_per_text_token", cfg.desperationMinKeepPerTextToken)
        cfg.endedTailSilenceRun = try i("ended_tail_silence_run", cfg.endedTailSilenceRun)
        cfg.endedTailBlipMax = try i("ended_tail_blip_max", cfg.endedTailBlipMax)
        cfg.endedTailWordMax = try i("ended_tail_word_max", cfg.endedTailWordMax)
        cfg.endedTailKeep = try i("ended_tail_keep", cfg.endedTailKeep)
        cfg.echoStrongEosProbability =
            try d("echo_strong_eos_probability", cfg.echoStrongEosProbability)
        cfg.echoStrongMaxTail = try i("echo_strong_max_tail", cfg.echoStrongMaxTail)
        cfg.echoStrongMinPositionPct =
            try i("echo_strong_min_position_pct", cfg.echoStrongMinPositionPct)
        cfg.echoWeakEosProbability = try d("echo_weak_eos_probability", cfg.echoWeakEosProbability)
        cfg.echoWeakMaxTail = try i("echo_weak_max_tail", cfg.echoWeakMaxTail)
        cfg.echoWeakMinPositionPct =
            try i("echo_weak_min_position_pct", cfg.echoWeakMinPositionPct)
        // Every postprocess parameter, because a hand-written list that
        // misses one is a manifest key this port does not read. Python takes
        // its fields off the dataclass precisely so a new constant cannot be
        // left out; the four ports write the list by hand, so the list has to
        // be complete. Defaults matching hides the gap until a checkpoint
        // sets one of them, at which point the manifest declares one recipe
        // and the engine runs another.
        cfg.dropoutMinTokens = try i("dropout_min_tokens", cfg.dropoutMinTokens)
        cfg.retryMaxAttempts = try i("retry_max_attempts", cfg.retryMaxAttempts)
        cfg.pacingTolerance = try d("pacing_tolerance", cfg.pacingTolerance)
        cfg.repetitionMaxPeriod = try i("repetition_max_period", cfg.repetitionMaxPeriod)
        cfg.repetitionMinCycles = try i("repetition_min_cycles", cfg.repetitionMinCycles)
        cfg.repetitionMinSpan = try i("repetition_min_span", cfg.repetitionMinSpan)
        // A string field like mode, and refused like mode: a law this port
        // does not implement must not fall back to a default: the resolver
        // would cut where the manifest said to condemn.
        if let raw = try ManifestReader.text(block, path, "repetition_resume") {
            guard let law = Postprocess.RepetitionResume(rawValue: raw) else {
                throw LoudKitError.manifest(
                    "unknown repetition_resume \(raw); expected one of "
                    + Postprocess.RepetitionResume.allCases.map(\.rawValue)
                        .joined(separator: ", "))
            }
            cfg.repetitionResume = law
        }
        // A string field like mode, and refused like mode: a family this
        // port does not implement must not fall back to a default: the loop
        // exemption would read one silence list under a manifest declaring
        // another.
        if let raw = try ManifestReader.text(block, path, "repetition_silence") {
            guard let family = Postprocess.RepetitionSilence(rawValue: raw) else {
                throw LoudKitError.manifest(
                    "unknown repetition_silence \(raw); expected one of "
                    + Postprocess.RepetitionSilence.allCases.map(\.rawValue)
                        .joined(separator: ", "))
            }
            cfg.repetitionSilence = family
        }
        cfg.stallRunTokens = try i("stall_run_tokens", cfg.stallRunTokens)
        return cfg
    }
}

/// How a backend gets there. Free to differ from the Python side; declared so
/// the parity table can label every row.
public struct ExecutionConfig: Sendable, Equatable {
    /// Informational: the token generator runs natively on the CPU in fp32,
    /// weights upcast from the packed fp16, and nothing reads this field to
    /// choose. That is the measured-right placement on Apple silicon, where the
    /// autoregressive stage is faster on CPU than GPU or ANE at batch one, and
    /// fp32 is the declared precision of the conformance fixture ("same
    /// precision, same tokens").
    public var tokenGeneratorPrecision: String = "fp32"
    /// Compute units per CoreML stage. The estimator is the ANE citizen; the
    /// encoder and vocoder are fp32 graphs and stay on CPU.
    public var encoderComputeUnits: ComputeUnits = .cpuOnly
    /// Where the flow estimator runs. The one stage the ANE is faster at.
    public var estimatorComputeUnits: ComputeUnits = .cpuAndNeuralEngine
    /// Where the vocoder runs. An fp32 graph, so CPU.
    public var vocoderComputeUnits: ComputeUnits = .cpuOnly

    /// The CoreML placements this package names, mapped onto
    /// `MLComputeUnits` at load.
    public enum ComputeUnits: String, Sendable {
        case cpuOnly, cpuAndNeuralEngine, all
    }

    public init() {}

    /// One line for a log: the three placements and the generator's precision.
    public func describe() -> String {
        "exec[swift-native-t3(\(tokenGeneratorPrecision)) "
            + "coreml enc=\(encoderComputeUnits.rawValue) "
            + "est=\(estimatorComputeUnits.rawValue) voc=\(vocoderComputeUnits.rawValue)]"
    }
}

/// Everything this package throws. Six cases, so a caller can branch on the
/// kind of failure without matching on message text, and a ``code`` in the
/// shared catalog every port and every transport names failures with.
public enum LoudKitError: Error, CustomStringConvertible, LocalizedError {
    /// A checkpoint manifest declares something this build cannot run, or
    /// something the reference implementation refuses.
    case manifest(String)
    /// A file the engine needs is absent, too large, or not what it claims.
    case asset(String)
    /// A caller's argument is outside the contract: a token id, a length, a
    /// rate, a recording.
    case shape(String)
    /// A CoreML stage failed or returned an output the host cannot read.
    case prediction(String)
    /// `shouldCancel` returned true and the partial utterance was discarded.
    ///
    /// Thrown rather than returned as a short `Result`, because a caller who
    /// interrupted wants to know that nothing was produced: a `Result` with a
    /// fraction of the audio is indistinguishable from a short utterance.
    case cancelled
    /// A one-window render whose text did not fit the window.
    ///
    /// Thrown rather than returned with `hitTokenCap`, because shipping the
    /// window's worth of audio with the rest of the text never spoken is
    /// silent data loss. `synthesize` splits at sentence boundaries instead.
    case windowOverflow(tokens: Int, window: Int)

    /// The kind, then the sentence. The kind is a prefix rather than a code so
    /// one printed line says both.
    public var description: String {
        switch self {
        case .manifest(let s): return "manifest: \(s)"
        case .asset(let s): return "asset: \(s)"
        case .shape(let s): return "shape: \(s)"
        case .prediction(let s): return "prediction: \(s)"
        case .cancelled: return "cancelled before the audio was rendered"
        case .windowOverflow(let tokens, let window):
            return
                "the text did not fit one \(window)-token window (\(tokens) produced) "
                + "and its tail was not spoken. Use synthesize, which splits at "
                + "sentence boundaries and joins the audio."
        }
    }

    /// This condition's name in the frozen error-code catalog, the one
    /// vocabulary every transport and every port shares.
    ///
    /// Mirrors `loudkit.errors.error_code`, including its fallback: a condition
    /// the catalog does not name yet is `invalid_request`, which is also
    /// `LoudkitError.code`'s own default. So a Swift caller and a Python caller
    /// classify the same failure with the same string, and a transport in front
    /// of either sends the same body.
    ///
    /// Six cases are coarser than the catalog, so the case alone cannot say
    /// which condition a `.asset` or a `.shape` is, and a seventh case would
    /// change what a caller matching the existing six sees. What can move is
    /// where the condition is *written down*: the sites that raise a named
    /// condition already say which one it is, in the message a caller reads,
    /// so this reads it back from there instead of guessing from the case.
    /// Each fragment is a constant the raising site builds its message from
    /// (``noSuchVoice`` and the three beside it), so the site and this
    /// classifier cannot drift apart, and no message byte moved to make the
    /// reading possible.
    ///
    /// Five of the nine codes are reachable here. The four that are not:
    /// `number_grammar` because this port has no such refusal at all
    /// (`Numbers.cardinal` returns `nil` for a value past the grammar's
    /// largest scale and the funnel leaves the digits alone), `audio_not_found`
    /// and `provenance_invalid` because this port has neither an enrollment
    /// audio reader nor a provenance reader, and `invalid_request` which is the
    /// fallback rather than a condition. A `.manifest` or a `.prediction` is
    /// `invalid_request` on both sides: neither has a catalog name.
    public var code: String {
        switch self {
        case .cancelled: return "cancelled"
        case .windowOverflow: return "window_overflow"
        case .asset(let message):
            if message.contains(Self.noSuchVoice) { return "voice_not_found" }
            if message.contains(Self.unsupportedLanguage) { return "unsupported_language" }
            return "invalid_request"
        case .shape(let message):
            if message.contains(Self.notASpeechToken) || message.contains(Self.nothingToRender) {
                return "invalid_tokens"
            }
            return "invalid_request"
        case .manifest, .prediction: return "invalid_request"
        }
    }

    /// `ModelBundle.voice(named:)` on a name the release does not ship.
    static let noSuchVoice = ": no such voice in "
    /// `TextFrontend.encode` on a language off the twelve-id roster.
    static let unsupportedLanguage = ". Supported: "
    /// `Engine.validateSpeechTokens` on an id outside the acoustic codebook.
    static let notASpeechToken = ", which is not an acoustic speech token"
    /// `Engine.synthesizeTokens` on an empty sequence. The catalog counts an
    /// empty sequence as `invalid_tokens`, not as a bare bad request.
    static let nothingToRender = "tokens is empty: there is nothing to render."

    /// No voice by that name in the release at `directory`.
    ///
    /// A factory rather than a bare `.asset` at the site, so the sentence that
    /// makes ``code`` read `voice_not_found` is written once. The case and the
    /// message are what they were.
    static func voiceNotFound(name: String, directory: String, shipped: [String]) -> Self {
        .asset(
            "\(name)\(noSuchVoice)\(directory). This release ships "
                + (shipped.isEmpty ? "none" : shipped.joined(separator: ", ")) + ".")
    }

    /// A language this build's text layer is not written for.
    static func languageUnsupported(_ language: String, why: String, roster: [String]) -> Self {
        .asset("language \(language) \(why)\(unsupportedLanguage)\(roster.joined(separator: ", "))")
    }

    /// A speech token id outside the acoustic codebook.
    static func invalidToken(field: String, token: Int, limit: Int) -> Self {
        .shape(
            "\(field) contains \(token)\(notASpeechToken) "
                + "(expected 0 <= id < \(limit)). Pass `Result.tokens` "
                + "from an earlier call; the generator's own control tokens are "
                + "already stripped from it.")
    }

    /// An empty speech token sequence.
    static var noTokensToRender: Self { .shape(nothingToRender) }

    /// LocalizedError, not just CustomStringConvertible: without it the
    /// messages above never reach the user. Swift bridges a plain `Error` to
    /// NSError, and `localizedDescription`, what a caller prints, what an
    /// alert shows, becomes "The operation couldn't be completed.
    /// (LoudKit.LoudKitError error 0.)". Every asset path, every shape
    /// mismatch, every "export with tools/export_coreml.py" hint is discarded
    /// at exactly the moment someone needs it.
    public var errorDescription: String? { description }
}

/// Identifies the text funnel: what its code does, and what data it reads.
///
/// The digest is of this port's own copies of the funnel data files, so a copy
/// that has drifted from the reference produces a different fingerprint and the
/// engine refuses to start, rather than silently speaking something else.
public struct TextConfig: Sendable, Equatable {
    /// Bumped when the funnel's passes change what they emit for text they
    /// already handled. A new language or table moves `grammar` on its own.
    public var recipe: String = "funnel-6"
    /// Digest of the three shared funnel data files, in the order
    /// `numbers.json`, `pl_en_respell.json`, `numerals.json`. See
    /// `Numbers.grammarDigest`.
    public var grammar: String

    /// Both halves default to what this build is, so the common construction
    /// describes the funnel that is actually going to run.
    public init(recipe: String = "funnel-6", grammar: String = Numbers.grammarDigest) {
        self.recipe = recipe
        self.grammar = grammar
    }
}
