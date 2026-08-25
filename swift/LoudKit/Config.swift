import CryptoKit
import Foundation
import LoudKitText

/// What the engine computes, and how fast it gets there — kept apart, exactly
/// as in `loudkit/config.py`. `AlgorithmConfig` is identical on every backend
/// *and in every language*: the fingerprint here is computed over the same
/// canonical JSON the Python side hashes, so a Swift engine and a Python
/// engine can refuse to be compared before they disagree about what they are
/// computing — which is how the founding guidance defect (mel corr 0.979
/// between two "identical" pipelines) becomes a loud error instead of a
/// campaign-long mystery.

public enum GuidanceMode: String, Sendable {
    /// One estimator call per Euler step. Correct for the guidance-distilled
    /// student — the only mode the packed checkpoint supports.
    case singlePath = "single_path"
    /// Teacher-only CFG. Running it on the distilled student applies guidance
    /// twice (EXP-016); `MelDecoder` refuses it.
    case cfgDualPath = "cfg_dual_path"
}

public struct SamplingConfig: Sendable, Equatable {
    public var temperature: Double = 0.8
    public var repetitionPenalty: Double = 1.2
    public var minP: Double = 0.05
    public var maxNewTokens: Int = 255
    /// Tokens exempt from *both* the repetition penalty and the min_p floor.
    /// A reader pauses repeatedly; penalising silence suppresses pausing
    /// (measured: pause ratio 0.112 -> 0.085 with a plausible wrong list).
    public var silenceTokenIds: [Int] = []
    /// EOS floor (len-prior gate): the stop token is masked until
    /// `max(minTokensFloor, floor(nTextTokens * minTokensTextRatio))`.
    public var minTokensFloor: Int = 0
    public var minTokensTextRatio: Double = 0.0

    public init() {}
}

/// How text longer than one window is split. Algorithm layer — where the
/// splits fall and what each piece is conditioned on decide where the reader
/// breathes, so they are configuration shared with Python, not caller whim.
///
/// Where `Chunking.splitText` decides to cut, and how a chunk is conditioned on
/// the previous one. The values are load-bearing: the fingerprint hashes them,
/// and `Engine.stream`/`synthesizeLong` apply them (see `Chunking.swift`).
public struct ChunkConfig: Sendable, Equatable {
    public var enabled: Bool = true
    public var maxTokens: Int = 255
    /// Speech tokens from the previous chunk fed back as context. Zero means
    /// independent chunks, which stutters at joins (the F0 contour restarts
    /// ~74 Hz at the join on the reference voice); 6 tokens of context carry
    /// it across (~7 Hz, a natural phrase boundary).
    public var prefixTokens: Int = 6
    /// Split candidates, strongest first.
    public var splitOn: [String] = [". ", "! ", "? ", "; ", ", "]

    public init() {}
}

public struct WindowConfig: Sendable, Equatable {
    public var maxSpeechTokens: Int = 255
    public var staticLength: Int?
    /// Pad token for unused static slots. The shipped engine pads with
    /// silence unit 4254 — padding with token 0 bleeds +3 dB of high-band mel
    /// energy into the tail through the encoder's attention.
    public var padTokenId: Int?
    public var staticPromptTokens: Int?

    public init() {}
}

public struct AlgorithmConfig: Sendable, Equatable {
    public var recipeVersion: String = "loudkit-1"
    public var guidance: GuidanceMode = .singlePath
    public var guidanceRate: Double = 0.0
    public var eulerSteps: Int = 2
    public var eulerGrid: [Double]?
    public var sampling = SamplingConfig()
    public var window = WindowConfig()
    public var chunking = ChunkConfig()
    /// The artifact detectors. They remove tokens, so they change the audio
    /// and are read from the manifest for the same reason the joins are: a
    /// backend that re-guesses where a chunk ended cuts somewhere else, and
    /// the difference is a hallucinated word that either does or does not
    /// reach a listener.
    public var postprocess = PostprocessConfig()

    /// The funnel's identity — its code version and the digest of the grammar
    /// file this port reads. In the fingerprint because the funnel decides what
    /// string the model is handed, and therefore what it says.
    public var text = TextConfig()
    public var sampleRate: Int = 24_000
    public var tokenRateHz: Double = 25.0
    public var speechVocabSize: Int = 8194
    public var startSpeechToken: Int = 6561
    public var stopSpeechToken: Int = 6562

    public init() {}

    /// The Euler time grid: explicit if configured, else the cosine schedule
    /// `t_i = 1 − cos(i/K · π/2)` — one formula, shared with
    /// `loudkit.models.flow.time_grid` (the fixture pins the values).
    public func timeGrid() -> [Double] {
        if let grid = eulerGrid { return grid }
        let k = Double(eulerSteps)
        return (0...eulerSteps).map { 1.0 - Foundation.cos(Double($0) / k * Double.pi / 2.0) }
    }

    public func eosFloor(nTextTokens: Int) -> Int {
        max(sampling.minTokensFloor, Int(Double(nTextTokens) * sampling.minTokensTextRatio))
    }

    // MARK: identity

    /// The fingerprint schema version — bump in lockstep with Python's
    /// `loudkit.config.FINGERPRINT_SCHEMA`.
    public static let fingerprintSchema = 1

    /// The exact string that gets hashed — `AlgorithmConfig.canonical_form()`
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
                    if ch.value < 0x20 {
                        out += String(format: "\\u%04x", ch.value)
                    } else {
                        out.unicodeScalars.append(ch)
                    }
                }
            }
            return out + "\""
        }
        func opt(_ v: Int?) -> String { v.map(String.init) ?? "null" }

        let grid = eulerGrid.map { "[" + $0.map(num).joined(separator: ",") + "]" } ?? "null"
        let sil = "[" + sampling.silenceTokenIds.map(String.init).joined(separator: ",") + "]"
        let splitOn = "[" + chunking.splitOn.map(str).joined(separator: ",") + "]"
        let chunkingJSON = "{"
            + "\"enabled\":\(chunking.enabled),"
            + "\"max_tokens\":\(chunking.maxTokens),"
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
        // audio — exactly the silent drift a whole-config hash exists to catch.
        let pp = postprocess
        let postprocessJSON = "{"
            + "\"ceiling_slack_tokens\":\(pp.ceilingSlackTokens),"
            + "\"ceiling_speech_per_text_token\":\(num(pp.ceilingSpeechPerTextToken)),"
            + "\"desperation_band_floor\":\(pp.desperationBandFloor),"
            + "\"desperation_band_ratio\":\(num(pp.desperationBandRatio)),"
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
            + "\"repetition_max_period\":\(pp.repetitionMaxPeriod),"
            + "\"repetition_min_cycles\":\(pp.repetitionMinCycles),"
            + "\"repetition_min_span\":\(pp.repetitionMinSpan),"
            + "\"retry_max_attempts\":\(pp.retryMaxAttempts),"
            + "\"trailing_filler_threshold\":\(num(pp.trailingFillerThreshold)),"
            + "\"trailing_silence_run_tokens\":\(pp.trailingSilenceRunTokens)"
            + "}"
        let body = "{"
            + "\"chunking\":\(chunkingJSON),"
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

    /// First 16 hex chars of SHA-256 over the canonical form — comparable
    /// with `AlgorithmConfig.fingerprint()` on the Python side.
    public func fingerprint() -> String {
        let digest = SHA256.hash(data: Data(canonicalForm().utf8))
        return digest.map { String(format: "%02x", $0) }.joined().prefix(16).description
    }

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
        // while Python (`config.py`) parsed and honoured them — two engines,
        // one manifest, different chunk boundaries and therefore different
        // audio.
        if let chunking = manifest["chunking"] as? [String: Any] {
            if let enabled = chunking["enabled"] as? Bool {
                config.chunking.enabled = enabled
            }
            if let maxTokens = chunking["max_tokens"] as? Int {
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
            if let prefix = chunking["prefix_tokens"] as? Int {
                config.chunking.prefixTokens = prefix
            }
            if let splitOn = chunking["split_on"] {
                // An array of separators. A bare string is eight separators of
                // one character each once Swift iterates it, which is not what
                // any manifest means — refuse rather than reinterpret.
                guard let list = splitOn as? [String] else {
                    throw LoudKitError.manifest(
                        "chunking.split_on must be a list of strings, got "
                        + "\(type(of: splitOn))")
                }
                config.chunking.splitOn = list
            }
        }
        config.postprocess = try Self.postprocess(from: manifest)
        let guidanceRaw = manifest["guidance"] as? String ?? "single_path"
        guard let guidance = GuidanceMode(rawValue: guidanceRaw) else {
            throw LoudKitError.manifest("unknown guidance mode \(guidanceRaw)")
        }
        config.guidance = guidance
        config.guidanceRate = (manifest["guidance_rate"] as? NSNumber)?.doubleValue ?? 0.0
        config.eulerSteps = (manifest["n_cfm_timesteps"] as? NSNumber)?.intValue ?? 2
        config.sampleRate = (manifest["sample_rate"] as? NSNumber)?.intValue ?? 24_000
        // Python refuses a manifest with a non-positive `sample_rate` and the other four
        // took it: every duration this engine reports is `samples / sample_rate`, so a
        // zero divides by zero and a negative reports negative seconds. A rate is the one
        // manifest field whose wrongness is not caught by any shape.
        guard config.sampleRate > 0 else {
            throw LoudKitError.manifest("sample_rate must be > 0: \(config.sampleRate)")
        }
        config.speechVocabSize = (manifest["speech_vocab_size"] as? NSNumber)?.intValue ?? 8194
        if let speech = manifest["speech_tokens"] as? [String: Any] {
            config.startSpeechToken = (speech["start"] as? NSNumber)?.intValue ?? 6561
            config.stopSpeechToken = (speech["stop"] as? NSNumber)?.intValue ?? 6562
        }

        var sampling = SamplingConfig()
        if let defaults = manifest["sampling_defaults"] as? [String: Any] {
            sampling.temperature = (defaults["temperature"] as? NSNumber)?.doubleValue ?? 0.8
            sampling.repetitionPenalty =
                (defaults["repetition_penalty"] as? NSNumber)?.doubleValue ?? 1.2
            sampling.minP = (defaults["min_p"] as? NSNumber)?.doubleValue ?? 0.05
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
            sampling.maxNewTokens = (defaults["max_new_tokens"] as? NSNumber)?.intValue ?? 255
        }
        sampling.silenceTokenIds =
            (manifest["silence_token_ids"] as? [NSNumber])?.map(\.intValue) ?? []

        guard let eos = manifest["eos_floor"] as? [String: Any],
              let win = manifest["window"] as? [String: Any] else {
            throw LoudKitError.manifest(
                "checkpoint manifest carries no window/eos_floor — this is an "
                + "un-amended pack; run tools/amend_manifest.py. LoudKit refuses "
                + "to re-guess the window recipe: two backends framing the "
                + "prompt differently are different algorithms.")
        }
        sampling.minTokensFloor = (eos["min_tokens_floor"] as? NSNumber)?.intValue ?? 0
        sampling.minTokensTextRatio =
            (eos["min_tokens_text_ratio"] as? NSNumber)?.doubleValue ?? 0.0
        guard sampling.minTokensFloor >= 0 else {
            throw LoudKitError.manifest(
                "min_tokens_floor must be >= 0: \(sampling.minTokensFloor)")
        }
        guard sampling.minTokensTextRatio >= 0 else {
            throw LoudKitError.manifest(
                "min_tokens_text_ratio must be >= 0: \(sampling.minTokensTextRatio)")
        }
        config.sampling = sampling

        var window = WindowConfig()
        window.maxSpeechTokens = (win["max_speech_tokens"] as? NSNumber)?.intValue ?? 255
        window.staticLength = (win["static_length"] as? NSNumber)?.intValue
        window.padTokenId = (win["pad_token_id"] as? NSNumber)?.intValue
        window.staticPromptTokens = (win["static_prompt_tokens"] as? NSNumber)?.intValue
        config.window = window
        return config
    }

    /// The `postprocess` block, or the shipping detectors when it is absent.
    ///
    /// An unknown mode is refused rather than defaulted: it would trim where
    /// the manifest said not to, under a matching `recipe_version`.
    private static func postprocess(from manifest: [String: Any]) throws -> PostprocessConfig {
        var cfg = PostprocessConfig()
        guard let block = manifest["postprocess"] as? [String: Any] else { return cfg }

        if let raw = block["mode"] as? String {
            guard let mode = Postprocess.Mode(rawValue: raw) else {
                throw LoudKitError.manifest(
                    "unknown postprocess mode \(raw); expected one of "
                    + Postprocess.Mode.allCases.map(\.rawValue).joined(separator: ", "))
            }
            cfg.mode = mode
        }
        func d(_ key: String, _ current: Double) -> Double {
            (block[key] as? NSNumber)?.doubleValue ?? current
        }
        func i(_ key: String, _ current: Int) -> Int {
            (block[key] as? NSNumber)?.intValue ?? current
        }
        cfg.ceilingSpeechPerTextToken =
            d("ceiling_speech_per_text_token", cfg.ceilingSpeechPerTextToken)
        cfg.ceilingSlackTokens = i("ceiling_slack_tokens", cfg.ceilingSlackTokens)
        cfg.trailingFillerThreshold = d("trailing_filler_threshold", cfg.trailingFillerThreshold)
        cfg.trailingSilenceRunTokens =
            i("trailing_silence_run_tokens", cfg.trailingSilenceRunTokens)
        cfg.desperationBandRatio = d("desperation_band_ratio", cfg.desperationBandRatio)
        cfg.desperationBandFloor = i("desperation_band_floor", cfg.desperationBandFloor)
        cfg.fillerMinEosProbability =
            d("filler_min_eos_probability", cfg.fillerMinEosProbability)
        cfg.fillerMaxSpeechAfterRun =
            i("filler_max_speech_after_run", cfg.fillerMaxSpeechAfterRun)
        cfg.desperationSpeechPerTextToken =
            d("desperation_speech_per_text_token", cfg.desperationSpeechPerTextToken)
        cfg.desperationMinTextTokens =
            i("desperation_min_text_tokens", cfg.desperationMinTextTokens)
        cfg.endedTailSilenceRun = i("ended_tail_silence_run", cfg.endedTailSilenceRun)
        cfg.endedTailBlipMax = i("ended_tail_blip_max", cfg.endedTailBlipMax)
        cfg.endedTailWordMax = i("ended_tail_word_max", cfg.endedTailWordMax)
        cfg.endedTailKeep = i("ended_tail_keep", cfg.endedTailKeep)
        cfg.echoStrongEosProbability =
            d("echo_strong_eos_probability", cfg.echoStrongEosProbability)
        cfg.echoStrongMaxTail = i("echo_strong_max_tail", cfg.echoStrongMaxTail)
        cfg.echoStrongMinPositionPct =
            i("echo_strong_min_position_pct", cfg.echoStrongMinPositionPct)
        cfg.echoWeakEosProbability = d("echo_weak_eos_probability", cfg.echoWeakEosProbability)
        cfg.echoWeakMaxTail = i("echo_weak_max_tail", cfg.echoWeakMaxTail)
        cfg.echoWeakMinPositionPct =
            i("echo_weak_min_position_pct", cfg.echoWeakMinPositionPct)
        // The six this wall was missing. Python reads its fields off the
        // dataclass precisely so that a new constant cannot be left out of a
        // hand-written list; the four ports write the list by hand, and every
        // one of them had drifted six fields behind. Defaults matched, so
        // nothing sounded wrong — until a checkpoint set one of them, at which
        // point the manifest declares one recipe and four engines run another.
        cfg.dropoutMinTokens = i("dropout_min_tokens", cfg.dropoutMinTokens)
        cfg.retryMaxAttempts = i("retry_max_attempts", cfg.retryMaxAttempts)
        cfg.pacingTolerance = d("pacing_tolerance", cfg.pacingTolerance)
        cfg.repetitionMaxPeriod = i("repetition_max_period", cfg.repetitionMaxPeriod)
        cfg.repetitionMinCycles = i("repetition_min_cycles", cfg.repetitionMinCycles)
        cfg.repetitionMinSpan = i("repetition_min_span", cfg.repetitionMinSpan)
        return cfg
    }
}

/// How a backend gets there. Free to differ from the Python side; declared so
/// the parity table can label every row.
public struct ExecutionConfig: Sendable {
    /// The token generator runs natively on the CPU in fp32 (weights upcast
    /// from the packed fp16). This is the measured-right placement on Apple
    /// silicon — the autoregressive stage is faster on CPU than GPU/ANE at
    /// batch one — and fp32 is the declared precision of the conformance
    /// fixture ("same precision, same tokens").
    public var tokenGeneratorPrecision: String = "fp32"
    /// Compute units per CoreML stage. The estimator is the ANE citizen; the
    /// encoder and vocoder are fp32 graphs and stay on CPU.
    public var encoderComputeUnits: ComputeUnits = .cpuOnly
    public var estimatorComputeUnits: ComputeUnits = .cpuAndNeuralEngine
    public var vocoderComputeUnits: ComputeUnits = .cpuOnly

    public enum ComputeUnits: String, Sendable {
        case cpuOnly, cpuAndNeuralEngine, all
    }

    public init() {}

    public func describe() -> String {
        "exec[swift-native-t3(\(tokenGeneratorPrecision)) "
            + "coreml enc=\(encoderComputeUnits.rawValue) "
            + "est=\(estimatorComputeUnits.rawValue) voc=\(vocoderComputeUnits.rawValue)]"
    }
}

public enum LoudKitError: Error, CustomStringConvertible, LocalizedError {
    case manifest(String)
    case asset(String)
    case shape(String)
    case prediction(String)
    /// `shouldCancel` returned true and the partial utterance was discarded.
    ///
    /// Thrown rather than returned as a short `Result`, because a caller who
    /// interrupted wants to know that nothing was produced — a `Result` with a
    /// fraction of the audio is indistinguishable from a short utterance.
    case cancelled

    public var description: String {
        switch self {
        case .manifest(let s): return "manifest: \(s)"
        case .asset(let s): return "asset: \(s)"
        case .shape(let s): return "shape: \(s)"
        case .prediction(let s): return "prediction: \(s)"
        case .cancelled: return "cancelled before the audio was rendered"
        }
    }

    /// LocalizedError, not just CustomStringConvertible: without it the
    /// messages above never reach the user. Swift bridges a plain `Error` to
    /// NSError, and `localizedDescription` — what a caller prints, what an
    /// alert shows — becomes "The operation couldn't be completed.
    /// (LoudKit.LoudKitError error 0.)". Every asset path, every shape
    /// mismatch, every "export with tools/export_coreml.py" hint is discarded
    /// at exactly the moment someone needs it.
    public var errorDescription: String? { description }
}


/// Identifies the text funnel: what its code does, and what data it reads.
///
/// The digest is of *this* port's copy of `numbers.json`, so a copy that has
/// drifted from the reference produces a different fingerprint and the engine
/// refuses to start, rather than silently speaking something else.
public struct TextConfig: Sendable, Equatable {
    /// Bumped when the funnel's passes change what they emit for text they
    /// already handled. A new language or table moves `grammar` on its own.
    public var recipe: String = "funnel-2"
    /// Digest of the shared grammar file, `numbers.json`.
    ///
    /// Hashes the grammar and **not** `pl_en_respell.json`, which is also a
    /// funnel input and also moves the spoken tokens: a build whose lexicon has
    /// drifted speaks different words and reports the same sixteen hex digits.
    /// Left as it is deliberately — all five implementations hash the same one
    /// file, so widening it here alone would break the identity it exists to
    /// establish. Widening it is a five-way change plus a fixture
    /// regeneration, and it is worth making.
    public var grammar: String

    public init(recipe: String = "funnel-2", grammar: String = Numbers.grammarDigest) {
        self.recipe = recipe
        self.grammar = grammar
    }
}
