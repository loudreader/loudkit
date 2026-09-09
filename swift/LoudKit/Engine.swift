import CoreML
import Foundation
import LoudKitText

/// The engine: the same five-component composition as `loudkit.engine.Engine`,
/// with the same public shape:
///
///     let engine = try await Engine.load("loudreader/loudr-1")
///     let voice  = try engine.voice(named: "joe")
///     let result = try engine.synthesize("Hello there.", voice: voice, seed: 7)
///     try result.saveWav("hello.wav")
///
/// Seeds are derived per stage with the identical splitting constants, so the
/// sampler, the flow prior and the vocoder excitation consume the same Philox
/// numbers as the Python engine given the same user seed.
public final class Engine {
    static let streamFlow: UInt64 = 1
    static let streamVocoder: UInt64 = 2
    /// Where the seeds of chunks *after the first* start, clear of the
    /// per-stage streams. Chunk 0 draws the caller's seed itself, so a text
    /// that fits one window renders the same through ``synthesizeWindow`` and
    /// through ``synthesize``, in every implementation.
    static let streamChunkBase: UInt64 = 16

    /// Mirrors `_STREAM_RESPLIT` in `loudkit.window`: the second half of a
    /// re-split chunk draws from its own stream off the chunk's seed. Chunk
    /// streams run from ``streamChunkBase`` upwards with no ceiling, so there
    /// is no room above them to claim; deriving off the chunk seed leaves only
    /// the values already drawn from it to avoid, which are the flow at 1, the
    /// vocoder at 2, and the retry ladder from 8 up.
    static let streamResplit: UInt64 = 4096

    /// What this engine computes, and the fingerprint it reports.
    public let algorithm: AlgorithmConfig
    /// Where each stage runs. Free to differ between builds without moving the
    /// fingerprint.
    public let execution: ExecutionConfig
    /// Text to token ids, with the language tag.
    public let frontend: TextFrontend
    /// The autoregressive stage: text tokens to speech tokens.
    public let tokenGenerator: TokenGenerator
    /// The flow decoder: speech tokens to a mel spectrogram.
    public let melDecoder: MelDecoder
    /// The vocoder: a mel spectrogram to a waveform.
    public let vocoder: Vocoder
    /// Where this engine was loaded from, kept so `withExecution` can rebuild
    /// the CoreML stages without re-reading the generator weights.
    private var checkpointURL: URL?
    private var assetsURL: URL?
    /// The release this engine was opened from, when it was one: what
    /// `voiceNames`, `voice(named:)` and `enroll` read.
    var bundle: ModelBundle?

    /// Wall time in each of the three stages, in seconds.
    public struct StageTimings: Sendable, Equatable {
        /// Token generation.
        public let tokens: Double
        /// Flow decoding to mel.
        public let mel: Double
        /// Vocoding to a waveform.
        public let audio: Double
        /// The three added up.
        public var total: Double { tokens + mel + audio }
        /// Audio seconds produced per second spent. Above one is faster than
        /// real time; a zero total reports infinity rather than dividing by it.
        public func rtf(audioSeconds: Double) -> Double {
            total > 0 ? audioSeconds / total : .infinity
        }
    }

    /// One finished synthesis: the waveform, and everything about how it was
    /// produced that a caller may need to reproduce or judge it.
    public struct Result: Sendable {
        /// The waveform, mono float32 in [-1, 1] at `sampleRate`.
        public let audio: [Float]
        /// The acoustic speech tokens that were rendered, control tokens
        /// stripped. Pass these as `previousTokens` to continue the contour.
        public let tokens: [Int]
        /// The mel spectrogram the vocoder read, 80 bins, frame major.
        public let mel: [Float]
        /// Frames in `mel`, since `mel` is flat.
        public let melFrames: Int
        /// The seed this render was drawn from. Same seed, same audio.
        public let seed: UInt64
        /// Sample rate of `audio`.
        public let sampleRate: Int
        /// Wall time in each stage.
        public let timings: StageTimings
        /// The algorithm fingerprint at render time, so a stored result can be
        /// compared with a later one without guessing what produced it.
        public let algorithmFingerprint: String
        /// Some chunk stopped at a cap rather than a stop token, so it is cut
        /// off mid-sentence. ORed across the passage.
        public let hitTokenCap: Bool
        /// What the artifact detectors concluded, one entry per chunk.
        ///
        /// A list rather than a single verdict because a passage is many chunks
        /// and they fail independently: one hallucinated tail in the middle of
        /// six clean ones is the case worth seeing, and an aggregate hides it.
        public var inspections: [Postprocess.Inspection] = []

        /// The time-stretch this render was asked for. `1.0` means none was
        /// applied: the waveform came straight out of the vocoder.
        ///
        /// Recorded rather than inferred, because it cannot be inferred: a
        /// stretched reading and a naturally faster one are the same numbers
        /// afterwards, and `duration` alone cannot tell a caller which it is
        /// holding.
        public var speed: Double = 1.0

        /// Where each chunk lands in `audio`, and where its words probably do.
        ///
        /// One entry per chunk, in order and adjacent: chunk *k*'s `end` is the
        /// same `Double` as chunk *k+1*'s `start`, and the last `end` is
        /// `duration`. A single-window synthesis gets one entry covering the
        /// whole result.
        ///
        /// Chunk boundaries are exact: they are sample offsets, which the
        /// engine already knows because it concatenated the chunks. The
        /// per-word times inside each entry are an **estimate**; read
        /// ``Timing`` before building anything that depends on them.
        ///
        /// Measured on the returned waveform, so they already account for
        /// `speed`.
        public var chunks: [ChunkTiming] = []

        /// Any chunk came back marked suspect. The five things that mark one
        /// are listed on ``Postprocess/Inspection/suspect``; whether anything
        /// was removed depends on which, so read `inspections` rather than
        /// assuming. Not an error: a report, and the engine's signal to retry.
        public var suspect: Bool { inspections.contains { $0.suspect } }

        /// Length of `audio` in seconds.
        public var duration: Double { Double(audio.count) / Double(sampleRate) }

        /// Write a 32-bit float WAV, for the conformance harness: rounding
        /// to 16 bits would put a quantisation floor under a correlation
        /// gated at 0.999. `saveWav` is the file to hand a person.
        public func saveFloat32Wav(to url: URL) throws {
            var data = Data()
            func append<T>(_ value: T) {
                var v = value
                withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
            }
            let byteCount = audio.count * 4
            // RIFF carries its sizes in 32 bits, so a WAV cannot describe more
            // than about 4 GB, roughly twelve hours at 24 kHz mono float32.
            // `UInt32(...)` on a larger value does not truncate in Swift, it
            // traps: the process dies with no message and no file written. Say
            // what happened instead.
            guard 36 + byteCount <= Int(UInt32.max) else {
                throw LoudKitError.shape(
                    "\(audio.count) samples is \(byteCount) bytes of audio; a WAV "
                    + "header cannot describe more than \(UInt32.max): split the "
                    + "passage and write several files"
                )
            }
            data.append(contentsOf: Array("RIFF".utf8))
            append(UInt32(36 + byteCount).littleEndian)
            data.append(contentsOf: Array("WAVE".utf8))
            data.append(contentsOf: Array("fmt ".utf8))
            append(UInt32(16).littleEndian)
            append(UInt16(3).littleEndian)  // IEEE float
            append(UInt16(1).littleEndian)
            append(UInt32(sampleRate).littleEndian)
            append(UInt32(sampleRate * 4).littleEndian)
            append(UInt16(4).littleEndian)
            append(UInt16(32).littleEndian)
            data.append(contentsOf: Array("data".utf8))
            append(UInt32(byteCount).littleEndian)
            audio.withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
            try data.write(to: url)
        }
    }

    init(algorithm: AlgorithmConfig, execution: ExecutionConfig, frontend: TextFrontend,
         tokenGenerator: TokenGenerator, melDecoder: MelDecoder, vocoder: Vocoder) {
        self.algorithm = algorithm
        self.execution = execution
        self.frontend = frontend
        self.tokenGenerator = tokenGenerator
        self.melDecoder = melDecoder
        self.vocoder = vocoder
    }

    /// Build an engine from a packed (and amended) checkpoint plus the CoreML
    /// packages exported by `tools/export_coreml.py`.
    ///
    /// - Parameters:
    ///   - checkpoint: `loudr-1.safetensors` (tokenizer.json beside it).
    ///   - coremlAssets: directory holding `flow_encoder.mlpackage`,
    ///     `flow_estimator.mlpackage`, `vocoder.mlpackage`. Defaults to
    ///     `coreml/` beside the checkpoint.
    public static func load(
        checkpoint checkpointURL: URL,
        coremlAssets: URL? = nil,
        execution: ExecutionConfig = ExecutionConfig()
    ) throws -> Engine {
        let checkpoint = try Checkpoint(url: checkpointURL)
        let algorithm = try checkpoint.algorithm()
        let assets = coremlAssets ?? checkpoint.coremlAssetsURL
        try checkpoint.verifyCoreMLExport(at: assets, algorithm: algorithm)
        func package(_ stem: String) throws -> URL { try stageURL(in: assets, stem) }
        func units(_ u: ExecutionConfig.ComputeUnits) -> MLComputeUnitsWrapper { .init(u) }

        let frontend = try TextFrontend(tokenizerURL: checkpoint.tokenizerURL)
        let generator = try TokenGenerator(checkpoint: checkpoint, config: algorithm)
        let encoder = try MLHelpers.loadModel(
            packageURL: package("flow_encoder"), computeUnits: units(execution.encoderComputeUnits).value)
        let estimator = try MLHelpers.loadModel(
            packageURL: package("flow_estimator"), computeUnits: units(execution.estimatorComputeUnits).value)
        let hift = try MLHelpers.loadModel(
            packageURL: package("vocoder"), computeUnits: units(execution.vocoderComputeUnits).value)
        let melDecoder = try MelDecoder(
            config: algorithm, encoder: encoder, estimator: estimator,
            spkWeight: try checkpoint.store.floats("s3gen.flow.spk_embed_affine_layer.weight"),
            spkBias: try checkpoint.store.floats("s3gen.flow.spk_embed_affine_layer.bias"))
        let vocoder = Vocoder(config: algorithm, hift: hift)
        let engine = Engine(
            algorithm: algorithm, execution: execution, frontend: frontend,
            tokenGenerator: generator, melDecoder: melDecoder, vocoder: vocoder)
        engine.checkpointURL = checkpointURL
        engine.assetsURL = assets
        return engine
    }

    /// The same checkpoint under a different `ExecutionConfig`: rebuilds only
    /// the CoreML stages, sharing the already-loaded generator and frontend.
    /// Execution is the axis that is *allowed* to differ, and reloading
    /// ~0.7 GB of generator weights to move the renderer between compute
    /// units would make placement comparisons needlessly expensive.
    public func withExecution(_ execution: ExecutionConfig) throws -> Engine {
        guard let checkpointURL, let assetsURL else {
            throw LoudKitError.asset("engine was not built by Engine.load: no checkpoint to rebuild from")
        }
        func units(_ u: ExecutionConfig.ComputeUnits) -> MLComputeUnitsWrapper { .init(u) }
        func package(_ stem: String) throws -> URL { try Self.stageURL(in: assetsURL, stem) }
        let checkpoint = try Checkpoint(url: checkpointURL)
        let encoder = try MLHelpers.loadModel(
            packageURL: package("flow_encoder"), computeUnits: units(execution.encoderComputeUnits).value)
        let estimator = try MLHelpers.loadModel(
            packageURL: package("flow_estimator"), computeUnits: units(execution.estimatorComputeUnits).value)
        let hift = try MLHelpers.loadModel(
            packageURL: package("vocoder"), computeUnits: units(execution.vocoderComputeUnits).value)
        let melDecoder = try MelDecoder(
            config: algorithm, encoder: encoder, estimator: estimator,
            spkWeight: try checkpoint.store.floats("s3gen.flow.spk_embed_affine_layer.weight"),
            spkBias: try checkpoint.store.floats("s3gen.flow.spk_embed_affine_layer.bias"))
        let engine = Engine(
            algorithm: algorithm, execution: execution, frontend: frontend,
            tokenGenerator: tokenGenerator, melDecoder: melDecoder,
            vocoder: Vocoder(config: algorithm, hift: hift))
        engine.checkpointURL = checkpointURL
        engine.assetsURL = assetsURL
        engine.bundle = bundle
        return engine
    }

    private func release(_ what: String) throws -> ModelBundle {
        guard let bundle else {
            throw LoudKitError.asset(
                "this engine was opened with Engine.load(checkpoint:), which has no "
                + "release to take \(what) from")
        }
        return bundle
    }

    /// The names `voice(named:)` will accept, sorted. Empty for an engine
    /// opened with `Engine.load(checkpoint:)`.
    public var voiceNames: [String] { bundle?.voiceNames ?? [] }

    /// One shipped voice by name, such as `"joe"`.
    public func voice(named name: String) throws -> VoiceProfile {
        try release("voices").voice(named: name)
    }

    /// Clone a voice from a recording on disk: anything AVFoundation opens.
    /// `language` is what the voice reads in by default.
    ///
    /// The three enrollment packages are not in a plain fetch. An engine
    /// loaded by repo id fetches them into its own cache directory the first
    /// time; one loaded from a directory needs a fetch made with
    /// `cloning: true`.
    public func enroll(contentsOf url: URL, name: String = "", language: String = "en")
        async throws -> VoiceProfile {
        let bundle = try release("the enrollment packages")
        try await bundle.fetchCloning()
        return try bundle.enroller().enroll(contentsOf: url, name: name, language: language)
    }

    /// A stage may be present as the exported .mlpackage or as the
    /// precompiled .mlmodelc only (an app bundle ships the compiled form and
    /// skips the on-device CoreML compile). The package wins when both exist,
    /// and `MLHelpers.loadModel` then compiles it into a temporary directory on
    /// every load: the compiled sibling is used only when it is the one form
    /// present, so a package and a compiled copy of the same stem cannot drift
    /// apart unnoticed.
    private static func stageURL(in assets: URL, _ stem: String) throws -> URL {
        for ext in ["mlpackage", "mlmodelc"] {
            let url = assets.appendingPathComponent("\(stem).\(ext)")
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        throw LoudKitError.asset(
            "\(stem).mlpackage/.mlmodelc not found in \(assets.path): "
            + "export with tools/export_coreml.py")
    }

    /// One line for a log: what this engine computes, then where it runs.
    public func describe() -> String {
        "\(algorithm.describe()) | \(execution.describe())"
    }

    // MARK: synthesis

    /// The one path that produces speech tokens.
    ///
    /// Single-shot and streaming both go through it so they cannot drift: the
    /// generation ceiling, the stop-token observation and the artifact
    /// detectors are applied once, here, rather than twice and eventually
    /// differently.
    ///
    /// `isTerminal` says whether this chunk ends the passage. A continuation
    /// chunk has no sentence end, so its stop peak means nothing and its
    /// trailing pause is the sentence's rhythm rather than dead air, the
    /// detectors that cut a tail are told so and hold off.
    private func generateInspected(
        textTokens: [Int], voice: VoiceProfile, seed: UInt64, prefix: [Int],
        isTerminal: Bool, maxNewTokens: Int?, onStep: (() -> Void)?,
        shouldCancel: (() -> Bool)?
    ) throws -> (
        tokens: [Int], inspection: Postprocess.Inspection, hitCap: Bool, hitWindow: Bool,
        generatedCount: Int
    ) {
        let pp = algorithm.postprocess
        let floor = algorithm.eosFloor(nTextTokens: textTokens.count)
        // Refused rather than passed through. A negative cap reached the
        // generator, which produced nothing and returned an empty result with
        // no error anywhere. The reference makes the same refusal for the
        // same reason: a count of tokens is at least one.
        if let asked = maxNewTokens, asked < 1 {
            throw LoudKitError.shape("maxNewTokens must be >= 1, got \(asked)")
        }
        var cap = maxNewTokens ?? algorithm.sampling.maxNewTokens
        if pp.mode != .off {
            // Applied during generation, not after it: the tokens past the
            // ceiling cost real time on a device and are certain to be
            // discarded. It only ever stops a row that was going to run away.
            cap = min(
                cap,
                Postprocess.ceiling(
                    forTextTokens: textTokens.count, config: pp,
                    window: algorithm.window.maxSpeechTokens))
        }

        // Selective re-roll: a window whose verdict is unfixable (dropout
        // content missing, or suspect, certainly wrong with nowhere to cut) is
        // regenerated from a derived seed, up to retryMaxAttempts times. Only
        // condemned windows pay; the ladder is a pure function of the caller's
        // seed, so the same seed still gives the same audio, retries included.
        var gen: [Int] = []
        var verdict = Postprocess.Inspection(keep: 0, reason: .clean, suspect: false)
        var hitCap = false
        var hitWindow = false
        // When the ladder exhausts with every attempt condemned, the attempt
        // that ships is the *best* seen, not the last: fewest tokens in the
        // true-silence set, integer and portable, like the detectors.
        // Measured: on the worst voices 30% of condemned fires exhaust the
        // ladder, and keeping the last attempt shipped rows worse than the
        // first. The render census gates the count where the checkpoint
        // carries one; the configured silence list is the fallback.
        let deadAir = Set(
            pp.silenceRenderIds.isEmpty
                ? algorithm.sampling.silenceTokenIds : pp.silenceRenderIds)
        var best: (
            count: Int, gen: [Int], verdict: Postprocess.Inspection, hitCap: Bool,
            hitWindow: Bool
        )?
        var attempt = 0
        while true {
            // Retry attempts draw derive(seed, 8 + attempt): clear of the
            // stage streams (1, 2) and below the chunk streams at 16.
            let attemptSeed = attempt == 0 ? seed : Self.derive(seed, 8 + UInt64(attempt))
            let sampler = LRSamplerV1(config: algorithm.sampling, seed: attemptSeed)
            if pp.mode != .off {
                sampler.observeEOS(stopToken: algorithm.stopSpeechToken, floor: floor)
            }
            // Token-level barge-in throws `LoudKitError.cancelled` from the
            // generator, at the poll that fired; a partial row never gets here.
            let generation = try tokenGenerator.generate(
                textTokens: textTokens, voice: voice, sampler: sampler, maxNewTokens: cap,
                prefix: prefix, onStep: onStep, shouldCancel: shouldCancel)

            // `gen` is what the shipped engine calls a row: every token the
            // model committed to, with the stop marker itself excluded.
            // Indices into it are decode-step indices, which is what makes the
            // observed peak comparable against it, so the detectors run here,
            // before `stripSpecials` is free to renumber anything.
            gen = generation.rawTokens
            let ended = gen.last == algorithm.stopSpeechToken
            if ended { gen.removeLast() }

            let peak = sampler.eosPeak
            hitCap = !ended && gen.count >= cap
            // The window, asked separately and asked here, where `gen` is still
            // what the model produced. `cap` is `min(maxNewTokens,
            // ceilingFor(...))`, so `hitCap` cannot tell a filled window from a
            // runaway short text; and the trim below can cut a filled window
            // down to a few tokens, which is how a caller measuring the
            // returned array saw room to spare.
            hitWindow = !ended && gen.count >= algorithm.window.maxSpeechTokens
            verdict = Postprocess.inspect(
                gen,
                request: Postprocess.Request(
                    textTokenCount: textTokens.count, minTokens: floor,
                    eosPeakAt: peak.at, eosPeakProb: peak.probability,
                    ended: ended, isTerminal: isTerminal, hitCeiling: hitCap),
                silence: Set(algorithm.sampling.silenceTokenIds), config: pp)
            let condemned = verdict.reason == .dropout || verdict.suspect
            if !condemned || pp.mode == .off { break }
            let silenceCount = gen.lazy.filter { deadAir.contains($0) }.count
            // Strict `<`, and `Int.max` for "nothing recorded yet": on a tie
            // the earlier attempt stands, so the ladder stays a pure function
            // of the caller's seed with no dependence on iteration order.
            if silenceCount < (best?.count ?? Int.max) {
                best = (silenceCount, gen, verdict, hitCap, hitWindow)
            }
            if attempt >= pp.retryMaxAttempts {
                // Always set: the loop only reaches here after a condemned
                // attempt, and a condemned attempt is recorded just above.
                if let chosen = best { (_, gen, verdict, hitCap, hitWindow) = chosen }
                break
            }
            attempt += 1
        }
        // How many tokens the model committed to, before the trim: the
        // overflow error reports this rather than what survived postprocess.
        let generatedCount = gen.count
        if pp.mode == .trim, verdict.keep < gen.count {
            gen = Array(gen.prefix(verdict.keep))
        }
        let speech = try stripSpecials(gen)
        try Self.requireSpeechProduced(speech)
        return (speech, verdict, hitCap, hitWindow, generatedCount)
    }

    /// Render text that fits one model window, and refuse text that does not.
    ///
    /// `synthesize` is the call for any length; this one is for the
    /// conformance harness and for a caller who wants the refusal. `language`
    /// is `nil` for the voice's own; `speed` is in `[0.5, 2.0]`, pitch
    /// preserved, and `1.0` is an exact bypass; `previousTokens` are the
    /// `Result.tokens` of the call before, so this one continues its pitch
    /// contour; `shouldCancel` is polled on every decode step and throws
    /// `LoudKitError.cancelled` when it returns true.
    public func synthesizeWindow(
        _ text: String, voice: VoiceProfile, seed: UInt64 = 0, language: String? = nil,
        maxNewTokens: Int? = nil, speed: Double = 1.0, previousTokens: [Int]? = nil,
        onStep: (() -> Void)? = nil,
        shouldCancel: (() -> Bool)? = nil
    ) throws -> Result {
        // Both refused here, before the six seconds of generation they would
        // otherwise be discovered after.
        try TimeStretch.validateSpeed(speed)
        let prefix = try carryFrom(previousTokens)
        let language = Self.resolveLanguage(language, voice: voice)
        // The speech funnel, before tokenising: the same `SpeechText.prepared`
        // Python calls `speech_text` and runs in `Engine._generate_window`, on
        // the one path that renders. This module could not reach it while it
        // lived in a separate target, so it encoded raw text: "Rabat 15% na
        // weekend!" gave 17 token ids where Python gives 27 for "Rabat
        // piętnaście procent na łikend!". Different tokens are different
        // speech, for the package the README presents beside Python.
        let prepared = SpeechText.prepared(text, languageId: language)
        // The funnel may remove everything (a footnote marker, an emoji).
        // Refused here, where `synthesize` refuses through an empty split:
        // encoding what is left tokenises the bare language tag and renders
        // `min_tokens_floor` tokens of babble under a caller's own text.
        try Self.requireSomethingToSpeak(prepared)
        let textTokens = try frontend.encode(prepared, language: language)

        let t0 = Date()
        // A single window is the whole passage, so it is terminal.
        let generated = try generateInspected(
            textTokens: textTokens, voice: voice, seed: seed, prefix: prefix, isTerminal: true,
            maxNewTokens: maxNewTokens, onStep: onStep, shouldCancel: shouldCancel)
        let t1 = Date()
        // Discarded, not rendered: every port polls here, between the token
        // phase and the render, because the render is the larger half of the
        // barge-in latency on an edge device.
        if shouldCancel?() == true { throw LoudKitError.cancelled }
        // Refused rather than truncated: one window's audio with the rest of
        // the text unspoken is silent data loss.
        //
        // The *window*, not the cap: `hitCap` is also set by the postprocess
        // length ceiling, which stops a short text that ran away. That text
        // fitted and there is nothing to split.
        //
        // Read from the generation rather than from the returned array:
        // postprocess trims, so a window that filled and was then cut back
        // measured short here and the refusal never fired. The overflow was
        // gone from the evidence, not from the audio.
        if generated.hitWindow {
            throw LoudKitError.windowOverflow(
                tokens: generated.generatedCount, window: algorithm.window.maxSpeechTokens)
        }
        let speech = generated.tokens
        let (mel, frames) = try melDecoder.decode(
            tokens: speech, voice: voice, seed: Self.derive(seed, Self.streamFlow))
        let t2 = Date()
        if shouldCancel?() == true { throw LoudKitError.cancelled }
        let rendered = try vocoder.synthesize(
            mel: mel, frames: frames, seed: Self.derive(seed, Self.streamVocoder))
        let t3 = Date()
        // Last, and after `generateInspected` above rather than before it: the
        // detectors judge pacing by duration per token, and stretching first
        // would move every number they compare against. `speed == 1.0` returns
        // the vocoder's own array, so the default costs nothing and changes no
        // byte. Outside the stage timings for the same reason it is outside the
        // fingerprint: it is delivery, not synthesis.
        let audio = TimeStretch.fadeEdges(
            try TimeStretch.timeStretch(
            rendered, sampleRate: algorithm.sampleRate, speed: speed),
            sampleRate: algorithm.sampleRate, seconds: algorithm.edgeFadeSeconds)

        if shouldCancel?() == true { throw LoudKitError.cancelled }
        return Result(
            audio: audio, tokens: speech, mel: mel, melFrames: frames, seed: seed,
            sampleRate: algorithm.sampleRate,
            timings: StageTimings(
                tokens: t1.timeIntervalSince(t0),
                mel: t2.timeIntervalSince(t1),
                audio: t3.timeIntervalSince(t2)),
            algorithmFingerprint: algorithm.fingerprint(),
            hitTokenCap: generated.hitCap,
            inspections: [generated.inspection],
            speed: speed,
            // One window is one chunk, and it starts at zero: a rendered chunk
            // is its own result and cannot know what preceded it. The text is
            // the post-funnel text, because that is what was tokenised and
            // therefore what the samples say. Timed on `audio`, after the
            // stretch, so there is no `1/speed` correction anywhere.
            chunks: Timing.timeline(
                [ChunkSpan(text: prepared, samples: audio.count, tokens: speech.count)],
                sampleRate: algorithm.sampleRate))
    }

    /// Render a token sequence that already exists: the single most useful
    /// diagnostic when two implementations disagree (it removes sampling from
    /// the comparison and asks only whether the renderer agrees).
    public func synthesizeTokens(
        _ tokens: [Int], voice: VoiceProfile, seed: UInt64 = 0
    ) throws -> Result {
        // A caller's sequence, so it is checked rather than filtered.
        // `stripSpecials` drops ids at or above the limit because the generator
        // legitimately emits them; here they came from outside, and dropping
        // them renders something the caller did not ask for. Empty is refused
        // for the same reason: `stripSpecials` would return an empty row and
        // the renderer would hand back silence. Python refuses both.
        guard !tokens.isEmpty else { throw LoudKitError.noTokensToRender }
        try Self.validateSpeechTokens(tokens, limit: algorithm.startSpeechToken, field: "tokens")
        let speech = try stripSpecials(tokens)
        let t0 = Date()
        let (mel, frames) = try melDecoder.decode(
            tokens: speech, voice: voice, seed: Self.derive(seed, Self.streamFlow))
        let t1 = Date()
        let audio = try vocoder.synthesize(
            mel: mel, frames: frames, seed: Self.derive(seed, Self.streamVocoder))
        let t2 = Date()
        return Result(
            audio: audio, tokens: speech, mel: mel, melFrames: frames, seed: seed,
            sampleRate: algorithm.sampleRate,
            timings: StageTimings(tokens: 0, mel: t1.timeIntervalSince(t0), audio: t2.timeIntervalSince(t1)),
            algorithmFingerprint: algorithm.fingerprint(),
            hitTokenCap: false,
            // No text reached this path, so there are no words to estimate,
            // but the span still covers the whole render, so a caller stitching
            // results does not have to special-case it.
            chunks: Timing.timeline(
                [ChunkSpan(text: "", samples: audio.count, tokens: speech.count)],
                sampleRate: algorithm.sampleRate))
    }

    /// One rendered chunk, handed to a `stream` callback as soon as it exists.
    public struct Chunk: Sendable {
        /// Zero-based position in the split, which is also what the chunk's
        /// seed was derived from.
        public let index: Int
        /// The piece of the passage this chunk speaks, after the speech
        /// funnel: what was tokenised, which is not always what the caller
        /// passed in.
        public let text: String
        /// This chunk's waveform, already faded at both edges and stretched.
        public let audio: [Float]
        /// This chunk's acoustic speech tokens, control tokens stripped.
        public let tokens: [Int]
        /// This chunk's mel spectrogram, 80 bins, frame major.
        public let mel: [Float]
        /// Frames in `mel`, since `mel` is flat.
        public let melFrames: Int
        /// What the artifact detectors concluded about this chunk. Per chunk
        /// rather than aggregated because chunks fail independently: one
        /// hallucinated tail among six clean ones is the case worth seeing.
        public let inspection: Postprocess.Inspection

        /// True when generation stopped at a cap rather than at a stop token,
        /// so this chunk is cut off mid-sentence. Per chunk, because chunks
        /// truncate independently; `synthesize` ORs it across the passage.
        public let hitTokenCap: Bool
        /// This chunk's own span, starting at zero.
        ///
        /// A streamed chunk cannot know what preceded it and the caller decides
        /// what it has already queued, so reporting anything but zero would be
        /// a guess about someone else's playback. Add the offset with
        /// ``ChunkTiming/shifted(by:)``, or let
        /// ``Engine/synthesize(_:voice:seed:language:speed:previousTokens:shouldCancel:)``
        /// stitch the timeline for you.
        public let timing: ChunkTiming
    }

    /// Speak `text` chunk by chunk, calling `onChunk` as each becomes ready.
    ///
    /// The same synthesis as `synthesize`, delivered as it is made: time to
    /// first audio is set by the first chunk. Return `false` from `onChunk`
    /// to stop. `shouldCancel` stops within one decode step, and the stream
    /// then ends without throwing: the chunks already delivered are the
    /// partial, the one in flight is discarded. Chunk 0 draws the caller's
    /// seed and every later chunk `derive(seed, 16 + index)`, so a chunk's
    /// audio does not depend on how many came before it; the last
    /// `chunking.prefixTokens` tokens of each chunk condition the next, and
    /// `previousTokens` seed that carry for the first.
    public func stream(
        _ text: String, voice: VoiceProfile, seed: UInt64 = 0, language: String? = nil,
        speed: Double = 1.0, previousTokens: [Int]? = nil,
        shouldCancel: (() -> Bool)? = nil,
        onChunk: (Chunk) throws -> Bool
    ) throws {
        do {
            try chunks(
                text, voice: voice, seed: seed, language: language, speed: speed,
                previousTokens: previousTokens, shouldCancel: shouldCancel, onChunk: onChunk)
        } catch LoudKitError.cancelled {
            return
        }
    }

    /// `stream`, throwing `LoudKitError.cancelled` where the flag stopped it.
    private func chunks(  // swiftlint:disable:this function_parameter_count
        _ text: String, voice: VoiceProfile, seed: UInt64, language: String?,
        speed: Double, previousTokens: [Int]?,
        shouldCancel: (() -> Bool)?,
        onChunk: (Chunk) throws -> Bool
    ) throws {
        try TimeStretch.validateSpeed(speed)
        let language = Self.resolveLanguage(language, voice: voice)
        // The funnel runs on the whole text BEFORE splitting: Polish
        // respelling changes the length ("download" -> "dałnloud"), so a budget
        // computed first would be a budget for text the engine never speaks.
        let prepared = SpeechText.prepared(text, languageId: language)
        let chunks = Chunking.splitText(prepared, config: algorithm.chunking)
        if chunks.isEmpty { throw LoudKitError.shape("nothing to speak") }

        let prefixLength = algorithm.chunking.prefixTokens
        var carry: [Int] = try carryFrom(previousTokens)

        // A work queue rather than a walk over `chunks`: a chunk the window
        // could not hold is replaced, in place, by its two halves. The decision
        // needs a generated window: the overrun is a fact about this voice and
        // this text together, and nothing before generation knows it, so a
        // queue is the only structure that lets one entry become two after the
        // fact.
        //
        // Both halves keep the ORIGINAL chunk's index, so a repair cannot move
        // the seed of any later chunk. Seeds only: chunk k+1 is conditioned on
        // the tail of chunk k, which after a repair comes from the second half,
        // so later audio in THIS passage does move. What the index buys is that
        // the change stops at this passage.
        struct Part {
            let text: String
            let index: Int
            let seed: UInt64
            let terminal: Bool
            /// False on a half, so a half that still overruns ships as it is.
            /// One did, measured through the engine over the 51 passages
            /// carrying a cap hit, and its audio ended on a 0.42 s tail, it
            /// finished its clause rather than being cut. An unbounded split is
            /// a new way to fail.
            let splittable: Bool
        }
        var queue = chunks.enumerated().map { index, text in
            Part(
                text: text, index: index,
                seed: index == 0
                    ? seed : Self.derive(seed, Self.streamChunkBase + UInt64(index)),
                terminal: index == chunks.count - 1, splittable: true)
        }

        var qi = 0
        while qi < queue.count {
            let part = queue[qi]
            let index = part.index
            let chunk = part.text
            if shouldCancel?() == true { throw LoudKitError.cancelled }
            let chunkSeed = part.seed
            let textTokens = try frontend.encode(chunk, language: language)
            // Only the last chunk ends the passage.
            let generated = try generateInspected(
                textTokens: textTokens, voice: voice, seed: chunkSeed, prefix: carry,
                isTerminal: part.terminal, maxNewTokens: nil, onStep: nil,
                shouldCancel: shouldCancel)
            // The window has to be what stopped it, not the length-proportional
            // ceiling: generateInspected caps at min(maxNewTokens,
            // ceilingFor(...)), and the second fires when a short text runs
            // away. Halving a runaway gives two runaways with smaller ceilings.
            // Measured before the trim, for the reason `synthesize` states.
            if generated.hitWindow, part.splittable,
                algorithm.chunking.capResplit == .word,
                let halves = Chunking.splitInHalf(chunk)
            {
                // Discard this window and do the two halves instead. The second
                // draws from a stream of its own off the chunk seed: the flow
                // takes 1, the vocoder 2, and the retry ladder 8 up.
                queue.replaceSubrange(
                    qi...qi,
                    with: [
                        Part(
                            text: halves.0, index: index, seed: chunkSeed,
                            terminal: false, splittable: false),
                        Part(
                            text: halves.1, index: index,
                            seed: Self.derive(chunkSeed, Self.streamResplit),
                            terminal: part.terminal, splittable: false),
                    ])
                continue
            }
            // Discarded, not rendered: see `synthesizeWindow`.
            if shouldCancel?() == true { throw LoudKitError.cancelled }

            let speech = generated.tokens
            let (mel, frames) = try melDecoder.decode(
                tokens: speech, voice: voice, seed: Self.derive(chunkSeed, Self.streamFlow))
            if shouldCancel?() == true { throw LoudKitError.cancelled }
            let rendered = try vocoder.synthesize(
                mel: mel, frames: frames, seed: Self.derive(chunkSeed, Self.streamVocoder))
            // Applied per chunk and last, see `synthesize`. Per chunk rather
            // than once over the joined passage because `stream` has no joined
            // passage to apply it to, and the two paths have to produce the
            // same waveform.
            let audio = TimeStretch.fadeEdges(
            try TimeStretch.timeStretch(
                rendered, sampleRate: algorithm.sampleRate, speed: speed),
            sampleRate: algorithm.sampleRate, seconds: algorithm.edgeFadeSeconds)

            if shouldCancel?() == true { throw LoudKitError.cancelled }
            carry = try Self.carryFrom(speech, prefixTokens: prefixLength,
                                       startSpeechToken: algorithm.startSpeechToken, decode: algorithm.decode)
            let keepGoing = try onChunk(
                Chunk(
                    index: index, text: chunk, audio: audio, tokens: speech, mel: mel,
                    melFrames: frames, inspection: generated.inspection,
                    hitTokenCap: generated.hitCap,
                    // Timed on the stretched audio, so a caller who plays the
                    // chunk gets spans that match what they hear.
                    timing: Timing.timeline(
                        [ChunkSpan(text: chunk, samples: audio.count, tokens: speech.count)],
                        sampleRate: algorithm.sampleRate)[0]))
            if !keepGoing { break }
            qi += 1
        }
    }

    /// Speak text of any length as one `Result`.
    ///
    /// Exactly `stream` with the chunks concatenated, one loop, so the two
    /// paths cannot drift. `hitTokenCap` is ORed across chunks: one truncated
    /// chunk truncates the passage. Throws `LoudKitError.cancelled` when
    /// `shouldCancel` returned true: a passage cut short is never handed back
    /// as a `Result`.
    public func synthesize(
        _ text: String, voice: VoiceProfile, seed: UInt64 = 0, language: String? = nil,
        speed: Double = 1.0, previousTokens: [Int]? = nil,
        shouldCancel: (() -> Bool)? = nil
    ) throws -> Result {
        var audio: [Float] = []
        var tokens: [Int] = []
        var hitCapAnywhere = false
        var melParts: [(mel: [Float], frames: Int)] = []
        var frames = 0
        var inspections: [Postprocess.Inspection] = []
        var spans: [ChunkSpan] = []
        let t0 = Date()

        try chunks(
            text, voice: voice, seed: seed, language: language, speed: speed,
            previousTokens: previousTokens, shouldCancel: shouldCancel)
        { chunk in
            audio.append(contentsOf: chunk.audio)
            tokens.append(contentsOf: chunk.tokens)
            inspections.append(chunk.inspection)
            // ORed across chunks, matching the other four ports: one truncated
            // chunk truncates the passage.
            hitCapAnywhere = hitCapAnywhere || chunk.hitTokenCap
            spans.append(
                ChunkSpan(
                    text: chunk.text, samples: chunk.audio.count, tokens: chunk.tokens.count))
            // Along time, not end to end: a mel is (bins, frames) row-major, so
            // concatenating the flat arrays would interleave the bins. Collected
            // here, joined once by `joinMelsAlongTime`.
            melParts.append((chunk.mel, chunk.melFrames))
            frames += chunk.melFrames
            return true
        }

        let elapsed = Date().timeIntervalSince(t0)
        let mel = Self.joinMelsAlongTime(melParts, frames: frames)
        return Result(
            audio: audio, tokens: tokens, mel: mel, melFrames: frames, seed: seed,
            sampleRate: algorithm.sampleRate,
            timings: StageTimings(tokens: elapsed, mel: 0, audio: 0),
            algorithmFingerprint: algorithm.fingerprint(),
            hitTokenCap: hitCapAnywhere,
            inspections: inspections,
            speed: speed,
            // Rebuilt from the chunks rather than shifting each chunk's own
            // timing by a running Double: `timeline` accumulates sample offsets
            // as integers, so the joins are exact and every chunk's `end` is
            // the next one's `start` down to the last bit.
            chunks: Timing.timeline(spans, sampleRate: algorithm.sampleRate))
    }

    /// Concatenate `(bins, frames)` row-major mels along the time axis.
    ///
    /// The flat arrays cannot simply be appended: that puts the second mel's
    /// first bin after the first mel's last bin, which is not a spectrogram,
    /// and it still renders, into audio that sounds like a fault in the model.
    ///
    /// One allocation, one pass, bins-major, each part's frames written in
    /// place. Joining pairwise copied the whole accumulator per chunk, so a
    /// passage of N chunks copied the mel N(N+1)/2 times, quadratic in the
    /// length of the thing being read and on the largest array in the process.
    ///
    /// `frames` is the total across `parts`, which the caller is already
    /// counting.
    static func joinMelsAlongTime(_ parts: [(mel: [Float], frames: Int)], frames: Int) -> [Float] {
        let bins = parts.first.map { $0.mel.count / max($0.frames, 1) } ?? 0
        guard bins > 0 else { return [] }
        var out = [Float](repeating: 0, count: bins * frames)
        for bin in 0..<bins {
            var at = bin * frames
            for part in parts {
                for f in 0..<part.frames { out[at + f] = part.mel[bin * part.frames + f] }
                at += part.frames
            }
        }
        return out
    }

    /// The conditioning context this call inherits from the one before it,
    /// read from this engine's config.
    private func carryFrom(_ previousTokens: [Int]?) throws -> [Int] {
        try Self.carryFrom(
            previousTokens, prefixTokens: algorithm.chunking.prefixTokens,
            startSpeechToken: algorithm.startSpeechToken, decode: algorithm.decode)
    }

    /// Refuse text the funnel emptied, before it is tokenised.
    ///
    /// A footnote marker or a lone emoji leaves nothing, and encoding nothing
    /// tokenises the bare language tag: the generator then renders
    /// `min_tokens_floor` tokens of babble under the caller's own text, with no
    /// error anywhere. `synthesize` refuses through an empty split; this is the
    /// same door on the one-window path. Static, so it is exercised without a
    /// checkpoint. Mirrors the `NothingToSpeakError` in `generate_window`.
    static func requireSomethingToSpeak(_ prepared: String) throws {
        guard prepared.isEmpty else { return }
        throw LoudKitError.shape(
            "nothing to speak: the text funnel removed every character. "
            + "Footnote markers, emoji and symbols with no word in the "
            + "render language are dropped, and this input was only those.")
    }

    /// Refuse a window that generated no speech at all.
    ///
    /// Silence with no error is the one failure a caller does not notice, and
    /// the remedy is a setting, so the message names it. Python raises here too.
    static func requireSpeechProduced(_ speech: [Int]) throws {
        guard speech.isEmpty else { return }
        throw LoudKitError.shape(
            "generation produced no speech tokens: the stop token was accepted "
            + "immediately. Set sampling.min_tokens_floor above 0 to refuse that "
            + "during sampling.")
    }

    /// Refuse a token sequence the renderer cannot look up, naming the first
    /// offending id and the bound.
    ///
    /// One function for both doors that take ids from a caller, so
    /// `previousTokens` and `tokens` refuse the same values with the same
    /// sentence. Mirrors `loudkit.window.validate_speech_tokens`.
    static func validateSpeechTokens(_ tokens: [Int], limit: Int, field: String) throws {
        for token in tokens where !(0 <= token && token < limit) {
            throw LoudKitError.invalidToken(field: field, token: token, limit: limit)
        }
    }

    /// The conditioning context a call inherits from the one before it.
    ///
    /// The same slice the streaming loop takes between two chunks, last
    /// `chunking.prefixTokens`, applied to tokens that came from a different
    /// call. There is deliberately no second mechanism: a request boundary and
    /// a chunk boundary are the same join, and the reason chunk joins do not
    /// stutter is the reason request joins should not either.
    ///
    /// Any length is accepted because only the tail is used, so
    /// `previousTokens: result.tokens` is the intended call and a caller should
    /// never have to know the prefix length to make it.
    ///
    /// Static, and taking the two config values rather than reading them off
    /// `self`, so it can be tested on a machine with no checkpoint and no
    /// CoreML packages: an `Engine` cannot be built without them, and this
    /// arithmetic, which is the whole of feature C, would otherwise only ever
    /// be exercised on the developer machines that have the weights.
    ///
    /// - Throws: `LoudKitError.shape` for an id outside the acoustic codebook.
    ///   The whole input is checked rather than only the slice that will be
    ///   used: an id out of range means the sequence was built wrong, and
    ///   reporting that only when it happens to land in the last six tokens
    ///   would make the failure depend on the length of the caller's text.
    static func carryFrom(
        _ previousTokens: [Int]?, prefixTokens: Int, startSpeechToken: Int, decode: String = "single"
    ) throws -> [Int] {
        guard let previousTokens else { return [] }
        try validateSpeechTokens(previousTokens, limit: startSpeechToken,
                                 field: "previousTokens")
        // Guarded rather than `suffix(prefixTokens)` alone for the same reason
        // Python does not write `tokens[-wanted:]`: at zero that slice is the
        // whole list rather than nothing, which would condition on the entire
        // previous utterance at exactly the setting that means "chunks are
        // independent". Swift's `suffix(0)` is empty, so this guard is about
        // saying the rule out loud in all five ports, not about the arithmetic.
        guard prefixTokens > 0 else { return [] }
        var end = previousTokens.count
        if decode == "fusion_mtp2" { end -= end % 2 }
        var start = end - prefixTokens
        if decode == "fusion_mtp2", start % 2 != 0 { start -= 1 }
        return Array(previousTokens[max(0, start)..<end])
    }

    /// Drop the generator's control tokens, and refuse, rather than slice, a
    /// sequence longer than the render window.
    ///
    /// `.prefix(maxSpeechTokens)` leaves the end of a passage nonexistent
    /// while the audio still sounds perfectly fine: silent data loss, noticed
    /// only by a listener who knows the text. `window.strip_specials` raises,
    /// and Rust, Go and JS all return an error; truncating instead, in two
    /// places with `Renderer.decode` doing it again independently, hands a
    /// caller clipped audio and no error anywhere.
    private func stripSpecials(_ tokens: [Int]) throws -> [Int] {
        let limit = algorithm.startSpeechToken
        // Refused, not filtered, and both ends. `filter { $0 < limit }` dropped
        // the specials above the range and said nothing about below it, so a
        // negative id went straight to the renderer: on CoreML that is an
        // out-of-bounds read of the embedding table, and in torch it indexes
        // from the *end* and returns a plausible vector. Python, Go, Rust and
        // JS all refuse it; this was the one port that did not, on the public
        // API; the transports validate before they get here.
        //
        // Filtering is right for a special and wrong for a negative: a special
        // is a token the caller legitimately has and this layer removes, while
        // a negative is not a token at all, and dropping it silently renders
        // something the caller did not ask for.
        if let bad = tokens.first(where: { $0 < 0 }) {
            // The same sentence `LoudKitError.invalidToken` writes, from the
            // same constant, so this site keeps the `invalid_tokens` code
            // rather than holding it by a wording coincidence. Its own range
            // clause, because this check has a range and no lower bound to
            // state.
            throw LoudKitError.shape(
                "tokens contains \(bad)\(LoudKitError.notASpeechToken) "
                    + "(expected 0 to \(limit - 1))")
        }
        let speech = tokens.filter { $0 < limit }
        try Windowing.requireFits(speech.count, algorithm.window.maxSpeechTokens)
        return speech
    }

    /// Per-stage seed from one user seed, same constants as
    /// `loudkit.window._derive`, so the streams line up across languages.
    /// Public because the derivation is part of the seed contract (the
    /// conformance fixture pins its outputs), not an implementation detail.
    public static func derive(_ seed: UInt64, _ stream: UInt64) -> UInt64 {
        seed &* 0x9E37_79B9_7F4A_7C15 &+ stream &* 0xBF58_476D_1CE4_E5B9
    }

    /// What a synthesis reads as when neither the caller nor the voice says.
    ///
    /// Reached less often than it looks: ``VoiceProfile/load(url:)`` defaults a
    /// *missing* header key to `"en"`, and Python writes the key,
    /// so an empty string only arrives from a profile built in memory or a
    /// header hand-edited to `""`. A profile file with no language field
    /// inherits nothing: it loads as `"en"`.
    static let fallbackLanguage = "en"

    /// The language chain: the argument, then the voice's recorded language,
    /// then English.
    ///
    /// Without the voice link, `engine.synthesize("Cześć", voice: polishVoice)`
    /// runs Polish text
    /// through the English frontend, English number words, English
    /// abbreviation expansion, no Polish respelling, and says so nowhere. A
    /// profile records the language of the audio it was enrolled from,
    /// so the voice is the better answer than a constant.
    ///
    /// Passing a language is how cross-lingual synthesis is requested: an
    /// English voice reading Polish text is `language: "pl"`, and the argument
    /// always wins over the profile.
    ///
    /// An empty profile language is treated as absent: it is not a language,
    /// and `TextFrontend.encode` would tag the text `[]` with it.
    ///
    /// Mirrors `loudkit.window.resolve_language`.
    static func resolveLanguage(_ language: String?, voice: VoiceProfile) -> String {
        if let language { return language }
        return voice.language.isEmpty ? fallbackLanguage : voice.language
    }
}

/// Small indirection so Config.swift stays CoreML-free.
struct MLComputeUnitsWrapper {
    let value: CoreML.MLComputeUnits
    init(_ units: ExecutionConfig.ComputeUnits) {
        switch units {
        case .cpuOnly: value = .cpuOnly
        case .cpuAndNeuralEngine: value = .cpuAndNeuralEngine
        case .all: value = .all
        }
    }
}
