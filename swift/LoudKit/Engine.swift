import CoreML
import Foundation
import LoudKitText

/// The engine: the same five-component composition as `loudkit.engine.Engine`,
/// with the same public shape, so the README can show both languages side by
/// side:
///
///     let engine = try Engine.load(checkpoint: url)
///     let voice  = try VoiceProfile.load(url: voiceURL)
///     let result = try engine.synthesize("Hello there.", voice: voice, seed: 7)
///     try result.save(to: outURL)
///
/// Seeds are derived per stage with the identical splitting constants, so the
/// sampler, the flow prior and the vocoder excitation consume the same Philox
/// numbers as the Python engine given the same user seed.
public final class Engine {
    static let streamFlow: UInt64 = 1
    static let streamVocoder: UInt64 = 2
    /// Chunk seeds start here, clear of the per-stage streams.
    static let streamChunkBase: UInt64 = 16

    public let algorithm: AlgorithmConfig
    public let execution: ExecutionConfig
    public let frontend: TextFrontend
    public let tokenGenerator: TokenGenerator
    public let melDecoder: MelDecoder
    public let vocoder: Vocoder
    /// Where this engine was loaded from — kept so `withExecution` can
    /// rebuild the CoreML stages without re-reading the generator weights.
    private var checkpointURL: URL?
    private var assetsURL: URL?

    public struct StageTimings {
        public let tokens: Double
        public let mel: Double
        public let audio: Double
        public var total: Double { tokens + mel + audio }
        public func rtf(audioSeconds: Double) -> Double {
            total > 0 ? audioSeconds / total : .infinity
        }
    }

    public struct Result {
        public let audio: [Float]
        public let tokens: [Int]
        public let mel: [Float]
        public let melFrames: Int
        public let seed: UInt64
        public let sampleRate: Int
        public let timings: StageTimings
        public let algorithmFingerprint: String
        public let hitTokenCap: Bool
        /// What the artifact detectors concluded, one entry per chunk.
        ///
        /// A list rather than a single verdict because a passage is many chunks
        /// and they fail independently: one hallucinated tail in the middle of
        /// six clean ones is the case worth seeing, and an aggregate hides it.
        public var inspections: [Postprocess.Inspection] = []

        /// The time-stretch this render was asked for. `1.0` means none was
        /// applied — the waveform came straight out of the vocoder.
        ///
        /// Recorded rather than inferred, because it cannot be inferred: a
        /// stretched reading and a naturally faster one are the same numbers
        /// afterwards, and `duration` alone cannot tell a caller which it is
        /// holding.
        public var speed: Double = 1.0

        /// Whether `shouldCancel` stopped this passage before its last chunk.
        ///
        /// `synthesize` throws `LoudKitError.cancelled` for the same event,
        /// because a fraction of one utterance is not useful. A long passage is
        /// different — the chunks already rendered are playable — so `stream`
        /// and `synthesizeLong` return what they have. Breaking and saying
        /// nothing leaves a caller unable to tell an interrupted
        /// reading from a short one, and the two want opposite handling.
        public var cancelled: Bool = false

        /// Where each chunk lands in `audio`, and where its words probably do.
        ///
        /// One entry per chunk, in order and adjacent: chunk *k*'s `end` is the
        /// same `Double` as chunk *k+1*'s `start`, and the last `end` is
        /// `duration`. A single-window synthesis gets one entry covering the
        /// whole result.
        ///
        /// Chunk boundaries are exact — they are sample offsets, which the
        /// engine already knows because it concatenated the chunks. The
        /// per-word times inside each entry are an **estimate**; read
        /// ``Timing`` before building anything that depends on them.
        ///
        /// Measured on the returned waveform, so they already account for
        /// `speed`.
        public var chunks: [ChunkTiming] = []

        /// Any chunk was impossibly long for its text and no rule could say
        /// where to cut. Nothing was removed; you are being told.
        public var suspect: Bool { inspections.contains { $0.suspect } }

        public var duration: Double { Double(audio.count) / Double(sampleRate) }

        /// Write a 32-bit float WAV.
        ///
        /// **Not** the same container as Python's `Result.save`, which is a
        /// bare `sf.write(...)` and therefore soundfile's default PCM_16.
        /// Same audio, different encoding — a byte comparison of the two
        /// files can never match.
        /// Float32 is kept deliberately: this file is what the conformance
        /// harness reads back, and rounding to 16 bits would put a
        /// quantisation floor under a correlation gated at 0.999.
        public func save(to url: URL) throws {
            var data = Data()
            func append<T>(_ value: T) {
                var v = value
                withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
            }
            let byteCount = audio.count * 4
            // RIFF carries its sizes in 32 bits, so a WAV cannot describe more
            // than about 4 GB — roughly twelve hours at 24 kHz mono float32.
            // `UInt32(...)` on a larger value does not truncate in Swift, it
            // traps: the process dies with no message and no file written. Say
            // what happened instead.
            guard 36 + byteCount <= Int(UInt32.max) else {
                throw LoudKitError.shape(
                    "\(audio.count) samples is \(byteCount) bytes of audio; a WAV "
                    + "header cannot describe more than \(UInt32.max) — split the "
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
            throw LoudKitError.asset("engine was not built by Engine.load — no checkpoint to rebuild from")
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
        return engine
    }

    /// A stage may be present as the exported .mlpackage or as the
    /// precompiled .mlmodelc only (an app bundle ships the compiled form and
    /// skips the on-device CoreML compile). The package wins when both exist
    /// because MLHelpers.loadModel already prefers a compiled sibling of the
    /// same stem.
    private static func stageURL(in assets: URL, _ stem: String) throws -> URL {
        for ext in ["mlpackage", "mlmodelc"] {
            let url = assets.appendingPathComponent("\(stem).\(ext)")
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        throw LoudKitError.asset(
            "\(stem).mlpackage/.mlmodelc not found in \(assets.path) — "
            + "export with tools/export_coreml.py")
    }

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
    /// trailing pause is the sentence's rhythm rather than dead air — the
    /// detectors that cut a tail are told so and hold off.
    private func generateInspected(
        textTokens: [Int], voice: VoiceProfile, seed: UInt64, prefix: [Int],
        isTerminal: Bool, maxNewTokens: Int?, onStep: (() -> Void)?,
        shouldCancel: (() -> Bool)?
    ) throws -> (tokens: [Int], inspection: Postprocess.Inspection, hitCap: Bool) {
        let pp = algorithm.postprocess
        let floor = algorithm.eosFloor(nTextTokens: textTokens.count)
        // Refused rather than passed through. A negative cap reached the
        // generator, which produced nothing and returned an empty result with
        // no error anywhere — the same defect Python carried at `engine.py`,
        // and the same fix: a count of tokens is at least one.
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

        // Selective re-roll: a window whose verdict is unfixable — dropout
        // (content missing) or suspect (certainly wrong, nowhere to cut) — is
        // regenerated from a derived seed, up to retryMaxAttempts times. Only
        // condemned windows pay; the ladder is a pure function of the caller's
        // seed, so the same seed still gives the same audio, retries included.
        var gen: [Int] = []
        var verdict = Postprocess.Inspection(keep: 0, reason: .clean, suspect: false)
        var hitCap = false
        var attempt = 0
        while true {
            // Retry attempts draw derive(seed, 8 + attempt): clear of the
            // stage streams (1, 2) and below the chunk streams at 16.
            let attemptSeed = attempt == 0 ? seed : Self.derive(seed, 8 + UInt64(attempt))
            let sampler = LRSamplerV1(config: algorithm.sampling, seed: attemptSeed)
            if pp.mode != .off {
                sampler.observeEOS(stopToken: algorithm.stopSpeechToken, floor: floor)
            }
            let generation = tokenGenerator.generate(
                textTokens: textTokens, voice: voice, sampler: sampler, maxNewTokens: cap,
                prefix: prefix, onStep: onStep, shouldCancel: shouldCancel)

            // `gen` is what the shipped engine calls a row: every token the
            // model committed to, with the stop marker itself excluded.
            // Indices into it are decode-step indices, which is what makes the
            // observed peak comparable against it — so the detectors run here,
            // before `stripSpecials` is free to renumber anything.
            gen = generation.rawTokens
            let ended = gen.last == algorithm.stopSpeechToken
            if ended { gen.removeLast() }

            let peak = sampler.eosPeak
            hitCap = !ended && gen.count >= cap
            verdict = Postprocess.inspect(
                gen,
                request: Postprocess.Request(
                    textTokenCount: textTokens.count, minTokens: floor,
                    eosPeakAt: peak.at, eosPeakProb: peak.probability,
                    ended: ended, isTerminal: isTerminal, hitCeiling: hitCap),
                silence: Set(algorithm.sampling.silenceTokenIds), config: pp)
            let condemned = verdict.reason == .dropout || verdict.suspect
            if !condemned || pp.mode == .off || attempt >= pp.retryMaxAttempts { break }
            attempt += 1
        }
        if pp.mode == .trim, verdict.keep < gen.count {
            gen = Array(gen.prefix(verdict.keep))
        }
        return (try stripSpecials(gen), verdict, hitCap)
    }

    /// `onStep` — see `TokenGenerator.generate`; forwarded verbatim.
    ///
    /// `language` is `nil` for "the voice's own language" — see
    /// ``resolveLanguage(_:voice:)``. Pass one to read text in a language the
    /// voice was not enrolled in; that is what cross-lingual synthesis is, and
    /// the argument always wins.
    ///
    /// `speed` is playback speed, in `[0.5, 2.0]`; greater than one is faster
    /// and pitch is preserved. `1.0` — the default — is an exact bypass: the
    /// waveform is the vocoder's own samples, untouched. It is applied last,
    /// after the postprocess detectors have inspected the render, because those
    /// detectors measure pacing against the text (duration per token) and a
    /// stretch applied first would move every measurement they make. It is a
    /// change to the *delivery*, not to the reading, and like the seed it is an
    /// execution input rather than an algorithm value — the fingerprint does
    /// not move.
    ///
    /// `previousTokens` are the speech tokens this utterance continues from —
    /// `Result.tokens` of the call before it. The single window is then
    /// conditioned on their tail exactly as an interior chunk is conditioned on
    /// its predecessor, which is what stops a second request from restarting
    /// the pitch contour like a fresh sentence. Only the last
    /// `chunking.prefixTokens` are used, so passing a whole previous result is
    /// the intended usage and costs the caller no arithmetic.
    public func synthesize(
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
        // The speech funnel, before tokenising — the same `SpeechText.prepared`
        // Python calls `speech_text` and runs in `Engine._generate_window`, on
        // the one path that renders. This module could not reach it while it
        // lived in a separate target, so it encoded raw text: "Rabat 15% na
        // weekend!" gave 17 token ids where Python gives 27 for "Rabat
        // piętnaście procent na łikend!". Different tokens are different
        // speech, for the package the README presents beside Python.
        let prepared = SpeechText.prepared(text, languageId: language)
        let textTokens = try frontend.encode(prepared, language: language)

        let t0 = Date()
        // A single window is the whole passage, so it is terminal.
        let generated = try generateInspected(
            textTokens: textTokens, voice: voice, seed: seed, prefix: prefix, isTerminal: true,
            maxNewTokens: maxNewTokens, onStep: onStep, shouldCancel: shouldCancel)
        let t1 = Date()
        // Discarded, not rendered. The partial tokens belong to speech the
        // listener has already interrupted, and the mel decode plus vocode is
        // the larger half of the barge-in latency on an edge device — so
        // rendering them adds exactly the wait the cancellation exists to
        // remove, and then plays audio nobody asked for. Python discards at
        // engine.py:298 and JS at engine.ts:473, both with the same reasoning.
        if shouldCancel?() == true { throw LoudKitError.cancelled }
        let speech = generated.tokens
        let (mel, frames) = try melDecoder.decode(
            tokens: speech, voice: voice, seed: Self.derive(seed, Self.streamFlow))
        let t2 = Date()
        let rendered = try vocoder.synthesize(
            mel: mel, frames: frames, seed: Self.derive(seed, Self.streamVocoder))
        let t3 = Date()
        // Last, and after `generateInspected` above rather than before it: the
        // detectors judge pacing by duration per token, and stretching first
        // would move every number they compare against. `speed == 1.0` returns
        // the vocoder's own array, so the default costs nothing and changes no
        // byte. Outside the stage timings for the same reason it is outside the
        // fingerprint — it is delivery, not synthesis.
        let audio = try TimeStretch.timeStretch(
            rendered, sampleRate: algorithm.sampleRate, speed: speed)

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

    /// Render a token sequence that already exists — the single most useful
    /// diagnostic when two implementations disagree (it removes sampling from
    /// the comparison and asks only whether the renderer agrees).
    public func synthesizeTokens(
        _ tokens: [Int], voice: VoiceProfile, seed: UInt64 = 0
    ) throws -> Result {
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
            // No text reached this path, so there are no words to estimate —
            // but the span still covers the whole render, so a caller stitching
            // results does not have to special-case it.
            chunks: Timing.timeline(
                [ChunkSpan(text: "", samples: audio.count, tokens: speech.count)],
                sampleRate: algorithm.sampleRate))
    }

    /// Drop the generator's control tokens, and **refuse** — rather than slice
    /// — a sequence longer than the render window.
    ///
    /// `.prefix(maxSpeechTokens)` leaves the end of a passage nonexistent
    /// while the audio still sounds perfectly fine: silent data loss,
    /// noticed only by a listener who knows the text. Python raises here
    /// (engine.py:466), and Rust, Go and JS all return an error; truncating
    /// instead — in two places, `Renderer.decode` doing it again independently —
    /// hands a caller clipped audio and no error anywhere.
    /// One rendered chunk, handed to a `stream` callback as soon as it exists.
    public struct Chunk {
        /// Zero-based position in the split, which is also what the chunk's
        /// seed was derived from.
        public let index: Int
        /// The piece of the passage this chunk speaks, after the speech funnel
        /// — what was tokenised, which is not always what the caller passed in.
        public let text: String
        public let audio: [Float]
        public let tokens: [Int]
        public let mel: [Float]
        public let melFrames: Int
        /// What the artifact detectors concluded about this chunk. Per chunk
        /// rather than aggregated because chunks fail independently: one
        /// hallucinated tail among six clean ones is the case worth seeing.
        public let inspection: Postprocess.Inspection
        /// This chunk's own span, starting at zero.
        ///
        /// A streamed chunk cannot know what preceded it — the caller decides
        /// what it has already queued — so reporting anything but zero would be
        /// a guess about someone else's playback. Add the offset with
        /// ``ChunkTiming/shifted(by:)``, or let
        /// ``Engine/synthesizeLong(_:voice:seed:language:speed:previousTokens:shouldCancel:)``
        /// stitch the timeline for you.
        public let timing: ChunkTiming
    }

    /// Speak `text` chunk by chunk, calling `onChunk` as each becomes ready.
    ///
    /// The difference from
    /// ``synthesizeLong(_:voice:seed:language:speed:previousTokens:shouldCancel:)``
    /// is delivery, not synthesis: time to first audio is set by the first
    /// chunk rather than by the whole passage, which is what lets a reading app
    /// start playing a sentence while the rest is still being made.
    ///
    /// A callback rather than an `AsyncSequence`, so the caller decides whether
    /// this runs on an actor: the engine's CoreML models are not `Sendable` and
    /// making the API async would impose a concurrency model on hosts that do
    /// not want one. Return `false` from `onChunk` to stop.
    ///
    /// Two things make the joins match Python's rather than merely existing:
    ///
    /// * **Per-chunk seeds.** Each chunk draws from `derive(seed, 16 + index)`,
    ///   so a chunk's audio does not depend on how many came before it and
    ///   stopping early cannot change what was already produced.
    /// * **Prefix carry.** The last `chunking.prefixTokens` speech tokens of a
    ///   chunk are fed into the next as context. Without it every chunk
    ///   restarts its pitch contour like a fresh sentence, and the restart is
    ///   audible at every join.
    ///
    /// `language` is `nil` for "the voice's own language" — see
    /// ``resolveLanguage(_:voice:)``. Resolved once here, before splitting, so
    /// every chunk of a passage is read in the same language.
    ///
    /// `speed` stretches each chunk independently, which is the same
    /// independence the seeds and the prefix already have: a chunk's audio must
    /// not depend on how many came before it, or a listener who stops early
    /// would have heard something different from one who did not.
    ///
    /// `previousTokens` seeds the carry, so the first chunk of *this* call is
    /// conditioned on the tail of a *previous* one. It is the same conditioning
    /// the joins inside a passage already use — the carry variable below simply
    /// starts non-empty — which is why a request boundary stops being audible
    /// without a second mechanism existing to maintain.
    public func stream(
        _ text: String, voice: VoiceProfile, seed: UInt64 = 0, language: String? = nil,
        speed: Double = 1.0, previousTokens: [Int]? = nil,
        shouldCancel: (() -> Bool)? = nil,
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

        for (index, chunk) in chunks.enumerated() {
            if shouldCancel?() == true { break }
            let chunkSeed = Self.derive(seed, Self.streamChunkBase + UInt64(index))
            let textTokens = try frontend.encode(chunk, language: language)
            // Only the last chunk ends the passage.
            let generated = try generateInspected(
                textTokens: textTokens, voice: voice, seed: chunkSeed, prefix: carry,
                isTerminal: index == chunks.count - 1, maxNewTokens: nil, onStep: nil,
                shouldCancel: shouldCancel)
            // Discarded, not rendered — see `synthesize`.
            if shouldCancel?() == true { break }

            let speech = generated.tokens
            let (mel, frames) = try melDecoder.decode(
                tokens: speech, voice: voice, seed: Self.derive(chunkSeed, Self.streamFlow))
            let rendered = try vocoder.synthesize(
                mel: mel, frames: frames, seed: Self.derive(chunkSeed, Self.streamVocoder))
            // Applied per chunk and last — see `synthesize`. Per chunk rather
            // than once over the joined passage because `stream` has no joined
            // passage to apply it to, and the two paths have to produce the
            // same waveform.
            let audio = try TimeStretch.timeStretch(
                rendered, sampleRate: algorithm.sampleRate, speed: speed)

            carry = prefixLength > 0 ? Array(speech.suffix(prefixLength)) : []
            let keepGoing = try onChunk(
                Chunk(
                    index: index, text: chunk, audio: audio, tokens: speech, mel: mel,
                    melFrames: frames, inspection: generated.inspection,
                    // Timed on the stretched audio, so a caller who plays the
                    // chunk gets spans that match what they hear.
                    timing: Timing.timeline(
                        [ChunkSpan(text: chunk, samples: audio.count, tokens: speech.count)],
                        sampleRate: algorithm.sampleRate)[0]))
            if !keepGoing { break }
        }
    }

    /// Speak text of any length as one waveform.
    ///
    /// Exactly
    /// ``stream(_:voice:seed:language:speed:previousTokens:shouldCancel:onChunk:)``
    /// with the chunks concatenated — one loop, so the streaming and
    /// whole-passage paths cannot drift apart.
    ///
    /// Before this existed, `synthesize` refused anything past one window (~127
    /// characters) and the caller had to split the text themselves. A caller
    /// who splits differently gets different chunk boundaries, therefore
    /// different derived seeds, therefore different audio from every other port
    /// — while `AlgorithmConfig.fingerprint()` goes on declaring the chunking
    /// recipe this module was not applying.
    ///
    /// `language` is `nil` for "the voice's own language"; left unresolved here
    /// so ``stream(_:voice:seed:language:speed:previousTokens:shouldCancel:onChunk:)``
    /// resolves it once, on the one path that renders.
    ///
    /// `speed` is applied per chunk, exactly as `stream` applies it, so the two
    /// paths still produce the same waveform. `previousTokens` conditions the
    /// *first* chunk; every chunk after it is conditioned on the one before, as
    /// always.
    public func synthesizeLong(
        _ text: String, voice: VoiceProfile, seed: UInt64 = 0, language: String? = nil,
        speed: Double = 1.0, previousTokens: [Int]? = nil,
        shouldCancel: (() -> Bool)? = nil
    ) throws -> Result {
        var audio: [Float] = []
        var tokens: [Int] = []
        // Set when `shouldCancel` stops the loop, so the caller can tell an
        // interrupted passage from a finished one. `synthesize` throws
        // `.cancelled` for the same event; here a partial result is genuinely
        // useful — the audio produced so far is playable — so it is returned
        // and flagged rather than discarded. What it must not be is
        // indistinguishable from a short passage, which is what a bare `break`
        // made it.
        var wasCancelled = false
        var melParts: [(mel: [Float], frames: Int)] = []
        var frames = 0
        var inspections: [Postprocess.Inspection] = []
        var spans: [ChunkSpan] = []
        let t0 = Date()

        try stream(
            text, voice: voice, seed: seed, language: language, speed: speed,
            previousTokens: previousTokens, shouldCancel: shouldCancel)
        { chunk in
            audio.append(contentsOf: chunk.audio)
            tokens.append(contentsOf: chunk.tokens)
            inspections.append(chunk.inspection)
            spans.append(
                ChunkSpan(
                    text: chunk.text, samples: chunk.audio.count, tokens: chunk.tokens.count))
            // Along time, not end to end: a mel is (bins, frames) row-major, so
            // concatenating the flat arrays would interleave the bins.
            // Collected, joined once. `appendMelAlongTime` copies the whole
            // accumulator every call, so a passage of N chunks copied the mel
            // N(N+1)/2 times — quadratic in the length of the thing being read,
            // at the point where it is already the largest array in the process.
            melParts.append((chunk.mel, chunk.melFrames))
            frames += chunk.melFrames
            return true
        }
        // Asked once more, after the stream has stopped: `stream` breaks out of
        // its own loop and cannot report why, so this is where an interrupted
        // passage becomes distinguishable from a finished one.
        wasCancelled = shouldCancel?() == true

        let elapsed = Date().timeIntervalSince(t0)
        // One allocation, one pass: bins-major, each chunk's frames written in
        // place. This is what `appendMelAlongTime` did pairwise, without the
        // N(N+1)/2 copies.
        let bins = melParts.first.map { $0.mel.count / max($0.frames, 1) } ?? 0
        var mel = [Float](repeating: 0, count: bins * frames)
        if bins > 0 {
            for bin in 0..<bins {
                var at = bin * frames
                for part in melParts {
                    for f in 0..<part.frames { mel[at + f] = part.mel[bin * part.frames + f] }
                    at += part.frames
                }
            }
        }
        return Result(
            audio: audio, tokens: tokens, mel: mel, melFrames: frames, seed: seed,
            sampleRate: algorithm.sampleRate,
            timings: StageTimings(tokens: elapsed, mel: 0, audio: 0),
            algorithmFingerprint: algorithm.fingerprint(),
            hitTokenCap: false,
            inspections: inspections,
            speed: speed,
            cancelled: wasCancelled,
            // Rebuilt from the chunks rather than shifting each chunk's own
            // timing by a running Double: `timeline` accumulates sample offsets
            // as integers, so the joins are exact and every chunk's `end` is
            // the next one's `start` down to the last bit.
            chunks: Timing.timeline(spans, sampleRate: algorithm.sampleRate))
    }

    /// Concatenate two `(bins, frames)` row-major mels along the time axis.
    ///
    /// The flat arrays cannot simply be appended: that puts the second mel's
    /// first bin after the first mel's last bin, which is not a spectrogram.
    static func appendMelAlongTime(
        _ left: [Float], _ leftFrames: Int, _ right: [Float], _ rightFrames: Int
    ) -> [Float] {
        if left.isEmpty { return right }
        let bins = left.count / max(leftFrames, 1)
        var out = [Float](repeating: 0, count: bins * (leftFrames + rightFrames))
        for bin in 0..<bins {
            let target = bin * (leftFrames + rightFrames)
            for f in 0..<leftFrames { out[target + f] = left[bin * leftFrames + f] }
            for f in 0..<rightFrames {
                out[target + leftFrames + f] = right[bin * rightFrames + f]
            }
        }
        return out
    }

    /// The conditioning context this call inherits from the one before it,
    /// read from this engine's config.
    private func carryFrom(_ previousTokens: [Int]?) throws -> [Int] {
        try Self.carryFrom(
            previousTokens, prefixTokens: algorithm.chunking.prefixTokens,
            startSpeechToken: algorithm.startSpeechToken)
    }

    /// The conditioning context a call inherits from the one before it.
    ///
    /// The same slice the streaming loop takes between two chunks — last
    /// `chunking.prefixTokens` — applied to tokens that came from a different
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
    /// arithmetic — which is the whole of feature C — would otherwise only ever
    /// be exercised on the developer machines that have the weights.
    ///
    /// - Throws: `LoudKitError.shape` for an id outside the acoustic codebook.
    ///   The whole input is checked rather than only the slice that will be
    ///   used: an id out of range means the sequence was built wrong, and
    ///   reporting that only when it happens to land in the last six tokens
    ///   would make the failure depend on the length of the caller's text.
    static func carryFrom(
        _ previousTokens: [Int]?, prefixTokens: Int, startSpeechToken: Int
    ) throws -> [Int] {
        guard let previousTokens else { return [] }
        for token in previousTokens where !(0 <= token && token < startSpeechToken) {
            throw LoudKitError.shape(
                "previousTokens contains \(token), which is not an acoustic speech "
                    + "token (expected 0 <= id < \(startSpeechToken)). Pass `Result.tokens` "
                    + "from an earlier call; the generator's own control tokens are "
                    + "already stripped from it.")
        }
        // Guarded rather than `suffix(prefixTokens)` alone for the same reason
        // Python does not write `tokens[-wanted:]`: at zero that slice is the
        // whole list rather than nothing, which would condition on the entire
        // previous utterance at exactly the setting that means "chunks are
        // independent". Swift's `suffix(0)` is empty, so this guard is about
        // saying the rule out loud in all five ports, not about the arithmetic.
        return prefixTokens > 0 ? Array(previousTokens.suffix(prefixTokens)) : []
    }

    private func stripSpecials(_ tokens: [Int]) throws -> [Int] {
        let limit = algorithm.startSpeechToken
        // Refused, not filtered, and both ends. `filter { $0 < limit }` dropped
        // the specials above the range and said nothing about below it, so a
        // negative id went straight to the renderer: on CoreML that is an
        // out-of-bounds read of the embedding table, and in torch it indexes
        // from the *end* and returns a plausible vector. Python, Go, Rust and
        // JS all refuse it; this was the one port that did not, on the public
        // API — the transports validate before they get here.
        //
        // Filtering is right for a special and wrong for a negative: a special
        // is a token the caller legitimately has and this layer removes, while
        // a negative is not a token at all, and dropping it silently renders
        // something the caller did not ask for.
        if let bad = tokens.first(where: { $0 < 0 }) {
            throw LoudKitError.shape(
                "tokens contains \(bad), which is not an acoustic speech token "
                    + "(expected 0 to \(limit - 1))")
        }
        let speech = tokens.filter { $0 < limit }
        try Windowing.requireFits(speech.count, algorithm.window.maxSpeechTokens)
        return speech
    }

    /// Per-stage seed from one user seed — same constants as
    /// `loudkit.engine._derive`, so the streams line up across languages.
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
    /// inherits nothing — it loads as `"en"`.
    static let fallbackLanguage = "en"

    /// The language chain: the argument, then the voice's recorded language,
    /// then English.
    ///
    /// Without the voice link, `engine.synthesize("Cześć", voice: polishVoice)`
    /// runs Polish text
    /// through the English frontend — English number words, English
    /// abbreviation expansion, no Polish respelling — and says so nowhere. A
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
    /// Mirrors `loudkit.engine._resolve_language`.
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
