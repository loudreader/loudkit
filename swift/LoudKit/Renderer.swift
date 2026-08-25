import CoreML
import Foundation

/// The renderer: the CoreML stage graphs exported by `tools/export_coreml.py`,
/// driven with the same window recipe and the same Philox-addressed noise as
/// the Python coreml backend. All randomness is data; the graphs are pure
/// functions; a cross-language waveform difference is arithmetic.

enum MLHelpers {
    static func loadModel(packageURL: URL, computeUnits: MLComputeUnits) throws -> MLModel {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        let compiled: URL
        if packageURL.pathExtension == "mlmodelc" {
            compiled = packageURL
        } else {
            // sibling .mlmodelc wins (pre-compiled cache); otherwise compile
            let sibling = packageURL.deletingPathExtension().appendingPathExtension("mlmodelc")
            if FileManager.default.fileExists(atPath: sibling.path) {
                compiled = sibling
            } else {
                let fresh = try MLModel.compileModel(at: packageURL)
                // best-effort cache beside the package so the next load skips
                // compilation; falls back to the temp copy on a read-only dir
                if (try? FileManager.default.moveItem(at: fresh, to: sibling)) != nil {
                    compiled = sibling
                } else {
                    compiled = fresh
                }
            }
        }
        return try MLModel(contentsOf: compiled, configuration: config)
    }

    static func floatArray(_ shape: [Int], _ data: [Float]) throws -> MLMultiArray {
        let m = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        data.withUnsafeBufferPointer {
            m.dataPointer.bindMemory(to: Float.self, capacity: data.count)
                .update(from: $0.baseAddress!, count: data.count)
        }
        return m
    }

    static func intArray(_ shape: [Int], _ data: [Int32]) throws -> MLMultiArray {
        let m = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .int32)
        data.withUnsafeBufferPointer {
            m.dataPointer.bindMemory(to: Int32.self, capacity: data.count)
                .update(from: $0.baseAddress!, count: data.count)
        }
        return m
    }

    static func predict(_ model: MLModel, _ inputs: [String: MLMultiArray], stage: String) throws -> [Float] {
        let provider = try MLDictionaryFeatureProvider(
            dictionary: inputs.mapValues { MLFeatureValue(multiArray: $0) })
        let out: MLFeatureProvider
        do {
            out = try model.prediction(from: provider)
        } catch {
            throw LoudKitError.prediction("\(stage): \(error.localizedDescription)")
        }
        guard let name = out.featureNames.first,
              let array = out.featureValue(for: name)?.multiArrayValue else {
            throw LoudKitError.prediction("\(stage): no multiarray output")
        }
        return floats(array)
    }

    /// Stride-aware MLMultiArray -> [Float]. CoreML can return NON-CONTIGUOUS
    /// multiarrays (padded strides); reading `dataPointer` linearly scrambles
    /// the values — the production engine hit exactly this (2026-07-22, mu
    /// corr 0.025 vs 1.0 once fixed; MLArrayReader.swift), and this port hit
    /// it again on its first end-to-end run (mel corr 0.13). Every model
    /// output goes through here.
    static func floats(_ m: MLMultiArray) -> [Float] {
        let shape = m.shape.map { $0.intValue }
        let strides = m.strides.map { $0.intValue }
        let count = m.count
        let nd = shape.count
        let fp16 = m.dataType == .float16
        guard count > 0, nd > 0 else { return [] }
        let base = m.dataPointer

        var contiguous = true
        var expected = 1
        for d in stride(from: nd - 1, through: 0, by: -1) {
            if strides[d] != expected {
                contiguous = false
                break
            }
            expected *= shape[d]
        }

        var res = [Float](repeating: 0, count: count)
        res.withUnsafeMutableBufferPointer { rb in
            let dst = rb.baseAddress!
            if contiguous && !fp16 {
                _ = memcpy(dst, base, count * 4)
                return
            }
            let last = shape[nd - 1]
            if strides[nd - 1] == 1 && last > 0 {
                // last dim contiguous: copy row by row
                let rows = count / last
                var idx = [Int](repeating: 0, count: max(1, nd - 1))
                for r in 0..<rows {
                    var off = 0
                    for d in 0..<(nd - 1) { off += idx[d] * strides[d] }
                    if fp16 {
                        let sp = base.advanced(by: off * 2).bindMemory(to: Float16.self, capacity: last)
                        for k in 0..<last { dst[r * last + k] = Float(sp[k]) }
                    } else {
                        _ = memcpy(dst + r * last, base.advanced(by: off * 4), last * 4)
                    }
                    var d = nd - 2
                    while d >= 0 {
                        idx[d] += 1
                        if idx[d] < shape[d] { break }
                        idx[d] = 0
                        d -= 1
                    }
                }
            } else {
                // fully strided fallback
                var idx = [Int](repeating: 0, count: nd)
                for lin in 0..<count {
                    var off = 0
                    for d in 0..<nd { off += idx[d] * strides[d] }
                    dst[lin] = fp16
                        ? Float(base.load(fromByteOffset: off * 2, as: Float16.self))
                        : base.load(fromByteOffset: off * 4, as: Float.self)
                    var d = nd - 1
                    while d >= 0 {
                        idx[d] += 1
                        if idx[d] < shape[d] { break }
                        idx[d] = 0
                        d -= 1
                    }
                }
            }
        }
        return res
    }
}

/// Speech tokens + voice -> mel, via the CoreML encoder and estimator.
/// Geometry and recipe identical to `loudkit.backends.coreml_backend`.
public final class MelDecoder {
    static let flowNoiseStream: UInt32 = 0
    static let melBins = 80

    private let config: AlgorithmConfig
    private let encoder: MLModel
    private let estimator: MLModel
    /// The 192->80 speaker projection lives in the torch flow module and is
    /// baked into neither graph; it is read from the checkpoint
    /// (`s3gen.flow.spk_embed_affine_layer`) so this path computes the
    /// identical `spks`.
    private let spkWeight: [Float]  // [80, 192]
    private let spkBias: [Float]

    init(config: AlgorithmConfig, encoder: MLModel, estimator: MLModel,
         spkWeight: [Float], spkBias: [Float]) throws {
        guard config.guidance == .singlePath else {
            throw LoudKitError.manifest(
                "the exported estimator is the guidance-distilled student; "
                + "cfg_dual_path would apply guidance twice (EXP-016)")
        }
        guard config.window.staticLength == 255, config.window.staticPromptTokens == 238 else {
            throw LoudKitError.shape(
                "the exported graphs are static at query 255 / prompt 238; this "
                + "AlgorithmConfig frames \(String(describing: config.window.staticLength))/"
                + "\(String(describing: config.window.staticPromptTokens)). A different "
                + "window is a different algorithm — re-export rather than reframe.")
        }
        self.config = config
        self.encoder = encoder
        self.estimator = estimator
        self.spkWeight = spkWeight
        self.spkBias = spkBias
    }

    /// Returns `(mel, frames)` — mel is `(80, 2n)` row-major, the prompt
    /// region already cut.
    public func decode(tokens: [Int], voice: VoiceProfile, seed: UInt64) throws -> ([Float], Int) {
        let w = config.window
        guard let staticLen = w.staticLength, let promptLen = w.staticPromptTokens,
              let pad = w.padTokenId else {
            throw LoudKitError.shape("static window not configured")
        }
        // Refused, not truncated — see Engine.stripSpecials. This site sliced
        // independently of that one, so a caller reaching the renderer
        // directly lost the tail of the passage with nothing raised anywhere.
        try Windowing.requireFits(tokens.count, w.maxSpeechTokens)
        let toks = tokens.map { Int32($0) }
        let n = toks.count
        let tMel = 2 * (promptLen + staticLen)

        // window recipe: prompt framed to exactly promptLen (truncate long,
        // silence-pad short), query padded to staticLen with the silence unit
        var prompt = [Int32](repeating: Int32(pad), count: promptLen)
        for (i, t) in voice.promptTokens.prefix(promptLen).enumerated() { prompt[i] = Int32(t) }
        var query = [Int32](repeating: Int32(pad), count: staticLen)
        for (i, t) in toks.enumerated() { query[i] = t }

        let mu = try MLHelpers.predict(
            encoder,
            ["prompt_token": try MLHelpers.intArray([1, promptLen], prompt),
             "speech_tokens": try MLHelpers.intArray([1, staticLen], query)],
            stage: "encoder")
        guard mu.count == Self.melBins * tMel else {
            throw LoudKitError.shape("encoder output \(mu.count) != \(Self.melBins * tMel)")
        }

        // spks = affine(normalize(flow_embedding))
        var norm: Float = 0
        for v in voice.flowEmbedding { norm += v * v }
        norm = norm.squareRoot()
        let emb = voice.flowEmbedding.map { $0 / norm }
        var spks = [Float](repeating: 0, count: Self.melBins)
        let k = emb.count
        for r in 0..<Self.melBins {
            var acc: Float = 0
            for c in 0..<k { acc += spkWeight[r * k + c] * emb[c] }
            spks[r] = acc + spkBias[r]
        }

        // prompt mel condition, zero-padded to the full window
        var cond = [Float](repeating: 0, count: Self.melBins * tMel)
        let promptFrames = 2 * promptLen
        let keep = min(voice.promptMelFrames, promptFrames)
        for c in 0..<Self.melBins {
            for f in 0..<keep {
                cond[c * tMel + f] = voice.promptMel[c * voice.promptMelFrames + f]
            }
        }

        var x = Noise.gaussianField(seed: seed, stream: Self.flowNoiseStream,
                                    rows: Self.melBins, cols: tMel)
        let grid = config.timeGrid()
        let muArr = try MLHelpers.floatArray([1, Self.melBins, tMel], mu)
        let condArr = try MLHelpers.floatArray([1, Self.melBins, tMel], cond)
        let spksArr = try MLHelpers.floatArray([1, Self.melBins], spks)
        for i in 0..<(grid.count - 1) {
            let v = try MLHelpers.predict(
                estimator,
                ["x": try MLHelpers.floatArray([1, Self.melBins, tMel], x),
                 "mu": muArr,
                 "t": try MLHelpers.floatArray([1], [Float(grid[i])]),
                 "spks": spksArr,
                 "cond": condArr],
                stage: "estimator[\(i)]")
            guard v.count == x.count else {
                throw LoudKitError.shape("estimator output \(v.count) != \(x.count)")
            }
            let dt = Float(grid[i + 1] - grid[i])
            for j in 0..<x.count { x[j] += dt * v[j] }
        }

        // cut the prompt reconstruction; return only the real speech region
        let frames = 2 * n
        var mel = [Float](repeating: 0, count: Self.melBins * frames)
        for c in 0..<Self.melBins {
            for f in 0..<frames {
                mel[c * frames + f] = x[c * tMel + promptFrames + f]
            }
        }
        return (mel, frames)
    }
}

/// Mel -> waveform via the fp32 HiFT graph. Phase offsets and excitation
/// noise are Philox-addressed inputs, identical bits to the Python backend.
public final class Vocoder {
    static let phaseStream: UInt32 = 0
    static let noiseStream: UInt32 = 1
    static let nHarmonics = 9
    static let upsamplePerFrame = 480

    private let config: AlgorithmConfig
    private let hift: MLModel

    init(config: AlgorithmConfig, hift: MLModel) {
        self.config = config
        self.hift = hift
    }

    public func synthesize(mel: [Float], frames melFrames: Int, seed: UInt64) throws -> [Float] {
        let bins = MelDecoder.melBins
        let staticFrames = 2 * config.window.maxSpeechTokens
        let nFrames = min(melFrames, staticFrames)
        var padded = [Float](repeating: 0, count: bins * staticFrames)
        for c in 0..<bins {
            for f in 0..<nFrames {
                padded[c * staticFrames + f] = mel[c * melFrames + f]
            }
        }
        let nSamples = staticFrames * Self.upsamplePerFrame
        var phase = [Float](repeating: 0, count: Self.nHarmonics)
        let offsets = Noise.symmetricUniforms(
            seed: seed, stream: Self.phaseStream, n: Self.nHarmonics - 1, halfWidth: Double.pi)
        for h in 1..<Self.nHarmonics { phase[h] = offsets[h - 1] }
        let noise = Noise.gaussianField(
            seed: seed, stream: Self.noiseStream, rows: Self.nHarmonics, cols: nSamples)

        let wav = try MLHelpers.predict(
            hift,
            ["mel": try MLHelpers.floatArray([1, bins, staticFrames], padded),
             "phase": try MLHelpers.floatArray([1, Self.nHarmonics, 1], phase),
             "noise": try MLHelpers.floatArray([1, Self.nHarmonics, nSamples], noise)],
            stage: "vocoder")
        return Array(wav.prefix(nFrames * Self.upsamplePerFrame))
    }
}
