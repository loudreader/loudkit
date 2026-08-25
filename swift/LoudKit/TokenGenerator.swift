import Accelerate
import Foundation

/// The token generator: a Llama-architecture decoder that writes speech,
/// implemented natively (Accelerate BLAS, fp32) against the packed `t3.*`
/// weights.
///
/// Why native rather than the app's CoreML T3: the stateful multi-function
/// export has no validated cross-implementation harness (the sample wall's
/// "Not covered: T3 on the ANE", and torch segfaults beside CoreML in one
/// process on the Python side), and a wrong row is worse than a missing one.
/// A native fp32 generator is the same *declared execution* as the Python
/// conformance engine (torch cpu, fp32 compute over fp16-stored weights), so
/// "same tokens from the same seed" is a meaningful cross-language claim
/// rather than a coincidence of two unvalidated graphs. The algorithm layer —
/// sequence layout, learned positions, EOS floor, sampler injection — is
/// lifted from the shipped runner (ChatterboxT3Runner.swift) and from
/// `loudkit.models.generator`, which are the same algorithm by construction.
///
/// Sequence layout, identical to both references:
///
///     [ cond (34) | [START] text [STOP] | speech: START, s0, s1, ... ]
///
/// The EOS floor is applied here, before the sampler sees the logits — an EOS
/// *policy* belongs to the generator, not to sampler exemptions.
public final class TokenGenerator {
    static let startTextToken = 255
    static let stopTextToken = 0

    private let config: AlgorithmConfig
    private let hidden: Int
    private let nLayers: Int
    private let nHeads: Int
    private let nKV: Int
    private let headDim: Int
    private let intermediate: Int
    private let rmsEps: Float

    private struct Layer {
        var qProj: [Float]  // [nHeads*headDim, hidden]
        var kProj: [Float]  // [nKV*headDim, hidden]
        var vProj: [Float]
        var oProj: [Float]  // [hidden, nHeads*headDim]
        var gate: [Float]   // [intermediate, hidden]
        var up: [Float]
        var down: [Float]   // [hidden, intermediate]
        var inputNorm: [Float]
        var postNorm: [Float]
    }

    private var layers: [Layer] = []
    private var finalNorm: [Float]
    private let speechEmb: [Float]     // [8194, hidden]
    private let textEmb: [Float]       // [textVocab, hidden]
    private let speechPos: [Float]     // [4100, hidden]
    private let textPos: [Float]       // [2050, hidden]
    private let speechHead: [Float]    // [8194, hidden]
    // cond encoder
    private let spkrW: [Float]         // [hidden, 256]
    private let spkrB: [Float]
    private let emotionW: [Float]      // [hidden, 1], no bias
    private let percQuery: [Float]     // [32, hidden]
    private let percNormW: [Float]
    private let percNormB: [Float]
    private let percQ: [Float]
    private let percQb: [Float]
    private let percK: [Float]
    private let percKb: [Float]
    private let percV: [Float]
    private let percVb: [Float]
    private let percOut: [Float]
    private let percOutB: [Float]

    private var invFreq: [Float] = []  // [headDim/2]

    public init(checkpoint: Checkpoint, config: AlgorithmConfig) throws {
        self.config = config
        guard let llama = checkpoint.manifest["llama_config"] as? [String: Any] else {
            throw LoudKitError.manifest("manifest is missing llama_config")
        }
        hidden = (llama["hidden_size"] as? NSNumber)?.intValue ?? 1024
        nLayers = (llama["num_hidden_layers"] as? NSNumber)?.intValue ?? 16
        nHeads = (llama["num_attention_heads"] as? NSNumber)?.intValue ?? 16
        nKV = (llama["num_key_value_heads"] as? NSNumber)?.intValue ?? 4
        headDim = (llama["head_dim"] as? NSNumber)?.intValue ?? 64
        intermediate = (llama["intermediate_size"] as? NSNumber)?.intValue ?? 2100
        rmsEps = (llama["rms_norm_eps"] as? NSNumber)?.floatValue ?? 1e-5

        // Corroborated against the weights before a single allocation.
        //
        // This was the only port that took the manifest's word for its
        // dimensions. Python refuses a manifest the tensors cannot fill --
        // `intermediate_size: 16_000_000` beside a 26 MB file asks for 197 GB
        // and gets a `ValueError` naming the field -- while here the same
        // manifest reached `matmulT`, which trusts the shapes it is handed and
        // reads past the end of the buffer.
        //
        // Whole shapes, not row counts: a `(16_000_000, 0)` matrix weighs
        // almost nothing on disk and satisfies a row check, which is the hole
        // the Python version had until this week. A projection here is always
        // `(something, hidden_size)`, so both halves are knowable.
        //
        // Layer zero only. If layer zero is consistent, the rest are the same
        // architecture or `floats(_:)` refuses them by size when it reads them;
        // checking all sixteen would read every header to say the same thing.
        let expected: [(String, [Int])] = [
            ("t3.tfmr.layers.0.mlp.gate_proj.weight", [intermediate, hidden]),
            ("t3.tfmr.layers.0.self_attn.q_proj.weight", [nHeads * headDim, hidden]),
            ("t3.tfmr.layers.0.self_attn.k_proj.weight", [nKV * headDim, hidden]),
        ]
        for (name, want) in expected {
            let got = try checkpoint.store.shape(name)
            guard got == want else {
                throw LoudKitError.manifest(
                    "\(name) is \(got) and the manifest describes \(want). Refusing "
                        + "before building a model the checkpoint cannot fill.")
            }
        }
        guard nHeads > 0, nKV > 0, hidden > 0, headDim > 0, intermediate > 0, nLayers > 0
        else {
            throw LoudKitError.manifest(
                "llama_config has a non-positive dimension: hidden=\(hidden) "
                    + "layers=\(nLayers) heads=\(nHeads) kv=\(nKV) "
                    + "head_dim=\(headDim) intermediate=\(intermediate)")
        }

        let s = checkpoint.store
        func f(_ name: String) throws -> [Float] { try s.floats("t3." + name) }

        for i in 0..<nLayers {
            let p = "tfmr.layers.\(i)."
            layers.append(Layer(
                qProj: try f(p + "self_attn.q_proj.weight"),
                kProj: try f(p + "self_attn.k_proj.weight"),
                vProj: try f(p + "self_attn.v_proj.weight"),
                oProj: try f(p + "self_attn.o_proj.weight"),
                gate: try f(p + "mlp.gate_proj.weight"),
                up: try f(p + "mlp.up_proj.weight"),
                down: try f(p + "mlp.down_proj.weight"),
                inputNorm: try f(p + "input_layernorm.weight"),
                postNorm: try f(p + "post_attention_layernorm.weight")))
        }
        finalNorm = try f("tfmr.norm.weight")
        speechEmb = try f("speech_emb.weight")
        textEmb = try f("text_emb.weight")
        speechPos = try f("speech_pos_emb.emb.weight")
        textPos = try f("text_pos_emb.emb.weight")
        speechHead = try f("speech_head.weight")
        spkrW = try f("cond_enc.spkr_enc.weight")
        spkrB = try f("cond_enc.spkr_enc.bias")
        emotionW = try f("cond_enc.emotion_adv_fc.weight")
        percQuery = try f("cond_enc.perceiver.pre_attention_query")
        percNormW = try f("cond_enc.perceiver.attn.norm.weight")
        percNormB = try f("cond_enc.perceiver.attn.norm.bias")
        percQ = try f("cond_enc.perceiver.attn.to_q.weight")
        percQb = try f("cond_enc.perceiver.attn.to_q.bias")
        percK = try f("cond_enc.perceiver.attn.to_k.weight")
        percKb = try f("cond_enc.perceiver.attn.to_k.bias")
        percV = try f("cond_enc.perceiver.attn.to_v.weight")
        percVb = try f("cond_enc.perceiver.attn.to_v.bias")
        percOut = try f("cond_enc.perceiver.attn.proj_out.weight")
        percOutB = try f("cond_enc.perceiver.attn.proj_out.bias")

        // RoPE inverse frequencies, llama3 wavelength-dependent rescale,
        // fp64 until the end — verbatim the rule in loudkit.models.generator.
        let rope = llama["rope_scaling"] as? [String: Any] ?? [:]
        let theta = (llama["rope_theta"] as? NSNumber)?.doubleValue ?? 500_000.0
        let factor = (rope["factor"] as? NSNumber)?.doubleValue ?? 8.0
        let lowFF = (rope["low_freq_factor"] as? NSNumber)?.doubleValue ?? 1.0
        let highFF = (rope["high_freq_factor"] as? NSNumber)?.doubleValue ?? 4.0
        let origMax = (rope["original_max_position_embeddings"] as? NSNumber)?.doubleValue ?? 8192
        var freqs: [Float] = []
        for d in stride(from: 0, to: headDim, by: 2) {
            let exponent = Double(d) / Double(headDim)
            let inv = 1.0 / Foundation.pow(theta, exponent)
            let wavelen = 2.0 * Double.pi / inv
            let lowWavelen = origMax / lowFF
            let highWavelen = origMax / highFF
            let value: Double
            if wavelen < highWavelen {
                value = inv
            } else if wavelen > lowWavelen {
                value = inv / factor
            } else {
                let smooth = (origMax / wavelen - lowFF) / (highFF - lowFF)
                value = (1.0 - smooth) * inv / factor + smooth * inv
            }
            freqs.append(Float(value))
        }
        invFreq = freqs
    }

    // MARK: linear algebra helpers

    /// c[m x n] = a[m x k] @ w[n x k]^T  (torch Linear layout)
    private func matmulT(_ a: [Float], m: Int, k: Int, w: [Float], n: Int) -> [Float] {
        var c = [Float](repeating: 0, count: m * n)
        a.withUnsafeBufferPointer { ap in
            w.withUnsafeBufferPointer { wp in
                c.withUnsafeMutableBufferPointer { cp in
                    cblas_sgemm(
                        CblasRowMajor, CblasNoTrans, CblasTrans,
                        Int32(m), Int32(n), Int32(k),
                        1.0, ap.baseAddress, Int32(k),
                        wp.baseAddress, Int32(k),
                        0.0, cp.baseAddress, Int32(n))
                }
            }
        }
        return c
    }

    private func addBias(_ x: inout [Float], bias: [Float], rows: Int) {
        let n = bias.count
        for r in 0..<rows {
            for i in 0..<n { x[r * n + i] += bias[i] }
        }
    }

    private func rmsNorm(_ x: [Float], rows: Int, weight: [Float]) -> [Float] {
        var out = [Float](repeating: 0, count: x.count)
        let d = hidden
        for r in 0..<rows {
            var ss: Double = 0
            for i in 0..<d { ss += Double(x[r * d + i]) * Double(x[r * d + i]) }
            let scale = Float(1.0 / (ss / Double(d) + Double(rmsEps)).squareRoot())
            for i in 0..<d { out[r * d + i] = weight[i] * (x[r * d + i] * scale) }
        }
        return out
    }

    private func layerNorm(_ x: [Float], rows: Int, weight: [Float], bias: [Float]) -> [Float] {
        var out = [Float](repeating: 0, count: x.count)
        let d = weight.count
        for r in 0..<rows {
            var mean: Double = 0
            for i in 0..<d { mean += Double(x[r * d + i]) }
            mean /= Double(d)
            var varSum: Double = 0
            for i in 0..<d {
                let dv = Double(x[r * d + i]) - mean
                varSum += dv * dv
            }
            let inv = 1.0 / (varSum / Double(d) + 1e-5).squareRoot()
            for i in 0..<d {
                out[r * d + i] = Float((Double(x[r * d + i]) - mean) * inv) * weight[i] + bias[i]
            }
        }
        return out
    }

    /// Softmax in place over `count` scores.
    ///
    /// Scalar on purpose. The vDSP form (vDSP_maxv, vvexpf, vDSP_sve,
    /// vDSP_vsdiv) was measured at 98.9 tok/s against 98.5 for this one, which
    /// is noise: softmax is O(visible) beside two gemv calls that are
    /// O(visible x 64), so it is a sixty-fourth of the attention it sits in.
    /// The scalar version is the one whose summation order matches the
    /// reference implementations, and it costs nothing to keep.
    private func softmaxRow(_ x: UnsafeMutablePointer<Float>, count: Int) {
        var maxV = -Float.infinity
        for i in 0..<count where x[i] > maxV { maxV = x[i] }
        var sum: Float = 0
        for i in 0..<count {
            let e = Foundation.expf(x[i] - maxV)
            x[i] = e
            sum += e
        }
        for i in 0..<count { x[i] /= sum }
    }

    // MARK: rope

    private func ropeTables(length: Int) -> (cos: [Float], sin: [Float]) {
        // angle layout matches cat(freqs, freqs): dim d and d + headDim/2
        // share the frequency invFreq[d]
        let half = headDim / 2
        var cosT = [Float](repeating: 0, count: length * headDim)
        var sinT = [Float](repeating: 0, count: length * headDim)
        for p in 0..<length {
            for d in 0..<half {
                let angle = Float(p) * invFreq[d]
                let c = Foundation.cosf(angle)
                let s = Foundation.sinf(angle)
                cosT[p * headDim + d] = c
                cosT[p * headDim + d + half] = c
                sinT[p * headDim + d] = s
                sinT[p * headDim + d + half] = s
            }
        }
        return (cosT, sinT)
    }

    /// In-place rotary embedding on one head-vector at `offset`:
    /// q <- q*cos + rotate_half(q)*sin
    private func applyRope(
        _ x: inout [Float], offset: Int, cosRow: Int, tables: (cos: [Float], sin: [Float])
    ) {
        let half = headDim / 2
        let cBase = cosRow * headDim
        var rotated = [Float](repeating: 0, count: headDim)
        for i in 0..<half { rotated[i] = -x[offset + half + i] }
        for i in 0..<half { rotated[half + i] = x[offset + i] }
        for i in 0..<headDim {
            x[offset + i] = x[offset + i] * tables.cos[cBase + i] + rotated[i] * tables.sin[cBase + i]
        }
    }

    // MARK: conditioning

    private func condEmbeds(voice: VoiceProfile) -> [Float] {
        // speaker slot
        var spkr = matmulT(voice.speakerEmbedding, m: 1, k: voice.speakerEmbedding.count,
                           w: spkrW, n: hidden)
        addBias(&spkr, bias: spkrB, rows: 1)

        // perceiver over the conditioning prompt (speech emb + speech pos)
        let plen = voice.condPromptTokens.count
        var prompt = [Float](repeating: 0, count: plen * hidden)
        for (i, tok) in voice.condPromptTokens.enumerated() {
            for d in 0..<hidden {
                prompt[i * hidden + d] = speechEmb[tok * hidden + d] + speechPos[i * hidden + d]
            }
        }
        let pass1 = perceiverPass(x1: Array(percQuery), t: 32, x2: prompt, s: plen)
        let resampled = perceiverPass(x1: pass1, t: 32, x2: pass1, s: 32)

        // emotion slot: Linear(1 -> hidden, bias=false) on a scalar. The axis
        // is dead on these weights; the slot is fed the training constant.
        var emo = [Float](repeating: 0, count: hidden)
        for d in 0..<hidden { emo[d] = emotionW[d] * VoiceProfile.emotionNeutral }

        return spkr + resampled + emo
    }

    /// One shared perceiver attention block (the original uses the same norm
    /// and projections for both passes; the weights exist once).
    private func perceiverPass(x1: [Float], t: Int, x2: [Float], s: Int) -> [Float] {
        let heads = 4
        let hd = hidden / heads  // 256
        var q = matmulT(layerNorm(x1, rows: t, weight: percNormW, bias: percNormB),
                        m: t, k: hidden, w: percQ, n: hidden)
        addBias(&q, bias: percQb, rows: t)
        let x2n = layerNorm(x2, rows: s, weight: percNormW, bias: percNormB)
        var k = matmulT(x2n, m: s, k: hidden, w: percK, n: hidden)
        addBias(&k, bias: percKb, rows: s)
        var v = matmulT(x2n, m: s, k: hidden, w: percV, n: hidden)
        addBias(&v, bias: percVb, rows: s)

        var attnOut = [Float](repeating: 0, count: t * hidden)
        let scale = 1.0 / Float(Double(hd).squareRoot())
        for h in 0..<heads {
            for i in 0..<t {
                var scores = [Float](repeating: 0, count: s)
                for j in 0..<s {
                    var dot: Float = 0
                    for d in 0..<hd {
                        dot += q[i * hidden + h * hd + d] * k[j * hidden + h * hd + d]
                    }
                    scores[j] = dot * scale
                }
                scores.withUnsafeMutableBufferPointer { softmaxRow($0.baseAddress!, count: s) }
                for j in 0..<s {
                    let w = scores[j]
                    for d in 0..<hd {
                        attnOut[i * hidden + h * hd + d] += w * v[j * hidden + h * hd + d]
                    }
                }
            }
        }
        var proj = matmulT(attnOut, m: t, k: hidden, w: percOut, n: hidden)
        addBias(&proj, bias: percOutB, rows: t)
        for i in 0..<(t * hidden) { proj[i] += x1[i] }
        return proj
    }

    // MARK: transformer

    private final class Cache {
        var k: [[Float]]  // per layer: [nKV * maxLen * headDim]
        var v: [[Float]]
        var length = 0
        let maxLen: Int
        init(layers: Int, nKV: Int, maxLen: Int, headDim: Int) {
            self.maxLen = maxLen
            k = Array(repeating: [Float](repeating: 0, count: nKV * maxLen * headDim), count: layers)
            v = Array(repeating: [Float](repeating: 0, count: nKV * maxLen * headDim), count: layers)
        }
    }

    /// Scaled dot-product attention for one layer, over the cache.
    ///
    /// BLAS rather than scalar loops, which is what the cache layout was
    /// already shaped for: at a fixed layer and kv head,
    /// `k[(kv * maxLen + j) * headDim + d]` is a contiguous row-major
    /// [maxLen x headDim] block, so the visible prefix is a matrix and each of
    /// the two loops is one matrix-vector product. Scalar, the pair cost about
    /// 6.5 million operations per token and grew with the context.
    ///
    /// `scores` is the caller's buffer, at least `maxLen` long, so nothing is
    /// allocated per head. `out` is zero on entry and each (head, row) writes
    /// its own `headDim` slice exactly once, which is why beta is 0.
    private func attend(
        q: [Float], kCache: [Float], vCache: [Float],
        scores: inout [Float], out: inout [Float],
        rows: Int, past: Int, maxLen: Int
    ) {
        let qDim = nHeads * headDim
        let scale = 1.0 / Float(Double(headDim).squareRoot())
        let group = nHeads / nKV
        kCache.withUnsafeBufferPointer { kp in
            vCache.withUnsafeBufferPointer { vp in
                q.withUnsafeBufferPointer { qp in
                    scores.withUnsafeMutableBufferPointer { sp in
                        out.withUnsafeMutableBufferPointer { op in
                            for h in 0..<nHeads {
                                let kvBase = (h / group) * maxLen * headDim
                                for r in 0..<rows {
                                    let visible = past + r + 1  // causal: keys 0...(past+r)
                                    let base = r * qDim + h * headDim
                                    // alpha is left at 1 and the scale is its
                                    // own pass: BLAS may fold alpha in
                                    // anywhere, and the scalar version scaled
                                    // the finished sum. Scores must not move.
                                    cblas_sgemv(
                                        CblasRowMajor, CblasNoTrans,
                                        Int32(visible), Int32(headDim),
                                        1.0, kp.baseAddress! + kvBase, Int32(headDim),
                                        qp.baseAddress! + base, 1,
                                        0.0, sp.baseAddress!, 1)
                                    var s = scale
                                    vDSP_vsmul(sp.baseAddress!, 1, &s, sp.baseAddress!, 1,
                                               vDSP_Length(visible))
                                    softmaxRow(sp.baseAddress!, count: visible)
                                    cblas_sgemv(
                                        CblasRowMajor, CblasTrans,
                                        Int32(visible), Int32(headDim),
                                        1.0, vp.baseAddress! + kvBase, Int32(headDim),
                                        sp.baseAddress!, 1,
                                        0.0, op.baseAddress! + base, 1)
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// One forward over `rows` new tokens (prefill: rows > 1 causal; decode:
    /// rows == 1 attending over the cache). Returns final hidden states.
    private func forward(
        _ embeds: [Float], rows: Int, positions: [Int], cache: Cache
    ) -> [Float] {
        let maxPos = positions.max()! + 1
        let tables = ropeTables(length: maxPos)
        let qDim = nHeads * headDim
        let kvDim = nKV * headDim
        let past = cache.length
        var x = embeds
        // One score buffer for every layer, head and row, sized to the cache
        // rather than to the visible prefix. Allocated inside the head loop it
        // was 16 heads x 16 layers = 256 heap allocations per token.
        var scores = [Float](repeating: 0, count: cache.maxLen)

        for (li, layer) in layers.enumerated() {
            let xn = rmsNorm(x, rows: rows, weight: layer.inputNorm)
            var q = matmulT(xn, m: rows, k: hidden, w: layer.qProj, n: qDim)
            var kNew = matmulT(xn, m: rows, k: hidden, w: layer.kProj, n: kvDim)
            let vNew = matmulT(xn, m: rows, k: hidden, w: layer.vProj, n: kvDim)

            for r in 0..<rows {
                for h in 0..<nHeads {
                    applyRope(&q, offset: r * qDim + h * headDim, cosRow: positions[r], tables: tables)
                }
                for h in 0..<nKV {
                    applyRope(&kNew, offset: r * kvDim + h * headDim, cosRow: positions[r], tables: tables)
                }
            }

            // append to cache
            for r in 0..<rows {
                let slot = past + r
                for h in 0..<nKV {
                    let dst = (h * cache.maxLen + slot) * headDim
                    let src = r * kvDim + h * headDim
                    for d in 0..<headDim {
                        cache.k[li][dst + d] = kNew[src + d]
                        cache.v[li][dst + d] = vNew[src + d]
                    }
                }
            }

            var attnOut = [Float](repeating: 0, count: rows * qDim)
            attend(q: q, kCache: cache.k[li], vCache: cache.v[li],
                   scores: &scores, out: &attnOut,
                   rows: rows, past: past, maxLen: cache.maxLen)
            let attnProj = matmulT(attnOut, m: rows, k: qDim, w: layer.oProj, n: hidden)
            for i in 0..<(rows * hidden) { x[i] += attnProj[i] }

            let xn2 = rmsNorm(x, rows: rows, weight: layer.postNorm)
            var gate = matmulT(xn2, m: rows, k: hidden, w: layer.gate, n: intermediate)
            let up = matmulT(xn2, m: rows, k: hidden, w: layer.up, n: intermediate)
            for i in 0..<(rows * intermediate) {
                let g = gate[i]
                gate[i] = (g / (1 + Foundation.expf(-g))) * up[i]  // silu(g) * up
            }
            let downOut = matmulT(gate, m: rows, k: intermediate, w: layer.down, n: hidden)
            for i in 0..<(rows * hidden) { x[i] += downOut[i] }
        }
        cache.length += rows
        return rmsNorm(x, rows: rows, weight: finalNorm)
    }

    private func speechLogits(_ hiddenRow: [Float]) -> [Float] {
        matmulT(hiddenRow, m: 1, k: hidden, w: speechHead, n: config.speechVocabSize)
    }

    // MARK: contract

    public struct Generation {
        public let rawTokens: [Int]  // includes the natural stop token when it fired
        public let hitTokenCap: Bool
    }

    /// Autoregressive decode to the stop token or the cap. The sampler owns
    /// the law; this loop owns only the EOS floor and the `seen` bookkeeping.
    ///
    /// `prefix` holds speech tokens from the preceding chunk — fed through
    /// for context, seeded into the repetition-penalty state, and dropped
    /// from the result. Same contract as the Python protocol, present from
    /// the first release because adding it later would break implementers.
    /// `onStep` is a cooperative hook called once per decoded token — for
    /// progress UIs, and for harnesses that must duty-cycle a long burst
    /// (iOS kills a background process that holds >80% CPU over 60 s; the
    /// demo app's benchmark sleeps inside this hook and subtracts the slept
    /// time from what it reports). It runs on the decoding thread; whatever
    /// it spends is part of the caller's wall clock.
    public func generate(
        textTokens: [Int], voice: VoiceProfile, sampler: LRSamplerV1,
        maxNewTokens: Int? = nil, prefix: [Int] = [], onStep: (() -> Void)? = nil,
        shouldCancel: (() -> Bool)? = nil
    ) -> Generation {
        let cap = maxNewTokens ?? config.sampling.maxNewTokens
        let floor = config.eosFloor(nTextTokens: textTokens.count)
        let stop = config.stopSpeechToken

        // [ cond | START text STOP | speech START | prefix... ]
        let cond = condEmbeds(voice: voice)
        let framed = [Self.startTextToken] + textTokens + [Self.stopTextToken]
        var embeds = cond
        embeds.reserveCapacity((cond.count / hidden + framed.count + 1 + prefix.count) * hidden)
        for (k, tok) in framed.enumerated() {
            for d in 0..<hidden {
                embeds.append(textEmb[tok * hidden + d] + textPos[k * hidden + d])
            }
        }
        for d in 0..<hidden {
            embeds.append(speechEmb[config.startSpeechToken * hidden + d] + speechPos[d])
        }
        var seen = [Bool](repeating: false, count: config.speechVocabSize)
        for (k, tok) in prefix.enumerated() {
            for d in 0..<hidden {
                embeds.append(speechEmb[tok * hidden + d] + speechPos[(k + 1) * hidden + d])
            }
            seen[tok] = true
        }
        let prefillLen = embeds.count / hidden

        let cache = Cache(layers: nLayers, nKV: nKV, maxLen: prefillLen + cap + 1, headDim: headDim)
        let hiddenStates = forward(embeds, rows: prefillLen, positions: Array(0..<prefillLen), cache: cache)
        var logits = speechLogits(Array(hiddenStates[(prefillLen - 1) * hidden..<prefillLen * hidden]))

        var out: [Int] = []
        for step in 0..<cap {
            if shouldCancel?() == true { break } // token-level barge-in
            if out.count < floor { logits[stop] = -Float.infinity }
            let token = sampler.sample(logits: logits, step: step, seen: seen)
            out.append(token)
            onStep?()
            if token == stop { break }
            seen[token] = true
            var emb = [Float](repeating: 0, count: hidden)
            for d in 0..<hidden {
                emb[d] = speechEmb[token * hidden + d]
                    + speechPos[(prefix.count + step + 1) * hidden + d]
            }
            let h = forward(emb, rows: 1, positions: [prefillLen + step], cache: cache)
            logits = speechLogits(h)
        }
        return Generation(rawTokens: out, hitTokenCap: out.count >= cap && !out.contains(stop))
    }
}
