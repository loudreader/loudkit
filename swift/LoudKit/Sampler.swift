import Foundation

/// LR-SAMPLER-v1 — the third independent implementation of the sampling law.
///
/// Same semantics, to the bit, as `loudkit.sampler.LRSamplerV1`, and verified
/// against it through the conformance fixture: the RNG is Philox-addressed
/// (a token's randomness is a pure function of `(seed, step)`), min_p is
/// evaluated in logit space (no softmax, hence no reduction whose order a
/// backend could vary), and selection is Gumbel-argmax with ties broken
/// toward the lowest index. All sampling arithmetic runs in Double, exactly
/// as the Python side promotes to float64.
public final class LRSamplerV1 {
    public static let version = "LR-SAMPLER-v1"
    static let samplingStream: UInt32 = 0

    private let config: SamplingConfig
    private let seed: UInt64
    private let block: Int
    private var noise: [Double] = []
    private var noiseBase = -1
    private var noiseWidth = -1
    private let isSilence: Set<Int>

    /// Observation of how close each step came to stopping. Never feeds back
    /// into the draw; read by the postprocess detectors after generation. `nil`
    /// disables it, and with it its cost — one exponential and one sum over the
    /// vocabulary per step.
    private var stopToken: Int?
    private var eosFloor = 0
    private var peakAt = -1
    private var peakProb = 0.0

    public init(config: SamplingConfig, seed: UInt64, block: Int = 256) {
        self.config = config
        self.seed = seed
        self.block = block
        self.isSilence = Set(config.silenceTokenIds)
    }

    /// Enable the stop-token observation the postprocess layer reads.
    ///
    /// Done here, in the sampler, rather than by changing the generator: every
    /// backend already calls the sampler on every step — it owns the RNG
    /// stream, so a backend that skipped it would produce different tokens —
    /// which means the observation reaches every generation path without a new
    /// seam.
    ///
    /// `floor` is the EOS floor this generation runs under. The peak is only
    /// recorded past it, matching the shipped engine: below the floor the
    /// generator masks the stop token, so its probability there describes the
    /// mask rather than the model.
    public func observeEOS(stopToken: Int, floor: Int) {
        self.stopToken = stopToken
        self.eosFloor = floor
        self.peakAt = -1
        self.peakProb = 0.0
    }

    /// Where the model came closest to stopping, as `(step, probability)`.
    ///
    /// `(-1, 0)` when the stop token was never plausible, or when
    /// ``observeEOS(stopToken:floor:)`` was not called. **If the model never
    /// stops, that peak is where the sentence really ended** — which is what
    /// makes the number worth carrying.
    public var eosPeak: (at: Int, probability: Double) { (peakAt, peakProb) }

    /// Record how close this step came to stopping. Never changes the draw.
    ///
    /// The quantity is the shipped engine's, reproduced exactly: the stop
    /// token's softmax weight over the sum of the weights that survived
    /// `min_p`. The numerator is taken **before** the cutoff is applied, so a
    /// step where the stop token was itself filtered out still reports how near
    /// it came — the number answers "how close was this to being the end", not
    /// "what was the chance of stopping", and the first question is the one the
    /// detectors need, because the rows they exist to rescue are precisely the
    /// ones where stopping never won.
    ///
    /// The floor is `>` and not `>=`: at exactly the floor step the generator
    /// has only just unmasked the stop token, and the shipped engine records
    /// from the step after.
    private func observe(_ z: [Double], maxS: Double, threshold: Double, step: Int) {
        guard let stop = stopToken, step > eosFloor, stop < z.count else { return }
        var total = 0.0
        for i in 0..<z.count where z[i] >= threshold || isSilence.contains(i) {
            total += Foundation.exp(z[i] - maxS)
        }
        guard total > 0 else { return }
        let prob = Foundation.exp(z[stop] - maxS) / total
        if prob > peakProb {
            peakProb = prob
            peakAt = step
        }
    }

    private func noiseRow(step: Int, width: Int) -> ArraySlice<Double> {
        if noiseBase < 0 || step < noiseBase || step >= noiseBase + block || noiseWidth != width {
            noiseBase = (step / block) * block
            noiseWidth = width
            noise = Philox.gumbelNoise(
                seed: seed, stream: Self.samplingStream,
                // `UInt32(...)` traps rather than wrapping in Swift, so a run
                // past 2^32 steps would kill the process with no message. Not
                // reachable at 25 tokens a second — that is five and a half
                // years of continuous speech — but a trap is a bad way to find
                // out, and the clamp costs one comparison.
                step0: UInt32(clamping: noiseBase), nSteps: block, width: width)
        }
        let row = step - noiseBase
        return noise[row * width..<(row + 1) * width]
    }

    /// Choose the next token from raw, unnormalised logits.
    ///
    /// - Parameters:
    ///   - logits: `(vocab,)` scores straight from the model head.
    ///   - step: decode step index — it addresses the RNG, so the result does
    ///     not depend on how many tokens were drawn before.
    ///   - seen: which tokens have already been emitted (repetition penalty;
    ///     silence ids are exempt).
    public func sample(logits: [Float], step: Int, seen: [Bool]) -> Int {
        let vocab = logits.count
        var z = [Double](repeating: 0, count: vocab)
        for i in 0..<vocab { z[i] = Double(logits[i]) }

        if config.repetitionPenalty != 1.0 {
            let rp = config.repetitionPenalty
            for i in 0..<vocab where seen[i] && !isSilence.contains(i) {
                z[i] = z[i] > 0 ? z[i] / rp : z[i] * rp
            }
        }

        // division, not multiplication by the reciprocal: z/T and z*(1/T) can
        // differ in the last ulp, and the Python side divides
        let temperature = config.temperature
        var maxS = -Double.infinity
        for i in 0..<vocab {
            z[i] /= temperature
            if z[i] > maxS { maxS = z[i] }
        }

        // min_p in logit space: identical selection to p_i >= min_p * p_max,
        // with no softmax and therefore no order-dependent normalisation.
        let threshold = config.minP > 0 ? maxS + Foundation.log(config.minP) : -Double.infinity

        if stopToken != nil {
            observe(z, maxS: maxS, threshold: threshold, step: step)
        }

        let g = noiseRow(step: step, width: vocab)
        let gBase = g.startIndex
        var bestIdx = 0
        var best = -Double.infinity
        for i in 0..<vocab {
            let keep = z[i] >= threshold || isSilence.contains(i)
            guard keep else { continue }
            let v = z[i] + g[gBase + i]
            if v > best {  // strict: ties break toward the lowest index
                best = v
                bestIdx = i
            }
        }
        return bestIdx
    }
}
