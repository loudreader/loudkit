import Foundation

/// Render randomness as data, mirroring `loudkit.models.noise`.
///
/// The flow prior and the vocoder excitation are inputs that happen to be
/// random. Both sides of the conformance table draw them from the same Philox
/// counters, so a cross-language waveform comparison measures arithmetic, not
/// RNG plumbing. The Gaussian transform draws a fresh uniform pair per sample
/// (cos-only Box–Muller) — the cached-spare variant puts a period-2 structure
/// exactly on Nyquist, measured at +5.3 dB and audible after the vocoder.
enum Noise {
    /// `rows x cols` standard normals, row-major float32. Consumes Philox
    /// sub-streams `stream` and `stream + 1` (the Box–Muller pair), so
    /// callers space their stream ids by two — same rule as Python.
    static func gaussianField(seed: UInt64, stream: UInt32, rows: Int, cols: Int) -> [Float] {
        let u1 = Philox.uniforms(seed: seed, stream: stream, step0: 0, nSteps: rows, width: cols)
        let u2 = Philox.uniforms(seed: seed, stream: stream + 1, step0: 0, nSteps: rows, width: cols)
        var out = [Float](repeating: 0, count: rows * cols)
        for i in 0..<out.count {
            let z = (-2.0 * Foundation.log(u1[i])).squareRoot()
                * Foundation.cos(2.0 * Double.pi * u2[i])
            out[i] = Float(z)
        }
        return out
    }

    /// `n` uniforms in `(-halfWidth, halfWidth)`, counter-addressed, float32.
    static func symmetricUniforms(seed: UInt64, stream: UInt32, n: Int, halfWidth: Double) -> [Float] {
        let u = Philox.uniforms(seed: seed, stream: stream, step0: 0, nSteps: 1, width: n)
        return u.map { Float(($0 * 2.0 - 1.0) * halfWidth) }
    }
}
