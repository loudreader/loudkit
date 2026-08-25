import Foundation

/// Philox-4x32-10, the counter-based RNG every loudkit implementation shares.
///
/// The stream has independent implementations — numpy in
/// `loudkit.rng`, the CUDA-side variant in the research repo, and this one —
/// which is the entire point of choosing Philox: it is integer-only, so a
/// correct port produces identical bits by construction, and it is checkable
/// against the published Random123 known-answer vectors rather than against
/// another implementation's output. `swift test` runs those
/// vectors from the shared conformance fixture.
///
/// The n-th random number is a pure function of `(seed, stream, step, index)`
/// — not of how many numbers were drawn before it — so this side may generate
/// per token while the Python side generates a block ahead, and the streams
/// still agree.
public enum Philox {
    static let m0: UInt64 = 0xD251_1F53
    static let m1: UInt64 = 0xCD9E_8D57
    static let w0: UInt32 = 0x9E37_79B9  // golden ratio
    static let w1: UInt32 = 0xBB67_AE85  // sqrt(3) - 1

    /// Ten rounds over one 4x32 counter block.
    public static func philox4x32_10(
        counter: (UInt32, UInt32, UInt32, UInt32), key: (UInt32, UInt32)
    ) -> (UInt32, UInt32, UInt32, UInt32) {
        var (c0, c1, c2, c3) = counter
        var (k0, k1) = key
        for _ in 0..<10 {
            let p0 = UInt64(c0) * m0
            let p1 = UInt64(c2) * m1
            let hi0 = UInt32(truncatingIfNeeded: p0 >> 32)
            let lo0 = UInt32(truncatingIfNeeded: p0)
            let hi1 = UInt32(truncatingIfNeeded: p1 >> 32)
            let lo1 = UInt32(truncatingIfNeeded: p1)
            (c0, c1, c2, c3) = (hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0)
            k0 = k0 &+ w0
            k1 = k1 &+ w1
        }
        return (c0, c1, c2, c3)
    }

    /// `nSteps x width` uniforms in the open interval (0, 1), identical bits
    /// to `loudkit.rng.uniforms`. Row-major.
    ///
    /// Counter layout (must never drift from the Python side): c0 = quad
    /// index, c1 = step, c2 = stream, c3 = 0; key = (seed lo, seed hi). Each
    /// quad yields its four outputs consecutively. The `+ 0.5` before scaling
    /// keeps every value clear of both interval ends with no branch.
    public static func uniforms(
        seed: UInt64, stream: UInt32, step0: UInt32, nSteps: Int, width: Int
    ) -> [Double] {
        let quads = (width + 3) / 4
        let k0 = UInt32(truncatingIfNeeded: seed)
        let k1 = UInt32(truncatingIfNeeded: seed >> 32)
        var out = [Double](repeating: 0, count: nSteps * width)
        for s in 0..<nSteps {
            let step = step0 &+ UInt32(s)
            for q in 0..<quads {
                let r = philox4x32_10(
                    counter: (UInt32(q), step, stream, 0), key: (k0, k1))
                let bits = [r.0, r.1, r.2, r.3]
                for j in 0..<4 {
                    let idx = q * 4 + j
                    if idx < width {
                        out[s * width + idx] = (Double(bits[j]) + 0.5) / 4_294_967_296.0
                    }
                }
            }
        }
        return out
    }

    /// `-log(-log(u))` for one block of steps — the additive form of a
    /// categorical draw. Row-major `nSteps x width`.
    public static func gumbelNoise(
        seed: UInt64, stream: UInt32, step0: UInt32, nSteps: Int, width: Int
    ) -> [Double] {
        var u = uniforms(seed: seed, stream: stream, step0: step0, nSteps: nSteps, width: width)
        for i in 0..<u.count {
            u[i] = -Foundation.log(-Foundation.log(u[i]))
        }
        return u
    }
}
