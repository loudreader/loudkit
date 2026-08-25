//! Philox-4x32-10 — a bit-parity port of `loudkit.rng`. Native u32/u64
//! arithmetic, so the bits match the Python, JS and Go implementations by
//! construction. The n-th random number is a pure function of
//! `(seed, stream, step, index)`.

const M0: u32 = 0xd2511f53;
const M1: u32 = 0xcd9e8d57;
const W0: u32 = 0x9e3779b9;
const W1: u32 = 0xbb67ae85;
const ROUNDS: usize = 10;

/// Ten rounds of Philox-4x32 over one counter quad. Returns the four u32
/// streams.
pub fn philox4x32(c0: u32, c1: u32, c2: u32, c3: u32, k0: u32, k1: u32) -> [u32; 4] {
    let (mut x0, mut x1, mut x2, mut x3) = (c0, c1, c2, c3);
    let (mut key0, mut key1) = (k0, k1);
    for _ in 0..ROUNDS {
        let (hi0, lo0) = mulhilo(x0, M0);
        let (hi1, lo1) = mulhilo(x2, M1);
        x0 = hi1 ^ x1 ^ key0;
        x1 = lo1;
        x2 = hi0 ^ x3 ^ key1;
        x3 = lo0;
        key0 = key0.wrapping_add(W0);
        key1 = key1.wrapping_add(W1);
    }
    [x0, x1, x2, x3]
}

/// 64-bit product of two u32s as (hi, lo).
#[inline]
fn mulhilo(a: u32, b: u32) -> (u32, u32) {
    let p = u64::from(a) * u64::from(b);
    ((p >> 32) as u32, p as u32)
}

/// `n_steps * width` uniforms in the open interval (0, 1).
pub fn uniforms(seed: u64, stream: u32, step0: usize, n_steps: usize, width: usize) -> Vec<f64> {
    let mut out = vec![0.0; n_steps * width];
    let quads = width.div_ceil(4);
    for s in 0..n_steps {
        let step = (s + step0) as u32;
        for q in 0..quads {
            let r = philox4x32(q as u32, step, stream, 0, seed as u32, (seed >> 32) as u32);
            for (i, v) in r.iter().enumerate() {
                let idx = s * width + q * 4 + i;
                if idx < out.len() {
                    out[idx] = (f64::from(*v) + 0.5) / 4294967296.0;
                }
            }
        }
    }
    out
}

/// `-log(-log(u))`, the additive form of a categorical draw.
pub fn gumbel_noise(
    seed: u64,
    stream: u32,
    step0: usize,
    n_steps: usize,
    width: usize,
) -> Vec<f64> {
    uniforms(seed, stream, step0, n_steps, width)
        .iter()
        .map(|u| -(-u.ln()).ln())
        .collect()
}
