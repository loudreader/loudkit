//! Render randomness as Philox data — a bit-parity port of
//! `loudkit.models.noise`.

use crate::rng;

/// rows*cols standard-normal field.
pub fn gaussian_field(seed: u64, stream: u32, rows: usize, cols: usize) -> Vec<f32> {
    let u1 = rng::uniforms(seed, stream, 0, rows, cols);
    let u2 = rng::uniforms(seed, stream.wrapping_add(1), 0, rows, cols);
    u1.iter()
        .zip(u2.iter())
        .map(|(a, b)| ((-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()) as f32)
        .collect()
}

/// n uniforms in (-half_width, half_width).
pub fn symmetric_uniforms(seed: u64, stream: u32, n: usize, half_width: f64) -> Vec<f32> {
    let u = rng::uniforms(seed, stream, 0, 1, n);
    u.iter()
        .map(|v| ((v * 2.0 - 1.0) * half_width) as f32)
        .collect()
}
