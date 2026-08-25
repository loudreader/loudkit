//! LR-SAMPLER-v1 — a bit-parity port of `loudkit.sampler`. min_p is evaluated
//! in logit space and selection is Gumbel-argmax, so the choice is identical
//! on every backend.

use std::collections::HashSet;

use crate::rng;

/// The sampling law (mirror of `loudkit.config.SamplingConfig`).
#[derive(Clone)]
pub struct Config {
    pub temperature: f64,
    pub repetition_penalty: f64,
    pub min_p: f64,
    pub max_new_tokens: usize,
    pub silence_token_ids: Vec<usize>,
    pub min_tokens_floor: usize,
    pub min_tokens_text_ratio: f64,
}

const SAMPLING_STREAM: u32 = 0;

/// Chooses the next token from raw logits. Caches a block of precomputed
/// Gumbel noise, because generating ten Philox rounds per token costs more
/// than running the entire model.
pub struct Sampler {
    config: Config,
    seed: u64,
    block: usize,
    noise: Vec<f64>,
    base: usize,
    silence: HashSet<usize>,

    /// Observation of how close each step came to stopping. Never feeds back
    /// into the draw; read by the postprocess detectors after generation.
    /// `None` disables it, and with it its cost — one exponential and one sum
    /// over the vocabulary per step.
    stop_token: Option<usize>,
    eos_floor: usize,
    peak_at: i64,
    peak_prob: f64,
}

impl Sampler {
    pub fn new(config: Config, seed: u64) -> Self {
        Self::with_block(config, seed, 256)
    }

    pub fn with_block(config: Config, seed: u64, block: usize) -> Self {
        let silence: HashSet<usize> = config.silence_token_ids.iter().copied().collect();
        Sampler {
            config,
            seed,
            block,
            noise: Vec::new(),
            base: 0,
            silence,
            stop_token: None,
            eos_floor: 0,
            peak_at: -1,
            peak_prob: 0.0,
        }
    }

    /// Enable the stop-token observation the postprocess layer reads.
    ///
    /// Done here, in the sampler, rather than by changing the generator: every
    /// backend already calls the sampler on every step — it owns the RNG
    /// stream, so a backend that skipped it would produce different tokens —
    /// which means the observation reaches every generation path without a new
    /// seam.
    ///
    /// `eos_floor` is the floor this generation runs under. The peak is only
    /// recorded past it, matching the shipped engine: below the floor the
    /// generator masks the stop token, so its probability there describes the
    /// mask rather than the model.
    pub fn observe_eos(&mut self, stop_token: usize, eos_floor: usize) {
        self.stop_token = Some(stop_token);
        self.eos_floor = eos_floor;
        self.peak_at = -1;
        self.peak_prob = 0.0;
    }

    /// Where the model came closest to stopping, as `(step, probability)`.
    ///
    /// `(-1, 0.0)` when the stop token was never plausible, or when
    /// [`Sampler::observe_eos`] was not called. **If the model never stops,
    /// that peak is where the sentence really ended** — which is what makes the
    /// number worth carrying.
    #[must_use]
    pub fn eos_peak(&self) -> (i64, f64) {
        (self.peak_at, self.peak_prob)
    }

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
    fn observe(&mut self, scaled: &[f64], max_s: f64, threshold: f64, step: usize) {
        let Some(stop) = self.stop_token else { return };
        if step <= self.eos_floor || stop >= scaled.len() {
            return;
        }
        let has_minp = self.config.min_p != 0.0;
        let mut total = 0.0;
        for (i, &value) in scaled.iter().enumerate() {
            if !has_minp || value >= threshold || self.silence.contains(&i) {
                total += (value - max_s).exp();
            }
        }
        if total <= 0.0 {
            return;
        }
        let prob = (scaled[stop] - max_s).exp() / total;
        if prob > self.peak_prob {
            self.peak_prob = prob;
            self.peak_at = step as i64;
        }
    }

    fn noise_for(&mut self, step: usize, width: usize) -> &[f64] {
        if self.noise.is_empty()
            || step < self.base
            || step >= self.base + self.block
            || self.noise.len() != self.block * width
        {
            self.base = (step / self.block) * self.block;
            self.noise =
                rng::gumbel_noise(self.seed, SAMPLING_STREAM, self.base, self.block, width);
        }
        let start = (step - self.base) * width;
        &self.noise[start..start + width]
    }

    /// Choose the next token from raw, unnormalised logits.
    pub fn call(&mut self, logits: &[f32], step: usize, seen: &[bool]) -> usize {
        let n = logits.len();
        let mut z = vec![0.0f64; n];
        for (i, l) in logits.iter().enumerate() {
            z[i] = f64::from(*l);
        }

        if self.config.repetition_penalty != 1.0 {
            for i in 0..n {
                if seen[i] && !self.silence.contains(&i) {
                    z[i] = if z[i] > 0.0 {
                        z[i] / self.config.repetition_penalty
                    } else {
                        z[i] * self.config.repetition_penalty
                    };
                }
            }
        }

        let mut scaled = vec![0.0f64; n];
        let mut max_s = f64::NEG_INFINITY;
        for i in 0..n {
            scaled[i] = z[i] / self.config.temperature;
            if scaled[i] > max_s {
                max_s = scaled[i];
            }
        }

        // min_p in logit space: keep i iff s[i] >= max(s) + ln(min_p).
        let threshold = if self.config.min_p > 0.0 {
            max_s + self.config.min_p.ln()
        } else {
            f64::NEG_INFINITY
        };

        if self.stop_token.is_some() {
            self.observe(&scaled, max_s, threshold, step);
        }

        let g = self.noise_for(step, n).to_vec();
        let min_p = self.config.min_p;
        let has_minp = min_p != 0.0;
        let silence = &self.silence;
        let mut best = f64::NEG_INFINITY;
        let mut best_idx = -1isize;
        for i in 0..n {
            let keep = !has_minp || scaled[i] >= threshold || silence.contains(&i);
            if !keep {
                continue;
            }
            let v = scaled[i] + g[i];
            if v > best {
                best = v;
                best_idx = i as isize;
            }
        }
        if best_idx == -1 {
            return 0; // all kept values -inf; argmax falls back to index 0
        }
        best_idx as usize
    }
}
