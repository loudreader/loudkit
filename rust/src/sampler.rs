//! LR-SAMPLER-v1: a bit-parity port of `loudkit.sampler`. min_p is evaluated
//! in logit space and selection is Gumbel-argmax, so the choice is identical
//! on every backend.

use std::collections::HashSet;

use crate::fingerprint::repr_float;
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

impl Config {
    /// `SamplingConfig.__post_init__`, rule for rule and message for message.
    ///
    /// The law had no range checks at all on this side, so a manifest could
    /// hand [`Sampler::call`] a temperature of zero and it would divide the
    /// logit row by it; a `repetition_penalty` below 1.0, which rewards the
    /// repetition it exists to punish; a `min_p` of 1.0 or more, which masks
    /// every candidate including the argmax; and a negative
    /// `min_tokens_text_ratio`, which pushes the end-of-speech floor below
    /// zero. The reference refuses all four at the manifest door, before any
    /// of it reaches audio.
    ///
    /// `min_tokens_floor` carries no check here because it is a `usize`: the
    /// reader refuses a negative one at the shape door with its own message.
    ///
    /// # Errors
    ///
    /// One `ValueError` of the reference's, as a `String`.
    pub fn validate(&self) -> Result<(), String> {
        if !(self.temperature > 0.0 && self.temperature <= 4.0) {
            return Err(format!(
                "temperature out of range: {}",
                repr_float(self.temperature)
            ));
        }
        if self.repetition_penalty < 1.0 {
            return Err(format!(
                "repetition_penalty below 1.0 rewards repetition: {}",
                repr_float(self.repetition_penalty)
            ));
        }
        if !(self.min_p >= 0.0 && self.min_p < 1.0) {
            return Err(format!("min_p out of range: {}", repr_float(self.min_p)));
        }
        // The other three ports took a cap of zero and decoded nothing, which
        // reaches a caller as silence they have to diagnose rather than an
        // error they can read. A cap of zero is not a configuration, it is a
        // typo in a manifest.
        if self.max_new_tokens == 0 {
            return Err("max_new_tokens must be positive: 0".to_string());
        }
        if self.min_tokens_text_ratio < 0.0 {
            return Err(format!(
                "min_tokens_text_ratio must be >= 0: {}",
                repr_float(self.min_tokens_text_ratio)
            ));
        }
        Ok(())
    }
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

    /// The two vocabulary-wide scratch rows [`Sampler::call`] works in, held
    /// across calls rather than allocated inside one. `call` runs once per
    /// token, so a pair of fresh `n`-wide `Vec<f64>` there was two
    /// allocations per token for the length of a passage. Nothing is carried
    /// between calls: both are refilled from the logits at the top.
    z: Vec<f64>,
    scaled: Vec<f64>,

    /// Observation of how close each step came to stopping. Never feeds back
    /// into the draw; read by the postprocess detectors after generation.
    /// `None` disables it, and with it its cost: one exponential and one sum
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
            z: Vec::new(),
            scaled: Vec::new(),
            stop_token: None,
            eos_floor: 0,
            peak_at: -1,
            peak_prob: 0.0,
        }
    }

    /// Enable the stop-token observation the postprocess layer reads.
    ///
    /// Done here, in the sampler, rather than by changing the generator: every
    /// backend already calls the sampler on every step: it owns the RNG
    /// stream, so a backend that skipped it would produce different tokens,
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
    /// that peak is where the sentence really ended**, which is what makes the
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
    /// it came: the number answers "how close was this to being the end", not
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

    /// Make sure the drawn block covers `step`, and say where its row starts.
    ///
    /// The offset rather than the row, because the caller reads the row beside
    /// `self.silence` and a borrow of one is a borrow of the other. Returning
    /// the slice from a `&mut self` method forced the caller to copy it, a
    /// vocabulary-wide row of `f64` per token.
    fn noise_row(&mut self, step: usize, width: usize) -> usize {
        if self.noise.is_empty()
            || step < self.base
            || step >= self.base + self.block
            || self.noise.len() != self.block * width
        {
            self.base = (step / self.block) * self.block;
            self.noise =
                rng::gumbel_noise(self.seed, SAMPLING_STREAM, self.base, self.block, width);
        }
        (step - self.base) * width
    }

    /// Choose the next token from raw, unnormalised logits.
    pub fn call(&mut self, logits: &[f32], step: usize, seen: &[bool]) -> usize {
        let n = logits.len();
        // Moved out of `self` and put back at the end, so that `observe`
        // below can take `&mut self` while the row it reads is a local.
        let mut z = std::mem::take(&mut self.z);
        let mut scaled = std::mem::take(&mut self.scaled);
        z.clear();
        z.extend(logits.iter().map(|l| f64::from(*l)));

        if self.config.repetition_penalty != 1.0 {
            // The penalty applies to every seen token, silence included.
            // Silence was exempt here until the interior-stall study: immune
            // to the penalty and re-admitted below the min_p floor (the
            // exemption below), a silence run had no exit: zero escapes in
            // 1,031 instrumented trap steps, and 33.0% of long-form
            // paragraphs carried a >1 s hole, with 74 of 1705 passages
            // rendering chunks of no speech at all. Penalising seen silence
            // closes the trap: holes 33.0% -> 4.3% and mute chunks 74 -> 1 at
            // 120 passages/arm across ten languages, natural-band pause rates
            // inside noise on 8/9 healthy voices, WER flat or better.
            for i in 0..n {
                if seen[i] {
                    z[i] = if z[i] > 0.0 {
                        z[i] / self.config.repetition_penalty
                    } else {
                        z[i] * self.config.repetition_penalty
                    };
                }
            }
        }

        scaled.clear();
        let mut max_s = f64::NEG_INFINITY;
        for logit in &z {
            let s = logit / self.config.temperature;
            scaled.push(s);
            if s > max_s {
                max_s = s;
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

        let row = self.noise_row(step, n);
        let g = &self.noise[row..row + n];
        let min_p = self.config.min_p;
        let has_minp = min_p != 0.0;
        let silence = &self.silence;
        let mut best = f64::NEG_INFINITY;
        let mut best_idx: Option<usize> = None;
        for i in 0..n {
            // Silence stays available even when min_p would drop it: a pause
            // token is what makes a reader pause, and a filter that removes
            // the only way to pause is a filter that removes prosody. This is
            // the one exemption silence keeps: removing it was measured
            // catastrophic (median long-form gap 2.46 s -> 4.64 s). The
            // repetition penalty above now applies to silence like everything
            // else, so a pause that overstays decays instead of never ending.
            let keep = !has_minp || scaled[i] >= threshold || silence.contains(&i);
            if !keep {
                continue;
            }
            let v = scaled[i] + g[i];
            if v > best {
                best = v;
                best_idx = Some(i);
            }
        }
        self.z = z;
        self.scaled = scaled;
        // `None` is every kept value at -inf; argmax falls back to index 0.
        best_idx.unwrap_or(0)
    }
}
