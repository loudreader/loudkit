//! The renderer's geometry, a bit-parity port of `loudkit.models.windowing`:
//! the window framing recipe, the Euler grid, the EOS floor and the Philox
//! stream ids.

/// Philox sub-stream, under the stage seed, for the CFM prior. Streams 0 and 1
/// are consumed by the Box-Muller pair, so any future draw starts at 2.
pub const FLOW_NOISE_STREAM: u32 = 0;
/// Philox sub-stream for the eight harmonic phase offsets. Row 0 is pinned to
/// zero: the voiced fundamental must start at a zero crossing.
pub const VOCODER_PHASE_STREAM: u32 = 0;
/// Philox sub-streams 1 and 2, the Box-Muller pair, for the excitation noise.
pub const VOCODER_NOISE_STREAM: u32 = 1;

/// Text framing token that opens the transcript segment. A property of the T3
/// text tokenizer family, held here so this backend frames the same way.
pub const START_TEXT_TOKEN: usize = 255;
/// Text framing token that closes the transcript segment. See
/// [`START_TEXT_TOKEN`].
pub const STOP_TEXT_TOKEN: usize = 0;

const TOKEN_MEL_RATIO: usize = 2;
const MEL_BINS: usize = 80;

/// Output of the window recipe.
#[derive(Debug)]
pub struct Framed {
    /// The `(P+Q,)` token row: prompt tokens then speech tokens.
    pub row: Vec<i64>,
    /// The mel condition, `80 * 2*(P+Q)` values in `[bin][frame]` order.
    pub cond: Vec<f32>,
    /// How many frames of `cond` belong to the reference prompt.
    pub prompt_frames: usize,
    /// The row's length, `P+Q`.
    pub n: usize,
}

/// The Euler time grid: the manifest's explicit one if it has one, else cosine.
///
/// An explicit grid exists because "cosine" is a formula two codebases can
/// write two ways, and a checkpoint that ships one must integrate on that
/// schedule in every port. A port that ignored the parameter would integrate
/// on its own schedule under a fingerprint recording the grid it ignored. The
/// shipping manifest has `euler_grid: null`, so only a checkpoint that sets
/// the key exercises this arm.
#[must_use]
pub fn time_grid(euler_steps: usize, euler_grid: Option<&[f64]>) -> Vec<f64> {
    if let Some(grid) = euler_grid {
        if !grid.is_empty() {
            return grid.to_vec();
        }
    }
    (0..=euler_steps)
        .map(|i| 1.0 - ((i as f64) / (euler_steps as f64) * std::f64::consts::PI / 2.0).cos())
        .collect()
}

/// Minimum speech tokens before the stop token becomes sampleable.
pub fn eos_floor(n_text_tokens: usize, min_floor: usize, ratio: f64) -> usize {
    let r = (n_text_tokens as f64 * ratio) as usize;
    min_floor.max(r)
}

/// The token that fills unused static-window slots.
///
/// # Errors
///
/// Returns an error when a static-length window is configured but neither
/// `pad` nor `silence` gives it a token to pad with.
fn pad_token_id(pad: Option<usize>, silence: &[usize]) -> Result<usize, String> {
    if let Some(p) = pad {
        return Ok(p);
    }
    if let Some(&s) = silence.first() {
        return Ok(s);
    }
    Err(
        "static window needs a pad token: set WindowConfig.pad_token_id or provide \
         silence token ids: padding with token 0 bleeds +3 dB of high-band energy \
         into the tail through the encoder's attention"
            .to_string(),
    )
}

/// Window recipe values, resolved from the manifest.
///
/// Algorithm layer, as in `config.WindowConfig`: the pad-and-truncate recipe
/// was the whole measured deviation between two backends' renders.
pub struct WindowConfig {
    /// Longest token sequence one window carries, about 10.2 s at 25 Hz.
    pub max_speech_tokens: usize,
    /// Pad every window to this many tokens, or `None` for ragged.
    pub static_length: Option<usize>,
    /// Padding token. `None` means the silence token; token 0 bleeds into the
    /// tail.
    pub pad_token_id: Option<usize>,
    /// Fixed length of the reference-prompt window, or `None` for ragged.
    pub static_prompt_tokens: Option<usize>,
}

/// The ragged window: 255 tokens, no static padding on either side.
///
/// The same values as `config.WindowConfig`'s field defaults, so a manifest
/// that declares no window frames the same way in both. The shipped
/// checkpoint declares all four keys, so this is the recipe for a manifest
/// that leaves them unstated, not the shipped one.
impl Default for WindowConfig {
    fn default() -> Self {
        WindowConfig {
            max_speech_tokens: 255,
            static_length: None,
            pad_token_id: None,
            static_prompt_tokens: None,
        }
    }
}

impl WindowConfig {
    /// `WindowConfig.__post_init__`, rule for rule and message for message.
    ///
    /// Both refusals were made at frame time, once a render was already under
    /// way, or not at all. A static window shorter than the token budget is
    /// the one that matters: `frame_windows` refuses it, but only on the
    /// utterance that overruns, so a manifest declaring it loaded, reported a
    /// fingerprint, and failed mid-stream after earlier chunks had played. The
    /// reference refuses it at the door, before anything is rendered.
    ///
    /// # Errors
    ///
    /// One `ValueError` of the reference's, as a `String`.
    pub fn validate(&self) -> Result<(), String> {
        if let Some(length) = self.static_length {
            if length < self.max_speech_tokens {
                return Err(format!(
                    "static_length {length} cannot be shorter than max_speech_tokens {}",
                    self.max_speech_tokens
                ));
            }
        }
        // `usize`, so "positive" is the whole of the reference's `<= 0`: the
        // reader refuses a negative one at the shape door.
        if self.static_prompt_tokens == Some(0) {
            return Err("static_prompt_tokens must be positive: 0".to_string());
        }
        Ok(())
    }
}

/// The shipped static-window recipe.
pub fn production_window() -> WindowConfig {
    WindowConfig {
        max_speech_tokens: 255,
        static_length: Some(255),
        pad_token_id: Some(4254),
        static_prompt_tokens: Some(238),
    }
}

/// Apply the window recipe.
///
/// # Errors
///
/// Returns an error when a static-length window is configured but neither
/// `pad` nor `silence` gives it a token to pad with, see `pad_token_id`,
/// and when more tokens are handed in than the window holds.
///
/// An over-window input is refused with the amount of speech that would have
/// been lost; silent truncation (`take(max_speech_tokens)`) hides missing text
/// behind correct-sounding audio: the end of a passage does not exist while
/// the audio sounds perfectly fine, and the only listener who notices is one
/// who knows the text. The Python engine refuses it loudly and says how much
/// speech would be lost; this does the same.
pub fn frame_windows(
    cfg: &WindowConfig,
    pad: Option<usize>,
    silence: &[usize],
    tokens: &[usize],
    prompt_tokens: &[i64],
    prompt_mel: &[f32],
) -> Result<Framed, String> {
    if tokens.len() > cfg.max_speech_tokens {
        return Err(format!(
            "{} speech tokens exceed the {}-token window by {}; split the text first",
            tokens.len(),
            cfg.max_speech_tokens,
            tokens.len() - cfg.max_speech_tokens
        ));
    }
    let toks: Vec<usize> = tokens.to_vec();
    let n = toks.len();
    let prompt_toks: Vec<usize> = prompt_tokens.iter().map(|t| *t as usize).collect();
    let prompt_mel_frames = prompt_mel.len() / MEL_BINS;

    let (prompt, query, cond_width, prompt_frames);
    if let Some(sl) = cfg.static_length {
        // The guard above checks `max_speech_tokens`; the buffer below is
        // `static_length` long. A manifest declaring `max_speech_tokens: 300`
        // with `static_length: 255` therefore index-panicked in `q[i] = *t`,
        // killing the process where Python raises a ValueError from numpy.
        // Unreachable with the shipping manifest (both 255) and reachable
        // through the one input this library treats as authoritative.
        if cfg.max_speech_tokens > sl {
            return Err(format!(
                "window.static_length {sl} is shorter than max_speech_tokens {}: the \
                 static query buffer cannot hold a full window",
                cfg.max_speech_tokens
            ));
        }
        let p = pad_token_id(pad, silence)?;
        let p_len = cfg.static_prompt_tokens.unwrap_or(prompt_toks.len());
        let mut pr = vec![p; p_len];
        for (i, t) in prompt_toks.iter().take(p_len).enumerate() {
            pr[i] = *t;
        }
        let mut q = vec![p; sl];
        for (i, t) in toks.iter().enumerate() {
            q[i] = *t;
        }
        prompt = pr;
        query = q;
        cond_width = TOKEN_MEL_RATIO * (p_len + sl);
        prompt_frames = TOKEN_MEL_RATIO * p_len;
    } else {
        prompt = prompt_toks.clone();
        query = toks;
        cond_width = TOKEN_MEL_RATIO * (prompt_toks.len() + n);
        prompt_frames = TOKEN_MEL_RATIO * prompt_toks.len();
    }

    let mut row = Vec::with_capacity(prompt.len() + query.len());
    for t in &prompt {
        row.push(*t as i64);
    }
    for t in &query {
        row.push(*t as i64);
    }

    let mut cond = vec![0.0f32; cond_width * MEL_BINS];
    let keep_f = prompt_mel_frames.min(prompt_frames);
    for b in 0..MEL_BINS {
        for f in 0..keep_f {
            cond[b * cond_width + f] = prompt_mel[b * prompt_mel_frames + f];
        }
    }

    Ok(Framed {
        row,
        cond,
        prompt_frames,
        n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Over-window is refused, not trimmed.
    ///
    /// `take(max_speech_tokens)` leaves the end of a long passage nonexistent
    /// while the audio still sounds fine: the only listener who notices is
    /// one who already knows the text. Python refuses it loudly; this port
    /// does too.
    #[test]
    fn frame_windows_refuses_more_tokens_than_the_window_holds() {
        let cfg = WindowConfig {
            max_speech_tokens: 4,
            static_length: None,
            pad_token_id: None,
            static_prompt_tokens: None,
        };
        let prompt_tokens = vec![1i64, 2, 3];
        let prompt_mel = vec![0.0f32; MEL_BINS * 6];
        let err = frame_windows(
            &cfg,
            None,
            &[],
            &[1, 2, 3, 4, 5],
            &prompt_tokens,
            &prompt_mel,
        )
        .expect_err("5 tokens in a 4-token window must be refused");
        assert!(err.contains("exceed"), "unhelpful message: {err}");
    }

    // Pins the error path: a static-length window configured without a pad
    // token or silence token ids to pad with is a catchable error, not a
    // panic on a malformed checkpoint manifest: matching Python
    // (ValueError), JS and Swift.
    #[test]
    fn frame_windows_errors_without_pad_token() {
        // max_speech_tokens matches static_length: a window whose budget
        // exceeds its own static buffer is a config Python refuses at
        // construction (WindowConfig.__post_init__), and this port refuses
        // it too rather than index-panicking on it.
        let cfg = WindowConfig {
            max_speech_tokens: 4,
            static_length: Some(4),
            pad_token_id: None,
            static_prompt_tokens: Some(2),
        };
        let prompt_tokens = [1i64, 2];
        let prompt_mel = [0.0f32; MEL_BINS * 4];

        let err = frame_windows(&cfg, None, &[], &[10, 20], &prompt_tokens, &prompt_mel)
            .expect_err("expected an error");
        assert!(err.contains("pad token"), "unexpected message: {err}");
    }

    #[test]
    fn frame_windows_succeeds_with_silence_fallback() {
        // Same window as the test above, for the reason stated there.
        let cfg = WindowConfig {
            max_speech_tokens: 4,
            static_length: Some(4),
            pad_token_id: None,
            static_prompt_tokens: Some(2),
        };
        let prompt_tokens = [1i64, 2];
        let prompt_mel = [0.0f32; MEL_BINS * 4];

        let framed = frame_windows(&cfg, None, &[7], &[10, 20], &prompt_tokens, &prompt_mel)
            .expect("silence_token_ids should provide a pad token");
        assert_eq!(framed.row.len(), 6); // 2 static prompt + 4 static query
    }
}
