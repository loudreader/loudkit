//! Enrollment: reference audio to a [`voice::Profile`]. A bit-parity port of
//! `loudkit.models.enroll` over the exported enrollment ONNX graphs.
//!
//! The DSP (resampler, filterbanks, trim) is implemented here and held to the
//! enrollment fixture; the model stages run through `s3_tokenizer.onnx`,
//! `camp.onnx` and `voice_encoder.onnx`. The filter tables and windows are
//! embedded as the same float32 data every port loads.
//!
//! `needless_range_loop` is allowed for this module, and only this one. In
//! signal processing the index *is* a physical quantity, a filter phase, a
//! frame number, a frequency bin, and several loops here read a second buffer
//! at an offset (`padded[start + i]`), which iterator form expresses worse than
//! it expresses the arithmetic. This file's job is to be checkable line by line
//! against `loudkit.models.enroll` and torchaudio's kernel; keeping the indices
//! visible is what makes that possible.
#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, LazyLock, Mutex};

use ndarray::Array3;
use ort::session::Session;
use ort::value::{DynValue, Value};

use crate::execution::{ort_err, Execution, ExecutionConfig};
use crate::voice;

const MEL_SR: usize = 24_000;
const S3_SR: usize = 16_000;
const MAX_REF_SECONDS: usize = 10;
const COND_SECONDS: usize = 6;

const FRAME_400: usize = 400;
const HOP_160: usize = 160;
const KALDI_FFT: usize = 512;
const MATCHA_NFFT: usize = 1920;
const MATCHA_HOP: usize = 480;

const PARTIAL_FRAMES: usize = 160;
const PARTIAL_STEP: usize = 77;

// ---------------------------------------------------- reference recording

/// Shortest clip a speaker can be estimated from. The utterance encoder's
/// first partial alone covers 1.6 s and is zero-padded under it, so a
/// sub-second clip enrolls mostly padding.
const MIN_ENROLL_SECONDS: f64 = 1.0;

/// Longest clip the input contract stays honest over. The prompt uses the
/// first 10 s and the speaker embedding reads the whole clip, so a five-minute
/// recording produces a voice mostly shaped by audio the docs say is ignored.
/// Refused rather than truncated: the user picked that recording for a reason,
/// and silently using a different slice of it is worse than asking them to
/// choose.
const MAX_ENROLL_SECONDS: f64 = 30.0;

/// A clip whose loudest sample is under this is silence at any playback level;
/// there is no voice in it to enroll.
const SILENCE_PEAK: f64 = 1e-4;

const GOOD_INPUT: &str = "A good input is 5 to 10 seconds of one person speaking, clean, \
                          without music or a second voice.";

/// `%.1e` as the reference prints it: one decimal and a signed exponent of at
/// least two digits. Rust's own `{:.1e}` writes `5.0e-5` where the reference
/// writes `5.0e-05`, and the peak is inside a sentence the ports are matched
/// on word for word.
fn exponent_1(value: f64) -> String {
    let text = format!("{value:.1e}");
    let (mantissa, exponent) = text.split_once('e').unwrap_or((text.as_str(), "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let sign = if exponent < 0 { '-' } else { '+' };
    format!("{mantissa}e{sign}{:02}", exponent.abs())
}

/// Refuse a recording the enrollment contract cannot honour.
///
/// The whole preflight, before any DSP runs, with the same five sentences as
/// `loudkit.models.enrollment_audio.validate_reference_audio`. Without it a
/// clip shorter than 721 samples indexes past the end of the reflect padding
/// in [`matcha_mel`] and panics, a sub-second clip enrolls padding, and NaN
/// samples poison every statistic downstream.
///
/// See `docs/design/models-notes.md`.
///
/// # Errors
///
/// The sentence naming whichever of the five conditions the clip meets.
pub fn validate_reference_audio(audio: &[f32], sample_rate: usize) -> Result<(), String> {
    // A non-positive rate reaches the resampler as a division by zero. Callers
    // and tests in every port match on this one message, so it stays word for
    // word.
    if sample_rate == 0 {
        return Err("sample rate must be positive, got 0".to_string());
    }
    // Finiteness before anything arithmetic: one NaN poisons every statistic
    // below and every tensor downstream.
    if audio.iter().any(|v| !v.is_finite()) {
        return Err(format!(
            "the recording contains NaN or Inf samples, so no voice can be derived \
             from it. Re-export the file. {GOOD_INPUT}"
        ));
    }
    let seconds = audio.len() as f64 / sample_rate as f64;
    if seconds < MIN_ENROLL_SECONDS {
        return Err(format!(
            "the recording is {seconds:.2} s: too short to enroll a speaker from \
             (minimum {MIN_ENROLL_SECONDS} s). {GOOD_INPUT}"
        ));
    }
    if seconds > MAX_ENROLL_SECONDS {
        return Err(format!(
            "the recording is {seconds:.1} s. Only the first 10 s become the voice \
             prompt, and the whole clip shapes the speaker embedding, so a long \
             recording enrolls something the prompt does not carry. Trim it to the \
             best 5 to 10 seconds (at most {MAX_ENROLL_SECONDS} s). {GOOD_INPUT}"
        ));
    }
    // The peak is taken in f64 off f32 samples, as the reference takes it, so
    // the floor is the same number on both sides of the comparison.
    let peak = audio
        .iter()
        .fold(0.0f64, |acc, &v| acc.max(f64::from(v).abs()));
    if peak < SILENCE_PEAK {
        return Err(format!(
            "the recording is silent (peak {}); there is no voice in it to enroll. \
             {GOOD_INPUT}",
            exponent_1(peak)
        ));
    }
    Ok(())
}

// ------------------------------------------------------------------ tables

static S3_MEL: &[u8] = include_bytes!("enroll_data/s3_mel_filters.f32");
static S3_HANN: &[u8] = include_bytes!("enroll_data/s3_hann400.f32");
static MATCHA_MEL: &[u8] = include_bytes!("enroll_data/matcha_mel_filters.f32");
static MATCHA_HANN: &[u8] = include_bytes!("enroll_data/matcha_hann1920.f32");
static VE_MEL: &[u8] = include_bytes!("enroll_data/voiceenc_mel_filters.f32");
static VE_HANN: &[u8] = include_bytes!("enroll_data/voiceenc_hann400.f32");
static KALDI_MEL: &[u8] = include_bytes!("enroll_data/kaldi_mel_filters.f32");
static KALDI_POVEY: &[u8] = include_bytes!("enroll_data/kaldi_povey400.f32");

fn f32_table(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

// ---------------------------------------------------------------- resampler

/// The one portable Hann-windowed-sinc resampler, a bit-parity port of
/// `loudkit.models.resample`: float64 kernel rounded to float32 once, FIR
/// accumulated left to right in float32 with no fused multiply-add.
pub fn resample(waveform: &[f32], orig_freq: usize, new_freq: usize) -> Vec<f32> {
    if orig_freq == new_freq {
        return waveform.to_vec();
    }
    let g = gcd(orig_freq, new_freq);
    let (orig, new) = (orig_freq / g, new_freq / g);

    let (kernel, width) = sinc_hann_kernel(orig, new);
    let taps = kernel[0].len();

    let mut padded = vec![0.0f32; width + waveform.len() + width + orig];
    padded[width..width + waveform.len()].copy_from_slice(waveform);

    let n_out = (padded.len() - taps) / orig + 1;
    let mut out = vec![0.0f32; n_out * new];
    for i in 0..n_out {
        let base = i * orig;
        for phase in 0..new {
            let mut acc = 0.0f32;
            for c in 0..taps {
                acc += kernel[phase][c] * padded[base + c];
            }
            out[i * new + phase] = acc;
        }
    }

    let target = (new * waveform.len()).div_ceil(orig);
    out.truncate(target);
    out
}

fn sinc_hann_kernel(orig: usize, new: usize) -> (Vec<Vec<f32>>, usize) {
    let base = (orig.min(new) as f64) * 0.99;
    let width = (6.0 * orig as f64 / base).ceil() as usize;

    let mut kernel = vec![vec![0.0f32; 2 * width + orig]; new];
    for phase in 0..new {
        for idx in 0..(2 * width + orig) {
            let mut t = -(phase as f64) / new as f64 + (idx as f64 - width as f64) / orig as f64;
            t *= base;
            t = t.clamp(-6.0, 6.0);
            let window = (t * std::f64::consts::PI / 6.0 / 2.0).cos().powi(2);
            let tt = t * std::f64::consts::PI;
            let sinc = if tt == 0.0 { 1.0 } else { tt.sin() / tt };
            kernel[phase][idx] = (sinc * window * (base / orig as f64)) as f32;
        }
    }
    (kernel, width)
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

// ------------------------------------------------------------------ filterbanks

/// The cosine and sine rows of a DFT of one size, `[bin][sample]`.
type Basis = (Vec<Vec<f64>>, Vec<Vec<f64>>);

/// The bases already built, one per `nfft`.
///
/// A basis is a pure function of `nfft`, so building it again always answers
/// the same values; three mel functions asked for one per call, and the
/// `MATCHA_NFFT` basis alone is about 29 MB of trig. The trade is that each
/// size stays resident for the life of the process rather than for the length
/// of one mel, which is worth stating on a small device: three sizes are
/// reachable here, 400 from both `centred_power_spectra` callers, `KALDI_FFT`
/// and `MATCHA_NFFT`.
static BASES: LazyLock<Mutex<HashMap<usize, Arc<Basis>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn build_basis(nfft: usize) -> Basis {
    let bins = nfft / 2 + 1;
    let mut cos = vec![vec![0.0f64; nfft]; bins];
    let mut sin = vec![vec![0.0f64; nfft]; bins];
    for k in 0..bins {
        for n in 0..nfft {
            let a = -2.0 * std::f64::consts::PI * k as f64 * n as f64 / nfft as f64;
            cos[k][n] = a.cos();
            sin[k][n] = a.sin();
        }
    }
    (cos, sin)
}

fn load_basis(nfft: usize) -> Arc<Basis> {
    // A poisoned lock is taken rather than propagated: the map is a cache of a
    // pure function, so the worst a panic mid-insert leaves behind is a
    // missing entry, and refusing to enroll over it would be the larger fault.
    if let Some(hit) = BASES.lock().unwrap_or_else(|e| e.into_inner()).get(&nfft) {
        return Arc::clone(hit);
    }
    // Built outside the lock. Two threads meeting a cold size both build it
    // and the second insert wins, which costs one discarded basis and never a
    // thread stalled behind someone else's 29 MB of cosines.
    let built = Arc::new(build_basis(nfft));
    BASES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(nfft, Arc::clone(&built));
    built
}

fn power_spectrum(frame: &[f64], nfft: usize, basis: &Basis) -> Vec<f64> {
    let (cos, sin) = basis;
    let bins = nfft / 2 + 1;
    let mut out = vec![0.0f64; bins];
    for k in 0..bins {
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for n in 0..nfft {
            re += cos[k][n] * frame[n];
            im += sin[k][n] * frame[n];
        }
        out[k] = re * re + im * im;
    }
    out
}

fn mel_multiply(
    filters: &[f32],
    rows: usize,
    bins: usize,
    spectra: &[Vec<f64>],
    frames: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * frames];
    for r in 0..rows {
        for f in 0..frames {
            let mut acc = 0.0f32;
            for b in 0..bins {
                acc += filters[r * bins + b] * spectra[b][f] as f32;
            }
            out[r * frames + f] = acc;
        }
    }
    out
}

/// torch.stft(center=True, pad reflect) as power spectra: rows bins, columns frames.
fn centred_power_spectra(samples: &[f64], window: &[f32], drop_last: bool) -> Vec<Vec<f64>> {
    let nfft = window.len();
    let half = nfft / 2;
    let mut padded = vec![0.0f64; samples.len() + nfft];
    for i in 0..half {
        padded[i] = samples[half - i];
    }
    padded[half..half + samples.len()].copy_from_slice(samples);
    for i in 0..half {
        padded[half + samples.len() + i] = samples[samples.len() - 2 - i];
    }

    let mut frames = samples.len() / HOP_160 + 1;
    if drop_last {
        frames -= 1;
    }
    let bins = nfft / 2 + 1;
    let basis = load_basis(nfft);
    let mut out = vec![vec![0.0f64; frames]; bins];
    for f in 0..frames {
        let start = f * HOP_160;
        let mut frame = vec![0.0f64; nfft];
        for i in 0..nfft {
            frame[i] = padded[start + i] * window[i] as f64;
        }
        let sp = power_spectrum(&frame, nfft, &basis);
        for k in 0..bins {
            out[k][f] = sp[k];
        }
    }
    out
}

/// `_S3Tokenizer._log_mel`: 128-bin log mel, `[bin][frame]`, log10, eight
/// decades of headroom, shifted into [0, 1].
fn tokenizer_mel(samples: &[f64]) -> (Vec<f32>, usize) {
    let s3_hann = f32_table(S3_HANN);
    let s3_mel = f32_table(S3_MEL);
    let spectra = centred_power_spectra(samples, &s3_hann, true);
    let frames = spectra[0].len();
    let mel = mel_multiply(&s3_mel, 128, 201, &spectra, frames);

    let mut peak = f32::NEG_INFINITY;
    for &v in &mel {
        let v = (v as f64).max(1e-10).log10() as f32;
        if v > peak {
            peak = v;
        }
    }
    let ceiling = peak - 8.0;
    let out: Vec<f32> = mel
        .iter()
        .map(|&v| {
            let mut v = (v as f64).max(1e-10).log10() as f32;
            if v < ceiling {
                v = ceiling;
            }
            (v + 4.0) * 0.25
        })
        .collect();
    (out, frames)
}

/// The 24 kHz flow conditioning mel: `[80, frames]`, Slaney mels, log clamp 1e-5.
fn matcha_mel(samples: &[f64]) -> Vec<f32> {
    let matcha_hann = f32_table(MATCHA_HANN);
    let matcha_mel = f32_table(MATCHA_MEL);
    let pad = (MATCHA_NFFT - MATCHA_HOP) / 2;
    let mut padded = vec![0.0f64; samples.len() + 2 * pad];
    for i in 0..pad {
        padded[i] = samples[pad - i];
    }
    padded[pad..pad + samples.len()].copy_from_slice(samples);
    for i in 0..pad {
        padded[pad + samples.len() + i] = samples[samples.len() - 2 - i];
    }

    let frames = (padded.len() - MATCHA_NFFT) / MATCHA_HOP + 1;
    let bins = MATCHA_NFFT / 2 + 1;
    let basis = load_basis(MATCHA_NFFT);
    let mut spectra = vec![vec![0.0f64; frames]; bins];
    for f in 0..frames {
        let start = f * MATCHA_HOP;
        let mut frame = vec![0.0f64; MATCHA_NFFT];
        for i in 0..MATCHA_NFFT {
            frame[i] = padded[start + i] * matcha_hann[i] as f64;
        }
        let mut sp = power_spectrum(&frame, MATCHA_NFFT, &basis);
        for v in &mut sp {
            *v = (*v + 1e-9).sqrt();
        }
        for k in 0..bins {
            spectra[k][f] = sp[k];
        }
    }

    let mel = mel_multiply(&matcha_mel, 80, bins, &spectra, frames);
    mel.iter()
        .map(|&v| (v as f64).max(1e-5).ln() as f32)
        .collect()
}

/// torchaudio's Kaldi fbank: DC removal, 0.97 pre-emphasis, Povey window,
/// 512-point power spectrum, Kaldi mels, natural log, per-bin mean removed.
/// Returns `[frame][bin]`, matching torchaudio and the fixture.
fn kaldi_fbank(samples: &[f64]) -> Vec<f32> {
    let kaldi_mel = f32_table(KALDI_MEL);
    let kaldi_povey = f32_table(KALDI_POVEY);
    let frames = (samples.len() - FRAME_400) / HOP_160 + 1;
    let bins = KALDI_FFT / 2 + 1;
    let basis = load_basis(KALDI_FFT);

    let mut spectra = vec![vec![0.0f64; frames]; bins];
    for f in 0..frames {
        let start = f * HOP_160;
        let mut frame = vec![0.0f64; KALDI_FFT];
        let mut mean = 0.0f64;
        for i in 0..FRAME_400 {
            frame[i] = samples[start + i];
            mean += frame[i];
        }
        mean /= FRAME_400 as f64;
        for i in 0..FRAME_400 {
            frame[i] -= mean;
        }
        let prev = frame[0];
        for i in (1..FRAME_400).rev() {
            frame[i] -= 0.97 * frame[i - 1];
        }
        frame[0] -= 0.97 * prev;
        for i in 0..FRAME_400 {
            frame[i] *= kaldi_povey[i] as f64;
        }
        let sp = power_spectrum(&frame, KALDI_FFT, &basis);
        for k in 0..bins {
            spectra[k][f] = sp[k];
        }
    }

    let mut mel = mel_multiply(&kaldi_mel, 80, 256, &spectra, frames);
    let epsilon = 1.192_092_9e-7f32;
    for v in &mut mel {
        if *v < epsilon {
            *v = epsilon;
        }
        *v = (*v as f64).ln() as f32;
    }
    for b in 0..80 {
        let mut m = 0.0f64;
        for f in 0..frames {
            m += mel[b * frames + f] as f64;
        }
        m /= frames as f64;
        for f in 0..frames {
            mel[b * frames + f] -= m as f32;
        }
    }
    let mut out = vec![0.0f32; mel.len()];
    for f in 0..frames {
        for b in 0..80 {
            out[f * 80 + b] = mel[b * frames + f];
        }
    }
    out
}

/// The 40-bin power mel the utterance voice encoder reads, `[frame][bin]`,
/// computed on librosa's symmetric hann.
fn voice_encoder_mel(samples: &[f64]) -> (Vec<f32>, usize) {
    let ve_hann = f32_table(VE_HANN);
    let ve_mel = f32_table(VE_MEL);
    let spectra = centred_power_spectra(samples, &ve_hann, false);
    let frames = spectra[0].len();
    let bin_major = mel_multiply(&ve_mel, 40, 201, &spectra, frames);
    let mut out = vec![0.0f32; frames * 40];
    for f in 0..frames {
        for b in 0..40 {
            out[f * 40 + b] = bin_major[b * frames + f];
        }
    }
    (out, frames)
}

/// librosa.effects.trim(top_db=20) with the default reference (max): frame RMS
/// with center=True reflection padding, threshold 20 dB below peak, the sample
/// span from the first to the last frame above it.
fn trim(samples: &[f64]) -> Vec<f64> {
    const FRAME: usize = 2048;
    const HOP: usize = 512;
    let half = FRAME / 2;

    let mut padded = vec![0.0f64; samples.len() + FRAME];
    for i in 0..half {
        padded[i] = samples[half - i];
    }
    padded[half..half + samples.len()].copy_from_slice(samples);
    for i in 0..half {
        padded[half + samples.len() + i] = samples[samples.len() - 2 - i];
    }

    let n_frames = 1 + samples.len() / HOP;
    let mut rms = vec![0.0f64; n_frames];
    let mut peak = 0.0f64;
    for f in 0..n_frames {
        let start = f * HOP;
        let mut sum = 0.0f64;
        for i in start..start + FRAME {
            sum += padded[i] * padded[i];
        }
        let r = (sum / FRAME as f64).sqrt();
        rms[f] = r;
        if r > peak {
            peak = r;
        }
    }

    let (mut first, mut last) = (None, None);
    for (f, &r) in rms.iter().enumerate() {
        if r > 0.1 * peak {
            if first.is_none() {
                first = Some(f);
            }
            last = Some(f);
        }
    }
    let (Some(first), Some(last)) = (first, last) else {
        return samples.to_vec();
    };
    let start = first * HOP;
    let end = (last * HOP + HOP).min(samples.len());
    if end <= start {
        return samples.to_vec();
    }
    samples[start..end].to_vec()
}

// -------------------------------------------------------------------- enroller

/// An enrollment pipeline over the three exported graphs.
pub struct Enroller {
    execution: Execution,
    tokenizer: Session,
    camp: Session,
    ve: Session,
}

impl Enroller {
    /// Load the graphs on whatever execution provider `auto` finds.
    ///
    /// # Errors
    ///
    /// Everything [`Enroller::load_with`] returns.
    pub fn load(onnx_dir: &Path) -> Result<Self, String> {
        Self::load_with(onnx_dir, &ExecutionConfig::default())
    }

    /// Load the graphs on a named execution provider.
    ///
    /// # Errors
    ///
    /// Returns an error when a graph is missing, and when
    /// `execution.onnx_provider` names a provider this build or this machine
    /// does not offer.
    pub fn load_with(onnx_dir: &Path, execution: &ExecutionConfig) -> Result<Self, String> {
        let execution = Execution::resolve(execution, &crate::execution::available_providers()?)?;
        let open = |name: &str| -> Result<Session, String> {
            crate::execution::session_builder(execution.provider(), name)?
                .commit_from_file(onnx_dir.join(name))
                .map_err(ort_err)
        };
        let tokenizer = open("s3_tokenizer.onnx")?;
        let camp = open("camp.onnx")?;
        let ve = open("voice_encoder.onnx")?;
        Ok(Enroller {
            execution,
            tokenizer,
            camp,
            ve,
        })
    }

    /// The execution provider these sessions actually run on.
    #[must_use]
    pub fn execution(&self) -> Execution {
        self.execution
    }

    /// An enrolled voice from a WAV file: 16-bit PCM or 32-bit float, any
    /// rate, any channel count.
    ///
    /// The clip is resampled by [`enroll`](Enroller::enroll), which is the one
    /// resampler every port agrees to the bit; nothing here rewrites the
    /// samples first.
    ///
    /// # Errors
    ///
    /// Everything [`crate::wav::Audio::read_wav`] and
    /// [`enroll`](Enroller::enroll) return.
    pub fn enroll_wav(&mut self, path: impl AsRef<Path>) -> Result<Enrolled, String> {
        let clip = crate::wav::Audio::read_wav(path)?;
        self.enroll(&clip.samples, clip.sample_rate as usize)
    }

    /// An enrolled voice, before wrapping in a [`voice::Profile`].
    ///
    /// # Errors
    ///
    /// Everything [`validate_reference_audio`] refuses, and whatever the three
    /// graphs return.
    pub fn enroll(&mut self, audio: &[f32], sample_rate: usize) -> Result<Enrolled, String> {
        validate_reference_audio(audio, sample_rate)?;
        let wav: Vec<f64> = audio.iter().map(|&v| v as f64).collect();
        let wav24_full = if sample_rate == MEL_SR {
            wav
        } else {
            resample(audio, sample_rate, MEL_SR)
                .into_iter()
                .map(f64::from)
                .collect()
        };
        let max_samples = MAX_REF_SECONDS * MEL_SR;
        let wav24: Vec<f64> = wav24_full.iter().take(max_samples).copied().collect();

        let wav16_flow: Vec<f64> = resample(&to_f32(&wav24), MEL_SR, S3_SR)
            .into_iter()
            .map(f64::from)
            .collect();
        let wav16_t3: Vec<f64> = resample(&to_f32(&wav24_full), MEL_SR, S3_SR)
            .into_iter()
            .map(f64::from)
            .collect();

        let prompt_mel = matcha_mel(&wav24);
        let prompt_mel_frames = prompt_mel.len() / 80;

        let (tok_mel, _) = tokenizer_mel(&wav16_flow);
        let tokens = self.tokenize(&tok_mel)?;
        let n_tok = tokens.len().min(prompt_mel_frames / 2);
        let prompt_tokens = tokens[..n_tok].to_vec();
        let prompt_mel = prompt_mel[..80 * (2 * n_tok)].to_vec();

        let cond_samples = (COND_SECONDS * S3_SR).min(wav16_t3.len());
        let (cond_mel, _) = tokenizer_mel(&wav16_t3[..cond_samples]);
        let cond_tokens = self.tokenize_capped(&cond_mel, 150)?;

        let fbank = kaldi_fbank(&wav16_flow);
        let flow_emb = self.cam_embedding(&fbank)?;

        let speaker_emb = self.speaker_embedding(&wav16_t3)?;

        Ok(Enrolled {
            speaker_embedding: speaker_emb,
            flow_embedding: flow_emb,
            prompt_tokens,
            prompt_mel,
            prompt_mel_frames: 2 * n_tok,
            cond_prompt_tokens: cond_tokens,
            source_sample_rate: sample_rate,
        })
    }

    fn tokenize(&mut self, mel: &[f32]) -> Result<Vec<i64>, String> {
        let frames = mel.len() / 128;
        let input =
            Array3::from_shape_vec((1, 128, frames), mel.to_vec()).map_err(|e| e.to_string())?;
        let mut inputs: HashMap<String, DynValue> = HashMap::new();
        inputs.insert(
            "mel".to_string(),
            Value::from_array(input)
                .map_err(|e| e.to_string())?
                .into_dyn(),
        );
        let outputs = self.tokenizer.run(inputs).map_err(|e| e.to_string())?;
        let (_, data) = outputs[0]
            .try_extract_tensor::<i64>()
            .map_err(|e| e.to_string())?;
        Ok(data.to_vec())
    }

    fn tokenize_capped(&mut self, mel: &[f32], cap: usize) -> Result<Vec<i64>, String> {
        if mel.len() / 128 > cap * 4 {
            return self.tokenize(&mel[..128 * cap * 4]);
        }
        self.tokenize(mel)
    }

    fn cam_embedding(&mut self, fbank: &[f32]) -> Result<Vec<f32>, String> {
        let frames = fbank.len() / 80;
        let mut transposed = vec![0.0f32; fbank.len()];
        for f in 0..frames {
            for b in 0..80 {
                transposed[b * frames + f] = fbank[f * 80 + b];
            }
        }
        let input =
            Array3::from_shape_vec((1, 80, frames), transposed).map_err(|e| e.to_string())?;
        let mut inputs: HashMap<String, DynValue> = HashMap::new();
        inputs.insert(
            "fbank".to_string(),
            Value::from_array(input)
                .map_err(|e| e.to_string())?
                .into_dyn(),
        );
        let outputs = self.camp.run(inputs).map_err(|e| e.to_string())?;
        let (_, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        Ok(data.to_vec())
    }

    fn speaker_embedding(&mut self, wav16_t3: &[f64]) -> Result<Vec<f32>, String> {
        let trimmed = trim(wav16_t3);
        let (mel, frames) = voice_encoder_mel(&trimmed);

        let mut n_wins = 0;
        let mut rem = 0;
        let span = mel.len() / 40;
        if span > PARTIAL_FRAMES - PARTIAL_STEP {
            n_wins = (span - PARTIAL_FRAMES + PARTIAL_STEP) / PARTIAL_STEP;
            rem = (span - PARTIAL_FRAMES + PARTIAL_STEP) % PARTIAL_STEP;
        }
        if n_wins == 0
            || (rem + (PARTIAL_FRAMES - PARTIAL_STEP)) as f64 / PARTIAL_FRAMES as f64 >= 0.8
        {
            n_wins += 1;
        }
        let target = PARTIAL_FRAMES + PARTIAL_STEP * (n_wins - 1);
        let mut mel = mel;
        if target > frames {
            mel.resize(target * 40, 0.0);
        }

        let mut partials = vec![0.0f32; n_wins * PARTIAL_FRAMES * 40];
        for i in 0..n_wins {
            let start = i * PARTIAL_STEP * 40;
            partials[i * PARTIAL_FRAMES * 40..(i + 1) * PARTIAL_FRAMES * 40]
                .copy_from_slice(&mel[start..start + PARTIAL_FRAMES * 40]);
        }

        let input = Array3::from_shape_vec((n_wins, PARTIAL_FRAMES, 40), partials)
            .map_err(|e| e.to_string())?;
        let mut inputs: HashMap<String, DynValue> = HashMap::new();
        inputs.insert(
            "partials".to_string(),
            Value::from_array(input)
                .map_err(|e| e.to_string())?
                .into_dyn(),
        );
        let outputs = self.ve.run(inputs).map_err(|e| e.to_string())?;
        let (_, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        let per_partial = data.to_vec();

        let mut pooled = vec![0.0f32; 256];
        for i in 0..n_wins {
            for d in 0..256 {
                pooled[d] += per_partial[i * 256 + d];
            }
        }
        let norm = pooled
            .iter()
            .map(|&v| (v as f64) * (v as f64))
            .sum::<f64>()
            .sqrt();
        if norm > 0.0 {
            for v in &mut pooled {
                *v = (*v as f64 / norm) as f32;
            }
        }
        Ok(pooled)
    }
}

/// The enrolled voice's five tensors, before wrapping.
pub struct Enrolled {
    pub speaker_embedding: Vec<f32>,
    pub flow_embedding: Vec<f32>,
    pub prompt_tokens: Vec<i64>,
    pub prompt_mel: Vec<f32>,
    pub prompt_mel_frames: usize,
    pub cond_prompt_tokens: Vec<i64>,
    /// The recording's own rate, recorded for provenance.
    pub source_sample_rate: usize,
}

impl Enrolled {
    /// Wrap in a [`voice::Profile`]. `language` is what the voice reads in by
    /// default; `""` means `"en"`.
    #[must_use]
    pub fn profile(&self, name: &str, language: &str) -> voice::Profile {
        voice::Profile {
            enrolment: "first-10s".to_string(),
            name: name.to_string(),
            speaker_embedding: self.speaker_embedding.clone(),
            flow_embedding: self.flow_embedding.clone(),
            prompt_tokens: self.prompt_tokens.clone(),
            prompt_mel: self.prompt_mel.clone(),
            cond_prompt_tokens: self.cond_prompt_tokens.clone(),
            source_sample_rate: self.source_sample_rate,
            language: if language.is_empty() {
                "en".to_string()
            } else {
                language.to_string()
            },
        }
    }
}

fn to_f32(x: &[f64]) -> Vec<f32> {
    x.iter().map(|&v| v as f32).collect()
}
