//! Playing faster without talking higher — WSOLA, from first principles.
//!
//! "Speed" in a reading app means what it means on a video player: 1.5x is the
//! same voice, sooner. Resampling gives you a chipmunk; what is wanted is *time*
//! stretched while *pitch* is left alone.
//!
//! **Why WSOLA and not a phase vocoder.** The phase vocoder is the other
//! standard answer and is better on sustained, harmonic material — held notes,
//! chords. Speech is the opposite kind of signal: it is mostly transients
//! (plosives, the attack of every syllable) sitting on a pitch that moves
//! continuously. A phase vocoder resynthesises from magnitudes and unwrapped
//! phases, and its characteristic failure on that material is transient smearing
//! — a /t/ arriving as a soft thud, "phasiness" on voiced segments — which is
//! precisely the part of speech intelligibility rests on. WSOLA never leaves the
//! time domain: it copies real waveform segments and only chooses *where* to
//! copy them from, so a plosive is either included whole or not at all. It
//! cannot smear what it never transforms.
//!
//! **The algorithm.** Cut the input into overlapping ~25 ms frames. Write them
//! back out at a hop fixed by the output rate (50 % overlap), and read them in
//! at a hop scaled by `speed`. The read position is not used as computed: it is
//! moved by up to ±10 ms to whichever offset best matches what the previously
//! written frame *would* naturally have been followed by. That search is the
//! "waveform similarity" in the name, and it keeps successive frames in phase
//! with each other, so the overlap-add reinforces
//! rather than cancels. A plain OLA without the search is the same code with the
//! search window set to zero, and it sounds like it: periodic warble at the
//! frame rate.
//!
//! Everything here is deterministic — no RNG, no adaptivity, no libraries. The
//! constants are derived from the sample rate rather than written as sample
//! counts, so the same code is correct at 16 kHz or 48 kHz, and the five
//! implementations derive them the same way.
//!
//! **What it costs.** At 1.25x this is hard to tell from a native reading. At
//! 2x, or at 0.5x, it is audibly processed: the alignment search cannot always
//! find a match, and the artefact is a faint roughness or a doubled consonant.
//! That is the practical range, and the bounds below are set where the result stops
//! being worth offering rather than where the arithmetic stops working.
//!
//! Mirrors `loudkit.models.timestretch` in Python.

use std::f64::consts::PI;

/// The range worth offering, not the range that runs.
///
/// Outside it the alignment search stops finding matches often enough — the
/// required shift exceeds the ±10 ms it may look over — and the output is
/// recognisably processed rather than merely faster. Refused rather than
/// clamped: a caller who asked for 3x and silently got 2x has a bug that only a
/// stopwatch finds.
pub const MIN_SPEED: f64 = 0.5;
/// The upper end of [`MIN_SPEED`]'s range.
pub const MAX_SPEED: f64 = 2.0;

/// Analysis/synthesis frame. Long enough to hold two periods of the lowest
/// voiced pitch this is used on (~80 Hz), short enough that a frame is inside
/// one phone.
const FRAME_MS: f64 = 25.0;

/// How far the read position may move to find a better join — a bit under one
/// pitch period at the low end of the voiced range, which is what the search is
/// looking for.
const SEARCH_MS: f64 = 10.0;

/// Frames overlap by half. A periodic Hann window at hop = frame/2 sums to
/// exactly one, so the overlap-add needs no normalisation of its own — the
/// denominator below only ever corrects the ends and the places the alignment
/// search moved a frame off the grid.
const HANN_COLA_HOP: usize = 2;

/// `Ok(())` if `speed` is usable, or the refusal that names the range.
///
/// Kept here rather than in the engine so that every entry point — three engine
/// methods and the CLI — refuses the same values with the same words, and a new
/// entry point cannot forget to.
///
/// # Errors
/// For a non-finite speed, and for one outside `[0.5, 2.0]`. The wording matches
/// Python's, so a user who hits it in two languages reads the same sentence
/// twice.
pub fn validate_speed(speed: f64) -> Result<(), String> {
    // Checked before the range, and not folded into it: `NaN` compares false
    // against both bounds, so a naive `MIN <= s <= MAX` would let it through and
    // produce an empty waveform three stages later.
    if !speed.is_finite() {
        return Err(format!("speed must be a finite number, not {speed:?}"));
    }
    // Formatted with `{:?}` rather than `{}`: Rust's `Display` prints 2.0 as
    // "2", and the refusal a user reads should name the same range in all five
    // languages rather than one that looks like a different number.
    if !(MIN_SPEED..=MAX_SPEED).contains(&speed) {
        return Err(format!(
            "speed {speed:?} is outside [{MIN_SPEED:?}, {MAX_SPEED:?}]. Beyond that \
             range the time-stretch is audibly processed rather than merely \
             faster or slower, so it is refused rather than clamped."
        ));
    }
    Ok(())
}

/// How long `n_samples` becomes at `speed`.
///
/// Written as `floor(n / speed + 0.5)` rather than with a rounding helper on
/// purpose: Python rounds halves to even, Go, Rust, Swift and JavaScript do not,
/// and a one-sample disagreement between ports on an exact half is the kind of
/// thing that is found six months later in a conformance run.
#[must_use]
pub fn stretched_length(n_samples: usize, speed: f64) -> usize {
    let scaled = (n_samples as f64 / speed + 0.5).floor();
    // A negative or non-finite `scaled` only reaches here from a caller that
    // skipped `validate_speed`; saturating to zero is what the `as` cast does
    // and it is the same answer Python's `int(...)` would refuse to reach.
    if scaled > 0.0 {
        scaled as usize
    } else {
        0
    }
}

/// `audio` played at `speed`, same pitch.
///
/// `speed` greater than one shortens, less than one lengthens. `1.0` is the
/// bypass: the samples come back untouched, having entered no arithmetic at all.
/// Rust cannot hand a borrowed slice back as an owned `Vec` without copying it,
/// so unlike Python — which returns the caller's own array — this returns a
/// fresh allocation; the *values* are bit-identical, every byte the vocoder
/// produced, and that is the property the default depends on.
///
/// Returns exactly `stretched_length(audio.len(), speed)` samples.
///
/// # Panics
/// For a `speed` that [`validate_speed`] refuses. Every engine entry point
/// validates at the door — before the six seconds of generation the caller would
/// otherwise wait to discover a typo — so this is unreachable through the
/// engine, and it is a panic rather than a `Result` because there is no
/// waveform to return for a speed that has no meaning. Silently substituting one
/// is the clamp this feature exists to refuse.
#[must_use]
pub fn time_stretch(audio: &[f32], sample_rate: usize, speed: f64) -> Vec<f32> {
    if let Err(why) = validate_speed(speed) {
        panic!("{why}");
    }
    if speed == 1.0 {
        return audio.to_vec();
    }

    let n = audio.len();
    let out_len = stretched_length(n, speed);
    let frame = (sample_rate as f64 * FRAME_MS / 1000.0 + 0.5).floor() as usize;
    let hop = frame / HANN_COLA_HOP;
    if n <= frame || out_len == 0 || hop == 0 {
        // Nothing to overlap-add: a fragment shorter than one frame has no
        // second frame to align against. Cut or zero-padded to the right length
        // instead, which is wrong in the way silence is wrong rather than in the
        // way a pitch shift is. At 24 kHz a frame is 600 samples — a fortieth of
        // a second, below anything the engine renders.
        //
        // A zero hop joins that branch rather than looping forever. It takes a
        // sample rate under 60 Hz to reach, so it is not a behaviour difference
        // in any case a caller can hit — it turns a hang, which no backtrace
        // explains, into the short-fragment path.
        let mut out = vec![0.0f32; out_len];
        let take = out_len.min(n);
        out[..take].copy_from_slice(&audio[..take]);
        return out;
    }

    let search = (sample_rate as f64 * SEARCH_MS / 1000.0 + 0.5).floor() as usize;
    // Periodic Hann, i.e. 2*pi*i/frame and not /(frame-1). The periodic form is
    // the one that sums to exactly one at 50 % overlap; the symmetric form is off
    // by a hair at every frame boundary, which reads as a low-level buzz at the
    // frame rate — 40 Hz here, right in the range a listener notices.
    let window: Vec<f64> = (0..frame)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / frame as f64).cos())
        .collect();

    // Every intermediate is f64. The waveform is f32, but a sum of six hundred
    // products accumulated in f32 loses the low bits the alignment search is
    // ranking candidates by, and the five ports agree on f64 arithmetic.
    let x: Vec<f64> = audio.iter().map(|s| f64::from(*s)).collect();
    // Room for the last frame to be written whole; trimmed at the end.
    let mut acc = vec![0.0f64; out_len + frame];
    let mut weight = vec![0.0f64; out_len + frame];

    let mut last_frame_at: usize = 0;
    let mut write_at: usize = 0;
    let mut k: usize = 0;
    while write_at < out_len {
        let ideal = (k as f64 * hop as f64 * speed + 0.5).floor() as i64;
        let read_at = if k == 0 {
            0
        } else {
            // What the previous frame would naturally have been followed by. The
            // search asks which nearby segment continues *this*, not which one
            // the arithmetic pointed at.
            let from = (last_frame_at + hop).min(n);
            let to = (from + frame).min(n);
            best_match(&x, &x[from..to], ideal, search, frame)
        };
        let read_at = read_at.min(n - frame);

        let segment = &x[read_at..read_at + frame];
        for (i, (w, s)) in window.iter().zip(segment).enumerate() {
            acc[write_at + i] += w * s;
            weight[write_at + i] += w;
        }

        last_frame_at = if n >= frame + hop {
            read_at.min(n - frame - hop)
        } else {
            read_at
        };
        write_at += hop;
        k += 1;
    }

    // The Hann pair sums to one in the interior, so this division is the
    // identity almost everywhere; it earns its place at the two ends, where only
    // one frame contributes and the raw sum would fade in and out.
    let mut out = vec![0.0f32; out_len];
    for (o, (a, w)) in out.iter_mut().zip(acc.iter().zip(&weight)) {
        if *w > 1e-12 {
            *o = (a / w) as f32;
        }
    }
    out
}

/// The offset within ±`search` of `ideal` whose frame best continues `target`.
///
/// Scored by cross-correlation normalised by the *candidate's* energy only — the
/// target's is the same for every candidate and cancels out of the ranking.
/// Without that normalisation the search prefers whichever candidate is loudest
/// rather than whichever fits, which at a syllable onset is exactly the wrong
/// one.
///
/// Ties go to the lower offset (the comparison is a strict `>`), so the choice
/// does not depend on iteration order and the five ports agree.
///
/// `ideal` is signed and `x.len() - frame` is computed as a signed bound because
/// both ends of the search window can fall outside the input: an unsigned
/// subtraction there is the wrap-around that turns "before the start" into "past
/// the end".
fn best_match(x: &[f64], target: &[f64], ideal: i64, search: usize, frame: usize) -> usize {
    let n = x.len() as i64;
    let last = n - frame as i64;
    let lo = (ideal - search as i64).max(0);
    let hi = last.min(ideal + search as i64);
    if hi < lo || target.len() < frame {
        return ideal.clamp(0, last) as usize;
    }

    let mut best_at = lo;
    let mut best_score = f64::NEG_INFINITY;
    for at in lo..=hi {
        let candidate = &x[at as usize..at as usize + frame];
        let energy: f64 = candidate.iter().map(|c| c * c).sum();
        // A silent candidate scores zero rather than dividing by nothing.
        let score = if energy <= 0.0 {
            0.0
        } else {
            let dot: f64 = candidate.iter().zip(target).map(|(c, t)| c * t).sum();
            dot / energy.sqrt()
        };
        if score > best_score {
            best_score = score;
            best_at = at;
        }
    }
    best_at as usize
}
