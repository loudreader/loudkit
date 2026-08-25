//! WSOLA: the properties, not the samples.
//!
//! Nothing here pins a waveform. A time-stretch has no golden output that
//! survives a change of compiler, and the five ports sum the same floats in
//! their own order — so what is asserted is what a listener would notice if it
//! broke: the length, the pitch, the loudness, and the fact that `speed = 1.0`
//! is not a stretch at all but a bypass.
//!
//! The same properties are asserted in Python, Go, TypeScript and Swift. **A
//! shared byte-level fixture was considered and rejected**: the alignment search
//! picks between candidates whose cross-correlations can differ in the last bit
//! across languages, and one offset chosen differently moves every sample after
//! it — so the fixture would fail for a reason that is not a defect, which is
//! the worst kind of test to own.
//!
//! Needs no assets; this is arithmetic over a generated signal.

use std::f64::consts::PI;

use loudkit::timestretch::{stretched_length, time_stretch, validate_speed, MAX_SPEED, MIN_SPEED};

const SAMPLE_RATE: usize = 24_000;
const SPEEDS: [f64; 6] = [0.5, 0.75, 0.9, 1.25, 1.5, 2.0];

/// A voiced-ish test signal: a low fundamental, a harmonic, and a sweep.
///
/// Deterministic by construction — no RNG anywhere in this file, because the
/// stretcher has none either and a flaky DSP test is worse than no DSP test.
fn signal(seconds: f64, f0: f64) -> Vec<f32> {
    let n = (SAMPLE_RATE as f64 * seconds) as usize;
    (0..n)
        .map(|i| {
            let t = i as f64 / SAMPLE_RATE as f64;
            let wave = 0.5 * (2.0 * PI * f0 * t).sin()
                + 0.25 * (2.0 * PI * 2.0 * f0 * t).sin()
                + 0.15 * (2.0 * PI * 600.0 * t * (1.0 + t)).sin();
            wave as f32
        })
        .collect()
}

/// Fundamental by autocorrelation. Enough to catch a chipmunk.
fn pitch_hz(x: &[f32]) -> f64 {
    let from = SAMPLE_RATE / 4;
    let window: Vec<f64> = x[from..from + 4096].iter().map(|v| f64::from(*v)).collect();
    let mean = window.iter().sum::<f64>() / window.len() as f64;
    let window: Vec<f64> = window.iter().map(|v| v - mean).collect();

    let lo = SAMPLE_RATE / 500;
    let hi = SAMPLE_RATE / 80;
    let mut best_lag = lo;
    let mut best = f64::NEG_INFINITY;
    for lag in lo..hi {
        let ac: f64 = window[..window.len() - lag]
            .iter()
            .zip(&window[lag..])
            .map(|(a, b)| a * b)
            .sum();
        if ac > best {
            best = ac;
            best_lag = lag;
        }
    }
    SAMPLE_RATE as f64 / best_lag as f64
}

fn rms(x: &[f32]) -> f64 {
    let sum: f64 = x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
    (sum / x.len() as f64).sqrt()
}

/// Bit-identical, not merely equal. The engine's default must not depend on a
/// DSP path being lossless — it must not enter the DSP path at all.
///
/// Rust cannot return the borrowed slice as an owned `Vec` without copying it,
/// so this is the closest thing to Python's identity check that the ownership
/// model allows: every sample is the same 32 bits it went in as.
#[test]
fn unity_speed_is_a_bypass() {
    let x = signal(0.3, 220.0);
    let got = time_stretch(&x, SAMPLE_RATE, 1.0);
    assert_eq!(got.len(), x.len());
    for (a, b) in x.iter().zip(&got) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn the_output_is_exactly_as_long_as_asked() {
    let x = signal(1.0, 220.0);
    for speed in SPEEDS {
        let got = time_stretch(&x, SAMPLE_RATE, speed);
        assert_eq!(got.len(), stretched_length(x.len(), speed), "speed {speed}");
    }
}

/// Python rounds halves to even; Rust, Go, Swift and JavaScript do not. A
/// one-sample disagreement on an exact half is found six months later, in a
/// conformance run, by somebody else — so the formula is spelled
/// `floor(n / speed + 0.5)` in all five rather than handed to a rounding
/// helper.
#[test]
fn the_length_formula_is_half_up_not_half_even() {
    assert_eq!(stretched_length(5, 2.0), 3);
}

/// No overlap to align, so it is cut or padded. At 24 kHz this is under 25 ms —
/// below anything the engine renders, and the alternative is a panic on the
/// degenerate case.
#[test]
fn a_fragment_shorter_than_a_frame_is_still_the_right_length() {
    let tiny = vec![1.0f32; 64];
    assert_eq!(time_stretch(&tiny, SAMPLE_RATE, 2.0).len(), 32);
    let stretched = time_stretch(&tiny, SAMPLE_RATE, 0.5);
    assert_eq!(stretched.len(), 128);
    // Padded with silence rather than with whatever was in the allocation.
    assert!(stretched[64..].iter().all(|s| *s == 0.0));
}

/// The entire point. A resampler would move the fundamental by exactly `speed`;
/// this must not move it at all.
#[test]
fn pitch_is_preserved() {
    let x = signal(1.0, 220.0);
    let before = pitch_hz(&x);
    for speed in SPEEDS {
        let got = time_stretch(&x, SAMPLE_RATE, speed);
        let after = pitch_hz(&got);
        assert!(
            (after - before).abs() / before < 0.03,
            "speed {speed}: {before} Hz became {after} Hz"
        );
    }
}

/// Overlap-add with a window that does not sum to one is the classic way to get
/// a 6 dB drop or a comb filter. The periodic Hann at 50 % overlap sums to one,
/// and the denominator corrects the ends.
#[test]
fn loudness_survives() {
    let x = signal(1.0, 220.0);
    let before = rms(&x);
    for speed in SPEEDS {
        let got = time_stretch(&x, SAMPLE_RATE, speed);
        let after = rms(&got);
        assert!(
            (after - before).abs() / before < 0.15,
            "speed {speed}: RMS {before} became {after}"
        );
    }
}

#[test]
fn nothing_clips_or_goes_non_finite() {
    let x = signal(1.0, 220.0);
    for speed in SPEEDS {
        for sample in time_stretch(&x, SAMPLE_RATE, speed) {
            assert!(sample.is_finite(), "speed {speed}");
            assert!(sample.abs() <= 1.2, "speed {speed}: {sample}");
        }
    }
}

/// The correlation search divides by candidate energy; a silent frame is where
/// that division has to not happen.
#[test]
fn silence_stays_silent() {
    let got = time_stretch(&vec![0.0f32; SAMPLE_RATE], SAMPLE_RATE, 1.5);
    assert!(got.iter().all(|s| *s == 0.0));
}

/// No RNG, no adaptivity, no wall clock. Same in, same out, forever.
#[test]
fn two_calls_agree_bit_for_bit() {
    let x = signal(0.5, 220.0);
    let first = time_stretch(&x, SAMPLE_RATE, 1.4);
    let second = time_stretch(&x, SAMPLE_RATE, 1.4);
    assert_eq!(first.len(), second.len());
    for (a, b) in first.iter().zip(&second) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

/// A caller who asked for 3x and silently got 2x has a bug only a stopwatch
/// finds.
#[test]
fn out_of_range_is_refused_not_clamped() {
    for speed in [0.49, 2.01, 0.0, -1.0, 10.0] {
        let err = validate_speed(speed).expect_err("a speed outside the range must be refused");
        // The same sentence Python, Go, TypeScript and Swift print, range
        // included: a user who hits this in two languages should read it twice.
        assert!(err.contains("outside [0.5, 2.0]"), "speed {speed}: {err}");
    }
}

/// `NaN` compares false against both bounds, so a naive range test would let it
/// through and produce an empty waveform three stages later.
#[test]
fn non_finite_is_refused_before_the_range_check() {
    for speed in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let err = validate_speed(speed).expect_err("a non-finite speed must be refused");
        assert!(err.contains("finite"), "{speed}: {err}");
    }
}

#[test]
fn the_bounds_themselves_are_allowed() {
    for speed in [MIN_SPEED, 1.0, MAX_SPEED] {
        assert!(validate_speed(speed).is_ok(), "speed {speed}");
    }
}

/// The guard that no implementation tested, which is how two of the five
/// shipped without it.
///
/// Below ~60 Hz the derived frame is one sample, so the hop — `frame / 2` — is
/// zero, and the overlap-add loop advances by it. TypeScript and Swift computed
/// the hop *after* the degenerate-shape guard and never tested it, so both
/// looped forever on an input Rust returned from in microseconds; nothing was
/// red, because nothing asked. Now all five ask.
///
/// Bounded by the clock as well as by the length: a regression here is a hang,
/// and a hang is a suite that never finishes rather than a suite that fails.
#[test]
fn a_sample_rate_too_low_to_have_a_hop_does_not_hang() {
    let started = std::time::Instant::now();
    let got = time_stretch(&vec![0.0f32; 64], 40, 1.5);
    assert!(
        started.elapsed() < std::time::Duration::from_secs(5),
        "the overlap-add loop did not terminate"
    );
    assert_eq!(got.len(), stretched_length(64, 1.5));
    assert_eq!(got.len(), 43);
}
