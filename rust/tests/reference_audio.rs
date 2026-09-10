//! The five refusals `validate_reference_audio` makes on the Python side, run
//! here without weights. Each message is the reference's, word for word,
//! because a caller reads it and a port is held to it.

use loudkit::enroll::validate_reference_audio;

const SR: usize = 24_000;

const GOOD_INPUT: &str = "A good input is 5 to 10 seconds of one person speaking, clean, \
                          without music or a second voice.";

const NAN_REFUSAL: &str = "the recording contains NaN or Inf samples, so no voice can be \
                           derived from it. Re-export the file. A good input is 5 to 10 \
                           seconds of one person speaking, clean, without music or a \
                           second voice.";

/// A deterministic clip loud enough to pass the silence floor.
fn speech_like(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (0.5 * (2.0 * std::f64::consts::PI * 220.0 * i as f64 / SR as f64).sin()) as f32)
        .collect()
}

fn refusal(audio: &[f32], sample_rate: usize) -> String {
    validate_reference_audio(audio, sample_rate).expect_err("the recording was accepted")
}

#[test]
fn a_non_positive_rate_is_refused() {
    // `usize` makes a negative rate unrepresentable here, so zero is the whole
    // of the condition this port can meet.
    assert_eq!(
        refusal(&speech_like(SR), 0),
        "sample rate must be positive, got 0"
    );
}

#[test]
fn nan_and_inf_samples_are_refused() {
    for (at, value) in [
        (0, f32::NAN),
        (2 * SR - 1, f32::NAN),
        (0, f32::INFINITY),
        (2 * SR - 1, f32::NEG_INFINITY),
    ] {
        let mut audio = speech_like(2 * SR);
        audio[at] = value;
        assert_eq!(refusal(&audio, SR), NAN_REFUSAL, "sample {at} = {value}");
    }
}

/// 720 samples is one short of the reflect padding `matcha_mel` reads, so
/// without this guard it indexes past the end of the slice and panics.
#[test]
fn a_clip_shorter_than_a_second_is_refused_before_the_reflect_padding() {
    for (samples, seconds) in [
        (720, "0.03"),
        (0, "0.00"),
        // Half a second: no panic, but the utterance encoder pads it out to
        // its 1.6 s first partial and enrolls mostly padding.
        (SR / 2, "0.50"),
        // One sample under the minimum. The reported seconds round to 1.00 and
        // the refusal still stands: the comparison is on the exact length.
        (SR - 1, "1.00"),
    ] {
        assert_eq!(
            refusal(&speech_like(samples), SR),
            format!(
                "the recording is {seconds} s: too short to enroll a speaker from \
                 (minimum 1 s). {GOOD_INPUT}"
            ),
            "{samples} samples"
        );
    }
}

#[test]
fn a_recording_longer_than_thirty_seconds_is_refused() {
    assert_eq!(
        refusal(&speech_like(31 * SR), SR),
        format!(
            "the recording is 31.0 s. Only the first 10 s become the voice prompt, and \
             the whole clip shapes the speaker embedding, so a long recording enrolls \
             something the prompt does not carry. Trim it to the best 5 to 10 seconds \
             (at most 30 s). {GOOD_INPUT}"
        )
    );
}

#[test]
fn a_silent_recording_is_refused() {
    for (value, peak) in [
        (0.0f32, "0.0e+00"),
        (5e-5, "5.0e-05"),
        // The floor itself. f32 rounds 1e-4 down, so the loudest sample an f32
        // clip can hold at this level is still under the f64 floor the
        // reference compares against.
        (1e-4, "1.0e-04"),
    ] {
        assert_eq!(
            refusal(&vec![value; 2 * SR], SR),
            format!(
                "the recording is silent (peak {peak}); there is no voice in it to \
                 enroll. {GOOD_INPUT}"
            ),
            "peak {value}"
        );
    }
}

/// Finiteness is checked before anything arithmetic, because one NaN poisons
/// every statistic below it. A clip that breaks two rules names the first.
#[test]
fn the_order_of_the_checks_is_the_reference_order() {
    let mut short_and_nan = speech_like(720);
    short_and_nan[0] = f32::NAN;
    assert_eq!(refusal(&short_and_nan, SR), NAN_REFUSAL);

    let mut silent_and_nan = vec![0.0f32; 2 * SR];
    silent_and_nan[7] = f32::NAN;
    assert_eq!(refusal(&silent_and_nan, SR), NAN_REFUSAL);

    // Length before loudness, as the reference orders them.
    assert!(refusal(&vec![0.0f32; 31 * SR], SR).starts_with("the recording is 31.0 s"));
}

/// The clip every port's enrollment conformance runs on. It needs no graphs to
/// be judged, so the guard is held to real reference audio and not only to
/// synthetic tones.
#[test]
fn the_fixture_clip_still_enrolls() {
    let dir = std::env::var("LOUDKIT_ENROLL_FIXTURE")
        .unwrap_or_else(|_| "../tests/data/enrollment".to_string());
    let path = std::path::Path::new(&dir).join("ref_audio.f32");
    let Ok(bytes) = std::fs::read(&path) else {
        eprintln!("SKIPPED (not a pass): enrollment fixture not found: {dir}");
        return;
    };
    let audio: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    validate_reference_audio(&audio, SR).expect("the fixture clip was refused");
}

/// The bounds are inclusive at both ends and the fixture clip sits inside
/// them, so tightening the guard cannot start refusing what already enrolls.
#[test]
fn the_band_itself_is_accepted() {
    for samples in [SR, SR + 1, 15 * SR, 30 * SR] {
        // One sample just over the f32 floor, and silence everywhere else.
        let mut audio = vec![0.0f32; samples];
        audio[0] = 1.01e-4;
        assert!(
            validate_reference_audio(&audio, SR).is_ok(),
            "{samples} samples at 24 kHz"
        );
    }
}
