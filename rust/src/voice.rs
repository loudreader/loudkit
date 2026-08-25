//! Voice profile loading — a port of `loudkit.voice.VoiceProfile.load`.

use serde_json::Value;

use crate::safetensors;

pub struct Profile {
    pub name: String,
    pub speaker_embedding: Vec<f32>,
    pub flow_embedding: Vec<f32>,
    pub prompt_tokens: Vec<i64>,
    pub prompt_mel: Vec<f32>,
    pub cond_prompt_tokens: Vec<i64>,
    /// Language of the reference audio, for provenance — and, since the engine
    /// consults it, the language a synthesis reads as when the caller names
    /// none. Written by `loudkit.voice.VoiceProfile.save` into the same header
    /// as `name`; dropping the key makes a Polish voice read Polish text as
    /// English.
    pub language: String,
}

const FORMAT_VERSION: u64 = 1;

/// The constant fed to the generator's emotion conditioning slot.
///
/// The checkpoint reserves one of its 34 conditioning slots for an emotion
/// scalar. On these weights the axis is dead (distillation collapsed it), so
/// the slot is not a control and not part of the profile format — but it must
/// be fed the value the model was distilled with. Every port uses this.
pub const EMOTION_NEUTRAL: f32 = 0.5;

/// The two speaker encoders' output widths, and the mel bin count. Mirrors
/// `loudkit.voice.VoiceProfile`, which validates the same three.
const SPEAKER_DIM: usize = 256;
const FLOW_DIM: usize = 192;
const MEL_BINS: usize = 80;

/// Smallest speaker-vector norm a profile may carry.
///
/// Below this the renderers stop agreeing: this port and CoreML divide by the
/// raw norm and yield NaN, torch's `F.normalize` carries an epsilon and yields
/// a finite — but arbitrary — direction. Enrolled vectors are order-1; anything
/// this small is a corrupt or synthetic file, not a quiet voice.
const MIN_EMBEDDING_NORM: f32 = 1e-6;

/// Reject an embedding the renderers would disagree about.
///
/// A profile is a file that gets copied, mailed and downloaded, so these belong
/// at the boundary rather than in each backend. Python has validated them since
/// the degenerate-profile fix; the ports accepted anything shaped like floats
/// and blew up deeper in inference, where the error names a matrix rather than
/// a file.
fn check_embedding(name: &str, values: &[f32], expected: usize) -> Result<(), String> {
    if values.len() != expected {
        return Err(format!("{name} must be {expected}-d, got {}", values.len()));
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(format!("{name} contains NaN or infinity"));
    }
    let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm < MIN_EMBEDDING_NORM {
        return Err(format!(
            "{name} has norm {norm:e}, below {MIN_EMBEDDING_NORM:e}: a zero or near-zero \
             speaker vector normalises to NaN here and to a finite arbitrary direction on \
             torch, so the same file would speak differently per backend"
        ));
    }
    Ok(())
}

/// The shipped model's dimensions, the same two Python reads out of its
/// `AlgorithmConfig`.
///
/// Both ends, not just the floor: without the ceiling `prompt_tokens = [9000]`
/// loads cleanly and then indexes past the end of the embedding table. The ceilings are
/// the shipped model's — prompt tokens index the speech codebook below the
/// start-of-speech marker, conditioning tokens the whole speech vocabulary.
const START_SPEECH_TOKEN: i64 = 6561;
const SPEECH_VOCAB_SIZE: i64 = 8194;

/// Matches Python's `MAX_VOICE_BYTES`, which the other four readers never had.
///
/// A voice profile is a handful of small tensors, and a safetensors file
/// claiming otherwise is not one. The cap is on the file, before it is opened,
/// because the shape checks that follow only run once a header has been parsed.
pub const MAX_VOICE_BYTES: u64 = 8 * 1024 * 1024;

pub fn load(path: &str) -> Result<Profile, String> {
    if let Ok(meta) = std::fs::metadata(path) {
        if meta.len() > MAX_VOICE_BYTES {
            return Err(format!(
                "{path}: {} bytes, over the {MAX_VOICE_BYTES} byte limit for a voice",
                meta.len()
            ));
        }
    }
    let f = safetensors::File::open(path)?;
    let header: Value = f
        .metadata
        .get("voice")
        .map(|s| serde_json::from_str(s).unwrap_or(Value::Null))
        .unwrap_or(Value::Null);
    let version = header["format_version"].as_u64().unwrap_or(0);
    if version != FORMAT_VERSION {
        return Err(format!(
            "{path}: voice format version {version}, this build reads {FORMAT_VERSION}"
        ));
    }
    let speaker_embedding = f.f32("speaker_embedding")?;
    let flow_embedding = f.f32("flow_embedding")?;
    let prompt_tokens = f.i64("prompt_tokens")?;
    let prompt_mel = f.f32("prompt_mel")?;
    let cond_prompt_tokens = f.i64("cond_prompt_tokens")?;

    check_embedding("speaker_embedding", &speaker_embedding, SPEAKER_DIM)?;
    check_embedding("flow_embedding", &flow_embedding, FLOW_DIM)?;
    if prompt_mel.iter().any(|v| !v.is_finite()) {
        return Err("prompt_mel contains NaN or infinity".to_string());
    }
    if prompt_mel.len() % MEL_BINS != 0 {
        return Err(format!(
            "prompt_mel must be ({MEL_BINS}, frames), got {} values",
            prompt_mel.len()
        ));
    }
    for (name, tokens, ceiling) in [
        ("prompt_tokens", &prompt_tokens, START_SPEECH_TOKEN),
        ("cond_prompt_tokens", &cond_prompt_tokens, SPEECH_VOCAB_SIZE),
    ] {
        if let Some(bad) = tokens.iter().find(|t| **t >= ceiling) {
            return Err(format!(
                "{name} contains id {bad}, at or past the {ceiling} the model has"
            ));
        }
        // Negative ids index an embedding table from the end — silently.
        if let Some(bad) = tokens.iter().find(|t| **t < 0) {
            return Err(format!("{name} contains a negative id: {bad}"));
        }
    }
    Ok(Profile {
        name: header["name"]
            .as_str()
            .map(|s| s.to_string())
            .unwrap_or_else(|| "voice".to_string()),
        speaker_embedding,
        flow_embedding,
        prompt_tokens,
        prompt_mel,
        cond_prompt_tokens,
        language: header_language(&header),
    })
}

/// The profile's language, or `"en"` when the header does not say.
///
/// A named function rather than an inline `unwrap_or_else` so the absent-key
/// branch is reachable from a test without writing a whole safetensors file:
/// it is the branch every profile written before this port read the field back
/// actually takes, and the reason the engine's language chain does not
/// retrofit them — they load as English, not as blank, so they inherit nothing.
fn header_language(header: &Value) -> String {
    header["language"].as_str().unwrap_or("en").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A profile the renderers would disagree about must not load.
    ///
    /// Shape validation alone let a well-shaped but degenerate file through,
    /// and the three renderers then differed on what it meant: torch's
    /// `F.normalize` carries an epsilon and returns a finite (arbitrary)
    /// direction for a zero speaker vector, while this port and CoreML divide
    /// by the raw norm and produce NaN.
    #[test]
    fn degenerate_embeddings_are_refused() {
        let good = vec![0.0625f32; FLOW_DIM];
        assert!(check_embedding("flow_embedding", &good, FLOW_DIM).is_ok());

        assert!(check_embedding("flow_embedding", &vec![0.0; FLOW_DIM], FLOW_DIM).is_err());
        assert!(check_embedding("flow_embedding", &good[..8], FLOW_DIM).is_err());

        let mut nan = good.clone();
        nan[3] = f32::NAN;
        assert!(check_embedding("flow_embedding", &nan, FLOW_DIM).is_err());

        let mut inf = good.clone();
        inf[0] = f32::INFINITY;
        assert!(check_embedding("flow_embedding", &inf, FLOW_DIM).is_err());
    }

    /// A header that omits `language` reads as `"en"`, matching
    /// `loudkit.voice.VoiceProfile.load` and the other three ports.
    ///
    /// Dropping the key makes every profile English
    /// whatever the file says. The absent-key branch is the one real files
    /// take — Python writes the key, but files written by anything
    /// older do not — and it is what stops the engine's language chain from
    /// retrofitting them.
    #[test]
    fn a_header_without_a_language_reads_as_english() {
        assert_eq!(header_language(&serde_json::json!({"name": "x"})), "en");
        assert_eq!(header_language(&Value::Null), "en");
        assert_eq!(
            header_language(&serde_json::json!({"language": "pl"})),
            "pl"
        );
        // An explicitly empty value is preserved rather than defaulted: it is
        // the one input the engine's fallback branch exists for, and turning it
        // into "en" here would make that branch unreachable.
        assert_eq!(header_language(&serde_json::json!({"language": ""})), "");
    }
}
