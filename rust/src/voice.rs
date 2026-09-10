//! Voice profiles, read and written: a port of `loudkit.voice.VoiceProfile`.

use serde_json::Value;

use crate::safetensors;

pub struct Profile {
    /// Prompt preparation strategy; empty means legacy first-10s.
    pub enrolment: String,
    pub name: String,
    pub speaker_embedding: Vec<f32>,
    pub flow_embedding: Vec<f32>,
    pub prompt_tokens: Vec<i64>,
    pub prompt_mel: Vec<f32>,
    pub cond_prompt_tokens: Vec<i64>,
    /// The recording's own sample rate, for provenance.
    pub source_sample_rate: usize,
    /// Language of the reference audio, and the language a synthesis reads
    /// as when the caller names none.
    pub language: String,
}

const FORMAT_VERSION: u64 = 1;

/// The prompt cuts this build knows how to reproduce, sorted the way the
/// reference sorts them for its message.
///
/// `loudkit.voice.KNOWN_ENROLMENTS`. The string was read and stored and never
/// checked, so a profile cut by a build that prepares its prompt some other
/// way loaded here and spoke in a different voice under the same name, which
/// is the sentence the reference refuses it with.
const KNOWN_ENROLMENTS: [&str; 2] = ["first-10s", "first-10s-pause"];

/// Longest voice name kept, as `loudkit.voice._MAX_NAME_CHARS`. Characters,
/// not bytes: the reference slices a `str`.
const MAX_NAME_CHARS: usize = 200;

/// The constant fed to the generator's emotion conditioning slot.
///
/// The checkpoint reserves one of its 34 conditioning slots for an emotion
/// scalar. On these weights the axis is dead (distillation collapsed it), so
/// the slot is not a control and not part of the profile format, but it must
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
/// a finite, but arbitrary, direction. Enrolled vectors are order-1; anything
/// this small is a corrupt or synthetic file, not a quiet voice.
const MIN_EMBEDDING_NORM: f32 = 1e-6;

/// A tensor shape spelled the way a numpy shape prints, so a refusal here
/// reads word for word like the reference's: `(80, 7)`, `(80,)`, `()`.
fn shape_repr(shape: &[i64]) -> String {
    let body: Vec<String> = shape.iter().map(i64::to_string).collect();
    if body.len() == 1 {
        return format!("({},)", body[0]);
    }
    format!("({})", body.join(", "))
}

/// A string spelled the way Python's `repr` spells it, for the refusals that
/// quote a value back.
///
/// Single quotes unless the value contains one and no double quote, which is
/// the whole of `repr`'s rule for the strings that reach here.
fn py_repr_str(s: &str) -> String {
    if s.contains('\'') && !s.contains('"') {
        return format!("\"{s}\"");
    }
    format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'"))
}

/// A JSON value spelled the way the reference's refusals quote it back, so the
/// same bad header reads the same in both.
fn py_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => py_repr_str(s),
        Value::Array(items) => {
            let body: Vec<String> = items.iter().map(py_repr).collect();
            format!("[{}]", body.join(", "))
        }
        Value::Object(map) => {
            let body: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{}: {}", py_repr_str(k), py_repr(v)))
                .collect();
            format!("{{{}}}", body.join(", "))
        }
    }
}

/// What a decoded JSON value is, in the words a header is written in, as a
/// phrase that reads after "is": "an array", "a string", "null".
fn json_type_phrase(value: &Value) -> &'static str {
    match value {
        Value::Object(_) => "an object",
        Value::Array(_) => "an array",
        Value::String(_) => "a string",
        Value::Bool(_) => "a boolean",
        Value::Null => "null",
        Value::Number(_) => "a number",
    }
}

/// The dimensions of an embedding, as `VoiceProfile._validate_shapes` reads
/// them.
///
/// Split from the value checks below because the reference runs every shape
/// check before any value check, and a profile with two faults has to name the
/// same one here as there.
///
/// The declared shape, not only the value count: this reader flattens every
/// tensor, so a `[2, 128]` speaker vector arrived as 256 floats and passed a
/// check that was reading the count.
fn check_embedding_shape(name: &str, shape: &[i64], expected: usize) -> Result<(), String> {
    if shape.len() != 1 {
        return Err(format!(
            "{name} must be 1-D, got shape {}",
            shape_repr(shape)
        ));
    }
    if shape[0] != expected as i64 {
        return Err(format!("{name} must be {expected}-d, got {}", shape[0]));
    }
    Ok(())
}

/// Reject an embedding the renderers would disagree about.
///
/// A profile is a file that gets copied, mailed and downloaded, so this
/// belongs at the boundary rather than in each backend, and every port
/// validates here. Accepting anything shaped like floats moves the failure
/// deeper into inference, where the error names a matrix rather than a file.
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
/// the shipped model's: prompt tokens index the speech codebook below the
/// start-of-speech marker, conditioning tokens the whole speech vocabulary.
const START_SPEECH_TOKEN: i64 = 6561;
const SPEECH_VOCAB_SIZE: i64 = 8194;

/// Matches Python's `MAX_VOICE_BYTES`, which the other four readers never had.
///
/// A voice profile is a handful of small tensors, and a safetensors file
/// claiming otherwise is not one. The cap is on the file, before it is opened,
/// because the shape checks that follow only run once a header has been parsed.
pub const MAX_VOICE_BYTES: u64 = 8 * 1024 * 1024;

impl Profile {
    /// Write the profile as safetensors, with the header
    /// `loudkit.voice.VoiceProfile.save` writes, so every implementation reads
    /// it back. Owner-only permissions: a profile derives from a recording of
    /// a person.
    ///
    /// # Errors
    ///
    /// For a mel that is not `(80, frames)`, and everything the file write
    /// returns.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), String> {
        if !self.prompt_mel.len().is_multiple_of(MEL_BINS) {
            return Err(format!(
                "prompt_mel must be ({MEL_BINS}, frames), got {} values",
                self.prompt_mel.len()
            ));
        }
        let header = serde_json::json!({
            "format_version": FORMAT_VERSION,
            "name": self.name,
            "source_sample_rate": self.source_sample_rate,
            "language": self.language,
            "enrolment": if self.enrolment.is_empty() { "first-10s" } else { &self.enrolment },
        })
        .to_string();
        let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let i64s = |v: &[i64]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let entries = vec![
            safetensors::Entry {
                name: "speaker_embedding".into(),
                dtype: "F32".into(),
                shape: vec![self.speaker_embedding.len() as i64],
                data: f32s(&self.speaker_embedding),
            },
            safetensors::Entry {
                name: "flow_embedding".into(),
                dtype: "F32".into(),
                shape: vec![self.flow_embedding.len() as i64],
                data: f32s(&self.flow_embedding),
            },
            safetensors::Entry {
                name: "prompt_tokens".into(),
                dtype: "I64".into(),
                shape: vec![self.prompt_tokens.len() as i64],
                data: i64s(&self.prompt_tokens),
            },
            safetensors::Entry {
                name: "prompt_mel".into(),
                dtype: "F32".into(),
                shape: vec![MEL_BINS as i64, (self.prompt_mel.len() / MEL_BINS) as i64],
                data: f32s(&self.prompt_mel),
            },
            safetensors::Entry {
                name: "cond_prompt_tokens".into(),
                dtype: "I64".into(),
                shape: vec![self.cond_prompt_tokens.len() as i64],
                data: i64s(&self.cond_prompt_tokens),
            },
        ];
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("voice".to_string(), header);
        safetensors::write(path.as_ref(), &entries, &metadata)
    }
}

pub fn load(path: &str) -> Result<Profile, String> {
    // The file, not the path it was found at: the reference and the other
    // three ports all put the file's own name in front of a voice refusal.
    let file = std::path::Path::new(path)
        .file_name()
        .map_or_else(|| path.to_string(), |s| s.to_string_lossy().into_owned());
    if let Ok(meta) = std::fs::metadata(path) {
        if meta.len() > MAX_VOICE_BYTES {
            return Err(format!(
                "{file}: {} bytes, over the {MAX_VOICE_BYTES} byte limit for a voice",
                meta.len()
            ));
        }
    }
    let f = safetensors::File::open(path)?;
    // The parse failure is named where it happens. Folded to `Value::Null` it
    // reached the version check as 0, so an unreadable header was reported as
    // a file too old for this build, which points the caller at the wrong
    // fault.
    let raw: Value = match f.metadata.get("voice") {
        Some(s) => serde_json::from_str(s)
            .map_err(|e| format!("{file}: voice header is not readable JSON: {e}"))?,
        None => Value::Object(serde_json::Map::new()),
    };
    // An object, so that a key that is absent and a key holding `null` are two
    // different answers: indexing a `Value` folds them together, and the second
    // is a value the field's reader has to refuse.
    let header = match raw {
        Value::Object(map) => map,
        other => {
            return Err(format!(
                "{file}: voice header is {}, expected a JSON object",
                json_type_phrase(&other)
            ))
        }
    };
    let version = header_int(&header, &file, "format_version", 0)?;
    if version != FORMAT_VERSION as i64 {
        return Err(format!(
            "{file}: voice format version {version}, this build reads {FORMAT_VERSION}"
        ));
    }
    let speaker_embedding = f.f32("speaker_embedding")?;
    let flow_embedding = f.f32("flow_embedding")?;
    let prompt_tokens = f.i64("prompt_tokens")?;
    let prompt_mel = f.f32("prompt_mel")?;
    let cond_prompt_tokens = f.i64("cond_prompt_tokens")?;

    // The shapes the file declares, which the flat accessors above drop.
    let shape = |name: &str| -> Vec<i64> {
        f.tensors
            .get(name)
            .map(|t| t.shape.clone())
            .unwrap_or_default()
    };
    // Every shape first, then every value, then the enrolment law: the order
    // `VoiceProfile.__post_init__` runs its three passes in, so a profile with
    // two faults names the same one in both.
    check_embedding_shape(
        "speaker_embedding",
        &shape("speaker_embedding"),
        SPEAKER_DIM,
    )?;
    check_embedding_shape("flow_embedding", &shape("flow_embedding"), FLOW_DIM)?;
    // `(80, frames)`, read from the declared shape rather than inferred from a
    // value count. `len % 80 == 0` is true of a `[2, 80]` mel, which is the
    // same 160 floats transposed: this reader loaded it and the renderer then
    // read frames as bins. Go and JS check the same way and have the same
    // hole; the reference and Swift read `ndim` and `shape[0]`.
    let mel_shape = shape("prompt_mel");
    if mel_shape.len() != 2 || mel_shape[0] != MEL_BINS as i64 {
        return Err(format!(
            "prompt_mel must be ({MEL_BINS}, frames), got {}",
            shape_repr(&mel_shape)
        ));
    }
    if shape("prompt_tokens").len() != 1 || shape("cond_prompt_tokens").len() != 1 {
        return Err("token prompts must be 1-D".to_string());
    }

    check_embedding("speaker_embedding", &speaker_embedding, SPEAKER_DIM)?;
    check_embedding("flow_embedding", &flow_embedding, FLOW_DIM)?;
    if prompt_mel.iter().any(|v| !v.is_finite()) {
        return Err("prompt_mel contains NaN or infinity".to_string());
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
        // Negative ids index an embedding table from the end: silently.
        if let Some(bad) = tokens.iter().find(|t| **t < 0) {
            return Err(format!("{name} contains a negative id: {bad}"));
        }
    }

    // A rate is provenance, but a wrong one is still a wrong claim about the
    // recording, and `as_u64` is `None` for a negative number: `-48000` was
    // silently rewritten to the 24000 default, so a profile that lied about
    // its source came back saying something else. The reference and Go both
    // refuse it.
    let source_sample_rate = header_sample_rate(&header, &file)?;
    let name = header_name(&header, &file)?;
    let enrolment = header_str(&header, &file, "enrolment", "first-10s")?;
    if !KNOWN_ENROLMENTS.contains(&enrolment.as_str()) {
        return Err(format!(
            "{}: enrolment strategy {} is not one this build implements ({}). \
             The profile was made by a build that cuts its prompt differently, so loading \
             it here would speak in a different voice under the same name.",
            if name.is_empty() { "voice" } else { &name },
            py_repr_str(&enrolment),
            KNOWN_ENROLMENTS.join(", ")
        ));
    }
    let language = header_str(&header, &file, "language", "en")?;
    Ok(Profile {
        enrolment,
        name,
        speaker_embedding,
        flow_embedding,
        prompt_tokens,
        prompt_mel,
        cond_prompt_tokens,
        source_sample_rate,
        language,
    })
}

/// A header, as `serde_json` hands one back after the object check.
type Header = serde_json::Map<String, Value>;

/// A whole JSON number under `key`, or `default` when the key is absent.
///
/// # Errors
///
/// Returns the reference's refusal when the key holds anything that is not a
/// whole number. `true` is one of those: it read as version one here, so a
/// header saying `format_version: true` loaded as a version-1 voice.
fn header_int(header: &Header, file: &str, key: &str, default: i64) -> Result<i64, String> {
    let Some(raw) = header.get(key) else {
        return Ok(default);
    };
    let number = raw.as_f64().filter(|_| !raw.is_boolean()).ok_or_else(|| {
        format!(
            "{file}: voice header '{key}' must be a number, got {}",
            py_repr(raw)
        )
    })?;
    if number.fract() != 0.0 {
        return Err(format!(
            "{file}: voice header '{key}' must be a whole number, got {}",
            py_repr(raw)
        ));
    }
    Ok(raw.as_i64().unwrap_or(number as i64))
}

/// A JSON string under `key`, or `default` when the key is absent.
///
/// # Errors
///
/// Returns the reference's refusal when the key holds anything that is not a
/// string. Not a fallback to `default`: `language: 5` took English here, and
/// language selects the text funnel, so a voice whose header says 5 was spoken
/// by whichever funnel the fallback happened to land on.
fn header_str(header: &Header, file: &str, key: &str, default: &str) -> Result<String, String> {
    match header.get(key) {
        None => Ok(default.to_string()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(raw) => Err(format!(
            "{file}: voice header '{key}' must be a string, got {}",
            py_repr(raw)
        )),
    }
}

/// The recording's own sample rate, or the 24 kHz a header that names none
/// stands for.
///
/// # Errors
///
/// Returns the reference's refusal when the header names a rate that is not a
/// whole number, or one that is not positive.
fn header_sample_rate(header: &Header, file: &str) -> Result<usize, String> {
    let declared = header_int(header, file, "source_sample_rate", 24_000)?;
    if declared <= 0 {
        return Err(format!("source_sample_rate must be positive: {declared}"));
    }
    Ok(declared as usize)
}

/// The profile's name: the header's, cut to [`MAX_NAME_CHARS`], or the file's
/// own stem when the header does not say.
///
/// The stem and not the word "voice": the reference names an unnamed profile
/// after the file it came out of, so `alice.safetensors` is "alice" there and
/// was "voice" here, in the string a caller sees and a server echoes back.
///
/// # Errors
///
/// Returns the reference's refusal when the header's name is not a string.
fn header_name(header: &Header, file: &str) -> Result<String, String> {
    let stem = std::path::Path::new(file)
        .file_stem()
        .map_or_else(String::new, |s| s.to_string_lossy().into_owned());
    let declared = header_str(header, file, "name", &stem)?;
    Ok(declared.chars().take(MAX_NAME_CHARS).collect())
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
    /// take, since Python writes the key but files written by anything
    /// older do not, and it is what stops the engine's language chain from
    /// retrofitting them.
    #[test]
    fn a_header_without_a_language_reads_as_english() {
        let language = |h: Value| header_str(h.as_object().unwrap(), "v", "language", "en");
        assert_eq!(language(serde_json::json!({"name": "x"})), Ok("en".into()));
        assert_eq!(language(serde_json::json!({})), Ok("en".into()));
        assert_eq!(
            language(serde_json::json!({"language": "pl"})),
            Ok("pl".into())
        );
        // An explicitly empty value is preserved rather than defaulted: it is
        // the one input the engine's fallback branch exists for, and turning it
        // into "en" here would make that branch unreachable.
        assert_eq!(language(serde_json::json!({"language": ""})), Ok("".into()));
    }

    /// A header value that is present but not the field's JSON type is refused
    /// by name, not coerced.
    ///
    /// `language: 5` took the English default here, and language selects the
    /// text funnel, so a voice whose header says 5 was spoken by whichever
    /// funnel the fallback happened to land on. `_header_str` and `_header_int`
    /// refuse the five fields in these sentences.
    #[test]
    fn a_header_value_of_the_wrong_type_is_refused() {
        let text = |v: Value| {
            header_str(
                serde_json::json!({"language": v}).as_object().unwrap(),
                "v.safetensors",
                "language",
                "en",
            )
        };
        for (value, want) in [
            (serde_json::json!(5), "got 5"),
            (serde_json::json!(null), "got None"),
            (serde_json::json!(true), "got True"),
            (serde_json::json!(false), "got False"),
            (serde_json::json!(5.0), "got 5.0"),
            (serde_json::json!(["en"]), "got ['en']"),
            (serde_json::json!({"id": "en"}), "got {'id': 'en'}"),
        ] {
            assert_eq!(
                text(value),
                Err(format!(
                    "v.safetensors: voice header 'language' must be a string, {want}"
                ))
            );
        }

        let number = |v: Value| {
            header_int(
                serde_json::json!({"format_version": v})
                    .as_object()
                    .unwrap(),
                "v.safetensors",
                "format_version",
                0,
            )
        };
        // `true` is not a number, and it read as version one: a header saying
        // `format_version: true` loaded as a version-1 voice.
        for (value, want) in [
            (serde_json::json!(true), "must be a number, got True"),
            (serde_json::json!("1"), "must be a number, got '1'"),
            (serde_json::json!(null), "must be a number, got None"),
            (serde_json::json!(1.9), "must be a whole number, got 1.9"),
        ] {
            assert_eq!(
                number(value),
                Err(format!(
                    "v.safetensors: voice header 'format_version' {want}"
                ))
            );
        }
        assert_eq!(number(serde_json::json!(1)), Ok(1));
        assert_eq!(number(serde_json::json!(1.0)), Ok(1));
    }

    /// An unreadable header says so, rather than quoting a version.
    ///
    /// Folded to `Value::Null`, a header that is not JSON reaches the version
    /// check as 0 and comes back as "voice format version 0, this build reads
    /// 1", which points the caller at a file too old instead of a file that is
    /// damaged.
    #[test]
    fn an_unreadable_header_is_not_reported_as_an_old_version() {
        let dir = std::env::temp_dir().join(format!("loudkit-voice-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("broken.safetensors");
        let header = serde_json::json!({"__metadata__": {"voice": "{not json"}}).to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        std::fs::write(&path, bytes).expect("write");

        let err = load(&path.to_string_lossy())
            .err()
            .expect("an unreadable header must be refused");
        assert!(err.contains("not readable JSON"), "{err}");
        assert!(
            !err.contains("format version 0"),
            "the message must not blame the version: {err}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A rate the header declares is read, and a rate that cannot be one is
    /// refused rather than replaced.
    ///
    /// `as_u64` is `None` for a negative number, so `-48000` fell to the
    /// 24000 default: a profile that lied about its source came back saying
    /// something else, in the field the provenance chain reads.
    #[test]
    fn a_declared_sample_rate_is_read_or_refused_never_replaced() {
        let rate = |v: Value| {
            header_sample_rate(
                serde_json::json!({"source_sample_rate": v})
                    .as_object()
                    .unwrap(),
                "v.safetensors",
            )
        };
        assert_eq!(rate(serde_json::json!(48_000)), Ok(48_000));
        assert_eq!(rate(serde_json::json!(16_000)), Ok(16_000));
        // A whole number however it is written, and nothing else: a rate with
        // a fraction divides every duration the profile reports by a rate no
        // reader would agree on.
        assert_eq!(rate(serde_json::json!(24_000.0)), Ok(24_000));
        assert_eq!(
            rate(serde_json::json!(24_000.9)),
            Err(
                "v.safetensors: voice header 'source_sample_rate' must be a whole number, \
                 got 24000.9"
                    .to_string()
            )
        );
        assert_eq!(
            rate(serde_json::json!(-48_000)),
            Err("source_sample_rate must be positive: -48000".to_string())
        );
        assert_eq!(
            rate(serde_json::json!(0)),
            Err("source_sample_rate must be positive: 0".to_string())
        );
        // A header that names no rate stands for 24 kHz, as it always has.
        assert_eq!(
            header_sample_rate(serde_json::json!({}).as_object().unwrap(), "v.safetensors"),
            Ok(24_000)
        );
    }

    /// A prompt mel is `(80, frames)` by its declared shape, not by its value
    /// count.
    ///
    /// `len % 80 == 0` is true of a `[2, 80]` mel, which is the same 160
    /// floats transposed, so the file loaded and the renderer read frames as
    /// bins. Go and JS check the same way and have the same hole.
    #[test]
    fn a_transposed_prompt_mel_is_refused() {
        let dir = std::env::temp_dir().join(format!("loudkit-voice-mel-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        for (shape, want) in [
            (
                vec![2i64, 80],
                "prompt_mel must be (80, frames), got (2, 80)",
            ),
            (vec![80], "prompt_mel must be (80, frames), got (80,)"),
            (
                vec![1, 80, 4],
                "prompt_mel must be (80, frames), got (1, 80, 4)",
            ),
            (
                vec![160, 2],
                "prompt_mel must be (80, frames), got (160, 2)",
            ),
        ] {
            let path = dir.join(format!("mel-{}.safetensors", shape.len() * 100 + 1));
            write_probe(&path, &shape);
            let err = load(&path.to_string_lossy()).err().unwrap_or_else(|| {
                panic!("a prompt_mel of shape {shape:?} must be refused");
            });
            assert_eq!(err, want, "{shape:?}");
        }
        // The shape the enroller writes still loads.
        let path = dir.join("mel-ok.safetensors");
        write_probe(&path, &[MEL_BINS as i64, 2]);
        let profile = load(&path.to_string_lossy()).expect("(80, frames) must load");
        assert_eq!(profile.prompt_mel.len(), MEL_BINS * 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A prompt this build cannot reproduce is refused, with the reference's
    /// own sentence.
    ///
    /// The law was read and stored and never checked, so a profile cut by a
    /// build that prepares its prompt some other way loaded and spoke in a
    /// different voice under the same name.
    #[test]
    fn an_unknown_enrolment_law_is_refused() {
        let dir = std::env::temp_dir().join(format!("loudkit-voice-law-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        for law in ["first-10s", "first-10s-pause"] {
            let path = dir.join(format!("law-{law}.safetensors"));
            write_probe_with(&path, &[MEL_BINS as i64, 2], law, "probe");
            let profile = load(&path.to_string_lossy())
                .unwrap_or_else(|e| panic!("{law} is a law this build implements: {e}"));
            assert_eq!(profile.enrolment, law);
        }
        let path = dir.join("law-bad.safetensors");
        write_probe_with(&path, &[MEL_BINS as i64, 2], "first-30s-nonsense", "probe");
        let err = load(&path.to_string_lossy())
            .err()
            .expect("an unknown law must be refused");
        assert_eq!(
            err,
            "probe: enrolment strategy 'first-30s-nonsense' is not one this build implements \
             (first-10s, first-10s-pause). The profile was made by a build that cuts its \
             prompt differently, so loading it here would speak in a different voice under \
             the same name."
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The name is the file's own stem when the header does not carry one, and
    /// is cut where the reference cuts it.
    #[test]
    fn an_unnamed_profile_is_named_after_its_file() {
        let name = |h: Value, file: &str| header_name(h.as_object().unwrap(), file);
        assert_eq!(
            name(serde_json::json!({}), "alice.safetensors"),
            Ok("alice".into())
        );
        assert_eq!(
            name(serde_json::json!({"name": "bob"}), "alice.safetensors"),
            Ok("bob".into())
        );
        // Characters, not bytes: the reference slices a `str`.
        let long = "\u{17c}".repeat(300);
        assert_eq!(
            name(serde_json::json!({"name": long}), "x.safetensors")
                .unwrap()
                .chars()
                .count(),
            MAX_NAME_CHARS
        );
    }

    /// A profile probe: valid everywhere except the `prompt_mel` shape.
    fn write_probe(path: &std::path::Path, mel_shape: &[i64]) {
        write_probe_with(path, mel_shape, "first-10s", "probe");
    }

    fn write_probe_with(path: &std::path::Path, mel_shape: &[i64], law: &str, name: &str) {
        let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let i64s = |v: &[i64]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let count: i64 = mel_shape.iter().product();
        let entries = vec![
            safetensors::Entry {
                name: "speaker_embedding".into(),
                dtype: "F32".into(),
                shape: vec![SPEAKER_DIM as i64],
                data: f32s(&vec![0.0625; SPEAKER_DIM]),
            },
            safetensors::Entry {
                name: "flow_embedding".into(),
                dtype: "F32".into(),
                shape: vec![FLOW_DIM as i64],
                data: f32s(&vec![0.0625; FLOW_DIM]),
            },
            safetensors::Entry {
                name: "prompt_mel".into(),
                dtype: "F32".into(),
                shape: mel_shape.to_vec(),
                data: f32s(&vec![0.5; count.max(0) as usize]),
            },
            safetensors::Entry {
                name: "prompt_tokens".into(),
                dtype: "I64".into(),
                shape: vec![2],
                data: i64s(&[1, 2]),
            },
            safetensors::Entry {
                name: "cond_prompt_tokens".into(),
                dtype: "I64".into(),
                shape: vec![2],
                data: i64s(&[1, 2]),
            },
        ];
        let header = serde_json::json!({
            "format_version": FORMAT_VERSION,
            "name": name,
            "language": "en",
            "source_sample_rate": 24_000,
            "enrolment": law,
        })
        .to_string();
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("voice".to_string(), header);
        safetensors::write(path, &entries, &metadata).expect("write the probe");
    }
}
