//! The packed checkpoint: manifest + the embedding tables the generator needs
//! on the host. Port of the `loudkit.checkpoint` reads.

use std::collections::HashMap;

use serde_json::Value;

use crate::chunking::ChunkConfig;
use crate::postprocess::{
    Config as PostprocessConfig, Mode as PostprocessMode, RepetitionResume, RepetitionSilence,
};
use crate::safetensors;
use crate::windowing::WindowConfig;

/// The four fp32 embedding tables the generator reads (fp16 storage upcasts
/// exactly).
///
/// Named fields rather than a four-tuple, because all four are `Vec<f32>` of
/// the same element type and the same hidden width: a caller that took them in
/// the wrong order would compile, load and render, and the only report of the
/// mistake would be the audio. A field name is checked by the compiler; a
/// tuple position is checked by the reader.
pub struct GeneratorTables {
    /// Text token embeddings, one row of `HIDDEN_DIM` per tokenizer id.
    pub text_emb: Vec<f32>,
    /// Speech token embeddings, one row per speech token id.
    pub speech_emb: Vec<f32>,
    /// Text position embeddings, added to `text_emb` by position.
    pub text_pos: Vec<f32>,
    /// Speech position embeddings, added to `speech_emb` by position.
    pub speech_pos: Vec<f32>,
}

pub struct Checkpoint {
    pub manifest: HashMap<String, Value>,
    file: safetensors::File,
}

/// Manifest versions and decoder loops implemented by this engine.
pub const SUPPORTED_FORMAT_VERSIONS: [i64; 2] = [1, 2];
pub const SUPPORTED_DECODE_MODES: [&str; 2] = ["single", "fusion_mtp2"];

impl Checkpoint {
    pub fn open(path: &str) -> Result<Self, String> {
        let file = safetensors::File::open(path)?;
        let manifest_str = file
            .metadata
            .get("manifest")
            .ok_or_else(|| format!("{path}: no embedded manifest: not a loudkit checkpoint"))?;
        let manifest: Value = serde_json::from_str(manifest_str)
            .map_err(|e| format!("{path}: bad manifest JSON: {e}"))?;
        if manifest["format"].as_str() != Some("loudkit-checkpoint") {
            return Err(format!(
                "{path}: no embedded manifest: not a loudkit checkpoint"
            ));
        }
        // `format_version` is checked, not only `format`. Python refuses a
        // version it does not read; a port that accepts any version will
        // happily load a future checkpoint whose fields mean something else,
        // the loader would still "work", and the audio would be wrong for
        // reasons no error names.
        let version = manifest["format_version"].as_i64().unwrap_or(-1);
        if !SUPPORTED_FORMAT_VERSIONS.contains(&version) {
            return Err(format!(
                "{path}: manifest format_version {version}; this build reads \
                 {SUPPORTED_FORMAT_VERSIONS:?}"
            ));
        }
        check_block_shapes(path, &manifest)?;
        check_chunking_keys(path, &manifest)?;
        let m: HashMap<String, Value> = manifest
            .as_object()
            .map(|o| o.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();
        // Every value the readers below will take from this manifest, read
        // once here through one accumulator, so a value this port cannot read
        // is refused before it becomes a default that nothing reports.
        check_readable(&m).map_err(|e| format!("{path}: {e}"))?;
        // The decode mode, for the file that understates its version. See
        // SUPPORTED_DECODE_MODES: the number alone is only as good as the
        // packer that wrote it.
        let mode = decode_mode_from(&m, &mut Numbers::default());
        if mode == "fusion_mtp2" && version < 2 {
            return Err(format!(
                "{path}: decode.mode fusion_mtp2 requires format_version 2"
            ));
        }
        if !SUPPORTED_DECODE_MODES.contains(&mode.as_str()) {
            return Err(format!(
                "{path}: manifest declares decode.mode {mode:?}; this build \
                 decodes {SUPPORTED_DECODE_MODES:?}"
            ));
        }
        Ok(Checkpoint { manifest: m, file })
    }

    /// The window recipe from the manifest.
    ///
    /// An absent block is the ragged window, matching
    /// `manifest.py::_window_from`, which returns a default `WindowConfig`.
    /// Falling back to the shipped static recipe instead framed a manifest
    /// that declares no window differently here than in Python.
    #[must_use]
    pub fn window(&self) -> WindowConfig {
        window_from(&self.manifest, &mut Numbers::default())
    }

    /// Sampling values from the manifest.
    #[must_use]
    pub fn sampling(&self) -> (f64, f64, f64, usize, Vec<usize>, usize, f64) {
        sampling_from(&self.manifest, &mut Numbers::default())
    }

    /// Speech token ids, and the vocabulary the sampler draws from.
    #[must_use]
    pub fn speech_tokens(&self) -> (usize, usize, usize) {
        speech_tokens_from(&self.manifest, &mut Numbers::default())
    }

    #[must_use]
    pub fn sample_rate(&self) -> usize {
        Numbers::default().top_count(&self.manifest, "sample_rate", 24_000)
    }

    /// Guidance strength. Zero in `single_path`, which is the shipping mode.
    #[must_use]
    pub fn guidance_rate(&self) -> f64 {
        Numbers::default().top(&self.manifest, "guidance_rate", 0.0)
    }

    /// The edge ramp in seconds.
    ///
    /// An absent key is the legacy 5 ms, not the shipped 20 ms, because a
    /// manifest packed before the key existed was measured at that length.
    /// `manifest.py::edge_fade_from` reads it the same way, and reads an
    /// explicit null as the same absence.
    #[must_use]
    pub fn edge_fade_seconds(&self) -> f64 {
        edge_fade_from(&self.manifest, &mut Numbers::default())
    }

    /// Speech tokens per second: 25 Hz for this model family.
    #[must_use]
    pub fn token_rate_hz(&self) -> f64 {
        Numbers::default().top(&self.manifest, "token_rate_hz", 25.0)
    }

    #[must_use]
    pub fn euler_steps(&self) -> usize {
        Numbers::default().top_count(&self.manifest, "n_cfm_timesteps", 2)
    }

    /// The explicit Euler time grid, or `None` for the cosine schedule.
    ///
    /// A JSON string is refused rather than iterated, matching Python's guard
    /// on this key: a manifest one port misreads while another defaults is the
    /// divergence class this library exists to prevent.
    #[must_use]
    pub fn euler_grid(&self) -> Option<Vec<f64>> {
        euler_grid_from(&self.manifest, &mut Numbers::default())
    }

    /// The recipe tag, mirroring
    /// `loudkit.config.AlgorithmConfig.from_manifest`: absent means
    /// `loudkit-1`, and anything else is refused.
    ///
    /// # Errors
    ///
    /// Returns an error naming the declared value when it is not `loudkit-1`.
    pub fn recipe_version(&self) -> Result<String, String> {
        recipe_version_from(&self.manifest)
    }

    /// The artifact detectors, read from the manifest or defaulted to the
    /// shipping constants.
    ///
    /// # Errors
    ///
    /// Returns an error for a mode this port does not implement.
    pub fn postprocess(&self) -> Result<PostprocessConfig, String> {
        let mut nums = Numbers::default();
        let cfg = postprocess_from(&self.manifest, &mut nums)?;
        nums.settle()?;
        Ok(cfg)
    }

    /// The chunking recipe: where the reader breathes, and the prefix carry.
    ///
    /// Read from the manifest rather than defaulted. A checkpoint that declares
    /// its own boundaries and a runtime that silently uses different ones agree
    /// on `recipe_version` and disagree on the reading, which is the drift the
    /// fingerprint exists to prevent.
    #[must_use]
    pub fn chunking(&self) -> ChunkConfig {
        chunking_from(&self.manifest, &mut Numbers::default())
    }

    /// The declared guidance mode, refused when this port cannot honour it.
    ///
    /// This binding calls the estimator once per Euler step and never forms
    /// `(1+w)·v_cond − w·v_uncond`, so an accepted `cfg_dual_path` checkpoint
    /// would load, produce plausible audio, and disagree with the Python
    /// engine under a matching `recipe_version`. The JS, Go, Python and CoreML
    /// paths all refuse the same mode for the same reason.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown mode, and for `cfg_dual_path`.
    pub fn guidance(&self) -> Result<String, String> {
        guidance_from(&self.manifest)
    }

    #[must_use]
    pub fn decode_mode(&self) -> String {
        decode_mode_from(&self.manifest, &mut Numbers::default())
    }

    /// The fusion MLP, when the manifest declares the two-head loop.
    ///
    /// # Errors
    ///
    /// Returns an error when a tensor is missing or the wrong size.
    pub fn fusion_weights(&self) -> Result<Option<[Vec<f32>; 4]>, String> {
        if self.decode_mode() == "single" {
            return Ok(None);
        }
        let weights = [
            self.file.f32("t3.fuse.0.weight")?,
            self.file.f32("t3.fuse.0.bias")?,
            self.file.f32("t3.fuse.2.weight")?,
            self.file.f32("t3.fuse.2.bias")?,
        ];
        if weights.iter().map(Vec::len).collect::<Vec<_>>()
            != [2 * 1024 * 1024, 1024, 1024 * 1024, 1024]
        {
            return Err("invalid fusion MLP tensor sizes".to_string());
        }
        Ok(Some(weights))
    }

    /// The fp32 embedding tables the generator uses (fp16 storage upcasts
    /// exactly).
    pub fn generator_tables(&self) -> Result<GeneratorTables, String> {
        Ok(GeneratorTables {
            text_emb: self.file.f32("t3.text_emb.weight")?,
            speech_emb: self.file.f32("t3.speech_emb.weight")?,
            text_pos: self.file.f32("t3.text_pos_emb.emb.weight")?,
            speech_pos: self.file.f32("t3.speech_pos_emb.emb.weight")?,
        })
    }

    /// The 192->80 speaker affine the flow decoder conditions on.
    pub fn speaker_affine(&self) -> Result<(Vec<f32>, Vec<f32>), String> {
        Ok((
            self.file.f32("s3gen.flow.spk_embed_affine_layer.weight")?,
            self.file.f32("s3gen.flow.spk_embed_affine_layer.bias")?,
        ))
    }
}

/// The shape of every manifest block and list the readers below consult.
///
/// `manifest.py::_block` refuses a key of the wrong type rather than
/// defaulting it, and names the string-as-list case on its own because
/// `silence_token_ids: "123"` must not load as three tokens. Every reader
/// below returns a value rather than a `Result`, so the door is the one place
/// that refusal fits: without it a mistyped key takes the fallback in silence
/// here while Python raises.
///
/// # Errors
///
/// Returns an error naming the key and the shape it must have.
fn check_block_shapes(path: &str, manifest: &Value) -> Result<(), String> {
    let Some(object) = manifest.as_object() else {
        return Ok(());
    };
    for (key, value) in object {
        // `null` is the ragged window and the cosine grid said out loud, which
        // is how `_window_from` and `algorithm_from` read them.
        let expected = match key.as_str() {
            "window" if !value.is_object() && !value.is_null() => "a mapping or null",
            // `decode: null` is the single-token loop said out loud, which is
            // how `decode_from` reads it.
            "decode" if !value.is_object() && !value.is_null() => "a mapping or null",
            "sampling_defaults" | "speech_tokens" | "eos_floor" | "chunking" | "postprocess"
                if !value.is_object() =>
            {
                "a mapping"
            }
            "silence_token_ids" | "silence_render_ids" | "quiet_render_ids"
                if !value.is_array() =>
            {
                "a list"
            }
            "euler_grid" if !value.is_array() && !value.is_null() => "a list or null",
            _ => continue,
        };
        return Err(format!(
            "{path}: manifest key {key:?} must be {expected}, got {}",
            json_kind(value)
        ));
    }
    Ok(())
}

/// Chunking keys the splitter here does not honour.
///
/// `config.ChunkConfig` carries `first_chunk_max_tokens` and cuts the first
/// chunk short by it; this port neither honours the key nor refuses it, so a
/// manifest that sets it splits differently here and in Python under one
/// `recipe_version`. Refused rather than ignored, for the reason the guidance
/// door already gives: a divergence a fingerprint cannot see is the one this
/// library exists to prevent. Honouring it moves where the first chunk is cut,
/// so that is a 0.1.2 change and this is the refusal until then.
///
/// # Errors
///
/// Returns an error naming the key.
fn check_chunking_keys(path: &str, manifest: &Value) -> Result<(), String> {
    let Some(block) = manifest.get("chunking").and_then(Value::as_object) else {
        return Ok(());
    };
    for key in UNHONOURED_CHUNKING_KEYS {
        if block.contains_key(key) {
            return Err(format!(
                "{path}: manifest['chunking'][{key:?}] is not implemented by this \
                 binding, which would split the text differently from the Python \
                 engine under a matching recipe_version"
            ));
        }
    }
    Ok(())
}

/// The chunking keys Python reads and this splitter does not.
const UNHONOURED_CHUNKING_KEYS: [&str; 1] = ["first_chunk_max_tokens"];

/// The JSON type name, for an error that says what the manifest carried.
fn json_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "a list",
        Value::Object(_) => "a mapping",
    }
}

/// One manifest block, and the path that names it in a refusal.
///
/// An absent block reads as an empty one, so every key below it takes the
/// fallback written beside it, which is what `_block` answers for a key the
/// manifest omits.
#[derive(Clone, Copy)]
struct Block<'a> {
    map: Option<&'a serde_json::Map<String, Value>>,
    name: &'a str,
}

impl<'a> Block<'a> {
    fn of(manifest: &'a HashMap<String, Value>, name: &'a str) -> Self {
        Block {
            map: manifest.get(name).and_then(Value::as_object),
            name,
        }
    }

    fn get(self, key: &str) -> Option<&'a Value> {
        self.map.and_then(|m| m.get(key))
    }

    fn path(self, key: &str) -> String {
        format!("{}['{key}']", top_path(self.name))
    }
}

/// A top-level key the way a refusal quotes it back, so the same bad file reads
/// the same here as under `manifest._where`.
fn top_path(key: &str) -> String {
    format!("manifest['{key}']")
}

/// The one numeric reading of a decoded JSON value.
///
/// `Value::as_f64` answers for a JSON number and for nothing else: a boolean, a
/// null, a list and a string have no numeric reading here. `config.go::asFloat`
/// is the same rule in Go.
///
/// The reference is looser on one input and this port does not follow it there.
/// `int("255")` is 255 in Python, so a manifest carrying a quoted length loads
/// in the reference and is refused here. Refusing is the safe side: the
/// alternative is not accepting it, it is accepting it *differently* in five
/// languages, each with its own reading of `"255 "`, `"0x10"` and `"1e3"`. A
/// refusal is one error somebody fixes in the packer that wrote it; a coercion
/// is a divergence the fingerprint cannot see.
fn as_number(value: &Value) -> Option<f64> {
    value.as_f64()
}

/// The manifest's numbers, names and flags, refusing a value it cannot read.
///
/// Answering the default for a value that has no reading hands the engine a
/// setting the manifest never declared. `static_length: "255"` ignored is a
/// ragged window under a manifest that asked for a padded one, and
/// `temperature: null` ignored is 0.8 under a manifest that asked for
/// something else. The fallback is then written into the canonical form as if
/// the manifest had declared it, so the fingerprint agrees with the misreading
/// instead of reporting it, and that is the divergence class this library
/// exists to prevent.
///
/// `algorithm_from` converts each of these with `int()`, `float()` or `str()`,
/// which refuse a null, a list and a non-numeric string.
///
/// The first refusal is remembered rather than returned, so the forty reads
/// below stay expressions and are checked once, at the door
/// [`Checkpoint::open`] holds. Reads after a refusal answer their default and
/// are discarded with the checkpoint.
#[derive(Default)]
struct Numbers {
    err: Option<String>,
}

impl Numbers {
    /// Keep the first refusal: the ones after it are consequences of reading on
    /// past it, and the first names the key somebody has to fix.
    fn refuse(&mut self, message: String) {
        if self.err.is_none() {
            self.err = Some(message);
        }
    }

    /// The accumulated refusal, or nothing to report.
    ///
    /// # Errors
    ///
    /// Returns the first value this reader could not read.
    fn settle(self) -> Result<(), String> {
        self.err.map_or(Ok(()), Err)
    }

    /// A numeric sub-key of `block`.
    ///
    /// Absent is not zero: only a missing key takes the default, so a manifest
    /// declaring `min_p: 0` (no truncation, a legal and meaningful setting)
    /// keeps its zero, and so does a deliberately disabled EOS floor.
    fn read(&mut self, block: Block<'_>, key: &str, default: f64) -> f64 {
        let Some(raw) = block.get(key) else {
            return default;
        };
        as_number(raw).unwrap_or_else(|| {
            self.refuse(format!("{} should be a number, got {raw}", block.path(key)));
            default
        })
    }

    /// A JSON number held here as a count or an id, under the two rules such a
    /// key answers to.
    ///
    /// A count of 2.7 is refused rather than truncated. A token id or a token
    /// budget given as a fraction was computed wrong, and truncating it to 2
    /// hides the arithmetic that produced it behind a chunker that breathes in
    /// a different place, under a `recipe_version` saying the five
    /// implementations agree. `manifest._int` refuses it in this sentence, and
    /// the fraction is tested before the sign because the reference tests only
    /// the fraction: -1.5 is refused for what it is in all five.
    ///
    /// A negative is refused rather than clamped: the fields these feed are
    /// `usize` and a cast would land every negative on zero, which is the
    /// silent default this reader exists to close, one type down. Python and Go
    /// carry these as signed and keep the negative for a range check further
    /// on; the refusal here names the key instead.
    ///
    /// `null_reading` is what the caller accepts besides a number, so the
    /// window's optional lengths say so and the rest do not.
    fn whole(&mut self, raw: &Value, path: &str, null_reading: &str) -> Option<usize> {
        match as_number(raw) {
            Some(v) if v.fract() != 0.0 => {
                self.refuse(format!("{path} must be a whole number, got {raw}"));
                None
            }
            Some(v) if v >= 0.0 => Some(v as usize),
            _ => {
                self.refuse(format!(
                    "{path} should be a non-negative number{null_reading}, got {raw}"
                ));
                None
            }
        }
    }

    /// A numeric sub-key held here as a count or an id. See [`Numbers::whole`].
    fn count(&mut self, block: Block<'_>, key: &str, default: usize) -> usize {
        let Some(raw) = block.get(key) else {
            return default;
        };
        self.whole(raw, &block.path(key), "").unwrap_or(default)
    }

    /// An optional length: a whole number, an explicit null for the unset
    /// reading, or a refusal for anything else.
    ///
    /// [`Numbers::read`] has no null reading because the values it reads have
    /// none: a temperature is a number or it is a mistake. A window length is a
    /// number or the ragged reading, and `_window_from` spells that with one
    /// `opt` helper that reads an absent key and an explicit null the same way.
    fn opt(&mut self, block: Block<'_>, key: &str, default: Option<usize>) -> Option<usize> {
        match block.get(key) {
            None => default,
            Some(Value::Null) => None,
            Some(raw) => match self.whole(raw, &block.path(key), " or null") {
                Some(v) => Some(v),
                None => default,
            },
        }
    }

    /// A top-level numeric key. `manifest.py::_number` refuses the same values,
    /// a boolean among them.
    fn top(&mut self, manifest: &HashMap<String, Value>, key: &str, default: f64) -> f64 {
        let Some(raw) = manifest.get(key) else {
            return default;
        };
        as_number(raw).unwrap_or_else(|| {
            self.refuse(format!("manifest[{key:?}] should be a number, got {raw}"));
            default
        })
    }

    /// A top-level numeric key held here as a count. See [`Numbers::whole`].
    fn top_count(&mut self, manifest: &HashMap<String, Value>, key: &str, default: usize) -> usize {
        let Some(raw) = manifest.get(key) else {
            return default;
        };
        self.whole(raw, &top_path(key), "").unwrap_or(default)
    }

    /// A top-level list of token ids, or the default for an absent key.
    ///
    /// The shape is settled by [`check_block_shapes`]; the elements are settled
    /// here, for the reason the whole list is: the reference builds the census
    /// with `int()` per element, and an element dropped for having no numeric
    /// reading enters neither the silence set nor the fingerprint, so the
    /// canonical form records a census the manifest never declared.
    fn ids(&mut self, manifest: &HashMap<String, Value>, key: &str) -> Vec<usize> {
        let Some(list) = manifest.get(key).and_then(Value::as_array) else {
            return Vec::new();
        };
        let path = top_path(key);
        list.iter()
            .enumerate()
            .map(|(i, raw)| self.whole(raw, &format!("{path}[{i}]"), "").unwrap_or(0))
            .collect()
    }

    /// A string-valued key of `block`, or nothing for an absent one.
    ///
    /// Every caller names a closed set and checks the answer against it; this
    /// settles only the shape, so the "unknown mode" refusals keep naming their
    /// own options. A present key skipped for having the wrong type would be a
    /// law the manifest declared and this port did not run. The reference
    /// arrives at the same refusal from the other side: `str()` renders the
    /// value and the membership check fails on what it rendered.
    ///
    /// A refused key answers nothing rather than a default, so the caller
    /// leaves its own value in place and this reader's message is the one that
    /// reaches the caller, not a second complaint about the empty string it
    /// would otherwise have parsed.
    fn text(&mut self, block: Block<'_>, key: &str) -> Option<String> {
        let raw = block.get(key)?;
        raw.as_str().map_or_else(
            || {
                self.refuse(format!("{} must be a string, got {raw}", block.path(key)));
                None
            },
            |s| Some(s.to_string()),
        )
    }

    /// A top-level key that has to be a name, checked for shape only.
    ///
    /// The closed-set refusals stay with the readers that name their own
    /// options; this is the door noticing that the value is not a name at all,
    /// which the reference notices from the other side: `str()` renders it and
    /// the membership check fails on what it rendered.
    fn top_text(&mut self, manifest: &HashMap<String, Value>, key: &str) {
        if let Some(raw) = manifest.get(key) {
            if !raw.is_string() {
                self.refuse(format!("manifest[{key:?}] must be a string, got {raw}"));
            }
        }
    }

    /// A boolean sub-key of `block`.
    ///
    /// Not ignored when it is not a boolean: a manifest that turns the splitter
    /// off and a runtime that leaves it on read the same `recipe_version` and
    /// deliver different audio, and `enabled` is the one key whose misreading
    /// moves every join at once. The reference coerces with `bool()`, which
    /// reads `0` as off and `"no"` as on; a refusal is the safe side of a
    /// coercion no two languages spell alike.
    fn flag(&mut self, block: Block<'_>, key: &str, default: bool) -> bool {
        let Some(raw) = block.get(key) else {
            return default;
        };
        raw.as_bool().unwrap_or_else(|| {
            self.refuse(format!(
                "{} must be JSON true or false, got {raw}",
                block.path(key)
            ));
            default
        })
    }

    /// A list of strings under `block`, or nothing for an absent key.
    ///
    /// An element dropped for having the wrong type is a separator or an
    /// abbreviation the manifest declared and the splitter never saw, so the
    /// text breaks somewhere else and the canonical form records the shorter
    /// list as if it were what the manifest said. The reference renders each
    /// element with `str()`, a coercion this port does not copy, for the reason
    /// [`as_number`] gives.
    fn strings(&mut self, block: Block<'_>, key: &str) -> Option<Vec<String>> {
        let raw = block.get(key)?;
        let Some(list) = raw.as_array() else {
            self.refuse(format!(
                "{} must be a list of strings, got {raw}",
                block.path(key)
            ));
            return None;
        };
        let mut out = Vec::with_capacity(list.len());
        for (i, item) in list.iter().enumerate() {
            let Some(s) = item.as_str() else {
                self.refuse(format!(
                    "{}[{i}] must be a string, got {item}",
                    block.path(key)
                ));
                return None;
            };
            out.push(s.to_string());
        }
        Some(out)
    }
}

/// The window recipe a manifest declares, or the ragged window it does not.
///
/// A `static_length` with no reading, ignored, pads nothing under a manifest
/// that asked for a padded window, and the canonical form then records the
/// ragged window it fell back to. Only the three lengths have a null reading,
/// and it is the ragged one: `_window_from` reads them through an `opt` that
/// spells an absent key and an explicit null the same way, and reads
/// `max_speech_tokens` with a bare `int()` that refuses a null like any other
/// non-number.
fn window_from(manifest: &HashMap<String, Value>, nums: &mut Numbers) -> WindowConfig {
    let mut w = WindowConfig::default();
    let win = Block::of(manifest, "window");
    w.max_speech_tokens = nums.count(win, "max_speech_tokens", w.max_speech_tokens);
    w.static_length = nums.opt(win, "static_length", w.static_length);
    w.static_prompt_tokens = nums.opt(win, "static_prompt_tokens", w.static_prompt_tokens);
    w.pad_token_id = nums.opt(win, "pad_token_id", w.pad_token_id);
    w
}

/// Sampling values from the manifest.
///
/// Zero and zero are `manifest.py::algorithm_from`'s defaults for the two EOS
/// floor keys; 10 and 1.2 are the shipped manifest's declared values, which is
/// not the same thing.
fn sampling_from(
    manifest: &HashMap<String, Value>,
    nums: &mut Numbers,
) -> (f64, f64, f64, usize, Vec<usize>, usize, f64) {
    let sd = Block::of(manifest, "sampling_defaults");
    let eos = Block::of(manifest, "eos_floor");
    (
        nums.read(sd, "temperature", 0.8),
        nums.read(sd, "repetition_penalty", 1.2),
        nums.read(sd, "min_p", 0.05),
        nums.count(sd, "max_new_tokens", 255),
        nums.ids(manifest, "silence_token_ids"),
        nums.count(eos, "min_tokens_floor", 0),
        nums.read(eos, "min_tokens_text_ratio", 0.0),
    )
}

/// The speech token ids, and the vocabulary the sampler draws from.
fn speech_tokens_from(
    manifest: &HashMap<String, Value>,
    nums: &mut Numbers,
) -> (usize, usize, usize) {
    let sp = Block::of(manifest, "speech_tokens");
    (
        nums.count(sp, "start", 6561),
        nums.count(sp, "stop", 6562),
        // `speech_vocab_size` 0 sizes the sampler's seen-token slice to nothing
        // and the first penalised step indexes past it, which is why this
        // default is `algorithm_from`'s and not zero.
        nums.top_count(manifest, "speech_vocab_size", 8194),
    )
}

/// The edge ramp in seconds.
///
/// An explicit null is the one place absence is spelled out: a manifest older
/// than the field and one that writes `edge_fade_seconds: null` both mean the
/// 5 ms those releases shipped. Anything else with no numeric reading is
/// refused, as `edge_fade_from` refuses it.
fn edge_fade_from(manifest: &HashMap<String, Value>, nums: &mut Numbers) -> f64 {
    match manifest.get("edge_fade_seconds") {
        None | Some(Value::Null) => 0.005,
        Some(_) => nums.top(manifest, "edge_fade_seconds", 0.005),
    }
}

/// The explicit Euler time grid, or `None` for the cosine schedule.
///
/// Absent and null are the cosine schedule. A point with no numeric reading is
/// refused rather than dropped: a shorter grid integrates on a schedule the
/// manifest never declared, and the canonical form records the shorter one.
fn euler_grid_from(manifest: &HashMap<String, Value>, nums: &mut Numbers) -> Option<Vec<f64>> {
    let grid = manifest.get("euler_grid")?.as_array()?;
    Some(
        grid.iter()
            .enumerate()
            .map(|(i, raw)| {
                as_number(raw).unwrap_or_else(|| {
                    nums.refuse(format!(
                        "manifest[\"euler_grid\"][{i}] should be a number, got {raw}"
                    ));
                    0.0
                })
            })
            .collect(),
    )
}

/// The decode loop the manifest declares.
///
/// `decode_from` renders the mode with `str()` and checks the membership on
/// what it rendered, so a mode that is not a string is refused there too.
fn decode_mode_from(manifest: &HashMap<String, Value>, nums: &mut Numbers) -> String {
    nums.text(Block::of(manifest, "decode"), "mode")
        .unwrap_or_else(|| "single".to_string())
}

/// The chunking recipe: where the reader breathes, and the prefix carry.
fn chunking_from(manifest: &HashMap<String, Value>, nums: &mut Numbers) -> ChunkConfig {
    let mut cfg = ChunkConfig::default();
    let block = Block::of(manifest, "chunking");
    cfg.enabled = nums.flag(block, "enabled", cfg.enabled);
    cfg.max_tokens = nums.count(block, "max_tokens", cfg.max_tokens);
    cfg.prefix_tokens = nums.count(block, "prefix_tokens", cfg.prefix_tokens);
    // Presence, not length. An absent key takes the default; an explicitly
    // empty list is a value the manifest states, and a length check reads the
    // two the same way. `ChunkConfig::validate` carries the reference's
    // refusal for an empty set, and only a presence check keeps that refusal
    // reachable. A `split_on` this port cannot read is a refusal `nums`
    // carries to the door, not a silent default.
    if let Some(seps) = nums.strings(block, "split_on") {
        cfg.split_on = seps;
    }
    // Presence, not length, for a second reason: an empty list is meaningful
    // here, being the old law spelled as data, so it must not fall back to
    // the shipping set.
    if let Some(abbreviations) = nums.strings(block, "abbreviations") {
        cfg.abbreviations = abbreviations;
    }
    // Set, not checked: `ChunkConfig::validate` refuses an unknown spelling,
    // and it is the one place this port states that refusal.
    if let Some(v) = nums.text(block, "mid_sentence_period") {
        cfg.mid_sentence_period = v;
    }
    if let Some(v) = nums.text(block, "cap_resplit") {
        cfg.cap_resplit = v;
    }
    cfg
}

/// Every value the readers above take from the manifest, read once through one
/// accumulator.
///
/// The door is the one place this refusal fits: the readers answer a value
/// rather than a `Result`, so without it a value this port cannot read takes
/// the fallback in silence here while Python raises. Reading through the
/// readers themselves rather than a hand-written list of keys is deliberate: a
/// key added to a reader is covered here without being named here, which is the
/// mistake `postprocess_from` made twice.
///
/// # Errors
///
/// Returns the first value the manifest carries that this port cannot read.
fn check_readable(manifest: &HashMap<String, Value>) -> Result<(), String> {
    let mut nums = Numbers::default();
    let _ = window_from(manifest, &mut nums);
    let _ = sampling_from(manifest, &mut nums);
    let _ = speech_tokens_from(manifest, &mut nums);
    let _ = chunking_from(manifest, &mut nums);
    let _ = euler_grid_from(manifest, &mut nums);
    let _ = edge_fade_from(manifest, &mut nums);
    let _ = decode_mode_from(manifest, &mut nums);
    let _ = nums.top(manifest, "guidance_rate", 0.0);
    let _ = nums.top(manifest, "token_rate_hz", 25.0);
    let _ = nums.top_count(manifest, "sample_rate", 24_000);
    let _ = nums.top_count(manifest, "n_cfm_timesteps", 2);
    nums.top_text(manifest, "guidance");
    nums.top_text(manifest, "recipe_version");
    postprocess_from(manifest, &mut nums)?;
    nums.settle()
}

fn guidance_from(manifest: &HashMap<String, Value>) -> Result<String, String> {
    // A mode that is not a string is refused rather than defaulted: the
    // reference renders it with `str()` and fails the membership check on what
    // it rendered, so `guidance: 5` is a refusal there and must not be
    // single_path here under a fingerprint that says the manifest asked for it.
    let declared = match manifest.get("guidance") {
        None => "single_path".to_string(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => {
            return Err(format!(
                "manifest[\"guidance\"] must be a string, got {other}"
            ))
        }
    };
    match declared.as_str() {
        "single_path" => Ok(declared),
        "cfg_dual_path" => Err("manifest declares guidance mode cfg_dual_path, which this \
             binding does not implement: it would render single-path audio and silently \
             disagree with the Python engine"
            .to_string()),
        other => Err(format!(
            "manifest declares unknown guidance mode {other:?}; \
             expected single_path or cfg_dual_path"
        )),
    }
}

fn recipe_version_from(manifest: &HashMap<String, Value>) -> Result<String, String> {
    // One recipe means one accepted value. A foreign tag believed here would
    // ride into every fingerprint this engine reports; a foreign tag rewritten
    // to `loudkit-1` claims this recipe for a checkpoint that named another.
    // Absence is not a tag, it is the shipping default left unstated, and
    // all five ports read it the same way.
    match manifest.get("recipe_version") {
        None => Ok("loudkit-1".to_string()),
        Some(Value::String(s)) if s == "loudkit-1" => Ok(s.clone()),
        Some(other) => Err(format!(
            "manifest declares recipe_version {other}; the only recipe is \"loudkit-1\""
        )),
    }
}

/// The artifact detectors declared by the manifest, or the shipping defaults.
///
/// # Errors
///
/// Returns an error for an unknown mode. A mode this port does not implement
/// must not fall back to a default: it would trim where the manifest said not
/// to, under a matching `recipe_version`.
fn postprocess_from(
    manifest: &HashMap<String, Value>,
    nums: &mut Numbers,
) -> Result<PostprocessConfig, String> {
    // The render censuses: which ids actually render as digital silence
    // (`silence_render_ids`) and which as contextually-quiet breath/decay
    // (`quiet_render_ids`). Optional: a checkpoint packed before the census
    // has neither, and the stall detector then falls back to
    // `silence_token_ids`, degraded but safe. Read from the manifest top
    // level, before the block check, because they are properties of the
    // weights, like `silence_token_ids`, not detector constants someone
    // tuned. A manifest with no postprocess block still carries them.
    let mut cfg = PostprocessConfig {
        silence_render_ids: nums.ids(manifest, "silence_render_ids"),
        quiet_render_ids: nums.ids(manifest, "quiet_render_ids"),
        ..PostprocessConfig::default()
    };
    let Some(block) = manifest.get("postprocess").and_then(|v| v.as_object()) else {
        return Ok(cfg);
    };
    // The censuses live at the manifest top level, where they are read above.
    // Accepting them here too would give one value two homes in one file;
    // Python's block reader refuses them the same way.
    for key in ["silence_render_ids", "quiet_render_ids"] {
        if block.contains_key(key) {
            return Err(format!(
                "manifest['postprocess'][{key:?}] belongs at the manifest top \
                 level, beside 'silence_token_ids'"
            ));
        }
    }
    // The block the numeric and name reads below go through, so a value with
    // no reading is refused rather than left as the shipping constant beside
    // it. Python reads this block off the dataclass and refuses a member that
    // is not a number by name.
    let pp = Block::of(manifest, "postprocess");
    if let Some(v) = nums.text(pp, "mode") {
        cfg.mode = PostprocessMode::parse(&v)?;
    }
    // A string field like mode, and refused like mode: the resolver must not
    // cut where the manifest said to condemn.
    if let Some(v) = nums.text(pp, "repetition_resume") {
        cfg.repetition_resume = RepetitionResume::parse(&v)?;
    }
    // A string field like mode, and refused like mode: the loop exemption
    // must not read one silence list under a manifest declaring another.
    if let Some(v) = nums.text(pp, "repetition_silence") {
        cfg.repetition_silence = RepetitionSilence::parse(&v)?;
    }
    cfg.ceiling_speech_per_text_token = nums.read(
        pp,
        "ceiling_speech_per_text_token",
        cfg.ceiling_speech_per_text_token,
    );
    cfg.trailing_filler_threshold = nums.read(
        pp,
        "trailing_filler_threshold",
        cfg.trailing_filler_threshold,
    );
    cfg.filler_min_eos_probability = nums.read(
        pp,
        "filler_min_eos_probability",
        cfg.filler_min_eos_probability,
    );
    cfg.desperation_speech_per_text_token = nums.read(
        pp,
        "desperation_speech_per_text_token",
        cfg.desperation_speech_per_text_token,
    );
    cfg.echo_strong_eos_probability = nums.read(
        pp,
        "echo_strong_eos_probability",
        cfg.echo_strong_eos_probability,
    );
    cfg.echo_weak_eos_probability = nums.read(
        pp,
        "echo_weak_eos_probability",
        cfg.echo_weak_eos_probability,
    );

    cfg.ceiling_slack_tokens = nums.count(pp, "ceiling_slack_tokens", cfg.ceiling_slack_tokens);
    cfg.trailing_silence_run_tokens = nums.count(
        pp,
        "trailing_silence_run_tokens",
        cfg.trailing_silence_run_tokens,
    );
    cfg.desperation_band_ratio =
        nums.read(pp, "desperation_band_ratio", cfg.desperation_band_ratio);
    cfg.desperation_band_floor =
        nums.count(pp, "desperation_band_floor", cfg.desperation_band_floor);
    cfg.filler_max_speech_after_run = nums.count(
        pp,
        "filler_max_speech_after_run",
        cfg.filler_max_speech_after_run,
    );
    cfg.desperation_min_text_tokens = nums.count(
        pp,
        "desperation_min_text_tokens",
        cfg.desperation_min_text_tokens,
    );
    cfg.desperation_min_keep_per_text_token = nums.read(
        pp,
        "desperation_min_keep_per_text_token",
        cfg.desperation_min_keep_per_text_token,
    );
    cfg.ended_tail_silence_run =
        nums.count(pp, "ended_tail_silence_run", cfg.ended_tail_silence_run);
    cfg.ended_tail_blip_max = nums.count(pp, "ended_tail_blip_max", cfg.ended_tail_blip_max);
    cfg.ended_tail_word_max = nums.count(pp, "ended_tail_word_max", cfg.ended_tail_word_max);
    cfg.ended_tail_keep = nums.count(pp, "ended_tail_keep", cfg.ended_tail_keep);
    cfg.echo_strong_max_tail = nums.count(pp, "echo_strong_max_tail", cfg.echo_strong_max_tail);
    cfg.echo_strong_min_position_pct = nums.count(
        pp,
        "echo_strong_min_position_pct",
        cfg.echo_strong_min_position_pct,
    );
    cfg.echo_weak_max_tail = nums.count(pp, "echo_weak_max_tail", cfg.echo_weak_max_tail);
    cfg.echo_weak_min_position_pct = nums.count(
        pp,
        "echo_weak_min_position_pct",
        cfg.echo_weak_min_position_pct,
    );
    // Every postprocess parameter, because a hand-written list that misses
    // one is a manifest key this port does not read. Python takes its fields
    // off the dataclass precisely so a new constant cannot be left out; the
    // four ports write the list by hand, so the list has to be complete.
    // Defaults matching hides the gap until a checkpoint sets one of them, at
    // which point the manifest declares one recipe and the engine runs
    // another.
    cfg.dropout_min_tokens = nums.count(pp, "dropout_min_tokens", cfg.dropout_min_tokens);
    cfg.retry_max_attempts = nums.count(pp, "retry_max_attempts", cfg.retry_max_attempts);
    cfg.pacing_tolerance = nums.read(pp, "pacing_tolerance", cfg.pacing_tolerance);
    cfg.repetition_max_period = nums.count(pp, "repetition_max_period", cfg.repetition_max_period);
    cfg.repetition_min_cycles = nums.count(pp, "repetition_min_cycles", cfg.repetition_min_cycles);
    cfg.repetition_min_span = nums.count(pp, "repetition_min_span", cfg.repetition_min_span);
    cfg.stall_run_tokens = nums.count(pp, "stall_run_tokens", cfg.stall_run_tokens);
    // Checked once, where the manifest is read, exactly as the chunking recipe
    // is checked at the engine door: a detector constant out of range is not a
    // configuration, and two of them took the run down rather than declining.
    cfg.validate()?;
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `postprocess_from` with its accumulator, so a test reads a block the
    /// way `Checkpoint::open` reads one.
    fn postprocess_of(manifest: &HashMap<String, Value>) -> Result<PostprocessConfig, String> {
        let mut nums = Numbers::default();
        let cfg = postprocess_from(manifest, &mut nums)?;
        nums.settle()?;
        Ok(cfg)
    }

    /// The manifest contract is a contract.
    ///
    /// Every engine refuses a `format_version` it does not read; a port that
    /// accepts any version will happily load a checkpoint whose fields mean
    /// something else: the loader would still "work", and the audio would be
    /// wrong for reasons no error names. Only implemented formats are accepted.
    #[test]
    fn supported_format_versions_are_the_ones_implemented() {
        assert_eq!(SUPPORTED_FORMAT_VERSIONS, [1, 2]);
    }

    /// And the mode, because the version is only as honest as the packer.
    ///
    /// A manifest saying `format_version 1` beside `decode.mode =
    /// "fusion_mtp2"` is inconsistent and must fail before loading graphs.
    #[test]
    fn supported_decode_modes_are_the_ones_implemented() {
        assert_eq!(SUPPORTED_DECODE_MODES, ["single", "fusion_mtp2"]);
    }
    /// A safetensors file carrying nothing but a manifest.
    ///
    /// Small enough to write in a test and complete enough for `open` to
    /// reach every check it makes, which is the point: asserting on the two
    /// constants alone left both guards deletable with the tests still green.
    #[cfg(test)]
    fn write_manifest_only(dir: &std::path::Path, name: &str, manifest: &str) -> String {
        let header = serde_json::json!({"__metadata__": {"manifest": manifest}}).to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        let path = dir.join(name);
        std::fs::write(&path, bytes).unwrap();
        path.to_string_lossy().into_owned()
    }

    /// The loader, not the constant.
    ///
    /// Three manifests: the one every 0.1.0 checkpoint carries, the one this
    /// engine has not implemented the loop for, and the one that understates
    /// its version, which clears the number gate and would have been run
    /// through the one-token loop, producing fluent speech that is not the
    /// text.
    #[test]
    fn the_loader_refuses_what_the_constants_declare() {
        let dir = std::env::temp_dir().join(format!("loudkit-fmt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let ok = write_manifest_only(
            &dir,
            "v1.safetensors",
            r#"{"format":"loudkit-checkpoint","format_version":1}"#,
        );
        assert!(
            Checkpoint::open(&ok).is_ok(),
            "a version-1 checkpoint must load"
        );

        let single = write_manifest_only(
            &dir,
            "v1-single.safetensors",
            r#"{"format":"loudkit-checkpoint","format_version":1,"decode":{"mode":"single"}}"#,
        );
        assert!(
            Checkpoint::open(&single).is_ok(),
            "an explicit single is the same loop"
        );

        let v2 = write_manifest_only(
            &dir,
            "v2.safetensors",
            r#"{"format":"loudkit-checkpoint","format_version":2,"decode":{"mode":"fusion_mtp2"}}"#,
        );
        assert!(Checkpoint::open(&v2).is_ok());

        // The one the version gate alone cannot catch.
        let lying = write_manifest_only(
            &dir,
            "v1-fusion.safetensors",
            r#"{"format":"loudkit-checkpoint","format_version":1,"decode":{"mode":"fusion_mtp2"}}"#,
        );
        let err = Checkpoint::open(&lying)
            .err()
            .expect("a fusion mode under version 1 must be refused");
        assert!(err.contains("decode.mode"), "{err}");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Where the reader breathes is declared by the checkpoint, not assumed by
    /// the runtime.
    #[test]
    fn chunking_comes_from_the_manifest() {
        let mut manifest = HashMap::new();
        manifest.insert(
            "chunking".to_string(),
            serde_json::json!({
                "enabled": false,
                "max_tokens": 99,
                "prefix_tokens": 3,
                "split_on": ["|"],
            }),
        );
        // Built directly rather than through `open`: the chunking recipe is a
        // property of the manifest, and requiring a 1.27 GB file to check that
        // would make it untestable without one.
        let ckpt = Checkpoint {
            manifest,
            file: safetensors::File {
                tensors: HashMap::new(),
                metadata: HashMap::new(),
            },
        };
        let cfg = ckpt.chunking();
        assert!(!cfg.enabled);
        assert_eq!(cfg.max_tokens, 99);
        assert_eq!(cfg.prefix_tokens, 3);
        assert_eq!(cfg.split_on, vec!["|".to_string()]);
    }

    /// A manifest document as the map every reader below takes.
    #[cfg(test)]
    fn map_of(manifest: &Value) -> HashMap<String, Value> {
        manifest
            .as_object()
            .map(|o| o.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default()
    }

    /// A `Checkpoint` over a manifest, without the 1.27 GB file behind it.
    #[cfg(test)]
    fn from_manifest(manifest: Value) -> Checkpoint {
        Checkpoint {
            manifest: map_of(&manifest),
            file: safetensors::File {
                tensors: HashMap::new(),
                metadata: HashMap::new(),
            },
        }
    }

    /// A block that carries some of its keys and not the rest is read, not
    /// aborted on.
    ///
    /// Indexing a `serde_json::Map` with `[]` panics on a key the map does not
    /// hold, so the four blocks below took the whole process down instead of
    /// the fallback written beside each key.
    #[test]
    fn a_partial_block_falls_back_instead_of_panicking() {
        let ckpt = from_manifest(serde_json::json!({
            "window": {"max_speech_tokens": 200},
            "sampling_defaults": {"temperature": 0.5},
            "eos_floor": {"min_tokens_text_ratio": 1.5},
            "speech_tokens": {"stop": 42},
        }));

        let w = ckpt.window();
        assert_eq!(w.max_speech_tokens, 200);
        assert_eq!(w.static_length, None);
        assert_eq!(w.static_prompt_tokens, None);
        assert_eq!(w.pad_token_id, None);

        let (temp, rep, minp, max_new, _sil, floor, ratio) = ckpt.sampling();
        assert!((temp - 0.5).abs() < f64::EPSILON);
        assert!((rep - 1.2).abs() < f64::EPSILON);
        assert!((minp - 0.05).abs() < f64::EPSILON);
        assert_eq!(max_new, 255);
        assert_eq!(floor, 0);
        assert!((ratio - 1.5).abs() < f64::EPSILON);

        let (start, stop, _vocab) = ckpt.speech_tokens();
        assert_eq!(start, 6561);
        assert_eq!(stop, 42);
    }

    /// Every default this reader falls back to is `algorithm_from`'s.
    ///
    /// Sample rate, speech vocabulary and Euler step count have to be the
    /// reference reader's exactly: a `speech_vocab_size` of 0 sizes the
    /// sampler's seen-token slice to nothing, and `n_cfm_timesteps` 0 leaves
    /// the Euler loop with no step to take, so the vocoder reads the prior
    /// noise.
    #[test]
    fn absent_keys_take_the_python_defaults() {
        let ckpt = from_manifest(serde_json::json!({}));

        let (_, _, vocab) = ckpt.speech_tokens();
        assert_eq!(
            vocab, 8194,
            "manifest.py defaults speech_vocab_size to 8194"
        );
        assert_eq!(ckpt.euler_steps(), 2, "n_cfm_timesteps defaults to 2");
        assert_eq!(ckpt.sample_rate(), 24_000, "sample_rate defaults to 24000");

        let (_, _, _, _, _, floor, ratio) = ckpt.sampling();
        assert_eq!(floor, 0, "eos_floor.min_tokens_floor defaults to 0");
        assert!(
            ratio.abs() < f64::EPSILON,
            "eos_floor.min_tokens_text_ratio defaults to 0.0"
        );

        // A manifest with no window block is the ragged window, as
        // `_window_from` returns a default `WindowConfig`.
        let w = ckpt.window();
        assert_eq!(w.max_speech_tokens, 255);
        assert_eq!(w.static_length, None);
        assert_eq!(w.pad_token_id, None);
        assert_eq!(w.static_prompt_tokens, None);
    }

    /// A key of the wrong shape is refused at the door, not defaulted.
    ///
    /// `silence_token_ids: "123"` is the case Python names on its own: read as
    /// a sequence it would load as three tokens, and read with `as_array` it
    /// loads as none, which is a different recipe from the one the manifest
    /// declares.
    #[test]
    fn a_key_of_the_wrong_shape_is_refused() {
        for (key, bad) in [
            ("silence_token_ids", serde_json::json!("123")),
            ("window", serde_json::json!(255)),
            ("sampling_defaults", serde_json::json!([1, 2])),
            ("speech_tokens", serde_json::json!("6561")),
            ("eos_floor", serde_json::json!(10)),
            ("euler_grid", serde_json::json!("cosine")),
        ] {
            let manifest = serde_json::json!({ key: bad });
            let err = check_block_shapes("m.safetensors", &manifest)
                .expect_err("a mistyped key must be refused");
            assert!(err.contains(key), "the error must name the key: {err}");
        }
    }

    /// The shapes the readers below rely on, including the two spellings of
    /// "unstated" that `algorithm_from` accepts.
    #[test]
    fn the_shapes_the_readers_read_are_accepted() {
        let manifest = serde_json::json!({
            "window": null,
            "euler_grid": null,
            "sampling_defaults": {"temperature": 0.8},
            "speech_tokens": {"start": 6561, "stop": 6562},
            "eos_floor": {"min_tokens_floor": 10},
            "silence_token_ids": [1, 2, 3],
        });
        assert!(check_block_shapes("m.safetensors", &manifest).is_ok());
        assert!(check_block_shapes("m.safetensors", &serde_json::json!({})).is_ok());
    }

    /// A chunking key this splitter does not honour is refused by name.
    ///
    /// Ignored, it splits the text differently here and in Python under one
    /// `recipe_version`, which is the divergence class the guidance door
    /// already refuses.
    #[test]
    fn an_unhonoured_chunking_key_is_refused() {
        for key in UNHONOURED_CHUNKING_KEYS {
            let manifest = serde_json::json!({ "chunking": { key: 96 } });
            let err = check_chunking_keys("m.safetensors", &manifest)
                .expect_err("a key this splitter ignores must be refused");
            assert!(err.contains(key), "the error must name the key: {err}");
        }
        // The keys it does honour still load, block or no block.
        let honoured = serde_json::json!({
            "chunking": {"enabled": true, "max_tokens": 255, "prefix_tokens": 6},
        });
        assert!(check_chunking_keys("m.safetensors", &honoured).is_ok());
        assert!(check_chunking_keys("m.safetensors", &serde_json::json!({})).is_ok());
    }

    /// A guidance mode this port does not implement must be refused, not run
    /// as single_path under a fingerprint that says otherwise. It was not
    /// modelled here at all, which made that outcome silent rather than merely
    /// possible.
    #[test]
    fn guidance_refuses_what_this_port_cannot_run() {
        fn with(mode: Option<&str>) -> Result<String, String> {
            let mut manifest = HashMap::new();
            if let Some(m) = mode {
                manifest.insert("guidance".to_string(), Value::String(m.to_string()));
            }
            guidance_from(&manifest)
        }
        assert_eq!(with(None).unwrap(), "single_path");
        assert_eq!(with(Some("single_path")).unwrap(), "single_path");
        assert!(with(Some("cfg_dual_path")).is_err());
        assert!(with(Some("sorta_guided")).is_err());
    }

    // Pins recipe_version defaulting: a manifest missing the key defaults,
    // as in Python/JS/Swift (non-amended checkpoints).
    #[test]
    fn recipe_version_defaults_when_absent() {
        let manifest = HashMap::new();
        assert_eq!(recipe_version_from(&manifest).unwrap(), "loudkit-1");
    }

    #[test]
    fn recipe_version_accepts_the_one_recipe() {
        let mut manifest = HashMap::new();
        manifest.insert(
            "recipe_version".to_string(),
            Value::String("loudkit-1".to_string()),
        );
        assert_eq!(recipe_version_from(&manifest).unwrap(), "loudkit-1");
    }

    // One recipe means one accepted value, and the error names what the
    // manifest declared. Believing a foreign tag would fingerprint it;
    // rewriting it would claim this recipe for a checkpoint that named
    // another. All five ports refuse it identically.
    #[test]
    fn recipe_version_refuses_a_foreign_tag() {
        let mut manifest = HashMap::new();
        manifest.insert(
            "recipe_version".to_string(),
            Value::String("loudkit-9".to_string()),
        );
        manifest.insert("chunking".to_string(), Value::Object(Default::default()));
        manifest.insert("postprocess".to_string(), Value::Object(Default::default()));
        let err = recipe_version_from(&manifest).unwrap_err();
        assert!(err.contains("loudkit-9"), "error must name the tag: {err}");
    }

    // A tag that is not even a string is refused, not defaulted: a manifest
    // one port misreads while another defaults is the divergence class this
    // library exists to prevent.
    #[test]
    fn recipe_version_refuses_a_non_string() {
        let mut manifest = HashMap::new();
        manifest.insert("recipe_version".to_string(), Value::from(9));
        assert!(recipe_version_from(&manifest).is_err());
    }

    // The detectors default on even when the block is absent; the tag does
    // not move for it: there is one recipe, and a manifest that omits a
    // block left a shipping default unstated.
    #[test]
    fn absent_postprocess_block_defaults_the_detectors_on() {
        let mut manifest = HashMap::new();
        manifest.insert(
            "recipe_version".to_string(),
            Value::String("loudkit-1".to_string()),
        );
        assert_eq!(recipe_version_from(&manifest).unwrap(), "loudkit-1");
        assert_eq!(
            postprocess_of(&manifest).unwrap().mode,
            PostprocessMode::Trim
        );
    }

    #[test]
    fn an_unknown_postprocess_mode_is_refused() {
        let mut manifest = HashMap::new();
        let mut block = serde_json::Map::new();
        block.insert("mode".to_string(), Value::String("shave".to_string()));
        manifest.insert("postprocess".to_string(), Value::Object(block));
        assert!(postprocess_of(&manifest).is_err());
    }

    /// A law this port does not implement must be refused, not defaulted:
    /// the resolver would cut where the manifest said to condemn. "cut"
    /// names the pre-amendment law and is read.
    #[test]
    fn an_unknown_repetition_resume_is_refused() {
        let mut manifest = HashMap::new();
        manifest.insert(
            "postprocess".to_string(),
            serde_json::json!({"repetition_resume": "maybe"}),
        );
        let err = postprocess_of(&manifest).unwrap_err();
        assert!(
            err.contains("repetition_resume"),
            "the error must name the field: {err}"
        );

        let mut manifest = HashMap::new();
        manifest.insert(
            "postprocess".to_string(),
            serde_json::json!({"repetition_resume": "cut"}),
        );
        assert_eq!(
            postprocess_of(&manifest).unwrap().repetition_resume,
            RepetitionResume::Cut
        );
    }

    /// A family this port does not implement must be refused, not defaulted:
    /// the loop exemption would read one silence list under a manifest
    /// declaring another. "sampling" names the pre-amendment law and is read.
    #[test]
    fn an_unknown_repetition_silence_is_refused() {
        let mut manifest = HashMap::new();
        let mut block = serde_json::Map::new();
        block.insert(
            "repetition_silence".to_string(),
            Value::String("both".to_string()),
        );
        manifest.insert("postprocess".to_string(), Value::Object(block));
        let err = postprocess_of(&manifest).unwrap_err();
        assert!(
            err.contains("repetition_silence"),
            "the error must name the field: {err}"
        );

        let mut manifest = HashMap::new();
        manifest.insert(
            "postprocess".to_string(),
            serde_json::json!({"repetition_silence": "sampling"}),
        );
        assert_eq!(
            postprocess_of(&manifest).unwrap().repetition_silence,
            RepetitionSilence::Sampling
        );
    }

    /// The render censuses ride the manifest top level, beside
    /// `silence_token_ids`; declaring them inside the postprocess block would
    /// give one value two homes in one file, so it is refused by name, the
    /// same way Python's block reader refuses it.
    #[test]
    fn render_ids_inside_the_postprocess_block_are_refused() {
        for key in ["silence_render_ids", "quiet_render_ids"] {
            let mut manifest = HashMap::new();
            let mut block = serde_json::Map::new();
            block.insert(key.to_string(), serde_json::json!([1, 2]));
            manifest.insert("postprocess".to_string(), Value::Object(block));
            let err = postprocess_of(&manifest).unwrap_err();
            assert!(err.contains(key), "the error must name the key: {err}");
        }
    }

    /// The top-level censuses reach the detectors, with or without a
    /// postprocess block: a manifest with no detector overrides still
    /// carries the properties of its weights.
    #[test]
    fn top_level_censuses_are_read() {
        let mut manifest = HashMap::new();
        manifest.insert("silence_render_ids".to_string(), serde_json::json!([7, 8]));
        manifest.insert("quiet_render_ids".to_string(), serde_json::json!([9]));
        let cfg = postprocess_of(&manifest).unwrap();
        assert_eq!(cfg.silence_render_ids, vec![7, 8]);
        assert_eq!(cfg.quiet_render_ids, vec![9]);

        manifest.insert(
            "postprocess".to_string(),
            serde_json::json!({"stall_run_tokens": 30}),
        );
        let with_block = postprocess_of(&manifest).unwrap();
        assert_eq!(with_block.stall_run_tokens, 30);
        assert_eq!(with_block.silence_render_ids, vec![7, 8]);
    }

    /// A manifest key that counts things takes a whole number.
    ///
    /// Truncating 2.7 to 2 turns a packer's arithmetic mistake into a chunker
    /// that breathes in a different place, a window framed to a different
    /// length or a census naming a token nobody wrote, under a
    /// `recipe_version` saying the five implementations agree. `manifest._int`
    /// refuses each of these in the sentence pinned below, and the reference's
    /// postprocess reader refuses the counts in that block.
    #[test]
    fn a_fractional_count_is_refused() {
        for (manifest, want) in [
            (
                serde_json::json!({"chunking": {"max_tokens": 200.5}}),
                "manifest['chunking']['max_tokens'] must be a whole number, got 200.5",
            ),
            (
                serde_json::json!({"chunking": {"prefix_tokens": 2.7}}),
                "manifest['chunking']['prefix_tokens'] must be a whole number, got 2.7",
            ),
            (
                serde_json::json!({"window": {"max_speech_tokens": 255.5}}),
                "manifest['window']['max_speech_tokens'] must be a whole number, got 255.5",
            ),
            (
                serde_json::json!({"window": {"static_length": 255.5}}),
                "manifest['window']['static_length'] must be a whole number, got 255.5",
            ),
            (
                serde_json::json!({"window": {"pad_token_id": 2.7}}),
                "manifest['window']['pad_token_id'] must be a whole number, got 2.7",
            ),
            (
                serde_json::json!({"window": {"static_prompt_tokens": 238.5}}),
                "manifest['window']['static_prompt_tokens'] must be a whole number, got 238.5",
            ),
            (
                serde_json::json!({"speech_tokens": {"start": 6561.5}}),
                "manifest['speech_tokens']['start'] must be a whole number, got 6561.5",
            ),
            (
                serde_json::json!({"speech_tokens": {"stop": 6562.5}}),
                "manifest['speech_tokens']['stop'] must be a whole number, got 6562.5",
            ),
            (
                serde_json::json!({"sampling_defaults": {"max_new_tokens": 255.5}}),
                "manifest['sampling_defaults']['max_new_tokens'] must be a whole number, got 255.5",
            ),
            (
                serde_json::json!({"eos_floor": {"min_tokens_floor": 10.5}}),
                "manifest['eos_floor']['min_tokens_floor'] must be a whole number, got 10.5",
            ),
            (
                serde_json::json!({"n_cfm_timesteps": 2.7}),
                "manifest['n_cfm_timesteps'] must be a whole number, got 2.7",
            ),
            (
                serde_json::json!({"sample_rate": 24000.5}),
                "manifest['sample_rate'] must be a whole number, got 24000.5",
            ),
            (
                serde_json::json!({"speech_vocab_size": 8194.5}),
                "manifest['speech_vocab_size'] must be a whole number, got 8194.5",
            ),
            (
                serde_json::json!({"postprocess": {"stall_run_tokens": 2.7}}),
                "manifest['postprocess']['stall_run_tokens'] must be a whole number, got 2.7",
            ),
            (
                serde_json::json!({"silence_token_ids": [4137, 4218.5]}),
                "manifest['silence_token_ids'][1] must be a whole number, got 4218.5",
            ),
            (
                serde_json::json!({"silence_render_ids": [2.7]}),
                "manifest['silence_render_ids'][0] must be a whole number, got 2.7",
            ),
            (
                serde_json::json!({"quiet_render_ids": [2.7]}),
                "manifest['quiet_render_ids'][0] must be a whole number, got 2.7",
            ),
            // The fraction is tested before the sign, as the reference tests
            // only the fraction: a negative fraction is refused for what it is.
            (
                serde_json::json!({"window": {"pad_token_id": -1.5}}),
                "manifest['window']['pad_token_id'] must be a whole number, got -1.5",
            ),
        ] {
            let err =
                check_readable(&map_of(&manifest)).expect_err("a fractional count must be refused");
            assert_eq!(err, want);
        }
    }

    /// And the fields that hold a rate, a threshold or a probability still
    /// take one. Getting the boundary wrong in this direction refuses a
    /// manifest that is correct, which is the same defect facing the other
    /// way.
    #[test]
    fn a_fractional_rate_is_a_value() {
        let manifest = serde_json::json!({
            "token_rate_hz": 25.5,
            "edge_fade_seconds": 0.03,
            "euler_grid": [0.0, 0.27, 1.0],
            "sampling_defaults": {
                "temperature": 0.75, "repetition_penalty": 1.25, "min_p": 0.055
            },
            "eos_floor": {"min_tokens_text_ratio": 1.25},
            "postprocess": {"pacing_tolerance": 0.17},
        });
        let m = map_of(&manifest);
        assert!(check_readable(&m).is_ok());
        let ckpt = from_manifest(manifest);
        assert!((ckpt.token_rate_hz() - 25.5).abs() < 1e-12);
        assert!((ckpt.edge_fade_seconds() - 0.03).abs() < 1e-12);
        let (temp, rep, min_p, _, _, _, ratio) = ckpt.sampling();
        assert!((temp - 0.75).abs() < 1e-12);
        assert!((rep - 1.25).abs() < 1e-12);
        assert!((min_p - 0.055).abs() < 1e-12);
        assert!((ratio - 1.25).abs() < 1e-12);
        assert_eq!(ckpt.euler_grid(), Some(vec![0.0, 0.27, 1.0]));
    }

    /// A count written as a float is the count: 255.0 is 255 in JSON and in
    /// every reader here, and only the fraction is refused.
    #[test]
    fn a_whole_count_written_as_a_float_is_read() {
        let manifest = serde_json::json!({
            "sample_rate": 24000.0,
            "n_cfm_timesteps": 2.0,
            "window": {"max_speech_tokens": 255.0},
            "silence_token_ids": [4254.0],
        });
        assert!(check_readable(&map_of(&manifest)).is_ok());
        let ckpt = from_manifest(manifest);
        assert_eq!(ckpt.sample_rate(), 24_000);
        assert_eq!(ckpt.euler_steps(), 2);
        assert_eq!(ckpt.window().max_speech_tokens, 255);
        assert_eq!(ckpt.sampling().4, vec![4254]);
    }
}
