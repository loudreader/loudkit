//! The packed checkpoint: manifest + the embedding tables the generator needs
//! on the host. Port of the `loudkit.checkpoint` reads.

use std::collections::HashMap;

use serde_json::Value;

use crate::chunking::ChunkConfig;
use crate::postprocess::{Config as PostprocessConfig, Mode as PostprocessMode};
use crate::safetensors;
use crate::windowing::WindowConfig;

/// The four fp32 embedding tables the generator reads: text, speech, text
/// positions, speech positions (fp16 storage upcasts exactly).
type GeneratorTables = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

pub struct Checkpoint {
    pub manifest: HashMap<String, Value>,
    file: safetensors::File,
}

/// Manifest format versions this build understands, mirroring Python's.
pub const SUPPORTED_FORMAT_VERSIONS: [i64; 1] = [1];

impl Checkpoint {
    pub fn open(path: &str) -> Result<Self, String> {
        let file = safetensors::File::open(path)?;
        let manifest_str = file
            .metadata
            .get("manifest")
            .ok_or_else(|| format!("{path}: no embedded manifest — not a loudkit checkpoint"))?;
        let manifest: Value = serde_json::from_str(manifest_str)
            .map_err(|e| format!("{path}: bad manifest JSON: {e}"))?;
        if manifest["format"].as_str() != Some("loudkit-checkpoint") {
            return Err(format!(
                "{path}: no embedded manifest — not a loudkit checkpoint"
            ));
        }
        // `format_version` is checked, not only `format`. Python refuses a
        // version it does not read; a port that accepts any version will
        // happily load a future checkpoint whose fields mean something else —
        // the loader would still "work", and the audio would be wrong for
        // reasons no error names.
        let version = manifest["format_version"].as_i64().unwrap_or(-1);
        if !SUPPORTED_FORMAT_VERSIONS.contains(&version) {
            return Err(format!(
                "{path}: manifest format_version {version}; this build reads \
                 {SUPPORTED_FORMAT_VERSIONS:?}"
            ));
        }
        let m = manifest
            .as_object()
            .map(|o| o.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();
        Ok(Checkpoint { manifest: m, file })
    }

    fn num(&self, key: &str) -> f64 {
        self.manifest
            .get(key)
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0)
    }

    /// The window recipe from the manifest (or the production fallback).
    pub fn window(&self) -> WindowConfig {
        let mut w = crate::windowing::production_window();
        if let Some(win) = self.manifest.get("window").and_then(|v| v.as_object()) {
            if let Some(v) = win["max_speech_tokens"].as_u64() {
                w.max_speech_tokens = v as usize;
            }
            if let Some(v) = win["static_length"].as_u64() {
                w.static_length = Some(v as usize);
            }
            if let Some(v) = win["static_prompt_tokens"].as_u64() {
                w.static_prompt_tokens = Some(v as usize);
            }
            if let Some(v) = win["pad_token_id"].as_u64() {
                w.pad_token_id = Some(v as usize);
            }
        }
        w
    }

    /// Sampling values from the manifest.
    pub fn sampling(&self) -> (f64, f64, f64, usize, Vec<usize>, usize, f64) {
        let sd = self
            .manifest
            .get("sampling_defaults")
            .and_then(|v| v.as_object());
        let get = |k: &str, def: f64| sd.and_then(|o| o[k].as_f64()).unwrap_or(def);
        let sil: Vec<usize> = self
            .manifest
            .get("silence_token_ids")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_u64())
                    .map(|x| x as usize)
                    .collect()
            })
            .unwrap_or_default();
        let eos = self.manifest.get("eos_floor").and_then(|v| v.as_object());
        let floor = eos
            .and_then(|o| o["min_tokens_floor"].as_u64())
            .unwrap_or(10) as usize;
        let ratio = eos
            .and_then(|o| o["min_tokens_text_ratio"].as_f64())
            .unwrap_or(1.2);
        (
            get("temperature", 0.8),
            get("repetition_penalty", 1.2),
            get("min_p", 0.05),
            get("max_new_tokens", 255.0) as usize,
            sil,
            floor,
            ratio,
        )
    }

    /// Speech token ids.
    pub fn speech_tokens(&self) -> (usize, usize, usize) {
        let sp = self
            .manifest
            .get("speech_tokens")
            .and_then(|v| v.as_object());
        let start = sp.and_then(|o| o["start"].as_u64()).unwrap_or(6561) as usize;
        let stop = sp.and_then(|o| o["stop"].as_u64()).unwrap_or(6562) as usize;
        let vocab = self.num("speech_vocab_size") as usize;
        (start, stop, vocab)
    }

    pub fn sample_rate(&self) -> usize {
        self.num("sample_rate") as usize
    }

    /// Guidance strength. Zero in `single_path`, which is the shipping mode.
    pub fn guidance_rate(&self) -> f64 {
        self.manifest
            .get("guidance_rate")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0)
    }

    /// Speech tokens per second — 25 Hz for this model family.
    pub fn token_rate_hz(&self) -> f64 {
        self.manifest
            .get("token_rate_hz")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(25.0)
    }

    pub fn euler_steps(&self) -> usize {
        self.num("n_cfm_timesteps") as usize
    }

    /// The explicit Euler time grid, or `None` for the cosine schedule.
    ///
    /// A JSON string is refused rather than iterated, matching Python's guard
    /// on this key: a manifest one port misreads while another defaults is the
    /// divergence class this library exists to prevent.
    pub fn euler_grid(&self) -> Option<Vec<f64>> {
        let grid = self.manifest.get("euler_grid")?.as_array()?;
        Some(grid.iter().filter_map(serde_json::Value::as_f64).collect())
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
        postprocess_from(&self.manifest)
    }

    /// The chunking recipe: where the reader breathes, and the prefix carry.
    ///
    /// Read from the manifest rather than defaulted. A checkpoint that declares
    /// its own boundaries and a runtime that silently uses different ones agree
    /// on `recipe_version` and disagree on the reading, which is the drift the
    /// fingerprint exists to prevent.
    pub fn chunking(&self) -> ChunkConfig {
        let mut cfg = ChunkConfig::default();
        let Some(block) = self.manifest.get("chunking").and_then(|v| v.as_object()) else {
            return cfg;
        };
        if let Some(v) = block.get("enabled").and_then(serde_json::Value::as_bool) {
            cfg.enabled = v;
        }
        if let Some(v) = block.get("max_tokens").and_then(serde_json::Value::as_u64) {
            cfg.max_tokens = v as usize;
        }
        if let Some(v) = block
            .get("prefix_tokens")
            .and_then(serde_json::Value::as_u64)
        {
            cfg.prefix_tokens = v as usize;
        }
        if let Some(v) = block.get("split_on").and_then(|v| v.as_array()) {
            let seps: Vec<String> = v
                .iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect();
            if !seps.is_empty() {
                cfg.split_on = seps;
            }
        }
        cfg
    }

    /// The declared guidance mode, refused when this port cannot honour it.
    ///
    /// This binding calls the estimator once per Euler step and never forms
    /// `(1+w)·v_cond − w·v_uncond`, so a `cfg_dual_path` checkpoint would load,
    /// produce plausible audio, and disagree with the Python engine under a
    /// matching `recipe_version`. It was not modelled here at all, which made
    /// that outcome not merely possible but silent. The JS, Go, Python and
    /// CoreML paths all refuse the same mode for the same reason.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown mode, and for `cfg_dual_path`.
    pub fn guidance(&self) -> Result<String, String> {
        guidance_from(&self.manifest)
    }

    /// The fp32 embedding tables the generator uses (fp16 storage upcasts
    /// exactly).
    pub fn generator_tables(&self) -> Result<GeneratorTables, String> {
        Ok((
            self.file.f32("t3.text_emb.weight")?,
            self.file.f32("t3.speech_emb.weight")?,
            self.file.f32("t3.text_pos_emb.emb.weight")?,
            self.file.f32("t3.speech_pos_emb.emb.weight")?,
        ))
    }

    /// The 192->80 speaker affine the flow decoder conditions on.
    pub fn speaker_affine(&self) -> Result<(Vec<f32>, Vec<f32>), String> {
        Ok((
            self.file.f32("s3gen.flow.spk_embed_affine_layer.weight")?,
            self.file.f32("s3gen.flow.spk_embed_affine_layer.bias")?,
        ))
    }
}

fn guidance_from(manifest: &HashMap<String, Value>) -> Result<String, String> {
    let declared = manifest
        .get("guidance")
        .and_then(|v| v.as_str())
        .unwrap_or("single_path")
        .to_string();
    match declared.as_str() {
        "single_path" => Ok(declared),
        "cfg_dual_path" => Err("manifest declares guidance mode cfg_dual_path, which this \
             binding does not implement — it would render single-path audio and silently \
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
fn postprocess_from(manifest: &HashMap<String, Value>) -> Result<PostprocessConfig, String> {
    let mut cfg = PostprocessConfig::default();
    let Some(block) = manifest.get("postprocess").and_then(|v| v.as_object()) else {
        return Ok(cfg);
    };
    if let Some(v) = block.get("mode").and_then(Value::as_str) {
        cfg.mode = PostprocessMode::parse(v)?;
    }
    let num = |key: &str, current: f64| -> f64 {
        block.get(key).and_then(Value::as_f64).unwrap_or(current)
    };
    cfg.ceiling_speech_per_text_token = num(
        "ceiling_speech_per_text_token",
        cfg.ceiling_speech_per_text_token,
    );
    cfg.trailing_filler_threshold = num("trailing_filler_threshold", cfg.trailing_filler_threshold);
    cfg.filler_min_eos_probability =
        num("filler_min_eos_probability", cfg.filler_min_eos_probability);
    cfg.desperation_speech_per_text_token = num(
        "desperation_speech_per_text_token",
        cfg.desperation_speech_per_text_token,
    );
    cfg.echo_strong_eos_probability = num(
        "echo_strong_eos_probability",
        cfg.echo_strong_eos_probability,
    );
    cfg.echo_weak_eos_probability = num("echo_weak_eos_probability", cfg.echo_weak_eos_probability);

    let count = |key: &str, current: usize| -> usize {
        block
            .get(key)
            .and_then(Value::as_u64)
            .map_or(current, |v| usize::try_from(v).unwrap_or(current))
    };
    cfg.ceiling_slack_tokens = count("ceiling_slack_tokens", cfg.ceiling_slack_tokens);
    cfg.trailing_silence_run_tokens = count(
        "trailing_silence_run_tokens",
        cfg.trailing_silence_run_tokens,
    );
    cfg.desperation_band_ratio = num("desperation_band_ratio", cfg.desperation_band_ratio);
    cfg.desperation_band_floor = count("desperation_band_floor", cfg.desperation_band_floor);
    cfg.filler_max_speech_after_run = count(
        "filler_max_speech_after_run",
        cfg.filler_max_speech_after_run,
    );
    cfg.desperation_min_text_tokens = count(
        "desperation_min_text_tokens",
        cfg.desperation_min_text_tokens,
    );
    cfg.ended_tail_silence_run = count("ended_tail_silence_run", cfg.ended_tail_silence_run);
    cfg.ended_tail_blip_max = count("ended_tail_blip_max", cfg.ended_tail_blip_max);
    cfg.ended_tail_word_max = count("ended_tail_word_max", cfg.ended_tail_word_max);
    cfg.ended_tail_keep = count("ended_tail_keep", cfg.ended_tail_keep);
    cfg.echo_strong_max_tail = count("echo_strong_max_tail", cfg.echo_strong_max_tail);
    cfg.echo_strong_min_position_pct = count(
        "echo_strong_min_position_pct",
        cfg.echo_strong_min_position_pct,
    );
    cfg.echo_weak_max_tail = count("echo_weak_max_tail", cfg.echo_weak_max_tail);
    cfg.echo_weak_min_position_pct =
        count("echo_weak_min_position_pct", cfg.echo_weak_min_position_pct);
    // The six this wall was missing. Python reads its fields off the dataclass
    // precisely so a new constant cannot be left out of a hand-written list;
    // the four ports write the list by hand, and every one of them had drifted
    // the same six fields behind. Defaults matched, so nothing sounded wrong —
    // until a checkpoint sets one, at which point the manifest declares one
    // recipe and four engines run another.
    cfg.dropout_min_tokens = count("dropout_min_tokens", cfg.dropout_min_tokens);
    cfg.retry_max_attempts = count("retry_max_attempts", cfg.retry_max_attempts);
    cfg.pacing_tolerance = num("pacing_tolerance", cfg.pacing_tolerance);
    cfg.repetition_max_period = count("repetition_max_period", cfg.repetition_max_period);
    cfg.repetition_min_cycles = count("repetition_min_cycles", cfg.repetition_min_cycles);
    cfg.repetition_min_span = count("repetition_min_span", cfg.repetition_min_span);
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The manifest contract is a contract.
    ///
    /// Python refuses a `format_version` it does not read; a port that accepts
    /// any version will happily load a future checkpoint whose fields mean
    /// something else — the loader would still "work", and the audio would be
    /// wrong for reasons no error names.
    #[test]
    fn supported_format_versions_match_python() {
        assert_eq!(SUPPORTED_FORMAT_VERSIONS, [1]);
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
            postprocess_from(&manifest).unwrap().mode,
            PostprocessMode::Trim
        );
    }

    #[test]
    fn an_unknown_postprocess_mode_is_refused() {
        let mut manifest = HashMap::new();
        let mut block = serde_json::Map::new();
        block.insert("mode".to_string(), Value::String("shave".to_string()));
        manifest.insert("postprocess".to_string(), Value::Object(block));
        assert!(postprocess_from(&manifest).is_err());
    }
}
