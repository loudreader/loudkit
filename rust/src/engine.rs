//! The full synthesis pipeline over the exported ONNX graphs, fp32, no torch.
//! A bit-parity port of the Python/JS/Go engines: same text, voice and seed
//! give the same tokens and the same render band.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ndarray::{Array1, Array2, Array3};
use ort::session::Session;
use ort::value::{DynValue, Value};

use crate::checkpoint::{Checkpoint, GeneratorTables};
use crate::chunking::{self, ChunkConfig};
use crate::execution::{ort_err, Execution, ExecutionConfig};
use crate::fingerprint::repr_float;
use crate::frontend::Frontend;
use crate::noise;
use crate::postprocess;
use crate::sampler::{self, Sampler};
use crate::speechtext;
use crate::timestretch;
use crate::timing::{self, ChunkSpan, ChunkTiming};
use crate::voice;
use crate::windowing::{self, WindowConfig};

const MEL_BINS: usize = 80;
const N_HARMONICS: usize = 9;
const UPSAMPLE_PER_FRAME: usize = 480;
const HIDDEN_DIM: usize = 1024;

/// What varies about one synthesis besides the text and the voice. The
/// default is seed 0, the voice's own language, normal speed, no previous
/// tokens and no cancellation.
#[derive(Clone)]
pub struct Options {
    pub seed: u64,
    /// Language of the text. `None` means the voice's own; name one only to
    /// read text in a language the voice was not enrolled in.
    pub language: Option<String>,
    /// Playback speed in `[0.5, 2.0]`, pitch preserved. `1.0` is an exact
    /// bypass.
    pub speed: f64,
    /// The `tokens` of an earlier [`Synthesis`], so this utterance continues
    /// its pitch contour instead of restarting. Any length; only the tail is
    /// used.
    pub previous_tokens: Option<Vec<usize>>,
    /// Polled on every decode step. When it returns true the chunk being
    /// generated is discarded: [`Engine::synthesize`] returns `Err` with
    /// [`crate::error::CANCELLED`], and [`Engine::stream`] ends after the
    /// chunks it already delivered. `Fn`, not `FnMut`, because `Options` is
    /// borrowed shared; count from an `AtomicUsize`.
    pub should_cancel: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            seed: 0,
            language: None,
            speed: 1.0,
            previous_tokens: None,
            should_cancel: None,
        }
    }
}

impl std::fmt::Debug for Options {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Options")
            .field("seed", &self.seed)
            .field("language", &self.language)
            .field("speed", &self.speed)
            .field("previous_tokens", &self.previous_tokens)
            .field("should_cancel", &self.should_cancel.as_ref().map(|_| "Fn"))
            .finish()
    }
}

impl Options {
    /// `should_cancel` as the `FnMut` the decode loop polls, if any.
    fn cancel(&self) -> Option<impl FnMut() -> bool + 'static> {
        self.should_cancel.clone().map(|f| move || f())
    }
}

/// One rendered passage.
pub struct Synthesis {
    pub audio: Vec<f32>,
    pub sample_rate: usize,
    pub tokens: Vec<usize>,
    /// The `[80, frames]` mel the audio was vocoded from, row-major.
    pub mel: Vec<f32>,
    /// Where each chunk landed, in order and adjacent: chunk *k*'s `end` is
    /// chunk *k+1*'s `start`. The per-word times inside are an estimate.
    pub chunks: Vec<ChunkTiming>,
    /// Generation stopped at the token cap rather than at a stop token, so
    /// the audio is real but the reading is probably cut off. ORed across
    /// chunks.
    pub hit_token_cap: bool,
}

impl Synthesis {
    /// Duration in seconds.
    #[must_use]
    pub fn duration(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.audio.len() as f64 / self.sample_rate as f64
    }

    /// Write the audio as a mono 16-bit PCM WAV.
    ///
    /// # Errors
    ///
    /// Everything [`crate::wav::Audio::save_wav`] returns.
    pub fn save_wav(&self, path: impl AsRef<std::path::Path>) -> Result<(), String> {
        let rate = u32::try_from(self.sample_rate).map_err(|_| "sample rate does not fit a WAV")?;
        crate::wav::Audio::new(self.audio.clone(), rate).save_wav(path)
    }
}

/// One loaded checkpoint, its graphs and its text funnel.
///
/// Holds `ort::session::Session` values, so it is not `Sync`; a server hands
/// one out under its own lock. [`Engine::load`] is the only constructor, which
/// is what keeps every configuration on this type resolved from a manifest.
pub struct Engine {
    config: EngineConfig,
    /// The release this engine was opened from, when it was one: what
    /// `voice`, `voices` and `enroll` read.
    bundle: Option<crate::hub::Bundle>,
    execution: Execution,
    frontend: Frontend,
    text_emb: Vec<f32>,
    speech_emb: Vec<f32>,
    text_pos: Vec<f32>,
    speech_pos: Vec<f32>,
    spk_weight: Vec<f32>,
    spk_bias: Vec<f32>,
    cond: Session,
    prefill: Session,
    step: Session,
    head2: Option<Session>,
    fusion: Option<[Vec<f32>; 4]>,
    n_layers: usize,
    encoder: Session,
    estimator: Session,
    vocoder: Session,
}

/// One rendered chunk, handed to an [`Engine::stream`] callback as soon as it
/// exists.
pub struct Chunk<'a> {
    /// Zero-based position in the split, which is also what its seed was
    /// derived from.
    pub index: usize,
    /// What this chunk was asked to say, *after* the speech funnel: the text
    /// that was tokenised, which is not always what the caller passed in
    /// (numbers are read as words, and Polish respells embedded English). A
    /// caller matching a highlight back against the input would drift the moment
    /// a digit appeared, so what was spoken is what is reported.
    pub text: &'a str,
    pub audio: &'a [f32],
    pub tokens: &'a [usize],
    pub mel: &'a [f32],
    /// What the artifact detectors concluded about this chunk. Carried per
    /// chunk rather than aggregated because chunks fail independently: one
    /// hallucinated tail among six clean ones is the case worth seeing.
    pub inspection: postprocess::Inspection,
    /// True when generation stopped at the token cap rather than at a stop
    /// token, so this chunk is cut off mid-sentence. Per chunk, for the same
    /// reason the inspection is: chunks truncate independently.
    pub hit_token_cap: bool,
    /// Where this chunk's words fall inside *this chunk's* audio.
    ///
    /// `start` is zero and `end` is the chunk's own duration: a streamed chunk
    /// is its own result and cannot know what preceded it, so reporting anything
    /// else would be a guess about the caller's playback. A caller stitching the
    /// stream adds the offsets with [`ChunkTiming::shifted`].
    pub timing: ChunkTiming,
}

/// Every value the checkpoint declares, resolved once at load.
///
/// The algorithm layer of `loudkit.config.AlgorithmConfig`: each field here
/// changes the audio, which is why they are read from the manifest rather than
/// defaulted, and why [`EngineConfig::describe`] hashes them into the
/// fingerprint.
pub struct EngineConfig {
    pub decode_mode: String,
    /// The raised-cosine ramp on both edges of every rendered window, in seconds.
    /// The shipped value is the unset one: only another value enters the
    /// canonical form, so the fingerprint moves only when the ramp does.
    pub edge_fade_seconds: f64,
    pub recipe_version: String,
    /// Guidance mode, kept rather than only checked at load.
    ///
    /// The value decides whether the estimator runs once or twice per step, so
    /// it belongs in everything this port compares, its own fingerprint
    /// included.
    pub guidance: String,
    pub guidance_rate: f64,
    pub sample_rate: usize,
    /// Speech tokens per second. Algorithm-bearing: it converts a token count
    /// into the seconds of speech an over-window refusal reports.
    pub token_rate_hz: f64,
    pub euler_steps: usize,
    /// Explicit Euler time grid, or `None` for the cosine schedule.
    pub euler_grid: Option<Vec<f64>>,
    pub start_speech: usize,
    pub stop_speech: usize,
    pub speech_vocab_size: usize,
    pub window: WindowConfig,
    pub sampling: sampler::Config,
    pub chunking: ChunkConfig,
    /// The artifact detectors. They remove tokens, so they change the
    /// audio and are read from the manifest for the same reason the joins
    /// are: a backend that re-guesses where a chunk ended cuts somewhere
    /// else, and the difference is a hallucinated word that either does or
    /// does not reach a listener.
    pub postprocess: postprocess::Config,
    /// The funnel's identity: its code version and the digest of the grammar
    /// file this port reads. In the fingerprint because the funnel decides what
    /// string the model is handed, and therefore what it says.
    pub text: TextConfig,
}

impl EngineConfig {
    /// The algorithm a checkpoint declares, read from its manifest.
    ///
    /// The whole reader path in one call, which is what `config.FromManifest`
    /// is in Go and `manifest.algorithm_from` is in Python. Separate from
    /// [`Engine::load_paths_with`] because the values here are a property of
    /// the manifest and nothing else: a test can pin what a released manifest
    /// reads to without a 1.27 GB checkpoint and a directory of graphs beside
    /// it. A fingerprint check has to go through this reader: one against a
    /// configuration written by hand holds even if the reader ignores every key
    /// and answers its defaults.
    ///
    /// # Errors
    ///
    /// Returns an error when the manifest declares an algorithm this port
    /// cannot run, and when a value it carries is one no recipe can use.
    pub fn from_checkpoint(ckpt: &Checkpoint) -> Result<Self, String> {
        let (start, stop, vocab) = ckpt.speech_tokens();
        let (temp, rep, minp, max_new, sil, floor, ratio) = ckpt.sampling();
        let config = EngineConfig {
            decode_mode: ckpt.decode_mode(),
            edge_fade_seconds: ckpt.edge_fade_seconds(),
            text: TextConfig::default(),
            recipe_version: ckpt.recipe_version()?,
            guidance: ckpt.guidance()?,
            guidance_rate: ckpt.guidance_rate(),
            sample_rate: ckpt.sample_rate(),
            token_rate_hz: ckpt.token_rate_hz(),
            euler_steps: ckpt.euler_steps(),
            euler_grid: ckpt.euler_grid(),
            start_speech: start,
            stop_speech: stop,
            speech_vocab_size: vocab,
            window: ckpt.window(),
            chunking: ckpt.chunking(),
            postprocess: ckpt.postprocess()?,
            sampling: sampler::Config {
                temperature: temp,
                repetition_penalty: rep,
                min_p: minp,
                max_new_tokens: max_new,
                silence_token_ids: sil,
                min_tokens_floor: floor,
                min_tokens_text_ratio: ratio,
            },
        };
        // Checked once, where the manifest is read, rather than per utterance,
        // and in the order the reference builds the dataclasses: chunking,
        // sampling and window run as each block is constructed, the cross-field
        // rules in `AlgorithmConfig.__post_init__` after all three. A manifest
        // with two faults must name the same one this port and the reference.
        //
        // A chunking recipe with no character budget makes `split_text` cut
        // nothing and loop forever; the ports read the same manifest key.
        config.chunking.validate()?;
        config.sampling.validate()?;
        config.window.validate()?;
        config.validate()?;
        Ok(config)
    }

    /// `AlgorithmConfig.__post_init__`: the rules that need more than one
    /// field, in the reference's order and its wording.
    ///
    /// Without them a manifest the reference refuses loads here and computes a
    /// fingerprint for an algorithm that cannot run: `n_cfm_timesteps: 0`
    /// makes the flow loop run zero times and renders the prior noise as
    /// audio, `token_rate_hz: 0` divides every duration by zero,
    /// `speech_vocab_size: 0` leaves no vocabulary, a start token past the
    /// embedding table indexes off the end, a start equal to the stop never
    /// stops, and a grid of the wrong length or the wrong direction integrates
    /// the ODE on a schedule that is not one.
    ///
    /// # Errors
    ///
    /// One `ValueError` of the reference's, as a `String`.
    fn validate(&self) -> Result<(), String> {
        if self.guidance == "single_path" && self.guidance_rate != 0.0 {
            return Err("guidance_rate must be 0.0 in single_path mode".to_string());
        }
        // The `cfg_dual_path` half of the reference's pair has no call site
        // here: this port refuses that mode outright at `Checkpoint::guidance`,
        // because it would render single-path audio under a manifest that says
        // otherwise.
        if self.euler_steps < 1 {
            return Err(format!("euler_steps must be >= 1: {}", self.euler_steps));
        }
        self.validate_numeric_core()?;
        if let Some(grid) = &self.euler_grid {
            let want = self.euler_steps + 1;
            if grid.len() != want {
                return Err(format!(
                    "euler_grid has {} points, expected {want}",
                    grid.len()
                ));
            }
            if !grid.windows(2).all(|p| p[1] > p[0]) {
                return Err("euler_grid must be strictly increasing".to_string());
            }
            // `grid` is non-empty here: a zero-length grid cannot have
            // `euler_steps + 1` points, and `euler_steps` is at least one.
            let first = grid[0];
            let last = grid[grid.len() - 1];
            if first.abs() > 1e-6 || (last - 1.0).abs() > 1e-6 {
                return Err("euler_grid must run from 0.0 to 1.0".to_string());
            }
        }
        // The three token budgets have to agree, or a chunk overruns the render
        // window mid-stream, after earlier chunks have already played.
        let window = self.window.max_speech_tokens;
        if self.chunking.enabled && self.chunking.max_tokens > window {
            return Err(format!(
                "chunking.max_tokens {} exceeds the render window ({window}): every chunk \
                 would be sized past what the renderer accepts, and the refusal would land \
                 mid-stream, after audio had already been delivered",
                self.chunking.max_tokens
            ));
        }
        if self.sampling.max_new_tokens > window {
            return Err(format!(
                "sampling.max_new_tokens {} exceeds the render window ({window}): generation \
                 is allowed to produce more speech than the renderer will accept, so a long \
                 utterance fails after it has been generated rather than before",
                self.sampling.max_new_tokens
            ));
        }
        Ok(())
    }

    /// `AlgorithmConfig._validate_numeric_core`: the single-field ranges, split
    /// out in the reference and split out here for the same reason.
    fn validate_numeric_core(&self) -> Result<(), String> {
        // The reference exempts the one value its reader turns into the unset
        // sentinel, today's 20 ms, and that value is inside this range: so the
        // rule is the range, on whatever the ramp actually is. An absent key is
        // the legacy 5 ms, which is inside it too.
        if !(self.edge_fade_seconds >= 0.001 && self.edge_fade_seconds <= 0.05) {
            return Err(format!(
                "edge_fade_seconds must be in [0.001, 0.05]: {}",
                repr_float(self.edge_fade_seconds)
            ));
        }
        // Every duration this engine reports is `samples / sample_rate`, so a
        // zero divides by zero. A rate is the one manifest field whose
        // wrongness is not caught by any shape.
        if self.sample_rate == 0 {
            return Err("sample_rate must be > 0: 0".to_string());
        }
        if self.token_rate_hz <= 0.0 {
            return Err(format!(
                "token_rate_hz must be > 0: {}",
                repr_float(self.token_rate_hz)
            ));
        }
        if self.speech_vocab_size < 1 {
            return Err(format!(
                "speech_vocab_size must be >= 1: {}",
                self.speech_vocab_size
            ));
        }
        for (name, value) in [
            ("start_speech_token", self.start_speech),
            ("stop_speech_token", self.stop_speech),
        ] {
            if value >= self.speech_vocab_size {
                return Err(format!(
                    "{name} must be in [0, {}): {value}",
                    self.speech_vocab_size
                ));
            }
        }
        if self.start_speech == self.stop_speech {
            return Err(format!(
                "start_speech_token and stop_speech_token must differ: both are {}",
                self.start_speech
            ));
        }
        Ok(())
    }

    /// The algorithm half of [`Engine::describe`], field for field the same as
    /// `AlgorithmConfig.describe()` in Python: including the float spelling,
    /// which goes through `fingerprint::repr_float` so that `1.0`
    /// prints as Python prints it rather than as `1`.
    #[must_use]
    pub fn describe(&self) -> String {
        let guidance = if self.guidance == "single_path" {
            self.guidance.clone()
        } else {
            format!("cfg@{}", crate::fingerprint::repr_float(self.guidance_rate))
        };
        // An empty explicit grid falls back to the cosine schedule in
        // `windowing::time_grid`, so it is reported as cosine here too.
        let grid = if self.euler_grid.as_ref().is_some_and(|g| !g.is_empty()) {
            "explicit"
        } else {
            "cosine"
        };
        let window = match self.window.static_length {
            Some(n) => n.to_string(),
            None => "ragged".to_string(),
        };
        format!(
            "algo[{}] {} {guidance} euler={}({grid}) temp={} rep={} min_p={} sil={} win={window}",
            crate::fingerprint::fingerprint(self),
            self.recipe_version,
            self.euler_steps,
            crate::fingerprint::repr_float(self.sampling.temperature),
            crate::fingerprint::repr_float(self.sampling.repetition_penalty),
            crate::fingerprint::repr_float(self.sampling.min_p),
            self.sampling.silence_token_ids.len(),
        )
    }
}

/// Identifies the text funnel: what its code does, and what data it reads.
///
/// The digest is of *this* crate's embedded `numbers.json`, so a copy that has
/// drifted from the reference produces a different fingerprint and the engine
/// refuses to start rather than silently speaking something else.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextConfig {
    /// Bumped when the funnel's passes change what they emit for text they
    /// already handled. A new language or table moves `grammar` on its own.
    pub recipe: String,
    pub grammar: String,
}

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            recipe: "funnel-6".to_string(),
            grammar: crate::numbers::grammar_digest(),
        }
    }
}

/// The exported graphs, under the release's own names. One list, read by the
/// loader and by the export record it checks itself against, so the two cannot
/// name different sets.
const COND_GRAPH: &str = "t3_cond.onnx";
const PREFILL_GRAPH: &str = "t3_prefill.onnx";
const STEP_GRAPH: &str = "t3_step.onnx";
const PAIR_STEP_GRAPH: &str = "t3_pair_step.onnx";
const HEAD2_GRAPH: &str = "t3_head2.onnx";
const ENCODER_GRAPH: &str = "flow_encoder.onnx";
const ESTIMATOR_GRAPH: &str = "flow_estimator.onnx";
const HIFT_GRAPH: &str = "vocoder.onnx";

/// The set a decode mode loads, in the order it opens them.
fn graph_names(fused: bool) -> &'static [&'static str] {
    if fused {
        &[
            COND_GRAPH,
            PREFILL_GRAPH,
            PAIR_STEP_GRAPH,
            HEAD2_GRAPH,
            ENCODER_GRAPH,
            ESTIMATOR_GRAPH,
            HIFT_GRAPH,
        ]
    } else {
        &[
            COND_GRAPH,
            PREFILL_GRAPH,
            STEP_GRAPH,
            ENCODER_GRAPH,
            ESTIMATOR_GRAPH,
            HIFT_GRAPH,
        ]
    }
}

/// Refuse an embedding table a live id can index past the end of.
///
/// `flat` is the table flattened row-major at `HIDDEN_DIM` per row, `max_id` the
/// largest id the engine will ever look up in it. The message names the file the
/// caller can change, not the index that would otherwise have blown up.
fn embedding_fits(which: &str, max_id: usize, flat: usize, source: &str) -> Result<(), String> {
    let rows = flat / HIDDEN_DIM;
    if max_id >= rows {
        return Err(format!(
            "{source}: {which} token id {max_id} is past the end of the \
             checkpoint's {which} embedding table ({rows} rows)"
        ));
    }
    Ok(())
}

/// The decode loop's speech-side bookkeeping: which learned positional row each
/// generated token reads, and which ids the repetition penalty has already
/// seen.
///
/// Both numbers depend on the carried prefix, so both have to agree with the
/// prefill directly above the loop. Keeping them in one value the loop
/// delegates to is what stops one site from disagreeing with the other.
pub struct DecodeState {
    prefix_len: usize,
    seen: Vec<bool>,
}

impl DecodeState {
    /// Start a decode with the prefix already spoken.
    ///
    /// The repetition mask is seeded from the prefix because those ids were
    /// spoken moments ago and the penalty exists to keep them from being
    /// repeated. An empty mask agrees with Python
    /// (`backends/onnx_backend.py:337`) and JS only while every carried token
    /// is a manifest silence id, which the penalty exempts anyway; a carried
    /// tail holding a repeated non-silence token near a decision boundary
    /// picks a different token.
    ///
    /// Ids are the caller's to keep in the codebook (`carry_from` is the
    /// guard), so an out-of-range one panics here rather than being dropped,
    /// the same as it would in the embedding lookup a few lines later.
    #[must_use]
    pub fn new(prefix: &[usize], speech_vocab_size: usize) -> Self {
        let mut seen = vec![false; speech_vocab_size];
        for token in prefix {
            seen[*token] = true;
        }
        Self {
            prefix_len: prefix.len(),
            seen,
        }
    }

    /// The learned speech positional row for the `i`-th carried prefix token.
    ///
    /// BOS holds row 0, so a prefix of length P owns rows 1..=P. This is the
    /// prefill's layout and therefore the authority [`Self::position`] has to
    /// continue from.
    #[must_use]
    pub fn prefix_position(i: usize) -> usize {
        i + 1
    }

    /// The learned speech positional row for the `step`-th generated token.
    ///
    /// `step + 1` re-reads a row the prefill just wrote for a carried token
    /// and never reaches P+1 and above at all: it is right only for a chunk
    /// with no prefix, so it matches Python on a single window and diverges on
    /// every long-form chunk after the first. Python
    /// (`backends/onnx_backend.py:353`) and Swift (`TokenGenerator.swift:586`)
    /// index the same way this does.
    #[must_use]
    pub fn position(&self, step: usize) -> usize {
        self.prefix_len + step + 1
    }

    /// The repetition mask, as the sampler takes it.
    #[must_use]
    pub fn seen(&self) -> &[bool] {
        &self.seen
    }

    /// Record a generated token as spoken.
    pub fn mark(&mut self, token: usize) {
        self.seen[token] = true;
    }
}

impl Engine {
    /// Open a release: a directory holding one, or a repo id such as
    /// `"loudreader/loudr-1"`, which is fetched into [`crate::hub::cache_dir`]
    /// the first time and read from there afterwards.
    ///
    /// The onnxruntime shared library is found here (`ORT_DYLIB_PATH`,
    /// `LOUDKIT_ONNXRUNTIME_LIB`, then the usual install paths), so nothing
    /// above this has to name it.
    ///
    /// # Errors
    ///
    /// Everything [`crate::hub::Bundle::open`] and [`Engine::load_paths_with`]
    /// return.
    pub fn load(dir_or_repo: impl AsRef<std::path::Path>) -> Result<Self, String> {
        Self::load_with(dir_or_repo, &ExecutionConfig::default())
    }

    /// [`Engine::load`] on a named execution provider.
    ///
    /// # Errors
    ///
    /// Everything [`Engine::load`] returns.
    pub fn load_with(
        dir_or_repo: impl AsRef<std::path::Path>,
        execution: &ExecutionConfig,
    ) -> Result<Self, String> {
        let bundle = crate::hub::Bundle::open(dir_or_repo)?;
        let mut engine = Self::load_paths_with(
            &bundle.checkpoint.to_string_lossy(),
            &bundle.onnx_dir.to_string_lossy(),
            &bundle.tokenizer.to_string_lossy(),
            execution,
        )?;
        engine.bundle = Some(bundle);
        Ok(engine)
    }

    /// Build an engine from a layout of your own: the checkpoint, the
    /// directory of exported graphs and `tokenizer.json`, named one by one.
    /// [`Engine::voice`] and [`Engine::enroll_wav`] need a release and are
    /// not available on an engine opened this way.
    ///
    /// # Errors
    ///
    /// Everything [`Engine::load_paths_with`] returns.
    pub fn load_paths(
        ckpt_path: &str,
        onnx_dir: &str,
        tokenizer_path: &str,
    ) -> Result<Self, String> {
        Self::load_paths_with(
            ckpt_path,
            onnx_dir,
            tokenizer_path,
            &ExecutionConfig::default(),
        )
    }

    /// [`Engine::load_paths`] on a named execution provider.
    ///
    /// # Errors
    ///
    /// Returns an error when the onnxruntime shared library cannot be found,
    /// when an asset is missing or malformed, when the manifest declares an
    /// algorithm this port cannot run, and when `execution.onnx_provider`
    /// names a provider this build or this machine does not offer, which is a
    /// refusal rather than a fallback.
    pub fn load_paths_with(
        ckpt_path: &str,
        onnx_dir: &str,
        tokenizer_path: &str,
        execution: &ExecutionConfig,
    ) -> Result<Self, String> {
        // First: a provider this build cannot register is not a missing-file
        // problem and must not be reported as one.
        let execution = Execution::resolve(execution, &crate::execution::available_providers()?)?;
        crate::execution::locate_runtime()?;
        let ckpt = Checkpoint::open(ckpt_path)?;
        // Refused before any graph is loaded: an algorithm this port cannot
        // run is not a missing-file problem and must not be reported as one.
        ckpt.guidance()?;
        let GeneratorTables {
            text_emb,
            speech_emb,
            text_pos,
            speech_pos,
        } = ckpt.generator_tables()?;
        let (spk_weight, spk_bias) = ckpt.speaker_affine()?;
        let frontend = Frontend::load(tokenizer_path)?;
        let (start, stop, vocab) = ckpt.speech_tokens();
        // The tokenizer and the checkpoint are separate files a caller can pair
        // by hand: `LOUDKIT_TOKENIZER` exists precisely so they can. A vocabulary
        // wider than the checkpoint's table makes `text_row` read past the end of
        // it, which is an out-of-bounds panic several seconds into a synthesis
        // rather than a refusal naming the file that is wrong. The same reasoning
        // as `loudkit.models.generator.check_manifest_sizes`, one layer out: this
        // port reads the table itself and can measure it.
        embedding_fits(
            "text",
            frontend.max_token_id(),
            text_emb.len(),
            tokenizer_path,
        )?;
        // Same read, one table over: `speech_row` is indexed by the manifest's
        // own start/stop ids and by sampler draws below `speech_vocab_size`, so
        // the manifest can outrun its own weights.
        embedding_fits(
            "speech",
            start.max(stop).max(vocab.saturating_sub(1)),
            speech_emb.len(),
            ckpt_path,
        )?;
        // The algorithm, read from the manifest and checked, before anything
        // expensive loads.
        let config = EngineConfig::from_checkpoint(&ckpt)?;

        let fused = config.decode_mode == "fusion_mtp2";
        // The graphs are one export of one checkpoint or they are not a set,
        // and a mixed set speaks this checkpoint's tokens through another
        // one's renderer. Checked before any graph is opened, so a refusal
        // costs no session.
        let d = PathBuf::from(onnx_dir);
        crate::export::check_export_record(
            &d,
            Path::new(ckpt_path),
            &ckpt.manifest,
            &crate::fingerprint::fingerprint(&config),
            config.euler_steps,
            graph_names(fused),
        )?;
        let open = |name: &str| -> Result<Session, String> {
            crate::execution::session_builder(execution.provider(), name)?
                .commit_from_file(d.join(name))
                .map_err(ort_err)
        };
        let cond = open(COND_GRAPH)?;
        let prefill = open(PREFILL_GRAPH)?;
        let step = open(if fused { PAIR_STEP_GRAPH } else { STEP_GRAPH })?;
        let head2 = if fused {
            Some(open(HEAD2_GRAPH)?)
        } else {
            None
        };
        let fusion = ckpt.fusion_weights()?;
        let n_layers = ckpt
            .manifest
            .get("llama_config")
            .and_then(|v| v.get("num_hidden_layers"))
            .and_then(serde_json::Value::as_u64)
            .filter(|v| *v > 0)
            .ok_or("manifest requires positive llama_config.num_hidden_layers")?
            as usize;
        let encoder = open(ENCODER_GRAPH)?;
        let estimator = open(ESTIMATOR_GRAPH)?;
        let vocoder = open(HIFT_GRAPH)?;

        Ok(Engine {
            config,
            bundle: None,
            execution,
            frontend,
            text_emb,
            speech_emb,
            text_pos,
            speech_pos,
            spk_weight,
            spk_bias,
            cond,
            prefill,
            step,
            head2,
            fusion,
            n_layers,
            encoder,
            estimator,
            vocoder,
        })
    }

    /// This engine's algorithm fingerprint.
    ///
    /// Comparable with `AlgorithmConfig.fingerprint()` in Python and
    /// `AlgorithmConfig.fingerprint()` in Swift. Two engines whose fingerprints
    /// differ are computing different things, whatever their audio sounds like.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        crate::fingerprint::fingerprint(&self.config)
    }

    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// One line naming both layers. Log it on every run.
    ///
    /// Same two halves as `Engine.describe()` in Python: the algorithm, which
    /// every port must agree on, then the execution, which they are free to
    /// differ on. The provider is in the second half because that is the
    /// question a benchmark row and a bug report both have to answer: a
    /// figure taken on CUDA and a figure taken on MLAS are not comparable, and
    /// nothing else in the output said which one ran.
    #[must_use]
    pub fn describe(&self) -> String {
        format!("{} | {}", self.config.describe(), self.execution.describe())
    }

    fn release(&self, what: &str) -> Result<&crate::hub::Bundle, String> {
        self.bundle.as_ref().ok_or_else(|| {
            format!(
                "this engine was opened with Engine::load_paths, which has no \
                 release to take {what} from"
            )
        })
    }

    /// The names [`Engine::voice`] will accept, sorted.
    ///
    /// # Errors
    ///
    /// For an engine opened with [`Engine::load_paths`].
    pub fn voices(&self) -> Result<Vec<String>, String> {
        Ok(self.release("voices")?.voices())
    }

    /// One shipped voice by name, such as `"joe"`.
    ///
    /// # Errors
    ///
    /// For a name the release does not ship, listing the ones it does, and
    /// everything [`voice::load`] returns.
    pub fn voice(&self, name: &str) -> Result<voice::Profile, String> {
        let path = self.release("voices")?.voice_path(name)?;
        voice::load(&path.to_string_lossy())
    }

    /// Clone a voice from a WAV recording: 16-bit PCM or 32-bit float, any
    /// rate, any channel count. `language` is what the voice reads in by
    /// default; `""` means `"en"`.
    ///
    /// The three enrollment graphs are not in a plain fetch. An engine
    /// loaded by repo id fetches them into its own cache directory the first
    /// time; one loaded from a directory needs a fetch made with `cloning`.
    ///
    /// # Errors
    ///
    /// For a directory without the enrollment graphs, and everything
    /// [`crate::wav::Audio::read_wav`] and [`Engine::enroll`] return.
    pub fn enroll_wav(
        &self,
        path: impl AsRef<std::path::Path>,
        name: &str,
        language: &str,
    ) -> Result<voice::Profile, String> {
        let clip = crate::wav::Audio::read_wav(path)?;
        self.enroll(&clip.samples, clip.sample_rate as usize, name, language)
    }

    /// [`Engine::enroll_wav`] for samples already in memory: mono `f32` in
    /// `[-1, 1]` at any rate.
    ///
    /// # Errors
    ///
    /// Everything [`crate::hub::Bundle::fetch_cloning`] and
    /// [`crate::enroll::Enroller::enroll`] return.
    pub fn enroll(
        &self,
        samples: &[f32],
        sample_rate: usize,
        name: &str,
        language: &str,
    ) -> Result<voice::Profile, String> {
        let bundle = self.release("the enrollment graphs")?;
        bundle.fetch_cloning()?;
        let mut enroller = crate::enroll::Enroller::load_with(
            &bundle.onnx_dir,
            &ExecutionConfig {
                onnx_provider: self.execution.requested(),
            },
        )?;
        Ok(enroller
            .enroll(samples, sample_rate)?
            .profile(name, language))
    }

    /// The execution provider these sessions actually run on.
    #[must_use]
    pub fn execution(&self) -> Execution {
        self.execution
    }

    pub fn encode(&self, text: &str, language: &str) -> Result<Vec<usize>, String> {
        // The speech funnel the shipped Swift/Python engines run before
        // tokenising (SpeechText.prepared), Polish English-respelling
        // included; see speechtext.
        let spoken = crate::speechtext::speech_text(text, language);
        self.frontend.encode(&spoken, language)
    }

    // ------------------------------------------------------------ generator

    fn cond_row(&mut self, v: &voice::Profile) -> Result<Vec<f32>, String> {
        let speaker = Array2::from_shape_vec((1, 256), v.speaker_embedding.clone())
            .map_err(|e| e.to_string())?;
        let prompt: Vec<i64> = v.cond_prompt_tokens.clone();
        let prompt =
            Array2::from_shape_vec((1, prompt.len()), prompt).map_err(|e| e.to_string())?;
        // dead axis on these weights; fed the training constant
        let emotion = Array2::from_shape_vec((1, 1), vec![crate::voice::EMOTION_NEUTRAL])
            .map_err(|e| e.to_string())?;
        let mut inputs: HashMap<String, DynValue> = HashMap::new();
        inputs.insert(
            "speaker_emb".to_string(),
            Value::from_array(speaker)
                .map_err(|e| e.to_string())?
                .into_dyn(),
        );
        inputs.insert(
            "prompt_tokens".to_string(),
            Value::from_array(prompt)
                .map_err(|e| e.to_string())?
                .into_dyn(),
        );
        inputs.insert(
            "emotion".to_string(),
            Value::from_array(emotion)
                .map_err(|e| e.to_string())?
                .into_dyn(),
        );
        let outputs = self.cond.run(inputs).map_err(|e| e.to_string())?;
        let (_, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        Ok(data.to_vec())
    }

    fn text_row(&self, text_tokens: &[usize]) -> Vec<f32> {
        let mut framed = vec![windowing::START_TEXT_TOKEN];
        framed.extend_from_slice(text_tokens);
        framed.push(windowing::STOP_TEXT_TOKEN);
        let mut out = vec![0.0f32; framed.len() * HIDDEN_DIM];
        for (i, id) in framed.iter().enumerate() {
            let base = id * HIDDEN_DIM;
            for j in 0..HIDDEN_DIM {
                out[i * HIDDEN_DIM + j] =
                    self.text_emb[base + j] + self.text_pos[i * HIDDEN_DIM + j];
            }
        }
        out
    }

    fn speech_row(&self, token: usize, position: usize) -> Vec<f32> {
        let sbase = token * HIDDEN_DIM;
        let pbase = position * HIDDEN_DIM;
        self.speech_emb[sbase..sbase + HIDDEN_DIM]
            .iter()
            .zip(&self.speech_pos[pbase..pbase + HIDDEN_DIM])
            .map(|(e, p)| e + p)
            .collect()
    }

    fn pair_row(&self, first: usize, second: usize, position: usize) -> Vec<f32> {
        let [w0, b0, w2, b2] = self.fusion.as_ref().expect("fused decoder weights");
        let a = &self.speech_emb[first * HIDDEN_DIM..(first + 1) * HIDDEN_DIM];
        let b = &self.speech_emb[second * HIDDEN_DIM..(second + 1) * HIDDEN_DIM];
        let input = Array1::from_iter(a.iter().chain(b).copied());
        let w0 = ndarray::ArrayView2::from_shape((HIDDEN_DIM, 2 * HIDDEN_DIM), w0)
            .expect("fusion shape");
        let mut hidden = w0.dot(&input);
        for (x, bias) in hidden.iter_mut().zip(b0) {
            *x += bias;
            *x = 0.5 * *x * (1.0 + libm::erff(*x / std::f32::consts::SQRT_2));
        }
        let w2 =
            ndarray::ArrayView2::from_shape((HIDDEN_DIM, HIDDEN_DIM), w2).expect("fusion shape");
        let mut fused = w2.dot(&hidden).to_vec();
        for i in 0..HIDDEN_DIM {
            fused[i] = 0.5 * (a[i] + b[i])
                + (fused[i] + b2[i])
                + self.speech_pos[position * HIDDEN_DIM + i];
        }
        fused
    }

    fn prefill_embeds(
        &mut self,
        text_tokens: &[usize],
        v: &voice::Profile,
        prefix: &[usize],
    ) -> Result<(Vec<f32>, usize), String> {
        let cond = self.cond_row(v)?;
        let text = self.text_row(text_tokens);
        let bos = self.speech_row(self.config.start_speech, 0);
        let mut rows: Vec<&[f32]> = vec![&cond, &text, &bos];
        let pe;
        if !prefix.is_empty() {
            let prefix_len = prefix.len();
            pe = if self.fusion.is_some() {
                prefix
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .enumerate()
                    .flat_map(|(i, p)| self.pair_row(p[0], p[1], i + 1))
                    .collect()
            } else {
                let mut v = vec![0.0f32; prefix_len * HIDDEN_DIM];
                for (i, tok) in prefix.iter().enumerate() {
                    let sbase = tok * HIDDEN_DIM;
                    let pbase = DecodeState::prefix_position(i) * HIDDEN_DIM;
                    for j in 0..HIDDEN_DIM {
                        v[i * HIDDEN_DIM + j] =
                            self.speech_emb[sbase + j] + self.speech_pos[pbase + j];
                    }
                }
                v
            };
            rows.push(&pe);
        }
        let total: usize = rows.iter().map(|r| r.len()).sum();
        let mut out = Vec::with_capacity(total);
        for r in rows {
            out.extend_from_slice(r);
        }
        Ok((out, total / HIDDEN_DIM))
    }

    /// Run the autoregressive loop to the stop token or cap.
    pub fn generate(
        &mut self,
        text_tokens: &[usize],
        v: &voice::Profile,
        s: &mut Sampler,
        max_new_tokens: Option<usize>,
        mut should_cancel: Option<&mut dyn FnMut() -> bool>,
        prefix: &[usize],
    ) -> Result<Vec<usize>, String> {
        let prefix = if self.fusion.is_some() {
            &prefix[..prefix.len() - prefix.len() % 2]
        } else {
            prefix
        };
        let cap = max_new_tokens.unwrap_or(self.config.sampling.max_new_tokens);
        let floor = windowing::eos_floor(
            text_tokens.len(),
            self.config.sampling.min_tokens_floor,
            self.config.sampling.min_tokens_text_ratio,
        );
        let stop = self.config.stop_speech;

        // `prefix` holds speech tokens from the preceding chunk: fed in as
        // context and NOT returned. `prefill_embeds` accepts it, and a caller
        // that passes `&[]` restarts its pitch contour at every chunk
        // boundary: the audible stutter the prefix exists to remove.
        let (embeds, prefill_len) = self.prefill_embeds(text_tokens, v, prefix)?;
        let embeds = Array3::from_shape_vec((1, prefill_len, HIDDEN_DIM), embeds)
            .map_err(|e| e.to_string())?;
        let positions = Array1::from_iter(0..prefill_len as i64);
        let mut inputs: HashMap<String, DynValue> = HashMap::new();
        inputs.insert(
            "embeds".to_string(),
            Value::from_array(embeds)
                .map_err(|e| e.to_string())?
                .into_dyn(),
        );
        inputs.insert(
            "positions".to_string(),
            Value::from_array(positions)
                .map_err(|e| e.to_string())?
                .into_dyn(),
        );
        let mut outputs = self.prefill.run(inputs).map_err(|e| e.to_string())?;
        let logits_last;
        let mut hidden = Vec::new();
        {
            let (_, all_logits) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| e.to_string())?;
            logits_last = all_logits[(prefill_len - 1) * self.config.speech_vocab_size..].to_vec();
        }
        if self.head2.is_some() {
            hidden = outputs["hidden"]
                .try_extract_tensor::<f32>()
                .map_err(ort_err)?
                .1
                .to_vec();
        }
        let mut kv = collect_kv(&mut outputs, "kv_", self.n_layers)?;
        drop(outputs);

        let mut state = DecodeState::new(prefix, self.config.speech_vocab_size);
        let mut out = Vec::new();
        let mut logits_last = logits_last;
        while out.len() < cap {
            let step = out.len();
            if should_cancel.as_mut().is_some_and(|c| c()) {
                // Token-level barge-in: the partial row is discarded, not returned.
                return Err(crate::error::CANCELLED.to_string());
            }
            // Taken, not copied: the row is replaced at the end of the step,
            // and every path that leaves before then leaves the loop.
            let mut row = std::mem::take(&mut logits_last);
            if out.len() < floor {
                row[stop] = f32::NEG_INFINITY;
            }
            let token = s.call(&row, step, state.seen());
            out.push(token);
            if token == stop {
                break;
            }
            state.mark(token);

            let mut step_inputs: HashMap<String, DynValue> = HashMap::new();
            if let Some(head2) = self.head2.as_mut() {
                if out.len() >= cap {
                    break;
                }
                if should_cancel.as_mut().is_some_and(|c| c()) {
                    return Err(crate::error::CANCELLED.to_string());
                }
                let h =
                    ndarray::Array2::from_shape_vec((1, HIDDEN_DIM), std::mem::take(&mut hidden))
                        .map_err(|e| e.to_string())?;
                let first = Array1::from_vec(vec![token as i64]);
                let second_out = head2
                    .run(ort::inputs![
                        "hidden" => Value::from_array(h).map_err(ort_err)?,
                        "first_id" => Value::from_array(first).map_err(ort_err)?
                    ])
                    .map_err(ort_err)?;
                let mut row = second_out["logits"]
                    .try_extract_tensor::<f32>()
                    .map_err(ort_err)?
                    .1
                    .to_vec();
                if out.len() < floor {
                    row[stop] = f32::NEG_INFINITY;
                }
                let second = s.call(&row, out.len(), state.seen());
                out.push(second);
                if second == stop {
                    break;
                }
                state.mark(second);
                if out.len() >= cap {
                    break;
                }
                let pair = step / 2;
                step_inputs.insert(
                    "pair_ids".to_string(),
                    Value::from_array(
                        ndarray::Array2::from_shape_vec((1, 2), vec![token as i64, second as i64])
                            .map_err(|e| e.to_string())?,
                    )
                    .map_err(ort_err)?
                    .into_dyn(),
                );
                step_inputs.insert(
                    "speech_position".to_string(),
                    Value::from_array(Array1::from_vec(vec![(prefix.len() / 2 + pair + 1) as i64]))
                        .map_err(ort_err)?
                        .into_dyn(),
                );
                step_inputs.insert(
                    "position".to_string(),
                    Value::from_array(Array1::from_vec(vec![(prefill_len + pair) as i64]))
                        .map_err(ort_err)?
                        .into_dyn(),
                );
            } else {
                let emb = Array3::from_shape_vec(
                    (1, 1, HIDDEN_DIM),
                    self.speech_row(token, state.position(step)),
                )
                .map_err(|e| e.to_string())?;
                step_inputs.insert(
                    "embeds".to_string(),
                    Value::from_array(emb).map_err(ort_err)?.into_dyn(),
                );
                step_inputs.insert(
                    "position".to_string(),
                    Value::from_array(Array1::from_vec(vec![(prefill_len + step) as i64]))
                        .map_err(ort_err)?
                        .into_dyn(),
                );
            }
            // The cache the step reads is the previous run's own tensors,
            // handed straight back. Nothing is copied out of a run or into the
            // next one, and the shape the graph reaches is the shape the graph
            // wrote rather than one rebuilt from a length.
            for (i, (past_k, past_v)) in kv.k.drain(..).zip(kv.v.drain(..)).enumerate() {
                step_inputs.insert(format!("past_k_{i}"), past_k);
                step_inputs.insert(format!("past_v_{i}"), past_v);
            }
            let mut step_out = self.step.run(step_inputs).map_err(|e| e.to_string())?;
            {
                let (_, sl) = step_out[0]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| e.to_string())?;
                logits_last = sl.to_vec();
            }
            if self.head2.is_some() {
                hidden = step_out["hidden"]
                    .try_extract_tensor::<f32>()
                    .map_err(ort_err)?
                    .1
                    .to_vec();
            }
            kv = collect_kv(&mut step_out, "present_", self.n_layers)?;
            drop(step_out);
        }
        Ok(out)
    }

    // -------------------------------------------------------------- renderer

    /// Tokens -> mel via the exported encoder + estimator.
    pub fn decode_mel(
        &mut self,
        tokens: &[usize],
        v: &voice::Profile,
        seed: u64,
    ) -> Result<Vec<f32>, String> {
        let framed = windowing::frame_windows(
            &self.config.window,
            self.config.window.pad_token_id,
            &self.config.sampling.silence_token_ids,
            tokens,
            &v.prompt_tokens,
            &v.prompt_mel,
        )?;
        let p_len = self
            .config
            .window
            .static_prompt_tokens
            .unwrap_or(framed.row.len() - framed.n);
        let prompt = framed.row[..p_len].to_vec();
        let query = framed.row[p_len..].to_vec();
        let t_mel = 2 * framed.row.len();

        let prompt = Array2::from_shape_vec((1, p_len), prompt).map_err(|e| e.to_string())?;
        let query = Array2::from_shape_vec((1, query.len()), query).map_err(|e| e.to_string())?;
        let mut inputs: HashMap<String, DynValue> = HashMap::new();
        inputs.insert(
            "prompt_token".to_string(),
            Value::from_array(prompt)
                .map_err(|e| e.to_string())?
                .into_dyn(),
        );
        inputs.insert(
            "speech_tokens".to_string(),
            Value::from_array(query)
                .map_err(|e| e.to_string())?
                .into_dyn(),
        );
        let outputs = self.encoder.run(inputs).map_err(|e| e.to_string())?;
        let (_, mu) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        let mu = mu.to_vec();

        // speaker affine
        let emb = &v.flow_embedding;
        let norm: f64 = emb
            .iter()
            .map(|x| f64::from(*x) * f64::from(*x))
            .sum::<f64>()
            .sqrt();
        let mut spks = vec![0.0f32; MEL_BINS];
        for (i, bias) in self.spk_bias.iter().enumerate() {
            let mut acc = f64::from(*bias);
            for j in 0..emb.len() {
                acc += f64::from(self.spk_weight[i * emb.len() + j]) * f64::from(emb[j]) / norm;
            }
            spks[i] = acc as f32;
        }

        let grid = windowing::time_grid(self.config.euler_steps, self.config.euler_grid.as_deref());
        let mut x = noise::gaussian_field(seed, windowing::FLOW_NOISE_STREAM, MEL_BINS, t_mel);
        let cond = framed.cond.clone();
        for i in 0..grid.len() - 1 {
            let t0 = grid[i];
            let dt = grid[i + 1] - t0;
            let xa = Array3::from_shape_vec((1, MEL_BINS, t_mel), x.clone())
                .map_err(|e| e.to_string())?;
            let mua = Array3::from_shape_vec((1, MEL_BINS, t_mel), mu.clone())
                .map_err(|e| e.to_string())?;
            let ta = Array1::from_vec(vec![t0 as f32]);
            let spa =
                Array2::from_shape_vec((1, MEL_BINS), spks.clone()).map_err(|e| e.to_string())?;
            let conda = Array3::from_shape_vec((1, MEL_BINS, t_mel), cond.clone())
                .map_err(|e| e.to_string())?;
            let mut vin: HashMap<String, DynValue> = HashMap::new();
            vin.insert(
                "x".to_string(),
                Value::from_array(xa).map_err(|e| e.to_string())?.into_dyn(),
            );
            vin.insert(
                "mu".to_string(),
                Value::from_array(mua)
                    .map_err(|e| e.to_string())?
                    .into_dyn(),
            );
            vin.insert(
                "t".to_string(),
                Value::from_array(ta).map_err(|e| e.to_string())?.into_dyn(),
            );
            vin.insert(
                "spks".to_string(),
                Value::from_array(spa)
                    .map_err(|e| e.to_string())?
                    .into_dyn(),
            );
            vin.insert(
                "cond".to_string(),
                Value::from_array(conda)
                    .map_err(|e| e.to_string())?
                    .into_dyn(),
            );
            let vout = self.estimator.run(vin).map_err(|e| e.to_string())?;
            let (_, vdata) = vout[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| e.to_string())?;
            for j in 0..x.len() {
                x[j] += (dt as f32) * vdata[j];
            }
        }

        let n = framed.n;
        let prompt_frames = framed.prompt_frames;
        let out_len = 2 * n;
        let mut mel = vec![0.0f32; MEL_BINS * out_len];
        for b in 0..MEL_BINS {
            for f in 0..out_len {
                mel[b * out_len + f] = x[b * t_mel + prompt_frames + f];
            }
        }
        Ok(mel)
    }

    /// Mel -> audio via the exported HiFT graph.
    pub fn vocode(&mut self, mel: &[f32], seed: u64) -> Result<Vec<f32>, String> {
        let frames = 2 * self.config.window.max_speech_tokens;
        let mel_frames = mel.len() / MEL_BINS;
        let n_frames = mel_frames.min(frames);
        let mut padded = vec![0.0f32; MEL_BINS * frames];
        for b in 0..MEL_BINS {
            for f in 0..n_frames {
                padded[b * frames + f] = mel[b * mel_frames + f];
            }
        }
        let n_samples = frames * UPSAMPLE_PER_FRAME;
        let mut phase = vec![0.0f32; N_HARMONICS];
        let offsets = noise::symmetric_uniforms(
            seed,
            windowing::VOCODER_PHASE_STREAM,
            N_HARMONICS - 1,
            std::f64::consts::PI,
        );
        phase[1..].copy_from_slice(&offsets);
        let noise_data = noise::gaussian_field(
            seed,
            windowing::VOCODER_NOISE_STREAM,
            N_HARMONICS,
            n_samples,
        );

        let mela =
            Array3::from_shape_vec((1, MEL_BINS, frames), padded).map_err(|e| e.to_string())?;
        let phasea =
            Array3::from_shape_vec((1, N_HARMONICS, 1), phase).map_err(|e| e.to_string())?;
        let noisea = Array3::from_shape_vec((1, N_HARMONICS, n_samples), noise_data)
            .map_err(|e| e.to_string())?;
        let mut inputs: HashMap<String, DynValue> = HashMap::new();
        inputs.insert(
            "mel".to_string(),
            Value::from_array(mela)
                .map_err(|e| e.to_string())?
                .into_dyn(),
        );
        inputs.insert(
            "phase".to_string(),
            Value::from_array(phasea)
                .map_err(|e| e.to_string())?
                .into_dyn(),
        );
        inputs.insert(
            "noise".to_string(),
            Value::from_array(noisea)
                .map_err(|e| e.to_string())?
                .into_dyn(),
        );
        let outputs = self.vocoder.run(inputs).map_err(|e| e.to_string())?;
        let (_, wav) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        Ok(wav[..n_frames * UPSAMPLE_PER_FRAME].to_vec())
    }

    /// The one path that produces speech tokens.
    ///
    /// Single-shot and streaming both go through it so they cannot drift: the
    /// generation ceiling, the stop-token observation and the artifact
    /// detectors are applied once, here, rather than twice and eventually
    /// differently.
    ///
    /// `is_terminal` says whether this chunk ends the passage. A continuation
    /// chunk has no sentence end, so its stop peak means nothing and its
    /// trailing pause is the sentence's rhythm rather than dead air, so the
    /// detectors that cut a tail are told and hold off.
    ///
    /// `should_cancel` is polled at every decode step, the same as
    /// [`Engine::generate`]; `None` means no cancellation.
    fn generate_inspected(
        &mut self,
        text_ids: &[usize],
        v: &voice::Profile,
        seed: u64,
        prefix: &[usize],
        is_terminal: bool,
        should_cancel: Option<&mut dyn FnMut() -> bool>,
    ) -> Result<Generated, String> {
        let pp = self.config.postprocess.clone();
        let floor = windowing::eos_floor(
            text_ids.len(),
            self.config.sampling.min_tokens_floor,
            self.config.sampling.min_tokens_text_ratio,
        );
        let mut cap = self.config.sampling.max_new_tokens;
        if pp.mode != postprocess::Mode::Off {
            // Applied during generation, not after it: the tokens past the
            // ceiling cost real time on a device and are certain to be
            // discarded. It only ever stops a row that was going to run away.
            cap = cap.min(postprocess::ceiling_for(
                text_ids.len(),
                &pp,
                self.config.window.max_speech_tokens,
            ));
        }

        let silence: HashSet<usize> = self
            .config
            .sampling
            .silence_token_ids
            .iter()
            .copied()
            .collect();
        // Selective re-roll. A window whose verdict is unfixable, dropout
        // (content missing) or suspect (certainly wrong, nowhere to cut), is
        // regenerated from a derived seed, up to retry_max_attempts times.
        // Only condemned windows pay; the ladder is a pure function of the
        // caller's seed, so the same seed still gives the same audio.
        let mut should_cancel = should_cancel;
        let mut gen;
        let mut verdict;
        // True when the row stopped at the ceiling rather than at a stop
        // token: the utterance is cut off mid-sentence. Computed here: where
        // `ended` and the effective cap are both in hand, and carried out,
        // because a caller cannot recompute it after the specials are stripped
        // and the cap is forgotten.
        let mut hit_cap;
        let mut hit_window;
        // When the ladder exhausts with every attempt condemned, the attempt
        // that ships is the *best* seen, not the last: fewest tokens in the
        // true-silence set: integer and portable, like the detectors.
        // Measured: on the worst voices 30% of condemned fires exhaust the
        // ladder, and keeping the last attempt shipped rows worse than the
        // first. The render census gates the count where the checkpoint
        // carries one; the configured silence list is the fallback.
        let dead_air: HashSet<usize> = if pp.silence_render_ids.is_empty() {
            self.config
                .sampling
                .silence_token_ids
                .iter()
                .copied()
                .collect()
        } else {
            pp.silence_render_ids.iter().copied().collect()
        };
        let mut best: Option<(usize, Generated)> = None;
        let mut attempt = 0usize;
        loop {
            let attempt_seed = if attempt == 0 {
                seed
            } else {
                derive_seed(seed, RETRY_STREAM_BASE + attempt as u64)
            };
            let mut s = Sampler::new(self.config.sampling.clone(), attempt_seed);
            if pp.mode != postprocess::Mode::Off {
                s.observe_eos(self.config.stop_speech, floor);
            }
            let cancel = should_cancel
                .as_mut()
                .map(|f| &mut **f as &mut dyn FnMut() -> bool);
            let raw = self.generate(text_ids, v, &mut s, Some(cap), cancel, prefix)?;

            // `gen` is what the shipped engine calls a row: every token the
            // model committed to, with the stop marker itself excluded.
            // Indices into it are decode-step indices, which is what makes the
            // observed peak comparable against it, so the detectors run here,
            // before the specials are stripped and free to renumber anything.
            gen = raw;
            let ended = gen.last() == Some(&self.config.stop_speech);
            if ended {
                gen.pop();
            }
            let (peak_at, peak_prob) = s.eos_peak();
            hit_cap = !ended && gen.len() >= cap;
            // The window, asked separately and asked here, where `gen` is still
            // what the model produced. `cap` is `min(max_new_tokens,
            // ceiling_for(...))`, so `hit_cap` cannot tell a filled window from
            // a runaway short text; and the trim below can cut a filled window
            // down to a few tokens, which is how a caller measuring the
            // returned vector saw room to spare.
            hit_window = !ended && gen.len() >= self.config.window.max_speech_tokens;
            verdict = postprocess::inspect(
                &gen,
                &postprocess::Request {
                    text_token_count: text_ids.len(),
                    min_tokens: floor,
                    eos_peak_at: peak_at,
                    eos_peak_prob: peak_prob,
                    ended,
                    is_terminal,
                    hit_ceiling: hit_cap,
                },
                &silence,
                &pp,
            );
            let condemned = verdict.reason == postprocess::Reason::Dropout || verdict.suspect;
            if !condemned || pp.mode == postprocess::Mode::Off {
                break;
            }
            let silence_count = gen.iter().filter(|t| dead_air.contains(t)).count();
            if best.as_ref().is_none_or(|b| silence_count < b.0) {
                // Strict `<`: on a tie the earlier attempt stands, so the
                // ladder stays a pure function of the caller's seed with no
                // dependence on iteration order.
                best = Some((
                    silence_count,
                    Generated {
                        tokens: gen.clone(),
                        verdict,
                        hit_cap,
                        hit_window,
                    },
                ));
            }
            if attempt >= pp.retry_max_attempts {
                let (_, chosen) = best.take().expect("a condemned attempt was recorded above");
                gen = chosen.tokens;
                verdict = chosen.verdict;
                hit_cap = chosen.hit_cap;
                hit_window = chosen.hit_window;
                break;
            }
            attempt += 1;
        }
        if pp.mode == postprocess::Mode::Trim && verdict.keep < gen.len() {
            gen.truncate(verdict.keep);
        }
        let tokens = gen
            .into_iter()
            .filter(|t| *t < self.config.start_speech)
            .collect();
        Ok(Generated {
            tokens,
            verdict,
            hit_cap,
            hit_window,
        })
    }

    /// Render text that fits one model window, and refuse text that does not.
    ///
    /// [`Engine::synthesize`] is the call for any length; this one is for the
    /// conformance harness and for a caller who wants the refusal.
    ///
    /// # Errors
    /// For text longer than one window, for an `Options` value out of range,
    /// [`crate::error::CANCELLED`] when `o.should_cancel` returned true, and
    /// for anything the graphs refuse.
    pub fn synthesize_window(
        &mut self,
        text: &str,
        v: &voice::Profile,
        o: &Options,
    ) -> Result<Synthesis, String> {
        let (seed, speed) = (o.seed, o.speed);
        let language = o.language.as_deref();
        let previous_tokens = o.previous_tokens.as_deref();
        let mut cancel = o.cancel();
        let should_cancel: Option<&mut dyn FnMut() -> bool> =
            cancel.as_mut().map(|c| c as &mut dyn FnMut() -> bool);
        // Both refused here, before the six seconds of generation they would
        // otherwise be discovered after.
        timestretch::validate_speed(speed)?;
        carry_from(previous_tokens, 0, self.config.start_speech)?;
        let prefix = carry_pair_aligned(
            previous_tokens.unwrap_or_default(),
            self.config.chunking.prefix_tokens,
            self.fusion.is_some(),
        );
        let language = resolve_language(language, v);
        // The funnelled text, kept rather than thrown away inside `encode`: it
        // is what was tokenised, and therefore the text the timeline must carry.
        let spoken = speechtext::speech_text(text, language);
        let text_ids = self.frontend.encode(&spoken, language)?;
        // A single window is the whole passage, so it is terminal.
        let Generated {
            tokens,
            hit_cap: hit_token_cap,
            hit_window: hit_window_cap,
            ..
        } = self.generate_inspected(&text_ids, v, seed, &prefix, true, should_cancel)?;
        // Refused rather than truncated. Shipping the window's worth of audio
        // with the rest of the text never spoken is silent data loss: what a
        // caller hears is fluent, complete-sounding and short. This crate's
        // README documents the error.
        //
        // The *window*, not the cap: `hit_token_cap` is also set by the
        // postprocess length ceiling, which stops a short text that ran away.
        // That text fitted and there is nothing to split, so refusing there
        // would hand the caller an error they can do nothing about.
        //
        // Read from the generation rather than from the returned vector:
        // postprocess trims, so a window that filled and was then cut back
        // measures short there. A trimmed length cannot say whether generation
        // hit the cap; only the flag captured before postprocess can.
        if hit_window_cap {
            return Err(format!(
                "the text did not fit one {}-token window and its tail was not \
                 spoken. Use synthesize, which splits at sentence boundaries \
                 and joins the audio.",
                self.config.window.max_speech_tokens
            ));
        }
        // Discarded, not rendered: every port polls here, between the token
        // phase and the render.
        if cancel.as_mut().is_some_and(|c| c()) {
            return Err(crate::error::CANCELLED.to_string());
        }
        let mel = self.decode_mel(&tokens, v, derive_seed(seed, 1))?;
        if cancel.as_mut().is_some_and(|c| c()) {
            return Err(crate::error::CANCELLED.to_string());
        }
        let audio = self.vocode(&mel, derive_seed(seed, 2))?;
        let audio = timestretch::time_stretch(&audio, self.config.sample_rate, speed);
        let audio = timestretch::fade_edges(
            audio,
            self.config.sample_rate,
            self.config.edge_fade_seconds,
        );
        if cancel.as_mut().is_some_and(|c| c()) {
            return Err(crate::error::CANCELLED.to_string());
        }
        // One window is one chunk, and it starts at zero. Measured on the
        // stretched waveform, the one the caller receives, so there is no
        // `1/speed` correction to apply anywhere, and applying one would
        // double-count.
        let chunks = timing::timeline(
            &[ChunkSpan {
                text: spoken,
                samples: audio.len(),
                tokens: tokens.len(),
            }],
            self.config.sample_rate,
        );
        Ok(Synthesis {
            audio,
            sample_rate: self.config.sample_rate,
            tokens,
            mel,
            chunks,
            hit_token_cap,
        })
    }

    /// Speak `text` chunk by chunk, calling `on_chunk` as each becomes ready.
    ///
    /// The difference from [`Engine::synthesize`] is delivery, not
    /// synthesis: time to first audio is set by the first chunk rather than by
    /// the whole passage, which is what lets a reading app start playing a
    /// sentence while the rest is still being made.
    ///
    /// A callback rather than an `Iterator`, because the engine is borrowed
    /// mutably for the whole render: an iterator would have to hand out items
    /// borrowing from something it is still using. Return `false` from
    /// `on_chunk` to stop. `should_cancel` (or `o.should_cancel` when it is
    /// `None`) is polled on every decode step; when it returns true the chunk
    /// being generated is discarded and the stream ends with `Ok(())`: the
    /// chunks already delivered are the partial, and the caller flipped the
    /// flag.
    ///
    /// Splitting it across windows:
    ///
    /// [`Engine::synthesize`] delegates to the same splitter, and
    /// [`Engine::synthesize_window`] is the call that refuses anything longer
    /// than one window. Two things make the joins match Python's rather than
    /// merely existing:
    ///
    /// * **Per-chunk seeds.** Chunk 0 draws the caller's seed itself, so a
    ///   text that fits one window reads the same here as through
    ///   [`Engine::synthesize`]; every later chunk draws
    ///   `derive(seed, 16 + index)`, so its audio does not depend on how many
    ///   came before it and stopping early cannot change what was already
    ///   produced.
    /// * **Prefix carry.** The last `chunking.prefix_tokens` speech tokens of a
    ///   chunk are fed into the next as context and dropped from its output.
    ///
    /// `language` is `None` for "the voice's own language", see
    /// [`resolve_language`]. Resolved once here, before splitting, so every
    /// chunk of a passage is read in the same language.
    ///
    /// `speed` stretches each chunk independently, which is the same
    /// independence the seeds and the prefix already have: a chunk's audio must
    /// not depend on how many came before it, or a listener who stopped early
    /// would have heard something different from one who did not.
    ///
    /// `previous_tokens` seeds the carry, so the first chunk of *this* call is
    /// conditioned on the tail of a *previous* one. It is the same conditioning
    /// the joins inside a passage already use: the carry variable below simply
    /// starts non-empty, which is why a request boundary stops being audible
    /// without a second mechanism existing to maintain.
    ///
    /// # Errors
    /// For a `speed` outside the range, for a `previous_tokens` entry that is not
    /// an acoustic speech token, for text that funnels away to nothing, and for
    /// anything the graphs refuse.
    pub fn stream(
        &mut self,
        text: &str,
        v: &voice::Profile,
        o: &Options,
        should_cancel: Option<&mut dyn FnMut() -> bool>,
        on_chunk: &mut dyn FnMut(Chunk<'_>) -> bool,
    ) -> Result<(), String> {
        // One `&mut dyn FnMut` for the whole loop, reborrowed per call.
        // `Option<&mut dyn FnMut>` is not Copy, so consulting it before a chunk
        // *and* handing it to `generate` in the same iteration cannot both move
        // it; collapsing to a single reference (with a no-op stand-in when the
        // caller passed none) makes the reborrow the only thing happening.
        let mut from_options = o.cancel();
        let mut never = || false;
        let cancel: &mut dyn FnMut() -> bool = match (should_cancel, from_options.as_mut()) {
            (Some(callback), _) => callback,
            (None, Some(callback)) => callback,
            (None, None) => &mut never,
        };
        match self.chunks(text, v, o, cancel, on_chunk) {
            Err(e) if crate::error::is_cancelled(&e) => Ok(()),
            other => other,
        }
    }

    /// [`Engine::stream`], returning [`crate::error::CANCELLED`] where the
    /// flag stopped it.
    fn chunks(
        &mut self,
        text: &str,
        v: &voice::Profile,
        o: &Options,
        cancel: &mut dyn FnMut() -> bool,
        on_chunk: &mut dyn FnMut(Chunk<'_>) -> bool,
    ) -> Result<(), String> {
        let (seed, speed) = (o.seed, o.speed);
        let language = o.language.as_deref();
        let previous_tokens = o.previous_tokens.as_deref();
        // Refused before the split, not per chunk: the answer cannot change
        // between chunks, and the caller should hear about a bad value before
        // the first one is generated rather than after.
        timestretch::validate_speed(speed)?;
        let language = resolve_language(language, v);

        // The funnel runs on the whole text BEFORE splitting: Polish
        // respelling changes the length ("download" -> "dałnloud"), so a budget
        // computed first would be a budget for text the engine never speaks.
        let prepared = speechtext::speech_text(text, language);
        let chunks = chunking::split_text(&prepared, &self.config.chunking);
        if chunks.is_empty() {
            return Err("nothing to speak".to_string());
        }

        let prefix_len = self.config.chunking.prefix_tokens;
        // Seeded from the caller's history rather than starting empty. Every
        // chunk after the first still takes the one before it: the caller's
        // tokens seed the carry, they do not replace it.
        carry_from(previous_tokens, 0, self.config.start_speech)?;
        let mut carry: Vec<usize> = carry_pair_aligned(
            previous_tokens.unwrap_or_default(),
            prefix_len,
            self.fusion.is_some(),
        );

        // A work queue rather than a walk over `chunks`: a chunk the window
        // could not hold is replaced, in place, by its two halves. The decision
        // needs a generated window -- the overrun is a fact about this voice and
        // this text together, and nothing before generation knows it -- so a
        // queue is the only structure that lets one entry become two after the
        // fact.
        //
        // Both halves keep the ORIGINAL chunk's index, so a repair cannot move
        // the seed of any later chunk. Seeds only: chunk k+1 is conditioned on
        // the tail of chunk k, which after a repair comes from the second half,
        // so later audio in THIS passage does move. What the index buys is that
        // the change stops at this passage.
        struct Part {
            text: String,
            index: usize,
            seed: u64,
            terminal: bool,
            /// False on a half, so a half that still overruns ships as it is.
            /// One did, measured through the engine over the 51 passages
            /// carrying a cap hit, and its audio ended on a 0.42 s tail -- it
            /// finished its clause rather than being cut. An unbounded split is
            /// a new way to fail.
            splittable: bool,
        }
        let mut queue: Vec<Part> = chunks
            .iter()
            .enumerate()
            .map(|(i, c)| Part {
                text: c.clone(),
                index: i,
                seed: seed_for_chunk(seed, i),
                terminal: i == chunks.len() - 1,
                splittable: true,
            })
            .collect();

        let mut qi = 0;
        while qi < queue.len() {
            if cancel() {
                return Err(crate::error::CANCELLED.to_string());
            }
            let (index, chunk_seed, terminal, splittable) = {
                let p = &queue[qi];
                (p.index, p.seed, p.terminal, p.splittable)
            };
            let chunk = queue[qi].text.clone();
            let ids = self.frontend.encode(&chunk, language)?;
            // Only the last chunk ends the passage.
            let Generated {
                tokens: chunk_tokens,
                verdict,
                hit_cap: chunk_hit_cap,
                hit_window: chunk_filled_window,
            } =
                self.generate_inspected(&ids, v, chunk_seed, &carry, terminal, Some(&mut *cancel))?;
            // The window has to be what stopped it, not the length-proportional
            // ceiling: generate_inspected caps at min(max_new_tokens,
            // ceiling_for(...)), and the second fires when a short text runs
            // away. Halving a runaway gives two runaways with smaller ceilings.
            // Measured before the trim, for the reason `synthesize` states.
            if chunk_filled_window
                && splittable
                && self.config.chunking.cap_resplit == chunking::WORD_CAP_RESPLIT
            {
                if let Some((first, second)) = chunking::split_in_half(&chunk) {
                    // Discard this window and do the two halves instead. The
                    // second draws from a stream of its own off the chunk seed:
                    // the flow takes 1, the vocoder 2, the retry ladder 8 up.
                    queue.splice(
                        qi..=qi,
                        [
                            Part {
                                text: first,
                                index,
                                seed: chunk_seed,
                                terminal: false,
                                splittable: false,
                            },
                            Part {
                                text: second,
                                index,
                                seed: derive_seed(chunk_seed, RESPLIT_STREAM),
                                terminal,
                                splittable: false,
                            },
                        ],
                    );
                    continue;
                }
            }
            // Discarded, not rendered. The partial tokens belong to speech
            // the listener has already interrupted, and the mel decode plus
            // vocode is the larger half of the barge-in latency on an edge
            // device, so running them adds exactly the wait the cancellation
            // exists to remove, and then plays audio nobody asked for.
            // Every port polls here, between the token phase and the render.
            if cancel() {
                return Err(crate::error::CANCELLED.to_string());
            }
            let chunk_mel = self.decode_mel(&chunk_tokens, v, derive_seed(chunk_seed, 1))?;
            if cancel() {
                return Err(crate::error::CANCELLED.to_string());
            }
            let chunk_audio = self.vocode(&chunk_mel, derive_seed(chunk_seed, 2))?;
            // Last stage, after the inspection above rather than before it.
            let chunk_audio =
                timestretch::time_stretch(&chunk_audio, self.config.sample_rate, speed);
            let chunk_audio = timestretch::fade_edges(
                chunk_audio,
                self.config.sample_rate,
                self.config.edge_fade_seconds,
            );

            if cancel() {
                return Err(crate::error::CANCELLED.to_string());
            }
            carry = carry_pair_aligned(&chunk_tokens, prefix_len, self.fusion.is_some());

            // Through `timeline` rather than built by hand, so a streamed chunk
            // and a stitched passage share one piece of arithmetic and cannot
            // come to disagree about where a word falls.
            let timing = timing::timeline(
                &[ChunkSpan {
                    text: chunk.clone(),
                    samples: chunk_audio.len(),
                    tokens: chunk_tokens.len(),
                }],
                self.config.sample_rate,
            )
            .pop()
            .ok_or("timeline dropped the only span it was given")?;

            let keep_going = on_chunk(Chunk {
                index,
                text: &chunk,
                audio: &chunk_audio,
                tokens: &chunk_tokens,
                inspection: verdict,
                hit_token_cap: chunk_hit_cap,
                mel: &chunk_mel,
                timing,
            });
            if !keep_going {
                break;
            }
            qi += 1;
        }
        Ok(())
    }

    /// Speak text of any length as one waveform.
    ///
    /// Exactly [`Engine::stream`] with the chunks concatenated: one loop, so
    /// the streaming and whole-passage paths cannot drift apart. Use `stream`
    /// when you want to start playing before the passage is finished.
    ///
    /// `language` is `None` for "the voice's own language"; left unresolved
    /// here so [`Engine::stream`] resolves it once, on the one path that
    /// renders.
    ///
    /// `speed` is applied per chunk, exactly as [`Engine::stream`] applies it,
    /// so the two paths still produce the same waveform. `previous_tokens`
    /// conditions the *first* chunk; every chunk after it is conditioned on the
    /// one before, as always.
    ///
    /// # Errors
    /// See [`Engine::stream`], plus [`crate::error::CANCELLED`] when
    /// `o.should_cancel` returned true: a passage cut short is never handed
    /// back as a `Synthesis`.
    pub fn synthesize(
        &mut self,
        text: &str,
        v: &voice::Profile,
        o: &Options,
    ) -> Result<Synthesis, String> {
        let mut audio: Vec<f32> = Vec::new();
        let mut tokens: Vec<usize> = Vec::new();
        let mut mel: Vec<f32> = Vec::new();
        let mut spans: Vec<ChunkSpan> = Vec::new();
        let mut hit_token_cap = false;
        let mut from_options = o.cancel();
        let mut never = || false;
        let cancel: &mut dyn FnMut() -> bool = match from_options.as_mut() {
            Some(callback) => callback,
            None => &mut never,
        };
        self.chunks(text, v, o, cancel, &mut |chunk| {
            audio.extend_from_slice(chunk.audio);
            tokens.extend_from_slice(chunk.tokens);
            mel = append_mel_along_time(std::mem::take(&mut mel), chunk.mel);
            // ORed across chunks: one truncated chunk truncates the
            // passage, which is the fact a caller of the joined waveform
            // needs.
            hit_token_cap |= chunk.hit_token_cap;
            spans.push(ChunkSpan {
                text: chunk.text.to_string(),
                samples: chunk.audio.len(),
                tokens: chunk.tokens.len(),
            });
            true
        })?;
        // Rebuilt from the spans rather than shifting each chunk's own timing by
        // a running float: `timeline` accumulates sample offsets as integers, so
        // the joins are exact and every chunk's `end` is the next one's `start`
        // down to the last bit.
        let chunks = timing::timeline(&spans, self.config.sample_rate);
        Ok(Synthesis {
            audio,
            sample_rate: self.config.sample_rate,
            tokens,
            mel,
            chunks,
            hit_token_cap,
        })
    }
}

/// Select a suffix whose boundaries align with complete decoder pairs.
pub fn carry_pair_aligned(tokens: &[usize], wanted: usize, fused: bool) -> Vec<usize> {
    if wanted == 0 {
        return Vec::new();
    }
    let end = if fused {
        tokens.len() - tokens.len() % 2
    } else {
        tokens.len()
    };
    let mut start = end.saturating_sub(wanted);
    if fused {
        start -= start % 2;
    }
    tokens[start..end].to_vec()
}

/// The conditioning context a call inherits from the one before it.
///
/// The same slice the streaming loop takes between two chunks: the last
/// `prefix_tokens`, applied to tokens that came from a different call. There is
/// deliberately no second mechanism: a request boundary and a chunk boundary are
/// the same join, and the reason chunk joins do not stutter is the reason request
/// joins should not either.
///
/// Any length is accepted because only the tail is used, so
/// `previous_tokens = Some(&result_tokens)` is the intended call and a caller
/// should never have to know the prefix length to make it.
///
/// A free function rather than a method so it can be exercised without a
/// checkpoint: `Engine` holds six concrete `ort::session::Session` values and
/// `Engine::load` is its only constructor, so nothing can drive the pipeline
/// without the weights and the runtime library. This slice is the whole of
/// the behaviour, and it is the unit under test.
///
/// Mirrors `Engine._carry_from` in `loudkit.engine`.
///
/// # Errors
/// For an id outside the acoustic codebook. The whole input is checked rather
/// than only the slice that will be used: an id out of range means the sequence
/// was built wrong, and reporting that only when it happens to land in the last
/// six tokens would make the failure depend on the length of the caller's text.
pub fn carry_from(
    previous_tokens: Option<&[usize]>,
    prefix_tokens: usize,
    start_speech: usize,
) -> Result<Vec<usize>, String> {
    let Some(tokens) = previous_tokens else {
        return Ok(Vec::new());
    };
    for token in tokens {
        // Only the upper bound is spelled out. Python and JavaScript check
        // `0 <= id` too because a negative can reach them; here the token type
        // is unsigned, so the lower half of the same guard is the type system's.
        if *token >= start_speech {
            return Err(format!(
                "previous_tokens contains {token}, which is not an acoustic \
                 speech token (expected 0 <= id < {start_speech}). Pass the \
                 token vector from an earlier call; the generator's own control \
                 tokens are already stripped from it."
            ));
        }
    }
    // Not `tokens[len - prefix_tokens..]` unguarded: a zero there is the whole
    // list rather than nothing, which would condition on the entire previous
    // utterance at exactly the setting that means "chunks are independent".
    if prefix_tokens == 0 {
        return Ok(Vec::new());
    }
    let take = prefix_tokens.min(tokens.len());
    Ok(tokens[tokens.len() - take..].to_vec())
}

/// Chunk seeds start here, clear of the per-stage streams (1 = flow,
/// 2 = vocoder). Mirrors `_STREAM_CHUNK` in loudkit.engine.
const CHUNK_STREAM_BASE: usize = 16;

/// Mirrors `_STREAM_RESPLIT` in `loudkit.engine`: the second half of a re-split
/// chunk draws from its own stream off the chunk's seed. Chunk streams run from
/// `CHUNK_STREAM_BASE` upwards with no ceiling, so there is no room above them
/// to claim; deriving off the chunk seed leaves only the values already drawn
/// from it to avoid, which are the flow at 1, the vocoder at 2, and the retry
/// ladder from 8 up.
const RESPLIT_STREAM: u64 = 4096;

/// What a synthesis reads as when neither the caller nor the voice says.
///
/// Reached less often than it looks: `voice::load` defaults a *missing* header
/// key to `"en"`, and Python writes the key, so an empty string only arrives
/// from a `Profile` built in memory or a header hand-edited to `""`. A profile
/// file with no language field inherits nothing: it loads as `"en"`.
const FALLBACK_LANGUAGE: &str = "en";

/// The language chain: the argument, then the voice's recorded language, then
/// `"en"`.
///
/// Without the voice link, Polish text reads through the English
/// frontend: English number words, English abbreviation expansion, no Polish
/// respelling, and nothing in the audio reports the mismatch. A profile
/// records the language of the audio it was enrolled from, so the voice is the
/// better answer than a constant.
///
/// Passing `Some(..)` is how cross-lingual synthesis is requested: an English
/// voice reading Polish text is `Some("pl")`, and the argument always wins over
/// the profile.
///
/// Mirrors `loudkit.engine._resolve_language`. Public here, where Python's is
/// private, because Rust's CLI is a separate crate: it has to report the
/// language a run actually used, and a second copy of the chain there is a
/// second thing to keep in agreement.
#[must_use]
pub fn resolve_language<'a>(language: Option<&'a str>, v: &'a voice::Profile) -> &'a str {
    if let Some(explicit) = language {
        return explicit;
    }
    if v.language.is_empty() {
        return FALLBACK_LANGUAGE;
    }
    &v.language
}

/// Concatenate two row-major `[MEL_BINS, frames]` mels along the TIME axis.
///
/// Appending the flat buffers end to end, the obvious thing, is not
/// concatenation: after the first chunk the next chunk's bin 0 lands after the
/// previous chunk's bin 79, so every row but the first is wrong. The audio is
/// unaffected (it is vocoded per chunk) but the mel is the diagnostic people
/// reach for when two backends disagree, and a mis-shaped one sends them
/// looking in the wrong place.
fn append_mel_along_time(dst: Vec<f32>, src: &[f32]) -> Vec<f32> {
    if dst.is_empty() {
        return src.to_vec();
    }
    let dst_frames = dst.len() / MEL_BINS;
    let src_frames = src.len() / MEL_BINS;
    let frames = dst_frames + src_frames;
    let mut out = vec![0.0f32; MEL_BINS * frames];
    for b in 0..MEL_BINS {
        out[b * frames..b * frames + dst_frames]
            .copy_from_slice(&dst[b * dst_frames..(b + 1) * dst_frames]);
        out[b * frames + dst_frames..(b + 1) * frames]
            .copy_from_slice(&src[b * src_frames..(b + 1) * src_frames]);
    }
    out
}

/// What one inspected generation produced.
///
/// Named fields rather than a tuple: `hit_cap` and `hit_window` are both
/// `bool`, they answer different questions, and the caller that swaps them
/// refuses a passage that fitted or splits one that did not.
struct Generated {
    tokens: Vec<usize>,
    verdict: postprocess::Inspection,
    hit_cap: bool,
    hit_window: bool,
}

/// The decoder's key-value cache between two steps.
///
/// The tensors are the run that produced them, not copies of it: ort hands its
/// outputs back as owned values and takes the same values as the next run's
/// inputs, so a token costs no copy of the cache in either direction.
struct KVCache {
    k: Vec<DynValue>,
    v: Vec<DynValue>,
}

fn collect_kv(
    outputs: &mut ort::session::SessionOutputs,
    prefix: &str,
    n_layers: usize,
) -> Result<KVCache, String> {
    let mut kv = KVCache {
        k: Vec::with_capacity(n_layers),
        v: Vec::with_capacity(n_layers),
    };
    for i in 0..n_layers {
        let (k, v) = (format!("{prefix}k_{i}"), format!("{prefix}v_{i}"));
        kv.k.push(kv_tensor(outputs.remove(&k), &k)?);
        kv.v.push(kv_tensor(outputs.remove(&v), &v)?);
    }
    Ok(kv)
}

/// One KV tensor out of a step or prefill run, by name.
///
/// Takes the looked-up value rather than the outputs: `outputs[name]` is
/// `get(name).unwrap_or_else(|| panic!(...))` in ort, so an onnx dir whose
/// graph names its outputs differently: the user-supplied half of the same
/// mistake as a wrong dtype, reached the caller as a panic. Both halves are
/// this one error now, and the split lets the message be tested without a
/// session, which `SessionOutputs` cannot be built outside ort.
///
/// The dtype is read and thrown away. A cache tensor is never looked at by
/// this crate, only passed back to the graph, but a graph that returns
/// something other than f32 has to be refused where it is named rather than
/// several steps later inside ort.
fn kv_tensor(found: Option<DynValue>, name: &str) -> Result<DynValue, String> {
    let found = found.ok_or_else(|| format!("missing or non-f32 output {name}"))?;
    found
        .try_extract_tensor::<f32>()
        .map_err(|_| format!("missing or non-f32 output {name}"))?;
    Ok(found)
}

/// Where the retry ladder starts drawing.
///
/// Attempt `n` draws `derive_seed(seed, RETRY_STREAM_BASE + n)`: clear of the
/// stage streams (1, 2) and below the chunk streams at
/// [`CHUNK_STREAM_BASE`]. [`crate::postprocess::RETRY_LADDER_HEADROOM`] is how
/// many of those streams the ladder may use.
const RETRY_STREAM_BASE: u64 = 8;

/// The seed chunk `index` of a passage draws.
///
/// Chunk 0 takes the caller's seed itself, so a text that fits one window
/// renders the same through [`Engine::synthesize`] and through
/// [`Engine::synthesize_window`]; every later chunk derives from its own
/// stream, so its audio does not depend on how many chunks came before it.
fn seed_for_chunk(seed: u64, index: usize) -> u64 {
    if index == 0 {
        seed
    } else {
        derive_seed(seed, (CHUNK_STREAM_BASE + index) as u64)
    }
}

/// Mirrors `engine._derive`: one stream id off a caller's seed.
fn derive_seed(seed: u64, stream: u64) -> u64 {
    const PHI: u64 = 0x9e3779b97f4a7c15;
    const PSI: u64 = 0xbf58476d1ce4e5b9;
    seed.wrapping_mul(PHI)
        .wrapping_add(stream.wrapping_mul(PSI))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(language: &str) -> voice::Profile {
        voice::Profile {
            enrolment: "first-10s".to_string(),
            name: "fake".to_string(),
            speaker_embedding: Vec::new(),
            flow_embedding: Vec::new(),
            prompt_tokens: Vec::new(),
            prompt_mel: Vec::new(),
            cond_prompt_tokens: Vec::new(),
            source_sample_rate: 24_000,
            language: language.to_string(),
        }
    }

    /// The obvious call must not be the wrong one.
    ///
    /// Without the voice link, `synthesize("Cześć", polish_voice, seed, ..)`
    /// runs Polish text through the English frontend: the argument alone has no
    /// answer but `"en"`, and a profile's own language, recorded at
    /// enrollment, is only visible through the voice. The chain is argument,
    /// then voice, then `"en"`, and these are its links.
    ///
    /// Tested against the resolver rather than through `synthesize` because
    /// this port has no weight-free engine seam: `Engine` holds six concrete
    /// `ort::session::Session` values and `Engine::load` is the only
    /// constructor, so nothing can drive the pipeline without a checkpoint and
    /// a runtime library. The resolver is the whole of the behaviour under
    /// test.
    #[test]
    fn language_comes_from_the_voice() {
        let polish = profile("pl");
        assert_eq!(resolve_language(None, &polish), "pl");
        assert_eq!(resolve_language(Some("en"), &polish), "en");
        // A hand-built profile can carry an empty language, and an empty
        // language id is not a language: it would tag the text `[]`. A file
        // whose header simply omits the key loads as "en" instead, so it never
        // reaches this branch.
        assert_eq!(resolve_language(None, &profile("")), "en");
    }

    /// A mel is row-major `[MEL_BINS, frames]`. Appending two flat buffers puts
    /// the second chunk's bin 0 after the first chunk's bin 79, so every row but
    /// the first is wrong. The audio is unaffected: each chunk is vocoded on
    /// its own, but the mel is the diagnostic people reach for when two
    /// backends disagree, and a mis-shaped one sends them looking in the wrong
    /// place.
    #[test]
    fn mel_is_concatenated_along_time() {
        fn chunk(frames: usize, offset: usize) -> Vec<f32> {
            let mut m = vec![0.0f32; MEL_BINS * frames];
            for b in 0..MEL_BINS {
                for f in 0..frames {
                    m[b * frames + f] = (b * 1000 + offset + f) as f32;
                }
            }
            m
        }
        let joined = append_mel_along_time(
            append_mel_along_time(Vec::new(), &chunk(3, 0)),
            &chunk(2, 100),
        );
        assert_eq!(joined.len(), MEL_BINS * 5);
        for b in 0..MEL_BINS {
            let want: Vec<f32> = [0, 1, 2, 100, 101]
                .iter()
                .map(|o| (b * 1000 + o) as f32)
                .collect();
            assert_eq!(&joined[b * 5..(b + 1) * 5], &want[..], "row {b}");
        }
    }

    /// The defaults every port shares, seed 0, the voice's language, normal
    /// speed, nothing to continue from.
    #[test]
    fn options_default_to_the_shared_defaults() {
        let o = Options::default();
        assert_eq!(o.seed, 0);
        assert_eq!(o.language, None);
        assert_eq!(o.speed, 1.0);
        assert_eq!(o.previous_tokens, None);
    }

    #[test]
    fn a_synthesis_knows_its_duration() {
        let s = Synthesis {
            audio: vec![0.0; 12_000],
            sample_rate: 24_000,
            tokens: Vec::new(),
            mel: Vec::new(),
            chunks: Vec::new(),
            hit_token_cap: false,
        };
        assert_eq!(s.duration(), 0.5);
    }

    /// A tokenizer wider than the checkpoint refuses at the door, naming it.
    ///
    /// `text_row` indexes the embedding table by raw token id. Paired with a
    /// checkpoint from another release the widest ids read past the end of it,
    /// an out-of-bounds panic several seconds into a synthesis, pointing at
    /// neither of the two files the caller chose.
    #[test]
    fn an_id_past_the_embedding_table_is_refused_at_load() {
        let table = vec![0.0f32; 4 * HIDDEN_DIM];
        assert!(embedding_fits("text", 3, table.len(), "tokenizer.json").is_ok());
        let err = embedding_fits("text", 4, table.len(), "tokenizer.json").unwrap_err();
        assert!(err.contains("tokenizer.json"), "{err}");
        assert!(err.contains("4 rows"), "{err}");
    }

    /// A graph whose outputs are named differently is an error, not a panic.
    ///
    /// The onnx dir is user-supplied and need not be this checkpoint's export.
    /// ort's `outputs[name]` panics on a name it does not hold, so the missing
    /// half of the mistake has to be read the same way as the wrong-dtype half,
    /// which returns a readable error.
    #[test]
    fn a_missing_kv_output_is_an_error() {
        let err = kv_tensor(None, "present_k_0").unwrap_err();
        assert_eq!(err, "missing or non-f32 output present_k_0");
    }
}

#[cfg(test)]
mod derive_seed_tests {
    /// The shipping derivation against the shared fixture.
    ///
    /// The assertion calls `derive_seed`, the function every chunk seed in a
    /// real render goes through, rather than declaring `PHI` and `PSI` here
    /// and recomputing the product: a test that carries its own copy of the
    /// arithmetic stays green when the constant in `derive_seed` changes.
    ///
    /// A unit test rather than an integration one because `derive_seed` is
    /// private and should stay private: making it public to be testable would
    /// grow the crate's surface to serve its own test.
    #[test]
    fn seed_derivation_matches_the_shared_fixture() {
        let Some(v) = crate::shared_fixture("vectors.json") else {
            return;
        };
        let cases = v["seeds"]["derivation"]
            .as_array()
            .expect("seeds.derivation");
        assert!(
            !cases.is_empty(),
            "seeds.derivation is empty; nothing was compared"
        );
        for c in cases {
            let seed = c["seed"].as_u64().unwrap();
            let stream = c["stream"].as_u64().unwrap();
            let want =
                u64::from_str_radix(c["derived"].as_str().unwrap().trim_start_matches("0x"), 16)
                    .unwrap();
            assert_eq!(
                super::derive_seed(seed, stream),
                want,
                "seed {seed} stream {stream}"
            );
        }
    }

    /// The headroom the postprocess validator refuses against is the real gap.
    ///
    /// `retry_max_attempts` is bounded by
    /// [`crate::postprocess::RETRY_LADDER_HEADROOM`], which lives one layer
    /// down and cannot see these two constants. Moving either of them without
    /// moving it would let the ladder derive a seed from a stream a chunk
    /// already owns, and the re-roll would render that chunk's audio.
    #[test]
    fn the_retry_ladder_fits_below_the_chunk_streams() {
        let base = super::RETRY_STREAM_BASE;
        let chunks = super::CHUNK_STREAM_BASE as u64;
        let top = base + crate::postprocess::RETRY_LADDER_HEADROOM as u64;
        assert!(
            top <= chunks,
            "retry streams {base}..{top} must stay below the chunk streams at {chunks}"
        );
    }
}
