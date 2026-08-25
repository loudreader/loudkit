//! The full synthesis pipeline over the exported ONNX graphs, fp32, no torch.
//! A bit-parity port of the Python/JS/Go engines: same text, voice and seed
//! give the same tokens and the same render band.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use ndarray::{Array1, Array2, Array3, Array4};
use ort::session::Session;
use ort::value::{DynValue, Value};

use crate::checkpoint::Checkpoint;
use crate::chunking::{self, ChunkConfig};
use crate::execution::{Execution, ExecutionConfig};
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
const N_LAYERS: usize = 16;
const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 64;
const HIDDEN_DIM: usize = 1024;

/// The speech synthesis result: audio samples, speech tokens, mel frames, the
/// sample rate, where each chunk landed in the audio, and whether generation
/// hit the token cap.
///
/// The timeline is second-to-last because it was the newest field and a caller
/// that does not want it writes one `_` — the shape the tutorial's snippet
/// already had. Its entries are adjacent and in order: chunk *k*'s `end` is the
/// same `f64` as chunk *k+1*'s `start`, and the last `end` is the whole
/// duration. A single-window render is one entry covering everything. Chunk
/// boundaries are exact; the per-word times inside them are an estimate — see
/// [`crate::timing`] before building anything that depends on those.
///
/// The trailing bool is `hit_token_cap`: true when generation stopped at the
/// token cap rather than at a stop token, so the reading is probably truncated.
/// Truncation is not an error — the audio is real, it is just incomplete — so it
/// travels as a value rather than an `Err`, and a caller must be able to report
/// it. For `synthesize_long` it is ORed across chunks: one truncated chunk
/// truncates the passage.
type SynthResult = (
    Vec<f32>,
    Vec<usize>,
    Vec<f32>,
    usize,
    Vec<ChunkTiming>,
    bool,
);

pub struct Engine {
    config: EngineConfig,
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
    encoder: Session,
    estimator: Session,
    vocoder: Session,
}

/// The algorithm values the engine reads from the manifest.
/// One rendered chunk, handed to an [`Engine::stream`] callback as soon as it
/// exists.
pub struct Chunk<'a> {
    /// Zero-based position in the split, which is also what its seed was
    /// derived from.
    pub index: usize,
    /// What this chunk was asked to say, *after* the speech funnel — the text
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

pub struct EngineConfig {
    pub recipe_version: String,
    /// Guidance mode, kept rather than only checked at load.
    ///
    /// It was validated on the way in and then thrown away, which meant the
    /// value that decides whether the estimator runs once or twice was not part
    /// of anything this port could compare — including its own fingerprint.
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
    /// The funnel's identity — its code version and the digest of the grammar
    /// file this port reads. In the fingerprint because the funnel decides what
    /// string the model is handed, and therefore what it says.
    pub text: TextConfig,
}

impl EngineConfig {
    /// The algorithm half of [`Engine::describe`], field for field the same as
    /// `AlgorithmConfig.describe()` in Python — including the float spelling,
    /// which goes through [`crate::fingerprint::repr_float`] so that `1.0`
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
            recipe: "funnel-2".to_string(),
            grammar: crate::numbers::grammar_digest(),
        }
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
/// Both numbers depend on the carried prefix, and both were wrong here in the
/// same way — the loop was written as if the prefix did not exist, contradicting
/// the prefill directly above it. Keeping them in one value the loop delegates
/// to is what stops one site from disagreeing with the other again.
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
    /// (`backends/onnx_backend.py:337`) and JS only while every carried token is
    /// a manifest silence id, which the penalty exempts anyway; the first
    /// carried tail holding a repeated non-silence token near a decision
    /// boundary picks a different token.
    ///
    /// Ids are the caller's to keep in the codebook — `carry_from` is the guard
    /// — so an out-of-range one panics here rather than being dropped, the same
    /// as it would in the embedding lookup a few lines later.
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
    /// `step + 1` re-reads a row the prefill just wrote for a carried token and
    /// never reaches P+1 and above at all. It is right only for a chunk with no
    /// prefix, which is why a single-window call matched Python and every
    /// long-form chunk after the first did not. Python
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
    /// Build an engine from a checkpoint, an onnx graph dir and a tokenizer,
    /// on whatever execution provider `auto` finds.
    ///
    /// # Errors
    ///
    /// Everything [`Engine::load_with`] returns.
    pub fn load(ckpt_path: &str, onnx_dir: &str, tokenizer_path: &str) -> Result<Self, String> {
        Self::load_with(
            ckpt_path,
            onnx_dir,
            tokenizer_path,
            &ExecutionConfig::default(),
        )
    }

    /// Build an engine on a named execution provider.
    ///
    /// # Errors
    ///
    /// Returns an error when an asset is missing or malformed, when the
    /// manifest declares an algorithm this port cannot run, and when
    /// `execution.onnx_provider` names a provider this build or this machine
    /// does not offer — see [`crate::execution`] for why that is a refusal
    /// rather than a fallback.
    pub fn load_with(
        ckpt_path: &str,
        onnx_dir: &str,
        tokenizer_path: &str,
        execution: &ExecutionConfig,
    ) -> Result<Self, String> {
        // First, for the same reason the guidance check is early: a provider
        // this build cannot register is not a missing-file problem and must not
        // be reported as one. All six sessions then run on the one provider
        // this resolves to.
        let execution = Execution::resolve(execution, &crate::execution::available_providers()?)?;
        let ckpt = Checkpoint::open(ckpt_path)?;
        // Refused before any graph is loaded: an algorithm this port cannot
        // run is not a missing-file problem and must not be reported as one.
        ckpt.guidance()?;
        let (text_emb, speech_emb, text_pos, speech_pos) = ckpt.generator_tables()?;
        let (spk_weight, spk_bias) = ckpt.speaker_affine()?;
        let frontend = Frontend::load(tokenizer_path)?;
        let (start, stop, vocab) = ckpt.speech_tokens();
        // The tokenizer and the checkpoint are separate files a caller can pair
        // by hand — `LOUDKIT_TOKENIZER` exists precisely so they can. A vocabulary
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
        let (temp, rep, minp, max_new, sil, floor, ratio) = ckpt.sampling();
        // Python and Swift refuse a non-positive cap; the other three took it
        // and decoded nothing, which reaches a caller as silence they have to
        // diagnose rather than an error they can read. A cap of zero is not a
        // configuration, it is a typo in a manifest. Checked here rather than
        // in `sampling()`, which returns a tuple and cannot fail.
        if max_new == 0 {
            return Err("max_new_tokens must be positive: 0".to_string());
        }
        // Python refuses a manifest with a non-positive `sample_rate` and the other four
        // took it: every duration this engine reports is `samples / sample_rate`, so a
        // zero divides by zero and a negative reports negative seconds. A rate is the one
        // manifest field whose wrongness is not caught by any shape.
        if ckpt.sample_rate() == 0 {
            return Err("sample_rate must be > 0: 0".to_string());
        }
        let window = ckpt.window();
        let config = EngineConfig {
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
            window,
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
        // Checked once, at the door, rather than per utterance — and before
        // anything expensive loads. A chunking recipe with no character budget
        // makes `split_text` cut nothing and loop forever; Python has refused
        // it since d8742aa and the ports read the same manifest key.
        config.chunking.validate()?;

        let d = PathBuf::from(onnx_dir);
        let open = |name: &str| -> Result<Session, String> {
            crate::execution::session_builder(execution.provider(), name)?
                .commit_from_file(d.join(name))
                .map_err(ort_err)
        };
        let cond = open("t3_cond.onnx")?;
        let prefill = open("t3_prefill.onnx")?;
        let step = open("t3_step.onnx")?;
        let encoder = open("flow_encoder.onnx")?;
        let estimator = open("flow_estimator.onnx")?;
        let vocoder = open("vocoder.onnx")?;

        Ok(Engine {
            config,
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
    /// question a benchmark row and a bug report both have to answer — a
    /// figure taken on CUDA and a figure taken on MLAS are not comparable, and
    /// nothing else in the output said which one ran.
    #[must_use]
    pub fn describe(&self) -> String {
        format!("{} | {}", self.config.describe(), self.execution.describe())
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
            pe = {
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
        // boundary — the audible stutter the prefix exists to remove.
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
        let outputs = self.prefill.run(inputs).map_err(|e| e.to_string())?;
        let logits_last;
        let mut kv;
        {
            let (_, all_logits) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| e.to_string())?;
            logits_last = all_logits[(prefill_len - 1) * self.config.speech_vocab_size..].to_vec();
            kv = collect_kv(&outputs, "kv_")?;
        }
        drop(outputs);

        let mut state = DecodeState::new(prefix, self.config.speech_vocab_size);
        let mut out = Vec::new();
        let mut logits_last = logits_last;
        for step in 0..cap {
            if let Some(c) = should_cancel.as_mut() {
                if c() {
                    break; // token-level barge-in, mirroring the Python engine
                }
            }
            let mut row = logits_last.clone();
            if out.len() < floor {
                row[stop] = f32::NEG_INFINITY;
            }
            let token = s.call(&row, step, state.seen());
            out.push(token);
            if token == stop {
                break;
            }
            state.mark(token);

            let emb = self.speech_row(token, state.position(step));
            let emb = Array3::from_shape_vec((1, 1, HIDDEN_DIM), emb).map_err(|e| e.to_string())?;
            let pos = Array1::from_vec(vec![(prefill_len + step) as i64]);
            let mut step_inputs: HashMap<String, DynValue> = HashMap::new();
            step_inputs.insert(
                "embeds".to_string(),
                Value::from_array(emb)
                    .map_err(|e| e.to_string())?
                    .into_dyn(),
            );
            step_inputs.insert(
                "position".to_string(),
                Value::from_array(pos)
                    .map_err(|e| e.to_string())?
                    .into_dyn(),
            );
            for i in 0..N_LAYERS {
                let seq = kv.k[i].len() / (KV_HEADS * HEAD_DIM);
                let k = Array4::from_shape_vec((1, KV_HEADS, seq, HEAD_DIM), kv.k[i].clone())
                    .map_err(|e| e.to_string())?;
                let vv = Array4::from_shape_vec((1, KV_HEADS, seq, HEAD_DIM), kv.v[i].clone())
                    .map_err(|e| e.to_string())?;
                step_inputs.insert(
                    format!("past_k_{i}"),
                    Value::from_array(k).map_err(|e| e.to_string())?.into_dyn(),
                );
                step_inputs.insert(
                    format!("past_v_{i}"),
                    Value::from_array(vv).map_err(|e| e.to_string())?.into_dyn(),
                );
            }
            let step_out = self.step.run(step_inputs).map_err(|e| e.to_string())?;
            {
                let (_, sl) = step_out[0]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| e.to_string())?;
                logits_last = sl.to_vec();
                kv = collect_kv(&step_out, "present_")?;
            }
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

    /// Full pipeline: text -> tokens -> mel -> audio. `should_cancel` is
    /// polled at every decode step, same as [`Engine::generate`]; pass `None`
    /// for no cancellation.
    /// The one path that produces speech tokens.
    ///
    /// Single-shot and streaming both go through it so they cannot drift: the
    /// generation ceiling, the stop-token observation and the artifact
    /// detectors are applied once, here, rather than twice and eventually
    /// differently.
    ///
    /// `is_terminal` says whether this chunk ends the passage. A continuation
    /// chunk has no sentence end, so its stop peak means nothing and its
    /// trailing pause is the sentence's rhythm rather than dead air — the
    /// detectors that cut a tail are told so and hold off.
    fn generate_inspected(
        &mut self,
        text_ids: &[usize],
        v: &voice::Profile,
        seed: u64,
        prefix: &[usize],
        is_terminal: bool,
        should_cancel: Option<&mut dyn FnMut() -> bool>,
    ) -> Result<(Vec<usize>, postprocess::Inspection, bool), String> {
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
        // Selective re-roll: a window whose verdict is unfixable — dropout
        // (content missing) or suspect (certainly wrong, nowhere to cut) — is
        // regenerated from a derived seed, up to retry_max_attempts times.
        // Only condemned windows pay; the ladder is a pure function of the
        // caller's seed, so the same seed still gives the same audio.
        let mut should_cancel = should_cancel;
        let mut gen;
        let mut verdict;
        // True when the row stopped at the ceiling rather than at a stop
        // token: the utterance is cut off mid-sentence. Computed here — where
        // `ended` and the effective cap are both in hand — and carried out,
        // because a caller cannot recompute it after the specials are stripped
        // and the cap is forgotten.
        let mut hit_cap;
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
            // observed peak comparable against it — so the detectors run here,
            // before the specials are stripped and free to renumber anything.
            gen = raw;
            let ended = gen.last() == Some(&self.config.stop_speech);
            if ended {
                gen.pop();
            }
            let (peak_at, peak_prob) = s.eos_peak();
            hit_cap = !ended && gen.len() >= cap;
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
            if !condemned || pp.mode == postprocess::Mode::Off || attempt >= pp.retry_max_attempts {
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
        Ok((tokens, verdict, hit_cap))
    }

    /// Speak `text` in `v`.
    ///
    /// `language` is `None` for "the voice's own language" — see
    /// [`resolve_language`]. Pass `Some(..)` to read text in a language the
    /// voice was not enrolled in; that is what cross-lingual synthesis is, and
    /// the argument always wins.
    ///
    /// `speed` is playback speed in `[0.5, 2.0]`; greater than one is faster and
    /// the pitch does not move. `1.0` is the bypass — the vocoder's own samples,
    /// untouched. It is applied last, after the artifact detectors have inspected
    /// the render, because those detectors judge pacing as duration per token and
    /// a stretch applied first would move every number they compare against. It
    /// is a change to the *delivery*, not to the reading, which is also why it is
    /// not part of the algorithm fingerprint: it is an execution input like the
    /// seed.
    ///
    /// `previous_tokens` are the speech tokens this utterance continues from —
    /// the token vector returned by the call before it. The window is then
    /// conditioned on their tail exactly as an interior chunk is conditioned on
    /// its predecessor, which is what stops a second request from restarting the
    /// pitch contour like a fresh sentence. Only the last
    /// `chunking.prefix_tokens` are used, so passing a whole previous result is
    /// the intended usage and costs nothing. `None` is byte-for-byte the
    /// behaviour before the parameter existed.
    ///
    /// # Errors
    /// For a `speed` outside the range, for a `previous_tokens` entry that is not
    /// an acoustic speech token, and for anything the graphs refuse.
    // Over clippy's seven. Every one of these is a distinct execution input the
    // caller chooses per call, and folding them into an options struct would
    // give this port a signature none of the other four has — the tutorials, the
    // arity check in `tests/test_docs.py` and anyone reading two bindings side
    // by side all key off the parameter list being the same list everywhere.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize(
        &mut self,
        text: &str,
        v: &voice::Profile,
        seed: u64,
        language: Option<&str>,
        speed: f64,
        previous_tokens: Option<&[usize]>,
        should_cancel: Option<&mut dyn FnMut() -> bool>,
    ) -> Result<SynthResult, String> {
        // Both refused here, before the six seconds of generation they would
        // otherwise be discovered after.
        timestretch::validate_speed(speed)?;
        let prefix = carry_from(
            previous_tokens,
            self.config.chunking.prefix_tokens,
            self.config.start_speech,
        )?;
        let language = resolve_language(language, v);
        // The funnelled text, kept rather than thrown away inside `encode`: it
        // is what was tokenised, and therefore the text the timeline must carry.
        let spoken = speechtext::speech_text(text, language);
        let text_ids = self.frontend.encode(&spoken, language)?;
        // A single window is the whole passage, so it is terminal.
        let (tokens, _, hit_token_cap) =
            self.generate_inspected(&text_ids, v, seed, &prefix, true, should_cancel)?;
        let mel = self.decode_mel(&tokens, v, derive_seed(seed, 1))?;
        let audio = self.vocode(&mel, derive_seed(seed, 2))?;
        let audio = timestretch::time_stretch(&audio, self.config.sample_rate, speed);
        // One window is one chunk, and it starts at zero. Measured on the
        // stretched waveform — the one the caller receives — so there is no
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
        Ok((
            audio,
            tokens,
            mel,
            self.config.sample_rate,
            chunks,
            hit_token_cap,
        ))
    }

    /// Speak `text` chunk by chunk, calling `on_chunk` as each becomes ready.
    ///
    /// The difference from [`Engine::synthesize_long`] is delivery, not
    /// synthesis: time to first audio is set by the first chunk rather than by
    /// the whole passage, which is what lets a reading app start playing a
    /// sentence while the rest is still being made.
    ///
    /// A callback rather than an `Iterator`, because the engine is borrowed
    /// mutably for the whole render — an iterator would have to hand out items
    /// borrowing from something it is still using. Return `false` from
    /// `on_chunk` to stop; the effect is the same as `should_cancel`.
    ///
    /// Splitting it across windows:
    ///
    /// This port had no long-form path: [`Engine::synthesize`] renders one
    /// window and refuses anything longer, while the documentation called the
    /// binding supported. Two things make the joins match Python's rather than
    /// merely existing:
    ///
    /// * **Per-chunk seeds.** Each chunk draws from `derive(seed, 16 + index)`,
    ///   so a chunk's audio does not depend on how many came before it and
    ///   stopping early cannot change what was already produced.
    /// * **Prefix carry.** The last `chunking.prefix_tokens` speech tokens of a
    ///   chunk are fed into the next as context and dropped from its output.
    ///
    /// `language` is `None` for "the voice's own language" — see
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
    /// the joins inside a passage already use — the carry variable below simply
    /// starts non-empty — which is why a request boundary stops being audible
    /// without a second mechanism existing to maintain.
    ///
    /// # Errors
    /// For a `speed` outside the range, for a `previous_tokens` entry that is not
    /// an acoustic speech token, for text that funnels away to nothing, and for
    /// anything the graphs refuse.
    // Over clippy's seven. Every one of these is a distinct execution input the
    // caller chooses per call, and folding them into an options struct would
    // give this port a signature none of the other four has — the tutorials, the
    // arity check in `tests/test_docs.py` and anyone reading two bindings side
    // by side all key off the parameter list being the same list everywhere.
    #[allow(clippy::too_many_arguments)]
    pub fn stream(
        &mut self,
        text: &str,
        v: &voice::Profile,
        seed: u64,
        language: Option<&str>,
        speed: f64,
        previous_tokens: Option<&[usize]>,
        should_cancel: Option<&mut dyn FnMut() -> bool>,
        on_chunk: &mut dyn FnMut(Chunk<'_>) -> bool,
    ) -> Result<(), String> {
        // Refused before the split, not per chunk: the answer cannot change
        // between chunks, and the caller should hear about a bad value before
        // the first one is generated rather than after.
        timestretch::validate_speed(speed)?;
        let language = resolve_language(language, v);
        // One `&mut dyn FnMut` for the whole loop, reborrowed per call.
        // `Option<&mut dyn FnMut>` is not Copy, so consulting it before a chunk
        // *and* handing it to `generate` in the same iteration cannot both move
        // it; collapsing to a single reference (with a no-op stand-in when the
        // caller passed none) makes the reborrow the only thing happening.
        let mut never = || false;
        let cancel: &mut dyn FnMut() -> bool = match should_cancel {
            Some(callback) => callback,
            None => &mut never,
        };

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
        let mut carry: Vec<usize> =
            carry_from(previous_tokens, prefix_len, self.config.start_speech)?;

        for (index, chunk) in chunks.iter().enumerate() {
            if cancel() {
                break;
            }
            let chunk_seed = derive_seed(seed, (CHUNK_STREAM_BASE + index) as u64);
            let ids = self.frontend.encode(chunk, language)?;
            // Only the last chunk ends the passage.
            let (chunk_tokens, verdict, chunk_hit_cap) = self.generate_inspected(
                &ids,
                v,
                chunk_seed,
                &carry,
                index == chunks.len() - 1,
                Some(&mut *cancel),
            )?;
            // Discarded, not rendered. The partial tokens belong to speech
            // the listener has already interrupted, and the mel decode plus
            // vocode is the larger half of the barge-in latency on an edge
            // device — so running them adds exactly the wait the cancellation
            // exists to remove, and then plays audio nobody asked for.
            // Python does this at engine.py:298; JS at engine.ts:473.
            if cancel() {
                break;
            }
            let chunk_mel = self.decode_mel(&chunk_tokens, v, derive_seed(chunk_seed, 1))?;
            let chunk_audio = self.vocode(&chunk_mel, derive_seed(chunk_seed, 2))?;
            // Last stage, after the inspection above rather than before it.
            let chunk_audio =
                timestretch::time_stretch(&chunk_audio, self.config.sample_rate, speed);

            carry = if prefix_len > 0 && !chunk_tokens.is_empty() {
                let take = prefix_len.min(chunk_tokens.len());
                chunk_tokens[chunk_tokens.len() - take..].to_vec()
            } else {
                Vec::new()
            };

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
                text: chunk,
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
        }
        Ok(())
    }

    /// Speak text of any length as one waveform.
    ///
    /// Exactly [`Engine::stream`] with the chunks concatenated — one loop, so
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
    /// See [`Engine::stream`].
    // Over clippy's seven. Every one of these is a distinct execution input the
    // caller chooses per call, and folding them into an options struct would
    // give this port a signature none of the other four has — the tutorials, the
    // arity check in `tests/test_docs.py` and anyone reading two bindings side
    // by side all key off the parameter list being the same list everywhere.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_long(
        &mut self,
        text: &str,
        v: &voice::Profile,
        seed: u64,
        language: Option<&str>,
        speed: f64,
        previous_tokens: Option<&[usize]>,
        should_cancel: Option<&mut dyn FnMut() -> bool>,
    ) -> Result<SynthResult, String> {
        let mut audio: Vec<f32> = Vec::new();
        let mut tokens: Vec<usize> = Vec::new();
        let mut mel: Vec<f32> = Vec::new();
        let mut spans: Vec<ChunkSpan> = Vec::new();
        let mut hit_token_cap = false;
        self.stream(
            text,
            v,
            seed,
            language,
            speed,
            previous_tokens,
            should_cancel,
            &mut |chunk| {
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
            },
        )?;
        // Rebuilt from the spans rather than shifting each chunk's own timing by
        // a running float: `timeline` accumulates sample offsets as integers, so
        // the joins are exact and every chunk's `end` is the next one's `start`
        // down to the last bit.
        let chunks = timing::timeline(&spans, self.config.sample_rate);
        Ok((
            audio,
            tokens,
            mel,
            self.config.sample_rate,
            chunks,
            hit_token_cap,
        ))
    }
}

/// The conditioning context a call inherits from the one before it.
///
/// The same slice the streaming loop takes between two chunks — the last
/// `prefix_tokens` — applied to tokens that came from a different call. There is
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

/// What a synthesis reads as when neither the caller nor the voice says.
///
/// Reached less often than it looks: `voice::load` defaults a *missing* header
/// key to `"en"`, and Python writes the key, so an empty string only arrives
/// from a `Profile` built in memory or a header hand-edited to `""`. A profile
/// file with no language field inherits nothing — it loads as `"en"`.
const FALLBACK_LANGUAGE: &str = "en";

/// The language chain: the argument, then the voice, then English.
///
/// The language chain: the argument, then the voice's recorded language, then
/// `"en"`. Without the voice link, Polish text reads through the English
/// frontend — English number words, English abbreviation expansion, no Polish
/// respelling — and nothing in the audio reports the mismatch. A profile
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
/// Appending the flat buffers end to end — the obvious thing — is not
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

struct KVCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

fn collect_kv(outputs: &ort::session::SessionOutputs, prefix: &str) -> Result<KVCache, String> {
    let mut kv = KVCache {
        k: Vec::new(),
        v: Vec::new(),
    };
    for i in 0..N_LAYERS {
        let (k, v) = (format!("{prefix}k_{i}"), format!("{prefix}v_{i}"));
        kv.k.push(kv_tensor(outputs.get(&k), &k)?);
        kv.v.push(kv_tensor(outputs.get(&v), &v)?);
    }
    Ok(kv)
}

/// One KV tensor out of a step or prefill run, by name.
///
/// Takes the looked-up value rather than the outputs: `outputs[name]` is
/// `get(name).unwrap_or_else(|| panic!(...))` in ort, so an onnx dir whose
/// graph names its outputs differently — the user-supplied half of the same
/// mistake as a wrong dtype — reached the caller as a panic. Both halves are
/// this one error now, and the split lets the message be tested without a
/// session, which `SessionOutputs` cannot be built outside ort.
fn kv_tensor(found: Option<&ort::value::DynValue>, name: &str) -> Result<Vec<f32>, String> {
    let (_, data) = found
        .ok_or_else(|| format!("missing or non-f32 output {name}"))?
        .try_extract_tensor::<f32>()
        .map_err(|_| format!("missing or non-f32 output {name}"))?;
    Ok(data.to_vec())
}

/// Mirrors engine._derive.
/// Retry attempts draw derive(seed, 8 + attempt): clear of the stage streams
/// (1, 2) and below the chunk streams at 16.
const RETRY_STREAM_BASE: u64 = 8;

fn derive_seed(seed: u64, stream: u64) -> u64 {
    const PHI: u64 = 0x9e3779b97f4a7c15;
    const PSI: u64 = 0xbf58476d1ce4e5b9;
    seed.wrapping_mul(PHI)
        .wrapping_add(stream.wrapping_mul(PSI))
}

/// ort errors don't convert to String via `?`; this is the single conversion
/// point. Generic over the recovery type, because a builder call hands back the
/// builder it failed on rather than a bare error.
fn ort_err<R>(e: ort::Error<R>) -> String {
    format!("{e}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(language: &str) -> voice::Profile {
        voice::Profile {
            name: "fake".to_string(),
            speaker_embedding: Vec::new(),
            flow_embedding: Vec::new(),
            prompt_tokens: Vec::new(),
            prompt_mel: Vec::new(),
            cond_prompt_tokens: Vec::new(),
            language: language.to_string(),
        }
    }

    /// The obvious call must not be the wrong one.
    ///
    /// Without the voice link, `synthesize("Cześć", polish_voice, seed, ..)`
    /// runs Polish text through the English frontend: the argument alone has no
    /// answer but `"en"`, and a profile's own language — recorded at
    /// enrollment — is only visible through the voice. The chain is argument,
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
        // language id is not a language — it would tag the text `[]`. A file
        // whose header simply omits the key loads as "en" instead, so it never
        // reaches this branch.
        assert_eq!(resolve_language(None, &profile("")), "en");
    }

    /// A mel is row-major `[MEL_BINS, frames]`. Appending two flat buffers puts
    /// the second chunk's bin 0 after the first chunk's bin 79, so every row but
    /// the first is wrong. The audio is unaffected — each chunk is vocoded on
    /// its own — but the mel is the diagnostic people reach for when two
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

    /// The result carries the truncation flag.
    ///
    /// `SynthResult` is a bare tuple, so nothing forces a new field to be
    /// threaded through `synthesize` and `synthesize_long` — a field that only
    /// exists in one of them compiles anyway and lies to whichever caller reads
    /// it. Pinning the arity here means dropping the flag or threading it
    /// through one path alone breaks this test rather than a caller's
    /// destructuring. This pins the shape; the flag's value on a real render is
    /// covered where the weights are available.
    #[test]
    fn synth_result_carries_hit_token_cap() {
        let result: SynthResult = (
            Vec::new(),
            Vec::new(),
            Vec::new(),
            24_000,
            Vec::new(),
            false,
        );
        let (_, _, _, _, _, capped) = result;
        assert!(!capped, "a normal render does not hit the token cap");
    }

    /// A tokenizer wider than the checkpoint refuses at the door, naming it.
    ///
    /// `text_row` indexes the embedding table by raw token id. Paired with a
    /// checkpoint from another release the widest ids read past the end of it —
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
    /// half of the same mistake used to abort the process while the wrong-dtype
    /// half returned a readable error.
    #[test]
    fn a_missing_kv_output_is_an_error() {
        let err = kv_tensor(None, "present_k_0").unwrap_err();
        assert_eq!(err, "missing or non-f32 output present_k_0");
    }
}
