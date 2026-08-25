//! Deciding where a generated chunk actually ended.
//!
//! Mirrors `loudkit.postprocess`. This is a **detector**, not a filter: it reads
//! the speech tokens a chunk produced, answers one question — where did the
//! sentence really stop? — and returns a verdict. It never touches a sample of
//! audio.
//!
//! The artifact it removes is generated, not spectral. The decoder is
//! free-running, and silence tokens are exempt from both the repetition penalty
//! and the `min_p` cutoff (penalising silence measurably removes pauses), so
//! once the sentence is over those tokens keep probability mass indefinitely.
//! The decoder free-runs silence, and any step where a non-silence token
//! survives the cutoff becomes a hallucinated word — heard as "it finished, then
//! a long gap, then one random word".
//!
//! Every constant came from a device trace or a regression, and every rule is
//! pinned by `tests/data/conformance/postprocess.json`, which all five ports
//! run. Provenance is in `docs/reference/postprocess.md`.
//! Python reference: `loudkit/postprocess.py`.

use std::collections::HashSet;

/// What the engine does with a verdict.
///
/// `Trim` applies the cut, which changes the audio and therefore travels in the
/// fingerprint like every other audible decision. `Report` runs the detectors
/// and attaches the verdict without acting on it. `Off` skips them entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Off,
    Report,
    Trim,
}

impl Mode {
    /// The manifest spelling, which is also what the fingerprint hashes.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Report => "report",
            Self::Trim => "trim",
        }
    }

    /// Parse a manifest value.
    ///
    /// # Errors
    ///
    /// Returns an error naming the unknown mode. A mode this port does not
    /// implement must not fall back to a default: it would trim where the
    /// manifest said not to, or not trim where it said to, under a matching
    /// `recipe_version`.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "off" => Ok(Self::Off),
            "report" => Ok(Self::Report),
            "trim" => Ok(Self::Trim),
            other => Err(format!(
                "manifest declares unknown postprocess mode {other:?}; \
                 expected off, report or trim"
            )),
        }
    }
}

/// Which rule fired.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reason {
    Clean,
    Dropout,
    Repetition,
    SilenceTail,
    TerminalEcho,
    Desperation,
    EndedTail,
}

impl Reason {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Clean => "clean",
            Self::SilenceTail => "silence_tail",
            Self::TerminalEcho => "terminal_echo",
            Self::Desperation => "desperation",
            Self::Dropout => "dropout",
            Self::Repetition => "repetition",
            Self::EndedTail => "ended_tail",
        }
    }
}

/// The detector constants. Algorithm layer: a port that uses a
/// different number produces different audio, so these are hashed into the
/// fingerprint rather than left as module constants.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub mode: Mode,

    /// Hard stop for generation, as a multiple of the text-token count.
    ///
    /// Device trace of the showcase render: `t3.overrun gen=92 ceiling=92
    /// bestEOS=74@0.003 floor=31` — ~26 text tokens stopped only because it hit
    /// the ceiling, mid-sentence, already at 3.5 speech tokens per text token.
    /// NOT the chunker's 2.6: there, guessing high only wastes window; here,
    /// guessing low cuts a sentence off.
    pub ceiling_speech_per_text_token: f64,
    /// Carries the very short texts, where a ratio alone is unsafe (1.6 s).
    pub ceiling_slack_tokens: usize,

    /// Share of a tail that must be silence before it counts as one.
    pub trailing_filler_threshold: f64,
    /// An unbroken silence run marking a structural boundary (~0.5 s at 25 Hz).
    ///
    /// A hallucinated word sits *behind* such a seam; under the share test
    /// alone its burst lowers the silence ratio below threshold, so the
    /// ugliest tails are exactly the ones the rescue refuses to cut.
    pub trailing_silence_run_tokens: usize,
    /// Top of the stop-peak acceptance band in [`desperation_cut`], as a
    /// multiple of the text-token count.
    ///
    /// Measured reads run 1.75–2.35 speech tokens per text token, so the band
    /// reaches past every legitimate ending while staying well under the 4.5x
    /// garbage threshold.
    pub desperation_band_ratio: f64,
    /// Slack above the proportional band, in speech tokens (~0.5 s). Carries
    /// the short texts, where the ratio alone would close the band on endings
    /// a legitimate read had already reached.
    pub desperation_band_floor: usize,
    /// How confident the best stop must be before the share/run test is
    /// consulted at all. EOS-defence bench, variant B.
    pub filler_min_eos_probability: f64,
    /// How much speech may follow a seam and still be a hallucinated word
    /// rather than a continuing clause (~0.4 s).
    ///
    /// Deliberately separate from `ended_tail_word_max` despite holding the
    /// same number: they govern different rows, so
    /// loosening the trim on terminal chunks must not silently loosen this.
    pub filler_max_speech_after_run: usize,

    /// Past this ratio the row certainly contains garbage, whatever its stop
    /// confidence said.
    ///
    /// "It was as he expected." — 14 text tokens — came back as 96 speech
    /// tokens of sentence-then-dense-babble, with the stop peak at the right
    /// *place* (45) but confidence 0.000, so every probability-gated rescue
    /// refused. Real speech runs 1.75–2.35 speech tokens per text token.
    pub desperation_speech_per_text_token: f64,
    /// Tiny texts are exempt: fixed overheads give a clean "No!" a ratio of 6+
    /// by itself.
    pub desperation_min_text_tokens: usize,

    /// Silence before a blip that counts as stranding it (~0.24 s).
    pub ended_tail_silence_run: usize,
    /// <= 80 ms of "speech" is a click, not a word.
    pub ended_tail_blip_max: usize,
    /// A stray word behind a full seam on a *terminal* chunk is cut with it.
    /// Continuation chunks keep their tails — their pauses are the sentence's
    /// rhythm and their "end" is not an end.
    pub ended_tail_word_max: usize,
    /// Pause left in place after trimming (~0.2 s).
    pub ended_tail_keep: usize,

    /// The ordinary terminal echo: a confident stop, late, with at most ~1.2 s
    /// after it. The position rule keeps a real clause pause from reading as an
    /// ending.
    pub echo_strong_eos_probability: f64,
    pub echo_strong_max_tail: usize,
    pub echo_strong_min_position_pct: usize,

    /// The narrow second path, for one regression ("...but a brigand. Pass.
    /// Four.": `gen=124/124, bestEOS=109@0.004`). Confidence this weak is
    /// accepted only with every corroborator at once.
    pub echo_weak_eos_probability: f64,
    pub echo_weak_max_tail: usize,
    pub echo_weak_min_position_pct: usize,

    /// How many re-rolls a condemned window may get before shipping as is.
    /// Only dropout and suspect retry; each attempt draws a derived seed, so
    /// the ladder is a pure function of the caller's seed.
    pub retry_max_attempts: usize,

    /// How far a chunk's pace may drift from the passage's median before it
    /// is flagged (multiplicative, both directions).
    pub pacing_tolerance: f64,

    /// Longest cycle, in tokens (~0.5 s), that counts as a stuck decoder.
    /// Above this a repeated block is a phrase, and a repeated phrase is
    /// rhetoric rather than a lock-up.
    pub repetition_max_period: usize,

    /// How many consecutive identical cycles before it can be a loop. Two is a
    /// repeated phrase; three is a necessary condition, not a sufficient one.
    pub repetition_min_cycles: usize,

    /// How many tokens the repeating region must cover (~1.0 s). The constant
    /// that does the work: measured across 27 renders in nine languages, a
    /// healthy row repeats for at most 10 tokens. Keying on cycle count alone
    /// fired on 22 of those 27.
    pub repetition_min_span: usize,

    /// Early truncation: the row is too short to be the text it was asked for.
    /// Reported, never cut — there is nothing to cut, and it is the most damaging
    /// failure in the set because a listener cannot hear that content is absent.
    /// The 25-token floor is the published criterion for a catastrophic
    /// neural-codec TTS failure; the proportional test exempts a genuinely short
    /// line, since the shortest healthy reads measured run 35 tokens.
    pub dropout_min_tokens: usize,
}

impl Default for Config {
    /// The shipping detector configuration.
    fn default() -> Self {
        Self {
            mode: Mode::Trim,
            ceiling_speech_per_text_token: 4.0,
            ceiling_slack_tokens: 40,
            trailing_filler_threshold: 0.7,
            trailing_silence_run_tokens: 12,
            desperation_band_ratio: 2.6,
            desperation_band_floor: 12,
            filler_min_eos_probability: 0.05,
            filler_max_speech_after_run: 10,
            desperation_speech_per_text_token: 4.5,
            desperation_min_text_tokens: 10,
            ended_tail_silence_run: 6,
            ended_tail_blip_max: 2,
            ended_tail_word_max: 10,
            ended_tail_keep: 5,
            echo_strong_eos_probability: 0.1,
            echo_strong_max_tail: 30,
            echo_strong_min_position_pct: 68,
            echo_weak_eos_probability: 0.003,
            echo_weak_max_tail: 16,
            echo_weak_min_position_pct: 85,
            retry_max_attempts: 2,
            pacing_tolerance: 1.6,
            repetition_max_period: 12,
            repetition_min_cycles: 3,
            repetition_min_span: 24,
            dropout_min_tokens: 25,
        }
    }
}

/// What the detectors concluded about one chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Inspection {
    /// How many leading tokens survive — equal to the input length when nothing
    /// fired, so a caller can always slice by it without branching.
    pub keep: usize,
    pub reason: Reason,
    /// The row is impossibly long for its text and no anchor agreed where to
    /// cut. Not an error and not a cut: a report. Shipping such a row silently
    /// is how the artifact reached listeners in the first place.
    pub suspect: bool,
}

impl Inspection {
    /// Whether anything was removed.
    #[must_use]
    pub fn cut(&self) -> bool {
        self.reason != Reason::Clean
    }
}

/// Everything the detectors need to know about one generated chunk.
#[derive(Debug, Clone, Copy)]
pub struct Request {
    /// The denominator of every ratio rule.
    pub text_token_count: usize,
    /// The EOS floor this row was generated under.
    pub min_tokens: usize,
    /// Step at which the stop token was most probable, or negative if it was
    /// never observed.
    pub eos_peak_at: i64,
    pub eos_peak_prob: f64,
    /// Whether generation stopped at the stop token rather than a cap.
    pub ended: bool,
    /// Whether this chunk ends the passage. A continuation chunk has no
    /// sentence end, so its stop peak means nothing.
    pub is_terminal: bool,
    /// Whether generation was stopped by the length ceiling.
    pub hit_ceiling: bool,
}

/// Speech tokens at which the decoder is stopped whatever it thinks.
///
/// Applied *during* generation: the tokens past it cost real time on a device
/// and are certain to be discarded. It only ever stops a row that was going to
/// run away — a model that stops on its own never reaches it.
#[must_use]
pub fn ceiling_for(text_token_count: usize, cfg: &Config, window: usize) -> usize {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    #[allow(clippy::cast_precision_loss)]
    let proportional = (text_token_count as f64 * cfg.ceiling_speech_per_text_token) as usize;
    (proportional + cfg.ceiling_slack_tokens).min(window)
}

fn silence_flags(tokens: &[usize], silence: &HashSet<usize>) -> Vec<bool> {
    tokens.iter().map(|t| silence.contains(t)).collect()
}

/// Whether what follows `index` is a trailing tail rather than more sentence.
///
/// The overrun rescue cuts back to where the model came closest to stopping,
/// and that peak is a hint, not a verdict. Trusting it alone truncated whole
/// sentences: a voice reading a language its tag does not match may never
/// commit to stopping, so its best moment of hesitation lands a third of the
/// way in.
///
/// So the peak is corroborated by *what it proposes to discard* — either the
/// tail is mostly silence by share, or it holds a long unbroken run with only a
/// stray word behind it. Without that second half, a rhetorical pause mid-tail
/// (25 silent tokens, then 80 of speech) matched the run rule and the rescue cut
/// the rest of the sentence off.
#[must_use]
pub fn is_trailing_filler(
    tokens: &[usize],
    index: usize,
    silence: &HashSet<usize>,
    cfg: &Config,
) -> bool {
    if index >= tokens.len() {
        return false;
    }
    let flags = silence_flags(&tokens[index..], silence);

    let (mut silent, mut run, mut longest_run) = (0usize, 0usize, 0usize);
    for &is_silent in &flags {
        if is_silent {
            silent += 1;
            run += 1;
            longest_run = longest_run.max(run);
        } else {
            run = 0;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    if silent as f64 / flags.len() as f64 >= cfg.trailing_filler_threshold {
        return true;
    }
    if longest_run < cfg.trailing_silence_run_tokens {
        return false;
    }

    // Collect qualifying runs, then require every gap of speech between them
    // — and after the last — to be a stray word or less. [seam][real
    // sentence][seam][word] fails: the tokens between the two seams are the
    // sentence itself, not filler trailing the first boundary.
    struct RunSpan {
        start: usize,
        end: usize,
    }
    let mut runs: Vec<RunSpan> = Vec::new();
    let (mut scan_run, mut scan_start) = (0usize, 0usize);
    for (i, &is_silent) in flags.iter().enumerate() {
        if is_silent {
            if scan_run == 0 {
                scan_start = i;
            }
            scan_run += 1;
            if scan_run == cfg.trailing_silence_run_tokens {
                runs.push(RunSpan {
                    start: scan_start,
                    end: i + 1,
                });
            }
        } else {
            scan_run = 0;
        }
    }
    if runs.is_empty() {
        return false;
    }
    if runs[0].start > cfg.filler_max_speech_after_run {
        return false;
    }
    let last = runs.last().expect("non-empty above");
    if flags.len() - last.end > cfg.filler_max_speech_after_run {
        return false;
    }
    for pair in runs.windows(2) {
        if pair[1].start - pair[0].end > cfg.filler_max_speech_after_run {
            return false;
        }
    }
    true
}

/// The rescue for rows whose *length* is the evidence.
///
/// Past the ratio the row is certainly broken, so the question is where to cut,
/// not whether: at the first long silence run that starts past the floor (a run
/// straddling the floor belongs to the sentence, which is why the run's *start*
/// is tested), else at the stop peak if it sits in a band a real read could have
/// ended in. The band protects the mislabeled-language case (92 generated / 26
/// text = 3.5x), whose kind of row must never be cut at a peak landing a third
/// of the way in.
///
/// `peak_allowed` is false for a continuation chunk: it has no sentence end, so
/// Indices of chunks whose pace drifts past the tolerance from the median.
/// Long-form drift: per-chunk pace (speech tokens / text tokens) against the passage's own median, report-only. The median rather than the mean, so one broken chunk cannot drag the baseline toward itself and hide.
#[must_use]
pub fn pacing_outliers(ratios: &[f64], cfg: &Config) -> Vec<usize> {
    if ratios.len() < 3 {
        // One chunk has no neighbours; two cannot say which of them drifted.
        return Vec::new();
    }
    let mut ordered = ratios.to_vec();
    ordered.sort_by(|a, b| a.partial_cmp(b).expect("ratios are finite"));
    let mid = ordered.len() / 2;
    let median = if ordered.len().is_multiple_of(2) {
        (ordered[mid - 1] + ordered[mid]) / 2.0
    } else {
        ordered[mid]
    };
    if median <= 0.0 {
        return Vec::new();
    }
    ratios
        .iter()
        .enumerate()
        .filter(|(_, r)| **r > median * cfg.pacing_tolerance || **r < median / cfg.pacing_tolerance)
        .map(|(i, _)| i)
        .collect()
}

/// Whether the row is too short to be the text it was asked for.
///
/// Two conditions, both required. The absolute floor catches a row that stopped
/// almost immediately whatever the text was; the proportional one keeps a
/// genuinely short line exempt, because a read producing less than one speech
/// token per text token has not said the text under any pronunciation.
#[must_use]
pub fn is_dropout(token_count: usize, text_token_count: usize, cfg: &Config) -> bool {
    if token_count >= cfg.dropout_min_tokens {
        return false;
    }
    text_token_count > 0 && token_count < text_token_count
}

/// Where a stuck decoder started looping, or `None`.
///
/// The failure the tail rules cannot see, because it happens *inside* the row.
/// The mechanism is the one behind the trailing hallucinated word — the model's
/// own output becomes its context — but it strikes mid-sequence, so no rule that
/// reads the end can find it.
///
/// Deliberately hard to trigger, because it is the only rule here that cuts
/// mid-sequence: a short cycle, repeated many times, matched exactly. A decoder
/// that has genuinely locked up emits the same tokens rather than similar ones,
/// and a fuzzy match on a signal this destructive would truncate real speech.
///
/// A cycle that is entirely silence is never a loop — silence repeating is what
/// silence is, and the tail rules already judge pauses against where they sit.
///
/// Returns one full cycle past the loop's start: the first instance is
/// plausibly the word the sentence wanted.
#[must_use]
pub fn repetition_cut(tokens: &[usize], silence: &HashSet<usize>, cfg: &Config) -> Option<usize> {
    let n = tokens.len();
    if n < cfg.repetition_min_span {
        return None;
    }
    let quiet: Vec<bool> = tokens.iter().map(|t| silence.contains(t)).collect();

    // Earliest loop wins: a row that locks up twice locked up first at the
    // first one, and everything after it is already inside the failure.
    let mut best: Option<usize> = None;
    let longest_period = cfg.repetition_max_period.min(n / cfg.repetition_min_cycles);
    for period in 1..=longest_period {
        let mut start = 0;
        while start + period * cfg.repetition_min_cycles <= n {
            let mut cycles = 1;
            let mut at = start + period;
            while at + period <= n && tokens[at..at + period] == tokens[start..start + period] {
                cycles += 1;
                at += period;
            }
            if cycles >= cfg.repetition_min_cycles
                && cycles * period >= cfg.repetition_min_span
                && !quiet[start..start + period].iter().all(|q| *q)
            {
                let candidate = start + period;
                if best.is_none_or(|b| candidate < b) {
                    best = Some(candidate);
                }
                break;
            }
            start += 1;
        }
    }
    best
}

/// its stop peak means nothing.
#[must_use]
pub fn desperation_cut(
    tokens: &[usize],
    text_token_count: usize,
    min_tokens: usize,
    eos_peak_at: i64,
    silence: &HashSet<usize>,
    cfg: &Config,
    peak_allowed: bool,
) -> Option<usize> {
    if text_token_count < cfg.desperation_min_text_tokens {
        return None;
    }
    #[allow(clippy::cast_precision_loss)]
    if (tokens.len() as f64) < text_token_count as f64 * cfg.desperation_speech_per_text_token {
        return None;
    }
    let earliest = min_tokens.max(10);
    let flags = silence_flags(tokens, silence);

    let (mut run_start, mut run) = (0usize, 0usize);
    for (i, &is_silent) in flags.iter().enumerate() {
        if is_silent {
            if run == 0 {
                run_start = i;
            }
            run += 1;
            if run >= cfg.trailing_silence_run_tokens && run_start >= earliest {
                return Some(run_start);
            }
        } else {
            run = 0;
        }
    }

    // No seam — the babble is dense; fall back to the model's own best stop, if
    // it lands where a real read could have ended.
    if !peak_allowed || eos_peak_at < 0 {
        return None;
    }
    #[allow(clippy::cast_sign_loss)]
    let peak = eos_peak_at as usize;
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    let band_top = (cfg.desperation_band_ratio * text_token_count as f64) as usize
        + cfg.desperation_band_floor;
    if peak >= earliest && peak <= band_top && peak < tokens.len() {
        return Some(peak);
    }
    None
}

/// Dead air past the sentence on a row that stopped when it meant to.
///
/// Walked backward as `[sentence][r1 silence][burst][r2 silence]`. Three shapes
/// come off: a bare silence run half a second long; a silence run with a 1–2
/// token blip right before the stop (the device specimen ended `.......#`); and,
/// on a *terminal* chunk only, a stray word behind a full seam.
#[must_use]
pub fn ended_tail_trim(
    tokens: &[usize],
    silence: &HashSet<usize>,
    cfg: &Config,
    is_terminal: bool,
) -> Option<usize> {
    let flags = silence_flags(tokens, silence);
    // `i64` so walking off the front is representable; every comparison below
    // needs "we ran out of tokens" to be distinguishable from index 0.
    let mut j = tokens.len() as i64 - 1;

    let mut r2 = 0usize;
    while j >= 0 && flags[j as usize] {
        r2 += 1;
        j -= 1;
    }
    if j < 0 {
        return None;
    }
    #[allow(clippy::cast_sign_loss)]
    if r2 >= cfg.trailing_silence_run_tokens {
        let n = j as usize + 1 + r2.min(cfg.ended_tail_keep);
        return (n < tokens.len()).then_some(n);
    }

    let mut burst = 0usize;
    while j >= 0 && !flags[j as usize] {
        burst += 1;
        j -= 1;
    }
    let mut r1 = 0usize;
    while j >= 0 && flags[j as usize] {
        r1 += 1;
        j -= 1;
    }
    if j < 0 {
        return None; // the "burst" was the sentence
    }

    let stranded_click = burst <= cfg.ended_tail_blip_max && r1 >= cfg.ended_tail_silence_run;
    let stranded_word =
        is_terminal && burst <= cfg.ended_tail_word_max && r1 >= cfg.trailing_silence_run_tokens;
    if !stranded_click && !stranded_word {
        return None;
    }
    #[allow(clippy::cast_sign_loss)]
    let n = j as usize + 1 + r1.min(cfg.ended_tail_keep);
    (n < tokens.len()).then_some(n)
}

/// A terminal chunk that ended correctly and then free-ran an extra word.
///
/// There is no silence seam here, so [`is_trailing_filler`] has nothing to
/// anchor on. Instead the earlier stop candidate must be strong, late and
/// followed by a short tail. The second acceptance path is narrower and exists
/// for one regression where the model never sampled a stop token but its best —
/// very weak — stop was 15 tokens before the hard ceiling.
#[must_use]
pub fn terminal_echo_cut(
    token_count: usize,
    eos_peak_at: i64,
    eos_peak_prob: f64,
    min_tokens: usize,
    is_terminal: bool,
    hit_ceiling: bool,
    cfg: &Config,
) -> Option<usize> {
    if !is_terminal || eos_peak_at < 0 {
        return None;
    }
    #[allow(clippy::cast_sign_loss)]
    let peak = eos_peak_at as usize;
    if peak <= min_tokens.max(10) || peak >= token_count {
        return None;
    }

    let tail = token_count - peak;
    let strong_peak = eos_peak_prob >= cfg.echo_strong_eos_probability
        && tail <= cfg.echo_strong_max_tail
        && peak * 100 >= token_count * cfg.echo_strong_min_position_pct;
    let weak_late_peak_at_ceiling = hit_ceiling
        && eos_peak_prob >= cfg.echo_weak_eos_probability
        && tail <= cfg.echo_weak_max_tail
        && peak * 100 >= token_count * cfg.echo_weak_min_position_pct;
    (strong_peak || weak_late_peak_at_ceiling).then_some(peak)
}

/// Run every detector in precedence order and return one verdict.
///
/// The shipped reader grew five entry points, one per field bug, and left the
/// ordering to each call site. Here they are one resolver with the precedence
/// written down, because an order that lives in a caller is an order the next
/// caller gets wrong.
///
/// Peak-anchored rescues first, then the length-anchored one — it is the
/// bluntest, and it applies to *ended* rows too, because a model that babbles
/// past its sentence and only then samples a stop token has forfeited the trust
/// that stopping implies. The ended-tail trim runs only when nothing above
/// fired.
#[must_use]
pub fn inspect(
    tokens: &[usize],
    req: &Request,
    silence: &HashSet<usize>,
    cfg: &Config,
) -> Inspection {
    if cfg.mode == Mode::Off || tokens.is_empty() {
        return Inspection {
            keep: tokens.len(),
            reason: Reason::Clean,
            suspect: false,
        };
    }

    let floor = req.min_tokens.max(10) as i64;
    let mut cut: Option<usize> = None;
    let mut reason = Reason::Clean;

    // Terminal chunks only, like its three siblings. `is_terminal` means a
    // continuation chunk's stop peak is meaningless and its pauses are rhythm
    // rather than dead air — and this rule reads exactly those two signals, so
    // it was trimming mid-passage chunks on evidence the contract says is not
    // evidence. Changed in all five implementations together; postprocess is a
    // bit-parity surface.
    let filler_cut = req.is_terminal
        && !req.ended
        && req.eos_peak_prob > cfg.filler_min_eos_probability
        && req.eos_peak_at > floor
        && (req.eos_peak_at as usize) < tokens.len()
        && is_trailing_filler(tokens, req.eos_peak_at as usize, silence, cfg);
    // Early truncation first: nothing below can help a row that is already too
    // short, and the verdict is "incomplete" rather than "wrongly ended".
    if is_dropout(tokens.len(), req.text_token_count, cfg) {
        return Inspection {
            keep: tokens.len(),
            reason: Reason::Dropout,
            suspect: true,
        };
    }

    // Then repetition, because it is the only rule that knows *exactly* where
    // the failure began. Every other anchor here is inferred from a signal that might mean
    // something else; an exactly repeated cycle is not.
    if let Some(looped) = repetition_cut(tokens, silence, cfg) {
        cut = Some(looped);
        reason = Reason::Repetition;
    } else if filler_cut {
        #[allow(clippy::cast_sign_loss)]
        {
            cut = Some(req.eos_peak_at as usize);
        }
        reason = Reason::SilenceTail;
    } else if let Some(echo) = terminal_echo_cut(
        tokens.len(),
        req.eos_peak_at,
        req.eos_peak_prob,
        req.min_tokens,
        req.is_terminal,
        req.hit_ceiling,
        cfg,
    ) {
        cut = Some(echo);
        reason = Reason::TerminalEcho;
    } else if let Some(desperate) = desperation_cut(
        tokens,
        req.text_token_count,
        req.min_tokens,
        req.eos_peak_at,
        silence,
        cfg,
        req.is_terminal,
    ) {
        cut = Some(desperate);
        reason = Reason::Desperation;
    }

    if cut.is_none() && req.ended {
        if let Some(trimmed) = ended_tail_trim(tokens, silence, cfg, req.is_terminal) {
            cut = Some(trimmed);
            reason = Reason::EndedTail;
        }
    }

    // A condemned row that dodged every token anchor. Reported, never cut: no
    // rule could say where, and cutting at a guess is how the rescue truncated
    // whole sentences before the corroboration rules were added.
    #[allow(clippy::cast_precision_loss)]
    let suspect = cut.is_none()
        && req.text_token_count >= cfg.desperation_min_text_tokens
        && tokens.len() as f64
            >= req.text_token_count as f64 * cfg.desperation_speech_per_text_token;
    Inspection {
        keep: cut.unwrap_or(tokens.len()),
        reason,
        suspect,
    }
}
