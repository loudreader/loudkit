//! Where each chunk — and, approximately, each word — lands in the waveform.
//!
//! A reading app highlights the sentence it is speaking. That needs two
//! different kinds of answer, and this module keeps them apart on purpose,
//! because conflating them is how a feature like this becomes a lie.
//!
//! **Chunk times are exact.** The engine renders each chunk to its own waveform
//! and concatenates them, so it knows every chunk's sample offset and sample
//! length without estimating anything. [`ChunkTiming`] reports those, converted
//! to seconds. Chunk *k*'s `end` is bit-identical to chunk *k+1*'s `start`:
//! both are the same integer sample offset divided by the same sample rate, so
//! a highlight driven by them can neither gap nor overlap.
//!
//! **Word times are estimated.** The model emits speech tokens, not an
//! alignment; nothing in this pipeline knows where a word begins.
//! [`WordTiming`] distributes a chunk's real duration across its words in
//! proportion to how long each word is in characters, and that is all it is. It
//! is right often enough to be useful for a highlight at sentence scale and
//! wrong in the ways you would expect: a long word said fast, a short word
//! held, a pause before a clause. The error grows with the length of the chunk,
//! because a single bad guess early shifts everything after it — one sentence is
//! usually fine, a long paragraph read as one chunk is not. If you need real
//! alignment you need a forced aligner; this is not one, and pretending
//! otherwise would be worse than the estimate.
//!
//! Both are computed *after* any time-stretch, on the waveform the caller
//! actually receives, so a `speed` other than 1.0 needs no correction applied to
//! them.
//!
//! Mirrors `loudkit.timing` in Python, arithmetic for arithmetic.

/// What one rendered chunk contributes to a timeline.
///
/// The three facts the engine has at concatenation time and nothing else: the
/// text it was asked to speak (post-funnel, which is what was tokenised), how
/// many samples it rendered to, and how many speech tokens it took. Kept as an
/// input type rather than assembling a [`ChunkTiming`] per chunk, because the
/// offsets are only knowable once the order is known.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkSpan {
    pub text: String,
    pub samples: usize,
    pub tokens: usize,
}

/// One word's estimated span, in seconds from the start of the synthesis.
///
/// **Estimated, by proportional allocation.** The chunk's real duration is
/// divided among its words in proportion to their length in characters. There
/// is no alignment model here and no per-word measurement — see the module
/// documentation for what that costs you.
#[derive(Debug, Clone, PartialEq)]
pub struct WordTiming {
    /// The word as it appears in the chunk, punctuation included.
    ///
    /// Punctuation stays attached because the split is on whitespace: a caller
    /// highlighting `"end."` wants the full stop lit with the word, and a caller
    /// matching back against their own text needs the substring to be a
    /// substring.
    pub text: String,
    pub start: f64,
    pub end: f64,
}

/// One chunk's exact span, and its words' estimated ones.
///
/// The two tiers in one value on purpose: a caller that trusts only the exact
/// tier reads `start`/`end` and ignores `words`, and the field names make it
/// impossible to reach the estimate by accident.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkTiming {
    /// The chunk's text after the speech funnel — what was tokenised, which is
    /// not always what the caller passed in (Polish respells embedded English,
    /// and numbers are read as words).
    pub text: String,
    /// Seconds from the start of this result's audio.
    ///
    /// Zero for the first chunk, and for every chunk handed to an
    /// [`crate::engine::Engine::stream`] callback: a streamed chunk is its own
    /// result and does not know what preceded it, so the caller stitching the
    /// stream adds the offsets — [`ChunkTiming::shifted`] is that.
    pub start: f64,
    pub end: f64,
    /// Speech tokens this chunk generated. Duration over tokens is the pacing
    /// the postprocess detectors measure against, which is the other reason to
    /// carry it.
    pub tokens: usize,
    pub words: Vec<WordTiming>,
}

impl ChunkTiming {
    #[must_use]
    pub fn duration(&self) -> f64 {
        self.end - self.start
    }

    /// This timing moved later by `by` seconds, words included.
    ///
    /// What a caller of [`crate::engine::Engine::stream`] needs and the engine
    /// cannot do for them: every streamed chunk starts at zero, because it is
    /// the caller who knows whether this chunk follows five others or begins a
    /// fresh playback.
    #[must_use]
    pub fn shifted(&self, by: f64) -> Self {
        Self {
            text: self.text.clone(),
            start: self.start + by,
            end: self.end + by,
            tokens: self.tokens,
            words: self
                .words
                .iter()
                .map(|w| WordTiming {
                    text: w.text.clone(),
                    start: w.start + by,
                    end: w.end + by,
                })
                .collect(),
        }
    }
}

/// Lay rendered chunks end to end and time them.
///
/// Offsets accumulate in **samples**, not seconds, and are divided by the rate
/// once at the end. Accumulating seconds instead would make chunk *k*'s `end`
/// and chunk *k+1*'s `start` two different sums of the same floats, differing in
/// the last bit — a gap or an overlap of a few nanoseconds, invisible in a test
/// that compares with a tolerance and visible as a flicker in a highlight that
/// switches on `time >= start`.
#[must_use]
pub fn timeline(spans: &[ChunkSpan], sample_rate: usize) -> Vec<ChunkTiming> {
    let rate = sample_rate as f64;
    let mut out = Vec::with_capacity(spans.len());
    let mut at: usize = 0;
    for span in spans {
        let start = at as f64 / rate;
        at += span.samples;
        let end = at as f64 / rate;
        out.push(ChunkTiming {
            text: span.text.clone(),
            start,
            end,
            tokens: span.tokens,
            words: estimate_words(&span.text, start, end),
        });
    }
    out
}

/// Split `text` on whitespace and share `[start, end]` out by length.
///
/// The allocation is by **character count**, not by token count or by any
/// acoustic measure: a word's characters are the only thing known here, and they
/// correlate with duration well enough at sentence scale to drive a highlight.
/// Whitespace itself is not charged for — the gap between two words belongs to
/// whichever side of the boundary the caller's player is on, and splitting it
/// would only invent a third kind of span.
///
/// Boundaries are computed from a running character total rather than by adding
/// per-word durations, so the spans cannot drift: the first `start` is exactly
/// `start`, the last `end` is exactly `end`, and every interior boundary is
/// shared by the two words that meet at it.
///
/// Characters means **code points** — `chars().count()`, matching Python's
/// `len(w)`, Go's `utf8.RuneCountInString` and Swift's `unicodeScalars.count`.
/// Counting bytes instead would give Polish and Japanese text different word
/// weights in one port than in another, for text that reads identically.
#[must_use]
pub fn estimate_words(text: &str, start: f64, end: f64) -> Vec<WordTiming> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let lengths: Vec<usize> = words.iter().map(|w| w.chars().count()).collect();
    let total: usize = lengths.iter().sum();
    if total == 0 {
        return Vec::new();
    }
    let total = total as f64;
    let span = end - start;
    let mut out = Vec::with_capacity(words.len());
    let mut seen: usize = 0;
    for (word, length) in words.iter().zip(&lengths) {
        let at = start + span * (seen as f64 / total);
        seen += length;
        out.push(WordTiming {
            text: (*word).to_string(),
            start: at,
            end: start + span * (seen as f64 / total),
        });
    }
    out
}
