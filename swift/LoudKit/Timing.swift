import Foundation

/// Where each chunk — and, approximately, each word — lands in the waveform.
///
/// Mirrors `loudkit.timing`. A reading app highlights the sentence it is
/// speaking, and that needs two different kinds of answer. This module is
/// careful to keep them apart, because conflating them is how a feature like
/// this becomes a lie:
///
/// **Chunk times are exact.** The engine renders each chunk to its own waveform
/// and concatenates them, so it knows every chunk's sample offset and sample
/// length without estimating anything. ``ChunkTiming`` reports those, converted
/// to seconds. Chunk *k*'s `end` is bit-identical to chunk *k+1*'s `start` —
/// both are the same integer sample offset divided by the same sample rate — so
/// a highlight driven by them can neither gap nor overlap.
///
/// **Word times are estimated.** The model emits speech tokens, not an
/// alignment; nothing in this pipeline knows where a word begins.
/// ``WordTiming`` distributes a chunk's real duration across its words in
/// proportion to how long each word is in characters, and that is all it is. It
/// is right often enough to be useful for a highlight at sentence scale and
/// wrong in the ways you would expect: a long word said fast, a short word
/// held, a pause before a clause. The error grows with the length of the chunk,
/// because a single bad guess early shifts everything after it — one sentence
/// is usually fine, a long paragraph read as one chunk is not. If you need real
/// alignment you need a forced aligner; this is not one, and pretending
/// otherwise would be worse than the estimate.
///
/// Both are computed *after* any time-stretch, on the waveform the caller
/// actually receives, so `Result.speed` needs no correction applied to them.
public enum Timing {
    /// Lay rendered chunks end to end and time them.
    ///
    /// Offsets accumulate in **samples**, not seconds, and are divided by the
    /// rate once at the end. Accumulating seconds instead would make chunk
    /// *k*'s `end` and chunk *k+1*'s `start` two different sums of the same
    /// doubles, differing in the last bit — a gap or an overlap of a few
    /// nanoseconds, invisible to a test that compares with a tolerance and
    /// visible as a flicker in a highlight that switches on `time >= start`.
    public static func timeline(_ spans: [ChunkSpan], sampleRate: Int) -> [ChunkTiming] {
        var out: [ChunkTiming] = []
        out.reserveCapacity(spans.count)
        var at = 0
        for span in spans {
            let start = Double(at) / Double(sampleRate)
            at += span.samples
            let end = Double(at) / Double(sampleRate)
            out.append(
                ChunkTiming(
                    text: span.text, start: start, end: end, tokens: span.tokens,
                    words: estimateWords(span.text, start: start, end: end)))
        }
        return out
    }

    /// Split `text` on whitespace and share `[start, end]` out by length.
    ///
    /// The allocation is by **character count**, not by token count or by any
    /// acoustic measure: a word's characters are the only thing known here, and
    /// they correlate with duration well enough at sentence scale to drive a
    /// highlight. Whitespace itself is not charged for — the gap between two
    /// words belongs to whichever side of the boundary the caller's player is
    /// on, and splitting it would only invent a third kind of span.
    ///
    /// Boundaries are computed from a running character total rather than by
    /// adding per-word durations, so the spans cannot drift: the first `start`
    /// is exactly `start`, the last `end` is exactly `end`, and every interior
    /// boundary is shared by the two words that meet at it.
    ///
    /// Length is counted in Unicode scalars, the same way ``Chunking`` counts
    /// them and the same way the other four ports do (`utf8.RuneCountInString`,
    /// `chars().count`, `[...w].length`). `String.count` counts grapheme
    /// clusters, which is a third answer again, and it would give Polish and
    /// Japanese text different word weights in Swift than in Python for text
    /// that reads identically.
    public static func estimateWords(
        _ text: String, start: Double, end: Double
    ) -> [WordTiming] {
        let words = text.split(whereSeparator: { $0.isWhitespace })
        let lengths = words.map { $0.unicodeScalars.count }
        let total = lengths.reduce(0, +)
        if total == 0 { return [] }
        let span = end - start
        var out: [WordTiming] = []
        out.reserveCapacity(words.count)
        var seen = 0
        for (word, length) in zip(words, lengths) {
            let at = start + span * (Double(seen) / Double(total))
            seen += length
            out.append(
                WordTiming(
                    text: String(word), start: at,
                    end: start + span * (Double(seen) / Double(total))))
        }
        return out
    }
}

/// What one rendered chunk contributes to a timeline.
///
/// The three facts the engine has at concatenation time and nothing else: the
/// text it was asked to speak (post-funnel, which is what was tokenised), how
/// many samples it rendered to, and how many speech tokens it took. Kept as an
/// input type rather than assembling ``ChunkTiming`` per chunk, because the
/// offsets are only knowable once the order is known.
public struct ChunkSpan: Sendable, Equatable {
    public var text: String
    public var samples: Int
    public var tokens: Int

    public init(text: String, samples: Int, tokens: Int) {
        self.text = text
        self.samples = samples
        self.tokens = tokens
    }
}

/// One word's estimated span, in seconds from the start of the synthesis.
///
/// **Estimated, by proportional allocation.** The chunk's real duration is
/// divided among its words in proportion to their length in characters. There
/// is no alignment model here and no per-word measurement — see ``Timing`` for
/// what that costs you.
public struct WordTiming: Sendable, Equatable {
    /// The word as it appears in the chunk, punctuation included.
    ///
    /// Punctuation stays attached because the split is on whitespace: a caller
    /// highlighting `"end."` wants the full stop lit with the word, and a
    /// caller matching back against their own text needs the substring to be a
    /// substring.
    public var text: String
    public var start: Double
    public var end: Double

    public init(text: String, start: Double, end: Double) {
        self.text = text
        self.start = start
        self.end = end
    }
}

/// One chunk's exact span, and its words' estimated ones.
///
/// The two tiers in one object on purpose: a caller that trusts only the exact
/// tier reads `start`/`end` and ignores `words`, and the field names make it
/// impossible to reach the estimate by accident.
public struct ChunkTiming: Sendable, Equatable {
    /// The chunk's text after the speech funnel — what was tokenised, which is
    /// not always what the caller passed in (Polish respells embedded English,
    /// and numbers are read as words).
    public var text: String

    /// Seconds from the start of this `Engine.Result`'s audio.
    ///
    /// Zero for the first chunk, and for every chunk delivered by
    /// ``Engine/stream(_:voice:seed:language:speed:previousTokens:shouldCancel:onChunk:)``:
    /// a streamed chunk is its own render and does not know what preceded it,
    /// so the caller stitching the stream adds the offsets.
    public var start: Double

    public var end: Double

    /// Speech tokens this chunk generated. Duration over tokens is the pacing
    /// the postprocess detectors measure against, which is the other reason to
    /// carry it.
    public var tokens: Int

    public var words: [WordTiming]

    public init(
        text: String, start: Double, end: Double, tokens: Int, words: [WordTiming] = []
    ) {
        self.text = text
        self.start = start
        self.end = end
        self.tokens = tokens
        self.words = words
    }

    public var duration: Double { end - start }

    /// This timing moved later by `by` seconds, words included.
    ///
    /// What a caller stitching a stream applies: each delivered chunk starts at
    /// zero, and only the caller knows how much audio it has already queued.
    public func shifted(by: Double) -> ChunkTiming {
        ChunkTiming(
            text: text, start: start + by, end: end + by, tokens: tokens,
            words: words.map { WordTiming(text: $0.text, start: $0.start + by, end: $0.end + by) })
    }
}
