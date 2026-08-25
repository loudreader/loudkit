// Package timing mirrors loudkit.timing: where each chunk — and,
// approximately, each word — lands in the waveform.
//
// A reading app highlights the sentence it is speaking. That needs two
// different kinds of answer, and this package is careful to keep them apart,
// because conflating them is how a feature like this becomes a lie.
//
// Chunk times are exact. The engine renders each chunk to its own waveform and
// concatenates them, so it knows every chunk's sample offset and sample length
// without estimating anything. ChunkTiming reports those, converted to seconds.
// Chunk k's End is bit-identical to chunk k+1's Start: both are the same
// integer sample offset divided by the same sample rate, so a highlight driven
// by them can neither gap nor overlap.
//
// Word times are estimated. The model emits speech tokens, not an alignment;
// nothing in this pipeline knows where a word begins. WordTiming distributes a
// chunk's real duration across its words in proportion to how long each word is
// in code points, and that is all it is. It is right often enough to be useful
// for a highlight at sentence scale and wrong in the ways you would expect: a
// long word said fast, a short word held, a pause before a clause. The error
// grows with the length of the chunk, because a single bad guess early shifts
// everything after it — one sentence is usually fine, a long paragraph read as
// one chunk is not. If you need real alignment, you need a forced aligner; this
// is not one, and pretending otherwise would be worse than the estimate.
//
// Both are computed after any time-stretch, on the waveform the caller actually
// receives, so a result rendered at speed 1.5 needs no 1/speed correction
// applied to them.
package timing

import (
	"strings"
	"unicode/utf8"
)

// Span is what one rendered chunk contributes to a timeline.
//
// The three facts the engine has at concatenation time and nothing else: the
// text it was asked to speak (post-funnel, which is what was tokenised), how
// many samples it rendered to, and how many speech tokens it took. An input
// type rather than a per-chunk ChunkTiming, because the offsets are only
// knowable once the order is known.
type Span struct {
	Text    string
	Samples int
	Tokens  int
}

// WordTiming is one word's estimated span, in seconds from the start of the
// synthesis.
//
// Estimated, by proportional allocation. The chunk's real duration is divided
// among its words in proportion to their length in code points. There is no
// alignment model here and no per-word measurement — see the package comment
// for what that costs you.
type WordTiming struct {
	// Text is the word as it appears in the chunk, punctuation included.
	//
	// Punctuation stays attached because the split is on whitespace: a caller
	// highlighting "end." wants the full stop lit with the word, and a caller
	// matching back against their own text needs the substring to be a
	// substring.
	Text  string
	Start float64
	End   float64
}

// ChunkTiming is one chunk's exact span, and its words' estimated ones.
//
// The two tiers in one value on purpose: a caller that trusts only the exact
// tier reads Start/End and ignores Words, and the field names make it
// impossible to reach the estimate by accident.
type ChunkTiming struct {
	// Text is the chunk's text after the speech funnel — what was tokenised,
	// which is not always what the caller passed in (Polish respells embedded
	// English, and numbers are read as words).
	Text string

	// Start is seconds from the start of this synthesis's audio.
	//
	// Zero for the first chunk, and for every chunk handed to a Stream
	// callback: a streamed chunk is its own result and does not know what
	// preceded it, so the caller stitching the stream adds the offsets.
	Start float64

	End float64

	// Tokens is the speech tokens this chunk generated. Duration over tokens is
	// the pacing the postprocess detectors measure against, which is the other
	// reason to carry it.
	Tokens int

	Words []WordTiming
}

// Duration is End - Start.
func (c ChunkTiming) Duration() float64 { return c.End - c.Start }

// Shifted is this timing moved later by by seconds, words included.
//
// What a caller stitching a stream needs: each streamed chunk starts at zero,
// and the offsets are the caller's running total. Timeline does the same job in
// samples when the whole passage is in hand, which is exact; this is the best
// available when it is not.
func (c ChunkTiming) Shifted(by float64) ChunkTiming {
	words := make([]WordTiming, len(c.Words))
	for i, w := range c.Words {
		words[i] = WordTiming{Text: w.Text, Start: w.Start + by, End: w.End + by}
	}
	return ChunkTiming{
		Text:   c.Text,
		Start:  c.Start + by,
		End:    c.End + by,
		Tokens: c.Tokens,
		Words:  words,
	}
}

// Timeline lays rendered chunks end to end and times them.
//
// Offsets accumulate in samples, not seconds, and are divided by the rate once
// at the end. Accumulating seconds instead would make chunk k's End and chunk
// k+1's Start two different sums of the same floats, differing in the last bit
// — a gap or an overlap of a few nanoseconds, invisible in a test that compares
// with a tolerance and visible as a flicker in a highlight that switches on
// time >= start.
func Timeline(spans []Span, sampleRate int) []ChunkTiming {
	out := make([]ChunkTiming, 0, len(spans))
	at := 0
	for _, span := range spans {
		start := float64(at) / float64(sampleRate)
		at += span.Samples
		end := float64(at) / float64(sampleRate)
		out = append(out, ChunkTiming{
			Text:   span.Text,
			Start:  start,
			End:    end,
			Tokens: span.Tokens,
			Words:  EstimateWords(span.Text, start, end),
		})
	}
	return out
}

// EstimateWords splits text on whitespace and shares [start, end] out by length.
//
// The allocation is by code-point count, not by token count or by any acoustic
// measure: a word's characters are the only thing known here, and they
// correlate with duration well enough at sentence scale to drive a highlight.
// Code points rather than bytes so that the same text weights the same way in
// all five ports — a Polish "ł" is one character to a reader and two bytes to
// Go, and byte length would make an accented passage drift against the same
// passage read by the Python engine.
//
// Whitespace itself is not charged for — the gap between two words belongs to
// whichever side of the boundary the caller's player is on, and splitting it
// would only invent a third kind of span.
//
// Boundaries are computed from a running character total rather than by adding
// per-word durations, so the spans cannot drift: the first Start is exactly
// start, the last End is exactly end, and every interior boundary is shared by
// the two words that meet at it.
func EstimateWords(text string, start, end float64) []WordTiming {
	words := strings.Fields(text)
	total := 0
	lengths := make([]int, len(words))
	for i, w := range words {
		lengths[i] = utf8.RuneCountInString(w)
		total += lengths[i]
	}
	if total == 0 {
		return nil
	}
	span := end - start
	out := make([]WordTiming, 0, len(words))
	seen := 0
	for i, word := range words {
		at := start + span*(float64(seen)/float64(total))
		seen += lengths[i]
		out = append(out, WordTiming{
			Text:  word,
			Start: at,
			End:   start + span*(float64(seen)/float64(total)),
		})
	}
	return out
}
