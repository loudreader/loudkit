// Timestamps: exact at the chunk, estimated at the word.
//
// The whole value of this feature is that a reading app can trust the first
// tier and be told, loudly, not to trust the second in the same way. So the
// tests are split the same way: the chunk assertions are equalities — down to
// the last bit, because a tolerance would hide exactly the defect that matters
// — and the word assertions are invariants (monotonic, inside the chunk, every
// word present). Nothing here claims a word lands where a listener would say it
// does.

package timing

import (
	"math"
	"testing"
)

const sampleRate = 24_000

// A highlight that switches on time >= start flickers on a gap and double-lights
// on an overlap, and both are invisible to a comparison with a tolerance.
// Offsets accumulate as integer samples for exactly this reason.
func TestChunksAreAdjacentToTheLastBit(t *testing.T) {
	got := Timeline([]Span{{"a b", 7_001, 3}, {"c d e", 13_337, 5}}, sampleRate)
	if got[0].Start != 0.0 {
		t.Errorf("the first chunk starts at %v, want 0", got[0].Start)
	}
	if got[1].Start != got[0].End {
		t.Errorf("chunk 1 starts at %v but chunk 0 ends at %v — a gap of %v",
			got[1].Start, got[0].End, got[1].Start-got[0].End)
	}
	if want := float64(7_001+13_337) / sampleRate; got[1].End != want {
		t.Errorf("the timeline ends at %v, want %v", got[1].End, want)
	}
}

func TestTheSpansCoverTheWholeRenderWithNothingLeftOver(t *testing.T) {
	got := Timeline([]Span{{"one", 100, 1}, {"two", 200, 2}, {"three", 300, 3}}, sampleRate)
	total := 0.0
	for _, c := range got {
		total += c.Duration()
	}
	if want := 600.0 / sampleRate; math.Abs(total-want) > 1e-12 {
		t.Errorf("the spans total %v seconds, want %v", total, want)
	}
	for i, want := range []int{1, 2, 3} {
		if got[i].Tokens != want {
			t.Errorf("chunk %d reports %d tokens, want %d", i, got[i].Tokens, want)
		}
	}
}

func TestAnEmptyRenderIsAnEmptyTimeline(t *testing.T) {
	if got := Timeline(nil, sampleRate); len(got) != 0 {
		t.Errorf("no spans gave %d timings", len(got))
	}
}

func TestWordsTileTheChunkWithoutGaps(t *testing.T) {
	words := EstimateWords("alpha beta gamma", 1.0, 4.0)
	if len(words) != 3 {
		t.Fatalf("got %d words, want 3", len(words))
	}
	if words[0].Start != 1.0 {
		t.Errorf("the first word starts at %v, want the chunk's own start", words[0].Start)
	}
	if words[2].End != 4.0 {
		t.Errorf("the last word ends at %v, want the chunk's own end", words[2].End)
	}
	for i := 0; i+1 < len(words); i++ {
		if words[i].End != words[i+1].Start {
			t.Errorf("word %d ends at %v but word %d starts at %v",
				i, words[i].End, i+1, words[i+1].Start)
		}
	}
}

func TestTimesAreMonotonicAndInsideTheChunk(t *testing.T) {
	words := EstimateWords("a bb ccc dddd e", 2.5, 3.25)
	previous := 2.5
	for _, w := range words {
		if !(2.5 <= w.Start && w.Start <= w.End && w.End <= 3.25) {
			t.Errorf("%q spans [%v, %v], outside the chunk's [2.5, 3.25]", w.Text, w.Start, w.End)
		}
		if w.Start < previous {
			t.Errorf("%q starts at %v, before the previous word's start %v", w.Text, w.Start, previous)
		}
		previous = w.Start
	}
}

// The whole content of the estimate: characters stand in for seconds. Nothing
// else here knows how long a word takes.
func TestALongerWordIsGivenLonger(t *testing.T) {
	words := EstimateWords("hi internationalisation", 0.0, 1.0)
	if len(words) != 2 {
		t.Fatalf("got %d words, want 2", len(words))
	}
	short := words[0].End - words[0].Start
	long := words[1].End - words[1].Start
	if long <= short {
		t.Errorf("the long word got %v s and the short one %v s", long, short)
	}
}

// A caller highlighting "end." wants the full stop lit with the word, and a
// caller matching back against their own text needs the substring to be a
// substring.
func TestPunctuationStaysWithItsWord(t *testing.T) {
	words := EstimateWords("Hello, world!", 0.0, 1.0)
	if len(words) != 2 || words[0].Text != "Hello," || words[1].Text != "world!" {
		t.Errorf("got %v, want [Hello, world!] with the punctuation attached", words)
	}
}

func TestNoTextIsNoWordsRatherThanADivisionByZero(t *testing.T) {
	if got := EstimateWords("   ", 0.0, 1.0); len(got) != 0 {
		t.Errorf("whitespace gave %d words", len(got))
	}
	if got := EstimateWords("", 0.0, 1.0); len(got) != 0 {
		t.Errorf("an empty string gave %d words", len(got))
	}
}

// The other four ports count code points too. A byte count would give Polish and
// Japanese text different word weights in Go than in Python, for text that reads
// identically — "żółć" is four characters to a reader and eight bytes to Go.
func TestLengthIsCountedInCodePointsNotBytes(t *testing.T) {
	words := EstimateWords("aaaa żółć", 0.0, 1.0)
	if len(words) != 2 {
		t.Fatalf("got %d words, want 2", len(words))
	}
	ascii := words[0].End - words[0].Start
	accented := words[1].End - words[1].Start
	if math.Abs(ascii-accented) > 1e-12 {
		t.Errorf("four ASCII characters got %v s and four accented ones %v s", ascii, accented)
	}
}

func TestEstimateWordsSurvivesAChunkOfOneWord(t *testing.T) {
	words := EstimateWords("word", 0.5, 0.75)
	if len(words) != 1 || words[0].Start != 0.5 || words[0].End != 0.75 {
		t.Errorf("got %v, want one word spanning exactly [0.5, 0.75]", words)
	}
}

// What a caller stitching a stream does with each chunk's own timing.
func TestShiftingMovesTheWordsWithTheChunk(t *testing.T) {
	span := Timeline([]Span{{"a bb", 240, 2}}, sampleRate)[0]
	moved := span.Shifted(1.0)
	if moved.Start != span.Start+1.0 || moved.End != span.End+1.0 {
		t.Errorf("the chunk moved to [%v, %v], want [%v, %v]",
			moved.Start, moved.End, span.Start+1.0, span.End+1.0)
	}
	if len(moved.Words) != len(span.Words) {
		t.Fatalf("shifting changed the word count: %d -> %d", len(span.Words), len(moved.Words))
	}
	for i, w := range moved.Words {
		if w.Start != span.Words[i].Start+1.0 || w.Text != span.Words[i].Text {
			t.Errorf("word %d moved to %v (%q), want %v (%q)",
				i, w.Start, w.Text, span.Words[i].Start+1.0, span.Words[i].Text)
		}
	}
	// The original is untouched: Shifted returns a new timing rather than
	// editing the words in place, and a caller stitching a stream keeps both.
	if span.Start != 0.0 {
		t.Errorf("shifting moved the original too: it now starts at %v", span.Start)
	}
}
