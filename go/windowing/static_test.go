package windowing

import (
	"strings"
	"testing"

	"github.com/loudreader/loudkit/go/config"
	"github.com/loudreader/loudkit/go/voice"
)

// TestAStaticBufferShorterThanItsWindowIsRefused pins the truncation that used
// to be silent.
//
// The over-window guard measures against max_speech_tokens; the static query
// buffer is static_length long. When the second is the smaller, copy fills what
// fits and drops the rest: max_speech_tokens 300 with static_length 255 lost 45
// speech tokens, about 1.8 seconds, and Framed.N still reported 300, so a
// caller deriving duration from it announced 12.0 s of audio for 10.2 s of
// sound. Python and Swift refuse the manifest and Rust refuses the framing;
// this port now does both.
func TestAStaticBufferShorterThanItsWindowIsRefused(t *testing.T) {
	cfg := config.AlgorithmConfig{
		Window: config.WindowConfig{
			MaxSpeechTokens:    300,
			StaticLength:       intp(255),
			StaticPromptTokens: intp(238),
			PadTokenID:         intp(4254),
		},
	}
	v := &voice.Profile{
		PromptTokens: []int64{1, 2},
		PromptMel:    make([]float32, melBins*4),
	}
	tokens := make([]int, 300)

	_, err := FrameWindows(cfg, tokens, v)
	if err == nil {
		t.Fatal("a 255-token buffer framed a 300-token window without complaint")
	}
	const want = "static_length 255 cannot be shorter than max_speech_tokens 300"
	if err.Error() != want {
		t.Errorf("got %q, want %q", err.Error(), want)
	}
}

// TestFramedNCountsWhatTheRowCarries is the reason the refusal above matters:
// N is what a caller turns into seconds.
func TestFramedNCountsWhatTheRowCarries(t *testing.T) {
	cfg := config.AlgorithmConfig{Window: config.ProductionWindow()}
	v := &voice.Profile{
		PromptTokens: []int64{1, 2},
		PromptMel:    make([]float32, melBins*4),
	}
	tokens := make([]int, 255)
	for i := range tokens {
		tokens[i] = i % 100
	}

	framed, err := FrameWindows(cfg, tokens, v)
	if err != nil {
		t.Fatal(err)
	}
	if framed.N != len(tokens) {
		t.Fatalf("N = %d, want %d", framed.N, len(tokens))
	}
	// The query half of the row, after the static prompt.
	query := framed.Row[framed.PromptTokens:]
	for i, tok := range tokens {
		if query[i] != int64(tok) {
			t.Fatalf("query[%d] = %d, want %d: the row does not carry the tokens N counts",
				i, query[i], tok)
		}
	}
}

// TestAStaticBufferEqualToItsWindowStillFrames guards the fix from the other
// side: the shipped recipe has both at 255 and must keep working.
func TestAStaticBufferEqualToItsWindowStillFrames(t *testing.T) {
	cfg := config.AlgorithmConfig{Window: config.ProductionWindow()}
	v := &voice.Profile{
		PromptTokens: []int64{1, 2},
		PromptMel:    make([]float32, melBins*4),
	}
	if _, err := FrameWindows(cfg, make([]int, 255), v); err != nil {
		t.Fatalf("the shipped window recipe is refused: %v", err)
	}
}

// TestTheOverWindowRefusalStillNamesTheOverflow keeps the two refusals apart:
// too many tokens for the window is the caller's problem, and a buffer shorter
// than the window is the manifest's.
func TestTheOverWindowRefusalStillNamesTheOverflow(t *testing.T) {
	cfg := config.AlgorithmConfig{Window: config.ProductionWindow()}
	v := &voice.Profile{PromptTokens: []int64{1}, PromptMel: make([]float32, melBins*2)}
	_, err := FrameWindows(cfg, make([]int, 300), v)
	if err == nil || !strings.Contains(err.Error(), "exceed the 255-token window by 45") {
		t.Fatalf("over-window refusal reads %v", err)
	}
}
