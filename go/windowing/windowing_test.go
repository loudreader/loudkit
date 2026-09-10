package windowing

import (
	"errors"
	"testing"

	"github.com/loudreader/loudkit/go/config"
	"github.com/loudreader/loudkit/go/voice"
)

// Pins the error path: a static-length window configured without a pad token
// is a catchable error (ErrNoPadToken), not a crash of the whole process on a
// malformed checkpoint manifest, matching Python (ValueError), JS and Swift.
func TestFrameWindowsReturnsErrorWithoutPadToken(t *testing.T) {
	cfg := config.AlgorithmConfig{
		Window: config.WindowConfig{
			MaxSpeechTokens:    4,
			StaticLength:       intp(4),
			StaticPromptTokens: intp(2),
			// PadTokenID left nil, and no SilenceTokenIds below.
		},
	}
	v := &voice.Profile{
		PromptTokens: []int64{1, 2},
		PromptMel:    make([]float32, 80*4),
	}

	_, err := FrameWindows(cfg, []int{10, 20}, v)
	if err == nil {
		t.Fatal("expected an error, got nil")
	}
	if !errors.Is(err, ErrNoPadToken) {
		t.Fatalf("expected ErrNoPadToken, got %v", err)
	}
}

func TestFrameWindowsSucceedsWithSilenceFallback(t *testing.T) {
	cfg := config.AlgorithmConfig{
		Sampling: config.SamplingConfig{SilenceTokenIds: []int{7}},
		Window: config.WindowConfig{
			MaxSpeechTokens:    4,
			StaticLength:       intp(4),
			StaticPromptTokens: intp(2),
		},
	}
	v := &voice.Profile{
		PromptTokens: []int64{1, 2},
		PromptMel:    make([]float32, 80*4),
	}

	framed, err := FrameWindows(cfg, []int{10, 20}, v)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(framed.Row) != 6 { // 2 static prompt + 4 static query
		t.Fatalf("Row length = %d, want 6", len(framed.Row))
	}
}

func intp(v int) *int { return &v }

// A static window with no prompt count frames the prompt as it is.
// Python is `p_len = w.static_prompt_tokens or len(prompt_tokens)`, so the
// absent count means "as many as the voice brought"; dereferencing the nil
// pointer instead panicked on a window Python frames.
func TestFrameWindowsFramesAStaticWindowWithNoPromptCount(t *testing.T) {
	v := &voice.Profile{
		PromptTokens: []int64{1, 2, 3},
		PromptMel:    make([]float32, melBins*6),
	}
	base := config.WindowConfig{
		MaxSpeechTokens: 4,
		StaticLength:    intp(4),
		PadTokenID:      intp(7),
	}

	absent, err := FrameWindows(config.AlgorithmConfig{Window: base}, []int{10, 20}, v)
	if err != nil {
		t.Fatalf("static window with no prompt count: %v", err)
	}

	spelled := base
	spelled.StaticPromptTokens = intp(len(v.PromptTokens))
	explicit, err := FrameWindows(config.AlgorithmConfig{Window: spelled}, []int{10, 20}, v)
	if err != nil {
		t.Fatalf("static window with the count spelled out: %v", err)
	}

	if absent.PromptTokens != len(v.PromptTokens) {
		t.Errorf("PromptTokens = %d, want %d", absent.PromptTokens, len(v.PromptTokens))
	}
	if len(absent.Row) != len(explicit.Row) {
		t.Fatalf("Row length %d, want %d", len(absent.Row), len(explicit.Row))
	}
	for i := range explicit.Row {
		if absent.Row[i] != explicit.Row[i] {
			t.Fatalf("Row[%d] = %d, want %d", i, absent.Row[i], explicit.Row[i])
		}
	}
	if absent.PromptFrames != explicit.PromptFrames {
		t.Errorf("PromptFrames = %d, want %d", absent.PromptFrames, explicit.PromptFrames)
	}
}

// The ragged window, which is Python's default, carries no static fields at
// all, so recomputing the prompt length from them is a nil dereference.
// Framed.PromptTokens is the number DecodeMel reads instead, and the ragged
// branch has to report it.
func TestFrameWindowsReportsPromptTokensOnTheRaggedWindow(t *testing.T) {
	v := &voice.Profile{
		PromptTokens: []int64{1, 2, 3},
		PromptMel:    make([]float32, melBins*6),
	}
	cfg := config.AlgorithmConfig{Window: config.WindowConfig{MaxSpeechTokens: 255}}

	framed, err := FrameWindows(cfg, []int{10, 20}, v)
	if err != nil {
		t.Fatalf("ragged window: %v", err)
	}
	if framed.PromptTokens != len(v.PromptTokens) {
		t.Fatalf("PromptTokens = %d, want %d", framed.PromptTokens, len(v.PromptTokens))
	}
	if got := len(framed.Row) - framed.PromptTokens; got != 2 {
		t.Fatalf("query length = %d, want 2", got)
	}
	if framed.PromptFrames != tokenMelRatio*framed.PromptTokens {
		t.Fatalf("PromptFrames = %d, want %d", framed.PromptFrames, tokenMelRatio*framed.PromptTokens)
	}
}

// Over-window is refused, not trimmed.
//
// Slicing to MaxSpeechTokens leaves the end of a long passage nonexistent
// while the audio still sounds fine: the only listener who notices is
// one who already knows the text. Python refuses it loudly; this port does
// too.
func TestFrameWindowsRefusesMoreTokensThanTheWindowHolds(t *testing.T) {
	cfg := config.AlgorithmConfig{Window: config.WindowConfig{MaxSpeechTokens: 4}}
	v := &voice.Profile{
		PromptTokens: []int64{1, 2, 3},
		PromptMel:    make([]float32, melBins*6),
	}
	if _, err := FrameWindows(cfg, []int{1, 2, 3, 4, 5}, v); err == nil {
		t.Fatal("5 tokens in a 4-token window were accepted; the tail would vanish")
	}
}

// TestTimeGridHonoursAnExplicitGrid: the doc comment claims it, and the field
// exists to be honoured. An explicit grid exists because "cosine" is a formula
// five codebases can write five ways, so ignoring one means rendering on a
// different integration schedule, silently, under a fingerprint that records
// the grid being ignored.
func TestTimeGridHonoursAnExplicitGrid(t *testing.T) {
	cosine := TimeGrid(config.AlgorithmConfig{EulerSteps: 2})
	if len(cosine) != 3 || cosine[0] != 0.0 {
		t.Fatalf("cosine schedule looks wrong: %v", cosine)
	}
	explicit := TimeGrid(config.AlgorithmConfig{
		EulerSteps: 2, EulerGrid: []float64{0.0, 0.25, 1.0}})
	if len(explicit) != 3 || explicit[1] != 0.25 {
		t.Fatalf("explicit grid was ignored: %v", explicit)
	}
}
