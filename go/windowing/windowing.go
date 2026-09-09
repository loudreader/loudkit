// Package windowing mirrors loudkit.models.windowing: the renderer's pure
// geometry, which is the window framing recipe, the Euler grid, the EOS floor
// and the Philox stream ids.
package windowing

import (
	"errors"
	"fmt"
	"math"

	"github.com/loudreader/loudkit/go/config"
	"github.com/loudreader/loudkit/go/voice"
)

// The Philox sub-streams each stage draws from, under its own stage seed.
// The names and the values are Python's, in loudkit.models.windowing.
const (
	// FlowNoiseStream is the sub-stream for the CFM prior. Streams 0 and 1
	// are consumed by the Box-Muller pair, so keep any future draw at 2 or
	// above.
	FlowNoiseStream uint32 = 0
	// VocoderPhaseStream is the sub-stream for the eight harmonic phase
	// offsets. Row 0 is pinned to 0: the voiced fundamental must start at a
	// zero crossing.
	VocoderPhaseStream uint32 = 0
	// VocoderNoiseStream is the first of the Box-Muller pair (1 and 2) for
	// the excitation noise.
	VocoderNoiseStream uint32 = 1
)

// The tokens that frame a transcript segment. A property of the T3 text
// tokenizer family, kept here rather than beside the generator so the ONNX
// path can frame the same way, which is where Python keeps them too.
const (
	// StartTextToken opens the transcript segment.
	StartTextToken = 255
	// StopTextToken closes it.
	StopTextToken = 0
)

const (
	tokenMelRatio = 2
	melBins       = 80
)

// Framed is the output of the window recipe.
type Framed struct {
	Row          []int64   // (P+Q,) token row
	Cond         []float32 // (80 * 2*(P+Q)) mel condition
	PromptFrames int
	PromptTokens int // how many of Row's leading entries are prompt
	N            int
}

// TimeGrid returns the Euler time grid: the explicit one if configured, else
// cosine.
func TimeGrid(cfg config.AlgorithmConfig) []float64 {
	// The explicit path realises the doc comment above: "the explicit one if
	// configured". See config.EulerGrid.
	if len(cfg.EulerGrid) > 0 {
		out := make([]float64, len(cfg.EulerGrid))
		copy(out, cfg.EulerGrid)
		return out
	}
	k := cfg.EulerSteps
	grid := make([]float64, k+1)
	for i := 0; i <= k; i++ {
		grid[i] = 1.0 - math.Cos(float64(i)/float64(k)*math.Pi/2.0)
	}
	return grid
}

// ErrNoPadToken is returned by FrameWindows when a static-length window is
// configured but neither WindowConfig.PadTokenID nor Sampling.SilenceTokenIds
// gives it a token to pad with.
var ErrNoPadToken = errors.New(
	"static window needs a pad token: set WindowConfig.PadTokenID or " +
		"provide SilenceTokenIds: padding with token 0 bleeds +3 dB of " +
		"high-band energy into the tail through the encoder's attention",
)

func padTokenID(cfg config.AlgorithmConfig) (int, error) {
	if cfg.Window.PadTokenID != nil {
		return *cfg.Window.PadTokenID, nil
	}
	if len(cfg.Sampling.SilenceTokenIds) > 0 {
		return cfg.Sampling.SilenceTokenIds[0], nil
	}
	return 0, ErrNoPadToken
}

// staticPromptLen is how many prompt slots a static window reserves.
//
// A static window with no prompt count is Python's
// `p_len = w.static_prompt_tokens or len(prompt_tokens)`
// (models/windowing.py): the prompt fills the row as it is. Dereferencing the
// pointer instead panics on a configuration Python frames.
func staticPromptLen(w config.WindowConfig, promptTokens []int) int {
	if w.StaticPromptTokens != nil {
		return *w.StaticPromptTokens
	}
	return len(promptTokens)
}

// FrameWindows applies the window recipe. Returns ErrNoPadToken when the
// checkpoint manifest configures a static-length window without a pad token,
// and an error when more tokens are handed in than the window holds.
//
// An over-window input is refused with the amount of speech that would have
// been lost; silent truncation in a reading tool
// leaves the end of a passage nonexistent while the audio still
// sounds fine: the only listener who notices is one who knows the text. The
// Python engine refuses it loudly; so does this.
func FrameWindows(cfg config.AlgorithmConfig, tokens []int, v *voice.Profile) (Framed, error) {
	w := cfg.Window
	if len(tokens) > w.MaxSpeechTokens {
		return Framed{}, fmt.Errorf(
			"%d speech tokens exceed the %d-token window by %d; split the text first",
			len(tokens), w.MaxSpeechTokens, len(tokens)-w.MaxSpeechTokens)
	}
	toks := tokens
	n := len(toks)

	promptTokens := make([]int, len(v.PromptTokens))
	for i, t := range v.PromptTokens {
		promptTokens[i] = int(t)
	}
	promptMel := v.PromptMel
	promptMelFrames := len(promptMel) / melBins

	var prompt, query []int
	var condWidth, promptFrames int
	if w.StaticLength != nil {
		// The guard above measures against max_speech_tokens; the buffer below
		// is static_length long. A recipe whose static buffer is the shorter
		// of the two does not fail here, it truncates: max_speech_tokens 300
		// with static_length 255 dropped 45 speech tokens, 1.8 seconds of the
		// passage, and returned audio that sounds finished to everyone who
		// does not know the text.
		//
		// config.FromManifest refuses this recipe with the reference's
		// sentence, so no manifest reaches it. This is the second door,
		// because AlgorithmConfig is an ordinary struct a caller may build by
		// hand, and Framed.N would then report the count that went in rather
		// than the count the row carries.
		if err := w.Validate(); err != nil {
			return Framed{}, err
		}
		pad, err := padTokenID(cfg)
		if err != nil {
			return Framed{}, err
		}
		pLen := staticPromptLen(w, promptTokens)
		prompt = make([]int, pLen)
		for i := range prompt {
			prompt[i] = pad
		}
		for i := 0; i < min(len(promptTokens), pLen); i++ {
			prompt[i] = promptTokens[i]
		}
		query = make([]int, *w.StaticLength)
		for i := range query {
			query[i] = pad
		}
		copy(query, toks)
		condWidth = tokenMelRatio * (pLen + *w.StaticLength)
		promptFrames = tokenMelRatio * pLen
	} else {
		prompt = promptTokens
		query = toks
		condWidth = tokenMelRatio * (len(promptTokens) + n)
		promptFrames = tokenMelRatio * len(promptTokens)
	}

	row := make([]int64, len(prompt)+len(query))
	for i, t := range prompt {
		row[i] = int64(t)
	}
	for i, t := range query {
		row[len(prompt)+i] = int64(t)
	}

	cond := make([]float32, condWidth*melBins)
	keepF := min(promptMelFrames, promptFrames)
	for b := 0; b < melBins; b++ {
		for f := 0; f < keepF; f++ {
			cond[b*condWidth+f] = promptMel[b*promptMelFrames+f]
		}
	}

	return Framed{
		Row:          row,
		Cond:         cond,
		PromptFrames: promptFrames,
		PromptTokens: len(prompt),
		N:            n,
	}, nil
}
