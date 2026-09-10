package config

import (
	"errors"
	"fmt"
	"math"

	"github.com/loudreader/loudkit/go/internal/pyfmt"
)

// Validate refuses an algorithm the engine cannot run, or can run only by
// meaning something the manifest did not say.
//
// These are the reference's rules, from AlgorithmConfig.__post_init__ and the
// three dataclasses under it, in the reference's order and with its sentences.
// They live behind one call because a manifest is external data: the values
// here are hashed into the fingerprint that is supposed to catch a divergence,
// so a setting accepted and then quietly ignored is recorded as agreement.
//
// The reference validates each block as it is constructed, so the first
// refusal a malformed manifest gets is the same one in both. FromManifest
// calls this last, after every key has been read.
func (c AlgorithmConfig) Validate() error {
	if err := c.Postprocess.Validate(); err != nil {
		return err
	}
	if err := c.Chunking.Validate(); err != nil {
		return err
	}
	if err := c.Sampling.Validate(); err != nil {
		return err
	}
	if err := c.Window.Validate(); err != nil {
		return err
	}
	if c.Guidance == "single_path" && c.GuidanceRate != 0 {
		return errors.New("guidance_rate must be 0.0 in single_path mode")
	}
	if c.Guidance == "cfg_dual_path" && c.GuidanceRate <= 0 {
		return errors.New("cfg_dual_path with a zero rate does twice the work for nothing")
	}
	if c.EulerSteps < 1 {
		return fmt.Errorf("euler_steps must be >= 1: %d", c.EulerSteps)
	}
	if err := c.validateNumericCore(); err != nil {
		return err
	}
	if err := c.validateEulerGrid(); err != nil {
		return err
	}
	// The three token budgets have to agree, or a chunk overruns the render
	// window mid-stream, after earlier chunks have already played.
	window := c.Window.MaxSpeechTokens
	if c.Chunking.Enabled && c.Chunking.MaxTokens > window {
		return fmt.Errorf(
			"chunking.max_tokens %d exceeds the render window (%d): every chunk "+
				"would be sized past what the renderer accepts, and the refusal "+
				"would land mid-stream, after audio had already been delivered",
			c.Chunking.MaxTokens, window)
	}
	if c.Sampling.MaxNewTokens > window {
		return fmt.Errorf(
			"sampling.max_new_tokens %d exceeds the render window (%d): generation "+
				"is allowed to produce more speech than the renderer will accept, so "+
				"a long utterance fails after it has been generated rather than before",
			c.Sampling.MaxNewTokens, window)
	}
	return nil
}

// validateNumericCore is the reference's _validate_numeric_core: the values
// every duration, every index and every ramp is computed from.
func (c AlgorithmConfig) validateNumericCore() error {
	// Zero is this port's spelling of "the config never named a ramp", which a
	// manifest cannot say: FromManifest turns an absent key and an explicit
	// null into the 5 ms those releases shipped, so a zero arriving here was
	// written down.
	if c.EdgeFadeSeconds < 0.001 || c.EdgeFadeSeconds > 0.05 {
		return fmt.Errorf("edge_fade_seconds must be in [0.001, 0.05]: %s",
			pyfmt.Float(c.EdgeFadeSeconds))
	}
	if c.SampleRate <= 0 {
		return fmt.Errorf("sample_rate must be > 0: %d", c.SampleRate)
	}
	if c.TokenRateHz <= 0 {
		return fmt.Errorf("token_rate_hz must be > 0: %s", pyfmt.Float(c.TokenRateHz))
	}
	if c.SpeechVocabSize < 1 {
		return fmt.Errorf("speech_vocab_size must be >= 1: %d", c.SpeechVocabSize)
	}
	for _, t := range []struct {
		name  string
		value int
	}{
		{"start_speech_token", c.StartSpeech},
		{"stop_speech_token", c.StopSpeech},
	} {
		if t.value < 0 || t.value >= c.SpeechVocabSize {
			return fmt.Errorf("%s must be in [0, %d): %d",
				t.name, c.SpeechVocabSize, t.value)
		}
	}
	if c.StartSpeech == c.StopSpeech {
		return fmt.Errorf(
			"start_speech_token and stop_speech_token must differ: both are %d",
			c.StartSpeech)
	}
	return nil
}

// validateEulerGrid checks the explicit time grid against the step count it
// has to schedule. A grid of the wrong length integrates a different number of
// steps than the manifest declared; one that does not run from 0 to 1 stops
// the flow ODE somewhere the checkpoint was never trained for.
func (c AlgorithmConfig) validateEulerGrid() error {
	if c.EulerGrid == nil {
		return nil
	}
	if len(c.EulerGrid) != c.EulerSteps+1 {
		return fmt.Errorf("euler_grid has %d points, expected %d",
			len(c.EulerGrid), c.EulerSteps+1)
	}
	for i := 1; i < len(c.EulerGrid); i++ {
		if !(c.EulerGrid[i] > c.EulerGrid[i-1]) {
			return errors.New("euler_grid must be strictly increasing")
		}
	}
	if math.Abs(c.EulerGrid[0]) > 1e-6 || math.Abs(c.EulerGrid[len(c.EulerGrid)-1]-1.0) > 1e-6 {
		return errors.New("euler_grid must run from 0.0 to 1.0")
	}
	return nil
}

// Validate refuses a sampling law the sampler cannot run, with the sentences
// of SamplingConfig.__post_init__.
//
// A temperature of zero is the one that bites hardest: it divides by zero in
// the sampler and degenerates to greedy argmax, which is a different sampling
// law from the one the fingerprint records.
func (s SamplingConfig) Validate() error {
	if !(s.Temperature > 0 && s.Temperature <= 4.0) {
		return fmt.Errorf("temperature out of range: %s", pyfmt.Float(s.Temperature))
	}
	if s.RepetitionPenalty < 1.0 {
		return fmt.Errorf("repetition_penalty below 1.0 rewards repetition: %s",
			pyfmt.Float(s.RepetitionPenalty))
	}
	if !(s.MinP >= 0 && s.MinP < 1.0) {
		return fmt.Errorf("min_p out of range: %s", pyfmt.Float(s.MinP))
	}
	if s.MaxNewTokens <= 0 {
		return fmt.Errorf("max_new_tokens must be positive: %d", s.MaxNewTokens)
	}
	if s.MinTokensFloor < 0 {
		return fmt.Errorf("min_tokens_floor must be >= 0: %d", s.MinTokensFloor)
	}
	if s.MinTokensTextRatio < 0 {
		return fmt.Errorf("min_tokens_text_ratio must be >= 0: %s",
			pyfmt.Float(s.MinTokensTextRatio))
	}
	return nil
}

// Validate refuses a framing recipe the renderer cannot pad to, with the
// sentences of WindowConfig.__post_init__.
//
// A static query buffer shorter than the window it frames does not fail: it
// truncates. A manifest declaring max_speech_tokens 300 with static_length 255
// dropped 45 speech tokens, 1.8 seconds of the passage, and returned audio
// that sounds finished to everyone who does not know the text.
func (w WindowConfig) Validate() error {
	if w.StaticLength != nil && *w.StaticLength < w.MaxSpeechTokens {
		return fmt.Errorf(
			"static_length %d cannot be shorter than max_speech_tokens %d",
			*w.StaticLength, w.MaxSpeechTokens)
	}
	if w.StaticPromptTokens != nil && *w.StaticPromptTokens <= 0 {
		return fmt.Errorf("static_prompt_tokens must be positive: %d", *w.StaticPromptTokens)
	}
	return nil
}
