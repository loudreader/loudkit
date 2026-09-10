package postprocess

import (
	"fmt"
	"math"

	"github.com/loudreader/loudkit/go/internal/pyfmt"
)

// RetryLadderHeadroom is how many retry streams the seed ladder has room for:
// the gap between the engine's retry stream and its chunk streams, restated
// here because the engine cannot be imported without a cycle. Pinned to the
// originals by a test, as the reference pins its own.
const RetryLadderHeadroom = 8

// Validate refuses a detector preset that would make a rule mean nothing.
//
// The rules and their sentences are PostprocessConfig._validate_ranges in
// loudkit.postprocess, and they are here for the same reason they are there:
// these constants remove tokens, so a preset that reads as configuration and
// behaves as a disabled rule changes what a listener hears with nothing on
// screen to say so.
//
// The three closed-set keys (mode, repetition_resume, repetition_silence) are
// settled at the manifest door in config.FromManifest, which names the options
// it expected.
func (c Config) Validate() error {
	for _, f := range []struct {
		name  string
		value float64
	}{
		{"ceiling_speech_per_text_token", c.CeilingSpeechPerTextToken},
		{"trailing_filler_threshold", c.TrailingFillerThreshold},
		{"filler_min_eos_probability", c.FillerMinEosProbability},
		{"desperation_band_ratio", c.DesperationBandRatio},
		{"desperation_speech_per_text_token", c.DesperationSpeechPerTextToken},
		{"desperation_min_keep_per_text_token", c.DesperationMinKeepPerTextToken},
		{"echo_strong_eos_probability", c.EchoStrongEosProbability},
		{"echo_weak_eos_probability", c.EchoWeakEosProbability},
		{"pacing_tolerance", c.PacingTolerance},
	} {
		// Listed one by one so a NaN cannot walk through the comparisons
		// below: every ordering test against a NaN is false, so a threshold
		// set to one passes every range check and then never fires.
		if math.IsNaN(f.value) || math.IsInf(f.value, 0) {
			return fmt.Errorf("%s must be a finite number: %s", f.name, pyfmt.Float(f.value))
		}
	}
	if c.RetryMaxAttempts < 0 || c.RetryMaxAttempts >= RetryLadderHeadroom {
		return fmt.Errorf(
			"retry_max_attempts must be in [0, %d): %d. Above that the ladder's "+
				"derived seeds run into the streams the chunk seeds use.",
			RetryLadderHeadroom, c.RetryMaxAttempts)
	}
	if c.RepetitionMinCycles < 2 {
		// One cycle is not a repetition and two is the definition of one; a
		// threshold below two would cut every row that says a word twice.
		return fmt.Errorf("repetition_min_cycles must be at least 2: %d", c.RepetitionMinCycles)
	}
	if c.RepetitionMaxPeriod < 1 {
		return fmt.Errorf("repetition_max_period must be positive: %d", c.RepetitionMaxPeriod)
	}
	if c.StallRunTokens < 1 {
		// At zero every row with a single silence token before speech is a
		// stall, and "condemned" stops meaning anything.
		return fmt.Errorf("stall_run_tokens must be positive: %d", c.StallRunTokens)
	}
	if c.RepetitionMinSpan < c.RepetitionMinCycles {
		// A span shorter than the cycle count is unreachable: the shortest
		// qualifying loop is min_cycles copies of a one-token cycle.
		return fmt.Errorf(
			"repetition_min_span (%d) must be at least repetition_min_cycles (%d)",
			c.RepetitionMinSpan, c.RepetitionMinCycles)
	}
	return c.validateSeams()
}

// validateSeams carries the rules that relate one threshold to another, split
// out only because Validate had grown past the point where a reader could hold
// the whole of it.
func (c Config) validateSeams() error {
	if c.CeilingSpeechPerTextToken <= 0 {
		return fmt.Errorf("ceiling_speech_per_text_token must be positive: %s",
			pyfmt.Float(c.CeilingSpeechPerTextToken))
	}
	if c.DesperationSpeechPerTextToken <= c.CeilingSpeechPerTextToken {
		// The desperation rule exists for rows the ceiling let through. If it
		// triggered at or below the ceiling it would fire on every
		// ceiling-stopped row, including the ones the ceiling stopped
		// correctly, and "certainly broken" would stop meaning anything.
		return fmt.Errorf(
			"desperation_speech_per_text_token (%s) must exceed "+
				"ceiling_speech_per_text_token (%s): below it, the rule that means "+
				"'certainly broken' fires on rows the ceiling stopped correctly",
			pyfmt.Float(c.DesperationSpeechPerTextToken), pyfmt.Float(c.CeilingSpeechPerTextToken))
	}
	if c.DesperationMinKeepPerTextToken < 0 {
		return fmt.Errorf("desperation_min_keep_per_text_token must be >= 0: %s",
			pyfmt.Float(c.DesperationMinKeepPerTextToken))
	}
	if c.DesperationMinKeepPerTextToken > c.DesperationBandRatio {
		// The band top is where a real read could still have ended. Demanding
		// a keep above it condemns cuts landing exactly where the band admits
		// them, and "starved" stops meaning anything.
		return fmt.Errorf(
			"desperation_min_keep_per_text_token (%s) must not exceed "+
				"desperation_band_ratio (%s)",
			pyfmt.Float(c.DesperationMinKeepPerTextToken), pyfmt.Float(c.DesperationBandRatio))
	}
	if c.TrailingFillerThreshold <= 0 || c.TrailingFillerThreshold > 1 {
		return fmt.Errorf("trailing_filler_threshold must be in (0, 1]: %s",
			pyfmt.Float(c.TrailingFillerThreshold))
	}
	if c.FillerMinEosProbability < 0 || c.FillerMinEosProbability >= 1 {
		return fmt.Errorf("filler_min_eos_probability out of range: %s",
			pyfmt.Float(c.FillerMinEosProbability))
	}
	for _, f := range []struct {
		name  string
		value int
	}{
		{"ceiling_slack_tokens", c.CeilingSlackTokens},
		{"trailing_silence_run_tokens", c.TrailingSilenceRunTokens},
		// In the reference's order, which names these two among the counts:
		// a negative band floor closes the band the field exists to open, and
		// a negative dropout minimum leaves no row short enough to be one.
		{"desperation_band_floor", c.DesperationBandFloor},
		{"desperation_min_text_tokens", c.DesperationMinTextTokens},
		{"ended_tail_silence_run", c.EndedTailSilenceRun},
		{"ended_tail_blip_max", c.EndedTailBlipMax},
		{"ended_tail_word_max", c.EndedTailWordMax},
		{"filler_max_speech_after_run", c.FillerMaxSpeechAfterRun},
		{"ended_tail_keep", c.EndedTailKeep},
		{"echo_strong_max_tail", c.EchoStrongMaxTail},
		{"echo_weak_max_tail", c.EchoWeakMaxTail},
		{"dropout_min_tokens", c.DropoutMinTokens},
	} {
		if f.value < 0 {
			return fmt.Errorf("%s must be >= 0: %d", f.name, f.value)
		}
	}
	for _, f := range []struct {
		name  string
		value int
	}{
		{"echo_strong_min_position_pct", c.EchoStrongMinPositionPct},
		{"echo_weak_min_position_pct", c.EchoWeakMinPositionPct},
	} {
		if f.value < 0 || f.value > 100 {
			return fmt.Errorf("%s is a percentage: %d", f.name, f.value)
		}
	}
	return nil
}
