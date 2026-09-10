package config

import (
	"strings"
	"testing"
)

// TestTheReaderRefusesWhatTheReferenceRefuses walks the manifests the reference
// will not load.
//
// Every row is a value this reader used to accept and hash into a fingerprint,
// so the divergence it records is agreement. The messages are the reference's,
// checked in full rather than by substring: the wording is what tells a
// checkpoint author which key to fix, and a paraphrase here is a second
// vocabulary for the same refusal.
func TestTheReaderRefusesWhatTheReferenceRefuses(t *testing.T) {
	for _, c := range []struct {
		name string
		over map[string]interface{}
		want string
	}{
		// The window recipe. A static buffer shorter than the window it frames
		// truncates rather than fails, so the loss is 1.8 seconds of audio and
		// no error anywhere.
		{"static under window", map[string]interface{}{"window": map[string]interface{}{
			"max_speech_tokens": float64(300), "static_length": float64(255),
			"static_prompt_tokens": float64(238), "pad_token_id": float64(4254),
		}}, "static_length 255 cannot be shorter than max_speech_tokens 300"},
		{"prompt count zero", map[string]interface{}{"window": map[string]interface{}{
			"max_speech_tokens": float64(255), "static_length": float64(255),
			"static_prompt_tokens": float64(0), "pad_token_id": float64(4254),
		}}, "static_prompt_tokens must be positive: 0"},

		// The flow schedule. Zero steps runs the loop no times and renders the
		// prior noise field; the grid is what a checkpoint ships when the
		// schedule has to match across implementations.
		{"no steps", map[string]interface{}{"n_cfm_timesteps": float64(0)},
			"euler_steps must be >= 1: 0"},
		{"short grid", map[string]interface{}{"euler_grid": []interface{}{float64(0), float64(1)}},
			"euler_grid has 2 points, expected 3"},
		{"empty grid", map[string]interface{}{"euler_grid": []interface{}{}},
			"euler_grid has 0 points, expected 3"},
		{"descending grid", map[string]interface{}{
			"euler_grid": []interface{}{float64(1), float64(0.5), float64(0)}},
			"euler_grid must be strictly increasing"},
		{"grid off the unit interval", map[string]interface{}{
			"euler_grid": []interface{}{float64(0.1), float64(0.5), float64(1)}},
			"euler_grid must run from 0.0 to 1.0"},

		// The sampling law. A zero temperature divides by zero in the sampler
		// and degenerates to greedy argmax under a fingerprint that records
		// the law it is not running.
		{"zero temperature", sampling(map[string]interface{}{"temperature": float64(0)}),
			"temperature out of range: 0.0"},
		{"hot", sampling(map[string]interface{}{"temperature": float64(5)}),
			"temperature out of range: 5.0"},
		{"rewarding repetition", sampling(map[string]interface{}{"repetition_penalty": float64(0.5)}),
			"repetition_penalty below 1.0 rewards repetition: 0.5"},
		{"min_p over one", sampling(map[string]interface{}{"min_p": float64(1.5)}),
			"min_p out of range: 1.5"},
		{"min_p negative", sampling(map[string]interface{}{"min_p": float64(-1)}),
			"min_p out of range: -1.0"},
		{"no tokens", sampling(map[string]interface{}{"max_new_tokens": float64(0)}),
			"max_new_tokens must be positive: 0"},

		// The values every duration and every index is computed from.
		{"no rate", map[string]interface{}{"token_rate_hz": float64(0)},
			"token_rate_hz must be > 0: 0.0"},
		{"no vocabulary", map[string]interface{}{"speech_vocab_size": float64(0)},
			"speech_vocab_size must be >= 1: 0"},
		{"one marker", map[string]interface{}{"speech_tokens": map[string]interface{}{
			"start": float64(6561), "stop": float64(6561)}},
			"start_speech_token and stop_speech_token must differ: both are 6561"},
		{"marker past the table", map[string]interface{}{"speech_tokens": map[string]interface{}{
			"start": float64(99999), "stop": float64(6562)}},
			"start_speech_token must be in [0, 8194): 99999"},

		// The edge ramp. Zero is this port's spelling of "never named one", and
		// a manifest that writes it down is asking for something else.
		{"no fade", map[string]interface{}{"edge_fade_seconds": float64(0)},
			"edge_fade_seconds must be in [0.001, 0.05]: 0.0"},
		{"ten seconds of fade", map[string]interface{}{"edge_fade_seconds": float64(10)},
			"edge_fade_seconds must be in [0.001, 0.05]: 10.0"},

		// The splitter, whose rules chunking.Validate has always carried and
		// which nothing on this path called.
		{"no chunk budget", chunkingOver(map[string]interface{}{"max_tokens": float64(0)}),
			"chunking.max_tokens must be positive: 0"},
		{"prefix is the whole chunk", chunkingOver(map[string]interface{}{"prefix_tokens": float64(255)}),
			"chunking.prefix_tokens must be in [0, max_tokens): 255"},
		{"an abbreviation of nothing", chunkingOver(map[string]interface{}{
			"abbreviations": []interface{}{""}}),
			"chunking.abbreviations cannot contain an empty string"},
		{"chunks past the window", chunkingOver(map[string]interface{}{"max_tokens": float64(99999)}),
			"chunking.max_tokens 99999 exceeds the render window (255): every chunk " +
				"would be sized past what the renderer accepts, and the refusal " +
				"would land mid-stream, after audio had already been delivered"},

		// The detectors, which had no validator at all.
		{"negative retries", postprocessOver(map[string]interface{}{
			"retry_max_attempts": float64(-1)}),
			"retry_max_attempts must be in [0, 8): -1. Above that the ladder's " +
				"derived seeds run into the streams the chunk seeds use."},
		{"a share above one", postprocessOver(map[string]interface{}{
			"trailing_filler_threshold": float64(1.5)}),
			"trailing_filler_threshold must be in (0, 1]: 1.5"},
		{"a probability above one", postprocessOver(map[string]interface{}{
			"filler_min_eos_probability": float64(2)}),
			"filler_min_eos_probability out of range: 2.0"},
		{"a percentage past a hundred", postprocessOver(map[string]interface{}{
			"echo_strong_min_position_pct": float64(200)}),
			"echo_strong_min_position_pct is a percentage: 200"},
	} {
		m := shipped011()
		for k, v := range c.over {
			m[k] = v
		}
		_, err := FromManifest(m)
		if err == nil {
			t.Errorf("%s: loaded; the reference refuses it with %q", c.name, c.want)
			continue
		}
		if err.Error() != c.want {
			t.Errorf("%s:\n got %q\nwant %q", c.name, err.Error(), c.want)
		}
	}
}

// TestTheShippedManifestStillLoads is the other half: nothing above may refuse
// the checkpoint the release ships.
func TestTheShippedManifestStillLoads(t *testing.T) {
	cfg := mustFromManifest(t, shipped011())
	if err := cfg.Validate(); err != nil {
		t.Fatalf("the shipped 0.1.1 manifest is refused by its own validator: %v", err)
	}
	if got := Fingerprint(cfg); got != "7cd75498ad4e7531" {
		t.Errorf("the shipped manifest reads as %s", got)
	}
}

// sampling, chunkingOver and postprocessOver build a one-key override of a block
// that the shipped manifest already carries, so a probe changes one value and
// keeps the rest of the recipe.
func sampling(over map[string]interface{}) map[string]interface{} {
	return blockOver("sampling_defaults", over)
}

func chunkingOver(over map[string]interface{}) map[string]interface{} {
	return blockOver("chunking", over)
}

func postprocessOver(over map[string]interface{}) map[string]interface{} {
	return blockOver("postprocess", over)
}

func blockOver(key string, over map[string]interface{}) map[string]interface{} {
	base, _ := shipped011()[key].(map[string]interface{})
	merged := make(map[string]interface{}, len(base)+len(over))
	for k, v := range base {
		merged[k] = v
	}
	for k, v := range over {
		merged[k] = v
	}
	return map[string]interface{}{key: merged}
}

// TestValidateReadsTheSameOnAHandBuiltConfig: FromManifest is one door, and
// AlgorithmConfig is an ordinary struct a caller may fill in itself.
func TestValidateReadsTheSameOnAHandBuiltConfig(t *testing.T) {
	cfg := Defaults()
	cfg.Sampling.Temperature = 0
	err := cfg.Validate()
	if err == nil || !strings.Contains(err.Error(), "temperature out of range") {
		t.Fatalf("Validate on a hand-built config: %v", err)
	}
}
