package config

import "testing"

// The manifest 0.1.1 ships, as tools/amend_manifest.py writes it over the
// values the packed checkpoint already carried.
//
// Written out rather than assembled from ProductionWindow and the Production
// presets, because those are what the reader is being checked against: a
// manifest built from them would agree with a reader that ignored every key
// and answered its defaults, which is the exact failure this pins.
func shipped011() map[string]interface{} {
	return map[string]interface{}{
		"edge_fade_seconds": 0.02,
		"guidance":          "single_path",
		"guidance_rate":     0.0,
		"recipe_version":    "loudkit-1",
		"postprocess": map[string]interface{}{
			"mode":                              "trim",
			"ceiling_speech_per_text_token":     4.0,
			"ceiling_slack_tokens":              float64(40),
			"trailing_filler_threshold":         0.7,
			"trailing_silence_run_tokens":       float64(12),
			"filler_min_eos_probability":        0.05,
			"filler_max_speech_after_run":       float64(10),
			"desperation_speech_per_text_token": 4.5,
			"desperation_min_text_tokens":       float64(10),
			"ended_tail_silence_run":            float64(6),
			"ended_tail_blip_max":               float64(2),
			"ended_tail_word_max":               float64(10),
			"ended_tail_keep":                   float64(5),
			"echo_strong_eos_probability":       0.1,
			"echo_strong_max_tail":              float64(30),
			"echo_strong_min_position_pct":      float64(68),
			"echo_weak_eos_probability":         0.003,
			"echo_weak_max_tail":                float64(16),
			"echo_weak_min_position_pct":        float64(85),
		},
		"window": map[string]interface{}{
			"max_speech_tokens":    float64(255),
			"static_length":        float64(255),
			"pad_token_id":         float64(4254),
			"static_prompt_tokens": float64(238),
		},
		"eos_floor": map[string]interface{}{
			"min_tokens_floor":      float64(10),
			"min_tokens_text_ratio": 1.2,
		},
		"chunking": map[string]interface{}{
			"enabled":       true,
			"max_tokens":    float64(255),
			"prefix_tokens": float64(6),
			"split_on":      []interface{}{". ", "! ", "? ", "; ", ", "},
			"abbreviations": []interface{}{
				"A", "B", "Cpn", "D", "Dr", "F", "H", "Hr", "I", "J", "K", "M",
				"Mr", "Mrs", "R", "S", "St", "T", "V", "Vors", "dr", "mrs",
				"prof", "św",
			},
			"mid_sentence_period": "hold",
		},
		"sample_rate":       float64(24000),
		"token_rate_hz":     25.0,
		"speech_vocab_size": float64(8194),
		"n_cfm_timesteps":   float64(2),
		"speech_tokens": map[string]interface{}{
			"start": float64(6561),
			"stop":  float64(6562),
		},
		"sampling_defaults": map[string]interface{}{
			"temperature":        0.8,
			"repetition_penalty": 1.2,
			"min_p":              0.05,
			"max_new_tokens":     float64(255),
		},
		"silence_token_ids": []interface{}{
			float64(1731), float64(1821), float64(1822), float64(1824), float64(1975),
			float64(2058), float64(2068), float64(3190), float64(3377), float64(3918),
			float64(3927), float64(3928), float64(3930), float64(4008), float64(4009),
			float64(4011), float64(4012), float64(4137), float64(4146), float64(4161),
			float64(4171), float64(4173), float64(4174), float64(4218), float64(4245),
			float64(4251), float64(4252), float64(4254), float64(4255), float64(4260),
			float64(4282),
		},
		"silence_render_ids": []interface{}{
			float64(4137), float64(4215), float64(4218), float64(4299), float64(6162),
			float64(6324), float64(6405), float64(6486),
		},
		"quiet_render_ids": []interface{}{
			float64(1458), float64(1461), float64(1488), float64(1701), float64(1704),
			float64(1707), float64(1716), float64(1731), float64(1785), float64(1788),
			float64(1869), float64(1947), float64(1950), float64(1951), float64(1959),
			float64(1978), float64(2028), float64(2031), float64(2040), float64(2058),
			float64(2076), float64(2112), float64(2139), float64(3645), float64(3648),
			float64(3651), float64(3704), float64(3888), float64(3894), float64(4188),
			float64(5838), float64(6081), float64(6183), float64(6537),
		},
	}
}

// The shipped manifest reads to the shipped fingerprint, through the reader
// rather than around it.
//
// Every other check of this number builds the config by hand and compares the
// canonical form, which pins the *writer*: it holds even if FromManifest
// ignores every key and answers its defaults. This one starts where a user
// starts, at the manifest a released checkpoint carries, so a reader change
// that drops a key or reads it differently moves the number here instead of
// reaching a listener. `7cd75498ad4e7531` is the algorithm this build
// implements, recorded in CHANGELOG.md and in the conformance vectors. A
// published voice records the fingerprint it was rendered under, which is a
// fact about that artefact and does not move with this one.
func TestTheShippedManifestStillReadsToTheShippedFingerprint(t *testing.T) {
	const shippedFingerprint = "7cd75498ad4e7531"
	cfg := mustFromManifest(t, shipped011())
	if got := Fingerprint(cfg); got != shippedFingerprint {
		t.Errorf("the shipped 0.1.1 manifest now reads as %s, not %s\ncanonical form: %s",
			got, shippedFingerprint, CanonicalForm(cfg))
	}
	// Named separately because it is the block whose sub-keys are easiest to
	// drop silently: the window recipe is the whole measured deviation between
	// two backends' renders, and a dropped static_length frames it ragged.
	// Compared field by field: WindowConfig carries three pointers, so `==`
	// would compare where the lengths live rather than what they are.
	want := ProductionWindow()
	for _, c := range []struct {
		name     string
		got, exp *int
	}{
		{"static_length", cfg.Window.StaticLength, want.StaticLength},
		{"static_prompt_tokens", cfg.Window.StaticPromptTokens, want.StaticPromptTokens},
		{"pad_token_id", cfg.Window.PadTokenID, want.PadTokenID},
	} {
		if c.got == nil || *c.got != *c.exp {
			t.Errorf("window.%s = %s, want %d", c.name, jsonOptInt(c.got), *c.exp)
		}
	}
	if cfg.Window.MaxSpeechTokens != want.MaxSpeechTokens {
		t.Errorf("window.max_speech_tokens = %d, want %d",
			cfg.Window.MaxSpeechTokens, want.MaxSpeechTokens)
	}
}
