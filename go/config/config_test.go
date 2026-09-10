package config

import (
	"strings"
	"testing"

	"github.com/loudreader/loudkit/go/postprocess"
)

// Pins recipe_version defaulting: a manifest that omits the key falls back the
// way Python does, so a non-amended checkpoint does not get an empty recipe
// version in Go while every other port has one.
func TestFromManifestDefaultsRecipeVersionWhenAbsent(t *testing.T) {
	cfg := mustFromManifest(t, map[string]interface{}{})
	if cfg.RecipeVersion != "loudkit-1" {
		t.Fatalf("RecipeVersion = %q, want fallback %q", cfg.RecipeVersion, "loudkit-1")
	}
}

func TestFromManifestAcceptsTheOneRecipe(t *testing.T) {
	cfg := mustFromManifest(t, map[string]interface{}{
		"recipe_version": "loudkit-1",
		"chunking":       map[string]interface{}{},
		"postprocess":    map[string]interface{}{},
	})
	if cfg.RecipeVersion != "loudkit-1" {
		t.Fatalf("RecipeVersion = %q, want %q", cfg.RecipeVersion, "loudkit-1")
	}
}

// One recipe means one accepted value, and the error names what the manifest
// declared. Believing a foreign tag would fingerprint it; defaulting it would
// claim this recipe for a checkpoint that named another. All five ports
// refuse it identically.
func TestFromManifestRefusesAForeignRecipeVersion(t *testing.T) {
	_, err := FromManifest(map[string]interface{}{
		"recipe_version": "loudkit-9",
	})
	if err == nil {
		t.Fatal("a foreign recipe_version was accepted")
	}
	if got := err.Error(); !strings.Contains(got, "loudkit-9") {
		t.Fatalf("the error must name the declared tag: %q", got)
	}
}

// A tag that is not even a string is refused, not defaulted: a manifest one
// port misreads while another defaults is the divergence class this library
// exists to prevent.
func TestFromManifestRefusesANonStringRecipeVersion(t *testing.T) {
	if _, err := FromManifest(map[string]interface{}{
		"recipe_version": 9,
	}); err == nil {
		t.Fatal("a non-string recipe_version was accepted")
	}
}

// The detectors default on when the block is absent; the tag does not move
// for it: there is one recipe, and a manifest that omits a block left a
// shipping default unstated.
func TestFromManifestDefaultsTheDetectorsOnWhenPostprocessAbsent(t *testing.T) {
	cfg := mustFromManifest(t, map[string]interface{}{
		"recipe_version": "loudkit-1",
		"chunking":       map[string]interface{}{},
	})
	if cfg.RecipeVersion != "loudkit-1" {
		t.Fatalf("RecipeVersion = %q, want %q", cfg.RecipeVersion, "loudkit-1")
	}
	if cfg.Postprocess.Mode != "trim" {
		t.Fatalf("Postprocess.Mode = %q, want the shipping default", cfg.Postprocess.Mode)
	}
}

func TestFromManifestRefusesUnknownPostprocessMode(t *testing.T) {
	_, err := FromManifest(map[string]interface{}{
		"postprocess": map[string]interface{}{"mode": "shave"},
	})
	if err == nil {
		t.Fatal("expected an error for an unknown postprocess mode")
	}
}

// A law this port does not implement must be refused, not defaulted: the
// resolver would cut where the manifest said to condemn. "cut" names the
// pre-amendment law and is read.
func TestFromManifestRefusesUnknownRepetitionResume(t *testing.T) {
	_, err := FromManifest(map[string]interface{}{
		"postprocess": map[string]interface{}{"repetition_resume": "maybe"},
	})
	if err == nil {
		t.Fatal("an unknown repetition_resume was accepted")
	}
	cfg := mustFromManifest(t, map[string]interface{}{
		"postprocess": map[string]interface{}{"repetition_resume": "cut"},
	})
	if cfg.Postprocess.RepetitionResume != "cut" {
		t.Fatalf("RepetitionResume = %q, want %q", cfg.Postprocess.RepetitionResume, "cut")
	}
}

// A family this port does not implement must be refused, not defaulted: the
// loop exemption would read one silence list under a manifest declaring
// another. "sampling" names the pre-amendment law and is read.
func TestFromManifestRefusesUnknownRepetitionSilence(t *testing.T) {
	_, err := FromManifest(map[string]interface{}{
		"postprocess": map[string]interface{}{"repetition_silence": "both"},
	})
	if err == nil {
		t.Fatal("an unknown repetition_silence was accepted")
	}
	cfg := mustFromManifest(t, map[string]interface{}{
		"postprocess": map[string]interface{}{"repetition_silence": "sampling"},
	})
	if cfg.Postprocess.RepetitionSilence != "sampling" {
		t.Fatalf("RepetitionSilence = %q, want %q",
			cfg.Postprocess.RepetitionSilence, "sampling")
	}
}

// The render censuses ride the manifest top level, beside silence_token_ids;
// declaring them inside the postprocess block would give one value two homes
// in one file, so it is refused by name: the same way Python's block reader
// refuses it.
func TestFromManifestRefusesRenderIdsInsideThePostprocessBlock(t *testing.T) {
	for _, key := range []string{"silence_render_ids", "quiet_render_ids"} {
		_, err := FromManifest(map[string]interface{}{
			"postprocess": map[string]interface{}{key: []interface{}{1.0, 2.0}},
		})
		if err == nil {
			t.Fatalf("%s inside the postprocess block was accepted", key)
		}
	}
}

// The top-level censuses reach the detectors, with or without a postprocess
// block: a manifest with no detector overrides still carries the properties
// of its weights.
func TestFromManifestReadsTheTopLevelCensuses(t *testing.T) {
	cfg := mustFromManifest(t, map[string]interface{}{
		"silence_render_ids": []interface{}{7.0, 8.0},
		"quiet_render_ids":   []interface{}{9.0},
	})
	if len(cfg.Postprocess.SilenceRenderIds) != 2 || cfg.Postprocess.SilenceRenderIds[0] != 7 {
		t.Fatalf("SilenceRenderIds = %v, want [7 8]", cfg.Postprocess.SilenceRenderIds)
	}
	if len(cfg.Postprocess.QuietRenderIds) != 1 || cfg.Postprocess.QuietRenderIds[0] != 9 {
		t.Fatalf("QuietRenderIds = %v, want [9]", cfg.Postprocess.QuietRenderIds)
	}

	withBlock := mustFromManifest(t, map[string]interface{}{
		"silence_render_ids": []interface{}{7.0},
		"postprocess":        map[string]interface{}{"stall_run_tokens": 30.0},
	})
	if withBlock.Postprocess.StallRunTokens != 30 {
		t.Fatalf("StallRunTokens = %d, want 30", withBlock.Postprocess.StallRunTokens)
	}
	if len(withBlock.Postprocess.SilenceRenderIds) != 1 {
		t.Fatalf("SilenceRenderIds = %v, want [7]", withBlock.Postprocess.SilenceRenderIds)
	}
}

// mustFromManifest fails the test rather than returning a zero config, so a
// manifest the loader now refuses cannot look like a manifest with empty
// fields.
func mustFromManifest(t *testing.T, m map[string]interface{}) AlgorithmConfig {
	t.Helper()
	cfg, err := FromManifest(m)
	if err != nil {
		t.Fatalf("FromManifest(%v): %v", m, err)
	}
	return cfg
}

// A guidance mode this binding does not implement must be refused, not run as
// single_path under a fingerprint that says otherwise. The estimator is called
// once per step here and never forms (1+w)·v_cond − w·v_uncond.
func TestFromManifestRefusesDualPathGuidance(t *testing.T) {
	if _, err := FromManifest(map[string]interface{}{"guidance": "cfg_dual_path"}); err == nil {
		t.Fatal("cfg_dual_path was accepted; this binding renders single-path audio")
	}
	if _, err := FromManifest(map[string]interface{}{"guidance": "sorta_guided"}); err == nil {
		t.Fatal("an unknown guidance mode was accepted")
	}
}

// Zero is a value; only absence is absence. `min_p: 0` means no truncation;
// replacing it with the 0.05 default changes which tokens the
// sampler may pick: a different reading from the one the checkpoint declares.
func TestFromManifestKeepsExplicitZeroes(t *testing.T) {
	cfg := mustFromManifest(t, map[string]interface{}{
		"sampling_defaults": map[string]interface{}{"min_p": 0.0},
		"eos_floor": map[string]interface{}{
			"min_tokens_floor": 0.0, "min_tokens_text_ratio": 0.0,
		},
	})
	if cfg.Sampling.MinP != 0 {
		t.Fatalf("explicit min_p 0 became %v", cfg.Sampling.MinP)
	}
	if cfg.Sampling.MinTokensFloor != 0 || cfg.Sampling.MinTokensTextRatio != 0 {
		t.Fatalf("an explicitly disabled EOS floor came back as %d/%v",
			cfg.Sampling.MinTokensFloor, cfg.Sampling.MinTokensTextRatio)
	}
	// And an absent block gets what Defaults says an absent block gets, which
	// is what manifest.algorithm_from gives it: no floor at all. The floor is
	// a law a checkpoint declares, and 10 was this port's guess at one.
	def := mustFromManifest(t, map[string]interface{}{})
	if def.Sampling.MinP != Defaults().Sampling.MinP ||
		def.Sampling.MinTokensFloor != Defaults().Sampling.MinTokensFloor {
		t.Fatalf("absent defaults drifted: min_p=%v floor=%d",
			def.Sampling.MinP, def.Sampling.MinTokensFloor)
	}
	if def.Sampling.MinTokensFloor != 0 || def.Sampling.MinTokensTextRatio != 0 {
		t.Fatalf("an absent eos_floor came back as %d/%v, not the reference zero",
			def.Sampling.MinTokensFloor, def.Sampling.MinTokensTextRatio)
	}
}

// TestAbsentKeysMeanWhatTheReferenceSaysTheyMean pins every default the empty
// manifest gets, against manifest.algorithm_from's own.
//
// The fingerprint cannot stand in for this test: it is computed from the
// values a default supplied, so two ports whose defaults disagree hash their
// disagreement rather than report it. The loud key is the step count, because
// a zero there runs the flow loop no times and renders the prior noise field
// as audio with no error anywhere. The quiet one is the window: the shipped
// static recipe on one side and the ragged window on the other frame the same
// checkpoint's speech two ways.
func TestAbsentKeysMeanWhatTheReferenceSaysTheyMean(t *testing.T) {
	cfg := mustFromManifest(t, map[string]interface{}{})
	for _, c := range []struct {
		key       string
		got, want interface{}
	}{
		{"n_cfm_timesteps", cfg.EulerSteps, 2},
		{"speech_vocab_size", cfg.SpeechVocabSize, 8194},
		{"speech_tokens.start", cfg.StartSpeech, 6561},
		{"speech_tokens.stop", cfg.StopSpeech, 6562},
		{"sample_rate", cfg.SampleRate, 24000},
		{"token_rate_hz", cfg.TokenRateHz, 25.0},
		{"guidance", cfg.Guidance, "single_path"},
		{"guidance_rate", cfg.GuidanceRate, 0.0},
		{"edge_fade_seconds", cfg.EdgeFadeSeconds, 0.005},
		{"eos_floor.min_tokens_floor", cfg.Sampling.MinTokensFloor, 0},
		{"eos_floor.min_tokens_text_ratio", cfg.Sampling.MinTokensTextRatio, 0.0},
		{"sampling_defaults.temperature", cfg.Sampling.Temperature, 0.8},
		{"sampling_defaults.repetition_penalty", cfg.Sampling.RepetitionPenalty, 1.2},
		{"sampling_defaults.min_p", cfg.Sampling.MinP, 0.05},
		{"sampling_defaults.max_new_tokens", cfg.Sampling.MaxNewTokens, 255},
		{"window.max_speech_tokens", cfg.Window.MaxSpeechTokens, 255},
	} {
		if c.got != c.want {
			t.Errorf("an absent %s gave %v, the reference gives %v", c.key, c.got, c.want)
		}
	}
	// The ragged window: no static length, no pad token, no prompt tokens.
	if cfg.Window.StaticLength != nil || cfg.Window.PadTokenID != nil ||
		cfg.Window.StaticPromptTokens != nil {
		t.Errorf("an absent window block came back static, not ragged: %+v", cfg.Window)
	}
	// `window: null` is that same window said out loud.
	if got := mustFromManifest(t, map[string]interface{}{"window": nil}); got.Window != cfg.Window {
		t.Errorf("`window: null` gave %+v, not the ragged window %+v", got.Window, cfg.Window)
	}
	// And Defaults is where every one of them is written down.
	if cfg.Window != Defaults().Window || cfg.EulerSteps != Defaults().EulerSteps {
		t.Error("FromManifest's absent-key answers are not Defaults'")
	}
}

// TestFromManifestRefusesWhatTheReferenceRefuses pins the typed reads.
//
// A value with no numeric reading has no safe default: `n_cfm_timesteps: "2"`
// taken as zero is no flow steps at all, and `window: 3` ignored is the
// shipped static window under a manifest that named something else. Neither
// says anything on its way through. manifest._number, _block and _window_from
// refuse all of these by key, and a manifest one port reads while another
// mis-reads is the divergence this library exists to prevent.
func TestFromManifestRefusesWhatTheReferenceRefuses(t *testing.T) {
	for _, c := range []struct {
		name string
		m    map[string]interface{}
	}{
		{"a string step count", map[string]interface{}{"n_cfm_timesteps": "2"}},
		{"a boolean step count", map[string]interface{}{"n_cfm_timesteps": true}},
		{"a string vocabulary size", map[string]interface{}{"speech_vocab_size": "8194"}},
		{"a null sample rate", map[string]interface{}{"sample_rate": nil}},
		{"a string guidance rate", map[string]interface{}{"guidance_rate": "0.5"}},
		{"a string token rate", map[string]interface{}{"token_rate_hz": "25"}},
		{"a string edge fade", map[string]interface{}{"edge_fade_seconds": "0.02"}},
		{"a string eos_floor block", map[string]interface{}{"eos_floor": "x"}},
		{"a string sampling block", map[string]interface{}{"sampling_defaults": "x"}},
		{"a string speech_tokens block", map[string]interface{}{"speech_tokens": "x"}},
		{"a string chunking block", map[string]interface{}{"chunking": "x"}},
		{"a string postprocess block", map[string]interface{}{"postprocess": "x"}},
		{"a string decode block", map[string]interface{}{"decode": "x"}},
		{"a numeric window", map[string]interface{}{"window": float64(3)}},
		{"a string silence list", map[string]interface{}{"silence_token_ids": "123"}},
		{"a string silence render list", map[string]interface{}{"silence_render_ids": "1"}},
		{"a string quiet render list", map[string]interface{}{"quiet_render_ids": "1"}},
	} {
		if _, err := FromManifest(c.m); err == nil {
			t.Errorf("%s was accepted", c.name)
		}
	}
	// A null edge fade is the one deliberate exception: a manifest older than
	// the field and one that writes the null both mean the 5 ms those releases
	// shipped, as edge_fade_from reads it.
	if cfg := mustFromManifest(t, map[string]interface{}{"edge_fade_seconds": nil}); cfg.EdgeFadeSeconds != 0.005 {
		t.Errorf("`edge_fade_seconds: null` gave %v, want the legacy 0.005", cfg.EdgeFadeSeconds)
	}
}

// TestFromManifestRefusesAnUnhonouredChunkingKey pins a refusal the fingerprint
// cannot stand in for.
//
// ChunkConfig here carries no first-chunk cap, so the canonical form has no
// slot for one: a manifest that sets first_chunk_max_tokens and a manifest that
// omits it hash identically, while Python cuts the first chunk short and this
// splitter does not. Ignoring it is a different reading under one recipe
// version with nothing to report it. Rust and JS refuse it by this name too.
func TestFromManifestRefusesAnUnhonouredChunkingKey(t *testing.T) {
	if _, err := FromManifest(map[string]interface{}{
		"chunking": map[string]interface{}{"first_chunk_max_tokens": float64(8)},
	}); err == nil {
		t.Fatal("chunking.first_chunk_max_tokens was accepted; this splitter ignores it")
	}
	// Null is still the key being set, and Python reads it as "no cap": the
	// refusal is about the key this port cannot honour, not about its value.
	if _, err := FromManifest(map[string]interface{}{
		"chunking": map[string]interface{}{"first_chunk_max_tokens": nil},
	}); err == nil {
		t.Error("a null first_chunk_max_tokens was accepted")
	}
	// And the keys it does honour still load.
	cfg := mustFromManifest(t, map[string]interface{}{
		"chunking": map[string]interface{}{"max_tokens": float64(128)},
	})
	if cfg.Chunking.MaxTokens != 128 {
		t.Errorf("chunking.max_tokens = %d, want 128", cfg.Chunking.MaxTokens)
	}
}

// TestFromManifestRefusesANonNumberSubKey pins the reads inside the blocks.
//
// A sub-key is a manifest value like any other, and the blocks it lives in are
// where the audible ones are: `temperature: "0.8"` answered as zero is greedy
// decoding under a manifest asking for 0.8, and the fingerprint over it hashes
// the zero rather than reporting it. algorithm_from converts each of these
// with int() or float(), which refuse a null, a list and a non-numeric string.
func TestFromManifestRefusesANonNumberSubKey(t *testing.T) {
	for _, c := range []struct {
		name string
		m    map[string]interface{}
	}{
		{"a string temperature", map[string]interface{}{
			"sampling_defaults": map[string]interface{}{"temperature": "0.8"}}},
		{"a null temperature", map[string]interface{}{
			"sampling_defaults": map[string]interface{}{"temperature": nil}}},
		{"a list min_p", map[string]interface{}{
			"sampling_defaults": map[string]interface{}{"min_p": []interface{}{0.05}}}},
		{"a string cap", map[string]interface{}{
			"sampling_defaults": map[string]interface{}{"max_new_tokens": "255"}}},
		{"a string start token", map[string]interface{}{
			"speech_tokens": map[string]interface{}{"start": "6561"}}},
		{"a null stop token", map[string]interface{}{
			"speech_tokens": map[string]interface{}{"stop": nil}}},
		{"a string eos floor", map[string]interface{}{
			"eos_floor": map[string]interface{}{"min_tokens_floor": "10"}}},
		{"a string eos ratio", map[string]interface{}{
			"eos_floor": map[string]interface{}{"min_tokens_text_ratio": "1.2"}}},
		{"a string chunk budget", map[string]interface{}{
			"chunking": map[string]interface{}{"max_tokens": "128"}}},
		{"a string prefix carry", map[string]interface{}{
			"chunking": map[string]interface{}{"prefix_tokens": "6"}}},
		{"a string detector threshold", map[string]interface{}{
			"postprocess": map[string]interface{}{"pacing_tolerance": "1.6"}}},
		{"a null detector count", map[string]interface{}{
			"postprocess": map[string]interface{}{"stall_run_tokens": nil}}},
		{"a string id in the silence census", map[string]interface{}{
			"silence_token_ids": []interface{}{float64(4137), "4218"}}},
		{"a null id in the render census", map[string]interface{}{
			"silence_render_ids": []interface{}{nil}}},
		{"a string id in the quiet census", map[string]interface{}{
			"quiet_render_ids": []interface{}{"1458"}}},
		{"a string euler grid", map[string]interface{}{"euler_grid": "0.5"}},
		{"a string point in the euler grid", map[string]interface{}{
			"euler_grid": []interface{}{float64(0), "0.5"}}},
		// The window block reads through the same accumulator, so it gives the
		// same answer. A string static_length ignored is the ragged window
		// under a manifest that asked for a padded one, and the reference
		// coerces that string with int() rather than ignoring it.
		{"a string static length", map[string]interface{}{
			"window": map[string]interface{}{"static_length": "1024"}}},
		{"a list static length", map[string]interface{}{
			"window": map[string]interface{}{"static_length": []interface{}{float64(1024)}}}},
		{"a string window cap", map[string]interface{}{
			"window": map[string]interface{}{"max_speech_tokens": "255"}}},
		// The cap has no null reading: _window_from reads it with a bare
		// int(), which refuses a null the way it refuses a list.
		{"a null window cap", map[string]interface{}{
			"window": map[string]interface{}{"max_speech_tokens": nil}}},
		{"a string prompt window", map[string]interface{}{
			"window": map[string]interface{}{"static_prompt_tokens": "238"}}},
		{"a string pad token", map[string]interface{}{
			"window": map[string]interface{}{"pad_token_id": "4254"}}},
		{"an object pad token", map[string]interface{}{
			"window": map[string]interface{}{"pad_token_id": map[string]interface{}{}}}},
	} {
		if _, err := FromManifest(c.m); err == nil {
			t.Errorf("%s was accepted", c.name)
		}
	}
}

// The refusal names the block and the key, so a reader is told which line of
// the manifest to fix rather than that the manifest is bad.
func TestANonNumberSubKeyRefusalNamesTheKey(t *testing.T) {
	_, err := FromManifest(map[string]interface{}{
		"sampling_defaults": map[string]interface{}{"temperature": "0.8"},
	})
	if err == nil {
		t.Fatal("a string temperature was accepted")
	}
	for _, want := range []string{"manifest['sampling_defaults']", "temperature", "number"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("refusal %q does not name %q", err, want)
		}
	}
}

// The doors that read a name or a flag give the same answer the numeric ones
// do: a present key this port cannot read is refused, not ignored.
//
// Ignoring one is worse than it looks, because every one of these keys names a
// law. A mode read as the default is a law the manifest declared and the
// engine did not run, and the canonical form records the default, so the
// fingerprint agrees with the misreading instead of reporting it. The
// reference refuses each of these too, reaching it from the other side: str()
// renders the value and the membership check fails on what it rendered.
func TestFromManifestRefusesASubKeyOfTheWrongType(t *testing.T) {
	for _, c := range []struct {
		name string
		m    map[string]interface{}
	}{
		{"a numeric decode mode", map[string]interface{}{
			"decode": map[string]interface{}{"mode": float64(2)}}},
		{"a null decode mode", map[string]interface{}{
			"decode": map[string]interface{}{"mode": nil}}},
		{"a numeric guidance", map[string]interface{}{"guidance": float64(1)}},
		{"a numeric chunking switch", map[string]interface{}{
			"chunking": map[string]interface{}{"enabled": float64(0)}}},
		{"a string chunking switch", map[string]interface{}{
			"chunking": map[string]interface{}{"enabled": "false"}}},
		{"a string separator set", map[string]interface{}{
			"chunking": map[string]interface{}{"split_on": ". "}}},
		{"a numeric separator", map[string]interface{}{
			"chunking": map[string]interface{}{"split_on": []interface{}{". ", float64(1)}}}},
		// Not a shape but a substitution: an empty separator set used to take
		// the shipping five, so the text broke in five places the manifest
		// named none of. ChunkConfig.__post_init__ refuses it with this
		// sentence, and so does chunking.Validate here.
		{"an empty separator set", map[string]interface{}{
			"chunking": map[string]interface{}{"split_on": []interface{}{}}}},
		{"a numeric abbreviation", map[string]interface{}{
			"chunking": map[string]interface{}{"abbreviations": []interface{}{float64(1)}}}},
		{"a numeric mid-sentence law", map[string]interface{}{
			"chunking": map[string]interface{}{"mid_sentence_period": float64(1)}}},
		{"a null cap resplit", map[string]interface{}{
			"chunking": map[string]interface{}{"cap_resplit": nil}}},
		{"a numeric postprocess mode", map[string]interface{}{
			"postprocess": map[string]interface{}{"mode": float64(1)}}},
		{"a boolean postprocess mode", map[string]interface{}{
			"postprocess": map[string]interface{}{"mode": true}}},
		{"a null repetition resume", map[string]interface{}{
			"postprocess": map[string]interface{}{"repetition_resume": nil}}},
		{"a numeric repetition silence", map[string]interface{}{
			"postprocess": map[string]interface{}{"repetition_silence": float64(1)}}},
	} {
		if _, err := FromManifest(c.m); err == nil {
			t.Errorf("%s was accepted", c.name)
		}
	}
}

// And the names a well-formed manifest declares still arrive.
func TestFromManifestReadsTheNamesItAccepts(t *testing.T) {
	cfg := mustFromManifest(t, map[string]interface{}{
		"decode": map[string]interface{}{"mode": "fusion_mtp2"},
		"chunking": map[string]interface{}{
			"enabled":             false,
			"split_on":            []interface{}{"! "},
			"abbreviations":       []interface{}{"Dr"},
			"mid_sentence_period": "break",
			"cap_resplit":         "off",
		},
		"postprocess": map[string]interface{}{
			"mode":               "report",
			"repetition_resume":  "cut",
			"repetition_silence": "sampling",
		},
	})
	for _, c := range []struct{ name, got, want string }{
		{"decode mode", cfg.DecodeMode, "fusion_mtp2"},
		{"mid_sentence_period", cfg.Chunking.MidSentencePeriod, "break"},
		{"cap_resplit", cfg.Chunking.CapResplit, "off"},
		{"postprocess mode", cfg.Postprocess.Mode, "report"},
		{"repetition_resume", cfg.Postprocess.RepetitionResume, "cut"},
		{"repetition_silence", cfg.Postprocess.RepetitionSilence, "sampling"},
	} {
		if c.got != c.want {
			t.Errorf("%s = %q, want %q", c.name, c.got, c.want)
		}
	}
	if cfg.Chunking.Enabled {
		t.Error("a manifest that turned the splitter off came back with it on")
	}
	if len(cfg.Chunking.SplitOn) != 1 || cfg.Chunking.SplitOn[0] != "! " {
		t.Errorf("split_on = %q, want the one declared separator", cfg.Chunking.SplitOn)
	}
	if len(cfg.Chunking.Abbreviations) != 1 || cfg.Chunking.Abbreviations[0] != "Dr" {
		t.Errorf("abbreviations = %q, want the one declared abbreviation", cfg.Chunking.Abbreviations)
	}
}

// Null is the ragged reading for the three optional lengths, and only for
// them. _window_from reads those through an `opt` that spells an absent key
// and an explicit null the same way, so a manifest saying `static_length:
// null` gets the same window as one that says nothing.
func TestFromManifestReadsANullWindowLengthAsRagged(t *testing.T) {
	cfg := mustFromManifest(t, map[string]interface{}{
		"window": map[string]interface{}{
			"max_speech_tokens":    float64(255),
			"static_length":        nil,
			"static_prompt_tokens": nil,
			"pad_token_id":         nil,
		},
	})
	if cfg.Window != (WindowConfig{MaxSpeechTokens: 255}) {
		t.Errorf("null lengths gave %+v, want the ragged window", cfg.Window)
	}
}

// And the window a well-formed manifest declares still arrives whole: the
// shipped static recipe, read from the four keys rather than assumed.
func TestFromManifestReadsTheStaticWindow(t *testing.T) {
	cfg := mustFromManifest(t, map[string]interface{}{
		"window": map[string]interface{}{
			"max_speech_tokens":    float64(255),
			"static_length":        float64(255),
			"static_prompt_tokens": float64(238),
			"pad_token_id":         float64(4254),
		},
	})
	if CanonicalForm(cfg) != CanonicalForm(withWindow(ProductionWindow())) {
		t.Errorf("declared static window gave %+v, want %+v", cfg.Window, ProductionWindow())
	}
}

func withWindow(w WindowConfig) AlgorithmConfig {
	cfg := Defaults()
	cfg.Window = w
	return cfg
}

// The window refusal names the block and the key, like every other one.
func TestAWindowSubKeyRefusalNamesTheKey(t *testing.T) {
	_, err := FromManifest(map[string]interface{}{
		"window": map[string]interface{}{"static_length": "1024"},
	})
	if err == nil {
		t.Fatal("a string static_length was accepted")
	}
	for _, want := range []string{"manifest['window']", "static_length", "number"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("refusal %q does not name %q", err, want)
		}
	}
}

// And the values a well-formed manifest declares still arrive, including the
// explicit grid this port reads rather than falls back from.
func TestFromManifestReadsTheNumbersItAccepts(t *testing.T) {
	cfg := mustFromManifest(t, map[string]interface{}{
		"sampling_defaults": map[string]interface{}{"temperature": 0.5},
		"speech_tokens":     map[string]interface{}{"start": float64(6561)},
		"euler_grid":        []interface{}{float64(0), 0.25, float64(1)},
		"silence_token_ids": []interface{}{float64(4137), float64(4218)},
	})
	if cfg.Sampling.Temperature != 0.5 {
		t.Errorf("temperature = %v, want 0.5", cfg.Sampling.Temperature)
	}
	if cfg.StartSpeech != 6561 {
		t.Errorf("start speech token = %d, want 6561", cfg.StartSpeech)
	}
	want := []float64{0, 0.25, 1}
	if len(cfg.EulerGrid) != len(want) {
		t.Fatalf("euler grid = %v, want %v", cfg.EulerGrid, want)
	}
	for i, at := range want {
		if cfg.EulerGrid[i] != at {
			t.Errorf("euler grid[%d] = %v, want %v", i, cfg.EulerGrid[i], at)
		}
	}
	if len(cfg.Sampling.SilenceTokenIds) != 2 {
		t.Errorf("silence census = %v, want two ids", cfg.Sampling.SilenceTokenIds)
	}
}

// A manifest key that counts things takes a whole number.
//
// Truncating 2.7 to 2 turns a packer's arithmetic mistake into a chunker that
// breathes in a different place, a window framed to a different length or a
// census naming a token nobody wrote, under a recipe_version saying the five
// implementations agree. manifest._int refuses each of these, and the
// reference's postprocess reader refuses the counts in that block.
func TestFromManifestRefusesAFractionalCount(t *testing.T) {
	for _, c := range []struct {
		name string
		m    map[string]interface{}
		want string
	}{
		{"a chunk budget", map[string]interface{}{
			"chunking": map[string]interface{}{"max_tokens": 200.5}},
			"manifest['chunking']['max_tokens'] must be a whole number, got 200.5"},
		{"a prefix carry", map[string]interface{}{
			"chunking": map[string]interface{}{"prefix_tokens": 2.7}},
			"manifest['chunking']['prefix_tokens'] must be a whole number, got 2.7"},
		{"a window cap", map[string]interface{}{
			"window": map[string]interface{}{"max_speech_tokens": 255.5}},
			"manifest['window']['max_speech_tokens'] must be a whole number, got 255.5"},
		{"a static length", map[string]interface{}{
			"window": map[string]interface{}{"static_length": 255.5}},
			"manifest['window']['static_length'] must be a whole number, got 255.5"},
		{"a pad token", map[string]interface{}{
			"window": map[string]interface{}{"pad_token_id": 2.7}},
			"manifest['window']['pad_token_id'] must be a whole number, got 2.7"},
		{"a prompt window", map[string]interface{}{
			"window": map[string]interface{}{"static_prompt_tokens": 238.5}},
			"manifest['window']['static_prompt_tokens'] must be a whole number, got 238.5"},
		{"a start token", map[string]interface{}{
			"speech_tokens": map[string]interface{}{"start": 6561.5}},
			"manifest['speech_tokens']['start'] must be a whole number, got 6561.5"},
		{"a stop token", map[string]interface{}{
			"speech_tokens": map[string]interface{}{"stop": 6562.5}},
			"manifest['speech_tokens']['stop'] must be a whole number, got 6562.5"},
		{"a generation budget", map[string]interface{}{
			"sampling_defaults": map[string]interface{}{"max_new_tokens": 255.5}},
			"manifest['sampling_defaults']['max_new_tokens'] must be a whole number, got 255.5"},
		{"an EOS floor", map[string]interface{}{
			"eos_floor": map[string]interface{}{"min_tokens_floor": 10.5}},
			"manifest['eos_floor']['min_tokens_floor'] must be a whole number, got 10.5"},
		{"a step count", map[string]interface{}{"n_cfm_timesteps": 2.7},
			"manifest['n_cfm_timesteps'] must be a whole number, got 2.7"},
		{"a sample rate", map[string]interface{}{"sample_rate": 24000.5},
			"manifest['sample_rate'] must be a whole number, got 24000.5"},
		{"a vocabulary size", map[string]interface{}{"speech_vocab_size": 8194.5},
			"manifest['speech_vocab_size'] must be a whole number, got 8194.5"},
		{"a detector count", map[string]interface{}{
			"postprocess": map[string]interface{}{"stall_run_tokens": 2.7}},
			"manifest['postprocess']['stall_run_tokens'] must be a whole number, got 2.7"},
		{"an id in the silence census", map[string]interface{}{
			"silence_token_ids": []interface{}{float64(4137), 4218.5}},
			"manifest['silence_token_ids'][1] must be a whole number, got 4218.5"},
		{"an id in the render census", map[string]interface{}{
			"silence_render_ids": []interface{}{2.7}},
			"manifest['silence_render_ids'][0] must be a whole number, got 2.7"},
		{"an id in the quiet census", map[string]interface{}{
			"quiet_render_ids": []interface{}{2.7}},
			"manifest['quiet_render_ids'][0] must be a whole number, got 2.7"},
	} {
		_, err := FromManifest(c.m)
		if err == nil {
			t.Errorf("%s: a fractional value was accepted", c.name)
			continue
		}
		if err.Error() != c.want {
			t.Errorf("%s: refusal is %q, want %q", c.name, err, c.want)
		}
	}
}

// And the fields that hold a rate, a threshold or a probability still take
// one. Getting the boundary wrong in this direction refuses a manifest that is
// correct, which is the same defect facing the other way.
func TestFromManifestKeepsTheFractionalFields(t *testing.T) {
	cfg := mustFromManifest(t, map[string]interface{}{
		"token_rate_hz":     25.5,
		"edge_fade_seconds": 0.03,
		"euler_grid":        []interface{}{float64(0), 0.27, float64(1)},
		"sampling_defaults": map[string]interface{}{
			"temperature": 0.75, "repetition_penalty": 1.25, "min_p": 0.055},
		"eos_floor":   map[string]interface{}{"min_tokens_text_ratio": 1.25},
		"postprocess": map[string]interface{}{"pacing_tolerance": 0.17},
	})
	if cfg.TokenRateHz != 25.5 {
		t.Errorf("token rate = %v, want 25.5", cfg.TokenRateHz)
	}
	if cfg.EdgeFadeSeconds != 0.03 {
		t.Errorf("edge fade = %v, want 0.03", cfg.EdgeFadeSeconds)
	}
	if cfg.Sampling.Temperature != 0.75 || cfg.Sampling.MinP != 0.055 {
		t.Errorf("sampling = %+v, want the values the manifest declared", cfg.Sampling)
	}
	if cfg.Sampling.MinTokensTextRatio != 1.25 {
		t.Errorf("eos ratio = %v, want 1.25", cfg.Sampling.MinTokensTextRatio)
	}
	if cfg.Postprocess.PacingTolerance != 0.17 {
		t.Errorf("pacing tolerance = %v, want 0.17", cfg.Postprocess.PacingTolerance)
	}
}

// A count written as a float is the count. 255.0 is 255 in JSON and in every
// reader here; only the fraction is refused.
func TestFromManifestAcceptsAWholeCountWrittenAsAFloat(t *testing.T) {
	cfg := mustFromManifest(t, map[string]interface{}{
		"sample_rate":       float64(24000),
		"n_cfm_timesteps":   float64(2),
		"window":            map[string]interface{}{"max_speech_tokens": float64(255)},
		"silence_token_ids": []interface{}{float64(4254)},
	})
	if cfg.SampleRate != 24000 || cfg.EulerSteps != 2 {
		t.Errorf("config = %+v, want the values the manifest declared", cfg)
	}
	if cfg.Window.MaxSpeechTokens != 255 {
		t.Errorf("window cap = %d, want 255", cfg.Window.MaxSpeechTokens)
	}
	if len(cfg.Sampling.SilenceTokenIds) != 1 || cfg.Sampling.SilenceTokenIds[0] != 4254 {
		t.Errorf("silence census = %v, want [4254]", cfg.Sampling.SilenceTokenIds)
	}
}

// Every branch of PostprocessConfig._validate_ranges, in its order, with its
// sentence.
//
// The reference validates in `__post_init__`, so a manifest naming an
// impossible constant never becomes a config there. This wall read every
// number and asked nothing of it: `ceiling_slack_tokens: -1` loaded and
// widened the ceiling, `stall_run_tokens: 0` loaded and called every row with
// one leading silence token a stall.
func TestFromManifestRefusesAPostprocessConstantOutOfRange(t *testing.T) {
	for _, c := range []struct {
		block map[string]interface{}
		want  string
	}{
		{map[string]interface{}{"retry_max_attempts": 8.0},
			"retry_max_attempts must be in [0, 8): 8. Above that the ladder's " +
				"derived seeds run into the streams the chunk seeds use."},
		{map[string]interface{}{"retry_max_attempts": -1.0},
			"retry_max_attempts must be in [0, 8): -1"},
		{map[string]interface{}{"repetition_min_cycles": 1.0},
			"repetition_min_cycles must be at least 2: 1"},
		{map[string]interface{}{"repetition_max_period": 0.0},
			"repetition_max_period must be positive: 0"},
		{map[string]interface{}{"stall_run_tokens": 0.0},
			"stall_run_tokens must be positive: 0"},
		{map[string]interface{}{"stall_run_tokens": -1.0},
			"stall_run_tokens must be positive: -1"},
		{map[string]interface{}{"repetition_min_span": 2.0},
			"repetition_min_span (2) must be at least repetition_min_cycles (3)"},
		{map[string]interface{}{"ceiling_speech_per_text_token": 0.0},
			"ceiling_speech_per_text_token must be positive: 0"},
		{map[string]interface{}{"desperation_speech_per_text_token": 4.0},
			"desperation_speech_per_text_token (4.0) must exceed " +
				"ceiling_speech_per_text_token (4.0): below it, the rule that means " +
				"'certainly broken' fires on rows the ceiling stopped correctly"},
		{map[string]interface{}{"desperation_min_keep_per_text_token": -0.1},
			"desperation_min_keep_per_text_token must be >= 0: -0.1"},
		{map[string]interface{}{"desperation_min_keep_per_text_token": 2.7},
			"desperation_min_keep_per_text_token (2.7) must not exceed " +
				"desperation_band_ratio (2.6)"},
		{map[string]interface{}{"trailing_filler_threshold": 0.0},
			"trailing_filler_threshold must be in (0, 1]: 0"},
		{map[string]interface{}{"trailing_filler_threshold": 1.1},
			"trailing_filler_threshold must be in (0, 1]: 1.1"},
		{map[string]interface{}{"filler_min_eos_probability": 1.0},
			"filler_min_eos_probability out of range: 1"},
		{map[string]interface{}{"filler_min_eos_probability": -0.1},
			"filler_min_eos_probability out of range: -0.1"},
		{map[string]interface{}{"ceiling_slack_tokens": -1.0},
			"ceiling_slack_tokens must be >= 0: -1"},
		{map[string]interface{}{"desperation_band_floor": -1.0},
			"desperation_band_floor must be >= 0: -1"},
		{map[string]interface{}{"dropout_min_tokens": -1.0},
			"dropout_min_tokens must be >= 0: -1"},
		{map[string]interface{}{"echo_weak_max_tail": -1.0},
			"echo_weak_max_tail must be >= 0: -1"},
		{map[string]interface{}{"echo_strong_min_position_pct": 101.0},
			"echo_strong_min_position_pct is a percentage: 101"},
		{map[string]interface{}{"echo_weak_min_position_pct": -1.0},
			"echo_weak_min_position_pct is a percentage: -1"},
	} {
		_, err := FromManifest(map[string]interface{}{"postprocess": c.block})
		if err == nil {
			t.Fatalf("%v was accepted", c.block)
		}
		if !strings.Contains(err.Error(), c.want) {
			t.Fatalf("%v: error %q does not carry %q", c.block, err, c.want)
		}
	}
}

// The rules are one-sided, and a validator that also refused the last legal
// rung would be a second defect wearing the fix's clothes.
func TestFromManifestAcceptsTheBoundaryValuesTheReferenceAdmits(t *testing.T) {
	for _, block := range []map[string]interface{}{
		{"retry_max_attempts": float64(postprocess.RetryLadderHeadroom - 1)},
		{"retry_max_attempts": 0.0},
		{"repetition_min_cycles": 2.0, "repetition_min_span": 2.0},
		{"repetition_max_period": 1.0},
		{"stall_run_tokens": 1.0},
		{"trailing_filler_threshold": 1.0},
		{"filler_min_eos_probability": 0.0},
		{"desperation_min_keep_per_text_token": 2.6},
		{"ceiling_slack_tokens": 0.0},
		{"echo_strong_min_position_pct": 0.0},
		{"echo_weak_min_position_pct": 100.0},
	} {
		if _, err := FromManifest(map[string]interface{}{"postprocess": block}); err != nil {
			t.Fatalf("%v was refused: %v", block, err)
		}
	}
}
