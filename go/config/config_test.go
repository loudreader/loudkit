package config

import (
	"strings"
	"testing"
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
// sampler may pick — a different reading from the one the checkpoint declares.
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
	// And an absent block still gets the production defaults.
	def := mustFromManifest(t, map[string]interface{}{})
	if def.Sampling.MinP != 0.05 || def.Sampling.MinTokensFloor != 10 {
		t.Fatalf("absent defaults drifted: min_p=%v floor=%d",
			def.Sampling.MinP, def.Sampling.MinTokensFloor)
	}
}
