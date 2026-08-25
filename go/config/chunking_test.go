package config

import "testing"

// Where the reader breathes is declared by the checkpoint, not assumed by the
// runtime. A manifest that carries its own boundaries and a port that ignores
// them agree on recipe_version and disagree on the reading.
func TestChunkingComesFromTheManifest(t *testing.T) {
	cfg := mustFromManifest(t, map[string]interface{}{
		"chunking": map[string]interface{}{
			"enabled":       false,
			"max_tokens":    99.0,
			"prefix_tokens": 3.0,
			"split_on":      []interface{}{"|"},
		},
	})
	if cfg.Chunking.Enabled {
		t.Fatal("enabled=false was ignored")
	}
	if cfg.Chunking.MaxTokens != 99 || cfg.Chunking.PrefixTokens != 3 {
		t.Fatalf("got %d/%d, want 99/3", cfg.Chunking.MaxTokens, cfg.Chunking.PrefixTokens)
	}
	if len(cfg.Chunking.SplitOn) != 1 || cfg.Chunking.SplitOn[0] != "|" {
		t.Fatalf("split_on is %q", cfg.Chunking.SplitOn)
	}

	// A manifest that says nothing keeps the shipping recipe.
	def := mustFromManifest(t, map[string]interface{}{})
	if def.Chunking.MaxTokens != 255 || def.Chunking.PrefixTokens != 6 {
		t.Fatalf("defaults drifted: %d/%d", def.Chunking.MaxTokens, def.Chunking.PrefixTokens)
	}
}
