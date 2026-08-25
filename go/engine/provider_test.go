package engine

import (
	"strings"
	"testing"

	"github.com/loudreader/loudkit/go/config"
)

// The provider an engine resolved has to reach the line a benchmark row and a
// bug report are cut from, or nothing downstream can say which device produced
// a number.
func TestDescribeCarriesTheProvider(t *testing.T) {
	cfg, err := config.FromManifest(map[string]interface{}{
		"n_cfm_timesteps": float64(10),
		"sample_rate":     float64(24000),
		"chunking":        map[string]interface{}{},
		"postprocess":     map[string]interface{}{},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{"cpu", "cuda", "coreml", "directml"} {
		eng := &Engine{config: cfg, provider: p}
		if eng.Provider() != p {
			t.Errorf("Provider() = %q, want %q", eng.Provider(), p)
		}
		got := eng.Describe()
		if !strings.Contains(got, "exec[onnx provider="+p+"]") {
			t.Errorf("describe %q does not name provider %q", got, p)
		}
		// Beside the algorithm, not instead of it: the two halves answer
		// different questions and a run needs both.
		if !strings.Contains(got, "algo["+config.Fingerprint(cfg)+"]") {
			t.Errorf("describe %q does not name the algorithm", got)
		}
		// The separator is Python's and Rust's. Pinned because a log scraper
		// that splits four ports' lines has to split them the same way, and a
		// space here would make this the one port it cannot.
		if !strings.Contains(got, " | exec[") {
			t.Errorf("describe %q does not join the halves with %q", got, " | ")
		}
	}
}

// A provider name is checked before the checkpoint is read, so a typo costs a
// message rather than the seconds it takes to load a couple of gigabytes — and
// so a run that cannot honour the request never starts.
func TestLoadWithRefusesAnUnknownProviderBeforeReadingAnything(t *testing.T) {
	_, err := LoadWith("/nonexistent/checkpoint.safetensors", "/nonexistent/onnx",
		"/nonexistent/tokenizer.json", config.ExecutionConfig{ONNXProvider: "metal"})
	if err == nil {
		t.Fatal("metal accepted")
	}
	if !strings.Contains(err.Error(), "metal") {
		t.Fatalf("error does not name the provider: %v", err)
	}
	if strings.Contains(err.Error(), "checkpoint.safetensors") {
		t.Fatalf("the checkpoint was opened before the provider was checked: %v", err)
	}
}
