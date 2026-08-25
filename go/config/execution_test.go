package config

import (
	"strings"
	"testing"
)

func TestExecutionDefaultIsAuto(t *testing.T) {
	if got := DefaultExecution().RequestedProvider(); got != ProviderAuto {
		t.Fatalf("default provider = %q, want %q", got, ProviderAuto)
	}
	// The zero struct is the same request as the named default. A caller who
	// writes ExecutionConfig{} has not asked for CPU.
	if got := (ExecutionConfig{}).RequestedProvider(); got != ProviderAuto {
		t.Fatalf("zero-value provider = %q, want %q", got, ProviderAuto)
	}
	if err := (ExecutionConfig{}).Validate(); err != nil {
		t.Fatalf("zero value rejected: %v", err)
	}
}

func TestExecutionAcceptsTheFiveNames(t *testing.T) {
	for _, name := range []string{"auto", "cpu", "cuda", "coreml", "directml"} {
		if err := (ExecutionConfig{ONNXProvider: name}).Validate(); err != nil {
			t.Errorf("%q rejected: %v", name, err)
		}
	}
	// The list the error message quotes is the list that is accepted.
	if len(ONNXProviders) != 5 {
		t.Fatalf("ONNXProviders = %v, want the five contract names", ONNXProviders)
	}
}

func TestExecutionRefusesUnknownProvider(t *testing.T) {
	// "CUDA" and "metal" are the two shapes of this mistake: right provider
	// spelled wrong, and a provider that does not exist. Both must be refused
	// as spelling, before anything is probed — a resolver answer would read as
	// a missing library and send the caller off to install something.
	for _, name := range []string{"CUDA", "Auto", "metal", "gpu", "mps", " cpu"} {
		err := (ExecutionConfig{ONNXProvider: name}).Validate()
		if err == nil {
			t.Fatalf("%q accepted", name)
		}
		if !strings.Contains(err.Error(), name) {
			t.Errorf("error for %q does not name it: %v", name, err)
		}
		for _, known := range ONNXProviders {
			if !strings.Contains(err.Error(), known) {
				t.Errorf("error for %q does not offer %q: %v", name, known, err)
			}
		}
	}
}

func TestProviderPreferenceOrder(t *testing.T) {
	// auto takes only a provider a measurement backs. CoreML measured slower
	// than CPU and moved the tokens; DirectML has never been run. Both stay
	// selectable by name, neither is a default. CPU is last, so auto always
	// has an answer.
	want := []string{"cuda", "cpu"}
	if len(ProviderPreference) != len(want) {
		t.Fatalf("ProviderPreference = %v, want %v", ProviderPreference, want)
	}
	for i, p := range want {
		if ProviderPreference[i] != p {
			t.Fatalf("ProviderPreference = %v, want %v", ProviderPreference, want)
		}
	}
	if ProviderPreference[len(ProviderPreference)-1] != ProviderCPU {
		t.Fatal("CPU must be last in the preference order")
	}
	// Auto is a request, never a destination: it must not be something the
	// resolver can hand back.
	for _, p := range ProviderPreference {
		if p == ProviderAuto {
			t.Fatal("auto is in the preference order")
		}
	}
}

func TestDescribeExecutionCarriesTheProvider(t *testing.T) {
	if got := DescribeExecution(ProviderCUDA); got != "exec[onnx provider=cuda]" {
		t.Fatalf("DescribeExecution(cuda) = %q", got)
	}
	if got := DescribeExecution(ProviderCPU); got != "exec[onnx provider=cpu]" {
		t.Fatalf("DescribeExecution(cpu) = %q", got)
	}
}

func TestDescribeNamesAlgorithmAndProvider(t *testing.T) {
	cfg, err := FromManifest(map[string]interface{}{
		"recipe_version":    "loudkit-1",
		"n_cfm_timesteps":   float64(10),
		"speech_vocab_size": float64(6561),
		"sample_rate":       float64(24000),
		"chunking":          map[string]interface{}{},
		"postprocess":       map[string]interface{}{},
		"silence_token_ids": []interface{}{float64(1), float64(2)},
		"speech_tokens": map[string]interface{}{
			"start": float64(6561), "stop": float64(6562),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	line := Describe(cfg) + " | " + DescribeExecution(ProviderCoreML)
	for _, want := range []string{
		"algo[" + Fingerprint(cfg) + "]",
		"loudkit-1",
		"single_path",
		"euler=10(cosine)",
		// Floats render as Python's repr does, so a Go log line and a Python
		// one can be diffed rather than read.
		"temp=0.8",
		"rep=1.2",
		"min_p=0.05",
		"sil=2",
		"win=255",
		"exec[onnx provider=coreml]",
	} {
		if !strings.Contains(line, want) {
			t.Errorf("describe line %q is missing %q", line, want)
		}
	}
}

func TestDescribeMarksAnExplicitGrid(t *testing.T) {
	cfg, err := FromManifest(map[string]interface{}{
		"n_cfm_timesteps": float64(4),
		"euler_grid":      []interface{}{float64(0), float64(0.5), float64(1)},
		"sample_rate":     float64(24000),
		"chunking":        map[string]interface{}{},
		"postprocess":     map[string]interface{}{},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := Describe(cfg); !strings.Contains(got, "euler=4(explicit)") {
		t.Fatalf("describe %q does not mark the explicit grid", got)
	}
}
