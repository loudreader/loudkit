package onnx

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/yalue/onnxruntime_go"

	"github.com/loudreader/loudkit/go/config"
)

func TestTelemetryIsDisabledBeforeTheEnvironmentStarts(t *testing.T) {
	t.Setenv(disableTelemetryEnv, "0")
	if err := disableTelemetry(); err != nil {
		t.Fatal(err)
	}
	if got := os.Getenv(disableTelemetryEnv); got != "1" {
		t.Fatalf("%s = %q, want 1", disableTelemetryEnv, got)
	}
}

// A misspelled provider is refused as spelling, with no library involved: the
// name is wrong wherever this runs, and a machine without onnxruntime must
// still get the message that says so.
func TestResolveRefusesAnUnknownName(t *testing.T) {
	if _, err := Resolve("metal"); err == nil {
		t.Fatal("metal accepted")
	} else if !strings.Contains(err.Error(), "metal") ||
		!strings.Contains(err.Error(), "coreml") {
		t.Fatalf("unhelpful error: %v", err)
	}
}

// CPU is answered without probing. This is the behaviour that keeps a
// -provider cpu run from paying for a CUDA context it did not ask for, and it
// is observable here as the one Resolve that works with no environment up.
func TestResolveCPUNeedsNoEnvironment(t *testing.T) {
	if onnxruntime_go.IsInitialized() {
		t.Skip("environment already initialized by another test")
	}
	got, err := Resolve(config.ProviderCPU)
	if err != nil {
		t.Fatalf("cpu: %v", err)
	}
	if got != config.ProviderCPU {
		t.Fatalf("cpu resolved to %q", got)
	}
}

// Everything else needs a library to measure, and says which one it could not
// measure rather than reporting the provider as missing.
func TestResolveWithoutEnvironmentSaysSo(t *testing.T) {
	if onnxruntime_go.IsInitialized() {
		t.Skip("environment already initialized by another test")
	}
	for _, req := range []string{config.ProviderAuto, config.ProviderCUDA} {
		_, err := Resolve(req)
		if err == nil {
			t.Fatalf("%q resolved with no environment", req)
		}
		if !strings.Contains(err.Error(), "not initialized") {
			t.Errorf("%q: error does not name the cause: %v", req, err)
		}
	}
}

// "auto" is a question. It must never reach session options as if it were an
// answer — six graphs opened on an unresolved request could land on two
// devices.
func TestApplyProviderRefusesAuto(t *testing.T) {
	err := applyProvider(nil, config.ProviderAuto, "vocoder.onnx")
	if err == nil {
		t.Fatal("auto accepted as a provider")
	}
	if !strings.Contains(err.Error(), "Resolve") {
		t.Fatalf("error does not point at the resolver: %v", err)
	}
}

func TestApplyProviderRefusesAnUnknownName(t *testing.T) {
	if err := applyProvider(nil, "metal", "vocoder.onnx"); err == nil {
		t.Fatal("metal accepted")
	}
}

// The CPU provider is onnxruntime's fallback and needs no append, so it must
// not touch the (here nil) options.
func TestApplyProviderCPUIsANoOp(t *testing.T) {
	if err := applyProvider(nil, config.ProviderCPU, "vocoder.onnx"); err != nil {
		t.Fatalf("cpu: %v", err)
	}
}

// Everything above is machine-independent. This one measures a real library
// and skips without one: what it asserts is that the answer comes from the
// library rather than from an assumption about the operating system.
func TestResolveAgainstTheLoadedLibrary(t *testing.T) {
	lib := os.Getenv("LOUDKIT_ONNXRUNTIME_LIB")
	if lib == "" {
		t.Skip("set LOUDKIT_ONNXRUNTIME_LIB to measure a real library")
	}
	if onnxruntime_go.IsInitialized() {
		t.Skip("environment already initialized by another test")
	}
	SetSharedLibraryPath(lib)
	if err := InitializeEnvironment(); err != nil {
		t.Fatal(err)
	}
	defer DestroyEnvironment()

	available, err := Available()
	if err != nil {
		t.Fatal(err)
	}
	if len(available) == 0 || available[len(available)-1] != config.ProviderCPU {
		t.Fatalf("available = %v, want CPU present and last", available)
	}
	// What auto promises is the first *preferred* provider that is available,
	// which is a different list from the report order Available returns. On
	// this Mac the report begins with coreml and auto answers cpu, because
	// coreml is offered by the library and deliberately absent from
	// config.ProviderPreference: reachable by name, never a default. Asserting
	// auto == available[0] would encode the report order as if it were the
	// preference order, and fail on exactly the machines the split exists for.
	want := ""
	for _, p := range config.ProviderPreference {
		if contains(available, p) {
			want = p
			break
		}
	}
	if want == "" {
		// CPU is in both lists and is in every build, so auto cannot run out
		// of candidates. Reaching here means one of those two invariants
		// broke, and the assertion below would be measuring nothing.
		t.Fatalf("no preferred provider is available: preference %v, available %v",
			config.ProviderPreference, available)
	}
	auto, err := Resolve(config.ProviderAuto)
	if err != nil {
		t.Fatal(err)
	}
	if auto != want {
		t.Fatalf("auto chose %q; the first available entry of preference %v is %q",
			auto, config.ProviderPreference, want)
	}
	// auto must hand back something the loaded library can actually run, and
	// never the request it was given.
	if auto == config.ProviderAuto {
		t.Fatal("auto resolved to itself")
	}
	if !contains(available, auto) {
		t.Fatalf("auto chose %q, which %s does not offer: %v",
			auto, Library(), available)
	}
	t.Logf("%s offers %v; auto chose %s", Library(), available, auto)

	// Every provider this build offers must resolve to itself, and every one
	// it does not must be refused with a message naming the library, what it
	// does offer, and how to get the missing one.
	//
	// Over ConcreteProviders, not ProviderPreference: the point of the split
	// is that a provider auto declines is still reachable by name, and only
	// the wider list asks coreml and directml that question.
	for _, p := range config.ConcreteProviders {
		got, err := Resolve(p)
		if contains(available, p) {
			if err != nil || got != p {
				t.Errorf("available provider %q resolved to (%q, %v)", p, got, err)
			}
			continue
		}
		if err == nil {
			t.Errorf("unavailable provider %q resolved to %q", p, got)
			continue
		}
		msg := err.Error()
		if !strings.Contains(msg, p) {
			t.Errorf("%q: error does not name the provider asked for: %v", p, err)
		}
		if !strings.Contains(msg, lib) {
			t.Errorf("%q: error does not name the library that was measured: %v", p, err)
		}
		for _, have := range available {
			if !strings.Contains(msg, have) {
				t.Errorf("%q: error does not list %q as offered: %v", p, have, err)
			}
		}
		if !strings.Contains(msg, "LOUDKIT_ONNXRUNTIME_LIB") {
			t.Errorf("%q: error does not say how to get the provider: %v", p, err)
		}
	}
}

func contains(list []string, want string) bool {
	for _, v := range list {
		if v == want {
			return true
		}
	}
	return false
}

// CoreML is asked for once and lands on three of the six graphs. t3_step runs
// once per speech token and CPU does it in 9.8 ms against CoreML's best
// 17.6 ms; t3_prefill and t3_step also fail to compile under MLProgram at all.
// Keeping the generator on CPU is what makes the token stream identical to a
// CPU run.
func TestCoreMLRunsTheRendererAndNothingElse(t *testing.T) {
	for _, graph := range []string{"flow_encoder.onnx", "flow_estimator.onnx", "vocoder.onnx"} {
		if got := placement(config.ProviderCoreML, graph); got != config.ProviderCoreML {
			t.Errorf("%s: placed on %q, want coreml", graph, got)
		}
	}
	// The enrollment graphs share this package's Load. The voice encoder
	// decides what a cloned voice sounds like and has never been measured on
	// CoreML, so an allowlist keeps it, and anything added later, on CPU.
	for _, graph := range []string{
		"t3_cond.onnx", "t3_prefill.onnx", "t3_step.onnx",
		"s3_tokenizer.onnx", "camp.onnx", "voice_encoder.onnx", "added_later.onnx",
	} {
		if got := placement(config.ProviderCoreML, graph); got != config.ProviderCPU {
			t.Errorf("%s: placed on %q, want cpu", graph, got)
		}
	}
}

// Every provider but CoreML is applied to all graphs alike.
func TestEveryOtherProviderTakesEveryGraph(t *testing.T) {
	for _, p := range []string{config.ProviderCPU, config.ProviderCUDA, config.ProviderDirectML} {
		for _, graph := range []string{"t3_step.onnx", "vocoder.onnx", "voice_encoder.onnx"} {
			if got := placement(p, graph); got != p {
				t.Errorf("%s/%s: placed on %q, want %q", p, graph, got, p)
			}
		}
	}
}

// Compiling the renderer graphs takes about 146 s, so the cache directory is
// not optional; without one that cost is paid on every session.
func TestCoreMLCacheDirectory(t *testing.T) {
	t.Setenv(coremlCacheEnv, "/tmp/loudkit-coreml-test")
	if got := coremlCacheDir(); got != "/tmp/loudkit-coreml-test" {
		t.Errorf("override ignored: %q", got)
	}
	t.Setenv(coremlCacheEnv, "")
	if got := coremlCacheDir(); !strings.HasSuffix(got, filepath.Join("loudkit", "coreml")) {
		t.Errorf("default is not under loudkit/coreml: %q", got)
	}
}
