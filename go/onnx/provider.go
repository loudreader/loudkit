package onnx

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/yalue/onnxruntime_go"

	"github.com/loudreader/loudkit/go/config"
)

var (
	libMu sync.Mutex
	// sharedLibrary is the path this process was pointed at. Kept because
	// onnxruntime_go has no getter for it and the provider error is only
	// actionable if it names the library that was measured: "cuda is not
	// available" is a bug report nobody can act on, "this libonnxruntime.so
	// offers cpu" is one they can.
	sharedLibrary string
	// probed is the availability answer for the library above, nil until
	// measured. Cached because it is a property of a loaded library, which
	// cannot change while the environment is up, and because probing CUDA
	// costs a CUDA context.
	probed map[string]bool
)

// SetSharedLibraryPath points the binding at an onnxruntime shared library and
// records which one, for the provider error.
//
// Callers who reach past this to onnxruntime_go.SetSharedLibraryPath still
// work; their errors name the library by version rather than by path.
func SetSharedLibraryPath(path string) {
	libMu.Lock()
	sharedLibrary = path
	probed = nil // a different library offers different providers
	libMu.Unlock()
	onnxruntime_go.SetSharedLibraryPath(path)
}

const disableTelemetryEnv = "ORT_DISABLE_TELEMETRY"

// InitializeEnvironment starts ONNX Runtime with its built-in telemetry off.
//
// Official native builds enable telemetry by default. The environment switch
// suppresses the initialization event and persistent device identifier; the
// API call keeps the environment disabled after it exists. Use this wrapper
// rather than initializing onnxruntime_go directly.
func InitializeEnvironment(opts ...onnxruntime_go.EnvironmentOption) error {
	if err := disableTelemetry(); err != nil {
		return err
	}
	if err := onnxruntime_go.InitializeEnvironment(opts...); err != nil {
		return err
	}
	if err := onnxruntime_go.DisableTelemetry(); err != nil {
		_ = onnxruntime_go.DestroyEnvironment()
		return fmt.Errorf("disable onnxruntime telemetry: %w", err)
	}
	return nil
}

// DestroyEnvironment releases the process-wide ONNX Runtime environment.
func DestroyEnvironment() error {
	return onnxruntime_go.DestroyEnvironment()
}

func disableTelemetry() error {
	if err := os.Setenv(disableTelemetryEnv, "1"); err != nil {
		return fmt.Errorf("disable onnxruntime telemetry: %w", err)
	}
	return nil
}

// Library names what was loaded, for an error a reader has to act on.
func Library() string {
	libMu.Lock()
	path := sharedLibrary
	libMu.Unlock()
	version := ""
	if onnxruntime_go.IsInitialized() {
		version = onnxruntime_go.GetVersion()
	}
	switch {
	case path != "" && version != "":
		return fmt.Sprintf("%s (onnxruntime %s)", path, version)
	case path != "":
		return path
	case version != "":
		return "the loaded onnxruntime " + version
	}
	return "the loaded onnxruntime"
}

// Available reports which providers the loaded shared library offers, in
// config.ConcreteProviders order.
//
// Measured, not assumed. The binding exposes AppendExecutionProvider* but not
// OrtApi::GetAvailableProviders, so the only honest test is to append the
// provider to a throwaway SessionOptions and read the status: a build without
// it answers ORT_NOT_IMPLEMENTED or "not enabled in this build", a build with
// it answers nil. Guessing from runtime.GOOS instead would call CoreML
// available on every Mac, including the ones running a linux-built library
// through Rosetta or a stripped minimal build.
func Available() ([]string, error) {
	if !onnxruntime_go.IsInitialized() {
		return nil, fmt.Errorf(
			"cannot tell which execution providers %s offers: the onnxruntime "+
				"environment is not initialized (call "+
				"onnx.InitializeEnvironment first)", Library())
	}
	libMu.Lock()
	defer libMu.Unlock()
	if probed == nil {
		probed = make(map[string]bool, len(config.ConcreteProviders))
		for _, p := range config.ConcreteProviders {
			probed[p] = probe(p) == nil
		}
	}
	out := make([]string, 0, len(probed))
	for _, p := range config.ConcreteProviders {
		if probed[p] {
			out = append(out, p)
		}
	}
	return out, nil
}

// Resolve answers the provider question against the library that is loaded,
// and returns the concrete provider the graphs will run on.
//
// An explicit provider that is missing is an error. It is never quietly
// downgraded to CPU: a run that was asked for CUDA and silently delivered CPU
// publishes the wrong number under the right name, which is the whole defect
// this change exists to close.
func Resolve(requested string) (string, error) {
	cfg := config.ExecutionConfig{ONNXProvider: requested}
	if err := cfg.Validate(); err != nil {
		return "", err
	}
	req := cfg.RequestedProvider()
	if req == config.ProviderCPU {
		// Answered without probing. The CPU provider is in every build, and
		// probing the others carries costs a caller who asked for CPU did not
		// ask for: appending CUDA initialises a CUDA context and replaces Go's
		// signal handlers (yalue/onnxruntime_go#140).
		return config.ProviderCPU, nil
	}
	available, err := Available()
	if err != nil {
		return "", err
	}
	if req == config.ProviderAuto {
		// Available is in report order, which is not preference order, so the
		// choice walks the preference list rather than taking the first entry.
		// CPU is in both and always available, so this cannot fall through.
		for _, p := range config.ProviderPreference {
			for _, a := range available {
				if a == p {
					return p, nil
				}
			}
		}
		return "", fmt.Errorf(
			"onnx_provider %q found no usable execution provider: %s offers %s",
			config.ProviderAuto, Library(), strings.Join(available, ", "))
	}
	for _, p := range available {
		if p == req {
			return req, nil
		}
	}
	return "", fmt.Errorf(
		"onnx execution provider %q is not available: %s offers %s. %s",
		req, Library(), strings.Join(available, ", "), remedy(req))
}

// remedy names the way to get a provider this build does not have.
//
// In Go the answer is always the shared library. Nothing about the provider
// set is compiled into this module — no build tag, no cgo flag, no second
// package — so `go get` cannot change it and only the library the binding
// loads can.
func remedy(provider string) string {
	switch provider {
	case config.ProviderCUDA:
		return "Point LOUDKIT_ONNXRUNTIME_LIB at a CUDA-enabled onnxruntime: " +
			"`pip install onnxruntime-gpu` puts one at " +
			"onnxruntime/capi/libonnxruntime.so, and the official " +
			"onnxruntime-linux-x64-gpu release archive carries the same library."
	case config.ProviderCoreML:
		return "CoreML ships only in the macOS builds of onnxruntime " +
			"(the onnxruntime-silicon or onnxruntime wheel for macOS, or the " +
			"osx- release archive); point LOUDKIT_ONNXRUNTIME_LIB at one of those."
	case config.ProviderDirectML:
		return "DirectML ships only in the onnxruntime-directml build on " +
			"Windows; point LOUDKIT_ONNXRUNTIME_LIB at its onnxruntime.dll."
	}
	return "Point LOUDKIT_ONNXRUNTIME_LIB at a build that carries it."
}

// rendererGraphs are the three graphs CoreML is allowed to run.
//
// An allowlist, not a denylist: enroll.go opens three more graphs through the
// same Load, and the voice encoder decides what a cloned voice sounds like.
// Nobody has measured CoreML on it, so it stays on CPU, along with any graph
// added later.
var rendererGraphs = map[string]bool{
	"flow_encoder.onnx":   true,
	"flow_estimator.onnx": true,
	"vocoder.onnx":        true,
}

// coremlCacheEnv overrides where CoreML keeps its compiled models.
const coremlCacheEnv = "LOUDKIT_COREML_CACHE"

// coremlCacheDir is where CoreML writes compiled models, and it is not
// optional.
//
// Compiling the renderer graphs takes about 146 s. With a cache directory that
// is paid once per machine and later loads cost about 25 s; without one it is
// paid on every session, which no interactive use can absorb. The cache runs
// to roughly 1.6 GB.
func coremlCacheDir() string {
	if dir := os.Getenv(coremlCacheEnv); dir != "" {
		return dir
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return filepath.Join(".", "loudkit", "coreml")
	}
	return filepath.Join(home, "Library", "Caches", "loudkit", "coreml")
}

// placement answers which provider one graph actually runs on.
//
// Every provider but CoreML is applied to all graphs alike. CoreML is a
// placement: the renderer on CoreML, everything else on CPU. t3_step runs once
// per speech token and CPU does it in 9.8 ms against CoreML's best 17.6 ms;
// t3_prefill and t3_step also fail to compile under MLProgram outright, and
// MLProgram is the only setting worth having. Keeping the generator on CPU is
// what makes the token stream identical to a CPU run, index for index. The
// waveform is not bit-identical, which is what the identity contract already
// says about running the renderer elsewhere.
func placement(provider, graph string) string {
	if provider == config.ProviderCoreML && !rendererGraphs[graph] {
		return config.ProviderCPU
	}
	return provider
}

// probe measures one provider by appending it to session options that are then
// thrown away. Nothing is loaded and no graph is run.
func probe(provider string) error {
	if provider == config.ProviderCPU {
		return nil
	}
	opts, err := onnxruntime_go.NewSessionOptions()
	if err != nil {
		return err
	}
	defer opts.Destroy()
	// Probing asks whether the library carries the provider at all, so it uses
	// a graph the placement will not divert: a CoreML probe that landed on CPU
	// would report CoreML available on every machine.
	return applyProvider(opts, provider, "vocoder.onnx")
}

// applyProvider puts one concrete provider on session options.
//
// It refuses "auto" rather than resolving it, so that six graphs cannot be
// opened on two answers to the same question: the caller resolves once and
// passes the name down.
func applyProvider(opts *onnxruntime_go.SessionOptions, provider, graph string) error {
	switch placement(provider, graph) {
	case config.ProviderCPU:
		// No append. The CPU provider is onnxruntime's fallback for whatever
		// the appended providers do not claim, and appending it explicitly is
		// not part of the C API.
		return nil
	case config.ProviderCUDA:
		cuda, err := onnxruntime_go.NewCUDAProviderOptions()
		if err != nil {
			return err
		}
		defer cuda.Destroy()
		return opts.AppendExecutionProviderCUDA(cuda)
	case config.ProviderCoreML:
		// The string-option form, not the deprecated flags one: the flags API
		// is frozen at the pre-1.20 option set and cannot name a cache
		// directory at all.
		//
		// MLProgram is not a tuning knob. At the default (NeuralNetwork) the
		// renderer shatters into hundreds of partitions -- flow_estimator 342,
		// flow_encoder 47, vocoder 51 against 2, 1 and 25 -- and it changes the
		// numbers: a NeuralNetwork vocoder sums 217.70 where CPU sums 211.15,
		// while MLProgram sums 211.149. The fast setting is the faithful one.
		return opts.AppendExecutionProviderCoreMLV2(map[string]string{
			"ModelFormat":         "MLProgram",
			"ModelCacheDirectory": coremlCacheDir(),
		})
	case config.ProviderDirectML:
		// DirectML requires these two, per the onnxruntime docs: memory
		// pattern off and sequential execution. Set before the append so a
		// failure to set them is reported as itself.
		if err := opts.SetMemPattern(false); err != nil {
			return err
		}
		if err := opts.SetExecutionMode(onnxruntime_go.ExecutionModeSequential); err != nil {
			return err
		}
		// Device 0 is the primary display GPU.
		return opts.AppendExecutionProviderDirectML(0)
	case config.ProviderAuto:
		return fmt.Errorf(
			"onnx: %q is a request, not a provider; call Resolve first", provider)
	}
	return fmt.Errorf("unknown onnx execution provider %q; expected one of %s",
		provider, strings.Join(config.ONNXProviders, ", "))
}
