package config

import (
	"fmt"
	"strconv"
	"strings"
)

// The execution providers, spelled as the shared contract spells them: the
// same five words in Python, Rust, Go and JS, so a benchmark row and a bug
// report name the same thing in every port.
const (
	ProviderAuto     = "auto"
	ProviderCPU      = "cpu"
	ProviderCUDA     = "cuda"
	ProviderCoreML   = "coreml"
	ProviderDirectML = "directml"
)

// ONNXProviders are the accepted values of ExecutionConfig.ONNXProvider.
var ONNXProviders = []string{
	ProviderAuto, ProviderCPU, ProviderCUDA, ProviderCoreML, ProviderDirectML,
}

// ProviderPreference is the order ProviderAuto tries the concrete providers,
// best first. CPU is last and always reachable, so auto always has an answer.
//
// auto prefers a provider only where a measurement says it is faster. CoreML
// is faster -- the split placement in the onnx package measures RTF 1.35-1.70
// on an M3 Pro against 0.85-1.02 for all-CPU -- and is still not here, for a
// reason that is not speed: compiling the renderer graphs costs about 146 s
// the first time on a machine and leaves 1.6 GB of cache behind, and a default
// may not spend either without being asked. DirectML has never been run by
// this project. Both stay selectable by name; neither is a default. CUDA leads
// until it is measured, and drops out the same way if it loses.
//
// This is what auto *prefers*, which is not the same question as what the
// library *offers*: see ConcreteProviders.
var ProviderPreference = []string{
	ProviderCUDA, ProviderCPU,
}

// ConcreteProviders are every provider that can actually run graphs, in the
// order an availability report lists them. ProviderAuto is absent: it is a
// request, not a provider.
//
// Kept apart from ProviderPreference because the two answer different
// questions. Probing only the preference list would make a provider that auto
// declines unreachable by name as well -- a caller asking for coreml would be
// told the library does not offer it, on a machine where it does.
var ConcreteProviders = []string{
	ProviderCUDA, ProviderCoreML, ProviderDirectML, ProviderCPU,
}

// ExecutionConfig is how the engine runs, never what it computes.
//
// Nothing here reaches the fingerprint. Two machines running one
// AlgorithmConfig on two providers agree on the recipe and may disagree in the
// last bits of the audio; that is the split loudkit.config draws between the
// two structs, and this port keeps it.
//
// This port carries the one field an ONNX binding can honour. Precision,
// attention kernel and the rest of Python's ExecutionConfig are properties of
// the torch path, and the graphs here are exported fp32.
type ExecutionConfig struct {
	// ONNXProvider is which execution provider runs the graphs, one of
	// ONNXProviders.
	//
	// The zero value "" reads as ProviderAuto, so ExecutionConfig{} and
	// ExecutionConfig{ONNXProvider: "auto"} are the same request. Go has no
	// unset field, and a struct literal must not mean something the named
	// default does not.
	ONNXProvider string
}

// DefaultExecution is what a caller who says nothing gets.
func DefaultExecution() ExecutionConfig {
	return ExecutionConfig{ONNXProvider: ProviderAuto}
}

// RequestedProvider is ONNXProvider with the zero value read as auto.
func (e ExecutionConfig) RequestedProvider() string {
	if e.ONNXProvider == "" {
		return ProviderAuto
	}
	return e.ONNXProvider
}

// Validate refuses a provider name outside ONNXProviders.
//
// Spelling is checked before anything is probed, because a typo that reached
// the resolver would come back as "this build does not offer metal" — which
// reads as a missing library and sends the caller off to install something
// that does not exist.
func (e ExecutionConfig) Validate() error {
	p := e.RequestedProvider()
	for _, known := range ONNXProviders {
		if p == known {
			return nil
		}
	}
	return fmt.Errorf("unknown onnx provider %q; expected one of %s",
		p, strings.Join(ONNXProviders, ", "))
}

// DescribeExecution is the execution line, mirroring Python's
// ExecutionConfig.describe.
//
// It takes the provider that was chosen, not the one that was asked for:
// "auto" is a question, and a benchmark row needs the answer.
//
// The two tokens are Python's, not this port's: "onnx" is the placement, where
// Python prints self.device, and "provider=" is the flag it appends beside it
// (config.py ExecutionConfig.describe). Rust prints the same pair. A grep for
// `provider=` across four ports' logs has to find four hits, so this spells it
// the way the others do rather than the shorter way.
func DescribeExecution(chosen string) string {
	return "exec[onnx provider=" + chosen + "]"
}

// Describe is the one-line algorithm summary, mirroring Python's
// AlgorithmConfig.describe field for field so the two ports' log lines can be
// diffed. It is a log line and not an identity: the fingerprint it opens with
// is the identity, and nothing in the rest of it is hashed.
func Describe(cfg AlgorithmConfig) string {
	g := cfg.Guidance
	if g != "single_path" {
		g = "cfg@" + pyFloat(cfg.GuidanceRate)
	}
	grid := "cosine"
	if len(cfg.EulerGrid) > 0 {
		grid = "explicit"
	}
	win := "ragged"
	if cfg.Window.StaticLength != nil {
		win = strconv.Itoa(*cfg.Window.StaticLength)
	}
	return fmt.Sprintf(
		"algo[%s] %s %s euler=%d(%s) temp=%s rep=%s min_p=%s sil=%d win=%s",
		Fingerprint(cfg), cfg.RecipeVersion, g, cfg.EulerSteps, grid,
		pyFloat(cfg.Sampling.Temperature),
		pyFloat(cfg.Sampling.RepetitionPenalty),
		pyFloat(cfg.Sampling.MinP),
		len(cfg.Sampling.SilenceTokenIds), win)
}
