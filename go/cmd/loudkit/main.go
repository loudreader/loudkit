// Command loudkit speaks a line of text through the Go engine — the CLI of
// the Go binding, used by the conformance runner and by hand.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/loudreader/loudkit/go/config"
	"github.com/loudreader/loudkit/go/engine"
	"github.com/loudreader/loudkit/go/onnx"
	"github.com/loudreader/loudkit/go/voice"
)

// usage is what this binary takes, and — because someone comparing the two
// ports will reach for both — how it differs from the Rust CLI.
//
// The two argv surfaces are deliberately not the same: each grew around what
// that port needed in order to be driven by hand. Saying so here is the
// alternative to converging them, and it keeps a reader from reading the
// difference as a port that fell behind. Parity between the ports is the
// library API and the conformance fixture, never these two flag lists.
const usage = `usage: loudkit -checkpoint CKPT -onnx DIR -voice VOICE -text TEXT
               [-seed N] [-tokens] [-speed X] [-timestamps] [-provider NAME]

A dev tool for driving this port by hand. The Rust CLI (rust/src/main.rs) is
deliberately a different surface: it carries --language and --json, which this
one has no equivalent for, and not -timestamps, which this one has.`

func main() {
	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, usage)
		fmt.Fprintln(os.Stderr)
		flag.PrintDefaults()
	}
	ckpt := flag.String("checkpoint", "", "packed checkpoint")
	onnxDir := flag.String("onnx", "", "exported onnx graph dir")
	voicePath := flag.String("voice", "", "voice profile")
	text := flag.String("text", "", "text to speak")
	seed := flag.Uint64("seed", 0, "seed")
	tokensOnly := flag.Bool("tokens", false, "print tokens only")
	// 1.0 is the default here as it is everywhere else, and it is a bypass
	// rather than a stretch of factor one — a run without -speed produces the
	// vocoder's own bytes, so this flag existing changes no existing output.
	speed := flag.Float64("speed", 1.0, "playback speed in [0.5, 2.0]; pitch is preserved")
	timestamps := flag.Bool("timestamps", false, "print per-chunk spans and estimated word times")
	// auto, not cpu: the default asks for the best provider the shared library
	// offers, and the run prints which one answered. An explicit name that the
	// library does not carry is refused rather than downgraded, so a timing
	// taken with -provider cuda cannot be a CPU timing.
	provider := flag.String("provider", config.ProviderAuto,
		"onnx execution provider: "+strings.Join(config.ONNXProviders, ", "))
	flag.Parse()

	if *ckpt == "" || *onnxDir == "" || *voicePath == "" || *text == "" {
		flag.Usage()
		os.Exit(2)
	}
	lib := os.Getenv("LOUDKIT_ONNXRUNTIME_LIB")
	if lib == "" {
		fmt.Fprintln(os.Stderr, "set LOUDKIT_ONNXRUNTIME_LIB to the onnxruntime shared library")
		os.Exit(2)
	}
	// Through the onnx package, not the binding: it records which library was
	// loaded, so a "provider not available" message can name the file whose
	// build decided that.
	onnx.SetSharedLibraryPath(lib)
	if err := onnx.InitializeEnvironment(); err != nil {
		fmt.Fprintln(os.Stderr, "init:", err)
		os.Exit(1)
	}
	defer onnx.DestroyEnvironment()

	// The release ships `tokenizer.json` beside the checkpoint. Appending
	// ".tokenizer.json" to the whole filename asks for
	// "loudr-1.safetensors.tokenizer.json", which no release contains, so
	// this CLI could not run on an official layout without an override.
	tokPath := os.Getenv("LOUDKIT_TOKENIZER")
	if tokPath == "" {
		tokPath = filepath.Join(filepath.Dir(*ckpt), "tokenizer.json")
	}
	eng, err := engine.LoadWith(*ckpt, *onnxDir, tokPath,
		config.ExecutionConfig{ONNXProvider: *provider})
	if err != nil {
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(1)
	}
	defer eng.Close()
	// On stderr, so a caller parsing the result line keeps parsing it, and on
	// every run rather than behind a flag: a timing with no provider beside it
	// is a number that cannot be compared with another one.
	fmt.Fprintln(os.Stderr, eng.Describe())

	v, err := voice.Load(*voicePath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "voice:", err)
		os.Exit(1)
	}

	// SynthesizeLong, not Synthesize: the single-window path refuses anything
	// longer than one window, so the CLI could not speak a paragraph. Text
	// that fits one window takes the same route through a single chunk.
	// Empty language, not "en": the engine resolves the argument, then the
	// voice, then English, so a Polish voice reads Polish without a flag this
	// CLI does not have. No previous tokens: each invocation of a CLI is its own
	// utterance, and there is nothing before it to continue from.
	audio, tokens, _, chunks, sr, capped, err := eng.SynthesizeLong(*text, v, *seed, "", *speed, nil, nil)
	if err != nil {
		fmt.Fprintln(os.Stderr, "synthesize:", err)
		os.Exit(1)
	}
	if capped {
		// The flag exists so truncation cannot pass silently: the audio is
		// real but incomplete. Same warning the Python CLI prints.
		fmt.Fprintln(os.Stderr, "warning: generation stopped at the token cap rather than at a stop token, so the reading is probably truncated")
	}
	if *tokensOnly {
		// The stream from the long-form path, so `-tokens` in this CLI names
		// the same thing as `--tokens` in the Rust one. A single-window
		// generate here would stop at 255 speech tokens on text that the
		// synthesis path reads as several chunks, which reads as a port
		// disagreeing with the others when only the flag disagreed. The seed
		// travels into SynthesizeLong, so `-tokens -seed 7` is the stream for
		// seed 7 and not for seed 0, and token identity under a named seed is
		// the whole point of this mode. Language stays empty for the same
		// reason as below: the engine resolves the voice's own tag, so a
		// Polish voice does not print the ids of an English read.
		fmt.Println(tokens)
		return
	}
	fmt.Printf("tokens=%d audio=%d samples @ %d Hz = %.2fs peak=%.3f\n",
		len(tokens), len(audio), sr, float64(len(audio))/float64(sr), maxAbs(audio))
	if *timestamps {
		// Chunk spans are exact; the per-word times inside them are an estimate
		// by proportional allocation, which is why only the exact tier is printed
		// unprefixed and the words are marked with a tilde.
		for _, c := range chunks {
			fmt.Printf("  %6.3f–%6.3f  %s\n", c.Start, c.End, c.Text)
			for _, w := range c.Words {
				fmt.Printf("      ~%6.3f  %s\n", w.Start, w.Text)
			}
		}
	}
}

func maxAbs(a []float32) float32 {
	m := float32(0)
	for _, x := range a {
		if x < 0 {
			x = -x
		}
		if x > m {
			m = x
		}
	}
	return m
}
