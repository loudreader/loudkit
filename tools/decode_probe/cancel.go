// The Go half of the edge-triggered cancellation measurement.
//
// Counterpart to tools/decode_probe/cancel_py.py: the same two callback
// shapes, so "Python returns a partial where Go returns an error" is measured
// rather than read.
//
//	go run tools/decode_probe/cancel.go BUNDLE VOICE TEXT
package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/loudreader/loudkit/go/config"
	"github.com/loudreader/loudkit/go/engine"
	"github.com/loudreader/loudkit/go/onnx"
	"github.com/loudreader/loudkit/go/sampler"
	"github.com/loudreader/loudkit/go/voice"
)

func main() {
	if len(os.Args) != 4 {
		fmt.Fprintln(os.Stderr, "usage: cancel BUNDLE VOICE TEXT")
		os.Exit(2)
	}
	bundle, voicePath, text := os.Args[1], os.Args[2], os.Args[3]
	self, _ := os.Executable()
	fmt.Fprintf(os.Stderr, "go cancel probe: %s\n", self)

	lib := os.Getenv("LOUDKIT_ONNXRUNTIME_LIB")
	onnx.SetSharedLibraryPath(lib)
	if err := onnx.InitializeEnvironment(); err != nil {
		fmt.Fprintln(os.Stderr, "init:", err)
		os.Exit(1)
	}
	defer onnx.DestroyEnvironment()

	eng, err := engine.LoadWith(filepath.Join(bundle, "loudr-1.safetensors"),
		filepath.Join(bundle, "onnx"), filepath.Join(bundle, "tokenizer.json"),
		config.ExecutionConfig{ONNXProvider: "cpu"})
	if err != nil {
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(1)
	}
	defer eng.Close()
	v, _ := voice.Load(voicePath)
	cfg := eng.Config()
	textTokens, _ := eng.Encode(text, "en")

	newSampler := func(seed uint64) *sampler.Sampler {
		return sampler.New(sampler.Config{
			Temperature:       cfg.Sampling.Temperature,
			RepetitionPenalty: cfg.Sampling.RepetitionPenalty,
			MinP:              cfg.Sampling.MinP,
			MaxNewTokens:      cfg.Sampling.MaxNewTokens,
			SilenceTokenIds:   cfg.Sampling.SilenceTokenIds,
		}, seed)
	}

	full, err := eng.Generate(textTokens, v, newSampler(7), nil, nil, nil)
	fmt.Printf("baseline generate: %d tokens, err=%v\n", len(full), err)

	// Level-triggered: true from poll 21 onward.
	n := 0
	level := func() bool { n++; return n > 20 }
	got, err := eng.Generate(textTokens, v, newSampler(7), nil, level, nil)
	fmt.Printf("Generate, level-triggered cancel: returned %d tokens, err=%v\n", len(got), err)

	// Edge-triggered: true exactly once, on poll 21.
	m, done := 0, false
	edge := func() bool {
		m++
		if m == 21 && !done {
			done = true
			return true
		}
		return false
	}
	got2, err := eng.Generate(textTokens, v, newSampler(7), nil, edge, nil)
	fmt.Printf("Generate, edge-triggered cancel: returned %d tokens, err=%v\n", len(got2), err)

	// The public API with the edge-triggered flag: the case Python absorbs.
	k, done2 := 0, false
	edge2 := func() bool {
		k++
		if k == 21 && !done2 {
			done2 = true
			return true
		}
		return false
	}
	out, err := eng.Synthesize(text, v, engine.Options{Seed: 7, Language: "en", ShouldCancel: edge2})
	if err != nil {
		fmt.Printf("Synthesize, edge-triggered cancel: err=%v\n", err)
	} else {
		fmt.Printf("Synthesize, edge-triggered cancel: RETURNED %d tokens, %d samples\n",
			len(out.Tokens), len(out.Audio))
	}
}
