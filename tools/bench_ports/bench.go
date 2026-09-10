// Run from go/: go run ../tools/bench_ports/bench.go BUNDLE VOICE TEXT
package main

import (
	"encoding/json"
	"fmt"
	"github.com/loudreader/loudkit/go/config"
	"github.com/loudreader/loudkit/go/engine"
	"github.com/loudreader/loudkit/go/onnx"
	"github.com/loudreader/loudkit/go/voice"
	"os"
	"path/filepath"
	"time"
)

func check(err error) {
	if err != nil {
		panic(err)
	}
}
func main() {
	if len(os.Args) != 4 {
		fmt.Fprintln(os.Stderr, "usage: bench BUNDLE VOICE TEXT")
		os.Exit(2)
	}
	bundle := os.Args[1]
	onnx.SetSharedLibraryPath(os.Getenv("LOUDKIT_ONNXRUNTIME_LIB"))
	check(onnx.InitializeEnvironment())
	defer onnx.DestroyEnvironment()
	checkpoint := filepath.Join(bundle, filepath.Base(bundle)+".safetensors")
	start := time.Now()
	eng, err := engine.LoadWith(checkpoint, filepath.Join(bundle, "onnx"), filepath.Join(bundle, "tokenizer.json"), config.ExecutionConfig{ONNXProvider: config.ProviderCPU})
	check(err)
	defer eng.Close()
	load := time.Since(start).Seconds()
	// The checkpoint's rate, not a literal 24000: at any other rate a hardcoded
	// divisor reports the wrong audio duration, and so the wrong RTF, for every run.
	sampleRate := float64(eng.Config().SampleRate)
	v, err := voice.Load(os.Args[2])
	check(err)
	rows := []map[string]any{}
	for run := 0; run < 4; run++ {
		start = time.Now()
		first := 0.
		samples, tokens, chunks := 0, 0, 0
		check(eng.Stream(os.Args[3], v, engine.Options{Seed: 7}, func(c engine.Chunk) bool {
			if chunks == 0 {
				first = time.Since(start).Seconds()
			}
			chunks++
			samples += len(c.Audio)
			tokens += len(c.Tokens)
			return true
		}))
		rows = append(rows, map[string]any{"run": run, "seconds": time.Since(start).Seconds(), "ttfa_s": first, "audio_s": float64(samples) / sampleRate, "tokens": tokens, "chunks": chunks})
	}
	check(json.NewEncoder(os.Stdout).Encode(map[string]any{"runtime": "go", "bundle": bundle, "execution": eng.Describe(), "load_s": load, "text": os.Args[3], "seed": 7, "runs": rows}))
}
