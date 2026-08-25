# loudkit Go binding

The loudkit engine over `github.com/yalue/onnxruntime_go`. The rng, sampler,
tokenizer, windowing, engine and the Polish respelling lexicon are ported to
Go, with **no torch** at runtime.

## Status: supported

Passes the shared conformance fixture: weight-free vectors exact, free-run
tokens and render band against the checkpoint. The full walkthrough is
`docs/guides/08-go.md` in the repo root.

## Requirements

- Go 1.25.13 or newer. The patch floor includes the current standard-library
  security fixes; `golang.org/x/text` v0.40 sets the Go 1.25 language floor.
- a `libonnxruntime` shared library (load-dynamic; point
  `LOUDKIT_ONNXRUNTIME_LIB` at it)
- the exported ONNX graphs, the packed checkpoint, a voice profile, and
  `tokenizer.json`

## Synthesise

```bash
go get github.com/loudreader/loudkit/go
```

```bash
pip install "loudkit[hub]"
loudkit download loudreader/loudr-1 --for onnx --local-dir loudr-1
```

Everything lands inside `loudr-1/`, which is what the paths below are relative
to. `--with-cloning` adds the three enrollment graphs.

```go
package main

import (
	"fmt"
	"log"

	"github.com/loudreader/loudkit/go/engine"
	"github.com/loudreader/loudkit/go/onnx"
	"github.com/loudreader/loudkit/go/voice"
)

func main() {
	onnx.SetSharedLibraryPath("/path/to/libonnxruntime.dylib")
	if err := onnx.InitializeEnvironment(); err != nil {
		log.Fatal(err)
	}
	defer onnx.DestroyEnvironment()

	eng, err := engine.Load(
		"loudr-1/loudr-1.safetensors", // packed checkpoint
		"loudr-1/onnx",                // exported graphs
		"loudr-1/tokenizer.json",      // text tokenizer
	)
	if err != nil {
		log.Fatal(err)
	}
	defer eng.Close()

	v, err := voice.Load("loudr-1/voices/joe.safetensors")
	if err != nil {
		log.Fatal(err)
	}

	// Use SynthesizeLong. Synthesize renders one window and errors on
	// anything longer instead of clipping it. The empty language means the voice's own; 1.0 is
	// normal speed; the nils are previousTokens and shouldCancel.
	audio, tokens, _, _, sr, _, err := eng.SynthesizeLong(
		"Hello from loudkit.", v, 7, "", 1.0, nil, nil)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%d tokens, %.2fs\n", len(tokens), float64(len(audio))/float64(sr))
	// audio is []float32 at sr. Hand it to your own WAV writer or audio
	// device. This module writes no files.
}
```

Streaming, timestamps, speed and barge-in: `docs/guides/08-go.md`.

## Execution provider

`engine.Load` and `enroll.LoadEnroller` pick the best provider the shared
library offers. `engine.LoadWith` and `enroll.LoadEnrollerWith` take an
`ExecutionConfig` and name one instead:

```go
eng, err := engine.LoadWith(ckpt, onnxDir, tokPath, config.ExecutionConfig{
	ONNXProvider: config.ProviderCUDA, // auto, cpu, cuda, coreml, directml
})
fmt.Println(eng.Describe()) // ... | exec[onnx provider=cuda]
```

`auto` is the default. It takes cuda where the shared library offers it and cpu
otherwise; it reaches neither coreml nor directml.
`Engine.Provider` and `Describe` report the one it took. A named provider the
library does not carry is an error, never a quiet fall back to cpu, so a timing
taken under one provider cannot be a timing from another.

Which providers exist is decided by the shared library alone. Nothing here is
compiled in, so `go get` cannot add one and only the library
`LOUDKIT_ONNXRUNTIME_LIB` points at can:

| provider | shared library |
| --- | --- |
| `cpu` | any build |
| `cuda` | `onnxruntime-gpu` wheel, or the `onnxruntime-linux-x64-gpu` archive |
| `coreml` | a macOS build |
| `directml` | the `onnxruntime-directml` build, on Windows |

Point the binding through `onnx.SetSharedLibraryPath` rather than the binding's
own setter. Both load the library, but only the first records which file it
was, and the refusal above is worth reading only if it names the build that
decided the answer.

A GPU provider can change the token stream and waveform. CoreML is available by
name on a compatible macOS build, but `auto` does not select it because its
first compile is expensive. The conformance fixture pins CPU. See
[`docs/benchmarks.md`](../docs/benchmarks.md#onnx-execution-providers).

CUDA has been measured on an RTX 3090. DirectML is unit-tested for resolution
and refusal text but has not been measured on Windows.

The CLI carries the same knob as `-provider`.

## Build and test

```bash
gofmt -l . && go vet ./...
go test ./conformance/        # weight-free conformance vectors
go test ./...                 # + engine conformance (needs LOUDKIT_* assets)
```

## Runtime libraries and assets

The ONNX Runtime shared library and the checkpoint/graphs are **not** bundled.
The package loads them from the paths you provide.
