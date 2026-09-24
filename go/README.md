# loudkit for Go

Text to speech in Go, on ONNX Runtime. It does not need Python or PyTorch.

## Hello

You need Go 1.25 or newer and the ONNX Runtime shared library, version 1.28
or newer. The library is not part of the module. Install it separately:

- macOS: `brew install onnxruntime`. Check that its version is 1.28 or newer.
- Linux: unpack the `onnxruntime-linux-x64` archive from the
  [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases).
- Windows: unpack the `onnxruntime-win-x64` archive from the same page.

`loudkit.Load` looks for the library in these directories:

- macOS: `/opt/homebrew/lib`, `/usr/local/lib`
- Linux: `/usr/local/lib`, `/usr/lib`, `/usr/lib/x86_64-linux-gnu`,
  `/usr/lib/aarch64-linux-gnu`
- Windows: `C:\Program Files\onnxruntime\lib`, the working directory

If the library is somewhere else, for example in the unpacked archive, set
`LOUDKIT_ONNXRUNTIME_LIB` to the library file. Distribution packages such as
`libonnxruntime-dev` can be older than 1.28.

```bash
mkdir hello && cd hello
go mod init hello
go get github.com/loudreader/loudkit/go@v0.1.1
```

`main.go`:

```go
// Command hello fetches loudr-1, speaks one line and writes hello.wav.
package main

import (
	"log"

	loudkit "github.com/loudreader/loudkit/go"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

// run holds the body so the deferred Close runs on the way out: log.Fatal
// ends the process through os.Exit, which runs no deferred function.
func run() error {
	eng, err := loudkit.Load("loudreader/loudr-1")
	if err != nil {
		return err
	}
	defer eng.Close()

	v, err := eng.Voice("joe")
	if err != nil {
		return err
	}

	out, err := eng.Synthesize("Hello from loudkit.", v, loudkit.Options{Seed: 7})
	if err != nil {
		return err
	}
	if err := out.SaveWav("hello.wav"); err != nil {
		return err
	}
	log.Printf("hello.wav: %.2fs", out.Duration().Seconds())
	return nil
}
```

```bash
go run .
```

The first run downloads the model files into the user cache:

- macOS: `~/Library/Caches/loudkit/loudreader--loudr-1`
- Linux: `$XDG_CACHE_HOME/loudkit/loudreader--loudr-1`, or
  `~/.cache/loudkit/loudreader--loudr-1` when `XDG_CACHE_HOME` is not set
- Windows: `%LOCALAPPDATA%\loudkit\loudreader--loudr-1`

Set `LOUDKIT_CACHE` to use `$LOUDKIT_CACHE/loudreader--loudr-1` instead. The
Rust, JS and Swift ports use the same directory. Each downloaded file is
checked against the release's `SHA256SUMS`. `eng.Voices()` lists the 28
voices. `Close` releases the native runtime's memory, which matters when a
process loads a second engine.

The examples on this page need loudkit 0.1.1. To run the example from a
repository checkout, run `go run ./examples/hello` in `go/`.
`examples/hello/main.go` is the program above.

`loudr-1` and `loudr-1-turbo` use the same API. To switch models, change the
model name. The same voice profiles work with both models. `loudkit.Load` also
accepts a local release directory. Give it with a path prefix, such as
`./loudr-1`: `Load` reads the bare names `loudr-1` and `loudr-1-turbo` as repo
ids.


## API overview

```go
loudkit.Download(repo, dir)                            // download to a local directory
loudkit.DownloadWith(repo, dir, loudkit.Fetch{Revision: "v0.1.1", Cloning: true})
loudkit.Load(dirOrRepoID)                              // a local directory or a repo id
eng.Voices()                                           // the voice names in the release
eng.Voice("joe")                                       // load one voice by name
eng.Synthesize(text, v, loudkit.Options{Seed: 7, Language: "pl", Speed: 1.25, PreviousTokens: earlier.Tokens})
eng.Stream(text, v, loudkit.Options{}, func(c loudkit.Chunk) bool { play(c.Audio); return true })
mine, err := eng.Enroll("me.wav", "mine", "en")           // clone; fetches the enrollment graphs once
if err != nil { log.Fatal(err) }
if err := mine.Save("mine.safetensors"); err != nil { log.Fatal(err) }
voice.Load("mine.safetensors")
out.SaveWav(path); out.WriteWav(w)                     // 16-bit PCM
```

`Synthesize` splits long text into chunks at sentence boundaries and joins
the audio in memory. The zero `Options` value selects seed 0, the voice's own
language and speed 1.0. `Stream` passes each chunk to the callback when it is
ready. Return false to stop, or set `Options.ShouldCancel`, which is checked on
every decode step.

On an engine loaded by repo id, the first `Enroll` call fetches the enrollment
graphs. For a local directory, fetch them with `Fetch{Cloning: true}`.
`loudkit.LoadPaths(checkpoint, onnxDir, tokenizer)` loads assets from the
paths you give.

A release is hundreds of megabytes, and this package sets no deadline on a
download. These calls take a `context.Context`: `DownloadContext`,
`LoadContext`, `EnrollContext` and `EnrollPCMContext`. Use them in a server to
set a deadline or to cancel. A cancelled download stops the transfer, and the
next call resumes the partial file. A cancelled call returns the context's
error. A call that cannot reach the Hub uses the cached release instead.

Streaming, timestamps, speed and barge-in: `docs/guides/08-go.md`.

## Execution provider

`loudkit.Load` uses the `auto` provider. To choose a provider, build the
engine with `engine.LoadWith`. `LoadWith` does not start ONNX Runtime, so call
`onnx.SetSharedLibraryPath` and `onnx.InitializeEnvironment` first:

```go
onnx.SetSharedLibraryPath("/usr/local/lib/libonnxruntime.so") // the library file on your machine
if err := onnx.InitializeEnvironment(); err != nil {
	log.Fatal(err)
}
defer onnx.DestroyEnvironment()

b, err := loudkit.Open("./loudr-1") // finds the checkpoint, graphs and tokenizer
if err != nil {
	log.Fatal(err)
}
eng, err := engine.LoadWith(b.Checkpoint, b.ONNXDir, b.Tokenizer, config.ExecutionConfig{
	ONNXProvider: config.ProviderCUDA, // auto, cpu, cuda, coreml, directml
})
if err != nil {
	log.Fatal(err)
}
defer eng.Close()
fmt.Println(eng.Describe())
```

The block imports `fmt`, `log`, this module's root package, and its `config`,
`engine` and `onnx` packages. `docs/guides/08-go.md` has the complete program.

`auto` selects CUDA where the shared library has it, and CPU otherwise. It
never selects CoreML or DirectML. If you name a provider that the library
does not have, the load returns an error. The shared library decides which
providers are available:

| provider | shared library |
| --- | --- |
| `cpu` | any build |
| `cuda` | `onnxruntime-gpu` wheel, or the `onnxruntime-linux-x64-gpu` archive |
| `coreml` | a macOS build |
| `directml` | the `onnxruntime-directml` build, on Windows |

A GPU provider can change the token stream and waveform. Conformance runs use
CPU. See [`docs/benchmarks.md`](../docs/benchmarks.md#onnx-execution-providers).

## Build and test

```bash
gofmt -l . && go vet ./...
go test ./...                 # weight-free; the engine conformance needs LOUDKIT_* assets
```
