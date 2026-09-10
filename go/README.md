# loudkit for Go

Text to speech in Go, on onnxruntime. No Python, no torch.

## Hello

Go 1.25 or newer, and one shared library that cannot be vendored: `brew
install onnxruntime` on macOS, `apt install libonnxruntime-dev` on Linux, the
[onnxruntime-win-x64 archive](https://github.com/microsoft/onnxruntime/releases)
on Windows. It is found automatically; set `LOUDKIT_ONNXRUNTIME_LIB` if yours is
somewhere unusual.

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

The first run downloads the model files into
`~/Library/Caches/loudkit/loudreader--loudr-1` on macOS and
`~/.cache/loudkit/loudreader--loudr-1` on Linux (`$LOUDKIT_CACHE` moves it),
the directory the Rust, JS and Swift ports share, and checks every file
against the release's own `SHA256SUMS`; later runs read what is there.
`eng.Voices()` names the 28 voices. `Close` hands back the runtime's
memory; it matters when you build a second engine.

The snippets on this page need loudkit 0.1.1. From a checkout, `go run
./examples/hello` in `go/` runs `examples/hello/main.go`, which is this file.

Both `loudr-1` and `loudr-1-turbo` use this API in 0.1.1. Change the model
name to switch; keep the same voice profile. A local release directory works
as well as a published model name.


## The rest of the front door

```go
loudkit.Download(repo, dir)                            // a directory of your own
loudkit.DownloadWith(repo, dir, loudkit.Fetch{Revision: "v0.1.1", Cloning: true})
loudkit.Load(dirOrRepoID)                              // a directory or a repo id
eng.Voices()                                           // the names in the release
eng.Voice("joe")                                       // one of them
eng.Synthesize(text, v, loudkit.Options{Seed, Language, Speed, PreviousTokens})
eng.Stream(text, v, loudkit.Options{}, func(c loudkit.Chunk) bool { play(c.Audio); return true })
mine, err := eng.Enroll("me.wav", "mine", "en")           // clone; fetches the enrollment graphs once
if err != nil { log.Fatal(err) }
if err := mine.Save("mine.safetensors"); err != nil { log.Fatal(err) }
voice.Load("mine.safetensors")
out.SaveWav(path); out.WriteWav(w)                     // 16-bit PCM
```

`Synthesize` takes text of any length: it splits at sentence boundaries and
joins the audio. The zero `Options` is seed 0, the voice's own language and
normal speed. `Stream` hands out chunks as they are made; return false to
stop, or set `Options.ShouldCancel` to stop within one decode step. `Enroll`
on an engine loaded by repo id fetches the enrollment graphs once; a directory
of your own needs `Fetch{Cloning: true}`.
`loudkit.LoadPaths(checkpoint, onnxDir, tokenizer)` opens a layout of your
own.

A release is hundreds of megabytes and this package sets no deadline on the
link, so every call that can fetch one has a sibling taking a
`context.Context`: `DownloadContext`, `LoadContext`, `EnrollContext` and
`EnrollPCMContext`. Reach for those in a server. Cancelling stops the
transfer, and the next call resumes the part-file it left. A cancelled call
returns the context's error rather than falling back to the cached release,
which a call that could not reach the hub still does.

Streaming, timestamps, speed and barge-in: `docs/guides/08-go.md`.

## Execution provider

`Load` takes the best provider the shared library offers. To name one, build
the engine yourself:

```go
eng, err := engine.LoadWith(ckpt, onnxDir, tokPath, config.ExecutionConfig{
	ONNXProvider: config.ProviderCUDA, // auto, cpu, cuda, coreml, directml
})
fmt.Println(eng.Describe())
```

`auto` takes cuda where the shared library offers it and cpu otherwise; it
reaches neither coreml nor directml. A named provider the library does not
carry is an error, never a quiet fall back to cpu. Which providers exist is
decided by the shared library alone:

| provider | shared library |
| --- | --- |
| `cpu` | any build |
| `cuda` | `onnxruntime-gpu` wheel, or the `onnxruntime-linux-x64-gpu` archive |
| `coreml` | a macOS build |
| `directml` | the `onnxruntime-directml` build, on Windows |

A GPU provider can change the token stream and waveform; conformance runs pin
CPU. See [`docs/benchmarks.md`](../docs/benchmarks.md#onnx-execution-providers).

## Build and test

```bash
gofmt -l . && go vet ./...
go test ./...                 # weight-free; the engine conformance needs LOUDKIT_* assets
```
