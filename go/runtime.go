package loudkit

import (
	"fmt"
	"os"
	"runtime"
	"sync"

	onnxruntime "github.com/yalue/onnxruntime_go"

	"github.com/loudreader/loudkit/go/onnx"
)

// LibraryEnv names the shared library to load. Set it when the library is
// somewhere the search below does not look.
const LibraryEnv = "LOUDKIT_ONNXRUNTIME_LIB"

var (
	runtimeMu    sync.Mutex
	runtimeReady bool
)

// initRuntime points onnxruntime at a shared library and starts it, once per
// process.
//
// The library is not bundled and cannot be: it is a platform binary the caller
// installs. What this removes is having to name it, which for the usual
// install is a path the caller would have had to look up.
//
// Success latches; a failure does not. A sync.Once would remember "no
// onnxruntime shared library" for the life of the process, so the caller who
// reads that sentence, sets LOUDKIT_ONNXRUNTIME_LIB and calls Load again gets
// the same refusal from a lookup that never ran. The message names a fix, so
// the fix has to be able to work.
func initRuntime() error {
	runtimeMu.Lock()
	defer runtimeMu.Unlock()
	if runtimeReady {
		return nil
	}
	if onnxruntime.IsInitialized() {
		// A caller who set this up themselves keeps their own setup.
		runtimeReady = true
		return nil
	}
	lib, err := findLibrary()
	if err != nil {
		return err
	}
	// Through the onnx package rather than the binding, both times: it records
	// which library was loaded, so a "provider not available" message can name
	// the file whose build decided that, and its InitializeEnvironment is the
	// one that turns onnxruntime's telemetry off.
	onnx.SetSharedLibraryPath(lib)
	if err := onnx.InitializeEnvironment(); err != nil {
		return err
	}
	runtimeReady = true
	return nil
}

// findLibrary is the shared library this machine has, or a message with the
// one command that installs it.
func findLibrary() (string, error) {
	if named := os.Getenv(LibraryEnv); named != "" {
		if isFile(named) {
			return named, nil
		}
		return "", fmt.Errorf("%s points at %s, which is not a file", LibraryEnv, named)
	}
	for _, candidate := range libraryCandidates(runtime.GOOS) {
		if isFile(candidate) {
			return candidate, nil
		}
	}
	return "", fmt.Errorf("no onnxruntime shared library. %s\nOr set %s to one you "+
		"already have.", installHint(runtime.GOOS), LibraryEnv)
}

// libraryCandidates is where an onnxruntime install puts its shared library on
// one platform, in the order a machine is likely to have them.
func libraryCandidates(goos string) []string {
	switch goos {
	case "darwin":
		return []string{
			"/opt/homebrew/lib/libonnxruntime.dylib",
			"/usr/local/lib/libonnxruntime.dylib",
		}
	case "windows":
		return []string{
			`C:\Program Files\onnxruntime\lib\onnxruntime.dll`,
			"onnxruntime.dll",
		}
	default:
		return []string{
			"/usr/local/lib/libonnxruntime.so",
			"/usr/lib/libonnxruntime.so",
			"/usr/lib/x86_64-linux-gnu/libonnxruntime.so",
			"/usr/lib/aarch64-linux-gnu/libonnxruntime.so",
		}
	}
}

func installHint(goos string) string {
	switch goos {
	case "darwin":
		return "Install it with `brew install onnxruntime`."
	case "windows":
		return "Download onnxruntime-win-x64 from " +
			"https://github.com/microsoft/onnxruntime/releases and unpack it."
	default:
		return "Unpack onnxruntime-linux-x64 from " +
			"https://github.com/microsoft/onnxruntime/releases into /usr/local."
	}
}
