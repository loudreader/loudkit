package loudkit

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLibraryCandidatesPerPlatform(t *testing.T) {
	for goos, want := range map[string]string{
		"darwin":  ".dylib",
		"linux":   ".so",
		"windows": ".dll",
	} {
		got := libraryCandidates(goos)
		if len(got) == 0 {
			t.Fatalf("%s: no candidates", goos)
		}
		for _, path := range got {
			if !strings.HasSuffix(path, want) {
				t.Errorf("%s: %s is not a %s", goos, path, want)
			}
		}
	}
	// An unknown platform gets the ELF list rather than nothing to look at.
	if len(libraryCandidates("plan9")) == 0 {
		t.Error("an unknown platform still needs somewhere to look")
	}
}

func TestFindLibraryPrefersTheNamedOne(t *testing.T) {
	lib := filepath.Join(t.TempDir(), "libonnxruntime.dylib")
	if err := os.WriteFile(lib, []byte("not really a library"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv(LibraryEnv, lib)
	got, err := findLibrary()
	if err != nil || got != lib {
		t.Errorf("findLibrary = %q, %v", got, err)
	}

	// A name that points at nothing is an error about that name, never a
	// silent fall through to whatever this machine happens to have.
	t.Setenv(LibraryEnv, lib+".missing")
	if _, err := findLibrary(); err == nil || !strings.Contains(err.Error(), LibraryEnv) {
		t.Errorf("got %v", err)
	}
}

func TestInstallHintNamesACommand(t *testing.T) {
	if hint := installHint("darwin"); !strings.Contains(hint, "brew install onnxruntime") {
		t.Errorf("darwin hint = %q", hint)
	}
	for _, goos := range []string{"linux", "windows"} {
		if hint := installHint(goos); !strings.Contains(hint, "github.com/microsoft/onnxruntime/releases") {
			t.Errorf("%s hint = %q", goos, hint)
		}
	}
}
