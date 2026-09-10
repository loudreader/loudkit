package loudkit

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeFile writes one file inside a fake release, creating its directory.
func writeFile(t *testing.T, root, rel string, body []byte) string {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, body, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// writeCheckpoint writes a safetensors file carrying one manifest and no
// tensors: enough for the header reader that decides which of two files is the
// synthesis artefact.
func writeCheckpoint(t *testing.T, root, rel string, manifest map[string]any) string {
	t.Helper()
	encoded := []byte("{}")
	if manifest != nil {
		var err error
		encoded, err = json.Marshal(manifest)
		if err != nil {
			t.Fatal(err)
		}
	}
	header, err := json.Marshal(map[string]any{
		"__metadata__": map[string]string{"manifest": string(encoded)},
	})
	if err != nil {
		t.Fatal(err)
	}
	body := make([]byte, 8, 8+len(header))
	binary.LittleEndian.PutUint64(body, uint64(len(header)))
	return writeFile(t, root, rel, append(body, header...))
}

// fakeRelease is a directory shaped like a downloaded onnx release.
func fakeRelease(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	writeCheckpoint(t, root, checkpointName, map[string]any{"artifact_role": "synthesis"})
	writeFile(t, root, tokenizerName, []byte(`{"model":{}}`))
	writeFile(t, root, manifestName, []byte(`{}`))
	for _, rel := range onnxSynthesis {
		writeFile(t, root, rel, []byte("graph"))
	}
	writeFile(t, root, "voices/joe"+voiceSuffix, []byte("voice"))
	writeFile(t, root, "voices/gosia"+voiceSuffix, []byte("voice"))
	return root
}

func TestOpenFindsTheThreePaths(t *testing.T) {
	root := fakeRelease(t)
	b, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	if b.Checkpoint != filepath.Join(root, checkpointName) {
		t.Errorf("checkpoint = %s", b.Checkpoint)
	}
	if b.ONNXDir != filepath.Join(root, "onnx") {
		t.Errorf("onnx dir = %s", b.ONNXDir)
	}
	if b.Tokenizer != filepath.Join(root, tokenizerName) {
		t.Errorf("tokenizer = %s", b.Tokenizer)
	}
	if got := b.Voices(); len(got) != 2 || got[0] != "gosia" || got[1] != "joe" {
		t.Errorf("voices = %v, want [gosia joe] sorted", got)
	}
	if b.CanEnroll() {
		t.Error("a synthesis-only release must not claim it can enroll")
	}
}

func TestOpenNamesEveryMissingPiece(t *testing.T) {
	root := fakeRelease(t)
	if err := os.Remove(filepath.Join(root, "onnx", "vocoder.onnx")); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(root, tokenizerName)); err != nil {
		t.Fatal(err)
	}
	_, err := Open(root)
	if err == nil {
		t.Fatal("a release missing a graph and its tokenizer must not open")
	}
	for _, want := range []string{"onnx/vocoder.onnx", tokenizerName} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error does not name %s: %v", want, err)
		}
	}
}

func TestOpenRefusesAFileAndAnAbsentDirectory(t *testing.T) {
	root := t.TempDir()
	path := writeFile(t, root, "loudr-1.safetensors", []byte("x"))
	if _, err := Open(path); err == nil {
		t.Error("a file is not a release directory")
	}
	if _, err := Open(filepath.Join(root, "nowhere")); err == nil {
		t.Error("an absent directory is not a release")
	}
}

func TestCanEnrollSeesTheEnrollmentGraphs(t *testing.T) {
	root := fakeRelease(t)
	for _, rel := range onnxEnroll {
		writeFile(t, root, rel, []byte("graph"))
	}
	b, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	if !b.CanEnroll() {
		t.Error("a cloning release carries all three enrollment graphs")
	}
}

func TestFindCheckpointRules(t *testing.T) {
	t.Run("canonical name wins", func(t *testing.T) {
		root := t.TempDir()
		writeCheckpoint(t, root, checkpointName, nil)
		writeCheckpoint(t, root, "something-else.safetensors", nil)
		got, err := findCheckpoint(root)
		if err != nil {
			t.Fatal(err)
		}
		if filepath.Base(got) != checkpointName {
			t.Errorf("got %s", got)
		}
	})

	t.Run("the declared synthesis role wins when renamed", func(t *testing.T) {
		root := t.TempDir()
		writeCheckpoint(t, root, "mine.safetensors", map[string]any{"artifact_role": "synthesis"})
		writeCheckpoint(t, root, "mine-enrollment.safetensors",
			map[string]any{"artifact_role": "enrollment"})
		got, err := findCheckpoint(root)
		if err != nil {
			t.Fatal(err)
		}
		if filepath.Base(got) != "mine.safetensors" {
			t.Errorf("got %s", got)
		}
	})

	t.Run("a pre-split file claiming nothing still loads", func(t *testing.T) {
		root := t.TempDir()
		writeCheckpoint(t, root, "packed.safetensors", map[string]any{})
		got, err := findCheckpoint(root)
		if err != nil {
			t.Fatal(err)
		}
		if filepath.Base(got) != "packed.safetensors" {
			t.Errorf("got %s", got)
		}
	})

	t.Run("the voice encoder is not a checkpoint", func(t *testing.T) {
		root := t.TempDir()
		writeCheckpoint(t, root, "packed.safetensors", nil)
		writeCheckpoint(t, root, voiceEncoderName, nil)
		if _, err := findCheckpoint(root); err != nil {
			t.Fatalf("ve.safetensors must not make the release ambiguous: %v", err)
		}
	})

	t.Run("two undeclared candidates are ambiguous", func(t *testing.T) {
		root := t.TempDir()
		writeCheckpoint(t, root, "one.safetensors", nil)
		writeCheckpoint(t, root, "two.safetensors", nil)
		_, err := findCheckpoint(root)
		if err == nil || !strings.Contains(err.Error(), "name the one you mean") {
			t.Fatalf("got %v", err)
		}
	})

	t.Run("the enrollment half alone cannot speak", func(t *testing.T) {
		root := t.TempDir()
		writeCheckpoint(t, root, enrollmentName, map[string]any{"artifact_role": "enrollment"})
		_, err := findCheckpoint(root)
		if err == nil || !strings.Contains(err.Error(), "enrollment artefact") {
			t.Fatalf("got %v", err)
		}
	})

	t.Run("an empty directory says what a release is", func(t *testing.T) {
		_, err := findCheckpoint(t.TempDir())
		if err == nil || !strings.Contains(err.Error(), "no checkpoint here") {
			t.Fatalf("got %v", err)
		}
	})

	t.Run("both models under their own names is not a pick to make", func(t *testing.T) {
		root := t.TempDir()
		writeCheckpoint(t, root, checkpointName, nil)
		writeCheckpoint(t, root, turboName, nil)
		_, err := findCheckpoint(root)
		if err == nil || !strings.Contains(err.Error(), "2 models here") {
			t.Fatalf("got %v", err)
		}
	})

	t.Run("the canonical name does not outrank the manifest's claim", func(t *testing.T) {
		root := t.TempDir()
		writeCheckpoint(t, root, checkpointName, map[string]any{"artifact_role": "enrollment"})
		_, err := findCheckpoint(root)
		if err == nil || !strings.Contains(err.Error(), "enrollment artefact") {
			t.Fatalf("got %v", err)
		}
	})
}

func TestVoicePathRefusesAnAddress(t *testing.T) {
	b, err := Open(fakeRelease(t))
	if err != nil {
		t.Fatal(err)
	}
	got, err := b.VoicePath("joe")
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Base(got) != "joe"+voiceSuffix {
		t.Errorf("got %s", got)
	}
	if _, err := b.VoicePath("joe" + voiceSuffix); err != nil {
		t.Errorf("a name with the suffix on it is still a name: %v", err)
	}
	for _, bad := range []string{"../../id_rsa", `..\id_rsa`, "sub/joe", ".hidden", ""} {
		if _, err := b.VoicePath(bad); err == nil {
			t.Errorf("%q was resolved; a voice is named, not addressed", bad)
		}
	}
	err = func() error { _, err := b.VoicePath("nobody"); return err }()
	if err == nil || !strings.Contains(err.Error(), "gosia, joe") {
		t.Errorf("an unknown voice should list the ones there are: %v", err)
	}
}
