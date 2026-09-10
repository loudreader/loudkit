package conformance

import (
	"bytes"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/loudreader/loudkit/go/config"
	"github.com/loudreader/loudkit/go/engine"
	"github.com/loudreader/loudkit/go/enroll"
	"github.com/loudreader/loudkit/go/onnx"
)

// One enrollment result must remain reusable across both decode families.
func TestCloneOnceAcrossBothModels(t *testing.T) {
	first, second := os.Getenv("LOUDKIT_CKPT"), os.Getenv("LOUDKIT_SECOND_CKPT")
	graphs, secondGraphs := os.Getenv("LOUDKIT_ONNX_DIR"), os.Getenv("LOUDKIT_SECOND_ONNX_DIR")
	lib := os.Getenv("LOUDKIT_ONNXRUNTIME_LIB")
	if first == "" || second == "" || graphs == "" || secondGraphs == "" || lib == "" {
		skipOrFail(t, "set both model checkpoint/graph paths and ONNX runtime library")
	}
	fixture := os.Getenv("LOUDKIT_FIXTURE_DIR")
	if fixture == "" {
		fixture = "../../tests/data/conformance"
	}
	enrollment := os.Getenv("LOUDKIT_ENROLL_FIXTURE")
	if enrollment == "" {
		enrollment = "../../tests/data/enrollment"
	}
	onnx.SetSharedLibraryPath(lib)
	if err := onnx.InitializeEnvironment(); err != nil {
		t.Fatal(err)
	}
	defer onnx.DestroyEnvironment()
	execution := config.ExecutionConfig{ONNXProvider: config.ProviderCPU}
	enr, err := enroll.LoadEnrollerWith(graphs, execution)
	if err != nil {
		t.Fatal(err)
	}
	result, err := enr.Enroll(readF32(t, filepath.Join(enrollment, "ref_audio.f32")), 24000)
	enr.Close()
	if err != nil {
		t.Fatal(err)
	}
	profile := result.Profile("shared", 24000, "en")
	path := filepath.Join(t.TempDir(), "shared.voice.safetensors")
	if err := profile.Save(path); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	modes := map[string]bool{}
	for i, checkpoint := range []string{first, second} {
		dir := []string{graphs, secondGraphs}[i]
		eng, err := engine.LoadWith(checkpoint, dir, filepath.Join(fixture, "tokenizer.json"), execution)
		if err != nil {
			t.Fatal(err)
		}
		modes[eng.Config().DecodeMode] = true
		speech, err := eng.Synthesize("Hello, this is a shared voice.", profile, engine.Options{Seed: 7, Language: "en"})
		eng.Close()
		if err != nil {
			t.Fatal(err)
		}
		if len(speech.Audio) == 0 {
			t.Fatal("empty speech")
		}
		for _, sample := range speech.Audio {
			if math.IsNaN(float64(sample)) || math.IsInf(float64(sample), 0) {
				t.Fatal("non-finite speech")
			}
		}
	}
	if !modes["single"] || !modes["fusion_mtp2"] {
		t.Fatal("the test requires one model from each decode family")
	}
	if err := profile.Save(path); err != nil {
		t.Fatal(err)
	}
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) {
		t.Fatal("synthesis mutated the shared voice profile")
	}
}
