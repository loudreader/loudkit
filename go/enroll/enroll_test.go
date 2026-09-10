package enroll

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/loudreader/loudkit/go/config"
	"github.com/loudreader/loudkit/go/onnx"
)

// The enrollment port, gated against the shared fixture: the same reference
// clip must yield the fixture's prompt tokens exactly, and its embeddings to
// cosine > 0.9999. Needs the exported enrollment graphs and the onnxruntime
// shared library; skips with a named reason otherwise.
func testEnroll(t *testing.T) *Result {
	enr, audio := testEnroller(t)
	res, err := enr.Enroll(audio, 24000)
	if err != nil {
		t.Fatalf("enroll: %v", err)
	}
	return res
}

// testEnroller is testEnroll's setup, split out so a test can run the clone
// more than once against one set of graphs.
func testEnroller(t *testing.T) (*Enroller, []float32) {
	onnxDir := os.Getenv("LOUDKIT_ONNX_DIR")
	lib := os.Getenv("LOUDKIT_ONNXRUNTIME_LIB")
	fixture := os.Getenv("LOUDKIT_ENROLL_FIXTURE")
	if fixture == "" {
		fixture = filepath.Join("..", "..", "tests", "data", "enrollment")
	}
	if onnxDir == "" || lib == "" {
		t.Skip("set LOUDKIT_ONNX_DIR and LOUDKIT_ONNXRUNTIME_LIB")
	}
	if _, err := os.Stat(filepath.Join(fixture, "ref_audio.f32")); err != nil {
		t.Skip("enrollment fixture not found: " + fixture)
	}

	onnx.SetSharedLibraryPath(lib)
	if err := onnx.InitializeEnvironment(); err != nil {
		t.Skipf("onnxruntime: %v", err)
	}
	t.Cleanup(func() { _ = onnx.DestroyEnvironment() })

	// CPU by name, not the auto default: the fixture this compares against was
	// produced on CPU (tools/make_conformance.py pins the same), so a machine
	// that happens to offer CoreML would otherwise measure a different device
	// and report it as a port that disagrees with Python. What a GPU provider
	// does to these numbers is a measurement, and it does not belong inside a
	// parity gate.
	enr, err := LoadEnrollerWith(onnxDir, config.ExecutionConfig{
		ONNXProvider: config.ProviderCPU,
	})
	if err != nil {
		t.Fatalf("load enroller: %v", err)
	}
	t.Cleanup(enr.Close)

	return enr, readF32(t, filepath.Join(fixture, "ref_audio.f32"))
}

func readF32(t *testing.T, path string) []float32 {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("%s: %v", path, err)
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

func readI64(t *testing.T, path string) []int64 {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("%s: %v", path, err)
	}
	out := make([]int64, len(b)/8)
	for i := range out {
		out[i] = int64(binary.LittleEndian.Uint64(b[8*i:]))
	}
	return out
}

func fixtureDir(t *testing.T) string {
	t.Helper()
	d := os.Getenv("LOUDKIT_ENROLL_FIXTURE")
	if d == "" {
		d = filepath.Join("..", "..", "tests", "data", "enrollment")
	}
	return d
}

func TestPromptTokensExact(t *testing.T) {
	res := testEnroll(t)
	want := readI64(t, filepath.Join(fixtureDir(t), "prompt_tokens.i64"))
	if len(res.PromptTokens) != len(want) {
		t.Fatalf("prompt tokens: got %d, want %d", len(res.PromptTokens), len(want))
	}
	for i := range want {
		if res.PromptTokens[i] != want[i] {
			t.Fatalf("prompt token %d: got %d, want %d", i, res.PromptTokens[i], want[i])
		}
	}
}

func TestCondTokensExact(t *testing.T) {
	res := testEnroll(t)
	want := readI64(t, filepath.Join(fixtureDir(t), "cond_prompt_tokens.i64"))
	if len(res.CondPromptTokens) != len(want) {
		t.Fatalf("cond tokens: got %d, want %d", len(res.CondPromptTokens), len(want))
	}
	for i := range want {
		if res.CondPromptTokens[i] != want[i] {
			t.Fatalf("cond token %d: got %d, want %d", i, res.CondPromptTokens[i], want[i])
		}
	}
}

func TestEmbeddingsMatch(t *testing.T) {
	res := testEnroll(t)
	fx := fixtureDir(t)

	flow := readF32(t, filepath.Join(fx, "flow_embedding.f32"))
	if cos(res.FlowEmbedding, flow) <= 0.9999 {
		t.Fatalf("flow embedding cosine %f <= 0.9999", cos(res.FlowEmbedding, flow))
	}
	speaker := readF32(t, filepath.Join(fx, "speaker_embedding.f32"))
	if cos(res.SpeakerEmbedding, speaker) <= 0.9999 {
		t.Fatalf("speaker embedding cosine %f <= 0.9999", cos(res.SpeakerEmbedding, speaker))
	}
}

// Two clones from one enroller in one process must agree element for element.
// The enroller destroys every tensor it builds and every tensor a Run hands
// back, and a Result that still aliased a released output buffer would read
// differently, or crash, once the second clone reused the memory.
//
// The length-against-capacity check is the cheap half and needs no second run:
// a view returned straight from GetData over a larger output tensor carries a
// capacity past its length, and a copy does not.
func TestEnrollmentIsStableAcrossTwoRuns(t *testing.T) {
	enr, audio := testEnroller(t)

	first, err := enr.Enroll(audio, 24000)
	if err != nil {
		t.Fatalf("first enroll: %v", err)
	}
	second, err := enr.Enroll(audio, 24000)
	if err != nil {
		t.Fatalf("second enroll: %v", err)
	}

	// Two enrolments must not hand out slices over one array, and neither may
	// carry spare capacity: a caller appending to a profile's slice would then
	// write into that profile's own array, and two such appends would alias.
	// Length against capacity alone does not say this, because a grown copy
	// lands on an exact capacity whenever the length happens to sit on one of
	// the growth steps, so both are checked.
	if len(first.FlowEmbedding) != cap(first.FlowEmbedding) {
		t.Errorf("flow embedding carries slack: len %d, cap %d",
			len(first.FlowEmbedding), cap(first.FlowEmbedding))
	}
	if len(first.CondPromptTokens) != cap(first.CondPromptTokens) {
		t.Errorf("cond prompt tokens carry slack: len %d, cap %d",
			len(first.CondPromptTokens), cap(first.CondPromptTokens))
	}
	if len(first.CondPromptTokens) > 0 {
		was := second.CondPromptTokens[0]
		first.CondPromptTokens[0] = was + 1
		if second.CondPromptTokens[0] != was {
			t.Error("the two enrolments share one array of cond prompt tokens")
		}
		first.CondPromptTokens[0] = was
	}
	if len(first.FlowEmbedding) > 0 {
		was := second.FlowEmbedding[0]
		first.FlowEmbedding[0] = was + 1
		if second.FlowEmbedding[0] != was {
			t.Error("the two enrolments share one flow embedding array")
		}
		first.FlowEmbedding[0] = was
	}

	sameI64(t, "prompt tokens", first.PromptTokens, second.PromptTokens)
	sameI64(t, "cond prompt tokens", first.CondPromptTokens, second.CondPromptTokens)
	sameF32(t, "flow embedding", first.FlowEmbedding, second.FlowEmbedding)
	sameF32(t, "speaker embedding", first.SpeakerEmbedding, second.SpeakerEmbedding)
	sameF32(t, "prompt mel", first.PromptMel, second.PromptMel)
	if first.PromptMelFrames != second.PromptMelFrames {
		t.Errorf("prompt mel frames: %d then %d", first.PromptMelFrames, second.PromptMelFrames)
	}
}

func sameI64(t *testing.T, what string, a, b []int64) {
	t.Helper()
	if len(a) != len(b) {
		t.Errorf("%s: %d values then %d", what, len(a), len(b))
		return
	}
	for i := range a {
		if a[i] != b[i] {
			t.Errorf("%s[%d]: %d then %d", what, i, a[i], b[i])
			return
		}
	}
}

func sameF32(t *testing.T, what string, a, b []float32) {
	t.Helper()
	if len(a) != len(b) {
		t.Errorf("%s: %d values then %d", what, len(a), len(b))
		return
	}
	for i := range a {
		if a[i] != b[i] {
			t.Errorf("%s[%d]: %g then %g", what, i, a[i], b[i])
			return
		}
	}
}

func cos(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
