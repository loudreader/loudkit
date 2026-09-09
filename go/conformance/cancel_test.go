package conformance

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/loudreader/loudkit/go/config"
	"github.com/loudreader/loudkit/go/engine"
	"github.com/loudreader/loudkit/go/onnx"
	"github.com/loudreader/loudkit/go/voice"
)

// TestCancelAtADecodeStep holds the cancellation contract every port shares:
// cancelled at a decode step inside the second chunk, Synthesize returns
// ErrCancelled and no Result, and Stream delivers the first chunk exactly as
// an uncancelled run does and nothing after, with no error. Needs the same
// assets as TestEngineConformance; skips without them.
func TestCancelAtADecodeStep(t *testing.T) {
	ckpt := os.Getenv("LOUDKIT_CKPT")
	onnxDir := os.Getenv("LOUDKIT_ONNX_DIR")
	voicePath := os.Getenv("LOUDKIT_VOICE")
	lib := os.Getenv("LOUDKIT_ONNXRUNTIME_LIB")
	fixture := os.Getenv("LOUDKIT_FIXTURE_DIR")
	if fixture == "" {
		fixture = filepath.Join("..", "..", "tests", "data", "conformance")
	}
	if ckpt == "" || onnxDir == "" || voicePath == "" || lib == "" {
		skipOrFail(t, "set LOUDKIT_CKPT/LOUDKIT_ONNX_DIR/LOUDKIT_VOICE/LOUDKIT_ONNXRUNTIME_LIB")
	}

	onnx.SetSharedLibraryPath(lib)
	if err := onnx.InitializeEnvironment(); err != nil {
		t.Fatal(err)
	}
	defer onnx.DestroyEnvironment()
	eng, err := engine.LoadWith(ckpt, onnxDir, filepath.Join(fixture, "tokenizer.json"),
		config.ExecutionConfig{ONNXProvider: config.ProviderCPU})
	if err != nil {
		t.Fatal(err)
	}
	defer eng.Close()
	buf, err := os.ReadFile(filepath.Join(fixture, fixtureName(eng.Config().DecodeMode)))
	if err != nil {
		skipOrFail(t, "fixture not found: "+fixture)
	}
	var vectors map[string]interface{}
	if err := json.Unmarshal(buf, &vectors); err != nil {
		t.Fatal(err)
	}
	longForm, ok := vectors["long_form"].(map[string]interface{})
	if !ok {
		skipOrFail(t, "fixture has no long_form section")
	}
	kase := longForm["cases"].([]interface{})[0].(map[string]interface{})
	text := kase["text"].(string)

	v, err := voice.Load(voicePath)
	if err != nil {
		t.Fatal(err)
	}
	o := engine.Options{Seed: uint64(toFloat(kase["seed"])), Language: kase["language"].(string)}

	// Uncancelled first, counting the polls: the step to cancel at has to
	// land inside the second chunk's decode loop, and only a run that
	// finished can say where that is.
	polls := 0
	o.ShouldCancel = func() bool { polls++; return false }
	var pollsAtChunk []int
	var first []int
	if err := eng.Stream(text, v, o, func(c engine.Chunk) bool {
		pollsAtChunk = append(pollsAtChunk, polls)
		if first == nil {
			first = append([]int(nil), c.Tokens...)
		}
		return true
	}); err != nil {
		t.Fatal(err)
	}
	if len(pollsAtChunk) < 2 {
		t.Fatalf("the passage must split; it streamed as %d chunk(s)", len(pollsAtChunk))
	}
	step := pollsAtChunk[0] + 5
	if step >= pollsAtChunk[1] {
		t.Fatalf("step %d is not inside chunk 1 (polls %v)", step, pollsAtChunk)
	}

	// Stream: the first chunk arrives as it was, nothing after, no error.
	polls = 0
	o.ShouldCancel = func() bool { polls++; return polls > step }
	var got [][]int
	if err := eng.Stream(text, v, o, func(c engine.Chunk) bool {
		got = append(got, append([]int(nil), c.Tokens...))
		return true
	}); err != nil {
		t.Fatalf("Stream returned %v; a cancel ends the stream without an error", err)
	}
	if len(got) != 1 {
		t.Fatalf("%d chunks arrived after a cancel inside chunk 1; want the 1 that finished", len(got))
	}
	if !equalInts(got[0], first) {
		t.Fatalf("the chunk before the cancel is not the chunk an uncancelled run delivered")
	}

	// Synthesize: no Result, ErrCancelled.
	polls = 0
	res, err := eng.Synthesize(text, v, o)
	if !errors.Is(err, engine.ErrCancelled) {
		t.Fatalf("Synthesize returned err=%v; want ErrCancelled", err)
	}
	if res != nil {
		t.Fatalf("Synthesize handed back a Result of %d tokens after a cancel", len(res.Tokens))
	}

	// SynthesizeWindow, cancelled three steps in: the same signal.
	polls = 0
	o.ShouldCancel = func() bool { polls++; return polls > 3 }
	res, err = eng.SynthesizeWindow("Hello from loudkit.", v, o)
	if !errors.Is(err, engine.ErrCancelled) || res != nil {
		t.Fatalf("SynthesizeWindow returned (%v, %v); want (nil, ErrCancelled)", res, err)
	}
}
