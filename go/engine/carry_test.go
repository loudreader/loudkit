package engine

import (
	"strings"
	"testing"

	"github.com/loudreader/loudkit/go/chunking"
	"github.com/loudreader/loudkit/go/config"
)

// Cross-call prosody context: the tail of the previous call's tokens conditions
// the first window of this one, exactly as the tail of chunk k conditions chunk
// k+1 inside a passage. A request boundary and a chunk boundary are the same
// join, so there is one mechanism and this is it.
//
// Tested against carryFrom rather than through Synthesize because this port has
// no weight-free engine seam: engine.Engine holds six concrete *onnx.Session
// values, so nothing can drive the pipeline without a checkpoint and a runtime.
// The slice is the whole of the behaviour and it is a pure function of the
// config, so an Engine carrying nothing but a config is the practical unit here —
// language_test.go and mel_test.go test in-package for the same reason.
//
// What this therefore does NOT cover is the wiring: if the helper's result
// stopped being handed to the generator's prefix, every assertion here would
// still pass. That half is pinned in Python, by
// tests/test_engine.py::TestCrossRequestContext, against a fake generator that
// records the context it was given — building an equivalent seam in four more
// languages would cost four engine refactors to re-assert one fact.
func carryEngine(prefixTokens int) *Engine {
	return &Engine{config: config.AlgorithmConfig{
		// The shipping value. Anything at or above it is a control token the
		// generator emits and the renderer cannot read.
		StartSpeech: 6561,
		Chunking:    chunking.Config{Enabled: true, MaxTokens: 255, PrefixTokens: prefixTokens},
	}}
}

func TestCarryFromTakesTheTailOfTheHistory(t *testing.T) {
	e := carryEngine(6)
	got, err := e.carryFrom([]int{10, 11, 12, 13, 14, 15, 16, 17, 18, 19})
	if err != nil {
		t.Fatalf("a valid history was refused: %v", err)
	}
	want := []int{14, 15, 16, 17, 18, 19}
	if len(got) != len(want) {
		t.Fatalf("got %v, want the last six: %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v, want the last six: %v", got, want)
		}
	}
}

// The intended call is previousTokens = the whole previous result, with no
// arithmetic at the call site: any length is accepted because only the tail is
// used, and a caller should never have to know the prefix length to make it.
func TestCarryFromAcceptsAHistoryShorterThanThePrefix(t *testing.T) {
	e := carryEngine(6)
	got, err := e.carryFrom([]int{7, 8})
	if err != nil {
		t.Fatalf("a short history was refused: %v", err)
	}
	if len(got) != 2 || got[0] != 7 || got[1] != 8 {
		t.Errorf("got %v, want the whole two-token history", got)
	}
}

// No history is byte-for-byte the behaviour this engine had before the
// parameter existed: nothing is fed in, and the first window starts cold.
func TestCarryFromWithoutAHistoryCarriesNothing(t *testing.T) {
	e := carryEngine(6)
	for _, previous := range [][]int{nil, {}} {
		got, err := e.carryFrom(previous)
		if err != nil {
			t.Fatalf("%v was refused: %v", previous, err)
		}
		if len(got) != 0 {
			t.Errorf("%v carried %v", previous, got)
		}
	}
}

// The bug this guards: previousTokens[len-0:] is the whole slice rather than
// nothing, so a naive tail would condition on the entire previous utterance at
// exactly the setting that means "chunks are independent".
func TestAZeroPrefixCarriesNothingRatherThanEverything(t *testing.T) {
	e := carryEngine(0)
	got, err := e.carryFrom([]int{1, 2, 3, 4, 5})
	if err != nil {
		t.Fatalf("a valid history was refused: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("prefix_tokens = 0 carried %v", got)
	}
}

// The whole input is validated, not only the slice that will be used: an id out
// of range means the sequence was built wrong, and reporting that only when it
// lands in the last six tokens would make the failure depend on
// the length of the caller's text.
func TestAnIdOutsideTheCodebookIsRefusedWhereverItIs(t *testing.T) {
	e := carryEngine(6)
	// 6562 is the stop token — an id the generator emits and the renderer
	// cannot read. First in the history, so the tail alone would never see it.
	_, err := e.carryFrom([]int{6562, 1, 2, 3, 4, 5, 6, 7})
	if err == nil {
		t.Fatal("a control token in the history was accepted")
	}
	if !strings.Contains(err.Error(), "6562") || !strings.Contains(err.Error(), "6561") {
		t.Errorf("the error names neither the id nor the bound: %v", err)
	}
	if _, err := e.carryFrom([]int{1, -1, 2}); err == nil {
		t.Error("a negative id was accepted")
	}
}

// The carry outlives the call that produced it — it is fed to the generator one
// synthesis later — so it must not alias a slice the caller still owns.
func TestCarryFromCopiesRatherThanAliasing(t *testing.T) {
	e := carryEngine(2)
	previous := []int{1, 2, 3, 4}
	got, err := e.carryFrom(previous)
	if err != nil {
		t.Fatalf("a valid history was refused: %v", err)
	}
	previous[3] = 99
	if got[1] != 4 {
		t.Errorf("the carry followed the caller's slice: %v", got)
	}
}
