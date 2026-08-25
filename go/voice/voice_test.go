package voice

import (
	"math"
	"testing"
)

// A profile the renderers would disagree about must not load.
//
// Shape validation alone let a well-shaped but degenerate file through, and the
// three renderers then differed on what it meant: torch's F.normalize carries
// an epsilon and returns a finite (arbitrary) direction for a zero speaker
// vector, while this port and CoreML divide by the raw norm and produce NaN.
func TestDegenerateEmbeddingsAreRefused(t *testing.T) {
	good := make([]float32, flowDim)
	for i := range good {
		good[i] = 0.0625
	}
	if err := checkEmbedding("flow_embedding", good, flowDim); err != nil {
		t.Fatalf("a healthy embedding was refused: %v", err)
	}
	if err := checkEmbedding("flow_embedding", make([]float32, flowDim), flowDim); err == nil {
		t.Fatal("a zero embedding was accepted; it normalises to NaN here")
	}
	if err := checkEmbedding("flow_embedding", good[:8], flowDim); err == nil {
		t.Fatal("a short embedding was accepted")
	}
	nan := append([]float32(nil), good...)
	nan[3] = float32(math.NaN())
	if err := checkEmbedding("flow_embedding", nan, flowDim); err == nil {
		t.Fatal("NaN was accepted")
	}
	inf := append([]float32(nil), good...)
	inf[0] = float32(math.Inf(1))
	if err := checkEmbedding("flow_embedding", inf, flowDim); err == nil {
		t.Fatal("infinity was accepted")
	}
}
