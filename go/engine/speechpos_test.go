package engine

import "testing"

// The learned speech positional embedding is indexed per generated token, and
// the index is not the step: prefillEmbeds has already written the carried
// prefix at rows 1..len(prefix), so generation continues above it.
//
// Tested against speechPosition and speechRow rather than through decodeStep
// for the reason carry_test.go gives: engine.Engine holds six concrete
// *onnx.Session values, so nothing drives a decode step without a checkpoint
// and a runtime. The arithmetic is the whole of the defect and it is pure.
//
// positionEngine leaves speechEmb zero and fills speechPos row r with r, so
// the row an embedding was built from is readable off the embedding itself.
func positionEngine(rows int) *Engine {
	e := &Engine{
		speechEmb: make([]float32, rows*hiddenDim),
		speechPos: make([]float32, rows*hiddenDim),
	}
	for r := 0; r < rows; r++ {
		for j := 0; j < hiddenDim; j++ {
			e.speechPos[r*hiddenDim+j] = float32(r)
		}
	}
	return e
}

// The bug: step+1 asks for row 1 for the first generated token, which is the
// row the prefix's own first token occupies.
func TestTheFirstGeneratedTokenSitsAboveThePrefix(t *testing.T) {
	const prefixLen = 6
	if got := speechPosition(prefixLen, 0); got != prefixLen+1 {
		t.Fatalf("the first generated token asked for row %d, want %d", got, prefixLen+1)
	}
	e := positionEngine(64)
	row := e.speechRow(0, speechPosition(prefixLen, 0))
	if row[0] != float32(prefixLen+1) {
		t.Errorf("the embedding was built from row %v, want row %d", row[0], prefixLen+1)
	}
}

// Every generated token, not only the first: with a prefix of P the positions
// run P+1, P+2, ... and none of them re-enters the block the prefill wrote.
func TestGeneratedPositionsNeverReenterThePrefix(t *testing.T) {
	const prefixLen = 6
	for step := 0; step < 10; step++ {
		got := speechPosition(prefixLen, step)
		if got <= prefixLen {
			t.Fatalf("step %d landed on row %d, inside the prefix's rows 1..%d", step, got, prefixLen)
		}
		if want := prefixLen + step + 1; got != want {
			t.Fatalf("step %d asked for row %d, want %d", step, got, want)
		}
	}
}

// A single window carries nothing, generation starts at row 1, and the fix
// changes nothing there — which is why no single-window test ever caught this.
func TestWithoutAPrefixGenerationStillStartsAtRowOne(t *testing.T) {
	for step := 0; step < 4; step++ {
		if got := speechPosition(0, step); got != step+1 {
			t.Errorf("step %d asked for row %d, want %d", step, got, step+1)
		}
	}
}
