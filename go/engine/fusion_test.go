package engine

import (
	"errors"
	"github.com/loudreader/loudkit/go/sampler"
	"math"
	"slices"
	"testing"
)

func TestPairAlignedCarry(t *testing.T) {
	for _, c := range []struct {
		length, wanted int
		want           []int
	}{
		{9, 5, []int{2, 3, 4, 5, 6, 7}}, {9, 6, []int{2, 3, 4, 5, 6, 7}},
		{8, 5, []int{2, 3, 4, 5, 6, 7}}, {3, 6, []int{0, 1}}, {1, 6, nil}, {9, 0, nil},
	} {
		e := carryEngine(c.wanted)
		e.config.DecodeMode = "fusion_mtp2"
		tokens := make([]int, c.length)
		for i := range tokens {
			tokens[i] = i
		}
		got, err := e.carryFrom(tokens)
		if err != nil || !slices.Equal(got, c.want) {
			t.Fatalf("length%d wanted%d: %v %v", c.length, c.wanted, got, err)
		}
	}
}

func TestPrefixFusionIncludesMeanMLPAndPosition(t *testing.T) {
	e := &Engine{speechEmb: make([]float32, 3*hiddenDim), speechPos: make([]float32, 3*hiddenDim), fusion: map[string][]float32{
		"0.weight": make([]float32, 2*hiddenDim*hiddenDim), "0.bias": make([]float32, hiddenDim),
		"2.weight": make([]float32, hiddenDim*hiddenDim), "2.bias": make([]float32, hiddenDim),
	}}
	e.fusion["0.weight"][0] = 1
	e.fusion["2.weight"][0] = 1
	e.speechEmb[hiddenDim] = 3
	e.speechPos[2*hiddenDim] = .5
	for _, x := range []float32{-1, 1} {
		e.speechEmb[0] = x
		got := e.pairRow(0, 1, 2)[0]
		want := float32(.5)*(x+3) + float32(float64(x)*.5*(1+math.Erf(float64(x)/math.Sqrt2))) + .5
		if math.Abs(float64(got-want)) > 1e-6 {
			t.Fatalf("x%v got%v want%v", x, got, want)
		}
	}
}

func TestFusionCancellationBeforeSecondHead(t *testing.T) {
	e := &Engine{}
	polls := 0
	s := sampler.New(sampler.Config{Temperature: 0, RepetitionPenalty: 1, MaxNewTokens: 2}, 7)
	got, err := e.generatePairs([]float32{10, 0}, nil, kvCache{}, make([]bool, 2), 2, 0, 1, 0, 0, s, func() bool { polls++; return polls == 2 })
	if !errors.Is(err, ErrCancelled) || got != nil || polls != 2 {
		t.Fatalf("got %v, %v after %d polls", got, err, polls)
	}
}

func TestFusionGraphSignatureUsesCheckpointLayerCount(t *testing.T) {
	outputs := prefillOutputs(true, 12)
	if len(outputs) != 26 || outputs[len(outputs)-1] != "kv_v_11" {
		t.Fatalf("prefill outputs: %v", outputs)
	}
	inputs := stepInputs(true, 12)
	if len(inputs) != 27 || inputs[len(inputs)-1] != "past_v_11" {
		t.Fatalf("pair inputs: %v", inputs)
	}
	outputs = stepOutputs(true, 12)
	if len(outputs) != 26 || outputs[len(outputs)-1] != "present_v_11" {
		t.Fatalf("pair outputs: %v", outputs)
	}
}
