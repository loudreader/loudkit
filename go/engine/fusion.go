package engine

import (
	"fmt"
	"math"

	"github.com/loudreader/loudkit/go/onnx"
	"github.com/loudreader/loudkit/go/sampler"
	ort "github.com/yalue/onnxruntime_go"
)

// pairRow builds one prefix slot with the checkpoint's two-layer fusion MLP.
//
// Every accumulation here rounds the product before adding it. The row feeds
// the transformer's prefix, so a fused rounding does not stay a last-bit
// difference: it moves a logit, and a logit that crosses its neighbour changes
// a token and with it the whole utterance. Exact tokens are the promise the
// ports make to each other, so the arithmetic that decides them is written out
// rather than left to the compiler.
func (e *Engine) pairRow(first, second, position int) []float32 {
	a := e.speechEmb[first*hiddenDim : (first+1)*hiddenDim]
	b := e.speechEmb[second*hiddenDim : (second+1)*hiddenDim]
	input := append(append(make([]float32, 0, 2*hiddenDim), a...), b...)
	w0, b0 := e.fusion["0.weight"], e.fusion["0.bias"]
	w2, b2 := e.fusion["2.weight"], e.fusion["2.bias"]
	middle := make([]float32, hiddenDim)
	for i := range middle {
		var sum float32
		for j, x := range input {
			sum += float32(x * w0[i*2*hiddenDim+j])
		}
		sum += b0[i]
		z := float32(math.Erf(float64(sum / float32(math.Sqrt2))))
		middle[i] = float32(.5) * sum * (1 + z)
	}
	result := make([]float32, hiddenDim)
	for i := range result {
		var sum float32
		for j, x := range middle {
			sum += float32(x * w2[i*hiddenDim+j])
		}
		result[i] = float32(float32(.5)*(a[i]+b[i])) + (sum + b2[i]) + e.speechPos[position*hiddenDim+i]
	}
	return result
}

// secondLogits runs the second head over the pair's hidden state, which is the
// half of fusion_mtp2 the transformer does not produce itself.
func (e *Engine) secondLogits(hidden []float32, first int) ([]float32, error) {
	h, err := onnx.NewFloat32(ort.Shape{1, hiddenDim}, hidden)
	if err != nil {
		return nil, err
	}
	defer h.Destroy()
	token, err := onnx.NewInt64(ort.Shape{1}, []int64{int64(first)})
	if err != nil {
		return nil, err
	}
	defer token.Destroy()
	outputs, err := e.head2.Run([]ort.Value{h, token}, nil)
	if err != nil {
		return nil, err
	}
	defer destroyAll(outputs)
	values, err := onnx.DataF32(outputs[0])
	return append([]float32(nil), values...), err
}

// pairStep decodes one two-token step: the fused row in, the first head's
// logits, the hidden state the second head needs, and the KV cache out.
func (e *Engine) pairStep(first, second, pair, prefixLen, prefillLen int, kv kvCache) ([]float32, []float32, kvCache, error) {
	inputs := []ort.Value{}
	defer func() { destroyAll(inputs) }()
	for i, values := range [][]int64{{int64(first), int64(second)}, {int64(prefixLen/2 + pair + 1)}, {int64(prefillLen + pair)}} {
		shape := ort.Shape{1}
		if i == 0 {
			shape = ort.Shape{1, 2}
		}
		value, err := onnx.NewInt64(shape, values)
		if err != nil {
			return nil, nil, kv, err
		}
		inputs = append(inputs, value)
	}
	for i := 0; i < len(kv.k); i++ {
		for _, data := range [][]float32{kv.k[i], kv.v[i]} {
			value, err := onnx.NewFloat32(ort.Shape{1, kvHeads, int64(len(data) / (kvHeads * headDim)), headDim}, data)
			if err != nil {
				return nil, nil, kv, err
			}
			inputs = append(inputs, value)
		}
	}
	outputs, err := e.step.Run(inputs, nil)
	if err != nil {
		return nil, nil, kv, err
	}
	defer destroyAll(outputs)
	logits, err := onnx.DataF32(outputs[0])
	if err != nil {
		return nil, nil, kv, err
	}
	hidden, err := onnx.DataF32(outputs[1])
	if err != nil {
		return nil, nil, kv, err
	}
	present, err := collectKV(outputs[2:])
	return append([]float32(nil), logits...), append([]float32(nil), hidden...), present, err
}

// generatePairs is Generate's loop for fusion_mtp2: two tokens per step, the
// second drawn from secondLogits, and the same cap, floor and stop token as
// the single-token loop.
func (e *Engine) generatePairs(logits, hidden []float32, kv kvCache, seen []bool, cap, floor, stop, prefixLen, prefillLen int, s *sampler.Sampler, cancel func() bool) ([]int, error) {
	out := []int{}
	for len(out) < cap {
		if cancel != nil && cancel() {
			return nil, fmt.Errorf("at decode step %d: %w", len(out), ErrCancelled)
		}
		if len(out) < floor {
			logits[stop] = float32(math.Inf(-1))
		}
		first := s.Call(logits, len(out), seen)
		out = append(out, first)
		if first == stop || len(out) >= cap {
			break
		}
		seen[first] = true
		if cancel != nil && cancel() {
			return nil, fmt.Errorf("at decode step %d: %w", len(out), ErrCancelled)
		}
		secondRow, err := e.secondLogits(hidden, first)
		if err != nil {
			return nil, err
		}
		if len(out) < floor {
			secondRow[stop] = float32(math.Inf(-1))
		}
		second := s.Call(secondRow, len(out), seen)
		out = append(out, second)
		if second == stop || len(out) >= cap {
			break
		}
		seen[second] = true
		logits, hidden, kv, err = e.pairStep(first, second, len(out)/2-1, prefixLen, prefillLen, kv)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}
