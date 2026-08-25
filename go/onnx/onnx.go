// Package onnx wraps onnxruntime_go for the loudkit graphs. Mirrors the JS
// Session wrapper: the six exported graphs are loaded once and run per stage.
package onnx

import (
	"fmt"
	"path/filepath"

	"github.com/yalue/onnxruntime_go"
)

// Session is one loaded graph, run with dynamic inputs/outputs per call.
type Session struct {
	sess     *onnxruntime_go.DynamicAdvancedSession
	inNames  []string
	outNames []string
}

// Load opens a graph on one concrete execution provider — a name from
// config.ONNXProviders that Resolve has already answered, never "auto".
//
// The provider is a parameter and not a default because nil session options
// are onnxruntime's CPU provider: a session created that way puts a caller on
// a CUDA box on the CPU numbers, under a figure that says otherwise and with
// nothing to say so.
//
// Caller must have called onnx.InitializeEnvironment.
func Load(path string, inputs, outputs []string, provider string) (*Session, error) {
	opts, err := onnxruntime_go.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("session options for %s: %w", path, err)
	}
	// Destroyed here rather than kept: onnxruntime copies the options into the
	// session at creation, and the session outlives them.
	defer opts.Destroy()
	if err := applyProvider(opts, provider, filepath.Base(path)); err != nil {
		return nil, fmt.Errorf("execution provider %s for %s: %w", provider, path, err)
	}
	s, err := onnxruntime_go.NewDynamicAdvancedSession(path, inputs, outputs, opts)
	if err != nil {
		return nil, fmt.Errorf("session %s: %w", path, err)
	}
	return &Session{sess: s, inNames: inputs, outNames: outputs}, nil
}

// Close releases the session.
func (s *Session) Close() {
	if s.sess != nil {
		s.sess.Destroy()
	}
}

// Run runs the graph with the given input tensors. The outputs slice may be
// shorter than the session's output count; remaining outputs are auto-allocated.
// Returns the full outputs slice (with any auto-allocated tensors filled in).
func (s *Session) Run(inputs []onnxruntime_go.Value, outputs []onnxruntime_go.Value) ([]onnxruntime_go.Value, error) {
	if len(outputs) < len(s.outNames) {
		outputs = append(outputs, make([]onnxruntime_go.Value, len(s.outNames)-len(outputs))...)
	}
	if err := s.sess.Run(inputs, outputs); err != nil {
		return nil, err
	}
	return outputs, nil
}

// DataF32 extracts a value's data as []float32. The value must have been
// auto-allocated as float32 (every loudkit graph output is fp32).
func DataF32(v onnxruntime_go.Value) ([]float32, error) {
	t, ok := v.(*onnxruntime_go.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("output is not *Tensor[float32] (got %T)", v)
	}
	return t.GetData(), nil
}

// DataI64 extracts a value's data as []int64. The value must have been
// auto-allocated as int64 (the tokenizer graph's only output).
func DataI64(v onnxruntime_go.Value) ([]int64, error) {
	t, ok := v.(*onnxruntime_go.Tensor[int64])
	if !ok {
		return nil, fmt.Errorf("output is not *Tensor[int64] (got %T)", v)
	}
	return t.GetData(), nil
}

// NewFloat32 builds a float32 input tensor for a graph feed.
func NewFloat32(shape onnxruntime_go.Shape, data []float32) (*onnxruntime_go.Tensor[float32], error) {
	return onnxruntime_go.NewTensor(shape, data)
}

// NewInt64 builds an int64 input tensor for a graph feed.
func NewInt64(shape onnxruntime_go.Shape, data []int64) (*onnxruntime_go.Tensor[int64], error) {
	return onnxruntime_go.NewTensor(shape, data)
}
