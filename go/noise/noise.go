// Package noise mirrors loudkit.models.noise: render randomness as Philox
// data, so the same seed produces the same bytes on every backend.
package noise

import (
	"math"

	"github.com/loudreader/loudkit/go/rng"
)

// GaussianField returns a rows*cols standard-normal field.
func GaussianField(seed uint64, stream uint32, rows, cols int) []float32 {
	u1 := rng.Uniforms(seed, stream, 0, rows, cols)
	u2 := rng.Uniforms(seed, stream+1, 0, rows, cols)
	out := make([]float32, rows*cols)
	for i := range out {
		out[i] = float32(math.Sqrt(-2.0*math.Log(u1[i])) * math.Cos(2.0*math.Pi*u2[i]))
	}
	return out
}

// SymmetricUniforms returns n uniforms in (-halfWidth, halfWidth).
func SymmetricUniforms(seed uint64, stream uint32, n int, halfWidth float64) []float32 {
	u := rng.Uniforms(seed, stream, 0, 1, n)
	out := make([]float32, n)
	for i := range out {
		out[i] = float32((u[i]*2.0 - 1.0) * halfWidth)
	}
	return out
}
