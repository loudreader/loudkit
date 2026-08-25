// Package rng implements Philox-4x32-10, a bit-parity port of
// loudkit.rng. All arithmetic is native uint32/uint64, so the bits match the
// Python and JS implementations by construction. The n-th random number is a
// pure function of (seed, stream, step, index).
package rng

import "math"

const (
	m0    = uint32(0xd2511f53)
	m1    = uint32(0xcd9e8d57)
	w0    = uint32(0x9e3779b9)
	w1    = uint32(0xbb67ae85)
	round = 10
)

// Philox4x32 runs ten rounds of Philox-4x32 over one counter quad and returns
// the four uint32 streams.
func Philox4x32(c0, c1, c2, c3, k0, k1 uint32) [4]uint32 {
	x0, x1, x2, x3 := c0, c1, c2, c3
	key0, key1 := k0, k1
	for i := 0; i < round; i++ {
		hi0, lo0 := mulhilo(x0, m0)
		hi1, lo1 := mulhilo(x2, m1)
		x0 = hi1 ^ x1 ^ key0
		x1 = lo1
		x2 = hi0 ^ x3 ^ key1
		x3 = lo0
		key0 += w0
		key1 += w1
	}
	return [4]uint32{x0, x1, x2, x3}
}

// mulhilo computes the 64-bit product of two uint32s and returns (hi, lo).
func mulhilo(a, b uint32) (uint32, uint32) {
	p := uint64(a) * uint64(b)
	return uint32(p >> 32), uint32(p)
}

// Uniforms returns nSteps*width uniforms in the open interval (0,1).
func Uniforms(seed uint64, stream uint32, step0, nSteps, width int) []float64 {
	out := make([]float64, nSteps*width)
	quads := (width + 3) / 4
	for s := 0; s < nSteps; s++ {
		step := uint32(s + step0)
		for q := 0; q < quads; q++ {
			r := Philox4x32(uint32(q), step, stream, 0, uint32(seed), uint32(seed>>32))
			for i := 0; i < 4; i++ {
				idx := s*width + q*4 + i
				if idx < len(out) {
					out[idx] = (float64(r[i]) + 0.5) / 4294967296.0
				}
			}
		}
	}
	return out
}

// GumbelNoise returns -log(-log(u)) for a block of uniforms.
func GumbelNoise(seed uint64, stream uint32, step0, nSteps, width int) []float64 {
	u := Uniforms(seed, stream, step0, nSteps, width)
	out := make([]float64, len(u))
	for i, v := range u {
		out[i] = -math.Log(-math.Log(v))
	}
	return out
}
