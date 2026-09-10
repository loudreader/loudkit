package engine

import (
	"math"
	"math/rand"
	"testing"
)

// The embedding norms accumulate float64(x) * float64(x) where x is a float32,
// and the compiler fuses the multiply into the add. That is left alone because
// the product is exact: two 24-bit significands multiply to at most 48 bits and
// float64 carries 53, so there is no rounding for the fusion to skip and the
// fused and unfused sums agree bit for bit.
//
// The argument rests entirely on the operands being float32. Widen either one
// and the product starts rounding, the fusion starts mattering, and this port
// starts disagreeing with the other four on a value that conditions the flow
// decoder and gets written into a voice profile. This test fails when that
// happens, at the two shapes that would carry it.
func TestWidenedSquareIsContractionProof(t *testing.T) {
	r := rand.New(rand.NewSource(20250908))

	kinds := []struct {
		name string
		next func() float32
	}{
		{"normal", func() float32 { return float32(r.NormFloat64()) }},
		{"tiny", func() float32 { return float32(r.NormFloat64()) * 1e-20 }},
		{"huge", func() float32 { return float32(r.NormFloat64()) * 1e20 }},
		{"subnormal", func() float32 { return math.Float32frombits(r.Uint32() & 0x007fffff) }},
		{"anybits", func() float32 { return math.Float32frombits(r.Uint32()) }},
	}

	for _, k := range kinds {
		t.Run(k.name, func(t *testing.T) {
			for trial := 0; trial < 2000; trial++ {
				x := make([]float32, 256)
				for i := range x {
					x[i] = k.next()
				}
				fused, split := 0.0, 0.0
				for _, v := range x {
					fused += float64(v) * float64(v)
					split += float64(float64(v) * float64(v))
				}
				if math.IsNaN(fused) && math.IsNaN(split) {
					continue
				}
				if math.Float64bits(fused) != math.Float64bits(split) {
					t.Fatalf("trial %d: fused %#016x, rounded-first %#016x",
						trial, math.Float64bits(fused), math.Float64bits(split))
				}
			}
		})
	}

	// The same claim at the single accumulation, against the correctly rounded
	// fused multiply-add rather than against whatever the compiler chose.
	for i := 0; i < 200000; i++ {
		v := float64(math.Float32frombits(r.Uint32()))
		acc := r.NormFloat64() * math.Pow(2, float64(r.Intn(80)-40))
		if math.IsNaN(v) || math.IsInf(v, 0) {
			continue
		}
		if got, want := math.FMA(v, v, acc), acc+float64(v*v); math.Float64bits(got) != math.Float64bits(want) {
			t.Fatalf("FMA(%g,%g,%g) = %#016x, unfused %#016x",
				v, v, acc, math.Float64bits(got), math.Float64bits(want))
		}
	}
}
