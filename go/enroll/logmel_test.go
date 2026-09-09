package enroll

import (
	"math"
	"testing"
)

// The log mel keeps eight decades under whatever the clip's own peak is, so a
// quiet recording is stretched over the same range as a loud one. Seeding the
// peak at zero instead of below every value pinned the ceiling at -8 for any
// clip whose mel energies were all under 1, which is an ordinary quiet
// recording: at a 0.05 amplitude it moved 11915 of the 12800 cells away from
// the reference, and enrolling from such a clip gave a different speaker.
func TestTheCeilingFollowsAQuietPeak(t *testing.T) {
	for _, amp := range []float64{0.5, 0.05} {
		samples := make([]float64, 16000)
		for i := range samples {
			samples[i] = amp * math.Sin(2*math.Pi*200*float64(i)/16000)
		}
		mel, _ := tokenizerMel(samples)
		lo, hi := float32(math.MaxFloat32), float32(-math.MaxFloat32)
		for _, v := range mel {
			lo = min(lo, v)
			hi = max(hi, v)
		}
		// Eight decades, over the transform's own divisor of four.
		if span := float64(hi - lo); math.Abs(span-2) > 1e-4 {
			t.Errorf("amplitude %v spans %v decades of the eight it must span", amp, span*4)
		}
	}
}
