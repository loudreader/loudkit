package engine

import "testing"

// A mel is row-major [80, frames]. Appending two flat buffers puts the second
// chunk's bin 0 after the first chunk's bin 79, so every row but the first is
// wrong. The audio is unaffected — each chunk is vocoded on its own — but the
// mel is the diagnostic people reach for when two backends disagree, and a
// mis-shaped one sends them looking in the wrong place.
func TestAppendMelAlongTime(t *testing.T) {
	make2 := func(frames, offset int) []float32 {
		m := make([]float32, melBins*frames)
		for b := 0; b < melBins; b++ {
			for f := 0; f < frames; f++ {
				m[b*frames+f] = float32(b*1000 + offset + f)
			}
		}
		return m
	}
	joined := appendMelAlongTime(appendMelAlongTime(nil, make2(3, 0)), make2(2, 100))
	if len(joined) != melBins*5 {
		t.Fatalf("length %d, want %d", len(joined), melBins*5)
	}
	for b := 0; b < melBins; b++ {
		want := []float32{
			float32(b*1000 + 0), float32(b*1000 + 1), float32(b*1000 + 2),
			float32(b*1000 + 100), float32(b*1000 + 101),
		}
		got := joined[b*5 : (b+1)*5]
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("row %d: got %v, want %v", b, got, want)
			}
		}
	}
}
