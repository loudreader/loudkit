package timestretch

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// Every ramp the fixture pins, the historical 5 ms and the shipped 20 ms, has
// to come out of FadeEdges bit for bit. A cosine computed here cannot: numpy
// takes it in float32, and the ramp a release actually applies is the 20 ms one.
func TestFadeRampMatchesPythonBits(t *testing.T) {
	data, err := os.ReadFile("../../tests/data/conformance/edge_fade.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		SampleRate int `json:"sample_rate"`
		Ramps      []struct {
			Seconds float64  `json:"seconds"`
			Samples int      `json:"samples"`
			Bits    []uint32 `json:"bits"`
		} `json:"ramps"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Ramps) < 2 {
		t.Fatalf("fixture pins %d ramps, want the 5 ms and the 20 ms", len(fixture.Ramps))
	}
	for _, ramp := range fixture.Ramps {
		if len(ramp.Bits) != ramp.Samples || ramp.Samples == 0 {
			t.Fatalf("%vs: %d bits for %d samples", ramp.Seconds, len(ramp.Bits), ramp.Samples)
		}
		input := make([]float32, 4*ramp.Samples)
		for i := range input {
			input[i] = 1
		}
		out := FadeEdges(input, fixture.SampleRate, ramp.Seconds)
		for i, bits := range ramp.Bits {
			if math.Float32bits(out[i]) != bits {
				t.Fatalf("%vs ramp %d: got %08x want %08x",
					ramp.Seconds, i, math.Float32bits(out[i]), bits)
			}
			if math.Float32bits(out[len(out)-1-i]) != bits {
				t.Fatalf("%vs tail %d is not the head reversed", ramp.Seconds, i)
			}
		}
	}
}
