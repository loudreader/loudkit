package enroll

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestResampleIsBitExactAgainstTheFixture holds the FIR to byte equality with
// the reference, which is what docs/design/models-notes.md promises for it and
// what the ports match each other on.
//
// Byte equality rather than a tolerance, because the failure this guards is one
// float32 ulp wide: a fused multiply-add moves a third of these samples by that
// much, which any threshold loose enough to be called a tolerance admits.
func TestResampleIsBitExactAgainstTheFixture(t *testing.T) {
	fx := fixtureDir(t)
	if _, err := os.Stat(filepath.Join(fx, "ref_audio.f32")); err != nil {
		t.Skip("enrollment fixture not found: " + fx)
	}
	audio := readF32(t, filepath.Join(fx, "ref_audio.f32"))
	full := audio
	capped := full
	if max := int(maxRef * melSR); len(capped) > max {
		capped = capped[:max]
	}

	for _, tc := range []struct {
		name  string
		input []float32
		want  string
	}{
		{"flow", capped, "wav16_flow.f32"},
		{"t3", full, "wav16_t3.f32"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			want := readF32(t, filepath.Join(fx, tc.want))
			got := resample(tc.input, melSR, s3SR)
			if len(got) != len(want) {
				t.Fatalf("length = %d, want %d", len(got), len(want))
			}
			for i := range want {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("sample %d = %.9g (%#08x), want %.9g (%#08x)",
						i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
				}
			}
		})
	}
}

func TestResampleMatchesPythonReference(t *testing.T) {
	// Fixed outputs from loudkit.models.resample.resample, not the Go kernel.
	// Cover both directions and a rate ratio with many distinct phases.
	input := []float32{0, .25, -.5, 1, -.75, .5, .125, -.25}
	cases := []struct {
		name     string
		from, to int
		want     []float32
	}{
		{"down", 24000, 16000, []float32{.0776673555, -.0292089209, .2211000323, -.0460677072, .20756118, -.2059810162}},
		{"up", 16000, 24000, []float32{.0134814419, .3992075324, -.1998347789, -.475638777, .742939353, .5641857982, -.7256599665, -.1171451658, .7229516506, .138287276, -.3024970293, -.1519623548}},
		{"fractional", 44100, 16000, []float32{.0188409779, .0996189341, .0587580577}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := resample(input, tc.from, tc.to)
			if len(got) != len(tc.want) {
				t.Fatalf("length = %d, want %d", len(got), len(tc.want))
			}
			for i, want := range tc.want {
				if math.Abs(float64(got[i]-want)) > 1e-7 {
					t.Fatalf("sample %d = %.9g, want %.9g", i, got[i], want)
				}
			}
		})
	}
}
