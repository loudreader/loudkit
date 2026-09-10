package enroll

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

// The five refusals validate_reference_audio makes on the Python side, run
// here without weights. Each message is the reference's, word for word,
// because a caller reads it and a port is held to it.

const testGoodInput = "A good input is 5 to 10 seconds of one person speaking, clean, " +
	"without music or a second voice."

// speechLike is a deterministic clip loud enough to pass the silence floor.
func speechLike(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(0.5 * math.Sin(2*math.Pi*220.0*float64(i)/melSR))
	}
	return out
}

func refusal(t *testing.T, audio []float32, sampleRate int) string {
	t.Helper()
	err := ValidateReferenceAudio(audio, sampleRate)
	if err == nil {
		t.Fatal("the recording was accepted")
	}
	return err.Error()
}

func TestANonPositiveRateIsRefused(t *testing.T) {
	audio := speechLike(melSR)
	if got, want := refusal(t, audio, 0), "sample rate must be positive, got 0"; got != want {
		t.Errorf("got %q, want %q", got, want)
	}
	if got, want := refusal(t, audio, -24000), "sample rate must be positive, got -24000"; got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestNaNAndInfSamplesAreRefused(t *testing.T) {
	want := "the recording contains NaN or Inf samples, so no voice can be derived " +
		"from it. Re-export the file. " + testGoodInput
	for _, tc := range []struct {
		name  string
		at    int
		value float64
	}{
		{"NaN first", 0, math.NaN()},
		{"NaN last", 2*melSR - 1, math.NaN()},
		{"Inf first", 0, math.Inf(1)},
		{"Inf last", 2*melSR - 1, math.Inf(-1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			audio := speechLike(2 * melSR)
			audio[tc.at] = float32(tc.value)
			if got := refusal(t, audio, melSR); got != want {
				t.Errorf("got %q, want %q", got, want)
			}
		})
	}
}

// 720 samples is one short of the reflect padding matchaMel reads, so without
// this guard it indexes past the end of the slice and panics.
func TestAClipShorterThanASecondIsRefusedBeforeTheReflectPadding(t *testing.T) {
	for _, tc := range []struct {
		samples int
		want    string
	}{
		{720, "the recording is 0.03 s"},
		{0, "the recording is 0.00 s"},
		// Half a second: no panic, but the utterance encoder pads it out to
		// its 1.6 s first partial and enrolls mostly padding.
		{melSR / 2, "the recording is 0.50 s"},
		// One sample under the minimum. The reported seconds round to 1.00 and
		// the refusal still stands: the comparison is on the exact length.
		{melSR - 1, "the recording is 1.00 s"},
	} {
		want := tc.want + ": too short to enroll a speaker from (minimum 1 s). " + testGoodInput
		if got := refusal(t, speechLike(tc.samples), melSR); got != want {
			t.Errorf("%d samples: got %q, want %q", tc.samples, got, want)
		}
	}
}

func TestARecordingLongerThanThirtySecondsIsRefused(t *testing.T) {
	want := "the recording is 31.0 s. Only the first 10 s become the voice prompt, " +
		"and the whole clip shapes the speaker embedding, so a long recording " +
		"enrolls something the prompt does not carry. Trim it to the best 5 to 10 " +
		"seconds (at most 30 s). " + testGoodInput
	if got := refusal(t, speechLike(31*melSR), melSR); got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestASilentRecordingIsRefused(t *testing.T) {
	for _, tc := range []struct {
		name  string
		value float32
		want  string
	}{
		{"digital silence", 0, "0.0e+00"},
		{"under the floor", 5e-5, "5.0e-05"},
		// The floor itself. float32 rounds 1e-4 down, so the loudest sample a
		// float32 clip can hold at this level is still under the float64 floor
		// the reference compares against.
		{"at the floor", 1e-4, "1.0e-04"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			audio := make([]float32, 2*melSR)
			for i := range audio {
				audio[i] = tc.value
			}
			want := "the recording is silent (peak " + tc.want + "); there is no voice " +
				"in it to enroll. " + testGoodInput
			if got := refusal(t, audio, melSR); got != want {
				t.Errorf("got %q, want %q", got, want)
			}
		})
	}
}

// Finiteness is checked before anything arithmetic, because one NaN poisons
// every statistic below it. A clip that breaks two rules names the first.
func TestTheOrderOfTheChecksIsTheReferenceOrder(t *testing.T) {
	nan := "the recording contains NaN or Inf samples, so no voice can be derived " +
		"from it. Re-export the file. " + testGoodInput

	shortAndNaN := speechLike(720)
	shortAndNaN[0] = float32(math.NaN())
	if got := refusal(t, shortAndNaN, melSR); got != nan {
		t.Errorf("short and NaN: got %q, want the NaN refusal", got)
	}

	silentAndNaN := make([]float32, 2*melSR)
	silentAndNaN[7] = float32(math.NaN())
	if got := refusal(t, silentAndNaN, melSR); got != nan {
		t.Errorf("silent and NaN: got %q, want the NaN refusal", got)
	}

	// Length before loudness, as the reference orders them.
	if got := refusal(t, make([]float32, 31*melSR), melSR); got[:23] != "the recording is 31.0 s" {
		t.Errorf("silent and too long: got %q, want the length refusal", got)
	}
}

// The clip every port's enrollment conformance runs on. It needs no graphs to
// be judged, so the guard is held to real reference audio and not only to
// synthetic tones.
func TestTheFixtureClipStillEnrolls(t *testing.T) {
	fixture := os.Getenv("LOUDKIT_ENROLL_FIXTURE")
	if fixture == "" {
		fixture = filepath.Join("..", "..", "tests", "data", "enrollment")
	}
	path := filepath.Join(fixture, "ref_audio.f32")
	if _, err := os.Stat(path); err != nil {
		t.Skip("enrollment fixture not found: " + fixture)
	}
	if err := ValidateReferenceAudio(readF32(t, path), melSR); err != nil {
		t.Fatalf("the fixture clip was refused: %v", err)
	}
}

// The bounds are inclusive at both ends and the fixture clip sits inside them,
// so tightening the guard cannot start refusing what already enrolls.
func TestTheBandItselfIsAccepted(t *testing.T) {
	for _, samples := range []int{melSR, melSR + 1, 15 * melSR, 30 * melSR} {
		audio := make([]float32, samples)
		// One sample just over the float32 floor, and silence everywhere else.
		audio[0] = 1.01e-4
		if err := ValidateReferenceAudio(audio, melSR); err != nil {
			t.Errorf("%d samples at 24 kHz: %v", samples, err)
		}
	}
}
