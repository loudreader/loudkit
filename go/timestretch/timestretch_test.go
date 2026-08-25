// WSOLA: the properties, not the samples.
//
// Nothing here pins a waveform, and there is deliberately no shared byte-level
// fixture across the five ports. The alignment search ranks candidates by a
// cross-correlation whose last bit can differ between languages — Go sums a
// frame of products in a plain loop, NumPy sums it pairwise — and one offset
// chosen differently moves every sample after it. A fixture would then fail for
// a reason that is not a defect, which is the worst kind of red test: the one
// people learn to re-record.
//
// So what is asserted is what a listener would notice if this broke: the
// length, the pitch, the loudness, and the fact that speed 1.0 is not a stretch
// of factor one but a bypass. Python, Rust, TypeScript and Swift assert the same
// properties.

package timestretch

import (
	"math"
	"testing"
	"time"
)

const sampleRate = 24_000

var speeds = []float64{0.5, 0.75, 0.9, 1.25, 1.5, 2.0}

// signal is a voiced-ish test signal: a low fundamental, a harmonic, and a
// sweep. Deterministic by construction — no RNG anywhere in this file, because
// the stretcher has none either and a flaky DSP test is worse than no DSP test.
func signal(seconds, f0 float64) []float32 {
	n := int(float64(sampleRate) * seconds)
	out := make([]float32, n)
	for i := range out {
		t := float64(i) / sampleRate
		out[i] = float32(0.5*math.Sin(2*math.Pi*f0*t) +
			0.25*math.Sin(2*math.Pi*2*f0*t) +
			0.15*math.Sin(2*math.Pi*600*t*(1+t)))
	}
	return out
}

// pitchHz estimates the fundamental by autocorrelation. Enough to catch a
// chipmunk, which is the failure this is looking for — a resampler would move
// the fundamental by exactly speed.
func pitchHz(x []float32) float64 {
	const window = 4096
	at := sampleRate / 4
	w := make([]float64, window)
	mean := 0.0
	for i := range w {
		w[i] = float64(x[at+i])
		mean += w[i]
	}
	mean /= window
	for i := range w {
		w[i] -= mean
	}
	lo, hi := sampleRate/500, sampleRate/80
	bestLag, best := lo, math.Inf(-1)
	for lag := lo; lag < hi; lag++ {
		sum := 0.0
		for i := 0; i+lag < window; i++ {
			sum += w[i] * w[i+lag]
		}
		if sum > best {
			best, bestLag = sum, lag
		}
	}
	return sampleRate / float64(bestLag)
}

func rms(x []float32) float64 {
	sum := 0.0
	for _, s := range x {
		sum += float64(s) * float64(s)
	}
	return math.Sqrt(sum / float64(len(x)))
}

// stretch is TimeStretch with the error turned into a failure, because every
// call below passes a speed the range already allows — the refusal has its own
// test, and repeating four lines of error handling around each property would
// bury the property.
func stretch(t *testing.T, audio []float32, rate int, speed float64) []float32 {
	t.Helper()
	got, err := TimeStretch(audio, rate, speed)
	if err != nil {
		t.Fatal(err)
	}
	return got
}

// Identity, not equality. The engine's default must not depend on a DSP path
// being lossless — it must not enter the DSP path at all.
func TestUnitySpeedHandsBackTheSameSlice(t *testing.T) {
	x := signal(0.3, 220)
	got := stretch(t, x, sampleRate, 1.0)
	if len(got) != len(x) || &got[0] != &x[0] {
		t.Fatal("speed 1.0 returned a new slice; it must be an exact bypass")
	}
}

func TestTheOutputIsExactlyAsLongAsAsked(t *testing.T) {
	x := signal(1.0, 220)
	for _, speed := range speeds {
		got := stretch(t, x, sampleRate, speed)
		if want := StretchedLength(len(x), speed); len(got) != want {
			t.Errorf("speed %v gave %d samples, want %d", speed, len(got), want)
		}
	}
}

// Python rounds halves to even; Go, Rust, Swift and JavaScript do not. A
// one-sample disagreement on an exact half is found six months later, in a
// conformance run, by somebody else — so the formula is written as
// floor(n/speed + 0.5) in all five and never as the language's round().
func TestTheLengthFormulaIsHalfUpNotHalfEven(t *testing.T) {
	if got := StretchedLength(5, 2.0); got != 3 {
		t.Errorf("StretchedLength(5, 2.0) = %d, want 3 (2.5 rounds up, not to even)", got)
	}
}

// The guard that no implementation tested, which is how two of the five shipped
// without it.
//
// Below ~60 Hz the derived frame is one sample, so the hop — frame/2 — is zero,
// and the overlap-add loop advances by it. TypeScript and Swift computed the hop
// *after* the degenerate-shape guard and never tested it, so both looped forever
// on an input Go returned from in microseconds; nothing was red, because nothing
// asked. Now all five ask.
func TestASampleRateTooLowToHaveAHopDoesNotHang(t *testing.T) {
	done := make(chan []float32, 1)
	go func() {
		got, err := TimeStretch(make([]float32, 64), 40, 1.5)
		if err != nil {
			t.Error(err)
		}
		done <- got
	}()
	select {
	case got := <-done:
		if want := StretchedLength(64, 1.5); len(got) != want || want != 43 {
			t.Fatalf("got %d samples, want %d", len(got), want)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the overlap-add loop did not terminate")
	}
}

// No overlap to align, so it is cut or padded. At 24 kHz this is under 25 ms —
// below anything the engine renders, and the alternative is an index out of
// range on the degenerate case.
func TestAFragmentShorterThanAFrameIsStillTheRightLength(t *testing.T) {
	tiny := make([]float32, 64)
	for i := range tiny {
		tiny[i] = 1
	}
	if got := stretch(t, tiny, sampleRate, 2.0); len(got) != 32 {
		t.Errorf("half of 64 samples came back as %d", len(got))
	}
	got := stretch(t, tiny, sampleRate, 0.5)
	if len(got) != 128 {
		t.Fatalf("double of 64 samples came back as %d", len(got))
	}
	for i := 64; i < 128; i++ {
		if got[i] != 0 {
			t.Fatalf("the pad at %d is %v, want silence", i, got[i])
		}
	}
}

// The entire point of WSOLA rather than a resampler.
func TestPitchIsUnmoved(t *testing.T) {
	x := signal(2.0, 220)
	want := pitchHz(x)
	for _, speed := range speeds {
		got := pitchHz(stretch(t, x, sampleRate, speed))
		if math.Abs(got-want)/want > 0.03 {
			t.Errorf("speed %v moved the fundamental from %.1f Hz to %.1f Hz", speed, want, got)
		}
	}
}

// Overlap-add with a window that does not sum to one is the classic way to get
// a 6 dB drop or a comb filter. The periodic Hann at 50 % overlap sums to one,
// and the denominator corrects the ends.
func TestLoudnessSurvives(t *testing.T) {
	x := signal(1.0, 220)
	want := rms(x)
	for _, speed := range speeds {
		got := rms(stretch(t, x, sampleRate, speed))
		if math.Abs(got-want)/want > 0.15 {
			t.Errorf("speed %v moved RMS from %.4f to %.4f", speed, want, got)
		}
	}
}

func TestNothingClipsOrGoesNonFinite(t *testing.T) {
	x := signal(1.0, 220)
	for _, speed := range speeds {
		for i, s := range stretch(t, x, sampleRate, speed) {
			if math.IsNaN(float64(s)) || math.IsInf(float64(s), 0) {
				t.Fatalf("speed %v produced %v at sample %d", speed, s, i)
			}
			if math.Abs(float64(s)) > 1.2 {
				t.Fatalf("speed %v produced %v at sample %d, well past the input's range", speed, s, i)
			}
		}
	}
}

// The correlation search divides by the candidate's energy; a silent frame is
// where that division has to not happen.
func TestSilenceStaysSilent(t *testing.T) {
	got := stretch(t, make([]float32, sampleRate), sampleRate, 1.5)
	for i, s := range got {
		if s != 0 {
			t.Fatalf("silence came back as %v at sample %d", s, i)
		}
	}
}

// No RNG, no adaptivity, no wall clock. Same in, same out, forever.
func TestTwoCallsAgreeBitForBit(t *testing.T) {
	x := signal(1.0, 220)
	first := stretch(t, x, sampleRate, 1.4)
	second := stretch(t, x, sampleRate, 1.4)
	for i := range first {
		if first[i] != second[i] {
			t.Fatalf("two runs differ at sample %d: %v vs %v", i, first[i], second[i])
		}
	}
}

// A caller who asked for 3x and silently got 2x has a bug only a stopwatch
// finds.
func TestOutOfRangeIsRefusedNotClamped(t *testing.T) {
	for _, speed := range []float64{0.49, 2.01, 0.0, -1.0, 10.0} {
		if err := ValidateSpeed(speed); err == nil {
			t.Errorf("speed %v was accepted", speed)
		}
	}
}

// NaN compares false against both bounds, so a naive range test would let it
// through and produce a waveform of NaNs.
func TestNaNAndInfinityAreRefusedBeforeTheRangeCheck(t *testing.T) {
	for _, speed := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		if err := ValidateSpeed(speed); err == nil {
			t.Errorf("speed %v was accepted", speed)
		}
	}
}

func TestTheBoundsThemselvesAreAllowed(t *testing.T) {
	for _, speed := range []float64{MinSpeed, 1.0, MaxSpeed} {
		if err := ValidateSpeed(speed); err != nil {
			t.Errorf("speed %v was refused: %v", speed, err)
		}
	}
}

// An out-of-range speed reaching this far is a caller that skipped
// ValidateSpeed, and it is refused rather than silently bypassed: handing back
// the input would be the exact failure MinSpeed refuses by name — asked for 3x,
// silently got 1x, and only a stopwatch finds it. The engine refuses first,
// where the refusal saves six seconds of generation; this is the last line.
func TestAnUnvalidatedSpeedIsRefusedRatherThanBypassed(t *testing.T) {
	x := signal(0.3, 220)
	// TimeStretch directly, not the stretch helper: the helper turns an error
	// into a test failure, which is the wrong shape for the one case where the
	// error is the behaviour under test.
	got, err := TimeStretch(x, sampleRate, 4.0)
	if err == nil {
		t.Fatal("an out-of-range speed was accepted")
	}
	if got != nil {
		t.Fatal("a refusal must not also hand back samples")
	}
}
