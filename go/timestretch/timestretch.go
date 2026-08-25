// Package timestretch mirrors loudkit.models.timestretch: playing faster
// without talking higher — WSOLA, from first principles.
//
// "Speed" in a reading app means what it means on a video player: 1.5x is the
// same voice, sooner. Resampling gives you a chipmunk; what is wanted is time
// stretched while pitch is left alone.
//
// Why WSOLA and not a phase vocoder. The phase vocoder is the other standard
// answer and is better on sustained, harmonic material — held notes, chords.
// Speech is the opposite kind of signal: it is mostly transients (plosives, the
// attack of every syllable) sitting on a pitch that moves continuously. A phase
// vocoder resynthesises from magnitudes and unwrapped phases, and its
// characteristic failure on that material is transient smearing — a /t/
// arriving as a soft thud, "phasiness" on voiced segments — which is precisely
// the part of speech intelligibility rests on. WSOLA never leaves the time
// domain: it copies real waveform segments and only chooses where to copy them
// from, so a plosive is either included whole or not at all. It cannot smear
// what it never transforms.
//
// The algorithm. Cut the input into overlapping ~25 ms frames. Write them back
// out at a hop that is fixed by the output rate (50 % overlap), and read them in
// at a hop scaled by speed. The read position is not used as computed: it is
// moved by up to ±10 ms to whichever offset best matches what the previously
// written frame would naturally have been followed by. That search is the
// "waveform similarity" in the name, and it keeps successive frames in phase
// with each other, so the overlap-add reinforces
// rather than cancels. A plain OLA without the search is the same code with the
// search window set to zero, and it sounds like it: periodic warble at the
// frame rate.
//
// Everything here is deterministic — no RNG, no adaptivity, no dependencies.
// The constants are derived from the sample rate rather than written as sample
// counts, so the same code is correct at 16 kHz or 48 kHz, and the five
// implementations derive them the same way.
//
// What it costs. At 1.25x this is hard to tell from a native reading. At 2x, or
// at 0.5x, it is audibly processed: the alignment search cannot always find a
// match, and the artefact is a faint roughness or a doubled consonant. That is
// the practical range, and the bounds below are set where the result stops being
// worth offering rather than where the arithmetic stops working.
package timestretch

import (
	"fmt"
	"math"
)

// MinSpeed and MaxSpeed are the range worth offering, not the range that runs.
//
// Outside it the alignment search stops finding matches often enough — the
// required shift exceeds the ±10 ms it may look over — and the output is
// recognisably processed rather than merely faster. Refused rather than
// clamped: a caller who asked for 3x and silently got 2x has a bug that only a
// stopwatch finds.
const (
	MinSpeed = 0.5
	MaxSpeed = 2.0
)

// frameMS is the analysis/synthesis frame: long enough to hold two periods of
// the lowest voiced pitch this is used on (~80 Hz), short enough that a frame is
// inside one phone.
const frameMS = 25.0

// searchMS is how far the read position may move to find a better join — a bit
// under one pitch period at the low end of the voiced range, which is what the
// search is looking for.
const searchMS = 10.0

// hannCOLAHop: frames overlap by half. A periodic Hann window at hop =
// frame/2 sums to exactly one, so the overlap-add needs no normalisation of its
// own — the denominator in TimeStretch only ever corrects the ends and the
// places the alignment search moved a frame off the grid.
const hannCOLAHop = 2

// ValidateSpeed returns nil if speed is usable, or an error that says the range.
//
// Kept here rather than in the engine so that every entry point — Synthesize,
// SynthesizeLong, Stream, and whatever a caller wraps around them — refuses the
// same values with the same words, and a new entry point cannot forget to.
func ValidateSpeed(speed float64) error {
	if math.IsNaN(speed) || math.IsInf(speed, 0) {
		return fmt.Errorf("speed must be a finite number, not %v", speed)
	}
	if speed < MinSpeed || speed > MaxSpeed {
		return fmt.Errorf("speed %g is outside [%g, %g]: beyond that range the "+
			"time-stretch is audibly processed rather than merely faster or "+
			"slower, so it is refused rather than clamped", speed, MinSpeed, MaxSpeed)
	}
	return nil
}

// StretchedLength is how long n samples become at speed.
//
// Written as floor(n/speed + 0.5) rather than with math.Round on purpose:
// Python rounds halves to even, Go, Rust, Swift and JavaScript do not, and a
// one-sample disagreement between ports on an exact half is the kind of thing
// that is found six months later in a conformance run. The literal form is what
// keeps the five agreeing.
func StretchedLength(n int, speed float64) int {
	return int(math.Floor(float64(n)/speed + 0.5))
}

// TimeStretch returns audio played at speed, same pitch.
//
// speed > 1 shortens, < 1 lengthens, and 1.0 returns the input slice itself —
// not a copy that happens to be equal, because the engine's default must be a
// bypass and "bit-identical" is easier to trust when there is no arithmetic to
// be identical about. Go has no default arguments, so every caller names the
// 1.0 that Python omits; naming it costs nothing.
//
// sampleRate is not decorative: the frame, hop and search window are derived
// from it.
//
// The result is StretchedLength(len(audio), speed) samples long.
//
// An out-of-range speed is an error, not a quiet bypass. The three engine entry
// points already refuse one before they generate a single token — where the
// refusal is worth six seconds — so this is the last line rather than the first,
// and reaching it means a caller skipped ValidateSpeed. Returning the input
// instead would be the exact failure MinSpeed refuses by name: asked for 3x,
// silently got 1x, and only a stopwatch finds it. Python raises here, Rust
// panics, Swift and TypeScript throw; an error is how Go says the same thing.
func TimeStretch(audio []float32, sampleRate int, speed float64) ([]float32, error) {
	if err := ValidateSpeed(speed); err != nil {
		return nil, err
	}
	if speed == 1.0 {
		return audio, nil
	}

	n := len(audio)
	outLen := StretchedLength(n, speed)
	frame := int(math.Floor(float64(sampleRate)*frameMS/1000.0 + 0.5))
	hop := frame / hannCOLAHop
	// Nothing to overlap-add: a fragment shorter than one frame has no second
	// frame to align against. Cut or zero-padded to the right length instead,
	// which is wrong in the way silence is wrong rather than in the way a pitch
	// shift is. At 24 kHz a frame is 600 samples — a fortieth of a second, below
	// anything the engine renders.
	//
	// hop <= 0 joins that branch rather than looping forever. It needs a sample
	// rate under 60 Hz to happen, so it is not a behaviour difference from the
	// reference in any case a caller can reach — it turns a hang, which no stack
	// trace explains, into the short-fragment path.
	if n <= frame || outLen <= 0 || hop <= 0 {
		out := make([]float32, max(outLen, 0))
		copy(out, audio[:min(max(outLen, 0), n)])
		return out, nil
	}

	search := int(math.Floor(float64(sampleRate)*searchMS/1000.0 + 0.5))
	// Periodic Hann, i.e. 2*pi*i/frame and not /(frame-1). The periodic form is
	// the one that sums to exactly one at 50 % overlap; the symmetric form is off
	// by a hair at every frame boundary, which reads as a low-level buzz at the
	// frame rate — 40 Hz here, right in the range a listener notices.
	window := make([]float64, frame)
	for i := range window {
		window[i] = 0.5 - 0.5*math.Cos(2.0*math.Pi*float64(i)/float64(frame))
	}

	// Every intermediate is float64. The input is float32 and so is the output,
	// but the accumulator sums up to two windowed frames per sample and the
	// correlation sums a whole frame of products; doing either in float32 loses
	// bits the other four ports keep.
	x := make([]float64, n)
	for i, s := range audio {
		x[i] = float64(s)
	}
	// Room for the last frame to be written whole; trimmed at the end.
	acc := make([]float64, outLen+frame)
	weight := make([]float64, outLen+frame)

	lastFrameAt, writeAt, k := 0, 0, 0
	for writeAt < outLen {
		ideal := int(math.Floor(float64(k)*float64(hop)*speed + 0.5))
		readAt := 0
		if k > 0 {
			// What the previous frame would naturally have been followed by. The
			// search asks which nearby segment continues this, not which one the
			// arithmetic pointed at.
			readAt = bestMatch(x, slice(x, lastFrameAt+hop, lastFrameAt+hop+frame), ideal, search, frame)
		}
		readAt = clamp(readAt, 0, n-frame)

		segment := x[readAt : readAt+frame]
		for i := 0; i < frame; i++ {
			acc[writeAt+i] += window[i] * segment[i]
			weight[writeAt+i] += window[i]
		}

		lastFrameAt = readAt
		if n >= frame+hop {
			lastFrameAt = min(readAt, n-frame-hop)
		}
		writeAt += hop
		k++
	}

	// The Hann pair sums to one in the interior, so this division is the
	// identity almost everywhere; it earns its place at the two ends, where only
	// one frame contributes and the raw sum would fade in and out.
	out := make([]float32, outLen)
	for i := 0; i < outLen; i++ {
		if weight[i] > 1e-12 {
			out[i] = float32(acc[i] / weight[i])
		}
	}
	return out, nil
}

// bestMatch is the offset within ±search of ideal whose frame best continues
// target.
//
// Scored by cross-correlation normalised by the candidate's energy only — the
// target's is the same for every candidate and cancels out of the ranking.
// Without that normalisation the search prefers whichever candidate is loudest
// rather than whichever fits, which at a syllable onset is exactly the wrong
// one.
//
// Ties go to the lower offset (the comparison is strictly greater), so the
// choice does not depend on iteration order and the five ports agree.
func bestMatch(x, target []float64, ideal, search, frame int) int {
	n := len(x)
	lo := max(0, ideal-search)
	hi := min(n-frame, ideal+search)
	if hi < lo || len(target) < frame {
		return clamp(ideal, 0, n-frame)
	}

	bestAt := lo
	bestScore := math.Inf(-1)
	for at := lo; at <= hi; at++ {
		candidate := x[at : at+frame]
		energy, correlation := 0.0, 0.0
		for i := 0; i < frame; i++ {
			energy += candidate[i] * candidate[i]
			correlation += candidate[i] * target[i]
		}
		// A silent candidate scores zero rather than dividing by nothing.
		score := 0.0
		if energy > 0.0 {
			score = correlation / math.Sqrt(energy)
		}
		if score > bestScore {
			bestScore = score
			bestAt = at
		}
	}
	return bestAt
}

// slice is x[lo:hi] with the ends taken as far as they go, the way the
// reference's Python slicing already behaves: past the end is short, entirely
// past it is empty. A short target is what tells bestMatch there is nothing left
// to align against.
func slice(x []float64, lo, hi int) []float64 {
	lo = clamp(lo, 0, len(x))
	hi = clamp(hi, lo, len(x))
	return x[lo:hi]
}

func clamp(v, lo, hi int) int { return min(max(v, lo), hi) }
