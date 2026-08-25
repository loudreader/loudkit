// Package enroll turns reference audio into a voice profile — a bit-parity
// port of loudkit.models.enroll over the exported enrollment ONNX graphs.
//
// The DSP (resampler, filterbanks) is implemented here in Go and held to the
// enrollment fixture; the model stages run through s3_tokenizer.onnx,
// camp.onnx and voice_encoder.onnx.
package enroll

import "math"

// resample downsamples/upsamples a 1-D float32 signal with the one portable
// Hann-windowed-sinc law, a bit-parity port of loudkit.models.resample.
//
// The kernel is computed in float64 and rounded to float32 once; the FIR
// accumulates left to right in float32, never a fused multiply-add. That
// contract is what keeps the five ports bit-identical, and it must not drift.
func resample(waveform []float32, origFreq, newFreq int) []float32 {
	if origFreq == newFreq {
		out := make([]float32, len(waveform))
		copy(out, waveform)
		return out
	}
	g := gcd(origFreq, newFreq)
	orig, new := origFreq/g, newFreq/g

	kernel, width := sincHannKernel(orig, new, 6, 0.99)
	taps := len(kernel[0][0])

	padded := make([]float32, width+len(waveform)+width+orig)
	for i, v := range waveform {
		padded[width+i] = v
	}

	nOut := (len(padded)-taps)/orig + 1
	out := make([]float32, nOut*new)
	for i := 0; i < nOut; i++ {
		base := i * orig
		for phase := 0; phase < new; phase++ {
			var acc float32
			for c := 0; c < taps; c++ {
				acc += kernel[phase][0][c] * padded[base+c]
			}
			out[i*new+phase] = acc
		}
	}

	target := int(math.Ceil(float64(new*len(waveform)) / float64(orig)))
	return out[:target]
}

// sincHannKernel returns the float32 Hann-windowed-sinc kernel and its
// half-width, after GCD reduction. Mirrors loudkit.models.resample and
// torchaudio's sinc_interp_hann.
func sincHannKernel(orig, new, lowpassFilterWidth int, rolloff float64) ([][][]float32, int) {
	base := float64(min(orig, new)) * rolloff
	width := int(math.Ceil(float64(lowpassFilterWidth) * float64(orig) / base))

	kernel := make([][][]float32, new)
	for phase := 0; phase < new; phase++ {
		kernel[phase] = make([][]float32, 1)
		kernel[phase][0] = make([]float32, 2*width+orig)
		for idx := 0; idx < 2*width+orig; idx++ {
			t := float64(-phase)/float64(new) + float64(idx-width)/float64(orig)
			t *= base
			if t < -float64(lowpassFilterWidth) {
				t = -float64(lowpassFilterWidth)
			} else if t > float64(lowpassFilterWidth) {
				t = float64(lowpassFilterWidth)
			}
			window := math.Cos(t * math.Pi / float64(lowpassFilterWidth) / 2)
			window *= window
			tt := t * math.Pi
			var sinc float64
			if tt == 0 {
				sinc = 1
			} else {
				sinc = math.Sin(tt) / tt
			}
			kernel[phase][0][idx] = float32(sinc * window * (base / float64(orig)))
		}
	}
	return kernel, width
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
