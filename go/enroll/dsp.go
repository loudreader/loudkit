package enroll

import (
	_ "embed"
	"math"
	"sync"
)

// The enrollment filterbanks. Each is a bit-parity port of the corresponding
// Python reference (models/enroll.py) or, for the Kaldi path, of the Swift
// EnrollmentDSP — all four held to the enrollment fixture through the exported
// ONNX graphs. The mel filters and windows are shipped as float32 data (see
// tools/gen_dsp_assets.py) so every port multiplies the same tables.

//go:embed data/s3_mel_filters.f32
var s3MelFilters []byte

//go:embed data/s3_hann400.f32
var s3Hann400 []byte

//go:embed data/matcha_mel_filters.f32
var matchaMelFilters []byte

//go:embed data/matcha_hann1920.f32
var matchaHann1920 []byte

//go:embed data/voiceenc_mel_filters.f32
var voiceencMelFilters []byte

//go:embed data/voiceenc_hann400.f32
var voiceencHann400 []byte

//go:embed data/kaldi_mel_filters.f32
var kaldiMelFilters []byte

//go:embed data/kaldi_povey400.f32
var kaldiPovey400 []byte

const (
	frameLen400 = 400
	hopLen160   = 160
	kaldiFFT    = 512
	matchaNFFT  = 1920
	matchaHop   = 480
)

// f32 decodes a raw little-endian float32 table (the native endianness here).
func f32(b []byte) []float32 {
	out := make([]float32, len(b)/4)
	for i := range out {
		u := uint32(b[4*i]) | uint32(b[4*i+1])<<8 | uint32(b[4*i+2])<<16 | uint32(b[4*i+3])<<24
		out[i] = math.Float32frombits(u)
	}
	return out
}

var (
	once           sync.Once
	s3mel          []float32
	s3hann         []float32
	matchaMelTable []float32
	matchaHann     []float32
	veMel          []float32
	veHann         []float32
	kaldiMel       []float32
	kaldiPovey     []float32
)

func loadTables() {
	once.Do(func() {
		s3mel = f32(s3MelFilters)
		s3hann = f32(s3Hann400)
		matchaMelTable = f32(matchaMelFilters)
		matchaHann = f32(matchaHann1920)
		veMel = f32(voiceencMelFilters)
		veHann = f32(voiceencHann400)
		kaldiMel = f32(kaldiMelFilters)
		kaldiPovey = f32(kaldiPovey400)
	})
}

// dftBasis is a cached cosine/sine transform matrix for one length, so the
// 400- and 1920-point transforms (neither a power of two) are computed once.
// Direct DFT rather than FFT: enrollment runs once per voice, correctness over
// speed, and the lengths are not radix-2.
type dftBasis struct {
	cos [][]float64
	sin [][]float64
}

var dftCache = map[int]*dftBasis{}
var dftMu sync.Mutex

func basis(nfft int) *dftBasis {
	dftMu.Lock()
	defer dftMu.Unlock()
	if b, ok := dftCache[nfft]; ok {
		return b
	}
	bins := nfft/2 + 1
	b := &dftBasis{
		cos: make([][]float64, bins),
		sin: make([][]float64, bins),
	}
	for k := 0; k < bins; k++ {
		b.cos[k] = make([]float64, nfft)
		b.sin[k] = make([]float64, nfft)
		for n := 0; n < nfft; n++ {
			a := -2 * math.Pi * float64(k) * float64(n) / float64(nfft)
			b.cos[k][n] = math.Cos(a)
			b.sin[k][n] = math.Sin(a)
		}
	}
	dftCache[nfft] = b
	return b
}

// powerSpectrum returns |X[k]|^2 for k in 0..nfft/2 of a real frame, computed
// in float64 and returned as float64. The frame is the windowed (or raw) input
// of length nfft.
func powerSpectrum(frame []float64, nfft int) []float64 {
	b := basis(nfft)
	bins := nfft/2 + 1
	out := make([]float64, bins)
	for k := 0; k < bins; k++ {
		var re, im float64
		for n := 0; n < nfft; n++ {
			re += b.cos[k][n] * frame[n]
			im += b.sin[k][n] * frame[n]
		}
		out[k] = re*re + im*im
	}
	return out
}

// magnitudeSpectrum returns sqrt(|X|^2 + 1e-9), the matcha mel's measure.
func magnitudeSpectrum(frame []float64, nfft int) []float64 {
	p := powerSpectrum(frame, nfft)
	for k := range p {
		p[k] = math.Sqrt(p[k] + 1e-9)
	}
	return p
}

// centredPowerSpectra is torch.stft(center=True, pad reflect) as a matrix of
// power spectra: rows are bins, columns are frames. dropLast mirrors the S3
// tokenizer's stft[..., :-1].
func centredPowerSpectra(samples []float64, window []float32, dropLast bool) [][]float64 {
	nfft := len(window)
	half := nfft / 2
	padded := make([]float64, len(samples)+nfft)
	for i := 0; i < half; i++ {
		padded[i] = samples[half-i] // reflect
	}
	for i, v := range samples {
		padded[half+i] = v
	}
	for i := 0; i < half; i++ {
		padded[half+len(samples)+i] = samples[len(samples)-2-i]
	}

	frames := len(samples)/hopLen160 + 1
	if dropLast {
		frames--
	}
	bins := nfft/2 + 1
	out := make([][]float64, bins)
	for k := range out {
		out[k] = make([]float64, frames)
	}
	for f := 0; f < frames; f++ {
		start := f * hopLen160
		frame := make([]float64, nfft)
		for i := 0; i < nfft; i++ {
			frame[i] = padded[start+i] * float64(window[i])
		}
		sp := powerSpectrum(frame, nfft)
		for k := 0; k < bins; k++ {
			out[k][f] = sp[k]
		}
	}
	return out
}

// melMultiply is filters [rows, bins] @ spectra [bins, frames] in float32.
func melMultiply(filters []float32, rows, bins int, spectra [][]float64, frames int) []float32 {
	out := make([]float32, rows*frames)
	for r := 0; r < rows; r++ {
		for f := 0; f < frames; f++ {
			var acc float32
			for b := 0; b < bins; b++ {
				acc += filters[r*bins+b] * float32(spectra[b][f])
			}
			out[r*frames+f] = acc
		}
	}
	return out
}

// tokenizerMel is _S3Tokenizer._log_mel: 128-bin log mel, [bin][frame], log10,
// eight decades of headroom, shifted into [0, 1].
func tokenizerMel(samples []float64) ([]float32, int) {
	loadTables()
	spectra := centredPowerSpectra(samples, s3hann, true)
	frames := len(spectra[0])
	mel := melMultiply(s3mel, 128, 201, spectra, frames)

	var peak float32
	for _, v := range mel {
		if v < 1e-10 {
			v = 1e-10
		}
		v = float32(math.Log10(float64(v)))
		if v > peak {
			peak = v
		}
	}
	ceiling := peak - 8
	for i, v := range mel {
		if v < 1e-10 {
			v = 1e-10
		}
		v = float32(math.Log10(float64(v)))
		if v < ceiling {
			v = ceiling
		}
		mel[i] = (v + 4) * 0.25
	}
	return mel, frames
}

// matchaMel is the 24 kHz flow conditioning mel: [80, frames], 1920-point
// STFT, Slaney mels, log with a 1e-5 clamp.
func matchaMel(samples []float64) []float32 {
	loadTables()
	pad := (matchaNFFT - matchaHop) / 2
	padded := make([]float64, len(samples)+2*pad)
	for i := 0; i < pad; i++ {
		padded[i] = samples[pad-i]
	}
	for i, v := range samples {
		padded[pad+i] = v
	}
	for i := 0; i < pad; i++ {
		padded[pad+len(samples)+i] = samples[len(samples)-2-i]
	}

	frames := (len(padded)-matchaNFFT)/matchaHop + 1
	bins := matchaNFFT/2 + 1
	spectra := make([][]float64, bins)
	for k := range spectra {
		spectra[k] = make([]float64, frames)
	}
	for f := 0; f < frames; f++ {
		start := f * matchaHop
		frame := make([]float64, matchaNFFT)
		for i := 0; i < matchaNFFT; i++ {
			frame[i] = padded[start+i] * float64(matchaHann[i])
		}
		mg := magnitudeSpectrum(frame, matchaNFFT)
		for k := 0; k < bins; k++ {
			spectra[k][f] = mg[k]
		}
	}

	mel := melMultiply(matchaMelTable, 80, bins, spectra, frames)
	for i, v := range mel {
		if v < 1e-5 {
			v = 1e-5
		}
		mel[i] = float32(math.Log(float64(v)))
	}
	return mel
}

// kaldiFbank is torchaudio.compliance.kaldi.fbank with the loudkit defaults:
// DC removal, 0.97 pre-emphasis, Povey window, 512-point power spectrum,
// Kaldi mels, natural log, and the per-bin mean removed.
func kaldiFbank(samples []float64) []float32 {
	loadTables()
	frames := (len(samples)-frameLen400)/hopLen160 + 1
	bins := kaldiFFT/2 + 1
	spectra := make([][]float64, bins)
	for k := range spectra {
		spectra[k] = make([]float64, frames)
	}

	for f := 0; f < frames; f++ {
		start := f * hopLen160
		frame := make([]float64, kaldiFFT)
		var mean float64
		for i := 0; i < frameLen400; i++ {
			frame[i] = samples[start+i]
			mean += frame[i]
		}
		mean /= frameLen400
		for i := 0; i < frameLen400; i++ {
			frame[i] -= mean
		}
		// pre-emphasis, walked backwards so each sample sees its untouched
		// predecessor; the first stands in for its own.
		prev := frame[0]
		for i := frameLen400 - 1; i >= 1; i-- {
			frame[i] -= 0.97 * frame[i-1]
		}
		frame[0] -= 0.97 * prev

		for i := 0; i < frameLen400; i++ {
			frame[i] *= float64(kaldiPovey[i])
		}
		sp := powerSpectrum(frame, kaldiFFT)
		for k := 0; k < bins; k++ {
			spectra[k][f] = sp[k]
		}
	}

	mel := melMultiply(kaldiMel, 80, 256, spectra, frames)
	epsilon := float32(1.1920928955078125e-07)
	for i, v := range mel {
		if v < epsilon {
			v = epsilon
		}
		mel[i] = float32(math.Log(float64(v)))
	}
	// per-bin mean removal (the CAM++ input subtracts the utterance mean).
	for b := 0; b < 80; b++ {
		var m float64
		for f := 0; f < frames; f++ {
			m += float64(mel[b*frames+f])
		}
		m /= float64(frames)
		for f := 0; f < frames; f++ {
			mel[b*frames+f] -= float32(m)
		}
	}
	// torchaudio returns [frame][bin]; the fixture and the graph feed both
	// want that orientation.
	out := make([]float32, len(mel))
	for f := 0; f < frames; f++ {
		for b := 0; b < 80; b++ {
			out[f*80+b] = mel[b*frames+f]
		}
	}
	return out
}

// voiceEncoderMel is the 40-bin power mel the utterance voice encoder reads,
// [frame][bin], computed on librosa's symmetric hann.
func voiceEncoderMel(samples []float64) ([]float32, int) {
	loadTables()
	spectra := centredPowerSpectra(samples, veHann, false)
	frames := len(spectra[0])
	// [bin][frame] from the shared helper; transpose to [frame][bin].
	binMajor := melMultiply(veMel, 40, 201, spectra, frames)
	out := make([]float32, frames*40)
	for f := 0; f < frames; f++ {
		for b := 0; b < 40; b++ {
			out[f*40+b] = binMajor[b*frames+f]
		}
	}
	return out, frames
}

// trim is librosa.effects.trim(top_db=20) with the default reference (np.max):
// frame RMS with center=True reflection padding, a threshold 20 dB below the
// peak RMS, and the sample span from the first to the last frame above it.
func trim(samples []float64) []float64 {
	const frameLen = 2048
	const hop = 512
	half := frameLen / 2

	padded := make([]float64, len(samples)+frameLen)
	for i := 0; i < half; i++ {
		padded[i] = samples[half-i]
	}
	for i, v := range samples {
		padded[half+i] = v
	}
	for i := 0; i < half; i++ {
		padded[half+len(samples)+i] = samples[len(samples)-2-i]
	}

	nFrames := 1 + len(samples)/hop
	rms := make([]float64, nFrames)
	var peak float64
	for f := 0; f < nFrames; f++ {
		start := f * hop
		var sum float64
		for i := start; i < start+frameLen; i++ {
			sum += padded[i] * padded[i]
		}
		r := math.Sqrt(sum / frameLen)
		rms[f] = r
		if r > peak {
			peak = r
		}
	}

	first, last := -1, -1
	for f, r := range rms {
		// 20*log10(r/peak) > -20  <=>  r > 0.1*peak
		if r > 0.1*peak {
			if first == -1 {
				first = f
			}
			last = f
		}
	}
	if first == -1 {
		return samples
	}
	start := first * hop
	end := last*hop + hop
	if end > len(samples) {
		end = len(samples)
	}
	if end <= start {
		return samples
	}
	return samples[start:end]
}
