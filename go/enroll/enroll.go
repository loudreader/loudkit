package enroll

import (
	"fmt"
	"math"

	"github.com/yalue/onnxruntime_go"

	"github.com/loudreader/loudkit/go/config"
	"github.com/loudreader/loudkit/go/onnx"
	"github.com/loudreader/loudkit/go/voice"
)

const (
	melSR   = 24000
	s3SR    = 16000
	maxRef  = 10.0
	condSec = 6.0

	// The voice encoder's 1.6 s partial windows.
	partialFrames = 160
	partialStep   = 77 // round((s3SR / 1.3) / partialFrames)

	tokenizerMelBins = 128
	kaldiBins        = 80
	veMelBins        = 40
)

// Enroller turns a recording into a voice profile, running the enrollment
// ONNX graphs over the portable DSP here.
type Enroller struct {
	tokenizer *onnx.Session
	camp      *onnx.Session
	ve        *onnx.Session
}

// LoadEnroller opens the three enrollment graphs on the default execution
// config. The onnxruntime shared library must be initialised first.
func LoadEnroller(onnxDir string) (*Enroller, error) {
	return LoadEnrollerWith(onnxDir, config.DefaultExecution())
}

// LoadEnrollerWith is LoadEnroller with the execution config named.
//
// Enrollment honours the same setting as synthesis rather than pinning itself
// to CPU: it is the same onnxruntime and the same shared library, and a
// caller who moved the engine to a device did not ask for one graph in three
// to stay behind.
func LoadEnrollerWith(onnxDir string, execution config.ExecutionConfig) (*Enroller, error) {
	provider, err := onnx.Resolve(execution.RequestedProvider())
	if err != nil {
		return nil, err
	}
	tok, err := onnx.Load(onnxDir+"/s3_tokenizer.onnx", []string{"mel"}, []string{"tokens"}, provider)
	if err != nil {
		return nil, err
	}
	camp, err := onnx.Load(onnxDir+"/camp.onnx", []string{"fbank"}, []string{"out"}, provider)
	if err != nil {
		return nil, err
	}
	ve, err := onnx.Load(onnxDir+"/voice_encoder.onnx", []string{"partials"}, []string{"out"}, provider)
	if err != nil {
		return nil, err
	}
	return &Enroller{tokenizer: tok, camp: camp, ve: ve}, nil
}

// Close releases the three sessions.
func (e *Enroller) Close() {
	e.tokenizer.Close()
	e.camp.Close()
	e.ve.Close()
}

// Result is an enrolled voice before it is wrapped in a Profile.
type Result struct {
	SpeakerEmbedding []float32
	FlowEmbedding    []float32
	PromptTokens     []int64
	PromptMel        []float32
	PromptMelFrames  int
	CondPromptTokens []int64
}

// Enroll derives a voice from up to ten seconds of reference audio. The clip
// is used at 24 kHz (prompt mel) and 16 kHz (tokens and both encoders), all
// through the one portable resampler.
func (e *Enroller) Enroll(audio []float32, sampleRate int) (*Result, error) {
	if sampleRate <= 0 {
		return nil, fmt.Errorf("sample rate must be positive, got %d", sampleRate)
	}

	wav := make([]float64, len(audio))
	for i, v := range audio {
		wav[i] = float64(v)
	}

	wav24Full := wav
	if sampleRate != melSR {
		wav24Full = toFloat64(resample(audio, sampleRate, melSR))
	}
	maxSamples := int(maxRef * melSR)
	wav24 := wav24Full
	if len(wav24) > maxSamples {
		wav24 = wav24[:maxSamples]
	}

	wav16Flow := toFloat64(resample(toFloat32(wav24), melSR, s3SR))
	wav16T3 := toFloat64(resample(toFloat32(wav24Full), melSR, s3SR))

	// prompt mel, 24 kHz
	promptMel := matchaMel(wav24)
	promptMelFrames := len(promptMel) / 80

	// prompt tokens
	tokMel, _ := tokenizerMel(wav16Flow)
	tokens, err := e.tokenize(tokMel)
	if err != nil {
		return nil, err
	}
	nTok := len(tokens)
	if mf := promptMelFrames / 2; nTok > mf {
		nTok = mf
	}
	promptTokens := tokens[:nTok]
	promptMel = promptMel[:80*(2*nTok)]
	promptMelFrames = 2 * nTok

	// conditioning tokens: the librosa-rate clip, truncated to 6 s, capped at 150
	condSamples := int(condSec * s3SR)
	if condSamples > len(wav16T3) {
		condSamples = len(wav16T3)
	}
	condMel, _ := tokenizerMel(wav16T3[:condSamples])
	condTokens, err := e.tokenizeCapped(condMel, 150)
	if err != nil {
		return nil, err
	}

	// flow embedding (CAM++)
	fbank := kaldiFbank(wav16Flow)
	flowEmb, err := e.camEmbedding(fbank)
	if err != nil {
		return nil, err
	}

	// speaker embedding (utterance voice encoder)
	speakerEmb, err := e.speakerEmbedding(wav16T3)
	if err != nil {
		return nil, err
	}

	return &Result{
		SpeakerEmbedding: speakerEmb,
		FlowEmbedding:    flowEmb,
		PromptTokens:     promptTokens,
		PromptMel:        promptMel,
		PromptMelFrames:  promptMelFrames,
		CondPromptTokens: condTokens,
	}, nil
}

func (e *Enroller) tokenize(mel []float32) ([]int64, error) {
	t, err := onnx.NewFloat32(onnxruntime_go.Shape{1, tokenizerMelBins, int64(len(mel) / tokenizerMelBins)}, mel)
	if err != nil {
		return nil, err
	}
	outs, err := e.tokenizer.Run([]onnxruntime_go.Value{t}, nil)
	if err != nil {
		return nil, err
	}
	data, err := onnx.DataI64(outs[0])
	if err != nil {
		return nil, err
	}
	return data, nil
}

// tokenizeCapped mirrors tokenize(max_tokens=N): the mel is truncated to N*4
// frames before the graph.
func (e *Enroller) tokenizeCapped(mel []float32, cap int) ([]int64, error) {
	if len(mel)/tokenizerMelBins > cap*4 {
		mel = mel[:tokenizerMelBins*(cap*4)]
	}
	return e.tokenize(mel)
}

func (e *Enroller) camEmbedding(fbank []float32) ([]float32, error) {
	frames := len(fbank) / kaldiBins
	// transpose [frame][bin] to [bin][frame] for the graph
	transposed := make([]float32, len(fbank))
	for f := 0; f < frames; f++ {
		for b := 0; b < kaldiBins; b++ {
			transposed[b*frames+f] = fbank[f*kaldiBins+b]
		}
	}
	t, err := onnx.NewFloat32(onnxruntime_go.Shape{1, kaldiBins, int64(frames)}, transposed)
	if err != nil {
		return nil, err
	}
	outs, err := e.camp.Run([]onnxruntime_go.Value{t}, nil)
	if err != nil {
		return nil, err
	}
	return onnx.DataF32(outs[0])
}

func (e *Enroller) speakerEmbedding(wav16T3 []float64) ([]float32, error) {
	trimmed := trim(wav16T3)
	mel, frames := voiceEncoderMel(trimmed)

	// partial windowing, matching _VoiceEncoder.embed
	nWins := 0
	rem := 0
	if span := len(mel)/veMelBins - partialFrames + partialStep; span > 0 {
		nWins, rem = span/partialStep, span%partialStep
	}
	if nWins == 0 || float64(rem+(partialFrames-partialStep))/partialFrames >= 0.8 {
		nWins++
	}
	target := partialFrames + partialStep*(nWins-1)
	if target > frames {
		padded := make([]float32, target*veMelBins)
		copy(padded, mel)
		mel = padded
	}

	partials := make([]float32, nWins*partialFrames*veMelBins)
	for i := 0; i < nWins; i++ {
		start := i * partialStep * veMelBins
		copy(partials[i*partialFrames*veMelBins:], mel[start:start+partialFrames*veMelBins])
	}

	t, err := onnx.NewFloat32(onnxruntime_go.Shape{int64(nWins), partialFrames, veMelBins}, partials)
	if err != nil {
		return nil, err
	}
	outs, err := e.ve.Run([]onnxruntime_go.Value{t}, nil)
	if err != nil {
		return nil, err
	}
	perPartial, err := onnx.DataF32(outs[0]) // [nWins, 256]
	if err != nil {
		return nil, err
	}

	// mean-pool the per-partial embeddings and L2-normalise
	pooled := make([]float32, 256)
	for i := 0; i < nWins; i++ {
		for d := 0; d < 256; d++ {
			pooled[d] += perPartial[i*256+d]
		}
	}
	var norm float64
	for _, v := range pooled {
		norm += float64(v) * float64(v)
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		for i := range pooled {
			pooled[i] = float32(float64(pooled[i]) / norm)
		}
	}
	return pooled, nil
}

// Profile wraps a result in a voice.Profile.
func (r *Result) Profile(name string, sourceSampleRate int) *voice.Profile {
	return &voice.Profile{
		Name:             name,
		SpeakerEmbedding: r.SpeakerEmbedding,
		FlowEmbedding:    r.FlowEmbedding,
		PromptTokens:     r.PromptTokens,
		PromptMel:        r.PromptMel,
		CondPromptTokens: r.CondPromptTokens,
		SourceSampleRate: sourceSampleRate,
		Language:         "en",
	}
}

func toFloat64(x []float32) []float64 {
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = float64(v)
	}
	return out
}

func toFloat32(x []float64) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = float32(v)
	}
	return out
}
