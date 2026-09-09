package enroll

import (
	"fmt"
	"math"
	"path/filepath"

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
	// flowMelBins is the 24 kHz conditioning mel, the same 80 voice.melBins
	// names and a different 80 from kaldiBins.
	flowMelBins = 80
)

// ------------------------------------------------------- reference recording

const (
	// minEnrollSeconds is the shortest clip a speaker can be estimated from.
	// The utterance encoder's first partial alone covers 1.6 s and is
	// zero-padded under it, so a sub-second clip enrolls mostly padding.
	minEnrollSeconds = 1.0

	// maxEnrollSeconds is the longest clip the input contract stays honest
	// over. The prompt uses the first 10 s and the speaker embedding reads the
	// whole clip, so a five-minute recording produces a voice mostly shaped by
	// audio the docs say is ignored. Refused rather than truncated: the user
	// picked that recording for a reason, and silently using a different slice
	// of it is worse than asking them to choose.
	maxEnrollSeconds = 30.0

	// silencePeak is the level under which a clip's loudest sample makes it
	// silence at any playback level; there is no voice in it to enroll.
	silencePeak = 1e-4

	goodInput = "A good input is 5 to 10 seconds of one person speaking, clean, " +
		"without music or a second voice."
)

// ValidateReferenceAudio refuses a recording the enrollment contract cannot
// honour.
//
// The whole preflight, before any DSP runs, with the same five sentences as
// loudkit.models.enrollment_audio.validate_reference_audio. Without it a clip
// shorter than 721 samples indexes past the end of the reflect padding in
// matchaMel and panics, a sub-second clip enrolls padding, and NaN samples
// poison every statistic downstream.
//
// See docs/design/models-notes.md.
func ValidateReferenceAudio(audio []float32, sampleRate int) error {
	// A non-positive rate reaches the resampler as a division by zero. Callers
	// and tests in every port match on this one message, so it stays word for
	// word.
	if sampleRate <= 0 {
		return fmt.Errorf("sample rate must be positive, got %d", sampleRate)
	}
	// Finiteness before anything arithmetic: one NaN poisons every statistic
	// below and every tensor downstream.
	for _, v := range audio {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("the recording contains NaN or Inf samples, so no voice "+
				"can be derived from it. Re-export the file. %s", goodInput)
		}
	}
	seconds := float64(len(audio)) / float64(sampleRate)
	if seconds < minEnrollSeconds {
		return fmt.Errorf("the recording is %.2f s: too short to enroll a speaker from "+
			"(minimum %g s). %s", seconds, float64(minEnrollSeconds), goodInput)
	}
	if seconds > maxEnrollSeconds {
		return fmt.Errorf("the recording is %.1f s. Only the first 10 s become the voice "+
			"prompt, and the whole clip shapes the speaker embedding, so a long recording "+
			"enrolls something the prompt does not carry. Trim it to the best 5 to 10 "+
			"seconds (at most %g s). %s", seconds, float64(maxEnrollSeconds), goodInput)
	}
	// The peak is taken in float64 off float32 samples, as the reference takes
	// it, so the floor is the same number on both sides of the comparison.
	peak := 0.0
	for _, v := range audio {
		if a := math.Abs(float64(v)); a > peak {
			peak = a
		}
	}
	if peak < silencePeak {
		return fmt.Errorf("the recording is silent (peak %.1e); there is no voice in it "+
			"to enroll. %s", peak, goodInput)
	}
	return nil
}

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
	// The graphs opened so far are closed unless all three arrive, as the
	// engine's loader does: without this a second graph that fails to load
	// leaves the first one open, and EnrollPCM opens an enroller per call.
	var opened []*onnx.Session
	complete := false
	defer func() {
		if !complete {
			for _, session := range opened {
				session.Close()
			}
		}
	}()
	load := func(name string, inputs, outputs []string) (*onnx.Session, error) {
		session, err := onnx.Load(filepath.Join(onnxDir, name), inputs, outputs, provider)
		if err == nil {
			opened = append(opened, session)
		}
		return session, err
	}
	tok, err := load("s3_tokenizer.onnx", []string{"mel"}, []string{"tokens"})
	if err != nil {
		return nil, err
	}
	camp, err := load("camp.onnx", []string{"fbank"}, []string{"out"})
	if err != nil {
		return nil, err
	}
	ve, err := load("voice_encoder.onnx", []string{"partials"}, []string{"out"})
	if err != nil {
		return nil, err
	}
	complete = true
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
	if err := ValidateReferenceAudio(audio, sampleRate); err != nil {
		return nil, err
	}

	var wav24Full []float64
	if sampleRate == melSR {
		wav24Full = toFloat64(audio)
	} else {
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
	promptMelFrames := len(promptMel) / flowMelBins

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
	promptMel = promptMel[:flowMelBins*(2*nTok)]
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

// destroyAll releases a slice of runtime-allocated tensors. Session.Run
// hands ownership of its outputs to the caller, and nothing in this package
// took it: every enrollment leaked three graphs' worth of native values. The
// engine states the same rule at Engine.condRow and fixes it the same way.
func destroyAll(vs []onnxruntime_go.Value) {
	for _, v := range vs {
		if v != nil {
			v.Destroy()
		}
	}
}

func (e *Enroller) tokenize(mel []float32) ([]int64, error) {
	t, err := onnx.NewFloat32(onnxruntime_go.Shape{1, tokenizerMelBins, int64(len(mel) / tokenizerMelBins)}, mel)
	if err != nil {
		return nil, err
	}
	defer t.Destroy()
	outs, err := e.tokenizer.Run([]onnxruntime_go.Value{t}, nil)
	if err != nil {
		return nil, err
	}
	defer destroyAll(outs)
	data, err := onnx.DataI64(outs[0])
	if err != nil {
		return nil, err
	}
	// Copied, not returned as is: DataI64 is a view over the output tensor's
	// buffer, and the destroy above hands that buffer back. The copy must land
	// with the destroy or the two together are a read of released memory.
	//
	// Sized exactly rather than grown, because the result is handed out on a
	// voice profile: a caller appending to a slice with slack writes into the
	// profile's own array, and two such appends alias each other.
	out := make([]int64, len(data))
	copy(out, data)
	return out, nil
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
	defer t.Destroy()
	outs, err := e.camp.Run([]onnxruntime_go.Value{t}, nil)
	if err != nil {
		return nil, err
	}
	defer destroyAll(outs)
	data, err := onnx.DataF32(outs[0])
	if err != nil {
		return nil, err
	}
	// The copy is what lets the destroy above be correct: this embedding is
	// stored on the voice profile and outlives the tensor it is read from.
	// Sized exactly, for the reason `tokenize` gives.
	out := make([]float32, len(data))
	copy(out, data)
	return out, nil
}

func (e *Enroller) speakerEmbedding(wav16T3 []float64) ([]float32, error) {
	trimmed := trim(wav16T3)
	mel, frames := voiceEncoderMel(trimmed)

	// partial windowing, matching _VoiceEncoder.embed
	nWins := 0
	rem := 0
	if span := frames - partialFrames + partialStep; span > 0 {
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
	defer t.Destroy()
	outs, err := e.ve.Run([]onnxruntime_go.Value{t}, nil)
	if err != nil {
		return nil, err
	}
	defer destroyAll(outs)
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
	// Squaring a widened float32 is exact in float64, so the contraction the
	// compiler applies here is not observable. engine.DecodeMel states the count
	// of bits that makes it so, and TestWidenedSquareIsContractionProof holds
	// the operands to the float32 the argument depends on.
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

// Profile wraps a result in a voice.Profile. language is the language of the
// recording, and what the voice reads in by default; "" means "en".
func (r *Result) Profile(name string, sourceSampleRate int, language string) *voice.Profile {
	if language == "" {
		language = "en"
	}
	return &voice.Profile{
		Name:             name,
		SpeakerEmbedding: r.SpeakerEmbedding,
		FlowEmbedding:    r.FlowEmbedding,
		PromptTokens:     r.PromptTokens,
		PromptMel:        r.PromptMel,
		CondPromptTokens: r.CondPromptTokens,
		SourceSampleRate: sourceSampleRate,
		Language:         language,
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
