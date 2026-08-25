// Package engine is the full synthesis pipeline over the exported ONNX
// graphs, fp32, no torch — a bit-parity port of loudkit.backends.onnx_backend
// and the JS engine. Same text, voice and seed give the same tokens and the
// same render band as the Python engine.
package engine

import (
	"errors"
	"fmt"
	"math"
	"path/filepath"

	"github.com/yalue/onnxruntime_go"

	"github.com/loudreader/loudkit/go/checkpoint"
	"github.com/loudreader/loudkit/go/chunking"
	"github.com/loudreader/loudkit/go/config"
	"github.com/loudreader/loudkit/go/frontend"
	"github.com/loudreader/loudkit/go/noise"
	"github.com/loudreader/loudkit/go/onnx"
	"github.com/loudreader/loudkit/go/postprocess"
	"github.com/loudreader/loudkit/go/sampler"
	"github.com/loudreader/loudkit/go/speechtext"
	"github.com/loudreader/loudkit/go/timestretch"
	"github.com/loudreader/loudkit/go/timing"
	"github.com/loudreader/loudkit/go/voice"
	"github.com/loudreader/loudkit/go/windowing"
)

const (
	melBins          = 80
	nHarmonics       = 9
	upsamplePerFrame = 480
	nLayers          = 16
	kvHeads          = 4
	headDim          = 64
	hiddenDim        = 1024
)

// Engine is a loaded engine: six ONNX graphs plus the checkpoint's embedding
// tables and the text frontend.
type Engine struct {
	config config.AlgorithmConfig
	// provider is the concrete execution provider all six sessions were opened
	// on — the answer to the ExecutionConfig's question, resolved once so the
	// six cannot land on two different devices.
	provider  string
	frontend  *frontend.Frontend
	textEmb   []float32
	speechEmb []float32
	textPos   []float32
	speechPos []float32
	spkWeight []float32
	spkBias   []float32
	cond      *onnx.Session
	prefill   *onnx.Session
	step      *onnx.Session
	encoder   *onnx.Session
	estimator *onnx.Session
	vocoder   *onnx.Session
}

// embeddingFits refuses an embedding table a live id can index past the end of.
//
// flat is the table flattened row-major at hiddenDim per row, maxID the largest
// id the engine will ever look up in it. The message names the file the caller
// can change, not the index that would otherwise have blown up.
func embeddingFits(which string, maxID, flat int, source string) error {
	rows := flat / hiddenDim
	if maxID >= rows {
		return fmt.Errorf("%s: %s token id %d is past the end of the checkpoint's %s embedding table (%d rows)",
			source, which, maxID, which, rows)
	}
	return nil
}

// Load builds an engine from a checkpoint, an onnx graph dir, and a
// tokenizer.json, on the default execution config. The onnxruntime shared
// library must be initialised first.
func Load(ckptPath, onnxDir, tokenizerPath string) (*Engine, error) {
	return LoadWith(ckptPath, onnxDir, tokenizerPath, config.DefaultExecution())
}

// LoadWith is Load with the execution config named.
//
// Separate from Load rather than a parameter on it because the two questions
// are separate: everything Load already took decides what the engine says, and
// this decides only how fast it says it. Nothing here reaches the fingerprint.
//
// The provider is resolved before the checkpoint is opened, so a misspelled or
// missing provider costs a message rather than the seconds it takes to read a
// couple of gigabytes off disk.
func LoadWith(ckptPath, onnxDir, tokenizerPath string, execution config.ExecutionConfig) (*Engine, error) {
	provider, err := onnx.Resolve(execution.RequestedProvider())
	if err != nil {
		return nil, err
	}
	ckpt, err := checkpoint.Open(ckptPath)
	if err != nil {
		return nil, err
	}
	textEmb, speechEmb, textPos, speechPos, err := ckpt.GeneratorTables()
	if err != nil {
		return nil, err
	}
	spkW, spkB, err := ckpt.SpeakerAffine()
	if err != nil {
		return nil, err
	}
	fe, err := frontend.Load(tokenizerPath)
	if err != nil {
		return nil, err
	}
	// Checked once, at the door, rather than per utterance — and before the
	// graphs load. A chunking recipe with no character budget makes SplitText
	// cut nothing and loop forever; Python has refused it since d8742aa and
	// this port reads the same manifest key.
	algorithm, err := ckpt.Algorithm()
	if err != nil {
		return nil, err
	}
	if err := algorithm.Chunking.Validate(); err != nil {
		return nil, err
	}
	// The tokenizer and the checkpoint are separate files a caller can pair by
	// hand — LOUDKIT_TOKENIZER exists precisely so they can. A vocabulary wider
	// than the checkpoint's table makes textRow read past the end of it, which
	// is an out-of-range panic several seconds into a synthesis rather than a
	// refusal naming the file that is wrong. The same reasoning as
	// loudkit.models.generator.check_manifest_sizes, one layer out: this port
	// reads the table itself and can measure it.
	if err := embeddingFits("text", fe.MaxTokenID(), len(textEmb), tokenizerPath); err != nil {
		return nil, err
	}
	// Same read, one table over: speechRow is indexed by the manifest's own
	// start/stop ids and by sampler draws below SpeechVocabSize, so the manifest
	// can outrun its own weights.
	maxSpeech := algorithm.StartSpeech
	if algorithm.StopSpeech > maxSpeech {
		maxSpeech = algorithm.StopSpeech
	}
	if algorithm.SpeechVocabSize-1 > maxSpeech {
		maxSpeech = algorithm.SpeechVocabSize - 1
	}
	if err := embeddingFits("speech", maxSpeech, len(speechEmb), ckptPath); err != nil {
		return nil, err
	}

	cond, err := onnx.Load(filepath.Join(onnxDir, "t3_cond.onnx"),
		[]string{"speaker_emb", "prompt_tokens", "emotion"}, []string{"t3_cond_out"}, provider)
	if err != nil {
		return nil, err
	}
	prefill, err := onnx.Load(filepath.Join(onnxDir, "t3_prefill.onnx"),
		[]string{"embeds", "positions"}, prefillOutputs(), provider)
	if err != nil {
		cond.Close()
		return nil, err
	}
	step, err := onnx.Load(filepath.Join(onnxDir, "t3_step.onnx"),
		stepInputs(), stepOutputs(), provider)
	if err != nil {
		cond.Close()
		prefill.Close()
		return nil, err
	}
	encoder, err := onnx.Load(filepath.Join(onnxDir, "flow_encoder.onnx"),
		[]string{"prompt_token", "speech_tokens"}, []string{"flow_encoder_out"}, provider)
	if err != nil {
		cond.Close()
		prefill.Close()
		step.Close()
		return nil, err
	}
	estimator, err := onnx.Load(filepath.Join(onnxDir, "flow_estimator.onnx"),
		[]string{"x", "mu", "t", "spks", "cond"}, []string{"flow_estimator_out"}, provider)
	if err != nil {
		cond.Close()
		prefill.Close()
		step.Close()
		encoder.Close()
		return nil, err
	}
	vocoder, err := onnx.Load(filepath.Join(onnxDir, "vocoder.onnx"),
		[]string{"mel", "phase", "noise"}, []string{"vocoder_out"}, provider)
	if err != nil {
		cond.Close()
		prefill.Close()
		step.Close()
		encoder.Close()
		estimator.Close()
		return nil, err
	}

	return &Engine{
		config:    algorithm,
		provider:  provider,
		frontend:  fe,
		textEmb:   textEmb,
		speechEmb: speechEmb,
		textPos:   textPos,
		speechPos: speechPos,
		spkWeight: spkW,
		spkBias:   spkB,
		cond:      cond,
		prefill:   prefill,
		step:      step,
		encoder:   encoder,
		estimator: estimator,
		vocoder:   vocoder,
	}, nil
}

// Close releases all sessions.
func (e *Engine) Close() {
	for _, s := range []*onnx.Session{e.cond, e.prefill, e.step, e.encoder, e.estimator, e.vocoder} {
		s.Close()
	}
}

// Config exposes the resolved algorithm.
func (e *Engine) Config() config.AlgorithmConfig { return e.config }

// Provider is the execution provider the six graphs are running on: one of
// cpu, cuda, coreml, directml, never "auto". A caller who asked for auto reads
// the answer here.
func (e *Engine) Provider() string { return e.provider }

// Describe is the one-line run summary — what this engine computes, and what
// it is computing it on. Print it beside a benchmark number and paste it into
// a bug report: without the provider, a row of timings does not say what
// hardware produced them, and two rows that differ by 8x look like a defect.
//
// Joined with " | ", which is Python's separator (engine.py Engine.describe)
// and Rust's. The halves answer different questions and the bar is where a
// reader and a log scraper both split them.
func (e *Engine) Describe() string {
	return config.Describe(e.config) + " | " + config.DescribeExecution(e.provider)
}

// Fingerprint is this engine's algorithm fingerprint, comparable with the
// Python and Swift ones. Two engines whose fingerprints differ are computing
// different things, whatever their audio sounds like.
func (e *Engine) Fingerprint() string { return config.Fingerprint(e.config) }

// Encode normalises and tokenises text, through the speech funnel the shipped
// Swift/Python engines run before tokenising (SpeechText.prepared), Polish
// English-respelling included; see speechtext.
func (e *Engine) Encode(text, language string) ([]int, error) {
	return e.frontend.Encode(speechtext.Prepared(text, language), language)
}

// fallbackLanguage is what a synthesis reads as when neither the caller nor the
// voice says.
//
// Reached less often than it looks: voice.Load defaults a *missing* header key
// to "en", and Python writes the key, so an empty Language only
// arrives from a Profile built in memory or a header hand-edited to "". A
// profile file with no language field inherits nothing — it loads as "en".
const fallbackLanguage = "en"

// resolveLanguage is the language chain: the argument, then the voice's
// recorded language, then English.
//
// Without the voice link, Synthesize("Cześć", polishVoice, seed, "", nil) runs
// Polish text through the English frontend — English number words, English
// abbreviation expansion, no Polish respelling — and says so nowhere. A profile
// records the language of the audio it was enrolled from, so the voice is the
// better answer than a constant.
//
// Passing a language explicitly is how cross-lingual synthesis is requested: an
// English voice reading Polish text is language "pl", and the argument always
// wins over the profile.
//
// The empty string is this port's "absent", as it already was here and as
// nil-slice and nil-func are elsewhere in the package. Python distinguishes an
// explicit "" from an omitted argument and Go cannot; an explicit "" therefore
// reaches the voice's language rather than tagging the text "[]", which is the
// better of the two behaviours available.
//
// Exported as ResolveLanguage below for the CLI, which has no -language flag
// and would otherwise keep its own copy of the chain — the Rust port made the
// same call for the same reason.
func resolveLanguage(language string, v *voice.Profile) string {
	if language != "" {
		return language
	}
	if v != nil && v.Language != "" {
		return v.Language
	}
	return fallbackLanguage
}

// ResolveLanguage is resolveLanguage for callers outside this package.
//
// Encode takes no voice and so cannot run the chain itself, which left the CLI
// passing v.Language raw: a profile whose header language is blank made Go emit
// a "[]" tag no other port emits, because every other port routes that same
// case through the fallback.
func ResolveLanguage(language string, v *voice.Profile) string {
	return resolveLanguage(language, v)
}

// ------------------------------------------------------------ generator

// f32 is a shortcut for onnx.DataF32 that returns the data or propagates err.
func (e *Engine) f32(v onnxruntime_go.Value, err error) ([]float32, error) {
	if err != nil {
		return nil, err
	}
	return onnx.DataF32(v)
}

func (e *Engine) condRow(v *voice.Profile) ([]float32, error) {
	speaker := make([]float32, len(v.SpeakerEmbedding))
	copy(speaker, v.SpeakerEmbedding)
	prompt := make([]int64, len(v.CondPromptTokens))
	copy(prompt, v.CondPromptTokens)
	// dead axis on these weights; fed the training constant (see voice.EmotionNeutral)
	emotion := []float32{float32(voice.EmotionNeutral)}

	spT, err := onnx.NewFloat32(onnxruntime_go.Shape{1, 256}, speaker)
	if err != nil {
		return nil, err
	}
	defer spT.Destroy()
	prT, err := onnx.NewInt64(onnxruntime_go.Shape{1, int64(len(prompt))}, prompt)
	if err != nil {
		return nil, err
	}
	defer prT.Destroy()
	emT, err := onnx.NewFloat32(onnxruntime_go.Shape{1, 1}, emotion)
	if err != nil {
		return nil, err
	}
	defer emT.Destroy()

	outs, err := e.cond.Run([]onnxruntime_go.Value{spT, prT, emT}, nil)
	if err != nil {
		return nil, err
	}
	// The wrapper hands ownership of a Run's outputs to the caller. Only
	// decodeStep destroyed them, so every other path leaked native memory on
	// each render — invisible in a CLI that exits, unbounded in a server.
	defer destroyAll(outs)
	return e.f32(outs[0], nil)
}

// prefillEmbeds builds the [cond | text | bos | prefix] embedding row.
func (e *Engine) prefillEmbeds(textTokens []int, v *voice.Profile, prefix []int) ([]float32, int, error) {
	cond, err := e.condRow(v)
	if err != nil {
		return nil, 0, err
	}
	text := e.textRow(textTokens)
	bos := e.speechRow(e.config.StartSpeech, 0)

	rows := [][]float32{cond, text, bos}
	prefixLen := 0
	if len(prefix) > 0 {
		prefixLen = len(prefix)
		pe := make([]float32, prefixLen*hiddenDim)
		for i, tok := range prefix {
			sbase := tok * hiddenDim
			pbase := (i + 1) * hiddenDim
			for j := 0; j < hiddenDim; j++ {
				pe[i*hiddenDim+j] = e.speechEmb[sbase+j] + e.speechPos[pbase+j]
			}
		}
		rows = append(rows, pe)
	}
	total := 0
	for _, r := range rows {
		total += len(r)
	}
	out := make([]float32, total)
	off := 0
	for _, r := range rows {
		copy(out[off:], r)
		off += len(r)
	}
	return out, off / hiddenDim, nil
}

func (e *Engine) textRow(textTokens []int) []float32 {
	framed := append([]int{windowing.StartTextToken}, textTokens...)
	framed = append(framed, windowing.StopTextToken)
	out := make([]float32, len(framed)*hiddenDim)
	for i, id := range framed {
		base := id * hiddenDim
		for j := 0; j < hiddenDim; j++ {
			out[i*hiddenDim+j] = e.textEmb[base+j] + e.textPos[i*hiddenDim+j]
		}
	}
	return out
}

// speechPosition is the learned speech positional-embedding row for the
// step-th generated token.
//
// prefillEmbeds writes BOS at row 0 and the prefix at rows 1..len(prefix), so
// generation continues at len(prefix)+1. step+1 re-requests rows that same
// prefill just wrote and never reaches the rows above the prefix: on the
// measured two-chunk input it costs six tokens of divergence from Python
// (386 against 392). The RoPE positions fed to the transformer are a separate
// sequence and stay contiguous either way.
func speechPosition(prefixLen, step int) int {
	return prefixLen + step + 1
}

func (e *Engine) speechRow(token, position int) []float32 {
	out := make([]float32, hiddenDim)
	sbase := token * hiddenDim
	pbase := position * hiddenDim
	for j := 0; j < hiddenDim; j++ {
		out[j] = e.speechEmb[sbase+j] + e.speechPos[pbase+j]
	}
	return out
}

type kvCache struct {
	k, v [][]float32
}

// Generate runs the autoregressive loop to the stop token or cap.
//
// prefix holds speech tokens from the preceding chunk: fed in as context and
// NOT returned. prefillEmbeds accepts it, and a caller that passes
// nil restarts its pitch contour at every
// chunk boundary — the audible stutter the prefix exists to remove (~74 Hz at
// the join against ~7 Hz with a 6-token prefix, measured on the reference
// voice). They also seed the repetition-penalty state, since a token repeated
// across a join is as repeated as one within a chunk.
func (e *Engine) Generate(textTokens []int, v *voice.Profile, s *sampler.Sampler, maxNewTokens *int, shouldCancel func() bool, prefix []int) ([]int, error) {
	cap_ := e.config.Sampling.MaxNewTokens
	if maxNewTokens != nil {
		cap_ = *maxNewTokens
	}
	floor := config.EosFloor(len(textTokens), e.config)
	stop := e.config.StopSpeech

	embeds, prefillLen, err := e.prefillEmbeds(textTokens, v, prefix)
	if err != nil {
		return nil, err
	}
	positions := make([]int64, prefillLen)
	for i := range positions {
		positions[i] = int64(i)
	}

	embT, err := onnxruntime_go.NewTensor(onnxruntime_go.Shape{1, int64(prefillLen), hiddenDim}, embeds)
	if err != nil {
		return nil, err
	}
	defer embT.Destroy()
	posT, err := onnxruntime_go.NewTensor(onnxruntime_go.Shape{int64(prefillLen)}, positions)
	if err != nil {
		return nil, err
	}
	defer posT.Destroy()

	prefillOuts, err := e.prefill.Run([]onnxruntime_go.Value{embT, posT}, nil)
	if err != nil {
		return nil, err
	}
	defer destroyAll(prefillOuts)
	logitsData, err := onnx.DataF32(prefillOuts[0])
	if err != nil {
		return nil, err
	}
	logitsLast := append([]float32(nil), logitsData[(prefillLen-1)*e.config.SpeechVocabSize:]...)
	kv, err := collectKV(prefillOuts[1:])
	if err != nil {
		return nil, err
	}

	seen := make([]bool, e.config.SpeechVocabSize)
	// A token carried across a join is as seen as one this chunk emitted;
	// without this the penalty restarts blind at every chunk boundary.
	for _, t := range prefix {
		seen[t] = true
	}
	out := []int{}
	for step := 0; step < cap_; step++ {
		if shouldCancel != nil && shouldCancel() {
			break // token-level barge-in, mirroring the Python engine
		}
		row := append([]float32(nil), logitsLast...)
		if len(out) < floor {
			row[stop] = float32(math.Inf(-1))
		}
		token := s.Call(row, step, seen)
		out = append(out, token)
		if token == stop {
			break
		}
		seen[token] = true

		logitsLast, kv, err = e.decodeStep(token, step, len(prefix), prefillLen, kv)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

// decodeStep runs one autoregressive step: embed the sampled token, feed it
// and the KV cache through the step graph, and return the next logits plus
// the grown cache.
//
// Its own function rather than the body of Generate's loop because every
// tensor here needs destroying per step. Inside the loop, `defer` would queue
// 16 layers x up to 255 steps of closures that only fire when Generate
// returns — every KV tensor for every step alive at once, memory growing
// quadratically with the token count.
func (e *Engine) decodeStep(token, step, prefixLen, prefillLen int, kv kvCache) ([]float32, kvCache, error) {
	emb := e.speechRow(token, speechPosition(prefixLen, step))
	pos := []int64{int64(prefillLen + step)}
	embT, err := onnx.NewFloat32(onnxruntime_go.Shape{1, 1, hiddenDim}, emb)
	if err != nil {
		return nil, kv, err
	}
	defer embT.Destroy()
	posT, err := onnx.NewInt64(onnxruntime_go.Shape{1}, pos)
	if err != nil {
		return nil, kv, err
	}
	defer posT.Destroy()

	inputs := []onnxruntime_go.Value{embT, posT}
	// Destroys every KV input built below, whichever way this function exits.
	defer func() {
		for _, t := range inputs[2:] {
			t.Destroy()
		}
	}()
	for i := 0; i < nLayers; i++ {
		kt, err := onnx.NewFloat32(onnxruntime_go.Shape{1, kvHeads, int64(len(kv.k[i]) / (kvHeads * headDim)), headDim}, kv.k[i])
		if err != nil {
			return nil, kv, err
		}
		inputs = append(inputs, kt)
		vt, err := onnx.NewFloat32(onnxruntime_go.Shape{1, kvHeads, int64(len(kv.v[i]) / (kvHeads * headDim)), headDim}, kv.v[i])
		if err != nil {
			return nil, kv, err
		}
		inputs = append(inputs, vt)
	}

	stepOuts, err := e.step.Run(inputs, nil)
	if err != nil {
		return nil, kv, err
	}
	// Run auto-allocates its outputs; DataF32 aliases their memory, so every
	// read below copies before this fires.
	defer destroyAll(stepOuts)

	sl, err := onnx.DataF32(stepOuts[0])
	if err != nil {
		return nil, kv, err
	}
	logits := append([]float32(nil), sl...)
	next, err := collectKV(stepOuts[1:])
	if err != nil {
		return nil, kv, err
	}
	return logits, next, nil
}

// destroyAll releases a slice of runtime-allocated tensors. Values returned
// by Session.Run are owned by the caller; without this they leak per step.
func destroyAll(vs []onnxruntime_go.Value) {
	for _, v := range vs {
		if v != nil {
			v.Destroy()
		}
	}
}

func prefillOutputs() []string {
	outs := []string{"logits"}
	for i := 0; i < nLayers; i++ {
		outs = append(outs, fmt.Sprintf("kv_k_%d", i), fmt.Sprintf("kv_v_%d", i))
	}
	return outs
}

func stepInputs() []string {
	ins := []string{"embeds", "position"}
	for i := 0; i < nLayers; i++ {
		ins = append(ins, fmt.Sprintf("past_k_%d", i), fmt.Sprintf("past_v_%d", i))
	}
	return ins
}

func stepOutputs() []string {
	outs := []string{"logits"}
	for i := 0; i < nLayers; i++ {
		outs = append(outs, fmt.Sprintf("present_k_%d", i), fmt.Sprintf("present_v_%d", i))
	}
	return outs
}

func collectKV(outs []onnxruntime_go.Value) (kvCache, error) {
	var kv kvCache
	for i := 0; i < nLayers; i++ {
		k, err := onnx.DataF32(outs[i*2])
		if err != nil {
			return kv, err
		}
		v, err := onnx.DataF32(outs[i*2+1])
		if err != nil {
			return kv, err
		}
		kv.k = append(kv.k, append([]float32(nil), k...))
		kv.v = append(kv.v, append([]float32(nil), v...))
	}
	return kv, nil
}

// -------------------------------------------------------------- renderer

// DecodeMel renders tokens to a mel via the exported encoder + estimator.
func (e *Engine) DecodeMel(tokens []int, v *voice.Profile, seed uint64) ([]float32, error) {
	framed, err := windowing.FrameWindows(e.config, tokens, v)
	if err != nil {
		return nil, err
	}
	pLen := *e.config.Window.StaticPromptTokens
	prompt := framed.Row[:pLen]
	query := framed.Row[pLen:]
	tMel := 2 * len(framed.Row)

	pT, err := onnx.NewInt64(onnxruntime_go.Shape{1, int64(pLen)}, prompt)
	if err != nil {
		return nil, err
	}
	defer pT.Destroy()
	qT, err := onnx.NewInt64(onnxruntime_go.Shape{1, int64(len(query))}, query)
	if err != nil {
		return nil, err
	}
	defer qT.Destroy()

	muOut, err := e.encoder.Run([]onnxruntime_go.Value{pT, qT}, nil)
	if err != nil {
		return nil, err
	}
	defer destroyAll(muOut)
	mu, err := onnx.DataF32(muOut[0])
	if err != nil {
		return nil, err
	}

	emb := v.FlowEmbedding
	var norm float64
	for _, x := range emb {
		norm += float64(x) * float64(x)
	}
	norm = math.Sqrt(norm)
	spks := make([]float32, melBins)
	for i := 0; i < melBins; i++ {
		acc := float64(e.spkBias[i])
		for j := 0; j < len(emb); j++ {
			acc += float64(e.spkWeight[i*len(emb)+j]) * float64(emb[j]) / norm
		}
		spks[i] = float32(acc)
	}

	grid := windowing.TimeGrid(e.config)
	x := noise.GaussianField(seed, windowing.FlowNoiseStream, melBins, tMel)
	cond := framed.Cond
	for i := 0; i < len(grid)-1; i++ {
		t0 := grid[i]
		dt := grid[i+1] - t0
		xT, err := onnx.NewFloat32(onnxruntime_go.Shape{1, melBins, int64(tMel)}, x)
		if err != nil {
			return nil, err
		}
		muT, err := onnx.NewFloat32(onnxruntime_go.Shape{1, melBins, int64(tMel)}, mu)
		if err != nil {
			xT.Destroy()
			return nil, err
		}
		tT, err := onnx.NewFloat32(onnxruntime_go.Shape{1}, []float32{float32(t0)})
		if err != nil {
			xT.Destroy()
			muT.Destroy()
			return nil, err
		}
		spT, err := onnx.NewFloat32(onnxruntime_go.Shape{1, melBins}, spks)
		if err != nil {
			xT.Destroy()
			muT.Destroy()
			tT.Destroy()
			return nil, err
		}
		condT, err := onnx.NewFloat32(onnxruntime_go.Shape{1, melBins, int64(tMel)}, cond)
		if err != nil {
			xT.Destroy()
			muT.Destroy()
			tT.Destroy()
			spT.Destroy()
			return nil, err
		}
		vOut, err := e.estimator.Run([]onnxruntime_go.Value{xT, muT, tT, spT, condT}, nil)
		xT.Destroy()
		muT.Destroy()
		tT.Destroy()
		spT.Destroy()
		condT.Destroy()
		if err != nil {
			return nil, err
		}
		v, err := onnx.DataF32(vOut[0])
		if err != nil {
			destroyAll(vOut)
			return nil, err
		}
		next := make([]float32, len(x))
		for j := range x {
			next[j] = x[j] + float32(dt)*v[j]
		}
		// Explicit, not deferred: this runs once per Euler step, and `defer`
		// inside a loop queues every output until the function returns — which
		// is the whole leak, one step later.
		destroyAll(vOut)
		x = next
	}

	n := framed.N
	promptFrames := framed.PromptFrames
	outLen := 2 * n
	mel := make([]float32, melBins*outLen)
	for b := 0; b < melBins; b++ {
		for f := 0; f < outLen; f++ {
			mel[b*outLen+f] = x[b*tMel+(promptFrames+f)]
		}
	}
	return mel, nil
}

// Vocode renders a mel to audio via the exported HiFT graph.
func (e *Engine) Vocode(mel []float32, seed uint64) ([]float32, error) {
	frames := 2 * e.config.Window.MaxSpeechTokens
	melFrames := len(mel) / melBins
	nFrames := min(melFrames, frames)
	padded := make([]float32, melBins*frames)
	for b := 0; b < melBins; b++ {
		for f := 0; f < nFrames; f++ {
			padded[b*frames+f] = mel[b*melFrames+f]
		}
	}
	nSamples := frames * upsamplePerFrame
	phase := make([]float32, nHarmonics)
	offsets := noise.SymmetricUniforms(seed, windowing.VocoderPhaseStream, nHarmonics-1, math.Pi)
	copy(phase[1:], offsets)
	noise_ := noise.GaussianField(seed, windowing.VocoderNoiseStream, nHarmonics, nSamples)

	mT, err := onnx.NewFloat32(onnxruntime_go.Shape{1, melBins, int64(frames)}, padded)
	if err != nil {
		return nil, err
	}
	defer mT.Destroy()
	pT, err := onnx.NewFloat32(onnxruntime_go.Shape{1, nHarmonics, 1}, phase)
	if err != nil {
		return nil, err
	}
	defer pT.Destroy()
	nT, err := onnx.NewFloat32(onnxruntime_go.Shape{1, nHarmonics, int64(nSamples)}, noise_)
	if err != nil {
		return nil, err
	}
	defer nT.Destroy()

	wavOut, err := e.vocoder.Run([]onnxruntime_go.Value{mT, pT, nT}, nil)
	if err != nil {
		return nil, err
	}
	defer destroyAll(wavOut)
	wav, err := onnx.DataF32(wavOut[0])
	if err != nil {
		return nil, err
	}
	return append([]float32(nil), wav[:nFrames*upsamplePerFrame]...), nil
}

// generateInspected is the one path that produces speech tokens.
//
// Single-shot and streaming both go through it so they cannot drift: the
// generation ceiling, the stop-token observation and the artifact detectors are
// applied once, here, rather than twice and eventually differently. It returns
// the tokens after the specials are stripped, plus the verdict and whether
// generation stopped at the token cap rather than at a stop token.
//
// isTerminal says whether this chunk ends the passage. A continuation chunk has
// no sentence end, so its stop peak means nothing and its trailing pause is the
// sentence's rhythm rather than dead air — the detectors that cut a tail are
// told so and hold off.
func (e *Engine) generateInspected(
	textIds []int, v *voice.Profile, seed uint64, prefix []int, isTerminal bool,
	shouldCancel func() bool,
) ([]int, postprocess.Inspection, bool, error) {
	pp := e.config.Postprocess
	floor := config.EosFloor(len(textIds), e.config)
	cap := e.config.Sampling.MaxNewTokens
	if pp.Mode != postprocess.ModeOff {
		// Applied during generation, not after it: the tokens past the ceiling
		// cost real time on a device and are certain to be discarded. It only
		// ever stops a row that was going to run away.
		if c := postprocess.CeilingFor(len(textIds), pp, e.config.Window.MaxSpeechTokens); c < cap {
			cap = c
		}
	}

	// Selective re-roll: a window whose verdict is unfixable — dropout
	// (content missing) or suspect (certainly wrong, nowhere to cut) — is
	// regenerated from a derived seed, up to RetryMaxAttempts times. Only
	// condemned windows pay; the ladder is a pure function of the caller's
	// seed, so the same seed still gives the same audio, retries included.
	var gen []int
	var verdict postprocess.Inspection
	// True when the row stopped at the ceiling rather than at a stop token:
	// the utterance is cut off mid-sentence. Computed here — where `ended`
	// and the effective cap are both in hand — and carried out, because a
	// caller cannot recompute it after the specials are stripped and the cap
	// is forgotten.
	hitCap := false
	for attempt := 0; ; attempt++ {
		attemptSeed := seed
		if attempt > 0 {
			attemptSeed = deriveSeed(seed, uint64(retryStreamBase+attempt))
		}
		s := sampler.New(sampler.Config{
			Temperature:       e.config.Sampling.Temperature,
			RepetitionPenalty: e.config.Sampling.RepetitionPenalty,
			MinP:              e.config.Sampling.MinP,
			MaxNewTokens:      e.config.Sampling.MaxNewTokens,
			SilenceTokenIds:   e.config.Sampling.SilenceTokenIds,
		}, attemptSeed)
		if pp.Mode != postprocess.ModeOff {
			s.ObserveEOS(e.config.StopSpeech, floor)
		}

		raw, err := e.Generate(textIds, v, s, &cap, shouldCancel, prefix)
		if err != nil {
			return nil, postprocess.Inspection{}, false, err
		}

		// `gen` is what the shipped engine calls a row: every token the model
		// committed to, with the stop marker itself excluded. Indices into it
		// are decode-step indices, which is what makes the observed peak
		// comparable against it — so the detectors run here, before the
		// specials are stripped and free to renumber anything.
		gen = append([]int(nil), raw...)
		ended := len(gen) > 0 && gen[len(gen)-1] == e.config.StopSpeech
		if ended {
			gen = gen[:len(gen)-1]
		}
		peakAt, peakProb := s.EOSPeak()
		hitCap = !ended && len(gen) >= cap
		verdict = postprocess.Inspect(gen, postprocess.Request{
			TextTokenCount: len(textIds),
			MinTokens:      floor,
			EosPeakAt:      peakAt,
			EosPeakProb:    peakProb,
			Ended:          ended,
			IsTerminal:     isTerminal,
			HitCeiling:     hitCap,
		}, e.config.Sampling.SilenceTokenIds, pp)
		condemned := verdict.Reason == postprocess.ReasonDropout || verdict.Suspect
		if !condemned || pp.Mode == postprocess.ModeOff || attempt >= pp.RetryMaxAttempts {
			break
		}
	}
	if pp.Mode == postprocess.ModeTrim && verdict.Keep < len(gen) {
		gen = gen[:verdict.Keep]
	}

	tokens := make([]int, 0, len(gen))
	for _, t := range gen {
		if t < e.config.StartSpeech {
			tokens = append(tokens, t)
		}
	}
	return tokens, verdict, hitCap, nil
}

// Synthesize runs the whole pipeline: text -> tokens -> mel -> audio.
// shouldCancel is polled at every decode step, same as Generate; pass nil for
// no cancellation.
//
// An empty language means the voice's own — see resolveLanguage.
//
// speed is playback speed in [timestretch.MinSpeed, timestretch.MaxSpeed]:
// greater than one is faster, and the pitch does not move. 1.0 is an exact
// bypass — the waveform is the vocoder's own slice, untouched — and Go has no
// default arguments, so it is written out at every call site rather than
// omitted the way Python omits it. Outside the range the call is refused here,
// before the seconds of generation an error would otherwise be discovered
// after. Not part of the algorithm config and not in the fingerprint: it is an
// execution input like the seed and the text.
//
// previousTokens are the speech tokens this utterance continues from — the
// tokens returned by the call before it. The single window is then conditioned
// on their tail exactly as an interior chunk is conditioned on its predecessor,
// which is what stops a second request from restarting the pitch contour like a
// fresh sentence. Pass the whole previous result; only the last
// chunking.PrefixTokens are used and the slice happens here. nil is
// byte-for-byte the behaviour this method had before the parameter existed.
//
// Seven return values is more than a signature should carry, and they are all
// here anyway: this port hands back the intermediates rather than a Result
// type, so every stage a caller might want to compare against another backend
// is reachable without a struct that would have to be kept in step with
// Python's field by field. hitTokenCap is the one that is not an intermediate:
// true when generation stopped at the token cap rather than at a stop token,
// meaning the reading is probably truncated. Truncation is not an error — the
// audio is real, it is just incomplete — so it travels as a value rather than
// a non-nil err, and a caller must be able to report it.
func (e *Engine) Synthesize(text string, v *voice.Profile, seed uint64, language string, speed float64, previousTokens []int, shouldCancel func() bool) (audio []float32, tokens []int, mel []float32, chunks []timing.ChunkTiming, sampleRate int, hitTokenCap bool, err error) {
	if err := timestretch.ValidateSpeed(speed); err != nil {
		return nil, nil, nil, nil, 0, false, err
	}
	prefix, err := e.carryFrom(previousTokens)
	if err != nil {
		return nil, nil, nil, nil, 0, false, err
	}
	language = resolveLanguage(language, v)
	// The funnel is run here rather than inside Encode because its output is
	// what was tokenised, and therefore what the timing below describes: a
	// caller highlighting "three" needs the text the engine spoke, not the "3"
	// they passed in.
	prepared := speechtext.Prepared(text, language)
	textIds, err := e.frontend.Encode(prepared, language)
	if err != nil {
		return nil, nil, nil, nil, 0, false, err
	}
	// A single window is the whole passage, so it is terminal.
	tokens, _, hitTokenCap, err = e.generateInspected(textIds, v, seed, prefix, true, shouldCancel)
	if err != nil {
		return nil, nil, nil, nil, 0, false, err
	}
	mel, err = e.DecodeMel(tokens, v, deriveSeed(seed, 1))
	if err != nil {
		return nil, nil, nil, nil, 0, false, err
	}
	audio, err = e.Vocode(mel, deriveSeed(seed, 2))
	if err != nil {
		return nil, nil, nil, nil, 0, false, err
	}
	// Last, and after generateInspected rather than before it: the detectors
	// judge pacing by duration per token, and stretching first would move every
	// number they compare against.
	audio, err = timestretch.TimeStretch(audio, e.config.SampleRate, speed)
	if err != nil {
		return nil, nil, nil, nil, 0, false, err
	}
	// One window is one chunk, and it starts at zero. Measured on the stretched
	// audio, so a caller applies no 1/speed correction anywhere.
	chunks = timing.Timeline(
		[]timing.Span{{Text: prepared, Samples: len(audio), Tokens: len(tokens)}},
		e.config.SampleRate)
	return audio, tokens, mel, chunks, e.config.SampleRate, hitTokenCap, nil
}

// chunkStreamBase mirrors _STREAM_CHUNK in loudkit.engine: chunk seeds start
// here, clear of the per-stage streams (1 = flow, 2 = vocoder).
// Retry attempts draw derive(seed, 8+attempt): clear of the stage streams
// (1, 2) and below the chunk streams at 16.
const retryStreamBase = 8

const chunkStreamBase = 16

// SynthesizeLong speaks text of any length, splitting it across windows.
//
// This port had no long-form path: Synthesize renders one window and refuses
// anything longer, while the documentation called the binding supported. Two
// things make the joins match Python's rather than merely existing:
//
//   - Per-chunk seeds. Each chunk draws from derive(seed, 16+index), so a
//     chunk's audio does not depend on how many came before it and stopping
//     early cannot change what was already produced.
//   - Prefix carry. The last chunking.PrefixTokens speech tokens of a chunk
//     are fed into the next as context and dropped from its output.

// Chunk is one rendered piece, handed to a Stream callback as soon as it
// exists.
type Chunk struct {
	// Index is the zero-based position in the split, which is also what the
	// chunk's seed was derived from.
	Index  int
	Audio  []float32
	Tokens []int
	Mel    []float32
	// Text is this chunk's text after the speech funnel — what was tokenised,
	// which is not always what the caller passed in. It is the string a
	// highlight should be matched against; the caller's own text will drift from
	// it the moment a digit or an abbreviation appears.
	Text string
	// Inspection is what the artifact detectors concluded about this chunk.
	// Carried per chunk rather than aggregated because chunks fail
	// independently: one hallucinated tail among six clean ones is the case
	// worth seeing.
	Inspection postprocess.Inspection
	// HitTokenCap is true when generation stopped at the token cap rather than
	// at a stop token, so the chunk is cut off mid-sentence. Per chunk, for the
	// same reason the inspection is: chunks truncate independently. SynthesizeLong
	// ORs the flag across chunks; a caller streaming must decide itself whether
	// one truncated chunk is worth reporting.
	HitTokenCap bool
	// Timing is where this chunk lands in its own audio, and where its words
	// probably do. Start is zero: a streamed chunk is its own result and cannot
	// know what preceded it, so a caller stitching the stream adds the offsets
	// (timing.ChunkTiming.Shifted does the arithmetic). SynthesizeLong, which
	// has the whole passage in hand, adds them in samples instead.
	Timing timing.ChunkTiming
}

// Stream speaks text chunk by chunk, calling onChunk as each becomes ready.
//
// The difference from SynthesizeLong is delivery, not synthesis: time to first
// audio is set by the first chunk rather than by the whole passage, which is
// what lets a reading app start playing a sentence while the rest is still
// being made.
//
// A callback rather than a channel: a channel would need a goroutine, and a
// caller who stops reading early would leak it — a streaming API whose failure
// mode is a leaked goroutine is worse than one that hands you the chunk. Return
// false from onChunk to stop; the effect is the same as shouldCancel.
//
// shouldCancel is polled on every decode step, so an interrupt is honoured
// within one forward pass rather than at the next chunk boundary. The partial
// chunk is discarded without being rendered.
//
// An empty language means the voice's own — see resolveLanguage. Resolved once
// here, before splitting, so every chunk of a passage is read the same way.
//
// speed stretches each chunk independently, which is the same independence the
// seeds and the prefix already have: a chunk's audio must not depend on how many
// came before it, or a listener who stops early would have heard something
// different from one who did not. See Synthesize for the range and the bypass.
//
// previousTokens seeds the carry, so the first chunk of this call is conditioned
// on the tail of a previous one. It is the same conditioning the joins inside a
// passage already use — the carry variable below simply starts non-empty — which
// is why a request boundary stops being audible without a second mechanism
// existing to maintain.
func (e *Engine) Stream(text string, v *voice.Profile, seed uint64, language string, speed float64, previousTokens []int, shouldCancel func() bool, onChunk func(Chunk) bool) error {
	if err := timestretch.ValidateSpeed(speed); err != nil {
		return err
	}
	carry, err := e.carryFrom(previousTokens)
	if err != nil {
		return err
	}
	language = resolveLanguage(language, v)
	// The funnel runs on the whole text BEFORE splitting: Polish respelling
	// changes the length ("download" -> "dałnloud"), so a budget computed
	// first would be a budget for text the engine never speaks.
	prepared := speechtext.Prepared(text, language)
	chunks := chunking.SplitText(prepared, e.config.Chunking)
	if len(chunks) == 0 {
		return errors.New("nothing to speak")
	}

	for index, chunk := range chunks {
		if shouldCancel != nil && shouldCancel() {
			break
		}
		chunkSeed := deriveSeed(seed, uint64(chunkStreamBase+index))
		ids, err := e.frontend.Encode(chunk, language)
		if err != nil {
			return err
		}
		// Only the last chunk ends the passage.
		chunkTokens, verdict, chunkCapped, err := e.generateInspected(
			ids, v, chunkSeed, carry, index == len(chunks)-1, shouldCancel)
		if err != nil {
			return err
		}
		// Discarded, not rendered. The partial tokens belong to speech the
		// listener has already interrupted, and the mel decode plus vocode is
		// the larger half of the barge-in latency on an edge device — so
		// running them adds exactly the wait the cancellation exists to
		// remove, and then plays audio nobody asked for. Python does this at
		// engine.py:298; JS at engine.ts:473.
		if shouldCancel != nil && shouldCancel() {
			break
		}
		chunkMel, err := e.DecodeMel(chunkTokens, v, deriveSeed(chunkSeed, 1))
		if err != nil {
			return err
		}
		chunkAudio, err := e.Vocode(chunkMel, deriveSeed(chunkSeed, 2))
		if err != nil {
			return err
		}
		// The last stage, per chunk, for the reason Synthesize gives: the
		// detectors have already measured this render's pacing, and they measured
		// it on the vocoder's own samples.
		chunkAudio, err = timestretch.TimeStretch(chunkAudio, e.config.SampleRate, speed)
		if err != nil {
			return err
		}
		if n := e.config.Chunking.PrefixTokens; n > 0 && len(chunkTokens) > 0 {
			if n > len(chunkTokens) {
				n = len(chunkTokens)
			}
			carry = append([]int(nil), chunkTokens[len(chunkTokens)-n:]...)
		} else {
			carry = nil
		}

		// Through Timeline rather than by filling in a ChunkTiming here, so a
		// streamed chunk's own timing and the stitched one SynthesizeLong builds
		// come out of the same arithmetic and cannot drift apart.
		span := timing.Span{Text: chunk, Samples: len(chunkAudio), Tokens: len(chunkTokens)}
		if !onChunk(Chunk{
			Index: index, Audio: chunkAudio, Tokens: chunkTokens, Mel: chunkMel,
			Text:        chunk,
			Inspection:  verdict,
			HitTokenCap: chunkCapped,
			Timing:      timing.Timeline([]timing.Span{span}, e.config.SampleRate)[0],
		}) {
			break
		}
	}
	return nil
}

// SynthesizeLong speaks text of any length as one waveform.
//
// Exactly Stream with the chunks concatenated — one loop, so the streaming and
// whole-passage paths cannot drift apart. Use Stream when you want to start
// playing before the passage is finished.
//
// speed and previousTokens mean what they mean on Stream: the stretch is applied
// per chunk, so the two paths still produce the same waveform, and
// previousTokens conditions the first chunk while every chunk after it is
// conditioned on the one before, as always.
//
// The returned chunks tile the audio in order: the first Start is zero, chunk
// k's End is the same float as chunk k+1's Start, and the last End is the
// duration. Seven return values is more than a signature should carry — see
// Synthesize for why they are handed back rather than boxed. hitTokenCap is
// ORed across chunks, matching Python: one truncated chunk truncates the
// passage.
func (e *Engine) SynthesizeLong(text string, v *voice.Profile, seed uint64, language string, speed float64, previousTokens []int, shouldCancel func() bool) (audio []float32, tokens []int, mel []float32, chunks []timing.ChunkTiming, sampleRate int, hitTokenCap bool, err error) {
	var spans []timing.Span
	err = e.Stream(text, v, seed, language, speed, previousTokens, shouldCancel, func(c Chunk) bool {
		audio = append(audio, c.Audio...)
		tokens = append(tokens, c.Tokens...)
		mel = appendMelAlongTime(mel, c.Mel)
		hitTokenCap = hitTokenCap || c.HitTokenCap
		// Collected as spans and timed once at the end rather than by shifting
		// each chunk's own timing by a running float: Timeline accumulates the
		// offsets as integer samples, so every join is exact and a highlight
		// switching on time >= Start can neither gap nor light two chunks.
		spans = append(spans, timing.Span{
			Text: c.Text, Samples: len(c.Audio), Tokens: len(c.Tokens)})
		return true
	})
	if err != nil {
		return nil, nil, nil, nil, 0, false, err
	}
	return audio, tokens, mel, timing.Timeline(spans, e.config.SampleRate), e.config.SampleRate, hitTokenCap, nil
}

// carryFrom is the conditioning context a call inherits from the one before it.
//
// The same slice the streaming loop takes between two chunks — last
// chunking.PrefixTokens — applied to tokens that came from a different call.
// There is deliberately no second mechanism: a request boundary and a chunk
// boundary are the same join, and the reason chunk joins do not stutter is the
// reason request joins should not either.
//
// Any length is accepted because only the tail is used, so passing a whole
// previous result is the intended call and a caller should never have to know
// the prefix length to make it.
//
// The whole input is validated rather than only the slice that will be used: an
// id outside the acoustic codebook means the sequence was built wrong, and
// reporting that only when it lands in the last six tokens would make
// the failure depend on the length of the caller's text.
func (e *Engine) carryFrom(previousTokens []int) ([]int, error) {
	if len(previousTokens) == 0 {
		return nil, nil
	}
	limit := e.config.StartSpeech
	for _, token := range previousTokens {
		if token < 0 || token >= limit {
			return nil, fmt.Errorf(
				"previousTokens contains %d, which is not an acoustic speech "+
					"token (expected 0 <= id < %d). Pass the tokens returned by an "+
					"earlier call; the generator's own control tokens are already "+
					"stripped from them", token, limit)
		}
	}
	wanted := e.config.Chunking.PrefixTokens
	// Not previousTokens[len-wanted:] unguarded: a zero there is the whole slice
	// rather than nothing, which would condition on the entire previous
	// utterance at exactly the setting that means "chunks are independent".
	if wanted <= 0 {
		return nil, nil
	}
	if wanted > len(previousTokens) {
		wanted = len(previousTokens)
	}
	return append([]int(nil), previousTokens[len(previousTokens)-wanted:]...), nil
}

// appendMelAlongTime concatenates two row-major [melBins, frames] mels along
// the TIME axis.
//
// Appending the flat buffers end to end — the obvious thing, and what the JS
// port did — is not concatenation: after the first chunk the next chunk's bin 0
// lands after the previous chunk's bin 79, so every row but the first is wrong.
// The audio is unaffected (it is vocoded per chunk) but the returned mel is the
// diagnostic people reach for when two backends disagree, and a mis-shaped one
// sends them looking in the wrong place.
func appendMelAlongTime(dst, src []float32) []float32 {
	if len(dst) == 0 {
		return append([]float32(nil), src...)
	}
	const bins = melBins
	dstFrames := len(dst) / bins
	srcFrames := len(src) / bins
	out := make([]float32, bins*(dstFrames+srcFrames))
	for b := 0; b < bins; b++ {
		copy(out[b*(dstFrames+srcFrames):], dst[b*dstFrames:(b+1)*dstFrames])
		copy(out[b*(dstFrames+srcFrames)+dstFrames:], src[b*srcFrames:(b+1)*srcFrames])
	}
	return out
}

// deriveSeed mirrors engine._derive.
func deriveSeed(seed, stream uint64) uint64 {
	const phi = uint64(0x9e3779b97f4a7c15)
	const psi = uint64(0xbf58476d1ce4e5b9)
	return seed*phi + stream*psi
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
