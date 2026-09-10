// Package engine is the full synthesis pipeline over the exported ONNX
// graphs, fp32, no torch: a bit-parity port of loudkit.engine, over the
// stages loudkit.backends.onnx_backend runs, and of the JS engine. Same
// text, voice and seed give the same tokens and the same render band as the
// Python engine.
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

// Engine is a loaded engine: the manifest-selected ONNX graphs plus the checkpoint's embedding
// tables and the text frontend.
type Engine struct {
	config config.AlgorithmConfig
	// provider is the concrete execution provider every session was opened on,
	// the fusion head included: the answer to the ExecutionConfig's question,
	// resolved once so they cannot land on two different devices.
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
	head2     *onnx.Session
	fusion    map[string][]float32
	encoder   *onnx.Session
	estimator *onnx.Session
	vocoder   *onnx.Session
}

// The exported graphs, under the release's own names. One list, read by the
// loader and by the export record it checks itself against, so the two cannot
// name different sets.
const (
	condGraph      = "t3_cond.onnx"
	prefillGraph   = "t3_prefill.onnx"
	stepGraph      = "t3_step.onnx"
	pairStepGraph  = "t3_pair_step.onnx"
	head2Graph     = "t3_head2.onnx"
	encoderGraph   = "flow_encoder.onnx"
	estimatorGraph = "flow_estimator.onnx"
	hiftGraph      = "vocoder.onnx"
)

// graphNames is the set a decode mode loads, in the order it opens them.
func graphNames(fused bool) []string {
	if fused {
		return []string{
			condGraph, prefillGraph, pairStepGraph, head2Graph,
			encoderGraph, estimatorGraph, hiftGraph,
		}
	}
	return []string{condGraph, prefillGraph, stepGraph, encoderGraph, estimatorGraph, hiftGraph}
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
	// Checked once, at the door, rather than per utterance, and before the
	// graphs load. A chunking recipe with no character budget makes SplitText
	// cut nothing and loop forever; Python refuses it too, and this port reads
	// the same manifest key.
	algorithm, err := ckpt.Algorithm()
	if err != nil {
		return nil, err
	}
	if err := algorithm.Chunking.Validate(); err != nil {
		return nil, err
	}
	// The tokenizer and the checkpoint are separate files a caller can pair by
	// hand: LOUDKIT_TOKENIZER exists precisely so they can. A vocabulary wider
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

	layers := nLayers
	if llama, ok := ckpt.Manifest["llama_config"].(map[string]interface{}); ok {
		if count, ok := llama["num_hidden_layers"].(float64); ok {
			if count < 1 || count != math.Trunc(count) {
				return nil, fmt.Errorf("invalid transformer layer count %v", count)
			}
			layers = int(count)
		}
	}
	fused := algorithm.DecodeMode == config.DecodeFusionMTP2
	var fusion map[string][]float32
	if fused {
		fusion, err = ckpt.FusionWeights()
		if err != nil {
			return nil, err
		}
	}
	// The graphs are one export of one checkpoint or they are not a set, and a
	// mixed set speaks this checkpoint's tokens through another one's renderer.
	// Checked before any graph is opened, so a refusal costs no session.
	if err := ckpt.VerifyExport(onnxDir, algorithm, graphNames(fused)); err != nil {
		return nil, err
	}
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
	cond, err := load(condGraph, []string{"speaker_emb", "prompt_tokens", "emotion"}, []string{"t3_cond_out"})
	if err != nil {
		return nil, err
	}
	prefill, err := load(prefillGraph, []string{"embeds", "positions"}, prefillOutputs(fused, layers))
	if err != nil {
		return nil, err
	}
	stepName := stepGraph
	if fused {
		stepName = pairStepGraph
	}
	step, err := load(stepName, stepInputs(fused, layers), stepOutputs(fused, layers))
	if err != nil {
		return nil, err
	}
	var head2 *onnx.Session
	if fused {
		head2, err = load(head2Graph, []string{"hidden", "first_id"}, []string{"logits"})
		if err != nil {
			return nil, err
		}
	}
	encoder, err := load(encoderGraph, []string{"prompt_token", "speech_tokens"}, []string{"flow_encoder_out"})
	if err != nil {
		return nil, err
	}
	estimator, err := load(estimatorGraph, []string{"x", "mu", "t", "spks", "cond"}, []string{"flow_estimator_out"})
	if err != nil {
		return nil, err
	}
	vocoder, err := load(hiftGraph, []string{"mel", "phase", "noise"}, []string{"vocoder_out"})
	if err != nil {
		return nil, err
	}
	complete = true

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
		head2:     head2,
		fusion:    fusion,
		encoder:   encoder,
		estimator: estimator,
		vocoder:   vocoder,
	}, nil
}

// Close releases all sessions.
func (e *Engine) Close() {
	for _, s := range []*onnx.Session{e.cond, e.prefill, e.step, e.head2, e.encoder, e.estimator, e.vocoder} {
		if s != nil {
			s.Close()
		}
	}
}

// Config exposes the resolved algorithm.
func (e *Engine) Config() config.AlgorithmConfig { return e.config }

// Provider is the execution provider the graphs are running on: one of
// cpu, cuda, coreml, directml, never "auto". A caller who asked for auto reads
// the answer here.
func (e *Engine) Provider() string { return e.provider }

// Describe is the one-line run summary: what this engine computes, and what
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
// profile file with no language field inherits nothing: it loads as "en".
const fallbackLanguage = "en"

// resolveLanguage is the language chain: the argument, then the voice's
// recorded language, then English.
//
// Without the voice link, Synthesize("Cześć", polishVoice, seed, "", nil) runs
// Polish text through the English frontend (English number words, English
// abbreviation expansion, no Polish respelling) and says so nowhere. A profile
// records the language of the audio it was enrolled from, so the voice is the
// better answer than a constant.
//
// Passing a language explicitly is how cross-lingual synthesis is requested: an
// English voice reading Polish text is language "pl", and the argument always
// wins over the profile.
//
// The empty string is this port's "absent", as nil-slice and nil-func are
// elsewhere in the package. Python distinguishes an explicit "" from an
// omitted argument and Go cannot; an explicit "" therefore reaches the voice's
// language rather than tagging the text "[]", which is the better of the two
// behaviours available.
//
// Exported as ResolveLanguage below, which is the same chain for a caller
// outside this package.
func resolveLanguage(language string, v *voice.Profile) string {
	if language != "" {
		return language
	}
	if v != nil && v.Language != "" {
		return v.Language
	}
	return fallbackLanguage
}

// ResolveLanguage is resolveLanguage for a caller outside this package: the
// argument, then the voice, then English.
//
// It exists because frontend.Encode takes no voice and cannot run the chain
// itself, so a caller who tokenises before it synthesises would otherwise pass
// v.Language raw. A profile whose header language is blank then tags the text
// "[]", which no other port emits, because every other port routes that case
// through the fallback.
func ResolveLanguage(language string, v *voice.Profile) string {
	return resolveLanguage(language, v)
}

// ------------------------------------------------------------ generator

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
	// The wrapper hands ownership of a Run's outputs to the caller, so every
	// path that takes them has to release them. A path that does not leaks
	// native memory per render: invisible in a CLI that exits, unbounded in a
	// server.
	defer destroyAll(outs)
	return onnx.DataF32(outs[0])
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
	if len(prefix) > 0 {
		var pe []float32
		if e.config.DecodeMode == config.DecodeFusionMTP2 {
			prefix = prefix[:len(prefix)-len(prefix)%2]
			for i := 0; i < len(prefix); i += 2 {
				pe = append(pe, e.pairRow(prefix[i], prefix[i+1], i/2+1)...)
			}
		} else {
			pe = make([]float32, len(prefix)*hiddenDim)
			for i, tok := range prefix {
				sbase := tok * hiddenDim
				pbase := (i + 1) * hiddenDim
				for j := 0; j < hiddenDim; j++ {
					pe[i*hiddenDim+j] = e.speechEmb[sbase+j] + e.speechPos[pbase+j]
				}
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
// chunk boundary: the audible stutter the prefix exists to remove (~74 Hz at
// the join against ~7 Hz with a 6-token prefix, measured on the reference
// voice). They also seed the repetition-penalty state, since a token repeated
// across a join is as repeated as one within a chunk.
//
// Both token rows are validated first: this is a public door, and an id
// outside either table was indexed straight into it and killed the process.
func (e *Engine) Generate(textTokens []int, v *voice.Profile, s *sampler.Sampler, maxNewTokens *int, shouldCancel func() bool, prefix []int) ([]int, error) {
	if err := e.checkTextRow(textTokens); err != nil {
		return nil, err
	}
	if err := e.checkSpeechTokens(prefix); err != nil {
		return nil, err
	}
	cap_ := e.config.Sampling.MaxNewTokens
	if maxNewTokens != nil {
		cap_ = *maxNewTokens
	}
	floor := config.EosFloor(len(textTokens), e.config)
	stop := e.config.StopSpeech

	fused := e.config.DecodeMode == config.DecodeFusionMTP2
	if fused {
		prefix = prefix[:len(prefix)-len(prefix)%2]
	}
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
	offset := 1
	var hidden []float32
	if fused {
		hidden, err = onnx.DataF32(prefillOuts[1])
		if err != nil {
			return nil, err
		}
		hidden = append([]float32(nil), hidden...)
		offset = 2
	}
	kv, err := collectKV(prefillOuts[offset:])
	if err != nil {
		return nil, err
	}

	seen := make([]bool, e.config.SpeechVocabSize)
	// A token carried across a join is as seen as one this chunk emitted;
	// without this the penalty restarts blind at every chunk boundary.
	for _, t := range prefix {
		seen[t] = true
	}
	if fused {
		return e.generatePairs(logitsLast, hidden, kv, seen, cap_, floor, stop, len(prefix), prefillLen, s, shouldCancel)
	}
	out := []int{}
	for step := 0; step < cap_; step++ {
		if shouldCancel != nil && shouldCancel() {
			// Token-level barge-in: the partial row is discarded, not returned.
			return nil, fmt.Errorf("at decode step %d: %w", step, ErrCancelled)
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
// returns: every KV tensor for every step alive at once, memory growing
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
	for i := 0; i < len(kv.k); i++ {
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

func prefillOutputs(fused bool, layers int) []string {
	outs := []string{"logits"}
	if fused {
		outs = append(outs, "hidden")
	}
	for i := 0; i < layers; i++ {
		outs = append(outs, fmt.Sprintf("kv_k_%d", i), fmt.Sprintf("kv_v_%d", i))
	}
	return outs
}

func stepInputs(fused bool, layers int) []string {
	ins := []string{"embeds", "position"}
	if fused {
		ins = []string{"pair_ids", "speech_position", "position"}
	}
	for i := 0; i < layers; i++ {
		ins = append(ins, fmt.Sprintf("past_k_%d", i), fmt.Sprintf("past_v_%d", i))
	}
	return ins
}

func stepOutputs(fused bool, layers int) []string {
	outs := []string{"logits"}
	if fused {
		outs = append(outs, "hidden")
	}
	for i := 0; i < layers; i++ {
		outs = append(outs, fmt.Sprintf("present_k_%d", i), fmt.Sprintf("present_v_%d", i))
	}
	return outs
}

func collectKV(outs []onnxruntime_go.Value) (kvCache, error) {
	var kv kvCache
	if len(outs)%2 != 0 {
		return kv, fmt.Errorf("odd KV output count %d", len(outs))
	}
	for i := 0; i < len(outs)/2; i++ {
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
	// From the framing result, not from Window.StaticPromptTokens. The
	// config field is a nil pointer on the ragged window, which is Python's
	// documented default, so reading it here killed the process on a
	// configuration FrameWindows frames without complaint.
	pLen := framed.PromptTokens
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
	// Both operands are one float32 widened, so the product needs at most 48
	// significand bits and float64 has 53: it is exact, and a fused rounding and
	// a separate one give the same bits. The compiler contracts this and it
	// changes nothing. Widening the input to float64 would end that, which is
	// what TestWidenedSquareIsContractionProof stands over.
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
		// One Euler step, and the explicit conversion holds it to one multiply
		// and one add. Fused, the step keeps a rounding the other ports drop,
		// and nothing downstream absorbs it: the mel goes to the vocoder at full
		// amplitude, and the difference reaches the written WAV.
		next := make([]float32, len(x))
		for j := range x {
			next[j] = x[j] + float32(float32(dt)*v[j])
		}
		// Explicit, not deferred: this runs once per Euler step, and `defer`
		// inside a loop queues every output until the function returns, which
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
// sentence's rhythm rather than dead air: the detectors that cut a tail are
// told so and hold off.
func (e *Engine) generateInspected(
	textIds []int, v *voice.Profile, seed uint64, prefix []int, isTerminal bool,
	shouldCancel func() bool,
) ([]int, postprocess.Inspection, bool, bool, error) {
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

	// Selective re-roll: a window whose verdict is unfixable, dropout
	// (content missing) or suspect (certainly wrong, nowhere to cut), is
	// regenerated from a derived seed, up to RetryMaxAttempts times. Only
	// condemned windows pay; the ladder is a pure function of the caller's
	// seed, so the same seed still gives the same audio, retries included.
	var gen []int
	var verdict postprocess.Inspection
	// True when the row stopped at the ceiling rather than at a stop token:
	// the utterance is cut off mid-sentence. Computed here, where `ended`
	// and the effective cap are both in hand, and carried out, because a
	// caller cannot recompute it after the specials are stripped and the cap
	// is forgotten.
	hitCap := false
	hitWindow := false
	// When the ladder exhausts with every attempt condemned, the attempt that
	// ships is the *best* seen, not the last: fewest tokens in the
	// true-silence set, integer and portable, like the detectors. Measured:
	// on the worst voices 30% of condemned fires exhaust the ladder, and
	// keeping the last attempt shipped rows worse than the first. The render
	// census gates the count where the checkpoint carries one; the configured
	// silence list is the fallback.
	deadAirIds := pp.SilenceRenderIds
	if len(deadAirIds) == 0 {
		deadAirIds = e.config.Sampling.SilenceTokenIds
	}
	deadAir := make(map[int]struct{}, len(deadAirIds))
	for _, id := range deadAirIds {
		deadAir[id] = struct{}{}
	}
	bestCount := -1
	var bestGen []int
	var bestVerdict postprocess.Inspection
	bestHitCap := false
	bestHitWindow := false
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
			return nil, postprocess.Inspection{}, false, false, err
		}

		// `gen` is what the shipped engine calls a row: every token the model
		// committed to, with the stop marker itself excluded. Indices into it
		// are decode-step indices, which is what makes the observed peak
		// comparable against it, so the detectors run here, before the
		// specials are stripped and free to renumber anything.
		gen = append([]int(nil), raw...)
		ended := len(gen) > 0 && gen[len(gen)-1] == e.config.StopSpeech
		if ended {
			gen = gen[:len(gen)-1]
		}
		peakAt, peakProb := s.EOSPeak()
		hitCap = !ended && len(gen) >= cap
		// The window, asked separately and asked here, where gen is still what
		// the model produced. cap is min(MaxNewTokens, ceilingFor(...)), so
		// hitCap cannot tell a filled window from a runaway short text; and
		// the trim below can cut a filled window down to a few tokens, which
		// is how a caller measuring the returned slice saw room to spare.
		hitWindow = !ended && len(gen) >= e.config.Window.MaxSpeechTokens
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
		if !condemned || pp.Mode == postprocess.ModeOff {
			break
		}
		silenceCount := 0
		for _, t := range gen {
			if _, ok := deadAir[t]; ok {
				silenceCount++
			}
		}
		if bestCount < 0 || silenceCount < bestCount {
			// Strict `<`: on a tie the earlier attempt stands, so the ladder
			// stays a pure function of the caller's seed with no dependence
			// on iteration order.
			bestCount, bestGen, bestVerdict, bestHitCap, bestHitWindow =
				silenceCount, gen, verdict, hitCap, hitWindow
		}
		if attempt >= pp.RetryMaxAttempts {
			gen, verdict, hitCap, hitWindow = bestGen, bestVerdict, bestHitCap, bestHitWindow
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
	return tokens, verdict, hitCap, hitWindow, nil
}

// SynthesizeWindow renders text that fits one model window, and refuses text
// that does not. Synthesize is the call for any length; this one is for the
// conformance harness and for a caller who wants the refusal.
func (e *Engine) SynthesizeWindow(text string, v *voice.Profile, o Options) (*Result, error) {
	speed := o.speed()
	if err := timestretch.ValidateSpeed(speed); err != nil {
		return nil, err
	}
	prefix, err := e.carryFrom(o.PreviousTokens)
	if err != nil {
		return nil, err
	}
	language := resolveLanguage(o.Language, v)
	// The funnel runs here rather than inside Encode: its output is what was
	// tokenised, and therefore what the timing describes.
	prepared := speechtext.Prepared(text, language)
	textIds, err := e.frontend.Encode(prepared, language)
	if err != nil {
		return nil, err
	}
	// A single window is the whole passage, so it is terminal.
	tokens, _, hitTokenCap, hitWindowCap, err := e.generateInspected(
		textIds, v, o.Seed, prefix, true, o.ShouldCancel)
	if err != nil {
		return nil, err
	}
	// Refused rather than truncated: one window's audio with the rest of the
	// text unspoken is silent data loss. Read from the generation, not the
	// trimmed tokens, because postprocess can cut a filled window short.
	if hitWindowCap {
		return nil, fmt.Errorf(
			"the text did not fit one %d-token window and its tail was not spoken. "+
				"Use Synthesize, which splits at sentence boundaries and joins the audio",
			e.config.Window.MaxSpeechTokens)
	}
	// Discarded, not rendered: every port polls here, between the token
	// phase and the render.
	if o.ShouldCancel != nil && o.ShouldCancel() {
		return nil, ErrCancelled
	}
	mel, err := e.DecodeMel(tokens, v, deriveSeed(o.Seed, 1))
	if err != nil {
		return nil, err
	}
	if o.ShouldCancel != nil && o.ShouldCancel() {
		return nil, ErrCancelled
	}
	audio, err := e.Vocode(mel, deriveSeed(o.Seed, 2))
	if err != nil {
		return nil, err
	}
	// Stretched last: the detectors judged pacing on the vocoder's own
	// samples, and the timeline is measured on what the caller receives.
	audio, err = timestretch.TimeStretch(audio, e.config.SampleRate, speed)
	if err != nil {
		return nil, err
	}
	audio = timestretch.FadeEdges(audio, e.config.SampleRate, e.config.EdgeFade())
	if o.ShouldCancel != nil && o.ShouldCancel() {
		return nil, ErrCancelled
	}
	chunks := timing.Timeline(
		[]timing.Span{{Text: prepared, Samples: len(audio), Tokens: len(tokens)}},
		e.config.SampleRate)
	return &Result{
		Audio: audio, SampleRate: e.config.SampleRate, Tokens: tokens, Mel: mel,
		Chunks: chunks, HitTokenCap: hitTokenCap,
	}, nil
}

// Retry attempts draw derive(seed, 8+attempt): clear of the stage streams
// (1, 2) and below the chunk streams at 16.
const retryStreamBase = 8

// chunkStreamBase mirrors _STREAM_CHUNK in loudkit.window: chunks after the
// first draw derive(seed, 16+index). Chunk 0 draws the caller's seed itself, so
// a text that fits one window renders the same through SynthesizeWindow and
// through Synthesize, in every implementation.
const chunkStreamBase = 16

// resplitStream mirrors _STREAM_RESPLIT in loudkit.window: the second half of
// a re-split chunk draws from its own stream off the chunk's seed, clear of
// the flow at 1, the vocoder at 2 and the retry ladder from 8 up.
const resplitStream = 4096

// Chunk is one rendered piece, handed to a Stream callback as soon as it
// exists.
type Chunk struct {
	// Index is the zero-based position in the split, which is also what the
	// chunk's seed was derived from.
	Index  int
	Audio  []float32
	Tokens []int
	Mel    []float32
	// Text is this chunk's text after the speech funnel: what was tokenised,
	// and what a highlight should be matched against.
	Text string
	// Inspection is what the artifact detectors concluded about this chunk.
	Inspection postprocess.Inspection
	// HitTokenCap is true when generation stopped at the token cap rather
	// than at a stop token, so the chunk is cut off mid-sentence.
	HitTokenCap bool
	// Timing is where this chunk lands in its own audio, starting at zero. A
	// caller stitching the stream adds the offsets (ChunkTiming.Shifted).
	Timing timing.ChunkTiming
}

// Stream speaks text chunk by chunk, calling onChunk as each becomes ready.
//
// The same synthesis as Synthesize, delivered as it is made: time to first
// audio is set by the first chunk. Return false from onChunk to stop;
// o.ShouldCancel stops within one decode step, and the stream then ends with
// no error: the chunks already delivered are the partial, the one in flight
// is discarded.
//
// Chunk 0 draws the caller's seed and every later chunk derive(seed,
// 16+index), so a chunk's audio does not depend on how many came before it.
// The last chunking.PrefixTokens tokens of each chunk condition the next, and
// o.PreviousTokens seed that carry for the first.
func (e *Engine) Stream(text string, v *voice.Profile, o Options, onChunk func(Chunk) bool) error {
	err := e.stream(text, v, o, onChunk)
	if errors.Is(err, ErrCancelled) {
		return nil
	}
	return err
}

// stream is Stream, returning ErrCancelled where the caller's flag stopped it.
func (e *Engine) stream(text string, v *voice.Profile, o Options, onChunk func(Chunk) bool) error {
	speed := o.speed()
	if err := timestretch.ValidateSpeed(speed); err != nil {
		return err
	}
	carry, err := e.carryFrom(o.PreviousTokens)
	if err != nil {
		return err
	}
	shouldCancel := o.ShouldCancel
	language := resolveLanguage(o.Language, v)
	// The funnel runs on the whole text before splitting: Polish respelling
	// changes the length, so a budget computed first would be a budget for
	// text the engine never speaks.
	prepared := speechtext.Prepared(text, language)
	chunks := chunking.SplitText(prepared, e.config.Chunking)
	if len(chunks) == 0 {
		return errors.New("nothing to speak")
	}

	// A work queue rather than a range: a chunk the window could not hold is
	// replaced, in place, by its two halves. Both halves keep the original
	// index, so a repair cannot move the seed of any later chunk.
	type part struct {
		text     string
		index    int
		seed     uint64
		terminal bool
		// False on a half, so a half that still overruns ships as it is.
		splittable bool
	}
	queue := make([]part, 0, len(chunks))
	for i, c := range chunks {
		queue = append(queue, part{
			text: c, index: i, seed: chunkSeed(o.Seed, i),
			terminal: i == len(chunks)-1, splittable: true,
		})
	}

	for qi := 0; qi < len(queue); qi++ {
		p := queue[qi]
		index, chunk, chunkSeed := p.index, p.text, p.seed
		if shouldCancel != nil && shouldCancel() {
			return ErrCancelled
		}
		ids, err := e.frontend.Encode(chunk, language)
		if err != nil {
			return err
		}
		// Only the last chunk ends the passage.
		chunkTokens, verdict, chunkCapped, chunkFilledWindow, err := e.generateInspected(
			ids, v, chunkSeed, carry, p.terminal, shouldCancel)
		if err != nil {
			return err
		}
		// The window has to be what stopped it, not the length ceiling that
		// stops a short text running away: halving a runaway gives two.
		if chunkFilledWindow && p.splittable &&
			e.config.Chunking.CapResplit == chunking.WordCapResplit {
			if first, second, ok := chunking.SplitInHalf(chunk); ok {
				queue = append(queue[:qi], append([]part{
					{text: first, index: index, seed: chunkSeed, terminal: false, splittable: false},
					{text: second, index: index, seed: deriveSeed(chunkSeed, resplitStream),
						terminal: p.terminal, splittable: false},
				}, queue[qi+1:]...)...)
				qi--
				continue
			}
		}
		// Discarded, not rendered: the decode and vocode are the larger half
		// of barge-in latency, and the audio is speech nobody asked for.
		if shouldCancel != nil && shouldCancel() {
			return ErrCancelled
		}
		chunkMel, err := e.DecodeMel(chunkTokens, v, deriveSeed(chunkSeed, 1))
		if err != nil {
			return err
		}
		if shouldCancel != nil && shouldCancel() {
			return ErrCancelled
		}
		chunkAudio, err := e.Vocode(chunkMel, deriveSeed(chunkSeed, 2))
		if err != nil {
			return err
		}
		chunkAudio, err = timestretch.TimeStretch(chunkAudio, e.config.SampleRate, speed)
		if err != nil {
			return err
		}
		chunkAudio = timestretch.FadeEdges(chunkAudio, e.config.SampleRate, e.config.EdgeFade())
		if shouldCancel != nil && shouldCancel() {
			return ErrCancelled
		}
		carry = e.carryAligned(chunkTokens)

		// Through Timeline, so a streamed chunk's own timing and the stitched
		// one Synthesize builds come out of the same arithmetic.
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

// Synthesize speaks text of any length as one Result.
//
// Exactly Stream with the chunks concatenated, one loop, so the two paths
// cannot drift. HitTokenCap is ORed across chunks: one capped chunk caps
// the passage. When o.ShouldCancel returned true the result is nil and the
// error is ErrCancelled: a passage cut short is never handed back as a Result.
func (e *Engine) Synthesize(text string, v *voice.Profile, o Options) (*Result, error) {
	out := &Result{SampleRate: e.config.SampleRate}
	var spans []timing.Span
	err := e.stream(text, v, o, func(c Chunk) bool {
		out.Audio = append(out.Audio, c.Audio...)
		out.Tokens = append(out.Tokens, c.Tokens...)
		out.Mel = appendMelAlongTime(out.Mel, c.Mel)
		out.HitTokenCap = out.HitTokenCap || c.HitTokenCap
		// Timed once at the end: Timeline accumulates the offsets as integer
		// samples, so every join is exact.
		spans = append(spans, timing.Span{
			Text: c.Text, Samples: len(c.Audio), Tokens: len(c.Tokens)})
		return true
	})
	if err != nil {
		return nil, err
	}
	out.Chunks = timing.Timeline(spans, e.config.SampleRate)
	return out, nil
}

// carryFrom is the conditioning context a call inherits from the one before it.
//
// The same slice the streaming loop takes between two chunks, the last
// chunking.PrefixTokens, applied to tokens that came from a different call.
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
	if err := e.checkSpeechTokens(previousTokens); err != nil {
		return nil, err
	}
	return e.carryAligned(previousTokens), nil
}

// checkSpeechTokens refuses an id that is not an acoustic speech token.
//
// Shared with Generate, which indexes its repetition table with every carried
// id and died on an out-of-range one instead of reporting it. The
// sentence is carryFrom's own, unchanged, because the two are the same refusal
// at two doors and a caller who already reads one should not meet a second.
func (e *Engine) checkSpeechTokens(tokens []int) error {
	limit := e.config.StartSpeech
	for _, token := range tokens {
		if token < 0 || token >= limit {
			return fmt.Errorf(
				"previousTokens contains %d, which is not an acoustic speech "+
					"token (expected 0 <= id < %d). Pass the tokens returned by an "+
					"earlier call; the generator's own control tokens are already "+
					"stripped from them", token, limit)
		}
	}
	return nil
}

// checkTextRow refuses text tokens textRow would read the tables past the end
// of.
//
// Load already refuses a tokenizer whose largest id does not fit the text
// embedding table, so this catches only a row a caller assembled by hand.
// Without it textRow panics with an index out of range inside the library,
// which is not a failure a caller can handle.
func (e *Engine) checkTextRow(textTokens []int) error {
	rows := len(e.textEmb) / hiddenDim
	for _, token := range textTokens {
		if token < 0 || token >= rows {
			return fmt.Errorf(
				"textTokens contains %d, which is past the end of the "+
					"checkpoint's text embedding table (expected 0 <= id < %d). "+
					"Pass the ids frontend.Encode returns", token, rows)
		}
	}
	// The row textRow builds is the tokens plus the start and stop markers,
	// and it indexes the positional table by position.
	if framed, positions := len(textTokens)+2, len(e.textPos)/hiddenDim; framed > positions {
		return fmt.Errorf(
			"%d text tokens frame a %d-position row, past the checkpoint's "+
				"%d text positions; split the text first",
			len(textTokens), framed, positions)
	}
	return nil
}

// appendMelAlongTime concatenates two row-major [melBins, frames] mels along
// the TIME axis.
//
// Appending the flat buffers end to end (the obvious thing, and what the JS
// port did) is not concatenation: after the first chunk the next chunk's bin 0
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

// chunkSeed is the seed chunk i of a passage draws. Chunk 0 takes the caller's
// seed itself, so a text that fits one window renders the same through
// SynthesizeWindow and through Synthesize; every later chunk derives from its own
// stream, so its audio does not depend on how many chunks came before it.
func chunkSeed(seed uint64, index int) uint64 {
	if index == 0 {
		return seed
	}
	return deriveSeed(seed, uint64(chunkStreamBase+index))
}

// deriveSeed mirrors engine._derive.
func deriveSeed(seed, stream uint64) uint64 {
	const phi = uint64(0x9e3779b97f4a7c15)
	const psi = uint64(0xbf58476d1ce4e5b9)
	return seed*phi + stream*psi
}

func (e *Engine) carryAligned(tokens []int) []int {
	n := e.config.Chunking.PrefixTokens
	if n <= 0 || len(tokens) == 0 {
		return nil
	}
	start := max(0, len(tokens)-n)
	if e.config.DecodeMode == config.DecodeFusionMTP2 {
		end := len(tokens) - len(tokens)%2
		start = max(0, end-n)
		start -= start % 2
		return append([]int(nil), tokens[start:end]...)
	}
	return append([]int(nil), tokens[start:]...)
}
