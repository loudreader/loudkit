// Package config mirrors loudkit.config: the algorithm values that are
// identical on every backend, read from the checkpoint manifest.
package config

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"

	"github.com/loudreader/loudkit/go/chunking"
	"github.com/loudreader/loudkit/go/postprocess"
	"github.com/loudreader/loudkit/go/speechtext"
)

// SamplingConfig is the sampling law.
type SamplingConfig struct {
	Temperature        float64
	RepetitionPenalty  float64
	MinP               float64
	MaxNewTokens       int
	SilenceTokenIds    []int
	MinTokensFloor     int
	MinTokensTextRatio float64
}

// WindowConfig is the window framing recipe.
type WindowConfig struct {
	MaxSpeechTokens    int
	StaticLength       *int
	PadTokenID         *int
	StaticPromptTokens *int
}

// timestretchEdgeFadeSeconds mirrors timestretch.EdgeFadeSeconds; config cannot
// import timestretch, and the two are pinned equal by a test.
const timestretchEdgeFadeSeconds = 0.02

// AlgorithmConfig is everything that determines what the engine produces.
type AlgorithmConfig struct {
	DecodeMode string
	// EdgeFadeSeconds is the raised-cosine ramp on both edges of every rendered
	// window. Zero means unset, like an empty DecodeMode: the shipped length
	// applies and is fingerprinted. Read it through EdgeFade.
	EdgeFadeSeconds float64
	RecipeVersion   string
	Guidance        string
	GuidanceRate    float64
	EulerSteps      int
	// EulerGrid is the explicit time grid, or nil for the cosine schedule.
	//
	// An explicit grid overrides the cosine schedule.
	//
	// An explicit grid is what a checkpoint ships when the schedule has to
	// match across implementations, because "cosine" is a formula five
	// codebases can write five ways. Without this field a checkpoint shipping
	// one integrates on a different schedule here, silently, and under a
	// fingerprint that records the grid it is ignoring. The shipping manifest
	// has `euler_grid: null`.
	EulerGrid []float64
	Sampling  SamplingConfig
	Window    WindowConfig
	// Chunking is where the reader breathes. Read from the manifest rather
	// than defaulted: a checkpoint that declares its own boundaries and prefix
	// carry, and a runtime that silently uses different ones, agree on
	// recipe_version and disagree on the reading.
	Chunking chunking.Config
	// Postprocess is the artifact detectors. They remove tokens, so they
	// change the audio and are read from the manifest for the same reason the
	// joins are: a backend that re-guesses where a chunk ended cuts somewhere
	// else, and the difference is a hallucinated word that either does or does
	// not reach a listener.
	Postprocess postprocess.Config
	// Text is the funnel's identity: its code version and the digest of the
	// grammar file this port reads. In the fingerprint because the funnel
	// decides what string the model is handed, and therefore what it says.
	Text       TextConfig
	SampleRate int
	// TokenRateHz is speech tokens per second. Algorithm-bearing: it converts
	// a token count into the seconds of speech an over-window refusal reports,
	// and it is hashed into the fingerprint.
	TokenRateHz     float64
	SpeechVocabSize int
	StartSpeech     int
	StopSpeech      int
}

// Defaults is the algorithm a manifest that declares nothing gets.
//
// Every absent key falls back to a field of this one value rather than to a
// literal retyped where it is read, so an absent key means one thing in one
// place. manifest.algorithm_from reads its own dataclass the same way, and
// this has to be the same value. A default that disagrees with the reference
// is invisible while no manifest omits the key and silent once one does: the
// loudest is the step count, because a zero there runs the flow loop no times
// and renders the prior noise field as audio without an error anywhere.
//
// The window is the ragged one, which is what an absent `window` block means:
// the shipped static window is a recipe a manifest declares, not a shape to
// assume when it says nothing. ProductionWindow is still that recipe.
func Defaults() AlgorithmConfig {
	return AlgorithmConfig{
		DecodeMode:      DecodeSingle,
		EdgeFadeSeconds: 0.005,
		RecipeVersion:   RecipeVersion,
		Guidance:        "single_path",
		GuidanceRate:    0,
		EulerSteps:      2,
		TokenRateHz:     25.0,
		SampleRate:      24_000,
		SpeechVocabSize: 8194,
		StartSpeech:     6561,
		StopSpeech:      6562,
		Sampling: SamplingConfig{
			Temperature:       0.8,
			RepetitionPenalty: 1.2,
			MinP:              0.05,
			MaxNewTokens:      255,
			SilenceTokenIds:   []int{},
		},
		Window:      WindowConfig{MaxSpeechTokens: 255},
		Chunking:    chunking.Production(),
		Postprocess: postprocess.Production(),
	}
}

// ProductionWindow is the shipped static-window recipe.
func ProductionWindow() WindowConfig {
	s := 255
	p := 238
	pad := 4254
	return WindowConfig{
		MaxSpeechTokens:    255,
		StaticLength:       &s,
		StaticPromptTokens: &p,
		PadTokenID:         &pad,
	}
}

func intp(v int) *int { return &v }

// DecodeSingle and DecodeFusionMTP2 are the decode loops a manifest may
// declare. They are the release's own spellings, and the closed set they
// form is checkpoint.SupportedDecodeModes.
const (
	DecodeSingle     = "single"
	DecodeFusionMTP2 = "fusion_mtp2"
)

// GuidanceModes are the modes a manifest may declare.
var GuidanceModes = []string{"single_path", "cfg_dual_path"}

// FromManifest reads the algorithm values out of a checkpoint manifest.
//
// It returns an error rather than a bare config because a manifest is external
// data and two of its values cannot be defaulted safely. Guidance is the one
// that matters most: this binding runs the estimator once per step and never
// forms (1+w)·v_cond − w·v_uncond, so a cfg_dual_path checkpoint would load,
// produce plausible audio and disagree with the Python engine under a matching
// recipe_version. Refusing costs one error; not refusing costs a defect nobody
// can see. The JS binding already refuses it, and the Python and CoreML
// backends do too.
func FromManifest(m map[string]interface{}) (AlgorithmConfig, error) {
	def := Defaults()

	decode, err := block(m, "decode")
	if err != nil {
		return AlgorithmConfig{}, err
	}
	mode, err := stringKey(decode, "manifest['decode']", "mode", def.DecodeMode)
	if err != nil {
		return AlgorithmConfig{}, err
	}
	if mode != DecodeSingle && mode != DecodeFusionMTP2 {
		return AlgorithmConfig{}, fmt.Errorf("unsupported decode mode %q", mode)
	}

	// One accumulator for every numeric sub-key below, checked once before the
	// config is built. Blocks stay interface{} so a read can tell an absent
	// key from an explicit zero; flattening to float64 first would make the
	// two indistinguishable.
	var nums numbers

	sampling, err := block(m, "sampling_defaults")
	if err != nil {
		return AlgorithmConfig{}, err
	}
	samplingNum := nums.in(sampling, "manifest['sampling_defaults']")
	samplingCount := nums.count(sampling, "manifest['sampling_defaults']")
	sil, err := idList(m, "silence_token_ids")
	if err != nil {
		return AlgorithmConfig{}, err
	}
	if sil == nil {
		sil = def.Sampling.SilenceTokenIds
	}
	speech, err := block(m, "speech_tokens")
	if err != nil {
		return AlgorithmConfig{}, err
	}
	speechCount := nums.count(speech, "manifest['speech_tokens']")

	win, err := windowFrom(m, &nums, def.Window)
	if err != nil {
		return AlgorithmConfig{}, err
	}

	// Absent means what the reference says it means, and present-and-zero means
	// the manifest disabled it deliberately. That second half is what the
	// numeric reads exist for, and TestFromManifestKeepsExplicitZeroes pins it.
	eos, err := block(m, "eos_floor")
	if err != nil {
		return AlgorithmConfig{}, err
	}
	eosNum := nums.in(eos, "manifest['eos_floor']")
	eosFloorValue := nums.count(eos, "manifest['eos_floor']")("min_tokens_floor",
		def.Sampling.MinTokensFloor)
	eosRatioValue := eosNum("min_tokens_text_ratio", def.Sampling.MinTokensTextRatio)

	guidance, err := stringKey(m, "manifest", "guidance", def.Guidance)
	if err != nil {
		return AlgorithmConfig{}, err
	}
	known := false
	for _, mode := range GuidanceModes {
		if guidance == mode {
			known = true
		}
	}
	if !known {
		return AlgorithmConfig{}, fmt.Errorf(
			"manifest declares unknown guidance mode %q; expected one of %s",
			guidance, strings.Join(GuidanceModes, ", "))
	}
	if guidance == "cfg_dual_path" {
		return AlgorithmConfig{}, fmt.Errorf(
			"manifest declares guidance mode cfg_dual_path, which this binding does " +
				"not implement: it would render single-path audio and silently " +
				"disagree with the Python engine")
	}

	chunk := def.Chunking
	if v, err := block(m, "chunking"); err != nil {
		return AlgorithmConfig{}, err
	} else if len(v) > 0 {
		// A chunking key this splitter does not honour is refused by name
		// rather than ignored. chunking.Config carries no first-chunk cap, so
		// a manifest that sets one splits differently here than in Python
		// under one recipe_version, and the fingerprint cannot see it: the
		// canonical form has no slot for a field this port does not have, so
		// the manifest that sets the key and the manifest that omits it hash
		// the same. That is the divergence class the guidance door is already
		// refused for. Rust and JS refuse it by this name too.
		for _, key := range []string{"first_chunk_max_tokens"} {
			if _, present := v[key]; present {
				return AlgorithmConfig{}, fmt.Errorf(
					"manifest['chunking'][%q] is not implemented by this binding, "+
						"which would split the text differently from the Python "+
						"engine under a matching recipe_version", key)
			}
		}
		// Not `.(bool)` ignored: a manifest that turns the splitter off and a
		// runtime that leaves it on read the same recipe_version and deliver
		// different audio, and `enabled` is the one key whose misreading
		// changes every join at once.
		if raw, present := v["enabled"]; present {
			x, ok := raw.(bool)
			if !ok {
				return AlgorithmConfig{}, fmt.Errorf(
					"manifest['chunking']['enabled'] must be JSON true or false, got %#v", raw)
			}
			chunk.Enabled = x
		}
		chunkCount := nums.count(v, "manifest['chunking']")
		chunk.MaxTokens = chunkCount("max_tokens", chunk.MaxTokens)
		chunk.PrefixTokens = chunkCount("prefix_tokens", chunk.PrefixTokens)
		// Presence, not length, as for the abbreviations below: the default
		// applies to an absent key only, and a declared list is read as it
		// stands, so the canonical form records what the manifest asked for.
		// chunking.Validate carries the reference's refusal for an empty set,
		// and only a presence check keeps that refusal reachable.
		if raw, present := v["split_on"]; present {
			seps, err := stringList(raw, "chunking.split_on")
			if err != nil {
				return AlgorithmConfig{}, err
			}
			if len(seps) == 0 {
				return AlgorithmConfig{}, errors.New(
					"chunking.split_on cannot be empty: there would be nowhere to break")
			}
			chunk.SplitOn = seps
		}
		// Presence, not length: an empty list is meaningful here, it is the
		// old law spelled as data, so it must not fall back to the shipping
		// set.
		if raw, present := v["abbreviations"]; present {
			abbreviations, err := stringList(raw, "chunking.abbreviations")
			if err != nil {
				return AlgorithmConfig{}, err
			}
			chunk.Abbreviations = abbreviations
		}
		mid, err := stringKey(v, "manifest['chunking']", "mid_sentence_period",
			chunk.MidSentencePeriod)
		if err != nil {
			return AlgorithmConfig{}, err
		}
		if mid != chunking.HoldMidSentencePeriod && mid != chunking.BreakMidSentencePeriod {
			return AlgorithmConfig{}, fmt.Errorf(
				"manifest declares unknown chunking.mid_sentence_period %q", mid)
		}
		chunk.MidSentencePeriod = mid
		resplit, err := stringKey(v, "manifest['chunking']", "cap_resplit", chunk.CapResplit)
		if err != nil {
			return AlgorithmConfig{}, err
		}
		if resplit != chunking.WordCapResplit && resplit != chunking.OffCapResplit {
			return AlgorithmConfig{}, fmt.Errorf(
				"manifest declares unknown chunking.cap_resplit %q", resplit)
		}
		chunk.CapResplit = resplit
	}

	pp := def.Postprocess
	if v, err := block(m, "postprocess"); err != nil {
		return AlgorithmConfig{}, err
	} else if len(v) > 0 {
		// The render-id censuses are properties of the weights and live at
		// the manifest top level beside `silence_token_ids`, where they are
		// read below. Accepting them here too would give one value two homes
		// in one file; Python's block reader refuses them the same way.
		for _, key := range []string{"silence_render_ids", "quiet_render_ids"} {
			if _, present := v[key]; present {
				return AlgorithmConfig{}, fmt.Errorf(
					"manifest['postprocess'][%q] belongs at the manifest top "+
						"level, beside 'silence_token_ids'", key)
			}
		}
		ppMode, err := stringKey(v, "manifest['postprocess']", "mode", pp.Mode)
		if err != nil {
			return AlgorithmConfig{}, err
		}
		if ppMode != postprocess.ModeOff && ppMode != postprocess.ModeReport &&
			ppMode != postprocess.ModeTrim {
			return AlgorithmConfig{}, fmt.Errorf(
				"manifest declares unknown postprocess mode %q; expected one of %s, %s, %s",
				ppMode, postprocess.ModeOff, postprocess.ModeReport, postprocess.ModeTrim)
		}
		pp.Mode = ppMode
		ppNum := nums.in(v, "manifest['postprocess']")
		ppInt := nums.count(v, "manifest['postprocess']")
		pp.CeilingSpeechPerTextToken = ppNum("ceiling_speech_per_text_token",
			pp.CeilingSpeechPerTextToken)
		pp.CeilingSlackTokens = ppInt("ceiling_slack_tokens", pp.CeilingSlackTokens)
		pp.TrailingFillerThreshold = ppNum("trailing_filler_threshold",
			pp.TrailingFillerThreshold)
		pp.TrailingSilenceRunTokens = ppInt("trailing_silence_run_tokens", pp.TrailingSilenceRunTokens)
		pp.DesperationBandRatio = ppNum("desperation_band_ratio",
			pp.DesperationBandRatio)
		pp.DesperationBandFloor = ppInt("desperation_band_floor", pp.DesperationBandFloor)
		pp.FillerMinEosProbability = ppNum("filler_min_eos_probability",
			pp.FillerMinEosProbability)
		pp.FillerMaxSpeechAfterRun = ppInt("filler_max_speech_after_run", pp.FillerMaxSpeechAfterRun)
		pp.DesperationSpeechPerTextToken = ppNum("desperation_speech_per_text_token",
			pp.DesperationSpeechPerTextToken)
		pp.DesperationMinTextTokens = ppInt("desperation_min_text_tokens", pp.DesperationMinTextTokens)
		pp.DesperationMinKeepPerTextToken = ppNum("desperation_min_keep_per_text_token",
			pp.DesperationMinKeepPerTextToken)
		pp.EndedTailSilenceRun = ppInt("ended_tail_silence_run", pp.EndedTailSilenceRun)
		pp.EndedTailBlipMax = ppInt("ended_tail_blip_max", pp.EndedTailBlipMax)
		pp.EndedTailWordMax = ppInt("ended_tail_word_max", pp.EndedTailWordMax)
		pp.EndedTailKeep = ppInt("ended_tail_keep", pp.EndedTailKeep)
		pp.EchoStrongEosProbability = ppNum("echo_strong_eos_probability",
			pp.EchoStrongEosProbability)
		pp.EchoStrongMaxTail = ppInt("echo_strong_max_tail", pp.EchoStrongMaxTail)
		pp.EchoStrongMinPositionPct = ppInt("echo_strong_min_position_pct", pp.EchoStrongMinPositionPct)
		pp.EchoWeakEosProbability = ppNum("echo_weak_eos_probability",
			pp.EchoWeakEosProbability)
		pp.EchoWeakMaxTail = ppInt("echo_weak_max_tail", pp.EchoWeakMaxTail)
		pp.EchoWeakMinPositionPct = ppInt("echo_weak_min_position_pct", pp.EchoWeakMinPositionPct)
		// Python reads its fields off the dataclass precisely so a new
		// constant cannot be left out of a hand-written list; the four ports
		// write the list by hand, so a field missing here is invisible while
		// the defaults match and becomes a manifest declaring one recipe while
		// the engine runs another the moment a checkpoint sets it.
		pp.DropoutMinTokens = ppInt("dropout_min_tokens", pp.DropoutMinTokens)
		pp.RetryMaxAttempts = ppInt("retry_max_attempts", pp.RetryMaxAttempts)
		pp.PacingTolerance = ppNum("pacing_tolerance", pp.PacingTolerance)
		pp.RepetitionMaxPeriod = ppInt("repetition_max_period", pp.RepetitionMaxPeriod)
		pp.RepetitionMinCycles = ppInt("repetition_min_cycles", pp.RepetitionMinCycles)
		pp.RepetitionMinSpan = ppInt("repetition_min_span", pp.RepetitionMinSpan)
		// A string field like mode, and refused like mode: a law this port
		// does not implement must not fall back to a default, since the
		// resolver would cut where the manifest said to condemn.
		resume, err := stringKey(v, "manifest['postprocess']", "repetition_resume",
			pp.RepetitionResume)
		if err != nil {
			return AlgorithmConfig{}, err
		}
		if resume != postprocess.RepetitionResumeCondemn &&
			resume != postprocess.RepetitionResumeCut {
			return AlgorithmConfig{}, fmt.Errorf(
				"manifest declares unknown repetition_resume %q; expected %s or %s",
				resume, postprocess.RepetitionResumeCondemn,
				postprocess.RepetitionResumeCut)
		}
		pp.RepetitionResume = resume
		// A string field like mode, and refused like mode: a family this
		// port does not implement must not fall back to a default, since the
		// loop exemption would read one silence list under a manifest declaring
		// another.
		silence, err := stringKey(v, "manifest['postprocess']", "repetition_silence",
			pp.RepetitionSilence)
		if err != nil {
			return AlgorithmConfig{}, err
		}
		if silence != postprocess.RepetitionSilenceAcoustic &&
			silence != postprocess.RepetitionSilenceSampling {
			return AlgorithmConfig{}, fmt.Errorf(
				"manifest declares unknown repetition_silence %q; expected %s or %s",
				silence, postprocess.RepetitionSilenceAcoustic,
				postprocess.RepetitionSilenceSampling)
		}
		pp.RepetitionSilence = silence
		pp.StallRunTokens = ppInt("stall_run_tokens", pp.StallRunTokens)
		// The reference validates its ranges in `__post_init__`, so a manifest
		// naming an impossible constant never becomes a config there. Reading
		// every number and asking nothing of it let `ceiling_slack_tokens: -1`
		// widen the ceiling and `stall_run_tokens: 0` call every row with one
		// leading silence token a stall. Checked after the whole block is read,
		// because three of the rules compare two fields.
		if err := pp.Validate(); err != nil {
			return AlgorithmConfig{}, err
		}
	}
	// The render censuses: which ids actually render as digital silence
	// (`silence_render_ids`) and which as contextually-quiet breath/decay
	// (`quiet_render_ids`). Optional: a checkpoint packed before the census
	// has neither, and the stall detector then falls back to
	// `silence_token_ids`, degraded but safe. Read here rather than in the
	// postprocess block because they are properties of the weights, like
	// `silence_token_ids`, not detector constants someone tuned.
	if v, err := idList(m, "silence_render_ids"); err != nil {
		return AlgorithmConfig{}, err
	} else {
		pp.SilenceRenderIds = append(pp.SilenceRenderIds, v...)
	}
	if v, err := idList(m, "quiet_render_ids"); err != nil {
		return AlgorithmConfig{}, err
	} else {
		pp.QuietRenderIds = append(pp.QuietRenderIds, v...)
	}

	// A rate is the one manifest field whose wrongness no shape catches: every
	// duration this engine reports is samples / sample_rate, so a zero divides
	// by zero and a negative reports negative seconds. Defaulted to 24 000 for
	// a manifest that omits it, as Python defaults it, then checked.
	sampleRate, err := intKey(m, "sample_rate", def.SampleRate)
	if err != nil {
		return AlgorithmConfig{}, err
	}
	maxNew := samplingCount("max_new_tokens", def.Sampling.MaxNewTokens)
	recipe, err := recipeVersion(m)
	if err != nil {
		return AlgorithmConfig{}, err
	}

	// An explicit null is the one place absence is spelled out: a manifest
	// older than the field and one that writes `edge_fade_seconds: null` both
	// mean the 5 ms those releases shipped. Anything else that is not a number
	// is refused, as edge_fade_from refuses it.
	fade := def.EdgeFadeSeconds
	if raw, present := m["edge_fade_seconds"]; present && raw != nil {
		if fade, err = number(m, "edge_fade_seconds", fade); err != nil {
			return AlgorithmConfig{}, err
		}
	}
	guidanceRate, err := number(m, "guidance_rate", def.GuidanceRate)
	if err != nil {
		return AlgorithmConfig{}, err
	}
	steps, err := intKey(m, "n_cfm_timesteps", def.EulerSteps)
	if err != nil {
		return AlgorithmConfig{}, err
	}
	tokenRate, err := number(m, "token_rate_hz", def.TokenRateHz)
	if err != nil {
		return AlgorithmConfig{}, err
	}
	vocab, err := intKey(m, "speech_vocab_size", def.SpeechVocabSize)
	if err != nil {
		return AlgorithmConfig{}, err
	}
	grid, err := eulerGrid(m)
	if err != nil {
		return AlgorithmConfig{}, err
	}

	startSpeech := speechCount("start", def.StartSpeech)
	stopSpeech := speechCount("stop", def.StopSpeech)
	temperature := samplingNum("temperature", def.Sampling.Temperature)
	repetitionPenalty := samplingNum("repetition_penalty", def.Sampling.RepetitionPenalty)
	minP := samplingNum("min_p", def.Sampling.MinP)

	// Every numeric sub-key has now been read, so the first one that was not a
	// number is reported before any of the defaults it stood in for reaches a
	// config or the fingerprint over it.
	if nums.err != nil {
		return AlgorithmConfig{}, nums.err
	}

	cfg := AlgorithmConfig{
		DecodeMode:      mode,
		EdgeFadeSeconds: fade,
		RecipeVersion:   recipe,
		Text:            TextConfig{Recipe: TextRecipe, Grammar: GrammarDigest()},
		Guidance:        guidance,
		GuidanceRate:    guidanceRate,
		EulerSteps:      steps,
		EulerGrid:       grid,
		TokenRateHz:     tokenRate,
		SampleRate:      sampleRate,
		SpeechVocabSize: vocab,
		StartSpeech:     startSpeech,
		StopSpeech:      stopSpeech,
		Sampling: SamplingConfig{
			Temperature:        temperature,
			RepetitionPenalty:  repetitionPenalty,
			MinP:               minP,
			MaxNewTokens:       maxNew,
			SilenceTokenIds:    sil,
			MinTokensFloor:     eosFloorValue,
			MinTokensTextRatio: eosRatioValue,
		},
		Window:      win,
		Chunking:    chunk,
		Postprocess: pp,
	}
	// Every shape has been settled; what is left is whether the values make
	// an algorithm. Checked here, on the public reader, so that a caller
	// reading a manifest without building an engine gets the same refusals
	// as one that does: chunking.Validate carries the reference's wording,
	// and the sampling, window, grid and rate rules follow it.
	if err := cfg.Validate(); err != nil {
		return AlgorithmConfig{}, err
	}
	return cfg, nil
}

// asFloat is the one numeric reading of a decoded JSON value.
//
// encoding/json decodes every JSON number into float64; the int case is for a
// manifest assembled in Go rather than parsed. Everything else has no numeric
// reading at all, which the second return says rather than answering zero.
func asFloat(x interface{}) (float64, bool) {
	switch v := x.(type) {
	case float64:
		return v, true
	case int:
		return float64(v), true
	}
	return 0, false
}

// numbers reads the numeric keys of a manifest block, refusing any value that
// is not a number.
//
// Answering zero for a value that has no numeric reading hands the engine a
// setting the manifest never declared: `temperature: "0.8"` read as zero is
// greedy decoding under a manifest that asked for 0.8, and it is hashed into
// the fingerprint that is meant to catch exactly that. algorithm_from converts
// each of these with int() or float(), which refuse a null, a list and a
// non-numeric string, and a manifest one port reads while another mis-reads is
// the divergence class this library exists to prevent.
//
// The first refusal is remembered rather than returned, so the forty reads in
// FromManifest stay expressions and are checked once. Reads after a refusal
// answer their default and are discarded with the config.
type numbers struct{ err error }

// whole is the one integrality reading, for a manifest key that counts things:
// a window length, a token id, a token budget, a step count.
//
// A count of 2.7 is not a value that got rounded, it is a value that was
// computed wrong, and truncating it to 2 hides the arithmetic that produced it
// behind a chunker that breathes in a different place, under a recipe_version
// saying the five implementations agree. manifest._int refuses it in this
// sentence.
//
// Not the float fields. A temperature, a token rate, a guidance rate and a
// fade length take 2.7 as a value, and a check applied to them would refuse a
// manifest that is correct.
func whole(v float64, where string) (int, error) {
	if v != math.Trunc(v) {
		return 0, fmt.Errorf("%s must be a whole number, got %v", where, v)
	}
	return int(v), nil
}

// whereKey names one sub-key of a block the way a refusal quotes it back, as
// manifest._where spells the same path.
func whereKey(where, name string) string { return fmt.Sprintf("%s['%s']", where, name) }

// in binds one block and the path that names it in a refusal, so a caller
// reads in(sampling, "manifest['sampling_defaults']")("temperature", 0.8).
//
// Absent is not zero: only a missing key takes the default, so a manifest
// declaring `min_p: 0` (no truncation, a legal and meaningful setting) keeps
// its zero, and so does a deliberately disabled EOS floor.
func (n *numbers) in(m map[string]interface{}, where string) func(string, float64) float64 {
	return func(key string, def float64) float64 {
		raw, present := m[key]
		if !present {
			return def
		}
		if v, ok := asFloat(raw); ok {
			return v
		}
		n.refuse(fmt.Errorf("%s[%q] should be a number, got %#v", where, key, raw))
		return def
	}
}

// count binds one block the way in does, for the keys that hold a count rather
// than a rate or a threshold. The reference reads the two through _number and
// _int, and the list of which key is which is the same one in five languages.
func (n *numbers) count(m map[string]interface{}, where string) func(string, int) int {
	read := n.in(m, where)
	return func(key string, def int) int {
		v, err := whole(read(key, float64(def)), whereKey(where, key))
		if err != nil {
			n.refuse(err)
			return def
		}
		return v
	}
}

// opt reads the optional count keys of a block: a whole number, an explicit
// null for the unset reading, or a refusal for anything else.
//
// `in` has no null reading because the values it reads have none: a
// temperature is a number or it is a mistake. A window length is a number or
// the ragged reading, and the reference spells that with one `opt` helper that
// reads an absent key and an explicit null the same way.
func (n *numbers) opt(m map[string]interface{}, where string) func(string, *int) *int {
	return func(key string, def *int) *int {
		raw, present := m[key]
		if !present {
			return def
		}
		if raw == nil {
			return nil
		}
		if v, ok := asFloat(raw); ok {
			w, err := whole(v, whereKey(where, key))
			if err != nil {
				n.refuse(err)
				return def
			}
			return intp(w)
		}
		n.refuse(fmt.Errorf("%s[%q] should be a number or null, got %#v", where, key, raw))
		return def
	}
}

// refuse keeps the first refusal: the later ones are consequences of reading
// on past it, and the first names the key a reader has to fix.
func (n *numbers) refuse(err error) {
	if n.err == nil {
		n.err = err
	}
}

// number reads a top-level numeric key, refusing a value that is not a number.
//
// manifest._number refuses the same way. The value that comes back here is
// hashed into the fingerprint that is supposed to catch a divergence, so a
// key silently read as zero is a divergence the fingerprint records rather
// than reports.
func number(m map[string]interface{}, key string, def float64) (float64, error) {
	raw, present := m[key]
	if !present {
		return def, nil
	}
	if v, ok := asFloat(raw); ok {
		return v, nil
	}
	return 0, fmt.Errorf("manifest[%q] should be a number, got %#v", key, raw)
}

// intKey reads a top-level key that holds a count. See whole.
func intKey(m map[string]interface{}, key string, def int) (int, error) {
	v, err := number(m, key, float64(def))
	if err != nil {
		return 0, err
	}
	return whole(v, fmt.Sprintf("manifest['%s']", key))
}

// block reads a top-level object key, refusing a value that is not one. An
// absent key gives an empty block, so every reader below it finds its default.
func block(m map[string]interface{}, key string) (map[string]interface{}, error) {
	raw, present := m[key]
	if !present {
		return map[string]interface{}{}, nil
	}
	v, ok := raw.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("manifest key %q must be an object, got %#v", key, raw)
	}
	return v, nil
}

// idList reads a top-level array of token ids, refusing anything else.
//
// A string is refused by name because it is a Sequence in the reference too,
// and `silence_token_ids: "123"` must not load as three tokens. An element
// that is not a number is refused for the reason a whole list is: the
// reference builds the census with int() per element, and an element read as
// zero enters both the silence set and the fingerprint as token 0.
func idList(m map[string]interface{}, key string) ([]int, error) {
	raw, present := m[key]
	if !present {
		return nil, nil
	}
	v, ok := raw.([]interface{})
	if !ok {
		return nil, fmt.Errorf("manifest key %q must be a list, got %#v", key, raw)
	}
	out := make([]int, 0, len(v))
	for i, x := range v {
		id, ok := asFloat(x)
		if !ok {
			return nil, fmt.Errorf("manifest[%q][%d] should be a number, got %#v", key, i, x)
		}
		w, err := whole(id, fmt.Sprintf("manifest['%s'][%d]", key, i))
		if err != nil {
			return nil, err
		}
		out = append(out, w)
	}
	return out, nil
}

// windowFrom reads the window block.
//
// `window: null` is the ragged window said out loud, and an absent block is the
// same window said by saying nothing. Any other non-object is refused rather
// than ignored: a manifest that names a window this port cannot read should not
// render under the shipped static one.
//
// The sub-keys are refused for the same reason one level down, through the
// same reader the sampling and postprocess blocks use. A `static_length` that
// has no numeric reading and is ignored pads nothing under a manifest that
// asked for a padded window, and the canonical form then records the ragged
// window it fell back to, so the fingerprint agrees with the misreading
// instead of reporting it. That is the whole divergence class this library
// exists to prevent, and the window recipe is the one it was measured on.
//
// Only the three lengths have a null reading, and it is the ragged one:
// _window_from reads them through an `opt` that spells an absent key and an
// explicit null the same way, and reads max_speech_tokens with a bare int()
// that refuses a null like any other non-number.
func windowFrom(m map[string]interface{}, nums *numbers, win WindowConfig) (WindowConfig, error) {
	raw, present := m["window"]
	if !present || raw == nil {
		return win, nil
	}
	v, ok := raw.(map[string]interface{})
	if !ok {
		return WindowConfig{}, fmt.Errorf(
			"manifest['window'] must be an object or null (ragged), got %#v", raw)
	}
	winCount := nums.count(v, "manifest['window']")
	winOpt := nums.opt(v, "manifest['window']")
	win.MaxSpeechTokens = winCount("max_speech_tokens", win.MaxSpeechTokens)
	win.StaticLength = winOpt("static_length", win.StaticLength)
	win.StaticPromptTokens = winOpt("static_prompt_tokens", win.StaticPromptTokens)
	win.PadTokenID = winOpt("pad_token_id", win.PadTokenID)
	return win, nil
}

// stringKey reads a string-valued key, refusing a present value that is not a
// string. An absent key answers the default, as everywhere in this reader.
//
// Every caller below names a closed set and checks the answer against it; this
// settles only the shape, so the "unknown mode" refusals keep naming their own
// options. A present key skipped for having the wrong type would be a law the
// manifest declared and this port did not run, with the canonical form
// recording the default it fell back to, so the fingerprint would agree with
// the misreading instead of reporting it. The reference arrives at the same
// refusal from the other side: str() renders the value and the membership
// check fails on what it rendered.
func stringKey(m map[string]interface{}, where, key, def string) (string, error) {
	raw, present := m[key]
	if !present {
		return def, nil
	}
	s, ok := raw.(string)
	if !ok {
		return "", fmt.Errorf("%s[%q] must be a string, got %#v", where, key, raw)
	}
	return s, nil
}

// stringList reads a list of strings, refusing a value that is not a list and
// an element that is not a string.
//
// An element dropped for having the wrong type is a separator or an
// abbreviation the manifest declared and the splitter never saw, so the text
// breaks somewhere else and the canonical form records the shorter list as if
// it were what the manifest said. The reference renders each element with
// str() instead, which is a coercion this port does not copy, for the reason
// asFloat does not coerce a numeric string.
func stringList(raw interface{}, where string) ([]string, error) {
	list, ok := raw.([]interface{})
	if !ok {
		return nil, fmt.Errorf("manifest key %q must be a list of strings, got %#v", where, raw)
	}
	out := make([]string, 0, len(list))
	for i, x := range list {
		s, ok := x.(string)
		if !ok {
			return nil, fmt.Errorf("manifest key %q[%d] must be a string, got %#v", where, i, x)
		}
		out = append(out, s)
	}
	return out, nil
}

// eulerGrid reads an explicit time grid, or nil for the cosine schedule.
//
// Absent and null are the cosine schedule; anything that is not a list is
// refused rather than ignored, and so is a point that is not a number. An
// ignored grid runs the cosine schedule under a manifest that declared an
// explicit one, and the canonical form then records the null it fell back to,
// so the fingerprint agrees with the reading rather than reporting it.
// manifest.algorithm_from refuses this key by name, with this sentence.
func eulerGrid(m map[string]interface{}) ([]float64, error) {
	raw, ok := m["euler_grid"]
	if !ok || raw == nil {
		return nil, nil
	}
	list, ok := raw.([]interface{})
	if !ok {
		return nil, fmt.Errorf(
			"manifest['euler_grid'] must be a list of floats or null, got %#v", raw)
	}
	grid := make([]float64, 0, len(list))
	for i, v := range list {
		t, ok := asFloat(v)
		if !ok {
			return nil, fmt.Errorf(
				"manifest['euler_grid'][%d] should be a number, got %#v", i, v)
		}
		grid = append(grid, t)
	}
	return grid, nil
}

// RecipeVersion is the one recipe. There is no other, and nothing predates it.
const RecipeVersion = "loudkit-1"

// recipeVersion reads the tag the manifest carries, or names the one recipe
// this library has when the manifest is silent. One recipe means one accepted
// value: a foreign tag believed here would ride into every fingerprint, so it
// is refused with the value named. Absence is not a tag: a manifest that
// omits the key left a shipping default unstated.
func recipeVersion(m map[string]interface{}) (string, error) {
	raw, ok := m["recipe_version"]
	if !ok {
		return RecipeVersion, nil
	}
	if s, ok := raw.(string); ok && s == RecipeVersion {
		return s, nil
	}
	return "", fmt.Errorf(
		"manifest declares recipe_version %#v; the only recipe is %q", raw, RecipeVersion)
}

// EosFloor is the minimum speech tokens before the stop token becomes
// sampleable.
func EosFloor(nTextTokens int, cfg AlgorithmConfig) int {
	f := cfg.Sampling.MinTokensFloor
	r := int(float64(nTextTokens) * cfg.Sampling.MinTokensTextRatio)
	if r > f {
		return r
	}
	return f
}

// TextConfig identifies the text funnel: what its code does, and what data it
// reads. See loudkit/frontend/textconfig.py: the digest is of this port's own
// copy of numbers.json, so a copy that has drifted from the reference produces
// a different fingerprint and the engine refuses to start rather than silently
// speaking something else.
type TextConfig struct {
	Recipe  string
	Grammar string
}

// TextRecipe is the funnel's code version, bumped when the passes change what
// they emit for text they already handled. A new language or a new table moves
// Grammar on its own and needs no bump here.
const TextRecipe = "funnel-6"

// GrammarDigest is the first 16 hex characters of the SHA-256 of the embedded
// numbers.json, then pl_en_respell.json, then numerals.json: computed once,
// from the bytes this binary actually carries.
//
// The lexicon is hashed alongside the grammar because it is a funnel input
// exactly as the grammar is and it changes the spoken tokens, so both files
// hash into the fingerprint. Leaving the lexicon out covers 55 KB of rules but
// not 6.5 MB of vocabulary, and a build whose lexicon has drifted says
// different words under the same sixteen hex digits.
//
// The numeral table is the third for the same reason and one more: it decides
// both what a numeral reads as and whether a character is a numeral at all, so
// it pins the Unicode version of the fold. Two builds hashing the same three
// files fold identically whatever their runtimes' own tables say.
//
// Every byte in these files is data the funnel reads, which is why the
// descriptions live beside the data (numbers.about.md,
// numerals.provenance.json) rather than inside it: a prose member in a hashed
// file moves this digest and the fingerprint above it while every sample
// renders identically. Neither of those two is hashed or embedded here.
func GrammarDigest() string {
	grammarOnce.Do(func() {
		h := sha256.New()
		h.Write(speechtext.GrammarBytes())
		h.Write(speechtext.RespellBytes())
		// And the numeral table, for the same reason: it decides the words a
		// numeral becomes, and it pins the Unicode version the fold uses.
		h.Write(speechtext.NumeralBytes())
		grammarDigest = hex.EncodeToString(h.Sum(nil))[:16]
	})
	return grammarDigest
}

var (
	grammarOnce   sync.Once
	grammarDigest string
)

// EdgeFade is the ramp length the engine applies: the manifest's value, or the
// shipped length when the config never named one. Python refuses a ramp under
// one millisecond, so zero is never a real value and can stand for unset.
func (c AlgorithmConfig) EdgeFade() float64 {
	if c.EdgeFadeSeconds == 0 {
		return timestretchEdgeFadeSeconds
	}
	return c.EdgeFadeSeconds
}
