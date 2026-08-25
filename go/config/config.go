// Package config mirrors loudkit.config — the algorithm values that are
// identical on every backend, read from the checkpoint manifest.
package config

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
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

// AlgorithmConfig is everything that determines what the engine produces.
type AlgorithmConfig struct {
	RecipeVersion string
	Guidance      string
	GuidanceRate  float64
	EulerSteps    int
	// EulerGrid is the explicit time grid, or nil for the cosine schedule.
	//
	// An explicit grid overrides the cosine schedule.
	//
	// `config.py:296` gives the reason: "An explicit grid is preferred for anything
	// that must match across implementations, because 'cosine' is a formula two
	// codebases can write two ways." Without this field, a
	// checkpoint shipping an explicit grid integrates on a different
	// schedule here — silently, and under a fingerprint that records the grid it is
	// ignoring. The shipping manifest has `euler_grid: null`.
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
	// Text is the funnel's identity — its code version and the digest of the
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

func intp(v int) *int           { return &v }
func floatp(v float64) *float64 { return &v }

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
	// Kept as interface{} so `pick` can tell an absent key from an explicit
	// zero; flattening to float64 first made them indistinguishable.
	sampling := map[string]interface{}{}
	if v, ok := m["sampling_defaults"].(map[string]interface{}); ok {
		sampling = v
	}
	sil := []int{}
	if v, ok := m["silence_token_ids"].([]interface{}); ok {
		for _, x := range v {
			sil = append(sil, int(toFloat(x)))
		}
	}
	speech := map[string]float64{}
	if v, ok := m["speech_tokens"].(map[string]interface{}); ok {
		for k, x := range v {
			speech[k] = toFloat(x)
		}
	}

	win := ProductionWindow()
	if v, ok := m["window"].(map[string]interface{}); ok {
		if x, ok := v["max_speech_tokens"].(float64); ok {
			win.MaxSpeechTokens = int(x)
		}
		if x, ok := v["static_length"].(float64); ok {
			win.StaticLength = intp(int(x))
		}
		if x, ok := v["static_prompt_tokens"].(float64); ok {
			win.StaticPromptTokens = intp(int(x))
		}
		if x, ok := v["pad_token_id"].(float64); ok {
			win.PadTokenID = intp(int(x))
		}
	}

	// Defaults match the Python loader: absent means "the production floor",
	// present-and-zero means the manifest disabled it deliberately.
	eosFloorValue := 10.0
	eosRatioValue := 1.2
	if v, ok := m["eos_floor"].(map[string]interface{}); ok {
		eosFloorValue = pick(v, "min_tokens_floor", eosFloorValue)
		eosRatioValue = pick(v, "min_tokens_text_ratio", eosRatioValue)
	}

	guidance := "single_path"
	if raw, present := m["guidance"]; present {
		guidance = toString(raw)
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
				"not implement — it would render single-path audio and silently " +
				"disagree with the Python engine")
	}

	chunk := chunking.Production()
	if v, ok := m["chunking"].(map[string]interface{}); ok {
		if x, ok := v["enabled"].(bool); ok {
			chunk.Enabled = x
		}
		chunk.MaxTokens = int(pick(v, "max_tokens", float64(chunk.MaxTokens)))
		chunk.PrefixTokens = int(pick(v, "prefix_tokens", float64(chunk.PrefixTokens)))
		if raw, ok := v["split_on"].([]interface{}); ok {
			seps := make([]string, 0, len(raw))
			for _, x := range raw {
				if str, ok := x.(string); ok {
					seps = append(seps, str)
				}
			}
			if len(seps) > 0 {
				chunk.SplitOn = seps
			}
		}
	}

	pp := postprocess.Production()
	if v, ok := m["postprocess"].(map[string]interface{}); ok {
		if s, ok := v["mode"].(string); ok {
			if s != postprocess.ModeOff && s != postprocess.ModeReport &&
				s != postprocess.ModeTrim {
				return AlgorithmConfig{}, fmt.Errorf(
					"manifest declares unknown postprocess mode %q; expected one of %s, %s, %s",
					s, postprocess.ModeOff, postprocess.ModeReport, postprocess.ModeTrim)
			}
			pp.Mode = s
		}
		pp.CeilingSpeechPerTextToken = pick(v, "ceiling_speech_per_text_token",
			pp.CeilingSpeechPerTextToken)
		pp.CeilingSlackTokens = int(pick(v, "ceiling_slack_tokens",
			float64(pp.CeilingSlackTokens)))
		pp.TrailingFillerThreshold = pick(v, "trailing_filler_threshold",
			pp.TrailingFillerThreshold)
		pp.TrailingSilenceRunTokens = int(pick(v, "trailing_silence_run_tokens",
			float64(pp.TrailingSilenceRunTokens)))
		pp.DesperationBandRatio = pick(v, "desperation_band_ratio",
			pp.DesperationBandRatio)
		pp.DesperationBandFloor = int(pick(v, "desperation_band_floor",
			float64(pp.DesperationBandFloor)))
		pp.FillerMinEosProbability = pick(v, "filler_min_eos_probability",
			pp.FillerMinEosProbability)
		pp.FillerMaxSpeechAfterRun = int(pick(v, "filler_max_speech_after_run",
			float64(pp.FillerMaxSpeechAfterRun)))
		pp.DesperationSpeechPerTextToken = pick(v, "desperation_speech_per_text_token",
			pp.DesperationSpeechPerTextToken)
		pp.DesperationMinTextTokens = int(pick(v, "desperation_min_text_tokens",
			float64(pp.DesperationMinTextTokens)))
		pp.EndedTailSilenceRun = int(pick(v, "ended_tail_silence_run",
			float64(pp.EndedTailSilenceRun)))
		pp.EndedTailBlipMax = int(pick(v, "ended_tail_blip_max",
			float64(pp.EndedTailBlipMax)))
		pp.EndedTailWordMax = int(pick(v, "ended_tail_word_max",
			float64(pp.EndedTailWordMax)))
		pp.EndedTailKeep = int(pick(v, "ended_tail_keep", float64(pp.EndedTailKeep)))
		pp.EchoStrongEosProbability = pick(v, "echo_strong_eos_probability",
			pp.EchoStrongEosProbability)
		pp.EchoStrongMaxTail = int(pick(v, "echo_strong_max_tail",
			float64(pp.EchoStrongMaxTail)))
		pp.EchoStrongMinPositionPct = int(pick(v, "echo_strong_min_position_pct",
			float64(pp.EchoStrongMinPositionPct)))
		pp.EchoWeakEosProbability = pick(v, "echo_weak_eos_probability",
			pp.EchoWeakEosProbability)
		pp.EchoWeakMaxTail = int(pick(v, "echo_weak_max_tail",
			float64(pp.EchoWeakMaxTail)))
		pp.EchoWeakMinPositionPct = int(pick(v, "echo_weak_min_position_pct",
			float64(pp.EchoWeakMinPositionPct)))
		// The six this wall was missing. Python reads its fields off the
		// dataclass precisely so a new constant cannot be left out of a
		// hand-written list; the four ports write the list by hand, and every
		// one of them had drifted the same six fields behind. Defaults
		// matched, so nothing sounded wrong — until a checkpoint sets one, at
		// which point the manifest declares one recipe and four engines run
		// another.
		pp.DropoutMinTokens = int(pick(v, "dropout_min_tokens", float64(pp.DropoutMinTokens)))
		pp.RetryMaxAttempts = int(pick(v, "retry_max_attempts", float64(pp.RetryMaxAttempts)))
		pp.PacingTolerance = pick(v, "pacing_tolerance", pp.PacingTolerance)
		pp.RepetitionMaxPeriod = int(pick(v, "repetition_max_period",
			float64(pp.RepetitionMaxPeriod)))
		pp.RepetitionMinCycles = int(pick(v, "repetition_min_cycles",
			float64(pp.RepetitionMinCycles)))
		pp.RepetitionMinSpan = int(pick(v, "repetition_min_span", float64(pp.RepetitionMinSpan)))
	}

	// Python and Swift refuse a non-positive cap; Go, Rust and JS took it and
	// decoded nothing, which reaches a caller as silence they have to
	// diagnose rather than an error they can read. A cap of zero is not a
	// configuration, it is a typo in a manifest.
	// Python refuses a manifest with a non-positive `sample_rate` and the other four
	// took it: every duration this engine reports is `samples / sample_rate`, so a
	// zero divides by zero and a negative reports negative seconds. A rate is the one
	// manifest field whose wrongness is not caught by any shape.
	// Defaulted, then checked. Python, JS and Swift all fall back to 24 000
	// for a manifest that omits the field and this port fell back to zero —
	// `toFloat` on a missing key — so an older pack loaded with a sample rate
	// that makes every duration a division by zero.
	sampleRate := int(pick(m, "sample_rate", 24_000))
	if sampleRate <= 0 {
		return AlgorithmConfig{}, fmt.Errorf("sample_rate must be > 0: %d", sampleRate)
	}
	if maxNew := int(pick(sampling, "max_new_tokens", 255)); maxNew <= 0 {
		return AlgorithmConfig{}, fmt.Errorf("max_new_tokens must be positive: %d", maxNew)
	}
	recipe, err := recipeVersion(m)
	if err != nil {
		return AlgorithmConfig{}, err
	}

	return AlgorithmConfig{
		RecipeVersion:   recipe,
		Text:            TextConfig{Recipe: TextRecipe, Grammar: GrammarDigest()},
		Guidance:        guidance,
		GuidanceRate:    toFloat(m["guidance_rate"]),
		EulerSteps:      int(toFloat(m["n_cfm_timesteps"])),
		EulerGrid:       eulerGrid(m),
		TokenRateHz:     pick(m, "token_rate_hz", 25.0),
		SampleRate:      sampleRate,
		SpeechVocabSize: int(toFloat(m["speech_vocab_size"])),
		StartSpeech:     int(speech["start"]),
		StopSpeech:      int(speech["stop"]),
		Sampling: SamplingConfig{
			Temperature:        pick(sampling, "temperature", 0.8),
			RepetitionPenalty:  pick(sampling, "repetition_penalty", 1.2),
			MinP:               pick(sampling, "min_p", 0.05),
			MaxNewTokens:       int(pick(sampling, "max_new_tokens", 255)),
			SilenceTokenIds:    sil,
			MinTokensFloor:     int(eosFloorValue),
			MinTokensTextRatio: eosRatioValue,
		},
		Window:      win,
		Chunking:    chunk,
		Postprocess: pp,
	}, nil
}

func toFloat(x interface{}) float64 {
	switch v := x.(type) {
	case float64:
		return v
	case int:
		return float64(v)
	case nil:
		return 0
	}
	return 0
}

// pick reads a numeric field, distinguishing "absent" from "explicitly zero".
//
// The previous helper took the already-extracted value and treated 0 as
// missing, so a manifest declaring `min_p: 0` (no truncation — a legal and
// meaningful setting) silently got 0.05, and a deliberately disabled EOS floor
// got 10. Zero is a value; only absence is absence.
func pick(m map[string]interface{}, key string, def float64) float64 {
	raw, present := m[key]
	if !present || raw == nil {
		return def
	}
	return toFloat(raw)
}

func toString(x interface{}) string {
	if s, ok := x.(string); ok {
		return s
	}
	return ""
}

// eulerGrid reads an explicit time grid, or nil for the cosine schedule.
//
// A JSON string is refused rather than iterated: Python's from_manifest guards
// this key by name, and a manifest one port misreads while another defaults is
// the divergence class this library exists to prevent.
func eulerGrid(m map[string]interface{}) []float64 {
	raw, ok := m["euler_grid"]
	if !ok || raw == nil {
		return nil
	}
	list, ok := raw.([]interface{})
	if !ok {
		return nil
	}
	grid := make([]float64, 0, len(list))
	for _, v := range list {
		grid = append(grid, toFloat(v))
	}
	return grid
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
// reads. See loudkit/frontend/textconfig.py — the digest is of this port's own
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
const TextRecipe = "funnel-2"

// GrammarDigest is the first 16 hex characters of the SHA-256 of the embedded
// numbers.json followed by pl_en_respell.json — computed once, from the bytes
// this binary actually carries.
//
// The lexicon is hashed alongside the grammar because it is a funnel input
// exactly as the grammar is and it changes the spoken tokens, so both files
// hash into the fingerprint. Leaving the lexicon out covers 55 KB of rules but
// not 6.5 MB of vocabulary, and a build whose lexicon has drifted says
// different words under the same sixteen hex digits.
func GrammarDigest() string {
	grammarOnce.Do(func() {
		h := sha256.New()
		h.Write(speechtext.GrammarBytes())
		h.Write(speechtext.RespellBytes())
		grammarDigest = hex.EncodeToString(h.Sum(nil))[:16]
	})
	return grammarDigest
}

var (
	grammarOnce   sync.Once
	grammarDigest string
)
