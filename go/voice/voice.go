// Package voice loads a VoiceProfile (mirror of loudkit.voice).
package voice

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/loudreader/loudkit/go/safetensors"
)

// Profile is everything needed to speak as one voice.
type Profile struct {
	Name             string
	SpeakerEmbedding []float32
	FlowEmbedding    []float32
	PromptTokens     []int64
	PromptMel        []float32
	CondPromptTokens []int64
	SourceSampleRate int
	Language         string
}

const formatVersion = 1

// EmotionNeutral is the constant fed to the generator's emotion conditioning
// slot. The checkpoint reserves one of its 34 conditioning slots for an
// emotion scalar. On these weights the axis is dead (distillation collapsed
// it), so the slot is not a control and not part of the profile format — but
// it must be fed the value the model was distilled with. Every port uses this.
const EmotionNeutral = 0.5

// The two speaker encoders' output widths and the mel bin count. Mirrors
// loudkit.voice.VoiceProfile, which validates the same three.
const (
	speakerDim = 256
	flowDim    = 192
	melBins    = 80
)

// minEmbeddingNorm is the smallest speaker-vector norm a profile may carry.
//
// Below this the renderers stop agreeing: this port and CoreML divide by the
// raw norm and yield NaN, torch's F.normalize carries an epsilon and yields a
// finite — but arbitrary — direction. Enrolled vectors are order-1; anything
// this small is a corrupt or synthetic file, not a quiet voice.
const minEmbeddingNorm = 1e-6

// checkEmbedding rejects an embedding the renderers would disagree about.
//
// A profile is a file that gets copied, mailed and downloaded, so these checks
// belong at the boundary rather than in each backend. Python has validated
// them since the degenerate-profile fix; the ports accepted anything shaped
// like floats and blew up deeper in inference, where the error names a matrix
// rather than a file.
func checkEmbedding(name string, values []float32, expected int) error {
	if len(values) != expected {
		return fmt.Errorf("%s must be %d-d, got %d", name, expected, len(values))
	}
	var sum float64
	for _, v := range values {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("%s contains NaN or infinity", name)
		}
		sum += float64(v) * float64(v)
	}
	if norm := math.Sqrt(sum); norm < minEmbeddingNorm {
		return fmt.Errorf(
			"%s has norm %g, below %g: a zero or near-zero speaker vector normalises to "+
				"NaN here and to a finite arbitrary direction on torch, so the same file "+
				"would speak differently per backend", name, norm, minEmbeddingNorm)
	}
	return nil
}

// Load reads a voice profile file.
// The shipped model's dimensions, the same two Python reads out of
// `AlgorithmConfig`.
//
// Both ends, not just the floor. Without the ceiling a profile carrying
// `prompt_tokens =
// [9000]` loads cleanly here and then indexes past the end of the embedding
// table — an out-of-range panic, or a read of whatever follows it. `load`
// promises a profile is safe to open from an untrusted source, and a bound the
// renderer relies on has to be checked where that promise is made.
//
// The ceilings are the shipped model's, the same two Python takes from
// `AlgorithmConfig`: prompt tokens index the speech codebook below the
// start-of-speech marker, conditioning tokens the whole speech vocabulary.
const (
	startSpeechToken int64 = 6561
	speechVocabSize  int64 = 8194
)

// MaxVoiceBytes matches Python's `MAX_VOICE_BYTES`.
//
// Python has capped this since the reader was written and the other four never did: a voice profile is a handful of small tensors, and a safetensors file claiming otherwise is not one.
// The cap is on the file, before it is opened, because the shape checks that follow only run after a header has been parsed.
const MaxVoiceBytes = 8 * 1024 * 1024

func Load(path string) (*Profile, error) {
	if info, err := os.Stat(path); err == nil && info.Size() > MaxVoiceBytes {
		return nil, fmt.Errorf("%s: %d bytes, over the %d byte limit for a voice",
			filepath.Base(path), info.Size(), MaxVoiceBytes)
	}
	f, err := safetensors.Open(path)
	if err != nil {
		return nil, err
	}
	header := map[string]interface{}{}
	if s, ok := f.Metadata["voice"]; ok {
		if err := json.Unmarshal([]byte(s), &header); err != nil {
			return nil, fmt.Errorf("%s: bad voice header: %w", path, err)
		}
	}
	version := 0.0
	if v, ok := header["format_version"].(float64); ok {
		version = v
	}
	if int(version) != formatVersion {
		return nil, fmt.Errorf("%s: voice format version %d, this build reads %d",
			path, int(version), formatVersion)
	}
	speaker, err := f.F32("speaker_embedding")
	if err != nil {
		return nil, err
	}
	flow, err := f.F32("flow_embedding")
	if err != nil {
		return nil, err
	}
	promptTok, err := f.I64("prompt_tokens")
	if err != nil {
		return nil, err
	}
	promptMel, err := f.F32("prompt_mel")
	if err != nil {
		return nil, err
	}
	condTok, err := f.I64("cond_prompt_tokens")
	if err != nil {
		return nil, err
	}
	if err := checkEmbedding("speaker_embedding", speaker, speakerDim); err != nil {
		return nil, err
	}
	if err := checkEmbedding("flow_embedding", flow, flowDim); err != nil {
		return nil, err
	}
	for _, v := range promptMel {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("prompt_mel contains NaN or infinity")
		}
	}
	if len(promptMel)%melBins != 0 {
		return nil, fmt.Errorf("prompt_mel must be (%d, frames), got %d values",
			melBins, len(promptMel))
	}
	for _, pair := range []struct {
		name    string
		tokens  []int64
		ceiling int64
	}{
		{"prompt_tokens", promptTok, startSpeechToken},
		{"cond_prompt_tokens", condTok, speechVocabSize},
	} {
		for _, t := range pair.tokens {
			// Negative ids index an embedding table from the end — silently.
			if t < 0 {
				return nil, fmt.Errorf("%s contains a negative id: %d", pair.name, t)
			}
			if t >= pair.ceiling {
				return nil, fmt.Errorf("%s contains id %d, at or past the %d the model has",
					pair.name, t, pair.ceiling)
			}
		}
	}
	name, _ := header["name"].(string)
	if name == "" {
		name = "voice"
	}
	return &Profile{
		Name:             name,
		SpeakerEmbedding: speaker,
		FlowEmbedding:    flow,
		PromptTokens:     promptTok,
		PromptMel:        promptMel,
		CondPromptTokens: condTok,
		SourceSampleRate: headerInt(header, "source_sample_rate", 24000),
		Language:         headerString(header, "language", "en"),
	}, nil
}

func headerFloat(h map[string]interface{}, k string, def float64) float64 {
	if v, ok := h[k].(float64); ok {
		return v
	}
	return def
}

func headerInt(h map[string]interface{}, k string, def int) int {
	if v, ok := h[k].(float64); ok {
		return int(v)
	}
	return def
}

func headerString(h map[string]interface{}, k, def string) string {
	if v, ok := h[k].(string); ok {
		return v
	}
	return def
}
