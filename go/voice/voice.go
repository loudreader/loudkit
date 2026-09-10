// Package voice loads a VoiceProfile (mirror of loudkit.voice).
package voice

import (
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/loudreader/loudkit/go/safetensors"
)

// Profile is everything needed to speak as one voice.
type Profile struct {
	// Enrolment names prompt preparation; empty means the legacy first-10s strategy.
	Enrolment        string
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
// it), so the slot is not a control and not part of the profile format: but
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
// finite (but arbitrary) direction. Enrolled vectors are order-1; anything
// this small is a corrupt or synthetic file, not a quiet voice.
const minEmbeddingNorm = 1e-6

// checkEmbedding rejects an embedding the renderers would disagree about.
//
// A profile is a file that gets copied, mailed and downloaded, so these checks
// belong at the boundary rather than in each backend: a vector that is merely
// shaped like floats passes every backend's own guard and fails inside
// inference, where the error names a matrix rather than the file that carried
// it. VoiceProfile._validate_values checks the same two.
func checkEmbedding(name string, values []float32, expected int) error {
	if len(values) != expected {
		return fmt.Errorf("%s must be %d-d, got %d", name, expected, len(values))
	}
	// Squaring a widened float32 is exact in float64, so whether the compiler
	// fuses this accumulation does not change the norm it reports.
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

// The token ceilings are the shipped model's, the same two Python takes from
// AlgorithmConfig: prompt tokens index the speech codebook below the
// start-of-speech marker, conditioning tokens the whole speech vocabulary.
//
// Both ends are checked, not only the floor. Without the ceiling a profile
// carrying prompt_tokens = [9000] loads cleanly here and then indexes past the
// end of the embedding table, which is an out-of-range panic or a read of
// whatever follows it. Load promises a profile is safe to open from an
// untrusted source, and a bound the renderer relies on has to be checked where
// that promise is made.
const (
	startSpeechToken int64 = 6561
	speechVocabSize  int64 = 8194
)

// MaxVoiceBytes matches Python's MAX_VOICE_BYTES.
//
// A voice profile is a handful of small tensors, and a safetensors file
// claiming otherwise is not one. The cap is on the file, before it is opened,
// because the shape checks that follow only run after a header has been
// parsed.
const MaxVoiceBytes = 8 * 1024 * 1024

// MaxNameChars matches Python's _MAX_NAME_CHARS.
//
// The name is metadata a caller supplies at enrolment and a server echoes back
// in responses; nothing downstream truncates it. Counted in runes, not bytes,
// because Python slices the string by code point and a byte cut would also
// split one.
const MaxNameChars = 200

// The prompt-preparation strategies this build implements, mirroring
// loudkit.voice.KNOWN_ENROLMENTS.
//
// firstWindowEnrolment is the original: the prompt is the first ten seconds of
// the clip, and every voice enrolled before the field existed was made this
// way, so a profile whose header names no strategy is read as this one.
// pauseCutEnrolment is what `loudkit clone` makes by default: the clip cut at
// its last pause before ten seconds, then 0.4 s of silence. A different prompt
// from the same recording, so a different label.
//
// Sorted, because the refusal below lists them and Python lists them sorted.
const (
	firstWindowEnrolment = "first-10s"
	pauseCutEnrolment    = "first-10s-pause"
)

var knownEnrolments = []string{firstWindowEnrolment, pauseCutEnrolment}

// Load reads a voice profile file and refuses one this build cannot render: a
// file over MaxVoiceBytes, a header that is not a JSON object or carries a
// field of the wrong JSON type, a format version this build does not read,
// wrong embedding dimensions, a prompt mel that is non-finite or the wrong
// shape, a token id outside the model's tables, an enrolment strategy this
// build does not implement, or a source sample rate that is not positive. An
// over-long name is truncated rather than refused, which is what Python does
// with it.
func Load(path string) (*Profile, error) {
	file := filepath.Base(path)
	if info, err := os.Stat(path); err == nil && info.Size() > MaxVoiceBytes {
		return nil, fmt.Errorf("%s: %d bytes, over the %d byte limit for a voice",
			file, info.Size(), MaxVoiceBytes)
	}
	f, err := safetensors.Open(path)
	if err != nil {
		return nil, err
	}
	header := map[string]interface{}{}
	if s, ok := f.Metadata["voice"]; ok {
		var raw any
		if err := json.Unmarshal([]byte(s), &raw); err != nil {
			return nil, fmt.Errorf("%s: bad voice header: %w", file, err)
		}
		// A JSON list, string, number or null answers nothing for every key,
		// so the file would be refused as version zero: a version it does not
		// carry, which points whoever wrote it at the wrong fault.
		object, ok := raw.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("%s: voice header is %s, expected a JSON object",
				file, jsonTypePhrase(raw))
		}
		header = object
	}
	version, err := headerInt(header, file, "format_version", 0)
	if err != nil {
		return nil, err
	}
	if version != formatVersion {
		return nil, fmt.Errorf("%s: voice format version %d, this build reads %d",
			file, version, formatVersion)
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
			// Negative ids index an embedding table from the end: silently.
			if t < 0 {
				return nil, fmt.Errorf("%s contains a negative id: %d", pair.name, t)
			}
			if t >= pair.ceiling {
				return nil, fmt.Errorf("%s contains id %d, at or past the %d the model has",
					pair.name, t, pair.ceiling)
			}
		}
	}
	// The file's own stem when the header names nothing, which is the name the
	// reference gives an unnamed profile: `alice.safetensors` is "alice" there
	// and was "voice" here, in the string a caller sees and a server echoes
	// back. A header that names the empty string keeps it; only the refusal
	// below stands "voice" in for a name there is none of.
	name, err := headerString(header, file, "name", strings.TrimSuffix(file, filepath.Ext(file)))
	if err != nil {
		return nil, err
	}
	if runes := []rune(name); len(runes) > MaxNameChars {
		name = string(runes[:MaxNameChars])
	}
	// A strategy this build does not implement is refused rather than read as
	// the default, because the two cut the prompt from different audio: the
	// profile would load and speak, in a voice this build never enrolled,
	// under the name the caller asked for. VoiceProfile._validate_enrolment
	// refuses it with this sentence.
	enrolment, err := headerString(header, file, "enrolment", firstWindowEnrolment)
	if err != nil {
		return nil, err
	}
	if !slices.Contains(knownEnrolments, enrolment) {
		who := name
		if who == "" {
			who = "voice"
		}
		return nil, fmt.Errorf(
			"%s: enrolment strategy %s is not one this build implements (%s). "+
				"The profile was made by a build that cuts its prompt differently, "+
				"so loading it here would speak in a different voice under the same name.",
			who, pyRepr(enrolment), strings.Join(knownEnrolments, ", "))
	}
	// The recording's own rate, and the divisor of every duration derived from
	// it: a zero divides by zero and a negative reports negative seconds.
	// _validate_values refuses it with this sentence.
	rate, err := headerInt(header, file, "source_sample_rate", 24000)
	if err != nil {
		return nil, err
	}
	if rate <= 0 {
		return nil, fmt.Errorf("source_sample_rate must be positive: %d", rate)
	}
	language, err := headerString(header, file, "language", "en")
	if err != nil {
		return nil, err
	}
	return &Profile{
		Name:             name,
		SpeakerEmbedding: speaker,
		FlowEmbedding:    flow,
		PromptTokens:     promptTok,
		PromptMel:        promptMel,
		CondPromptTokens: condTok,
		SourceSampleRate: rate,
		Language:         language,
		Enrolment:        enrolment,
	}, nil
}

// Save writes the profile as safetensors, with the header
// python/loudkit/voice.py writes, so every implementation reads it back.
// Owner-only permissions: a profile derives from a recording of a person.
func (p *Profile) Save(path string) error {
	if len(p.PromptMel)%melBins != 0 {
		return fmt.Errorf("prompt_mel must be (%d, frames), got %d values", melBins, len(p.PromptMel))
	}
	enrolment := p.Enrolment
	if enrolment == "" {
		enrolment = firstWindowEnrolment
	}
	header, err := json.Marshal(map[string]any{
		"format_version":     formatVersion,
		"name":               p.Name,
		"source_sample_rate": p.SourceSampleRate,
		"language":           p.Language,
		"enrolment":          enrolment,
	})
	if err != nil {
		return err
	}
	entries := []safetensors.Entry{
		{Name: "speaker_embedding", Dtype: "F32", Shape: []int64{int64(len(p.SpeakerEmbedding))},
			Data: safetensors.F32Bytes(p.SpeakerEmbedding)},
		{Name: "flow_embedding", Dtype: "F32", Shape: []int64{int64(len(p.FlowEmbedding))},
			Data: safetensors.F32Bytes(p.FlowEmbedding)},
		{Name: "prompt_tokens", Dtype: "I64", Shape: []int64{int64(len(p.PromptTokens))},
			Data: safetensors.I64Bytes(p.PromptTokens)},
		{Name: "prompt_mel", Dtype: "F32", Shape: []int64{melBins, int64(len(p.PromptMel) / melBins)},
			Data: safetensors.F32Bytes(p.PromptMel)},
		{Name: "cond_prompt_tokens", Dtype: "I64", Shape: []int64{int64(len(p.CondPromptTokens))},
			Data: safetensors.I64Bytes(p.CondPromptTokens)},
	}
	return safetensors.Write(path, entries, map[string]string{"voice": string(header)})
}

// jsonTypePhrase names a decoded JSON value in the words a header is written
// in, as a phrase that reads after "is": "an array", "a string", "null".
func jsonTypePhrase(v any) string {
	switch v.(type) {
	case map[string]interface{}:
		return "an object"
	case []interface{}:
		return "an array"
	case string:
		return "a string"
	case bool:
		return "a boolean"
	case nil:
		return "null"
	default:
		return "a number"
	}
}

// pyRepr spells a decoded JSON value the way the reference's refusals quote it
// back, so the same bad header reads the same in both.
//
// A whole number prints without a fraction, because encoding/json widens every
// JSON number to float64 and the digits a header carried are gone by the time
// a refusal quotes them: `5` and `5.0` both print as `5`. Object keys are
// sorted, because a Go map has no order to preserve.
func pyRepr(v any) string {
	switch x := v.(type) {
	case nil:
		return "None"
	case bool:
		if x {
			return "True"
		}
		return "False"
	case float64:
		return strconv.FormatFloat(x, 'g', -1, 64)
	case string:
		// repr's own rule: single quotes unless the value holds one and no
		// double quote.
		if strings.Contains(x, "'") && !strings.Contains(x, `"`) {
			return `"` + x + `"`
		}
		return "'" + strings.ReplaceAll(strings.ReplaceAll(x, `\`, `\\`), "'", `\'`) + "'"
	case []interface{}:
		parts := make([]string, len(x))
		for i, item := range x {
			parts[i] = pyRepr(item)
		}
		return "[" + strings.Join(parts, ", ") + "]"
	case map[string]interface{}:
		parts := make([]string, 0, len(x))
		for _, k := range slices.Sorted(maps.Keys(x)) {
			parts = append(parts, pyRepr(k)+": "+pyRepr(x[k]))
		}
		return "{" + strings.Join(parts, ", ") + "}"
	default:
		return fmt.Sprintf("%v", x)
	}
}

// headerInt reads a whole JSON number under k, or def when k is absent.
//
// Not a cast: JSON has one number type and Go's decoder widens it to float64,
// so `format_version: 1.9` read as version 1 and a profile written by a build
// this one does not implement loaded as one it does. A value of another type
// is refused rather than replaced by def, because a header field of the wrong
// type was written by something that did not mean it.
func headerInt(h map[string]interface{}, file, k string, def int) (int, error) {
	raw, ok := h[k]
	if !ok {
		return def, nil
	}
	v, ok := raw.(float64)
	if !ok {
		return 0, fmt.Errorf("%s: voice header '%s' must be a number, got %s",
			file, k, pyRepr(raw))
	}
	if v != math.Trunc(v) {
		return 0, fmt.Errorf("%s: voice header '%s' must be a whole number, got %s",
			file, k, pyRepr(raw))
	}
	return int(v), nil
}

// headerString reads a JSON string under k, or def when k is absent.
//
// A present value of another type is refused, not stringified: `language: 5`
// would be the language id "5", and language selects the text funnel, so the
// voice would be read through a funnel nobody chose.
func headerString(h map[string]interface{}, file, k, def string) (string, error) {
	raw, ok := h[k]
	if !ok {
		return def, nil
	}
	v, ok := raw.(string)
	if !ok {
		return "", fmt.Errorf("%s: voice header '%s' must be a string, got %s",
			file, k, pyRepr(raw))
	}
	return v, nil
}
