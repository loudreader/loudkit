package config

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// FingerprintSchema is bumped only when the *set* of hashed fields changes,
// never when a value does. Adding a field with a default must not
// re-fingerprint an algorithm that did not change — a check that cries wolf on
// every upgrade is a check people learn to override.
const FingerprintSchema = 1

// CanonicalForm is the exact string that gets hashed.
//
// Every other cross-language check in this project compares a behaviour
// somebody thought to compare: the speech funnel because there are 30 fixture
// cases for it, the splitter because there are 18. This compares the *whole*
// algorithm configuration in one string, so a field nobody wrote a test for
// still cannot drift silently.
//
// The failure mode is concrete: an euler_grid parsed by one port and ignored
// by another; a silence_token_ids that accepts a JSON string and iterates its
// characters; a chunking.prefix_tokens read from the manifest by some ports and
// guessed by others — each invisible to behaviour comparison alone. This
// finds all of them at once, and the next one for free.
//
// Built by hand rather than through encoding/json: the byte-for-byte output is
// the contract, and a marshaller is free to change how it renders a float or
// orders a map between releases. Three rules make it portable:
//
//   - floats are their shortest round-tripping decimal, as a JSON *string*.
//     Python emits repr(float) — "0.8", not 0.8 — quoted, so no JSON parser
//     anywhere gets to re-render the number with its own idea of precision.
//   - keys are sorted, at every level.
//   - only schema-known fields are hashed, with an explicit schema version.
func CanonicalForm(cfg AlgorithmConfig) string {
	splitOn := make([]string, 0, len(cfg.Chunking.SplitOn))
	for _, sep := range cfg.Chunking.SplitOn {
		splitOn = append(splitOn, jsonString(sep))
	}
	chunking := fmt.Sprintf(
		`{"enabled":%t,"max_tokens":%d,"prefix_tokens":%d,"split_on":[%s]}`,
		cfg.Chunking.Enabled, cfg.Chunking.MaxTokens, cfg.Chunking.PrefixTokens,
		strings.Join(splitOn, ","))

	// Sorted, because the manifest's order is whatever the packer wrote and
	// the hash must not depend on it.
	silence := append([]int(nil), cfg.Sampling.SilenceTokenIds...)
	sort.Ints(silence)
	ids := make([]string, 0, len(silence))
	for _, id := range silence {
		ids = append(ids, strconv.Itoa(id))
	}
	sampling := fmt.Sprintf(
		`{"max_new_tokens":%d,"min_p":%s,"min_tokens_floor":%d,`+
			`"min_tokens_text_ratio":%s,"repetition_penalty":%s,`+
			`"silence_token_ids":[%s],"temperature":%s}`,
		cfg.Sampling.MaxNewTokens, jsonFloat(cfg.Sampling.MinP),
		cfg.Sampling.MinTokensFloor, jsonFloat(cfg.Sampling.MinTokensTextRatio),
		jsonFloat(cfg.Sampling.RepetitionPenalty), strings.Join(ids, ","),
		jsonFloat(cfg.Sampling.Temperature))

	window := fmt.Sprintf(
		`{"max_speech_tokens":%d,"pad_token_id":%s,"static_length":%s,`+
			`"static_prompt_tokens":%s}`,
		cfg.Window.MaxSpeechTokens, jsonOptInt(cfg.Window.PadTokenID),
		jsonOptInt(cfg.Window.StaticLength), jsonOptInt(cfg.Window.StaticPromptTokens))

	// Keys sorted, as everywhere in this form. The detectors remove tokens, so
	// a port using a different threshold produces different audio — exactly the
	// silent drift a whole-config hash exists to catch.
	pp := cfg.Postprocess
	postprocess := fmt.Sprintf(
		`{"ceiling_slack_tokens":%d,"ceiling_speech_per_text_token":%s,`+
			`"desperation_band_floor":%d,"desperation_band_ratio":%s,`+
			`"desperation_min_text_tokens":%d,"desperation_speech_per_text_token":%s,`+
			`"dropout_min_tokens":%d,`+
			`"echo_strong_eos_probability":%s,"echo_strong_max_tail":%d,`+
			`"echo_strong_min_position_pct":%d,"echo_weak_eos_probability":%s,`+
			`"echo_weak_max_tail":%d,"echo_weak_min_position_pct":%d,`+
			`"ended_tail_blip_max":%d,"ended_tail_keep":%d,`+
			`"ended_tail_silence_run":%d,"ended_tail_word_max":%d,`+
			`"filler_max_speech_after_run":%d,"filler_min_eos_probability":%s,`+
			`"mode":%s,"pacing_tolerance":%s,`+
			`"repetition_max_period":%d,"repetition_min_cycles":%d,`+
			`"repetition_min_span":%d,"retry_max_attempts":%d,`+
			`"trailing_filler_threshold":%s,`+
			`"trailing_silence_run_tokens":%d}`,
		pp.CeilingSlackTokens, jsonFloat(pp.CeilingSpeechPerTextToken),
		pp.DesperationBandFloor, jsonFloat(pp.DesperationBandRatio),
		pp.DesperationMinTextTokens, jsonFloat(pp.DesperationSpeechPerTextToken),
		pp.DropoutMinTokens,
		jsonFloat(pp.EchoStrongEosProbability), pp.EchoStrongMaxTail,
		pp.EchoStrongMinPositionPct, jsonFloat(pp.EchoWeakEosProbability),
		pp.EchoWeakMaxTail, pp.EchoWeakMinPositionPct,
		pp.EndedTailBlipMax, pp.EndedTailKeep,
		pp.EndedTailSilenceRun, pp.EndedTailWordMax,
		pp.FillerMaxSpeechAfterRun, jsonFloat(pp.FillerMinEosProbability),
		jsonString(pp.Mode), jsonFloat(pp.PacingTolerance),
		pp.RepetitionMaxPeriod, pp.RepetitionMinCycles,
		pp.RepetitionMinSpan, pp.RetryMaxAttempts,
		jsonFloat(pp.TrailingFillerThreshold),
		pp.TrailingSilenceRunTokens)

	eulerGrid := "null"
	if len(cfg.EulerGrid) > 0 {
		points := make([]string, 0, len(cfg.EulerGrid))
		for _, t := range cfg.EulerGrid {
			points = append(points, jsonFloat(t))
		}
		eulerGrid = "[" + strings.Join(points, ",") + "]"
	}

	// The funnel's identity travels in the fingerprint: its code version, and
	// the digest of the grammar file this port reads. Each implementation hashes
	// its *own* copy, so a port whose data has drifted computes a different
	// fingerprint and the engine refuses to start — which is how the drift is
	// caught, rather than by someone eventually hearing it.
	// Go has no field defaults, so a config built as a literal arrives with an
	// empty Text. An empty digest is never a real one, so it means "unset"
	// rather than "different", and filling it here keeps every construction
	// path — literal, manifest, test — hashing the same algorithm.
	textCfg := cfg.Text
	if textCfg.Recipe == "" {
		textCfg.Recipe = TextRecipe
	}
	if textCfg.Grammar == "" {
		textCfg.Grammar = GrammarDigest()
	}
	text := fmt.Sprintf(`{"grammar":%s,"recipe":%s}`,
		jsonString(textCfg.Grammar), jsonString(textCfg.Recipe))

	body := fmt.Sprintf(
		`{"chunking":%s,"euler_grid":%s,"euler_steps":%d,"guidance":%s,`+
			`"guidance_rate":%s,"postprocess":%s,"recipe_version":%s,"sample_rate":%d,`+
			`"sampling":%s,"speech_vocab_size":%d,"start_speech_token":%d,`+
			`"stop_speech_token":%d,"text":%s,"token_rate_hz":%s,"window":%s}`,
		chunking, eulerGrid, cfg.EulerSteps, jsonString(cfg.Guidance),
		jsonFloat(cfg.GuidanceRate), postprocess, jsonString(cfg.RecipeVersion),
		cfg.SampleRate, sampling, cfg.SpeechVocabSize, cfg.StartSpeech,
		cfg.StopSpeech, text, jsonFloat(cfg.TokenRateHz), window)

	return fmt.Sprintf(`{"algorithm":%s,"schema":%d}`, body, FingerprintSchema)
}

// Fingerprint is the first 16 hex characters of SHA-256 over CanonicalForm.
//
// Two engines whose fingerprints differ are computing different things,
// whatever their outputs happen to sound like — which is the point: the
// guidance defect this project was built around produced plausible audio on
// both sides of the mismatch, so no listening test could have found it.
func Fingerprint(cfg AlgorithmConfig) string {
	sum := sha256.Sum256([]byte(CanonicalForm(cfg)))
	return hex.EncodeToString(sum[:])[:16]
}

// jsonFloat renders a float the way Python's repr() does, as a JSON string.
//
// 'g' with -1 precision gives the shortest decimal that round-trips, which is
// what repr() gives — except that Go renders 25.0 as "25" while Python renders
// it "25.0". That one character is the difference between a matching
// fingerprint and a mysterious one.
func jsonFloat(v float64) string {
	return jsonString(pyFloat(v))
}

// pyFloat is the repr() rendering itself, for the log lines that print a float
// unquoted. Shared with jsonFloat so the two cannot drift: a describe line that
// says temp=0.8 and a canonical form that says "0.80" would send a reader
// hunting for a difference that is not there.
func pyFloat(v float64) string {
	s := strconv.FormatFloat(v, 'g', -1, 64)
	if !strings.ContainsAny(s, ".eEni") { // no point, no exponent, not inf/nan
		s += ".0"
	}
	return s
}

// jsonString escapes the way encoding/json escapes, minus the HTML escaping
// that json.Marshal applies by default and Python's json.dumps does not.
func jsonString(s string) string {
	var b strings.Builder
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		case '\t':
			b.WriteString(`\t`)
		case '\r':
			b.WriteString(`\r`)
		default:
			if r < 0x20 {
				fmt.Fprintf(&b, `\u%04x`, r)
			} else {
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
	return b.String()
}

func jsonOptInt(v *int) string {
	if v == nil {
		return "null"
	}
	return strconv.Itoa(*v)
}
