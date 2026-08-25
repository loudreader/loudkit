package postprocess

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// The shared conformance file. Every port runs it, so a rule that drifts in one
// language fails in one language.
func fixturePath(t *testing.T) string {
	t.Helper()
	p := os.Getenv("LOUDKIT_POSTPROCESS_FIXTURE")
	if p == "" {
		p = filepath.Join("..", "..", "tests", "data", "conformance", "postprocess.json")
	}
	if _, err := os.Stat(p); err != nil {
		t.Skipf("fixture not found: %s", p)
	}
	return p
}

type fixture struct {
	SilenceTokenIDs []int          `json:"silence_token_ids"`
	Config          map[string]any `json:"config"`
	Ceiling         []struct {
		Name       string `json:"name"`
		Why        string `json:"why"`
		TextTokens int    `json:"text_tokens"`
		Window     int    `json:"window"`
		Expect     int    `json:"expect"`
	} `json:"ceiling"`
	Dropout struct {
		Cases []struct {
			Name       string `json:"name"`
			Why        string `json:"why"`
			Tokens     int    `json:"tokens"`
			TextTokens int    `json:"text_tokens"`
			Expect     bool   `json:"expect"`
		} `json:"cases"`
	} `json:"dropout"`
	Pacing struct {
		Cases []struct {
			Name   string    `json:"name"`
			Why    string    `json:"why"`
			Ratios []float64 `json:"ratios"`
			Expect []int     `json:"expect"`
		} `json:"cases"`
	} `json:"pacing"`
	Repetition []struct {
		Name   string  `json:"name"`
		Why    string  `json:"why"`
		Shape  [][]any `json:"shape"`
		Expect *int    `json:"expect"`
	} `json:"repetition"`
	TrailingFiller []struct {
		Name   string  `json:"name"`
		Why    string  `json:"why"`
		Shape  [][]any `json:"shape"`
		From   int     `json:"from"`
		Expect bool    `json:"expect"`
	} `json:"trailing_filler"`
	Desperation []struct {
		Name        string  `json:"name"`
		Why         string  `json:"why"`
		Shape       [][]any `json:"shape"`
		TextTokens  int     `json:"text_tokens"`
		MinTokens   int     `json:"min_tokens"`
		EosPeakAt   int     `json:"eos_peak_at"`
		PeakAllowed bool    `json:"peak_allowed"`
		Expect      *int    `json:"expect"`
	} `json:"desperation"`
	EndedTail []struct {
		Name       string  `json:"name"`
		Why        string  `json:"why"`
		Shape      [][]any `json:"shape"`
		IsTerminal bool    `json:"is_terminal"`
		Expect     *int    `json:"expect"`
	} `json:"ended_tail"`
	TerminalEcho []struct {
		Name        string  `json:"name"`
		Why         string  `json:"why"`
		TokenCount  int     `json:"token_count"`
		EosPeakAt   int     `json:"eos_peak_at"`
		EosPeakProb float64 `json:"eos_peak_prob"`
		MinTokens   int     `json:"min_tokens"`
		IsTerminal  bool    `json:"is_terminal"`
		HitCeiling  bool    `json:"hit_ceiling"`
		Expect      *int    `json:"expect"`
	} `json:"terminal_echo"`
	LanguageGuard struct {
		Cases []struct {
			Name                 string `json:"name"`
			Why                  string `json:"why"`
			TextTokens           int    `json:"text_tokens"`
			Window               int    `json:"window"`
			Expect               int    `json:"expect"`
			MeasuredSpeechTokens int    `json:"measured_speech_tokens"`
			ExpectStopped        bool   `json:"expect_stopped_by_ceiling"`
		} `json:"cases"`
	} `json:"language_guard"`
	Resolve []struct {
		Name        string  `json:"name"`
		Why         string  `json:"why"`
		Mode        string  `json:"mode"`
		Shape       [][]any `json:"shape"`
		TextTokens  int     `json:"text_tokens"`
		MinTokens   int     `json:"min_tokens"`
		EosPeakAt   int     `json:"eos_peak_at"`
		EosPeakProb float64 `json:"eos_peak_prob"`
		Ended       bool    `json:"ended"`
		IsTerminal  bool    `json:"is_terminal"`
		HitCeiling  bool    `json:"hit_ceiling"`
		Expect      struct {
			Keep    int    `json:"keep"`
			Reason  string `json:"reason"`
			Suspect bool   `json:"suspect"`
		} `json:"expect"`
	} `json:"resolve"`
}

func load(t *testing.T) fixture {
	t.Helper()
	raw, err := os.ReadFile(fixturePath(t))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fx fixture
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	return fx
}

// build is the fixture's token-shape builder, spelled out in its header.
func build(t *testing.T, shape [][]any) []int {
	t.Helper()
	out := []int{}
	for _, seg := range shape {
		kind, _ := seg[0].(string)
		count := int(seg[1].(float64))
		switch kind {
		case "speech":
			for i := 0; i < count; i++ {
				out = append(out, 20+i%60)
			}
		case "quiet":
			for i := 0; i < count; i++ {
				out = append(out, i%8)
			}
		case "cycle":
			// count is the period; seg[2] the repeat count.
			cycle := make([]int, count)
			for i := range cycle {
				cycle[i] = 20 + i%60
			}
			for r := 0; r < int(seg[2].(float64)); r++ {
				out = append(out, cycle...)
			}
		case "cycle_mixed":
			// Second half silence: the word-then-pause stutter.
			half := count / 2
			cycle := make([]int, 0, count)
			for i := 0; i < count-half; i++ {
				cycle = append(cycle, 20+i)
			}
			for i := 0; i < half; i++ {
				cycle = append(cycle, i%8)
			}
			for r := 0; r < int(seg[2].(float64)); r++ {
				out = append(out, cycle...)
			}
		default:
			t.Fatalf("unknown segment kind %q", kind)
		}
	}
	return out
}

// configFrom builds the detector config out of the fixture, so the numbers the
// test runs on are the numbers the fixture declares rather than this port's own
// defaults — which is the whole point of a shared file.
func configFrom(t *testing.T, fx fixture, mode string) Config {
	t.Helper()
	num := func(key string) float64 {
		v, ok := fx.Config[key].(float64)
		if !ok {
			t.Fatalf("fixture config missing %q", key)
		}
		return v
	}
	// The band keys predate the fixture; absent means the shipping value,
	// exactly as the manifest readers treat absence.
	optNum := func(key string, def float64) float64 {
		if v, ok := fx.Config[key].(float64); ok {
			return v
		}
		return def
	}
	if mode == "" {
		s, ok := fx.Config["mode"].(string)
		if !ok {
			t.Fatalf("fixture config missing \"mode\"")
		}
		mode = s
	}
	return Config{
		Mode:                      mode,
		CeilingSpeechPerTextToken: num("ceiling_speech_per_text_token"),
		CeilingSlackTokens:        int(num("ceiling_slack_tokens")),
		TrailingFillerThreshold:   num("trailing_filler_threshold"),
		TrailingSilenceRunTokens:  int(num("trailing_silence_run_tokens")),
		DesperationBandRatio: optNum("desperation_band_ratio",
			Production().DesperationBandRatio),
		DesperationBandFloor: int(optNum("desperation_band_floor",
			float64(Production().DesperationBandFloor))),
		FillerMinEosProbability:       num("filler_min_eos_probability"),
		FillerMaxSpeechAfterRun:       int(num("filler_max_speech_after_run")),
		DesperationSpeechPerTextToken: num("desperation_speech_per_text_token"),
		DesperationMinTextTokens:      int(num("desperation_min_text_tokens")),
		EndedTailSilenceRun:           int(num("ended_tail_silence_run")),
		EndedTailBlipMax:              int(num("ended_tail_blip_max")),
		EndedTailWordMax:              int(num("ended_tail_word_max")),
		EndedTailKeep:                 int(num("ended_tail_keep")),
		EchoStrongEosProbability:      num("echo_strong_eos_probability"),
		EchoStrongMaxTail:             int(num("echo_strong_max_tail")),
		EchoStrongMinPositionPct:      int(num("echo_strong_min_position_pct")),
		EchoWeakEosProbability:        num("echo_weak_eos_probability"),
		EchoWeakMaxTail:               int(num("echo_weak_max_tail")),
		EchoWeakMinPositionPct:        int(num("echo_weak_min_position_pct")),
		RepetitionMaxPeriod:           int(num("repetition_max_period")),
		RepetitionMinCycles:           int(num("repetition_min_cycles")),
		RepetitionMinSpan:             int(num("repetition_min_span")),
		DropoutMinTokens:              int(num("dropout_min_tokens")),
		RetryMaxAttempts:              int(num("retry_max_attempts")),
		PacingTolerance:               num("pacing_tolerance"),
	}
}

// want turns the fixture's nullable "expect" into this port's -1 sentinel.
func want(p *int) int {
	if p == nil {
		return -1
	}
	return *p
}

func TestProductionMatchesTheFixture(t *testing.T) {
	// The shipping constants are the fixture's, or the cases below prove
	// nothing about what actually runs.
	fx := load(t)
	if got, expect := Production(), configFrom(t, fx, ""); got != expect {
		t.Fatalf("Production() has drifted from the conformance fixture:\n got %+v\nwant %+v",
			got, expect)
	}
}

func TestCeiling(t *testing.T) {
	fx := load(t)
	cfg := configFrom(t, fx, "")
	for _, c := range fx.Ceiling {
		if got := CeilingFor(c.TextTokens, cfg, c.Window); got != c.Expect {
			t.Errorf("%s: got %d, want %d (%s)", c.Name, got, c.Expect, c.Why)
		}
	}
}

func TestTrailingFiller(t *testing.T) {
	fx := load(t)
	cfg := configFrom(t, fx, "")
	for _, c := range fx.TrailingFiller {
		got := IsTrailingFiller(build(t, c.Shape), c.From, fx.SilenceTokenIDs, cfg)
		if got != c.Expect {
			t.Errorf("%s: got %v, want %v (%s)", c.Name, got, c.Expect, c.Why)
		}
	}
}

func TestDesperation(t *testing.T) {
	fx := load(t)
	cfg := configFrom(t, fx, "")
	for _, c := range fx.Desperation {
		got := DesperationCut(build(t, c.Shape), c.TextTokens, c.MinTokens, c.EosPeakAt,
			fx.SilenceTokenIDs, cfg, c.PeakAllowed)
		if got != want(c.Expect) {
			t.Errorf("%s: got %d, want %d (%s)", c.Name, got, want(c.Expect), c.Why)
		}
	}
}

func TestEndedTail(t *testing.T) {
	fx := load(t)
	cfg := configFrom(t, fx, "")
	for _, c := range fx.EndedTail {
		got := EndedTailTrim(build(t, c.Shape), fx.SilenceTokenIDs, cfg, c.IsTerminal)
		if got != want(c.Expect) {
			t.Errorf("%s: got %d, want %d (%s)", c.Name, got, want(c.Expect), c.Why)
		}
	}
}

func TestTerminalEcho(t *testing.T) {
	fx := load(t)
	cfg := configFrom(t, fx, "")
	for _, c := range fx.TerminalEcho {
		got := TerminalEchoCut(c.TokenCount, c.EosPeakAt, c.EosPeakProb, c.MinTokens,
			c.IsTerminal, c.HitCeiling, cfg)
		if got != want(c.Expect) {
			t.Errorf("%s: got %d, want %d (%s)", c.Name, got, want(c.Expect), c.Why)
		}
	}
}

// The precedence, which is the part a caller cannot get right by itself.
func TestResolve(t *testing.T) {
	fx := load(t)
	for _, c := range fx.Resolve {
		cfg := configFrom(t, fx, c.Mode)
		got := Inspect(build(t, c.Shape), Request{
			TextTokenCount: c.TextTokens,
			MinTokens:      c.MinTokens,
			EosPeakAt:      c.EosPeakAt,
			EosPeakProb:    c.EosPeakProb,
			Ended:          c.Ended,
			IsTerminal:     c.IsTerminal,
			HitCeiling:     c.HitCeiling,
		}, fx.SilenceTokenIDs, cfg)
		if got.Keep != c.Expect.Keep || got.Reason != c.Expect.Reason ||
			got.Suspect != c.Expect.Suspect {
			t.Errorf("%s: got %+v, want %+v (%s)", c.Name, got, c.Expect, c.Why)
		}
	}
}

// The ceiling was settled on English traces; nine languages ship.
//
// Speech tokens per *text* token is a property of the orthography, so a constant
// tuned on one language is an assumption everywhere else — and the expensive
// direction of that assumption is a guard that truncates correct speech in a
// language nobody measured. Measured with one voice held constant across nine
// language tags, because the voice-to-voice spread on a single sentence is
// larger than the language-to-language spread.
func TestLanguageGuard(t *testing.T) {
	fx := load(t)
	cfg := configFrom(t, fx, "")
	if len(fx.LanguageGuard.Cases) == 0 {
		t.Fatal("the fixture has no language_guard cases; nothing was compared")
	}
	var stopped []string
	for _, c := range fx.LanguageGuard.Cases {
		ceiling := CeilingFor(c.TextTokens, cfg, c.Window)
		if ceiling != c.Expect {
			t.Errorf("%s: ceiling %d, want %d (%s)", c.Name, ceiling, c.Expect, c.Why)
		}
		hit := c.MeasuredSpeechTokens >= ceiling
		if hit != c.ExpectStopped {
			t.Errorf("%s changed side of the ceiling (%s)", c.Name, c.Why)
		}
		if hit {
			stopped = append(stopped, c.Name)
		}
	}
	// One row belongs here and it is not a false positive: a Spanish three-word
	// phrase whose decoder never emitted a stop token. The guard caught a
	// runaway; it did not cut a legitimate read.
	if len(stopped) != 1 || stopped[0] != "es_short" {
		t.Errorf("rows stopped by the ceiling = %v, want [es_short] — a new entry "+
			"is a language being truncated by an English-tuned constant", stopped)
	}
}

// The loop the tail rules cannot see, because it happens mid-row.
//
// Every other rule reads the end of the chunk. A stuck decoder repeats inside
// it, and the literature puts that failure first or second in every ranking of
// what goes wrong with autoregressive speech models.
func TestRepetition(t *testing.T) {
	fx := load(t)
	cfg := configFrom(t, fx, "")
	if len(fx.Repetition) == 0 {
		t.Fatal("the fixture has no repetition cases; nothing was compared")
	}
	negatives := 0
	for _, c := range fx.Repetition {
		if c.Expect == nil {
			negatives++
		}
		if got := RepetitionCut(build(t, c.Shape), fx.SilenceTokenIDs, cfg); got != want(c.Expect) {
			t.Errorf("%s: got %d, want %d (%s)", c.Name, got, want(c.Expect), c.Why)
		}
	}
	// A mid-sequence cut is the most destructive thing this layer can do, so
	// the cases that must NOT fire carry more weight than the ones that must.
	if negatives < 6 {
		t.Errorf("only %d negative cases; too few to trust a mid-row cut", negatives)
	}
}

// Early truncation — the failure a listener cannot hear.
//
// Every other rule says the end of the row is wrong. This one says the row is
// incomplete, which is why it reports rather than cuts.
func TestDropout(t *testing.T) {
	fx := load(t)
	cfg := configFrom(t, fx, "")
	if len(fx.Dropout.Cases) == 0 {
		t.Fatal("the fixture has no dropout cases; nothing was compared")
	}
	for _, c := range fx.Dropout.Cases {
		if got := IsDropout(c.Tokens, c.TextTokens, cfg); got != c.Expect {
			t.Errorf("%s: got %v, want %v (%s)", c.Name, got, c.Expect, c.Why)
		}
	}
}

// Long-form drift, report-only, in the same integer-derived domain.
func TestPacing(t *testing.T) {
	fx := load(t)
	cfg := configFrom(t, fx, "")
	if len(fx.Pacing.Cases) == 0 {
		t.Fatal("the fixture has no pacing cases")
	}
	for _, c := range fx.Pacing.Cases {
		got := PacingOutliers(c.Ratios, cfg)
		want := c.Expect
		if len(got) != len(want) {
			t.Errorf("%s: got %v, want %v (%s)", c.Name, got, want, c.Why)
			continue
		}
		for i := range got {
			if got[i] != want[i] {
				t.Errorf("%s: got %v, want %v (%s)", c.Name, got, want, c.Why)
				break
			}
		}
	}
}
