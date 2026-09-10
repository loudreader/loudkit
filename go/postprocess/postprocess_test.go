package postprocess

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
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
		// Not a skip. This fixture is committed to the repository, so a
		// missing one is a checkout that lost it or a path that moved, and
		// skipping turned every rule in this file green without comparing
		// anything.
		t.Fatalf("fixture not found: %s (%v)", p, err)
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
	StarvedRescue struct {
		Cases []inspectCase `json:"cases"`
	} `json:"starved_rescue"`
	RepetitionResume struct {
		Cases []struct {
			inspectCase
			Loop *int `json:"loop"`
		} `json:"cases"`
	} `json:"repetition_resume"`
	RepetitionSilence struct {
		SilenceRenderIds []int `json:"silence_render_ids"`
		QuietRenderIds   []int `json:"quiet_render_ids"`
		Cases            []struct {
			inspectCase
			Loop *int `json:"loop"`
		} `json:"cases"`
	} `json:"repetition_silence"`
	Stall struct {
		SilenceRenderIds []int `json:"silence_render_ids"`
		QuietRenderIds   []int `json:"quiet_render_ids"`
		Cases            []struct {
			inspectCase
			RenderIds bool `json:"render_ids"`
			Stalled   bool `json:"stalled"`
		} `json:"cases"`
	} `json:"stall"`
	Resolve []struct {
		inspectCase
		Mode string `json:"mode"`
	} `json:"resolve"`
}

// inspectCase is the shape every resolver case in the fixture shares: a window
// of tokens, the counters Inspect reads, and the verdict expected of it.
//
// One shape rather than one per section: a mistyped json tag reads as a zero
// rather than as an error, so a case built on a private copy of the shape runs
// against the wrong request and passes.
type inspectCase struct {
	Name        string  `json:"name"`
	Why         string  `json:"why"`
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
}

// request is the case as the resolver reads it.
func (c inspectCase) request() Request {
	return Request{
		TextTokenCount: c.TextTokens,
		MinTokens:      c.MinTokens,
		EosPeakAt:      c.EosPeakAt,
		EosPeakProb:    c.EosPeakProb,
		Ended:          c.Ended,
		IsTerminal:     c.IsTerminal,
		HitCeiling:     c.HitCeiling,
	}
}

// inspect runs one case through the resolver under cfg.
func (c inspectCase) inspect(t *testing.T, fx fixture, cfg Config) Inspection {
	t.Helper()
	return Inspect(build(t, c.Shape), c.request(), fx.SilenceTokenIDs, cfg)
}

// check runs one case and compares the verdict the fixture expects.
func (c inspectCase) check(t *testing.T, fx fixture, cfg Config) {
	t.Helper()
	got := c.inspect(t, fx, cfg)
	if got.Keep != c.Expect.Keep || got.Reason != c.Expect.Reason ||
		got.Suspect != c.Expect.Suspect {
		t.Errorf("%s: got %+v, want %+v (%s)", c.Name, got, c.Expect, c.Why)
	}
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
		case "sil":
			// True digital silence in the stall section's two-class scheme.
			for i := 0; i < count; i++ {
				out = append(out, i%4)
			}
		case "breath":
			// The contextually-quiet family: extends a dead-air run without
			// counting toward its gate.
			for i := 0; i < count; i++ {
				out = append(out, 4+i%4)
			}
		case "dead":
			// True silence only the render census knows (the 6405 class):
			// outside the fixture's sampler list, inside its silence census.
			for i := 0; i < count; i++ {
				out = append(out, 12)
			}
		case "sigh":
			// Breath only the quiet census knows.
			for i := 0; i < count; i++ {
				out = append(out, 13)
			}
		case "cycle_dead":
			// The stutter with its pause on a census-only id.
			half := count / 2
			cycle := make([]int, 0, count)
			for i := 0; i < count-half; i++ {
				cycle = append(cycle, 20+i)
			}
			for i := 0; i < half; i++ {
				cycle = append(cycle, 12)
			}
			for r := 0; r < int(seg[2].(float64)); r++ {
				out = append(out, cycle...)
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
// defaults, which is the whole point of a shared file.
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
		DesperationMinKeepPerTextToken: optNum("desperation_min_keep_per_text_token",
			Production().DesperationMinKeepPerTextToken),
		EndedTailSilenceRun:      int(num("ended_tail_silence_run")),
		EndedTailBlipMax:         int(num("ended_tail_blip_max")),
		EndedTailWordMax:         int(num("ended_tail_word_max")),
		EndedTailKeep:            int(num("ended_tail_keep")),
		EchoStrongEosProbability: num("echo_strong_eos_probability"),
		EchoStrongMaxTail:        int(num("echo_strong_max_tail")),
		EchoStrongMinPositionPct: int(num("echo_strong_min_position_pct")),
		EchoWeakEosProbability:   num("echo_weak_eos_probability"),
		EchoWeakMaxTail:          int(num("echo_weak_max_tail")),
		EchoWeakMinPositionPct:   int(num("echo_weak_min_position_pct")),
		RepetitionMaxPeriod:      int(num("repetition_max_period")),
		RepetitionMinCycles:      int(num("repetition_min_cycles")),
		RepetitionMinSpan:        int(num("repetition_min_span")),
		// String fields, absent from the fixture's config block like the
		// band keys, so the shipping values apply.
		RepetitionResume:  optStr(fx, "repetition_resume", Production().RepetitionResume),
		RepetitionSilence: optStr(fx, "repetition_silence", Production().RepetitionSilence),
		// Like the band keys: absent from the fixture's config block, so the
		// shipping value applies; Python builds its config the same way.
		StallRunTokens:   int(optNum("stall_run_tokens", float64(Production().StallRunTokens))),
		DropoutMinTokens: int(num("dropout_min_tokens")),
		RetryMaxAttempts: int(num("retry_max_attempts")),
		PacingTolerance:  num("pacing_tolerance"),
	}
}

// optStr reads an optional string key from the fixture's config block.
func optStr(fx fixture, key, def string) string {
	if v, ok := fx.Config[key].(string); ok {
		return v
	}
	return def
}

// repSilenceConfig is configFrom plus the repetition_silence section's
// render-id censuses: the ids only the censuses know, which the loop
// exemption must union in under the acoustic family.
func repSilenceConfig(t *testing.T, fx fixture) Config {
	t.Helper()
	cfg := configFrom(t, fx, "")
	cfg.SilenceRenderIds = fx.RepetitionSilence.SilenceRenderIds
	cfg.QuietRenderIds = fx.RepetitionSilence.QuietRenderIds
	return cfg
}

// stallConfig is configFrom plus the stall section's render-id censuses. The
// fallback arm (renderIds false) runs without them: a checkpoint packed
// before the census, where only the run trigger fires.
func stallConfig(t *testing.T, fx fixture, renderIds bool) Config {
	t.Helper()
	cfg := configFrom(t, fx, "")
	if renderIds {
		cfg.SilenceRenderIds = fx.Stall.SilenceRenderIds
		cfg.QuietRenderIds = fx.Stall.QuietRenderIds
	}
	return cfg
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
	// nothing about what actually runs. DeepEqual because the render-id
	// censuses made Config carry slices; both sides are empty here: the
	// censuses are checkpoint properties, not detector constants.
	fx := load(t)
	if got, expect := Production(), configFrom(t, fx, ""); !reflect.DeepEqual(got, expect) {
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
		c.check(t, fx, cfg)
	}
}

// The ceiling was settled on English traces; nine languages ship.
//
// Speech tokens per *text* token is a property of the orthography, so a constant
// tuned on one language is an assumption everywhere else, and the expensive
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
		t.Errorf("rows stopped by the ceiling = %v, want [es_short]: a new entry "+
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

// The decoder trapped in silence: the failure no tail rule can see.
//
// Every mute chunk and every mid-row hole in the interior-stall study shipped
// as clean, because all six other rules anchor on the tail. The stall rule
// condemns instead of cutting: the failure is a hole, and the fix is the retry
// ladder. Detection is two-class (a true-silence gate, a quiet-family
// continuation), which the fixture pins because single-set counting was
// measured broken.
func TestStall(t *testing.T) {
	fx := load(t)
	if len(fx.Stall.Cases) == 0 {
		t.Fatal("the fixture has no stall cases; nothing was compared")
	}
	for _, c := range fx.Stall.Cases {
		cfg := stallConfig(t, fx, c.RenderIds)
		got := IsStalled(build(t, c.Shape), c.HitCeiling, fx.SilenceTokenIDs, cfg)
		if got != c.Stalled {
			t.Errorf("%s: got %v, want %v (%s)", c.Name, got, c.Stalled, c.Why)
		}
	}
}

// The wiring is part of the contract: after repetition, before every tail
// rescue, condemned like dropout.
func TestStallResolver(t *testing.T) {
	fx := load(t)
	for _, c := range fx.Stall.Cases {
		cfg := stallConfig(t, fx, c.RenderIds)
		c.check(t, fx, cfg)
	}
}

// A cap-hit desperation cut that keeps less than any full read.
//
// The one row that survived the stall fix: soren/da0028 chunk 4 burned 132
// tokens to the ceiling and the seam cut kept 36, or 1.44 s in which 33 of the
// 36 kept tokens render near-silent through ids outside both manifest
// censuses, invisible to every set-membership rule. The keep's *length* is the
// only evidence there is: a keep under DesperationMinKeepPerTextToken per text
// token cannot hold a full read, so the verdict is condemned into the retry
// ladder. The cut stands as the keep: an exhausted ladder ships the trim,
// flagged, rather than the untrimmed babble. No censuses configured here on
// purpose: the trigger is a length test and must fire identically on a
// checkpoint packed before them.
func TestStarvedRescue(t *testing.T) {
	fx := load(t)
	cfg := configFrom(t, fx, "")
	if len(fx.StarvedRescue.Cases) == 0 {
		t.Fatal("the fixture has no starved_rescue cases; nothing was compared")
	}
	for _, c := range fx.StarvedRescue.Cases {
		c.check(t, fx, cfg)
	}
}

func TestStarvedRescueFloorIsExclusive(t *testing.T) {
	// keep == floor ships: `<`, not `<=`, so the pinned law has no ambiguity
	// at the boundary for a port to resolve differently. Text 20 puts the
	// floor at exactly 34.0.
	fx := load(t)
	cfg := configFrom(t, fx, "")
	row := build(t, [][]any{{"speech", 34.0}, {"sil", 12.0}, {"speech", 90.0}})
	got := Inspect(row, Request{
		TextTokenCount: 20, MinTokens: 24, EosPeakAt: -1, EosPeakProb: 0.0,
		Ended: false, IsTerminal: true, HitCeiling: true,
	}, fx.SilenceTokenIDs, cfg)
	if got.Reason != ReasonDesperation || got.Keep != 34 {
		t.Fatalf("got %+v, want a desperation cut at 34", got)
	}
	if got.Suspect {
		t.Errorf("a keep exactly at the floor is not under it")
	}
}

func TestStarvedRescueZeroDisablesTheTrigger(t *testing.T) {
	fx := load(t)
	cfg := configFrom(t, fx, "")
	cfg.DesperationMinKeepPerTextToken = 0.0
	c := fx.StarvedRescue.Cases[0]
	got := c.inspect(t, fx, cfg)
	if got.Reason != ReasonDesperation || got.Suspect {
		t.Errorf("zero must disable the trigger, got %+v", got)
	}
}

func TestStallNeverCuts(t *testing.T) {
	fx := load(t)
	cfg := stallConfig(t, fx, true)
	row := build(t, [][]any{{"speech", 30.0}, {"sil", 30.0}, {"speech", 30.0}})
	got := Inspect(row, Request{
		TextTokenCount: 40, MinTokens: 48, EosPeakAt: -1, EosPeakProb: 0.0,
		Ended: true, IsTerminal: true, HitCeiling: false,
	}, fx.SilenceTokenIDs, cfg)
	if got.Reason != ReasonStall {
		t.Fatalf("reason = %q, want %q", got.Reason, ReasonStall)
	}
	if got.Keep != len(row) {
		t.Errorf("a stalled row must be handed back whole; the hole is mid-row " +
			"and no cut can remove it")
	}
	if !got.Suspect {
		t.Errorf("the caller has to be told, since nothing was changed")
	}
}

func TestRepetitionOutranksStall(t *testing.T) {
	// A row that both loops and stalls answers to the loop: an exactly
	// repeated cycle pins where the failure began. Here the decoder resumed
	// after the region, so the loop condemns rather than cuts, but it still
	// outranks the stall's condemnation, and the verdict names the anchor
	// that was found.
	fx := load(t)
	cfg := stallConfig(t, fx, true)
	row := build(t, [][]any{{"cycle", 4.0, 8.0}, {"sil", 30.0}, {"speech", 20.0}})
	got := Inspect(row, Request{
		TextTokenCount: 40, MinTokens: 48, EosPeakAt: -1, EosPeakProb: 0.0,
		Ended: true, IsTerminal: true, HitCeiling: false,
	}, fx.SilenceTokenIDs, cfg)
	if got.Reason != ReasonRepetition {
		t.Fatalf("reason = %q, want %q: the exact anchor outranks the condemnation",
			got.Reason, ReasonRepetition)
	}
}

// The loop exemption keys on acoustic silence, not the sampler list.
//
// The specimen: kathleen/en0023 seed 1234 parked a mid-chunk pause on ids
// 6486 (x7) then 6405 (x24), both rendering true silence, both in the
// manifest's silence_render_ids, neither in the sampler's list alone. Keyed on
// the sampler list, the period-1 run fires as a loop and the cut deletes the
// pause plus two whole sentences of correctly-read speech behind it, verdict
// repetition, not suspect, audibly fluent. Six of the checkpoint's eight
// truly-silent ids sit outside the sampler list, so that key is blind on most
// real pauses. This one rule therefore unions the sampler list with both
// render censuses; the tail rules keep the list they were calibrated against.
func TestRepetitionSilence(t *testing.T) {
	fx := load(t)
	if len(fx.RepetitionSilence.Cases) == 0 {
		t.Fatal("the fixture has no repetition_silence cases; nothing was compared")
	}
	cfg := repSilenceConfig(t, fx)
	for _, c := range fx.RepetitionSilence.Cases {
		got := RepetitionCut(build(t, c.Shape), fx.SilenceTokenIDs, cfg)
		if got != want(c.Loop) {
			t.Errorf("%s: got %d, want %d (%s)", c.Name, got, want(c.Loop), c.Why)
		}
	}
}

// The cascade is part of the contract: a declined loop falls through to
// stall, which condemns the specimen's pause into the retry ladder instead of
// shipping the cut.
func TestRepetitionSilenceResolver(t *testing.T) {
	fx := load(t)
	cfg := repSilenceConfig(t, fx)
	for _, c := range fx.RepetitionSilence.Cases {
		c.check(t, fx, cfg)
	}
}

func TestSamplingNamesTheOldLaw(t *testing.T) {
	// The pre-amendment behaviour stays nameable: a checkpoint measured
	// under it can declare what it measured, and this is what it did: cut at
	// the pause and delete everything behind it.
	fx := load(t)
	cfg := repSilenceConfig(t, fx)
	cfg.RepetitionSilence = RepetitionSilenceSampling
	c := fx.RepetitionSilence.Cases[0]
	if got := RepetitionCut(build(t, c.Shape), fx.SilenceTokenIDs, cfg); got != 31 {
		t.Fatalf("got %d, want 31: the old law cut one token past the pause's start", got)
	}
}

func TestWithoutCensusesTheUnionIsTheSamplerList(t *testing.T) {
	// A checkpoint packed before the censuses changes nothing: nothing on
	// such a build knows id 12 is silent, so the run still reads as a loop
	// there, under either field value.
	fx := load(t)
	cfg := configFrom(t, fx, "")
	c := fx.RepetitionSilence.Cases[0]
	if got := RepetitionCut(build(t, c.Shape), fx.SilenceTokenIDs, cfg); got != 31 {
		t.Fatalf("got %d, want 31", got)
	}
}

// A loop the decoder resumed from is condemned, never cut.
//
// The census fix (RepetitionSilence) needs a manifest that names the silent
// ids, and the published pack has none: on it the en0023 pause fired again as
// a period-1 loop and the cut kept 52 of 206 tokens, deleting two sentences
// of correctly-read speech, verdict repetition, no retry. The guard here
// needs no silence knowledge at all: a genuine lock-up runs its cycle to the
// end of the row (a ceiling truncates at most one incomplete copy, period - 1
// tokens), so a qualifying loop followed by a full period or more of other
// content is a decoder that resumed, and a decoder that resumed was never
// locked. Such a row is handed back whole, suspect, into the retry ladder.
// The section configures no censuses on purpose: it is the arm
// RepetitionSilence cannot reach.
func TestRepetitionResume(t *testing.T) {
	// The bare rule still reports the loop; the law lives in the resolver.
	fx := load(t)
	if len(fx.RepetitionResume.Cases) == 0 {
		t.Fatal("the fixture has no repetition_resume cases; nothing was compared")
	}
	cfg := configFrom(t, fx, "")
	for _, c := range fx.RepetitionResume.Cases {
		got := RepetitionCut(build(t, c.Shape), fx.SilenceTokenIDs, cfg)
		if got != want(c.Loop) {
			t.Errorf("%s: got %d, want %d (%s)", c.Name, got, want(c.Loop), c.Why)
		}
	}
}

func TestRepetitionResumeResolver(t *testing.T) {
	fx := load(t)
	cfg := configFrom(t, fx, "")
	for _, c := range fx.RepetitionResume.Cases {
		c.check(t, fx, cfg)
	}
}

func TestCutNamesTheOldLaw(t *testing.T) {
	// The pre-amendment behaviour stays nameable: a checkpoint measured
	// under it can declare what it measured, and this is what it did: cut at
	// the pause and delete everything behind it.
	fx := load(t)
	cfg := configFrom(t, fx, "")
	cfg.RepetitionResume = RepetitionResumeCut
	c := fx.RepetitionResume.Cases[0]
	got := c.inspect(t, fx, cfg)
	if got.Reason != ReasonRepetition || got.Keep != want(c.Loop) || got.Suspect {
		t.Fatalf("got %+v, want the specimen's cut at %d: the old law shipped it",
			got, want(c.Loop))
	}
}

func TestTheCensusArmIsUntouched(t *testing.T) {
	// With the censuses configured the specimen's pause is exempt from the
	// loop rule entirely and stall condemns it: the RepetitionSilence
	// contract, byte for byte, guard or no guard.
	fx := load(t)
	cfg := repSilenceConfig(t, fx)
	c := fx.RepetitionSilence.Cases[0]
	got := c.inspect(t, fx, cfg)
	if got.Reason != ReasonStall || got.Keep != c.Expect.Keep {
		t.Fatalf("got %+v, want the stall condemnation at keep %d", got, c.Expect.Keep)
	}
}

// Early truncation: the failure a listener cannot hear.
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
