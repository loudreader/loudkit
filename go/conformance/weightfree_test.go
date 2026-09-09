package conformance

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/loudreader/loudkit/go/chunking"
	"github.com/loudreader/loudkit/go/config"
	"github.com/loudreader/loudkit/go/frontend"
	"github.com/loudreader/loudkit/go/postprocess"
	"github.com/loudreader/loudkit/go/rng"
	"github.com/loudreader/loudkit/go/sampler"
)

func fixturePath(t *testing.T) string {
	p := os.Getenv("LOUDKIT_FIXTURE")
	if p == "" {
		p = filepath.Join("..", "..", "tests", "data", "conformance", "vectors.json")
	}
	if _, err := os.Stat(p); err != nil {
		// Committed fixture, so absence is breakage. Skipping made the whole
		// weight-free suite report success on a tree that could not run it.
		t.Fatalf("fixture not found: %s (%v)", p, err)
	}
	return p
}

func tokenizerPath(t *testing.T) string {
	p := os.Getenv("LOUDKIT_TOKENIZER")
	if p == "" {
		p = filepath.Join("..", "..", "tests", "data", "conformance", "tokenizer.json")
	}
	if _, err := os.Stat(p); err != nil {
		t.Fatalf("tokenizer not found: %s (%v)", p, err)
	}
	return p
}

func loadVectors(t *testing.T) map[string]interface{} {
	buf, err := os.ReadFile(fixturePath(t))
	if err != nil {
		t.Fatal(err)
	}
	var v map[string]interface{}
	if err := json.Unmarshal(buf, &v); err != nil {
		t.Fatal(err)
	}
	return v
}

// requireCases fails when a fixture section is empty.
//
// Every loop in this file ranges over a slice pulled out of the fixture by key.
// A regeneration that renamed one (`philox` to `rng`, say) would leave the
// loop comparing nothing and the test reporting a pass, switching the entire
// cross-language determinism claim off silently.
func requireCases(t *testing.T, section map[string]interface{}, key string) []interface{} {
	t.Helper()
	raw, ok := section[key]
	if !ok {
		t.Fatalf("the fixture has no %q section; nothing was compared", key)
	}
	list, ok := raw.([]interface{})
	if !ok {
		t.Fatalf("fixture section %q is %T, not a list", key, raw)
	}
	if len(list) == 0 {
		t.Fatalf("fixture section %q is empty; nothing was compared", key)
	}
	return list
}

func TestPhiloxKAT(t *testing.T) {
	philox := loadVectors(t)["philox"].(map[string]interface{})
	katCases := requireCases(t, philox, "kat")
	for _, raw := range katCases {
		c := raw.(map[string]interface{})
		counter := c["counter"].([]interface{})
		key := c["key"].([]interface{})
		want := c["expected"].([]interface{})
		got := rng.Philox4x32(
			uint32(toFloat(counter[0])), uint32(toFloat(counter[1])),
			uint32(toFloat(counter[2])), uint32(toFloat(counter[3])),
			uint32(toFloat(key[0])), uint32(toFloat(key[1])),
		)
		for i := 0; i < 4; i++ {
			if got[i] != uint32(toFloat(want[i])) {
				t.Fatalf("counter %v: stream %d = %d, want %d", counter, i, got[i], int(toFloat(want[i])))
			}
		}
	}
}

func TestUniformBits(t *testing.T) {
	philox := loadVectors(t)["philox"].(map[string]interface{})
	uniformBitsCases := requireCases(t, philox, "uniform_bits")
	for _, raw := range uniformBitsCases {
		p := raw.(map[string]interface{})
		var seed uint64
		// Checked, not dropped: a fixture whose seed field changes
		// shape parses as nothing and the case then runs on seed 0, which
		// passes or fails for reasons that have nothing to do with the port.
		if n, err := fmt.Sscanf(p["seed"].(string), "0x%x", &seed); n != 1 || err != nil {
			t.Fatalf("seed %v is not a hex string: %v", p["seed"], err)
		}
		u := rng.Uniforms(seed, uint32(toFloat(p["stream"])), int(toFloat(p["step0"])),
			int(toFloat(p["n_steps"])), int(toFloat(p["width"])))
		got := make([]uint32, len(u))
		for i, x := range u {
			got[i] = uint32(math.Round(x*4294967296 - 0.5))
		}
		want := []uint32{}
		for _, row := range p["bits"].([]interface{}) {
			for _, b := range row.([]interface{}) {
				want = append(want, uint32(toFloat(b)))
			}
		}
		if len(got) != len(want) {
			t.Fatalf("seed %s: length %d vs %d", p["seed"], len(got), len(want))
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("seed %s idx %d: got %d want %d", p["seed"], i, got[i], want[i])
			}
		}
	}
}

func TestGumbel(t *testing.T) {
	philox := loadVectors(t)["philox"].(map[string]interface{})
	gumbelCases := requireCases(t, philox, "gumbel")
	for _, raw := range gumbelCases {
		p := raw.(map[string]interface{})
		var seed uint64
		// See the check in TestUniformBits.
		if n, err := fmt.Sscanf(fmt.Sprintf("%v", p["seed"]), "%d", &seed); n != 1 || err != nil {
			t.Fatalf("seed %v is not a decimal number: %v", p["seed"], err)
		}
		g := rng.GumbelNoise(seed, uint32(toFloat(p["stream"])), int(toFloat(p["step"])),
			1, int(toFloat(p["width"])))
		vals := p["values"].([]interface{})
		for i, w := range vals {
			rel := math.Abs((g[i] - toFloat(w)) / toFloat(w))
			if rel > 1e-12 {
				t.Fatalf("seed %d idx %d: rel %e", seed, i, rel)
			}
		}
	}
}

func TestSampler(t *testing.T) {
	sam := loadVectors(t)["sampler"].(map[string]interface{})
	cases := requireCases(t, sam, "cases")
	for _, raw := range cases {
		c := raw.(map[string]interface{})
		cfgMap := c["config"].(map[string]interface{})
		cfg := sampler.Config{
			Temperature:       toFloat(cfgMap["temperature"]),
			RepetitionPenalty: toFloat(cfgMap["repetition_penalty"]),
			MinP:              toFloat(cfgMap["min_p"]),
			MaxNewTokens:      int(toFloat(cfgMap["max_new_tokens"])),
			SilenceTokenIds:   toInts(cfgMap["silence_token_ids"]),
		}
		s := sampler.New(cfg, uint64(toFloat(c["seed"])))
		var rows [][]float32
		if recipe, ok := c["logits_recipe"].(map[string]interface{}); ok {
			r := recipe
			for step := 0; step < int(toFloat(r["steps"])); step++ {
				u := rng.Uniforms(uint64(toFloat(r["seed"])), uint32(toFloat(r["stream"])),
					step, 1, int(toFloat(r["vocab"])))
				row := make([]float32, int(toFloat(r["vocab"])))
				for i, x := range u {
					row[i] = float32(x*toFloat(r["scale"]) + toFloat(r["offset"]))
				}
				rows = append(rows, row)
			}
		} else {
			logits := c["logits"].([]interface{})
			row := []float32{}
			for _, x := range logits[0].([]interface{}) {
				row = append(row, float32(toFloat(x)))
			}
			repeat := int(toFloat(c["repeat_logits"]))
			if repeat == 0 {
				repeat = len(logits)
			}
			for i := 0; i < repeat; i++ {
				rows = append(rows, row)
			}
		}
		seen := make([]bool, len(rows[0]))
		got := []int{}
		for step, row := range rows {
			tok := s.Call(row, step, seen)
			got = append(got, tok)
			seen[tok] = true
		}
		want := toInts(c["expected"])
		if len(got) != len(want) {
			t.Fatalf("%s: %d tokens vs %d", c["name"], len(got), len(want))
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("%s step %d: got %d want %d", c["name"], i, got[i], want[i])
			}
		}
	}
}

func TestFrontend(t *testing.T) {
	fe, err := frontend.Load(tokenizerPath(t))
	if err != nil {
		t.Fatal(err)
	}
	fixture := loadVectors(t)["frontend"].(map[string]interface{})
	cases := requireCases(t, fixture, "cases")
	for _, raw := range cases {
		c := raw.(map[string]interface{})
		ids, err := fe.Encode(c["text"].(string), c["language"].(string))
		if err != nil {
			t.Fatal(err)
		}
		want := toInts(c["ids"])
		if len(ids) != len(want) {
			t.Fatalf("%q: %d ids vs %d", c["text"], len(ids), len(want))
		}
		for i := range ids {
			if ids[i] != want[i] {
				t.Fatalf("%q idx %d: got %d want %d", c["text"], i, ids[i], want[i])
			}
		}
	}
}

// TestTheVocabularyCeilingIsKnown pins the ceiling engine.Load checks the
// checkpoint's text embedding table against.
//
// Encode can return any id in the vocabulary, and every one of them indexes
// that table; without the check a tokenizer paired with a checkpoint from
// another release reads past its end mid-synthesis. The shipped weights carry 2454 rows
// (TorchTokenGenerator.TEXT_VOCAB), so 2453 is the last id that fits: the
// margin is one row, which is why a regenerated fixture must show up here, as a
// line to read, rather than in a panic on someone's laptop.
func TestTheVocabularyCeilingIsKnown(t *testing.T) {
	fe, err := frontend.Load(tokenizerPath(t))
	if err != nil {
		t.Fatal(err)
	}
	if got := fe.MaxTokenID(); got != 2453 {
		t.Fatalf("max token id %d, want 2453", got)
	}
}

// The seed contract's *fixture* rows are pinned against the shipping
// `engine.deriveSeed` in go/engine/derive_seed_test.go, which can reach the
// unexported function. A copy of the formula here, recomputed from constants
// declared beside it, would leave this suite green while deriveSeed itself
// drifted.

func toFloat(x interface{}) float64 {
	switch v := x.(type) {
	case float64:
		return v
	case int:
		return float64(v)
	case int64:
		return float64(v)
	case json.Number:
		f, _ := v.Float64()
		return f
	}
	return 0
}

func toInts(x interface{}) []int {
	out := []int{}
	switch v := x.(type) {
	case []interface{}:
		for _, e := range v {
			out = append(out, int(toFloat(e)))
		}
	case []float64:
		for _, e := range v {
			out = append(out, int(e))
		}
	}
	return out
}

// TestFingerprintMatchesTheSharedFixture pins the whole algorithm config in one
// comparison.
//
// Every other check in this file compares a behaviour somebody thought to
// compare. This compares the entire configuration, so a field nobody wrote a
// test for still cannot drift: the failure mode is concrete: an euler_grid
// ignored by one port, a silence_token_ids that accepts a string, and a
// chunking.prefix_tokens guessed rather than read. This finds the next one for
// free.
func TestFingerprintMatchesTheSharedFixture(t *testing.T) {
	algorithm, ok := loadVectors(t)["algorithm"].(map[string]interface{})
	if !ok {
		t.Fatal("the fixture has no algorithm section; nothing was compared")
	}

	// The production algorithm, spelled out rather than loaded, so this runs
	// with no checkpoint: the fingerprint is a property of the values, and the
	// values are what the fixture pins.
	//
	// The render-id censuses are properties of the weights: the manifest
	// carries them at its top level, beside silence_token_ids, so like that
	// list they are spelled out here rather than defaulted by Production().
	pp := postprocess.Production()
	pp.SilenceRenderIds = []int{4137, 4215, 4218, 4299, 6162, 6324, 6405, 6486}
	pp.QuietRenderIds = []int{
		1458, 1461, 1488, 1701, 1704, 1707, 1716, 1731, 1785, 1788, 1869, 1947,
		1950, 1951, 1959, 1978, 2028, 2031, 2040, 2058, 2076, 2112, 2139, 3645,
		3648, 3651, 3704, 3888, 3894, 4188, 5838, 6081, 6183, 6537,
	}
	cfg := config.AlgorithmConfig{
		RecipeVersion:   "loudkit-1",
		Guidance:        "single_path",
		GuidanceRate:    0.0,
		EulerSteps:      2,
		EulerGrid:       nil,
		SampleRate:      24000,
		TokenRateHz:     25.0,
		SpeechVocabSize: 8194,
		StartSpeech:     6561,
		StopSpeech:      6562,
		Window:          config.ProductionWindow(),
		Chunking:        chunking.Production(),
		Postprocess:     pp,
		Sampling: config.SamplingConfig{
			Temperature:        0.8,
			RepetitionPenalty:  1.2,
			MinP:               0.05,
			MaxNewTokens:       255,
			MinTokensFloor:     10,
			MinTokensTextRatio: 1.2,
			SilenceTokenIds: []int{
				1731, 1821, 1822, 1824, 1975, 2058, 2068, 3190, 3377, 3918, 3927, 3928,
				3930, 4008, 4009, 4011, 4012, 4137, 4146, 4161, 4171, 4173, 4174, 4218,
				4245, 4251, 4252, 4254, 4255, 4260, 4282,
			},
		},
	}

	// The blob first: a mismatch there names the field that drifted, while a
	// mismatch in the hash alone says only that something did.
	if got, want := config.CanonicalForm(cfg), algorithm["canonical_form"].(string); got != want {
		t.Errorf("canonical form differs\n got: %s\nwant: %s", got, want)
	}
	if got, want := config.Fingerprint(cfg), algorithm["fingerprint"].(string); got != want {
		t.Errorf("fingerprint = %s, want %s", got, want)
	}
}

// The stop-token observation the postprocess layer reads.
//
// Pinned across languages because it is hand-written in five of them and it is
// *audible*: two of the detector rules compare it against a threshold, so a
// port that computes it differently cuts a chunk somewhere else. The quantity
// has two subtleties either of which a reimplementation gets wrong silently,
// the numerator is the stop token's weight taken BEFORE the min_p cutoff, and
// the peak is recorded only PAST the floor.
func TestEOSPeakMatchesTheSharedFixture(t *testing.T) {
	section, ok := loadVectors(t)["eos_peak"].(map[string]interface{})
	if !ok {
		t.Fatal("the fixture has no eos_peak section; nothing was compared")
	}
	rtol := toFloat(section["prob_rtol"])
	for _, raw := range requireCases(t, section, "cases") {
		c := raw.(map[string]interface{})
		cfgMap := c["config"].(map[string]interface{})
		s := sampler.New(sampler.Config{
			Temperature:       toFloat(cfgMap["temperature"]),
			RepetitionPenalty: toFloat(cfgMap["repetition_penalty"]),
			MinP:              toFloat(cfgMap["min_p"]),
			SilenceTokenIds:   toInts(cfgMap["silence_token_ids"]),
		}, uint64(toFloat(c["seed"])))
		s.ObserveEOS(int(toFloat(c["stop_token"])), int(toFloat(c["eos_floor"])))

		r := c["logits_recipe"].(map[string]interface{})
		vocab := int(toFloat(r["vocab"]))
		seen := make([]bool, vocab)
		for step := 0; step < int(toFloat(r["steps"])); step++ {
			u := rng.Uniforms(uint64(toFloat(r["seed"])), uint32(toFloat(r["stream"])),
				step, 1, vocab)
			row := make([]float32, vocab)
			for i, x := range u {
				row[i] = float32(x*toFloat(r["scale"]) + toFloat(r["offset"]))
			}
			seen[s.Call(row, step, seen)] = true
		}
		at, prob := s.EOSPeak()
		wantAt := int(toFloat(c["expected_at"]))
		wantProb := toFloat(c["expected_prob"])
		if at != wantAt {
			t.Errorf("%s: peak at %d, want %d", c["name"], at, wantAt)
		}
		if math.Abs(prob-wantProb) > rtol*math.Abs(wantProb) {
			t.Errorf("%s: peak prob %g, want %g", c["name"], prob, wantProb)
		}
	}
}

// TestResplitPlan holds this port to the `cap_resplit` law without weights.
//
// The token streams need a model; the *plan* does not. Which window carries
// which index, where the split falls, and which seed each window draws are all
// computable from the fixture alone, and they are exactly the three things that
// went wrong while this law was being written: a second half seeded from the
// next chunk's stream, a Swift split that counted grapheme clusters, and a
// queue that did not advance. A port that gets any of them wrong produces
// different audio and every other test still passes.
func TestResplitPlan(t *testing.T) {
	vectors := loadVectors(t)
	raw, ok := vectors["resplit"].(map[string]interface{})
	if !ok {
		// The section this test exists for. Skipping when it is absent meant
		// deleting it from the fixture would have retired the law in silence.
		t.Fatal("fixture has no resplit section; nothing was compared")
	}
	resplitStream := uint64(toFloat(raw["resplit_stream"]))
	chunkBase := uint64(toFloat(raw["chunk_stream_base"]))
	prefixTokens := int(toFloat(raw["prefix_tokens"]))

	for _, kase := range raw["cases"].([]interface{}) {
		c := kase.(map[string]interface{})
		name := c["name"].(string)
		seed := uint64(toFloat(c["seed"]))
		// The case moves the window, so the chunk budget moves with it: the
		// config refuses a budget larger than the window, and the engine gates
		// the re-split on the window rather than on any cap.
		cfg := chunking.Production()
		cfg.MaxTokens = int(toFloat(c["window"]))
		if cfg.CapResplit != chunking.WordCapResplit {
			t.Fatalf("this port ships cap_resplit=%q; the fixture pins the law", cfg.CapResplit)
		}
		prepared := c["prepared"].(string)
		windows := c["windows"].([]interface{})
		texts := chunking.SplitText(prepared, cfg)

		tail := func(w map[string]interface{}) []int {
			all := toInts(w["tokens"])
			if len(all) <= prefixTokens {
				return all
			}
			return all[len(all)-prefixTokens:]
		}
		check := func(at int, text string, sd uint64, pre []int, split bool, index int) {
			w := windows[at].(map[string]interface{})
			if got := int(toFloat(w["index"])); got != index {
				t.Fatalf("%s window %d: index %d, fixture %d: a moved index moves "+
					"every later chunk's seed", name, at, index, got)
			}
			if w["text"].(string) != text {
				t.Fatalf("%s window %d: text %q, fixture %q", name, at, text, w["text"])
			}
			if w["split"].(bool) != split {
				t.Fatalf("%s window %d: split=%v, fixture %v", name, at, split, w["split"])
			}
			want := strings.TrimPrefix(w["seed"].(string), "0x")
			if fmt.Sprintf("%x", sd) != want {
				t.Fatalf("%s window %d: seed %x, fixture %s: the second half must draw "+
					"from its own stream off the chunk seed", name, at, sd, want)
			}
			got := toInts(w["prefix"])
			if len(got) != len(pre) {
				t.Fatalf("%s window %d: carry length %d, fixture %d", name, at, len(pre), len(got))
			}
			for i := range got {
				if got[i] != pre[i] {
					t.Fatalf("%s window %d: carry %v, fixture %v", name, at, pre, got)
				}
			}
		}

		// Chunk 0 draws the caller's seed itself; the base applies from chunk 1 up.
		chunkSeed := func(index int) uint64 {
			if index == 0 {
				return seed
			}
			return resplitDerive(seed, chunkBase+uint64(index))
		}

		wi := 0
		var carry []int
		for index, chunkText := range texts {
			wasSplit := windows[wi].(map[string]interface{})["split"].(bool)
			first, second, splittable := chunking.SplitInHalf(chunkText)
			if wasSplit {
				if !splittable {
					t.Fatalf("%s chunk %d: the fixture split it and this port "+
						"finds no word boundary", name, index)
				}
				check(wi, first, chunkSeed(index), carry, true, index)
				pre := tail(windows[wi].(map[string]interface{}))
				wi++
				check(wi, second,
					resplitDerive(chunkSeed(index), resplitStream),
					pre, true, index)
				carry = tail(windows[wi].(map[string]interface{}))
				wi++
				continue
			}
			check(wi, chunkText, chunkSeed(index), carry, false, index)
			carry = tail(windows[wi].(map[string]interface{}))
			wi++
		}
		if wi != len(windows) {
			t.Fatalf("%s: planned %d windows, fixture has %d", name, wi, len(windows))
		}
		t.Logf("%s re-split plan: PASS (%d windows from %d chunks)", name, wi, len(texts))
	}
}

// resplitDerive mirrors deriveSeed in package engine, which is unexported.
func resplitDerive(seed, stream uint64) uint64 {
	return seed*0x9E3779B97F4A7C15 + stream*0xBF58476D1CE4E5B9
}
