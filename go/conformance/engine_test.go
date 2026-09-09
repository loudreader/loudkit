package conformance

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/loudreader/loudkit/go/chunking"
	"github.com/loudreader/loudkit/go/config"
	"github.com/loudreader/loudkit/go/engine"
	"github.com/loudreader/loudkit/go/onnx"
	"github.com/loudreader/loudkit/go/sampler"
	"github.com/loudreader/loudkit/go/speechtext"
	"github.com/loudreader/loudkit/go/voice"
)

// TestEngineConformance runs the ONNX engine against the shared end-to-end
// fixture: free-run tokens must be exact, fixed-token renders must land inside
// the fixture's correlation bands, and a long-form passage must produce the
// fixture's exact token stream in every one of its chunks. Needs the
// checkpoint, the exported graphs, the reference voice and the onnxruntime
// shared library; skips when any are absent.
func TestEngineConformance(t *testing.T) {
	ckpt := os.Getenv("LOUDKIT_CKPT")
	onnxDir := os.Getenv("LOUDKIT_ONNX_DIR")
	voicePath := os.Getenv("LOUDKIT_VOICE")
	lib := os.Getenv("LOUDKIT_ONNXRUNTIME_LIB")
	fixture := os.Getenv("LOUDKIT_FIXTURE_DIR")
	if fixture == "" {
		fixture = filepath.Join("..", "..", "tests", "data", "conformance")
	}
	if ckpt == "" || onnxDir == "" || voicePath == "" || lib == "" {
		skipOrFail(t, "set LOUDKIT_CKPT/LOUDKIT_ONNX_DIR/LOUDKIT_VOICE/LOUDKIT_ONNXRUNTIME_LIB")
	}
	if _, err := os.Stat(filepath.Join(fixture, "vectors.json")); err != nil {
		skipOrFail(t, "fixture not found: "+fixture)
	}

	onnx.SetSharedLibraryPath(lib)
	if err := onnx.InitializeEnvironment(); err != nil {
		t.Fatal(err)
	}
	defer onnx.DestroyEnvironment()

	// CPU by name, not the auto default: this fixture was generated on CPU
	// (tools/make_conformance.py pins the device for the same reason), so a
	// machine that offers CoreML or CUDA would otherwise measure a different
	// device and report the difference as a port that disagrees with Python.
	// What a GPU provider does to these tokens is a measurement to record, and
	// a parity gate is not where it belongs.
	eng, err := engine.LoadWith(ckpt, onnxDir, filepath.Join(fixture, "tokenizer.json"),
		config.ExecutionConfig{ONNXProvider: config.ProviderCPU})
	if err != nil {
		t.Fatal(err)
	}
	defer eng.Close()

	v, err := voice.Load(voicePath)
	if err != nil {
		t.Fatal(err)
	}

	buf, err := os.ReadFile(filepath.Join(fixture, fixtureName(eng.Config().DecodeMode)))
	if err != nil {
		t.Fatal(err)
	}
	var vectors map[string]interface{}
	if err := json.Unmarshal(buf, &vectors); err != nil {
		t.Fatal(err)
	}
	wantFingerprint := vectors["algorithm"].(map[string]interface{})["fingerprint"].(string)
	if got, want := eng.Fingerprint(), wantFingerprint; got != want {
		t.Fatalf("algorithm fingerprint: got %s, want %s", got, want)
	}
	cases := vectors["end_to_end"].([]interface{})

	for _, raw := range cases {
		c := raw.(map[string]interface{})
		name := c["name"].(string)
		seed := uint64(toFloat(c["seed"]))
		cfg := eng.Config()

		// free-run tokens: exact
		ids, err := eng.Encode(c["text"].(string), c["language"].(string))
		if err != nil {
			t.Fatal(err)
		}
		s := sampler.New(sampler.Config{
			Temperature:       cfg.Sampling.Temperature,
			RepetitionPenalty: cfg.Sampling.RepetitionPenalty,
			MinP:              cfg.Sampling.MinP,
			MaxNewTokens:      cfg.Sampling.MaxNewTokens,
			SilenceTokenIds:   cfg.Sampling.SilenceTokenIds,
		}, seed)
		started := time.Now()
		rawTok, err := eng.Generate(ids, v, s, nil, nil, nil)
		t.Logf("%s generation: %d tokens in %s (no prefix)", name, len(rawTok), time.Since(started))
		if err != nil {
			t.Fatal(err)
		}
		stripped := []int{}
		for _, tok := range rawTok {
			if tok < cfg.StartSpeech {
				stripped = append(stripped, tok)
			}
		}
		want := toInts(c["tokens"])
		if len(stripped) != len(want) {
			t.Fatalf("%s tokens: %d vs %d", name, len(stripped), len(want))
		}
		for i := range stripped {
			if stripped[i] != want[i] {
				t.Fatalf("%s token %d: got %d want %d", name, i, stripped[i], want[i])
			}
		}
		t.Logf("%s tokens: PASS (%d)", name, len(stripped))

		// The two public paths agree with the generator: chunk 0 draws the
		// caller's seed, so a text that fits one window renders the same
		// tokens through SynthesizeWindow and through Synthesize.
		opts := engine.Options{Seed: seed, Language: c["language"].(string)}
		window, err := eng.SynthesizeWindow(c["text"].(string), v, opts)
		if err != nil {
			t.Fatal(err)
		}
		whole, err := eng.Synthesize(c["text"].(string), v, opts)
		if err != nil {
			t.Fatal(err)
		}
		if !equalInts(window.Tokens, stripped) || !equalInts(whole.Tokens, stripped) {
			t.Fatalf("%s: SynthesizeWindow %d tokens, Synthesize %d tokens, generator %d: "+
				"the public paths and the generator disagree",
				name, len(window.Tokens), len(whole.Tokens), len(stripped))
		}

		// fixed-token render: within the band
		mel, err := eng.DecodeMel(want, v, derive(seed, 1))
		if err != nil {
			t.Fatal(err)
		}
		audio, err := eng.Vocode(mel, derive(seed, 2))
		if err != nil {
			t.Fatal(err)
		}
		melRef := readF32(t, filepath.Join(fixture, c["mel"].(map[string]interface{})["file"].(string)))
		wavRef := readF32(t, filepath.Join(fixture, c["wav"].(map[string]interface{})["file"].(string)))
		gates := c["gates"].(map[string]interface{})

		melCorr := corr(t, mel, melRef)
		waveCorr := corr(t, audio, wavRef)
		if melCorr < toFloat(gates["mel_corr"]) {
			t.Errorf("%s mel corr %.6f below gate %.6f", name, melCorr, toFloat(gates["mel_corr"]))
		}
		if waveCorr < toFloat(gates["wave_corr"]) {
			t.Errorf("%s wave corr %.4f below gate %.4f", name, waveCorr, toFloat(gates["wave_corr"]))
		}
		level := levelDB(t, audio, wavRef)
		if math.Abs(level) > toFloat(gates["wave_rms_db"]) {
			t.Errorf("%s level %+.4f dB against the reference, outside the %.4f dB band; "+
				"correlation cannot see this", name, level, toFloat(gates["wave_rms_db"]))
		}
		if peak := peakOf(audio); peak > 1 {
			t.Errorf("%s peak %.4f is outside the declared [-1, 1] waveform", name, peak)
		}
		t.Logf("%s render: mel %.6f wave %.4f level %+.4f dB", name, melCorr, waveCorr, level)
	}

	longForm(t, eng, v, vectors)
}

// longForm checks a passage too long for one window, chunk by chunk.
//
// Everything above it is a single window with an empty prefix, and with an
// empty prefix len(prefix)+step+1 and step+1 are the same number and a
// repetition mask seeded from the prefix is the empty one. This port wrote both
// short forms and this fixture passed anyway. A carried prefix is what tells
// them apart.
//
// Asserted per chunk rather than on the concatenation: a divergence inside
// chunk k shifts every token after it, so a whole-passage comparison reports
// one enormous mismatch instead of naming the chunk and the step.
//
// Deliberately not a t.Helper: the failures below name a chunk and a token
// index, and attributing them to the one-line call site would throw that away.
func longForm(t *testing.T, eng *engine.Engine, v *voice.Profile, vectors map[string]interface{}) {
	raw, ok := vectors["long_form"].(map[string]interface{})
	if !ok {
		skipOrFail(t, "fixture has no long_form section")
		return
	}
	prefixTokens := int(toFloat(raw["prefix_tokens"]))
	if got := eng.Config().Chunking.PrefixTokens; got != prefixTokens {
		t.Fatalf("this port carries %d tokens across a join, the fixture %d", got, prefixTokens)
	}

	for _, kase := range raw["cases"].([]interface{}) {
		c := kase.(map[string]interface{})
		name := c["name"].(string)
		language := c["language"].(string)
		// Funnel first, then split: the order the engine uses, and the order
		// the character budget assumes.
		prepared := speechtext.Prepared(c["text"].(string), language)
		if prepared != c["prepared"].(string) {
			t.Fatalf("%s: the speech funnel drifted:\n got %q\nwant %q",
				name, prepared, c["prepared"].(string))
		}
		chunks := c["chunks"].([]interface{})
		if len(chunks) < 2 {
			t.Fatalf("%s is a single window and proves nothing", name)
		}
		wantTexts := make([]string, len(chunks))
		for i, ch := range chunks {
			wantTexts[i] = ch.(map[string]interface{})["text"].(string)
		}
		gotTexts := chunking.SplitText(prepared, eng.Config().Chunking)
		if len(gotTexts) != len(wantTexts) {
			t.Fatalf("%s: split into %d chunks, fixture has %d", name, len(gotTexts), len(wantTexts))
		}
		for i := range gotTexts {
			if gotTexts[i] != wantTexts[i] {
				t.Fatalf("%s: chunk %d text moved, so every token below is about "+
					"different text:\n got %q\nwant %q", name, i, gotTexts[i], wantTexts[i])
			}
		}

		for _, ch := range chunks {
			chunk := ch.(map[string]interface{})
			index := int(toFloat(chunk["index"]))
			prefix := toInts(chunk["prefix"])
			want := toInts(chunk["tokens"])
			// The chain the streaming path walks: chunk k is conditioned on the
			// tail of chunk k-1. Spelled out in the fixture so a mismatch names
			// the carry rather than the tokens that followed from it.
			if index > 0 {
				previous := toInts(chunks[index-1].(map[string]interface{})["tokens"])
				end := len(previous)
				if eng.Config().DecodeMode == "fusion_mtp2" {
					end -= end % 2
				}
				start := max(0, end-prefixTokens)
				if eng.Config().DecodeMode == "fusion_mtp2" {
					start -= start % 2
				}
				tail := previous[start:end]
				if len(prefix) != len(tail) {
					t.Fatalf("carry length: got %d, want %d", len(prefix), len(tail))
				}
				for i := range tail {
					if prefix[i] != tail[i] {
						t.Fatalf("%s chunk %d: carry %v is not the previous chunk's tail %v",
							name, index, prefix, tail)
					}
				}
			}
			// Hex, because a derived 64-bit seed does not survive a JSON double.
			seed, err := strconv.ParseUint(
				strings.TrimPrefix(chunk["seed"].(string), "0x"), 16, 64)
			if err != nil {
				t.Fatal(err)
			}
			ids, err := eng.Encode(chunk["text"].(string), language)
			if err != nil {
				t.Fatal(err)
			}
			cfg := eng.Config()
			s := sampler.New(sampler.Config{
				Temperature:       cfg.Sampling.Temperature,
				RepetitionPenalty: cfg.Sampling.RepetitionPenalty,
				MinP:              cfg.Sampling.MinP,
				MaxNewTokens:      cfg.Sampling.MaxNewTokens,
				SilenceTokenIds:   cfg.Sampling.SilenceTokenIds,
			}, seed)
			started := time.Now()
			rawTok, err := eng.Generate(ids, v, s, nil, nil, prefix)
			t.Logf("%s chunk %d generation: %d tokens, prefix %d, %s", name, index, len(rawTok), len(prefix), time.Since(started))
			if err != nil {
				t.Fatal(err)
			}
			got := []int{}
			for _, tok := range rawTok {
				if tok < cfg.StartSpeech {
					got = append(got, tok)
				}
			}
			if len(got) != len(want) {
				t.Fatalf("%s chunk %d: %d tokens, fixture has %d", name, index, len(got), len(want))
			}
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("%s chunk %d: token %d is %d, fixture has %d",
						name, index, i, got[i], want[i])
				}
			}
		}
		// The engine's own long-form path, not the generator driven with the
		// fixture's seeds: this is what holds the chunk seed law (chunk 0 the
		// caller's seed, chunk k derive(seed, 16+k)) and the carry to the
		// fixture rather than the fixture to the test.
		whole, err := eng.Synthesize(c["text"].(string), v, engine.Options{
			Seed: uint64(toFloat(c["seed"])), Language: language,
		})
		if err != nil {
			t.Fatal(err)
		}
		publicTokens := c["tokens"]
		if public, ok := c["public_tokens"]; ok {
			publicTokens = public
		}
		if want := toInts(publicTokens); !equalInts(whole.Tokens, want) {
			for i := 0; i < min(len(want), len(whole.Tokens)); i++ {
				if want[i] != whole.Tokens[i] {
					t.Logf("first whole-token mismatch %d: got %v want %v", i, whole.Tokens[max(0, i-3):min(len(whole.Tokens), i+5)], want[max(0, i-3):min(len(want), i+5)])
					break
				}
			}
			t.Fatalf("%s: Synthesize produced %d tokens, the fixture's public list has %d: "+
				"the engine's own seed law or carry has drifted", name, len(whole.Tokens), len(want))
		}

		var streamedTokens []int
		var streamedAudio []float32
		count := 0
		err = eng.Stream(c["text"].(string), v, engine.Options{Seed: uint64(toFloat(c["seed"])), Language: language}, func(chunk engine.Chunk) bool {
			count++
			streamedTokens = append(streamedTokens, chunk.Tokens...)
			streamedAudio = append(streamedAudio, chunk.Audio...)
			return true
		})
		if err != nil {
			t.Fatal(err)
		}
		if count != len(chunks) || !equalInts(streamedTokens, whole.Tokens) {
			t.Fatalf("%s: stream changed chunk count or tokens", name)
		}
		if len(streamedAudio) != len(whole.Audio) {
			t.Fatalf("%s: repeat audio length changed", name)
		}
		for i, sample := range streamedAudio {
			if math.Float32bits(sample) != math.Float32bits(whole.Audio[i]) {
				t.Fatalf("%s: repeat audio changed at sample %d", name, i)
			}
		}
		t.Logf("%s long-form tokens: PASS (%d chunks)", name, len(chunks))
	}
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func derive(seed, stream uint64) uint64 {
	const phi = uint64(0x9e3779b97f4a7c15)
	const psi = uint64(0xbf58476d1ce4e5b9)
	return seed*phi + stream*psi
}

// corr is Pearson correlation on the explicit condition that the two inputs are
// the same length.
//
// Correlating min(len(a), len(b)) samples scores a
// truncated render perfectly against the prefix it managed to produce. The
// length is the finding in that case, so it is checked rather than absorbed.
func corr(t *testing.T, a, b []float32) float64 {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("length mismatch %d vs %d: correlating a prefix would hide a truncated render", len(a), len(b))
	}
	n := len(a)
	var ma, mb float64
	for i := 0; i < n; i++ {
		ma += float64(a[i])
		mb += float64(b[i])
	}
	ma /= float64(n)
	mb /= float64(n)
	var num, da, db float64
	for i := 0; i < n; i++ {
		x := float64(a[i]) - ma
		y := float64(b[i]) - mb
		num += x * y
		da += x * x
		db += y * y
	}
	return num / math.Sqrt(da*db)
}

// levelDB is the RMS ratio of a render to its reference, in dB.
//
// Correlation subtracts the mean and divides by the deviation, so it reports
// 1.0 for a render at half volume, at twenty times volume, or with a DC
// offset. Level is exactly what that normalisation discards, so it is the one
// amplitude fact worth its own gate.
func levelDB(t *testing.T, a, b []float32) float64 {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("length mismatch %d vs %d", len(a), len(b))
	}
	var sa, sb float64
	for i := range a {
		sa += float64(a[i]) * float64(a[i])
		sb += float64(b[i]) * float64(b[i])
	}
	if sa == 0 {
		t.Fatalf("rendered silence")
	}
	return 20 * math.Log10(math.Sqrt(sa/sb))
}

// peakOf is the loudest sample, against the [-1, 1] a waveform is declared to
// occupy. Everything downstream clips to that range, so a render outside it is
// audibly wrong and needs no tolerance to say so.
func peakOf(a []float32) float64 {
	peak := 0.0
	for _, v := range a {
		if abs := math.Abs(float64(v)); abs > peak {
			peak = abs
		}
	}
	return peak
}

func readF32(t *testing.T, path string) []float32 {
	buf, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(buf)/4)
	for i := range out {
		bits := uint32(buf[i*4]) | uint32(buf[i*4+1])<<8 | uint32(buf[i*4+2])<<16 | uint32(buf[i*4+3])<<24
		out[i] = math.Float32frombits(bits)
	}
	return out
}

// TestSynthesizeReportsHitTokenCap pins the truncation flag on the long-form
// result: false for a normal render, which ends at a stop token well under the
// cap. Python's synthesis layer declares every transport must report
// hit_token_cap: silent truncation presented as complete audio reads as
// complete to an agent, and this port computed the flag and dropped it.
func TestSynthesizeReportsHitTokenCap(t *testing.T) {
	ckpt := os.Getenv("LOUDKIT_CKPT")
	onnxDir := os.Getenv("LOUDKIT_ONNX_DIR")
	voicePath := os.Getenv("LOUDKIT_VOICE")
	lib := os.Getenv("LOUDKIT_ONNXRUNTIME_LIB")
	if ckpt == "" || onnxDir == "" || voicePath == "" || lib == "" {
		skipOrFail(t, "set LOUDKIT_CKPT/LOUDKIT_ONNX_DIR/LOUDKIT_VOICE/LOUDKIT_ONNXRUNTIME_LIB")
	}

	onnx.SetSharedLibraryPath(lib)
	if err := onnx.InitializeEnvironment(); err != nil {
		t.Fatal(err)
	}
	defer onnx.DestroyEnvironment()

	eng, err := engine.LoadWith(ckpt, onnxDir, tokenizerPath(t),
		config.ExecutionConfig{ONNXProvider: config.ProviderCPU})
	if err != nil {
		t.Fatal(err)
	}
	defer eng.Close()
	v, err := voice.Load(voicePath)
	if err != nil {
		t.Fatal(err)
	}

	out, err := eng.Synthesize("Hello from loudkit.", v, engine.Options{Seed: 4242})
	if err != nil {
		t.Fatal(err)
	}
	if out.HitTokenCap {
		t.Fatalf("a short sentence that ends at its stop token must not report HitTokenCap")
	}
	if len(out.Tokens) == 0 || len(out.Chunks) == 0 {
		t.Fatalf("the render produced no speech: %d tokens, %d chunks", len(out.Tokens), len(out.Chunks))
	}
}

// skipOrFail reports a missing prerequisite as a skip, or as a failure when
// LOUDKIT_REQUIRE_ASSETS is set.
//
// A skip is the correct result on a developer machine with no 1.27 GB
// checkpoint. On a runner that is supposed to have one, a missing asset is a
// broken environment, and a skipped conformance test is indistinguishable from
// a passing one in a CI summary. Same switch, same meaning, as the Python
// suite's requires() and the Rust conformance test.
func skipOrFail(t *testing.T, reason string) {
	t.Helper()
	if v := os.Getenv("LOUDKIT_REQUIRE_ASSETS"); v != "" && v != "0" {
		t.Fatalf("LOUDKIT_REQUIRE_ASSETS is set but %s", reason)
	}
	t.Skip(reason)
}

func fixtureName(mode string) string {
	if mode == "fusion_mtp2" {
		return "vectors_fusion_mtp2.json"
	}
	return "vectors.json"
}
