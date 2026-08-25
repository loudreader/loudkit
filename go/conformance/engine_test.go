package conformance

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

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

	buf, err := os.ReadFile(filepath.Join(fixture, "vectors.json"))
	if err != nil {
		t.Fatal(err)
	}
	var vectors map[string]interface{}
	if err := json.Unmarshal(buf, &vectors); err != nil {
		t.Fatal(err)
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
		rawTok, err := eng.Generate(ids, v, s, nil, nil, nil)
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
		t.Logf("%s render: mel %.6f wave %.4f", name, melCorr, waveCorr)
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
		// Funnel first, then split — the order the engine uses, and the order
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
				tail := previous[len(previous)-prefixTokens:]
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
			rawTok, err := eng.Generate(ids, v, s, nil, nil, prefix)
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
		t.Logf("%s long-form tokens: PASS (%d chunks)", name, len(chunks))
	}
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
// hit_token_cap — silent truncation presented as complete audio reads as
// complete to an agent — and this port computed the flag and dropped it.
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

	_, tokens, _, chunks, _, capped, err := eng.SynthesizeLong(
		"Hello from loudkit.", v, 4242, "", 1.0, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if capped {
		t.Fatalf("a short sentence that ends at its stop token must not report hitTokenCap")
	}
	if len(tokens) == 0 || len(chunks) == 0 {
		t.Fatalf("the render produced no speech: %d tokens, %d chunks", len(tokens), len(chunks))
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
