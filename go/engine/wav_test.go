package engine

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// The WAV writer against the shared fixtures, which Python generates from its
// own quantiser and encoder: the same floats become the same bytes in five
// implementations.

func readFixture(t *testing.T, name string, into any) {
	t.Helper()
	path := filepath.Join("..", "..", "tests", "data", "conformance", name)
	if env := os.Getenv("LOUDKIT_FIXTURE_DIR"); env != "" {
		path = filepath.Join(env, name)
	}
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(body, into); err != nil {
		t.Fatal(err)
	}
}

// probe is a float or the string "nan", which JSON cannot spell.
func probe(t *testing.T, raw json.RawMessage) float32 {
	t.Helper()
	var s string
	if json.Unmarshal(raw, &s) == nil {
		if s != "nan" {
			t.Fatalf("unknown probe %q", s)
		}
		return float32(math.NaN())
	}
	var f float64
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	return float32(f)
}

func TestQuantiseIsTheSharedRule(t *testing.T) {
	var fixture struct {
		Cases []struct {
			X     json.RawMessage `json:"x"`
			PCM16 int16           `json:"pcm16"`
		} `json:"cases"`
	}
	readFixture(t, "wav_quantise.json", &fixture)
	if len(fixture.Cases) < 10 {
		t.Fatalf("the fixture holds %d probes; nothing was compared", len(fixture.Cases))
	}
	for _, c := range fixture.Cases {
		x := probe(t, c.X)
		if got := quantise(x); got != c.PCM16 {
			t.Errorf("quantise(%v) = %d, want %d", x, got, c.PCM16)
		}
	}
}

func TestWriteWavBytesAreTheFixtures(t *testing.T) {
	var fixture struct {
		Cases []struct {
			Samples    []json.RawMessage `json:"samples"`
			SampleRate int               `json:"sample_rate"`
			Hex        string            `json:"hex"`
		} `json:"cases"`
	}
	readFixture(t, "wav_header.json", &fixture)
	if len(fixture.Cases) < 3 {
		t.Fatalf("the fixture holds %d cases; nothing was compared", len(fixture.Cases))
	}
	for _, c := range fixture.Cases {
		audio := make([]float32, len(c.Samples))
		for i, raw := range c.Samples {
			audio[i] = probe(t, raw)
		}
		r := &Result{Audio: audio, SampleRate: c.SampleRate}
		var buf bytes.Buffer
		if err := r.WriteWav(&buf); err != nil {
			t.Fatal(err)
		}
		if got := hex.EncodeToString(buf.Bytes()); got != c.Hex {
			t.Errorf("wav bytes for %d samples at %d Hz\n got %s\nwant %s",
				len(audio), c.SampleRate, got, c.Hex)
		}
	}
}

func TestWriteWavRefusesNoSampleRate(t *testing.T) {
	r := &Result{Audio: []float32{0}}
	if err := r.WriteWav(&bytes.Buffer{}); err == nil {
		t.Error("a WAV without a sample rate is not a WAV")
	}
}

func TestResultDuration(t *testing.T) {
	r := &Result{Audio: make([]float32, 24000), SampleRate: 24000}
	if got := r.Duration().Seconds(); math.Abs(got-1) > 1e-9 {
		t.Errorf("duration = %v", got)
	}
	if got := (&Result{}).Duration(); got != 0 {
		t.Errorf("duration of nothing = %v", got)
	}
}
