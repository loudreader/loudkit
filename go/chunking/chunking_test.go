package chunking

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type chunkCase struct {
	Config       string   `json:"config"`
	MaxTokens    int      `json:"max_tokens"`
	PrefixTokens int      `json:"prefix_tokens"`
	SplitOn      []string `json:"split_on"`
	Text         string   `json:"text"`
	Chunks       []string `json:"chunks"`
}

// The splitter must cut where the shared fixture says.
//
// Where the splits fall is audible, so a different split is a different
// reading — not a formatting choice. This port had no long-form path at all
// while the documentation called it supported and conformance-verified.
func TestSplitTextMatchesTheSharedFixture(t *testing.T) {
	dir := os.Getenv("LOUDKIT_FIXTURE_DIR")
	if dir == "" {
		dir = filepath.Join("..", "..", "tests", "data", "conformance")
	}
	raw, err := os.ReadFile(filepath.Join(dir, "speechtext.json"))
	if err != nil {
		t.Fatalf("fixture not found: %v", err)
	}
	var payload struct {
		Chunking []chunkCase `json:"chunking"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatal(err)
	}
	if len(payload.Chunking) == 0 {
		t.Fatal("the fixture carries no chunking cases")
	}

	for _, c := range payload.Chunking {
		got := SplitText(c.Text, Config{
			Enabled:      true,
			MaxTokens:    c.MaxTokens,
			PrefixTokens: c.PrefixTokens,
			SplitOn:      c.SplitOn,
		})
		if len(got) != len(c.Chunks) {
			t.Fatalf("%s: %d chunks, want %d\n got: %q\nwant: %q",
				c.Config, len(got), len(c.Chunks), got, c.Chunks)
		}
		for i := range got {
			if got[i] != c.Chunks[i] {
				t.Fatalf("%s chunk %d:\n got: %q\nwant: %q",
					c.Config, i, got[i], c.Chunks[i])
			}
		}
	}
}

// The constant is shared arithmetic, not a tuning knob: a port that picks a
// different value splits in different places and reads the text differently.
func TestCharsPerTokenMatchesPython(t *testing.T) {
	if CharsPerToken != 0.5 {
		t.Fatalf("CharsPerToken is %v; loudkit.frontend.chunking.CHARS_PER_TOKEN is 0.5", CharsPerToken)
	}
}

// TestValidateRefusesTheConfigsPythonRefuses pins the four refusals
// loudkit.config.ChunkConfig.__post_init__ makes.
//
// This was a plain struct that read MaxTokens straight from the manifest and
// accepted all of them. The zero-budget one is why it matters: SplitText cuts
// nothing and loops forever, which on a server is a wedged request holding the
// single-flight engine. Python fixed that in d8742aa, on the Python side only.
func TestValidateRefusesTheConfigsPythonRefuses(t *testing.T) {
	good := Production()
	if err := good.Validate(); err != nil {
		t.Fatalf("the shipping recipe must validate: %v", err)
	}
	for _, c := range []struct {
		name string
		cfg  Config
		want string
	}{
		{"zero max", Config{MaxTokens: 0, SplitOn: []string{". "}}, "must be positive"},
		{"no budget", Config{MaxTokens: 1, SplitOn: []string{". "}}, "no character budget"},
		{"prefix >= max", Config{MaxTokens: 20, PrefixTokens: 20, SplitOn: []string{". "}},
			"prefix_tokens must be in"},
		{"no separators", Config{MaxTokens: 20, PrefixTokens: 6}, "nowhere to break"},
	} {
		err := c.cfg.Validate()
		if err == nil {
			t.Errorf("%s: accepted a config Python refuses", c.name)
			continue
		}
		if !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: got %q, want it to mention %q", c.name, err, c.want)
		}
	}
}
