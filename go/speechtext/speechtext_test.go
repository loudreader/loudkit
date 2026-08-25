package speechtext

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// TestFunnelAgainstTheSharedFixture runs the cases every port is checked with.
//
// Hand-written cases in five languages are five tests of five different
// things. tests/data/conformance/speechtext.json is one test of one thing, and
// a disagreement names itself. That file's own note says so: "Every port must
// reproduce these exactly; a difference is a divergence, not a dialect". All
// the bindings read the
// `chunking` section and hand-write their funnel expectations; hand-written
// expectations alone are how three separate divergences (an uppercase language
// tag, non-ASCII digits, a typographic apostrophe sliced by bytes) can stay
// green in all three at once.
func TestFunnelAgainstTheSharedFixture(t *testing.T) {
	var fixture struct {
		Cases []struct {
			Text     string  `json:"text"`
			Language *string `json:"language"`
			Expected string  `json:"expected"`
		} `json:"cases"`
	}
	// LOUDKIT_FIXTURE_DIR, not LOUDKIT_FIXTURE: the two names are not
	// interchangeable. _DIR is the conformance directory, LOUDKIT_FIXTURE is
	// the vectors.json file inside it, which the weight-free suites read
	// directly. Joining "speechtext.json" onto the file path resolves to
	// vectors.json/speechtext.json and fails with "not a directory".
	dir := os.Getenv("LOUDKIT_FIXTURE_DIR")
	if dir == "" {
		dir = filepath.Join("..", "..", "tests", "data", "conformance")
	}
	raw, err := os.ReadFile(filepath.Join(dir, "speechtext.json"))
	if err != nil {
		t.Fatalf("cannot read the shared fixture: %v", err)
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatalf("cannot parse the shared fixture: %v", err)
	}
	// A renamed key would leave this loop comparing nothing and reporting a
	// pass, which is the failure this whole file exists to prevent.
	if len(fixture.Cases) == 0 {
		t.Fatal("the fixture has no cases; nothing was compared")
	}
	for _, c := range fixture.Cases {
		lang := ""
		if c.Language != nil {
			lang = *c.Language
		}
		if got := Prepared(c.Text, lang); got != c.Expected {
			t.Errorf("Prepared(%q, %q) = %q, want %q", c.Text, lang, got, c.Expected)
		}
	}
	t.Logf("%d cases compared", len(fixture.Cases))
}
