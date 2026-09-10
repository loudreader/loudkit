package speechtext

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
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

// TestTheGrammarIsParsedOnce fails if a second reader of numbers.json appears.
//
// The four tables in this package (numbers, dates, letters, unit words) read
// overlapping slices of one schema. A parse per table is a parse per table at
// startup and a place per table to forget a field when the grammar grows, so
// they share grammarDocument and this pins that.
func TestTheGrammarIsParsedOnce(t *testing.T) {
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	var sites []string
	for _, e := range entries {
		name := e.Name()
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		src, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		for i, line := range strings.Split(string(src), "\n") {
			if strings.Contains(line, "json.Unmarshal(numbersJSON") {
				sites = append(sites, fmt.Sprintf("%s:%d", name, i+1))
			}
		}
	}
	if len(sites) != 1 {
		t.Errorf("numbers.json is unmarshalled at %d sites (%v); it is parsed once, in grammarDocument",
			len(sites), sites)
	}
	// And the parse is idempotent: every caller sees the one table.
	if a, b := grammarDocument(), grammarDocument(); len(a) == 0 || len(a) != len(b) {
		t.Errorf("grammarDocument returned %d languages then %d", len(a), len(b))
	}
}
