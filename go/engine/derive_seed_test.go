package engine

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// The shipping derivation against the shared fixture.
//
// go/conformance pinned this by declaring phi and psi itself and recomputing
// the product, so it held the fixture to arithmetic written beside it rather
// than to the function every chunk seed in a real render goes through.
// Verified by mutation: changing the constant in deriveSeed left that suite
// green. Rust and JS had the same shape.
//
// In package engine rather than in conformance because deriveSeed is
// unexported and should stay so: exporting it to be testable would grow the
// package's surface to serve its own test.
func TestDeriveSeedMatchesTheSharedFixture(t *testing.T) {
	path := os.Getenv("LOUDKIT_FIXTURE")
	if path == "" {
		path = filepath.Join("..", "..", "tests", "data", "conformance", "vectors.json")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("fixture unreadable at %s: %v", path, err)
	}
	var doc map[string]interface{}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatalf("fixture is not JSON: %v", err)
	}
	seeds, ok := doc["seeds"].(map[string]interface{})
	if !ok {
		t.Fatal("the fixture has no seeds section; nothing was compared")
	}
	cases, ok := seeds["derivation"].([]interface{})
	if !ok || len(cases) == 0 {
		t.Fatal("seeds.derivation is empty; nothing was compared")
	}
	for _, item := range cases {
		p := item.(map[string]interface{})
		seed := uint64(p["seed"].(float64))
		stream := uint64(p["stream"].(float64))
		var want uint64
		if _, err := fmt.Sscanf(p["derived"].(string), "0x%x", &want); err != nil {
			t.Fatalf("bad derived value %q: %v", p["derived"], err)
		}
		if got := deriveSeed(seed, stream); got != want {
			t.Fatalf("seed %d stream %d: got %#x want %#x", seed, stream, got, want)
		}
	}
}
