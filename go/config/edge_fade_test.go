package config

import (
	"strings"
	"testing"
)

// New releases name 20 ms. Historical manifests retain the implicit 5 ms identity.
func TestEdgeFadeIdentityLeavesTheShippedRampUnchanged(t *testing.T) {
	base, err := FromManifest(map[string]interface{}{"edge_fade_seconds": 0.02})
	if err != nil {
		t.Fatal(err)
	}
	legacy, err := FromManifest(map[string]interface{}{})
	if err != nil {
		t.Fatal(err)
	}
	if legacy.EdgeFade() != 0.005 || strings.Contains(CanonicalForm(legacy), "edge_fade_seconds") {
		t.Fatal("legacy manifests must keep their 5 ms identity")
	}
	if !strings.Contains(CanonicalForm(base), `"edge_fade_seconds":"0.02"`) {
		t.Fatal("the shipped ramp must appear in the canonical form")
	}
	zero := base
	zero.EdgeFadeSeconds = 0
	if CanonicalForm(zero) != CanonicalForm(base) || zero.EdgeFade() != timestretchEdgeFadeSeconds {
		t.Fatal("a config that never named a ramp must hash and fade like the shipped one")
	}
	other, err := FromManifest(map[string]interface{}{"edge_fade_seconds": 0.008})
	if err != nil {
		t.Fatal(err)
	}
	want := strings.Replace(CanonicalForm(base), `"edge_fade_seconds":"0.02"`, `"edge_fade_seconds":"0.008"`, 1)
	if CanonicalForm(other) != want {
		t.Fatalf("edge fade identity missing or misplaced: %s", CanonicalForm(other))
	}
	if other.EdgeFadeSeconds != 0.008 {
		t.Fatalf("EdgeFadeSeconds = %v, want 0.008", other.EdgeFadeSeconds)
	}
}
