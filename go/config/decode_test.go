package config

import (
	"strings"
	"testing"
)

func TestDecodeIdentityLeavesSingleUnchanged(t *testing.T) {
	single, err := FromManifest(map[string]interface{}{})
	if err != nil {
		t.Fatal(err)
	}
	zero := single
	zero.DecodeMode = ""
	if CanonicalForm(single) != CanonicalForm(zero) {
		t.Fatal("single fingerprint changed")
	}
	fused, err := FromManifest(map[string]interface{}{"decode": map[string]interface{}{"mode": "fusion_mtp2"}})
	if err != nil {
		t.Fatal(err)
	}
	want := strings.Replace(CanonicalForm(single), `,"euler_grid":`, `,"decode_mode":"fusion_mtp2","euler_grid":`, 1)
	if CanonicalForm(fused) != want {
		t.Fatalf("decode identity missing: %s", CanonicalForm(fused))
	}
}
