package checkpoint

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A safetensors file carrying nothing but a manifest.
//
// Small enough to write in a test and complete enough for Open to reach every
// check it makes, which is the point: asserting on the two constants alone
// left both guards deletable with the tests still green.
func manifestOnly(t *testing.T, manifest string) string {
	t.Helper()
	header, err := json.Marshal(map[string]any{
		"__metadata__": map[string]string{"manifest": manifest},
	})
	if err != nil {
		t.Fatal(err)
	}
	buf := make([]byte, 8, 8+len(header))
	binary.LittleEndian.PutUint64(buf, uint64(len(header)))
	buf = append(buf, header...)
	path := filepath.Join(t.TempDir(), "ckpt.safetensors")
	if err := os.WriteFile(path, buf, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// The loader, not the constant.
//
// Three manifests: the one every 0.1.0 checkpoint carries, the one this engine
// has not implemented the loop for, and the one that understates its version,
// which clears the number gate and would have been run through the one-token
// loop, producing fluent speech that is not the text.
func TestTheLoaderRefusesWhatTheConstantsDeclare(t *testing.T) {
	for _, tc := range []struct {
		name     string
		manifest string
		wantErr  string
	}{
		{
			"version 1 loads",
			`{"format":"loudkit-checkpoint","format_version":1}`,
			"",
		},
		{
			"an explicit single is the same loop",
			`{"format":"loudkit-checkpoint","format_version":1,"decode":{"mode":"single"}}`,
			"",
		},
		{
			"version 2 loads",
			`{"format":"loudkit-checkpoint","format_version":2,"decode":{"mode":"fusion_mtp2"}}`,
			"",
		},
		{
			"a fusion mode under version 1 is refused",
			`{"format":"loudkit-checkpoint","format_version":1,"decode":{"mode":"fusion_mtp2"}}`,
			"decode.mode",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Open(manifestOnly(t, tc.manifest))
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("expected this to load, got %v", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("expected a refusal naming %q, got none", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("refusal does not name %q: %v", tc.wantErr, err)
			}
		})
	}
}
