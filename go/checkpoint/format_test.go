package checkpoint

import "testing"

// The manifest contract is a contract: Python refuses a format_version it does
// not read, and a port that accepts any version will happily load a future
// checkpoint whose fields mean something else. The loader would still "work",
// and the audio would be wrong for reasons no error names.
func TestSupportedFormatVersionsMatchPython(t *testing.T) {
	if len(SupportedFormatVersions) != 1 || SupportedFormatVersions[0] != 1 {
		t.Fatalf("SupportedFormatVersions is %v; Python reads [1]", SupportedFormatVersions)
	}
}
