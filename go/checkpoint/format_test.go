package checkpoint

import "testing"

// The manifest contract is a contract: Python refuses a format_version it does
// not read, and a port that accepts any version will happily load a future
// checkpoint whose fields mean something else. The loader would still "work",
// and the audio would be wrong for reasons no error names.
func TestSupportedFormatVersionsMatchPython(t *testing.T) {
	if len(SupportedFormatVersions) != 2 || SupportedFormatVersions[0] != 1 || SupportedFormatVersions[1] != 2 {
		t.Fatalf("SupportedFormatVersions is %v; Python reads [1 2]", SupportedFormatVersions)
	}
}

// And the mode, because the version is only as honest as the packer.
//
// A manifest saying format_version 1 beside decode.mode "fusion_mtp2" clears
// the version gate. Nothing else in this port reads the decode block, so
// before the check in Open it would have run the one-token loop over two-token
// weights and spoken fluent nonsense.
func TestSupportedDecodeModesAreTheOnesImplemented(t *testing.T) {
	if len(SupportedDecodeModes) != 2 || SupportedDecodeModes[0] != "single" || SupportedDecodeModes[1] != "fusion_mtp2" {
		t.Fatalf("SupportedDecodeModes is %v; this engine runs both declared loops",
			SupportedDecodeModes)
	}
}
