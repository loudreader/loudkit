package engine

import (
	"testing"

	"github.com/loudreader/loudkit/go/voice"
)

// The obvious call must not be the wrong one.
//
// Without the voice link, Synthesize("Cześć", polishVoice, seed, "", nil) runs
// Polish text through the English frontend: an empty language becomes "en"
// outright and a profile's own Language — recorded at enrollment — is never
// consulted. The chain is argument, then voice, then "en", and these are its
// links.
//
// Tested against the resolver rather than through Synthesize because this port
// has no weight-free engine seam: engine.Engine holds six concrete
// *onnx.Session values, so nothing can drive the pipeline without a checkpoint
// and a runtime. The resolver is the whole of the behaviour under test;
// mel_test.go tests an unexported helper in-package for the same reason.
func TestResolveLanguage(t *testing.T) {
	polish := &voice.Profile{Language: "pl"}

	if got := resolveLanguage("", polish); got != "pl" {
		t.Errorf("a Polish voice should read Polish by default, got %q", got)
	}
	if got := resolveLanguage("en", polish); got != "en" {
		t.Errorf("an explicit language should override the profile, got %q", got)
	}
	// A hand-built profile can carry an empty Language, and an empty language id
	// is not a language — it would tag the text "[]". A header that simply omits
	// the key loads as "en" instead, so it never reaches this branch.
	if got := resolveLanguage("", &voice.Profile{Language: ""}); got != "en" {
		t.Errorf("a profile without a language should fall back to English, got %q", got)
	}
	if got := resolveLanguage("", nil); got != "en" {
		t.Errorf("no voice at all should fall back to English, got %q", got)
	}
}
