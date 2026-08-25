package frontend

import (
	"strings"
	"testing"

	"github.com/loudreader/loudkit/go/speechtext"
)

// A tag the tokenizer knows is not a language the kit can speak.
//
// The vocabulary carries tags for 31 languages; the text layer is written for
// twelve. A blacklist of only zh/ja/he/ko/ru lets the other 26 go
// straight through: Encode(text, "bg") NFKD-mangles Cyrillic into ids the
// model reads as sounds it never learned — no error, plausible-sounding audio,
// wrong language.
//
// The roster itself is asserted rather than the refusal alone: numbers.json is
// the one authority, and a port that hardcodes a second copy is a port that
// will disagree with Python the next time a grammar is added.
func TestTheRosterIsAnAllowlist(t *testing.T) {
	allowed, list := supported()

	if len(list) != 12 {
		t.Fatalf("roster should be the twelve in numbers.json, got %d: %v", len(list), list)
	}
	want := speechtext.SupportedNumberLanguages()
	if strings.Join(list, ",") != strings.Join(want, ",") {
		t.Errorf("roster %v is not the number roster %v", list, want)
	}
	for _, lang := range []string{"en", "pl", "sv"} {
		if !allowed[lang] {
			t.Errorf("%q is on the roster and must be accepted", lang)
		}
	}
	// Cyrillic: a tokenizer tag, never a language this build speaks.
	if allowed["bg"] {
		t.Error("bg is not on the roster and must be refused")
	}
	if allowed["zh"] {
		t.Error("zh must stay refused")
	}
}

// The refusal has to carry both halves: what was refused, and what would work.
func TestARefusalNamesTheAlternatives(t *testing.T) {
	_, list := supported()

	off := (&ErrUnsupported{Language: "bg", Supported: list}).Error()
	if !strings.Contains(off, "text layer is written for") {
		t.Errorf("an off-roster language should say so plainly: %s", off)
	}
	if !strings.Contains(off, "en, es") {
		t.Errorf("a refusal must list what works: %s", off)
	}

	// The five model-based ones keep their specific reason: *why* they are
	// refused is real information, and "not on the roster" throws it away.
	modelled := (&ErrUnsupported{Language: "zh", Supported: list}).Error()
	if !strings.Contains(modelled, "model-based") {
		t.Errorf("zh should still explain the model-based pipeline: %s", modelled)
	}
}
