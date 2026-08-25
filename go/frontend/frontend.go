// Package frontend is the text->text-token pipeline, a bit-parity port of
// loudkit.frontend.text: lowercase, NFKD, a language tag, spaces to [SPACE],
// then plain BPE over Unicode scalars.
package frontend

import (
	"strings"
	"sync"

	"golang.org/x/text/unicode/norm"

	"github.com/loudreader/loudkit/go/speechtext"
	"github.com/loudreader/loudkit/go/tokenizer"
)

const space = "[SPACE]"

// needsModelPreprocessing names the refused languages whose refusal has a
// specific reason worth stating: their upstream pipeline wants Cangjie codes,
// kana conversion, diacritisation, jamo decomposition or stress marks, none of
// which this frontend carries. A subset of "not on the roster", kept so the
// message can say why rather than just no.
var needsModelPreprocessing = map[string]bool{"zh": true, "ja": true, "he": true, "ko": true, "ru": true}

var (
	rosterOnce sync.Once
	roster     map[string]bool
	rosterList []string
)

// supported is the allowlist: the twelve ids in numbers.json, the same roster
// Python's loudkit.frontend.numbers.supported_languages reports.
//
// This was a blacklist of the five above, and the difference matters because
// the tokenizer's vocabulary carries tags for 31 languages. A blacklist let the
// other 26 through and the tag was emitted, so Encode(text, "bg") NFKD-mangled
// Cyrillic into ids the model reads as sounds it was never trained to make — no
// error, plausible audio, wrong language.
func supported() (map[string]bool, []string) {
	rosterOnce.Do(func() {
		rosterList = speechtext.SupportedNumberLanguages()
		roster = make(map[string]bool, len(rosterList))
		for _, lang := range rosterList {
			roster[lang] = true
		}
	})
	return roster, rosterList
}

// Frontend normalises and tokenises text.
type Frontend struct {
	tokenizer *tokenizer.Tokenizer
}

// Load builds a frontend from a tokenizer.json path.
func Load(tokenizerPath string) (*Frontend, error) {
	parsed, err := tokenizer.ParseJSON(tokenizerPath)
	if err != nil {
		return nil, err
	}
	return &Frontend{tokenizer: tokenizer.New(parsed)}, nil
}

// MaxTokenID is the largest id Encode can return. See tokenizer.Tokenizer.MaxID.
func (f *Frontend) MaxTokenID() int {
	return f.tokenizer.MaxID()
}

// Encode normalises and tokenises. Same text and language give the same ids.
func (f *Frontend) Encode(text, language string) ([]int, error) {
	lang := strings.ToLower(language)
	if allowed, list := supported(); !allowed[lang] {
		return nil, &ErrUnsupported{Language: lang, Supported: list}
	}
	normalised := norm.NFKD.String(strings.ToLower(text))
	// Square brackets never reach the tokenizer from user text: the vocabulary
	// holds 117 bracket control tokens ([sigh], [gasp], the language tags) and
	// matches them greedily, so "he [sigh]ed" would make the model sigh. The
	// language tag added below is the one bracket that belongs.
	normalised = strings.NewReplacer("[", " ", "]", " ").Replace(normalised)
	tagged := "[" + lang + "]" + strings.ReplaceAll(normalised, " ", space)
	return f.tokenizer.Encode(tagged), nil
}

// ErrUnsupported names a language this frontend deliberately cannot read, and
// the ones it can — a refusal that cannot say what would have worked leaves the
// caller guessing.
type ErrUnsupported struct {
	Language  string
	Supported []string
}

func (e *ErrUnsupported) Error() string {
	why := "is not one of the languages this build's text layer is written for"
	if needsModelPreprocessing[e.Language] {
		why = "needs model-based text preprocessing " +
			"(Cangjie/kana/diacritics/jamo/stress) that this frontend does not carry"
	}
	return "language '" + e.Language + "' " + why +
		". Supported: " + strings.Join(e.Supported, ", ")
}
