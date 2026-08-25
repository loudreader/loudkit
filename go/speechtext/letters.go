// Acronyms, spelled in the language being read.
//
// CIA is see-eye-ay in an English render and ce-i-a in a Polish one, and those
// are not two spellings of one thing — they are what the two languages actually
// say. The engine is grapheme-based with a single language tag per utterance, so
// the letter name has to be written in the target language's own orthography:
// English "see" reads as /siː/ under English letter-to-sound rules, Polish "ce"
// reads as /t͡sɛ/ under Polish ones, and putting either into the other's render
// produces a word nobody says.
//
// Without this module, acronyms are spelled only in Polish, inside respell.go,
// with a Polish letter table: FBI becomes ef-be-i in a Polish render and
// reaches the model as the raw graphemes FBI in the other eleven, where a
// grapheme engine reads them as a word-shaped thing rather than as letters. The
// tables are per
// language in the shared grammar file; this reads them for all twelve, out of
// the same numbers.json every other implementation reads.
//
// What is not spelled: an acronym that is a word in its language stays a word —
// NASA and NATO everywhere, SIDA and OVNI in the Romance three, PESEL and ZUS in
// Polish, TUTKA in Finnish. Those lists are per language because the fact is:
// LOT is an airline in Poland and a common noun in English, and only one of them
// should be spelled out.//
// Python reference: `loudkit/frontend/letters.py`.
package speechtext

import (
	"encoding/json"
	"strings"
	"sync"
	"unicode"
)

const (
	minAcronymLetters = 2
	// Above five letters an all-caps run is far more often a shout, a product
	// name or a heading than an initialism, and spelling one out is a worse
	// error than leaving it — the listener can read SIGGRAPH; they cannot
	// un-hear ess-eye-gee-gee-ar-ay-pee-aitch.
	maxAcronymLetters = 5
)

type letterTable struct {
	names map[string]string
	words map[string]bool
}

var (
	letterTablesOnce sync.Once
	letterTables     map[string]*letterTable
)

func loadLetterTables() {
	var doc struct {
		Languages map[string]struct {
			LetterNames  map[string]string `json:"letter_names"`
			WordAcronyms []string          `json:"word_acronyms"`
		} `json:"languages"`
	}
	if err := json.Unmarshal(numbersJSON, &doc); err != nil {
		panic("speechtext: embedded numbers.json is unreadable: " + err.Error())
	}
	letterTables = make(map[string]*letterTable, len(doc.Languages))
	for lang, e := range doc.Languages {
		if len(e.LetterNames) == 0 {
			continue
		}
		words := make(map[string]bool, len(e.WordAcronyms))
		for _, w := range e.WordAcronyms {
			words[w] = true
		}
		letterTables[lang] = &letterTable{names: e.LetterNames, words: words}
	}
}

func tables() map[string]*letterTable {
	letterTablesOnce.Do(loadLetterTables)
	return letterTables
}

// SpellsAcronyms reports whether this language has a letter table at all.
func SpellsAcronyms(language string) bool {
	_, ok := tables()[language]
	return ok
}

// LetterName is what language calls one letter, or "" if it has no name for it.
//
// Empty rather than a guess: a letter with no entry means the acronym is left
// alone entirely, because half-spelling one (ef-be-q) is worse than not spelling
// it at all.
func LetterName(letter, language string) string {
	t, ok := tables()[language]
	if !ok {
		return ""
	}
	return t.names[strings.ToLower(letter)]
}

// SpellAcronym returns word as spelled-out letters, or "" to leave it alone.
//
// Empty — "not an acronym, or not one I can spell" — for a word that is not
// all-caps, is too short or too long, is a word in this language, or contains a
// letter this language has no name for.
func SpellAcronym(word, language string) string {
	if len([]rune(word)) < minAcronymLetters || !isAllCapsWord(word) {
		return ""
	}
	t, ok := tables()[language]
	if !ok {
		return ""
	}
	lowered := strings.ToLower(word)
	if t.words[lowered] {
		// A word, not an initialism: read as itself, lowercased so no later
		// pass mistakes it for an acronym again.
		//
		// Checked before the length cap, and the order matters: the cap is
		// about how long a thing may be before spelling
		// it becomes worse than leaving it, and it has nothing to say about a
		// word. With the cap first, every entry over five letters is dead —
		// UNESCO, UNICEF and INTERPOL never reach this branch.
		return lowered
	}
	if len([]rune(word)) > maxAcronymLetters {
		return ""
	}
	names := make([]string, 0, len(lowered))
	for _, ch := range lowered {
		name, ok := t.names[string(ch)]
		if !ok {
			return ""
		}
		names = append(names, name)
	}
	// Hyphens rather than spaces: they keep the letters one prosodic unit, so
	// the model reads a run of names instead of a list of tiny words.
	return strings.Join(names, "-")
}

// spellAcronyms spells every lone acronym in text the way language spells it.
//
// Shouting is left alone, and the rule for telling it from an initialism is
// context rather than anything inside the word. An initialism appears as a
// single capitalised island in ordinary text — "the CIA said" — while emphasis
// comes in runs. That distinction is not available from the word itself: IT is a
// word, an initialism and a shout depending only on what sits beside it, and no
// table can separate those. So a capitalised word spells out only when neither
// neighbour is also capitalised, and a text that is entirely capitals is passed
// through whole, because someone pasted a headline and spelling all of it would
// be the loudest possible wrong answer.
func spellAcronyms(text, language string) string {
	if !SpellsAcronyms(language) || !strings.ContainsFunc(text, unicode.IsUpper) {
		return text
	}

	tokens := splitOnNonWord(text)
	wordCount, allCaps := 0, true
	for _, t := range tokens {
		if isWordToken(t) {
			wordCount++
			if !isAllCapsWord(t) {
				allCaps = false
			}
		}
	}
	if wordCount > 1 && allCaps {
		// The whole text is capitals: someone pasted a shout, or a headline.
		//
		// More than one word, though. A text that is a single capitalised token
		// — Prepared("GPT") — is an acronym on its own, not a shout: there is no
		// run to read emphasis from, and refusing it would mean the one call
		// shaped exactly like "say this acronym" was the one that did not.
		return text
	}

	isCaps := func(i int) bool {
		if i < 0 || i >= len(tokens) {
			return false
		}
		return isWordToken(tokens[i]) && isAllCapsWord(tokens[i])
	}

	out := make([]string, len(tokens))
	copy(out, tokens)
	for i := range tokens {
		if !isCaps(i) {
			continue
		}
		// Neighbours, skipping the separator token between words.
		if isCaps(i-2) || isCaps(i+2) {
			continue // part of a run: emphasis, not an initialism
		}
		if said := SpellAcronym(tokens[i], language); said != "" {
			out[i] = said
		}
	}
	return strings.Join(out, "")
}

// isAllCapsWord mirrors Python's `token.isalpha() and token.isupper()`.
//
// Python's isupper() is true when there is at least one cased character and no
// lowercase one, so a token is judged as a unit rather than rune by rune.
func isAllCapsWord(token string) bool {
	if token == "" {
		return false
	}
	sawCased := false
	for _, ch := range token {
		if !unicode.IsLetter(ch) || unicode.IsLower(ch) {
			return false
		}
		if unicode.IsUpper(ch) {
			sawCased = true
		}
	}
	return sawCased
}

func isWordToken(token string) bool {
	if len([]rune(token)) <= 1 {
		return false
	}
	for _, ch := range token {
		if !unicode.IsLetter(ch) {
			return false
		}
	}
	return true
}

// splitOnNonWord is Python's re.split(r"(\W+)", text): separators are kept, so
// the pieces rejoin exactly. Word characters are letters, digits and underscore,
// which is what Python's \w means under its default Unicode rules.
func splitOnNonWord(text string) []string {
	isWord := func(ch rune) bool {
		return unicode.IsLetter(ch) || unicode.IsDigit(ch) || ch == '_'
	}
	var out []string
	var current strings.Builder
	first := true
	var currentIsWord bool
	for _, ch := range text {
		w := isWord(ch)
		if first {
			currentIsWord, first = w, false
			// Python's split starts on a word field, even an empty one.
			if !w {
				out = append(out, "")
			}
			current.WriteRune(ch)
			continue
		}
		if w == currentIsWord {
			current.WriteRune(ch)
			continue
		}
		out = append(out, current.String())
		current.Reset()
		current.WriteRune(ch)
		currentIsWord = w
	}
	if current.Len() > 0 {
		out = append(out, current.String())
	}
	return out
}
