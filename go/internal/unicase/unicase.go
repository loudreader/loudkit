// Package unicase holds the one case mapping the text funnel needs and the
// standard library does not have: Unicode's full lowercase.
//
// strings.ToLower applies the simple mapping, one rune in and one rune out.
// Python's str.lower, Rust's str::to_lowercase, JavaScript's toLowerCase and
// Swift's lowercased all apply the full mapping instead, and the funnel is a
// promise that the five ports read the same text the same way, so Go applies
// it too.
package unicase

import (
	"strings"
	"unicode"
)

// The runes the full mapping treats differently from the simple one.
const (
	dottedCapitalI    = 'İ' // LATIN CAPITAL LETTER I WITH DOT ABOVE
	combiningDotAbove = '̇' // COMBINING DOT ABOVE
	capitalSigma      = 'Σ' // GREEK CAPITAL LETTER SIGMA
	smallSigma        = 'σ'
	finalSmallSigma   = 'ς'
)

// ToLower returns s in Unicode full lowercase.
//
// It differs from strings.ToLower in exactly two places, both of which the
// other four ports get for free from their standard libraries:
//
//   - U+0130 lowercases to "i" followed by U+0307 COMBINING DOT ABOVE, not to
//     a bare "i". A dotted capital I is what a Turkish name carries, and
//     dropping the dot hands the respell lexicon and the acronym speller a
//     word nobody wrote: Go alone said "Ystanbul" for Istanbul spelled with
//     one, and spelled the initialism in "the IETF spec" out letter by letter.
//   - U+03A3 lowercases to U+03C2, final sigma, when it ends a word, and to
//     U+03C3 anywhere else. Unicode calls the rule Final_Sigma. Two different
//     lowercase letters are two different tokenizer ids.
//
// No locale-sensitive mapping is applied. Turkish dotless i and the Lithuanian
// accent rules are conditioned on a language tag that none of the five ports
// passes, so the funnel takes the unconditional mapping everywhere.
func ToLower(s string) string {
	if !strings.ContainsRune(s, dottedCapitalI) && !strings.ContainsRune(s, capitalSigma) {
		return strings.ToLower(s)
	}
	runes := []rune(s)
	var b strings.Builder
	b.Grow(len(s) + 8)
	for i, r := range runes {
		switch r {
		case dottedCapitalI:
			b.WriteRune('i')
			b.WriteRune(combiningDotAbove)
		case capitalSigma:
			if finalSigma(runes, i) {
				b.WriteRune(finalSmallSigma)
			} else {
				b.WriteRune(smallSigma)
			}
		default:
			b.WriteRune(unicode.ToLower(r))
		}
	}
	return b.String()
}

// finalSigma reports whether the sigma at runes[i] is word-final, which
// Unicode spells as: preceded by a cased rune and then any number of
// case-ignorable ones, and not followed by any number of case-ignorable runes
// and then a cased one.
func finalSigma(runes []rune, i int) bool {
	before := false
	for j := i - 1; j >= 0; j-- {
		if !caseIgnorable(runes[j]) {
			before = cased(runes[j])
			break
		}
	}
	if !before {
		return false
	}
	for j := i + 1; j < len(runes); j++ {
		if !caseIgnorable(runes[j]) {
			return !cased(runes[j])
		}
	}
	return true
}

// cased is Unicode's derived Cased property: Lowercase, Uppercase or
// titlecase, where the first two carry their Other_ additions.
func cased(r rune) bool {
	return unicode.IsLower(r) || unicode.IsUpper(r) || unicode.IsTitle(r) ||
		unicode.Is(unicode.Other_Lowercase, r) || unicode.Is(unicode.Other_Uppercase, r)
}

// caseIgnorable is Unicode's derived Case_Ignorable property: the marks,
// formats and modifier letters that sit inside a word without being letters of
// it, plus the three word-break classes that join two letters.
func caseIgnorable(r rune) bool {
	if midWord(r) {
		return true
	}
	return unicode.Is(unicode.Mn, r) || unicode.Is(unicode.Me, r) ||
		unicode.Is(unicode.Cf, r) || unicode.Is(unicode.Lm, r) ||
		unicode.Is(unicode.Sk, r)
}

// midWord is Word_Break MidLetter, MidNumLet and Single_Quote: the colons,
// middle dots, apostrophes and full stops Unicode counts as inside a word
// rather than after it. Hand-written because the standard library ships no
// word-break tables; the list is short and closed.
func midWord(r rune) bool {
	switch r {
	case '\'', // APOSTROPHE, the whole of Single_Quote
		'.', // FULL STOP, and the rest of MidNumLet
		'‘', // LEFT SINGLE QUOTATION MARK
		'’', // RIGHT SINGLE QUOTATION MARK
		'․', // ONE DOT LEADER
		'﹒', // SMALL FULL STOP
		'＇', // FULLWIDTH APOSTROPHE
		'．', // FULLWIDTH FULL STOP
		':', // COLON, and the rest of MidLetter
		'·', // MIDDLE DOT
		'·', // GREEK ANO TELEIA
		'՟', // ARMENIAN ABBREVIATION MARK
		'״', // HEBREW PUNCTUATION GERSHAYIM
		'‧', // HYPHENATION POINT
		'︓', // PRESENTATION FORM FOR VERTICAL COLON
		'﹕', // SMALL COLON
		'：': // FULLWIDTH COLON
		return true
	}
	return false
}
