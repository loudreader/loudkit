package speechtext

import "testing"

// TestTwoAmountsSideBySideStayTwoAmounts pins the currency rule's boundary.
//
// RE2 has no lookbehind, and the pattern used to capture the character in
// front of the symbol instead of looking at it. That character is then eaten,
// so the second of two adjacent amounts had to start one place early and took
// the previous amount's digits with it: `$1$2$3` came out as "one dollars
// twenty-three dollars", which is a number nobody wrote. docs/design/
// text-funnel.md is explicit that this is the outcome to avoid.
//
// The expectations are Python's, which are also JavaScript's and Swift's. Raw
// digits reaching the tokenizer is deliberate: a digit run glued to a word
// stays written.
func TestTwoAmountsSideBySideStayTwoAmounts(t *testing.T) {
	for _, c := range []struct{ text, lang, want string }{
		{"$1$2$3", "en", "one dollars2 dollars3 dollars"},
		{"$5$10", "en", "five dollars10 dollars"},
		{"$5$10", "pt", "cinco dólares10 dólares"},
		{"€1€2", "de", "eins Euro2 Euro"},
		// A letter in front still means a currency mark this table cannot
		// name: the symbol is dropped rather than spoken as the wrong money,
		// and the amount reads as a plain number.
		{"R$3,14", "en", "R three point one four"},
	} {
		if got := Prepared(c.text, c.lang); got != c.want {
			t.Errorf("Prepared(%q, %q) = %q, want %q", c.text, c.lang, got, c.want)
		}
	}
}

// TestTwoAbbreviationsSideBySideBothExpand pins the same boundary in the
// abbreviation pass, which had the same pattern and the same defect: the
// trailing boundary character was consumed, so the next abbreviation had
// nothing left to sit behind.
func TestTwoAbbreviationsSideBySideBothExpand(t *testing.T) {
	for _, c := range []struct{ text, lang, want string }{
		{"np. itd. itp.", "pl", "na przykład i tak dalej i tym podobne"},
		{"e.g. e.g.", "en", "for example for example"},
		{"e.g.,e.g.", "en", "for example,for example"},
		{"z.B. usw. bzw.", "de", "zum Beispiel und so weiter beziehungsweise"},
		// A word character in front means the abbreviation is part of a word.
		// Python's `\w` is Unicode-aware and RE2's is ASCII, so this row read
		// one way here and another way in the four other ports.
		{"źnp.", "pl", "źnp."},
		{"anp.", "pl", "anp."},
		{"np.x", "pl", "np.x"},
	} {
		if got := ExpandAbbreviations(c.text, c.lang); got != c.want {
			t.Errorf("ExpandAbbreviations(%q, %q) = %q, want %q", c.text, c.lang, got, c.want)
		}
	}
}
