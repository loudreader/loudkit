package unicase

import (
	"strings"
	"testing"
)

// TestTheDottedCapitalIKeepsItsDot pins the one unconditional place where the
// full mapping and Go's simple one part company. The answers are Python's
// str.lower, which is also Rust's, JavaScript's and Swift's.
func TestTheDottedCapitalIKeepsItsDot(t *testing.T) {
	for _, c := range []struct{ in, want string }{
		{"İ", "i̇"},
		{"İstanbul", "i̇stanbul"},
		{"İETF", "i̇etf"},
		{"İZMİR", "i̇zmi̇r"},
		{"Iİ", "ii̇"},
		{"İ012", "i̇012"},
	} {
		if got := ToLower(c.in); got != c.want {
			t.Errorf("ToLower(%q) = %q, want %q", c.in, got, c.want)
		}
		if simple := strings.ToLower(c.in); simple == c.want {
			t.Errorf("ToLower(%q): the simple mapping already agrees, the case has stopped testing anything", c.in)
		}
	}
}

// TestSigmaIsFinalOnlyAtTheEndOfAWord pins Unicode's Final_Sigma condition:
// the same capital sigma is two different lowercase letters, and so two
// different tokenizer ids, depending on what follows it.
func TestSigmaIsFinalOnlyAtTheEndOfAWord(t *testing.T) {
	for _, c := range []struct{ in, want string }{
		{"aΣ", "aς"},
		{"aΣb", "aσb"},
		{"Σa", "σa"},
		{"Σ", "σ"},
		{"ΟΔΟΣ", "οδος"},
		{"ΟΔΟΣ.", "οδος."},
		{"ΟΔΟΣ α", "οδος α"},
		{"aΣ'", "aς'"}, // an apostrophe is case-ignorable, so still final
		{"aΣ́", "aς́"}, // and so is a combining accent
		{"aΣ́b", "aσ́b"},
	} {
		if got := ToLower(c.in); got != c.want {
			t.Errorf("ToLower(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestEverythingElseIsTheSimpleMapping is the other half of the promise: the
// full mapping is the simple one everywhere the two agree, so no ordinary text
// moves.
func TestEverythingElseIsTheSimpleMapping(t *testing.T) {
	for _, s := range []string{
		"", "a", "ABC", "Zażółć gęślą jaźń", "ÄÖÜ", "ǅ", "ǄǅǆǇ",
		"Wizyta w Warszawie.", "123 $5", "ß", "İ" + "İ",
	} {
		if strings.ContainsRune(s, 'İ') || strings.ContainsRune(s, 'Σ') {
			continue
		}
		if got, want := ToLower(s), strings.ToLower(s); got != want {
			t.Errorf("ToLower(%q) = %q, want the simple mapping's %q", s, got, want)
		}
	}
}
