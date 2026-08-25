package speechtext

import "testing"

// Python guards each written date with `(?<![\d.,:/-])` and, on two of the
// three, `(?![\d-])` or `(?![\d/])`. This port checks the characters either
// side instead, and the class was handed to it as the pattern text: `\d` was
// the two literal runes `\` and `d`, so no digit ever matched and every guard
// was blind to exactly the character it was written for. `42.3.2026` reached
// the German dotted-date reading as a date with a stray digit welded in front
// of it — and a wrong day, since the match began at the `2`.
func TestADigitBesideADateRefusesIt(t *testing.T) {
	cases := []struct{ text, lang, want string }{
		{"42.3.2026", "de", "42.3.2026"},
		{"112/3/2026", "de", "112/3/2026"},
		{"12026-01-02", "de", "12026-01-02"},
		{"2026-01-020", "de", "2026-01-020"},
		{"12/3/20267", "de", "12/3/20267"},
		// The separators in each class still refuse, as they always did.
		{".42.3.2026", "de", ".42.3.2026"},
		{"2026-01-02-0", "de", "2026-01-02-0"},
		// A date nothing touches is still read.
		{"12.3.2026", "de", "zwölfte März zweitausendsechsundzwanzig"},
	}
	for _, c := range cases {
		if got := ExpandDates(c.text, c.lang); got != c.want {
			t.Errorf("ExpandDates(%q, %s) = %q, want %q", c.text, c.lang, got, c.want)
		}
	}
}
