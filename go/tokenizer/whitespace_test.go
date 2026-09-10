package tokenizer

import (
	"strings"
	"testing"
)

// TestThePreTokenizerDropsUnicodeWhitespace pins the class the `\s` in
// `\w+|[^\w\s]+` stands for.
//
// It was the six ASCII blanks here and Unicode's White_Space everywhere else.
// Most of the difference never reached the tokenizer, because the frontend
// normalises NFKD first and every space-like character with a compatibility
// decomposition becomes a plain space on the way. Four have none, and those
// four became pretokens with no vocabulary entry, so a word with one inside it
// reached the model as two words with an [UNK] between them, in this port
// alone.
func TestThePreTokenizerDropsUnicodeWhitespace(t *testing.T) {
	// The four that survive NFKD, then the ASCII blanks that always worked.
	for _, sep := range []string{
		"\u0085", "\u1680", "\u2028", "\u2029", // NEL, ogham, line, paragraph
		" ", "\t", "\n", "\r", "\f", "\v",
	} {
		got := whitespaceRegex("Alpha" + sep + "beta")
		want := []string{"Alpha", "beta"}
		if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
			t.Errorf("whitespaceRegex(%q) = %q, want %q", "Alpha"+sep+"beta", got, want)
		}
	}
}

// TestThePreTokenizerKeepsEverythingElse is the other half: the class may not
// grow past White_Space, or punctuation the model reads would disappear.
func TestThePreTokenizerKeepsEverythingElse(t *testing.T) {
	for _, c := range []struct {
		in   string
		want []string
	}{
		{"Alpha, beta!", []string{"Alpha", ",", "beta", "!"}},
		// A zero width non-joiner is not a space.
		{"a\u200cb", []string{"a", "\u200c", "b"}},
		{"[en]Hello", []string{"[", "en", "]", "Hello"}},
		{"", nil},
	} {
		got := whitespaceRegex(c.in)
		if strings.Join(got, "|") != strings.Join(c.want, "|") {
			t.Errorf("whitespaceRegex(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}
