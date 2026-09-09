package speechtext

import (
	"testing"
	"unicode/utf8"
)

// The vowel fold drops one character off the respelling. The respellings are
// Polish, so most of them carry a letter that is two bytes wide, and cutting
// the bytes at a character index used to cut one of those letters in half:
// 101568 of the 3317272 word-and-ending pairs came out different from the
// reference, and 13685 of those were not valid UTF-8 at all.
func TestTheVowelFoldCountsCharacters(t *testing.T) {
	loadPayload()
	for _, c := range []struct{ word, want string }{
		{"angia", "endża"},
		{"moldaviae", "mołldejwie"},
		{"moldaviaom", "mołldejwiom"},
		{"moldaviau", "mołldejwiu"},
	} {
		got := respelled(c.word)
		if got != c.want {
			t.Errorf("%q respelled to %q, reference says %q", c.word, got, c.want)
		}
		if !utf8.ValidString(got) {
			t.Errorf("%q respelled to bytes that are not UTF-8: %q", c.word, got)
		}
	}
}
