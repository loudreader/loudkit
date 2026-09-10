package speechtext

import "testing"

// TestFullLowercaseReachesTheFunnel pins the funnel rows Go alone used to get
// wrong, because strings.ToLower dropped the dot from U+0130 and handed the
// respeller and the acronym speller a word that was never written. The
// expectations are Python's, which are also Rust's, JavaScript's and Swift's.
func TestFullLowercaseReachesTheFunnel(t *testing.T) {
	for _, c := range []struct{ text, lang, want string }{
		// The respell lexicon: a dotless i turned this into "Ystanbul".
		{"Wizyta w İstanbul.", "pl", "Wizyta w İstanbul."},
		// The acronym speller: every rune had an ASCII letter name, so Go
		// spelled a word the other four leave alone.
		{"The İETF spec", "en", "The İETF spec"},
		{"İZMİR", "en", "İZMİR"},
		{"İCIA", "en", "İCIA"},
		{"Iİ", "en", "Iİ"},
		// The code-token speller, which refuses a token holding a letter it
		// has no name for.
		{"İ012", "pl", "İ012"},
	} {
		if got := Prepared(c.text, c.lang); got != c.want {
			t.Errorf("Prepared(%q, %q) = %q, want %q", c.text, c.lang, got, c.want)
		}
	}
}
