package speechtext

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func loadNumbersFixture(t *testing.T, name string) map[string]any {
	t.Helper()
	p := filepath.Join("..", "..", "tests", "data", "conformance", name)
	raw, err := os.ReadFile(p)
	if err != nil {
		t.Skipf("fixture not found: %s", p)
	}
	var fx map[string]any
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("parse %s: %v", name, err)
	}
	return fx
}

// The hand-written fixture: expectations from each language's own reference
// description, not captured from any implementation's output.
func TestCardinalMatchesTheHandFixture(t *testing.T) {
	fx := loadNumbersFixture(t, "numbers.json")
	cardinals := fx["cardinals"].(map[string]any)
	if len(cardinals) == 0 {
		t.Fatal("the fixture has no cardinals; nothing was compared")
	}
	for lang, raw := range cardinals {
		for _, c := range raw.([]any) {
			kase := c.(map[string]any)
			value := int64(kase["value"].(float64))
			got, err := Cardinal(value, lang, "")
			if err != nil {
				t.Errorf("%s %d: %v", lang, value, err)
				continue
			}
			if got != kase["expect"].(string) {
				t.Errorf("%s %d: got %q, want %q", lang, value, got, kase["expect"])
			}
		}
	}
	for _, c := range fx["gendered"].([]any) {
		kase := c.(map[string]any)
		value := int64(kase["value"].(float64))
		lang := kase["language"].(string)
		got, err := Cardinal(value, lang, kase["gender"].(string))
		if err != nil {
			t.Errorf("%s %d: %v", lang, value, err)
			continue
		}
		if got != kase["expect"].(string) {
			t.Errorf("%s %d g=%s: got %q, want %q",
				lang, value, kase["gender"], got, kase["expect"])
		}
	}
}

// The CLDR differential: 1300 spellouts Unicode wrote. Disputed rows carry
// their reasons and are skipped; past-scale rows must refuse loudly.
func TestCardinalMatchesCLDR(t *testing.T) {
	fx := loadNumbersFixture(t, "numbers_cldr.json")
	checked := 0
	for lang, raw := range fx["cases"].(map[string]any) {
		for _, c := range raw.([]any) {
			kase := c.(map[string]any)
			if _, disputed := kase["disputed"]; disputed {
				continue
			}
			value := int64(kase["value"].(float64))
			gender := ""
			if g, ok := kase["gender"].(string); ok {
				gender = g
			}
			got, err := Cardinal(value, lang, gender)
			if err != nil {
				continue // past our scale: the refusal is the declared behaviour
			}
			checked++
			if got != kase["expect"].(string) {
				t.Errorf("%s %d g=%q: got %q, cldr %q", lang, value, gender, got, kase["expect"])
			}
		}
	}
	if checked < 1000 {
		t.Fatalf("only %d CLDR rows ran; the corpus went missing", checked)
	}
}

func TestExpandNumbers(t *testing.T) {
	cases := []struct{ text, lang, want string }{
		{"I have 21 apples.", "en", "I have twenty-one apples."},
		{"3.5", "en", "three point five"},
		{"1,200", "en", "one thousand two hundred"},
		{"3,5", "pl", "trzy przecinek pięć"},
		{"Es kostet 250 Euro.", "de", "Es kostet zweihundertfünfzig Euro."},
		{"21 apples", "xx", "21 apples"}, // unknown language: leave it alone
		{"no numbers here", "en", "no numbers here"},
	}
	for _, c := range cases {
		if got := ExpandNumbers(c.text, c.lang); got != c.want {
			t.Errorf("ExpandNumbers(%q, %s) = %q, want %q", c.text, c.lang, got, c.want)
		}
	}
}

// The parity fuzzer's digit-run cases. Every expectation here is Python's
// output on the same input, because Python's engine backtracks and this port's
// does not: where they disagree the difference is this port reading digits the
// reference leaves written, or the other way round.
func TestDigitRunParityCases(t *testing.T) {
	cases := []struct{ text, lang, want string }{
		// A ragged run's last segment keeps its fraction. Stopping at the last
		// digit read "…setenta y dos" and left ".5" standing as written text.
		{"4 5672.5", "es", "cuatro cinco mil seiscientos setenta y dos coma cinco"},
		// Ragged is judged where the grouped alternative stops, which is in
		// front of the fraction: a digit behind `.0` does not turn a grouped
		// thousand into three spoken zeros.
		{"1 000.0 3", "nl", "duizend komma nul drie"},
		// The fraction group repeats, and a segment carrying two marks is not
		// one readable number — it is left written, per segment.
		{"4 5671.2.3", "es", "cuatro 5671.2.3"},
		{"4 567 8901.2.3", "es", "cuatro quinientos sesenta y siete 8901.2.3"},
		// A run the lookbehind refuses is not a consumed run: `3 100` binds
		// first, the `e` refuses it, and the `1000` inside is still a number.
		{"e3 1000", "sv", "e3 ettusen"},
		{"e3 1000 x", "sv", "e3 ettusen x"},
		// Nor is a run the glue checks refuse. Here the `+` puts the binding
		// past the lookbehind and the backward walk refuses it at the `e`; the
		// thousand behind a space that never grouped is untouched by that.
		{"1e+3 1000", "sv", "1e+3 ettusen"},
		{"zł.-000 2024", "es", "zł.-000 dos mil veinticuatro"},
		// A ragged run is read as its first group and the rest is re-matched,
		// so a tail that does reach a boundary comes back as the grouped number
		// it is rather than as loose segments.
		{"1 000 1 234 567 1 234 567 x", "fr", "un zéro zéro zéro un deux cent " +
			"trente-quatre cinq cent soixante-sept un million deux cent " +
			"trente-quatre mille cinq cent soixante-sept x"},
		{"1 234 567192.168.0.1", "de", "eins zweihundertvierunddreißig 567192.168.0.1"},
		// The last segment of a ragged run is refused like any other match, so
		// nothing is welded to the identifier behind it.
		{"1 234 567 1e6", "fi", "yksi kaksisataakolmekymmentäneljä " +
			"viisisataakuusikymmentäseitsemän 1e6"},
		// Nothing readable lies to the right in a run glued digit by digit.
		{"iOS18", "en", "iOS18"},
		{"v1.2.3", "en", "v1.2.3"},
		// A thousands group is part of the token it touches, in both
		// directions. The backward walk crosses the space in the first two, the
		// forward walk in the next two.
		{"C0200 000", "it", "C0200 000"},
		{"x200 000", "it", "x200 000"},
		{"2024 200x", "it", "2024 200x"},
		{"200 000x", "it", "200 000x"},
		// The space that ends a word is not a thousands space, whatever
		// follows it.
		{"Sold 200 000", "en", "Sold two hundred thousand"},
		// The first group is legitimately one to three digits wide, so the
		// width tested is the group being crossed into.
		{"a1 000 000", "en", "a1 000 000"},
		// Three digits and no fourth: `5.1e+3` is not a group, so the walk
		// stays inside `1000` instead of finding the exponent's `e`.
		{"1000 5.1e+3", "en", "one thousand 5.1e+3"},
		// Unequal groups are each their own number, and a letter behind the
		// run refuses all of them.
		{"1 202 555 0199", "en",
			"one two hundred and two five hundred and fifty-five zero one nine nine"},
		{"1 234 567.é", "de", "1 234 567.é"},
		{"1 0023R", "da", "1 0023R"},
	}
	for _, c := range cases {
		if got := ExpandNumbers(c.text, c.lang); got != c.want {
			t.Errorf("ExpandNumbers(%q, %s) = %q, want %q", c.text, c.lang, got, c.want)
		}
	}
}

// German writes the time with the word the spoken form also carries: the
// reading puts the infix between hour and minutes, so the written "Uhr"
// behind the digits is that same token and is consumed, not duplicated.
func TestAWrittenInfixIsNotSaidTwice(t *testing.T) {
	cases := []struct{ text, want string }{
		{"um 14:30 Uhr", "um vierzehn Uhr dreißig"},
		// A tab before the word consumes exactly like a space.
		{"um 14:30\tUhr", "um vierzehn Uhr dreißig"},
		{"um 24:00 Uhr an.", "um vierundzwanzig Uhr an."},
		// The dotted form runs through the second pattern.
		{"Termin um 14.30 Uhr.", "Termin um vierzehn Uhr dreißig."},
		// Without the word nothing changes.
		{"um 14:30", "um vierzehn Uhr dreißig"},
		// The noun on its own is not part of any time.
		{"Es ist 14:30 Uhr und die Uhr tickt.", "Es ist vierzehn Uhr dreißig und die Uhr tickt."},
		// Infix inside a longer word keeps its head.
		{"Die Uhrzeit ist 14:30.", "Die Uhrzeit ist vierzehn Uhr dreißig."},
	}
	for _, c := range cases {
		if got := ExpandTimes(c.text, "de"); got != c.want {
			t.Errorf("ExpandTimes(%q, de) = %q, want %q", c.text, got, c.want)
		}
	}
	// Eleven of the twelve grammars carry an empty infix: nothing to consume.
	if got := ExpandTimes("at 14:30 sharp", "en"); got != "at fourteen thirty sharp" {
		t.Errorf("ExpandTimes en = %q", got)
	}
}

// An empty infix consumes nothing at all. Searched for anyway, it matched the
// empty string wherever the whitespace run ended and took the whitespace with
// it — everywhere the character behind was not an ASCII letter or digit, which
// includes every accented letter in nine of these languages.
func TestAnEmptyInfixConsumesNoWhitespace(t *testing.T) {
	cases := []struct{ text, lang, want string }{
		{"3.14 é", "pt", "três catorze é"},
		{"at 14:30 !", "en", "at fourteen thirty !"},
		{"at 14:30 sharp", "en", "at fourteen thirty sharp"},
	}
	for _, c := range cases {
		if got := ExpandTimes(c.text, c.lang); got != c.want {
			t.Errorf("ExpandTimes(%q, %s) = %q, want %q", c.text, c.lang, got, c.want)
		}
	}
}
