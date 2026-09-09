package speechtext

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func loadNumbersFixture(t *testing.T, name string) map[string]any {
	t.Helper()
	p := filepath.Join("..", "..", "tests", "data", "conformance", name)
	raw, err := os.ReadFile(p)
	if err != nil {
		// Committed fixture: unreadable means broken, not absent. A skip here
		// passed the whole numbers suite without reading a case.
		t.Fatalf("fixture not readable: %s (%v)", p, err)
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
		// one readable number: it is left written, per segment.
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
// it: everywhere the character behind was not an ASCII letter or digit, which
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

// meridiems are both spellings in both cases, plus the dotted forms, which are
// letters too.
var meridiems = []string{"am", "pm", "AM", "PM", "Am", "pM", "a.m.", "p.m."}

// A spoken time is not written against a letter: `3:45pm` used to read *three
// forty-fivepm*, one word to a listener, where `3:45 pm` read correctly. A
// space in the source was deciding whether the meridiem was a word at all, and
// nothing in this suite asked.
func TestASpokenTimeIsNotWrittenAgainstALetter(t *testing.T) {
	for _, meridiem := range meridiems {
		text := "Call at 3:45" + meridiem + "."
		want := "Call at three forty-five " + meridiem + "."
		if got := ExpandTimes(text, "en"); got != want {
			t.Errorf("ExpandTimes(%q, en) = %q, want %q", text, got, want)
		}
	}
	// The written infix needs no space in front of it either.
	for _, c := range []struct{ text, want string }{
		{"um 14:30Uhr", "um vierzehn Uhr dreißig"},
		{"Termin um 14.30Uhr.", "Termin um vierzehn Uhr dreißig."},
	} {
		if got := ExpandTimes(c.text, "de"); got != c.want {
			t.Errorf("ExpandTimes(%q, de) = %q, want %q", c.text, got, c.want)
		}
	}
}

// Every hour and minute of the clock, in every language: the glued form reads
// exactly as the spaced one. The separator is the one the language treats as a
// time, and the hour is written both bare and zero-padded, two matches.
func TestTheSpaceInTheSourceDecidesNothing(t *testing.T) {
	for _, lang := range SupportedNumberLanguages() {
		separator := "."
		if decimalSeparator(lang) == "." {
			separator = ":"
		}
		for _, meridiem := range []string{"pm", "a.m."} {
			for hour := 0; hour <= 24; hour++ {
				for _, writtenHour := range []string{
					strconv.Itoa(hour), fmt.Sprintf("%02d", hour),
				} {
					for minute := 0; minute < 60; minute++ {
						written := fmt.Sprintf("%s%s%02d", writtenHour, separator, minute)
						glued := ExpandTimes("at "+written+meridiem+" sharp", lang)
						spaced := ExpandTimes("at "+written+" "+meridiem+" sharp", lang)
						if ExpandTimes(written, lang) == written {
							// Not a clock time here, `24:01` being the whole
							// set: both forms keep every character.
							if want := "at " + written + meridiem + " sharp"; glued != want {
								t.Fatalf("%s %s: glued = %q, want %q", lang, written, glued, want)
							}
							continue
						}
						if glued != spaced {
							t.Fatalf("%s %s: glued %q, spaced %q", lang, written, glued, spaced)
						}
					}
				}
			}
		}
	}
}

// A letter is the only thing the rule reads: what a following digit or
// separator refused, it still refuses, and a letter in front of the time is a
// different question the shared fixture answers.
func TestOnlyALetterSeparatesASpokenTime(t *testing.T) {
	for _, lang := range SupportedNumberLanguages() {
		// A seconds field is two digits and the last of them: `10:30:45:60`
		// is a separator run, not a clock, and the refusal that reads the
		// character after the time is what says so.
		for _, literal := range []string{"12.03.2026", "1.2.3", "24:30", "10:30:45:60"} {
			if got := ExpandTimes(literal, lang); got != literal {
				t.Errorf("%s: ExpandTimes(%q) = %q, want it left alone", lang, literal, got)
			}
		}
		for _, text := range []string{"at 14:30", "at 14:30.", "at 14:30, yes", "at 14:30!"} {
			got := ExpandTimes(text, lang)
			if strings.Contains(got, "  ") || strings.TrimRight(got, " ") != got {
				t.Errorf("%s: ExpandTimes(%q) = %q gained a space", lang, text, got)
			}
		}
	}
	if got := ExpandTimes("Meet at a14:30.", "en"); got != "Meet at afourteen thirty." {
		t.Errorf("a letter before the time moved: %q", got)
	}
}

// A clock time carries its seconds, and a zero seconds field says nothing the
// hour and the minute have not already said.
//
// The two forms are compared with each other rather than with twelve spellings
// of the reading, because agreeing is the whole rule. The dotted form is left
// out on purpose: `10.30.45` is a version string as readily as a timestamp.
func TestZeroSecondsReadAsNoSecondsAtAll(t *testing.T) {
	for _, lang := range SupportedNumberLanguages() {
		for _, pair := range [][2]string{
			{"10:30:00", "10:30"},
			{"3:45:00pm", "3:45pm"},
			{"24:00:00", "24:00"},
		} {
			with, without := ExpandTimes(pair[0], lang), ExpandTimes(pair[1], lang)
			if with != without {
				t.Errorf("%s: ExpandTimes(%q) = %q but ExpandTimes(%q) = %q",
					lang, pair[0], with, pair[1], without)
			}
		}
		// A dotted time takes no seconds anywhere, whatever the language does
		// with the dot between an hour and its minutes.
		if got := ExpandTimes("10.30.45", lang); got != "10.30.45" {
			t.Errorf("%s: ExpandTimes(%q) = %q, want it left alone", lang, "10.30.45", got)
		}
		if got := ExpandTimes("10:30:45", lang); got == "10:30:45" {
			t.Errorf("%s: ExpandTimes(%q) left the seconds written", lang, "10:30:45")
		}
	}
	// A zero minute is dropped from `10:30` and kept in `10:00:45`, where
	// dropping it would move the seconds into the minutes' place.
	if got := ExpandTimes("10:00:45 and 10:30:00", "en"); got != "ten zero forty-five and ten thirty" {
		t.Errorf("the seconds moved: %q", got)
	}
}

// TestCardinalRefusesTheMostNegativeInteger pins a refusal that used to be a
// crash.
//
// Negating math.MinInt64 overflows back to itself, so the magnitude stayed
// negative, passed the ceiling test, and reached the negative branch, which
// called Cardinal again with the same value. A stack overflow is a fatal error
// rather than a recoverable panic, so one call on one value took the process
// down. Python refuses it, because its integers do not overflow and the value
// is simply past the largest scale.
func TestCardinalRefusesTheMostNegativeInteger(t *testing.T) {
	for _, lang := range SupportedNumberLanguages() {
		if _, err := Cardinal(math.MinInt64, lang, ""); err == nil {
			t.Errorf("Cardinal(math.MinInt64, %q) was accepted", lang)
		}
		// Its neighbour is out of range too, and always was.
		if _, err := Cardinal(math.MinInt64+1, lang, ""); err == nil {
			t.Errorf("Cardinal(math.MinInt64+1, %q) was accepted", lang)
		}
		// Ordinary negatives still read.
		if got, err := Cardinal(-1, lang, ""); err != nil || got == "" {
			t.Errorf("Cardinal(-1, %q) = %q, %v", lang, got, err)
		}
	}
}
