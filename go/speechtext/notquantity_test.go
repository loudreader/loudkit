package speechtext

import "testing"

// A digit run with two or more separators is a version, an address or a date —
// never a number. Reading one as a number says the segments as one value: with
// the comma as the decimal mark the dots are treated as thousands grouping and
// the segments concatenate, so 192.168.0.1 is spoken as "nineteen million two
// hundred sixteen thousand eight hundred one". The Python reference
// additionally crashes on these.
//
// Every literal below is one that shipped wrong; the tests pin them.
func TestDigitsThatAreNotQuantities(t *testing.T) {
	notQuantities := []string{"1.2.3", "1.2.3.4", "192.168.0.1", "12.03.2026", "10.0.0.255"}
	for _, lang := range SupportedNumberLanguages() {
		for _, literal := range notQuantities {
			if got := ExpandNumbers(literal, lang); got != literal {
				t.Errorf("%s: ExpandNumbers(%q) = %q, want it left alone", lang, literal, got)
			}
		}
	}
}

func TestRealNumbersStillRead(t *testing.T) {
	// The guard must not buy correctness by refusing everything.
	for _, lang := range SupportedNumberLanguages() {
		for _, literal := range []string{"7", "2,5", "2.5"} {
			if got := ExpandNumbers(literal, lang); got == literal {
				t.Errorf("%s: ExpandNumbers(%q) left it as digits", lang, literal)
			}
		}
	}
}

// Two separators that group are a number: 1.234.567 is a million and a bit in
// the eleven languages whose decimal mark is the comma. The rule is "three
// digits after the first separator", not "at most one separator".
func TestGroupedThousandsAreStillANumber(t *testing.T) {
	for _, lang := range SupportedNumberLanguages() {
		if lang == "en" {
			continue // English groups with commas, not dots
		}
		if got := ExpandNumbers("1.234.567", lang); got == "1.234.567" {
			t.Errorf("%s: grouped thousands were refused", lang)
		}
	}
	if got := ExpandNumbers("1,234,567", "en"); got == "1,234,567" {
		t.Errorf("en: grouped thousands were refused")
	}
}

// 12.03.2026 is a date, and ExpandTimes must not eat its first half: the
// pattern matches 12.03 inside it, so the ordinary written date of five of the
// twelve languages would be spoken as a clock time with the year trailing
// behind.
func TestATimeIsNotPartOfADate(t *testing.T) {
	for _, lang := range SupportedNumberLanguages() {
		for _, literal := range []string{"12.03.2026", "am 05.11.2025 kam"} {
			if got := ExpandTimes(literal, lang); got != literal {
				t.Errorf("%s: ExpandTimes(%q) = %q, want it left alone", lang, literal, got)
			}
		}
		// A dotted time reads only where the dot is not the decimal point: `14.30`
		// is half past two in eleven of these languages and a number in the
		// twelfth. Asserting it for all twelve made every English decimal with two
		// fraction digits a clock time.
		if lang != "en" {
			if got := ExpandTimes("14.30", lang); got == "14.30" {
				t.Errorf("ExpandTimes(%q, %q) left it as written", "14.30", lang)
			}
		} else if got := ExpandTimes("14.30", lang); got != "14.30" {
			t.Errorf("ExpandTimes(%q, %q) = %q, want it left as a decimal", "14.30", lang, got)
		}
		for _, literal := range []string{"14:30", "at 14:30."} {
			if got := ExpandTimes(literal, lang); got == literal {
				t.Errorf("%s: ExpandTimes(%q) stopped reading a real time", lang, literal)
			}
		}
	}
}
