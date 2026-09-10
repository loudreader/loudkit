package speechtext

import "testing"

func TestRespell(t *testing.T) {
	cases := []struct{ in, lang, want string }{
		{"Pobierz download i zrób code review.", "pl", "Pobierz dałnloud i zrób koud riwju."},
		{"Rabat 15% na weekend!", "pl", "Rabat piętnaście procent na łikend!"},
		{"The quick brown fox jumps over the lazy dog.", "pl", "Da kłyk brałn faks dżamps ołwer da lejzi dog."},
		{"Skończ deadline'u przed piątkiem.", "pl", "Skończ dedlajnu przed piątkiem."},
		{"GPT działa dobrze na USB.", "pl", "gie-pe-te działa dobrze na u-es-be."},
		{"2.5 GB to dużo.", "pl", "dwa przecinek pięć gie-be to dużo."},
		{"download", "pl", "dałnloud"},
		{"queue", "pl", "kju"},
		{"thought", "pl", "tot"},
		{"juice", "pl", "dżus"},
	}
	for _, c := range cases {
		if got := Prepared(c.in, c.lang); got != c.want {
			t.Errorf("Prepared(%q, %q) = %q, want %q", c.in, c.lang, got, c.want)
		}
	}
}

// TestTheRespellerSaysNumbersWithTheSharedGrammar pins what belongs to this
// path now that Cardinal does the saying: the two refusals that are the
// respeller's and not the grammar's. A leading zero is a code, a PIN or a
// house number rather than a quantity, and past six digits the reading is
// longer than the digits and no more use.
func TestTheRespellerSaysNumbersWithTheSharedGrammar(t *testing.T) {
	for _, v := range []int64{0, 1, 7, 12, 15, 21, 42, 100, 101, 111, 200, 999,
		1000, 1001, 1002, 1005, 1012, 1014, 2000, 3000, 5000, 12000, 14000,
		21000, 22000, 100000, 123456, 999999} {
		token := ""
		for n, digits := v, []byte(nil); ; n /= 10 {
			digits = append([]byte{byte('0' + n%10)}, digits...)
			if n < 10 {
				token = string(digits)
				break
			}
		}
		want, err := Cardinal(v, "pl", "")
		if err != nil {
			t.Fatalf("Cardinal(%d, pl): %v", v, err)
		}
		if got, ok := numberWords(token); !ok || got != want {
			t.Errorf("numberWords(%q) = %q (%v), want %q", token, got, ok, want)
		}
	}
	// The refusals, which send the token to the digit-by-digit fallback.
	for _, token := range []string{"01", "007", "0123456", "1234567", "12345678"} {
		if s, ok := numberWords(token); ok {
			t.Errorf("numberWords(%q) = %q, want a refusal", token, s)
		}
	}
	if s, ok := numberWords("0"); !ok || s != "zero" {
		t.Errorf(`numberWords("0") = %q, %v`, s, ok)
	}
	// Digits in any script still read: the fold walks runes, not bytes.
	if s, ok := numberWords("١٢٣"); !ok || s != "sto dwadzieścia trzy" {
		t.Errorf("Arabic-Indic 123 = %q, %v", s, ok)
	}
}

// TestTheCodeTokenTableIsTheSharedTableInASCII pins the width of the letter
// table this path spells with.
//
// It is derived from the shared Polish letter names, and only over ASCII. The
// shared table also names the nine Polish letters; taking those too would spell
// Ż1 as "żet jeden", where a token written with Polish letters is a Polish word
// that a reader reads rather than spells.
func TestTheCodeTokenTableIsTheSharedTableInASCII(t *testing.T) {
	if len(letterNames) != 26 {
		t.Errorf("the code-token table has %d entries, want the 26 ASCII letters", len(letterNames))
	}
	for r := 'a'; r <= 'z'; r++ {
		if got, want := letterNames[string(r)], LetterName(string(r), "pl"); got != want {
			t.Errorf("%q: code table %q, shared table %q", r, got, want)
		}
	}
	for _, r := range "ąćęłńóśźż" {
		if n, ok := letterNames[string(r)]; ok {
			t.Errorf("the code-token table names %q as %q; it is ASCII only", r, n)
		}
		if LetterName(string(r), "pl") == "" {
			t.Errorf("the shared table lost its name for %q", r)
		}
	}
	if s, ok := spelledCodeToken("Ż1"); ok {
		t.Errorf("spelledCodeToken(\"Ż1\") = %q, want it left written", s)
	}
	if s, ok := spelledCodeToken("R2"); !ok || s != "er dwa" {
		t.Errorf(`spelledCodeToken("R2") = %q, %v`, s, ok)
	}
}

// TestTheRespellerSpellsAcronymsWithTheSharedTable pins the cap this path keeps.
//
// SpellAcronym checks its word list before the five-letter cap, so UNESCO,
// UNICEF and INTERPOL come back from it lowercased. This path caps first,
// because it has never had an answer for a word that long.
func TestTheRespellerSpellsAcronymsWithTheSharedTable(t *testing.T) {
	for _, c := range []struct {
		word, want string
		ok         bool
	}{
		{"GPT", "gie-pe-te", true},
		{"USB", "u-es-be", true},
		{"NASA", "nasa", true},
		{"ZUS", "zus", true},
		{"UNESCO", "", false},
		{"INTERPOL", "", false},
		{"A", "", false},
		{"ABCDEF", "", false},
		{"Abc", "", false},
	} {
		got, ok := spelledAcronym(c.word)
		if got != c.want || ok != c.ok {
			t.Errorf("spelledAcronym(%q) = %q, %v; want %q, %v", c.word, got, ok, c.want, c.ok)
		}
	}
}

// TestATokenWithNoLetterIsReadAsDigits pins the guard on the respeller's digit
// branch. The word collector keeps an apostrophe inside a word, so a quoted
// address arrives as `'192`: "no letter" takes the branch, "every character is
// a digit" does not, and the four other ports ask the first question.
//
// The digit-by-digit fallback writes an unnamed character out as itself. The
// alternative drops it, and an apostrophe that disappears between two readings
// of the same sentence is a change the listener cannot hear happening.
func TestATokenWithNoLetterIsReadAsDigits(t *testing.T) {
	for _, c := range []struct{ word, want string }{
		{"'192", "' jeden dziewięć dwa"},
		{"192'", "jeden dziewięć dwa '"},
		{"192", "sto dziewięćdziesiąt dwa"},
		{"007", "zero zero siedem"},
		{"’254’", "’ dwa pięć cztery ’"},
		{"1234567", "jeden dwa trzy cztery pięć sześć siedem"},
	} {
		if got := respelled(c.word); got != c.want {
			t.Errorf("respelled(%q) = %q, want %q", c.word, got, c.want)
		}
	}
}

// TestAQuotedAddressReadsTheSameAsInTheOtherPorts is the funnel row the guard
// was found by.
func TestAQuotedAddressReadsTheSameAsInTheOtherPorts(t *testing.T) {
	const want = "Wpisz ' jeden dziewięć dwa.sto sześćdziesiąt osiem przecinek zero.jeden ' w przeglądarce."
	if got := Prepared("Wpisz '192.168.0.1' w przeglądarce.", "pl"); got != want {
		t.Errorf("Prepared = %q, want %q", got, want)
	}
}
