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
