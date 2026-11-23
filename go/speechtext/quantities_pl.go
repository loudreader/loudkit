package speechtext

import (
	"regexp"
	"strconv"
	"strings"
	"unicode/utf8"
)

var (
	// Consume an unmarked digit run whole so its last digits cannot become a quantity.
	polishQuantity = regexp.MustCompile(`([0-9]+(?:[ \x{00a0}\x{202f}][0-9]{3})*)(?:,([0-9]+))?(?:[ \t\x{00a0}\x{202f}]*(zł|kg|m²))?`)
	polishChoice   = regexp.MustCompile(`([Ww]ybierz|[Zz]aznacz|[Ww]skaż) +([1-9]) +z +(10|[2-9]) +(?:(ofert|opcji|pozycji|propozycji|wiadomości)|(produktów|plików|biletów|elementów))`)
	polishMessages = regexp.MustCompile(`([Mm]asz) +2( +(?:nowe +)?wiadomości)`)
)

// normalizePolishQuantities runs before numeral folding, while m² still names a unit.
func normalizePolishQuantities(text string) string {
	text = replacePolishMatches(text, polishChoice, func(text string, m []int) string {
		chosen := matchGroup(text, m, 2)
		noun := matchGroup(text, m, 4)
		if noun == "" {
			noun = matchGroup(text, m, 5)
		} else {
			switch chosen {
			case "1":
				chosen = "jedną"
			case "2":
				chosen = "dwie"
			}
		}
		total, _ := strconv.Atoi(matchGroup(text, m, 3))
		genitives := [...]string{"dwóch", "trzech", "czterech", "pięciu", "sześciu", "siedmiu", "ośmiu", "dziewięciu", "dziesięciu"}
		return matchGroup(text, m, 1) + " " + chosen + " z " + genitives[total-2] + " " + noun
	})
	text = replacePolishMatches(text, polishMessages, func(text string, m []int) string {
		return matchGroup(text, m, 1) + " dwie" + matchGroup(text, m, 2)
	})
	return replacePolishMatches(text, polishQuantity, polishQuantityWords)
}

func replacePolishMatches(text string, pattern *regexp.Regexp, words func(string, []int) string) string {
	var out strings.Builder
	last := 0
	for _, m := range pattern.FindAllStringSubmatchIndex(text, -1) {
		start, end := m[0], m[1]
		before, _ := utf8.DecodeLastRuneInString(text[:start])
		after, _ := utf8.DecodeRuneInString(text[end:])
		if wordCharacter(before) || wordCharacter(after) {
			continue
		}
		said := words(text, m)
		if said == "" {
			continue
		}
		out.WriteString(text[last:start])
		out.WriteString(said)
		last = end
	}
	if last == 0 {
		return text
	}
	out.WriteString(text[last:])
	return out.String()
}

func polishQuantityWords(text string, m []int) string {
	unit := matchGroup(text, m, 3)
	if unit == "" {
		return ""
	}
	start := m[0]
	if start >= 2 && (text[start-1] == '.' || text[start-1] == ',') && isASCIIDigit(text[start-2]) {
		return ""
	}
	groups := strings.Fields(matchGroup(text, m, 1))
	if len(groups) > 1 && len(groups[0]) > 3 {
		return ""
	}
	whole := strings.Join(groups, "")
	if len(whole) > 9 || (len(whole) > 1 && whole[0] == '0') {
		return ""
	}
	count, _ := strconv.ParseInt(whole, 10, 64)
	fraction := matchGroup(text, m, 2)
	if unit == "zł" {
		if len(fraction) > 2 {
			return ""
		}
		zlotys := whole + " " + scaleWord(count, []string{"złoty", "złote", "złotych"})
		cents, _ := strconv.ParseInt((fraction + "00")[:2], 10, 64)
		if cents == 0 {
			return zlotys
		}
		grosze := strconv.FormatInt(cents, 10) + " " + scaleWord(cents, []string{"grosz", "grosze", "groszy"})
		if count == 0 {
			return grosze
		}
		return zlotys + " i " + grosze
	}
	if len(fraction) > 6 {
		return ""
	}
	forms := []string{"kilogram", "kilogramy", "kilogramów"}
	fractionalUnit := "kilograma"
	if unit == "m²" {
		forms = []string{"metr kwadratowy", "metry kwadratowe", "metrów kwadratowych"}
		fractionalUnit = "metra kwadratowego"
	}
	if strings.Trim(fraction, "0") != "" {
		return whole + "," + fraction + " " + fractionalUnit
	}
	return whole + " " + scaleWord(count, forms)
}
