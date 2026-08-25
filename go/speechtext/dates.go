// Dates and ordinals, said the way each language says them.
//
// A port of loudkit.frontend.dates: 12.03.2026 is the ordinary
// written date of five of these twelve languages, and without this funnel it
// reads as a clock time with a stray year, or as one eight-digit number. 1st
// arrives as "onest", because the number pass expands the digits and leaves the
// suffix stuck to them.
//
// Every rule is data from the shared numbers.json — month names, day forms, the
// infixes Spanish and Portuguese speak between the parts, the German oblique
// triggers, the ordinal tables. What is code here is the shape: which written
// forms are dates at all, and how each language reads a year.
//
// Two refusals are as deliberate as anything it does. A yearless "12.3." is
// never matched — its closing period is indistinguishable from a sentence's, so
// "Die Zahl ist 3.5." would otherwise come out as "dritte Mai". And 3/12/2026 is left alone in
// English, where it is March twelfth to half the world and the third of December
// to the other half: a listener recovers from hearing digits, not from a
// confident wrong month.//
// Python reference: `loudkit/frontend/dates.py`.
package speechtext

import (
	"encoding/json"
	"regexp"
	"strconv"
	"strings"
	"sync"
)

const (
	// Above this a four-digit run is an identifier, not a year.
	maxYear = 2999
	// A three-digit year exists; a three-digit anything is far more often a
	// quantity, and nothing in the string separates them.
	minYear = 1000
)

// February is 29 on purpose: a plausibility bound, not a calendar. Refusing 29
// February in a common year would reject a date a human wrote deliberately, and
// accepting it costs nothing.
var daysInMonth = [12]int{31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}

type dateRules struct {
	DayWords        map[int]string
	DayWordsOblique map[int]string
	ObliqueTriggers []string
	DayOneWord      string
	Months          []string
	DayMonthInfix   string
	MonthYearInfix  string
	DayFirstPrefix  string
	DayFirstInfix   string
	YearRule        string
	YearUnits       map[int]string
	YearTeens       map[int]string
	YearTens        map[int]string
	YearTwoThousand string
	DottedAmbiguous bool
	NoDottedDates   bool
	OrdSuffixes     []string
	OrdUnits        map[int]string
	OrdTeens        map[int]string
	OrdTens         map[int]string
	OrdJoiner       string
}

var (
	dateRulesOnce sync.Once
	dateTable     map[string]*dateRules
)

func loadDateRules() {
	var doc struct {
		Languages map[string]struct {
			Dates *struct {
				DayWords        map[string]string `json:"day_words"`
				DayWordsOblique map[string]string `json:"day_words_oblique"`
				ObliqueTriggers []string          `json:"oblique_triggers"`
				DayOneWord      string            `json:"day_one_word"`
				Months          []string          `json:"months"`
				DayMonthInfix   string            `json:"day_month_infix"`
				MonthYearInfix  string            `json:"month_year_infix"`
				DayFirstPrefix  string            `json:"day_first_prefix"`
				DayFirstInfix   string            `json:"day_first_infix"`
				YearRule        string            `json:"year_rule"`
				YearUnits       map[string]string `json:"year_units"`
				YearTeens       map[string]string `json:"year_teens"`
				YearTens        map[string]string `json:"year_tens"`
				YearTwoThousand string            `json:"year_two_thousand"`
				DottedAmbiguous bool              `json:"dotted_is_ambiguous"`
				NoDottedDates   bool              `json:"no_dotted_dates"`
			} `json:"dates"`
			Ordinals *struct {
				Suffixes   []string          `json:"suffixes"`
				Units      map[string]string `json:"units"`
				Teens      map[string]string `json:"teens"`
				Tens       map[string]string `json:"tens"`
				TensJoiner string            `json:"tens_joiner"`
			} `json:"ordinals"`
		} `json:"languages"`
	}
	if err := json.Unmarshal(numbersJSON, &doc); err != nil {
		panic("speechtext: embedded numbers.json is unreadable: " + err.Error())
	}
	ints := func(m map[string]string) map[int]string {
		out := make(map[int]string, len(m))
		for k, v := range m {
			if n, err := strconv.Atoi(k); err == nil && v != "" {
				out[n] = v
			}
		}
		return out
	}
	dateTable = make(map[string]*dateRules, len(doc.Languages))
	for lang, e := range doc.Languages {
		if e.Dates == nil {
			continue
		}
		d := e.Dates
		r := &dateRules{
			DayWords: ints(d.DayWords), DayWordsOblique: ints(d.DayWordsOblique),
			ObliqueTriggers: d.ObliqueTriggers, DayOneWord: d.DayOneWord,
			Months: d.Months, DayMonthInfix: d.DayMonthInfix,
			MonthYearInfix: d.MonthYearInfix, DayFirstPrefix: d.DayFirstPrefix,
			DayFirstInfix: d.DayFirstInfix, YearRule: d.YearRule,
			YearUnits: ints(d.YearUnits), YearTeens: ints(d.YearTeens),
			YearTens: ints(d.YearTens), YearTwoThousand: d.YearTwoThousand,
			DottedAmbiguous: d.DottedAmbiguous, NoDottedDates: d.NoDottedDates,
			OrdJoiner: "-",
		}
		if e.Ordinals != nil {
			r.OrdSuffixes = e.Ordinals.Suffixes
			r.OrdUnits = ints(e.Ordinals.Units)
			r.OrdTeens = ints(e.Ordinals.Teens)
			r.OrdTens = ints(e.Ordinals.Tens)
			if e.Ordinals.TensJoiner != "" {
				r.OrdJoiner = e.Ordinals.TensJoiner
			}
		}
		dateTable[lang] = r
	}
}

func dates() map[string]*dateRules {
	dateRulesOnce.Do(loadDateRules)
	return dateTable
}

func card(n int, lang string) string {
	s, err := Cardinal(int64(n), lang, "")
	if err != nil {
		return ""
	}
	return s
}

// MonthName is the month's name in this language, or "" when it has no table.
func MonthName(month int, language string) string {
	r, ok := dates()[language]
	if !ok || month < 1 || month > 12 || len(r.Months) != 12 {
		return ""
	}
	return r.Months[month-1]
}

// OrdinalDay is the day-of-month word, in whatever form this language's dates
// take. `oblique` is German only — the -en ending that am/den/vom select.
func OrdinalDay(day int, language string, oblique bool) string {
	r, ok := dates()[language]
	if !ok || day < 1 || day > 31 {
		return ""
	}
	if oblique {
		if w, ok := r.DayWordsOblique[day]; ok {
			return w
		}
	}
	if w, ok := r.DayWords[day]; ok {
		return w
	}
	// Cardinal languages: the day is just a number, except where the first of
	// the month is lexicalised.
	if day == 1 && r.DayOneWord != "" {
		return r.DayOneWord
	}
	return card(day, language)
}

// SayYear reads a year the way this language reads years.
//
// English and Norwegian split it; German, Dutch and Swedish group it in
// hundreds; the rest say one plain cardinal. Spanish is the explicit case — the
// RAE writes that a year is read as its cardinal and not in two-figure blocks as
// in English, so 2021 is "dos mil veintiuno".
func SayYear(year int, language string) string {
	r, ok := dates()[language]
	if !ok {
		return card(year, language)
	}
	switch r.YearRule {
	case "en_split":
		return yearEnglish(year)
	case "de_hundreds":
		return yearHundreds(year, "de", "hundert", 1100, 1999)
	case "nl_hundreds":
		return yearHundreds(year, "nl", "honderd", 1100, 1999)
	case "sv_hundreds":
		return yearHundreds(year, "sv", "hundra", 1100, 2099)
	case "no_split":
		return yearNorwegian(year)
	case "da_long":
		return yearDanish(year)
	case "pl_ordinal_genitive":
		return yearPolish(year, r)
	}
	return card(year, language)
}

func yearEnglish(year int) string {
	if year == 1000 || year == 2000 || (year >= 2001 && year <= 2009) {
		return card(year, "en")
	}
	if (year > 1000 && year < 2000) || year >= 2100 {
		century, rest := year/100, year%100
		if rest == 0 {
			return card(century, "en") + " hundred"
		}
		// "nineteen oh five" — never "nineteen five", which nobody says.
		if rest < 10 {
			return card(century, "en") + " oh " + card(rest, "en")
		}
		return card(century, "en") + " " + card(rest, "en")
	}
	if year >= 2010 && year <= 2099 {
		return "twenty " + card(year%100, "en")
	}
	return card(year, "en")
}

// German, Dutch and Swedish all write <century><joiner><rest> solid; only the
// joiner and the range differ. German stops at 1999 because the GfdS explicitly
// rejects "zwanzighundert…"; Swedish runs to 2099 because Isof has recommended
// the "tjugohundra…" series for decades.
func yearHundreds(year int, lang, joiner string, lo, hi int) string {
	if year < lo || year > hi {
		return card(year, lang)
	}
	century, rest := year/100, year%100
	head := card(century, lang) + joiner
	if rest == 0 {
		return head
	}
	return head + card(rest, lang)
}

// Norwegian splits 1100–1999 and drops "hundre": 1972 is "nittensyttito".
func yearNorwegian(year int) string {
	if year < 1100 || year > 1999 {
		return card(year, "no")
	}
	century, rest := year/100, year%100
	if rest == 0 {
		return card(century, "no") + "hundre"
	}
	return card(century, "no") + card(rest, "no")
}

// Dansk Sprognævn: the long form works for every year, and the short
// "telephone-number" form is explicitly poor for a century's first decade.
func yearDanish(year int) string {
	if year < 1100 || year > 1999 {
		return card(year, "da")
	}
	century, rest := year/100, year%100
	head := card(century, "da") + " hundrede"
	if rest == 0 {
		return head
	}
	return head + " og " + card(rest, "da")
}

// Only the tens and units of a Polish year decline. PWN's worked example is
// "tysiąc dziewięćset dziewięćdziesiątego drugiego": the thousands and hundreds
// keep their cardinal form and the ordinal genitive lands on the last two
// digits. Where those are zero the declension moves left, which is why 2000 has
// its own word.
func yearPolish(year int, r *dateRules) string {
	if year == 2000 && r.YearTwoThousand != "" {
		return r.YearTwoThousand
	}
	head, rest := year/100, year%100
	lead := ""
	if head != 0 {
		lead = card(head*100, "pl")
	}
	if rest == 0 {
		return lead
	}
	var tail string
	if teen, ok := r.YearTeens[rest]; ok {
		tail = teen
	} else {
		var words []string
		if w := r.YearTens[(rest/10)*10]; w != "" {
			words = append(words, w)
		}
		if w := r.YearUnits[rest%10]; w != "" {
			words = append(words, w)
		}
		tail = strings.Join(words, " ")
	}
	return strings.TrimSpace(lead + " " + tail)
}

func dateValid(day, month int, year int, hasYear bool) bool {
	if month < 1 || month > 12 {
		return false
	}
	if day < 1 || day > daysInMonth[month-1] {
		return false
	}
	if !hasYear {
		return true
	}
	return year >= minYear && year <= maxYear
}

func spokenDate(day, month, year int, hasYear bool, language string, oblique bool) string {
	r, ok := dates()[language]
	if !ok {
		return ""
	}
	dayWord := OrdinalDay(day, language, oblique)
	monthWord := MonthName(month, language)
	if dayWord == "" || monthWord == "" {
		return ""
	}
	parts := []string{dayWord}
	if r.DayMonthInfix != "" {
		parts = append(parts, r.DayMonthInfix)
	}
	parts = append(parts, monthWord)
	if hasYear {
		if r.MonthYearInfix != "" {
			parts = append(parts, r.MonthYearInfix)
		}
		parts = append(parts, SayYear(year, language))
	}
	return strings.Join(parts, " ")
}

var (
	isoDate = regexp.MustCompile(`([12][0-9]{3})-([01][0-9])-([0-3][0-9])`)
	// With the year, which is what makes it a date rather than a guess. The
	// yearless "12.3." is deliberately not matched — see the package note.
	dottedDate = regexp.MustCompile(`([0-3]?[0-9])\.([01]?[0-9])\.([12][0-9]{3})`)
	// Day-first in every language here; English is handled in the callback,
	// where the field order is genuinely ambiguous.
	slashedDate = regexp.MustCompile(`([0-3]?[0-9])/([01]?[0-9])/([12][0-9]{3})`)
)

// ExpandDates says every written date in text the way language says it.
//
// Never panics and never invents: a run failing the bounds check, or whose field
// order cannot be resolved, comes back exactly as it was written.
func ExpandDates(text, language string) string {
	r, ok := dates()[language]
	if !ok {
		return text
	}
	out := replaceDates(text, isoDate, func(g []string, at int, whole string) string {
		y, _ := strconv.Atoi(g[1])
		m, _ := strconv.Atoi(g[2])
		d, _ := strconv.Atoi(g[3])
		if !boundedBefore(whole, at, ".,:/-") || !boundedAfter(whole, at+len(g[0]), "-") {
			return ""
		}
		if !dateValid(d, m, y, true) {
			return ""
		}
		return spokenDate(d, m, y, true, language, isObliqueGo(whole, at, r))
	})
	out = replaceDates(out, dottedDate, func(g []string, at int, whole string) string {
		// Swedish marks an ordinal with a colon (1:a), never a trailing period,
		// so "12." there is a list number or a sentence end. English writes
		// dotted dates almost never, and when it does the field order is as
		// unresolvable as in the slashed form.
		if r.NoDottedDates || r.DottedAmbiguous {
			return ""
		}
		d, _ := strconv.Atoi(g[1])
		m, _ := strconv.Atoi(g[2])
		y, _ := strconv.Atoi(g[3])
		if !boundedBefore(whole, at, ".,:/-") || !wordBoundaryAfter(whole, at+len(g[0])) {
			return ""
		}
		if !dateValid(d, m, y, true) {
			return ""
		}
		return spokenDate(d, m, y, true, language, isObliqueGo(whole, at, r))
	})
	out = replaceDates(out, slashedDate, func(g []string, at int, whole string) string {
		d, _ := strconv.Atoi(g[1])
		m, _ := strconv.Atoi(g[2])
		y, _ := strconv.Atoi(g[3])
		if !boundedBefore(whole, at, ".,:/-") || !boundedAfter(whole, at+len(g[0]), "/") {
			return ""
		}
		// 3/12/2026 is March twelfth to half the English-speaking world and the
		// third of December to the other half, and nothing says which.
		if language == "en" && d <= 12 {
			return ""
		}
		if !dateValid(d, m, y, true) {
			return ""
		}
		return spokenDate(d, m, y, true, language, isObliqueGo(whole, at, r))
	})
	return textualDates(out, language, r)
}

// Go's RE2 has no lookaround, so the guards Python writes as `(?<![\d.,:/-])`
// and `(?![\d/])` are checked here against the characters either side of a
// match. Same rule, expressed where the engine can express it.
//
// A digit either side is refused by both, always — every call site's class has
// `\d` in it. It is tested in code rather than passed in as text because the
// `\d` written into the class *was* passed in as text, and `strings.ContainsRune`
// read it as the two literal runes `\` and `d`: no digit ever matched, and
// `42.3.2026` in German reached the dotted-date reading as *4zweite März
// zweitausendsechsundzwanzig* — a wrong day welded to a stray digit.
//
// `marks` is what is left: the separators, as literal runes.
func boundedBefore(s string, at int, marks string) bool {
	if at == 0 {
		return true
	}
	c := s[at-1]
	return !isASCIIDigit(c) && !strings.ContainsRune(marks, rune(c))
}

func boundedAfter(s string, end int, marks string) bool {
	if end >= len(s) {
		return true
	}
	c := s[end]
	return !isASCIIDigit(c) && !strings.ContainsRune(marks, rune(c))
}

func wordBoundaryAfter(s string, end int) bool {
	if end >= len(s) {
		return true
	}
	c := s[end]
	return !(c == '_' || (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))
}

// isObliqueGo reports whether am/den/vom sits before the date. German only.
func isObliqueGo(whole string, at int, r *dateRules) bool {
	if len(r.ObliqueTriggers) == 0 || at > len(whole) {
		return false
	}
	before := strings.TrimRight(whole[:at], " \t\n")
	fields := strings.Fields(before)
	if len(fields) == 0 {
		return false
	}
	tail := strings.ToLower(strings.Trim(fields[len(fields)-1], ",;:"))
	for _, w := range r.ObliqueTriggers {
		if strings.ToLower(w) == tail {
			return true
		}
	}
	return false
}

// textualDates handles "12 marca 2026", "12. März 2026", "March 12, 2026" — a
// written month name beside a bare day. The name is the disambiguator, so this
// runs for every language including English.
func textualDates(text, language string, r *dateRules) string {
	if len(r.Months) != 12 {
		return text
	}
	names := make([]string, 0, 12)
	for _, m := range r.Months {
		names = append(names, regexp.QuoteMeta(m))
	}
	joined := strings.Join(names, "|")
	// Spanish and Portuguese speak a preposition between every part, so the
	// written form carries it too: "12 de marzo de 2026".
	infix, yinfix := "", ""
	if r.DayMonthInfix != "" {
		infix = `(?:\s+` + regexp.QuoteMeta(r.DayMonthInfix) + `)?`
	}
	if r.MonthYearInfix != "" {
		yinfix = `(?:\s+` + regexp.QuoteMeta(r.MonthYearInfix) + `)?`
	}
	dayFirst := regexp.MustCompile(
		`(?i)([0-3]?[0-9])\.?` + infix + `\s+(` + joined + `)(?:` + yinfix + `\s+([12][0-9]{3}))?`)
	out := replaceDates(text, dayFirst, func(g []string, at int, whole string) string {
		if !wordBoundaryBefore(whole, at) || !wordBoundaryAfter(whole, at+len(g[0])) {
			return ""
		}
		d, _ := strconv.Atoi(g[1])
		m := monthIndexGo(g[2], r)
		y, hasYear := 0, false
		if len(g) > 3 && g[3] != "" {
			y, _ = strconv.Atoi(g[3])
			hasYear = true
		}
		if m == 0 || !dateValid(d, m, y, hasYear) {
			return ""
		}
		if r.DayFirstPrefix != "" || r.DayFirstInfix != "" {
			// English written day-first reads "the twelfth of March": both
			// dialects say it that way, so no locale flag is needed.
			head := OrdinalDay(d, language, false)
			rest := []string{MonthName(m, language)}
			if hasYear {
				rest = append(rest, SayYear(y, language))
			}
			prefix := ""
			if r.DayFirstPrefix != "" {
				prefix = r.DayFirstPrefix + " "
			}
			join := " "
			if r.DayFirstInfix != "" {
				join = " " + r.DayFirstInfix + " "
			}
			return prefix + head + join + strings.Join(rest, " ")
		}
		return spokenDate(d, m, y, hasYear, language, isObliqueGo(whole, at, r))
	})

	// Month-first is an English shape. Reading it in a language that never
	// writes it would be inventing a construction nobody used.
	if r.DayFirstInfix == "" {
		return out
	}
	monthFirst := regexp.MustCompile(
		`(?i)(` + joined + `)\s+([0-3]?[0-9])(?:st|nd|rd|th)?,?(?:\s+([12][0-9]{3}))?`)
	return replaceDates(out, monthFirst, func(g []string, at int, whole string) string {
		if !wordBoundaryBefore(whole, at) || !wordBoundaryAfter(whole, at+len(g[0])) {
			return ""
		}
		m := monthIndexGo(g[1], r)
		d, _ := strconv.Atoi(g[2])
		y, hasYear := 0, false
		if len(g) > 3 && g[3] != "" {
			y, _ = strconv.Atoi(g[3])
			hasYear = true
		}
		if m == 0 || !dateValid(d, m, y, hasYear) {
			return ""
		}
		parts := []string{MonthName(m, language), OrdinalDay(d, language, false)}
		if hasYear {
			parts = append(parts, SayYear(y, language))
		}
		return strings.Join(parts, " ")
	})
}

func wordBoundaryBefore(s string, at int) bool {
	if at == 0 {
		return true
	}
	c := s[at-1]
	return !(c == '_' || (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))
}

func monthIndexGo(name string, r *dateRules) int {
	lowered := strings.ToLower(name)
	for i, candidate := range r.Months {
		if strings.ToLower(candidate) == lowered {
			return i + 1
		}
	}
	return 0
}

// Ordinal is value as a written-out ordinal, or "" when this language has no
// table for it.
//
// Composed rather than enumerated past ninety-nine: the hundreds and above stay
// cardinal and only the last two digits become an ordinal, so 101st is "one
// hundred and first".
func Ordinal(value int, language string) string {
	r, ok := dates()[language]
	if !ok || len(r.OrdUnits) == 0 || value < 0 {
		return ""
	}
	head, rest := value/100, value%100
	tail := twoDigitOrdinal(rest, r)
	if tail == "" {
		return ""
	}
	if head == 0 {
		return tail
	}
	lead := card(head*100, language)
	if rest != 0 {
		return lead + " " + tail
	}
	return lead
}

func twoDigitOrdinal(value int, r *dateRules) string {
	if teen, ok := r.OrdTeens[value]; ok {
		return teen
	}
	tens, units := value/10, value%10
	if units == 0 {
		return r.OrdTens[tens*10]
	}
	if tens == 0 {
		return r.OrdUnits[units]
	}
	unitWord, ok := r.OrdUnits[units]
	if !ok {
		return ""
	}
	// The tens word is English's, because English is the only language of the
	// twelve writing an ordinal as digits plus a suffix.
	return card(tens*10, "en") + r.OrdJoiner + unitWord
}

// ExpandOrdinals says 1st and 22nd as words.
//
// English is the only one of the twelve writing an ordinal as digits plus a
// letter suffix, so for every other language this is a no-op. It runs before the
// number pass, which would otherwise expand the digits and leave the suffix
// stuck to them: "onest", "fiveth place", "twenty-twond".
func ExpandOrdinals(text, language string) string {
	r, ok := dates()[language]
	if !ok || len(r.OrdSuffixes) == 0 {
		return text
	}
	re := regexp.MustCompile(`(?i)([0-9]+)(` + strings.Join(r.OrdSuffixes, "|") + `)`)
	return replaceDates(text, re, func(g []string, at int, whole string) string {
		if !wordBoundaryBefore(whole, at) || !wordBoundaryAfter(whole, at+len(g[0])) {
			return ""
		}
		n, err := strconv.Atoi(g[1])
		if err != nil {
			return ""
		}
		return Ordinal(n, language)
	})
}

// replaceDates rewrites every match, right to left so earlier offsets stay
// valid. The callback gets the capture groups (index 0 is the whole match), the
// match offset, and the string being scanned — the last two because the German
// oblique test reads the word before the date, and because RE2 cannot express
// the lookaround Python uses. Returning "" leaves that match exactly as
// written, which is this module's answer whenever the evidence runs out.
func replaceDates(text string, re *regexp.Regexp, body func([]string, int, string) string) string {
	locs := re.FindAllStringSubmatchIndex(text, -1)
	if len(locs) == 0 {
		return text
	}
	out := text
	for i := len(locs) - 1; i >= 0; i-- {
		loc := locs[i]
		groups := make([]string, 0, len(loc)/2)
		for g := 0; g < len(loc); g += 2 {
			if loc[g] < 0 {
				groups = append(groups, "")
				continue
			}
			groups = append(groups, text[loc[g]:loc[g+1]])
		}
		said := body(groups, loc[0], text)
		if said == "" {
			continue
		}
		out = out[:loc[0]] + said + out[loc[1]:]
	}
	return out
}
