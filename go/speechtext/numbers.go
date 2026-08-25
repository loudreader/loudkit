// Numbers, said out loud — the Go half of loudkit.frontend.numbers.
//
// The grammar is data and only the interpreter is code: this file reads the
// same numbers.json every other implementation reads, so a rule lives once.
// Twelve languages times five implementations would otherwise be sixty chances
// for a rule to drift, and the CLDR differential (1300 rows) plus the
// hand-written fixture are what catch the drift that remains.
//
// The composition mirrors loudkit/frontend/numbers.py function for function.
// Where a behaviour looks odd — the joiner carrying its own spacing, agreement
// scopes per value, a scale noun with its own gender — the reason lives in the
// Python docstrings and in docs/reference/preprocess.md, and the fixture pins
// it.
//
// Python reference: loudkit/frontend/numbers.py.
package speechtext

import (
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Scale is one scale noun (thousand, million …) and how it behaves. The CLDR
// differential showed the behaviours are per scale, not per language: German
// writes eintausend solid but "eine Million" as two words; Danish says
// "tusind og et" but "en million et".
type Scale struct {
	Value            int64
	Forms            []string
	OneWord          string // "~" composes; "" bare; else the literal word
	Separate         bool
	Link             string
	SmallJoiner      string
	MultiplierAgrees bool
	MultiplierGender string
}

// Grammar is how one language builds a number word.
type Grammar struct {
	Ones                       []string
	Teens                      []string
	Tens                       []string
	Hundred                    string
	Hundreds                   []string
	HundredsGendered           map[string][]string
	HundredPluralFinal         string
	Scales                     []Scale
	UnitsBeforeTens            bool
	UnitTensJoiner             string
	TimeInfix                  string
	Abbreviations              []AbbrevEntry
	TensJoinerExceptions       map[int64]string
	HundredJoiner              string
	ScaleJoinerOnRoundHundreds bool
	ScaleLargeJoiner           string
	OneBeforeHundred           bool
	OneBeforeScale             bool
	WordJoin                   string
	MinusWord                  string
	DecimalSeparator           string
	DecimalWord                string
	Exceptions                 map[int64]string
	Genders                    map[string]map[int64]string
	GenderScopes               map[int64]string
	CombiningOnes              map[int64]string
}

// AbbrevEntry is one written->spoken pair, kept ordered longest-written-first
// so fr.o.m. cannot be half-eaten by a shorter entry.
type AbbrevEntry struct{ Written, Spoken string }

var grammars map[string]*Grammar

func loadGrammars() {
	var doc struct {
		Languages map[string]struct {
			Ones               []string            `json:"ones"`
			Teens              []string            `json:"teens"`
			Tens               []string            `json:"tens"`
			Hundred            string              `json:"hundred"`
			Hundreds           []string            `json:"hundreds"`
			HundredsGendered   map[string][]string `json:"hundreds_gendered"`
			HundredPluralFinal string              `json:"hundred_plural_final"`
			Scales             []struct {
				Value            int64    `json:"value"`
				Forms            []string `json:"forms"`
				One              *string  `json:"one"`
				Separate         bool     `json:"separate"`
				Link             string   `json:"link"`
				SmallJoiner      string   `json:"small_joiner"`
				MultiplierAgrees bool     `json:"multiplier_agrees"`
				MultiplierGender string   `json:"multiplier_gender"`
			} `json:"scales"`
			UnitsBeforeTens            bool                         `json:"units_before_tens"`
			UnitTensJoiner             string                       `json:"unit_tens_joiner"`
			TimeInfix                  string                       `json:"time_infix"`
			Abbreviations              map[string]string            `json:"abbreviations"`
			TensJoinerExceptions       map[string]string            `json:"tens_joiner_exceptions"`
			HundredJoiner              string                       `json:"hundred_joiner"`
			ScaleJoinerOnRoundHundreds bool                         `json:"scale_joiner_on_round_hundreds"`
			ScaleLargeJoiner           string                       `json:"scale_large_joiner"`
			OneBeforeHundred           bool                         `json:"one_before_hundred"`
			OneBeforeScale             bool                         `json:"one_before_scale"`
			WordJoin                   string                       `json:"word_join"`
			MinusWord                  string                       `json:"minus_word"`
			DecimalSeparator           string                       `json:"decimal_separator"`
			DecimalWord                string                       `json:"decimal_word"`
			Exceptions                 map[string]string            `json:"exceptions"`
			Genders                    map[string]map[string]string `json:"genders"`
			GenderScopes               map[string]string            `json:"gender_scopes"`
			CombiningOnes              map[string]string            `json:"combining_ones"`
		} `json:"languages"`
	}
	if err := json.Unmarshal(numbersJSON, &doc); err != nil {
		panic("speechtext: embedded numbers.json is unreadable: " + err.Error())
	}
	grammars = make(map[string]*Grammar, len(doc.Languages))
	intKeys := func(m map[string]string) map[int64]string {
		out := make(map[int64]string, len(m))
		for k, v := range m {
			n, err := strconv.ParseInt(k, 10, 64)
			if err != nil {
				panic("speechtext: non-integer key in numbers.json: " + k)
			}
			out[n] = v
		}
		return out
	}
	for lang, e := range doc.Languages {
		g := &Grammar{
			Ones: e.Ones, Teens: e.Teens, Tens: e.Tens,
			Hundred: e.Hundred, Hundreds: e.Hundreds,
			HundredsGendered:           e.HundredsGendered,
			HundredPluralFinal:         e.HundredPluralFinal,
			UnitsBeforeTens:            e.UnitsBeforeTens,
			UnitTensJoiner:             e.UnitTensJoiner,
			TimeInfix:                  e.TimeInfix,
			TensJoinerExceptions:       intKeys(e.TensJoinerExceptions),
			HundredJoiner:              e.HundredJoiner,
			ScaleJoinerOnRoundHundreds: e.ScaleJoinerOnRoundHundreds,
			ScaleLargeJoiner:           e.ScaleLargeJoiner,
			OneBeforeHundred:           e.OneBeforeHundred,
			OneBeforeScale:             e.OneBeforeScale,
			WordJoin:                   e.WordJoin,
			MinusWord:                  e.MinusWord,
			DecimalSeparator:           e.DecimalSeparator,
			DecimalWord:                e.DecimalWord,
			Exceptions:                 intKeys(e.Exceptions),
			GenderScopes:               intKeys(e.GenderScopes),
			CombiningOnes:              intKeys(e.CombiningOnes),
		}
		for w, sp := range e.Abbreviations {
			g.Abbreviations = append(g.Abbreviations, AbbrevEntry{w, sp})
		}
		sort.Slice(g.Abbreviations, func(i, j int) bool {
			return len(g.Abbreviations[i].Written) > len(g.Abbreviations[j].Written)
		})
		g.Genders = make(map[string]map[int64]string, len(e.Genders))
		for name, forms := range e.Genders {
			g.Genders[name] = intKeys(forms)
		}
		for _, sc := range e.Scales {
			one := "~"
			if sc.One != nil {
				one = *sc.One
			}
			g.Scales = append(g.Scales, Scale{
				Value: sc.Value, Forms: sc.Forms, OneWord: one,
				Separate: sc.Separate, Link: sc.Link, SmallJoiner: sc.SmallJoiner,
				MultiplierAgrees: sc.MultiplierAgrees,
				MultiplierGender: sc.MultiplierGender,
			})
		}
		grammars[lang] = g
	}
}

// SupportedNumberLanguages lists the language ids Cardinal can verbalize —
// the roster in numbers.json, and the allowlist the text frontend enforces.
//
// The nil check is not decoration: grammars is loaded lazily by Cardinal, so
// without it this returns an empty slice rather than the roster, and
// an empty allowlist refuses every language there is.
func SupportedNumberLanguages() []string {
	if grammars == nil {
		loadGrammars()
	}
	out := make([]string, 0, len(grammars))
	for lang := range grammars {
		out = append(out, lang)
	}
	sort.Strings(out)
	return out
}

// gendered returns the form value takes in gender at the given position, or
// "" when it does not inflect. Position is "standalone" (the whole number),
// "tail" (ends a larger number) or "tens_pair" (inside the solid compound).
func (g *Grammar) gendered(value int64, gender, position string) string {
	if gender == "" {
		return ""
	}
	scope := g.GenderScopes[value]
	if scope == "standalone" && position != "standalone" {
		return ""
	}
	if scope == "outside_tens" && position == "tens_pair" {
		return ""
	}
	return g.Genders[gender][value]
}

// Cardinal says value as words. Gender "" gives the citation form; an unknown
// language or a value past the grammar's largest scale is an error — silently
// reading digits back would be indistinguishable from success.
func Cardinal(value int64, language, gender string) (string, error) {
	if grammars == nil {
		loadGrammars()
	}
	g, ok := grammars[language]
	if !ok {
		return "", fmt.Errorf("no number grammar for %q", language)
	}
	ceiling := int64(1000)
	if len(g.Scales) > 0 {
		ceiling = g.Scales[0].Value * 1000
	}
	abs := value
	if abs < 0 {
		abs = -abs
	}
	if abs >= ceiling {
		return "", fmt.Errorf("%d is past the largest scale %q has a word for", value, language)
	}
	if value < 0 {
		rest, err := Cardinal(-value, language, gender)
		if err != nil {
			return "", err
		}
		// Always a spaced word, even in solid-writing languages: minus eins.
		return g.MinusWord + " " + rest, nil
	}
	// Standalone agreement applies to the whole number only: Polish jedna
	// alone, but sto jeden.
	if w := g.gendered(value, gender, "standalone"); w != "" {
		return w, nil
	}
	return compose(value, g, gender, false), nil
}

func compose(value int64, g *Grammar, gender string, asMultiplier bool) string {
	if w, ok := g.Exceptions[value]; ok {
		return w
	}
	if value < 100 {
		return belowHundred(value, g, gender, asMultiplier)
	}
	for _, sc := range g.Scales {
		if value >= sc.Value {
			return scaleGroup(value, sc, g, gender)
		}
	}
	return hundredsGroup(value, g, gender)
}

func scaleGroup(value int64, sc Scale, g *Grammar, gender string) string {
	count, rest := value/sc.Value, value%sc.Value
	join := g.WordJoin
	if sc.Separate {
		join = " "
	}
	linkDefault := sc.Link
	if linkDefault == "" {
		linkDefault = join
	}

	var head string
	if count == 1 && sc.OneWord != "~" {
		if sc.OneWord == "" {
			head = scaleWord(1, sc.Forms)
		} else {
			head = sc.OneWord + join + scaleWord(1, sc.Forms)
		}
	} else {
		// Whether the counted noun's gender reaches the multiplier is a fact
		// about the scale noun: Portuguese "duas mil", Polish "dwa tysiące".
		mg := ""
		if sc.MultiplierGender != "" {
			mg = sc.MultiplierGender
		} else if sc.MultiplierAgrees {
			mg = gender
		}
		head = compose(count, g, mg, true) + join + scaleWord(count, sc.Forms)
	}
	if rest == 0 {
		return head
	}

	roundHundreds := g.ScaleJoinerOnRoundHundreds && rest >= 100 && rest%100 == 0
	var link string
	switch {
	case sc.SmallJoiner != "" && (rest < 100 || roundHundreds):
		link = " " + sc.SmallJoiner + " "
		if join == "" {
			link = " " + sc.SmallJoiner + " "
		}
	case rest >= 100 && count >= 100 && g.ScaleLargeJoiner != "":
		link = g.ScaleLargeJoiner
	default:
		link = linkDefault
	}
	return head + link + compose(rest, g, gender, false)
}

func scaleWord(count int64, forms []string) string {
	if len(forms) == 1 || count == 1 {
		return forms[0]
	}
	if len(forms) == 2 { // singular / plural: Million / Millionen
		return forms[1]
	}
	lastTwo, last := count%100, count%10
	if last >= 2 && last <= 4 && !(lastTwo >= 12 && lastTwo <= 14) {
		return forms[1]
	}
	return forms[2]
}

func hundredsGroup(value int64, g *Grammar, gender string) string {
	count, rest := value/100, value%100
	var parts []string
	hundreds := g.Hundreds
	if gender != "" {
		if forms, ok := g.HundredsGendered[gender]; ok {
			hundreds = forms
		}
	}
	switch {
	case len(hundreds) > 0:
		parts = append(parts, hundreds[count-1])
	case count == 1 && !g.OneBeforeHundred:
		parts = append(parts, g.Hundred)
	default:
		parts = append(parts, compose(count, g, "", true))
		// French deux cents / deux cent un: the plural mark appears only when
		// the multiplied hundred ends the number.
		if count > 1 && rest == 0 && g.HundredPluralFinal != "" {
			parts = append(parts, g.HundredPluralFinal)
		} else {
			parts = append(parts, g.Hundred)
		}
	}
	if rest != 0 {
		if g.HundredJoiner != "" {
			parts = append(parts, g.HundredJoiner)
		}
		parts = append(parts, belowHundred(rest, g, gender, false))
	}
	nonEmpty := parts[:0]
	for _, p := range parts {
		if p != "" {
			nonEmpty = append(nonEmpty, p)
		}
	}
	return strings.Join(nonEmpty, g.WordJoin)
}

func unitWordFor(value int64, g *Grammar, gender string, asMultiplier bool) string {
	position := "tail"
	if asMultiplier {
		position = "tens_pair"
	}
	if w := g.gendered(value, gender, position); w != "" {
		return w
	}
	if asMultiplier {
		if w, ok := g.CombiningOnes[value]; ok {
			return w
		}
	}
	return g.Ones[value]
}

func belowHundred(value int64, g *Grammar, gender string, asMultiplier bool) string {
	if w := g.gendered(value, gender, "tail"); w != "" {
		return w
	}
	if w, ok := g.Exceptions[value]; ok {
		return w
	}
	if value < 10 {
		return unitWordFor(value, g, gender, asMultiplier)
	}
	if value < 20 {
		return g.Teens[value-10]
	}
	ten, unit := value/10, value%10
	tenWord := g.gendered(ten*10, gender, "tail")
	if tenWord == "" {
		tenWord = g.Tens[ten-2]
	}
	if unit == 0 {
		return tenWord
	}
	// A unit inside a tens pair is always in composition: einundzwanzig holds
	// even when the pair ends the number.
	unitWord := unitWordFor(unit, g, gender, true)
	joiner := g.UnitTensJoiner
	if override, ok := g.TensJoinerExceptions[value]; ok {
		joiner = override
	}
	if g.UnitsBeforeTens {
		return unitWord + joiner + tenWord
	}
	return tenWord + joiner + unitWord
}

// ASCII digits only, explicitly — see the Python module for why.
// Python's `_DIGIT_RUN`, minus the lookbehind RE2 cannot express — that guard
// is applied in ExpandNumbers against the character before the match.
//
// The three parts of the pattern are each audible: a run glued to a
// word is part of that word (`iOS18` reads as *iOSeighteen*), a minus in front
// of digits belongs to the number (`-5` reads as *five*), and space-grouped
// thousands are one number (`1 000` reads as *one zero zero zero*).
var digitRunRe = regexp.MustCompile(`([0-9]{1,3}(?: [0-9]{3})+|[0-9]+)((?:[.,][0-9]+)*)`)

// phoneRunRe is Python's `_PHONE_RUN`: an E.164 number — a plus, then digits,
// possibly grouped by spaces — read digit by digit and taken before the digit
// run, which cannot decline it. "+48 123 456 789" is a valid
// one-to-three-then-threes grouping, so it was read as *forty-eight billion*.
// The plus is the evidence: E.164 requires one and a grouped thousand never has
// one.
var phoneRunRe = regexp.MustCompile(`\+[0-9][0-9 ]*[0-9]`)

// minE164Digits keeps the rule above away from a signed quantity: "+5 degrees"
// and "+1 000 000 users" are deltas and millions, not numbers to spell out.
// ISO 8601's 24:00. Admitted as an hour, and only with a zero minute.
const endOfDayHour = 24

const minE164Digits = 8

// unicodeMinusRe folds U+2212 MINUS SIGN and U+2010 HYPHEN to ASCII where a
// digit follows. Everything downstream reads the sign as `-`, so a
// typographically correct minus was not a sign at all: it reached the
// punctuation pass, became a space, and "−5" was read as *five*. Not U+2013,
// which writes a range, and not U+2014, which is punctuation.
var unicodeMinusRe = regexp.MustCompile(`[\x{2212}\x{2010}]([0-9])`)

func expandPhoneNumbers(text, language string, g *Grammar) string {
	return phoneRunRe.ReplaceAllStringFunc(text, func(match string) string {
		digits := make([]rune, 0, len(match))
		for _, r := range match {
			if r >= '0' && r <= '9' {
				digits = append(digits, r)
			}
		}
		if len(digits) < minE164Digits {
			return match
		}
		said := make([]string, 0, len(digits))
		for _, d := range digits {
			word, err := Cardinal(int64(d-'0'), language, "")
			if err != nil {
				return match
			}
			said = append(said, word)
		}
		_ = g
		return strings.Join(said, " ")
	})
}

// decimalSeparator is the mark `language` writes between a whole number and
// its fraction, defaulting to "." for a language with no grammar.
func decimalSeparator(language string) string {
	if grammars == nil {
		loadGrammars()
	}
	if g, ok := grammars[language]; ok {
		return g.DecimalSeparator
	}
	return "."
}

// FoldForeignDigits rewrites Arabic-Indic and Eastern Arabic-Indic digits, and
// their separators, as this language spells them.
//
// Foreign digit systems and their separators, as this language spells them.
// Beside NFC because it is the same kind of pass: one spelling for every pass
// that follows, and early enough that the symbol table still sees the folded
// percent sign.
//
// Language-dependent for the separators, and that is not a detail. U+066B is a
// *decimal* separator, so folding it to a dot everywhere turned "٣٫١٤" into
// "3.14" — which in the eleven languages that write decimals with a comma is the
// written form of a clock time, read out as *drei Uhr vierzehn*.
func FoldForeignDigits(text, language string) string {
	if grammars == nil {
		loadGrammars()
	}
	decimal := "."
	if g, ok := grammars[language]; ok {
		decimal = g.DecimalSeparator
	}
	grouping := "."
	if decimal == "." {
		grouping = ","
	}
	var b strings.Builder
	b.Grow(len(text))
	for _, r := range text {
		switch {
		case r >= 0x0660 && r <= 0x0669:
			b.WriteRune(rune('0' + (r - 0x0660)))
		case r >= 0x06F0 && r <= 0x06F9:
			b.WriteRune(rune('0' + (r - 0x06F0)))
		case r == 0x066B:
			b.WriteString(decimal)
		case r == 0x066C:
			b.WriteString(grouping)
		case r == 0x066A:
			b.WriteRune('%')
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}

// ExpandNumbers says every run of digits in text as words — the seam between
// the verbalizer and the funnel. It never errors and never leaves digits
// behind: a number past every scale is read digit by digit, which is what such
// a number almost always is — an identifier. A separator between digits is a
// decimal mark only when it is the language's own; the other mark is a
// grouping mark and is dropped, which is what a reader does with it.
func ExpandNumbers(text, language string) string {
	if grammars == nil {
		loadGrammars()
	}
	g, ok := grammars[language]
	if !ok {
		return text
	}
	// Both before anything looks for a digit run: the sign has to be ASCII by
	// the time the pattern matches one, and a phone number has to be gone
	// before the grouping rule sees a shape it cannot decline.
	text = unicodeMinusRe.ReplaceAllString(text, "-$1")
	text = expandPhoneNumbers(text, language, g)
	var b strings.Builder
	// `cursor` is how far the output has been written, `pos` how far the scan
	// has read. They are two variables rather than one because a refused match
	// is not always a consumed one: Python's lookbehind is part of the pattern,
	// so a failure there makes its engine retry one character to the right and
	// find a *shorter* run inside the same region. Scanning `FindAll`'s results
	// instead skipped the whole region: in `e3 1000` the pattern binds `3 100`,
	// the lookbehind refuses it for the `e`, and the `1000` Python reads as
	// *ettusen* went unspoken. The ragged branch moves `pos` the other way, past
	// several of the matches `FindAll` would have handed back one at a time.
	cursor, pos := 0, 0
	for pos < len(text) {
		loc := digitRunRe.FindStringSubmatchIndex(text[pos:])
		if loc == nil {
			break
		}
		for k, v := range loc {
			if v >= 0 {
				loc[k] = v + pos
			}
		}
		start, end := loc[0], loc[1]
		// The lookbehind, in code: a run glued to a word is part of that word,
		// so `iOS18` stays written. The sign is read backwards rather than
		// captured, because RE2 does not retry a failed match one position to
		// the right the way Python's engine does — a captured `-?` swallows
		// the hyphen in `1-5` and leaves the `5` unspoken.
		sign := false
		if start > 0 && text[start-1] == '-' && wordBoundaryBefore(text, start-1) {
			sign = true
			start--
		} else if !wordBoundaryBefore(text, start) {
			pos = start + 1
			continue
		}
		// Python's `(?! ?[0-9])`, in code: a space-grouped run is a grouped
		// number only if it *reaches a boundary*. RE2 has no lookahead and,
		// unlike Python's engine, does not retry the alternation one branch
		// down — so where Python backtracks to reading each segment on its
		// own, this takes the longest prefix that fits and abandons the rest:
		// "1 202 555 0199" matches "1 202 555 019" and is read as a
		// ten-digit cardinal with a bare "9" trailing behind it.
		//
		// Judged at the end of the whole-number group, not at the end of the
		// match: Python asks it where the grouped alternative stops, which is
		// before the fraction. Asking it behind the fraction instead made
		// `1 000.0 3` ragged — a digit does follow the `.0` — and read the
		// grouped thousand as three separate zeros where every other port says
		// *duizend komma nul drie*.
		ragged := strings.Contains(text[loc[2]:loc[3]], " ") && digitsFollow(text, loc[3])
		// A ragged run is read, and consumed, as its first group alone. That is
		// the match Python's engine ends up with — the grouped alternative is
		// refused outright, the `[0-9]+` fallback takes the digits in front of
		// the first space — and everything after it is re-matched from there,
		// which is why the tail can come back as a number in its own right:
		// `1 000 1 234 567 1 234 567` is *un*, *zéro zéro zéro*, four more
		// segments, and then a grouped million, because that last run does reach
		// a boundary. Reading the whole run segment-wise in one pass said
		// *un deux cent trente-quatre …* for a number that is not ragged at all,
		// and consuming the whole binding on a refusal lost the thousand in
		// `1e+3 1000` — the space in front of four digits never grouped, so the
		// `e` has nothing to do with it.
		wholeEnd, readEnd := loc[3], end
		if ragged {
			wholeEnd = loc[2] + strings.IndexByte(text[loc[2]:loc[3]], ' ')
			readEnd = wholeEnd
		}
		pos = readEnd
		// The glue checks are asked of the *whole* binding, not of the narrowed
		// reading, and that ordering is the refusal rule: a maximal run of
		// digits and separators touching a word is left written rather than
		// split into segments and half spoken. `1 234 567.é` stays written here
		// and reads as a cardinal in the engines that backtrack.
		if gluedToAWord(text, start) || gluedForward(text, end) ||
			truncatedByAFraction(text, end) {
			continue
		}
		// Python's `(?![\w])` and its backward walk, in code.
		//
		// The lookbehind has no mirror: a run glued to a word on the left is
		// left alone, a run glued to one on the right is expanded up to the
		// letter and then abandoned. "5x3" comes out *fivex3* and "1e6" comes
		// out *onee6* — a word welded to a digit, which is not a reading of
		// anything. And the lookbehind sees one character, so an identifier
		// that puts a dot between its letter and its digits slips past it:
		// in "v1.2.3" the scan starts at the `2` and the version comes out
		// "v1.two point three".
		//
		// Asked where the reading ends: a ragged run ends at a space, which is
		// a boundary, and the digits behind that space are the next match's
		// business.
		if readEnd < len(text) && !wordBoundaryBefore(text, readEnd+1) {
			continue
		}
		digits := strings.ReplaceAll(text[loc[2]:wholeEnd], " ", "")
		fraction := ""
		// A ragged run has a digit where the fraction group would start, so the
		// group is empty and there is nothing to carry; a run that is not ragged
		// keeps its fraction, and `4 5672.5` ends *setenta y dos coma cinco*.
		if loc[4] >= 0 && !ragged {
			fraction = text[loc[4]:loc[5]]
		}
		literal := digits + fraction
		if !isQuantity(literal, g) {
			continue
		}
		b.WriteString(text[cursor:start])
		said := sayNumber(literal, g, language)
		if sign && g.MinusWord != "" {
			said = g.MinusWord + " " + said
		}
		b.WriteString(said)
		cursor = readEnd
	}
	b.WriteString(text[cursor:])
	return b.String()
}

// gluedForward reports whether the token continues past the match into a
// letter.
//
// The mirror of `gluedToAWord`, and it was missing here while Python, JS and
// Swift had it: `123.de` is one token to them and two to this port, which read
// "einhundertdreiundzwanzig.de". A grouping space is crossed so `200 000x` is one
// token; the ordinary space in `2024 200 people` is not, because what follows it
// is a word.
func gluedForward(text string, end int) bool {
	i := end
	for i < len(text) {
		// Decoded as a rune, not read as a byte. `é` is two bytes and neither
		// of them is an ASCII letter, so a byte-wise test walks straight past
		// it and reads the `1 234 567` in `1 234 567.é` that Python refuses —
		// the kind of input the parity fuzzer generates, because the funnel is
		// meant for nine languages with accents in them.
		r, width := utf8.DecodeRuneInString(text[i:])
		if unicode.IsLetter(r) {
			return true
		}
		if r == '_' || r == '.' || r == ',' || r == '-' || r == '+' || unicode.IsDigit(r) {
			i += width
			continue
		}
		// A thousands space: a digit in front of it and a group behind it, or the
		// walk leaves one number and enters the next — `1000 5.1e+3` found the
		// `e` two tokens away and called the whole line one glued token.
		//
		// `startsAGroup` here and `continuesAGroup` backwards; see the two for
		// why the question is the looser one in this direction only. The indices
		// are the ones this direction needs: written with the backward walk's —
		// a digit at `i-2` and the group at `i` — the test asked whether a space
		// was a digit, was false at every space, and let `2024 200x` read as
		// *duemilaventiquattro 200x*: half a token spoken, which is the class
		// this guard exists to stop.
		if r == ' ' && i > 0 && isASCIIDigit(text[i-1]) && startsAGroup(text, i+1) {
			i += width
			continue
		}
		return false
	}
	return false
}

// truncatedByAFraction reports whether a decimal point with digits behind it
// follows the match.
//
// A decimal point with digits behind the match means the fraction group shrank
// to zero so the right-hand guard could land on the dot instead of a letter:
// `1.5e3` matched just the `1` and read "one.5e3". A number that really ends
// here has nothing of the sort behind it.
func truncatedByAFraction(text string, end int) bool {
	return end+1 < len(text) && (text[end] == '.' || text[end] == ',') &&
		isASCIIDigit(text[end+1])
}

// gluedToAWord reports whether the digit run at `start` sits inside a token
// containing a letter — Python's backward walk over word characters and dots,
// which is the question its one-character lookbehind could not ask.
func gluedToAWord(text string, start int) bool {
	i := start
	for i > 0 {
		// Decoded backwards as a rune, not read as a byte. `ł` is two bytes and
		// neither is an ASCII letter, so a byte-wise test walked past it and
		// this port read the `2,50` in `zł2,50` that Python refuses.
		//
		// `-` and `+` are in the walk because an exponent puts one between the
		// letter and the digits: in `1e-3` the scan starts at the `3`, walks
		// back over `-` to `e`, and stops calling it a number. A bare `-5` is
		// unaffected — the walk reaches a space and finds no letter.
		r, width := utf8.DecodeLastRuneInString(text[:i])
		switch {
		case r == '_' || r == '.' || r == ',' || r == '-' || r == '+' ||
			unicode.IsDigit(r) || unicode.IsLetter(r):
			i -= width
			if unicode.IsLetter(r) {
				return true
			}
		// A thousands space is crossed here too, and its absence was a parity
		// break of the kind the walk exists to stop. `C0200 000` binds as one
		// match in this port, the lookbehind refuses it for the `C`, and the
		// scan then finds the standalone `000` with nothing but a space behind
		// it — "C0200 zero zero zero", half a token spoken, where Python, JS and
		// Swift leave the whole thing written.
		//
		// The group being stepped out of is the one whose width the pattern
		// fixes, so it is the half tested; a digit in front of the space as
		// well, or the walk crosses space after space and `Sold 200 000` goes
		// from `000` to `200` and on into "Sold".
		case r == ' ' && i >= 2 && isASCIIDigit(text[i-2]) && continuesAGroup(text, i):
			i -= width
		default:
			return false
		}
	}
	return false
}

// startsAGroup reports whether a thousands group's worth of digits begins at
// `i`: three of them, with nothing said about a fourth. The forward walk's half
// of the question and the looser half, because forwards the walk finishes a run
// the pattern *refused* to bind and a ragged group is exactly why it refused —
// `1 0023R` binds as `1 002` in an engine that does not backtrack, and a walk
// that stopped at the ragged group read the `1` and left `0023R` written: half a
// run spoken with the rest welded to a letter.
//
// Three digits and not fewer, so the walk still stops where the run stops: the
// `5` of `R2 5 iOS` is its own number.
func startsAGroup(text string, i int) bool {
	if i+groupDigits > len(text) {
		return false
	}
	for k := i; k < i+groupDigits; k++ {
		if !isASCIIDigit(text[k]) {
			return false
		}
	}
	return true
}

// continuesAGroup reports whether the run at `i` is exactly a thousands group:
// three digits and no fourth, the shape every group after the first has in the
// digit-run pattern.
//
// The backward walk's half, and the strict one, because backwards the group *is*
// the match and the pattern already fixed its width. The fourth-digit clause is
// what keeps `e3 1000` readable — four digits behind the space are not a group,
// so the space never grouped, so the thousand is a token of its own with nothing
// glued to the `e`. Measured on the fuzzer: the loose question in both
// directions changes 60 readings and 56 of them are losses; asked forwards only,
// the losses are gone.
func continuesAGroup(text string, i int) bool {
	return startsAGroup(text, i) &&
		(i+groupDigits >= len(text) || !isASCIIDigit(text[i+groupDigits]))
}

// groupDigits is the digits in a thousands group: every group after the first is
// exactly this many.
const groupDigits = 3

func isASCIIDigit(c byte) bool { return c >= '0' && c <= '9' }

// digitsFollow reports whether a digit sits at `i`, or at `i+1` behind a space.
// The two shapes Python's `(?! ?[0-9])` rejects.
func digitsFollow(text string, i int) bool {
	if i < len(text) && isASCIIDigit(text[i]) {
		return true
	}
	return i+1 < len(text) && text[i] == ' ' && isASCIIDigit(text[i+1])
}

// isQuantity reports whether a digit run is a number rather than a version, an
// address or a date. See the Python module: `1.2.3`, `192.168.0.1` and
// `12.03.2026` all match the digit-run pattern and none of them is a quantity.
// Reading them as one produced "nineteen million two hundred sixteen thousand
// eight hundred one" for an IP address, and in Python a hard crash.
//
// A run is a quantity when it has at most one separator, or when its separators
// genuinely group: every segment after the first exactly three digits, the first
// one to three. Anything else is left as written.
func isQuantity(literal string, g *Grammar) bool {
	grouping := "."
	if g.DecimalSeparator == "." {
		grouping = ","
	}
	whole, fraction, hasFraction := strings.Cut(literal, g.DecimalSeparator)
	// A second mark in what should be the fraction: Cut splits once, so this is
	// where "1.2.3" left "2.3" and the crash began.
	if strings.Contains(fraction, grouping) || strings.Contains(fraction, g.DecimalSeparator) {
		return false
	}
	segments := strings.Split(whole, grouping)
	if len(segments) == 1 {
		return true
	}
	grouped := len(segments[0]) >= 1 && len(segments[0]) <= 3
	for _, seg := range segments[1:] {
		if len(seg) != 3 {
			grouped = false
			break
		}
	}
	if grouped {
		return true
	}
	// Two segments and no fraction is the "2.5 GB" shape: the mark that is not
	// this language's decimal separator, used as one anyway.
	return len(segments) == 2 && !hasFraction
}

func sayNumber(literal string, g *Grammar, language string) string {
	// The non-decimal mark is only grouping when it groups: every following
	// segment exactly three digits. Polish "1.000" is a thousand; Polish "2.5"
	// is a de-facto decimal, and 2.5 read as 25 is a changed meaning.
	grouping := "."
	if g.DecimalSeparator == "." {
		grouping = ","
	}
	whole, fraction, hasFraction := strings.Cut(literal, g.DecimalSeparator)
	segments := strings.Split(whole, grouping)
	if len(segments) > 1 {
		allThrees := true
		for _, seg := range segments[1:] {
			if len(seg) != 3 {
				allThrees = false
				break
			}
		}
		switch {
		case allThrees:
			whole = strings.Join(segments, "")
		case !hasFraction && len(segments) == 2:
			whole, fraction, hasFraction = segments[0], segments[1], true
		default:
			whole = strings.Join(segments, "")
		}
	}
	fraction = strings.ReplaceAll(fraction, grouping, "")

	parts := []string{sayInteger(whole, language)}
	if hasFraction && fraction != "" {
		parts = append(parts, g.DecimalWord)
		// Digit by digit — "point four nine", never "point forty-nine":
		// leading zeros carry meaning there that a cardinal would eat.
		parts = append(parts, digitByDigit(fraction, language)...)
	}
	return strings.Join(parts, " ")
}

func sayInteger(digits, language string) string {
	// Leading zeros mean a code, not a quantity: 0042 is zero zero four two.
	if len(digits) > 1 && digits[0] == '0' {
		return strings.Join(digitByDigit(digits, language), " ")
	}
	n, err := strconv.ParseInt(digits, 10, 64)
	if err == nil {
		if said, cerr := Cardinal(n, language, ""); cerr == nil {
			return said
		}
	}
	return strings.Join(digitByDigit(digits, language), " ")
}

func digitByDigit(digits, language string) []string {
	out := make([]string, 0, len(digits))
	for _, ch := range digits {
		said, err := Cardinal(int64(ch-'0'), language, "")
		if err != nil {
			return []string{digits} // unreachable for 0-9; belt and braces
		}
		out = append(out, said)
	}
	return out
}

// No `\b`: Python guards this with `(?<![\d.,:]) … (?![.,:]?\d)`, which rejects
// a digit or separator either side and says nothing about letters. `\b` fires
// between a letter and a digit too, so `a14:30` matched in Python and not here.
// Both guards live in the neighbour check below.
var timeRunRe = regexp.MustCompile(`([01]?[0-9]|2[0-4])[:.]([0-5][0-9])`)

// ExpandTimes reads clock times as words — see the Python module for the
// shape and the deliberate absence of the colloquial clock.
func ExpandTimes(text, language string) string {
	if grammars == nil {
		loadGrammars()
	}
	g, ok := grammars[language]
	if !ok {
		return text
	}
	// Rebuilt by index rather than with ReplaceAllStringFunc, because whether a
	// match is a time depends on what sits *outside* it and RE2 has no
	// lookaround. `12.03` matches inside `12.03.2026` — the ordinary written
	// date of German, Polish, Danish, Finnish and Norwegian — and must not be
	// read as twelve o'clock three with the year trailing behind it. A time is
	// a time only when nothing is attached to either end.
	matches := timeRunRe.FindAllStringSubmatchIndex(text, -1)
	if matches == nil {
		return text
	}
	var out strings.Builder
	last := 0
	for _, m := range matches {
		start, end := m[0], m[1]
		if attachedToDigits(text, start, end) {
			continue
		}
		// A dot between an hour and two minutes is a clock time in some of
		// these languages and a decimal point in others, and the grammar file
		// already says which: a language that writes 14.30 for half past two
		// does not use the dot as its decimal mark. German writes "14.30 Uhr"
		// and "2,50 €"; English writes "2:30" and "$2.50". Without this, every
		// English decimal with two fraction digits was read as the clock —
		// "$0.49" as *zero forty-nine*, "3.14" as *three fourteen* — and the
		// shared fixture pinned one of them, so all five agreed on it.
		if text[m[3]] == '.' && g.DecimalSeparator == "." {
			continue
		}
		hour, _ := strconv.ParseInt(text[m[2]:m[3]], 10, 64)
		minute, _ := strconv.ParseInt(text[m[4]:m[5]], 10, 64)
		// 24 is admitted only with a zero minute: ISO 8601 writes end-of-day
		// as 24:00, and without it the two halves were read as unrelated
		// numbers with the colon left standing between them. 24:30 is not a
		// time in any convention and stays as written.
		if hour == endOfDayHour && minute != 0 {
			continue
		}
		words := []string{}
		if said, err := Cardinal(hour, language, ""); err == nil {
			words = append(words, said)
		}
		if g.TimeInfix != "" {
			words = append(words, g.TimeInfix)
		}
		if minute != 0 {
			if said, err := Cardinal(minute, language, ""); err == nil {
				words = append(words, said)
			}
		}
		// German is the only grammar here with a written infix; asked with an
		// empty one the scan matches the empty string wherever the whitespace
		// run ends and eats the whitespace with it. `3.14 é` in Portuguese came
		// out *três catorzeé* — two words welded — because the guard against
		// that is a letter test on one ASCII byte and `é` is two.
		if g.TimeInfix != "" {
			end = consumeWrittenInfix(text, end, g.TimeInfix)
		}
		out.WriteString(text[last:start])
		out.WriteString(strings.Join(words, " "))
		last = end
	}
	out.WriteString(text[last:])
	return out.String()
}

// consumeWrittenInfix extends end past a written infix word — German writes
// "um 14.30 Uhr", and the spoken reading already puts the infix where it
// belongs, between hour and minutes (*vierzehn Uhr dreißig*). Leaving the
// written word standing said it twice. Consumed only when it is a whole word
// immediately after the time; *Uhrzeit* keeps its head.
//
// ASCII byte scan throughout: space and tab are single bytes in UTF-8 and a
// letter or digit touching the infix is detected by range, so this matches
// the other four implementations exactly.
func consumeWrittenInfix(text string, end int, infix string) int {
	i := end
	for i < len(text) && (text[i] == ' ' || text[i] == '\t') {
		i++
	}
	if i == end || i+len(infix) > len(text) || text[i:i+len(infix)] != infix {
		return end
	}
	if after := i + len(infix); after < len(text) {
		if c := text[after]; ('0' <= c && c <= '9') || ('A' <= c && c <= 'Z') || ('a' <= c && c <= 'z') {
			return end
		}
		return after
	}
	return i + len(infix)
}

// attachedToDigits reports whether text[start:end] has a digit or a separator
// touching either end — the test that tells `14:30` from the `12.03` inside a
// date. A trailing sentence period is fine: what follows it is not a digit.
func attachedToDigits(text string, start, end int) bool {
	if start > 0 {
		switch text[start-1] {
		case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', ',', ':':
			return true
		}
	}
	if end < len(text) {
		switch text[end] {
		case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
			return true
		case '.', ',', ':':
			// A separator only disqualifies when a digit follows it, so
			// "at 14:30." keeps reading while "12.03.2026" does not.
			if end+1 < len(text) && text[end+1] >= '0' && text[end+1] <= '9' {
				return true
			}
		}
	}
	return false
}

// ExpandAbbreviations writes out the authority-listed abbreviations, longest
// first, at word boundaries only — see the Python module.
func ExpandAbbreviations(text, language string) string {
	if grammars == nil {
		loadGrammars()
	}
	g, ok := grammars[language]
	if !ok || len(g.Abbreviations) == 0 {
		return text
	}
	out := text
	for _, entry := range g.Abbreviations {
		re := regexp.MustCompile(`(^|[^\w.])` + regexp.QuoteMeta(entry.Written) + `($|[^\w.])`)
		out = re.ReplaceAllString(out, "${1}"+entry.Spoken+"${2}")
	}
	return out
}
