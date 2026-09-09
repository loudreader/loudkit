// Package speechtext is the language-agnostic speech funnel: a bit-parity
// port of the Swift engine's SpeechText and the JS/Python funnels.
//
// Before tokenising, the shipped engine scrubs the raw text: invisible
// characters, symbols that carry meaning, footnote markers, and punctuation
// (prosodic marks stay exactly where they are, because the model is a
// language model trained on punctuated text; everything else becomes a
// space). Applied by
// the engine's Encode path, mirroring Engine._synthesize_one in Python and
// Engine.encode in JS.
//
// The Polish English-respelling lexicon is ported too: see respell.go, which
// embeds the generated dictionary and is wired into Prepared below.
//
// Python reference: loudkit/frontend/speechtext.py.
package speechtext

import (
	_ "embed"
	"encoding/json"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

var invisibles = map[rune]bool{
	'\u200B': true, '\u200C': true, '\u200D': true, '\u2060': true,
	'\uFEFF': true, '\u00AD': true, '\u180E': true, '\u200E': true,
	'\u200F': true,
}

// symbolRule is one replacement, in the order Python applies it.
//
// A slice, not a map: these rules are applied in sequence to the same string,
// so the order is part of the output. Go randomises map iteration, which would
// make text normalisation vary run to run for one input: the opposite of what
// this port exists to guarantee. Python's dict and JS's object both preserve
// insertion order, so the reference order is the literal order below and it
// must stay in step with `_SPOKEN_SYMBOLS` in loudkit.frontend.speechtext.
//
// Which word a symbol takes is a per-language fact and comes from `unit_words`
// in the shared grammar; `symbolMarks` carries the rest.
var spokenSymbols = []rune{
	'%', '°', '¢', '€', '£', '¥', '₹',
	'×', '÷', '≈', '≥', '≤', '≠', '±',
	'→', '←', '⇒', '✓', '✔', '✗', '✘',
	'•', '·', '▪', '◦', '…', '&', '@',
}

// An arrow, a bullet and an ellipsis are the same pause in every language, so
// they are a rule here rather than a row in twelve grammars.
var symbolMarks = map[rune]string{
	'→': ",",
	'←': ",",
	'⇒': ",",
	'•': ",",
	'·': ",",
	'▪': ",",
	'◦': ",",
	'…': "...",
}

// asciiOperators are the ASCII spellings of the comparison operators, longest
// first, each named by the mathematical symbol whose word it shares.
//
// Only these six: `-`, `/`, `.` and `+` are ranges, paths, decimals and hyphens
// far more often than operators, and a word put on one of them changes prose
// that reads correctly today.
var asciiOperators = []struct{ written, symbol string }{
	{"<=", "≤"},
	{">=", "≥"},
	{"!=", "≠"},
	{"==", "="},
	{"<", "<"},
	{">", ">"},
}

// operatorRunRe finds the spellings above. Whitespace on both sides is the
// evidence that the mark is an operator and not markup or an emoticon, and
// Python asserts it with lookaround: `<p>`, `</div>`, `<3` and `a<b` all keep
// the mark written. RE2 has neither lookbehind nor lookahead, so speakOperators
// reads the characters either side itself and the whitespace stays where it is.
var operatorRunRe = regexp.MustCompile(`<=|>=|!=|==|<|>`)

// markupTagRe matches a markup tag, comment or declaration, which is not text
// anyone reads aloud.
//
// The name inside the angle brackets otherwise reaches the model as a word, and
// `<!-- ... -->` additionally leaves its `!` behind as a sentence-final
// exclamation. A tag becomes a space rather than nothing, because two block
// tags meeting back to back are two paragraphs and not one glued word.
var markupTagRe = regexp.MustCompile(`</?[A-Za-z!][^<>]*>`)

// `$` and `£` before a number read as a prefix in writing and a SUFFIX in
// speech: "$5" is "five dollars", not "dollars five".
// The wording comes from unitWords (numbers.json); this list only says which
// symbols are written prefix. Ordered: these run in sequence over one string.
var currencyPrefixes = []rune{'$', '£', '€', '¥', '₹'}

// currencySymbols also carries `¢`, which nobody writes in front of a number,
// it is a suffix in every convention, which is why the prefix pass never saw it
// and "0.49¢" reached the clock reader intact.
var currencySymbols = append(append([]rune{}, currencyPrefixes...), '¢')

var currencySuffixRe = map[rune]*regexp.Regexp{}

// currencyPrefixCache holds one language's prefix-currency patterns, in the
// order currencyPrefixes lists them.
//
// Per language and built lazily, because the optional magnitude a price may
// carry spells out this language's own scale nouns: one table for twelve
// languages stopped being possible when `$5 million` had to read as five
// million dollars. A sync.Map rather than a plain one, because a server reaches
// Prepared from many goroutines at once and a plain map would be a concurrent
// write.
var currencyPrefixCache sync.Map // language -> []*regexp.Regexp

// currencyPrefixPatterns compiles this language's five prefix patterns once.
//
// A letter in front means a multi-character currency mark.
//
// `R$` is the Brazilian real, `HK$` the Hong Kong dollar, `NT$` the Taiwan
// dollar, and this table has a wording for none of them. Matching the `$` alone
// read `R$3,14` as "R3,14 Dollar": the wrong currency, said confidently. The mark
// itself is still dropped by the punctuation pass, so the amount reads as a plain
// decimal; losing a symbol is a smaller lie than naming the wrong money.
//
// RE2 has no lookbehind, and capturing the letter instead of looking
// at it consumes it, which breaks the amount after it: in `$1$2$3` the
// second match had to start one character early and swallowed the `2`,
// so two amounts fused into "one dollars twenty-three dollars", a
// number nobody wrote. The pattern matches from the symbol only and
// speakPrefixCurrency reads the character in front of the match itself, which
// is what the other four ports do.
//
// whitespaceClass, not `\s`: RE2's `\s` is `[\t\n\f\r ]` and omits the
// vertical tab, so `£\vE1` matched here and nowhere else: the currency
// rule fired in four ports and not in this one, which then spoke the
// symbol in place instead of behind its amount.
func currencyPrefixPatterns(language string) []*regexp.Regexp {
	if cached, ok := currencyPrefixCache.Load(language); ok {
		return cached.([]*regexp.Regexp)
	}
	scale := scalePattern(language)
	built := make([]*regexp.Regexp, len(currencyPrefixes))
	for i, sym := range currencyPrefixes {
		// The number, and NOT the sentence punctuation behind it: a greedy
		// `[0-9.,]*` would swallow the comma in "£250,".
		pattern := regexp.QuoteMeta(string(sym)) +
			`[` + whitespaceClass + `]?([0-9]+(?:[.,][0-9]+)*)`
		if scale != "" {
			pattern += `(?:` + scale + `)?`
		}
		built[i] = regexp.MustCompile(pattern)
	}
	actual, _ := currencyPrefixCache.LoadOrStore(language, built)
	return actual.([]*regexp.Regexp)
}

// scalePattern is the optional magnitude that may follow a currency amount.
//
// Two alternatives and two groups: the abbreviating letter glued to the digits,
// and the scale noun written beside them. Both cases of each noun are spelled
// out rather than asked of a case-insensitive flag, because JavaScript has no
// inline flag group and a pattern that needs one is a pattern the five
// implementations cannot share.
//
// The `(?![A-Za-z])` Python closes it with lives in speakPrefixCurrency, RE2
// having no lookahead: a letter after the magnitude means `$5kg`, where neither
// alternative holds and the amount is spoken alone.
func scalePattern(language string) string {
	nouns := scaleNouns(language)
	if len(nouns) == 0 {
		return ""
	}
	written := make([]string, 0, 2*len(nouns))
	for _, noun := range nouns {
		written = append(written, regexp.QuoteMeta(noun))
		if capitalized := capitalizeFirst(noun); capitalized != noun {
			written = append(written, regexp.QuoteMeta(capitalized))
		}
	}
	return `(` + scaleSuffixPattern + `)|[` + whitespaceClass + `](` +
		strings.Join(written, "|") + `)`
}

// capitalizeFirst upper-cases the first character and leaves the rest, which is
// how a scale noun is written where a sentence or a headline capitalises it.
func capitalizeFirst(word string) string {
	if word == "" {
		return word
	}
	r, size := utf8.DecodeRuneInString(word)
	return strings.ToUpper(string(r)) + word[size:]
}

//go:embed numbers.json
var numbersJSON []byte

// GrammarBytes is the embedded grammar file exactly as this binary carries it.
// Exported so the fingerprint can hash the bytes this port actually reads,
// hashing a file on disk would say nothing about what got compiled in.
func GrammarBytes() []byte { return numbersJSON }

// dateFields and ordinalFields are the two date sections of one language.
type dateFields struct {
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
}

type ordinalFields struct {
	Suffixes   []string          `json:"suffixes"`
	Units      map[string]string `json:"units"`
	Teens      map[string]string `json:"teens"`
	Tens       map[string]string `json:"tens"`
	TensJoiner string            `json:"tens_joiner"`
}

type scaleFields struct {
	Value            int64    `json:"value"`
	Forms            []string `json:"forms"`
	One              *string  `json:"one"`
	Separate         bool     `json:"separate"`
	Link             string   `json:"link"`
	SmallJoiner      string   `json:"small_joiner"`
	MultiplierAgrees bool     `json:"multiplier_agrees"`
	MultiplierGender string   `json:"multiplier_gender"`
}

// grammarLang is every field this package reads out of one language's entry in
// numbers.json: numbers, dates, ordinals, letters and unit words together.
//
// One type, because the file is parsed once. Four anonymous structs meant four
// passes over the same 43 KB at startup and four places to forget a field when
// the grammar grows.
type grammarLang struct {
	Ones                       []string                     `json:"ones"`
	Teens                      []string                     `json:"teens"`
	Tens                       []string                     `json:"tens"`
	Hundred                    string                       `json:"hundred"`
	Hundreds                   []string                     `json:"hundreds"`
	HundredsGendered           map[string][]string          `json:"hundreds_gendered"`
	HundredPluralFinal         string                       `json:"hundred_plural_final"`
	Scales                     []scaleFields                `json:"scales"`
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
	UnitWords                  map[string]string            `json:"unit_words"`
	LetterNames                map[string]string            `json:"letter_names"`
	WordAcronyms               []string                     `json:"word_acronyms"`
	Dates                      *dateFields                  `json:"dates"`
	Ordinals                   *ordinalFields               `json:"ordinals"`
}

var (
	grammarFileOnce sync.Once
	grammarFile     struct {
		Languages map[string]grammarLang `json:"languages"`
	}
)

// grammarDocument is the parsed grammar, language id to entry.
//
// The four tables below it (numbers, dates, letters, unit words) all read this
// one document. They each keep their own sync.Once for the table they derive,
// because they are derived lazily and independently; the parse underneath them
// happens exactly once.
func grammarDocument() map[string]grammarLang {
	grammarFileOnce.Do(func() {
		if err := json.Unmarshal(numbersJSON, &grammarFile); err != nil {
			panic("speechtext: embedded numbers.json is unreadable: " + err.Error())
		}
	})
	return grammarFile.Languages
}

// unitWords is symbol -> word per language, from the shared grammar file. One
// row per language on the roster, because a symbol is spoken in the language
// being read: "$5" in a German render says "5 Dollar", and `≈` says "ungefähr".
var unitWords map[string]map[string]string

func loadUnitWords() {
	langs := grammarDocument()
	unitWords = make(map[string]map[string]string, len(langs))
	for lang, entry := range langs {
		unitWords[lang] = entry.UnitWords
	}
}

// unitWord returns the word `symbol` takes in `language`, or "" when this
// language has no wording for it. No fall back to English: a symbol is spoken
// in the language being read or it is left written, which is what the funnel
// does everywhere the evidence runs out.
func unitWord(symbol, language string) string {
	return unitWords[language][symbol]
}

// Punctuation that carries prosody stays; the rest becomes a space.
var prosodic = map[rune]bool{}

func init() {
	for _, r := range ".,!?;:\u2014\u2013\u2026\"\u201C\u201D\u201E«»()'\u2019\u00BF\u00A1" {
		prosodic[r] = true
	}
	loadUnitWords()
	for _, sym := range currencySymbols {
		// No letter guard on this side: only whitespace may sit between the
		// amount and the mark, so `3,14 R$` never matches in the first place.
		currencySuffixRe[sym] = regexp.MustCompile(
			`(\d+(?:[.,]\d+)*)[` + whitespaceClass + `]?` + regexp.QuoteMeta(string(sym)))
	}
}

// Prepared scrubs text the way the shipped Swift engine does
// (SpeechText.prepared). Same order, same rules, same output.
func Prepared(text, languageID string) string {
	// The language id is lowercased once here and again in the respeller.
	// GraphemeTextFrontend lowercases its own tag, so "PL" produced Polish
	// *tokens* while silently skipping the Polish respelling: the same
	// utterance read half one way and half the other, with nothing to
	// indicate it. Python fixed this in loudkit.frontend.speechtext.speech_text,
	// and Swift's LexicalRespelling.applied carries the same .lowercased() and
	// the same reason.
	languageID = strings.ToLower(languageID)
	// NFC first, before anything inspects a character: the same opening pass
	// the Python funnel runs, and the one this funnel did not have.
	//
	// Unicode lets the same character arrive two ways: Polish ą as U+0105 or as
	// a + U+0328, Danish å as U+00E5 or a + U+030A. The tokenizer's vocabulary
	// holds one of them, so a decomposed spelling reaches it as a base letter
	// followed by an unknown combining mark, and every rule below, every
	// pattern and lexicon lookup and character class, is matching a string
	// nobody wrote a rule for.
	//
	// Ahead of stripInvisibles, which removes format characters: normalisation
	// can compose a sequence into a single character, and running it afterwards
	// would leave that composition unexamined.
	// Beside NFC, and before the symbol pass so the folded percent sign
	// reaches the table that turns it into a word.
	out := stripInvisibles(FoldForeignDigits(norm.NFC.String(text), languageID))
	// Before the symbol pass, which would otherwise read a tag's angle brackets
	// as comparison operators and its attributes as text.
	out = dropMarkupTags(out)
	// Before the symbol table: see the Python reference. Every pass downstream
	// asks "is this a digit" and the five ports spell it four ways, so folding
	// first means all of them see ASCII.
	out = foldNumerals(out)
	out = speakSymbols(out, languageID)
	out = dropFootnoteMarkers(out)
	// Before the acronym pass, which spells a Roman numeral letter by letter,
	// and after the numeral fold, which is what turns `Ⅳ` into the `IV` this
	// pass reads.
	out = expandRomanNumerals(out, languageID)
	// Acronyms while the capitals are still capitals: every later pass
	// lowercases or rewrites, and a spelled acronym has to be decided while the
	// only evidence (that the word stands alone in caps) still exists. The
	// pass belongs here rather than in respell.go: a Polish-only table there
	// spells FBI ef-be-i in a Polish render and leaves the model raw graphemes
	// in the other eleven.
	out = spellAcronyms(out, languageID)
	// Dates before times and numbers, and this ordering is the whole reason the
	// pass exists: 12.03.2026 is the ordinary written date of five of these
	// languages, and both passes below want a piece of it. The clock pattern
	// matches 12.03 and the digit run matches the lot, so a date recognised any
	// later has already been eaten and read as a time with a stray year.
	out = ExpandDates(out, languageID)
	// Ordinals before numbers, for the same reason: the number pass expands the
	// digits and leaves the suffix stuck to them, so 1st arrived as "onest".
	out = ExpandOrdinals(out, languageID)
	// Numbers after footnotes and before punctuation: see the Python funnel
	// for the ordering argument; the fixture pins it.
	out = ExpandAbbreviations(out, languageID)
	out = ExpandTimes(out, languageID)
	out = ExpandNumbers(out, languageID)
	out = punctuationForSpeech(out)
	// Polish: respell embedded English the way a Polish reader says it. This
	// is the shipped engine's LexicalRespelling; see respell.go.
	out = LexicalRespelling(out, languageID)
	out = spaceRunRe.ReplaceAllString(out, " ")
	out = spaceBeforeMarkRe.ReplaceAllString(out, "$1")
	out = markRunRe.ReplaceAllString(out, "$1")
	return strings.TrimSpace(out)
}

func stripInvisibles(text string) string {
	seen := false
	for _, r := range text {
		if invisibles[r] {
			seen = true
			break
		}
	}
	if !seen {
		return text
	}
	var b strings.Builder
	for _, r := range text {
		if !invisibles[r] {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// priced spells a currency amount's decimal mark the way `language` does.
//
// The one place a dot between digits is known not to be a clock time, and the
// last place that knows it: by `expandTimes` the symbol has become a trailing
// word and `$0.49` is indistinguishable from `14.30`, which in the eleven
// comma-decimal languages is how a time is written. German answered "null Uhr
// neunundvierzig Dollar". Only a lone dot with a plain fraction is touched,
// `$1,234.56` carries a grouping mark this cannot safely reinterpret.
func priced(amount, language string) string {
	sep := decimalSeparator(language)
	if sep == "." {
		return amount
	}
	if plainDecimal.MatchString(amount) {
		return strings.Replace(amount, ".", sep, 1)
	}
	return amount
}

var plainDecimal = regexp.MustCompile(`^\d+\.\d+$`)

// speakPrefixCurrency rewrites every `$12.34` in text as "12.34 dollars", and
// leaves one glued to the end of a word written.
//
// The letter in front is looked at, never consumed. `R$` is the Brazilian
// real, `HK$` the Hong Kong dollar, `NT$` the Taiwan dollar, and this table
// has a wording for none of them: matching the `$` alone read `R$3,14` as
// "R3,14 Dollar", the wrong currency said confidently. The mark itself is
// still dropped by the punctuation pass, so the amount reads as a plain
// decimal, and losing a symbol is a smaller lie than naming the wrong money.
//
// A price may also say its magnitude, and it is said before the currency word:
// `$2.5M` is two point five million dollars.
func speakPrefixCurrency(text string, re *regexp.Regexp, word, language string) string {
	spans := re.FindAllStringSubmatchIndex(text, -1)
	if spans == nil {
		return text
	}
	var b strings.Builder
	b.Grow(len(text) + len(spans)*(len(word)+1))
	last := 0
	for _, s := range spans {
		if letterBefore(text, s[0]) {
			continue
		}
		end := s[1]
		suffix, spelled := matchGroup(text, s, 2), matchGroup(text, s, 3)
		if (suffix != "" || spelled != "") && asciiLetterAt(text, end) {
			// The `(?![A-Za-z])` that closes the magnitude. A letter behind it
			// means `$5kg`, where neither alternative holds: the optional group
			// is given back, the match ends at the digits, and the amount is
			// spoken alone.
			suffix, spelled, end = "", "", s[3]
		}
		amount := priced(text[s[2]:s[3]], language)
		said := scaleAfter(text[s[2]:s[3]], suffix, spelled, language)
		b.WriteString(text[last:s[0]])
		switch {
		case suffix != "" && said == "":
			// A magnitude this language has no noun for. The letter stays
			// written, which is what it did before the amount was moved.
			b.WriteString(amount + " " + word + suffix)
		case said != "":
			b.WriteString(amount + " " + said + " " + word)
		default:
			b.WriteString(amount + " " + word)
		}
		last = end
	}
	b.WriteString(text[last:])
	return b.String()
}

// matchGroup is capture n of a submatch index run, or "" when the pattern has
// no such group or the group took no part in the match.
func matchGroup(text string, span []int, n int) string {
	if 2*n+1 >= len(span) || span[2*n] < 0 {
		return ""
	}
	return text[span[2*n]:span[2*n+1]]
}

// scaleAfter is the magnitude word standing between a price and its currency,
// or "" when the price carries none.
//
// A written scale reaches speech in two shapes and both belong before the
// currency word: the letter glued to the digits (`$2.5M`) and the noun beside
// them (`$5 million`). The noun is already this language's own word and is kept
// as written; the letter is a number, so the grammar is asked for the form this
// count takes.
func scaleAfter(amount, suffix, spelled, language string) string {
	if spelled != "" {
		return spelled
	}
	if suffix == "" {
		return ""
	}
	whole := amount
	if i := strings.IndexByte(whole, '.'); i >= 0 {
		whole = whole[:i]
	}
	if i := strings.IndexByte(whole, ','); i >= 0 {
		whole = whole[:i]
	}
	digits := strings.Map(func(r rune) rune {
		if r >= '0' && r <= '9' {
			return r
		}
		return -1
	}, whole)
	count := int64(0)
	if digits != "" {
		// A run too long for an integer is not a count anyone wrote, and zero
		// takes the same plural form a large number takes.
		if parsed, err := strconv.ParseInt(digits, 10, 64); err == nil {
			count = parsed
		}
	}
	return scaleSuffixWord(suffix, count, language)
}

// speakOperators says the ASCII comparison operators as words in this language.
//
// The whitespace on both sides is the evidence and is not consumed: only the
// operator is replaced. An operator no grammar covers stays written, like every
// other symbol this funnel has no word for.
func speakOperators(text, language string) string {
	spans := operatorRunRe.FindAllStringIndex(text, -1)
	if spans == nil {
		return text
	}
	var b strings.Builder
	b.Grow(len(text))
	last := 0
	for _, s := range spans {
		if !whitespaceBefore(text, s[0]) || !whitespaceAt(text, s[1]) {
			continue
		}
		word := ""
		for _, op := range asciiOperators {
			if op.written == text[s[0]:s[1]] {
				word = unitWord(op.symbol, language)
				break
			}
		}
		if word == "" {
			continue
		}
		b.WriteString(text[last:s[0]])
		b.WriteString(word)
		last = s[1]
	}
	b.WriteString(text[last:])
	return b.String()
}

// whitespaceAt and whitespaceBefore are the funnel's own White_Space class,
// asked of one position rather than of a pattern.
//
// unicode.IsSpace is the White_Space property itself, so it is the same set
// whitespaceClass writes out, and not RE2's `\s`, which is ASCII and omits the
// vertical tab.
func whitespaceAt(text string, at int) bool {
	if at >= len(text) {
		return false
	}
	r, _ := utf8.DecodeRuneInString(text[at:])
	return unicode.IsSpace(r)
}

func whitespaceBefore(text string, at int) bool {
	if at <= 0 {
		return false
	}
	r, _ := utf8.DecodeLastRuneInString(text[:at])
	return unicode.IsSpace(r)
}

// letterBefore reports whether the character before byte offset at is a
// letter, as Unicode category L means it.
func letterBefore(text string, at int) bool {
	if at <= 0 {
		return false
	}
	r, _ := utf8.DecodeLastRuneInString(text[:at])
	return unicode.IsLetter(r)
}

func speakSymbols(text, language string) string {
	out := text
	if _, ok := unitWords[language]; !ok {
		// A language without a wording table hears English rather than
		// silence: the symbol is at least said, if with an accent.
		language = "en"
	}
	out = speakOperators(out, language)
	// Prefix currencies first, while the digits still follow the symbol.
	for i, sym := range currencyPrefixes {
		word := unitWord(string(sym), language)
		// The pattern below needs the symbol itself, so a string without it
		// cannot match: the same guard the two loops after this one carry.
		if word == "" || !strings.ContainsRune(out, sym) {
			continue
		}
		out = speakPrefixCurrency(out, currencyPrefixPatterns(language)[i], word, language)
	}
	// The same amount with the symbol behind it. `2.50 €` and `0.49¢` are prices by
	// exactly the evidence `€2.50` is, and reached the time pass with the dot intact:
	// German answered "zwei Uhr fünfzig Euro". Currency written as a *word*,
	// `5.50 zł`, is not covered; telling those from a unit needs a
	// per-language lexicon.
	for _, sym := range currencySymbols {
		word := unitWord(string(sym), language)
		if word == "" || !strings.ContainsRune(out, sym) {
			continue
		}
		re := currencySuffixRe[sym]
		out = re.ReplaceAllStringFunc(out, func(m string) string {
			return priced(re.FindStringSubmatch(m)[1], language) + " " + word
		})
	}
	for _, sym := range spokenSymbols {
		if !strings.ContainsRune(out, sym) {
			continue
		}
		repl := unitWord(string(sym), language)
		if repl == "" {
			repl = symbolMarks[sym]
		}
		if repl == "" {
			continue
		}
		// A word replacement needs spaces around it; a punctuation one must
		// not gain a space BEFORE it or the comma floats.
		spaced := " " + repl + " "
		if repl == "," {
			spaced = repl + " "
		}
		out = strings.ReplaceAll(out, string(sym), spaced)
	}
	return out
}

// spaceRunRe, spaceBeforeMarkRe and markRunRe close the funnel: a run of
// blanks becomes one space, a blank in front of a clause mark goes, and a run
// of clause marks becomes one mark.
//
// Package level, not compiled inside Prepared: they are constant patterns and
// Prepared runs once per chunk.
var (
	spaceRunRe = regexp.MustCompile(`[ \t]{2,}`)
	// A symbol that became a comma inherits the space that sat in front of it
	// ("0.49 → 0.24" would read "zero point four nine ,").
	//
	// The class is spelled out because RE2's `\s` is ASCII (`[\t\n\f\r ]`)
	// while Python's re, Rust's regex, JS and Swift are Unicode-aware here.
	// With `\s` this port keeps the NBSP that French typography puts before
	// ! ? ; : and the frontend folds it to a plain space, one extra token per
	// mark. Measured on one French sentence: 79 token ids against 75 in the
	// other four.
	spaceBeforeMarkRe = regexp.MustCompile(`[` + whitespaceClass + `]+([.,;:!?])`)
	// A run, not a pair: regex substitution does not overlap its matches, so a
	// pair rule turns "..." into ".." on one pass and "." on the next, making
	// the funnel non-idempotent.
	markRunRe = regexp.MustCompile(`([.,;:])(?:[` + whitespaceClass + `]*[.,;:])+`)
)

// whitespaceClass is Unicode White_Space as an RE2 character class.
//
// Python reference: loudkit.frontend.speechtext.WHITE_SPACE. RE2's own `\s` is
// `[\t\n\f\r ]`, ASCII and missing the vertical tab, so a pattern here that
// spells "a space" as `\s` means something narrower than the other four ports
// mean.
const whitespaceClass = `\x{0009}\x{000a}\x{000b}\x{000c}\x{000d}\x{0020}\x{0085}\x{00a0}\x{1680}\x{2000}-\x{200a}\x{2028}\x{2029}\x{202f}\x{205f}\x{3000}`

// footnoteRe matches a bracketed footnote marker.
//
// [0-9] and the explicit space class, not `\d` and `\s`, because RE2 reads
// both as ASCII: a marker separated by a non-breaking or thin space (ordinary
// French and German typography) then survives here and is read aloud by the
// number pass, where the other four ports drop it. The three dash characters
// in the class are literal data, the marks a range in a marker is written
// with, not prose.
var footnoteRe = regexp.MustCompile(`\[[0-9` + whitespaceClass + `,;\-–—]{1,20}\]`)

// foldNumerals makes every number character something a reader can say aloud.
//
// Python reference: loudkit.frontend.speechtext.fold_numerals, which carries
// the reasoning. A non-ASCII decimal digit becomes the ASCII digit of the same
// value and joins the run it was in; every other number character becomes the
// text numerals.json names for it, separated from an adjacent alphanumeric so
// that "\u00b29" does not fold into "29" and read as twenty-nine. The table
// decides whether a character is a numeral too: see foldedNumeral.
func foldNumerals(text string) string {
	if !strings.ContainsFunc(text, func(r rune) bool { _, _, ok := foldedNumeral(r); return ok }) {
		return text
	}
	runes := []rune(text)
	var out []rune
	for i, r := range runes {
		spelled, isDigit, ok := foldedNumeral(r)
		if !ok {
			out = append(out, r)
			continue
		}
		if isDigit {
			// A digit replacing a digit joins the run it was already in.
			out = append(out, []rune(spelled)...)
			continue
		}
		// A slash inside a spelled numeral is a fraction bar, asserted by the
		// character itself: `½` is a half wherever it stands, where a typed
		// `1/2` is a fraction, a date or the `24/7` of ordinary prose. The
		// division sign is the mark the symbol table already has a word for in
		// every language, so the reading comes from the grammar and not here.
		spelled = strings.ReplaceAll(spelled, "/", "÷")
		if len(out) > 0 && isLetterOrDigit(out[len(out)-1]) {
			out = append(out, ' ')
		}
		out = append(out, []rune(spelled)...)
		if i+1 < len(runes) && isLetterOrDigit(runes[i+1]) {
			out = append(out, ' ')
		}
	}
	return string(out)
}

// isLetterOrDigit is `\p{L}` or `\p{Nd}`, the word class all five ports test.
func isLetterOrDigit(r rune) bool {
	return unicode.IsLetter(r) || unicode.IsDigit(r)
}

// foldedNumeral answers what a numeral reads as, whether it is a decimal digit,
// and whether the table names it at all.
//
// Both halves come from the shared numerals.json, detection included. Computing
// either was wrong: walking down to a decimal block's start walks out of the
// block where two blocks touch (MATHEMATICAL DOUBLE-STRUCK DIGIT ZERO follows
// MATHEMATICAL SANS-SERIF DIGIT NINE with nothing between), and NFKC does not
// reach
// ETHIOPIC NUMBER TEN, the Aegean numbers, the Kaktovik digits or the Meroitic
// numerals, which then vanished; and asking unicode.IsNumber *whether* to fold
// put the runtime's Unicode version back into an answer the table had taken it
// out of. Go's tables move with the toolchain, Python's with the interpreter,
// and the five ports report one fingerprint. The table is cut from one pinned
// UCD and hashed into the grammar digest.
//
// The third result is false for a character the table does not name: an ASCII
// digit, a letter, an ideographic numeral. Those are left exactly as written.
func foldedNumeral(r rune) (string, bool, bool) {
	if r >= '0' && r <= '9' {
		return "", false, false
	}
	t := numeralTable()
	if text, ok := t.spelled[r]; ok {
		return text, false, true
	}
	i := sort.Search(len(t.decimalZeros), func(i int) bool { return t.decimalZeros[i] > r })
	if i > 0 {
		if offset := r - t.decimalZeros[i-1]; offset >= 0 && offset <= 9 {
			return string(rune('0' + offset)), true, true
		}
	}
	return "", false, false
}

//go:embed numerals.json
var numeralsJSON []byte

// NumeralBytes is the embedded numeral table exactly as this binary carries it.
func NumeralBytes() []byte { return numeralsJSON }

type numeralData struct {
	decimalZeros []rune
	spelled      map[rune]string
}

var (
	numeralOnce  sync.Once
	numeralCache numeralData
)

// numeralTable parses numerals.json once.
func numeralTable() numeralData {
	numeralOnce.Do(func() {
		var raw struct {
			DecimalZeros []int             `json:"decimal_zeros"`
			Spelled      map[string]string `json:"spelled"`
		}
		if err := json.Unmarshal(numeralsJSON, &raw); err != nil {
			panic("numerals.json unreadable: " + err.Error())
		}
		numeralCache.decimalZeros = make([]rune, len(raw.DecimalZeros))
		for i, z := range raw.DecimalZeros {
			numeralCache.decimalZeros[i] = rune(z)
		}
		numeralCache.spelled = make(map[rune]string, len(raw.Spelled))
		for k, v := range raw.Spelled {
			cp, err := strconv.Atoi(k)
			if err != nil {
				panic("numerals.json: bad code point " + k)
			}
			numeralCache.spelled[rune(cp)] = v
		}
	})
	return numeralCache
}

func dropMarkupTags(text string) string {
	if !strings.Contains(text, "<") {
		return text
	}
	return markupTagRe.ReplaceAllString(text, " ")
}

func dropFootnoteMarkers(text string) string {
	if !strings.Contains(text, "[") {
		return text
	}
	return footnoteRe.ReplaceAllString(text, "")
}

func punctuationForSpeech(text string) string {
	runes := []rune(text)
	var b strings.Builder
	for i, sc := range runes {
		isLetter := unicode.IsLetter(sc)
		isDigit := unicode.IsDigit(sc)
		if isLetter || isDigit || unicode.IsSpace(sc) || prosodic[sc] {
			b.WriteRune(sc)
			continue
		}
		var prev, next rune
		hasPrev, hasNext := false, false
		if i > 0 {
			prev, hasPrev = runes[i-1], true
		}
		if i+1 < len(runes) {
			next, hasNext = runes[i+1], true
		}
		// Between digits, "." and "," are numeric separators and "-" and "/"
		// are ranges and fractions: meaning, not decoration.
		betweenDigits := hasPrev && unicode.IsDigit(prev) && hasNext && unicode.IsDigit(next)
		if betweenDigits && strings.ContainsRune("-/:.", sc) {
			b.WriteRune(sc)
			continue
		}
		// A hyphen inside a word is part of the word ("well-known").
		// Either end alphanumeric, not both letters: a both-letters test lets
		// the exponent in "1e-3" become a space, and the model is handed
		// "1e 3" after the number pass has already declined to read it. The
		// same holds for "+", which the number pass declines as a token with a
		// letter in it.
		if (sc == '-' || sc == '+') &&
			hasPrev && (unicode.IsLetter(prev) || unicode.IsDigit(prev)) &&
			hasNext && (unicode.IsLetter(next) || unicode.IsDigit(next)) {
			b.WriteRune(sc)
			continue
		}
		b.WriteRune(' ')
	}
	return b.String()
}
