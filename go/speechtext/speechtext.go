// Package speechtext is the language-agnostic speech funnel — a bit-parity
// port of the Swift engine's SpeechText and the JS/Python funnels.
//
// Before tokenising, the shipped engine scrubs the raw text: invisible
// characters, symbols that carry meaning, footnote markers, and punctuation
// (prosodic marks stay exactly where they are — the model is a language model
// trained on punctuated text — everything else becomes a space). Applied by
// the engine's Encode path, mirroring Engine._synthesize_one in Python and
// Engine.encode in JS.
//
// The Polish English-respelling lexicon is ported too — see respell.go, which
// embeds the generated dictionary and is wired into Prepared below.//
// Python reference: `loudkit/frontend/polish.py`.
package speechtext

import (
	_ "embed"
	"encoding/json"
	"regexp"
	"strings"
	"unicode"

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
// make text normalisation vary run to run for one input — the opposite of what
// this port exists to guarantee. Python's dict and JS's object both preserve
// insertion order, so the reference order is the literal order below and it
// must stay in step with `_SYMBOL_WORDS` in loudkit.frontend.polish.
type symbolRule struct {
	sym rune
	en  string
	pl  string
}

// Symbols the model cannot voice, as words: (en, pl).
var symbolWords = []symbolRule{
	{'%', "percent", "procent"},
	{'°', "degrees", "stopni"},
	{'¢', "cents", "centów"},
	{'€', "euro", "euro"},
	{'£', "pounds", "funtów"},
	{'¥', "yen", "jenów"},
	{'₹', "rupees", "rupii"},
	{'×', "times", "razy"},
	{'÷', "divided by", "podzielone przez"},
	{'≈', "about", "około"},
	{'≥', "at least", "co najmniej"},
	{'≤', "at most", "najwyżej"},
	{'≠', "not equal to", "różne od"},
	{'±', "plus minus", "plus minus"},
	{'→', ",", ","},
	{'←', ",", ","},
	{'⇒', ",", ","},
	{'✓', "yes", "tak"},
	{'✔', "yes", "tak"},
	{'✗', "no", "nie"},
	{'✘', "no", "nie"},
	{'•', ",", ","},
	{'·', ",", ","},
	{'▪', ",", ","},
	{'◦', ",", ","},
	{'…', "...", "..."},
	{'&', "and", "i"},
	{'@', "at", "małpa"},
}

// `$` and `£` before a number read as a prefix in writing and a SUFFIX in
// speech: "$5" is "five dollars", not "dollars five".
// The wording comes from unitWords (numbers.json); this list only says which
// symbols are written prefix. Ordered: these run in sequence over one string.
var currencyPrefixes = []rune{'$', '£', '€', '¥', '₹'}

// currencySymbols also carries `¢`, which nobody writes in front of a number —
// it is a suffix in every convention, which is why the prefix pass never saw it
// and "0.49¢" reached the clock reader intact.
var currencySymbols = []rune{'$', '£', '€', '¥', '₹', '¢'}

var currencySuffixRe = map[rune]*regexp.Regexp{}

//go:embed numbers.json
var numbersJSON []byte

// GrammarBytes is the embedded grammar file exactly as this binary carries it.
// Exported so the fingerprint can hash the bytes this port actually reads —
// hashing a file on disk would say nothing about what got compiled in.
func GrammarBytes() []byte { return numbersJSON }

// unitWords is symbol -> word per language, from the shared grammar file. The
// old table was an (en, pl) pair with `pl if polish else en`, which meant
// seven of the nine languages heard English: "$5" in a German render said
// "5 dollars".
var unitWords map[string]map[string]string

func loadUnitWords() {
	var doc struct {
		Languages map[string]struct {
			UnitWords map[string]string `json:"unit_words"`
		} `json:"languages"`
	}
	if err := json.Unmarshal(numbersJSON, &doc); err != nil {
		panic("speechtext: embedded numbers.json is unreadable: " + err.Error())
	}
	unitWords = make(map[string]map[string]string, len(doc.Languages))
	for lang, entry := range doc.Languages {
		unitWords[lang] = entry.UnitWords
	}
}

// unitWord returns the word `symbol` takes in `language`, falling back to
// English so a symbol is at least said, if with an accent.
func unitWord(symbol, language string) string {
	if w, ok := unitWords[language][symbol]; ok {
		return w
	}
	return unitWords["en"][symbol]
}

// Punctuation that carries prosody stays; the rest becomes a space.
var prosodic = map[rune]bool{}

var currencyRe = map[rune]*regexp.Regexp{}

func init() {
	for _, r := range ".,!?;:\u2014\u2013\u2026\"\u201C\u201D\u201E«»()'\u2019\u00BF\u00A1" {
		prosodic[r] = true
	}
	loadUnitWords()
	for _, sym := range currencyPrefixes {
		// A letter in front means a multi-character currency mark.
		//
		// `R$` is the Brazilian real, `HK$` the Hong Kong dollar, `NT$` the Taiwan
		// dollar, and this table has a wording for none of them. Matching the `$` alone
		// read `R$3,14` as "R3,14 Dollar" — the wrong currency, said confidently. The mark
		// itself is still dropped by the punctuation pass, so the amount reads as a plain
		// decimal; losing a symbol is a smaller lie than naming the wrong money.
		//
		// RE2 has no lookbehind, so the letter is captured and put back.
		currencyRe[sym] = regexp.MustCompile(
			`(^|[^\p{L}])` + regexp.QuoteMeta(string(sym)) + `\s?(\d+(?:[.,]\d+)*)`)
	}
	for _, sym := range currencySymbols {
		// No letter guard on this side: only whitespace may sit between the
		// amount and the mark, so `3,14 R$` never matches in the first place.
		currencySuffixRe[sym] = regexp.MustCompile(
			`(\d+(?:[.,]\d+)*)\s?` + regexp.QuoteMeta(string(sym)))
	}
}

// Prepared scrubs text the way the shipped Swift engine does
// (SpeechText.prepared). Same order, same rules, same output.
func Prepared(text, languageID string) string {
	// The language id is lowercased once here and again in the respeller.
	// GraphemeTextFrontend lowercases its own tag, so "PL" produced Polish
	// *tokens* while silently skipping the Polish respelling — the same
	// utterance read half one way and half the other, with nothing to
	// indicate it. Python fixed this in loudkit.frontend.polish.speech_text,
	// and Swift's LexicalRespelling.applied carries the same .lowercased() and
	// the same reason.
	languageID = strings.ToLower(languageID)
	// NFC first, before anything inspects a character — the same opening pass
	// the Python funnel runs, and the one this funnel did not have.
	//
	// Unicode lets the same character arrive two ways: Polish ą as U+0105 or as
	// a + U+0328, Danish å as U+00E5 or a + U+030A. The tokenizer's vocabulary
	// holds one of them, so a decomposed spelling reaches it as a base letter
	// followed by an unknown combining mark — and every rule below, every
	// pattern and lexicon lookup and character class, is matching a string
	// nobody wrote a rule for.
	//
	// Ahead of stripInvisibles, which removes format characters: normalisation
	// can compose a sequence into a single character, and running it afterwards
	// would leave that composition unexamined.
	// Beside NFC, and before the symbol pass so the folded percent sign
	// reaches the table that turns it into a word.
	out := stripInvisibles(FoldForeignDigits(norm.NFC.String(text), languageID))
	out = speakSymbols(out, languageID)
	out = dropFootnoteMarkers(out)
	// Acronyms while the capitals are still capitals: every later pass
	// lowercases or rewrites, and a spelled acronym has to be decided while the
	// only evidence — that the word stands alone in caps — still exists. The
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
	// Numbers after footnotes and before punctuation — see the Python funnel
	// for the ordering argument; the fixture pins it.
	out = ExpandAbbreviations(out, languageID)
	out = ExpandTimes(out, languageID)
	out = ExpandNumbers(out, languageID)
	out = punctuationForSpeech(out)
	// Polish: respell embedded English the way a Polish reader says it. This
	// is the shipped engine's LexicalRespelling; see respell.go.
	out = LexicalRespelling(out, languageID)
	// Collapse runs of spaces/tabs — same as the shipped engine.
	spaces := regexp.MustCompile(`[ \t]{2,}`)
	out = spaces.ReplaceAllString(out, " ")
	// A symbol that became a comma inherits the space that sat in
	// front of it ("0.49 → 0.24" would read "zero point four nine ,").
	clause := regexp.MustCompile(`\s+([.,;:!?])`)
	out = clause.ReplaceAllString(out, "$1")
	// Two clause marks in a row is one clause mark.
	// a run, not a pair: regex substitution does not overlap its matches, so a pair rule turns "..." into ".." on one pass and "." on the next, making the funnel non-idempotent.
	marks := regexp.MustCompile(`([.,;:])(?:[\s]*[.,;:])+`)
	out = marks.ReplaceAllString(out, "$1")
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
// neunundvierzig Dollar". Only a lone dot with a plain fraction is touched —
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

func speakSymbols(text, language string) string {
	out := text
	if _, ok := unitWords[language]; !ok {
		// A language without a wording table hears English rather than
		// silence: the symbol is at least said, if with an accent.
		language = "en"
	}
	// Prefix currencies first, while the digits still follow the symbol.
	for _, sym := range currencyPrefixes {
		word := unitWord(string(sym), language)
		if word == "" {
			continue
		}
		out = currencyRe[sym].ReplaceAllStringFunc(out, func(m string) string {
			g := currencyRe[sym].FindStringSubmatch(m)
			return g[1] + priced(g[2], language) + " " + word
		})
	}
	// The same amount with the symbol behind it. `2.50 €` and `0.49¢` are prices by
	// exactly the evidence `€2.50` is, and reached the time pass with the dot intact:
	// German answered "zwei Uhr fünfzig Euro". Currency written as a *word* — `5.50
	// zł` — is not covered; telling those from a unit needs a per-language lexicon.
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
	for _, rule := range symbolWords {
		if !strings.ContainsRune(out, rule.sym) {
			continue
		}
		repl := unitWord(string(rule.sym), language)
		if repl == "" {
			// Not a per-language word (arrows, ticks): the old pair table
			// still carries these.
			repl = rule.en
			if language == "pl" {
				repl = rule.pl
			}
		}
		// A word replacement needs spaces around it; a punctuation one must
		// not gain a space BEFORE it or the comma floats.
		spaced := " " + repl + " "
		if repl == "," {
			spaced = repl + " "
		}
		out = strings.ReplaceAll(out, string(rule.sym), spaced)
	}
	return out
}

var footnoteRe = regexp.MustCompile(`\[[\d\s,;\-–—]{1,20}\]`)

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
		// are ranges and fractions — meaning, not decoration.
		betweenDigits := hasPrev && unicode.IsDigit(prev) && hasNext && unicode.IsDigit(next)
		if betweenDigits && strings.ContainsRune("-/:.", sc) {
			b.WriteRune(sc)
			continue
		}
		// A hyphen inside a word is part of the word ("well-known").
		// Either end alphanumeric, not both letters: the old test left the
		// exponent in "1e-3" to become a space, so the model was handed
		// "1e 3" after the number pass had already declined to read it.
		// `+` alongside `-`: the number pass declines "1e+3" as a token with a
		// letter in it, and punctuation then took it apart into "1e 3".
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
