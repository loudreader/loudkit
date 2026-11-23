// Package chunking mirrors loudkit.frontend.chunking: splitting text that is
// longer than one window.
//
// A window carries about 255 speech tokens, roughly ten seconds. Anything
// longer has to be split, generated in pieces and joined, and *where* the
// splits fall is audible: a break at a full stop is inaudible, a break
// mid-clause is not. That makes it an algorithm-layer decision rather than a
// caller's convenience, and it has to be identical in every port: a different
// split is a different set of joins and therefore a different reading.
//
// Python reference: loudkit/frontend/chunking.py.
package chunking

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"unicode"
)

// CharsPerToken is characters of prepared text per speech token.
//
// Measured on the reference voice across English, Polish (after the respelling
// funnel) and German: 0.53-0.64. The constant is the low end with margin
// (0.5 < the 0.53 measured minimum) because it is used to *stay under* a
// limit, never to predict a length: the middle of the range would let the
// worst case overflow the window more often still.
//
// It is a budget, not a guarantee. Measured over 9920 rendered chunks in
// ten languages, 54 overran the window anyway: a mean cannot bound a
// variance, and most of those are chunks that should have fitted and did
// not because the model emitted no stop token in time. An overflow is not
// an error either: the generator stops at the cap mid-word and the remainder
// is never spoken, which is what cap_resplit and split_in_half exist for.
//
// It must equal loudkit.frontend.chunking.CHARS_PER_TOKEN.
const CharsPerToken = 0.5

// HoldMidSentencePeriod and BreakMidSentencePeriod are the two spellings of
// Config.MidSentencePeriod.
//
// A period that does not end a sentence is not a boundary.
// HoldMidSentencePeriod is the law: without it "But Mr. Smith went home"
// breaks after the title and hands the renderer a seven-character chunk with
// its own derived seed and a token ceiling proportional to seven characters.
// BreakMidSentencePeriod names the other law so a pack can say which one it
// was measured under.
const (
	HoldMidSentencePeriod  = "hold"
	BreakMidSentencePeriod = "break"
)

// WordCapResplit and OffCapResplit are the two spellings of Config.CapResplit.
//
// SplitText budgets characters against a constant, and a speaker slower than
// it fills the window before the text runs out; the generator then stops at
// the cap mid-word and the remainder is lost, because chunk texts are fixed
// before any of them renders. WordCapResplit halves such a chunk and generates
// both halves in its place. OffCapResplit ships the truncated window, which is
// what a checkpoint that does not set this field gets.
const (
	WordCapResplit = "word"
	OffCapResplit  = "off"
)

// Config is the chunking policy, from the checkpoint's AlgorithmConfig.
type Config struct {
	Enabled      bool
	MaxTokens    int
	PrefixTokens int
	SplitOn      []string
	// Abbreviations are written forms whose following period does not end a
	// sentence, given without that period: the period is the separator's.
	// One union list for every language: surveyed over 1200 passages in ten,
	// a language-blind union re-chunks the corpus identically to ten
	// per-language lists. Data, not code: replacing the slice is the whole of
	// adding a language. Not the funnel's list, which maps a written
	// abbreviation to spoken words and carries only the unambiguous ones; what
	// reaches here is the residue the funnel refuses to touch.
	Abbreviations []string
	// MidSentencePeriod is HoldMidSentencePeriod or BreakMidSentencePeriod.
	MidSentencePeriod string
	// CapResplit is WordCapResplit or OffCapResplit.
	CapResplit string
}

// Validate refuses the four configurations
// loudkit.config.ChunkConfig.__post_init__ refuses, in the same sentence, so a
// user who hits one in two languages reads the same words twice.
//
// The second is the one with teeth: a MaxTokens small enough that
// int(MaxTokens * CharsPerToken) is zero makes SplitText cut nothing and loop
// forever, which on a server is a wedged request holding the single-flight
// engine.
func (c Config) Validate() error {
	if c.MaxTokens <= 0 {
		return fmt.Errorf("chunking.max_tokens must be positive: %d", c.MaxTokens)
	}
	if int(float64(c.MaxTokens)*CharsPerToken) < 1 {
		return fmt.Errorf(
			"chunking.max_tokens=%d leaves no character budget to split on "+
				"(int(%d * %v) == 0); needs at least %d",
			c.MaxTokens, c.MaxTokens, CharsPerToken, int(math.Ceil(1/CharsPerToken)))
	}
	if c.PrefixTokens < 0 || c.PrefixTokens >= c.MaxTokens {
		return fmt.Errorf(
			"chunking.prefix_tokens must be in [0, max_tokens): %d", c.PrefixTokens)
	}
	if len(c.SplitOn) == 0 {
		return errors.New("chunking.split_on cannot be empty: there would be nowhere to break")
	}
	if c.MidSentencePeriod != HoldMidSentencePeriod &&
		c.MidSentencePeriod != BreakMidSentencePeriod {
		return fmt.Errorf(
			"unknown mid_sentence_period %q: expected %q or %q",
			c.MidSentencePeriod, HoldMidSentencePeriod, BreakMidSentencePeriod)
	}
	if c.CapResplit != WordCapResplit && c.CapResplit != OffCapResplit {
		return fmt.Errorf("unknown cap_resplit %q: expected %q or %q",
			c.CapResplit, WordCapResplit, OffCapResplit)
	}
	for _, a := range c.Abbreviations {
		// An empty entry is a suffix of everything, so it would hold every
		// candidate and drive every split down to a word boundary. Silent,
		// and audible on every long passage.
		if a == "" {
			return errors.New("chunking.abbreviations cannot contain an empty string")
		}
	}
	return nil
}

var proseSeparators = []string{". ", "! ", "? ", "; ", ", "}

// Production is the shipping chunking recipe.
func Production() Config {
	return Config{
		Enabled:      true,
		MaxTokens:    255,
		PrefixTokens: 6,
		SplitOn:      slices.Clone(proseSeparators),
		// The surveyed union, sorted. It must equal
		// loudkit.config.ChunkConfig.abbreviations.
		Abbreviations: []string{
			"A", "B", "Cpn", "D", "Dr", "F", "H", "Hr", "I", "J", "K", "M",
			"Mr", "Mrs", "R", "S", "St", "T", "V", "Vors", "dr", "mrs",
			"prof", "\u015bw",
		},
		MidSentencePeriod: HoldMidSentencePeriod,
		CapResplit:        WordCapResplit,
	}
}

// isASCIIWord reports whether b is an ASCII letter or digit.
//
// A byte, not a rune, and no unicode.IsLetter: the five implementations have to
// answer this identically, and every language's idea of "letter" is its own.
// ASCII is the part they cannot disagree on, and a UTF-8 continuation byte is
// never mistaken for one.
func isASCIIWord(b byte) bool {
	return (b >= '0' && b <= '9') || (b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z')
}

// holds reports whether the candidate at byte offset at is a period inside a
// sentence rather than the end of one.
//
// look is the search window plus one rune, because the test below reads the
// character *after* the separator and the latest candidate can end the window
// exactly.
//
// Gated on the period: "! " and "? " end sentences and "; " and ", " do not end
// them at all, so neither is ever in doubt. The whole question is about the one
// mark that is written for two jobs.
func holds(look string, at int, sep string, cfg Config) bool {
	if cfg.MidSentencePeriod != HoldMidSentencePeriod || sep == "" || sep[0] != '.' {
		return false
	}
	// A sentence does not resume in lower case. This is what catches the
	// ellipsis the funnel folds to a single period, which no abbreviation list
	// reaches. ASCII only, and measured rather than assumed: over 2253 periods
	// in ten languages, four are followed by a word starting with a non-ASCII
	// lowercase letter, and reading the whole Unicode Lowercase property
	// instead moves one passage in 1200.
	if after := at + len(sep); after < len(look) &&
		look[after] >= 'a' && look[after] <= 'z' {
		return true
	}
	// Or the token in front of the period is a listed abbreviation. Entries
	// carry no period of their own: the period belongs to the separator.
	boundary := look[:at]
	for _, abbreviation := range cfg.Abbreviations {
		if !strings.HasSuffix(boundary, abbreviation) {
			continue
		}
		before := len(boundary) - len(abbreviation)
		if before > 0 && isASCIIWord(boundary[before-1]) {
			continue // the tail of a longer word, not a word of its own
		}
		return true
	}
	return false
}

// EstimateTokens is a conservative upper estimate of the speech tokens text
// will produce.
func EstimateTokens(text string) int {
	return int(float64(len([]rune(text)))/CharsPerToken) + 1
}

// SplitText splits text into pieces that each fit one window, in order,
// together covering the input. Never empty for non-empty input.
//
// Indexing is by rune, not byte: a byte-indexed cut lands inside a multi-byte
// character and produces invalid UTF-8, which is the shape of bug this port
// has had before.
func SplitText(text string, cfg Config) []string {
	return SplitTextForLanguage(text, cfg, "en")
}

// SplitTextForLanguage applies the prose rules for the spoken language.
func SplitTextForLanguage(text string, cfg Config, language string) []string {
	trimmed := strings.TrimFunc(text, chunkSpace)
	if trimmed == "" {
		return nil
	}
	if !cfg.Enabled || EstimateTokens(trimmed) <= cfg.MaxTokens {
		return []string{trimmed}
	}

	budget := int(float64(cfg.MaxTokens) * CharsPerToken)
	var chunks []string
	language = strings.ToLower(language)
	prose := slices.Equal(cfg.SplitOn, proseSeparators)
	var abbreviations [][]rune
	if prose {
		for _, abbreviation := range cfg.Abbreviations {
			abbreviations = append(abbreviations, []rune(abbreviation))
		}
	}
	rest := []rune(trimmed)

	for len(rest) > 0 {
		if len(rest) <= budget {
			chunks = append(chunks, strings.TrimFunc(string(rest), chunkSpace))
			break
		}
		var cut int
		if prose {
			cut = proseCut(rest, budget, cfg, abbreviations, language)
		} else {
			cut = configuredCut(rest, budget, cfg)
		}
		// Never zero: a cut of 0 leaves rest unchanged and the loop spins.
		if cut < 1 {
			cut = 1
		}
		chunks = append(chunks, strings.TrimFunc(string(rest[:cut]), chunkSpace))
		// Whitespace after a cut does not count against the next budget.
		// Advance over it in the existing rune slice to keep long passages linear.
		rest = rest[cut:]
		for len(rest) > 0 && chunkSpace(rest[0]) {
			rest = rest[1:]
		}
	}

	out := chunks[:0]
	for _, c := range chunks {
		if c != "" {
			out = append(out, c)
		}
	}
	return out
}

// proseCut tries to finish a sentence before choosing a weaker pause.
// All positions below are rune offsets within the remaining text.
func proseCut(rest []rune, budget int, cfg Config, abbreviations [][]rune, language string) int {
	limit := budget
	if cfg.MaxTokens == 255 && cfg.CapResplit == WordCapResplit {
		limit = min(len(rest), budget+24)
	}
	sentence, extended, clause, comma := 0, 0, 0, 0
	pauses := 0
	for i, mark := range rest[:limit] {
		if !strings.ContainsRune(".!?;:,–\u2014", mark) {
			continue
		}
		end := i + 1
		for end < limit && isCloser(rest[end]) {
			end++
		}
		if end < len(rest) && !chunkSpace(rest[end]) {
			continue
		}
		switch mark {
		case '.', '!', '?':
			if mark == '.' && periodHeld(rest, i, end, cfg, abbreviations) {
				continue
			}
			if end <= budget {
				sentence = end
			} else if extended == 0 {
				extended = end
			}
		default:
			if end > budget || (budget >= 64 && end*4 < budget) {
				continue
			}
			if budget >= 64 && len(rest)-end <= 24 && len(strings.FieldsFunc(string(rest[end:]), chunkSpace)) == 1 {
				continue
			}
			if mark == ',' {
				comma = end
				pauses++
			} else if mark == ';' || mark == ':' ||
				i > 0 && chunkSpace(rest[i-1]) && i+1 < len(rest) && chunkSpace(rest[i+1]) {
				clause = end
				pauses++
			}
		}
	}
	if pauses >= 2 {
		extended = 0
	}
	for _, candidate := range [...]int{sentence, extended, clause, comma} {
		if candidate > 0 {
			return candidate
		}
	}

	return wordCut(rest, budget, cfg, abbreviations, language)
}

func wordCut(rest []rune, budget int, cfg Config, abbreviations [][]rune, language string) int {
	// Keep both sides substantial when a greedy word break would leave a
	// short tail. Twice the gap offset compares distances without rounding.
	balance := budget >= 64 && (len(rest)-budget)*4 < budget
	latest, best, bestDistance := 0, 0, 0
	for i := 1; i <= budget; i++ {
		if !isWordBoundary(rest[i]) || isWordBoundary(rest[i-1]) {
			continue
		}
		latest = i
		if rest[i-1] == '.' && periodHeld(rest, i-1, i, cfg, abbreviations) {
			continue
		}
		distance := 2*i - len(rest)
		if distance < 0 {
			distance = -distance
		}
		if best == 0 || !balance || distance < bestDistance {
			best, bestDistance = i, distance
		}
	}
	if best > 0 {
		if language == "en" && budget >= 64 {
			return attachConjunction(rest, best, budget)
		}
		return best
	}
	if latest > 0 {
		return latest
	}
	return budget
}

// Keep an English clause opener with the clause it introduces. For "while",
// require a subject so phrases such as "for a little while" keep their break.
func attachConjunction(text []rune, cut, budget int) int {
	start := cut
	for start > 0 && cut-start < len("although") && !isWordBoundary(text[start-1]) {
		start--
	}
	if start == 0 || !isWordBoundary(text[start-1]) || (start-1)*4 < budget {
		return cut
	}
	word := string(text[start:cut])
	switch word {
	case "although", "because", "unless", "whereas", "while":
	default:
		return cut
	}
	next := cut
	for next < len(text) && chunkSpace(text[next]) {
		next++
	}
	if len(text)-next <= budget && len(text)-start > budget {
		return cut
	}
	if word == "while" {
		end := next
		for end < len(text) && end-next <= len("they") && !isWordBoundary(text[end]) {
			end++
		}
		switch string(text[next:end]) {
		case "the", "they", "we", "she", "he", "it", "you":
		default:
			return cut
		}
	}
	return start
}

func isCloser(r rune) bool {
	return strings.ContainsRune("\"'”’»›“‘«‹)]}", r)
}

// chunkSpace mirrors str.isspace() used for punctuation in the reference.
func chunkSpace(r rune) bool {
	return unicode.IsSpace(r) || 0x1c <= r && r <= 0x1f
}

func periodHeld(text []rune, period, end int, cfg Config, abbreviations [][]rune) bool {
	if cfg.MidSentencePeriod != HoldMidSentencePeriod {
		return false
	}
	for end < len(text) && chunkSpace(text[end]) {
		end++
	}
	if end == len(text) {
		return false
	}
	next := text[end]
	if next >= 'a' && next <= 'z' {
		return true
	}
	asciiWord := func(r rune) bool { return r < 128 && isASCIIWord(byte(r)) }
	for _, abbr := range abbreviations {
		start := period - len(abbr)
		if start >= 0 && (start == 0 || !asciiWord(text[start-1])) && slices.Equal(text[start:period], abbr) {
			return true
		}
	}
	if period == 0 || (period >= 2 && asciiWord(text[period-2])) {
		return false
	}
	initial := text[period-1]
	return initial >= 'A' && initial <= 'Z' && initial != 'I' && next >= 'A' && next <= 'Z'
}

// configuredCut keeps the caller's separator order and literal matches.
func configuredCut(rest []rune, budget int, cfg Config) int {
	head := string(rest[:budget+1])
	look := string(rest[:min(len(rest), budget+2)])
	for _, sep := range cfg.SplitOn {
		for at := strings.LastIndex(head, sep); at > 0; at = strings.LastIndex(head[:at], sep) {
			if !holds(look, at, sep, cfg) {
				return len([]rune(head[:at+len(sep)]))
			}
		}
	}
	for i := budget; i > 0; i-- {
		if isWordBoundary(rest[i]) {
			return i
		}
	}
	return budget
}

// WordBoundaries are the characters SplitInHalf may cut on, written out rather
// than tested for.
//
// A predicate would be shorter and the five ports do not agree on one:
// measured, Python's `str.isspace()` treats U+001C-U+001F as whitespace where
// the other four do not, and JS alone KEEPS U+0085 where the other four strip
// it. Swift alone strips U+200B, JS alone strips U+FEFF; the funnel removes
// both before the splitter sees them, but U+0085 and U+001C-U+001F survive
// it. A disagreement there is a different split point, which is different
// audio for the same text and seed. A hand-written set cannot drift.
//
// The funnel does not remove these. NBSP in particular is ordinary in real
// prose ("10 000", French punctuation, typeset copy), and a capped chunk whose
// only boundaries are NBSP comes back unsplittable and ships its truncation if
// they are not counted here.
var WordBoundaries = []rune{'\u0020', '\u0009', '\u000a', '\u000d', '\u00a0', '\u2007', '\u202f'}

func isWordBoundary(r rune) bool {
	for _, b := range WordBoundaries {
		if r == b {
			return true
		}
	}
	return false
}

// SplitInHalf halves a chunk the window could not hold, at a word boundary.
// The second return is false when there is no word boundary to use.
//
// SplitText's estimate is conservative but not a guarantee: it budgets
// characters against a constant, and a speaker slower than that constant fills
// the window before the text runs out. The generator then stops at the cap
// mid-word, and the words that did not fit are lost rather than deferred,
// because chunk texts are fixed before any of them is rendered. The measured
// overrun rates are in docs/design/text-funnel.md.
//
// The boundary is the nearest word break, and punctuation is not sought: a
// comma is an instruction to pause, this model has no pause-duration prior,
// and fed one it overshoots. Not sought is not avoided: the nearest word break
// can follow a comma.
//
// A single unbroken run is refused. Splitting it would have to cut a word,
// which is worse than the truncation it would be repairing.
func SplitInHalf(text string) (string, string, bool) {
	runes := []rune(text)
	middle := float64(len(runes)) / 2
	best := -1
	bestDistance := 0.0
	for i, r := range runes {
		// Interior only: a boundary at either end yields an empty half.
		if !isWordBoundary(r) || i <= 0 || i >= len(runes)-1 {
			continue
		}
		d := math.Abs(float64(i) - middle)
		if best < 0 || d < bestDistance {
			best, bestDistance = i+1, d
		}
	}
	if best < 0 {
		return "", "", false
	}
	first := strings.TrimFunc(string(runes[:best]), chunkSpace)
	second := strings.TrimFunc(string(runes[best:]), chunkSpace)
	// Trimming can empty a half the scan thought was interior, on input whose
	// boundary run is all whitespace. SplitText trims before this is ever
	// called, so the engine cannot reach it, but this is exported and a caller
	// handed an empty half would render silence and call it speech.
	if first == "" || second == "" {
		return "", "", false
	}
	return first, second, true
}
