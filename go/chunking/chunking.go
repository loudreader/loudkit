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

// Production is the shipping chunking recipe.
func Production() Config {
	return Config{
		Enabled:      true,
		MaxTokens:    255,
		PrefixTokens: 6,
		SplitOn:      []string{". ", "! ", "? ", "; ", ", "},
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
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return nil
	}
	if !cfg.Enabled || EstimateTokens(trimmed) <= cfg.MaxTokens {
		return []string{trimmed}
	}

	budget := int(float64(cfg.MaxTokens) * CharsPerToken)
	var chunks []string
	rest := []rune(trimmed)

	for len(rest) > 0 {
		if len(rest) <= budget {
			chunks = append(chunks, strings.TrimSpace(string(rest)))
			break
		}
		head := string(rest[:budget+1])
		// One rune past the window, and used only by holds: the latest
		// candidate can end the window exactly, and the test reads the
		// character after it. The search itself stays inside the budget.
		lookEnd := budget + 2
		if lookEnd > len(rest) {
			lookEnd = len(rest)
		}
		look := string(rest[:lookEnd])
		cut := -1
		// Strongest separator first, and within a separator the LATEST break,
		// so chunks run as long as they may rather than as short as they can.
		for _, sep := range cfg.SplitOn {
			at := strings.LastIndex(head, sep)
			// A period inside a sentence is not a boundary, so the search
			// keeps walking back through this separator's own occurrences
			// before it gives up and tries a weaker one. Searching head[:at]
			// skips an occurrence overlapping the held one, which no separator
			// here can have.
			for at > 0 && holds(look, at, sep, cfg) {
				at = strings.LastIndex(head[:at], sep)
			}
			if at > 0 {
				cut = len([]rune(head[:at])) + len([]rune(sep))
				break
			}
		}
		if cut <= 0 {
			// No punctuation in a whole window's worth of text. Break at the
			// last word boundary. WordBoundaries, not U+0020: NBSP survives
			// the funnel and is ordinary in real prose, so it counts as a
			// boundary here like an ordinary space.
			best := -1
			for _, b := range WordBoundaries {
				if at := strings.LastIndex(head, string(b)); at > best {
					best = at
				}
			}
			if best >= 0 {
				cut = len([]rune(head[:best]))
			}
		}
		if cut <= 0 {
			cut = budget // one unbroken token longer than a window: mid-word
		}
		// Never zero: a cut of 0 leaves rest unchanged and the loop spins.
		if cut < 1 {
			cut = 1
		}
		chunks = append(chunks, strings.TrimSpace(string(rest[:cut])))
		// TrimLeftFunc(unicode.IsSpace), not a four-character cutset: Python
		// uses lstrip() and Rust trim_start(), both of which strip ALL Unicode
		// whitespace. An NBSP left at the head of `rest` was charged against
		// the next chunk's budget and then removed from the chunk itself by
		// TrimSpace, so every split after the first one drifted: 28 of 218
		// shared cases, typically ending in a one-character chunk that becomes
		// its own utterance with its own derived seed. NBSP is ordinary in
		// real prose ("10 000", French punctuation, typeset copy).
		rest = []rune(strings.TrimLeftFunc(string(rest[cut:]), unicode.IsSpace))
	}

	out := chunks[:0]
	for _, c := range chunks {
		if c != "" {
			out = append(out, c)
		}
	}
	return out
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
	first := strings.TrimSpace(string(runes[:best]))
	second := strings.TrimSpace(string(runes[best:]))
	// Trimming can empty a half the scan thought was interior, on input whose
	// boundary run is all whitespace. SplitText trims before this is ever
	// called, so the engine cannot reach it, but this is exported and a caller
	// handed an empty half would render silence and call it speech.
	if first == "" || second == "" {
		return "", "", false
	}
	return first, second, true
}
