// Package chunking mirrors loudkit.frontend.chunking: splitting text that is
// longer than one window.
//
// A window carries about 255 speech tokens, roughly ten seconds. Anything
// longer has to be split, generated in pieces and joined, and *where* the
// splits fall is audible — a break at a full stop is inaudible, a break
// mid-clause is not. That makes it an algorithm-layer decision rather than a
// caller's convenience, and it has to be identical in every port: a different
// split is a different set of joins and therefore a different reading.
//
// This binding had no long-form path at all while the documentation described
// it as supported and conformance-verified.//
// Python reference: `loudkit/frontend/chunking.py`.
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
// limit, never to predict a length — the middle of the range would let the
// worst case overflow the window, and an overflow is a hard failure.
//
// It must equal loudkit.frontend.chunking.CHARS_PER_TOKEN.
const CharsPerToken = 0.5

// Config is the chunking policy, from the checkpoint's AlgorithmConfig.
type Config struct {
	Enabled      bool
	MaxTokens    int
	PrefixTokens int
	SplitOn      []string
}

// Validate returns the same refusal Python raises, so a user who hits it in
// two languages reads the same sentence twice.
//
// Validate the recipe, the way loudkit.config.ChunkConfig.__post_init__ does.
//
// Python refuses four configurations here, and the ports were plain structs that
// read MaxTokens straight from the manifest and accepted all of them. The second
// is the one that matters: d8742aa fixed "split_text hangs forever on a config
// the validator accepts" on the Python side only — a MaxTokens small enough that
// int(MaxTokens * CharsPerToken) is zero makes the splitter cut nothing and loop
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
	return nil
}

// Production is the shipping chunking recipe.
func Production() Config {
	return Config{
		Enabled:      true,
		MaxTokens:    255,
		PrefixTokens: 6,
		SplitOn:      []string{". ", "! ", "? ", "; ", ", "},
	}
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
		cut := -1
		// Strongest separator first, and within a separator the LATEST break,
		// so chunks run as long as they may rather than as short as they can.
		for _, sep := range cfg.SplitOn {
			if at := strings.LastIndex(head, sep); at > 0 {
				cut = len([]rune(head[:at])) + len([]rune(sep))
				break
			}
		}
		if cut <= 0 {
			// No punctuation in a whole window's worth of text. Break at the
			// last word boundary; it will be heard, and that is the point.
			if at := strings.LastIndex(head, " "); at >= 0 {
				cut = len([]rune(head[:at]))
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
		// TrimSpace, so every split after the first one drifted — 28 of 218
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
