package chunking

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

type chunkCase struct {
	Config            string   `json:"config"`
	MaxTokens         int      `json:"max_tokens"`
	PrefixTokens      int      `json:"prefix_tokens"`
	SplitOn           []string `json:"split_on"`
	Abbreviations     []string `json:"abbreviations"`
	MidSentencePeriod string   `json:"mid_sentence_period"`
	Text              string   `json:"text"`
	Chunks            []string `json:"chunks"`
}

// The splitter must cut where the shared fixture says.
//
// Where the splits fall is audible, so a different split is a different
// reading, not a formatting choice. This port had no long-form path at all
// while the documentation called it supported and conformance-verified.
func TestSplitTextMatchesTheSharedFixture(t *testing.T) {
	dir := os.Getenv("LOUDKIT_FIXTURE_DIR")
	if dir == "" {
		dir = filepath.Join("..", "..", "tests", "data", "conformance")
	}
	raw, err := os.ReadFile(filepath.Join(dir, "speechtext.json"))
	if err != nil {
		t.Fatalf("fixture not found: %v", err)
	}
	var payload struct {
		Chunking []chunkCase `json:"chunking"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatal(err)
	}
	if len(payload.Chunking) == 0 {
		t.Fatal("the fixture carries no chunking cases")
	}

	for _, c := range payload.Chunking {
		got := SplitText(c.Text, Config{
			Enabled:           true,
			MaxTokens:         c.MaxTokens,
			PrefixTokens:      c.PrefixTokens,
			SplitOn:           c.SplitOn,
			Abbreviations:     c.Abbreviations,
			MidSentencePeriod: c.MidSentencePeriod,
		})
		if len(got) != len(c.Chunks) {
			t.Fatalf("%s: %d chunks, want %d\n got: %q\nwant: %q",
				c.Config, len(got), len(c.Chunks), got, c.Chunks)
		}
		for i := range got {
			if got[i] != c.Chunks[i] {
				t.Fatalf("%s chunk %d:\n got: %q\nwant: %q",
					c.Config, i, got[i], c.Chunks[i])
			}
		}
	}
}

// The constant is shared arithmetic, not a tuning knob: a port that picks a
// different value splits in different places and reads the text differently.
func TestCharsPerTokenMatchesPython(t *testing.T) {
	if CharsPerToken != 0.5 {
		t.Fatalf("CharsPerToken is %v; loudkit.frontend.chunking.CHARS_PER_TOKEN is 0.5", CharsPerToken)
	}
}

// TestValidateRefusesTheConfigsPythonRefuses pins the four refusals
// loudkit.config.ChunkConfig.__post_init__ makes.
//
// The zero-budget one is why it matters: without the refusal SplitText cuts
// nothing and loops forever, which on a server is a wedged request holding the
// single-flight engine.
func TestValidateRefusesTheConfigsPythonRefuses(t *testing.T) {
	good := Production()
	if err := good.Validate(); err != nil {
		t.Fatalf("the shipping recipe must validate: %v", err)
	}
	for _, c := range []struct {
		name string
		cfg  Config
		want string
	}{
		{"zero max", Config{MaxTokens: 0, SplitOn: []string{". "}}, "must be positive"},
		{"no budget", Config{MaxTokens: 1, SplitOn: []string{". "}}, "no character budget"},
		{"prefix >= max", Config{MaxTokens: 20, PrefixTokens: 20, SplitOn: []string{". "}},
			"prefix_tokens must be in"},
		{"no separators", Config{MaxTokens: 20, PrefixTokens: 6}, "nowhere to break"},
		{"unknown law", Config{MaxTokens: 20, PrefixTokens: 6, SplitOn: []string{". "},
			MidSentencePeriod: "hold-ish"}, "unknown mid_sentence_period"},
		{"empty abbreviation", Config{MaxTokens: 20, PrefixTokens: 6, SplitOn: []string{". "},
			MidSentencePeriod: HoldMidSentencePeriod, CapResplit: WordCapResplit,
			Abbreviations: []string{"Mr", ""}}, "empty string"},
		{"unknown resplit", Config{MaxTokens: 20, PrefixTokens: 6, SplitOn: []string{". "},
			MidSentencePeriod: HoldMidSentencePeriod, CapResplit: "halve"},
			"unknown cap_resplit"},
	} {
		err := c.cfg.Validate()
		if err == nil {
			t.Errorf("%s: accepted a config Python refuses", c.name)
			continue
		}
		if !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: got %q, want it to mention %q", c.name, err, c.want)
		}
	}
}

// A period that does not end a sentence must not end a chunk.
//
// Without the rule "But Mr. Smith went home" breaks after the title and hands
// the renderer a seven-character chunk: its own utterance, its own derived
// seed, and a token ceiling proportional to seven characters. It is the only
// chunk in a 9920-row rendered census that hit that ceiling. Surveyed over
// 1200 passages in ten languages, 59 of 3773 cuts landed on a period inside a
// sentence; under this law, one.
func TestMidSentencePeriodsAreNotBoundaries(t *testing.T) {
	const title = "But Mr. Smith went home to the house on the hill where he had lived " +
		"for forty years without ever once complaining about any of it at all."

	if got := SplitText(title, Production()); got[0] == "But Mr." {
		t.Fatalf("the hold did not fire: %q", got)
	}

	// The old law, kept namable so a pack can say what it was measured under.
	old := Production()
	old.MidSentencePeriod = BreakMidSentencePeriod
	if got := SplitText(title, old); got[0] != "But Mr." {
		t.Fatalf("%q does not reproduce the old law: %q", BreakMidSentencePeriod, got)
	}

	// The half of the law that no list could do: `speech_text` folds a
	// mid-sentence ellipsis to a single period, and the next word is lower
	// case. It is the dominant cause in Polish, which has no abbreviation cuts.
	const ellipsis = "Grzeja sie i swieca. ciepłem ktore pamietaja z lata i z kazdej " +
		"innej pory roku na swiecie, a potem gasna powoli i nikt juz nie pamieta."
	if got := SplitText(ellipsis, Production()); strings.HasSuffix(got[0], "swieca.") {
		t.Fatalf("the lower-case test did not fire: %q", got)
	}

	// Gated on the period. A comma is followed by a lower-case word almost
	// every time it is written, so a rule that did not gate would veto every
	// comma in the language.
	const commas = "Alpha beta gamma delta, epsilon zeta eta theta, iota kappa lambda mu, " +
		"nu xi omicron pi rho, sigma tau upsilon phi chi psi omega at the end."
	held, broke := SplitText(commas, Production()), SplitText(commas, old)
	if len(held) != len(broke) || held[0] != broke[0] {
		t.Fatalf("the comma split moved: %q vs %q", held, broke)
	}

	// "NASA" ends in "A", and "A" is a listed initial; the guard on the
	// character in front of the match is what keeps this one breaking.
	const nasa = "The rocket that carried them up there was built by NASA. And the rest " +
		"of the afternoon went by without anybody saying much about it to anyone."
	if got := SplitText(nasa, Production()); !strings.HasSuffix(got[0], "by NASA.") {
		t.Fatalf("the guard held a word ending: %q", got)
	}

	// Every sentence end in the window is held, so the split falls through to
	// the latest comma. A comma break is heard; a chunk of "Mr." is heard worse.
	const norrell = "Mr. Norrell, who had been waiting in the hall for the better part " +
		"of an hour, said nothing at all to either of them about what he had seen there."
	if got := SplitText(norrell, Production()); !strings.HasSuffix(got[0], "hour,") {
		t.Fatalf("the search did not fall through to a weaker separator: %q", got)
	}
}

// TestWordBoundaryFallback covers SplitText's last resort, when a window holds
// no punctuation. The boundary table was introduced for SplitInHalf and this
// fallback, ten lines away, was left matching U+0020 alone, so text whose
// every space is non-breaking was cut mid-word.
func TestWordBoundaryFallback(t *testing.T) {
	cfg := Production()
	words := make([]string, 30)
	for i := range words {
		words[i] = fmt.Sprintf("ord%02d", i)
	}
	chunks := SplitText(strings.Join(words, "\u00a0"), cfg)
	if len(chunks) < 2 {
		t.Fatalf("the case needs to cross a window, got %d chunks", len(chunks))
	}
	for _, c := range chunks {
		parts := strings.Split(c, "\u00a0")
		last := parts[len(parts)-1]
		if !slices.Contains(words, last) {
			t.Fatalf("cut mid-word: %q", last)
		}
	}
}

// TestSplitInHalf is the repair for a chunk the window could not hold. The
// Python reference's TestSplitInHalf, case for case.
func TestSplitInHalf(t *testing.T) {
	if a, b, ok := SplitInHalf("one two three four five six"); !ok ||
		a+" "+b != "one two three four five six" {
		t.Fatalf("halving lost text: %q | %q ok=%v", a, b, ok)
	}
	// Punctuation is not sought. A weaker separator sits nearer the middle
	// than the comma does, and the comma has no pull of its own: of 27
	// re-splits taken at the separator nearest the middle, every seam over a
	// second fell on a comma.
	if a, b, ok := SplitInHalf("aa bb, cc dddddddddddd ee"); !ok ||
		a != "aa bb, cc" || b != "dddddddddddd ee" {
		t.Fatalf("sought punctuation: %q | %q ok=%v", a, b, ok)
	}
	// CRLF is one grapheme cluster and two Unicode scalars; Swift's
	// Array(text) yields clusters and halved this one word later until it was
	// switched to unicodeScalars. The funnel passes CRLF through verbatim.
	if a, b, ok := SplitInHalf("xx\r\nxx xx xxxxx"); !ok ||
		a != "xx\r\nxx" || b != "xx xxxxx" {
		t.Fatalf("scalar counting: %q | %q ok=%v", a, b, ok)
	}
	// NBSP and tab survive the funnel and are word boundaries. Matching only
	// U+0020 made a capped chunk whose separators were all non-breaking come
	// back unsplittable, so it shipped its truncation.
	for _, sep := range []string{"\u00a0", "\t", "\u202f"} {
		a, b, ok := SplitInHalf("alpha" + sep + "beta" + sep + "gamma")
		if !ok || a != "alpha"+sep+"beta" || b != "gamma" {
			t.Fatalf("boundary %q: %q | %q ok=%v", sep, a, b, ok)
		}
	}
	// A single unbroken run: splitting it would have to cut a word, which is
	// worse than the truncation it would be repairing.
	// Trimming can empty a half the scan thought was interior. Unreachable
	// through the engine, which trims first, but this is exported.
	for _, text := range []string{"omringden.", "", "x \t"} {
		if _, _, ok := SplitInHalf(text); ok {
			t.Fatalf("split an unbreakable run %q", text)
		}
	}
	for _, text := range []string{"a bb", "aaaaaaaa b", "a bbbbbbbb"} {
		a, b, ok := SplitInHalf(text)
		if !ok || a == "" || b == "" {
			t.Fatalf("empty half for %q: %q | %q ok=%v", text, a, b, ok)
		}
	}
}
