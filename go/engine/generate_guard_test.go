package engine

import (
	"strings"
	"testing"

	"github.com/loudreader/loudkit/go/chunking"
	"github.com/loudreader/loudkit/go/config"
)

// Generate is a public door and it indexed two tables with the ids it was
// handed: the repetition table with every prefix id, the text embedding table
// with every text id. An out-of-range id was an index-out-of-range
// panic inside the library, which no caller can handle, while carryFrom next
// to it already refused exactly the same input with a sentence.
//
// The engine here carries the two tables and a config and nothing else: the
// guards run before the first graph is reached, so a refused call never gets
// as far as needing one, and carry_test.go builds its engine the same way.
func guardEngine(textRows int) *Engine {
	return &Engine{
		config: config.AlgorithmConfig{
			StartSpeech:     6561,
			SpeechVocabSize: 8194,
			Chunking:        chunking.Config{Enabled: true, MaxTokens: 255, PrefixTokens: 6},
		},
		textEmb: make([]float32, textRows*hiddenDim),
		textPos: make([]float32, textRows*hiddenDim),
	}
}

func TestGenerateRefusesIdsItWouldIndexTablesWith(t *testing.T) {
	const rows = 16
	e := guardEngine(rows)

	for _, tc := range []struct {
		name       string
		textTokens []int
		prefix     []int
		want       string
	}{
		{
			name:       "a negative text id",
			textTokens: []int{1, -1, 2},
			want:       "text embedding table",
		},
		{
			name:       "a text id one past the table",
			textTokens: []int{1, rows, 2},
			want:       "text embedding table",
		},
		{
			name:       "more text tokens than there are positions",
			textTokens: make([]int, rows),
			want:       "text positions",
		},
		{
			name:       "a negative carried id",
			textTokens: []int{1, 2},
			prefix:     []int{5, -1},
			want:       "not an acoustic speech token",
		},
		{
			name:       "a carried id at the control tokens",
			textTokens: []int{1, 2},
			prefix:     []int{5, 6561},
			want:       "not an acoustic speech token",
		},
		{
			name:       "a carried id past the speech vocabulary",
			textTokens: []int{1, 2},
			prefix:     []int{5, 8194},
			want:       "not an acoustic speech token",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := e.Generate(tc.textTokens, nil, nil, nil, nil, tc.prefix)
			if err == nil {
				t.Fatal("expected a refusal, got nil")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %q does not say %q", err, tc.want)
			}
		})
	}
}

// The refusal is only for ids that would have crashed. A row the tables hold
// passes both guards, so nothing that rendered before stops rendering.
func TestTheGuardsAcceptARowTheTablesHold(t *testing.T) {
	const rows = 16
	e := guardEngine(rows)

	if err := e.checkTextRow([]int{0, 1, rows - 1}); err != nil {
		t.Errorf("a text row inside the table was refused: %v", err)
	}
	if err := e.checkTextRow(make([]int, rows-2)); err != nil {
		t.Errorf("a row exactly filling the positions was refused: %v", err)
	}
	if err := e.checkSpeechTokens(nil); err != nil {
		t.Errorf("no carried tokens was refused: %v", err)
	}
	if err := e.checkSpeechTokens([]int{0, 6560}); err != nil {
		t.Errorf("carried tokens inside the codebook were refused: %v", err)
	}
}

// One sentence for one refusal: Generate's carried-token check is carryFrom's,
// so a caller reading either door meets the same words.
func TestTheCarriedTokenRefusalIsOneSentence(t *testing.T) {
	e := guardEngine(16)
	_, viaCarry := e.carryFrom([]int{1, -1})
	viaGenerate := e.checkSpeechTokens([]int{1, -1})
	if viaCarry == nil || viaGenerate == nil {
		t.Fatal("both doors must refuse a negative id")
	}
	if viaCarry.Error() != viaGenerate.Error() {
		t.Fatalf("two sentences for one refusal:\ncarryFrom: %v\nGenerate:  %v", viaCarry, viaGenerate)
	}
}
