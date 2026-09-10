package frontend

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A tokenizer.json this build cannot encode with is refused at Load.
//
// Before the refusal, a file with no model block parsed cleanly: every index of
// the nil map answered with a zero value, the vocabulary stayed empty, and
// Encode returned unk id 0 for a whole passage. The engine then rendered that
// passage as sound, with no error anywhere. Python refuses the same three
// missing names at load, in loudkit.frontend.text.
func TestLoadRefusesATokenizerEncodeCannotUse(t *testing.T) {
	for _, tc := range []struct {
		name string
		json string
		want string
	}{
		{
			name: "no model block",
			json: `{"version":"1.0","added_tokens":[]}`,
			want: "no model block",
		},
		{
			name: "model is not an object",
			json: `{"version":"1.0","model":"bpe"}`,
			want: "no model block",
		},
		{
			name: "missing [START]",
			json: `{"model":{"type":"BPE","vocab":{"[STOP]":1,"[SPACE]":2},"merges":[]}}`,
			want: `missing "[START]"`,
		},
		{
			name: "missing [STOP]",
			json: `{"model":{"type":"BPE","vocab":{"[START]":0,"[SPACE]":2},"merges":[]}}`,
			want: `missing "[STOP]"`,
		},
		{
			name: "missing [SPACE]",
			json: `{"model":{"type":"BPE","vocab":{"[START]":0,"[STOP]":1},"merges":[]}}`,
			want: `missing "[SPACE]"`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tokenizer.json")
			if err := os.WriteFile(path, []byte(tc.json), 0o600); err != nil {
				t.Fatal(err)
			}
			f, err := Load(path)
			if err == nil {
				ids, encErr := f.Encode("hello there", "en")
				t.Fatalf("loaded a tokenizer it cannot encode with; Encode gave %v, %v", ids, encErr)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %q does not name the problem (%q)", err, tc.want)
			}
		})
	}
}

// The three names may arrive as added tokens rather than in the model
// vocabulary, which is where the shipped file carries them too. Python reads
// get_vocab, which is the union, so the check has to be the union.
func TestLoadAcceptsTheRequiredNamesAsAddedTokens(t *testing.T) {
	const json = `{"model":{"type":"BPE","vocab":{"a":3},"merges":[]},
	"added_tokens":[{"id":0,"content":"[START]"},{"id":1,"content":"[STOP]"},
	{"id":2,"content":"[SPACE]"}]}`
	path := filepath.Join(t.TempDir(), "tokenizer.json")
	if err := os.WriteFile(path, []byte(json), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(path); err != nil {
		t.Fatalf("added tokens carry the three names, so this must load: %v", err)
	}
}

// The shipped tokenizer still loads: the refusal is for files Python already
// refuses, and nothing that worked stops working.
func TestLoadAcceptsTheShippedTokenizer(t *testing.T) {
	path := filepath.Join("..", "..", "tests", "data", "conformance", "tokenizer.json")
	if _, err := os.Stat(path); err != nil {
		t.Skip("conformance tokenizer not found: " + path)
	}
	if _, err := Load(path); err != nil {
		t.Fatalf("the shipped tokenizer must load: %v", err)
	}
}
