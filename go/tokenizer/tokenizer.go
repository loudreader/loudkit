// Package tokenizer implements the grapheme BPE tokenizer — a bit-parity port
// of loudkit's frontend over the HF tokenizers.json format.
//
// The algorithm mirrors the JS port (@huggingface/tokenizers) exactly:
//  1. split the input on added-token contents (a trie dictionary splitter);
//  2. emit added tokens as single tokens;
//  3. Whitespace-pretokenize the rest (`\w+|[^\w\s]+`);
//  4. BPE-encode each pretoken against the merges.
//
// The conformance fixture pins the result; a drift here makes free-running
// tokens diverge from the Python engine.
package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"unicode"
)

// AddedToken is one entry of the tokenizer's added_tokens list.
type AddedToken struct {
	ID      int
	Content string
	Special bool
	LStrip  bool
	RStrip  bool
}

// ModelConfig is the BPE model block of tokenizer.json.
type ModelConfig struct {
	Type                    string
	UnkToken                string
	ContinuingSubwordPrefix string
	EndOfWordSuffix         string
	FuseUnk                 bool
	Vocab                   map[string]int
	Merges                  [][2]string
}

// TokenizerJSON is the parsed tokenizer.json.
type TokenizerJSON struct {
	Model        ModelConfig
	AddedTokens  []AddedToken
	PreTokenizer map[string]interface{}
	Normalizer   map[string]interface{}
}

// ParseJSON loads a tokenizer.json.
func ParseJSON(path string) (*TokenizerJSON, error) {
	buf, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var raw map[string]interface{}
	if err := json.Unmarshal(buf, &raw); err != nil {
		return nil, fmt.Errorf("%s: bad tokenizer JSON: %w", path, err)
	}
	t := &TokenizerJSON{}
	model, _ := raw["model"].(map[string]interface{})
	t.Model.Type, _ = model["type"].(string)
	t.Model.UnkToken, _ = model["unk_token"].(string)
	t.Model.ContinuingSubwordPrefix, _ = model["continuing_subword_prefix"].(string)
	t.Model.EndOfWordSuffix, _ = model["end_of_word_suffix"].(string)
	if v, ok := model["fuse_unk"].(bool); ok {
		t.Model.FuseUnk = v
	}
	t.Model.Vocab = map[string]int{}
	if v, ok := model["vocab"].(map[string]interface{}); ok {
		for k, id := range v {
			t.Model.Vocab[k] = int(toFloat(id))
		}
	}
	if v, ok := model["merges"].([]interface{}); ok {
		for _, m := range v {
			pair := [2]string{}
			switch mm := m.(type) {
			case string:
				parts := strings.SplitN(mm, " ", 2)
				if len(parts) == 2 {
					pair = [2]string{parts[0], parts[1]}
				}
			case []interface{}:
				if len(mm) == 2 {
					pair = [2]string{toString(mm[0]), toString(mm[1])}
				}
			}
			if pair[0] != "" {
				t.Model.Merges = append(t.Model.Merges, pair)
			}
		}
	}
	if v, ok := raw["added_tokens"].([]interface{}); ok {
		for i, a := range v {
			// Comma-ok, like every other assertion in this file: a
			// tokenizer.json whose added_tokens holds anything but objects is
			// bad input, and a library reports it rather than panicking.
			at, ok := a.(map[string]interface{})
			if !ok {
				return nil, fmt.Errorf("added_tokens[%d]: expected an object, got %T", i, a)
			}
			t.AddedTokens = append(t.AddedTokens, AddedToken{
				ID:      int(toFloat(at["id"])),
				Content: toString(at["content"]),
				Special: boolVal(at["special"]),
				LStrip:  boolVal(at["lstrip"]),
				RStrip:  boolVal(at["rstrip"]),
			})
		}
	}
	if v, ok := raw["pre_tokenizer"].(map[string]interface{}); ok {
		t.PreTokenizer = v
	}
	return t, nil
}

func toFloat(x interface{}) float64 {
	switch v := x.(type) {
	case float64:
		return v
	case int:
		return float64(v)
	}
	return 0
}

func toString(x interface{}) string {
	if s, ok := x.(string); ok {
		return s
	}
	return ""
}

func boolVal(x interface{}) bool {
	if b, ok := x.(bool); ok {
		return b
	}
	return false
}

// Tokenizer is a loaded, runnable tokenizer.
type Tokenizer struct {
	vocab     map[string]int
	unkToken  string
	unkID     int
	merges    map[[2]string]int
	addedMap  map[string]int
	splitter  *dictSplitter
	endOfWord string
}

// New builds a runnable tokenizer from the parsed JSON.
func New(t *TokenizerJSON) *Tokenizer {
	tk := &Tokenizer{
		vocab:     t.Model.Vocab,
		unkToken:  t.Model.UnkToken,
		merges:    map[[2]string]int{},
		addedMap:  map[string]int{},
		endOfWord: t.Model.EndOfWordSuffix,
	}
	for i, m := range t.Model.Merges {
		tk.merges[m] = i
	}
	unkID, ok := t.Model.Vocab[t.Model.UnkToken]
	if ok {
		tk.unkID = unkID
	}
	contents := []string{}
	for _, a := range t.AddedTokens {
		tk.addedMap[a.Content] = a.ID
		contents = append(contents, a.Content)
	}
	tk.splitter = newDictSplitter(contents)
	return tk
}

// MaxID is the largest id this vocabulary can emit, added tokens included.
//
// Every id from Encode indexes the checkpoint's text embedding table directly,
// so this is the number the engine checks that table against.
func (t *Tokenizer) MaxID() int {
	max := 0
	for _, id := range t.vocab {
		if id > max {
			max = id
		}
	}
	for _, id := range t.addedMap {
		if id > max {
			max = id
		}
	}
	return max
}

// Encode returns the token ids for a pre-tagged, normalised string.
func (t *Tokenizer) Encode(text string) []int {
	ids := []int{}
	for _, section := range t.splitter.split(text) {
		if id, ok := t.addedMap[section]; ok {
			ids = append(ids, id)
			continue
		}
		for _, pretoken := range whitespaceRegex(section) {
			for _, sub := range t.bpe(pretoken) {
				if id, ok := t.vocab[sub]; ok {
					ids = append(ids, id)
				} else {
					ids = append(ids, t.unkID)
				}
			}
		}
	}
	return ids
}

// bpe splits one pretoken into subwords, exactly like the reference.
func (t *Tokenizer) bpe(token string) []string {
	if token == "" {
		return nil
	}
	cur := make([]string, 0, len(token))
	for _, r := range token {
		cur = append(cur, string(r))
	}
	if t.endOfWord != "" && len(cur) > 0 {
		cur[len(cur)-1] += t.endOfWord
	}
	if len(cur) == 1 {
		return cur
	}

	// classic bottom-up merges: repeatedly merge the lowest-rank adjacent pair
	for {
		bestRank := -1
		bestIdx := -1
		for i := 0; i+1 < len(cur); i++ {
			rank, ok := t.merges[[2]string{cur[i], cur[i+1]}]
			if ok && (bestRank == -1 || rank < bestRank) {
				bestRank = rank
				bestIdx = i
			}
		}
		if bestIdx == -1 {
			break
		}
		merged := cur[bestIdx] + cur[bestIdx+1]
		next := make([]string, 0, len(cur)-1)
		next = append(next, cur[:bestIdx]...)
		next = append(next, merged)
		next = append(next, cur[bestIdx+2:]...)
		cur = next
	}
	return cur
}

// whitespaceRegex implements the Whitespace pre-tokenizer: \w+|[^\w\s]+.
// \w is Unicode-aware (any letter, digit or underscore), matching the Rust
// reference regex the fixture was generated with.
func whitespaceRegex(text string) []string {
	out := []string{}
	runes := []rune(text)
	var cur []rune
	var curKind int // 1 = word, 2 = non-word-non-space
	flush := func() {
		if len(cur) > 0 {
			out = append(out, string(cur))
			cur = nil
		}
	}
	for _, r := range runes {
		var kind int
		switch {
		case isWord(r):
			kind = 1
		case r == ' ' || r == '\t' || r == '\n' || r == '\r' || r == '\f' || r == '\v':
			kind = 0 // whitespace drops out
		default:
			kind = 2
		}
		if kind == 0 {
			flush()
			continue
		}
		if len(cur) > 0 && curKind != kind {
			flush()
		}
		cur = append(cur, r)
		curKind = kind
	}
	flush()
	return out
}

func isWord(r rune) bool {
	return r == '_' || unicode.IsLetter(r) || unicode.IsDigit(r) || unicode.IsMark(r)
}

// dictSplitter splits text on dictionary strings, longest match, char indices.
type dictSplitter struct {
	trie map[rune]interface{}
}

func newDictSplitter(words []string) *dictSplitter {
	trie := map[rune]interface{}{}
	for _, w := range words {
		node := trie
		for _, c := range w {
			next, ok := node[c].(map[rune]interface{})
			if !ok {
				next = map[rune]interface{}{}
				node[c] = next
			}
			node = next
		}
		node['\x00'] = w // end marker
	}
	return &dictSplitter{trie: trie}
}

func (d *dictSplitter) split(text string) []string {
	runes := []rune(text)
	result := []string{}
	start := 0
	i := 0
	for i < len(runes) {
		node := d.trie
		var match *string
		j := i
		for j < len(runes) {
			child, ok := node[runes[j]].(map[rune]interface{})
			if !ok {
				break
			}
			node = child
			if end, ok := node['\x00'].(string); ok {
				m := end
				match = &m
			}
			j++
		}
		if match != nil {
			if i > start {
				result = append(result, string(runes[start:i]))
			}
			result = append(result, *match)
			i += len([]rune(*match))
			start = i
		} else {
			i++
		}
	}
	if start < len(runes) {
		result = append(result, string(runes[start:]))
	}
	return result
}
