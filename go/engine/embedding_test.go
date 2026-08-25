package engine

import (
	"strings"
	"testing"
)

// A tokenizer wider than the checkpoint refuses at the door, naming it.
//
// textRow indexes the embedding table by raw token id. Paired with a checkpoint
// from another release the widest ids read past the end of it — an out-of-range
// panic several seconds into a synthesis, pointing at neither of the two files
// the caller chose.
func TestAnIDPastTheEmbeddingTableIsRefusedAtLoad(t *testing.T) {
	flat := 4 * hiddenDim
	if err := embeddingFits("text", 3, flat, "tokenizer.json"); err != nil {
		t.Fatalf("the last row is in range: %v", err)
	}
	err := embeddingFits("text", 4, flat, "tokenizer.json")
	if err == nil {
		t.Fatal("an id one past the table must be refused")
	}
	if !strings.Contains(err.Error(), "tokenizer.json") || !strings.Contains(err.Error(), "4 rows") {
		t.Fatalf("the message must name the file and the table: %v", err)
	}
}
