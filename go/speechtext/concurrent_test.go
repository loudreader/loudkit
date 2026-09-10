package speechtext

import (
	"sync"
	"testing"
)

// The number grammars load lazily, and a server reaches every entry point below
// from many goroutines at once. A `grammars == nil` check is not enough to
// guard that: a second caller reads the map while the first is still filling
// it, which is a torn read of the map header, a missing grammar for a language
// whose row has not landed yet, and two callers inside the loader at once
// writing the same map. Run under -race, which is what makes this a gate
// rather than a smoke test.
//
// The Once and the table are reset first so the probe drives the lazy load
// itself, whatever ran before it in this binary. The reset happens before the
// goroutines start, so it carries its own happens-before edge; the package has
// no parallel tests, so nothing else is reading the tables here.
func TestGrammarsLoadOnceUnderConcurrentCallers(t *testing.T) {
	const goroutines = 8

	wantPrepared := Prepared("She paid 1234 dollars on 3 March 2021.", "en")
	wantCardinal, err := Cardinal(1234, "pl", "")
	if err != nil {
		t.Fatalf("Cardinal(1234, pl): %v", err)
	}
	wantFolded := FoldForeignDigits("٣٫١٤", "de")
	wantLanguages := SupportedNumberLanguages()

	grammarsOnce = sync.Once{}
	grammars = nil

	type answer struct {
		prepared  string
		cardinal  string
		folded    string
		languages []string
	}
	got := make([]answer, goroutines)

	var start, done sync.WaitGroup
	start.Add(1)
	done.Add(goroutines)
	for i := range got {
		go func(i int) {
			defer done.Done()
			start.Wait()
			card, cardErr := Cardinal(1234, "pl", "")
			if cardErr != nil {
				card = "error: " + cardErr.Error()
			}
			got[i] = answer{
				prepared:  Prepared("She paid 1234 dollars on 3 March 2021.", "en"),
				cardinal:  card,
				folded:    FoldForeignDigits("٣٫١٤", "de"),
				languages: SupportedNumberLanguages(),
			}
		}(i)
	}
	start.Done()
	done.Wait()

	for i, a := range got {
		if a.prepared != wantPrepared {
			t.Errorf("goroutine %d: Prepared = %q, want %q", i, a.prepared, wantPrepared)
		}
		if a.cardinal != wantCardinal {
			t.Errorf("goroutine %d: Cardinal = %q, want %q", i, a.cardinal, wantCardinal)
		}
		if a.folded != wantFolded {
			t.Errorf("goroutine %d: FoldForeignDigits = %q, want %q", i, a.folded, wantFolded)
		}
		if len(a.languages) != len(wantLanguages) {
			t.Errorf("goroutine %d: SupportedNumberLanguages has %d ids, want %d",
				i, len(a.languages), len(wantLanguages))
			continue
		}
		for j, lang := range a.languages {
			if lang != wantLanguages[j] {
				t.Errorf("goroutine %d: SupportedNumberLanguages[%d] = %q, want %q",
					i, j, lang, wantLanguages[j])
			}
		}
	}
}
