package engine

import (
	"testing"

	"github.com/loudreader/loudkit/go/postprocess"
)

// TestTheRetryLadderHeadroomMatchesTheSeedStreams pins the restated constant to
// the two it is derived from.
//
// postprocess.RetryLadderHeadroom is the gap between the retry stream and the
// chunk streams, and it bounds retry_max_attempts: past it a retry's derived
// seed is a chunk's seed, so two different windows render from one stream. It
// lives in postprocess because that is where the refusal is, and it cannot be
// computed there without importing this package. The reference restates it the
// same way and pins it the same way.
func TestTheRetryLadderHeadroomMatchesTheSeedStreams(t *testing.T) {
	if got := chunkStreamBase - retryStreamBase; got != postprocess.RetryLadderHeadroom {
		t.Errorf("the seed ladder leaves %d retry streams, postprocess says %d",
			got, postprocess.RetryLadderHeadroom)
	}
}
