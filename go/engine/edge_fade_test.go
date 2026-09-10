package engine

import (
	"testing"

	"github.com/loudreader/loudkit/go/config"
	"github.com/loudreader/loudkit/go/timestretch"
)

// config cannot import timestretch, so it carries a copy of the shipped ramp
// length; this is what keeps the copy honest.
func TestTheShippedRampIsOneNumber(t *testing.T) {
	cfg, err := config.FromManifest(map[string]interface{}{"edge_fade_seconds": 0.02})
	if err != nil {
		t.Fatal(err)
	}
	if cfg.EdgeFade() != timestretch.EdgeFadeSeconds {
		t.Fatalf("config ships %v, timestretch ships %v", cfg.EdgeFade(), timestretch.EdgeFadeSeconds)
	}
}
