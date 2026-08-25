package config

import "testing"

func TestGrammarDigestMatchesTheReference(t *testing.T) {
	// The digest is of this port's own embedded copy. If it differs from the
	// Python reference the data has drifted, which is the whole reason it is
	// hashed into the fingerprint.
	const reference = "d10073beca3c0f03"
	if got := GrammarDigest(); got != reference {
		t.Errorf("grammar digest = %s, reference = %s — this port's numbers.json has drifted", got, reference)
	}
}
