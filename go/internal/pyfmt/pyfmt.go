// Package pyfmt renders a value the way the reference implementation prints
// it, so a refusal reads the same in Go as in Python and a canonical form
// hashes the same bytes.
package pyfmt

import (
	"strconv"
	"strings"
)

// Float renders a float the way Python's repr() does.
//
// 'g' with -1 precision gives the shortest decimal that round-trips, which is
// what repr() gives: except that Go renders 25.0 as "25" while Python renders
// it "25.0". That one character is the difference between a matching
// fingerprint and a mysterious one, and between a refusal that reads like the
// reference's and one a reader has to compare by hand.
func Float(v float64) string {
	s := strconv.FormatFloat(v, 'g', -1, 64)
	if !strings.ContainsAny(s, ".eEni") { // no point, no exponent, not inf/nan
		s += ".0"
	}
	return s
}
