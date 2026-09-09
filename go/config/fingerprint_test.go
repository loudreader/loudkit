package config

import "testing"

// TestAnIDListIsHashedInTheOrderTheManifestGaveIt pins the one thing the
// fingerprint may not do: normalise.
//
// This form used to sort the three id lists before hashing them, on the theory
// that the manifest's order is whatever the packer wrote. Python, Rust and
// Swift hash them as written, nothing sorts on the way in, and the packer does
// not sort either, so one file had two fingerprints depending on which port
// was asked. The number that exists to catch cross-port drift cannot be the
// thing that drifts.
//
// The values are Python's, from loudkit.config.AlgorithmConfig.
func TestAnIDListIsHashedInTheOrderTheManifestGaveIt(t *testing.T) {
	base := shipped011()
	for _, c := range []struct {
		name string
		ids  []interface{}
		want string
	}{
		{"ascending", []interface{}{float64(1), float64(5), float64(9)}, "47a50fe9fe6f42bf"},
		{"unsorted", []interface{}{float64(9), float64(1), float64(5)}, "38bcc642490920d4"},
	} {
		m := make(map[string]interface{}, len(base))
		for k, v := range base {
			m[k] = v
		}
		m["silence_token_ids"] = c.ids
		if got := Fingerprint(mustFromManifest(t, m)); got != c.want {
			t.Errorf("%s: fingerprint %s, want %s", c.name, got, c.want)
		}
	}
}
