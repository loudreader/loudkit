// Package digest is the module's one file hash, so the download's integrity
// check and the checkpoint's export record read the same bytes the same way.
package digest

import (
	"crypto/sha256"
	"encoding/hex"
	"io"
	"os"
)

// SHA256File is the hex sha256 of a file, streamed.
func SHA256File(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	// A megabyte at a time: the checkpoint is 747 MB.
	if _, err := io.CopyBuffer(h, f, make([]byte, 1<<20)); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
