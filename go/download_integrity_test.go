package loudkit

import (
	"bytes"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestNoChecksumsMeansNoVerifiedLine pins what the progress writer is allowed
// to claim.
//
// The count used to rise once per file the loop walked, while verifyFetched
// returns before hashing anything when the repo ships no SHA256SUMS. A repo
// with nothing vouching for it therefore ended with "loudkit: verified 15
// files against SHA256SUMS", which is the sentence a reader checks before
// trusting a directory.
func TestNoChecksumsMeansNoVerifiedLine(t *testing.T) {
	repo := newFakeRepo("someone/loudr-1")
	delete(repo.files, sumsName)
	repo.serve(t)

	var progress bytes.Buffer
	dir := filepath.Join(t.TempDir(), "loudr-1")
	if _, err := DownloadWith("someone/loudr-1", dir, Fetch{Progress: &progress}); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(progress.String(), "verified") {
		t.Errorf("a repo with no %s reported:\n%s", sumsName, progress.String())
	}
}

// TestAnUnlistedFileIsNotCountedAsVerified is the same rule one level down: a
// stranger's repo that ships a SHA256SUMS which does not cover every file gets
// a warning for the uncovered ones, and they do not join the count.
func TestAnUnlistedFileIsNotCountedAsVerified(t *testing.T) {
	repo := newFakeRepo("someone/loudr-1")
	// Signed without the vocoder, then handed it back: a repo whose manifest
	// covers everything but one file.
	body := repo.files["onnx/vocoder.onnx"]
	delete(repo.files, "onnx/vocoder.onnx")
	repo.sign()
	repo.files["onnx/vocoder.onnx"] = body
	repo.serve(t)

	var progress bytes.Buffer
	dir := filepath.Join(t.TempDir(), "loudr-1")
	if _, err := DownloadWith("someone/loudr-1", dir, Fetch{Progress: &progress}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(progress.String(), "not covered by "+sumsName) {
		t.Errorf("the uncovered file drew no warning:\n%s", progress.String())
	}
	if !strings.Contains(progress.String(), "verified 11 files against "+sumsName) {
		t.Errorf("the uncovered file was counted as verified:\n%s", progress.String())
	}
}

// TestAVerifiedLineCountsTheFilesItHashed is the other half: with a manifest
// the count is real, and it is the number of files the manifest lists.
func TestAVerifiedLineCountsTheFilesItHashed(t *testing.T) {
	repo := newFakeRepo("someone/loudr-1")
	repo.serve(t)

	var progress bytes.Buffer
	dir := filepath.Join(t.TempDir(), "loudr-1")
	if _, err := DownloadWith("someone/loudr-1", dir, Fetch{Progress: &progress}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(progress.String(), "verified 12 files against "+sumsName) {
		t.Errorf("progress does not count what it hashed:\n%s", progress.String())
	}
}

// TestAShortBodyIsNotCached pins the length check.
//
// The listing already says how long every file is, and nothing compared the
// bytes that arrived against it: a body cut short by a dropped connection or a
// proxy was renamed into place, then written into the receipt, so every later
// run trusted the receipt and never looked again. SHA256SUMS catches it when a
// repo ships one; Download is public for any repo.
func TestAShortBodyIsNotCached(t *testing.T) {
	repo := newFakeRepo("someone/loudr-1")
	delete(repo.files, sumsName)
	const target = "onnx/vocoder.onnx"
	full := repo.files[target]
	inner := repo.handler()
	serveHandler(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, target) && r.Header.Get("Range") == "" {
			w.Write(full[:len(full)-3])
			return
		}
		inner(w, r)
	}))

	dir := filepath.Join(t.TempDir(), "loudr-1")
	_, err := DownloadWith("someone/loudr-1", dir, quiet())
	if err == nil {
		t.Fatal("a truncated download was accepted")
	}
	if !strings.Contains(err.Error(), "stopped at") {
		t.Errorf("refusal reads %v", err)
	}
	if isFile(filepath.Join(dir, filepath.FromSlash(target))) {
		t.Error("the short file was cached anyway")
	}
	// The part-file stays, because that is what the next call resumes.
	if !isFile(filepath.Join(dir, filepath.FromSlash(target)+".part")) {
		t.Error("the part-file was removed, so the transfer cannot resume")
	}
}

// TestAStaleResumeIsRefused pins the Content-Range check.
//
// A 206 was appended to whatever the part-file held, on the server's word
// alone. A server answering for a file that has changed since the interrupted
// run, or answering from a different offset than the one asked for, produced a
// file that is neither version and that no length check can see.
func TestAStaleResumeIsRefused(t *testing.T) {
	const target = "onnx/vocoder.onnx"
	for _, c := range []struct {
		name         string
		contentRange func(from, size int) string
		want         string
	}{
		{"no header", func(int, int) string { return "" }, "no Content-Range"},
		{"another offset", func(from, size int) string {
			return fmt.Sprintf("bytes %d-%d/%d", from+1, size-1, size)
		}, "sent from byte"},
		{"another file", func(from, size int) string {
			return fmt.Sprintf("bytes %d-%d/%d", from, size-1, size+99)
		}, "the file changed since the interrupted run"},
	} {
		t.Run(c.name, func(t *testing.T) {
			repo := newFakeRepo("someone/loudr-1")
			delete(repo.files, sumsName)
			body := repo.files[target]
			inner := repo.handler()
			serveHandler(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				from, ranged := rangeStart(r.Header.Get("Range"))
				if !strings.HasSuffix(r.URL.Path, target) || !ranged {
					inner(w, r)
					return
				}
				if header := c.contentRange(int(from), len(body)); header != "" {
					w.Header().Set("Content-Range", header)
				}
				w.WriteHeader(http.StatusPartialContent)
				w.Write(body[from:])
			}))

			dir := filepath.Join(t.TempDir(), "loudr-1")
			if err := os.MkdirAll(filepath.Join(dir, "onnx"), 0o755); err != nil {
				t.Fatal(err)
			}
			part := filepath.Join(dir, filepath.FromSlash(target)+".part")
			if err := os.WriteFile(part, body[:3], 0o644); err != nil {
				t.Fatal(err)
			}
			_, err := DownloadWith("someone/loudr-1", dir, quiet())
			if err == nil {
				t.Fatal("the resume was accepted")
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("refusal reads %v, want it to name %q", err, c.want)
			}
			if isFile(part) {
				t.Error("the unusable part-file was left to be resumed again")
			}
		})
	}
}

// TestALibraryLookupFailureIsNotRemembered pins that the process may recover.
//
// initRuntime latched its error in a sync.Once, so the caller who reads "no
// onnxruntime shared library. ... Or set LOUDKIT_ONNXRUNTIME_LIB", sets it and
// calls Load again got the same refusal from a lookup that never ran a second
// time. The message names a fix, so the fix has to be able to work.
func TestALibraryLookupFailureIsNotRemembered(t *testing.T) {
	t.Setenv(LibraryEnv, filepath.Join(t.TempDir(), "nowhere.dylib"))
	if err := initRuntime(); err == nil {
		t.Fatal("a library that is not there was accepted")
	}
	runtimeMu.Lock()
	remembered := runtimeReady
	runtimeMu.Unlock()
	if remembered {
		t.Fatal("a failed lookup latched as ready")
	}
	// The second call has to reach findLibrary again rather than replay the
	// first answer, which is what a different message from a different path
	// shows.
	t.Setenv(LibraryEnv, "")
	err := initRuntime()
	if err != nil && strings.Contains(err.Error(), "nowhere.dylib") {
		t.Errorf("the second call replayed the first refusal: %v", err)
	}
}
