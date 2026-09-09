package loudkit

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/loudreader/loudkit/go/internal/digest"
)

// readFixture reads one of the shared fixtures Python generates.
func readFixture(t *testing.T, name string, into any) {
	t.Helper()
	path := filepath.Join("..", "tests", "data", "conformance", name)
	if env := os.Getenv("LOUDKIT_FIXTURE_DIR"); env != "" {
		path = filepath.Join(env, name)
	}
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(body, into); err != nil {
		t.Fatal(err)
	}
}

// TestSelectFilesIsTheSharedPlan holds the plan to the fixture the reference
// planner wrote: the same listing, the same patterns, the same files.
func TestSelectFilesIsTheSharedPlan(t *testing.T) {
	var fixture struct {
		Listing []string `json:"listing"`
		Cases   []struct {
			Backend string   `json:"backend"`
			Cloning bool     `json:"cloning"`
			Ignore  []string `json:"ignore"`
			Wanted  []string `json:"wanted"`
		} `json:"cases"`
	}
	readFixture(t, "release_plan.json", &fixture)
	var listing []remoteFile
	for _, name := range fixture.Listing {
		listing = append(listing, remoteFile{Path: name, Size: 1})
	}
	seen := 0
	for _, c := range fixture.Cases {
		if c.Backend != "onnx" {
			continue
		}
		seen++
		if !equal(ignoreAlways, c.Ignore) {
			t.Errorf("cloning=%v ignore patterns\n got %v\nwant %v", c.Cloning, ignoreAlways, c.Ignore)
		}
		if got := paths(selectFiles(listing, c.Cloning)); !equal(got, c.Wanted) {
			t.Errorf("cloning=%v plan\n got %v\nwant %v", c.Cloning, got, c.Wanted)
		}
	}
	if seen != 2 {
		t.Fatalf("the fixture holds %d onnx cases; nothing was compared", seen)
	}
}

func TestGlobMatchIsFnmatch(t *testing.T) {
	var fixture struct {
		Cases []struct {
			Pattern string `json:"pattern"`
			Name    string `json:"name"`
			Match   bool   `json:"match"`
		} `json:"cases"`
	}
	readFixture(t, "glob.json", &fixture)
	if len(fixture.Cases) < 10 {
		t.Fatalf("the fixture holds %d probes; nothing was compared", len(fixture.Cases))
	}
	for _, c := range fixture.Cases {
		if got := globMatch(c.Pattern, c.Name); got != c.Match {
			t.Errorf("globMatch(%q, %q) = %v, want %v", c.Pattern, c.Name, got, c.Match)
		}
	}
}

func TestIsRepoIDIsTheSharedRule(t *testing.T) {
	var fixture struct {
		Cases []struct {
			Ref      string `json:"ref"`
			IsRepoID bool   `json:"is_repo_id"`
		} `json:"cases"`
	}
	readFixture(t, "repo_id.json", &fixture)
	if len(fixture.Cases) < 10 {
		t.Fatalf("the fixture holds %d probes; nothing was compared", len(fixture.Cases))
	}
	for _, c := range fixture.Cases {
		if got := isRepoID(c.Ref); got != c.IsRepoID {
			t.Errorf("isRepoID(%q) = %v, want %v", c.Ref, got, c.IsRepoID)
		}
	}
	// A path that exists is a path, however it is spelled.
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "org", "name"), 0o755); err != nil {
		t.Fatal(err)
	}
	if isRepoID(filepath.Join(dir, "org", "name")) {
		t.Error("a directory on disk is never a repo id")
	}
}

func TestSafeJoinRefusesAnEscape(t *testing.T) {
	for _, bad := range []string{"../evil", "a/../../evil", `a\b`, "/", ".", "/etc/passwd", ""} {
		if got, err := safeJoin("/tmp/rel", bad); err == nil {
			t.Errorf("safeJoin(%q) = %q, want a refusal", bad, got)
		}
	}
	got, err := safeJoin("/tmp/rel", "onnx/vocoder.onnx")
	if err != nil || got != filepath.FromSlash("/tmp/rel/onnx/vocoder.onnx") {
		t.Errorf("safeJoin = %q, %v", got, err)
	}
}

func TestNextLink(t *testing.T) {
	header := `<https://hf.co/api/models/a/b/tree/main?cursor=x>; rel="next"`
	if got := nextLink(header); !strings.HasSuffix(got, "cursor=x") {
		t.Errorf("nextLink = %q", got)
	}
	if got := nextLink(""); got != "" {
		t.Errorf("nextLink of nothing = %q", got)
	}
}

// fakeRepo is a release served over HTTP: the three endpoints Download uses
// and a count of what was actually asked for. commit is what the revision
// route answers; a test moves it to move the revision.
type fakeRepo struct {
	repo   string
	files  map[string][]byte
	commit string
	mu     sync.Mutex
	got    []string
}

const (
	commitA = "3f2c1e0d9b8a7f6e5d4c3b2a1f0e9d8c7b6a5f4e"
	commitB = "b1d2f3a4c5e6b7d8f9a0c1e2b3d4f5a6c7e8b9d0"
)

func newFakeRepo(repo string) *fakeRepo {
	f := &fakeRepo{repo: repo, files: map[string][]byte{}, commit: commitA}
	f.files[checkpointName] = checkpointBytes(map[string]any{"artifact_role": "synthesis"})
	f.files[tokenizerName] = []byte(`{"model":{}}`)
	f.files[manifestName] = []byte(`{}`)
	f.files[releaseRecord] = []byte(`{"profile":"full-0.1","verified":true}`)
	f.files["README.md"] = []byte("# not part of the set")
	f.files[voiceEncoderName] = []byte("torch only")
	f.files[enrollmentName] = []byte("torch only")
	f.files["coreml/vocoder.mlpackage/Manifest.json"] = []byte("{}")
	for _, rel := range onnxSynthesis {
		f.files[rel] = []byte("graph " + rel)
	}
	for _, rel := range onnxEnroll {
		f.files[rel] = []byte("graph " + rel)
	}
	f.files["voices/joe"+voiceSuffix] = []byte("joe")
	f.files["voices/gosia"+voiceSuffix] = []byte("gosia")
	f.sign()
	return f
}

// sign writes the SHA256SUMS a release ships, covering everything but itself.
func (f *fakeRepo) sign() {
	delete(f.files, sumsName)
	names := make([]string, 0, len(f.files))
	for name := range f.files {
		names = append(names, name)
	}
	sort.Strings(names)
	var out bytes.Buffer
	for _, name := range names {
		fmt.Fprintf(&out, "%x  %s\n", sha256.Sum256(f.files[name]), name)
	}
	f.files[sumsName] = out.Bytes()
}

func (f *fakeRepo) serve(t *testing.T) {
	t.Helper()
	serveHandler(t, f.handler())
}

// serveHandler stands h up as the hub for one test and points the client at
// it, so a test that wants to interpose on a route can wrap f.handler().
func serveHandler(t *testing.T, h http.Handler) {
	t.Helper()
	server := httptest.NewServer(h)
	t.Cleanup(server.Close)
	t.Setenv("HF_ENDPOINT", server.URL)
	t.Setenv("HF_TOKEN", "")
}

// handler is the hub's three routes over f's files: the revision record, the
// recursive tree, and one body per file.
func (f *fakeRepo) handler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		treePrefix := "/api/models/" + f.repo + "/tree/"
		revisionPrefix := "/api/models/" + f.repo + "/revision/"
		filePrefix := "/" + f.repo + "/resolve/main/"
		switch {
		case strings.HasPrefix(r.URL.Path, revisionPrefix):
			f.mu.Lock()
			commit := f.commit
			f.mu.Unlock()
			_ = json.NewEncoder(w).Encode(map[string]any{"sha": commit, "siblings": []any{}})
		case strings.HasPrefix(r.URL.Path, treePrefix):
			type entry struct {
				Type string `json:"type"`
				Path string `json:"path"`
				Size int64  `json:"size"`
			}
			var page []entry
			for name, body := range f.files {
				page = append(page, entry{Type: "file", Path: name, Size: int64(len(body))})
			}
			page = append(page, entry{Type: "directory", Path: "onnx"})
			sort.Slice(page, func(i, j int) bool { return page[i].Path < page[j].Path })
			_ = json.NewEncoder(w).Encode(page)
		case strings.HasPrefix(r.URL.Path, filePrefix):
			name := strings.TrimPrefix(r.URL.Path, filePrefix)
			body, ok := f.files[name]
			if !ok {
				http.NotFound(w, r)
				return
			}
			f.mu.Lock()
			f.got = append(f.got, name)
			f.mu.Unlock()
			if from, ok := rangeStart(r.Header.Get("Range")); ok {
				if from >= int64(len(body)) {
					w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
					return
				}
				// With the Content-Range a 206 is required to carry: it is
				// what says where these bytes belong, and appending without
				// it is how a resumed download becomes a corrupt file.
				w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d",
					from, len(body)-1, len(body)))
				w.WriteHeader(http.StatusPartialContent)
				w.Write(body[from:])
				return
			}
			w.Write(body)
		default:
			http.NotFound(w, r)
		}
	}
}

// rangeStart is the first byte of a `bytes=N-` request header.
func rangeStart(header string) (int64, bool) {
	if !strings.HasPrefix(header, "bytes=") || !strings.HasSuffix(header, "-") {
		return 0, false
	}
	n, err := strconv.ParseInt(strings.TrimSuffix(strings.TrimPrefix(header, "bytes="), "-"), 10, 64)
	return n, err == nil
}

// move makes the revision resolve to another commit, as a push would.
func (f *fakeRepo) move(commit string) {
	f.mu.Lock()
	f.commit = commit
	f.mu.Unlock()
}

func (f *fakeRepo) fetched() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := append([]string{}, f.got...)
	f.got = nil
	sort.Strings(out)
	return out
}

func checkpointBytes(manifest map[string]any) []byte {
	encoded, _ := json.Marshal(manifest)
	header, _ := json.Marshal(map[string]any{
		"__metadata__": map[string]string{"manifest": string(encoded)},
	})
	body := make([]byte, 8, 8+len(header))
	body[0] = byte(len(header))
	body[1] = byte(len(header) >> 8)
	return append(body, header...)
}

func quiet() Fetch { return Fetch{Progress: &bytes.Buffer{}} }

func TestAListingThatKeepsPointingAtItselfIsRefused(t *testing.T) {
	// The tree API pages by the thousand and a release is tens of files, so a
	// Link: rel="next" that never ends is a cycle rather than a large repo.
	// Uncapped, this call never returns.
	repo := newFakeRepo("someone/loudr-1")
	inner := repo.handler()
	var pages atomic.Int64
	serveHandler(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.URL.Path, "/tree/") {
			inner(w, r)
			return
		}
		pages.Add(1)
		w.Header().Set("Link", `<`+"http://"+r.Host+r.URL.RequestURI()+`>; rel="next"`)
		_, _ = w.Write([]byte("[]"))
	}))

	dir := filepath.Join(t.TempDir(), "loudr-1")
	_, err := DownloadWith("someone/loudr-1", dir, quiet())
	if err == nil || !strings.Contains(err.Error(), "the file listing does not end") {
		t.Fatalf("got %v", err)
	}
	if got := pages.Load(); got < 2 || got > 200 {
		t.Errorf("followed %d pages", got)
	}
}

func TestDownloadFetchesTheSetAndVerifiesIt(t *testing.T) {
	repo := newFakeRepo("someone/loudr-1")
	repo.serve(t)
	dir := filepath.Join(t.TempDir(), "loudr-1")

	var log bytes.Buffer
	got, err := DownloadWith("someone/loudr-1", dir, Fetch{Progress: &log})
	if err != nil {
		t.Fatalf("%v\n%s", err, log.String())
	}
	if got != dir {
		t.Errorf("Download returned %q, want the directory it wrote", got)
	}
	// Open is the proof the download is usable, not just present.
	b, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	if got := b.Voices(); !equal(got, []string{"gosia", "joe"}) {
		t.Errorf("voices = %v", got)
	}
	if b.CanEnroll() {
		t.Error("a plain download must not carry the enrollment graphs")
	}
	for _, unwanted := range []string{
		voiceEncoderName, enrollmentName, "README.md",
		"coreml/vocoder.mlpackage/Manifest.json",
	} {
		if isFile(filepath.Join(dir, filepath.FromSlash(unwanted))) {
			t.Errorf("%s was fetched and no onnx caller opens it", unwanted)
		}
	}
	if !strings.Contains(log.String(), "verified") {
		t.Errorf("the first fetch did not report its verification:\n%s", log.String())
	}
	// The bookkeeping files come first, so a repo that is not a release is
	// refused before its weights move.
	order := repo.fetched()
	if len(order) < 2 {
		t.Fatalf("fetched %v", order)
	}

	// A second call resolves the revision, finds the receipt names that
	// commit, and does nothing: no file is fetched again, and nothing is
	// hashed, because verification happened when the bytes arrived.
	log.Reset()
	if _, err := DownloadWith("someone/loudr-1", dir, Fetch{Progress: &log}); err != nil {
		t.Fatal(err)
	}
	if got := repo.fetched(); len(got) != 0 {
		t.Errorf("a second download refetched %v", got)
	}
	if strings.Contains(log.String(), "verified") {
		t.Errorf("a second download re-hashed the release:\n%s", log.String())
	}
}

// TestTheReceiptIsTheSharedFixture holds the receipt's shape, the predicate
// and the hit rule to the fixture every port reads, case by case on disk.
func TestTheReceiptIsTheSharedFixture(t *testing.T) {
	var fixture struct {
		Name    string   `json:"name"`
		Fields  []string `json:"fields"`
		Sums    string   `json:"sums"`
		Files   []string `json:"files"`
		Example struct {
			SHA256Sums string `json:"sha256sums"`
		} `json:"example"`
		Cases []struct {
			Name    string          `json:"name"`
			Receipt json.RawMessage `json:"receipt"`
			Sums    *string         `json:"sums"`
			Files   []string        `json:"files"`
			Repo    string          `json:"repo"`
			Commit  string          `json:"commit"`
			Offline string          `json:"offline"`
			Online  string          `json:"online"`
		} `json:"cases"`
	}
	readFixture(t, "release_receipt.json", &fixture)
	if fixture.Name != receiptName {
		t.Errorf("the receipt is %q, the fixture says %q", receiptName, fixture.Name)
	}
	if !equal(fixture.Fields, receiptFields) {
		t.Errorf("receipt fields = %v, the fixture says %v", receiptFields, fixture.Fields)
	}
	if len(fixture.Cases) < 20 {
		t.Fatalf("the fixture holds %d cases; nothing was compared", len(fixture.Cases))
	}
	for _, c := range fixture.Cases {
		dir := t.TempDir()
		if string(c.Receipt) != "null" {
			if err := os.WriteFile(filepath.Join(dir, receiptName), c.Receipt, 0o644); err != nil {
				t.Fatal(err)
			}
		}
		if c.Sums != nil {
			if err := os.WriteFile(filepath.Join(dir, sumsName), []byte(*c.Sums), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		for _, name := range c.Files {
			if err := os.WriteFile(filepath.Join(dir, name), []byte(name), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		if got := readReceipt(dir, c.Repo, false) != nil; got != (c.Offline == "use") {
			t.Errorf("%s: readReceipt accepted = %v, want offline %s", c.Name, got, c.Offline)
		}
		if got := receiptHit(dir, c.Repo, c.Commit, false); got != (c.Online == "hit") {
			t.Errorf("%s: receiptHit = %v, want online %s", c.Name, got, c.Online)
		}
	}
	// What this port writes has the fixture's keys, in the fixture's order,
	// and over the example's manifest, the example's digest.
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, sumsName), []byte(fixture.Sums), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, name := range fixture.Files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(name), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := writeReceipt(dir, "loudreader/loudr-1", "main", commitA); err != nil {
		t.Fatal(err)
	}
	body, err := os.ReadFile(filepath.Join(dir, receiptName))
	if err != nil {
		t.Fatal(err)
	}
	if got := topLevelKeys(t, body); !equal(got, fixture.Fields) {
		t.Errorf("receipt keys = %v, want %v", got, fixture.Fields)
	}
	r := readReceipt(dir, "loudreader/loudr-1", false)
	if r == nil || r.SHA256Sums == nil || *r.SHA256Sums != fixture.Example.SHA256Sums {
		t.Errorf("readReceipt = %+v", r)
	}
	if !receiptHit(dir, "loudreader/loudr-1", commitA, false) || receiptHit(dir, "loudreader/loudr-1", commitB, false) {
		t.Error("the written receipt does not follow the hit rule")
	}
}

// topLevelKeys are a JSON object's keys in the order they were written.
func topLevelKeys(t *testing.T, body []byte) []string {
	t.Helper()
	dec := json.NewDecoder(bytes.NewReader(body))
	if tok, err := dec.Token(); err != nil || tok != json.Delim('{') {
		t.Fatalf("not an object: %v %v", tok, err)
	}
	var keys []string
	for dec.More() {
		tok, err := dec.Token()
		if err != nil {
			t.Fatal(err)
		}
		keys = append(keys, tok.(string))
		var value any
		if err := dec.Decode(&value); err != nil {
			t.Fatal(err)
		}
	}
	return keys
}

// TestAMovedRevisionIsFetchedAgainAndKeepsWhatMatches is the drift check: a
// receipt for another commit means fetch and verify again, and what already
// hashes to the new manifest is kept rather than moved twice.
func TestAMovedRevisionIsFetchedAgainAndKeepsWhatMatches(t *testing.T) {
	repo := newFakeRepo("someone/loudr-1")
	repo.serve(t)
	dir := filepath.Join(t.TempDir(), "loudr-1")
	if _, err := DownloadWith("someone/loudr-1", dir, quiet()); err != nil {
		t.Fatal(err)
	}
	repo.fetched()
	if r := readReceipt(dir, "someone/loudr-1", false); r == nil || r.Commit != commitA || r.Revision != "main" {
		t.Fatalf("receipt after the first fetch = %+v", r)
	}

	repo.move(commitB)
	var log bytes.Buffer
	if _, err := DownloadWith("someone/loudr-1", dir, Fetch{Progress: &log}); err != nil {
		t.Fatal(err)
	}
	if got := repo.fetched(); !equal(got, []string{sumsName}) {
		t.Errorf("a moved revision fetched %v, want the manifest alone", got)
	}
	if !strings.Contains(log.String(), "verified") {
		t.Errorf("a moved revision was not re-verified:\n%s", log.String())
	}
	if r := readReceipt(dir, "someone/loudr-1", false); r == nil || r.Commit != commitB {
		t.Errorf("receipt after the move = %+v", r)
	}

	// A file edited in place at its published length is caught on the next
	// miss, and it alone is fetched again.
	voice := filepath.Join(dir, "voices", "joe"+voiceSuffix)
	if err := os.WriteFile(voice, []byte("eoj"), 0o644); err != nil {
		t.Fatal(err)
	}
	repo.move(commitA)
	if _, err := DownloadWith("someone/loudr-1", dir, quiet()); err != nil {
		t.Fatal(err)
	}
	if got := repo.fetched(); !equal(got, []string{sumsName, "voices/joe" + voiceSuffix}) {
		t.Errorf("fetched %v, want the manifest and the edited voice", got)
	}
	if body, _ := os.ReadFile(voice); string(body) != "joe" {
		t.Errorf("the edited voice was not replaced: %q", body)
	}
}

// TestAListedFileDeletedUnderAReceiptIsFetchedAgain: a receipt vouches for
// every file the plan selects from SHA256SUMS, not only the files the port
// cannot run without. A voice is one of many and its absence is a miss; a
// file edited in place is not, because a hit hashes nothing.
func TestAListedFileDeletedUnderAReceiptIsFetchedAgain(t *testing.T) {
	repo := newFakeRepo("someone/loudr-1")
	repo.serve(t)
	dir := filepath.Join(t.TempDir(), "loudr-1")
	if _, err := DownloadWith("someone/loudr-1", dir, quiet()); err != nil {
		t.Fatal(err)
	}
	repo.fetched()
	voice := filepath.Join(dir, "voices", "gosia"+voiceSuffix)
	if err := os.Remove(voice); err != nil {
		t.Fatal(err)
	}
	if _, err := DownloadWith("someone/loudr-1", dir, quiet()); err != nil {
		t.Fatal(err)
	}
	if got := repo.fetched(); !equal(got, []string{sumsName, "voices/gosia" + voiceSuffix}) {
		t.Errorf("fetched %v, want the manifest and the deleted voice", got)
	}
	if err := os.WriteFile(voice, []byte("aisog"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := DownloadWith("someone/loudr-1", dir, quiet()); err != nil {
		t.Fatal(err)
	}
	if got := repo.fetched(); len(got) != 0 {
		t.Errorf("a hit fetched %v", got)
	}
	if body, _ := os.ReadFile(voice); string(body) != "aisog" {
		t.Errorf("a hit looked at the voice: %q", body)
	}
}

// TestOfflineWithAReceiptUsesIt: the hub cannot be reached, the directory
// holds a receipt, and the load goes on with a line on stderr. Without a
// receipt there is nothing to stand in for the hub.
func TestOfflineWithAReceiptUsesIt(t *testing.T) {
	repo := newFakeRepo("someone/loudr-1")
	repo.serve(t)
	dir := filepath.Join(t.TempDir(), "loudr-1")
	if _, err := DownloadWith("someone/loudr-1", dir, quiet()); err != nil {
		t.Fatal(err)
	}
	// A port nobody listens on: the transport fails, the hub never answers.
	t.Setenv("HF_ENDPOINT", "http://127.0.0.1:9")
	var log bytes.Buffer
	got, err := DownloadWith("someone/loudr-1", dir, Fetch{Progress: &log})
	if err != nil || got != dir {
		t.Fatalf("offline with a receipt: %q, %v", got, err)
	}
	if !strings.Contains(log.String(), "cannot be reached") || !strings.Contains(log.String(), commitA) {
		t.Errorf("offline use was not announced:\n%s", log.String())
	}
	if _, err := DownloadWith("someone/loudr-1", filepath.Join(t.TempDir(), "empty"), quiet()); err == nil {
		t.Error("offline with no receipt loaded nothing and said nothing")
	}
	// A directory verified as one repo does not answer for another.
	if err := writeReceipt(dir, "someone/other", "main", commitA); err != nil {
		t.Fatal(err)
	}
	if _, err := DownloadWith("someone/loudr-1", dir, quiet()); err == nil {
		t.Error("offline, a receipt for another repo answered for this one")
	}
	// Nor does a receipt no download wrote: one field forged is no receipt.
	if err := writeReceipt(dir, "someone/loudr-1", "main", commitA); err != nil {
		t.Fatal(err)
	}
	good, err := os.ReadFile(filepath.Join(dir, receiptName))
	if err != nil {
		t.Fatal(err)
	}
	for name, forged := range map[string]string{
		"no commit":    strings.Replace(string(good), `"commit": "`+commitA+`",`, "", 1),
		"empty commit": strings.Replace(string(good), commitA, "", 1),
		"wrong digest": strings.Replace(string(good), `"sha256sums": "`, `"sha256sums": "0`, 1),
	} {
		if forged == string(good) {
			t.Fatalf("%s: nothing forged in\n%s", name, good)
		}
		if err := os.WriteFile(filepath.Join(dir, receiptName), []byte(forged), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := DownloadWith("someone/loudr-1", dir, quiet()); err == nil {
			t.Errorf("offline, a receipt with %s answered", name)
		}
	}
}

// TestAStaleReceiptDoesNotOutliveAFailedFetch: a receipt for a commit whose
// files were never verified would be a lie the next run believes.
func TestAStaleReceiptDoesNotOutliveAFailedFetch(t *testing.T) {
	repo := newFakeRepo("someone/loudr-1")
	repo.serve(t)
	dir := filepath.Join(t.TempDir(), "loudr-1")
	if _, err := DownloadWith("someone/loudr-1", dir, quiet()); err != nil {
		t.Fatal(err)
	}
	repo.move(commitB)
	repo.files["voices/joe"+voiceSuffix] = []byte("not joe") // signed as something else
	voice := filepath.Join(dir, "voices", "joe"+voiceSuffix)
	if err := os.WriteFile(voice, []byte("xxx"), 0o644); err != nil { // and not kept either
		t.Fatal(err)
	}
	if _, err := DownloadWith("someone/loudr-1", dir, quiet()); err == nil {
		t.Fatal("a corrupt fetch passed")
	}
	if _, err := os.Stat(filepath.Join(dir, receiptName)); err == nil {
		t.Error("the receipt survived a fetch that failed verification")
	}
}

func TestDownloadFetchesTheBookkeepingFirst(t *testing.T) {
	repo := newFakeRepo("loudreader/loudr-1")
	repo.serve(t)
	dir := filepath.Join(t.TempDir(), "loudr-1")
	if _, err := DownloadWith("loudreader/loudr-1", dir, quiet()); err != nil {
		t.Fatal(err)
	}
	repo.mu.Lock()
	first := append([]string{}, repo.got[:2]...)
	repo.mu.Unlock()
	if !equal(first, []string{sumsName, releaseRecord}) {
		t.Errorf("the first two fetches were %v, want SHA256SUMS then release.json", first)
	}
}

func TestDownloadWithCloningAddsTheEnrollmentGraphs(t *testing.T) {
	repo := newFakeRepo("someone/loudr-1")
	repo.serve(t)
	dir := filepath.Join(t.TempDir(), "loudr-1")
	if _, err := DownloadWith("someone/loudr-1", dir, Fetch{Cloning: true, Progress: &bytes.Buffer{}}); err != nil {
		t.Fatal(err)
	}
	b, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !b.CanEnroll() {
		t.Error("a cloning download carries the three enrollment graphs")
	}
}

func TestDownloadCatchesABadByteAndRemovesTheFile(t *testing.T) {
	repo := newFakeRepo("someone/loudr-1")
	repo.files["voices/joe"+voiceSuffix] = []byte("not joe") // signed as something else
	repo.serve(t)
	dir := filepath.Join(t.TempDir(), "loudr-1")
	_, err := DownloadWith("someone/loudr-1", dir, quiet())
	if err == nil || !strings.Contains(err.Error(), "failed the release checksum") {
		t.Fatalf("got %v", err)
	}
	if isFile(filepath.Join(dir, "voices", "joe"+voiceSuffix)) {
		t.Error("a file that failed its checksum was left on disk")
	}
}

func TestDownloadRefusesUnvouchedWeights(t *testing.T) {
	repo := newFakeRepo("someone/loudr-1")
	repo.files["voices/stranger"+voiceSuffix] = []byte("nothing vouches for me")
	repo.serve(t) // signed before this file existed, so it is unlisted
	dir := filepath.Join(t.TempDir(), "loudr-1")
	_, err := DownloadWith("someone/loudr-1", dir, quiet())
	if err == nil || !strings.Contains(err.Error(), "nothing vouching for them") {
		t.Fatalf("got %v", err)
	}
}

func TestDownloadHoldsAnOfficialRepoToItsClaims(t *testing.T) {
	t.Run("no manifest at all", func(t *testing.T) {
		repo := newFakeRepo("loudreader/loudr-1")
		delete(repo.files, sumsName)
		repo.serve(t)
		_, err := DownloadWith("loudreader/loudr-1", filepath.Join(t.TempDir(), "r"), quiet())
		if err == nil || !strings.Contains(err.Error(), "no SHA256SUMS") {
			t.Fatalf("got %v", err)
		}
		if got := repo.fetched(); len(got) != 0 {
			t.Errorf("bytes moved before the refusal: %v", got)
		}
	})

	t.Run("a development bundle is refused before its weights move", func(t *testing.T) {
		repo := newFakeRepo("loudreader/loudr-1")
		repo.files[releaseRecord] = []byte(`{"profile":"lenient","verified":true}`)
		repo.sign()
		repo.serve(t)
		_, err := DownloadWith("loudreader/loudr-1", filepath.Join(t.TempDir(), "r"), quiet())
		if err == nil || !strings.Contains(err.Error(), "development bundle") {
			t.Fatalf("got %v", err)
		}
		if got := repo.fetched(); !equal(got, []string{sumsName, releaseRecord}) {
			t.Errorf("fetched %v before the refusal, want the two bookkeeping files only", got)
		}
	})

	t.Run("an ungated build is not the release", func(t *testing.T) {
		repo := newFakeRepo("loudreader/loudr-1")
		repo.files[releaseRecord] = []byte(`{"profile":"full-0.1","verified":false}`)
		repo.sign()
		repo.serve(t)
		_, err := DownloadWith("loudreader/loudr-1", filepath.Join(t.TempDir(), "r"), quiet())
		if err == nil || !strings.Contains(err.Error(), "load-and-speak gate") {
			t.Fatalf("got %v", err)
		}
	})

	t.Run("the release itself passes", func(t *testing.T) {
		repo := newFakeRepo("loudreader/loudr-1")
		repo.serve(t)
		if _, err := DownloadWith("loudreader/loudr-1", filepath.Join(t.TempDir(), "loudr-1"), quiet()); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("the turbo profile is a release too", func(t *testing.T) {
		// The profile gate is one gate for both models; what keeps turbo out
		// of this port is the graphs check, which this listing passes.
		repo := newFakeRepo("loudreader/loudr-1")
		repo.files[releaseRecord] = []byte(`{"profile":"turbo-0.1","verified":true}`)
		repo.sign()
		repo.serve(t)
		if _, err := DownloadWith("loudreader/loudr-1", filepath.Join(t.TempDir(), "r"), quiet()); err != nil {
			t.Fatal(err)
		}
	})
}

func TestDownloadRefusesMissingGraphsBeforeFetching(t *testing.T) {
	repo := newFakeRepo("loudreader/loudr-1-turbo")
	delete(repo.files, checkpointName)
	repo.files["loudr-1-turbo.safetensors"] = checkpointBytes(map[string]any{})
	for _, rel := range onnxSynthesis {
		delete(repo.files, rel)
	}
	for _, rel := range onnxEnroll {
		delete(repo.files, rel)
	}
	delete(repo.files, onnxExportRecord)
	repo.files[releaseRecord] = []byte(`{"profile":"turbo-0.1","verified":true}`)
	repo.sign()
	repo.serve(t)
	_, err := DownloadWith("loudreader/loudr-1-turbo", filepath.Join(t.TempDir(), "r"), quiet())
	if err == nil || !strings.Contains(err.Error(), "ships no") {
		t.Fatalf("got %v", err)
	}
	if got := repo.fetched(); len(got) != 0 {
		t.Errorf("bytes moved before the refusal: %v", got)
	}
}

func TestDownloadSaysWhenAGraphIsMissing(t *testing.T) {
	repo := newFakeRepo("someone/loudr-1")
	delete(repo.files, "onnx/vocoder.onnx")
	repo.sign()
	repo.serve(t)
	_, err := DownloadWith("someone/loudr-1", filepath.Join(t.TempDir(), "r"), quiet())
	if err == nil || !strings.Contains(err.Error(), "onnx/vocoder.onnx") {
		t.Fatalf("a short fetch has to name what is short: %v", err)
	}
	if got := repo.fetched(); len(got) != 0 {
		t.Errorf("bytes moved before the refusal: %v", got)
	}
}

func TestDownloadRefusesHalfAPairDecodeGraphSet(t *testing.T) {
	repo := newFakeRepo("someone/loudr-1")
	delete(repo.files, "onnx/t3_step.onnx")
	repo.files["onnx/t3_pair_step.onnx"] = []byte("graph onnx/t3_pair_step.onnx")
	repo.sign()
	repo.serve(t)
	_, err := DownloadWith("someone/loudr-1", filepath.Join(t.TempDir(), "r"), quiet())
	if err == nil || !strings.Contains(err.Error(), "pair/head2") {
		t.Fatalf("a pair set without head2 decodes nothing: %v", err)
	}
	if got := repo.fetched(); len(got) != 0 {
		t.Errorf("bytes moved before the refusal: %v", got)
	}
}

func TestDownloadRefusesSomethingThatIsNotARepoID(t *testing.T) {
	if _, err := Download("./loudr-1", t.TempDir()); err == nil {
		t.Error("a path is not a repo id")
	}
}

func TestParseSumsRefusesAMangledManifest(t *testing.T) {
	dir := t.TempDir()
	digest := strings.Repeat("a", 64)
	for name, body := range map[string]string{
		"malformed": "not a checksum line\n",
		"absolute":  digest + "  /etc/passwd\n",
		"traversal": digest + "  ../../id_rsa\n",
		"duplicate": digest + "  a.json\n" + digest + "  a.json\n",
		"empty":     "\n\n",
	} {
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := parseSums(path); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	good := filepath.Join(dir, "good")
	if err := os.WriteFile(good, []byte(digest+"  onnx/t3_step.onnx\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	entries, err := parseSums(good)
	if err != nil || entries["onnx/t3_step.onnx"] != digest {
		t.Errorf("entries = %v, %v", entries, err)
	}
}

func TestFileSHA256(t *testing.T) {
	path := filepath.Join(t.TempDir(), "x")
	if err := os.WriteFile(path, []byte("abc"), 0o644); err != nil {
		t.Fatal(err)
	}
	// The published digest of "abc", not one computed here with the same
	// primitive.
	const want = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
	got, err := digest.SHA256File(path)
	if err != nil || got != want {
		t.Errorf("digest.SHA256File = %q, %v", got, err)
	}
}

func paths(files []remoteFile) []string {
	out := make([]string, len(files))
	for i, f := range files {
		out[i] = f.Path
	}
	return out
}

func equal(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestFetchResumesAPartAndRestartsAFinishedOne(t *testing.T) {
	repo := newFakeRepo("someone/loudr-1")
	repo.serve(t)
	graph := repo.files["onnx/vocoder.onnx"]

	for _, c := range []struct {
		name string
		part []byte
	}{
		// A short `.part` is the interrupted download the Range header is for.
		{"short", graph[:3]},
		// One at or past the length has nothing to resume: the range request
		// for it comes back 416, so it has to be started again instead.
		{"finished", graph},
		{"overlong", append(append([]byte{}, graph...), "stale"...)},
	} {
		t.Run(c.name, func(t *testing.T) {
			dir := filepath.Join(t.TempDir(), "loudr-1")
			if err := os.MkdirAll(filepath.Join(dir, "onnx"), 0o755); err != nil {
				t.Fatal(err)
			}
			part := filepath.Join(dir, "onnx", "vocoder.onnx.part")
			if err := os.WriteFile(part, c.part, 0o644); err != nil {
				t.Fatal(err)
			}
			if _, err := DownloadWith("someone/loudr-1", dir, quiet()); err != nil {
				t.Fatal(err)
			}
			got, err := os.ReadFile(filepath.Join(dir, "onnx", "vocoder.onnx"))
			if err != nil || !bytes.Equal(got, graph) {
				t.Errorf("resumed file = %q, %v", got, err)
			}
			if isFile(part) {
				t.Error("the .part outlived the download")
			}
		})
	}
}

// TestEnrollOnARepoEngineFetchesTheCloningSetOnce: an engine loaded by repo
// id fetches the three enrollment graphs into the directory it was loaded
// from, with the manifest that vouches for them, and the second enrollment
// fetches nothing. An engine loaded from a directory is told how to fetch
// them and moves no bytes.
func TestEnrollOnARepoEngineFetchesTheCloningSetOnce(t *testing.T) {
	repo := newFakeRepo("someone/loudr-1")
	repo.serve(t)
	dir := filepath.Join(t.TempDir(), "loudr-1")
	if _, err := DownloadWith("someone/loudr-1", dir, quiet()); err != nil {
		t.Fatal(err)
	}
	repo.fetched()
	b, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	eng := &Engine{bundle: b, repo: "someone/loudr-1"}
	if err := eng.fetchCloning(context.Background()); err != nil {
		t.Fatal(err)
	}
	want := append([]string{sumsName}, onnxEnroll...)
	sort.Strings(want)
	if got := repo.fetched(); !equal(got, want) {
		t.Errorf("the first enrollment fetched %v, want %v", got, want)
	}
	if !b.CanEnroll() {
		t.Error("the enrollment graphs did not land in the engine's own directory")
	}
	if readReceipt(dir, "someone/loudr-1", true) == nil {
		t.Error("the receipt was not widened to the cloning set")
	}
	if err := eng.fetchCloning(context.Background()); err != nil {
		t.Fatal(err)
	}
	if got := repo.fetched(); len(got) != 0 {
		t.Errorf("the second enrollment fetched %v", got)
	}

	plain := filepath.Join(t.TempDir(), "plain")
	if _, err := DownloadWith("someone/loudr-1", plain, quiet()); err != nil {
		t.Fatal(err)
	}
	repo.fetched()
	pb, err := Open(plain)
	if err != nil {
		t.Fatal(err)
	}
	err = (&Engine{bundle: pb}).fetchCloning(context.Background())
	if err == nil || !strings.Contains(err.Error(), "Fetch{Cloning: true}") {
		t.Errorf("a directory engine got %v, want the sentence naming DownloadWith", err)
	}
	if got := repo.fetched(); len(got) != 0 {
		t.Errorf("a directory engine fetched %v", got)
	}
}

func TestFusionInventoryComesFromManifest(t *testing.T) {
	root := t.TempDir()
	ckpt := writeCheckpoint(t, root, "renamed.safetensors", map[string]any{"decode": map[string]any{"mode": "fusion_mtp2"}})
	graphs, err := synthesisGraphs(ckpt)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range graphs {
		writeFile(t, root, name, []byte("graph"))
	}
	writeFile(t, root, tokenizerName, []byte("{}"))
	if _, err := Open(root); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(root, "onnx", "t3_head2.onnx")); err != nil {
		t.Fatal(err)
	}
	if _, err := Open(root); err == nil || !strings.Contains(err.Error(), "t3_head2") {
		t.Fatalf("missing head2: %v", err)
	}
}

// TestWithoutStepIsTheWrittenOrder pins the two inventories spelled out, so
// reordering onnxSynthesis fails here rather than quietly changing what a
// fetch refuses and what Open reports as missing.
func TestWithoutStepIsTheWrittenOrder(t *testing.T) {
	single := []string{
		"onnx/t3_cond.onnx", "onnx/t3_prefill.onnx", stepGraph,
		"onnx/flow_encoder.onnx", "onnx/flow_estimator.onnx", "onnx/vocoder.onnx",
	}
	if !equal(onnxSynthesis, single) {
		t.Errorf("onnxSynthesis = %v\nwant %v", onnxSynthesis, single)
	}
	fusion := []string{
		"onnx/t3_cond.onnx", "onnx/t3_prefill.onnx", pairStepGraph, head2Graph,
		"onnx/flow_encoder.onnx", "onnx/flow_estimator.onnx", "onnx/vocoder.onnx",
	}
	if got := withoutStep(pairStepGraph, head2Graph); !equal(got, fusion) {
		t.Errorf("fusion inventory = %v\nwant %v", got, fusion)
	}
	dropped := []string{
		"onnx/t3_cond.onnx", "onnx/t3_prefill.onnx",
		"onnx/flow_encoder.onnx", "onnx/flow_estimator.onnx", "onnx/vocoder.onnx",
	}
	if got := withoutStep(); !equal(got, dropped) {
		t.Errorf("the graphs both modes share = %v\nwant %v", got, dropped)
	}
}

func TestBothModelNamesResolveToTheirRepository(t *testing.T) {
	for _, name := range []string{"loudr-1", "loudr-1-turbo"} {
		if got := modelRepo(name); got != "loudreader/"+name {
			t.Fatal(got)
		}
	}
}

// TestTheCachePathIsTheSharedFixture holds the cache layout to the fixture
// the four ports read: one path for a given root and repo, and the one
// variable that moves it.
func TestTheCachePathIsTheSharedFixture(t *testing.T) {
	var fixture struct {
		Env   string `json:"env"`
		Cases []struct {
			Root string `json:"root"`
			Repo string `json:"repo"`
			Path string `json:"path"`
		} `json:"cases"`
		Override []struct {
			Cache string `json:"cache"`
			Repo  string `json:"repo"`
			Path  string `json:"path"`
		} `json:"override"`
	}
	readFixture(t, "cache_path.json", &fixture)
	if fixture.Env != "LOUDKIT_CACHE" || len(fixture.Cases) < 4 || len(fixture.Override) == 0 {
		t.Fatalf("the fixture holds %d cases under %q; nothing was compared", len(fixture.Cases), fixture.Env)
	}
	for _, c := range fixture.Cases {
		if got := filepath.ToSlash(cachePath(c.Root, c.Repo)); got != c.Path {
			t.Errorf("cachePath(%q, %q) = %q, want %q", c.Root, c.Repo, got, c.Path)
		}
	}
	for _, o := range fixture.Override {
		t.Setenv(fixture.Env, o.Cache)
		got, err := cacheDir(o.Repo)
		if err != nil || filepath.ToSlash(got) != o.Path {
			t.Errorf("cacheDir(%q) under %s=%q = %q, %v; want %q", o.Repo, fixture.Env, o.Cache, got, err, o.Path)
		}
	}
	t.Setenv(fixture.Env, "")
	got, err := cacheDir("loudreader/loudr-1")
	if err != nil || filepath.Base(filepath.Dir(got)) != "loudkit" || filepath.Base(got) != "loudreader--loudr-1" {
		t.Errorf("cacheDir without the override = %q, %v", got, err)
	}
}

// TestOfflineWithoutTheCloningSetNamesTheGraphs: a receipt for the synthesis
// set answers a plain load when the hub is away, and an enrollment gets the
// sentence naming the three graphs, not the transport error.
func TestOfflineWithoutTheCloningSetNamesTheGraphs(t *testing.T) {
	repo := newFakeRepo("someone/loudr-1")
	repo.serve(t)
	dir := filepath.Join(t.TempDir(), "loudr-1")
	if _, err := DownloadWith("someone/loudr-1", dir, quiet()); err != nil {
		t.Fatal(err)
	}
	t.Setenv("HF_ENDPOINT", "http://127.0.0.1:9")
	if _, err := DownloadWith("someone/loudr-1", dir, quiet()); err != nil {
		t.Fatalf("offline, the synthesis set was not used: %v", err)
	}
	_, err := DownloadWith("someone/loudr-1", dir, Fetch{Cloning: true, Progress: &bytes.Buffer{}})
	want := "holds no enrollment graphs (" + strings.Join(onnxEnroll, ", ") + "). Connect once to fetch them"
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Errorf("offline without the cloning set: %v\nwant a sentence with %q", err, want)
	}
}

// TestFusionHeaderIsReadWithoutTheWeights: recognition reads the header
// alone, so a checkpoint the size of the real one, all hole, is named at
// once.
func TestFusionHeaderIsReadWithoutTheWeights(t *testing.T) {
	dir := t.TempDir()
	path := writeCheckpoint(t, dir, "renamed.safetensors",
		map[string]any{"decode": map[string]any{"mode": "fusion_mtp2"}})
	if err := os.Truncate(path, 4<<30); err != nil {
		t.Skipf("no sparse files here: %v", err)
	}
	_, err := Open(dir)
	if err == nil || !strings.Contains(err.Error(), "t3_pair_step.onnx") {
		t.Errorf("a 4 GB turbo checkpoint got %v", err)
	}
}

// TestAReceiptThatIsNotARecordIsNothing: an empty file, garbage, and a file
// too large to be a receipt are no receipt, and the last is not read.
func TestAReceiptThatIsNotARecordIsNothing(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, receiptName)
	record := `{"repo": "someone/loudr-1", "revision": "main", "commit": "` + commitA +
		`", "sha256sums": null, "fetched_at": "2026-09-02T12:00:00Z"`
	write := func(body string) {
		t.Helper()
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write(record + "}\n")
	if readReceipt(dir, "someone/loudr-1", false) == nil {
		t.Fatal("the record itself was refused")
	}
	for name, body := range map[string]string{
		"an empty file":                  "",
		"garbage":                        strings.Repeat("\x00\xff{", 1<<15),
		"a record padded past the limit": record + `, "pad": "` + strings.Repeat("x", receiptLimit) + `"}`,
	} {
		write(body)
		if readReceipt(dir, "someone/loudr-1", false) != nil {
			t.Errorf("%s was accepted as a receipt", name)
		}
	}
}

func TestFusionDownloadAddsEnrollmentOnceAndWorksOffline(t *testing.T) {
	repo := newFakeRepo("someone/renamed-model")
	repo.files[checkpointName] = checkpointBytes(map[string]any{"artifact_role": "synthesis", "format_version": 2, "decode": map[string]any{"mode": "fusion_mtp2"}})
	delete(repo.files, "onnx/t3_step.onnx")
	for _, name := range []string{"onnx/t3_pair_step.onnx", "onnx/t3_head2.onnx"} {
		repo.files[name] = []byte("graph " + name)
	}
	repo.sign()
	repo.serve(t)
	dir := t.TempDir()
	if _, err := DownloadWith(repo.repo, dir, quiet()); err != nil {
		t.Fatal(err)
	}
	if _, err := DownloadWith(repo.repo, dir, Fetch{Cloning: true, Progress: &bytes.Buffer{}}); err != nil {
		t.Fatal(err)
	}
	repo.fetched() // clear the recording before the offline request
	t.Setenv("HF_ENDPOINT", "http://127.0.0.1:9")
	_, err := DownloadWith(repo.repo, dir, Fetch{Cloning: true, Progress: &bytes.Buffer{}})
	if err != nil {
		t.Fatal(err)
	}
	b, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !b.CanEnroll() {
		t.Fatal("fusion bundle lacks canonical enrollment graphs")
	}
	if len(repo.fetched()) != 0 {
		t.Fatal("offline load fetched files")
	}
}

// TestDownloadContextCancelStopsAFetchInFlight is what the context is for:
// the hub answers, the body never ends, and nothing else in this package can
// end the call. Without one, this test would hang.
func TestDownloadContextCancelStopsAFetchInFlight(t *testing.T) {
	repo := newFakeRepo("someone/loudr-1")
	inner := repo.handler()
	started, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	serveHandler(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.URL.Path, "/resolve/") {
			inner(w, r)
			return
		}
		once.Do(func() { close(started) })
		w.WriteHeader(http.StatusOK)
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
		<-release
	}))
	// Registered after serveHandler so it runs before the server closes:
	// Close waits for handlers, and this is what lets this one return.
	t.Cleanup(func() { close(release) })

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		<-started
		cancel()
	}()
	dir := filepath.Join(t.TempDir(), "loudr-1")
	if _, err := DownloadContext(ctx, "someone/loudr-1", dir, quiet()); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelling a fetch in flight gave %v, want context.Canceled", err)
	}
}

// TestDownloadContextCancelDoesNotStandInTheReceipt separates a caller who
// gave up from a hub that cannot be reached. Only the second may answer from
// the receipt; a cancelled call that returned the directory would read as a
// download that happened.
func TestDownloadContextCancelDoesNotStandInTheReceipt(t *testing.T) {
	repo := newFakeRepo("someone/loudr-1")
	repo.serve(t)
	dir := filepath.Join(t.TempDir(), "loudr-1")
	if _, err := DownloadWith("someone/loudr-1", dir, quiet()); err != nil {
		t.Fatal(err)
	}
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := DownloadContext(cancelled, "someone/loudr-1", dir, quiet()); !errors.Is(err, context.Canceled) {
		t.Errorf("a cancelled call over a good receipt gave %v, want context.Canceled", err)
	}
	expired, stop := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer stop()
	if _, err := DownloadContext(expired, "someone/loudr-1", dir, quiet()); !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("an expired deadline over a good receipt gave %v, want context.DeadlineExceeded", err)
	}
	// The receipt is still good, so the plain call is still a hit.
	if _, err := DownloadWith("someone/loudr-1", dir, quiet()); err != nil {
		t.Errorf("after two abandoned calls the plain one failed: %v", err)
	}
}
