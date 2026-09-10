package loudkit

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// releaseCore is what every backend fetches. `*.safetensors` is the checkpoint
// and, nested, every voice; ve.safetensors and the enrollment artefact are
// carved back out by ignoreAlways. The JSON names are spelled out rather than
// `*.json`, which also matches the small manifests inside a CoreML package.
var releaseCore = []string{
	"*.safetensors",
	manifestName,
	tokenizerName,
	releaseRecord,
	voiceDir + "/*",
	sumsName,
}

// ignoreAlways are the two files an onnx caller never opens: the utterance
// voice encoder and the enrollment artefact are torch's enrollment path, and
// this port clones through the enrollment graphs instead.
var ignoreAlways = []string{voiceEncoderName, enrollmentName}

// Fetch is what varies about a download. The zero value is the plain synthesis
// set from the repo's default branch, with progress on stderr.
type Fetch struct {
	// Revision is a branch, tag or commit. Empty means the default branch.
	// Pin it for anything reproducible: a moving main is a moving model.
	Revision string
	// Cloning adds the three enrollment graphs Enroll needs.
	Cloning bool
	// Progress is where per-file lines go. Nil means stderr; use io.Discard
	// for silence.
	Progress io.Writer
}

// Download fetches the onnx release for a Hugging Face repo id into dir, and
// returns dir.
//
// The whole set travels, not just the weights. dir carries a receipt naming
// the commit it was verified against: a call whose revision still resolves
// to that commit, over a directory whose receipt readReceipt accepts and
// which still holds the set, fetches and hashes no weight; a moved revision
// re-hashes what is here and fetches only what changed; a call that cannot
// reach the hub uses the receipt and says so. Every file that arrives is
// hashed against the release's SHA256SUMS, once, here.
func Download(repo, dir string) (string, error) {
	return DownloadWith(repo, dir, Fetch{})
}

// DownloadWith is Download with the revision, the enrollment graphs and the
// progress writer named.
func DownloadWith(repo, dir string, f Fetch) (string, error) {
	return DownloadContext(context.Background(), repo, dir, f)
}

// refuseACacheThatIsNotThePin answers a pinned request offline with those
// bytes or with nothing.
//
// A caller asking for one revision and getting another is the failure a pin
// exists to prevent, and a warning is not a refusal: a build that pins its
// weights would run on different ones and report success.
//
// Both halves of the pin are already on disk. A forty-hex request names the
// commit, which the receipt records, so it is answered exactly. A tag or a
// branch cannot be resolved with no hub, but the receipt also records the ref
// it was fetched for, and asking for a different one cannot be satisfied from
// this directory either. An empty revision is no pin at all, so the cache
// answers it.
func refuseACacheThatIsNotThePin(dir string, r *receipt, revision string) error {
	if revision == "" {
		return nil
	}
	if commitHex.MatchString(revision) {
		if revision == r.Commit {
			return nil
		}
		return fmt.Errorf("%s: holds %s at %s, and the hub cannot be reached to "+
			"fetch %s. A pinned revision is not a preference: rerun with the hub "+
			"reachable, or point the download directory at a copy of that commit",
			dir, r.Repo, r.Commit, revision)
	}
	if revision == r.Revision {
		return nil
	}
	return fmt.Errorf("%s: was fetched for %q (%s), the hub cannot be reached, and "+
		"%q cannot be resolved without it. Rerun with the hub reachable, or ask "+
		"for the revision this copy holds", dir, r.Revision, r.Commit, revision)
}

// DownloadContext is DownloadWith under a caller's context, which is the form
// to reach for inside a server: a release is hundreds of megabytes over a link
// this package sets no deadline on, so without one the fetch runs to its own
// end whatever the caller decided. Cancelling stops the transfer and returns
// the context's error; a part-file stays behind and the next call resumes it.
//
// A cancelled or expired context is never mistaken for a hub that cannot be
// reached, so it does not fall back to the receipt.
func DownloadContext(ctx context.Context, repo, dir string, f Fetch) (string, error) {
	repo = modelRepo(repo)
	if !isRepoID(repo) {
		return "", fmt.Errorf("%q is not a Hugging Face repo id (those look like "+
			"'org/name', such as 'loudreader/loudr-1')", repo)
	}
	progress := f.Progress
	if progress == nil {
		progress = os.Stderr
	}
	revision := f.Revision
	if revision == "" {
		revision = "main"
	}
	client := newHubClient()
	commit, err := client.commit(ctx, repo, revision)
	complete := func() bool { return checkInventory(dir, f.Cloning, isOfficial(repo)) == nil }
	if err != nil {
		if !unreachable(err) {
			return "", err
		}
		r := readReceipt(dir, repo, f.Cloning)
		if r == nil || !complete() {
			// A receipt for the synthesis set does not cover an enrollment:
			// the sentence names the graphs, not the transport.
			if f.Cloning && readReceipt(dir, repo, false) != nil && checkInventory(dir, false, isOfficial(repo)) == nil {
				return "", fmt.Errorf("%s: the hub cannot be reached, and %s holds no enrollment "+
					"graphs (%s). Connect once to fetch them", repo, dir, strings.Join(onnxEnroll, ", "))
			}
			return "", err
		}
		if err := refuseACacheThatIsNotThePin(dir, r, f.Revision); err != nil {
			return "", err
		}
		fmt.Fprintf(progress, "loudkit: the hub cannot be reached; using %s, which holds "+
			"%s at %s (fetched %s)\n", dir, r.Repo, r.Commit, r.FetchedAt)
		return dir, nil
	}
	if receiptHit(dir, repo, commit, f.Cloning) && complete() {
		return dir, nil
	}
	// A stale receipt must not outlive the files it vouched for.
	os.Remove(filepath.Join(dir, receiptName))
	listing, err := client.tree(ctx, repo, revision)
	if err != nil {
		return "", err
	}
	wanted := selectFiles(listing, f.Cloning)
	if len(wanted) == 0 {
		return "", fmt.Errorf("%s at revision %s holds none of the files an onnx "+
			"release is made of", repo, revision)
	}
	if err := refuseBeforeFetching(repo, revision, wanted); err != nil {
		return "", err
	}
	total := int64(0)
	for _, file := range wanted {
		total += file.Size
	}
	fmt.Fprintf(progress, "loudkit: %s -> %s (%d files, %s)\n",
		repo, dir, len(wanted), humanBytes(total))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	// The two bookkeeping files first, so a repo that is not a release is
	// refused before its weights move. SHA256SUMS cannot vouch for itself and
	// always travels; a file already here that hashes to its new entry is
	// kept, and anything else is fetched.
	var sums map[string]string
	hashed := 0
	for _, file := range bookkeepingFirst(wanted) {
		dest, err := safeJoin(dir, file.Path)
		if err != nil {
			return "", err
		}
		kept := file.Path != sumsName && keptByHash(dest, file.Path, sums)
		if !kept {
			fmt.Fprintf(progress, "  %s  %s\n", file.Path, humanBytes(file.Size))
			if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
				return "", err
			}
			os.Remove(dest)
			if err := client.fetch(ctx, repo, revision, file.Path, dest, file.Size); err != nil {
				return "", err
			}
		}
		switch file.Path {
		case sumsName:
			if sums, err = parseSums(dest); err != nil {
				return "", err
			}
			continue
		case releaseRecord:
			if isOfficial(repo) {
				if err := requireReleasable(dir, repo, sums); err != nil {
					return "", err
				}
			}
		}
		if kept {
			// keptByHash matched this file against its SHA256SUMS entry to
			// decide to keep it, so it is hashed and verified either way.
			hashed++
			continue
		}
		listed, err := verifyFetched(dir, repo, file.Path, sums, progress)
		if err != nil {
			return "", err
		}
		if listed {
			hashed++
		}
	}
	// Counted, not assumed: only files whose content hash was actually
	// checked. verifyFetched returns before hashing when the repo carries no
	// SHA256SUMS, and counting those would report a directory as verified
	// that nothing had vouched for.
	if hashed > 0 {
		fmt.Fprintf(progress, "loudkit: verified %d files against %s\n", hashed, sumsName)
	}
	if err := checkInventory(dir, f.Cloning, isOfficial(repo)); err != nil {
		return "", err
	}
	if err := writeReceipt(dir, repo, revision, commit); err != nil {
		return "", err
	}
	fmt.Fprintf(progress, "loudkit: %s is ready\n", dir)
	return dir, nil
}

// unreachable says whether err is the transport failing rather than the hub
// answering: only then may a receipt stand in for the hub. A caller who
// cancelled, or whose deadline passed, asked for no such substitute, and the
// transport reports both as its own failure.
func unreachable(err error) bool {
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	var transport *url.Error
	return errors.As(err, &transport)
}

// refuseBeforeFetching is what the listing alone can say about a repo.
//
// An official repo must ship SHA256SUMS and release.json, and a repo without
// the onnx graphs cannot run here at all, whatever else it carries. Both are
// refused before a byte moves: the alternative is a gigabyte followed by a
// sentence about missing files.
func refuseBeforeFetching(repo, revision string, wanted []remoteFile) error {
	has := func(name string) bool {
		for _, file := range wanted {
			if file.Path == name {
				return true
			}
		}
		return false
	}
	if isOfficial(repo) {
		for _, name := range []string{sumsName, releaseRecord} {
			if !has(name) {
				return fmt.Errorf("%s at %s: no %s. Every %s release ships one, so this "+
					"cannot be checked against anything and will not be fetched. Pin a "+
					"revision you trust", repo, revision, name, officialOrg)
			}
		}
	}
	// Every graph but the step one, which the two decode modes spell differently.
	for _, graph := range withoutStep() {
		if !has(graph) {
			return fmt.Errorf("%s at %s ships no %s, which this port runs on. Use "+
				"loudreader/loudr-1, or pin a revision that carries the graphs",
				repo, revision, graph)
		}
	}
	if !has(stepGraph) && !(has(pairStepGraph) && has(head2Graph)) {
		return fmt.Errorf("%s at %s ships no %s or complete pair/head2 graph set", repo, revision, stepGraph)
	}
	return nil
}

func modelRepo(ref string) string {
	if ref == "loudr-1" || ref == "loudr-1-turbo" {
		return "loudreader/" + ref
	}
	return ref
}

// bookkeepingFirst puts SHA256SUMS and release.json ahead of everything else,
// keeping the rest sorted.
func bookkeepingFirst(files []remoteFile) []remoteFile {
	rank := func(name string) int {
		switch name {
		case sumsName:
			return 0
		case releaseRecord:
			return 1
		}
		return 2
	}
	out := append([]remoteFile(nil), files...)
	sort.SliceStable(out, func(i, j int) bool { return rank(out[i].Path) < rank(out[j].Path) })
	return out
}

// selectFiles is the fetch plan: everything an allow pattern names and no
// ignore pattern does, sorted so a run reads the same way twice.
func selectFiles(listing []remoteFile, cloning bool) []remoteFile {
	allow := append([]string{}, releaseCore...)
	allow = append(allow, onnxSynthesis...)
	allow = append(allow, pairStepGraph, head2Graph)
	allow = append(allow, onnxExportRecord)
	if cloning {
		allow = append(allow, onnxEnroll...)
	}
	var out []remoteFile
	for _, file := range listing {
		if !matchAny(allow, file.Path) || matchAny(ignoreAlways, file.Path) {
			continue
		}
		out = append(out, file)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Path < out[j].Path })
	return out
}

// matchAny is Python's fnmatch, where `*` crosses `/`, which is why
// `*.safetensors` names the checkpoint and every voice under voices/.
func matchAny(patterns []string, name string) bool {
	for _, pattern := range patterns {
		if globMatch(pattern, name) {
			return true
		}
	}
	return false
}

func globMatch(pattern, name string) bool {
	p, n := 0, 0
	star, mark := -1, 0
	for n < len(name) {
		switch {
		case p < len(pattern) && (pattern[p] == '?' || pattern[p] == name[n]):
			p++
			n++
		case p < len(pattern) && pattern[p] == '*':
			star, mark = p, n
			p++
		case star >= 0:
			p = star + 1
			mark++
			n = mark
		default:
			return false
		}
	}
	for p < len(pattern) && pattern[p] == '*' {
		p++
	}
	return p == len(pattern)
}

// Letters and numbers, not \w: a repo id is a Hub name and Hub names are
// Unicode, so łódź/model is one. RE2's \w is ASCII, as JS's is, and the
// reference, Rust and Swift all classify by character rather than by byte,
// which made the same id a repo id in three ports and a missing file in two.
var repoID = regexp.MustCompile(`^[\p{L}\p{N}_.\-]+/[\p{L}\p{N}_.\-]+$`)

// isRepoID is org/name, and never something that exists on disk: a path that
// exists is a path, however it is spelled.
func isRepoID(ref string) bool {
	if ref == "" {
		return false
	}
	if _, err := os.Stat(ref); err == nil {
		return false
	}
	if strings.HasPrefix(ref, ".") || strings.HasPrefix(ref, "/") ||
		strings.HasPrefix(ref, "~") || strings.HasSuffix(ref, voiceSuffix) {
		return false
	}
	return repoID.MatchString(ref)
}

// isOfficial says whether repo is one this project publishes, which is the
// only case where a missing SHA256SUMS or release.json is evidence of a
// release that lost files rather than of a stranger's bare upload.
func isOfficial(repo string) bool {
	org, _, ok := strings.Cut(repo, "/")
	return ok && strings.EqualFold(org, officialOrg)
}

// safeJoin refuses a listed name that would write outside dir, by the same
// rule the release manifest is parsed with.
func safeJoin(dir, name string) (string, error) {
	if why := rejectedName(name); why != "" {
		return "", fmt.Errorf("%s: the listing names a file that %s; refusing to "+
			"write it", name, why)
	}
	return filepath.Join(dir, filepath.FromSlash(name)), nil
}

// remoteFile is one entry of a repo's file listing.
type remoteFile struct {
	Path string
	Size int64
}

const defaultEndpoint = "https://huggingface.co"

// maxListingPages is the most pages of a file listing that are followed, the
// same bound Swift keeps.
const maxListingPages = 100

// hubClient is the Hugging Face HTTP API, which is two endpoints: a file
// listing and a byte range per file.
type hubClient struct {
	endpoint string
	http     *http.Client
	token    string
}

func newHubClient() *hubClient {
	endpoint := os.Getenv("HF_ENDPOINT")
	if endpoint == "" {
		endpoint = defaultEndpoint
	}
	// No client timeout: a deadline that fits a listing cannot fit a 747 MB
	// checkpoint. The transport's own dial and header timeouts still apply,
	// and a caller who wants a bound on the whole fetch passes a context.
	return &hubClient{
		endpoint: strings.TrimRight(endpoint, "/"),
		http:     &http.Client{},
		token:    os.Getenv("HF_TOKEN"),
	}
}

// commit is the sha the hub resolves revision to today: one request, the
// same one that says whether the repo and the revision exist at all.
func (c *hubClient) commit(ctx context.Context, repo, revision string) (string, error) {
	url := fmt.Sprintf("%s/api/models/%s/revision/%s", c.endpoint, repo, revision)
	resp, err := c.get(ctx, url, repo, revision)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	var info struct {
		SHA string `json:"sha"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return "", fmt.Errorf("%s: unreadable revision record: %w", repo, err)
	}
	if info.SHA == "" {
		return "", fmt.Errorf("%s at %s: the hub named no commit for this revision", repo, revision)
	}
	return info.SHA, nil
}

// tree lists every file in a repo at one revision, following the API's
// pagination.
func (c *hubClient) tree(ctx context.Context, repo, revision string) ([]remoteFile, error) {
	url := fmt.Sprintf("%s/api/models/%s/tree/%s?recursive=1", c.endpoint, repo, revision)
	var out []remoteFile
	for pages := 0; url != ""; pages++ {
		// A release is tens of files and the tree API pages by the thousand, so
		// a listing past this is a Link: rel="next" cycle rather than a large
		// repository, and following it never returns.
		if pages >= maxListingPages {
			return nil, fmt.Errorf("%s: the file listing does not end", repo)
		}
		resp, err := c.get(ctx, url, repo, revision)
		if err != nil {
			return nil, err
		}
		var page []struct {
			Type string `json:"type"`
			Path string `json:"path"`
			Size int64  `json:"size"`
			LFS  *struct {
				Size int64 `json:"size"`
			} `json:"lfs"`
		}
		err = json.NewDecoder(resp.Body).Decode(&page)
		next := nextLink(resp.Header.Get("Link"))
		resp.Body.Close()
		if err != nil {
			return nil, fmt.Errorf("%s: unreadable file listing: %w", repo, err)
		}
		for _, e := range page {
			if e.Type != "file" {
				continue
			}
			size := e.Size
			if e.LFS != nil && e.LFS.Size > 0 {
				size = e.LFS.Size
			}
			out = append(out, remoteFile{Path: e.Path, Size: size})
		}
		url = next
	}
	return out, nil
}

// fetch downloads one file, resuming a `.part` left by an interrupted run.
func (c *hubClient) fetch(ctx context.Context, repo, revision, name, dest string, size int64) error {
	url := fmt.Sprintf("%s/%s/resolve/%s/%s", c.endpoint, repo, revision, name)
	part := dest + ".part"
	var have int64
	if info, err := os.Stat(part); err == nil && info.Size() < size {
		// Only a short `.part` resumes: one at or past the file's length has
		// nothing left to ask for, and the range request comes back 416.
		have = info.Size()
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
	if have > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", have))
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	defer resp.Body.Close()
	flags := os.O_CREATE | os.O_WRONLY
	switch resp.StatusCode {
	case http.StatusPartialContent:
		// Appending to a part-file on the server's word alone is how a
		// half-file becomes a corrupt whole one: a 206 for a range other than
		// the one asked for, or for a file that has changed since the
		// interrupted run, lands bytes at an offset they do not belong to.
		if err := resumable(resp, have, size); err != nil {
			os.Remove(part)
			return fmt.Errorf("%s: %w. The part-file has been removed; run the "+
				"download again", name, err)
		}
		flags |= os.O_APPEND
	case http.StatusOK:
		// The server ignored the range, or there was nothing to resume.
		have = 0
		flags |= os.O_TRUNC
	default:
		return httpError(resp, repo, revision, name)
	}
	f, err := os.OpenFile(part, flags, 0o644)
	if err != nil {
		return err
	}
	written, err := io.Copy(f, resp.Body)
	if err != nil {
		f.Close()
		return fmt.Errorf("%s: %w", name, err)
	}
	if err := f.Close(); err != nil {
		return err
	}
	// The listing already said how long this file is, and nothing else
	// checks. Unchecked, a body cut short by a dropped connection or a proxy
	// is renamed into place, hashed against nothing when the repo ships no
	// SHA256SUMS, and written into the receipt, after which every later run
	// trusts the receipt and never looks again.
	if got := have + written; size > 0 && got != size {
		if got > size {
			// Longer than the listing said: the two cannot both be right, and
			// resuming into it would only append more.
			os.Remove(part)
			return fmt.Errorf("%s: the listing says %d bytes and the download carried "+
				"%d. The part-file has been removed; run the download again",
				name, size, got)
		}
		// Short. The part-file stays: the next call resumes it, which is what
		// an interrupted transfer is supposed to do.
		return fmt.Errorf("%s: the listing says %d bytes and the download stopped at "+
			"%d. Run the download again to resume it", name, size, got)
	}
	return os.Rename(part, dest)
}

// resumable checks a 206 against the range that was asked for.
//
// Content-Range is `bytes <first>-<last>/<total>`, or `bytes <first>-<last>/*`
// from a server that will not say the total. The first byte has to be the one
// the part-file ended at, and a stated total has to be the length the listing
// gave, or the part-file and the body are two different files.
func resumable(resp *http.Response, have, size int64) error {
	raw := resp.Header.Get("Content-Range")
	if raw == "" {
		return errors.New("the server answered 206 with no Content-Range, so " +
			"where its bytes belong in the file is unknown")
	}
	var first, last, total int64
	if n, err := fmt.Sscanf(raw, "bytes %d-%d/%d", &first, &last, &total); n < 2 || err != nil {
		if n, err := fmt.Sscanf(raw, "bytes %d-%d/*", &first, &last); n < 2 || err != nil {
			return fmt.Errorf("the server answered 206 with an unreadable "+
				"Content-Range %q", raw)
		}
		total = 0
	}
	if first != have {
		return fmt.Errorf("the part-file ends at byte %d and the server sent from "+
			"byte %d", have, first)
	}
	if total > 0 && size > 0 && total != size {
		return fmt.Errorf("the listing says %d bytes and the server says %d, so the "+
			"file changed since the interrupted run", size, total)
	}
	return nil
}

func (c *hubClient) get(ctx context.Context, url, repo, revision string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", repo, err)
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		return nil, httpError(resp, repo, revision, "")
	}
	return resp, nil
}

// httpError says what a reader can act on. The hub's own body for a missing
// repo talks about invalid credentials, which diagnoses a problem the caller
// usually does not have.
func httpError(resp *http.Response, repo, revision, name string) error {
	where := repo
	if name != "" {
		where = repo + "/" + name
	}
	switch resp.StatusCode {
	case http.StatusUnauthorized, http.StatusForbidden:
		return fmt.Errorf("%s: not public. Set HF_TOKEN to a token that can read it", where)
	case http.StatusNotFound:
		return fmt.Errorf("%s: not found at revision %s. Is the release published, "+
			"and are you online?", where, revision)
	}
	return fmt.Errorf("%s: the hub answered %s", where, resp.Status)
}

var linkNext = regexp.MustCompile(`<([^>]+)>;\s*rel="next"`)

func nextLink(header string) string {
	if m := linkNext.FindStringSubmatch(header); m != nil {
		return m[1]
	}
	return ""
}

func humanBytes(n int64) string {
	const unit = 1000
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	value, exp := float64(n)/unit, 0
	for value >= unit && exp < 3 {
		value /= unit
		exp++
	}
	return fmt.Sprintf("%.1f %sB", value, []string{"k", "M", "G", "T"}[exp])
}
