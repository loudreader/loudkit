package loudkit

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/loudreader/loudkit/go/internal/digest"
)

const (
	sumsName      = "SHA256SUMS"
	releaseRecord = "release.json"
	officialOrg   = "loudreader"
)

// strictProfiles are the profiles a builder stamps on a releasable bundle:
// full-0.1 for loudr-1 and turbo-0.1 for loudr-1-turbo. Neither is written
// until the builder's load-and-speak gate passes.
var strictProfiles = map[string]bool{"full-0.1": true, "turbo-0.1": true}

// verifyFetched hashes one file that this run fetched against the release's
// SHA256SUMS. Three rules, shared with loudkit.checksums: no manifest is
// fine only for a stranger's repo; a fetched file the manifest does not list
// is refused when it is weights, refused under an official repo whatever it
// is, and reported otherwise; and a digest that does not match is refused,
// with the file removed so the next run fetches it again.
//
// The first return says whether the file was actually hashed against an entry.
// Two of the paths below verify nothing (a repo with no SHA256SUMS, and a
// stranger's file the manifest does not list), and the caller counts what it
// reports having verified.
func verifyFetched(root, repo, name string, sums map[string]string, progress io.Writer) (bool, error) {
	if sums == nil {
		return false, nil
	}
	target := filepath.Join(root, filepath.FromSlash(name))
	want, listed := sums[name]
	if !listed {
		if strings.HasSuffix(name, voiceSuffix) {
			os.Remove(target)
			return false, fmt.Errorf("%s: %s does not list %s. These are weights loudkit "+
				"would open with nothing vouching for them. Pin a revision you trust",
				root, sumsName, name)
		}
		if isOfficial(repo) {
			os.Remove(target)
			return false, fmt.Errorf("%s: %s does not list %s. A %s release checksums every "+
				"file it ships, so this did not come from the release", root, sumsName,
				name, officialOrg)
		}
		fmt.Fprintf(progress, "loudkit: warning: %s is not covered by %s and therefore "+
			"not verified\n", name, sumsName)
		return false, nil
	}
	got, err := digest.SHA256File(target)
	if err != nil {
		return false, err
	}
	if got != want {
		os.Remove(target)
		return false, fmt.Errorf("%s: %s failed the release checksum. The file has been "+
			"removed; run the download again, or pin a revision you trust", root, name)
	}
	return true, nil
}

var sumsLine = regexp.MustCompile(`^([0-9a-f]{64})  (\S.*)$`)

// parseSums reads a SHA256SUMS, refusing a line it cannot understand, a name
// that escapes the release and a name listed twice.
func parseSums(sums string) (map[string]string, error) {
	f, err := os.Open(sums)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	out := map[string]string{}
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), 1<<20)
	for number := 1; scanner.Scan(); number++ {
		line := strings.TrimRight(scanner.Text(), "\r")
		if strings.TrimSpace(line) == "" {
			continue
		}
		m := sumsLine.FindStringSubmatch(line)
		if m == nil {
			return nil, fmt.Errorf("%s: malformed line %d: %q. This does not look "+
				"like a loudkit release manifest; refusing to verify against it",
				sums, number, line)
		}
		name := m[2]
		if why := rejectedName(name); why != "" {
			return nil, fmt.Errorf("%s: line %d: %q %s; refusing to verify against "+
				"a manifest that names files outside the release it describes",
				sums, number, name, why)
		}
		if _, seen := out[name]; seen {
			return nil, fmt.Errorf("%s: line %d: duplicate entry for %q. The "+
				"manifest disagrees with itself about one file. Rebuild the release",
				sums, number, name)
		}
		out[name] = m[1]
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("%s: no checksum entries", sums)
	}
	return out, nil
}

var driveLetter = regexp.MustCompile(`^[A-Za-z]:/`)

// rejectedName says why a name cannot address bytes inside the release, or
// "" when it can.
func rejectedName(name string) string {
	if strings.Contains(name, `\`) {
		return "is not a POSIX path (it contains a backslash)"
	}
	if strings.HasPrefix(name, "/") || driveLetter.MatchString(name) {
		return "is absolute, and a manifest name is relative to the release root"
	}
	for _, p := range strings.Split(name, "/") {
		if p == "" || p == "." || p == ".." {
			return "is not a normalised relative path"
		}
	}
	if path.Clean(name) != name {
		return fmt.Sprintf("is not normalised (it names %q)", path.Clean(name))
	}
	return ""
}

// requireReleasable holds an official repo to its own release.json: a
// development bundle carries a perfectly valid SHA256SUMS and would otherwise
// verify and load like the release.
func requireReleasable(root, repo string, entries map[string]string) error {
	record := filepath.Join(root, releaseRecord)
	if !isFile(record) {
		return fmt.Errorf("%s (%s): no release.json. Every %s release records its "+
			"profile and its verified flag there, so this download cannot prove it "+
			"is a release. Pin a revision you trust", repo, root, officialOrg)
	}
	// The record is itself covered by the manifest, so a release.json nothing
	// vouches for is the same event as a missing one.
	want, listed := entries[releaseRecord]
	if !listed {
		return fmt.Errorf("%s: %s does not list release.json, so the record that "+
			"would vouch for this release is vouched for by nothing. Pin a "+
			"revision you trust", repo, sumsName)
	}
	got, err := digest.SHA256File(record)
	if err != nil {
		return err
	}
	if got != want {
		return fmt.Errorf("%s: release.json failed the release checksum. Delete the "+
			"directory and download again", record)
	}
	body, err := os.ReadFile(record)
	if err != nil {
		return err
	}
	var claim struct {
		Profile  string `json:"profile"`
		Verified bool   `json:"verified"`
	}
	if err := json.Unmarshal(body, &claim); err != nil {
		return fmt.Errorf("%s: release.json is unreadable (%v). Delete the "+
			"directory and download again", record, err)
	}
	if !strictProfiles[claim.Profile] {
		return fmt.Errorf("%s: release.json says profile %q, and an %s release is "+
			"one of %s. This is a development bundle, not the release", record,
			claim.Profile, officialOrg, strings.Join(profileNames(), ", "))
	}
	if !claim.Verified {
		return fmt.Errorf("%s: release.json does not record verified: true, so the "+
			"bundle never passed the builder's load-and-speak gate", record)
	}
	return nil
}

func profileNames() []string {
	out := make([]string, 0, len(strictProfiles))
	for name := range strictProfiles {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// keptByHash says whether the file already at target is the one the manifest
// lists under name: present, listed, and hashing to the listed digest. Size
// decides nothing; a file edited in place at its published length is caught
// here on the next miss.
func keptByHash(target, name string, sums map[string]string) bool {
	want, listed := sums[name]
	if !listed || !isFile(target) {
		return false
	}
	got, err := digest.SHA256File(target)
	return err == nil && got == want
}

const receiptName = ".loudkit-release.json"

// receiptLimit is the most a receipt file is read: five short fields. A
// file past it is not a receipt, and is not read.
const receiptLimit = 1 << 20

// receiptFields are the receipt's keys, in the order writeReceipt writes them.
var receiptFields = []string{"repo", "revision", "commit", "sha256sums", "fetched_at"}

// receipt is what a verified download directory carries, in every port: the
// commit the asked revision resolved to when the files were hashed. A later
// load whose revision still resolves to that commit, over a receipt
// readReceipt accepts, is a hit and hashes no weight. The shape and the
// rules are pinned by tests/data/conformance/release_receipt.json.
type receipt struct {
	Repo     string `json:"repo"`
	Revision string `json:"revision"`
	Commit   string `json:"commit"`
	// SHA256Sums is the digest of SHA256SUMS, or null for a release without one.
	SHA256Sums *string `json:"sha256sums"`
	FetchedAt  string  `json:"fetched_at"`
}

var commitHex = regexp.MustCompile(`^[0-9a-f]{40}$`)

// readReceipt is the receipt under dir when it vouches for repo, else nil:
// every field present with its type (null is the wrong type for all but
// sha256sums; keys it does not know are ignored), repo the one asked, commit
// forty lowercase hex, sha256sums the digest of the SHA256SUMS on disk (null
// for none), and every file it lists that the plan selects present. No
// weight is hashed: the receipt is checked, not the release. This is the
// only place the file is read.
func readReceipt(dir, repo string, cloning bool) *receipt {
	path := filepath.Join(dir, receiptName)
	if info, err := os.Stat(path); err != nil || info.Size() > receiptLimit {
		return nil
	}
	body, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	// Every string field decodes through a pointer: a plain string swallows
	// null as "", and null is a wrong type like any other.
	var raw struct {
		Repo       *string `json:"repo"`
		Revision   *string `json:"revision"`
		Commit     *string `json:"commit"`
		SHA256Sums *string `json:"sha256sums"`
		FetchedAt  *string `json:"fetched_at"`
	}
	var fields map[string]json.RawMessage
	if json.Unmarshal(body, &fields) != nil || json.Unmarshal(body, &raw) != nil {
		return nil
	}
	for _, key := range receiptFields {
		if _, ok := fields[key]; !ok {
			return nil
		}
	}
	if raw.Repo == nil || raw.Revision == nil || raw.Commit == nil || raw.FetchedAt == nil {
		return nil
	}
	r := &receipt{
		Repo:       *raw.Repo,
		Revision:   *raw.Revision,
		Commit:     *raw.Commit,
		SHA256Sums: raw.SHA256Sums,
		FetchedAt:  *raw.FetchedAt,
	}
	if r.Repo != repo || !commitHex.MatchString(r.Commit) {
		return nil
	}
	sums := filepath.Join(dir, sumsName)
	if !isFile(sums) {
		if r.SHA256Sums != nil {
			return nil
		}
		return r
	}
	got, err := digest.SHA256File(sums)
	if err != nil || r.SHA256Sums == nil || *r.SHA256Sums != got {
		return nil
	}
	listed, err := parseSums(sums)
	if err != nil {
		return nil
	}
	names := make([]remoteFile, 0, len(listed))
	for name := range listed {
		names = append(names, remoteFile{Path: name})
	}
	for _, file := range selectFiles(names, cloning) {
		if !isFile(filepath.Join(dir, filepath.FromSlash(file.Path))) {
			return nil
		}
	}
	return r
}

// writeReceipt records that dir holds repo at commit, verified.
func writeReceipt(dir, repo, revision, commit string) error {
	r := receipt{
		Repo:      repo,
		Revision:  revision,
		Commit:    commit,
		FetchedAt: time.Now().UTC().Format("2006-01-02T15:04:05Z"),
	}
	if sums := filepath.Join(dir, sumsName); isFile(sums) {
		got, err := digest.SHA256File(sums)
		if err != nil {
			return err
		}
		r.SHA256Sums = &got
	}
	body, err := json.MarshalIndent(r, "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, receiptName), append(body, '\n'), 0o644)
}

// receiptHit says whether dir already holds repo at commit: a receipt
// readReceipt accepts, naming that commit. An empty commit never hits:
// nothing was resolved.
func receiptHit(dir, repo, commit string, cloning bool) bool {
	r := readReceipt(dir, repo, cloning)
	return commit != "" && r != nil && r.Commit == commit
}

// checkInventory is the set the fetch plan promised, on disk. allow_patterns
// is a request and not a receipt: the hub answers with whatever subset the
// repo holds, so a release missing its graphs looks exactly like one that has
// them.
func checkInventory(root string, cloning, requireVoices bool) error {
	var missing []string
	ckpt, ckptErr := findCheckpoint(root)
	if ckptErr != nil {
		missing = append(missing, checkpointName)
	}
	expected := []string{manifestName, tokenizerName}
	graphs := onnxSynthesis
	if ckptErr == nil {
		var err error
		graphs, err = synthesisGraphs(ckpt)
		if err != nil {
			return err
		}
	}
	expected = append(expected, graphs...)
	if cloning {
		expected = append(expected, onnxEnroll...)
	}
	for _, rel := range expected {
		if !isFile(filepath.Join(root, filepath.FromSlash(rel))) {
			missing = append(missing, rel)
		}
	}
	if requireVoices && len(voiceFiles(root)) == 0 {
		missing = append(missing, voiceDir+"/*"+voiceSuffix)
	}
	if len(missing) == 0 {
		return nil
	}
	sort.Strings(missing)
	return fmt.Errorf("%s: this fetch does not add up to a usable onnx set. "+
		"missing: %s. The release does not carry these files, or the fetch was "+
		"interrupted; retry, or pin a revision that ships them",
		root, strings.Join(missing, ", "))
}

func voiceFiles(root string) []string {
	entries, err := os.ReadDir(filepath.Join(root, voiceDir))
	if err != nil {
		return nil
	}
	var out []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), voiceSuffix) {
			out = append(out, e.Name())
		}
	}
	return out
}
