package loudkit

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/loudreader/loudkit/go/config"
)

// The layout of a release, mirroring loudkit.release. One repository is
// the source of truth for every backend and every port, so what varies between
// ports is the fetch, never the names.
const (
	checkpointName   = "loudr-1.safetensors"
	turboName        = "loudr-1-turbo.safetensors"
	enrollmentName   = "loudr-1-enrollment.safetensors"
	voiceEncoderName = "ve.safetensors"
	voiceDir         = "voices"
	voiceSuffix      = ".safetensors"
	tokenizerName    = "tokenizer.json"
	manifestName     = "manifest.json"
	onnxDirName      = "onnx"

	synthesisRole  = "synthesis"
	enrollmentRole = "enrollment"
)

// The graph names that vary with the decode mode: a single-step release ships
// stepGraph, a fusion one the other two. They are named because three places
// ask which is which, and none of them should ask by position.
const (
	stepGraph     = "onnx/t3_step.onnx"
	pairStepGraph = "onnx/t3_pair_step.onnx"
	head2Graph    = "onnx/t3_head2.onnx"
)

// onnxSynthesis are the graphs the engine loads, and onnxEnroll the three more
// a clone needs. Both lists are the release's own names, not this port's.
var (
	onnxSynthesis = []string{
		"onnx/t3_cond.onnx",
		"onnx/t3_prefill.onnx",
		stepGraph,
		"onnx/flow_encoder.onnx",
		"onnx/flow_estimator.onnx",
		"onnx/vocoder.onnx",
	}
	// onnxExportRecord says the graphs came from one export of one checkpoint,
	// and the loader holds them to it. Fetched with them, never required of a
	// fetch: a set exported before the record exists loads on a warning, so
	// demanding one here would refuse exactly those releases at the download,
	// which is the harder failure.
	onnxExportRecord = "onnx/export.json"
	onnxEnroll       = []string{
		"onnx/s3_tokenizer.onnx",
		"onnx/camp.onnx",
		"onnx/voice_encoder.onnx",
	}
)

// Bundle is a release on disk: the three paths the engine needs, found rather
// than passed.
type Bundle struct {
	Root       string // the release directory
	Checkpoint string // packed synthesis checkpoint
	ONNXDir    string // exported graphs
	Tokenizer  string // tokenizer.json
}

// Open finds the checkpoint, the graph directory and the tokenizer inside a
// release directory: what Download wrote, or a tree laid out the same way.
func Open(dir string) (*Bundle, error) {
	info, err := os.Stat(dir)
	if err != nil {
		return nil, fmt.Errorf("%s: no release here. Download one first: "+
			"loudkit.Download(\"loudreader/loudr-1\", %q)", dir, dir)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("%s: a release is a directory holding a checkpoint, "+
			"onnx/ and tokenizer.json, not a single file", dir)
	}
	ckpt, err := findCheckpoint(dir)
	if err != nil {
		return nil, err
	}
	graphs, err := synthesisGraphs(ckpt)
	if err != nil {
		return nil, err
	}

	b := &Bundle{
		Root:       dir,
		Checkpoint: ckpt,
		ONNXDir:    filepath.Join(dir, onnxDirName),
		Tokenizer:  filepath.Join(dir, tokenizerName),
	}
	var missing []string
	for _, rel := range graphs {
		if !isFile(filepath.Join(dir, filepath.FromSlash(rel))) {
			missing = append(missing, rel)
		}
	}
	if !isFile(b.Tokenizer) {
		// Python lets a checkpoint carry the tokenizer as a packed asset; this
		// port loads it from a file, so the sibling is required here.
		missing = append(missing, tokenizerName)
	}
	if len(missing) > 0 {
		return nil, fmt.Errorf("%s: this is not a usable onnx release, missing: %s. "+
			"Re-run Download, which fetches the whole set",
			dir, strings.Join(missing, ", "))
	}
	return b, nil
}

// CanEnroll reports whether the bundle carries the three enrollment graphs,
// which only a Download with Cloning fetches.
func (b *Bundle) CanEnroll() bool {
	for _, rel := range onnxEnroll {
		if !isFile(filepath.Join(b.Root, filepath.FromSlash(rel))) {
			return false
		}
	}
	return true
}

// Voices are the names Voice will accept, sorted.
func (b *Bundle) Voices() []string {
	entries, err := os.ReadDir(filepath.Join(b.Root, voiceDir))
	if err != nil {
		return nil
	}
	var names []string
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), voiceSuffix) {
			continue
		}
		names = append(names, strings.TrimSuffix(e.Name(), voiceSuffix))
	}
	sort.Strings(names)
	return names
}

// VoicePath is the file for one voice name.
//
// A voice is named, not addressed: a name carrying a separator would join into
// a path outside the release, so it is refused rather than resolved.
func (b *Bundle) VoicePath(name string) (string, error) {
	base := strings.TrimSuffix(name, voiceSuffix)
	if base == "" || strings.ContainsAny(base, `/\`) || strings.HasPrefix(base, ".") {
		return "", fmt.Errorf("%q: a voice is named, not addressed. Pass a bare name "+
			"such as \"joe\"", name)
	}
	path := filepath.Join(b.Root, voiceDir, base+voiceSuffix)
	if !isFile(path) {
		known := b.Voices()
		if len(known) == 0 {
			return "", fmt.Errorf("%q: no such voice, and %s holds none",
				name, filepath.Join(b.Root, voiceDir))
		}
		return "", fmt.Errorf("%q: no such voice. This release has: %s",
			name, strings.Join(known, ", "))
	}
	return path, nil
}

// findCheckpoint is the synthesis artefact in dir, by the rules
// loudkit.release resolves with: exactly one canonical name if one is there,
// otherwise the file whose own manifest claims the synthesis role, otherwise
// the single remaining candidate.
//
// Voices are safetensors too and live in voices/, so the search is the root
// level only. ve.safetensors is a released file at that level and is not a
// checkpoint, which is why it is named here.
func findCheckpoint(dir string) (string, error) {
	var canonical []string
	for _, name := range []string{checkpointName, turboName} {
		if isFile(filepath.Join(dir, name)) {
			canonical = append(canonical, name)
		}
	}
	if len(canonical) == 1 {
		named := filepath.Join(dir, canonical[0])
		if err := refuseRole(named, synthesisRole); err != nil {
			return "", err
		}
		return named, nil
	}
	// Two released models under their own names is not an ambiguity a resolver
	// may settle: they are different releases with different decode loops, so
	// picking either is picking a model the caller did not ask for.
	if len(canonical) > 1 {
		return "", fmt.Errorf("%s: %d models here (%s): name the one you mean. They "+
			"are different releases with different decode loops, so there is no "+
			"right one to pick", dir, len(canonical), strings.Join(canonical, ", "))
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return "", err
	}
	var found []string
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, voiceSuffix) {
			continue
		}
		if name == voiceEncoderName || strings.HasPrefix(name, ".") {
			continue
		}
		found = append(found, filepath.Join(dir, name))
	}
	sort.Strings(found)
	var declared, candidates []string
	for _, path := range found {
		switch artifactRole(path) {
		case synthesisRole:
			declared = append(declared, path)
			candidates = append(candidates, path)
		case enrollmentRole:
		default:
			candidates = append(candidates, path)
		}
	}
	if len(declared) == 1 {
		return declared[0], nil
	}
	if len(declared) == 0 && len(candidates) == 1 {
		return candidates[0], nil
	}
	if len(found) == 0 {
		return "", fmt.Errorf("%s: no checkpoint here. A loudkit release is a "+
			"synthesis checkpoint beside onnx/, tokenizer.json and voices/", dir)
	}
	if len(candidates) == 0 {
		return "", fmt.Errorf("%s: the only checkpoint here is a release's enrollment "+
			"artefact. Fetch the release's %s beside it", dir, checkpointName)
	}
	return "", fmt.Errorf("%s: %d checkpoints (%s); name the one you mean",
		dir, len(candidates), strings.Join(baseNames(candidates), ", "))
}

// synthesisGraphs selects the runtime inventory from the checkpoint manifest.
func synthesisGraphs(path string) ([]string, error) {
	mode := config.DecodeSingle
	if block, ok := checkpointManifest(path)["decode"].(map[string]any); ok {
		if named, ok := block["mode"].(string); ok {
			mode = named
		}
	}
	switch mode {
	case config.DecodeSingle:
		return onnxSynthesis, nil
	case config.DecodeFusionMTP2:
		return withoutStep(pairStepGraph, head2Graph), nil
	default:
		return nil, fmt.Errorf("unsupported decode mode %q", mode)
	}
}

// withoutStep is onnxSynthesis with the single-step graph replaced by the
// graphs named, or dropped when none are. It finds the step graph by name and
// keeps the order the list is written in, so the list can be reordered
// without quietly changing what a fetch refuses or an inventory expects.
func withoutStep(replacement ...string) []string {
	out := make([]string, 0, len(onnxSynthesis)+len(replacement))
	for _, graph := range onnxSynthesis {
		if graph == stepGraph {
			out = append(out, replacement...)
			continue
		}
		out = append(out, graph)
	}
	return out
}

// artifactRole is manifest["artifact_role"] for a checkpoint, or "" for a file
// that makes no claim.
//
// Absence is the pre-split release, which carries every tensor and must keep
// loading, so the field is read to refuse a file and never to require one.
func artifactRole(path string) string {
	manifest := checkpointManifest(path)
	role, _ := manifest["artifact_role"].(string)
	return role
}

// refuseRole refuses a file whose manifest declares it to be the other
// artefact.
func refuseRole(path, expected string) error {
	if role := artifactRole(path); role != "" && role != expected {
		return fmt.Errorf("%s: this is a release's %s artefact, and the %s artefact "+
			"is what was asked for. Pass the release directory, or the repo id, and "+
			"let the resolver pick", path, role, expected)
	}
	return nil
}

// checkpointManifest reads a safetensors header without reading the payload.
//
// The checkpoint is 747 MB and this runs while deciding which of two files to
// open, so only the header is read.
func checkpointManifest(path string) map[string]any {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	var length [8]byte
	if _, err := io.ReadFull(f, length[:]); err != nil {
		return nil
	}
	size := binary.LittleEndian.Uint64(length[:])
	// A header is JSON describing tensors; anything this big is not one, and
	// reading it would be the whole-file read this exists to avoid.
	if size == 0 || size > 64<<20 {
		return nil
	}
	body := make([]byte, size)
	if _, err := f.ReadAt(body, 8); err != nil {
		return nil
	}
	var header struct {
		Metadata struct {
			Manifest string `json:"manifest"`
		} `json:"__metadata__"`
	}
	if err := json.Unmarshal(body, &header); err != nil {
		return nil
	}
	var manifest map[string]any
	if err := json.Unmarshal([]byte(header.Metadata.Manifest), &manifest); err != nil {
		return nil
	}
	return manifest
}

func isFile(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.Mode().IsRegular()
}

func baseNames(paths []string) []string {
	out := make([]string, len(paths))
	for i, p := range paths {
		out[i] = filepath.Base(p)
	}
	return out
}
