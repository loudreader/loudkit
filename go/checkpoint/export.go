package checkpoint

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/loudreader/loudkit/go/config"
	"github.com/loudreader/loudkit/go/internal/digest"
)

// ExportRecord is what the exporter writes beside the graphs it wrote.
const ExportRecord = "export.json"

// exportTool writes it, and is what a refusal tells the reader to run.
const exportTool = "tools/export_onnx.py"

// recordKeys are the fields that make one export's identity, in the order a
// message prints them. The first three are the engine the graphs were traced
// against; the fourth is the renderer inside it.
var recordKeys = [...]string{
	"checkpoint_sha256",
	"algorithm_fingerprint",
	"euler_steps",
	"estimator_sha256",
}

// identity is one member's record, rendered for comparison. Each field holds
// the value the way Python's repr() prints it, so a set that agrees here
// agrees there and a refusal reads the same in both.
type identity [len(recordKeys)]string

// VerifyExport refuses a graph set whose members did not come from one export
// of this checkpoint.
//
// Graphs in one folder look like a set and need not be one: the exporters take
// a stage list, so a run that names the renderer leaves the other graphs as
// they were, and a mixed folder speaks one checkpoint's tokens through another
// checkpoint's renderer. ExportRecord beside the set records, per member, the
// checkpoint digest, the fingerprint, the step count and the estimator digest;
// every member must agree with the others and with the checkpoint being
// loaded.
//
// An absent record is a note on stderr rather than a refusal, because sets
// exported before the record exist and stranding them buys nothing a sentence
// cannot say. This port has no warning channel on the load path, and stderr is
// where Download writes the one warning it has, so the note goes there under
// the same prefix.
func (c *Checkpoint) VerifyExport(assets string, algorithm config.AlgorithmConfig, graphs []string) error {
	path := filepath.Join(assets, ExportRecord)
	body, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "loudkit: warning: %s carries no %s, so nothing says its "+
			"%d graphs came from one export of one checkpoint. Re-export with %s to "+
			"record it.\n", assets, ExportRecord, len(graphs), exportTool)
		return nil
	}
	if err != nil {
		return fmt.Errorf("%s: unreadable export record (%w)", path, err)
	}
	seen, err := recordedGraphs(path, body)
	if err != nil {
		return err
	}
	for _, name := range graphs {
		if _, ok := seen[name]; !ok {
			return fmt.Errorf("%s does not record %s, so it came from some other run "+
				"than the ones it does record. Re-export the set", path, name)
		}
	}
	var agreed identity
	for _, got := range seen {
		if agreed != (identity{}) && got != agreed {
			return fmt.Errorf("%s is a mixed graph set: its members were exported from "+
				"different inputs:\n%s\nRe-export every stage together", assets, rows(seen))
		}
		agreed = got
	}
	fileDigest, err := digest.SHA256File(c.path)
	if err != nil {
		return err
	}
	want := identity{
		quote(fileDigest),
		quote(config.Fingerprint(algorithm)),
		strconv.Itoa(algorithm.EulerSteps),
	}
	if agreed[0] != want[0] || agreed[1] != want[1] || agreed[2] != want[2] {
		return fmt.Errorf("%s was exported from a different engine than the one loading "+
			"it:\n  graphs: %s\n  checkpoint: %s\nRe-export against %s",
			assets, engineFields(agreed), engineFields(want), filepath.Base(c.path))
	}
	// A set traced from a swapped-in estimator agrees with itself and describes
	// a renderer the checkpoint does not; compared only when both sides record
	// one.
	estimator := c.packedEstimator()
	if agreed[3] != "None" && estimator != "" && agreed[3] != quote(estimator) {
		return fmt.Errorf("%s was traced with an estimator the checkpoint was not "+
			"packed from:\n  traced:  %s\n  packed:  %s\nThat is a different renderer "+
			"than %s describes, and its fingerprint does not cover the difference. "+
			"Re-export without --estimator-ckpt, or pack the estimator you traced",
			assets, unquote(agreed[3]), estimator, filepath.Base(c.path))
	}
	return nil
}

// recordedGraphs reads the record's graph block. An absent block is an
// unreadable record; a block that is not a mapping of name to its fields is
// named as itself, because the remedy differs.
func recordedGraphs(path string, body []byte) (map[string]identity, error) {
	var root map[string]json.RawMessage
	if err := json.Unmarshal(body, &root); err != nil {
		return nil, fmt.Errorf("%s: unreadable export record (%w)", path, err)
	}
	block, ok := root["graphs"]
	if !ok {
		return nil, fmt.Errorf("%s: unreadable export record (no 'graphs' block)", path)
	}
	malformed := fmt.Errorf("%s: the 'graphs' block is not a mapping of name to its "+
		"export record. Re-export the set with %s", path, exportTool)
	var entries map[string]map[string]interface{}
	if err := json.Unmarshal(block, &entries); err != nil {
		return nil, malformed
	}
	seen := make(map[string]identity, len(entries))
	for name, entry := range entries {
		got, ok := asIdentity(entry)
		if !ok {
			return nil, malformed
		}
		seen[name] = got
	}
	return seen, nil
}

// asIdentity reads one member's entry, refusing a value shape the exporter
// never writes. An absent field and a null one are the same answer, so a
// record written before a field existed still compares.
func asIdentity(entry map[string]interface{}) (identity, bool) {
	var out identity
	if entry == nil {
		return out, false
	}
	for i, key := range recordKeys {
		switch value := entry[key].(type) {
		case nil:
			out[i] = "None"
		case string:
			out[i] = quote(value)
		case float64:
			if value != math.Trunc(value) {
				return out, false
			}
			out[i] = strconv.FormatFloat(value, 'f', -1, 64)
		default:
			return out, false
		}
	}
	return out, true
}

// packedEstimator is the sha256 of the estimator this checkpoint was packed
// from, if its sources say.
func (c *Checkpoint) packedEstimator() string {
	sources, ok := c.Manifest["sources"].(map[string]interface{})
	if !ok {
		return ""
	}
	for _, value := range sources {
		entry, ok := value.(map[string]interface{})
		if !ok {
			continue
		}
		if role, _ := entry["role"].(string); role == "estimator" {
			sha, _ := entry["sha256"].(string)
			return sha
		}
	}
	return ""
}

func quote(value string) string { return "'" + value + "'" }

func unquote(value string) string { return strings.Trim(value, "'") }

// engineFields prints the three fields the engine is compared on.
func engineFields(got identity) string {
	parts := make([]string, 0, 3)
	for i := range got[:3] {
		parts = append(parts, quote(recordKeys[i])+": "+got[i])
	}
	return "{" + strings.Join(parts, ", ") + "}"
}

// rows prints every member's record, one per line, for a set that disagrees
// with itself.
func rows(seen map[string]identity) string {
	names := make([]string, 0, len(seen))
	for name := range seen {
		names = append(names, name)
	}
	sort.Strings(names)
	lines := make([]string, 0, len(names))
	for _, name := range names {
		fields := make([]string, 0, len(recordKeys))
		for i, key := range recordKeys {
			fields = append(fields, quote(key)+": "+seen[name][i])
		}
		lines = append(lines, "  "+name+": {"+strings.Join(fields, ", ")+"}")
	}
	return strings.Join(lines, "\n")
}
