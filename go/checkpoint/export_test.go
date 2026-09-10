package checkpoint

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/loudreader/loudkit/go/config"
	"github.com/loudreader/loudkit/go/internal/digest"
)

var testGraphs = []string{
	"t3_cond.onnx",
	"t3_prefill.onnx",
	"t3_step.onnx",
	"flow_encoder.onnx",
	"flow_estimator.onnx",
	"vocoder.onnx",
}

// exportSet is a graph directory carrying record, or none when record is "".
func exportSet(t *testing.T, record string) string {
	t.Helper()
	dir := t.TempDir()
	for _, name := range testGraphs {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("graph"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if record != "" {
		if err := os.WriteFile(filepath.Join(dir, ExportRecord), []byte(record), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

// entries is a record whose members all carry one identity.
func entries(t *testing.T, fields map[string]any, odd map[string]any) string {
	t.Helper()
	graphs := map[string]any{}
	for _, name := range testGraphs {
		graphs[name] = fields
	}
	if odd != nil {
		graphs["vocoder.onnx"] = odd
	}
	body, err := json.Marshal(map[string]any{
		"format": "loudkit-onnx-export",
		"graphs": graphs,
	})
	if err != nil {
		t.Fatal(err)
	}
	return string(body)
}

// openTestCheckpoint is a manifest-only checkpoint and the algorithm it
// declares, with the fields a matching record has to carry.
func openTestCheckpoint(t *testing.T) (*Checkpoint, config.AlgorithmConfig, map[string]any) {
	t.Helper()
	path := manifestOnly(t, `{"format":"loudkit-checkpoint","format_version":1}`)
	ckpt, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	algorithm, err := ckpt.Algorithm()
	if err != nil {
		t.Fatal(err)
	}
	sha, err := digest.SHA256File(path)
	if err != nil {
		t.Fatal(err)
	}
	return ckpt, algorithm, map[string]any{
		"checkpoint_sha256":     sha,
		"algorithm_fingerprint": config.Fingerprint(algorithm),
		"euler_steps":           algorithm.EulerSteps,
	}
}

// The rule the reference holds, held here: a record that names this engine
// loads, one that names another is refused, and no record at all is a note.
func TestVerifyExportHoldsTheRecordToTheCheckpoint(t *testing.T) {
	ckpt, algorithm, match := openTestCheckpoint(t)

	t.Run("a matching record loads", func(t *testing.T) {
		set := exportSet(t, entries(t, match, nil))
		if err := ckpt.VerifyExport(set, algorithm, testGraphs); err != nil {
			t.Fatalf("expected this set to load, got %v", err)
		}
	})

	t.Run("no record at all still loads", func(t *testing.T) {
		if err := ckpt.VerifyExport(exportSet(t, ""), algorithm, testGraphs); err != nil {
			t.Fatalf("a set exported before the record must still load, got %v", err)
		}
	})

	for _, tc := range []struct {
		name    string
		record  string
		wantErr string
	}{
		{
			"another engine's fingerprint",
			entries(t, map[string]any{
				"checkpoint_sha256":     match["checkpoint_sha256"],
				"algorithm_fingerprint": "5cfefec451bcedd1",
				"euler_steps":           match["euler_steps"],
			}, nil),
			"was exported from a different engine than the one loading it",
		},
		{
			"another checkpoint's digest",
			entries(t, map[string]any{
				"checkpoint_sha256":     strings.Repeat("0", 64),
				"algorithm_fingerprint": match["algorithm_fingerprint"],
				"euler_steps":           match["euler_steps"],
			}, nil),
			"was exported from a different engine than the one loading it",
		},
		{
			"another step count",
			entries(t, map[string]any{
				"checkpoint_sha256":     match["checkpoint_sha256"],
				"algorithm_fingerprint": match["algorithm_fingerprint"],
				"euler_steps":           7,
			}, nil),
			"was exported from a different engine than the one loading it",
		},
		{
			"one stage re-exported on its own",
			entries(t, match, map[string]any{
				"checkpoint_sha256":     match["checkpoint_sha256"],
				"algorithm_fingerprint": "5cfefec451bcedd1",
				"euler_steps":           match["euler_steps"],
			}),
			"is a mixed graph set",
		},
		{
			"a member the record does not name",
			`{"format":"loudkit-onnx-export","graphs":{}}`,
			"does not record t3_cond.onnx",
		},
		{
			"a graphs block that is not a mapping",
			`{"format":"loudkit-onnx-export","graphs":[1,2]}`,
			"is not a mapping of name to its export record",
		},
		{
			"an entry field of a shape the exporter never writes",
			entries(t, map[string]any{"euler_steps": []int{2}}, nil),
			"is not a mapping of name to its export record",
		},
		{
			"no graphs block",
			`{"format":"loudkit-onnx-export"}`,
			"unreadable export record",
		},
		{
			"not JSON",
			"{",
			"unreadable export record",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := ckpt.VerifyExport(exportSet(t, tc.record), algorithm, testGraphs)
			if err == nil {
				t.Fatalf("expected a refusal naming %q, got none", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("refusal does not name %q: %v", tc.wantErr, err)
			}
		})
	}
}

// A field the record was written before, and one it carries as null, are the
// same answer: neither turns a set that agrees into a mixed one.
func TestVerifyExportReadsARecordWrittenBeforeAFieldExisted(t *testing.T) {
	ckpt, algorithm, match := openTestCheckpoint(t)
	withNull := map[string]any{"estimator_sha256": nil}
	for key, value := range match {
		withNull[key] = value
	}
	graphs := map[string]any{}
	for i, name := range testGraphs {
		if i%2 == 0 {
			graphs[name] = match
		} else {
			graphs[name] = withNull
		}
	}
	body, err := json.Marshal(map[string]any{"format": "loudkit-onnx-export", "graphs": graphs})
	if err != nil {
		t.Fatal(err)
	}
	if err := ckpt.VerifyExport(exportSet(t, string(body)), algorithm, testGraphs); err != nil {
		t.Fatalf("an absent field and a null one must compare equal, got %v", err)
	}
}

// A renderer traced from a swapped-in estimator agrees with itself and
// describes a checkpoint this one is not.
func TestVerifyExportRefusesASwappedEstimator(t *testing.T) {
	path := manifestOnly(t, `{"format":"loudkit-checkpoint","format_version":1,`+
		`"sources":{"flow.pt":{"role":"estimator","sha256":"`+strings.Repeat("a", 64)+`"}}}`)
	ckpt, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	algorithm, err := ckpt.Algorithm()
	if err != nil {
		t.Fatal(err)
	}
	sha, err := digest.SHA256File(path)
	if err != nil {
		t.Fatal(err)
	}
	fields := map[string]any{
		"checkpoint_sha256":     sha,
		"algorithm_fingerprint": config.Fingerprint(algorithm),
		"euler_steps":           algorithm.EulerSteps,
		"estimator_sha256":      strings.Repeat("b", 64),
	}
	err = ckpt.VerifyExport(exportSet(t, entries(t, fields, nil)), algorithm, testGraphs)
	if err == nil || !strings.Contains(err.Error(), "traced with an estimator the checkpoint was not") {
		t.Fatalf("expected the estimator refusal, got %v", err)
	}
	fields["estimator_sha256"] = strings.Repeat("a", 64)
	if err := ckpt.VerifyExport(exportSet(t, entries(t, fields, nil)), algorithm, testGraphs); err != nil {
		t.Fatalf("the estimator the checkpoint was packed from must load, got %v", err)
	}
}
