// Package checkpoint loads a packed loudkit checkpoint: the manifest plus
// the embedding tables the generator graphs need on the host.
package checkpoint

import (
	"encoding/json"
	"fmt"
	"slices"

	"github.com/loudreader/loudkit/go/config"
	"github.com/loudreader/loudkit/go/safetensors"
)

// Checkpoint is an opened packed checkpoint.
type Checkpoint struct {
	path string
	file *safetensors.File
	// Manifest is the embedded loudkit manifest, parsed. Exported because it
	// is what config.FromManifest reads and what a caller inspects to see
	// which recipe a file declares.
	Manifest map[string]interface{}
}

// Open reads a checkpoint file.
func Open(path string) (*Checkpoint, error) {
	file, err := safetensors.Open(path)
	if err != nil {
		return nil, err
	}
	manifestStr, ok := file.Metadata["manifest"]
	if !ok {
		return nil, fmt.Errorf("%s: no embedded manifest, not a loudkit checkpoint", path)
	}
	var manifest map[string]interface{}
	if err := json.Unmarshal([]byte(manifestStr), &manifest); err != nil {
		return nil, fmt.Errorf("%s: bad manifest JSON: %w", path, err)
	}
	if fmtStr, _ := manifest["format"].(string); fmtStr != "loudkit-checkpoint" {
		return nil, fmt.Errorf("%s: no embedded manifest, not a loudkit checkpoint", path)
	}
	// format_version is checked, not only format. Python refuses a version it
	// does not read; a port that accepts any version will happily load a future
	// checkpoint whose fields mean something else: the loader would still
	// "work", and the audio would be wrong for reasons no error names.
	version, _ := manifest["format_version"].(float64)
	if !slices.Contains(SupportedFormatVersions, int(version)) {
		return nil, fmt.Errorf("%s: manifest format_version %d; this build reads %v",
			path, int(version), SupportedFormatVersions)
	}
	// The declared mode selects its graph family independently of the filename.
	mode := config.DecodeSingle
	if decode, ok := manifest["decode"].(map[string]interface{}); ok {
		if named, ok := decode["mode"].(string); ok {
			mode = named
		}
	}
	if mode == config.DecodeFusionMTP2 && version < 2 {
		return nil, fmt.Errorf("decode.mode fusion_mtp2 requires format_version 2")
	}
	if !slices.Contains(SupportedDecodeModes, mode) {
		return nil, fmt.Errorf("%s: manifest declares decode.mode %q; this build decodes %v",
			path, mode, SupportedDecodeModes)
	}
	return &Checkpoint{path: path, file: file, Manifest: manifest}, nil
}

// SupportedFormatVersions are the manifest versions this build understands.
var SupportedFormatVersions = []int{1, 2}

// SupportedDecodeModes are the decode loops this build implements.
var SupportedDecodeModes = []string{config.DecodeSingle, config.DecodeFusionMTP2}

// Algorithm reads the shipping algorithm from the manifest.
//
// Returns an error for a manifest this binding cannot honour: an unknown
// guidance mode, or cfg_dual_path, which it does not implement.
func (c *Checkpoint) Algorithm() (config.AlgorithmConfig, error) {
	return config.FromManifest(c.Manifest)
}

// GeneratorTables returns the fp32 embedding tables the generator uses.
func (c *Checkpoint) GeneratorTables() (textEmb, speechEmb, textPos, speechPos []float32, err error) {
	if textEmb, err = c.file.F32("t3.text_emb.weight"); err != nil {
		return
	}
	if speechEmb, err = c.file.F32("t3.speech_emb.weight"); err != nil {
		return
	}
	if textPos, err = c.file.F32("t3.text_pos_emb.emb.weight"); err != nil {
		return
	}
	speechPos, err = c.file.F32("t3.speech_pos_emb.emb.weight")
	return
}

// SpeakerAffine returns the 192->80 speaker projection the flow conditions on.
func (c *Checkpoint) SpeakerAffine() (weight, bias []float32, err error) {
	if weight, err = c.file.F32("s3gen.flow.spk_embed_affine_layer.weight"); err != nil {
		return
	}
	bias, err = c.file.F32("s3gen.flow.spk_embed_affine_layer.bias")
	return
}

// FusionWeights returns the prefix-slot MLP in float32.
func (c *Checkpoint) FusionWeights() (map[string][]float32, error) {
	out := map[string][]float32{}
	for _, name := range []string{"0.weight", "0.bias", "2.weight", "2.bias"} {
		values, err := c.file.F32("t3.fuse." + name)
		if err != nil {
			return nil, err
		}
		out[name] = values
	}
	if len(out["0.weight"]) != 2*1024*1024 || len(out["2.weight"]) != 1024*1024 || len(out["0.bias"]) != 1024 || len(out["2.bias"]) != 1024 {
		return nil, fmt.Errorf("invalid fusion MLP dimensions")
	}
	return out, nil
}
