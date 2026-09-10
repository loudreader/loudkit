// Stage-by-stage probe of the Go decode loop, for cross-port comparison.
//
// The same five stages in the same order as tools/decode_probe/probe.py, so a
// comparison stops at the first stage that disagrees instead of reporting
// noise from everything downstream of a divergence.
//
//	go run tools/decode_probe/probe.go BUNDLE VOICE TEXT SEED LANGUAGE OUTDIR
//
// Writes OUTDIR/go.json and OUTDIR/go.<stage>.bin (raw little-endian float32).
package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"

	"github.com/loudreader/loudkit/go/config"
	"github.com/loudreader/loudkit/go/engine"
	"github.com/loudreader/loudkit/go/onnx"
	"github.com/loudreader/loudkit/go/sampler"
	"github.com/loudreader/loudkit/go/voice"
)

func f32bytes(a []float32) []byte {
	b := make([]byte, 4*len(a))
	for i, x := range a {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return b
}

func shaOf(a []float32) string {
	sum := sha256.Sum256(f32bytes(a))
	return hex.EncodeToString(sum[:])
}

func head(a []float32, n int) []float32 {
	if len(a) < n {
		n = len(a)
	}
	return a[:n]
}

func main() {
	if len(os.Args) != 7 {
		fmt.Fprintln(os.Stderr, "usage: probe BUNDLE VOICE TEXT SEED LANGUAGE OUTDIR")
		os.Exit(2)
	}
	bundle, voicePath, text := os.Args[1], os.Args[2], os.Args[3]
	seed, err := strconv.ParseUint(os.Args[4], 10, 64)
	if err != nil {
		fmt.Fprintln(os.Stderr, "bad seed:", err)
		os.Exit(2)
	}
	language, outdir := os.Args[5], os.Args[6]

	// Which binary and which library actually answered. A path that silently
	// resolves to another checkout is the failure this line exists to prevent.
	self, _ := os.Executable()
	fmt.Fprintf(os.Stderr, "go probe: %s\n", self)
	lib := os.Getenv("LOUDKIT_ONNXRUNTIME_LIB")
	fmt.Fprintf(os.Stderr, "onnxruntime: %s\n", lib)
	if lib == "" {
		fmt.Fprintln(os.Stderr, "set LOUDKIT_ONNXRUNTIME_LIB")
		os.Exit(2)
	}
	onnx.SetSharedLibraryPath(lib)
	if err := onnx.InitializeEnvironment(); err != nil {
		fmt.Fprintln(os.Stderr, "init:", err)
		os.Exit(1)
	}
	defer onnx.DestroyEnvironment()

	if err := os.MkdirAll(outdir, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	ckpt := filepath.Join(bundle, "loudr-1.safetensors")
	tok := filepath.Join(bundle, "tokenizer.json")
	eng, err := engine.LoadWith(ckpt, filepath.Join(bundle, "onnx"), tok,
		config.ExecutionConfig{ONNXProvider: "cpu"})
	if err != nil {
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(1)
	}
	defer eng.Close()
	fmt.Fprintln(os.Stderr, eng.Describe())

	v, err := voice.Load(voicePath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "voice:", err)
		os.Exit(1)
	}
	cfg := eng.Config()

	rec := map[string]any{
		"port":        "go",
		"loaded_from": self,
		"bundle":      bundle,
		"voice":       voicePath,
		"text":        text,
		"seed":        seed,
		"language":    language,
		"decode":      cfg.DecodeMode,
		"fingerprint": eng.Describe(),
		"sample_rate": cfg.SampleRate,
	}

	// 1. text tokens.
	textTokens, err := eng.Encode(text, language)
	if err != nil {
		fmt.Fprintln(os.Stderr, "encode:", err)
		os.Exit(1)
	}
	rec["text_tokens"] = textTokens

	// 3. the token sequence. (2, the prefill row, is unexported here; see
	//    probe_prefill_test.go in go/engine, which reaches it in-package.)
	// The same five fields the engine's own retry ladder sets on the sampler
	// (go/engine/engine.go:933): built here the same way, so this probe's
	// draw law is the shipped one rather than a second opinion.
	s := sampler.New(sampler.Config{
		Temperature:       cfg.Sampling.Temperature,
		RepetitionPenalty: cfg.Sampling.RepetitionPenalty,
		MinP:              cfg.Sampling.MinP,
		MaxNewTokens:      cfg.Sampling.MaxNewTokens,
		SilenceTokenIds:   cfg.Sampling.SilenceTokenIds,
	}, seed)
	raw, err := eng.Generate(textTokens, v, s, nil, nil, nil)
	if err != nil {
		fmt.Fprintln(os.Stderr, "generate:", err)
		os.Exit(1)
	}
	rec["speech_tokens_raw"] = raw
	limit := cfg.StartSpeech
	stripped := []int{}
	for _, t := range raw {
		if t < limit {
			stripped = append(stripped, t)
		}
	}
	rec["speech_tokens"] = stripped

	// 4. the mel frames.
	mel, err := eng.DecodeMel(stripped, v, seed)
	if err != nil {
		fmt.Fprintln(os.Stderr, "mel:", err)
		os.Exit(1)
	}
	rec["mel"] = map[string]any{"len": len(mel), "sha": shaOf(mel), "head": head(mel, 8)}
	os.WriteFile(filepath.Join(outdir, "go.mel.bin"), f32bytes(mel), 0o644)

	// 5. the rendered samples.
	audio, err := eng.Vocode(mel, seed)
	if err != nil {
		fmt.Fprintln(os.Stderr, "vocode:", err)
		os.Exit(1)
	}
	rec["audio"] = map[string]any{"len": len(audio), "sha": shaOf(audio), "head": head(audio, 8)}
	os.WriteFile(filepath.Join(outdir, "go.audio.bin"), f32bytes(audio), 0o644)

	// 6. the long-form path.
	out, err := eng.Synthesize(text, v, engine.Options{Seed: seed, Language: language})
	if err != nil {
		fmt.Fprintln(os.Stderr, "synthesize:", err)
		os.Exit(1)
	}
	os.WriteFile(filepath.Join(outdir, "go.longform.bin"), f32bytes(out.Audio), 0o644)
	rec["longform"] = map[string]any{
		"tokens":        out.Tokens,
		"audio_len":     len(out.Audio),
		"audio_sha":     shaOf(out.Audio),
		"audio_head":    head(out.Audio, 8),
		"n_chunks":      len(out.Chunks),
		"hit_token_cap": out.HitTokenCap,
	}

	b, _ := json.MarshalIndent(rec, "", " ")
	os.WriteFile(filepath.Join(outdir, "go.json"), b, 0o644)
	fmt.Fprintln(os.Stderr, "wrote", filepath.Join(outdir, "go.json"))
}
