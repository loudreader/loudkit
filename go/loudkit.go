// Package loudkit is the front door of the Go binding: download a release,
// load it, pick a voice, write a WAV.
//
//	eng, _ := loudkit.Load("loudreader/loudr-1")
//	v, _ := eng.Voice("joe")
//	out, _ := eng.Synthesize("Hello from loudkit.", v, loudkit.Options{Seed: 7})
//	out.SaveWav("hello.wav")
//
// The subpackages under it are the engine itself. Reach for them when you have
// your own layout; see docs/guides/08-go.md.
package loudkit

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/loudreader/loudkit/go/engine"
	"github.com/loudreader/loudkit/go/enroll"
	"github.com/loudreader/loudkit/go/voice"
)

// Options is what varies about one synthesis besides the text and the voice:
// Seed, Language, Speed, PreviousTokens, ShouldCancel. The zero value is seed
// 0, the voice's own language and normal speed.
type Options = engine.Options

// Result is one rendered passage: Audio, SampleRate, Tokens, Mel, Chunks,
// HitTokenCap, and SaveWav.
type Result = engine.Result

// Chunk is one rendered piece, handed to Stream's callback as it is made.
type Chunk = engine.Chunk

// ErrCancelled is what Synthesize returns when Options.ShouldCancel fired:
// nothing was produced. Test for it with errors.Is.
var ErrCancelled = engine.ErrCancelled

// Engine speaks. It wraps engine.Engine with the release it was loaded from,
// so voices and enrollment are asked for by name rather than by path.
// Synthesize, Stream, SynthesizeWindow and Describe are the inner engine's.
type Engine struct {
	*engine.Engine
	bundle *Bundle
	// repo is the id the release was fetched by, so Enroll can fetch the
	// rest of it into the same directory. Empty for a directory of your own.
	repo string
}

// Load opens a release: a directory holding one, or a Hugging Face repo id,
// which goes through Download into the user cache. A cached release whose
// receipt names the commit main resolves to today is read as it is; offline,
// the receipt stands in and Download says so on stderr.
//
// The onnxruntime shared library is found and started here, so nothing above
// this has to name it.
func Load(ref string) (*Engine, error) {
	return LoadContext(context.Background(), ref)
}

// LoadContext is Load under a caller's context, which bounds the download a
// repo id may set off. A directory needs no network and ignores it.
func LoadContext(ctx context.Context, ref string) (*Engine, error) {
	ref = modelRepo(ref)
	dir, repo := ref, ""
	if isRepoID(ref) {
		cached, err := cacheDir(ref)
		if err != nil {
			return nil, err
		}
		if _, err := DownloadContext(ctx, ref, cached, Fetch{}); err != nil {
			return nil, err
		}
		dir, repo = cached, ref
	}
	bundle, err := Open(dir)
	if err != nil {
		return nil, err
	}
	inner, err := LoadPaths(bundle.Checkpoint, bundle.ONNXDir, bundle.Tokenizer)
	if err != nil {
		return nil, err
	}
	inner.bundle = bundle
	inner.repo = repo
	return inner, nil
}

// LoadPaths is Load for a layout of your own: the checkpoint, the directory
// of exported graphs and tokenizer.json, named one by one. Voice and Enroll
// need a release directory and are not available on an engine opened this
// way.
func LoadPaths(checkpoint, onnxDir, tokenizer string) (*Engine, error) {
	if err := initRuntime(); err != nil {
		return nil, err
	}
	inner, err := engine.Load(checkpoint, onnxDir, tokenizer)
	if err != nil {
		return nil, err
	}
	return &Engine{Engine: inner}, nil
}

// Voices are the names Voice will accept, sorted.
func (e *Engine) Voices() []string {
	if e.bundle == nil {
		return nil
	}
	return e.bundle.Voices()
}

// Voice loads one voice from the release by name, such as "joe".
func (e *Engine) Voice(name string) (*voice.Profile, error) {
	if e.bundle == nil {
		return nil, fmt.Errorf("this engine was opened with LoadPaths, which has no " +
			"release to take voices from. Use voice.Load with a path")
	}
	path, err := e.bundle.VoicePath(name)
	if err != nil {
		return nil, err
	}
	return voice.Load(path)
}

// Enroll clones a voice from a WAV recording, up to ten seconds of it.
// language is the language of the recording, and what the voice reads in
// when Options.Language is empty; "" means "en".
//
// The three enrollment graphs are not in a plain fetch. An engine loaded by
// repo id fetches them into its own cache directory the first time; one
// loaded from a directory needs a fetch made with Fetch{Cloning: true}.
func (e *Engine) Enroll(wavPath, name, language string) (*voice.Profile, error) {
	return e.EnrollContext(context.Background(), wavPath, name, language)
}

// EnrollContext is Enroll under a caller's context, which bounds the fetch of
// the enrollment graphs when the release does not carry them yet.
func (e *Engine) EnrollContext(ctx context.Context, wavPath, name, language string) (*voice.Profile, error) {
	f, err := os.Open(wavPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	audio, rate, err := readWav(f)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", wavPath, err)
	}
	return e.EnrollPCMContext(ctx, audio, rate, name, language)
}

// EnrollPCM is Enroll for audio you already have in memory: mono float32
// samples in [-1, 1] at any sample rate.
func (e *Engine) EnrollPCM(audio []float32, sampleRate int, name, language string) (*voice.Profile, error) {
	return e.EnrollPCMContext(context.Background(), audio, sampleRate, name, language)
}

// EnrollPCMContext is EnrollPCM under a caller's context, which bounds the
// fetch of the enrollment graphs when the release does not carry them yet.
func (e *Engine) EnrollPCMContext(ctx context.Context, audio []float32, sampleRate int, name, language string) (*voice.Profile, error) {
	if e.bundle == nil {
		return nil, fmt.Errorf("this engine was opened with LoadPaths, which has no " +
			"release to take the enrollment graphs from. Use enroll.LoadEnroller")
	}
	if err := e.fetchCloning(ctx); err != nil {
		return nil, err
	}
	enroller, err := enroll.LoadEnroller(e.bundle.ONNXDir)
	if err != nil {
		return nil, err
	}
	defer enroller.Close()
	result, err := enroller.Enroll(audio, sampleRate)
	if err != nil {
		return nil, err
	}
	return result.Profile(name, sampleRate, language), nil
}

// fetchCloning makes sure the release holds the three enrollment graphs: an
// engine loaded by repo id fetches them into the directory it was loaded
// from, through the same receipt-aware Download, so the cache grows to the
// wider set once and stays a hit afterwards.
func (e *Engine) fetchCloning(ctx context.Context) error {
	if e.bundle.CanEnroll() {
		return nil
	}
	if e.repo == "" {
		return fmt.Errorf("%s carries no enrollment graphs. Fetch them with "+
			"loudkit.DownloadWith(repo, dir, loudkit.Fetch{Cloning: true})", e.bundle.Root)
	}
	_, err := DownloadContext(ctx, e.repo, e.bundle.Root, Fetch{Cloning: true})
	return err
}

// cacheDir is where Load puts a repo it had to download, so a second project
// on this machine shares the copy: cachePath under the user cache, or under
// $LOUDKIT_CACHE when that is set. The layout is the one every port uses,
// pinned by tests/data/conformance/cache_path.json.
func cacheDir(repo string) (string, error) {
	if dir := os.Getenv("LOUDKIT_CACHE"); dir != "" {
		return filepath.Join(dir, cacheSlug(repo)), nil
	}
	root, err := os.UserCacheDir()
	if err != nil {
		return "", err
	}
	return cachePath(root, repo), nil
}

// cachePath is <root>/loudkit/<org>--<name>. root is the platform's user
// cache: $XDG_CACHE_HOME or ~/.cache on Linux, ~/Library/Caches on macOS,
// %LocalAppData% on Windows.
func cachePath(root, repo string) string {
	return filepath.Join(root, "loudkit", cacheSlug(repo))
}

func cacheSlug(repo string) string {
	return strings.ReplaceAll(repo, "/", "--")
}
