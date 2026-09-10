# Changelog

Notable changes, in the shape [Keep a Changelog](https://keepachangelog.com)
suggests. Versioning and what counts as a breaking change are spelled out in
[docs/reference/COMPATIBILITY.md](docs/reference/COMPATIBILITY.md).


## [0.1.1] - 2026-09-07

**The fingerprint moves: `79f71f5821477353` -> `7cd75498ad4e7531`.** Changes
to sampling, postprocessing, chunking and text preparation are audible, so the same text, voice and seed render
differently than they did on 0.1.0. That is what a moved fingerprint means and
it is the only way this project is allowed to change a reading:
[the identity contract](docs/reference/IDENTITY-CONTRACT.md) lists what
`loudkit-1` now includes, and every amendment below carries the measurement
that bought it.

Both model bundles now include all 28 voices: ten English voices and two for
each of the other nine languages. Henry, Oliver, Miles, Oscar, Clara, Emma,
Lucy and Sophie load by name without a separate profile download.

Python voice-profile saving now packs strided tensors into contiguous storage
before passing them to safetensors. Cropped enrollment mels could otherwise be
silently saved with incorrect values, changing the voice after reload in both
models. Already affected profiles must be regenerated from their source audio;
loading and resaving a corrupted file cannot restore the missing values.

Why a patch release moves it: before 1.0 the patch component tracks the API
surface and the fingerprint carries the audible contract, a clause
[COMPATIBILITY.md](docs/reference/COMPATIBILITY.md) states. This entry meets it.

All five SDKs read both checkpoint formats. Turbo uses `format_version 2`;
version-1 checkpoints retain their existing interpretation.

Source-breaking in Swift: `TokenGenerator.generate` now `throws`
(`LoudKitError.cancelled` at the poll that fires), so code that reaches the
generator through `Engine.tokenGenerator` must `try`. No compatibility shim.

Source-breaking for manually constructed voice profiles: Rust `voice::Profile`
now stores `enrolment`; add `enrolment: "first-10s".into()` to a struct literal
(or preserve the actual method if known). Go callers using positional `voice.Profile`
literals must add the new `Enrolment` field; keyed literals can omit it. Loading
and enrolling profiles use the same API and now preserve the enrollment method
when saved across all five SDKs.

### Added

- `mp3` and `opus` from every door: `/v1/synthesize`, `/v1/audio/speech`,
  gRPC and MCP. Both come from the codecs of the libsndfile that `soundfile`
  bundles, so neither adds a package. Hermes Agent and OpenClaw ask an
  OpenAI-compatible server for mp3, or for Opus when the reply is a voice note,
  and now get it from `loudkit serve`. `aac` stays refused: libsndfile cannot
  write it.
- The long-form conformance case is now driven through `engine.stream` and
  `engine.synthesize` as well as chunk by chunk, so Python consumes the flat
  `long_form.cases[].tokens` list the four ports consume; the parity page's
  seven weighted rows are compared by a `slow` test that regenerates with
  `--checkpoint` (the `parity` job runs it); enrollment from a WAV file is held
  to the enrollment fixture; `lk.enroll` and `loudkit speak` on a turbo release
  directory have goldens; `Result.save()` is pinned to the shared quantise
  rule; the production fingerprint is pinned weight-free; and the Swift
  offline-receipt test asserts the stderr line the other four ports assert.
- `loudkit text` previews what will be spoken without loading a model: the
  prepared text on stdout, a word diff of what the funnel changed on stderr.
  `loudkit --help` lists eight commands.
- `loudkit speak --play` plays the WAV it just saved through the system
  player (`afplay` on macOS, `aplay` or `paplay` on Linux).
- Speed measured on 0.1.1 for both models: eight local paths on an Apple M3
  Pro (PyTorch split CPU/MPS, native CoreML, ONNX Runtime CPU, the CPU
  reference, and the Swift, Rust, Go and TypeScript ports) and six NVIDIA
  parts (A100, L4, RTX 3090, T4, GTX 1080 Ti, Jetson Orin Nano), eager and
  with CUDA graphs, plus batched throughput on four of them. loudr-1-turbo
  runs 1.4x to 2.1x faster than loudr-1 end to end and 2.6x to 2.8x faster in
  aggregate throughput ([Benchmarks](docs/benchmarks.md)).


- **A second model, `loudreader/loudr-1-turbo`, behind the same one line.**
  `lk.load("loudreader/loudr-1-turbo")` is the whole difference: same voices,
  same API, same seeds. It is the `fusion_mtp2` generator below beside a
  renderer distilled to one Euler step, published as a second repository with
  the same layout. Python, Swift, Go, Rust and TypeScript support both models,
  using the matching ONNX graphs or CoreML packages. One enrolled voice
  profile works with both models.
  [Choosing a model](docs/guides/11-choosing-a-model.md) says which is which.
- A second decode mode, `fusion_mtp2`, two speech tokens per transformer
  forward: one head reads the hidden state, a second reads that state beside
  the first token's embedding, and a fusion MLP folds both into the one KV
  slot the pair occupies. Checkpoints that carry it declare `decode.mode` in
  their manifest and ship as `format_version 2`. The mode is a property of
  the weights, not a runtime option: it is fingerprinted, and a mode/weights
  disagreement is a load error. Measured with CUDA graphs against 0.1.0 on
  the same lineage, RTF on the short / medium / long benchmark samples:

  | | 0.1.0 | 0.1.1 |
  |---|---|---|
  | RTX 3090 | 1.22 / 6.03 / 7.73 | 1.43 / 7.47 / 8.45 |
  | Jetson Orin Nano | 0.53 / 1.60 / 1.84 | 0.63 / 1.82 / 2.16 |

  Beside it, none fingerprinted: sampling runs on the device inside the
  captured graph, graphs are kept across calls, the excitation noise is drawn
  on the GPU, and the vocoder no longer vocodes padding it throws away
  (`ExecutionConfig.vocoder_ragged`, on by default on CUDA renderers, in the
  identity contract's `equivalent` class; `--no-vocoder-ragged` for byte
  agreement with a 0.1.0 build). End-to-end figures: `docs/benchmarks.md`.
- **New release bundles use a 20 ms raised-cosine ramp on both window edges**
  to soften onset taps. The tap is a hot first mel frame from the flow decoder
  on three of the 28 voices, not a vocoder edge. Measured on oscar, the hottest:
  the residual burst in the first 12 ms of a window falls from 2.2 dB below the
  window RMS with the old 5 ms ramp to 14.7 dB below with 20 ms, at a cost of
  0.7 dB over the first 50 ms of a window that opens on speech; 50 ms would
  cost 3 dB there. `edge_fade_seconds` is part of the algorithm and its
  fingerprint, including the new 0.02 s default. Only the historical 0.005 s
  spelling stays omitted for identity compatibility.
- **`loudkit clone` ends the reference prompt in silence by default**: the clip
  is cut at its last pause before the ten-second limit, or at 9.6 s when no
  pause fits, and padded with 0.4 s of silence, because the model speaks from
  where the prompt ends. On one cloned voice the onset moved from 0 to 50 ms
  and the first 60 ms from -32 to -51 dBFS. `--no-end-in-silence` keeps the
  recording as given, which is what the library's `enroll` does unless asked.
  Profiles made with the cut carry the enrolment label `first-10s-pause`.
- **Every download directory carries a receipt**, `.loudkit-release.json`:
  the repo, the revision asked for, the commit it resolved to, the digest of
  `SHA256SUMS` and when it was fetched. A load whose revision still resolves
  to that commit hashes nothing; any other receipt, or none, fetches and
  verifies again, keeping every file that already hashes to the new manifest;
  a load that cannot reach the hub uses the receipt and says so on stderr.
  The same rules in Python (`loudkit download --local-dir`), Go, Rust, JS
  and Swift, pinned by one fixture. One hub call per load; a hub answer
  naming no commit is an error in all five.
- **The JS binding no longer needs Python.** `download(repo, dir)` fetches
  the ONNX release over `fetch` with no new dependency, on the same plan and
  rules as `loudkit download --for onnx`; `examples/hello.mjs` is the
  quickstart. `Enroller.enroll` takes a WAV path or bytes as well as samples.
- Go: `safetensors.Write`, `voice.Profile.Save`, `engine.Result.SaveWav`.
  Rust: `safetensors::write`, `voice::Profile::save`, `hub::rejected_name`,
  `hub::parse_sums`, `hub::is_official`, `Bundle::can_enroll`,
  `execution::locate_runtime`. JS: `saveVoice`, `profileFrom`,
  `writeSafetensors`, `hub.rejectedName`, `hub.parseSums`, `hub.canEnroll`,
  `Enroller.close`. Swift: `Safetensors.write`, `VoiceProfile.save`,
  `Engine.voiceNames`, `Engine.voice(named:)`, `Engine.enroll(contentsOf:)`.
- `export.json` beside every exported graph set, ONNX and CoreML, naming
  what each file was traced from. The backends refuse a set whose members
  disagree with each other or with the checkpoint; a release requires one.

### Changed

- Turbo CoreML uses an fp32 estimator: waveform correlation with ONNX now
  exceeds 0.999 on both conformance cases. The stricter gate is shared.
- Python downloads the native generator and fusion graphs required by the
  chosen backend. Legacy renderer-only CoreML bundles remain readable.
  Complete CoreML bundles run without PyTorch; explicit fp16 token generation
  retains the faster PyTorch/CoreML path. See the measured tradeoff in
  [benchmarks](docs/design/benchmarking.md).

- Go, Rust, JS, Swift: `Enroll` / `enroll_wav` / `enroll` on an engine loaded
  by repo id fetches the three enrollment graphs (Swift: packages) into the
  engine's own cache directory the first time, through the same verified
  download, and reads them from there afterwards. An engine loaded from a
  directory of your own still says how to fetch them (`Cloning: true` /
  `cloning: true` / `{ cloning: true }`).
- Go, Rust, JS, Swift: one cache layout, `<user cache>/loudkit/<org>--<name>`
  (`~/Library/Caches` on macOS, `$XDG_CACHE_HOME` or `~/.cache` on Linux,
  `%LocalAppData%` on Windows); `$LOUDKIT_CACHE` replaces `<user cache>/loudkit`
  in all four. A release one port fetched is read by the other three. Pinned by
  `tests/data/conformance/cache_path.json`.
- Rust: `hub::cache_dir()` is now `hub::cache_dir(repo)` and returns the
  release directory; the layout moved from `<root>/loudkit/<org>/<name>` to
  `<root>/loudkit/<org>--<name>`, so a release fetched by 0.1.0 is fetched
  again once. `hub::cache_path(root, repo)` is the pure rule. `hub::Bundle`
  gained `repo` and `fetch_cloning()` / `fetch_cloning_with(&Options)`.
- JS: the cache moved from `$HF_HOME/loudkit/<org>--<name>` to the platform
  user cache; `HF_HOME` is no longer read, so a release fetched by 0.1.0 is
  fetched again once.
- Swift: `Engine.enroll(contentsOf:name:language:)` is now `async throws`;
  add `await`. `ModelBundle.fetchCloning()` is public.
  `LoudKit.cacheDirectory(repo:revision:)` is now `cacheDirectory(repo:)` with
  no per-revision subdirectory (the receipt carries the revision), and
  `LOUDKIT_HOME` is replaced by `LOUDKIT_CACHE`. `LoudKit.cachePath(root:repo:)`
  is the pure rule.
- Go: `$LOUDKIT_CACHE` is honoured.
- Docs: the four port READMEs and guides 07 to 10 clone in the same two lines
  as Python, name the cache directory, and no longer show a second download
  into a directory of your own as the way to clone. Guide 01 names the ports'
  cache; guide 11 explains choosing between the two models.

- Go, Rust, JS, Swift: offline, `enroll` on an engine whose cache holds the
  synthesis set says which enrollment graphs (Swift: packages) are missing
  and that one connection fetches them, instead of the transport error (Go,
  Rust), "holds no verified release" (JS) or "cannot reach" (Swift). A plain
  load offline still uses the cache.
- Python, Go, Rust, JS, Swift: the receipt reader reads nothing past 1 MiB;
  a bigger file is no receipt. The shared fixture pins a JSON array, a
  number and a string as receipts (28 cases).
- Go and Rust READMEs carry the one sentence about `Enroll` / `enroll_wav`
  on a repo engine that the JS and Swift READMEs had.


- **One front door in five languages.** `synthesize(text, voice, options)`
  speaks text of any length in every port and returns one result that saves
  itself; the one-window call is `synthesizeWindow` (`SynthesizeWindow`,
  `synthesize_window`). Options are named and default everywhere: seed 0,
  the voice's own language, speed 1.0, no previous tokens. Go: `Options`
  and `Result` in `go/engine`, re-exported by the root package. Rust:
  `engine::Options` with `Default`, `engine::Synthesis` with `save_wav`;
  `stream` takes `&Options`. JS: `SynthesisOptions` gains `seed`, `language`
  and `shouldCancel`.
- **`load(ref)` hands out voices in every port.** `engine.voices()` and
  `engine.voice(name)` (Swift: `voiceNames`, `voice(named:)`) read the
  release the engine was opened from. Rust `Engine::load(ref)` replaces
  `load_bundle` (three-path form: `load_paths`). JS `Engine.load(ref)` is
  one argument (three-path form: `Engine.loadPaths`). Swift `Engine.load(ref)`
  (async) replaces `load(repoOrDirectory:)`; `ModelBundle` stays behind
  `Engine.load(bundle:)`. Python gains `Engine.voice(name)`,
  `Engine.voices()` and `Engine.checkpoint_path`.
- **Cloning is two lines in every port.** `engine.enroll(...)` takes a
  recording, a name and a language (Go `Enroll(name, wav, language)`, Rust
  `enroll_wav(path, name, language)`, JS `enroll(wav, { name, language })`,
  Swift `enroll(contentsOf:name:language:)`), and the profile saves itself
  in one layout, so a voice cloned in one language loads in the other four.
- **`Engine.synthesize` reads text of any length** in Python: one window
  when the text fits, sentence-boundary splits when it does not, one
  `Result` either way, the same tokens `stream` yields for the same seed.
  `synthesize(..., single_window=True)` renders one window and raises
  `WindowOverflowError` rather than splitting (`long_form=false` on the wire
  maps to it). One seed law in all five: chunk 0 draws the caller's seed
  itself and only later chunks draw `derive(seed, 16 + index)`, so a text
  that fits one window reads the same through every call.
- **One cancellation contract in five implementations.** `synthesize`
  returns the whole passage or signals cancellation explicitly, never the
  chunks that finished: Python raises `CancelledError` (a `LoudkitError`,
  code `cancelled`; the `Result | None` overloads are gone), Go returns
  `ErrCancelled`, Rust `Err(error::CANCELLED)` (`error::is_cancelled`), JS
  throws `CancelledError`, Swift throws `LoudKitError.cancelled`
  (`Result.cancelled` is gone). `stream` delivers the chunks that finished
  and then ends without raising. Rust `Options` gains `should_cancel`,
  Python's `Engine.synthesize` takes it, and a departed HTTP or gRPC client
  no longer holds the engine's slot to the last token (HTTP answers 499).
- Swift `TokenGenerator.generate` throws `LoudKitError.cancelled` at the poll
  that fires and is declared `throws`; every caller, `Engine.tokenGenerator`
  included, must `try`. A generator that could not cancel was the wrong API
  to keep, so there is no shim.
- **`loudkit.__all__` is 21 names, down from 46**: the six verbs, `Engine`,
  `Result`, `VoiceProfile`, `MIN_SPEED`, `MAX_SPEED`, the error classes and
  `__version__`. Everything else stays importable from the module that owns
  it, which [COMPATIBILITY.md](docs/reference/COMPATIBILITY.md) names.
- **`ExecutionOverrides` is gone.** Every `ExecutionConfig` field defaults to
  `None`, "the checkpoint's default for this device"; a named field wins even
  when it equals the default; `ExecutionConfig.resolved()` fills the rest.
- **Postprocess is a per-checkpoint preset.** `PostprocessConfig` and the
  eight detector functions left the public surface; `inspect`, `Inspection`,
  `Reason` and `PostprocessMode` remain, and `mode` is the one knob.
- **The fingerprint is internal.** `Result` carries one
  `Result.provenance: Provenance` value that `save()` writes into the C2PA
  manifest, in place of eight fields. `FINGERPRINT_SCHEMA` and
  `DEFAULT_ALGORITHM` left `config.__all__`.
- **Eight CLI commands.** `speak`, `text`, `clone`, `voices`, `download`,
  `serve`, `verify`, `doctor`. `serve --grpc` and `serve --mcp` replace the `grpc` and
  `mcp` commands; `doctor --describe` replaces `describe`; `bench` and
  `profile` moved to `tools/`. `--checkpoint` defaults to
  `$LOUDKIT_CHECKPOINT`, else `loudreader/loudr-1`.
- **One name for the token-cap flag in every port.** `Result.hit_token_cap`
  in Python, `HitTokenCap` in Go (was `Truncated`), `hit_token_cap` in Rust
  (was `truncated`), `hitTokenCap` in JS (was `hitCap`) and in Swift. Go's
  `Engine.Enroll` and `EnrollPCM` take the recording first, as the other four
  SDKs do.
- **Python modules.** `loudkit.engine` is four modules (`engine`, `result`,
  `window`, `stream`); `loudkit.config` is three (`config`, `execution`,
  `manifest`); `loudkit.hub` is three (`hub`, `release`, `checksums`);
  `loudkit.transports.http` is three (`http`, `limits`, `schemas`); every
  public name stays importable where it was. `loudkit.frontend.polish` is
  `loudkit.frontend.speechtext`.
- **`lk.enroll(path)` reads the file at its native rate** (soundfile,
  channels averaged) and resamples with the Hann-sinc law every port ships,
  so a 44.1 or 48 kHz recording enrols the same profile from Python as from
  any port. `enroll(samples, ...)` takes `sample_rate` (default 24000).
- `loudkit.hub.read_receipt(root, repo)` returns a validated `Receipt` or
  `None` and is the only reader of the file; `receipt_hit` wraps it, and the
  ports' `readReceipt` / `read_receipt` follow the same shape. Python asks
  the hub once per `(repo, revision)` per process.
- HTTP, gRPC and MCP refuse the same over-long request with the same code;
  the MCP `synthesize` tool's cap refusals carry `code: "invalid_request"`.
  Every loopback address is "local" for all three transports.
- A second concurrent `generate` on one engine is refused rather than
  served. A checkpoint whose `decode.mode` needs a `format_version` above
  the one it declares is refused at load, in all five engines.
- **Rust finds the ONNX Runtime library the way Go does**: `ORT_DYLIB_PATH`,
  then `LOUDKIT_ONNXRUNTIME_LIB`, then the usual install paths.
- **JS `Engine.load(dir)` no longer demands `manifest.json`**; it reads the
  manifest the checkpoint embeds, as the other four do.
- **Swift `Result.save(to:)` is `saveFloat32Wav(to:)`**; `saveWav(to:)` and
  `saveWav(_ path:)` write 16-bit PCM.
- `tools/build_release.py` and `tools/build_turbo_release.py` are one
  package, `tools/release/`, run as `python -m tools.release --model
  loudr-1|turbo`. The two scripts stay as shims and refuse a user-supplied
  `--model`. `--skip-verify` under `--model turbo` is refused before any
  copying.
- **Documentation.** `docs/README.md` lists the twelve pages a user needs;
  everything else is under `docs/reference/`, `docs/platforms/` and
  `docs/design/`. The README is the five-language front door. Internal
  vocabulary left every user page, and the C2PA promise is narrowed to
  Python and the server. `docs/parity-measured.md` is regenerated with the
  release weights, names what the free-running row compares against (its own
  stream, re-based at `3960d30`) and states that row's per-architecture
  scope.

### Security

- **One verification law, at download, in all five.** `download` (and a
  repo-id `load` that has to fetch) fetches `SHA256SUMS` and `release.json`
  before anything else; under `loudreader/` the record must say a release
  profile (`full-0.1` or `turbo-0.1`) and `verified: true`, or the bundle is
  refused before a weight moves. Every fetched file is hashed against the
  manifest as it arrives; a mismatch is refused and the file removed; an
  unlisted weight is refused. A listing name that escapes the directory is
  refused in all four ports.
- **The `.loudkit-verified` marker and re-verification on every load are
  gone** from Python, Go, JS and Swift; the receipt replaces them.

### Removed

- Go `Say`, `SynthesizeLong` and the seven-argument `Synthesize`; Rust
  `Engine::load_bundle`, `load_bundle_with`, `synthesize_long`,
  `voice::from_bundle` and the six-tuple; JS `synthesizeLong`,
  `Engine.loadWithDefaults`, the free `voice(name, dir)` and the three-path
  `Engine.load` overload; Swift `synthesizeLong` and
  `Engine.load(repoOrDirectory:)`.
- Python `Engine.synthesize_long`, `ExecutionOverrides`,
  `AlgorithmConfig.tokens_per_forward`, `DecodeGeometry.cache_shape`,
  `loudkit.bench`, `loudkit.profile`, `loudkit.hub.receipt_valid`,
  `loudkit.transports.mcp.main`, and five CLI commands (see Changed).

### Fixed: the silence campaign

The engine could park in silence and not come out. Measured over 120 passages
per arm in ten languages, paragraphs carrying a gap over one second ran at
**33.0%**, with **74 of 1705** passages rendering a chunk of no speech at all.
Six laws close it, each moving the fingerprint and re-basing the goldens.

- **A new verdict, `stall`, for a row the decoder spent trapped in silence.**
  The four rescues that existed all cut a tail, and this failure is a hole.
  `stall` cuts nothing and condemns the row into the retry ladder, the way
  `dropout` does, and runs before every tail rescue. Seventh `Reason`, and
  every port runs it.
- **The repetition penalty applies to silence ids.** Exempt from both the
  penalty and the `min_p` cutoff, a silence run was absorbing: zero escapes
  in 1,031 instrumented trap steps. The penalty exemption is gone and the
  `min_p` one stays (removing it instead measured catastrophic). Holes
  **33.0% -> 4.3%**, mute chunks **74 -> 1**, WER flat or better.
- **A pause is not a loop** (`repetition_silence: acoustic`): a pause parked
  on an id only the render census knows stalls and retries instead of being
  cut. **A loop the decoder resumed from is condemned, not cut**
  (`repetition_resume: condemn`): a tail cut would have removed the speech
  that followed it.
- **A starved desperation rescue is condemned**: a cap-hit trim that keeps
  under `desperation_min_keep_per_text_token` tokens per text token goes to
  the retry ladder, with the trim as the fallback.
- **The ladder ships the best attempt, not the last**: the one with the
  fewest tokens in the true-silence set.

### Fixed: the text funnel

- **Every character class the five implementations disagreed about, closed
  as one family** (`funnel-2` -> `funnel-4`). Each pass asks some version of
  "is this a letter, a digit, or a space", and the five spelled those three
  questions eleven different ways. Every difference was audible.
  - *Numerals.* A number character with no ASCII spelling (a superscript, a
    vulgar fraction, a circled or Roman numeral, a digit from any script but
    Latin) was deleted: `Add ½ cup` read as *add cup*. `fold_numerals` now
    reads a generated table, `models/data/numerals.json`, cut from one pinned
    UCD (16.0.0): a digit of any script becomes the ASCII digit of the same
    value, every other numeral becomes the text the table names (`½` is
    `1/2`). Detection comes from the table too, so a runtime's own Unicode
    version no longer decides what is read aloud.
  - *Letters, whitespace, boundaries.* `\p{L}` everywhere (Swift's
    `CharacterSet.letters` deleted 6,145 Tangut ideographs), one written-out
    whitespace class (Go read footnote markers aloud after a thin space), and
    `(?![\p{L}\p{Nd}_])` written out rather than `\b`, whose ECMAScript and
    Go spellings are ASCII.

- **A symbol is spoken in the language being read.** The grammar carried a
  word per language for the seven marks that are also currency or measure,
  and the funnel carried an English/Polish pair for the other twenty-one, so
  ten of the twelve languages heard English: `≈` said *about* in a German
  render and in a Finnish one. Thirteen word-valued symbols
  (`× ÷ ≈ ≥ ≤ ≠ ± ✓ ✔ ✗ ✘ & @`) gain a row per language; the eight that are
  punctuation everywhere (`→ ← ⇒ • · ▪ ◦ …`) stay a rule in code. Measured
  over every spoken symbol in three carriers in twelve languages, 360 of
  1008 rows move; every conformance case read in its declared language is
  unmoved.
- **An uncovered symbol is left written in all five funnels.** Four ports
  fell back to the English row per symbol where the reference left the
  symbol written. Measured on a grammar with eight rows cut, one per pass
  the funnel runs, 46 of 576 cases diverged. Swift held the wording table
  twice, as a literal beside the grammar file its own resources carry, and
  reads the file now.

### Fixed

- **Voices load from a release the Hub cache holds.** The cache is one
  symlink per file from the snapshot into its blob store, and both the voice
  resolver and the server's voice library refused those links as escapes, so
  `engine.voice("joe")`, `loudkit speak --voice joe` and `/v1/voices` found
  nothing after a download. A link is now confined to the cache entry that
  owns the snapshot; a link out of it, or out of a plain directory, is still
  refused.

- CoreML generator exports use a weighted sum for single-query attention,
  avoiding a CPU matrix-kernel error that changed Turbo tokens in longer windows.
  Both model bundles include regenerated, token-verified generator graphs.

- An offline receipt could be forged by one field. All five hold
  `.loudkit-release.json` to one reader before using it, offline or as an
  online hit: every field present with its type (`null` is the wrong type),
  `repo` the one asked, `commit` forty lowercase hex, `sha256sums` the digest
  of the `SHA256SUMS` on disk, every listed file the plan selects present.
  The fixture holds twenty-five cases, `null` in every field among them.
- Python never refreshed a moving `main`, and the native caches were not
  bound to a revision; a repo id always resolves through `download` now.
- **The one-window path refuses text that does not fit, in all five
  implementations, instead of shipping the tail unspoken**; none of the five
  raised the `WindowOverflowError` their docs promised.
- **A streaming response is capped at ten minutes of wall clock**, on HTTP as
  on gRPC: a client that stopped reading held the engine slot indefinitely.
- The Swift snippet on the front page and the landing page did not compile
  (`saveWav(to:)` takes a `URL`; the string form is `saveWav("hello.wav")`),
  and the landing page's Go snippet never read its `err`. Each port's suite
  now compiles its page blocks. `SUPPORTED.md` no longer advertises the old
  CLI.

### Fixed: the chunker

- **Not every period ends a sentence** (`ChunkConfig.mid_sentence_period:
  hold`). `"But Mr. Smith went home"` produced the chunk `"But Mr."`, with
  its own seed and a ceiling too short for a closing pause. A period is held
  when the next word starts in lower case or the token before it is in
  `ChunkConfig.abbreviations`; `"break"` names the old law.
- **A chunk the window could not hold is split, not shipped**
  (`ChunkConfig.cap_resplit: word`). 54 of 9920 chunks overran the character
  budget, 30 still speaking when the cap closed, and the remainder was never
  spoken. Such a chunk is replaced by its two halves, cut at the word
  boundary nearest the middle. Windows at the cap **52 -> 2**, still speaking
  at the cap **29 -> 1**. `"off"` names the old law.
- **The word-boundary fallback matches the boundaries prose has**, not
  U+0020 alone: text whose every space is non-breaking was cut mid-word.
- **Six ways the five funnels disagreed**, each now pinned in the shared
  fixture: the reference dropped the inverted Spanish marks `¿¡` that four
  ports kept, and Swift halved on grapheme clusters, not scalars.


## [0.1.0] - 2026-08-25

First public release.

### Added

#### Speech

- A two-stage engine optimized for local inference: an autoregressive token
  generator at 25 Hz and a parallel renderer. It runs faster than real time on
  a laptop, a phone and a Jetson. Benchmarks with the command that reproduces
  each number are in
  [docs/benchmarks.md](docs/benchmarks.md).
- **20 voices across 10 languages**, every one enrolled from a recording made
  or released for speech-technology use — consented donations and CC0 / CC-BY
  corpora — with donor or source, licence and consent basis named. Quality is
  evaluated for **English**; the rest are read, not yet measured.
- **Voice cloning from about ten seconds** of audio, producing a ~150 KB profile
  that is a file, not a model.
- Long-form synthesis that splits at sentence boundaries and carries prosody
  across the joins, and streaming that yields the first sentence rather than
  waiting for the paragraph.
- Token-level cancellation, so a voice agent's barge-in goes quiet within one
  decode step instead of at the end of a chunk. It is cooperative: a kernel
  already running is not interrupted.

#### Latency

- Streaming renders window *k* while generating window *k+1* — the chunk chain
  is sequential only through the tokens, so the overlap changes no byte
  (asserted against the serial path). Measured on an M3 Pro: a six-window
  passage 1.29x faster, the Apple bench row 2.81x → 3.43x.
- `ChunkConfig.first_chunk_max_tokens` caps only the first chunk, so a stream
  opens on the first clause instead of a full window: first audio ~1.9 s →
  ~1.4 s at a 96-token budget (M3 Pro). An algorithm value — setting it
  re-fingerprints; unset it is absent from the fingerprint. Python first; the
  ports follow.
- The generator's conditioning row is memoised per voice (a content key), so
  chunks and repeated requests in one voice stop recomputing it.
- `Engine.warm()` pays the first-use costs (kernel autotune, graph capture,
  allocator pools) at startup; `serve`, gRPC and MCP call it after loading, so
  the first request pays warm latency (measured on a 3090 with graphs:
  1.09 s → 0.73 s first audio, for 1.3 s once at boot).

#### Command line

- `loudkit doctor` — what this machine can run and the one command that fixes
  each gap. `loudkit download <repo>` — the checkpoint, the graphs one backend
  needs and all twenty voices into the shared cache; `--with-cloning` adds the
  encoder and the enrollment graphs. `loudkit voices <repo>` — the menu without the gigabyte.
  `loudkit verify <path>` — a checkpoint, voice profile or rendered WAV checked
  against its own claims (payload digest, profile validation, C2PA binding).

#### Errors

- A frozen catalog of error codes, defined in `loudkit.errors` and spoken by
  every transport: `exc.code` on Python exceptions, `"code"` in HTTP error
  bodies, `"error_code"` on the SSE terminal event, `loudkit-error-code` in
  gRPC trailing metadata. Codes are never renamed or reused, only added. See
  [docs/reference/errors.md](docs/reference/errors.md).

#### Five implementations, one behaviour

- Python is the reference; **Swift, Go, Rust and TypeScript are full ports**, not
  wrappers, held to shared conformance fixtures. All five compute the same
  algorithm fingerprint independently and agree.
- The same text, voice and seed give a bit-identical waveform on a given build,
  device and backend. Waveforms are **not** guaranteed to match across backends
  or devices. The engine refuses to start if two of its components disagree
  about what to compute. See
  [docs/reference/IDENTITY-CONTRACT.md](docs/reference/IDENTITY-CONTRACT.md).

#### Text handling

- Numbers, decimals, currency, units, times and abbreviations verbalised in
  **12 languages** from first principles — no LGPL dependency — and checked
  against a 1300-row CLDR differential.
- Sentence-boundary chunking, NFC normalisation, output charset closure, and a
  Polish anglicism respeller.
- Control-tag injection closed: bracketed text in user input can no longer
  trigger model behaviours.

#### It checks its own output

- Six detectors read the tokens the generator just produced and find the failure
  no audio filter can — a hallucinated tail after a long silence, in the same
  voice. Every threshold comes from a measured trace, and was
  re-measured across the nine spoken languages the roster then carried
  (Swedish, added later, ships unmeasured on the same margin). A chunk that is
  certainly wrong but that no rule can place is *reported*, not silently
  shipped. See [docs/design/postprocess.md](docs/design/postprocess.md).
- Selective re-roll: a chunk that fails is re-rendered from a derived seed
  rather than returned.

#### For a reading app

- `Result.chunks` gives exact per-chunk spans from sample offsets, plus
  **estimated** word times by proportional allocation — documented as an
  estimate, because it is not a forced aligner
  ([docs/reference/timestamps.md](docs/reference/timestamps.md)).
- `speed=` is the video-player control: faster without a pitch shift, via WSOLA
  written from first principles ([docs/reference/speed.md](docs/reference/speed.md)).
- `previous_tokens=` carries prosody across two separate calls.

#### Interfaces

- A local HTTP server under `/v1` with one-shot and SSE streaming routes and
  four output encodings (wav, pcm16, flac, ogg) — a **working example, not a
  production deployment**. A public bind requires a bearer token; a loopback
  bind does not require one, and authenticates when given one (see the server
  guide).
- An MCP server for agents, and a Linux Speech Dispatcher module so screen
  readers can use it.
- Discovery without loading anything: `loudkit.languages()`, `loudkit.voices()`.

#### Provenance

- Saved WAVs and server replies carry an unsigned C2PA claim-only manifest
  binding the fingerprint, recipe and seed to a hash of the audio
  ([docs/reference/provenance.md](docs/reference/provenance.md)). The provenance
  guide documents the manifest fields, verification and signing boundary.

### Known limitations

- Quality evaluated for English only.
- Ordinals are English-only: it is the one language here that writes a letter
  suffix (`22nd`), and the other eleven need the inflection work the token layer
  is being built for.
- Yearless numeric dates (`12.3.`) are deliberately **not** read as dates in any
  language. The shape is indistinguishable from a decimal at the end of a
  sentence, and reading it wrong is worse than reading it plainly.
- There is no expressiveness control. The checkpoint architecture reserves a
  conditioning slot for an emotion scalar, but the value does not change the
  output in the released checkpoint. The field was
  removed from `VoiceProfile` and the profile format before release; the slot
  is fed the training constant, and a profile that carries a legacy `emotion`
  header key still loads (the key is ignored).
- Apache-2.0, derived from Chatterbox (MIT, Resemble AI). The enrollment
  architectures come from CosyVoice, S3Tokenizer and 3D-Speaker (Apache-2.0),
  FunASR and Real-Time-Voice-Cloning (MIT); the Polish respelling lexicon is
  derived from CMUdict (CMU, BSD-2-clause). Everything upstream is permissively
  licensed, nothing is copyleft, and each holder is named in [NOTICE](NOTICE).
