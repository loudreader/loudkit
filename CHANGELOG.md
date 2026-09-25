# Changelog

Notable changes, in the format [Keep a Changelog](https://keepachangelog.com)
describes. [docs/reference/COMPATIBILITY.md](docs/reference/COMPATIBILITY.md)
defines versioning and what counts as a breaking change.


## [0.1.1] - 2026-09-10

The fingerprint moves: `79f71f5821477353` -> `7cd75498ad4e7531`. Changes to
sampling, postprocessing, chunking and text preparation are audible, so the
same text, voice and seed render differently than on 0.1.0.
[The identity contract](docs/reference/IDENTITY-CONTRACT.md) lists what
`loudkit-1` now includes, and the sections below name the changes that moved
the fingerprint. Before 1.0, any `0.x` release, minor or patch, may
change the public API and move the fingerprint
([COMPATIBILITY.md](docs/reference/COMPATIBILITY.md)).

Both model bundles now include all 28 voices: ten English voices and two for
each of the other nine languages. Henry, Oliver, Miles, Oscar, Clara, Emma,
Lucy and Sophie load by name without a separate profile download.

Python voice-profile saving now packs strided tensors into contiguous storage
before it passes them to safetensors. Before this fix, cropped enrollment mels
could be saved with incorrect values and no error, which changed the voice
after reload in both models. Regenerate affected profiles from their source
audio: loading and saving a corrupted file again cannot restore the missing
values.

All five SDKs read both checkpoint formats. Turbo uses `format_version 2`;
version-1 checkpoints keep their existing interpretation.

Breaking changes:

- Swift, source-breaking: `TokenGenerator.generate` is now declared `throws`
  and throws `LoudKitError.cancelled` at the poll that fires. Every caller,
  including code that reaches the generator through `Engine.tokenGenerator`,
  must `try`. There is no compatibility shim.
- Manually constructed voice profiles, source-breaking: Rust `voice::Profile`
  now stores `enrolment`. Add `enrolment: "first-10s".into()` to a struct
  literal, or the actual method if known. Go callers that use positional
  `voice.Profile` literals must add the new `Enrolment` field; keyed literals
  can omit it. Loading and enrolling profiles use the same API, and all five
  SDKs now keep the enrollment method when they save a profile.
- WAV provenance: `save()` writes the provenance manifest into an `LKPV` RIFF
  chunk. 0.1.1 reads WAVs saved by either release. The 0.1.0 reader
  (`read_provenance`, `verify_provenance`, `loudkit verify`) raises
  `ProvenanceError` on a WAV saved by 0.1.1. WAV replies from the server
  still append the manifest after the `data` chunk, as 0.1.0 did.

### Added

- `mp3` and `opus` output from `/v1/synthesize`, `/v1/audio/speech`, gRPC and
  MCP. Both use the codecs of the libsndfile that `soundfile` bundles, so
  neither adds a package. OpenAI-compatible clients that request mp3, or Opus
  for a voice note, now get it from `loudkit serve`. `aac` stays refused,
  because libsndfile cannot write it.
- Conformance coverage:
  - The long-form conformance case also runs through `engine.stream` and
    `engine.synthesize`, as well as chunk by chunk, so Python consumes the
    flat `long_form.cases[].tokens` list that the four ports consume.
  - A `slow` test regenerates the seven weighted rows of the parity page with
    `--checkpoint` and compares them. The `parity` job runs it.
  - Enrollment from a WAV file is checked against the enrollment fixture.
  - `lk.enroll` and `loudkit speak` on a turbo release directory have goldens.
  - `Result.save()` is pinned to the shared quantise rule.
  - The production fingerprint is pinned without weights.
  - The Swift offline-receipt test asserts the stderr line that the other four
    ports assert.
- `loudkit text` previews what will be spoken, without loading a model: the
  prepared text on stdout, and a word diff of what the funnel changed on
  stderr. `loudkit --help` lists eight commands.
- `loudkit speak --play` plays the saved WAV through the system player
  (`afplay` on macOS, `aplay` or `paplay` on Linux).
- Speed measured on 0.1.1 for both models: eight local paths on an Apple M3
  Pro (PyTorch split CPU/MPS, native CoreML, ONNX Runtime CPU, the CPU
  reference, and the Swift, Rust, Go and TypeScript ports) and six NVIDIA
  parts (A100, L4, RTX 3090, T4, GTX 1080 Ti, Jetson Orin Nano), eager and
  with CUDA graphs, plus batched throughput on four of them. loudr-1-turbo
  runs 1.4x to 2.2x faster than loudr-1 end to end and 2.4x to 2.8x faster in
  aggregate throughput ([Benchmarks](docs/benchmarks.md)).
- A second model, `loudreader/loudr-1-turbo`.
  `lk.load("loudreader/loudr-1-turbo")` selects it, with the same voices, API
  and seeds. It combines the `fusion_mtp2` generator below with a renderer
  distilled to one Euler step, and ships as a second repository with the same
  layout. Python, Swift, Go, Rust and TypeScript support both models, with the
  matching ONNX graphs or CoreML packages. One enrolled voice profile works
  with both models. [Choosing a model](docs/guides/11-choosing-a-model.md)
  compares them.
- A second decode mode, `fusion_mtp2`, with two speech tokens per transformer
  forward. One head reads the hidden state, and a second head reads that state
  together with the first token's embedding. A fusion MLP folds both into the
  one KV slot the pair occupies. Checkpoints with this mode declare
  `decode.mode` in their manifest and ship as `format_version 2`. The
  checkpoint sets the mode; there is no runtime option for it. The mode is
  fingerprinted, and a mode that disagrees with the weights is a load error.
  RTF with CUDA graphs on one checkpoint lineage, on the short, medium and
  long `tools/bench.py` passages
  ([two-token decode](docs/design/two-token-decode.md)):

  | | 0.1.0 | 0.1.1 |
  |---|---|---|
  | RTX 3090 | 1.22 / 6.03 / 7.73 | 1.43 / 7.47 / 8.45 |
  | Jetson Orin Nano | 0.53 / 1.60 / 1.84 | 0.63 / 1.82 / 2.16 |

  These development figures compare the two decode modes. They match neither
  shipped model; [docs/benchmarks.md](docs/benchmarks.md) has the figures for
  the shipped models.
- Execution changes, none of them fingerprinted: sampling runs on the device
  inside the captured graph, graphs are kept across calls, the excitation
  noise is drawn on the GPU, and the vocoder no longer renders padding that
  is then discarded (`ExecutionConfig.vocoder_ragged`, on by default on CUDA
  renderers, in the identity contract's `equivalent` class).
  `ExecutionConfig(vocoder_ragged=False)` gives byte agreement with a 0.1.0
  build (`--no-vocoder-ragged` in `tools/bench.py`). End-to-end figures are
  in [docs/benchmarks.md](docs/benchmarks.md).
- New release bundles use a 20 ms raised-cosine ramp on both window edges to
  soften onset taps. On three of the 28 voices, the tap is a high-energy first
  mel frame from the flow decoder. It is not a vocoder edge. Measured on
  oscar, the voice with the strongest first frame: the residual burst in the
  first 12 ms of a window falls from 2.2 dB below the window RMS with the old
  5 ms ramp to 14.7 dB below with 20 ms. The cost is 0.7 dB over the first
  50 ms of a window that opens on speech; a 50 ms ramp would cost 3 dB there.
  `edge_fade_seconds` is part of the algorithm and its fingerprint, including
  the new 0.02 s default. Only the historical 0.005 s spelling stays omitted,
  for identity compatibility.
- `loudkit clone` ends the reference prompt in silence by default. It cuts the
  clip at its last pause before the ten-second limit, or at 9.6 s when no
  pause fits, and pads it with 0.4 s of silence, because generation continues
  from where the prompt ends. On one cloned voice, the onset moved from 0 to
  50 ms and the level of the first 60 ms from -32 to -51 dBFS.
  `--no-end-in-silence` keeps the recording as given. The library's `enroll`
  also keeps it as given, unless asked to cut. Profiles made with the cut
  carry the enrolment label `first-10s-pause`.
- Every download directory carries a receipt, `.loudkit-release.json`: the
  repo, the revision asked for, the commit it resolved to, the digest of
  `SHA256SUMS` and the fetch time. A load whose revision still resolves to
  that commit hashes nothing. Any other receipt, or none, fetches and verifies
  again, and keeps every file that already matches the new manifest. A load
  that cannot reach the Hub uses the receipt and says so on stderr. Python
  (`loudkit download --local-dir`), Go, Rust, JS and Swift apply the same
  rules, and one fixture pins them. A load makes one Hub call. A Hub answer
  that names no commit is an error in all five.
- The JS binding no longer needs Python. `download(repo, dir)` fetches the
  ONNX release with `fetch` and no new dependency, with the same plan and
  rules as `loudkit download --for onnx`. `examples/hello.mjs` is the
  quickstart. `Enroller.enroll` takes a WAV path or bytes as well as samples.
- New port functions. Go: `safetensors.Write`, `voice.Profile.Save`,
  `engine.Result.SaveWav`. Rust: `safetensors::write`, `voice::Profile::save`,
  `hub::rejected_name`, `hub::parse_sums`, `hub::is_official`,
  `Bundle::can_enroll`, `execution::locate_runtime`. JS: `saveVoice`,
  `profileFrom`, `writeSafetensors`, `hub.rejectedName`, `hub.parseSums`,
  `hub.canEnroll`, `Enroller.close`. Swift: `Safetensors.write`,
  `VoiceProfile.save`, `Engine.voiceNames`, `Engine.voice(named:)`,
  `Engine.enroll(contentsOf:)`.
- `export.json` beside every exported graph set, ONNX and CoreML, names what
  each file was traced from. The backends refuse a set whose members disagree
  with each other or with the checkpoint, and a release requires one.

### Changed

#### Models and downloads

- Turbo CoreML uses an fp32 estimator. Its waveform correlation with ONNX now
  exceeds 0.999 on both conformance cases, and the stricter gate is shared.
- Python downloads the native generator and fusion graphs that the chosen
  backend needs. Legacy renderer-only CoreML bundles remain readable.
  Complete CoreML bundles run without PyTorch.
  `ExecutionConfig(generator_device="cpu")` keeps the PyTorch generator with
  CoreML rendering, in fp16 if requested, which can be faster for loudr-1
  ([Apple platforms](docs/platforms/apple.md)).

#### Port caches and enrollment

- Go, Rust, JS, Swift: `Enroll` / `enroll_wav` / `enroll` on an engine loaded
  by repo id fetches the three enrollment graphs (Swift: packages) into the
  engine's own cache directory on first use, through the same verified
  download, and reads them from there afterwards. An engine loaded from a
  directory of your own reports how to fetch them (`Cloning: true` /
  `cloning: true` / `{ cloning: true }`).
- Go, Rust, JS, Swift: one cache layout, `<user cache>/loudkit/<org>--<name>`
  (`~/Library/Caches` on macOS, `$XDG_CACHE_HOME` or `~/.cache` on Linux,
  `%LocalAppData%` on Windows). `$LOUDKIT_CACHE` replaces
  `<user cache>/loudkit` in all four. The other three ports read a release
  that one port fetched. `tests/data/conformance/cache_path.json` pins the
  layout.
- Rust: `hub::cache_dir()` is now `hub::cache_dir(repo)` and returns the
  release directory. The layout moved from `<root>/loudkit/<org>/<name>` to
  `<root>/loudkit/<org>--<name>`, so a release fetched by 0.1.0 is fetched
  again once. `hub::cache_path(root, repo)` is the pure rule. `hub::Bundle`
  gained `repo` and `fetch_cloning()` / `fetch_cloning_with(&Options)`.
- JS: the cache moved from `$HF_HOME/loudkit/<org>--<name>` to the platform
  user cache. `HF_HOME` is no longer read, so a release fetched by 0.1.0 is
  fetched again once.
- Swift: `Engine.enroll(contentsOf:name:language:)` is now `async throws`;
  add `await`. `ModelBundle.fetchCloning()` is public.
  `LoudKit.cacheDirectory(repo:revision:)` is now `cacheDirectory(repo:)`,
  with no per-revision subdirectory (the receipt carries the revision), and
  `LOUDKIT_CACHE` replaces `LOUDKIT_HOME`. `LoudKit.cachePath(root:repo:)` is
  the pure rule.
- Go: `$LOUDKIT_CACHE` is honoured.
- Go, Rust, JS, Swift: offline, `enroll` on an engine whose cache holds the
  synthesis set names the missing enrollment graphs (Swift: packages) and says
  that one connection fetches them. Before, it gave the transport error (Go,
  Rust), "holds no verified release" (JS) or "cannot reach" (Swift). A plain
  load offline still uses the cache.
- Python, Go, Rust, JS, Swift: the receipt reader reads nothing past 1 MiB,
  and a larger file is not a receipt. The shared fixture includes a JSON
  array, a number and a string as receipts.

#### The API in all five implementations

- `synthesize(text, voice, options)` takes text of any length in every port
  and returns one result with a method that saves it. The one-window call is
  `synthesizeWindow` (`SynthesizeWindow`, `synthesize_window`). Options are
  named and have the same defaults everywhere: seed 0, the voice's own
  language, speed 1.0, no previous tokens. Go: `Options` and `Result` in
  `go/engine`, re-exported by the root package. Rust: `engine::Options` with
  `Default`, `engine::Synthesis` with `save_wav`; `stream` takes `&Options`.
  JS: `SynthesisOptions` gains `seed`, `language` and `shouldCancel`.
- `load(ref)` gives access to voices in every port. `engine.voices()` and
  `engine.voice(name)` (Swift: `voiceNames`, `voice(named:)`) read the release
  the engine was opened from. Rust `Engine::load(ref)` replaces `load_bundle`
  (three-path form: `load_paths`). JS `Engine.load(ref)` takes one argument
  (three-path form: `Engine.loadPaths`). Swift `Engine.load(ref)` (async)
  replaces `load(repoOrDirectory:)`; `ModelBundle` stays available through
  `Engine.load(bundle:)`. Python gains `Engine.voice(name)`, `Engine.voices()`
  and `Engine.checkpoint_path`.
- Every port clones with `engine.enroll(...)`, which takes a recording, a name
  and a language: Go `Enroll(wav, name, language)`, Rust
  `enroll_wav(path, name, language)`, JS `enroll(wav, { name, language })`,
  Swift `enroll(contentsOf:name:language:)`. The profile saves in one layout,
  so a voice cloned in one implementation loads in the other four.
- Python `Engine.synthesize` takes text of any length: one window when the
  text fits, splits at sentence boundaries when it does not, and one `Result`
  either way, with the same tokens that `stream` yields for the same seed.
  `synthesize(..., single_window=True)` renders one window and raises
  `WindowOverflowError` instead of splitting (`long_form=false` on the wire
  maps to it). All five use one seed rule: chunk 0 uses the caller's seed
  itself, and only later chunks use `derive(seed, 16 + index)`. A text that
  fits one window therefore reads the same through every call.
- Cancellation behaves the same in all five implementations. `synthesize`
  returns the whole passage or signals cancellation explicitly; it never
  returns only the chunks that finished. Python raises `CancelledError` (a
  `LoudkitError`, code `cancelled`; the `Result | None` overloads are
  removed), Go returns `ErrCancelled`, Rust `Err(error::CANCELLED)`
  (`error::is_cancelled`), JS throws `CancelledError`, and Swift throws
  `LoudKitError.cancelled` (`Result.cancelled` is removed). `stream` delivers
  the chunks that finished and then ends without raising. Rust `Options`
  gains `should_cancel`, and Python's `Engine.synthesize` takes it. A
  disconnected HTTP or gRPC client no longer holds the engine's slot until
  the last token (HTTP answers 499).
- One name for the token-cap flag in every port: `Result.hit_token_cap` in
  Python, `HitTokenCap` in Go (was `Truncated`), `hit_token_cap` in Rust (was
  `truncated`), `hitTokenCap` in JS (was `hitCap`) and in Swift. Go's
  `Engine.Enroll` and `EnrollPCM` take the recording first, as the other four
  SDKs do.
- HTTP, gRPC and MCP refuse the same over-long request with the same code.
  The cap refusals of the MCP `synthesize` tool carry
  `code: "invalid_request"`. Every loopback address counts as local for all
  three transports.
- A second concurrent `generate` on one engine is refused. A checkpoint whose
  `decode.mode` needs a higher `format_version` than it declares is refused at
  load, in all five engines.
- Rust finds the ONNX Runtime library in the same order as Go:
  `ORT_DYLIB_PATH`, then `LOUDKIT_ONNXRUNTIME_LIB`, then the usual install
  paths.
- JS `Engine.load(dir)` no longer requires `manifest.json`. It reads the
  manifest the checkpoint embeds, as the other four do.
- Swift `Result.save(to:)` is now `saveFloat32Wav(to:)`. `saveWav(to:)` and
  `saveWav(_ path:)` write 16-bit PCM.

#### Python

- `loudkit.__all__` has 23 names, down from 40: the six functions `load`,
  `voice`, `voices`, `enroll`, `languages` and `best_device`, then `Engine`,
  `Result`, `VoiceProfile`, `MIN_SPEED`, `MAX_SPEED`, eleven error classes
  and `__version__`. Everything else stays importable from the module that
  owns it, which [COMPATIBILITY.md](docs/reference/COMPATIBILITY.md) names.
- `ExecutionOverrides` is removed. Every `ExecutionConfig` field defaults to
  `None`, which means "the checkpoint's default for this device". A named
  field wins even when it equals the default, and `ExecutionConfig.resolved()`
  fills the rest.
- Postprocess is a per-checkpoint preset. `PostprocessConfig` and the eight
  detector functions left the public surface. `inspect`, `Inspection`,
  `Reason` and `PostprocessMode` remain, and `mode` is the only setting.
- The fingerprint is internal. `Result` carries one
  `Result.provenance: Provenance` value in place of eight fields, and `save()`
  writes it into the provenance manifest (see Breaking changes above).
  `FINGERPRINT_SCHEMA` and `DEFAULT_ALGORITHM` left `config.__all__`.
- `loudkit.engine` is four modules (`engine`, `result`, `window`, `stream`),
  `loudkit.config` is three (`config`, `execution`, `manifest`),
  `loudkit.hub` is three (`hub`, `release`, `checksums`) and
  `loudkit.transports.http` is three (`http`, `limits`, `schemas`). Every
  public name stays importable where it was. `loudkit.frontend.polish` is now
  `loudkit.frontend.speechtext`.
- `lk.enroll(path)` reads the file at its native rate with soundfile, averages
  the channels and resamples with the Hann-sinc resampler that every port
  ships. A 44.1 or 48 kHz recording therefore enrols the same profile from
  Python as from any port. `enroll(samples, ...)` takes `sample_rate`
  (default 24000).
- `loudkit.hub.read_receipt(root, repo)` returns a validated `Receipt` or
  `None` and is the only reader of the file. `receipt_hit` wraps it, and the
  ports' `readReceipt` / `read_receipt` follow the same shape. Python asks the
  Hub once per `(repo, revision)` per process.

#### Command line and tools

- Eight CLI commands: `speak`, `text`, `clone`, `voices`, `download`,
  `serve`, `verify`, `doctor`. `serve --grpc` and `serve --mcp` replace the
  `grpc` and `mcp` commands, `doctor --describe` replaces `describe`, and
  `bench` and `profile` moved to `tools/`. `--checkpoint` defaults to
  `$LOUDKIT_CHECKPOINT`, else `loudreader/loudr-1`.
- `tools/build_release.py` and `tools/build_turbo_release.py` are one package,
  `tools/release/`, run as `python -m tools.release --model loudr-1|turbo`.
  The two scripts stay as shims and refuse a user-supplied `--model`.
  `--skip-verify` under `--model turbo` is refused before any copying.

#### Documentation

- `docs/README.md` lists the twelve pages a user needs. Everything else is
  under `docs/reference/`, `docs/platforms/` and `docs/design/`. The README is
  the front page for all five languages. Internal vocabulary was removed from
  the user pages, and the provenance promise is narrowed to Python and the
  server. `docs/parity-measured.md` is regenerated with the release weights,
  names what the free-running row compares against (its own stream,
  re-based for this release) and states that row's per-architecture scope.
- The four port READMEs and guides 07 to 10 clone with the same two calls as
  Python and name the cache directory. They no longer show a second download
  into a directory of your own as the way to clone. The Go and Rust READMEs
  now carry the sentence about `Enroll` / `enroll_wav` on a repo engine that
  the JS and Swift READMEs had. Guide 01 names the ports' cache, and guide 11
  explains how to choose between the two models.

### Security

- One verification rule at download, in all five implementations. `download`,
  and a repo-id `load` that has to fetch, fetches `SHA256SUMS` and
  `release.json` before anything else. Under `loudreader/`, the record must
  name a release profile (`full-0.1` or `turbo-0.1`) and `verified: true`, or
  the bundle is refused before any weight is downloaded. Every fetched file is
  hashed against the manifest as it arrives. A mismatch is refused and the
  file removed, and an unlisted weight is refused. A listing name that escapes
  the directory is refused in all four ports.
- The `.loudkit-verified` marker and re-verification on every load are
  removed from Python, Go, JS and Swift. The receipt replaces them.

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

### Fixed

#### Silence

The decoder could enter a run of silence and not leave it. In a study of 1705
passages (20 voices, ten languages), 33.0% of paragraphs carried a gap longer
than one second, and 74 chunks rendered no speech at all. Six rules address
this. Each moved the fingerprint and re-based the goldens.

- A new verdict, `stall`, for a row that the decoder spent trapped in silence.
  The four earlier tail rescues all cut a tail, and this failure is a gap
  inside the row. `stall` cuts nothing. Like `dropout`, it sends the row to the
  retry ladder, and it runs before every tail rescue. It is the seventh
  `Reason`, and every port runs it.
- The repetition penalty applies to silence ids. Exempt from both the penalty
  and the `min_p` cutoff, a silence run had no exit: zero escapes in 1,031
  instrumented trap steps. The penalty exemption is removed and the `min_p`
  exemption stays; removing the `min_p` exemption instead was much worse
  (median gap 2.46 s to 4.64 s). Measured at 120 passages per arm across ten
  languages: paragraphs with a gap 33.0% -> 4.3%, mute chunks 74 -> 1, WER
  unchanged or better.
- `repetition_silence: acoustic` tells a pause from a loop. A pause on an id
  that only the render census knows goes to stall and retry, and is not cut.
- `repetition_resume: condemn` sends a loop that the decoder resumed from to
  the retry ladder. A tail cut would remove the speech that followed it.
- A cap-hit trim that keeps fewer than `desperation_min_keep_per_text_token`
  tokens per text token goes to the retry ladder, with the trim as the
  fallback.
- When the retry ladder ends, it ships the attempt with the fewest tokens in
  the true-silence set, instead of the last attempt.

#### Text funnel

The funnel marker moves from `funnel-2` to `funnel-6`. No released build has
`funnel-3`.

- `funnel-4`: the five implementations agree on every character class. Each
  pass asks whether a character is a letter, a digit or a space, and the five
  implementations answered those three questions in eleven different ways.
  Each difference changed the audio.
  - Numerals. A number character with no ASCII spelling (a superscript, a
    vulgar fraction, a circled or Roman numeral, or a digit from any script
    except Latin) was deleted: `Add ½ cup` read as *add cup*. `fold_numerals`
    now reads a generated table, `models/data/numerals.json`, built from one
    pinned UCD (16.0.0). A digit of any script becomes the ASCII digit of the
    same value, and every other numeral becomes the text the table names
    (`½` is `1/2`). Detection also comes from the table, so the Unicode
    version of a runtime does not decide what is read aloud.
  - Letters, whitespace and boundaries. `\p{L}` everywhere (Swift's
    `CharacterSet.letters` deleted 6,145 Tangut ideographs). One written-out
    whitespace class (Go read footnote markers aloud after a thin space).
    `(?![\p{L}\p{Nd}_])` written out in place of `\b`, whose ECMAScript and Go
    versions are ASCII-only.
- `funnel-5`: a clock time followed by letters reads as separate words.
  `3:45pm` read *three forty-fivepm* and now reads *three forty-five pm*, the
  same as `3:45 pm`. German `14:30Uhr` reads *vierzehn Uhr dreißig*.
- `funnel-6`: marks that the funnel had no rule for.
  - The ASCII operators `<`, `>`, `<=`, `>=`, `!=` and `==` are read when they
    have whitespace on both sides. `if latency > 200 ms` read
    *if latency two hundred ms* and now reads
    *if latency greater than two hundred ms*.
  - Markup tags and HTML comments are removed, and a tag becomes a space.
    `<p>Hello</p>` read *p Hello p*.
  - Roman numerals written with I, V and X, from 2 to 39, are read as
    numbers. `Chapter IV` read *Chapter eye-vee* and now reads
    *Chapter four*.
  - A fraction numeral reads as a division. `½` read *one two* and now reads
    *one divided by two*. A typed `1/2` is not affected.
  - A price with a scale reads amount, scale, currency. `$2.5M` read
    *two point five dollarsM* and now reads *two point five million dollars*.
    `$5 million` read *five dollars million*.
- Symbols are read in the language of the text. The grammar had a word per
  language for the seven marks that are also currency or measure, and the
  funnel had only an English/Polish pair for the other twenty-one. Ten of the
  twelve languages therefore read those in English: `≈` read *about* in a
  German render and in a Finnish one. Thirteen word-valued symbols
  (`× ÷ ≈ ≥ ≤ ≠ ± ✓ ✔ ✗ ✘ & @`) gain a row per language. The eight that are
  punctuation in every language (`→ ← ⇒ • · ▪ ◦ …`) stay a rule in code.
  Measured over every spoken symbol in three carriers in twelve languages,
  360 of 1008 rows change. Every conformance case read in its declared
  language is unchanged.
- All five funnels leave a symbol that has no row in the grammar as written.
  Four ports fell back to the English row, where the reference left the symbol
  as written. Measured on a grammar with eight rows removed, one per funnel
  pass, 46 of 576 cases differed. Swift had a second copy of the wording table
  as a literal, beside the grammar file in its own resources. It now reads the
  file.

#### Chunker

- `ChunkConfig.mid_sentence_period: hold` keeps a period that does not end a
  sentence. `"But Mr. Smith went home"` produced the chunk `"But Mr."`, with
  its own seed and a ceiling too short for a closing pause. A period is held
  when the next word starts in lower case or the token before it is in
  `ChunkConfig.abbreviations`. `"break"` keeps the 0.1.0 behaviour.
- `ChunkConfig.cap_resplit: word` splits a chunk that the window cannot hold.
  In a census of 9920 rendered chunks in ten languages, 54 overran the
  character budget and 30 were still speaking when the cap closed. The rest
  of their text was never spoken. Such a chunk is now replaced by its two
  halves, cut at the word boundary nearest the middle. Measured again over the
  51 passages that carry a cap hit: windows at the cap 52 -> 2, windows still
  speaking at the cap 29 -> 1. `"off"` keeps the 0.1.0 behaviour.
- The word-boundary fallback recognises the whitespace that prose uses, not
  only U+0020. Text whose every space was non-breaking was cut mid-word.
- Six chunking differences between the five implementations are now pinned in
  the shared fixture. Among them: the reference dropped the inverted Spanish
  marks `¿¡` that four ports kept, and Swift halved on grapheme clusters
  instead of scalars.

#### Other fixes

- Voices load from a release in the Hub cache. The cache holds one symlink per
  file from the snapshot into its blob store. Both the voice resolver and the
  server's voice library refused those links as escapes, so
  `engine.voice("joe")`, `loudkit speak --voice joe` and `/v1/voices` found
  nothing after a download. A link is now confined to the cache entry that
  owns the snapshot. A link out of it, or out of a plain directory, is still
  refused.
- CoreML generator exports use a weighted sum for single-query attention.
  This avoids a CPU matrix-kernel error that changed Turbo tokens in longer
  windows. Both model bundles include regenerated, token-verified generator
  graphs.
- A receipt with one wrong field could pass as valid offline. All five now
  check `.loudkit-release.json` with one reader before they use it, offline or
  as an online hit: every field present with its type (`null` is the wrong
  type), `repo` equal to the repo asked for, `commit` forty lowercase hex,
  `sha256sums` equal to the digest of the `SHA256SUMS` on disk, and every
  listed file that the plan selects present. The fixture holds 28 cases,
  including `null` in every field.
- Python never refreshed a moving `main`, and the native caches were not
  bound to a revision. A repo id now always resolves through `download`.
- The one-window path refuses text that does not fit, in all five
  implementations, instead of leaving the tail unspoken. None of the five
  raised the `WindowOverflowError` that their docs promised.
- A streaming response is capped at ten minutes of wall clock, on HTTP as on
  gRPC. A client that stopped reading held the engine slot indefinitely.
- The Swift snippet on the front page and the landing page did not compile
  (`saveWav(to:)` takes a `URL`; the string form is `saveWav("hello.wav")`),
  and the landing page's Go snippet never read its `err`. The suite of each
  port now compiles its page blocks. `SUPPORTED.md` no longer advertises the
  old CLI.


## [0.1.0] - 2026-08-25

First public release.

### Added

#### Speech

- A two-stage engine optimized for local inference: an autoregressive token
  generator at 25 Hz and a parallel renderer. It runs faster than real time on
  a laptop, a phone and a Jetson. Benchmarks with the command that reproduces
  each number are in [docs/benchmarks.md](docs/benchmarks.md).
- 20 voices across 10 languages. Each one is enrolled from a recording made or
  released for speech-technology use (consented donations and CC0 or CC-BY
  corpora), with the donor or source, licence and consent basis named. Quality
  is evaluated for English. The other languages are not yet measured.
- Voice cloning from about ten seconds of audio. It produces a voice profile
  file of about 150 KB; no model is fine-tuned.
- Long-form synthesis that splits at sentence boundaries and carries prosody
  across the joins, and streaming that yields the first sentence without
  waiting for the paragraph.
- Token-level cancellation: the generator stops within one decode step of a
  barge-in, instead of at the end of a chunk. It is cooperative, so a kernel
  that is already running is not interrupted.

#### Latency

- Streaming renders window *k* while it generates window *k+1*. The chunk
  chain is sequential only through the tokens, so the overlap changes no byte
  (asserted against the serial path). Measured on an M3 Pro: a six-window
  passage 1.29x faster, the Apple bench row 2.81x → 3.43x.
- `ChunkConfig.first_chunk_max_tokens` caps only the first chunk, so a stream
  opens on the first clause instead of a full window: first audio ~1.9 s →
  ~1.4 s at a 96-token budget (M3 Pro). It is an algorithm value: setting it
  changes the fingerprint, and when it is unset it is absent from the
  fingerprint. Python first; the ports follow.
- The generator's conditioning row is memoised per voice (a content key), so
  chunks and repeated requests in one voice do not compute it again.
- `Engine.warm()` runs the first-use work (kernel autotune, graph capture,
  allocator pools) at startup. `serve`, gRPC and MCP call it after loading, so
  the first request gets warm latency (measured on a 3090 with graphs: first
  audio 1.09 s → 0.73 s, for 1.3 s once at startup).

#### Command line

- `loudkit doctor` reports what this machine can run and the command that
  fixes each gap.
- `loudkit download <repo>` fetches the checkpoint, the graphs one backend
  needs and all twenty voices into the shared cache. `--with-cloning` adds the
  encoder and the enrollment graphs.
- `loudkit voices <repo>` lists the voices without downloading the model.
- `loudkit verify <path>` checks a checkpoint, voice profile or rendered WAV
  against its own records (payload digest, profile validation, provenance
  binding).

#### Errors

- A catalog of stable error codes, defined in `loudkit.errors` and used by
  every transport: `exc.code` on Python exceptions, `"code"` in HTTP error
  bodies, `"error_code"` on the SSE terminal event, and `loudkit-error-code`
  in gRPC trailing metadata. Codes are never renamed or reused, only added.
  See [docs/reference/errors.md](docs/reference/errors.md).

#### Five implementations

- Python is the reference. Swift, Go, Rust and TypeScript are full ports, not
  wrappers, and shared conformance fixtures check them. All five compute the
  same algorithm fingerprint independently and agree.
- The same text, voice and seed give a bit-identical waveform on a given
  build, device and backend. Waveforms are **not** guaranteed to match across
  backends or devices. The engine refuses to start if two of its components
  disagree about what to compute. See
  [docs/reference/IDENTITY-CONTRACT.md](docs/reference/IDENTITY-CONTRACT.md).

#### Text handling

- Numbers, decimals, currency, units, times and abbreviations are read out in
  **12 languages** by loudkit's own implementation, with no LGPL dependency.
  A 1300-row CLDR differential checks it.
- Sentence-boundary chunking, NFC normalisation, output charset closure, and a
  Polish anglicism respeller.
- Bracketed text in user input can no longer trigger model control tags.

#### Output checks

- Six detectors read the tokens the generator produced and find failures such
  as a hallucinated tail after a long silence, in the same voice. Every
  threshold comes from a measured trace and was measured again across the
  nine spoken languages the roster then carried (Swedish, added later, ships
  unmeasured with the same margin). A chunk that is wrong but has no trim
  point is *reported*, and is not shipped as clean. See
  [docs/design/postprocess.md](docs/design/postprocess.md).
- Selective re-roll: a chunk that fails is rendered again from a derived
  seed.

#### Timing, speed and prosody

- `Result.chunks` gives exact per-chunk spans from sample offsets, plus
  **estimated** word times by proportional allocation. Word times are
  estimates: there is no forced alignment
  ([docs/reference/timestamps.md](docs/reference/timestamps.md)).
- `speed=` changes tempo without changing pitch, with loudkit's own WSOLA
  implementation ([docs/reference/speed.md](docs/reference/speed.md)).
- `previous_tokens=` carries prosody across two separate calls.

#### Interfaces

- A local HTTP server under `/v1` with one-shot and SSE streaming routes and
  four output encodings (wav, pcm16, flac, ogg). It is a working example, not
  a production deployment. A public bind requires a bearer token. On a
  loopback bind, `loudkit serve` needs no token and ignores one that is given;
  an app built with `build_app(token=...)` checks the token on any bind (see
  the server guide).
- An MCP server for agents, and a Linux Speech Dispatcher module so screen
  readers can use it.
- Discovery without loading anything: `loudkit.languages()`,
  `loudkit.voices()`.

#### Provenance

- Saved WAVs carry an unsigned provenance manifest (JUMBF boxes appended after
  the WAV data chunk) that binds the fingerprint, recipe and seed to a hash of
  the audio. One-shot HTTP responses also carry it in the
  `X-Loudkit-Provenance` header
  ([docs/reference/provenance.md](docs/reference/provenance.md)). The
  provenance guide documents the manifest fields, verification and the signing
  boundary.

### Known limitations

- Quality evaluated for English only.
- Ordinals are English-only (`22nd`). The other eleven languages need
  inflection rules that 0.1.0 does not have.
- Yearless numeric dates (`12.3.`) are **not** read as dates in any language.
  The same text can be a decimal at the end of a sentence.
- There is no expressiveness control. The checkpoint architecture reserves a
  conditioning slot for an emotion scalar, but the value does not change the
  output in the released checkpoint. The field was removed from
  `VoiceProfile` and the profile format before release. The slot is fed the
  training constant, and a profile that carries a legacy `emotion` header key
  still loads (the key is ignored).

### Licence

- Apache-2.0, derived from Chatterbox (MIT, Resemble AI). The enrollment
  architectures come from CosyVoice, S3Tokenizer and 3D-Speaker (Apache-2.0),
  FunASR and Real-Time-Voice-Cloning (MIT). The Polish respelling lexicon is
  derived from CMUdict (CMU, BSD-2-clause). Every upstream licence is
  permissive and none is copyleft. [NOTICE](NOTICE) names each holder.
