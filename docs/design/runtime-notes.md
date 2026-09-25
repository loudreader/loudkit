# Runtime notes

Maintainer notes for the runtime's checkpoint, configuration, error, provenance
and voice modules: the reasoning and the measurements behind each symbol. Each
heading names the module and the symbol a note belongs to. User documentation
is under `docs/guides/` and `docs/reference/`.


## `loudkit/checkpoint.py`


### `module`

A loudkit checkpoint is one `safetensors` file with a JSON manifest embedded in
its metadata. A release also ships a `manifest.json` copy beside it. Tensors
live in three namespaces:

- `t3.*`: the token generator;
- `s3gen.*`: everything downstream of it (flow, vocoder, and the enrollment
  models `s3gen.tokenizer` and `s3gen.speaker_encoder`);
- `assets.*`: text artefacts packed into the file (see `asset` below).

A release splits the weights into two files by `manifest["artifact_role"]`.
The synthesis checkpoint (`loudr-1.safetensors`, 747 MB) holds `t3`,
`s3gen.flow` and `s3gen.mel2wav`. The enrollment checkpoint
(`loudr-1-enrollment.safetensors`, 523 MB) holds the speech tokenizer and the
speaker encoder. A manifest with no role is a pre-split checkpoint that holds
every tensor, and it still loads.

The manifest is the source for values that are properties of the weights: the
silence-token ids, the Euler step count, the vocabulary bounds and the
per-module dtype map. Loaders read these values from the manifest. A field an
older manifest may omit takes its default from one place, `AlgorithmConfig`
(and `production_algorithm` for the window, EOS floor and postprocess blocks).
A second implementation that guesses such a default again is the divergence
this format exists to prevent (see the module docstring of `AlgorithmConfig`).

Two facts about the tensor payload matter to loading code:

- Precision is mixed per module and recorded in `manifest["dtype_map"]`,
  matched by longest prefix. The packed dtype is the storage dtype. A backend
  may upcast (fp16 to fp32 is exact), and it takes the compute dtype from its
  own `ExecutionConfig.precision`.
- The vocoder's weight-norm reparametrisation is already folded. The packed
  weights are plain `weight` tensors, bit-exactly equal to what the
  parametrised forward computes.

The module is torch-free: it reads NumPy arrays, so the ONNX and CoreML
backends load the same file without importing torch.


### `read_manifest`

The embedded manifest is authoritative. The sibling `manifest.json` is a copy
for humans and can drift if someone edits it, so `read_manifest` never reads it.

Raises `ValueError` if the file carries no manifest, if the manifest is not a
JSON object, or if it declares a format or `format_version` this build does not
read. A checkpoint is refused when its numbers cannot be read with the meaning
this build assigns to them.


### `_check_decode_version`

Refuses a manifest whose `format_version` does not cover its `decode.mode`, when
the file is opened. `DECODE_FORMAT_VERSION` gives the minimum version per mode.
`fusion_mtp2` needs version 2. A mode absent from the table needs no minimum, so
`single`, and an absent `decode` block, which means `single`, load from any
version-1 file.

The version number is the portable half of the contract. A version-1 reader
that ignores the `decode` block finds weights it knows, runs the one-token loop
over two-token weights, and produces fluent wrong speech with no error. Every
current engine reads `decode.mode` and refuses `fusion_mtp2` under
`format_version 1`: Python here, `Checkpoint::open` in Rust, `checkpoint.Open`
in Go, `Checkpoint.open` in TypeScript and `Checkpoint(url:)` in Swift. The gate
protects readers that check only the version.

A manifest that contradicts itself is refused. When the version and the mode
disagree, the loader cannot tell which one the packer meant, and a guess would
ship weights under a description nobody can confirm.


### `require_decode_support`

Refuses a decode mode this build does not run. Every backend runs both known
modes (`single` and `fusion_mtp2`), so only an unknown mode is refused.

- `mode`: the loop that will run. It is the resolved `AlgorithmConfig.decode`,
  not the manifest's raw block, so an explicit `algorithm=` override is judged
  on what it asks for.
- `target`: a backend name (`torch`, `onnx`, `coreml`) or a torch device (`cpu`,
  `cuda`, `mps`, with or without an index). One parameter takes both because
  the model fetch knows backends and the engine builder knows devices.
- `name`: the checkpoint's `manifest["name"]`, when the caller has it.

Raises `ValueError` naming the backend, the checkpoint (when `name` is given)
and the unsupported mode.


### `Checkpoint`

Tensors are read on demand, by namespace, not loaded wholesale. The two stages
of the engine may run in different processes or on different devices, and a
consumer loads only the stages it needs.

```python
from loudkit.checkpoint import Checkpoint

ckpt = Checkpoint.open("loudr-1.safetensors")
t3 = ckpt.tensors("t3.")          # generator weights, prefix stripped
ckpt.manifest["n_cfm_timesteps"]  # 2
```


### `file_digest`

The SHA-256 of the whole checkpoint file. A release's `SHA256SUMS` lists this
value, and provenance manifests carry it as `checkpoint_sha256`: it names the
artefact that rendered a waveform. It differs from `tensor_payload_sha256`,
which the manifest records inside the file and which covers the tensor payload
only. The file digest is computed on first use with one chunked read, and
cached for the opened checkpoint.


### `shapes`

Reads tensor shapes from the safetensors header and touches no tensor data, so
it is cheap even on the 747 MB synthesis checkpoint. It exists because it is
available before anything is allocated. The manifest declares the architecture,
and the architecture decides how much memory the model constructor requests.
An unchecked manifest lets a 20 kB file demand gigabytes. The shapes are in the
same file and cannot grow without the file growing, so builders check the
manifest against them (see `docs/design/execution-config.md`).


### `verified_sibling`

A checkpoint does not always carry everything it needs. The tokenizer, and on
the graph backends the exported ONNX or CoreML packages, can be separate files,
resolved by name from the checkpoint's directory. Replacing `tokenizer.json`
with another valid tokenizer changes the text ids, the speech, and possibly
where EOS lands, while `AlgorithmConfig.fingerprint()` stays the same: the
tokenizer is not part of the algorithm config, and `TextFrontend` carries no
config for `Engine._assert_one_algorithm` to compare. Two different readings
would then report the same identity.

`verified_sibling` checks a sibling file against the digest the manifest
records for it. The check is skipped when the manifest records no digest, so
checkpoints packed before the field existed still load.
`tools/amend_manifest.py` records `tokenizer_sha256`, and the release preflight
in `tools/build_release.py` refuses a bundle whose tokenizer does not match it.


### `asset`

The tokenizer and the Polish respelling lexicon are not weights, but they
decide what the weights are asked to say. A different `tokenizer.json` reads
the same text as different ids, and a different lexicon reads embedded English
differently. Shipped as loose files, they must be kept in step with the weights
by hand, and a sibling is bound to the weights only when the manifest records
its digest.

`tools/pack_assets.py` packs both into the checkpoint as `uint8` tensors under
`assets.`. The format needs no new container, because every port already reads
safetensors, and the bytes are covered by the same file as the weights.

`asset` returns `None` when the checkpoint has no packed copy.
`resolve_asset` then falls back to the verified sibling.


## `loudkit/config.py`


### `module`

`AlgorithmConfig` holds every value that defines the output and is the same on
every backend. `ExecutionConfig` holds placement, precision, kernels and thread
counts. Execution settings may differ per backend, and some of them change the
computed samples within the identity contract's classes. A setting belongs in
`AlgorithmConfig` when changing it on one backend alone would make that backend
read the text differently. The reasons for the split, and the measurements
behind each value, are in `docs/design/algorithm-config.md` and
`docs/design/execution-config.md`.


## `loudkit/errors.py`


### `module`

Named exception classes separate a refusal the caller can act on from a
defect. A transport boundary maps exception types to responses. With builtin
exceptions alone it cannot tell an unsupported language from a backend stub:
both raise `NotImplementedError`, but the first is the caller's question
(HTTP 400) and the second is a server defect (HTTP 500).

**Each class keeps the builtin it replaces as a base.** `except ValueError`
still catches a `WindowOverflowError`, and `except FileNotFoundError` still
catches a `VoiceNotFoundError`. A caller can also catch
`loudkit.LoudkitError` for "loudkit refused this", or one named class for one
refusal.

A class exists only for a condition a caller would branch on, and each carries
the values a caller would otherwise parse out of the message.

**Error codes.** Each class also names its condition with a short stable
string, `code`. An HTTP error body carries it as `"code"`, the SSE `done` event
as `"error_code"`, and gRPC as `loudkit-error-code` trailing metadata, so a
refusal has the same name on every transport. The catalog is frozen: codes are
never renamed or reused, only added. A refusal that no class names yet is
`invalid_request`, and a defect is `server_fault`. The transports add their own
codes for conditions that exist only at a boundary, such as `unauthorized` and
`busy`. `error_code` is the one mapping, and `docs/reference/errors.md` lists
the full catalog.


### `LoudkitError`

The base class of every named loudkit exception. It is never raised directly.
A caller that embeds the library writes `except loudkit.LoudkitError` to catch
the refusals loudkit names, without also catching a `ValueError` from NumPy.

It carries nothing of its own; the subclasses carry the diagnostics.

`LoudkitError` does not cover every error a caller can cause. Some argument and
configuration checks raise builtin exceptions, for example a `speed` outside
[0.5, 2.0] or an invalid `ExecutionConfig` field raises a bare `ValueError`. A
transport treats a bare exception it did not classify as a refusal as a
defect, and answers `server_fault`.


### `error_code`

A `LoudkitError` names its own condition through `code`. Any other exception
maps to `invalid_request`. The transport calls this function only for errors it
has classified as refusals, such as a bare `ValueError` from a layer that has
no class yet. An exception the boundary did not classify (a stub method, a
NumPy failure, a bug) never reaches this function, and the transport reports it
as `server_fault`.


### `__reduce__`

`BaseException.__reduce__` returns `(type(self), self.args)`, so unpickling
calls `cls(*args)`. Four subclasses take required keyword-only diagnostics
(`UnsupportedLanguageError`, `VoiceNotFoundError`, `InvalidTokensError` and
`WindowOverflowError`), and that call does not supply them. Default pickling
therefore raises `TypeError` and hides the original error wherever exceptions
cross a process boundary, for example errors returned from a
`ProcessPoolExecutor` worker.

The base class rebuilds every subclass through `__new__` and
`BaseException.__init__`, then restores its attributes. Giving the keywords
defaults would also fix pickling, but it would make the diagnostics optional at
every raise site.


### `NumberGrammarError`

Raised by `loudkit.frontend.numbers` when a language has no grammar, or when a
value is larger than the grammar's largest scale. Returning the digits instead
would give the caller `"1000000000"` with no way to tell it from a number the
grammar handled, and reading digits aloud is the failure that module exists to
remove.

It carries nothing beyond its message.

It is defined in `loudkit.errors` because `loudkit.frontend.numbers` imports
that module, and the class must live below its base. `loudkit.frontend.numbers`
re-exports it, and both names refer to the same class.


### `UnsupportedLanguageError`

Raised by `GraphemeTextFrontend.encode` (in `loudkit.frontend.text`) for any
language outside the twelve this build's text layer supports. For the languages
whose upstream pipeline needs model-based preprocessing that this frontend does
not carry (Cangjie codes, kana conversion, diacritisation, jamo decomposition,
stress marks), the message says so. The tokenizer holds tags for 31 languages
and would emit ids for any of them. A language the model was not trained on
would come out as fluent nonsense with no error downstream, so it is refused.

It carries:

- `language`: the refused id, lowercased;
- `supported`: the ids this build accepts, sorted, from
  `loudkit.frontend.numbers.supported_languages()`.

It is also a `NotImplementedError`, so `except NotImplementedError` still
catches it. The HTTP server catches this class, not the builtin, so a genuine
`NotImplementedError` from an unfinished backend is reported as a server fault,
not as a bad request.


### `VoiceNotFoundError`

Raised by `loudkit.synthesis.VoiceLibrary` when a request names a voice the
library does not hold, and by `loudkit.hub.resolve_voice` when a reference is
neither a file nor a voice in the named release.

It carries:

- `ref`: what was asked for, a name over HTTP, a path or a name in-process;
- `available`: the names that were found, when listing them is cheap (a local
  directory). It is empty for a remote repo, where listing would cost a network
  call to report an error.

When `available` holds a close match, the message ends with `did you mean
'<name>'?`. The suggestion is built in the class, not at each raise site, so
every site that lists alternatives gets it, and the message and the list cannot
disagree. Voice names are short lowercase donor or character names (`kathleen`,
`gosia`, `thorsten`), easy to mistype and easy to miss in a list of 28.

It is also a `FileNotFoundError`, so a caller that catches that still catches
this, and the CLI's "not found:" path is unchanged.


### `InvalidTokensError`

Raised when a caller-supplied speech token sequence cannot be rendered:
`previous_tokens` on `synthesize` and `stream`, and `tokens` on
`synthesize_tokens`, which also refuses an empty sequence.
`window.validate_speech_tokens` refuses a fractional id and any id outside the
acoustic codebook: a negative id, a control token, or an id past the
vocabulary. These ids index an embedding table, so without the check the
failure is an index error several stages away from the argument that caused
it.

It carries:

- `token`: the first offending id, so a caller filtering a long sequence knows
  which entry to look at;
- `limit`: the exclusive upper bound. Ids must satisfy `0 <= id < limit`.

A boundary must classify this error as the client's to fix. The streaming route
reports a `LoudkitError` as `bad_request` and anything else as `server_fault`,
and the HTTP routes answer 422 for a `ValueError` that is a `LoudkitError` and
500 for a bare `ValueError`. As a bare `ValueError`, one bad integer from a
client would be reported as a broken server.

It is also a `ValueError`, so an existing `except ValueError` around a
synthesis call still catches it.


### `WindowOverflowError`

Raised by `Engine.synthesize` with `single_window=True` when the text does not
fit one window, and by `Engine.synthesize_tokens` for a sequence longer than
`window.max_speech_tokens`. Truncating instead would drop text while the audio
still sounds complete. Plain `synthesize` never raises it: it splits the text.

It carries:

- `n_tokens`: how many speech tokens were produced or supplied;
- `window`: how many the window holds. The `single_window` message also states
  the window in seconds of speech.

It is also a `ValueError`, and the HTTP server answers it with 422.


### `AudioNotFoundError`

Raised when a recording that `loudkit.enroll` was asked to read is not there.
It is a `LoudkitError`, so `except LoudkitError` catches it, and a
`FileNotFoundError`, like `VoiceNotFoundError`, so `except FileNotFoundError`
catches it too.


## `loudkit/provenance.py`


### `module`

Article 50(2) of the EU AI Act asks providers of systems that generate
synthetic audio to mark the output in a machine-readable format, detectable as
artificially generated. This module writes such a marking as metadata, the
loudkit provenance manifest. It is not C2PA, and it is not an in-audio
watermark. Whether it meets a particular legal obligation is outside this note.

The marking has these properties:

- It is an **unsigned** manifest in JUMBF-shaped boxes, carried in an `LKPV`
  RIFF chunk after the audio. A player skips the chunk, and the `data` chunk
  keeps the bytes it would have without the manifest.
- It carries an assertion shaped like `c2pa.actions` with
  `digitalSourceType = trainedAlgorithmicMedia`, the IPTC term for media a
  model generated.
- It carries loudkit's own assertion, which makes the marking traceable: the
  algorithm fingerprint, the recipe version, the seed, the sample rate and
  speed, the voice label, the digests of the checkpoint and voice-profile files,
  the backend and execution description, the language, a hash of the text, and
  the SHA-256 of the audio bytes it binds to.

It is unsigned because a C2PA signature needs a certificate, and choosing whose
certificate, and managing its key, is a deployment decision a library cannot
make for its caller. The boxes alone are not a C2PA manifest either (see
below), so interoperable Content Credentials would need a C2PA implementation
as well as a signature.

It is metadata, with metadata's limits. Re-encoding or metadata-stripping tools
remove it, so it provides neither tamper resistance nor the robustness that
Article 50(2) also asks for as far as technically feasible. loudkit ships no
in-audio watermark: a watermark prototype did not pass its null test, because
its detector did not separate marked audio from unmarked audio. Disclosure to
listeners remains the caller's responsibility.

The box layout follows JUMBF's outline without being JUMBF: a superbox holding
a claim box holding a JSON box, each big-endian, length-prefixed and
UUID-typed, uncompressed, and readable with no dependencies. Real JUMBF opens a
superbox with a `jumd` description box, which these boxes omit, and
`c2pa-python` reports that when pointed at them. The chunk therefore carries a
private identifier, `LKPV`, and not `C2PA`: a reader that finds a `C2PA` chunk
expects a manifest store, and would report this file as broken instead of
ignoring it.


### `build_manifest`

The loudkit assertion carries the values that make the marking traceable:

- the algorithm fingerprint and the seed identify the algorithm and the random
  stream;
- the audio hash binds the manifest to these audio bytes;
- the checkpoint and profile digests name the weights and the voice that
  spoke. A fingerprint pins the algorithm, not the artefact, and a voice name
  is a label anyone can reuse;
- `backend` and `execution` name the datapath. Execution settings such as
  reduced precision and the thread count can change the samples, and in some
  cases the tokens, within the identity contract's measured classes.

These fields help trace and repeat a render. Repeating it exactly also needs
the input text and the same build, device and execution configuration; the
manifest stores only a hash of the text. An empty string means "not known
here", never "does not apply".


### `_stamped_now`

The `c2pa.actions`-shaped assertion records a creation time (`when`). It is the
only value in a rendered file that is not a function of the input, so it is the
one byte range in which two identical renders legitimately differ. The
transport suites assert that a transport returns byte for byte what the library
returns, so they patch this function (see `conftest.py`). Otherwise two
identical renders on either side of a second boundary would differ in the
trailer.


## `loudkit/voice.py`


### `module`

A `VoiceProfile` is the set of tensors the two stages need to speak as one
voice: a speaker embedding, a prompt of speech tokens, the mel of the reference
audio, and the conditioning the token generator was trained to read. A shipped
profile is about 150 KB and holds no weights.

Because a profile holds no weights, cloning is cheap and a voice is a small
file. The prompt is an input to the graphs, not part of them, so a new voice
needs no new exported model. Voices can be enrolled once on a fast machine and
shipped, and a device that only synthesises never needs the enrollment models
(the speech tokenizer and the speaker encoders), which ship in the separate
523 MB enrollment checkpoint and the 5.7 MB `ve.safetensors`.

Profiles are saved as `safetensors` with a small JSON header, so they hold no
executable content (no pickle). `VoiceProfile.load` also checks the file before
use: the size limit (`MAX_VOICE_BYTES`), the header's format version and field
types, the tensor shapes, finite values, embedding norms, token-id bounds and
the enrolment label.


### `VoiceProfile`

The two stages read the outputs of two different speaker encoders, so a profile
carries two embeddings. The token generator was trained against a 256-d
utterance-level voice-encoder vector, and the flow decoder conditions on a 192-d
CAM++ x-vector. Neither can be derived from the other, so both are enrolled
once and stored.

Attributes:

- `name`: a human label, carried for provenance and error messages. Nothing
  dispatches on it.
- `speaker_embedding`: the `(256,)` speaker vector for the token generator's
  conditioning encoder.
- `flow_embedding`: the `(192,)` x-vector for the mel decoder.
- `prompt_tokens`: the speech tokens of the reference audio, the prosodic and
  timbral prompt the mel decoder continues from. Stored at natural length; the
  window recipe (`WindowConfig`) decides the framing.
- `prompt_mel`: the `(80, frames)` mel of the reference, conditioning the flow.
- `cond_prompt_tokens`: the token generator's own conditioning prompt, which may
  differ in length from `prompt_tokens`.
- `source_sample_rate`: the sample rate of the enrollment input. It is
  provenance metadata only: `enroll` writes it, and no renderer, loader or
  engine reads it or resamples because of it. It stays in the format because
  it records a fact about the source.
- `language`: the language this voice reads in. It is the default for every
  `synthesize` or `stream` call without an explicit `language=`, so a profile
  marked `en` reads Polish text through the English funnel unless the call
  overrides it.


### `KNOWN_ENROLMENTS` and `enrolment`

The `enrolment` field records how the prompt was cut from the reference clip.
Enrolment chooses a window of the clip before any model runs, so two strategies
make two different voices from one recording, and nothing in the tensors shows
which strategy made them. The field is recorded for the same reason
`TextConfig.recipe` is: a build must not assume a strategy it cannot name.

`KNOWN_ENROLMENTS` lists the accepted labels, and a profile that names any other
label is refused at load:

- `first-10s`: the prompt is the first ten seconds of the clip. Every port's
  enroller makes this, and a profile with no label is read as this one.
- `first-10s-pause`: the clip is cut at its last pause before ten seconds and
  padded with 0.4 s of silence. `loudkit clone` makes this by default, and
  `enroll(end_in_silence=True)` makes it on request.

Every port loads both labels and uses the stored tensors as they are: a saved
profile is reused, never cut again.


### `_validate_values`

Profile values are checked once, at construction, not discovered per backend.
A profile is a file that can be copied, mailed and loaded from a source the
caller does not control, and the renderers disagree about a degenerate one.
Torch's `F.normalize` adds an epsilon and returns finite values for a zero
vector, while ONNX and CoreML divide by the raw norm and produce 192 NaNs. The
same profile would then speak differently per backend with no error, so an
embedding norm below `MIN_EMBEDDING_NORM` is refused.

The check also bounds every token id. A prompt token must be below the speech
codebook (`start_speech_token`), and a conditioning token below the full speech
vocabulary (`speech_vocab_size`). Both limits come from the shipped
`AlgorithmConfig`, not from constants repeated here. An id outside them would
index past the end of an embedding table, so it is refused before any backend
sees it.


### `cond_key`

The generator's conditioning is a pure function of `speaker_embedding` and
`cond_prompt_tokens` (the third slot is the constant `EMOTION_NEUTRAL`), so two
profiles with the same bytes there get the same cached row. The key hashes the
contents, not the object identity: profiles are frozen but freely copied, and an
`id()` key would miss on every copy. Hashing a few kilobytes per call costs far
less than the perceiver pass a cache miss runs.


### `EMOTION_NEUTRAL`

The checkpoint architecture reserves one of its 34 conditioning slots for an
emotion scalar (`t3.cond_enc.emotion_adv_fc`). On these weights the axis has no
effect (distillation collapsed the response), so the slot is not a control and
is not part of the profile format. It still needs an input, and that input is
0.5, the value the model was trained with and the value every profile carried.
The token generator in every implementation feeds this constant.


### `MAX_VOICE_BYTES`

The limit is 8 MiB. A shipped voice is about 150 KB, and the format has no
field that grows with anything a caller controls. The limit is about fifty
times the real size, which leaves room for a format change. Without a limit, a
profile whose `prompt_mel` is `(80, 100_000_000)` float32 would be read as
32 GB of tensor data, from a file the server loads by name on an
unauthenticated request. The limit is checked on
the file size, before any tensor is read, because the file size bounds every
tensor in it at once.
