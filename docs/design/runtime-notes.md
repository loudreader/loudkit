# Runtime notes

Maintainer notes moved out of the runtime's docstrings: the reasoning and
the measurements behind each symbol. Each heading names the module and
the symbol the note belongs to. Not user documentation.


## `loudkit/checkpoint.py`


### `module`

A loudkit checkpoint is a single ``safetensors`` file whose tensors live in two
namespaces, ``t3.*`` for the token generator and ``s3gen.*`` for everything
downstream, with a JSON manifest embedded in the file's metadata and mirrored
in a ``manifest.json`` beside it.

The manifest is not documentation, it is authority. Values that are properties
of the weights, silence-token ids, Euler step count, vocabulary bounds, the
per-module dtype map, are read from it and nowhere else, because the worst
The divergence class this format exists to prevent is defaults re-guessed by a second
implementation (see ``AlgorithmConfig``'s module docstring).

Two facts about the tensor payload that loading code must know:

* precision is **mixed per module** and recorded in ``manifest["dtype_map"]``
  by longest-prefix match. The packed dtype is the *storage* dtype; a backend
  may upcast (fp16 -> fp32 is exact) but must consult its own
  ``ExecutionConfig.precision`` for the compute dtype.
* the vocoder's weight-norm reparametrisation is already folded, the packed
  weights are plain ``weight`` tensors, bit-exactly equal to what the
  parametrised forward would have computed.

This module is deliberately torch-free: it reads numpy arrays, so a future
runtime-only backend (ONNX, CoreML) can load the same file without dragging
torch in.


### `require_decode_support`

Args:
    mode: the loop that will actually run: the resolved
        ``AlgorithmConfig.decode``, not the manifest's raw block, so an
        explicit ``algorithm=`` override is judged on what it asked for.
    target: a backend name (``torch``, ``onnx``, ``coreml``) or a torch
        device (``cpu``, ``cuda``, ``mps``, with or without an index).
        One function for both because the two callers speak different
        halves of the same vocabulary: the fetch knows backends, the
        engine builder knows devices.
    name: the checkpoint's ``manifest["name"]``, when the caller has it,
        so the message names the model the user chose.

Raises:
    ValueError: naming the model, the backend that refused it, and the two
        ways on. Without the mode's name: the reader is choosing a model,
        and ``fusion_mtp2`` is a fact about the weights they cannot act
        on. ``AlgorithmConfig.describe()`` prints it into the log, which
        is where a maintainer reads it.


### `verified_sibling`

A checkpoint is not self-contained: the tokenizer, and on the graph
backends the exported ONNX or CoreML packages, are separate files
resolved by name from the checkpoint's directory. Nothing tied them to
the weights. Swapping ``tokenizer.json`` for another valid one changes
the text ids, the speech, and potentially where EOS lands, and
``AlgorithmConfig.fingerprint()`` does not move, because the tokenizer
is not part of the algorithm config and ``TextFrontend`` carries no
config for ``Engine._assert_one_algorithm`` to compare. The result is
two different readings reporting the same identity, which is the one
thing this library promises cannot happen.

Enforced when the manifest records the digest and skipped when it does
not, because packs predating the field are still loadable, a checkpoint
that cannot state what it expects cannot have its expectation checked.
`tools/build_release.py` records the digests it ships.


### `asset`

The tokenizer and the Polish respelling lexicon are not weights, but
they decide what the weights are asked to say: a different
``tokenizer.json`` reads the same text as different ids, and a different
lexicon reads embedded English a different way. Shipped beside the file
they are two more things to keep in step, and this project has now spent
five copies of the
lexicon, a sibling bound only by a digest the shipping manifest does not
carry, and three ports that disagreed about the funnel.

Carried as a ``uint8`` tensor under ``assets.`` rather than in a new
container format, because **every port already has a safetensors
reader**. Nothing new to write in five languages, and the bytes are
covered by the same file the weights live in.

Returns ``None`` when the checkpoint predates the convention, which is
what :meth:`resolve_asset` falls back on.


## `loudkit/errors.py`


### `module`

Until now loudkit defined exactly one exception of its own and otherwise raised
bare ``ValueError``, ``RuntimeError``, ``NotImplementedError`` and
``FileNotFoundError``. That is fine inside a function and expensive at a
boundary, because the boundary has to classify: the HTTP server maps exception
*types* to status codes, and with builtins the only thing it could map was
"something raised NotImplementedError", which is true of an unsupported
language, and equally true of a backend with a stub method. One is the caller's
question and answers 400; the other is a server defect and must answer 500. The
server could not tell them apart, so every backend bug arrived at the client as
"your fault".

**Every class here inherits from the builtin it replaces.** ``except
ValueError`` still catches a :class:`WindowOverflowError`, ``except
FileNotFoundError`` still catches a :class:`VoiceNotFoundError`. Nothing written
against the bare-exception shape breaks; what is added is the ability to be *specific* -
``except loudkit.LoudkitError`` for "loudkit refused this", or a named class for
one refusal.

The hierarchy is deliberately small. A class earns its place by being something
a caller would branch on, and each one carries the values a caller would
otherwise have to parse back out of the message.

**Error codes.** Each class also names its condition as a short stable string,
``code``, the vocabulary the transports speak. An HTTP error body carries it
as ``"code"``, the SSE ``done`` event as ``"error_code"``, gRPC as
``loudkit-error-code`` trailing metadata, so the same refusal has the same name
whether it crossed a process boundary or not. The catalog is frozen: codes are
never renamed or reused, only added. Conditions no class names yet fall back to
``invalid_request`` (the caller's question) or ``server_fault`` (a defect
here); the transports add ``unauthorized`` and ``busy`` for conditions that
exist only at a boundary. :func:`error_code` is the one mapping.


### `UnsupportedLanguageError`

Raised: by :class:`~loudkit.frontend.text.GraphemeTextFrontend` for the
languages whose upstream pipeline needs model-based preprocessing this
frontend does not carry, Cangjie codes, kana conversion, diacritisation,
jamo decomposition, stress marks. Refused rather than silently skipped: a
grapheme read of Chinese is the wrong sounds in the right order, and no
error downstream would say why.

Carries:
    language: the id that was refused, lowercased.
    supported: the language ids this build *does* accept, sorted. Read from
        the tokenizer's own vocabulary rather than hardcoded, so it cannot
        drift from the tokenizer that ships.

Still a ``NotImplementedError``, so existing ``except NotImplementedError``
handlers keep working, but the server now catches *this* rather than the
builtin, which is how a genuine ``NotImplementedError`` from a half-written
backend stopped being reported to the client as a bad request.


### `VoiceNotFoundError`

Raised: by :class:`~loudkit.transports.http.VoiceLibrary` when a request names a
voice the library does not hold, and by
:func:`~loudkit.hub.resolve_voice` when a reference is neither a file nor
resolvable against a repo.

Carries:
    ref: what was asked for, a name over HTTP, a path or name in-process.
    available: the names that *were* found, when listing them is cheap
        (a local directory). Empty when it is not, resolving a name
        against a remote repo would mean a network call to answer an error,
        and an error that goes slower than the thing that failed is its own
        problem.

When ``available`` holds a close match, the message ends with ``did you mean
'<name>'?``. Done here rather than at each raise site so that every site
which can afford to list alternatives gets the suggestion automatically -
the message and the list are the same fact, and letting them be assembled
separately is how one of them ends up stale. Voice names are short,
lowercase donor or character names (``kathleen``, ``gosia``, ``thorsten``),
which is the exact shape a caller retypes wrong and a reader scans past in
a list of twenty.

Still a ``FileNotFoundError``, so a caller catching that keeps catching
this, and the CLI's "not found:" path is unchanged.


### `InvalidTokensError`

Raised: by :meth:`~loudkit.engine.Engine.synthesize` and friends when
``previous_tokens`` holds an id outside the acoustic codebook, a control
token, a negative, or something past the vocabulary. Those ids index an
embedding table, so the alternative to refusing is an index error three
stages away from the argument that caused it.

Carries:
    token: the first offending id, so a caller filtering a long sequence
        knows which entry to look at rather than which list.
    limit: the exclusive upper bound. Ids must satisfy ``0 <= id < limit``.

It earns a class rather than a bare ``ValueError`` because a boundary has to
classify it, and getting that wrong was visible: the streaming route maps
exception *types* to ``bad_request`` or ``server_fault``, everything outside
this hierarchy is a server fault by definition, and so a client sending one
bad integer was told the server had broken. That is the one verdict a client
cannot act on, and it is exactly backwards, the request is the thing to fix.
The one-shot route was already correct, which is how the two disagreed.

Still a ``ValueError``, which is what the HTTP server maps to 422 and what an
existing ``except ValueError`` around a synthesis call already catches.


### `WindowOverflowError`

Raised: by :meth:`~loudkit.engine.Engine.synthesize` with
``single_window=True``, and by
:meth:`~loudkit.engine.Engine.synthesize_tokens` for a sequence longer
than ``window.max_speech_tokens``. Silent truncation is not an option:
text would go missing while the audio still sounds fine, and only a
listener who knows the passage would notice. Plain ``synthesize`` never
raises it, it splits.

Carries:
    n_tokens: how many speech tokens were produced.
    window: how many the window holds. The overflow is the difference, and
        the message states it in seconds of speech as well as in tokens.

Still a ``ValueError``, which is what the HTTP server maps to 422 and what
every existing test expects.


## `loudkit/provenance.py`


### `module`

The EU AI Act (Article 50) requires synthetic audio to carry a
machine-readable marking, and the Commission's draft Code of Practice names
**C2PA Content Credentials** as the reference implementation, the same shape
of answer ElevenLabs, OpenAI and Adobe ship: a manifest bound to the file,
not an in-audio mark. This module writes exactly that.

What it is, stated precisely rather than claimed:

* A **claim-only** manifest in a JUMBF ``c2pa`` box, appended to the WAV after
  the data chunk (players ignore the trailing bytes; the C2PA tools read
  them). It carries the ``c2pa.actions`` assertion with
  ``digitalSourceType = trainedAlgorithmicMedia``, the vocabulary that means
  "this was generated by a model", plus the loudkit-specific claims that make
  the marking *traceable*, not just labelled: the algorithm fingerprint, the
  recipe version, the seed, the voice and the digests of the checkpoint and
  voice-profile files that spoke, the backend and execution layer that ran,
  the language, and the SHA-256 of the audio bytes it binds to.
* **Unsigned.** A full C2PA chain signs the manifest with a certificate, which
  is a deployment decision (whose cert?) and a key-management burden; a
  library cannot choose that for its caller. The claim-only manifest is the
  generator half of the standard, and a signer can wrap it later without
  changing what it says.
* **Metadata, with metadata's limits.** Re-encoding strips it, exactly as it
  strips every metadata-based marking in the field. It is the disclosure duty
  of Article 50, a labelling obligation, not tamper-proofing. There is no
  in-audio watermark: one was written and measured, its detection never
  cleared an honest null, and shipping a mark that cannot be found would be
  the worse of the two failures. Disclosure stays the caller's job.

The box layout is the standard minimal manifest store: a ``jumbf`` superbox
holding a ``c2pa`` description box holding a ``json`` box. Big-endian
length-prefixed, UUID-typed, no compression, readable with no dependencies.


## `loudkit/voice.py`


### `module`

A :class:`VoiceProfile` is the handful of tensors the two stages need in order
to speak as someone: a speaker embedding, a prompt of speech tokens, the mel of
the reference audio, and the conditioning the token generator was trained to
read. A few hundred kilobytes, no weights.

That framing is deliberate and it is what makes cloning cheap. An earlier
version baked the prompt into the graph, so every voice was a separate exported
model of several hundred megabytes; taking the prompt as an input instead turned
a voice into a file you can email. It also means voices can be enrolled once on
a fast machine and shipped, rather than re-derived on a phone, which matters
because enrollment needs a speaker encoder and a speech tokenizer that synthesis
otherwise never touches, together about 40% of the checkpoint.

Profiles are saved as ``safetensors`` with a small JSON header, so they are
inspectable, versioned, and safe to load from an untrusted source.


### `VoiceProfile`

The two stages read two *different* speaker encoders' outputs, so a profile
carries two embeddings: the token generator was trained against a 256-d
utterance-level voice-encoder vector, while the flow decoder conditions on
a 192-d CAM++ x-vector. They are not interchangeable and neither can be
derived from the other, which is why both are enrolled once and stored.

Attributes:
    name: human label. Carried for provenance and error messages only;
        nothing dispatches on it.
    speaker_embedding: ``(256,)`` speaker vector for the token generator's
        conditioning encoder.
    flow_embedding: ``(192,)`` x-vector for the mel decoder.
    prompt_tokens: speech tokens of the reference audio, the prosodic and
        timbral prompt the mel decoder continues from. Stored at natural
        length; the window recipe (``WindowConfig``) decides framing.
    prompt_mel: ``(80, frames)`` mel of the reference, conditioning the flow.
    cond_prompt_tokens: the token generator's own conditioning prompt, which
        may be a different length from ``prompt_tokens``.
    source_sample_rate: sample rate of the audio this was enrolled from.
        Provenance only, **nothing reads it**. It said "kept so a mismatch
        is detectable rather than silently resampled", which described a
        check no layer performs: `enroll` writes it and no renderer, loader
        or engine looks at it again. Recorded here rather than removed
        because the fact is worth carrying and a future check would want it;
        described honestly because a promise in a docstring is the kind of
        thing a caller builds on.
    language: **the language this voice reads in.** Not provenance: it is
        the default every `synthesize` / `stream` call without an explicit
        `language=` resolves to, so a profile stamped `en` reads Polish
        text through the English funnel. It sits beside
        `source_sample_rate`, which is honestly documented as read by
        nothing, and used to be described the same way.


## Notes moved from the runtime docstrings


## `loudkit/checkpoint.py`


### `read_manifest`

The embedded copy is authoritative, the sibling ``manifest.json`` is a
convenience for humans and can drift if someone edits it, so it is never
read here.

Raises:
    ValueError: if the file carries no manifest or declares a format this
        build does not read. Failing loudly beats loading a checkpoint
        under wrong assumptions about what its numbers mean.


### `_check_decode_version`

Checked here rather than in :class:`~loudkit.config.AlgorithmConfig`, and
checked at all, because the version number is the *portable* half of this
contract: four engines gate on it and none of them reads the ``decode``
block. A manifest that understates its version is therefore not a Python
problem, Python reads it correctly, it is a file that four correct
engines will accept and misread, and the only place to catch that is where
the file is opened.

Refused rather than repaired. Whichever number is wrong, a checkpoint whose
manifest contradicts itself is one whose provenance nobody can state, and
guessing which half the packer meant is how a build ships weights nobody
can name.


### `Checkpoint`

Tensors are pulled on demand rather than loaded wholesale because the two
stages of the engine may live in different processes or devices, and
enrollment (~40% of the payload) is not needed for synthesis at all.

Example:
    >>> ckpt = Checkpoint.open("loudr-1.safetensors")
    >>> t3 = ckpt.tensors("t3.")            # generator weights, prefix stripped
    >>> ckpt.manifest["n_cfm_timesteps"]
    2


### `file_digest`

This is the value a release's ``SHA256SUMS`` lists and the value
provenance manifests carry as ``checkpoint_sha256``, the digest that
names *which artefact* rendered a waveform. It is not
``tensor_payload_sha256``, which lives inside the file and can only
say the payload survived the download. Computed on first use and
cached: one chunked read of the file, once per opened checkpoint.


### `shapes`

No tensor data is touched, so this is cheap on a 747 MB file and, the
reason it exists, it is available *before* anything is allocated. A
manifest declares the architecture and the architecture decides how much
memory the model constructor asks the allocator for, so a manifest that
nothing checks is a 20 kB file that can demand gigabytes. These shapes
are the other half of the same checkpoint and cannot be inflated without
inflating the file, which makes them the thing to check the manifest
against.


## `loudkit/config.py`


### `module`

:class:`AlgorithmConfig` holds every value that decides what comes out and is
identical on every backend. :class:`~loudkit.execution.ExecutionConfig` holds
what decides how fast and is free to differ. The test for a new setting: if it
changed on one backend only, would the output be a different reading of the
text? Then it is algorithm. Why the split exists, and the measurements behind
each value, are in ``docs/design/algorithm-config.md``.


## `loudkit/errors.py`


### `LoudkitError`

Raised: never directly. It exists so a caller embedding the library can
write ``except loudkit.LoudkitError`` and catch the refusals loudkit means,
without also catching the ``ValueError`` that came out of numpy.

Carries: nothing of its own. The subclasses carry the diagnostics.

An error that is *not* a ``LoudkitError`` coming out of loudkit is either a
bug here or a failure in a dependency, in both cases something to report,
not something to handle.


### `NumberGrammarError`

Raised: by :mod:`loudkit.frontend.numbers` when a language has no grammar, or a
value is larger than the grammar's largest scale. Raised rather than
returning the digits: a caller who gets ``"1000000000"`` back has no way to
tell it apart from a number the grammar handled, and silently reading
digits aloud is the failure that module exists to remove.

Carries: nothing beyond its message.

Defined here rather than in :mod:`loudkit.frontend.numbers`, where it started and
where it is still exported from, only because that module imports this one:
the class has to live below the base it now inherits.
``loudkit.frontend.numbers.NumberGrammarError`` remains the same object.


### `AudioNotFoundError`

A plain `FileNotFoundError` was what this raised, and
`docs/design/embedding.md` promises that every loudkit error is also a
`LoudkitError`, so `except LoudkitError` did not catch the most ordinary
failure of the most file-dependent entry point.

Still a ``FileNotFoundError``, which is what a caller who wrote
``except FileNotFoundError`` around it already expects, and what
:class:`VoiceNotFoundError` beside it also keeps.


### `error_code`

A :class:`LoudkitError` names its own condition. Anything else that a
boundary chose to report as a refusal (a bare ``ValueError`` from a layer
that has not earned a class yet) is ``invalid_request``; an exception the
boundary did *not* choose, a stub method, a numpy failure, a bug, is
``server_fault``, decided by the caller passing it here only for errors it
classified as refusals. This function does not guess: it reads the class.


### `__reduce__`

``BaseException.__reduce__`` returns ``(type(self), self.args)``, so
unpickling calls ``cls(*args)``, and every subclass below takes its
diagnostics as *required keyword-only* arguments, which that call does
not supply. Default pickling therefore raises ``TypeError`` and masks
the original error wherever exceptions cross a process boundary, such
as errors ferried back from a ``ProcessPoolExecutor`` worker.

Rebuilt through ``__new__`` and ``BaseException.__init__`` rather than
by giving the keywords defaults, because the diagnostics are the whole
reason these classes exist and a default would make them optional at
every raise site.


## `loudkit/provenance.py`


### `_stamped_now`

C2PA wants a creation time, and this is the only value in a rendered file
that is not a function of the input. It is therefore the one byte-range in
which two identical renders can legitimately differ, and the transport
suites, which assert that a transport returns *byte for byte* what the
library returns, patch this (see ``conftest.py``): two identical renders
straddling a second boundary would otherwise differ in the trailer.


### `build_manifest`

The loudkit assertion carries the values that make the label *traceable*:
the algorithm fingerprint and the seed reproduce the exact waveform, the
audio hash binds the manifest to these bytes and not some other file, and
the checkpoint and profile digests name which weights and which voice
spoke, a fingerprint pins the algorithm, not the artefact, and a voice
*name* is a label anyone can reuse. ``backend`` and ``execution`` name the
datapath: execution never changes what is computed, but reduced precision
perturbs it within measured bands, and a manifest that names the waveform
should name what produced it. Empty strings mean "not known here", never
"does not apply".


## `loudkit/voice.py`


### `_validate_values`

Checked here rather than discovered per backend. A profile is a file
that can be copied, mailed and loaded from an untrusted source, and the
three renderers disagree about what a degenerate one means: torch's
``F.normalize`` carries an epsilon and returns finite values for a zero
vector, while ONNX and CoreML divide by the raw norm and produce 192
NaNs. One accepted profile, two behaviours, no error anywhere, the
divergence class this library exists to make impossible, arriving
through data instead of through code.


### `cond_key`

The generator's conditioning is a pure function of
``speaker_embedding`` and ``cond_prompt_tokens`` (the third slot is the
constant :data:`EMOTION_NEUTRAL`), so two profiles that agree on these
bytes get the same row. Keyed by content rather than by object
identity: profiles are frozen but freely copied, and an ``id()`` key
would silently miss on every copy. A few hundred bytes of hashing per
call, against a perceiver pass per miss.


## Notes moved from attribute docstrings and comments


### `loudkit/checkpoint.py`


#### `DECODE_FORMAT_VERSION`

The version gate is the *only* thing standing between a fusion checkpoint and
four engines that would misread it: Rust, Go, TypeScript and Swift refuse
version 2 by number and none of them parses the ``decode`` block at all. So a
manifest saying ``format_version 1`` beside ``decode.mode = "fusion_mtp2"``
loads everywhere and speaks fluent nonsense in four of the five, and until
this table existed, the coupling was asserted in a design note and enforced by
the packer remembering to write a 2.

A mode absent from this table needs no minimum, which is how ``single`` (and
an absent block, its synonym) keeps loading out of every version-1 file.


### `loudkit/voice.py`


#### `EMOTION_NEUTRAL`

The checkpoint architecture reserves one of its 34 conditioning slots for an
emotion scalar (``t3.cond_enc.emotion_adv_fc``). On these weights the axis is
dead, distillation collapsed the response, so the slot is not a control and
is not part of the profile format. It still has to be fed *something*, and it
has to be the value the model was distilled with and every profile ever
written carried: 0.5. Every renderer in every port uses this constant.


#### `MAX_VOICE_BYTES`

Every voice the kit ships weighs about 165 KB, and the format has no field that
grows with anything a caller controls, so fifty times the real size is room for
a format change, not for a payload. Without a bound, ``prompt_mel`` declared as
``(80, 100_000_000)`` is 64 GB of allocation the moment it is read, from a file
the server loads by name on an unauthenticated request. Checked on the file
rather than per tensor because the file bounds every tensor in it at once, and
does so before anything is materialised.


#### `KNOWN_ENROLMENTS`

A profile naming anything else is refused at load. That is the point of the
field: a build without the strategy that produced a voice must say so, rather
than apply its own and hand back a different voice under the same name.

Two strategies exist. `first-10s` is the prompt as the first ten seconds of the
clip, which every port's enroller makes. `first-10s-pause` is the clip cut at its
last pause before ten seconds and padded with 0.4 s of silence, which `loudkit
clone` makes by default and `enroll(end_in_silence=True)` makes on request. The
ports write only the first and read both: a profile is reused, never re-cut.


#### `enrolment`

A profile is an artefact, and this says how it was made. Enrolment picks a
window of the reference clip before any of the tested transform runs, so two
strategies produce two different voices from one recording, with nothing in
the tensors to tell them apart.

Recorded rather than assumed, for the reason `TextConfig.recipe` exists: an
implementation that does not have the strategy named here must refuse the
profile instead of silently applying its own. Five implementations agreeing
on the transform is worth nothing if they disagree about which ten seconds
to feed it.


#### `_validate_values` bounds every id

A prompt token is refused above the speech codebook and a conditioning token
above the full speech vocabulary, both ceilings taken from the shipped
algorithm rather than repeated here. `load()` promises a profile is safe to
open from an untrusted source, and a bound the renderer relies on has to be
checked where that promise is made: an oversized id indexes past the end of
`nn.Embedding`, or on the ONNX path reads whatever follows the table.
