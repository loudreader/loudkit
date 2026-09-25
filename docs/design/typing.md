# Typing: what the annotations guarantee, and where they stop

`loudkit` passes `mypy --strict` (plus `warn_unreachable` and
`disallow_any_generics`) over `python/loudkit`, and ships a `py.typed` marker,
so downstream users get these annotations checked in their own code (PEP 561).
No `# type: ignore` is bare: every suppression carries its error code and a
reason. A `cast` appears only where the runtime type is known.

## What the annotations guarantee

- The five protocols in `contracts.py` (`TextFrontend`, `VoiceEnroller`,
  `TokenGenerator`, `MelDecoder`, `Vocoder`) and the `Sampler` protocol are
  fully typed, and every implementation is checked against them. mypy rejects
  code that reaches through a protocol to an implementation's internals (see
  the CoreML affine weights below).
- Stage values have value types: `SpeechTokens = Sequence[int]`,
  `Mel = NDArray[np.float32]`, `Waveform = NDArray[np.float32]`. What crosses
  between stages is data, and the annotations name the data.
- `AlgorithmConfig` and `ExecutionConfig` fields use Literals where the value
  set is closed (`Precision`, guidance mode, device), so a typo in a mode name
  is a type error.
- A checkpoint manifest is parsed JSON (`Mapping[str, object]`). Small helpers
  narrow its values (`_block` and `_number` in `manifest.py`, `_cfg_int` and
  `_cfg_float` in `models/generator.py`). They check with `isinstance` and
  raise a `ValueError` or `TypeError` that names the key, so a manifest value
  of the wrong type fails at load time.

## The torch boundary

torch's stubs type `nn.Module.__call__` as returning `Any`, since a module may
return anything. Without a policy, every submodule call inside a `forward`
would propagate `Any`.

The policy, applied in `models/{generator,flow,vocoder,enroll}.py` and stated
once per module header:

- Where the callee's own `forward` is typed and returns a `Tensor`, the
  returning expression is wrapped in `cast(Tensor, ...)`. The cast restates
  the callee's annotated return type.
- `Tensor.numpy()` is also untyped. The sites that hand a numpy array to the
  sampler or to enrollment cast it to the concrete `NDArray[np.float32]` that
  `.float()` or `.astype(np.float32)` guarantees.
- Registered buffers are `Tensor | Module` through `Module.__getattr__`. Each
  buffer used in code carries a class-level `Tensor` annotation
  (`inv_freq`, `stft_window`, `_mel_filters`, `window`).
- Indexing an `nn.ModuleList` is typed as bare `Module`. The flow estimator's
  nested stage lists (kept nested so parameter names match the checkpoint
  1:1) are unpacked through `cast(nn.ModuleList, ...)` at the use site. The
  cast restates what `__init__` constructed there.

The type system does not cover tensor shapes, dtypes or devices. Three other
mechanisms check parts of them: the fingerprint check at engine construction,
the conformance suite, and `strict=True` state-dict loading. None of them
checks every tensor statically.

## Untyped third parties

`pyproject.toml` carries two `ignore_missing_imports` overrides. The first is
`librosa` alone, which also sets `follow_imports = "skip"` because mypy cannot
parse librosa 1.0's own type information at `python_version = 3.10`. The
second names `torch`, `torchaudio`, `safetensors`*, `soundfile`, `librosa`,
`onnxruntime`, `coremltools`, `mcp`, `grpc`, `grpc_tools`, `tokenizers` and
`huggingface_hub`. Values from these packages are `Any`, and the call site
narrows them before they spread:

- **coremltools** enters through one `cast`, in
  `backends/coreml_backend._PinnedInputs.predict`, against a local
  `_MLModelLike` protocol (`predict(Mapping[str, Any]) -> dict[str, Any]`)
  that states the one method this backend uses. `_load_model` wraps the raw
  `MLModel` in `_PinnedInputs` and returns that protocol, so no code
  downstream holds the untyped object.
- **torchaudio** and **librosa** are used only inside enrollment, where the
  results immediately become typed tensors or arrays.
- **fastapi**, **uvicorn** and **pydantic** are not in the override. They ship
  type information, so `transports/http.py` is checked against their APIs.
  They are in the `dev` extra because the type gate needs them installed.

\* safetensors ships types in current releases. The override entry stays
because older versions inside the supported range (`>=0.4`) do not. No
`no-untyped-call` ignores are used for safetensors, because they would be
unused on a typed version.

## Suppressions that remain, and why

The package contains **thirteen** `type: ignore` comments, counted by
`tests/test_hygiene.py`, which fails if this number goes stale:

| site | code | reason |
|---|---|---|
| `config.AlgorithmConfig.with_` | `arg-type` | `dataclasses.replace` wants each field's own type; a `**kwargs: object` passthrough cannot express that. The dataclass re-validates in `__post_init__`. |
| `execution.ExecutionConfig.resolved` | `arg-type` | the same `replace` gap on the execution side, where `resolved` builds its `changes` map as `dict[str, object]`. |
| `cli._SafetensorError` | `no-redef` | the import fallback defines a class of the same name when `safetensors` is missing, so `--help` still parses. |
| `backends._default_execution` (device, x2) | `arg-type` | callers pass ordinal strings such as `"cuda:0"`, which the `Device` Literal does not cover; the registry lookup has already checked the string. |
| `backends._default_execution` (generator_device) | `arg-type` | the same `Device` gap, on the second field that carries a device string. |
| `models/enroll.py` (`librosa.filters`, `librosa.effects`, x2) | `attr-defined` | the override above skips librosa's stubs, so mypy knows the package but not that it re-exports its submodules. The `import librosa.filters` a line earlier makes the attribute exist at runtime. |
| `mcp.build_server` (`@server.tool`, x3) | `untyped-decorator` | FastMCP's `@server.tool(...)` ships without a typed decorator signature, so mypy cannot see the wrapped function's type through it. |
| `grpc` (generated stub base) | `misc,name-defined` | the servicer base class comes from generated code that mypy does not load. |
| `http` (`HTTPException.loudkit_code`) | `attr-defined` | `_refuse` sets `loudkit_code` on Starlette's `HTTPException`, which does not declare that attribute. |

Each suppression names the error code mypy reports for that line, and
`warn_unused_ignores` fails on one that suppresses nothing. A suppression with
the wrong code hides no current error and can hide a future one.

## Checked in code, not in annotations

- `backends/coreml_backend.build_coreml_engine` reads the speaker affine
  weights from the checkpoint (`s3gen.flow.spk_embed_affine_layer.*`), not
  through the `MelDecoder` protocol, which does not expose module internals.
  They are the same tensors, read from the same file.
- `TorchVocoder.half()` is annotated `-> NoReturn`: it always raises, because
  the vocoder is fp32 only.

## Known limits

- Tensor shape, dtype and device correctness is outside the type system (see
  above). `-> Tensor` does not mean "the right tensor".
- Inside `forward` bodies, intermediate locals fed by submodule calls stay
  `Any` until the cast at the boundary, and mypy checks those lines loosely.
  The casts sit at the protocol boundaries, where the guarantees apply.
- The `Device` Literal names the five backends but does not model device
  ordinals such as `"cuda:0"` (the `_default_execution` suppressions above).
- `tools/mypy-tests.ini` checks fourteen test modules with `tools/mypy.ini`'s
  flags: twelve suites and the `assets.py` and `conftest.py` helpers they use.
  The suites cover what a user downloads and what the five implementations
  must agree about. A type error in one of them can hide a test that passes
  while asserting nothing. The list is explicit and grows one module at a
  time. The rest of `tests/` (the engine, model, server, MCP, CLI and backend
  suites) is not type-checked, because covering it means annotating it first.
- `tests/test_hygiene.py` fails on any `# type: ignore` comment in `tests/`,
  inside the mypy test profile or outside it. Outside a profile no checker
  reads the comment; inside one, the error it names is fixed instead.
- `tools/` and `research/` use their own profile, `tools/mypy.ini`. It allows
  untyped defs, untyped globals and untyped calls, because these are scripts
  with argparse mains and tracer-fed `forward` methods. It checks function
  bodies, follows imports, keeps optionals strict and reports unused
  suppressions. `tools/release/`, which builds what users download, is held to
  the flags `--strict` implies.
