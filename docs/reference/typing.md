# Typing: what the annotations promise, and where they stop

`loudkit` passes `mypy --strict` (plus `warn_unreachable` and
`disallow_any_generics`) over `python/loudkit`, and ships a `py.typed` marker, so
downstream users get these annotations checked in their own code (PEP 561).
This document explains the guarantees, the boundaries with untyped
dependencies are, and what is suppressed and why. The standard throughout: no
bare `# type: ignore`. Every suppression carries its error code and a reason,
and a `cast` appears only where we can state what the runtime type actually is.

## What the annotations guarantee

- **The component seams are real.** The five protocols in `contracts.py`
  (`TextFrontend`, `VoiceEnroller`, `TokenGenerator`, `MelDecoder`, `Vocoder`)
  and the `Sampler` protocol are fully typed, and every implementation is
  checked against them. Code can no longer reach through a protocol to an
  implementation's internals without mypy objecting. That check caught a real
  defect (below).
- **Value boundaries carry value types.** `SpeechTokens = Sequence[int]`,
  `Mel = NDArray[np.float32]`, `Waveform = NDArray[np.float32]`. What crosses
  between stages is data, and the annotations say exactly which data.
- **Configs are precise.** `AlgorithmConfig` / `ExecutionConfig` fields use
  Literals where the value set is closed (`Precision`, guidance mode, device),
  so a typo in a mode name is a type error, not a silent default.
- **Manifest reads are checked, not assumed.** A checkpoint manifest is
  parsed JSON (`Mapping[str, object]`). Small helpers narrow numeric values
  (`geti`/`getf` in `config.from_manifest`, `_cfg_int` / `_cfg_float` in
  `models/generator.py`). They `isinstance`-check and raise a
  `ValueError`/`TypeError` naming the offending key. A manifest that lies
  fails loudly at load time.

## The torch boundary

torch's stubs type `nn.Module.__call__` as returning `Any`, since a module may
return anything. Every submodule call inside a `forward` would therefore
silently propagate `Any`.

Policy, applied uniformly in `models/{generator,flow,vocoder,enroll}.py` and
stated once per module header:

- Where the callee's own `forward` is typed and provably returns a `Tensor`,
  the returning expression is wrapped in `cast(Tensor, ...)`. The cast asserts
  a contract that exists one class away.
- `Tensor.numpy()` is also untyped. The two sites converting to numpy for the
  sampler/enrollment say so in a comment and cast to the concrete
  `NDArray[np.float32]` that `.float()` / `.astype(np.float32)` guarantees.
- Registered buffers are `Tensor | Module` through `Module.__getattr__`. Each
  buffer used in code carries a class-level `Tensor` annotation
  (`inv_freq`, `stft_window`, `_mel_filters`, `window`).
- Indexing an `nn.ModuleList` is typed as bare `Module`. The flow estimator's
  nested stage lists (kept nested so parameter names match the checkpoint
  1:1) are unpacked through `cast(nn.ModuleList, ...)` at the use site. The
  cast restates what `__init__` constructed there.

What this does **not** guarantee: shapes, dtypes and devices of tensors are
not in the type system. Three other mechanisms enforce those: the fingerprint
check at engine construction, the conformance suite, and load-time
`strict=True` state-dict loading.

## Untyped third parties

`pyproject.toml` carries one `ignore_missing_imports` override, reserved for
packages that genuinely ship no stubs / `py.typed` (verified per package):
`torch`, `torchaudio`, `safetensors`*, `soundfile`, `librosa`, `onnxruntime`,
`coremltools`. Values arriving from them are `Any`, and the call site pins them
down rather than letting them spread:

- **coremltools** enters through exactly one `cast` in
  `backends/coreml_backend._load_model`, against a local `_MLModelLike`
  protocol (`predict(Mapping[str, Any]) -> dict[str, Any]`) that states the
  one method this backend uses. Everything downstream of the cast is checked
  against that protocol.
- **torchaudio / librosa** are used only inside enrollment, where the
  results immediately become typed tensors/arrays.
- **fastapi / uvicorn / pydantic are *not* in the override.** They ship real
  type information, so `server.py` is checked against their actual APIs. They
  are in the `dev` extra because the type gate cannot run without them
  installed.

\* safetensors ships types nowadays. The override entry is kept because older
versions inside the supported range (`>=0.4`) did not. `no-untyped-call`
ignores that assumed an untyped safetensors would be stale, and are not used.

## Suppressions that remain, and why

The package contains **eleven** `type: ignore` comments:

| site | code | reason |
|---|---|---|
| `config.AlgorithmConfig.with_` | `arg-type` | `dataclasses.replace` wants each field's own type; a `**kwargs: object` passthrough cannot express that. The dataclass re-validates in `__post_init__`. |
| `config.AlgorithmConfig.from_manifest` (guidance) | `arg-type` | the value is narrowed by an explicit membership check two lines above; mypy cannot connect a `str` to the Literal through that check. |
| `config.ExecutionOverrides` merge | `arg-type` | `dataclasses.replace` again, on the overrides a command line names. |
| `backends._default_execution` (device, x2) | `arg-type` | callers legitimately pass `"cuda:0"`, which the `Device` Literal does not cover; the registry lookup has already vetted the string. This is a real modelling gap in `Device`, owned by the config work; the ignore documents it rather than hiding it. |
| `backends._default_execution` (generator_device) | `arg-type` | same modelling gap as `device` above, on the second field that carries a `Device`-shaped string. |
| `mcp.build_server` (`@server.tool`, x3) | `untyped-decorator` | FastMCP's `@server.tool(...)` ships without a typed decorator signature; mypy can't see through it to the wrapped function's type. Third-party gap, not ours. |
| `grpc` (generated stub base) | `misc,name-defined` | the servicer base class comes from generated code that mypy does not load. |
| `http` (ASGI attribute) | `attr-defined` | an attribute Starlette sets at runtime and does not declare. |

Everything else that looked like a suppression was worse than nothing and is
gone: ~30 ignores whose error code did not match what mypy actually reports
(`arg-type` vs `call-overload`, `union-attr` vs `attr-defined`,
`no-untyped-def` on an override). A mis-coded ignore suppresses no current
error while standing ready to hide a future real one.

## Defects the type checker surfaced (fixed in code, not annotation)

- `engine.py` used `np.concatenate` and `Iterator` without importing either.
  `synthesize_long` raised `NameError` on any text long enough to split.
- `backends/coreml_backend.build_coreml_engine` fetched
  `spk_embed_affine_layer` through the `MelDecoder` *protocol*, which does
  not (and should not) expose module internals. It only worked because the
  torch implementation happens to have that attribute. The affine weights now
  come from the checkpoint (`s3gen.flow.spk_embed_affine_layer.*`): same
  tensors, same one-file provenance, no dependency on implementation guts.
- `TorchVocoder.half()` is annotated `-> NoReturn`: it always raises, because
  the vocoder is fp32-only. The annotation is the guard's documentation now.

## Known limits

- Tensor shape/dtype/device correctness is outside the type system (see
  above). Do not read `-> Tensor` as "the right tensor".
- Inside `forward` bodies, intermediate locals fed by submodule calls are
  still `Any` until the cast at the boundary, and mypy checks those lines
  loosely. Casting every intermediate would roughly double the noise for no
  additional safety at the seams, which is where the guarantees matter.
- The `Device` Literal names the five backends but does not model device
  ordinals such as `"cuda:0"` (the `_default_execution` suppressions above).
- `tests/` are not type-checked (`files = ["python/loudkit"]`). The gate covers
  what ships.
