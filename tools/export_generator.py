"""Generator graph adapters and conversion shared by the two exporters."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn

GENERATOR_STAGES = {
    "single": ("cond", "prefill", "step"),
    "fusion_mtp2": ("cond", "prefill", "pair_step", "head2"),
}

PROVENANCE = "export.json"
"""Per-artifact source digests expose mixed sets after partial renderer exports."""

_RECORD_BLOCK = {"onnx": "graphs", "coreml": "packages"}
"""What each record format calls its map from artifact name to entry."""


def export_toolchain() -> dict[str, str]:
    """Record installed conversion tools per artifact, including partial exports."""
    import platform
    from contextlib import suppress
    from importlib.metadata import PackageNotFoundError, version

    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("torch", "numpy", "onnx", "onnxruntime", "coremltools"):
        with suppress(PackageNotFoundError):
            versions[package] = version(package)
    return versions


def generator_filename(stage: str, backend: str) -> str:
    return f"t3_{stage}.{'onnx' if backend == 'onnx' else 'mlpackage'}"


def require_decode(decode: str) -> None:
    if decode not in GENERATOR_STAGES:
        raise SystemExit(f"Unsupported decode mode {decode!r}; no export graph set is defined.")


def exporter_parser(description: str | None, backend: str) -> argparse.ArgumentParser:
    """The command line both synthesis exporters take, spelled once.

    The CoreML exporter adds its estimator flags to what comes back; a flag
    either backend should grow belongs here, where it cannot reach one and
    miss the other.
    """
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=None, help=f"output dir (default: <ckpt dir>/{backend})")
    ap.add_argument("--stages", default=None)
    return ap


def selected_stages(args: argparse.Namespace, algo) -> set[str]:
    """What `--stages` named, or every stage this decode mode exports.

    Validated here, before the caller has made a directory, so a stage name
    that does not belong costs nothing and leaves nothing behind.
    """
    from export_renderer import RENDERER_STAGES

    stages = (
        {name.strip() for name in args.stages.split(",")}
        if args.stages is not None
        else set(GENERATOR_STAGES[algo.decode]) | set(RENDERER_STAGES)
    )
    validate_stages(algo.decode, stages)
    return stages


def write_provenance(
    out_dir: Path,
    stages: set[str],
    ckpt,
    algo,
    *,
    backend: str,
    stage_files: dict[str, str],
    extra: dict[str, object] | None = None,
    per_stage: Callable[[str], dict[str, object]] | None = None,
) -> None:
    """Record what each artifact this run wrote was exported from.

    Merged into whatever is already there rather than replacing it, because a
    partial export is a legitimate thing to do, say re-exporting one stage after
    an opset or converter bump. Merging is what makes a *mixed* set visible: the
    stale artifact keeps its old entry and the check on the load side sees two
    answers.

    Written atomically. A half-written record is worse than none: none says
    "this set predates the record", and half says something false.
    """
    import json
    import os

    from distill_meta import file_sha256
    from export_renderer import RENDERER_STAGES

    entry: dict[str, object] = {
        "toolchain": export_toolchain(),
        "checkpoint": ckpt.path.name,
        "checkpoint_sha256": file_sha256(ckpt.path),
        "algorithm_fingerprint": algo.fingerprint(),
        "euler_steps": algo.euler_steps,
        "decode": algo.decode,
        "graph_set": [
            generator_filename(stage, backend) for stage in GENERATOR_STAGES[algo.decode]
        ]
        + [stage_files[stage] for stage in RENDERER_STAGES],
        **(extra or {}),
    }
    path = out_dir / PROVENANCE
    block = _RECORD_BLOCK[backend]
    written: dict[str, dict] = {}
    record: dict[str, object] = {"format": f"loudkit-{backend}-export", block: written}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing.get(block), dict):
                written.update(existing[block])
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass  # unreadable is the same as absent; this run rewrites it
    for stage in sorted(stages):
        written[stage_files[stage]] = entry | (per_stage(stage) if per_stage else {})

    tmp = path.with_suffix(".json.partial")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    print(f"[provenance] {path.name}: {sorted(written)}")


def install_ct_shims() -> None:
    """Torch ops the coremltools frontend lacks; register before convert.

    ``view_as`` (encoder rel-pos attention) and ``broadcast_tensors`` are the
    two the production export needed. Inlined so this script has no import
    outside the repo. Idempotent.
    """
    from coremltools.converters.mil import Builder as mb  # noqa: N813
    from coremltools.converters.mil.frontend.torch.ops import _get_inputs
    from coremltools.converters.mil.frontend.torch.torch_op_registry import (
        _TORCH_OPS_REGISTRY,
        register_torch_op,
    )

    registered = getattr(_TORCH_OPS_REGISTRY, "name_to_func_mapping", {})

    @register_torch_op(override=True)
    def triu(context, node):
        x, diagonal = _get_inputs(context, node, expected=2)
        shape = mb.shape(x=x)
        rows = mb.gather(x=shape, indices=0, axis=0)
        cols = mb.gather(x=shape, indices=1, axis=0)
        row = mb.expand_dims(x=mb.range_1d(start=0, end=rows, step=1), axes=[1])
        col = mb.expand_dims(x=mb.range_1d(start=0, end=cols, step=1), axes=[0])
        mask = mb.greater_equal(x=mb.sub(x=col, y=row), y=diagonal)
        context.add(mb.select(cond=mask, a=x, b=0.0, name=node.name))

    if "view_as" not in registered:

        @register_torch_op
        def view_as(context, node):
            x, ref = _get_inputs(context, node, expected=2)
            context.add(mb.reshape(x=x, shape=mb.shape(x=ref), name=node.name))

    if "broadcast_tensors" not in registered:

        @register_torch_op
        def broadcast_tensors(context, node):
            tensors = _get_inputs(context, node, expected=1)[0]
            ins = list(tensors) if isinstance(tensors, (list, tuple)) else [tensors]
            target = np.broadcast_shapes(*[tuple(int(d) for d in t.shape) for t in ins])
            outs = [
                mb.broadcast_to(x=t, shape=list(target), name=f"{node.name}_{i}")
                for i, t in enumerate(ins)
            ]
            context.add(outs, torch_name=node.outputs[0])


class Cond(nn.Module):
    """speaker_emb + cond_prompt_tokens + emotion -> the 34-slot conditioning row."""

    def __init__(self, gen) -> None:
        super().__init__()
        self.speech_emb = gen.speech_emb
        self.speech_pos_emb = gen.speech_pos_emb
        self.cond_enc = gen.cond_enc

    def forward(
        self, speaker_emb: torch.Tensor, prompt_tokens: torch.Tensor, emotion: torch.Tensor
    ) -> torch.Tensor:
        prompt_emb = self.speech_emb(prompt_tokens) + self.speech_pos_emb.range(
            prompt_tokens.shape[1], speaker_emb.device
        )
        return self.cond_enc(speaker_emb, prompt_emb, emotion)


class Step(nn.Module):
    """One decode step against the past cache: embeds + position + past -> logits + present."""

    def __init__(self, gen, n_layers: int) -> None:
        super().__init__()
        self.tfmr = gen.tfmr
        self.head = gen.speech_head
        self.n = n_layers

    def forward(
        self, embeds: torch.Tensor, position: torch.Tensor, *past: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        cache = [(past[2 * i], past[2 * i + 1]) for i in range(self.n)]
        hidden, new_cache = self.tfmr(embeds, position, cache, attention="eager")
        logits = self.head(hidden[:, -1])
        flat: list[torch.Tensor] = []
        for k, v in new_cache:
            flat += [k, v]
        return (logits, *flat)


class Prefill(nn.Module):
    def __init__(self, gen):
        super().__init__()
        self.tfmr, self.head = gen.tfmr, gen.speech_head
        self.fused = gen.config.decode == "fusion_mtp2"

    def forward(self, embeds, positions):
        hidden, cache = self.tfmr(embeds, positions, None, attention="eager")
        outputs = (self.head(hidden), hidden[:, -1]) if self.fused else (self.head(hidden),)
        return (*outputs, *(x for pair in cache for x in pair))


class PairStep(nn.Module):
    def __init__(self, gen):
        super().__init__()
        self.tfmr, self.head = gen.tfmr, gen.speech_head
        self.emb, self.pos, self.fuse = gen.speech_emb, gen.speech_pos_emb.emb, gen.fuse

    def forward(self, pair_ids, speech_position, position, *past):
        pair = self.emb(pair_ids)
        first, second = pair[:, :1], pair[:, 1:]
        slot = 0.5 * (first + second) + self.fuse(torch.cat((first, second), dim=-1))
        slot = slot + self.pos(speech_position)[None]
        cache = list(zip(past[::2], past[1::2], strict=True))
        hidden, cache = self.tfmr(slot, position, cache, attention="eager")
        return (self.head(hidden[:, -1]), hidden[:, -1], *(x for pair in cache for x in pair))


class Head2(nn.Module):
    def __init__(self, gen):
        super().__init__()
        self.emb, self.head = gen.speech_emb, gen.head2

    def forward(self, hidden, first_id):
        return self.head(torch.cat((hidden, self.emb(first_id)), dim=-1))


def _lower_coreml_decode_attention(program):
    """Avoid CoreML CPU's inaccurate vector-matrix kernel at KV length 113.

    Express the single-query attention value product as a weighted sum.
    Prefill matrix products and the model's mathematical operation stay unchanged.
    """
    from coremltools.converters.mil import Builder
    from coremltools.converters.mil.mil.scope import ScopeInfo

    for function in program.functions.values():
        for op in list(function.operations):
            if (
                op.op_type != "matmul"
                or op.x.rank != 4
                or op.x.shape[-2] != 1
                or op.transpose_x.val
                or op.transpose_y.val
            ):
                continue
            with (
                function,
                Builder.scope(
                    *(ScopeInfo(source=source, data=data) for source, data in op.scopes.items())
                ),
            ):
                weights = Builder.expand_dims(x=op.x, axes=[-1], before_op=op)
                values = Builder.expand_dims(x=op.y, axes=[-3], before_op=op)
                products = Builder.mul(x=weights, y=values, before_op=op)
                result = Builder.reduce_sum(
                    x=products, axes=[-2], keep_dims=False, before_op=op
                )
            function.replace_uses_of_var_after_op(op, op.outputs[0], result)
            function.remove_ops([op])


def _convert(mod, example, inputs, outputs, dimensions, path, backend):
    """Export one graph and check every output at an untraced length."""
    if backend == "onnx":
        import onnxruntime as ort

        torch.onnx.export(
            mod,
            example,
            str(path),
            input_names=inputs,
            output_names=outputs,
            dynamic_axes=dimensions,
            opset_version=17,
            dynamo=False,
        )
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

        def run(values):
            return session.run(outputs, dict(zip(inputs, values, strict=True)))
    else:
        import coremltools as ct

        install_ct_shims()
        traced = torch.jit.trace(mod, example, strict=False)
        ranges = {}
        specs = []
        for name, value in zip(inputs, example, strict=True):
            shape = list(value.shape)
            for axis, symbol in dimensions.get(name, {}).items():
                if symbol not in ranges:
                    ranges[symbol] = ct.RangeDim(
                        lower_bound=1, upper_bound=4096, default=shape[axis]
                    )
                shape[axis] = ranges[symbol]
            specs.append(
                ct.TensorType(
                    name=name,
                    shape=tuple(shape),
                    dtype=np.int32 if value.dtype == torch.int64 else np.float32,
                )
            )
        program = ct.convert(
            traced,
            convert_to="milinternal",
            inputs=specs,
            outputs=[ct.TensorType(name=n) for n in outputs],
            minimum_deployment_target=ct.target.iOS17,
            compute_precision=ct.precision.FLOAT32,
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )
        _lower_coreml_decode_attention(program)
        model = ct.convert(
            program,
            minimum_deployment_target=ct.target.iOS17,
            compute_precision=ct.precision.FLOAT32,
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )
        model.save(str(path))

        def run(values):
            feed = {
                n: v.astype(np.int32) if v.dtype == np.int64 else v
                for n, v in zip(inputs, values, strict=True)
            }
            result = model.predict(feed)
            return [result[n] for n in outputs]

    # Change each symbolic input length together, so trace-time shape constants cannot pass.
    alternate = []
    for name, value in zip(inputs, example, strict=True):
        shape = list(value.shape)
        for axis in dimensions.get(name, {}):
            shape[axis] += 3
        if value.dtype == torch.int64:
            new = (
                torch.arange(shape[0])
                if value.ndim == 1 and name == "positions"
                else torch.zeros(shape, dtype=torch.int64)
            )
        else:
            new = torch.randn(shape)
        alternate.append(new)
    for values in (example, tuple(alternate)):
        with torch.no_grad():
            reference = mod(*values)
        if isinstance(reference, torch.Tensor):
            reference = (reference,)
        got = run([v.numpy() for v in values])
        for name, actual, expected in zip(outputs, got, reference, strict=True):
            np.testing.assert_allclose(
                actual, expected.numpy(), atol=1e-3, rtol=1e-5, err_msg=f"{path.name}: {name}"
            )
    return run


def validate_stages(decode: str, stages: set[str]) -> None:
    require_decode(decode)
    generator_stages = set(GENERATOR_STAGES[decode])
    invalid = stages - generator_stages - {"encoder", "estimator", "vocoder"}
    if invalid:
        raise SystemExit(f"Stages {sorted(invalid)} do not belong to decode {decode!r}")
    selected = stages & generator_stages
    if selected and selected != generator_stages:
        raise SystemExit("Generator stages must be exported together to run the token gate.")


def export_generator(gen, out_dir: Path, stages: set[str], backend: str):
    validate_stages(gen.config.decode, stages)
    n = gen.tfmr.n_layers
    h = gen.speech_emb.embedding_dim
    a = gen.tfmr.layers[0].self_attn
    cache = tuple(torch.randn(1, a.n_kv_heads, 8, a.head_dim) for _ in range(2 * n))
    past_names = [f"past_{kind}_{i}" for i in range(n) for kind in ("k", "v")]
    kv_names = [f"kv_{kind}_{i}" for i in range(n) for kind in ("k", "v")]
    present_names = [f"present_{kind}_{i}" for i in range(n) for kind in ("k", "v")]
    fused = gen.config.decode == "fusion_mtp2"
    specs = {
        "cond": (
            Cond(gen),
            (torch.randn(1, 256), torch.zeros(1, 64, dtype=torch.int64), torch.tensor([[0.5]])),
            ["speaker_emb", "prompt_tokens", "emotion"],
            ["t3_cond_out"],
            {"prompt_tokens": {1: "prompt"}},
        ),
        "prefill": (
            Prefill(gen),
            (torch.randn(1, 64, h), torch.arange(64)),
            ["embeds", "positions"],
            ["logits"] + (["hidden"] if fused else []) + kv_names,
            {
                "embeds": {1: "seq"},
                "positions": {0: "seq"},
                "logits": {1: "seq"},
                **{k: {2: "seq"} for k in kv_names},
            },
        ),
    }
    if fused:
        specs["pair_step"] = (
            PairStep(gen),
            (torch.tensor([[1, 2]]), torch.tensor([1]), torch.tensor([8]), *cache),
            ["pair_ids", "speech_position", "position"] + past_names,
            ["logits", "hidden"] + present_names,
            {
                **{k: {2: "past"} for k in past_names},
                **{k: {2: "present"} for k in present_names},
            },
        )
        specs["head2"] = (
            Head2(gen),
            (torch.randn(1, h), torch.tensor([1])),
            ["hidden", "first_id"],
            ["logits"],
            {},
        )
    else:
        specs["step"] = (
            Step(gen, n),
            (torch.randn(1, 1, h), torch.tensor([8]), *cache),
            ["embeds", "position"] + past_names,
            ["logits"] + present_names,
            {
                **{k: {2: "past"} for k in past_names},
                **{k: {2: "present"} for k in present_names},
            },
        )
    runs = {}
    for stage in GENERATOR_STAGES[gen.config.decode]:
        if stage in stages:
            print(f"[{stage}]", flush=True)
            mod, example, inputs, outputs, dims = specs[stage]
            path = out_dir / generator_filename(stage, backend)
            runs[stage] = _convert(mod.eval(), example, inputs, outputs, dims, path, backend)
    return runs


def gate_generator(gen, runs, fixture_dir: Path):  # noqa: PLR0912, PLR0915 - reference decode gate
    """Require fp32 teacher-forced logits and exact freely sampled tokens."""
    import json

    from loudkit.frontend.text import GraphemeTextFrontend
    from loudkit.models.windowing import eos_floor
    from loudkit.sampler import LRSamplerV1
    from loudkit.voice import VoiceProfile

    if set(runs) != set(GENERATOR_STAGES[gen.config.decode]):
        raise SystemExit("Generator stages must be exported together to run the token gate.")
    records = json.loads((fixture_dir / "vectors.json").read_text())["end_to_end"]
    frontend = GraphemeTextFrontend(fixture_dir / "tokenizer.json")
    cfg = gen.config
    fused = cfg.decode == "fusion_mtp2"
    for rec in records:
        voice = VoiceProfile.load(fixture_dir / rec["voice"])
        text = frontend.encode(rec["text"], rec["language"])
        for prefix in ([], [int(t) for t in rec["tokens"][:8]]):
            floor = eos_floor(len(text), cfg)

            def sampler(rec=rec, floor=floor):
                return LRSamplerV1(
                    cfg.sampling,
                    seed=rec["seed"],
                    stop_token=cfg.stop_speech_token,
                    eos_floor=floor,
                )

            with torch.no_grad():
                want = list(gen.generate(text, voice, sampler=sampler(), prefix=prefix))
                embeds = gen.prefill_embeds(text, voice)
                if prefix:
                    if fused:
                        slots = [
                            gen._pair_slot_embed(prefix[i], prefix[i + 1], i // 2 + 1)
                            for i in range(0, len(prefix), 2)
                        ]
                    else:
                        slots = [
                            gen._speech_token_embed(t, i + 1) for i, t in enumerate(prefix)
                        ]
                    embeds = torch.cat((embeds, *slots), dim=1)
                base = embeds.numpy()
            # Replace the torch conditioning row with the exported conditioning.
            cond = runs["cond"](
                [
                    np.asarray(voice.speaker_embedding, dtype=np.float32)[None],
                    np.asarray(voice.cond_prompt_tokens, dtype=np.int64)[None],
                    np.array([[0.5]], dtype=np.float32),
                ]
            )[0]
            base[:, : cond.shape[1]] = cond
            outputs = runs["prefill"]([base, np.arange(base.shape[1], dtype=np.int64)])
            logits = outputs[0][0, -1]
            hidden, kv = (outputs[1], outputs[2:]) if fused else (None, outputs[1:])
            seen = np.zeros(gen.SPEECH_VOCAB, dtype=bool)
            seen[prefix] = True
            draw = sampler()
            got: list[int] = []
            rows = []
            while len(got) < cfg.sampling.max_new_tokens:
                rows.append(logits.copy())
                masked = logits.copy()
                if len(got) < floor:
                    masked[cfg.stop_speech_token] = -np.inf
                token = draw(masked, step=len(got), seen=seen)
                got.append(token)
                if token == cfg.stop_speech_token:
                    break
                seen[token] = True
                if fused and len(got) % 2:
                    logits = runs["head2"]([hidden, np.array([token], dtype=np.int64)])[0][0]
                    continue
                if fused:
                    outputs = runs["pair_step"](
                        [
                            np.array([got[-2:]], dtype=np.int64),
                            np.array([len(prefix) // 2 + len(got) // 2], dtype=np.int64),
                            np.array([base.shape[1] + len(got) // 2 - 1], dtype=np.int64),
                            *kv,
                        ]
                    )
                    logits, hidden, kv = outputs[0][0], outputs[1], outputs[2:]
                else:
                    with torch.no_grad():
                        emb = gen._speech_token_embed(token, len(prefix) + len(got)).numpy()
                    outputs = runs["step"](
                        [emb, np.array([base.shape[1] + len(got) - 1], dtype=np.int64), *kv]
                    )
                    logits, kv = outputs[0][0], outputs[1:]
            if got != want:
                index = next(
                    (i for i, (a, b) in enumerate(zip(got, want, strict=False)) if a != b),
                    min(len(got), len(want)),
                )
                raise SystemExit(
                    f"{rec['name']} prefix={len(prefix)}: exact token gate failed at {index}"
                )
            if not prefix:
                with torch.no_grad():
                    forced = gen.teacher_forced_logits(text, voice, got)
                np.testing.assert_allclose(
                    np.asarray(rows), forced[: len(rows)], atol=1e-3, rtol=1e-5
                )
            print(
                f"[gate] {rec['name']} prefix={len(prefix)}: {len(got)} exact tokens",
                flush=True,
            )


def staging_directory(destination: Path):
    import shutil
    import tempfile

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".export-", dir=destination.parent))
    import atexit

    atexit.register(shutil.rmtree, staging, ignore_errors=True)
    if (destination / "export.json").is_file():
        shutil.copyfile(destination / "export.json", staging / "export.json")
    return staging


def promote(staging: Path, destination: Path):
    import shutil

    destination.mkdir(parents=True, exist_ok=True)
    for path in staging.iterdir():
        if path.name == "export.json":
            continue
        target = destination / path.name
        if target.is_dir():
            shutil.rmtree(target)
        path.replace(target)
    (staging / "export.json").replace(destination / "export.json")
    staging.rmdir()


def load_generator(ckpt, algo):
    """The torch reference generator, forced fp32 like the conformance run."""
    from loudkit.models.generator import TorchTokenGenerator

    llama_config = ckpt.manifest["llama_config"]
    assert isinstance(llama_config, dict)
    gen = TorchTokenGenerator(algo, llama_config, attention="eager")
    gen.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in ckpt.tensors("t3.").items()})
    return gen.float().eval()
