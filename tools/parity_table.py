"""Measure every parity claim this project makes, and write the table.

The numbers that justify loudkit live in test docstrings, gate constants and
`docs/platforms/apple.md`. A reader deciding whether to trust the thing has to go and
find them, and a number that is only ever quoted in prose drifts from the
number the suite actually enforces — which is the failure mode this whole
repository is organised against.

So the table is **generated from a run**, not typed. Each row carries the gate
the suite enforces *and* what was measured, and a row that could not be
measured says so rather than being omitted: a table with a hole in it is
information, a table quietly missing a row is a claim.

    python tools/parity_table.py --out docs/parity-measured.md

Needs the packed checkpoint and the reference dumps for the weighted rows; the
weight-free rows (RNG, sampler, frontend, funnel, chunking) run anywhere.
`--out -` prints to stdout.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    # Names only, for the annotations below: loudkit is imported inside the
    # functions that need it so the table still runs without a backend.
    from loudkit.config import Device, Precision

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "data" / "conformance"
REFERENCE = REPO / "tests" / "data" / "reference"


@dataclass
class Row:
    """One claim: what is compared, against what, how tightly, and what happened."""

    stage: str
    against: str
    gate: str
    measured: str = "not measured"
    detail: str = ""
    ok: bool | None = None


@dataclass
class Report:
    rows: list[Row] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add(self, row: Row) -> Row:
        self.rows.append(row)
        return row


# ---------------------------------------------------------------- weight-free


def measure_weight_free(report: Report) -> None:
    """The rows that need no checkpoint: the arithmetic every port shares."""
    from loudkit.rng import KAT_VECTORS, philox_4x32_10

    row = report.add(
        Row(
            "Philox 4x32-10",
            "published known-answer vectors",
            "exact",
            detail="the RNG is checked against a standard, not against itself",
        )
    )
    bad = 0
    for ctr, key, want in KAT_VECTORS:
        c0, c1, c2, c3 = (np.array([c], dtype=np.uint64) for c in ctr)
        got = philox_4x32_10(c0, c1, c2, c3, key[0], key[1])
        if tuple(int(g[0]) for g in got) != want:
            bad += 1
    row.measured = f"{len(KAT_VECTORS) - bad}/{len(KAT_VECTORS)} vectors"
    row.ok = bad == 0

    vectors = json.loads((FIXTURE / "vectors.json").read_text(encoding="utf-8"))
    speechtext = json.loads((FIXTURE / "speechtext.json").read_text(encoding="utf-8"))

    _sampler_row(report, vectors)
    _funnel_row(report, speechtext)
    _chunking_row(report, speechtext)
    _postprocess_row(report)
    _eos_peak_row(report, vectors)
    _seed_row(report, vectors)


def _sampler_row(report: Report, vectors: dict[str, Any]) -> None:
    from loudkit.config import SamplingConfig
    from loudkit.sampler import LRSamplerV1

    row = report.add(
        Row(
            "LR-SAMPLER-v1",
            "shared fixture",
            "exact",
            detail="min_p in logit space, gumbel-argmax, ties to the low index",
        )
    )
    cases = vectors["sampler"]["cases"]
    bad = 0
    for case in cases:
        cfg = case["config"]
        sampler = LRSamplerV1(
            SamplingConfig(
                temperature=cfg["temperature"],
                repetition_penalty=cfg["repetition_penalty"],
                min_p=cfg["min_p"],
                silence_token_ids=tuple(cfg.get("silence_token_ids", ())),
            ),
            seed=int(case["seed"]),
        )
        # Two shapes, both of which the ports already consume: an explicit
        # logits row repeated `repeat_logits` times (so the repetition penalty
        # has something to bite on), or a `logits_recipe` that generates a
        # full-vocabulary row per step from the RNG — the fixture carries the
        # recipe rather than 24 x 8194 floats.
        if "logits_recipe" in case:
            from loudkit.rng import uniforms

            r = case["logits_recipe"]
            rows = [
                (
                    uniforms(
                        seed=int(r["seed"]),
                        stream=int(r["stream"]),
                        step0=step,
                        n_steps=1,
                        width=int(r["vocab"]),
                    )[0]
                    * r["scale"]
                    + r["offset"]
                ).tolist()
                for step in range(int(r["steps"]))
            ]
        else:
            rows = case["logits"]
            if int(case.get("repeat_logits", 0)) > 0:
                rows = [rows[0]] * int(case["repeat_logits"])
        seen: NDArray[np.bool_] = np.zeros(len(rows[0]), bool)
        got = []
        for step, row_logits in enumerate(rows):
            token = sampler(np.asarray(row_logits, dtype=np.float32), step=step, seen=seen)
            got.append(token)
            seen[token] = True
        if got != case["expected"]:
            bad += 1
    row.measured = f"{len(cases) - bad}/{len(cases)} cases"
    row.ok = bad == 0


def _funnel_row(report: Report, speechtext: dict[str, Any]) -> None:
    from loudkit.frontend.polish import speech_text

    row = report.add(
        Row(
            "Speech funnel",
            "shared fixture (Python, Swift, Go, Rust, JS)",
            "exact",
            detail="invisibles, symbols, footnotes, punctuation, Polish respelling",
        )
    )
    cases = speechtext["cases"]
    bad = sum(
        1 for c in cases if speech_text(c["text"], c.get("language") or "") != c["expected"]
    )
    row.measured = f"{len(cases) - bad}/{len(cases)} cases"
    row.ok = bad == 0


def _chunking_row(report: Report, speechtext: dict[str, Any]) -> None:
    from loudkit.config import ChunkConfig
    from loudkit.frontend.chunking import split_text

    row = report.add(
        Row(
            "Long-form splitting",
            "shared fixture (Python, Swift, Go, Rust, JS)",
            "exact",
            detail="where the reader breathes; a different split is a different reading",
        )
    )
    cases = speechtext["chunking"]
    bad = 0
    for case in cases:
        cfg = ChunkConfig(
            max_tokens=case["max_tokens"],
            prefix_tokens=case["prefix_tokens"],
            split_on=tuple(case["split_on"]),
        )
        if split_text(case["text"], cfg) != case["chunks"]:
            bad += 1
    row.measured = f"{len(cases) - bad}/{len(cases)} cases"
    row.ok = bad == 0


def _postprocess_row(report: Report) -> None:
    """The artifact detectors: five implementations, one verdict per case."""
    from loudkit.postprocess import (
        PostprocessConfig,
        ceiling_for,
        desperation_cut,
        ended_tail_trim,
        inspect,
        is_trailing_filler,
        terminal_echo_cut,
    )

    row = report.add(
        Row(
            "Postprocess detectors",
            "shared fixture (Python, Swift, Go, Rust, JS)",
            "exact",
            detail="where a chunk ended; a different verdict is a different cut",
        )
    )
    fx = json.loads((FIXTURE / "postprocess.json").read_text(encoding="utf-8"))
    sil = fx["silence_token_ids"]
    base = PostprocessConfig(**fx["config"])

    def build(shape: list[list[Any]]) -> list[int]:
        out: list[int] = []
        for kind, count in shape:
            out.extend((20 + i % 60) if kind == "speech" else i % 8 for i in range(count))
        return out

    total = 0
    bad = 0
    for case in fx["ceiling"]:
        total += 1
        bad += (
            ceiling_for(case["text_tokens"], config=base, window=case["window"])
            != case["expect"]
        )
    for case in fx["trailing_filler"]:
        total += 1
        bad += (
            is_trailing_filler(build(case["shape"]), case["from"], silence=sil, config=base)
            != case["expect"]
        )
    for case in fx["desperation"]:
        total += 1
        bad += (
            desperation_cut(
                build(case["shape"]),
                text_token_count=case["text_tokens"],
                min_tokens=case["min_tokens"],
                eos_peak_at=case["eos_peak_at"],
                silence=sil,
                config=base,
                peak_allowed=case["peak_allowed"],
            )
            != case["expect"]
        )
    for case in fx["ended_tail"]:
        total += 1
        bad += (
            ended_tail_trim(
                build(case["shape"]),
                silence=sil,
                config=base,
                is_terminal=case["is_terminal"],
            )
            != case["expect"]
        )
    for case in fx["terminal_echo"]:
        total += 1
        bad += (
            terminal_echo_cut(
                token_count=case["token_count"],
                eos_peak_at=case["eos_peak_at"],
                eos_peak_prob=case["eos_peak_prob"],
                min_tokens=case["min_tokens"],
                is_terminal=case["is_terminal"],
                hit_ceiling=case["hit_ceiling"],
                config=base,
            )
            != case["expect"]
        )
    for case in fx["resolve"]:
        total += 1
        overrides = {"mode": case["mode"]} if "mode" in case else {}
        cfg = PostprocessConfig(**{**fx["config"], **overrides})
        got = inspect(
            build(case["shape"]),
            text_token_count=case["text_tokens"],
            min_tokens=case["min_tokens"],
            eos_peak_at=case["eos_peak_at"],
            eos_peak_prob=case["eos_peak_prob"],
            ended=case["ended"],
            is_terminal=case["is_terminal"],
            hit_ceiling=case["hit_ceiling"],
            silence=sil,
            config=cfg,
        )
        want = case["expect"]
        bad += (got.keep, got.reason, got.suspect) != (
            want["keep"],
            want["reason"],
            want["suspect"],
        )
    row.measured = f"{total - bad}/{total} cases"
    row.ok = bad == 0


def _eos_peak_row(report: Report, vectors: dict[str, Any]) -> None:
    """The stop-token observation two of the detector rules threshold on."""
    from loudkit.config import SamplingConfig
    from loudkit.rng import uniforms
    from loudkit.sampler import LRSamplerV1

    section = vectors["eos_peak"]
    row = report.add(
        Row(
            "EOS peak observation",
            "shared fixture (Python, Swift, Go, Rust, JS)",
            f"step exact, probability rtol {section['prob_rtol']:g}",
            detail="audible despite never feeding back: two detector rules threshold on it",
        )
    )
    cases = section["cases"]
    bad = 0
    for case in cases:
        cfg = SamplingConfig(
            temperature=case["config"]["temperature"],
            repetition_penalty=case["config"]["repetition_penalty"],
            min_p=case["config"]["min_p"],
            silence_token_ids=tuple(case["config"]["silence_token_ids"]),
        )
        r = case["logits_recipe"]
        sampler = LRSamplerV1(
            cfg, seed=case["seed"], stop_token=case["stop_token"], eos_floor=case["eos_floor"]
        )
        seen = np.zeros(r["vocab"], dtype=bool)
        for step in range(r["steps"]):
            u = uniforms(r["seed"], r["stream"], step, 1, r["vocab"])[0]
            logits_row = (u * r["scale"] + r["offset"]).astype(np.float32)
            seen[sampler(logits_row, step=step, seen=seen)] = True
        at, prob = sampler.eos_peak
        if at != case["expected_at"] or abs(prob - case["expected_prob"]) > section[
            "prob_rtol"
        ] * abs(case["expected_prob"]):
            bad += 1
    row.measured = f"{len(cases) - bad}/{len(cases)} cases"
    row.ok = bad == 0


def _seed_row(report: Report, vectors: dict[str, Any]) -> None:
    from loudkit.engine import _derive

    row = report.add(
        Row(
            "Seed derivation",
            "shared fixture",
            "exact",
            detail="one user seed, independent per-stage streams",
        )
    )
    cases = vectors["seeds"]["derivation"]
    # The fixture records the derived value as a hex string, because a 64-bit
    # integer does not survive a JSON round-trip through every port.
    bad = sum(
        1
        for c in cases
        if _derive(int(c["seed"]), int(c["stream"])) != int(str(c["derived"]), 16)
    )
    row.measured = f"{len(cases) - bad}/{len(cases)} cases"
    row.ok = bad == 0


# ------------------------------------------------------------------- weighted


def measure_against_reference(  # noqa: PLR0915 - one block per measured row
    report: Report, checkpoint: Path
) -> None:
    """The rows that compare this engine to the implementation that shipped."""
    if not REFERENCE.exists() or not (REFERENCE / "meta.json").exists():
        report.notes.append(
            "reference dumps absent — the torch-vs-reference rows were not measured"
        )
        _placeholder_reference_rows(report)
        return

    import torch

    import loudkit
    from loudkit.sampler import LRSamplerV1
    from loudkit.voice import VoiceProfile

    engine = loudkit.load(str(checkpoint), device="cpu")
    voice = VoiceProfile.load(REFERENCE / "testvoice.voice.safetensors")
    meta = json.loads((REFERENCE / "meta.json").read_text(encoding="utf-8"))
    sentences = [k for k in ("0", "1", "2") if k in meta]

    tf = report.add(
        Row(
            "Token generator, teacher-forced",
            "reference implementation",
            "top-1 >= 99%, median KL < 1e-4",
            detail="the only generator comparison free of sampling chaos (EXP-010)",
        )
    )
    agree = steps = 0
    kls: list[float] = []
    for i in sentences:
        rec = meta[i]
        ref = np.load(REFERENCE / f"s{i}_tf_logits.npy")
        mine = engine.token_generator.teacher_forced_logits(
            np.asarray(rec["text_ids"], dtype=np.int64), voice, rec["speech_tokens"][:64]
        )
        n = min(len(mine), len(ref))
        agree += int((mine[:n].argmax(-1) == ref[:n].argmax(-1)).sum())
        steps += n
        p = torch.log_softmax(torch.tensor(ref[:n]), -1)
        q = torch.log_softmax(torch.tensor(mine[:n]), -1)
        kls.append(
            float(
                torch.nn.functional.kl_div(q, p, log_target=True, reduction="none")
                .sum(-1)
                .abs()
                .median()
            )
        )
    top1 = agree / steps
    tf.measured = f"top-1 {top1 * 100:.2f}% ({agree}/{steps}), max median KL {max(kls):.2e}"
    tf.ok = top1 >= 0.99 and max(kls) < 1e-4

    free = report.add(
        Row(
            "Token generator, free-running",
            "reference implementation",
            "exact",
            detail="same law, same seed — a mismatch means the logits moved",
        )
    )
    matched = total = 0
    for i in sentences:
        rec = meta[i]
        sampler = LRSamplerV1(engine.algorithm.sampling, seed=rec["seed"])
        tokens = list(
            engine.token_generator.generate(
                np.asarray(rec["text_ids"], dtype=np.int64), voice, sampler=sampler
            )
        )
        total += len(rec["tokens"])
        matched += sum(1 for a, b in zip(tokens, rec["tokens"], strict=False) if a == b) * (
            len(tokens) == len(rec["tokens"])
        )
    free.measured = f"{matched}/{total} tokens"
    free.ok = matched == total

    mel_row = report.add(
        Row(
            "Mel decoder, fixed tokens",
            "reference implementation",
            "corr >= 0.999",
            detail="same tokens and same injected noise, so a difference is arithmetic",
        )
    )
    wav_row = report.add(
        Row(
            "Vocoder, fixed tokens",
            "reference implementation",
            "corr >= 0.98",
            detail="gated loosely on purpose: predicted phase decorrelates, spectrum does not",
        )
    )
    mels, waves = [], []
    for i in sentences:
        rec = meta[i]
        result = engine.synthesize_tokens(rec["speech_tokens"], voice, seed=rec["seed"])
        ref_mel = np.load(REFERENCE / f"s{i}_mel.npy")
        ref_wav = np.load(REFERENCE / f"s{i}_wav.npy")
        mels.append(float(np.corrcoef(result.mel.ravel(), ref_mel.ravel())[0, 1]))
        n = min(len(result.audio), len(ref_wav))
        waves.append(float(np.corrcoef(result.audio[:n], ref_wav[:n])[0, 1]))
    mel_row.measured = f"corr {min(mels):.6f}–{max(mels):.6f} over {len(mels)} sentences"
    mel_row.ok = min(mels) >= 0.999
    wav_row.measured = f"corr {min(waves):.4f}–{max(waves):.4f} over {len(waves)} sentences"
    wav_row.ok = min(waves) >= 0.98

    rerender = report.add(
        Row(
            "Re-render, same seed and build",
            "itself",
            "bit-identical",
            detail="identity class I-2: determinism within one backend",
        )
    )
    rec = meta[sentences[0]]
    a = engine.synthesize_tokens(rec["speech_tokens"], voice, seed=rec["seed"])
    b = engine.synthesize_tokens(rec["speech_tokens"], voice, seed=rec["seed"])
    same = bool(np.array_equal(a.audio, b.audio))
    rerender.measured = "identical" if same else "DIVERGED"
    rerender.ok = same


def _placeholder_reference_rows(report: Report) -> None:
    for stage, gate in (
        ("Token generator, teacher-forced", "top-1 >= 99%, median KL < 1e-4"),
        ("Token generator, free-running", "exact"),
        ("Mel decoder, fixed tokens", "corr >= 0.999"),
        ("Vocoder, fixed tokens", "corr >= 0.98"),
        ("Re-render, same seed and build", "bit-identical"),
    ):
        report.add(Row(stage, "reference implementation", gate))


# The ONNX graphs are exported fp32 only (EXP-015: fp16 not worth a second
# artefact; EXP-017: int8 blocked), while the fixture's `execution` block
# records the CoreML precision map. Handing the fp16 map to ONNX is a refusal,
# not a measurement — the same explicit map `test_render_band_onnx` uses.
_ONNX_PRECISION: dict[str, Precision] = {
    "token_generator": "fp32",
    "mel_decoder.estimator": "fp32",
    "mel_decoder.encoder": "fp32",
    "vocoder": "fp32",
}


def measure_backend(report: Report, checkpoint: Path, device: Device, label: str) -> None:
    """One non-torch backend against the same fixture the others are gated on."""
    cases = (
        json.loads((FIXTURE / "vectors.json").read_text(encoding="utf-8")).get("end_to_end")
        or []
    )
    # Read from the fixture, not restated here. This row's whole claim is that a
    # second backend is held to "the same fixture the others are gated on" — and
    # it was restating the bar instead, as a literal that had drifted from the
    # one the fixture declares and Swift's end-to-end test enforces. A gate
    # asserted in two places is a gate that eventually means two things, and the
    # copy in a report generator is the one nobody re-measures.
    gates = cases[0]["gates"] if cases else {"mel_corr": 0.999, "wave_corr": 0.95}
    mel_gate = min(float(c["gates"]["mel_corr"]) for c in cases) if cases else gates["mel_corr"]
    wave_gate = (
        min(float(c["gates"]["wave_corr"]) for c in cases) if cases else gates["wave_corr"]
    )
    row = report.add(
        Row(
            f"{label} renderer",
            "shared fixture",
            f"mel corr >= {mel_gate:g}, wave corr >= {wave_gate:g}",
            detail="a second backend does not get a second, looser bar",
        )
    )
    if not cases:
        report.notes.append(f"{label}: the fixture carries no end_to_end cases")
        return
    try:
        import loudkit
        from loudkit.config import ExecutionConfig
        from loudkit.voice import VoiceProfile

        precision = _ONNX_PRECISION if device == "onnx" else cases[0]["execution"]
        # Pinned for the same reason make_conformance pins it: a measured row
        # must say which provider produced it, not inherit one from the host.
        execution = ExecutionConfig(device=device, precision=precision, onnx_provider="cpu")
        engine = loudkit.load(str(checkpoint), device=device, execution=execution)
        voice = VoiceProfile.load(FIXTURE / cases[0]["voice"])
    except Exception as exc:  # noqa: BLE001 - a missing backend is a note, not a crash
        report.notes.append(f"{label}: not measured ({type(exc).__name__}: {exc})")
        return

    mels, waves = [], []
    for case in cases:
        result = engine.synthesize_tokens(case["tokens"], voice, seed=case["seed"])
        ref_mel = np.fromfile(FIXTURE / case["mel"]["file"], dtype="<f4").reshape(
            case["mel"]["shape"]
        )
        ref_wav = np.fromfile(FIXTURE / case["wav"]["file"], dtype="<f4")
        if result.mel.shape != ref_mel.shape or len(result.audio) != len(ref_wav):
            row.measured = "SHAPE MISMATCH"
            row.ok = False
            return
        mels.append(float(np.corrcoef(result.mel.ravel(), ref_mel.ravel())[0, 1]))
        waves.append(float(np.corrcoef(result.audio, ref_wav)[0, 1]))
    row.measured = f"mel {min(mels):.6f}, wave {min(waves):.4f} (worst of {len(mels)})"
    row.ok = min(mels) >= mel_gate and min(waves) >= wave_gate


# --------------------------------------------------------------------- output


def render(report: Report, environment: str) -> str:
    """The table, plus what was not measured and why."""
    lines = [
        "# Parity, measured",
        "",
        "The companion to [`parity.md`](design/parity.md), which is written by hand and",
        "explains *what the reference is* and how it was produced. This file is the",
        "other half: the current numbers, regenerated rather than remembered.",
        "",
        "Generated by `tools/parity_table.py`. Every row is a comparison the test",
        "suite enforces; `gate` is the threshold that fails the build, `measured` is",
        "what this run actually observed. A row that says *not measured* was not run",
        "in this environment — it is left in rather than dropped, because a table",
        "quietly missing a row reads as a table with nothing to hide.",
        "",
        f"Environment: {environment}",
        "",
        "| stage | compared against | gate | measured |",
        "|---|---|---|---|",
    ]
    for row in report.rows:
        mark = "" if row.ok is None else ("✓ " if row.ok else "✗ ")
        lines.append(f"| {row.stage} | {row.against} | `{row.gate}` | {mark}{row.measured} |")
    lines.append("")

    detailed = [r for r in report.rows if r.detail]
    if detailed:
        lines += ["## Why each gate is where it is", ""]
        lines += [f"- **{r.stage}** — {r.detail}" for r in detailed]
        lines.append("")

    if report.notes:
        lines += ["## Not measured in this run", ""]
        lines += [f"- {n}" for n in report.notes]
        lines.append("")
    return "\n".join(lines)


def _environment() -> str:
    bits = [f"Python {sys.version.split()[0]}"]
    try:
        import torch

        bits.append(f"torch {torch.__version__}")
    except ImportError:  # pragma: no cover - torch is a dev dependency
        bits.append("torch absent")
    try:
        import onnxruntime

        bits.append(f"onnxruntime {onnxruntime.__version__}")
    except ImportError:
        bits.append("onnxruntime absent")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        bits.append(f"loudkit {commit}")
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        pass
    return ", ".join(bits)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="packed .safetensors; weighted rows need it")
    parser.add_argument(
        "--out",
        default="docs/parity-measured.md",
        help="'-' for stdout. Never docs/design/parity.md — that is the hand-written "
        "report, with the provenance and the experiment references this file "
        "deliberately does not carry.",
    )
    args = parser.parse_args()

    report = Report()
    measure_weight_free(report)

    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        measure_against_reference(report, checkpoint)
        measure_backend(report, checkpoint, "onnx", "ONNX")
        measure_backend(report, checkpoint, "coreml", "CoreML")
    else:
        report.notes.append(
            "no --checkpoint given — every weighted row was skipped, so this table "
            "covers the algorithm layer only"
        )
        _placeholder_reference_rows(report)

    text = render(report, _environment())
    if args.out == "-":
        print(text)
    else:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out} ({len(report.rows)} rows, {len(report.notes)} notes)")

    failed = [r.stage for r in report.rows if r.ok is False]
    if failed:
        print("FAILED: " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
