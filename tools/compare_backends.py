"""Cross-backend output comparison: cpu/mps/coreml/onnx on the same text.

For each backend: synthesize the same text+voice+seed, time it, and report
(1) byte determinism (same seed twice => identical waveform), (2) token
identity vs the cpu fp32 reference, (3) mel spectrogram correlation vs the cpu
fp32 reference, (4) time-domain correlation. Writes a JSON row per backend.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

import loudkit
from loudkit.config import ExecutionConfig

# Resolved the way the test suite resolves its assets: an environment variable
# with the repository's own `assets/` as the default, flat at the root, matching
# `tests/assets.py`. Hardcoding the author's path meant this tool ran on exactly
# one computer, which is the opposite of what a cross-backend comparison is for.
_DEFAULT_ROOT = str(Path(__file__).resolve().parents[1] / "assets")
_ROOT = os.environ.get("LOUDKIT_ASSET_ROOT", _DEFAULT_ROOT)
CKPT = os.environ.get(
    "LOUDKIT_CHECKPOINT",
    f"{_ROOT}/loudr-1.safetensors",
)
VOICE = "tests/data/reference/testvoice.voice.safetensors"
OUT = Path("out/compare/backends_compare.json")
TEXT = "Pobierz download i zrób code review na 15% szybciej, bo mamy deadline."
SEED = 7
LANG = "pl"

FP32 = ExecutionConfig(
    device="cpu",
    precision={
        "token_generator": "fp32",
        "mel_decoder.estimator": "fp32",
        "mel_decoder.encoder": "fp32",
        "vocoder": "fp32",
    },
)


def corr(a, b):
    a, b = np.asarray(a, np.float32).ravel(), np.asarray(b, np.float32).ravel()
    n = min(len(a), len(b))
    return float(np.corrcoef(a[:n], b[:n])[0, 1])


def run(name: str, device: str, execution=None) -> dict:
    t0 = time.perf_counter()
    engine = loudkit.load(CKPT, device=device, execution=execution)
    load_s = time.perf_counter() - t0
    voice = loudkit.VoiceProfile.load(VOICE)

    t0 = time.perf_counter()
    r1 = engine.synthesize(TEXT, voice, seed=SEED, language=LANG)
    t1 = time.perf_counter()
    r2 = engine.synthesize(TEXT, voice, seed=SEED, language=LANG)
    t2 = time.perf_counter()

    deterministic = np.array_equal(r1.audio, r2.audio)
    return {
        "backend": name,
        "load_s": round(load_s, 3),
        "synth_s": round(t1 - t0, 3),
        "synth2_s": round(t2 - t1, 3),
        "deterministic_bytes": deterministic,
        "n_tokens": len(r1.tokens),
        "duration_s": round(float(r1.duration), 3),
        "audio_len": len(r1.audio),
        "tokens": r1.tokens,
        "mel": r1.mel,
        "audio": r1.audio,
        "execution": engine.execution.describe(),
    }


def main() -> None:
    rows = {}
    ref = run("cpu_fp32", "cpu", FP32)
    rows["cpu_fp32"] = ref

    # torch shipping dtype map on cpu
    rows["cpu"] = run("cpu", "cpu")
    # mps, onnx. coreml runs separately here for its own reason: this script
    # compares whole-pipeline dtype maps, and the coreml backend supplies the
    # renderer only. The in-process crash that used to force the split is
    # gone; loudkit.backends.coreml_backend pins its prediction buffers.
    rows["mps"] = run("mps", "mps")
    rows["onnx"] = run("onnx", "onnx")
    # mps at matched fp32 precision — the honest same-precision comparison
    rows["mps_fp32"] = run(
        "mps_fp32",
        "mps",
        ExecutionConfig(
            device="mps",
            generator_device="cpu",
            precision={
                "token_generator": "fp32",
                "mel_decoder.estimator": "fp32",
                "mel_decoder.encoder": "fp32",
                "vocoder": "fp32",
            },
        ),
    )

    # comparisons vs the fp32 reference — only meaningful between backends
    # that produced IDENTICAL tokens. A token differs (fp16 vs fp32 band), the
    # mel/audio are a different utterance and correlation is meaningless
    # (measured 0.097 once; that is the precision-band contract, not a render
    # difference). So: compare every fp32 row against the fp32 reference, and
    # report the fp16 rows' token divergence separately.
    summary = {}
    for name, row in rows.items():
        same_tokens = row["tokens"] == ref["tokens"]
        summary[name] = {
            "synth_s": row["synth_s"],
            "deterministic_bytes": row["deterministic_bytes"],
            "n_tokens": row["n_tokens"],
            "tokens_exact": same_tokens,
            "mel_corr": corr(row["mel"], ref["mel"]) if same_tokens else None,
            "wave_corr": corr(row["audio"], ref["audio"]) if same_tokens else None,
            "audio_len": row["audio_len"],
        }

    print(f"text: {TEXT!r} seed={SEED} lang={LANG}")
    header = ("backend", "synth(s)", "det", "tok", "tok==ref", "mel_corr", "wave_corr", "len")
    print(
        f"{header[0]:<10} {header[1]:>9} {header[2]:>4} {header[3]:>4} "
        f"{header[4]:>8} {header[5]:>9} {header[6]:>9} {header[7]:>7}"
    )
    for name in ("cpu_fp32", "cpu", "mps", "onnx", "mps_fp32"):
        s = summary[name]
        mc = f"{s['mel_corr']:.6f}" if s["mel_corr"] is not None else "  n/a  "
        wc = f"{s['wave_corr']:.4f}" if s["wave_corr"] is not None else "  n/a "
        print(
            f"{name:<10} {s['synth_s']:>9.3f} {str(s['deterministic_bytes']):>4} "
            f"{s['n_tokens']:>4} {str(s['tokens_exact']):>8} {mc:>9} "
            f"{wc:>9} {s['audio_len']:>7}"
        )
    print(
        "\n(fp16 cpu/mps rows: tokens differ from fp32 — the documented"
        " precision-band contract, so mel/wave corr is reported as n/a)"
    )

    # save compact rows for cross-binding comparison later
    out = {}
    for name, row in rows.items():
        out[name] = {
            "synth_s": row["synth_s"],
            "deterministic_bytes": row["deterministic_bytes"],
            "tokens": row["tokens"],
            "mel": row["mel"].astype(np.float32).tobytes().hex(),
            "audio": row["audio"].astype(np.float32).tobytes().hex(),
        }
    # `out/compare/` is what the justfile documents and what .gitignore
    # reserves; this wrote /tmp, so `just compare` told you to look somewhere
    # the file had never been.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
