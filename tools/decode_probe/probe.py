"""Stage-by-stage probe of the Python decode loop, for cross-port comparison.

One record per port, the same five stages in the same order, so a comparison
stops at the first stage that disagrees instead of reporting noise from
everything downstream of a divergence.

    PYTHONPATH=python python tools/decode_probe/probe.py \
        BUNDLE VOICE TEXT SEED LANGUAGE OUTDIR [--device onnx] [--no-longform]

Writes OUTDIR/python-<device>.json and OUTDIR/python-<device>.<stage>.bin
(raw little-endian float32). See tools/decode_probe/compare.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main() -> int:  # noqa: PLR0915, one linear pass per stage, read in order
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("voice")
    ap.add_argument("text")
    ap.add_argument("seed", type=int)
    ap.add_argument("language")
    ap.add_argument("outdir")
    ap.add_argument("--device", default="onnx")
    ap.add_argument("--no-longform", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import loudkit as lk
    from loudkit.sampler import LRSamplerV1
    from loudkit.voice import VoiceProfile

    # Which module actually answered. A path that silently resolves to another
    # checkout is the failure this line exists to make impossible.
    print(f"loudkit module: {lk.__file__}", file=sys.stderr)
    print(f"python: {sys.executable}", file=sys.stderr)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    name = args.tag or f"python-{args.device}"

    eng = lk.load(args.bundle, device=args.device)
    voice = VoiceProfile.load(args.voice)
    cfg = eng.algorithm

    rec: dict = {
        "port": name,
        "loaded_from": lk.__file__,
        "bundle": args.bundle,
        "voice": args.voice,
        "text": args.text,
        "seed": args.seed,
        "language": args.language,
        "decode": cfg.decode,
        "fingerprint": eng.describe(),
        "sample_rate": cfg.sample_rate,
        "backend": eng.backend,
    }

    # 1. text tokens -- what the funnel handed the generator.
    text_tokens = [int(t) for t in eng.frontend.encode(args.text, language=args.language)]
    rec["text_tokens"] = text_tokens

    # 2. the prefill row and the conditioning row, where the backend exposes
    #    them. The torch generator has `prefill_embeds`; the ONNX one builds
    #    the same row through `_prefill_embeds`.
    gen = eng.token_generator
    try:
        if hasattr(gen, "prefill_embeds"):
            row = gen.prefill_embeds(np.asarray(text_tokens, dtype=np.int64), voice)
            row = row.detach().float().cpu().numpy()
        else:
            # Reached by name because the two backends expose the same row
            # under different ones; the probe is a driver, not a caller of the
            # public surface.
            row = gen._prefill_embeds(  # type: ignore[attr-defined]
                np.asarray(text_tokens, dtype=np.int64), voice, []
            )
            if isinstance(row, tuple):
                row = row[0]
            row = np.asarray(row, dtype=np.float32)
        row = np.ascontiguousarray(row.reshape(-1).astype(np.float32))
        (out / f"{name}.prefill.bin").write_bytes(row.tobytes())
        rec["prefill"] = {"len": int(row.size), "sha": sha(row), "head": row[:8].tolist()}
    except Exception as exc:  # pragma: no cover - reported, not raised
        rec["prefill"] = {"error": f"{type(exc).__name__}: {exc}"}

    try:
        cond = gen._cond_row(voice) if hasattr(gen, "_cond_row") else None
        if cond is not None:
            cond = np.ascontiguousarray(np.asarray(cond, dtype=np.float32).reshape(-1))
            (out / f"{name}.cond.bin").write_bytes(cond.tobytes())
            rec["cond"] = {"len": int(cond.size), "sha": sha(cond), "head": cond[:8].tolist()}
    except Exception as exc:  # pragma: no cover
        rec["cond"] = {"error": f"{type(exc).__name__}: {exc}"}

    # 3. the token sequence, from one generate call on the whole text.
    sampler = LRSamplerV1(cfg.sampling, seed=args.seed)
    tokens = gen.generate(np.asarray(text_tokens, dtype=np.int64), voice, sampler=sampler)
    tokens = [int(t) for t in tokens]
    rec["speech_tokens_raw"] = tokens
    # What the renderer is actually handed: `Engine.synthesize_tokens` strips
    # the start/stop markers, and the ports disagree about whether `generate`
    # hands the stop token back at all, so the comparison is made on both.
    limit = cfg.start_speech_token
    tokens = [t for t in tokens if t < limit]
    rec["speech_tokens"] = tokens

    # 4. the mel frames.
    mel = eng.mel_decoder.decode(tokens, voice, seed=args.seed)
    mel = np.ascontiguousarray(np.asarray(mel, dtype=np.float32))
    rec["mel"] = {
        "shape": list(mel.shape),
        "sha": sha(mel),
        "head": mel.reshape(-1)[:8].tolist(),
    }
    (out / f"{name}.mel.bin").write_bytes(mel.reshape(-1).tobytes())

    # 5. the rendered samples.
    audio = eng.vocoder.synthesize(mel, voice, seed=args.seed)
    audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).reshape(-1))
    rec["audio"] = {"len": int(audio.size), "sha": sha(audio), "head": audio[:8].tolist()}
    (out / f"{name}.audio.bin").write_bytes(audio.tobytes())

    # 6. the long-form path: chunking, prefix carry, retries, the WAV bytes.
    if not args.no_longform:
        res = eng.synthesize(args.text, voice, seed=args.seed, language=args.language)
        la = np.ascontiguousarray(np.asarray(res.audio, dtype=np.float32).reshape(-1))
        (out / f"{name}.longform.bin").write_bytes(la.tobytes())
        wav = res.wav_bytes() if hasattr(res, "wav_bytes") else b""
        rec["longform"] = {
            "tokens": [int(t) for t in res.tokens],
            "audio_len": int(la.size),
            "audio_sha": sha(la),
            "audio_head": la[:8].tolist(),
            "wav_sha": hashlib.sha256(wav).hexdigest() if wav else "",
            "n_chunks": len(res.chunks),
            "hit_token_cap": bool(getattr(res, "hit_token_cap", False)),
        }

    (out / f"{name}.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(f"wrote {out / (name + '.json')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
