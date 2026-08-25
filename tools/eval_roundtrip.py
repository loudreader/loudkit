"""ASR round-trip evaluation against each language's own floor.

Renders the probe corpus (tests/data/probes/probes.json), transcribes it with
an ASR the caller provides, and reports CER per language **next to that
language's human-speech floor** — because a round-trip number means nothing on
its own: Danish's floor is more than double English's, and one global threshold
would hold Danish to an impossible bar while letting English coast.

Advisory by construction, never a gate inside the engine: the blind spots in
tools/eval_floors.json (liaison invisible to WER, the pt variant undetectable,
ASR correcting mispronunciations from its own prior) are why an ASR verdict may
inform a release decision and must never cut audio. Run with two ASR families
when the number matters — a verifier from the evaluator's own family recovers
2–3x more apparent headroom than a cross-family pair.

Usage:
    python tools/eval_roundtrip.py <checkpoint> <voice> out/
    # then transcribe out/*.wav with your ASR of choice and:
    python tools/eval_roundtrip.py --score out/ transcripts.json
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROBES = REPO / "tests" / "data" / "probes" / "probes.json"
FLOORS = REPO / "tools" / "eval_floors.json"


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate, on NFC-normalised, lowercased, punctuation-free
    text — the usual round-trip normalisation, so a comma cannot count as an
    error while a missing clause does."""

    def clean(s: str) -> str:
        s = unicodedata.normalize("NFC", s).lower()
        return "".join(ch for ch in s if ch.isalnum() or ch.isspace())

    a, b = clean(reference).split(), clean(hypothesis).split()
    ref = " ".join(a)
    hyp = " ".join(b)
    if not ref:
        return 0.0
    # Levenshtein over characters, two rows.
    prev = list(range(len(hyp) + 1))
    for i, ca in enumerate(ref, 1):
        row = [i]
        for j, cb in enumerate(hyp, 1):
            row.append(min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = row
    return prev[-1] / len(ref)


def render(checkpoint: str, voice_path: str, out_dir: Path) -> None:
    import numpy as np
    import soundfile as sf

    import loudkit
    from loudkit.voice import VoiceProfile

    engine = loudkit.load(checkpoint, device="cpu")
    voice = VoiceProfile.load(voice_path)
    probes = json.loads(PROBES.read_text(encoding="utf-8"))["languages"]
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for lang, items in probes.items():
        for i, item in enumerate(items):
            r = engine.synthesize_long(item["text"], voice, seed=7, language=lang)
            name = f"{lang}_{i:02d}_{item['class']}.wav"
            sf.write(out_dir / name, np.asarray(r.audio), r.sample_rate)
            manifest.append(
                {"file": name, "language": lang, "text": item["text"], "class": item["class"]}
            )
            print(f"{name}: {len(r.tokens)} tokens", flush=True)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n{len(manifest)} renders in {out_dir}; transcribe them, then --score")


def score(out_dir: Path, transcripts_path: Path) -> None:
    floors = json.loads(FLOORS.read_text(encoding="utf-8"))["whisper_large_v3_fleurs_cer"]
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    transcripts = json.loads(transcripts_path.read_text(encoding="utf-8"))  # {file: hypothesis}
    by_lang: dict[str, list[float]] = {}
    missing: list[str] = []
    for entry in manifest:
        hyp = transcripts.get(entry["file"])
        if hyp is None:
            # Counted, not skipped. A transcript file covering half the manifest
            # produced a table of confident per-language numbers computed from
            # whatever happened to be there, with nothing saying the other half
            # was never scored — the shape of a green result that means nothing.
            missing.append(entry["file"])
            continue
        by_lang.setdefault(entry["language"], []).append(cer(entry["text"], hyp))
    if missing:
        print(
            f"warning: {len(missing)} of {len(manifest)} manifest entries have no "
            f"transcript and were not scored (first: {missing[0]})",
            file=sys.stderr,
        )
    print(f"{'lang':5} {'n':>3} {'CER':>7} {'floor':>7} {'ratio':>6}  verdict (factor 2.0)")
    for lang in sorted(by_lang):
        rates = by_lang[lang]
        mean = sum(rates) / len(rates)
        floor = floors.get(lang)
        if floor is None:
            print(
                f"{lang:5} {len(rates):3} {mean:7.3f} {'—':>7} {'—':>6}  "
                "no floor measured; do not gate"
            )
            continue
        ratio = mean * 100 / floor
        verdict = "within" if mean * 100 <= 2.0 * floor else "OVER"
        print(f"{lang:5} {len(rates):3} {mean:7.3f} {floor:6.1f}% {ratio:5.1f}x  {verdict}")


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--score",
        nargs=2,
        metavar=("OUT_DIR", "TRANSCRIPTS"),
        type=Path,
        help="score a finished render against its transcripts instead of rendering",
    )
    # Optional so --score can stand alone; required together in render mode,
    # which the check below states rather than the arity.
    ap.add_argument("checkpoint", nargs="?")
    ap.add_argument("voice", nargs="?")
    ap.add_argument("out_dir", nargs="?", type=Path)
    return ap


def main() -> int:
    ap = _parser()
    args = ap.parse_args()
    if args.score:
        score(*args.score)
        return 0
    if not (args.checkpoint and args.voice and args.out_dir):
        ap.error("render needs <checkpoint> <voice> <out-dir>; scoring needs --score")
    render(args.checkpoint, args.voice, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
