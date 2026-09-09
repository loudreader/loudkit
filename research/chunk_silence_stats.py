"""Per-chunk silence statistics, at a scale where rates mean something.

Renders a corpus chunk by chunk and records, for every chunk, how much silence
sits at its head, at its tail, and strictly inside it. A gap a listener
complains about is one of three different things, and lumping them together
hides which one a change actually fixed:

- **head** silence on a chunk that continues a sentence. Nothing in the
  pipeline trims it: ``Inspection.keep`` counts *leading* tokens that survive,
  so every postprocess rule cuts a tail.
- **seam** silence, which is the tail of one chunk plus the head of the next,
  glued by the join. No single chunk looks wrong; the joined audio does.
- **interior** silence, a run the decoder free-ran mid-clause. The postprocess
  rules are all tail-anchored and cannot reach it.

Twenty paragraphs cannot separate those. A one-in-twenty failure needs hundreds
of chunks before its rate is worth quoting, which is what this tool is for.

Resumable by passage: an interrupted run continues, and two profiles can be
compared by pointing ``--label`` at each in turn.

Usage::

    python research/chunk_silence_stats.py --label shipped --voice joe \\
        --passages 200 --out out/chunkstats/shipped.jsonl
    python research/chunk_silence_stats.py --label cleaned \\
        --profile /path/to/joe-clean.safetensors \\
        --passages 200 --out out/chunkstats/cleaned.jsonl
    python research/chunk_silence_stats.py --compare out/chunkstats/*.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
READING_SET = REPO / "tests/data/judge/reading-en.json"

# The same framing the rest of this investigation used, so numbers compare.
FRAME_SECONDS = 0.02
SILENCE_RMS = 0.012
MIN_RUN_SECONDS = 0.18


def silence_profile(audio, rate: int) -> dict:
    """Head, tail and largest interior silence of one chunk, in seconds."""
    import numpy as np

    samples = np.asarray(audio, dtype="float32").reshape(-1)
    width = int(rate * FRAME_SECONDS)
    if width < 1 or len(samples) < width:
        return {"seconds": len(samples) / rate, "head": 0.0, "tail": 0.0, "interior": 0.0}
    frames = np.array(
        [
            float(np.sqrt(np.mean(samples[i : i + width] ** 2)))
            for i in range(0, len(samples) - width, width)
        ]
    )
    quiet = frames < SILENCE_RMS

    head = 0
    while head < len(quiet) and quiet[head]:
        head += 1
    tail = 0
    while tail < len(quiet) - head and quiet[len(quiet) - 1 - tail]:
        tail += 1

    interior = run = 0
    for flag in quiet[head : len(quiet) - tail]:
        if flag:
            run += 1
            interior = max(interior, run)
        else:
            run = 0
    if interior * FRAME_SECONDS < MIN_RUN_SECONDS:
        interior = 0

    # The floor is recorded so a measurement can disqualify itself. An
    # absolute RMS threshold only separates speech from silence while the
    # silence sits well below it; a voice that renders with audible room tone
    # would have its pauses counted as speech and its gap rate reported as
    # zero. That failure is silent, so the evidence for ruling it out travels
    # with every row.
    quietest = float(np.percentile(frames, 5)) if len(frames) else 0.0
    # A chunk that is silence end to end is not a long pause, it is text that
    # was never spoken. It passes postprocess as clean, because every rule is
    # looking for where speech stopped and there was none. Counted apart from
    # the three pause classes: a pause is a defect of delivery, this is missing
    # content, and a fix that shortened it would be deleting the sentence
    # rather than repairing it.
    voiced = int((~quiet).sum())
    return {
        "seconds": round(len(samples) / rate, 3),
        "head": round(head * FRAME_SECONDS, 3),
        "tail": round(tail * FRAME_SECONDS, 3),
        "interior": round(interior * FRAME_SECONDS, 3),
        "floor": round(quietest, 6),
        "voiced_frames": voiced,
    }


def render(args) -> int:
    import loudkit as lk

    passages = json.loads(Path(args.reading_set).read_text())["passages"][: args.passages]
    engine = lk.load(args.checkpoint)
    if args.profile:
        voice = lk.VoiceProfile.load(args.profile)
    else:
        voice = lk.voice(args.voice, repo=args.checkpoint)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.is_file():
        for line in out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["passage"])

    todo = [p for p in passages if p["id"] not in done]
    print(f"{args.label}: {len(passages)} passages, {len(todo)} to render", flush=True)

    # Appended inside a `with`: an exception mid-loop closed this handle
    # because the interpreter did, not because the code did.
    with out.open("a") as handle:
        started = time.time()
        chunks_written = 0
        for index, passage in enumerate(todo, start=1):
            pieces = [
                silence_profile(chunk.audio, chunk.sample_rate)
                for chunk in engine.stream(
                    passage["text"], voice, seed=args.seed, language=args.language or None
                )
            ]
            # The seam is what the join produces: one chunk's tail against the
            # next chunk's head. It belongs to neither chunk, so it is recorded
            # here rather than inferred later.
            seams = [
                round(pieces[i]["tail"] + pieces[i + 1]["head"], 3)
                for i in range(len(pieces) - 1)
            ]
            handle.write(
                json.dumps(
                    {
                        "label": args.label,
                        "passage": passage["id"],
                        "chars": len(passage["text"]),
                        "seed": args.seed,
                        "language": args.language or voice.language,
                        "chunks": pieces,
                        "seams": seams,
                    }
                )
                + "\n"
            )
            handle.flush()
            chunks_written += len(pieces)
            if index % 10 == 0 or index == len(todo):
                rate = (time.time() - started) / index
                left = rate * (len(todo) - index)
                print(
                    f"  {index}/{len(todo)} passages, {chunks_written} chunks, "
                    f"~{left / 60:.0f} min left",
                    flush=True,
                )
    print(f"{args.label}: wrote {chunks_written} chunks to {out}", flush=True)
    return 0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def compare(paths: list[Path]) -> int:
    groups: dict[str, list[dict]] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                groups.setdefault(row["label"], []).append(row)

    # Match within a voice, not across the whole table. Two arms of one voice
    # must be compared on the passages both rendered, since one run being
    # further along than the other is normal while both are in flight. But two
    # different voices read different languages, so intersecting globally would
    # leave nothing at all.
    families: dict[str, list[str]] = {}
    for label in groups:
        families.setdefault(label.rsplit("-", 1)[0], []).append(label)
    for family, labels in families.items():
        if len(labels) < 2:
            continue
        shared = set.intersection(*({r["passage"] for r in groups[k]} for k in labels))
        dropped = {k: len(groups[k]) - len(shared) for k in labels}
        for k in labels:
            groups[k] = [r for r in groups[k] if r["passage"] in shared]
        if any(dropped.values()):
            print(
                f"{family}: matched on {len(shared)} passages present in both arms "
                f"(dropped {', '.join(f'{k}:{n}' for k, n in dropped.items() if n)})"
            )

    print(
        f"{'label':12} {'passages':>8} {'chunks':>7} {'seams':>6}   "
        f"{'head p95':>8} {'tail p95':>8} {'seam p95':>8} {'int p95':>8}   "
        f"{'seam>1s':>8} {'int>1s':>7} {'tail>2s':>8} {'any>1s':>7}"
    )
    for label, rows in sorted(groups.items()):
        chunks = [c for r in rows for c in r["chunks"]]
        floors = [c["floor"] for c in chunks if "floor" in c]
        # A first chunk has no carried prefix and cannot show the seam-head
        # failure, so it would dilute the head statistic if counted.
        heads = [c["head"] for r in rows for c in r["chunks"][1:]]
        seams = [s for r in rows for s in r["seams"]]
        interiors = [c["interior"] for c in chunks]
        # A chunk that hits the token cap can free-run silence into its tail.
        # That tail belongs to no seam when it is the last chunk, so a
        # seam-only view cannot see it at all; it is its own defect class.
        tails = [c["tail"] for c in chunks]
        over_tail = sum(1 for t in tails if t > 2.0)
        over_seam = sum(1 for s in seams if s > 1.0)
        over_int = sum(1 for v in interiors if v > 1.0)
        worst = [
            max([*r["seams"], *(c["interior"] for c in r["chunks"])], default=0.0) for r in rows
        ]
        over_any = sum(1 for w in worst if w > 1.0)

        # `voiced_frames` was added after some runs were recorded. Where it is
        # absent, fall back to geometry: a chunk whose head covers its whole
        # length has no speech in it. Without the fallback an older arm would
        # silently report zero and a newer one would look like a regression.
        def is_mute(c: dict) -> bool:
            if "voiced_frames" in c:
                return c["voiced_frames"] == 0
            return c["seconds"] > 0.3 and c["head"] >= c["seconds"] - 0.06

        mute = [c for c in chunks if is_mute(c)]
        print(
            f"{label:12} {len(rows):8} {len(chunks):7} {len(seams):6}   "
            f"{percentile(heads, 0.95):7.2f}s {percentile(tails, 0.95):7.2f}s "
            f"{percentile(seams, 0.95):7.2f}s "
            f"{percentile(interiors, 0.95):7.2f}s   "
            f"{over_seam:4}/{len(seams):<3} {over_int:3}/{len(interiors):<3} "
            f"{over_tail:4}/{len(tails):<3} {over_any:3}/{len(rows):<3}"
        )
        if mute:
            lost = sum(c["seconds"] for c in mute)
            print(
                f"    SILENT ROWS {label}: {len(mute)}/{len(chunks)} chunks rendered "
                f"no speech at all ({lost:.0f}s of text never spoken)"
            )
        if floors:
            # Within 12 dB of the threshold and the split stops being
            # trustworthy; say so rather than letting a clean-looking rate
            # stand on a measurement that could not have seen a gap.
            median_floor = sorted(floors)[len(floors) // 2]
            db = 20 * math.log10(max(median_floor, 1e-9))
            headroom = 20 * math.log10(SILENCE_RMS) - db
            if headroom < 12:
                print(
                    f"    WARNING {label}: median silence floor {db:.0f} dBFS is only "
                    f"{headroom:.0f} dB below the {20 * math.log10(SILENCE_RMS):.0f} dBFS "
                    f"threshold; gaps may be undercounted"
                )

    print("\nhead = silence opening a non-initial chunk; tail = silence closing one;")
    print("seam = tail+head across a join; int = largest run strictly inside a chunk.")
    print("A tail over 2s is a cap-hit row free-running silence, which belongs to no")
    print("seam when it is the last chunk. Rates are per seam, per chunk and per passage.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", type=Path, nargs="+", default=None)
    parser.add_argument("--label", default="")
    parser.add_argument("--voice", default="joe")
    parser.add_argument("--profile", default="", help="a .safetensors path, overrides --voice")
    parser.add_argument("--checkpoint", default="loudreader/loudr-1")
    parser.add_argument("--reading-set", type=Path, default=READING_SET)
    parser.add_argument("--passages", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--language",
        default="",
        help="read the corpus in this language regardless of the voice's own. "
        "Cross-lingual synthesis: it separates a voice's behaviour from "
        "the language's, which a same-language measurement cannot",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.compare:
        return compare(args.compare)
    if not args.label or not args.out:
        raise SystemExit("rendering needs --label and --out (or pass --compare)")
    return render(args)


if __name__ == "__main__":
    sys.exit(main())
