"""Compare the probe records five ports wrote, stage by stage.

Stops reporting at the first stage that disagrees, because everything
downstream of a divergence is noise: a token that differs at step 40 makes
every mel frame and every sample after it different for a reason that is not
the mel decoder's or the vocoder's.

    python tools/decode_probe/compare.py OUTDIR

Reports, per stage: which ports agree, the first divergent index, and what
each port said there.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

PORTS = ["python-onnx", "go", "rust", "js", "swift"]
# The order a comparison must walk. Everything after the first disagreement is
# a consequence of it, not an independent finding.
STAGES = ["text_tokens", "speech_tokens_raw", "speech_tokens", "mel", "audio"]


def read_f32(path: Path) -> list[float]:
    b = path.read_bytes()
    return list(struct.unpack(f"<{len(b) // 4}f", b))


def first_diff_int(a: list[int], b: list[int]) -> int | None:
    # Not strict: a length difference is one of the answers this looks for,
    # and the caller is told about it by the return below.
    for i, (x, y) in enumerate(zip(a, b)):  # noqa: B905
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def first_diff_f32(a: list[float], b: list[float], *, tol: float = 0.0) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):  # noqa: B905, see first_diff_int
        if abs(x - y) > tol:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def ulps(x: float, y: float) -> int:
    """Distance in representable float32 steps, for judging a last-float gap."""
    ax = struct.unpack("<i", struct.pack("<f", x))[0]
    ay = struct.unpack("<i", struct.pack("<f", y))[0]
    if ax < 0:
        ax = -2147483648 - ax
    if ay < 0:
        ay = -2147483648 - ay
    return abs(ax - ay)


def compare_tokens(stage: str, records: dict, ref: str) -> bool:
    """One token stage across the ports. True when they disagree.

    Its own function rather than a branch of `main`: the float stages bind the
    same names to `list[float]`, and one name for two element types is how a
    comparison silently reads one row as the other.
    """
    rows: dict[str, list[int]] = {}
    for port, record in records.items():
        row = record.get(stage)
        if row is not None:
            rows[port] = [int(x) for x in row]
    if len(rows) < 2:
        return False
    ref_row = rows[ref] if ref in rows else next(iter(rows.values()))
    print(f"[{stage}] lengths: " + ", ".join(f"{k}={len(r)}" for k, r in rows.items()))
    bad: dict[str, int] = {}
    for port, row in rows.items():
        if port == ref:
            continue
        at = first_diff_int(ref_row, row)
        if at is not None:
            bad[port] = at
    if not bad:
        print(f"    all {len(rows)} ports agree")
        return False
    for port, at in bad.items():
        a = ref_row[at] if at < len(ref_row) else None
        b = rows[port][at] if at < len(rows[port]) else None
        print(f"    first divergent step {at}: {ref}={a} {port}={b}")
        lo = max(0, at - 3)
        print(f"      {ref:<12} [...{ref_row[lo : at + 4]}...]")
        print(f"      {port:<12} [...{rows[port][lo : at + 4]}...]")
    print(f"\n  -> stopping at [{stage}]: everything downstream is a consequence.\n")
    return True


def main() -> int:  # noqa: PLR0912, PLR0915, one linear pass per stage, read in order
    outdir = Path(sys.argv[1])
    records: dict[str, dict] = {}
    for p in PORTS:
        f = outdir / f"{p}.json"
        if f.exists():
            records[p] = json.loads(f.read_text())
        else:
            print(f"{p}: no record ({f} missing)")
    if len(records) < 2:
        print("need at least two records to compare")
        return 1
    print(f"\nports present: {', '.join(records)}")
    ref = "python-onnx" if "python-onnx" in records else next(iter(records))
    print(f"reference: {ref}\n")

    for r in records.values():
        print(f"  {r['port']:<12} decode={r.get('decode')} loaded={r.get('loaded_from')}")
    print()

    diverged = False
    for stage in STAGES:
        if stage in ("mel", "audio"):
            vals = {}
            for p in records:
                f = outdir / f"{p}.{stage}.bin"
                if f.exists():
                    vals[p] = read_f32(f)
            if len(vals) < 2:
                continue
            base = vals[ref] if ref in vals else next(iter(vals.values()))
            shas = {p: records[p].get(stage, {}).get("sha", "") for p in vals}
            print(f"[{stage}] lengths: " + ", ".join(f"{p}={len(v)}" for p, v in vals.items()))
            print(f"[{stage}] sha groups:")
            groups: dict[str, list[str]] = {}
            for p, s in shas.items():
                groups.setdefault(s[:16], []).append(p)
            for s, ps in groups.items():
                print(f"    {s}  {', '.join(ps)}")
            if len(groups) > 1:
                diverged = True
                for p, v in vals.items():
                    if p == ref:
                        continue
                    i = first_diff_f32(base, v)
                    if i is None:
                        continue
                    if i < len(base) and i < len(v):
                        u = ulps(base[i], v[i])
                        print(
                            f"    first differing index {i}: {ref}={base[i]!r} "
                            f"{p}={v[i]!r} ({u} ulp)"
                        )
                    else:
                        print(f"    lengths differ at {i}: {ref}={len(base)} {p}={len(v)}")
                # A stage that disagrees makes every later stage a consequence.
                print(
                    f"\n  -> stopping at [{stage}]: everything downstream is a consequence.\n"
                )
                break
            print()
            continue

        if compare_tokens(stage, records, ref):
            diverged = True
            break
        print()

    # The long-form path is measured separately: it exercises chunking, the
    # prefix carry and the retry ladder, none of which the single-window
    # ladder above touches.
    print("[longform]")
    for p, record in records.items():
        lf = record.get("longform")
        if not lf:
            continue
        print(
            f"  {p:<12} tokens={len(lf['tokens']):<5} chunks={lf['n_chunks']:<3} "
            f"samples={lf['audio_len']:<8} sha={lf['audio_sha'][:16]} cap={lf['hit_token_cap']}"
        )
    lfs = {p: r["longform"] for p, r in records.items() if r.get("longform")}
    if len(lfs) >= 2 and ref in lfs:
        for p, lf in lfs.items():
            if p == ref:
                continue
            i = first_diff_int(lfs[ref]["tokens"], lf["tokens"])
            if i is not None:
                a = lfs[ref]["tokens"]
                print(
                    f"  token divergence {ref} vs {p} at step {i}: "
                    f"{a[i] if i < len(a) else None} vs "
                    f"{lf['tokens'][i] if i < len(lf['tokens']) else None}"
                )

    return 1 if diverged else 0


if __name__ == "__main__":
    raise SystemExit(main())
