"""Mechanics shared by the two batch-scaling benchmarks.

`bench_batch.py` measures the token generator and `bench_render.py` measures the
renderer. They ask different questions of different modules, but they take the
same command line and they both have to drain a device queue before they read a
clock. Those pieces were copied into both files, and one copy had already
drifted: `bench_batch.py` synchronised CUDA and not MPS, and nothing in it
refuses a non-CUDA device, so pointing it at MPS timed how long the work took to
enqueue rather than how long it took to run. Held once here, so the next fix
lands in both.

Not a package. research/ is a directory of scripts run from the repository root,
and each one finds this module because the interpreter puts the script's own
directory first on `sys.path`.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch


def sync(device: str) -> None:
    """Make the device finish what was queued, so a wall time is a wall time.

    MPS as well as CUDA, including for a tool aimed at CUDA: neither script
    refuses another device, and a queue that is never drained reports the time
    to enqueue the work rather than the time to do it. CPU is already
    synchronous, and torch.cuda would fail on a CPU-only build.
    """
    kind = device.split(":", maxsplit=1)[0]
    if kind == "cuda":
        torch.cuda.synchronize()
    elif kind == "mps":
        torch.mps.synchronize()


def batch_list(value: str) -> list[int]:
    try:
        return [int(x) for x in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"batches must be a comma list of integers: {exc}"
        ) from exc


def parser(
    description: str | None, device_help: str, default_batches: Sequence[int]
) -> argparse.ArgumentParser:
    """The command line both benchmarks take: four positionals and a batch list.

    The description and the device help stay the caller's, because the two tools
    document different devices; so does the default batch list, because the
    generator is worth measuring out to 64 and the renderer is not.
    """
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("checkpoint")
    ap.add_argument("voice")
    ap.add_argument("device", help=device_help)
    ap.add_argument("outdir", type=Path)
    ap.add_argument(
        "batches",
        nargs="?",
        type=batch_list,
        # A string default goes through `type` like a command-line value, so
        # the default and an explicit argument take exactly the same path.
        default=",".join(map(str, default_batches)),
        help="comma list of batch sizes (default: %(default)s)",
    )
    return ap
