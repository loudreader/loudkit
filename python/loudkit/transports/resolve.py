"""Turning a ``serve`` argument into a release, once for all three doors.

Every transport's entry point takes the same pair -- a checkpoint reference
and an optional voice directory -- and has to answer the same two questions
in the same order before it can load anything: which file the reference
names, and where the voices beside it live. Three copies of that answer is
three places for it to drift, and it drifted twice already (a backend the
device did not need, and a local release directory whose voices were looked
for one level above it), which is what ``tests/test_transport_resolution.py``
exists to catch. Not a transport, like :mod:`.limits` and :mod:`.schemas`:
what the peers share. See ``docs/design/transports.md``.

The same holds for the one ``serve`` flag that is not a wire setting:
``--first-chunk-tokens`` has to be read off the resolved release and folded
into an :class:`~loudkit.config.AlgorithmConfig` before the engine loads, and
that is the same twelve lines in every door that offers the flag.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..synthesis import VoiceLibrary

if TYPE_CHECKING:
    from ..config import AlgorithmConfig

__all__ = ["algorithm_override", "open_release"]


def open_release(
    checkpoint: str | Path, voices: str | Path | None, device: str | None
) -> tuple[Path, VoiceLibrary]:
    """The checkpoint ``checkpoint`` names, and the voice library beside it.

    ``device`` picks which of a release's file sets the hub fetches, because a
    repo id with no backend named brings back the torch set: a snapshot with
    no graphs in it, which ``device="onnx"`` then cannot run on a download
    that looked perfect.

    A repo id resolves to the checkpoint *inside the snapshot the hub
    returned*, so the default voice directory is that snapshot's own
    ``voices/``. Every shape goes through the resolver, a file and a local
    release directory included, or a directory keeps its own name and its
    voices are looked for beside it rather than inside it.

    The hub import is deferred so importing a transport does not require the
    ``hub`` extra, and so the callers that never resolve never pay for it.
    """
    from ..hub import backend_for_device, resolve_checkpoint

    ckpt = resolve_checkpoint(str(checkpoint), backend=backend_for_device(device))
    return ckpt, VoiceLibrary(Path(voices) if voices else ckpt.parent / "voices")


def algorithm_override(ckpt: Path, first_chunk_tokens: int | None) -> AlgorithmConfig | None:
    """The release's algorithm with ``first_chunk_tokens`` cut into its chunking.

    ``None`` when nothing was overridden, which is what ``load`` reads as
    "take the manifest yourself": building the config here in that case would
    reach the same values by a longer route, and one more place to disagree.

    Takes the resolved checkpoint rather than the reference the caller typed.
    A repo id resolves through the hub, two resolves can land on two
    snapshots, and the manifest read for the override has to be the manifest
    the engine then loads.

    The imports are deferred for the reason :func:`open_release` defers the
    hub: a door that never overrides never pays for them.
    """
    if first_chunk_tokens is None:
        return None

    import dataclasses

    from ..checkpoint import read_manifest
    from ..config import AlgorithmConfig

    base = AlgorithmConfig.from_manifest(read_manifest(ckpt))
    return dataclasses.replace(
        base,
        chunking=dataclasses.replace(base.chunking, first_chunk_max_tokens=first_chunk_tokens),
    )
