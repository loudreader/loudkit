"""What a step-distilled estimator says about itself, read the same way twice.

Two tools swap a distilled flow estimator into a model: `pack_turbo.py` bakes
it into a packed checkpoint, and `export_coreml.py` traces it into a CoreML
package. Both have to answer one question first: **how many Euler steps was
this estimator distilled for**. The recipe is explicit that running it
at any other K is a different algorithm rather than the same one at a different
speed, and neither the packed tensors nor the traced graph carries K: it lives
in the manifest, one layer away from the weights it governs.

One parser answers it for both, and the rule is the same on either side. K is
read from the estimator file wherever the file records it, under any of the
spellings below. Where it does not, the operator states it: `--n-cfm-timesteps`
in the packer, `--estimator-k` in the exporter, neither with a default. A file
and a flag that disagree are a refusal, not a preference. That order matters
because once the estimator is inside a packed checkpoint every later guard
reads the manifest, so a manifest that contradicts the weights it governs has
nothing left to check it against.

The parser is here rather than in
`loudkit.checkpoint` because it reads *research* checkpoints: torch pickles
from a training tree, in whatever shape that tree wrote them. Nothing at
runtime ever sees one.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from loudkit.checkpoint import file_sha256

__all__ = ["ESTIMATOR_K_KEYS", "estimator_k", "file_sha256"]

ESTIMATOR_K_KEYS = ("k_student", "n_cfm_timesteps", "euler_steps", "k")
"""Where a step-distill run might have written the K it was trained for.

Several spellings because the training side is a research tree, not this
repository, and its checkpoints predate anyone needing to read this back.
"""

_SCOPES = ("cfg", "config", "args", "hparams")
"""Blocks a training script parks its hyperparameters in, besides the top level."""


def _scopes(blob: Mapping[str, Any]) -> Iterator[tuple[str, Mapping[str, Any]]]:
    yield "", blob
    for name in _SCOPES:
        nested = blob.get(name)
        if isinstance(nested, Mapping):
            yield name, nested
        elif nested is not None and hasattr(nested, "__dict__"):
            yield name, vars(nested)


def estimator_k(blob: Mapping[str, Any]) -> int | None:
    """The Euler step count a distilled estimator was trained for, if it says.

    Returns ``None`` when the file records nothing, which is a checkpoint the
    caller has to have declared for it.

    **Every declaration is read, not the first one.** Returning the first hit
    made a file carrying `k = 1` at the top level and `cfg.k_student = 2`
    inside resolve to whichever the search order happened to reach, a
    checkpoint that contradicts itself answering as though it did not. A file
    that disagrees with itself has no K anyone can state, so it is refused,
    which is the same answer this project gives a manifest that contradicts
    itself.

    Raises:
        ValueError: if two declarations disagree, or if one is present and is
            not a positive integer -- `True`, `0`, a list. A malformed
            declaration is refused rather than skipped: skipping it is how a
            file carrying `k_student = 1` beside a nested `k_student = 0`
            answered 1, which is the self-contradiction this function exists to
            refuse, arriving through the one branch that did not look.
    """
    found: dict[str, int] = {}
    malformed: dict[str, object] = {}
    for scope, values in _scopes(blob):
        for key in ESTIMATOR_K_KEYS:
            if key not in values:
                continue
            value = values[key]
            name = f"{scope}.{key}" if scope else key
            # `True` is an `int` in Python and is not a step count. A present
            # declaration that is not a positive integer is *refused*, not
            # skipped: skipping it made a file carrying `k_student = 1` beside a
            # nested `k_student = 0` answer 1, which is the self-contradiction
            # this function exists to refuse, arriving through the one branch
            # that did not look.
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                malformed[name] = value
                continue
            found[name] = value
    if malformed:
        pairs = ", ".join(f"{k}={v!r}" for k, v in sorted(malformed.items()))
        raise ValueError(
            f"the checkpoint declares a Euler step count that is not a positive "
            f"integer ({pairs}). Fix the training checkpoint rather than reading "
            f"past it: everything downstream believes the manifest this number "
            f"goes into."
        )
    if not found:
        return None
    distinct = set(found.values())
    if len(distinct) > 1:
        pairs = ", ".join(f"{k}={v}" for k, v in sorted(found.items()))
        raise ValueError(
            f"the checkpoint declares more than one Euler step count ({pairs}). "
            f"A file that contradicts itself has no K anyone can state; fix the "
            f"training checkpoint rather than picking one of them here."
        )
    return distinct.pop()
