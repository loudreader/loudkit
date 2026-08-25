"""The text funnel: everything between raw text and the tokens the model reads.

One recipe, five implementations. ``TextConfig.recipe`` and the grammar digest
pin what this package does into the algorithm fingerprint, so two builds that
report the same sixteen hex digits hand the model the same string.

Import the passes from their submodules directly (``loudkit.frontend.numbers``,
``loudkit.frontend.polish``, ...). The package ``__init__`` stays import-free on
purpose: ``loudkit.config`` and this package's chunking pass are mutually
dependent by design, and an eager init would turn that into a circular import.
"""

from __future__ import annotations
