#!/usr/bin/env python3
"""``tools/release --model loudr-1``, under the name RELEASING.md and CI use."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from release import shim

if __name__ == "__main__":
    raise SystemExit(shim("tools/build_release.py", "loudr-1", sys.argv[1:]))
