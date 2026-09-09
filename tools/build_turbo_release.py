#!/usr/bin/env python3
"""``tools/release --model turbo``, under the name RELEASING.md uses."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from release import shim

if __name__ == "__main__":
    raise SystemExit(shim("tools/build_turbo_release.py", "turbo", sys.argv[1:]))
