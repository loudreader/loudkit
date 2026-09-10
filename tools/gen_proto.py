#!/usr/bin/env python3
"""Generate the gRPC stubs from `proto/loudkit.proto`.

    python tools/gen_proto.py

The output is committed, and `tests/test_grpc.py` regenerates it and compares,
the same rule the respelling lexicon follows. Generated code that is committed
without a check drifts from its source the first time someone edits the `.proto`
and forgets, and a stub that disagrees with the schema is a wire format nobody
declared.

`grpc_tools.protoc` writes `import loudkit_pb2` at the top of the `_grpc`
module, which only resolves if the output directory is on `sys.path`. Rewritten
to a package-relative import so the stubs work as `loudkit.proto.*` wherever the
package is installed.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROTO = ROOT / "proto" / "loudkit.proto"
OUT = ROOT / "python" / "loudkit" / "proto"


def main() -> int:
    try:
        import grpc_tools.protoc  # noqa: F401
    except ImportError:
        print('gen_proto needs grpcio-tools:\n  pip install "loudkit[grpc]"', file=sys.stderr)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    # `newline="\n"` here and below: the stubs are tracked, and
    # `tests/test_grpc.py` regenerates them to prove the checked-in copies are
    # what protoc writes. Text mode translates on Windows.
    (OUT / "__init__.py").write_text(
        '"""Generated gRPC stubs. Edit `proto/loudkit.proto` and run '
        '`python tools/gen_proto.py`."""\n',
        encoding="utf-8",
        newline="\n",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{PROTO.parent}",
            f"--python_out={OUT}",
            f"--pyi_out={OUT}",
            f"--grpc_python_out={OUT}",
            str(PROTO),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return result.returncode

    grpc_module = OUT / "loudkit_pb2_grpc.py"
    text = grpc_module.read_text(encoding="utf-8")
    fixed = re.sub(r"^import loudkit_pb2 as", "from . import loudkit_pb2 as", text, flags=re.M)
    if fixed != text:
        grpc_module.write_text(fixed, encoding="utf-8", newline="\n")
    print(f"wrote {len(list(OUT.glob('loudkit_pb2*')))} files to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    # In the entry point, not in `main`, which the suite calls as a function.
    # Both paths this writes are fixed, so there is nothing to choose and any
    # argument is a misreading -- but parsing is what makes `--help` print
    # help: a generator that reads no argv answers every question by
    # regenerating its output.
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()
    raise SystemExit(main())
