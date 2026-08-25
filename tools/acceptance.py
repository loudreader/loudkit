#!/usr/bin/env python3
"""Install loudkit the way a stranger does, and make it speak.

RELEASING.md §8 asks for this in prose: a machine that never saw the repository,
`pip install loudkit`, `download`, `speak`, `verify`. Prose is a hope. This is
the same steps as a gate, so the answer is a exit code rather than a memory of
having tried it.

The one property that makes it a clean room, and the only one worth writing a
script for: it proves the package under test came from the *installed*
distribution and not from the checkout sitting next to it. A venv created
inside a source tree will happily import that tree, and then the gate passes
while shipping a wheel that is missing half its data files. So the run
refuses to start unless `loudkit.__file__` resolves inside the new venv's
site-packages.

Two modes, because the gate is worth having before the release as well as
after:

    python tools/acceptance.py --wheel dist/loudkit-0.1.0-py3-none-any.whl
    python tools/acceptance.py --from-pypi

Weights are large, so the download step is opt-in:

    python tools/acceptance.py --wheel dist/loudkit-0.1.0-py3-none-any.whl --speak
    python tools/acceptance.py --from-pypi --repo loudreader/loudr-1 --speak

`--speak` installs the README runtime extras (`torch,audio,hub`) when
`--extras` is omitted. Pass `--extras` to test another supported runtime.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class GateError(RuntimeError):
    """A gate said no. The message is the report."""


def run(cmd: list[str], *, cwd: Path, env_note: str = "") -> str:
    """Run a command, returning stdout, raising with both streams on failure."""
    # check=False because the raise below carries both streams; CalledProcessError
    # would surface the exit code and drop the output that says why.
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise GateError(
            f"{env_note or cmd[0]} exited {proc.returncode}\n"
            f"$ {' '.join(cmd)}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return proc.stdout


def make_venv(root: Path) -> tuple[Path, Path]:
    """A fresh interpreter, outside the repository. Returns (python, bin dir)."""
    venv.create(root, with_pip=True, clear=True)
    bindir = root / ("Scripts" if sys.platform == "win32" else "bin")
    python = bindir / ("python.exe" if sys.platform == "win32" else "python")
    if not python.exists():
        raise GateError(f"the new venv has no interpreter at {python}")
    return python, bindir


def install(python: Path, spec: str, cwd: Path) -> None:
    run([str(python), "-m", "pip", "install", "--upgrade", "pip"], cwd=cwd, env_note="pip")
    run([str(python), "-m", "pip", "install", spec], cwd=cwd, env_note=f"pip install {spec}")


def assert_imported_from_the_venv(python: Path, root: Path, cwd: Path) -> Path:
    """The clean-room check itself.

    Without this the script is theatre: run it from the repository and Python
    imports `python/loudkit` off the path, so a wheel that ships no data files
    passes every step below.
    """
    probe = (
        "import json, loudkit;"
        "print(json.dumps({'file': loudkit.__file__, 'version': loudkit.__version__}))"
    )
    out = run([str(python), "-c", probe], cwd=cwd, env_note="import loudkit")
    info = json.loads(out.strip().splitlines()[-1])
    imported = Path(info["file"]).resolve()
    if not imported.is_relative_to(root.resolve()):
        raise GateError(
            f"loudkit was imported from {imported}, which is outside the test venv "
            f"at {root}. This is not a clean room: the checkout is shadowing the "
            "installed package, and every check after this one would be measuring "
            "the wrong code."
        )
    if imported.is_relative_to(REPO_ROOT):
        raise GateError(f"loudkit was imported from the checkout at {imported}")
    print(f"  import  {info['version']} from {imported}")
    return imported


def check_cli(bindir: Path, cwd: Path) -> None:
    exe = bindir / ("loudkit.exe" if sys.platform == "win32" else "loudkit")
    if not exe.exists():
        raise GateError(f"the distribution installed no `loudkit` entry point at {exe}")
    run([str(exe), "--version"], cwd=cwd, env_note="loudkit --version")
    # doctor reports an environment; it must run without weights and without
    # the repository, which is exactly the state a new user is in.
    print("  doctor  " + run([str(exe), "doctor"], cwd=cwd).strip().splitlines()[-1])


def check_speaks(bindir: Path, cwd: Path, repo: str) -> None:
    exe = bindir / ("loudkit.exe" if sys.platform == "win32" else "loudkit")
    run([str(exe), "download", repo], cwd=cwd, env_note="loudkit download")
    wav = cwd / "acceptance.wav"
    run(
        [
            str(exe),
            "speak",
            "--checkpoint",
            repo,
            "--voice",
            "joe",
            "--out",
            str(wav),
            "The clean room speaks.",
        ],
        cwd=cwd,
        env_note="loudkit speak",
    )
    if not wav.exists() or wav.stat().st_size < 1024:
        raise GateError(f"`loudkit speak` wrote no usable audio at {wav}")
    run([str(exe), "verify", str(wav)], cwd=cwd, env_note="loudkit verify")
    print(f"  speak   {wav.stat().st_size} bytes, verified")


def requested_extras(extras: str, *, speak: bool) -> str:
    """Choose a runnable install for the opt-in synthesis gate."""
    if extras:
        return extras
    return "torch,audio,hub" if speak else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--wheel", type=Path, help="a locally built wheel, for a dry run")
    source.add_argument(
        "--from-pypi", action="store_true", help="install the published loudkit"
    )
    ap.add_argument("--extras", default="", help='e.g. "torch,audio,hub"')
    ap.add_argument("--speak", action="store_true", help="also download weights and synthesise")
    ap.add_argument("--repo", default="loudreader/loudr-1", help="weights repo for --speak")
    ap.add_argument("--keep", action="store_true", help="leave the scratch directory behind")
    args = ap.parse_args()

    selected_extras = requested_extras(args.extras, speak=args.speak)
    extras = f"[{selected_extras}]" if selected_extras else ""
    if args.wheel:
        wheel = args.wheel.resolve()
        if not wheel.exists():
            print(f"no wheel at {wheel}", file=sys.stderr)
            return 2
        spec = f"{wheel}{extras}"
    else:
        spec = f"loudkit{extras}"

    # Outside the repository on purpose: a scratch directory under the checkout
    # puts `python/` on sys.path for anything run with cwd there.
    scratch = Path(tempfile.mkdtemp(prefix="loudkit-acceptance-"))
    print(f"clean room: {scratch}")
    try:
        python, bindir = make_venv(scratch / "venv")
        print(f"  install {spec}")
        install(python, spec, cwd=scratch)
        assert_imported_from_the_venv(python, scratch / "venv", cwd=scratch)
        check_cli(bindir, cwd=scratch)
        if args.speak:
            check_speaks(bindir, cwd=scratch, repo=args.repo)
        else:
            print("  speak   skipped (pass --speak to download weights and synthesise)")
    except GateError as exc:
        print(f"\nFAILED\n{exc}", file=sys.stderr)
        return 1
    finally:
        if args.keep:
            print(f"kept: {scratch}")
        else:
            shutil.rmtree(scratch, ignore_errors=True)
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
