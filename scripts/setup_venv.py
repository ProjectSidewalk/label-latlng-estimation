"""Create (or refresh) the Python environment every analysis in this repo runs in.

Works the same on macOS, WSL/Linux and native Windows -- it is Python rather than shell for
exactly that reason, since this repo is developed across all three.

Usage (from the repo root)::

    python scripts/setup_venv.py                 # build .venv from python/requirements.txt
    python scripts/setup_venv.py --recreate      # discard and rebuild it
    python scripts/setup_venv.py --venv .venv-wsl  # a second env beside the first

The last form matters if one checkout is shared between native Windows and WSL: a venv
records absolute interpreter paths and links against that platform's binaries, so the two
cannot share one directory. Give each its own (``.venv*`` is gitignored).

Why a venv at all, rather than ``pip install`` into whatever Python is on PATH: the
summaries under ``data/`` are claimed to regenerate from committed inputs, and that claim is
only as strong as the toolchain behind it. See "Python environment" in README.md, and
issue #22 for a measured case where the toolchain was not enough.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUIREMENTS = os.path.join(ROOT, "python", "requirements.txt")
MIN_PYTHON = (3, 10)

# Imported by name, reported by version. PIL is Pillow's import name; every other entry
# matches its distribution.
PACKAGES = ["numpy", "pandas", "scipy", "statsmodels", "pyproj", "matplotlib", "PIL",
            "lightgbm", "streetlevel", "pytest"]

IS_WINDOWS = os.name == "nt"


def venv_python(venv_dir: str) -> str:
    """The interpreter inside a venv. Windows puts it in Scripts/, POSIX in bin/."""
    return (os.path.join(venv_dir, "Scripts", "python.exe") if IS_WINDOWS
            else os.path.join(venv_dir, "bin", "python"))


def activation_hint(venv_dir: str) -> str:
    """How to activate it in the shell the user is most likely standing in."""
    rel = os.path.relpath(venv_dir, ROOT)
    if IS_WINDOWS:
        return (f"    {rel}\\Scripts\\Activate.ps1        # PowerShell\n"
                f"    {rel}\\Scripts\\activate.bat        # cmd.exe")
    return f"    source {rel}/bin/activate"


def say(message: str) -> None:
    """Print in step with the subprocesses, which write straight to the terminal."""
    print(message, flush=True)


def run(cmd: list[str]) -> None:
    """Run a subprocess, failing loudly with the command that failed."""
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"\n!! command failed ({result.returncode}): {' '.join(cmd)}")


def openmp_hint() -> str:
    """Platform-specific advice for the one dependency that installs and then won't load.

    LightGBM's wheels link against an OpenMP runtime that pip does not ship. On macOS that
    runtime is almost never present by default, which produces a dlopen failure that reads
    like a broken install rather than a missing system library.
    """
    system = platform.system()
    if system == "Darwin":
        return "brew install libomp"
    if system == "Linux":
        return ("install your distro's OpenMP runtime, e.g. "
                "`sudo apt-get install libgomp1` on Debian/Ubuntu/WSL")
    return ("install the Microsoft Visual C++ Redistributable (x64) — "
            "https://aka.ms/vs/17/release/vc_redist.x64.exe")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--venv", default=os.path.join(ROOT, ".venv"),
                    help="directory to create (default: .venv at the repo root)")
    ap.add_argument("--recreate", action="store_true",
                    help="delete an existing environment first")
    args = ap.parse_args()
    venv_dir = os.path.abspath(args.venv)

    if sys.version_info < MIN_PYTHON:
        raise SystemExit(f"!! this repo needs Python >= {'.'.join(map(str, MIN_PYTHON))}; "
                         f"this is {platform.python_version()}")

    if args.recreate and os.path.isdir(venv_dir):
        say(f"==> removing {venv_dir}")
        shutil.rmtree(venv_dir)

    if os.path.isdir(venv_dir):
        say(f"==> reusing existing {venv_dir}")
    else:
        say(f"==> creating {venv_dir} with Python {platform.python_version()} "
            f"on {platform.system()} {platform.machine()}")
        run([sys.executable, "-m", "venv", venv_dir])

    python = venv_python(venv_dir)
    say("==> installing python/requirements.txt")
    run([python, "-m", "pip", "install", "--upgrade", "--quiet", "pip"])
    run([python, "-m", "pip", "install", "--quiet", "-r", REQUIREMENTS])

    say("==> versions")
    report = (
        "import importlib, sys\n"
        "print('  {:<12}{}'.format('python', sys.version.split()[0]))\n"
        f"for name in {PACKAGES!r}:\n"
        "    try:\n"
        "        mod = importlib.import_module(name)\n"
        "    except Exception as exc:\n"
        "        print('  {:<12}FAILED TO IMPORT ({})'.format(name, type(exc).__name__))\n"
        "        continue\n"
        "    print('  {:<12}{}'.format(name, getattr(mod, '__version__', 'unknown')))\n"
    )
    run([python, "-c", report])

    if subprocess.run([python, "-c", "import lightgbm"], cwd=ROOT,
                      capture_output=True).returncode != 0:
        say(f"""
!! lightgbm installed but its native library will not load.
!! Fix: {openmp_hint()}
!! It is benchmark-only (issue #6): everything except run_gbm_ceiling.py and
!! run_gbm_transfer.py runs without it.""")

    say(f"""
Done. Activate it with:

{activation_hint(venv_dir)}

Then `pytest` should report 463 passed, 4 skipped (the skips are RUN_SLOW=1 re-derivations).""")


if __name__ == "__main__":
    main()
