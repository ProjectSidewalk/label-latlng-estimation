"""Which host built a committed summary — recorded so a divergence is diagnosable.

Every `data/*-summary.json` claims to regenerate from committed inputs, and for the
closed-form and geometric artifacts that claim holds on any machine: they are linear
algebra and trigonometry over fixed rows, and they reproduce to 1e-9 across platforms.

The two LightGBM artifacts are the exception (issue #22). A booster chooses its splits at
histogram bin boundaries, so a last-bit difference between two platforms' `libm` — in
`arctan`, for the one variant that eats a trig-derived feature — is enough to move a
split and change a metric in its fifth decimal. That is orders of magnitude below anything
either #6 report claims, but it is not nothing, and until this module existed "which host
built this" had to be inferred from a `newline="\\n"` comment in a runner.

Deliberately dependency-light: it imports only what it reports on, and a library it cannot
import is omitted rather than fatal, so any runner can record its provenance without
taking on the benchmark-only dependencies of another.
"""

from __future__ import annotations

import importlib
import platform
import sys

# The libraries whose version can plausibly change a committed number. LightGBM is the one
# that provably has (#22); the other three sit under every numeric path in the repo.
LIBRARIES = ("numpy", "pandas", "scipy", "lightgbm")


def library_versions(names: tuple[str, ...] = LIBRARIES) -> dict:
    """`{name: version}` for each importable library, skipping the ones that are absent."""
    out = {}
    for name in names:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        out[name] = getattr(module, "__version__", "unknown")
    return out


def host_provenance(libraries: tuple[str, ...] = LIBRARIES) -> dict:
    """The environment this process is running in, as a JSON-safe dict.

    `python_build` is `sys.version` with its whitespace collapsed: on some builds it
    carries an embedded newline, which is legal JSON but turns one field into two lines of
    a diff for no gain. The short `python` version is kept alongside it because that is
    the one a reader compares against, and parsing it back out of the build string is
    exactly the kind of chore this record exists to remove.
    """
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_build": " ".join(sys.version.split()),
        "libraries": library_versions(libraries),
        "note": "recorded so a rerun that lands on different numbers is diagnosable "
                "rather than forensic (issue #22)",
    }
