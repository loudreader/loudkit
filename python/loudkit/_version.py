"""The one place the package version is written.

A leaf, so the provenance writers can read it without importing the package root.
"""

__version__ = "0.1.1"


def package_version() -> str:
    """The version a provenance claim reports: the installed one, else the literal.

    Read here rather than written at each call site: a manifest that names a
    version the package is not is a false claim, and a literal default drifts
    at every bump.
    """
    try:
        from importlib.metadata import version

        return version("loudkit")
    except Exception:  # pragma: no cover - metadata present in any install
        return __version__
