"""Fixtures every test in this suite gets, whether it asks or not."""

from __future__ import annotations

import os
import tempfile

import pytest

# Set before numba is imported anywhere. librosa's mel/STFT helpers are
# numba-cached functions, and numba's default cache location is next to the
# *source* — inside site-packages. In any environment where site-packages is
# not writable (CI images, system installs, pip install --user with a read-only
# venv) the first enrollment test then fails with a cache write error that
# reads like a broken install. Pointing the cache at a temp directory costs a
# recompile per session and removes the failure class; honoured only when the
# caller has not already chosen a location.
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "numba-cache"))


@pytest.fixture(autouse=True)
def _hold_the_provenance_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Freeze the C2PA `when`, so byte-equality assertions mean what they say.

    The suite's central claim is that every transport returns byte for byte
    what calling the library returns — asserted for HTTP, gRPC and the
    OpenAI-compatible route. A rendered WAV carries a C2PA trailer, and the
    trailer carries a creation timestamp, which is the one value in the file
    that is not a function of the input.

    So those assertions were time-dependent. Two identical renders that
    straddled a second boundary differed by one digit at some offset deep in
    the trailer, and the test reported a parity failure in the synthesis path.
    Observed once in four full runs of the suite: `At index 11573 diff: b'9' !=
    b'8'`, on audio that was identical.

    Autouse, because the point is that no test has to remember. What is being
    checked is that two paths produce the same audio; the wall clock is not
    part of that and never was.
    """
    monkeypatch.setattr("loudkit.provenance._stamped_now", lambda: "2026-01-01T00:00:00+00:00")
