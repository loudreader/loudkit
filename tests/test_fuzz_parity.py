"""The fuzzer distinguishes comparison failures from a broken harness."""

from __future__ import annotations

import pytest
from tools.fuzz_parity import divergence_count


@pytest.mark.parametrize(
    ("port", "output", "expected"),
    [
        ("go", "speechtext_test.go:42: mismatch\nspeechtext_test.go:42: mismatch", 2),
        ("js", "actual: 3", 3),
        ("rust", "shared fixture in 4/300 cases", 4),
        ("swift", "shared fixture in 5/300 cases", 5),
        ("go", "build failed: missing dependency", None),
    ],
)
def test_counts_only_reported_mismatches(port: str, output: str, expected: int | None) -> None:
    assert divergence_count(port, output) == expected
