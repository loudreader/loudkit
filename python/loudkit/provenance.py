"""loudkit provenance: what made this audio, carried beside it.

JUMBF-shaped boxes in a private RIFF chunk, holding one JSON assertion and a
SHA-256 that binds it to the file's audio payload. **This is not C2PA.** A
C2PA manifest is a signed manifest store, this signs nothing and stores one
assertion, and the two are not interchangeable however similar the boxes look.
Read it with ``loudkit verify``.

See ``docs/design/runtime-notes.md``.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ._version import package_version
from .contracts import Waveform
from .errors import ProvenanceError

__all__ = [
    "JUMBF_UUID",
    "CLAIM_UUID",
    "JSON_UUID",
    "build_manifest",
    "manifest_bytes",
    "write_wav",
    "read_provenance",
    "verify_provenance",
]

JUMBF_UUID = uuid.UUID("6a756d62-0011-0010-8000-00aa00389b71")
CLAIM_UUID = uuid.UUID("63327061-0011-0010-8000-00aa00389b71")
"""The type of the box holding the claim.

The value JUMBF registers for a C2PA store, kept because the box layout is
borrowed from there and changing it would move every recorded digest for no
reader's benefit. It labels a box; it does not make the file a C2PA asset.
"""
JSON_UUID = uuid.UUID("6a736f6e-0011-0010-8000-00aa00389b71")

DIGITAL_SOURCE_TYPE = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
"""The IPTC vocabulary value for AI-generated media, what ``c2pa.actions``
carries to say "a model made this"."""

MANIFEST_LABEL = "loudkit.provenance"
"""The label of the loudkit-specific assertion inside the manifest."""


def _box(box_uuid: uuid.UUID, payload: bytes) -> bytes:
    body = box_uuid.bytes + payload
    return struct.pack(">I", len(body) + 4) + body


def manifest_bytes(manifest: dict[str, object]) -> bytes:
    """The wire form of a manifest: ``jumbf { claim { json { ... } } }``."""
    return _box(
        JUMBF_UUID,
        _box(
            CLAIM_UUID,
            _box(
                JSON_UUID, json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
            ),
        ),
    )


def _stamped_now() -> str:
    """The `when` a manifest records, as its own function so a test can hold it.

    See ``docs/design/runtime-notes.md``.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_manifest(
    *,
    audio: bytes,
    algorithm_fingerprint: str,
    recipe_version: str,
    seed: int,
    sample_rate: int,
    voice: str = "",
    language: str = "",
    text: str = "",
    speed: float = 1.0,
    version: str = package_version(),
    voice_sha256: str = "",
    checkpoint_sha256: str = "",
    backend: str = "",
    execution: str = "",
) -> dict[str, object]:
    """One manifest for one rendered waveform.

    See ``docs/design/runtime-notes.md``.
    """
    when = _stamped_now()
    return {
        "claim_generator": f"loudkit {version}",
        "claim_generator_info": [{"name": "loudkit", "version": version}],
        "assertions": [
            {
                "label": "c2pa.actions",
                "data": {
                    "actions": [
                        {
                            "action": "c2pa.created",
                            "softwareAgent": "loudkit",
                            "digitalSourceType": DIGITAL_SOURCE_TYPE,
                            "when": when,
                        }
                    ]
                },
            },
            {
                "label": MANIFEST_LABEL,
                "data": {
                    "algorithm_fingerprint": algorithm_fingerprint,
                    "recipe_version": recipe_version,
                    "seed": seed,
                    "sample_rate": sample_rate,
                    "speed": speed,
                    "voice": voice,
                    "voice_profile_sha256": voice_sha256,
                    "checkpoint_sha256": checkpoint_sha256,
                    "backend": backend,
                    "execution": execution,
                    "language": language,
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest() if text else "",
                    "audio_sha256": hashlib.sha256(audio).hexdigest(),
                },
            },
        ],
    }


def write_wav(
    path: str | Path,
    audio: Waveform,
    sample_rate: int,
    *,
    manifest: bool = True,
    algorithm_fingerprint: str = "",
    recipe_version: str = "",
    seed: int = 0,
    voice: str = "",
    language: str = "",
    text: str = "",
    speed: float = 1.0,
    version: str = package_version(),
    voice_sha256: str = "",
    checkpoint_sha256: str = "",
    backend: str = "",
    execution: str = "",
) -> Path:
    """Write a 16-bit WAV; with ``manifest``, carry the provenance box in a chunk.

    The audio hash binds the manifest to the data chunk as written, and that
    chunk is untouched by the manifest: adding provenance changes the header's
    declared size and appends a chunk, and nothing else. Written beside the
    target and renamed onto it, so a render that dies halfway leaves the
    previous file or none.
    """
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The suffix is kept: soundfile picks the container from it.
    tmp = path.with_name(f".{path.stem}.partial{path.suffix}")
    try:
        sf.write(str(tmp), audio, sample_rate)
        if not manifest:
            os.replace(tmp, path)
            return path
        payload = _riff_audio_payload(tmp.read_bytes())
        claim = build_manifest(
            audio=payload,
            algorithm_fingerprint=algorithm_fingerprint,
            recipe_version=recipe_version,
            seed=seed,
            sample_rate=sample_rate,
            voice=voice,
            language=language,
            text=text,
            speed=speed,
            version=version,
            voice_sha256=voice_sha256,
            checkpoint_sha256=checkpoint_sha256,
            backend=backend,
            execution=execution,
        )
        tmp.write_bytes(_with_provenance_chunk(tmp.read_bytes(), manifest_bytes(claim)))
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


PROVENANCE_CHUNK_ID = b"LKPV"
"""The RIFF chunk the manifest lives in.

Inside the RIFF structure, because a reader walks the chunks a file declares
and bytes past the last one are not part of the file as far as the format is
concerned.

Not ``C2PA``. That identifier promises a signed manifest store, and a reader
that finds it is entitled to parse one; handed these boxes it reports a
malformed manifest rather than passing over a file it has no business reading.
A private identifier is ignored by everything that does not know it, which is
the correct behaviour for metadata only loudkit understands.
"""


def _with_provenance_chunk(data: bytes, box: bytes) -> bytes:
    """``data`` with ``box`` added as a provenance chunk, and RIFF resized.

    After the audio, so a player that streams the file reaches the samples
    without walking past the manifest first, and so the `data` payload keeps
    the offset it had. The payload is what the manifest's hash binds to and it
    is untouched here: only the header's declared size changes, by the length
    of the chunk that follows.
    """
    pad = len(box) & 1
    chunk = PROVENANCE_CHUNK_ID + struct.pack("<I", len(box)) + box + b"\x00" * pad
    out = bytearray(data + chunk)
    struct.pack_into("<I", out, 4, len(out) - 8)
    return bytes(out)


def _find_provenance_chunk(data: bytes) -> bytes:
    """The payload of the file's provenance chunk, or empty.

    Walks the chunks the header declares, which is the same walk every reader
    of this format makes, including the ones that are not ours.
    """
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return b""
    offset = 12
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        size = struct.unpack_from("<I", data, offset + 4)[0]
        if chunk_id == PROVENANCE_CHUNK_ID:
            return data[offset + 8 : offset + 8 + size]
        # Chunks are word-aligned: an odd size is followed by a pad byte that
        # is not counted in it. The same step Go, JS and Rust walk with.
        offset += 8 + size + (size & 1)
    return b""


def _find_trailing_boxes(data: bytes) -> bytes:
    """The bytes after the last RIFF chunk, which is where 0.1.0 wrote them.

    That release appended the boxes past the length the header declares, and
    every WAV it saved still carries them. This reader is for those files; new
    files carry the ``LKPV`` chunk instead.
    """
    if len(data) < 12 or data[:4] != b"RIFF":
        return b""
    offset = 12
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        size = struct.unpack_from("<I", data, offset + 4)[0]
        # Chunks are word-aligned: an odd size is followed by a pad byte that
        # is not counted in it. The same step Go, JS and Rust walk with.
        offset += 8 + size + (size & 1)
        if chunk_id == b"data":
            return data[offset:]
    return b""


def read_provenance(path: str | Path) -> dict[str, object] | None:
    """The manifest a loudkit WAV carries, or ``None`` for any other file.

    The chunk first, then the trailing boxes 0.1.0 wrote, so a file saved by
    either release reads. Only the chunk is written.
    """
    data = Path(path).read_bytes()
    trailer = _find_provenance_chunk(data) or _find_trailing_boxes(data)
    if not trailer:
        return None
    try:
        _, inner = _unbox(trailer, JUMBF_UUID)
        _, inner = _unbox(inner, CLAIM_UUID)
        _, payload = _unbox(inner, JSON_UUID)
    except (ValueError, struct.error) as exc:
        # A malformed box is not an unmarked file. Swallowing both into `None`
        # meant an auditor asking "is this labelled?" could not tell "no" from
        # "yes, and the label is damaged", and the second is the interesting
        # answer, because it is what tampering looks like.
        raise ProvenanceError(f"provenance box present but unreadable: {exc}") from exc
    try:
        manifest = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ProvenanceError(f"provenance payload is not JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ProvenanceError("provenance payload is not an object")
    return manifest


def _unbox(data: bytes, expected: uuid.UUID) -> tuple[bytes, bytes]:
    if len(data) < 8:
        raise ValueError("truncated box")
    size = struct.unpack_from(">I", data, 0)[0]
    if size < 8 or size > len(data):
        raise ValueError(f"box size {size} out of range")
    if uuid.UUID(bytes=data[4:20]) != expected:
        raise ValueError("unexpected box type")
    return data[size:], data[20:size]


def verify_provenance(path: str | Path) -> tuple[dict[str, object] | None, bool]:
    """Read the manifest and check that its audio hash matches the file.

    ``(manifest, ok)``, ``(None, False)`` for a file with no provenance. The
    check proves the manifest belongs to *these* bytes; it does not prove who
    wrote them, because the manifest is unsigned (see the module docstring).
    """
    manifest = read_provenance(path)
    if manifest is None:
        return None, False
    data = Path(path).read_bytes()
    assertion = manifest.get("assertions")
    declared = ""
    if isinstance(assertion, list) and assertion:
        # By label, not by position: the label is an assertion's identity and
        # the order the writer emitted them in is not significant.
        entry: dict[str, object] = {}
        for candidate in assertion:
            if isinstance(candidate, dict) and candidate.get("label") == MANIFEST_LABEL:
                entry = candidate
                break
        inner = entry.get("data")
        if isinstance(inner, dict):
            declared = str(inner.get("audio_sha256", ""))
    actual = hashlib.sha256(_riff_audio_payload(data)).hexdigest()
    return manifest, declared == actual


def _riff_audio_payload(data: bytes) -> bytes:
    """The PCM payload of a WAV, for the hash the manifest binds to."""
    if len(data) < 12 or data[:4] != b"RIFF":
        return data
    offset = 12
    while offset + 8 <= len(data):
        size = struct.unpack_from("<I", data, offset + 4)[0]
        if data[offset : offset + 4] == b"data":
            return data[offset + 8 : offset + 8 + size]
        # Chunks are word-aligned: an odd size is followed by a pad byte that
        # is not counted in it. The same step Go, JS and Rust walk with.
        offset += 8 + size + (size & 1)
    return b""
