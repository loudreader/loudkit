#!/usr/bin/env bash
# Close the three silence classes at scale, one voice at a time.
#
# For each voice: render the shipped profile and the repaired one over the same
# per-language reading set, at a passage count high enough that a rate means
# something, and record head / seam / interior separately. A single max-gap
# number per render conflates a seam artifact with a decoder stall, and the two
# respond differently to a fix.
#
# Reference point, from joe at 200 passages / 844 chunks / 644 seams:
#   passages with any gap over 1s   25.0% -> 4.5%
#   seams over 1s                    6.1% -> 1.1%
#   interiors over 1s                2.7% -> 0.2%
# A repaired voice should land in that shape, and its natural pause band
# (0.4-0.8s) should get denser rather than thinner.
#
# Usage:
#   research/run_silence_scale.sh <voice> <lang> <repaired-profile.safetensors> [passages]
#
# Streams are meant to be run a few at a time, not all ten: rendering is CPU
# bound and oversubscribing the box makes every stream slower without
# finishing any of them sooner.
set -u

VOICE=${1:?voice name, e.g. nils}
LANG_CODE=${2:?language code, e.g. sv}
FIXED=${3:?path to the repaired .safetensors}
PASSAGES=${4:-120}

PY=${PY:-.venv/bin/python}
# Repository-relative, like OUT below. The default was a session scratchpad on
# one machine, so the script ran there and nowhere else; the reading sets are
# built by `research/build_reading_set.py` and land wherever it is pointed.
SETS=${SETS:-out/scale}
OUT=${OUT:-out/chunkstats}
SET_FILE="$SETS/reading-$LANG_CODE.json"

if [ ! -f "$SET_FILE" ]; then
  echo "no reading set for $LANG_CODE at $SET_FILE" >&2
  exit 1
fi
if [ ! -f "$FIXED" ]; then
  echo "no repaired profile at $FIXED" >&2
  exit 1
fi

mkdir -p "$OUT"

echo "=== $VOICE ($LANG_CODE), $PASSAGES passages, shipped $(date -Is) ==="
$PY research/chunk_silence_stats.py --label "$VOICE-shipped" --voice "$VOICE" \
  --reading-set "$SET_FILE" --passages "$PASSAGES" \
  --out "$OUT/$VOICE-shipped.jsonl" || exit 1

echo "=== $VOICE ($LANG_CODE), repaired $(date -Is) ==="
$PY research/chunk_silence_stats.py --label "$VOICE-repaired" --profile "$FIXED" \
  --reading-set "$SET_FILE" --passages "$PASSAGES" \
  --out "$OUT/$VOICE-repaired.jsonl" || exit 1

echo "=== $VOICE result $(date -Is) ==="
$PY research/chunk_silence_stats.py --compare \
  "$OUT/$VOICE-shipped.jsonl" "$OUT/$VOICE-repaired.jsonl"
