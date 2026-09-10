#!/usr/bin/env bash
# Tier 2.5 end to end: render every system, pair them, judge, report.
#
# Every step is resumable, so re-running after an interruption or a failure
# continues rather than repeating. Set PILOT=1 for a 20-passage dry run on a
# cheap model before committing to the full spend.
#
#   PILOT=1 research/run_judge_chain.sh          # ~20 passages, a few dollars
#   research/run_judge_chain.sh                  # 400 passages, ~$13/comparator
#
# See docs/design/pairwise-judge.md for what the numbers mean and what they
# do not.
set -u

PY=${PY:-.venv/bin/python}
OUT=${OUT:-out/judge}
LOUDKIT_VOICE=${LOUDKIT_VOICE:-joe}
KOKORO_VOICE=${KOKORO_VOICE:-af_heart}
POCKETTTS_VOICE=${POCKETTTS_VOICE:-alba}
KITTEN_VOICE=${KITTEN_VOICE:-expr-voice-2-m}
# Piper needs a downloaded .onnx path, ElevenLabs a paid key: both off unless set.
PIPER_VOICE=${PIPER_VOICE:-}
ELEVENLABS_VOICE=${ELEVENLABS_VOICE:-}

if [ "${PILOT:-0}" = "1" ]; then
  LIMIT_ARG="--limit 20"
  JUDGE_MODEL=${JUDGE_MODEL:-google/gemini-3.7-flash}
else
  LIMIT_ARG=""
  JUDGE_MODEL=${JUDGE_MODEL:-google/gemini-3.1-pro-preview}
fi

mkdir -p "$OUT"

echo "=== render loudkit ($LOUDKIT_VOICE) $(date -Is) ==="
$PY research/render_comparators.py --system loudkit --voice "$LOUDKIT_VOICE" \
  --audio-root "$OUT/audio" $LIMIT_ARG || exit 1

COMPARATORS=""
render_one() {  # name, voice; a comparator that is not set up is skipped, not fatal
  [ -z "$2" ] && { echo "--- skipping $1 (no voice configured)"; return; }
  echo "=== render $1 ($2) $(date -Is) ==="
  if $PY research/render_comparators.py --system "$1" --voice "$2" \
      --audio-root "$OUT/audio" $LIMIT_ARG; then
    COMPARATORS="$COMPARATORS $1"
  else
    echo "--- $1 did not render; continuing without it" >&2
  fi
}

render_one kokoro "$KOKORO_VOICE"
render_one pockettts "$POCKETTTS_VOICE"
render_one kitten "$KITTEN_VOICE"
render_one piper "$PIPER_VOICE"
render_one elevenlabs "$ELEVENLABS_VOICE"

if [ -z "$COMPARATORS" ]; then
  echo "no comparator rendered; nothing to judge" >&2
  exit 1
fi

for system in $COMPARATORS; do
  echo "=== judge loudkit vs $system $(date -Is) ==="
  $PY research/render_comparators.py --manifest --system-b "$system" \
    --audio-root "$OUT/audio" $LIMIT_ARG \
    --out "$OUT/loudkit-vs-$system.manifest.jsonl" || exit 1
  $PY research/judge_pairwise.py \
    --manifest "$OUT/loudkit-vs-$system.manifest.jsonl" \
    --out "$OUT/loudkit-vs-$system.jsonl" \
    --system-b "$system" --model "$JUDGE_MODEL" || true
done

echo "=== report $(date -Is) ==="
# The manifests share the loudkit-vs- prefix; name the verdict files explicitly.
VERDICTS=""
for system in $COMPARATORS; do VERDICTS="$VERDICTS $OUT/loudkit-vs-$system.jsonl"; done
$PY research/judge_report.py $VERDICTS \
  --svg "$OUT/pairwise-judge.svg" --json "$OUT/summary.json" --losses
$PY research/judge_contact_sheet.py --results $VERDICTS \
  --audio-root "$OUT/audio" --out "$OUT/listen.html"
echo "=== DONE $(date -Is) ==="
