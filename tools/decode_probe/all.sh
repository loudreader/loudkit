#!/bin/zsh
# Run every port's probe on the same input, into one directory.
#
#   tools/decode_probe/all.sh TEXT SEED LANGUAGE VOICE OUTDIR
#
# Each probe prints the module or binary it actually loaded before it measures:
# a path that silently resolves to a different checkout is the failure mode
# this whole harness exists to rule out.
set -e
here=${0:a:h}
source "$here/env.sh"
cd "$LOUDKIT_WORKTREE"

text=$1; seed=$2; language=$3; voice=$4; outdir=$5
if [[ -z "$outdir" ]]; then
  echo "usage: all.sh TEXT SEED LANGUAGE VOICE OUTDIR" >&2
  exit 2
fi
voicepath=$LOUDKIT_ASSET_ROOT/voices/$voice.safetensors
mkdir -p "$outdir"

run() {
  local name=$1; shift
  echo "=== $name ===" >&2
  # The status is the command's, not the pager's: a pipeline reports its last
  # stage, so piping into tail made the failure branch unreachable and an
  # unbuilt probe vanished into a run where the rest "agreed".
  local log
  log=$(mktemp)
  if "$@" >"$log" 2>&1; then
    tail -4 "$log" >&2
  else
    tail -4 "$log" >&2
    echo "$name FAILED" >&2
  fi
  rm -f "$log"
}

run python "$here/py.sh" "$LOUDKIT_ASSET_ROOT" "$voicepath" "$text" "$seed" "$language" "$outdir" --device onnx
run go out/probe-go "$LOUDKIT_ASSET_ROOT" "$voicepath" "$text" "$seed" "$language" "$outdir"
run rust rust/target/release/examples/probe "$LOUDKIT_ASSET_ROOT" "$voicepath" "$text" "$seed" "$language" "$outdir"
run js node tools/decode_probe/probe.mjs "$LOUDKIT_ASSET_ROOT" "$voicepath" "$text" "$seed" "$language" "$outdir"
run swift out/probe-swift "$LOUDKIT_ASSET_ROOT" "$voicepath" "$text" "$seed" "$language" "$outdir"

echo "--- comparing ---" >&2
"$PYTHON" "$here/compare.py" "$outdir"
