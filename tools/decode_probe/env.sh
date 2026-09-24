#!/bin/zsh
# The environment every probe in this directory needs.
#
# The runner scripts source this file, so the paths live in one place. The Go
# binding needs ORT API 28, which is ONNX Runtime 1.28 or newer; the Rust `ort`
# build refuses anything older than 1.27. The default Rust library below is
# 1.27, so it does not work for Go. A probe that picked the wrong one would fail
# in a way that looks like a port defect. The defaults below are macOS paths.
#
# Override any of these from the caller; each assignment keeps a value that is
# already set.
# The checkout this file is in, unless the caller names another. A worktree
# shares its assets and its virtualenv with the checkout it was cut from, so
# LOUDKIT_REPO is the place those live and LOUDKIT_WORKTREE is the code under
# test; they are the same directory unless you are working in a worktree.
: ${LOUDKIT_REPO:=$(cd -- "$(dirname -- "$0")/../.." && git rev-parse --path-format=absolute --git-common-dir 2>/dev/null | sed 's|/\.git$||' || pwd)}
: ${LOUDKIT_WORKTREE:=$(cd -- "$(dirname -- "$0")/../.." && pwd)}
: ${LOUDKIT_ASSET_ROOT:=$LOUDKIT_REPO/assets}
: ${LOUDKIT_ONNX_DIR:=$LOUDKIT_ASSET_ROOT/onnx}
# Go and Python: the binding wants ORT API 28.
: ${LOUDKIT_ONNXRUNTIME_LIB:=$LOUDKIT_REPO/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.1.29.0.dylib}
# Rust: `ort` feature api-27 refuses a dylib older than 1.27.
: ${ORT_DYLIB_PATH:=$LOUDKIT_REPO/js/node_modules/onnxruntime-node/bin/napi-v6/darwin/arm64/libonnxruntime.1.27.0.dylib}
: ${LOUDKIT_TOKENIZER:=$LOUDKIT_ASSET_ROOT/tokenizer.json}
: ${PYTHON:=$LOUDKIT_REPO/.venv/bin/python}
: ${PYTHONPATH:=$LOUDKIT_WORKTREE/python}

export LOUDKIT_REPO LOUDKIT_WORKTREE LOUDKIT_ASSET_ROOT LOUDKIT_ONNX_DIR
export LOUDKIT_ONNXRUNTIME_LIB ORT_DYLIB_PATH LOUDKIT_TOKENIZER PYTHON PYTHONPATH
