#!/bin/zsh
# Compile and run the Swift probe against the release build's objects.
#
# The same link line tools/bench_ports/README.md documents, for the same
# reason: SwiftPM builds no executable for a file outside the manifest, and
# adding one to Package.swift would change the shipped package to run a probe.
set -e
here=${0:a:h}
source "$here/env.sh"
cd "$LOUDKIT_WORKTREE"
mkdir -p out
if [[ ! -x out/probe-swift || tools/decode_probe/probe.swift -nt out/probe-swift ]]; then
  swiftc -O -I .build/arm64-apple-macosx/release/Modules \
    tools/decode_probe/probe.swift \
    .build/arm64-apple-macosx/release/LoudKit.build/*.o \
    .build/arm64-apple-macosx/release/LoudKitText.build/*.o \
    -o out/probe-swift
fi
exec out/probe-swift "$@"
