#!/bin/zsh
# Run the Go cancellation probe with this laptop's environment.
set -e
here=${0:a:h}
source "$here/env.sh"
cd "$LOUDKIT_WORKTREE"
exec out/cancel-go "$@"
