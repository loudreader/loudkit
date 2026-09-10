#!/bin/zsh
# Run an arbitrary python script with this laptop's loudkit environment.
# The first argument is the script; the rest go through to it.
set -e
here=${0:a:h}
source "$here/env.sh"
exec "$PYTHON" "$@"
