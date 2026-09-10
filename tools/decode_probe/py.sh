#!/bin/zsh
# Run the Python probe with this laptop's environment. Arguments go straight
# through to probe.py.
set -e
here=${0:a:h}
source "$here/env.sh"
exec "$PYTHON" "$here/probe.py" "$@"
