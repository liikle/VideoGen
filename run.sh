#!/usr/bin/env bash
# Start the VideoGen web UI. Extra arguments are passed through, e.g. ./run.sh --host 0.0.0.0
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Run ./install.sh first." >&2
  exit 1
fi
exec .venv/bin/python -m videogen serve "$@"
