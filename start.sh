#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
[[ -x .venv/bin/python ]] || { echo "Run make install first." >&2; exit 1; }
exec .venv/bin/python scripts/services.py start
