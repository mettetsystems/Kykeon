#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python_bin="${KYKEON_PYTHON:-python3.14}"
command -v "$python_bin" >/dev/null || { echo "Install Python 3.14 or set KYKEON_PYTHON to its executable." >&2; exit 1; }
command -v node >/dev/null || { echo "Install Node.js 22+ first." >&2; exit 1; }
command -v npm >/dev/null || { echo "Install npm first." >&2; exit 1; }
node -e 'if (Number(process.versions.node.split(".")[0]) < 22) { console.error("Node.js 22+ is required."); process.exit(1); }'
if [[ ! -d .venv ]]; then
  "$python_bin" -m venv .venv
fi
[[ -x .venv/bin/python ]] || { echo "Existing .venv is incomplete. Repair it before continuing." >&2; exit 1; }
mkdir -p .kykeon/models .kykeon/artifacts .kykeon/run .kykeon/logs
printf '%s\n' 'Setup ready. Run make install to install dependencies.'
