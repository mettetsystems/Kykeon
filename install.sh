#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
./setup.sh
npm ci
.venv/bin/python -m pip install 'torch==2.15.0+cu132' --index-url https://download.pytorch.org/whl/cu132
.venv/bin/python -m pip install -r backend/requirements.txt --upgrade-strategy only-if-needed
printf '%s\n' 'Dependencies installed. Run make start.'
