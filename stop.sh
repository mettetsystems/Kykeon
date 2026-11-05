#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec "${KYKEON_PYTHON:-python3.14}" scripts/services.py stop
