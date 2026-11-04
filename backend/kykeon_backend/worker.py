from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .errors import CancelledJob
from .pipeline import run_spec
from .paths import atomic_json


def _emit(path: Path, event: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")
        handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    if spec.get("device_id") == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if (spec.get("test_hook") or {}).get("bad_result"):
        Path(spec["result_path"]).write_text("{", encoding="utf-8")
        return
    progress = Path(spec["progress_path"])

    def emit(event: dict) -> None:
        _emit(progress, event)

    try:
        result = run_spec(spec, emit)
        atomic_json(Path(spec["artifact_root"]), Path(spec["result_path"]), result)
    except CancelledJob as exc:
        emit({"event": "cancelled", "detail": str(exc)})
        sys.exit(2)
    except Exception as exc:
        Path(spec["error_path"]).write_text(
            json.dumps({"error": str(exc), "type": type(exc).__name__}),
            encoding="utf-8",
        )
        emit({"event": "error", "detail": str(exc)})
        sys.exit(1)


if __name__ == "__main__":
    main()
