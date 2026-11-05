from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .checkpoints import inspect_checkpoint
from .errors import NotFound, ValidationFailed
from .paths import contained_any, atomic_json


class Registry:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if path.is_file():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.records = loaded if isinstance(loaded, dict) else {}
        else:
            self.records = {}

    def save(self) -> None:
        atomic_json(self.path.parent, self.path, self.records)

    def upsert(self, resolved: Path, inspection: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            existing = next(
                (record for record in self.records.values() if record["resolved_path"] == str(resolved)),
                None,
            )
            record_id = existing["id"] if existing else str(uuid.uuid4())
            record = {
                "id": record_id,
                "resolved_path": str(resolved),
                "name": resolved.name,
                "architecture": inspection["architecture"],
                "parameter_count": inspection["parameter_count"],
                "directory_bytes": inspection["directory_bytes"],
                "native_dtype": inspection["native_dtype"],
                "eligible_tensor_count": len(inspection["eligible_tensors"]),
                "inspection": inspection,
                "simulated": False,
                "registered_at": datetime.now(timezone.utc).isoformat(),
            }
            self.records[record_id] = record
            self.save()
            return record

    def get(self, record_id: str) -> dict[str, Any]:
        record = self.records.get(record_id)
        if record is None:
            raise NotFound(f"No registered checkpoint with id {record_id}.")
        return record

    def by_path(self, resolved: Path) -> dict[str, Any] | None:
        return next(
            (record for record in self.records.values() if record["resolved_path"] == str(resolved)),
            None,
        )

    def public(self, record: dict[str, Any]) -> dict[str, Any]:
        inspection = record["inspection"]
        return {
            "id": record["id"],
            "resolved_path": record["resolved_path"],
            "name": record["name"],
            "architecture": record["architecture"],
            "parameter_count": record["parameter_count"],
            "directory_bytes": record["directory_bytes"],
            "native_dtype": record["native_dtype"],
            "simulated": False,
            "registered_at": record["registered_at"],
            "config": inspection["config"],
            "eligible_tensors": inspection["eligible_tensors"],
            "tensors": inspection["tensors"],
            "memory_estimate": inspection["memory_estimate"],
        }


def register_path(registry: Registry, roots: list[Path] | tuple[Path, ...], raw_path: str) -> dict[str, Any]:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValidationFailed("Enter a checkpoint directory.")
    resolved = contained_any(tuple(roots), Path(raw_path))
    if not resolved.is_dir():
        raise ValidationFailed("The checkpoint path must be a directory.")
    inspection = inspect_checkpoint(resolved, roots)
    return registry.upsert(resolved, inspection)


def resolve_registered(
    registry: Registry,
    roots: list[Path] | tuple[Path, ...],
    raw_path: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    resolved = contained_any(tuple(roots), Path(raw_path))
    record = registry.by_path(resolved)
    if record is None:
        raise ValidationFailed(
            "Register this checkpoint before local execution. "
            "An unregistered path is not opened, and simulation results are not substituted."
        )
    inspection = inspect_checkpoint(resolved, roots)
    return resolved, record, inspection
