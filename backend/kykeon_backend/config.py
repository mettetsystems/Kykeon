from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _split_paths(value: str | None) -> list[Path]:
    if not value:
        return []
    return [Path(part).expanduser() for part in value.split(os.pathsep) if part]


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    model_roots: tuple[Path, ...]
    artifact_root: Path
    allowed_origins: tuple[str, ...]
    cancel_grace_seconds: float
    kill_grace_seconds: float
    allow_test_hooks: bool
    job_timeout_seconds: float = 1800

    @staticmethod
    def from_env() -> "Settings":
        host = os.environ.get("KYKEON_HOST", "127.0.0.1")
        if host not in {"127.0.0.1", "localhost"} and os.environ.get(
            "KYKEON_ALLOW_NON_LOOPBACK"
        ) != "1":
            raise RuntimeError(
                "Kykeon binds to 127.0.0.1 by default. "
                "KYKEON_ALLOW_NON_LOOPBACK=1 is required to bind another host."
            )
        roots = _split_paths(os.environ.get("KYKEON_MODEL_ROOTS"))
        artifact = Path(
            os.environ.get(
                "KYKEON_ARTIFACT_ROOT",
                str(Path.home() / ".local" / "share" / "kykeon" / "artifacts"),
            )
        ).expanduser()
        origins = tuple(
            part.strip()
            for part in os.environ.get(
                "KYKEON_ALLOWED_ORIGINS",
                "http://127.0.0.1:5173,http://localhost:5173",
            ).split(",")
            if part.strip()
        )
        return Settings(
            host=host,
            port=int(os.environ.get("KYKEON_PORT", "8765")),
            model_roots=tuple(roots),
            artifact_root=artifact,
            allowed_origins=origins,
            cancel_grace_seconds=float(os.environ.get("KYKEON_CANCEL_GRACE", "8")),
            kill_grace_seconds=float(os.environ.get("KYKEON_KILL_GRACE", "4")),
            allow_test_hooks=os.environ.get("KYKEON_ALLOW_TEST_HOOKS") == "1",
            job_timeout_seconds=float(os.environ.get("KYKEON_JOB_TIMEOUT", "1800")),
        )
