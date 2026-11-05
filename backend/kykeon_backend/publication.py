"""One atomic success record gates both results and artifact visibility."""
from pathlib import Path
import hashlib
import json
from .paths import output_path, atomic_json


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def success_result(root: Path, job_id: str):
    try:
        directory = output_path(root, root / "jobs" / job_id)
        record = json.loads(output_path(root, directory / "success.json").read_text())
        result_path = output_path(root, directory / "result.json")
        if record.get("version") != 1 or record.get("job_id") != job_id or record.get("result_sha256") != digest(result_path):
            return None
        result = json.loads(result_path.read_text())
        if not isinstance(result, dict) or result.get("execution") != "local":
            return None
        artifact = result.get("artifact")
        if artifact:
            final = output_path(root, root / "checkpoints" / artifact["id"])
            if Path(artifact["path"]) != final:
                return None
            provenance = output_path(root, final / "kykeon-provenance.json")
            if record.get("provenance_sha256") != digest(provenance):
                return None
            data = json.loads(provenance.read_text())
            if data.get("job_id") != job_id or data.get("artifact_id") != artifact["id"]:
                return None
        return result
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None
    except Exception:
        # A confinement failure must hide, rather than expose, an artifact.
        return None


def publish_success(root: Path, job_id: str, result: dict) -> None:
    directory = output_path(root, root / "jobs" / job_id)
    record = {"version": 1, "job_id": job_id,
              "result_sha256": digest(output_path(root, directory / "result.json"))}
    artifact = result.get("artifact")
    if artifact:
        final = output_path(root, root / "checkpoints" / artifact["id"])
        if Path(artifact["path"]) != final:
            raise ValueError("Artifact path does not match its identity")
        provenance_path = output_path(root, final / "kykeon-provenance.json")
        provenance = json.loads(provenance_path.read_text())
        if provenance.get("job_id") != job_id or provenance.get("artifact_id") != artifact["id"]:
            raise ValueError("Artifact provenance does not match the job")
        required = ("source_unchanged", "scaled_tensor_finite", "only_selected_tensor_changed", "reload_succeeded", "derived_inference_succeeded")
        if any(provenance.get("validation", {}).get(key) is not True for key in required):
            raise ValueError("Artifact validation is incomplete")
        files = provenance.get("derived_files")
        if not isinstance(files, list) or not files:
            raise ValueError("Artifact file hashes are missing")
        for item in files:
            path = output_path(final, final / item["path"])
            if digest(path) != item["sha256"]:
                raise ValueError("Derived checkpoint hash mismatch")
        record["provenance_sha256"] = digest(provenance_path)
    try:
        atomic_json(root, directory / "success.json", record)
    except OSError:
        # A directory fsync can fail after atomic rename. The visible validated
        # commit still decides the outcome; never label it failed and published.
        if success_result(root, job_id) is None:
            raise
