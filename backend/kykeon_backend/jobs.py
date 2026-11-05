from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .config import Settings
from .errors import CancelledJob, JobBusy, JobFailed, NotFound, ValidationFailed
from .paths import safe_rmtree, output_path, atomic_json
from .publication import publish_success, success_result
from .pipeline import build_plan
from .registry import resolve_registered
from .workflow import ensure_workflow_shape

TERMINAL = {"succeeded", "failed", "cancelled"}


def read_progress(data: bytes, offset: int) -> tuple[list[dict[str, Any]], int]:
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0 or offset > len(data):
        raise ValidationFailed(
            f"Progress offset {offset} is outside the {len(data)}-byte progress file."
        )
    chunk = data[offset:]
    newline = chunk.rfind(b"\n")
    if newline < 0:
        return [], offset
    complete = chunk[: newline + 1]
    new_offset = offset + newline + 1
    events: list[dict[str, Any]] = []
    for raw in complete.decode("utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {
                "event": "error",
                "detail": "A complete progress record was corrupt and was not applied.",
            }
        if isinstance(parsed, dict):
            events.append(parsed)
        else:
            events.append(
                {
                    "event": "error",
                    "detail": "A complete progress record was not an object and was not applied.",
                }
            )
    return events, new_offset


class JobManager:
    def __init__(self, settings: Settings, registry, roots: list[Path]):
        self.settings = settings
        self.registry = registry
        self.roots = roots
        self._lock = threading.Lock()
        self._active: str | None = None
        self._closing = False
        self.jobs: dict[str, dict[str, Any]] = {}
        self.by_submission: dict[str, dict[str, Any]] = {}
        settings.artifact_root.mkdir(parents=True, exist_ok=True)
        self._lock_file = open(output_path(settings.artifact_root, settings.artifact_root / "backend.lock"), "a+", encoding="utf-8")
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another Kykeon backend holds the execution lock. "
                "Refusing to start a second executor that could leave an orphan worker."
            ) from exc
        self._reconcile()

    def submit(
        self,
        workflow: dict[str, Any],
        device_id: str,
        warmup_runs: int,
        measured_runs: int,
        test_hook: dict[str, Any] | None,
        submission_id: str | None,
    ) -> dict[str, Any]:
        ensure_workflow_shape(workflow)
        submission_id = submission_id or str(uuid.uuid4())
        if not _is_uuid(submission_id):
            raise ValidationFailed("submission_id must be a UUID.")
        with self._lock:
            existing = self.by_submission.get(submission_id)
            if existing is not None:
                return self._public(existing) | {"idempotent": True}
            if self._closing:
                raise JobFailed("The backend is shutting down and is not accepting jobs.")
            if self._active is not None:
                raise JobBusy("A local job is already running. Only one execution job runs at a time.")
            job_id = submission_id
            self._active = job_id
        try:
            model_path = _model_path(workflow)
            resolved, _record, inspection = resolve_registered(self.registry, self.roots, model_path)
            plan = build_plan(workflow, inspection, resolved, warmup_runs, measured_runs, self.roots)
            job = self._prepare_job(job_id, submission_id, plan, device_id, test_hook or {})
        except Exception as exc:
            self._abort_slot(job_id, exc)
            raise
        self._start_thread(job)
        return self._public(job)

    def _prepare_job(self, job_id: str, submission_id: str, plan: dict, device_id: str, test_hook: dict) -> dict:
        job_dir = output_path(self.settings.artifact_root, self.settings.artifact_root / "jobs" / job_id)
        try:
            job_dir.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            raise JobFailed(f"Could not create the job directory: {exc}") from exc
        job = {
            "id": job_id,
            "submission_id": submission_id,
            "status": "queued",
            "directory": str(job_dir),
            "cancel_path": str(job_dir / "cancel"),
            "progress_path": str(job_dir / "progress.jsonl"),
            "result_path": str(job_dir / "result.json"),
            "error_path": str(job_dir / "error.json"),
            "error": None,
            "result": None,
            "created_at": time.time(),
            "process": None,
        }
        spec = {
            "job_id": job_id,
            "device_id": device_id,
            "artifact_root": str(self.settings.artifact_root),
            "model_roots": [str(path) for path in self.roots],
            "cancel_path": job["cancel_path"],
            "progress_path": job["progress_path"],
            "result_path": job["result_path"],
            "error_path": job["error_path"],
            "plan": json.loads(json.dumps(plan)),
            "test_hook": test_hook,
        }
        try:
            (job_dir / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
            (job_dir / "progress.jsonl").write_text("", encoding="utf-8")
        except OSError as exc:
            raise JobFailed(f"Could not write the job specification: {exc}") from exc
        with self._lock:
            self.jobs[job_id] = job
            self.by_submission[submission_id] = job
        self._persist(job)
        return job

    def _start_thread(self, job: dict[str, Any]) -> None:
        try:
            thread = threading.Thread(target=self._run, args=(job, Path(job["directory"]) / "spec.json"), daemon=True)
            thread.start()
            job["thread"] = thread
        except Exception as exc:
            self._mark(job, "failed", f"Could not start the job thread: {exc}")
            self._release(job["id"])
            _discard(self.settings.artifact_root, job["id"])

    def _abort_slot(self, job_id: str, exc: Exception) -> None:
        with self._lock:
            if self._active == job_id:
                self._active = None
        if isinstance(exc, (ValidationFailed, JobFailed, JobBusy, NotFound, CancelledJob)):
            return
        raise JobFailed(str(exc)) from exc

    def _run(self, job: dict[str, Any], spec_path: Path) -> None:
        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_HUB_DISABLE_TELEMETRY"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        backend_root = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = backend_root + os.pathsep + env.get("PYTHONPATH", "")
        started = time.time()
        process = None
        log = None
        try:
            spec = json.loads(output_path(self.settings.artifact_root, spec_path).read_text(encoding="utf-8"))
            if spec.get("device_id") == "cpu":
                env["CUDA_VISIBLE_DEVICES"] = ""
            log = output_path(self.settings.artifact_root, Path(job["directory"]) / "worker.log").open("ab")
            process = subprocess.Popen(
                [sys.executable, "-m", "kykeon_backend.worker", "--spec", str(spec_path)],
                cwd=backend_root,
                env=env,
                start_new_session=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            job["process"] = process
            job["pid"] = process.pid
            job["status"] = "running"
            output_path(self.settings.artifact_root, Path(job["directory"]) / "worker.pid").write_text(str(process.pid), encoding="utf-8")
            cancel_started = None
            kill_started = None
            while process.poll() is None:
                if self._closing or Path(job["cancel_path"]).exists():
                    if cancel_started is None:
                        cancel_started = time.time()
                        job["status"] = "cancel_requested"
                    elif kill_started is None and time.time() - cancel_started >= self.settings.cancel_grace_seconds:
                        _signal_group(process, signal.SIGTERM)
                        kill_started = time.time()
                    elif kill_started is not None and time.time() - kill_started >= self.settings.kill_grace_seconds:
                        _signal_group(process, signal.SIGKILL)
                if time.time() - started > self.settings.job_timeout_seconds:
                    _signal_group(process, signal.SIGKILL)
                    process.wait(timeout=5)
                    tail = _tail(Path(job["progress_path"]))
                    self._mark(
                        job,
                        "failed",
                        f"Worker exceeded {self.settings.job_timeout_seconds:.0f}s and was stopped. "
                        f"Last progress: {tail or 'none'}. Worker log: {_tail(Path(job['directory']) / 'worker.log')}.",
                    )
                    _discard(self.settings.artifact_root, job["id"])
                    return
                time.sleep(0.05)
            process.wait(timeout=5)
            self._collect(job, process.returncode)
        except OSError as exc:
            self._mark(job, "failed", f"Could not start the worker: {exc}")
            _discard(self.settings.artifact_root, job["id"])
        except Exception as exc:
            self._mark(job, "failed", f"Job supervision failed: {exc}")
            _discard(self.settings.artifact_root, job["id"])
        finally:
            if process is not None and process.poll() is None:
                _signal_group(process, signal.SIGKILL)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            if job["status"] not in TERMINAL:
                self._mark(job, "failed", job.get("error") or "The worker ended without a terminal status.")
                _discard(self.settings.artifact_root, job["id"])
            if log is not None:
                log.close()
            self._release(job["id"])

    def _collect(self, job: dict[str, Any], code: int | None) -> None:
        result_path = Path(job["result_path"])
        if result_path.is_file():
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._mark(job, "failed", "Worker result was malformed and was not accepted.")
                _discard(self.settings.artifact_root, job["id"])
                return
            if isinstance(payload, dict) and payload.get("execution") == "local" and code == 0:
                with self._lock:
                    if job["status"] == "cancel_requested" or Path(job["cancel_path"]).exists():
                        self._mark(job, "cancelled", "Run cancelled before publication.")
                    else:
                        publish_success(self.settings.artifact_root, job["id"], payload)
                        job["result"] = payload
                        self._mark(job, "succeeded", None)
                        return
                _discard(self.settings.artifact_root, job["id"])
                return
        if code == 2 or job["status"] == "cancel_requested":
            self._mark(job, "cancelled", "Run cancelled.")
        else:
            message = f"Worker exited with status {code} before writing a result."
            error_path = Path(job["error_path"])
            if error_path.is_file():
                try:
                    message = json.loads(error_path.read_text(encoding="utf-8")).get("error") or message
                except json.JSONDecodeError:
                    message = "Worker error file was malformed."
            self._mark(job, "failed", message)
        _discard(self.settings.artifact_root, job["id"])

    def _mark(self, job: dict[str, Any], status: str, error: str | None) -> None:
        job["status"] = status
        job["error"] = error
        self._persist(job)

    def _persist(self, job: dict[str, Any]) -> None:
        try:
            atomic_json(self.settings.artifact_root, Path(job["directory"]) / "state.json", self._public(job) | {"legacy_unverified": job.get("legacy_unverified", False)})
        except Exception:
            # The in-memory terminal state and success record remain authoritative.
            pass

    def _release(self, job_id: str) -> None:
        with self._lock:
            if self._active == job_id:
                self._active = None

    def get(self, job_id: str) -> dict[str, Any]:
        job = self.jobs.get(job_id) or self.by_submission.get(job_id)
        if job is None:
            raise NotFound(f"No job {job_id}.")
        return self._public(job)

    def by_submission_id(self, submission_id: str) -> dict[str, Any]:
        job = self.by_submission.get(submission_id)
        if job is None:
            raise NotFound(f"No submission {submission_id}.")
        return self._public(job)

    def result(self, job_id: str) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if job is None:
            raise NotFound(f"No job {job_id}.")
        if job["status"] == "succeeded":
            return job["result"]
        if job["status"] not in TERMINAL:
            raise JobBusy("The job is still running.")
        if job["status"] == "cancelled":
            raise CancelledJob(job["error"] or "Run cancelled.")
        raise JobFailed(job["error"] or "Job failed.")

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self.jobs.get(job_id) or self.by_submission.get(job_id)
            if job is None:
                raise NotFound(f"No job {job_id}.")
            if job["status"] in TERMINAL:
                return self._public(job)
            output_path(self.settings.artifact_root, Path(job["cancel_path"])).write_text("cancel\n", encoding="utf-8")
            self._mark(job, "cancel_requested", None)
            return self._public(job)

    def events(self, job_id: str, offset: int) -> tuple[list[dict[str, Any]], int]:
        job = self.jobs.get(job_id)
        if job is None:
            raise NotFound(f"No job {job_id}.")
        path = Path(job["progress_path"])
        data = path.read_bytes() if path.is_file() else b""
        return read_progress(data, offset)

    def shutdown(self) -> None:
        self._closing = True
        with self._lock:
            running = [job for job in self.jobs.values() if job["status"] not in TERMINAL]
        for job in running:
            try:
                self.cancel(job["id"])
            except OSError:
                pass
        deadline = time.time() + self.settings.cancel_grace_seconds + self.settings.kill_grace_seconds + 1
        while time.time() < deadline and any(job["status"] not in TERMINAL for job in running):
            time.sleep(0.05)
        for job in running:
            process = job.get("process")
            if process is not None and process.poll() is None:
                _signal_group(process, signal.SIGKILL)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
            if job["status"] not in TERMINAL:
                self._mark(job, "cancelled", "Backend shut down before the worker finished.")
                _discard(self.settings.artifact_root, job["id"])
            self._release(job["id"])

    def _public(self, job: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": job["id"],
            "submission_id": job.get("submission_id", job["id"]),
            "status": job["status"],
            "error": job.get("error"),
            "pid": job.get("pid"),
        }

    def _reconcile(self) -> None:
        jobs_root = output_path(self.settings.artifact_root, self.settings.artifact_root / "jobs")
        if not jobs_root.exists():
            return
        for job_dir in jobs_root.iterdir():
            output_path(self.settings.artifact_root, job_dir)
            if not job_dir.is_dir() or not _is_uuid(job_dir.name):
                continue
            pid_path = output_path(self.settings.artifact_root, job_dir / "worker.pid")
            if pid_path.is_file():
                _stop_recorded_worker(pid_path)
            result = success_result(self.settings.artifact_root, job_dir.name)
            state_path = output_path(self.settings.artifact_root, job_dir / "state.json")
            try:
                state = json.loads(state_path.read_text())
            except (OSError, ValueError):
                state = {}
            if not isinstance(state, dict):
                state = {}
            status = "succeeded" if result else state.get("status", "failed")
            if not isinstance(status, str):
                status = "failed"
            if status not in {"failed", "cancelled"} and result is None:
                status = "failed"
            job = {"id": job_dir.name, "submission_id": job_dir.name, "directory": str(job_dir),
                   "status": status, "result": result, "process": None,
                   "error": None if result else state.get("error") or "Backend restarted before a successful publication.",
                   "legacy_unverified": state.get("legacy_unverified", not state_path.exists())}
            for name, filename in (("cancel", "cancel"), ("progress", "progress.jsonl"), ("result", "result.json"), ("error", "error.json")):
                job[name + "_path"] = str(output_path(self.settings.artifact_root, job_dir / filename))
            self.jobs[job_dir.name] = job
            self.by_submission[job_dir.name] = job
            if result is None and not job["legacy_unverified"]:
                _discard(self.settings.artifact_root, job_dir.name)
            # Legacy jobs without state/success records stay hidden. Preserve
            # their files rather than deleting an older user's export.
            self._persist(job)


def _discard(artifact_root: Path, job_id: str) -> None:
    """Best-effort cleanup must never prevent releasing the executor."""
    try:
        safe_rmtree(artifact_root, output_path(artifact_root, artifact_root / "staging" / job_id))
    except Exception:
        pass
    try:
        checkpoints = output_path(artifact_root, artifact_root / "checkpoints")
        children = list(checkpoints.iterdir()) if checkpoints.exists() else []
    except Exception:
        return
    for child in children:
        try:
            provenance = output_path(artifact_root, child / "kykeon-provenance.json")
            payload = json.loads(provenance.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("job_id") != job_id:
                continue
            if success_result(artifact_root, job_id) is None:
                safe_rmtree(artifact_root, child)
        except Exception:
            continue


def _model_path(workflow: dict[str, Any]) -> str:
    for node in workflow.get("nodes") or []:
        if not isinstance(node, dict):
            raise ValidationFailed("Each node must be an object.")
        data = node.get("data")
        if not isinstance(data, dict):
            raise ValidationFailed("Each node needs a data object.")
        if data.get("definitionId") == "model-2":
            config = data.get("config")
            if not isinstance(config, dict):
                raise ValidationFailed("The model node configuration must be an object.")
            path = config.get("path")
            if isinstance(path, str) and path.strip():
                return path
    raise ValidationFailed("The workflow has no model path to register against.")


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True


def _signal_group(process: subprocess.Popen, sig: signal.Signals) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        process.send_signal(sig)


def _tail(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")[-500:]


def _stop_recorded_worker(pid_path: Path) -> None:
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except ValueError:
        return
    cmdline = Path(f"/proc/{pid}/cmdline")
    if not cmdline.exists():
        return
    args = cmdline.read_bytes().decode("utf-8", "replace").split("\x00")
    expected_spec = str(pid_path.parent / "spec.json")
    if "kykeon_backend.worker" not in args or "--spec" not in args:
        return
    index = args.index("--spec")
    if index + 1 >= len(args) or args[index + 1] != expected_spec:
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + 2
    while time.time() < deadline and Path(f"/proc/{pid}").exists():
        time.sleep(0.05)
    if Path(f"/proc/{pid}").exists():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def list_artifacts(artifact_root: Path) -> list[dict[str, Any]]:
    jobs_root = output_path(artifact_root, artifact_root / "jobs")
    if not jobs_root.exists():
        return []
    found = []
    for directory in sorted(jobs_root.iterdir()):
        if directory.is_symlink() or not _is_uuid(directory.name):
            continue
        result = success_result(artifact_root, directory.name)
        if not result or not result.get("artifact"):
            continue
        artifact = result["artifact"]
        provenance = result.get("provenance", {})
        found.append({**artifact, "operation": provenance.get("operation"),
                      "published_at": provenance.get("timestamps", {}).get("published_at")})
    return found
