from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from . import __version__
from .config import Settings
from .devices import list_devices, nvidia_smi_report, resolve_device
from .errors import CancelledJob, Forbidden, KykeonError
from .jobs import JobManager, list_artifacts
from .paths import contained, output_path
from .pipeline import build_plan, library_versions, tokenize_preflight
from .checkpoints import inspect_checkpoint
from .registry import Registry, register_path, resolve_registered

API = "/api/v1"


class RegisterBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str


class ValidateBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    workflow: dict[str, Any]
    device_id: str


class TestHook(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pause_seconds: float | None = None
    ignore_cancel_seconds: float | None = None
    fail: str | None = None
    bad_result: bool | None = None


class JobBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    workflow: dict[str, Any]
    device_id: str
    warmup_runs: int = Field(default=1, ge=0, le=2)
    measured_runs: int = Field(default=5, ge=1, le=10)
    test_hook: TestHook | None = None
    submission_id: str | None = None


def readable_roots(settings: Settings) -> list[Path]:
    roots = [path for path in settings.model_roots]
    checkpoints = output_path(settings.artifact_root, settings.artifact_root / "checkpoints")
    checkpoints.mkdir(parents=True, exist_ok=True)
    roots.append(checkpoints)
    return roots


def create_app(settings: Settings) -> FastAPI:
    settings.artifact_root.mkdir(parents=True, exist_ok=True)
    registry = Registry(output_path(settings.artifact_root, settings.artifact_root / "registry.json"))
    roots = readable_roots(settings)
    jobs = JobManager(settings, registry, roots)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        jobs.shutdown()

    app = FastAPI(title="Kykeon local execution", version=__version__, lifespan=lifespan)
    app.state.jobs = jobs
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Kykeon-Request"],
    )

    @app.middleware("http")
    async def loopback_and_origin(request: Request, call_next):
        client = request.client.host if request.client else ""
        if client not in {"127.0.0.1", "::1"}:
            return JSONResponse(
                {"error": "Kykeon accepts connections only from loopback.", "code": "forbidden"},
                status_code=403,
            )
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            if request.headers.get("x-kykeon-request") != "1":
                return JSONResponse(
                    {
                        "error": "Mutation requests must send X-Kykeon-Request: 1.",
                        "code": "forbidden",
                    },
                    status_code=403,
                )
            origin = request.headers.get("origin")
            if origin and origin not in settings.allowed_origins:
                return JSONResponse(
                    {"error": f"Origin {origin} is not allowed.", "code": "forbidden"},
                    status_code=403,
                )
        return await call_next(request)

    @app.exception_handler(KykeonError)
    async def kykeon_error(_request: Request, exc: KykeonError):
        return JSONResponse(
            {"error": exc.message, "code": exc.code},
            status_code=exc.status_code,
        )

    @app.exception_handler(CancelledJob)
    async def cancelled(_request: Request, exc: CancelledJob):
        return JSONResponse({"error": str(exc), "code": "cancelled"}, status_code=409)

    @app.get(f"{API}/health")
    def health():
        versions = library_versions()
        return {
            "status": "ok",
            "api": "v1",
            "backend_version": __version__,
            "libraries": versions,
            "nvidia_smi": nvidia_smi_report(),
            "simulation": "This process does not simulate. Simulation stays in the browser.",
        }

    @app.get(f"{API}/capabilities")
    def capabilities():
        return {
            "api": "v1",
            "concepts": {
                "catalog_reference": "Catalog entries stay in the workbench. This API executes only the allowlist below.",
                "simulation": "Simulation is a separate browser engine. This API never returns simulated text or metrics.",
                "real_local_execution": [
                    "Dense LlamaForCausalLM Hugging Face directories with safetensors",
                    "Sampling controls: temperature, top-k, top-p, seed, repetition penalty, max new tokens, stop sequences",
                    "operation-4 weight rescaling of one attention or MLP projection",
                ],
                "unsupported_configuration": [
                    "Quantized checkpoints",
                    "Architectures other than LlamaForCausalLM",
                    "Pickle and GGUF weights",
                    "Custom Python, JSON, or TensorFlow execution",
                    "Projected tool fields",
                    "Merging, quantization, PEFT, and LoRA",
                    "Batching, KV-cache tuning, and acceptance gates",
                ],
                "enterprise_classification": "An Enterprise flag is a provisional scope label, not a benchmark and not an entitlement.",
            },
            "operations": [
                {
                    "id": "operation-4",
                    "name": "Weight rescaling",
                    "version": "1.0.0",
                    "scale": {"minimum": 0.5, "maximum": 1.5, "default": 1.0},
                }
            ],
            "measurement_defaults": {"warmup_runs": 1, "measured_runs": 5},
        }

    @app.get(f"{API}/devices")
    def devices():
        return list_devices()

    @app.get(f"{API}/models/roots")
    def model_roots():
        return {
            "roots": [str(path) for path in settings.model_roots],
            "artifact_checkpoints": str(settings.artifact_root / "checkpoints"),
        }

    @app.get(f"{API}/models/browse")
    def browse(path: str = ""):
        if not path:
            entries = []
            for root in settings.model_roots:
                if root.expanduser().is_dir():
                    entries.append({"name": root.expanduser().name, "path": str(root.expanduser().resolve())})
            return {"entries": entries}
        resolved = None
        owner = None
        last_error = "That directory is outside the configured roots."
        for root in roots:
            try:
                resolved = contained(root, Path(path))
                owner = root
                break
            except KykeonError as exc:
                last_error = exc.message
        if resolved is None or owner is None or not resolved.is_dir():
            return JSONResponse(
                {"error": last_error, "code": "path_rejected"},
                status_code=400,
            )
        entries = []
        for child in sorted(resolved.iterdir()):
            if not child.is_dir():
                continue
            try:
                real = contained(owner, child)
            except KykeonError:
                continue
            entries.append({"name": child.name, "path": str(real)})
        return {"path": str(resolved), "entries": entries}

    @app.post(f"{API}/models/register")
    def register(body: RegisterBody):
        record = register_path(registry, roots, body.path)
        return registry.public(record)

    @app.get(f"{API}/models")
    def models():
        return {"models": [registry.public(record) for record in registry.records.values()]}

    @app.get(f"{API}/models/{{model_id}}")
    def model(model_id: str):
        return registry.public(registry.get(model_id))

    @app.post(f"{API}/models/{{model_id}}/preflight")
    def model_preflight(model_id: str, body: PreflightBody):
        record = registry.get(model_id)
        path = Path(record["resolved_path"])
        inspection = inspect_checkpoint(path, roots)
        selected = None
        if body.tensor:
            selected = next((row for row in inspection["tensors"] if row["name"] == body.tensor), None)
            if selected is None:
                from .errors import ValidationFailed

                raise ValidationFailed(f"Tensor {body.tensor!r} is not in this checkpoint.")
        return tokenize_preflight(path, roots, inspection, body.prompt, body.max_new_tokens, selected)

    @app.post(f"{API}/workflows/validate")
    def validate(body: ValidateBody):
        device = resolve_device(body.device_id)
        model_path = _workflow_model_path(body.workflow)
        resolved, _record, inspection = resolve_registered(registry, roots, model_path)
        plan = build_plan(body.workflow, inspection, resolved, 1, 5, roots)
        return {
            "accepted": True,
            "kind": plan["kind"],
            "device": device,
            "tensor": None if plan["tensor"] is None else plan["tensor"]["name"],
            "scale": plan["scale"],
            "controls_not_applied": plan["sampling"]["notes"],
            "preflight": plan["preflight"],
            "memory_estimate": plan["preflight"],
        }

    @app.post(f"{API}/jobs")
    def create_job(body: JobBody):
        if body.test_hook is not None and not settings.allow_test_hooks:
            raise Forbidden("Test hooks are disabled.")
        resolve_device(body.device_id)
        hook = body.test_hook.model_dump(exclude_none=True) if body.test_hook else None
        return jobs.submit(
            body.workflow,
            body.device_id,
            body.warmup_runs,
            body.measured_runs,
            hook,
            body.submission_id,
        )

    @app.get(f"{API}/submissions/{{submission_id}}")
    def submission(submission_id: str):
        return jobs.by_submission_id(submission_id)

    @app.get(f"{API}/jobs/{{job_id}}")
    def job(job_id: str):
        return jobs.get(job_id)

    @app.get(f"{API}/jobs/{{job_id}}/result")
    def job_result(job_id: str):
        return jobs.result(job_id)

    @app.post(f"{API}/jobs/{{job_id}}/cancel")
    def cancel(job_id: str):
        return jobs.cancel(job_id)

    @app.get(f"{API}/jobs/{{job_id}}/events")
    async def events(job_id: str, offset: int | None = None):
        jobs.get(job_id)
        if offset is not None:
            batch, new_offset = jobs.events(job_id, offset)
            status = jobs.get(job_id)
            return {
                "events": batch,
                "offset": new_offset,
                "status": status["status"],
                "error": status["error"],
            }

        async def stream():
            offset = 0
            while True:
                batch, offset = jobs.events(job_id, offset)
                for event in batch:
                    yield f"data: {json.dumps(event)}\n\n"
                status = jobs.get(job_id)
                if status["status"] in {"succeeded", "failed", "cancelled"}:
                    yield f"data: {json.dumps({'event': 'terminal', 'status': status['status'], 'error': status['error']})}\n\n"
                    break
                await asyncio.sleep(0.2)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get(f"{API}/artifacts")
    def artifacts():
        return {"artifacts": list_artifacts(settings.artifact_root)}

    @app.get(f"{API}/artifacts/{{artifact_id}}")
    def artifact(artifact_id: str):
        for item in list_artifacts(settings.artifact_root):
            if item["id"] == artifact_id:
                provenance_path = Path(item["path"]) / "kykeon-provenance.json"
                return {**item, "provenance": json.loads(provenance_path.read_text(encoding="utf-8"))}
        return JSONResponse({"error": "Artifact not found.", "code": "not_found"}, status_code=404)

    return app


class PreflightBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str
    max_new_tokens: int = Field(default=80, ge=1, le=512)
    tensor: str | None = None


def _workflow_model_path(workflow: dict[str, Any]) -> str:
    from .workflow import ensure_workflow_shape

    workflow = ensure_workflow_shape(workflow)
    for node in workflow.get("nodes") or []:
        data = node.get("data") or {}
        if data.get("definitionId") == "model-2":
            path = (data.get("config") or {}).get("path")
            if isinstance(path, str):
                return path
    raise KykeonError("The workflow has no model path.")


