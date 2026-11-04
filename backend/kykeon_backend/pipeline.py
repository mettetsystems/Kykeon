from __future__ import annotations

import gc
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .checkpoints import assert_checkpoint_contained, file_record, inspect_checkpoint, memory_estimate
from .paths import safe_rmtree, within_any, output_path
from .devices import resolve_device
from .errors import CancelledJob, DeviceUnavailable, KykeonError, UnsupportedModel
from .generate import SEED_NOTE, measure_generation
from .surgery import assert_rescale_effect, rescale_directory
from .workflow import validate_local_workflow

DTYPES = {"F16": "float16", "BF16": "bfloat16", "F32": "float32"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _torch_dtype(name: str):
    import torch

    return {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32}[name]


def library_versions() -> dict[str, str | None]:
    import safetensors
    import torch
    import transformers

    return {
        "kykeon_backend": __version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
    }


def snapshot_files(directory: Path, roots: list[Path] | tuple[Path, ...] | None = None) -> list[dict[str, Any]]:
    search = tuple(roots) if roots is not None else (directory,)
    records = []
    for path in sorted(directory.rglob("*")):
        if not path.is_symlink() and not path.is_file():
            continue
        real = within_any(search, path)
        if real.is_file():
            records.append(file_record(path, directory))
    return records


def release_slot(slot: dict[str, Any], device) -> None:
    """Drop every reference this slot holds, then collect.

    Callers must not keep a separate variable pointing at slot['model'].
    """
    slot["model"] = None
    slot["tokenizer"] = None
    gc.collect()
    if device is not None and getattr(device, "type", None) == "cuda":
        import torch

        torch.cuda.empty_cache()


def load_llama(path: Path, device, dtype_name: str, roots: list[Path] | tuple[Path, ...] | None = None):
    import torch
    from transformers import AutoTokenizer, LlamaForCausalLM

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    path = assert_checkpoint_contained(path, tuple(roots) if roots is not None else (path,))
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = _torch_dtype(dtype_name)
    try:
        model = LlamaForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
    except TypeError:
        model = LlamaForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    if type(model).__name__ != "LlamaForCausalLM":
        raise UnsupportedModel(
            f"Loaded {type(model).__name__}, which is not LlamaForCausalLM. The checkpoint was not executed."
        )
    try:
        model.to(device)
    except torch.cuda.OutOfMemoryError as exc:
        raise DeviceUnavailable(
            "The selected GPU ran out of memory while loading the checkpoint. "
            "The job was not moved to CPU. "
            f"{exc}"
        ) from exc
    model.eval()
    return model, tokenizer


def release(model, device) -> None:
    import torch

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _check_cancel(cancel_path: Path) -> None:
    if cancel_path.exists():
        raise CancelledJob("Run cancelled.")


def run_spec(spec: dict[str, Any], emit: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    import torch

    cancel_path = Path(spec["cancel_path"])
    hook = spec.get("test_hook") or {}
    if hook.get("fail"):
        raise RuntimeError(str(hook.get("fail")))
    if hook.get("ignore_cancel_seconds"):
        import time

        time.sleep(float(hook["ignore_cancel_seconds"]))
    if hook.get("pause_seconds"):
        import time

        deadline = time.time() + float(hook["pause_seconds"])
        while time.time() < deadline:
            _check_cancel(cancel_path)
            time.sleep(0.05)

    device_info = resolve_device(spec["device_id"])
    device = torch.device(spec["device_id"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    plan = spec["plan"]
    roots = [Path(item) for item in spec.get("model_roots") or []]
    source = assert_checkpoint_contained(Path(plan["resolved_path"]), roots)
    dtype_name = plan["inspection"]["native_dtype"]
    if dtype_name not in DTYPES:
        raise UnsupportedModel(f"No supported floating dtype was found on the weights ({dtype_name}).")

    def cancelled() -> None:
        _check_cancel(cancel_path)

    def emit_step(step: dict[str, str], state: str, detail: str) -> None:
        emit(
            {
                "event": "node",
                "nodeId": step["node_id"],
                "definitionId": step["definition_id"],
                "state": state,
                "detail": detail,
            }
        )

    started = _now()
    source_hashes = snapshot_files(source, roots)
    slot: dict[str, Any] = {"model": None, "tokenizer": None}
    slot["model"], slot["tokenizer"] = load_llama(source, device, dtype_name, roots)
    for step in plan.get("steps") or []:
        if step["definition_id"] == "model-2":
            emit_step(step, "complete", "Checkpoint loaded")
    try:
        if plan["kind"] == "inference":
            return _inference_job(
                plan, slot, device, device_info, emit_step, cancelled, started, source_hashes
            )
        return _rescale_job(
            spec,
            plan,
            slot,
            device,
            device_info,
            source,
            roots,
            emit_step,
            cancelled,
            started,
            source_hashes,
        )
    finally:
        release_slot(slot, device)


def _inference_job(plan, slot, device, device_info, emit_step, cancelled, started, source_hashes):
    sampling = plan["sampling"]
    configured = None
    comparison = None
    from .workflow import SAMPLING_DEFAULTS, normalize_sampling

    for step in plan["steps"]:
        definition = step["definition_id"]
        emit_step(step, "running", definition)
        if definition == "generate":
            configured = measure_generation(
                slot["model"],
                slot["tokenizer"],
                plan["prompt"],
                sampling,
                device,
                warmup_runs=plan["warmup_runs"],
                measured_runs=plan["measured_runs"],
                cancelled=cancelled,
            )
        elif definition == "compare":
            baseline = measure_generation(
                slot["model"],
                slot["tokenizer"],
                plan["prompt"],
                normalize_sampling({**SAMPLING_DEFAULTS, "seed": sampling["seed"]}),
                device,
                warmup_runs=plan["warmup_runs"],
                measured_runs=plan["measured_runs"],
                cancelled=cancelled,
            )
            comparison = {"default_settings": baseline, "configured": configured}
        emit_step(step, "complete", definition)
    if configured is None:
        raise UnsupportedModel("The inference chain did not generate text.")
    report = _inference_report(plan, configured, comparison, device_info)
    return {
        "execution": "local",
        "kind": "inference",
        "text": configured["text"],
        "report": report,
        "metrics": configured,
        "comparison": comparison,
        "preflight": plan.get("preflight"),
        "validity": {
            "finite_generation": True,
            "capability_retention_score": {
                "status": "unavailable",
                "reason": "Kykeon does not invent an accuracy or capability-retention score.",
            },
        },
        "device": device_info,
        "seed_note": SEED_NOTE,
        "source_hashes": source_hashes,
        "workflow": plan.get("workflow_document"),
        "started_at": started,
        "finished_at": _now(),
        "versions": library_versions(),
    }


def _rescale_job(
    spec, plan, slot, device, device_info, source, roots, emit_step, cancelled, started, source_hashes
):
    sampling = plan["sampling"]
    tensor = plan["tensor"]["name"]
    scale = float(plan["scale"])
    for step in plan["steps"]:
        if step["definition_id"] == "operation-4":
            emit_step(step, "running", "Measuring the source checkpoint")
    before = measure_generation(
        slot["model"],
        slot["tokenizer"],
        plan["prompt"],
        sampling,
        device,
        warmup_runs=plan["warmup_runs"],
        measured_runs=plan["measured_runs"],
        cancelled=cancelled,
    )
    release_slot(slot, device)
    cancelled()
    artifact_root = Path(spec["artifact_root"])
    within_any((artifact_root,), artifact_root)
    staging = output_path(artifact_root, artifact_root / "staging" / spec["job_id"])
    artifact_id = str(uuid.uuid4())
    rescale_directory(source, staging, tensor, scale, cancelled, roots, artifact_root)
    effect = assert_rescale_effect(source, staging, tensor, scale)
    after_hashes = snapshot_files(source, roots)
    if source_hashes != after_hashes:
        raise UnsupportedModel("The source checkpoint changed. Publication was aborted.")
    node_emit = {step["definition_id"]: step for step in plan["steps"]}
    emit_step(node_emit["operation-4"], "complete", "Derived weights written to staging")
    emit_step(node_emit["evaluate"], "running", "Reloading the derived checkpoint")
    derived: dict[str, Any] = {"model": None, "tokenizer": None}
    derived["model"], derived["tokenizer"] = load_llama(
        staging, device, plan["inspection"]["native_dtype"], (artifact_root, source)
    )
    try:
        after = measure_generation(
            derived["model"],
            derived["tokenizer"],
            plan["prompt"],
            sampling,
            device,
            warmup_runs=plan["warmup_runs"],
            measured_runs=plan["measured_runs"],
            cancelled=cancelled,
        )
    finally:
        release_slot(derived, device)
    validity = {
        "source_unchanged": True,
        "scaled_tensor_finite": effect["scaled_tensor_finite"],
        "only_selected_tensor_changed": effect["only_selected_tensor_changed"],
        "values_preserved_at_unit_scale": effect["values_preserved"],
        "reload_succeeded": True,
        "derived_inference_succeeded": True,
        "capability_retention_score": {
            "status": "unavailable",
            "reason": "Kykeon does not invent an accuracy or capability-retention score.",
        },
    }
    directory_bytes = sum(path.stat().st_size for path in staging.rglob("*") if path.is_file())
    provenance = {
        "kind": "kykeon-real-checkpoint",
        "simulated": False,
        "job_id": spec["job_id"],
        "artifact_id": artifact_id,
        "operation": {
            "id": "operation-4",
            "name": "Weight rescaling",
            "version": plan["operation"]["version"],
            "tensor": tensor,
            "scale": scale,
            "tensor_shape": plan["tensor"]["shape"],
            "tensor_dtype": plan["tensor"]["dtype"],
            "tensor_parameters": plan["tensor"]["parameters"],
        },
        "source": {
            "path": str(source),
            "architecture": "LlamaForCausalLM",
            "files": source_hashes,
        },
        "workflow": {
            "version": 1,
            "sha256": plan["workflow_sha256"],
            "kind": "rescale",
            "document": plan.get("workflow_document"),
            "steps": plan.get("steps"),
        },
        "lineage": _lineage(source, roots),
        "measurements": {
            "methodology": before.get("methodology"),
            "before": _measurement_summary(before),
            "after": _measurement_summary(after),
        },
        "preflight": plan.get("preflight"),
        "versions": library_versions(),
        "device": device_info,
        "dtype": plan["inspection"]["native_dtype"],
        "seed": sampling["seed"],
        "generation": {
            "prompt": plan["prompt"],
            "temperature": sampling["temperature"],
            "top_k": sampling["top_k"],
            "top_p": sampling["top_p"],
            "seed": sampling["seed"],
            "repetition_penalty": sampling["repetition_penalty"],
            "max_new_tokens": sampling["max_new_tokens"],
            "stop_sequences": sampling["stop_sequences"],
            "warmup_runs": plan["warmup_runs"],
            "measured_runs": plan["measured_runs"],
        },
        "validation": validity,
        "timestamps": {"started_at": started, "published_at": _now()},
        "seed_note": SEED_NOTE,
        "memory_estimate": memory_estimate(
            plan["inspection"]["config"],
            plan["inspection"]["tensors"],
            prompt_tokens=before["input_token_count"],
            max_new_tokens=sampling["max_new_tokens"],
            selected=plan["tensor"],
        ),
    }
    (staging / "kykeon-provenance.json").write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )
    final = output_path(artifact_root, artifact_root / "checkpoints" / artifact_id)
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise UnsupportedModel("Refusing to overwrite an existing artifact.")
    cancelled()
    os.replace(staging, final)
    try:
        cancelled()
    except CancelledJob:
        safe_rmtree(artifact_root, final)
        raise
    content_hashes = [
        item
        for item in snapshot_files(final, (artifact_root,))
        if item["path"] not in {"kykeon-provenance.json", "kykeon-commit.json"}
    ]
    provenance["derived_files"] = content_hashes
    (final / "kykeon-provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    emit_step(node_emit["evaluate"], "complete", "Derived checkpoint reloaded and generated text")
    emit_step(node_emit["export"], "running", "Publishing artifact")
    emit_step(node_emit["export"], "complete", "Artifact published")
    history = [
        f"{item.get('operation', {}).get('id', 'prior')} preserved from the source lineage"
        for item in provenance["lineage"]
    ]
    history.append(f"operation-4 weight rescaling {plan['operation']['version']}: {tensor} × {scale:g}")
    model_reference = {
        "name": f"{plan['model_name']} scale {scale:g}",
        "path": str(final),
        "architecture": "LlamaForCausalLM",
        "sizeGB": directory_bytes / 1_000_000_000,
        "history": history,
        "sources": [str(source)],
    }
    report = _rescale_report(plan, before, after, validity, final, device_info)
    return {
        "execution": "local",
        "kind": "rescale",
        "text": after["text"],
        "report": report,
        "model": model_reference,
        "metrics": {"before": before, "after": after},
        "comparison": {"before": before["text"], "after": after["text"]},
        "validity": validity,
        "provenance": provenance,
        "artifact": {
            "id": artifact_id,
            "path": str(final),
            "simulated": False,
            "kind": "kykeon-real-checkpoint",
        },
        "device": device_info,
        "seed_note": SEED_NOTE,
        "source_hashes": after_hashes,
        "started_at": started,
        "finished_at": _now(),
        "versions": library_versions(),
    }


def _fmt_metric(metric: dict[str, Any], unit: str) -> str:
    if not isinstance(metric, dict):
        return "unavailable"
    if metric.get("status") == "unavailable":
        return f"unavailable ({metric.get('reason')})"
    if "median" in metric and "tokens_per_second" not in metric and "samples" in metric:
        label = ""
        p95 = metric.get("p95")
        if isinstance(p95, dict) and p95.get("status") == "unavailable":
            p95_text = f"p95 unavailable ({p95.get('reason')})"
        elif isinstance(p95, dict):
            p95_text = (
                f"p95 {p95.get('value'):.4f} {unit} "
                f"[{p95.get('label')}, n={p95.get('sample_count')}]"
            )
        else:
            p95_text = "p95 unavailable"
        return (
            f"median {metric['median']:.4f} {unit} "
            f"(n={metric.get('sample_count')}; {p95_text})"
        )
    if "median" in metric and metric.get("definition"):
        return f"median {metric['median']:.4f} tokens/s (n={metric.get('sample_count')})"
    if "max_bytes" in metric:
        return f"max {metric['max_bytes']} bytes (n={metric.get('sample_count')})"
    return "unavailable"


def _inference_report(plan, configured, comparison, device) -> str:
    lines = [
        "LOCAL EXECUTION",
        "Real generation. This is not a simulated fixture.",
        f"Device: {device.get('name')} ({device.get('id')})",
        f"Input tokens: {configured['input_token_count']}",
        f"Output tokens: {configured['output_token_count']}",
        f"Time to first token: {_fmt_metric(configured['time_to_first_token_seconds'], 's')}",
        f"End-to-end latency: {_fmt_metric(configured['end_to_end_seconds'], 's')}",
        f"Output-token throughput: {_fmt_metric(configured['throughput'], 'tokens/s')}",
        f"Peak CUDA allocated: {_fmt_metric(configured['peak_cuda_allocated_bytes'], 'bytes')}",
        f"Peak CUDA reserved: {_fmt_metric(configured['peak_cuda_reserved_bytes'], 'bytes')}",
        f"Warm-up runs: {configured['warmup_runs']}. Measured runs: {configured['measured_runs']}.",
        SEED_NOTE,
        "",
        "Configured output",
        configured["text"],
    ]
    for note in configured.get("notes") or []:
        lines.append(note)
    if comparison:
        lines.extend(
            [
                "",
                "Default-settings output",
                "Same prompt, tokenizer, and seed. Sampling settings are the workbench defaults.",
                comparison["default_settings"]["text"],
            ]
        )
    lines.append("")
    lines.append("No capability-retention or accuracy score was calculated.")
    return "\n".join(lines)


def _rescale_report(plan, before, after, validity, final, device) -> str:
    tensor = plan["tensor"]
    lines = [
        "LOCAL WEIGHT RESCALING",
        "The source checkpoint was not modified. A derived checkpoint was reloaded before this export was recorded.",
        "Saving the checkpoint does not mean the model improved.",
        f"Tensor: {tensor['name']}",
        f"Shape: {tensor['shape']}",
        f"Dtype: {tensor['dtype']}",
        f"Parameters: {tensor['parameters']}",
        f"Scale: {plan['scale']:g}",
        f"Device: {device.get('name')} ({device.get('id')})",
        f"Artifact: {final}",
        "",
        "Source output",
        before["text"],
        f"Source input tokens: {before['input_token_count']}",
        f"Source output tokens: {before['output_token_count']}",
        f"Source time to first token: {_fmt_metric(before['time_to_first_token_seconds'], 's')}",
        f"Source end-to-end latency: {_fmt_metric(before['end_to_end_seconds'], 's')}",
        f"Source throughput: {_fmt_metric(before['throughput'], 'tokens/s')}",
        f"Source peak CUDA allocated: {_fmt_metric(before['peak_cuda_allocated_bytes'], 'bytes')}",
        "",
        "Derived output",
        after["text"],
        f"Derived input tokens: {after['input_token_count']}",
        f"Derived output tokens: {after['output_token_count']}",
        f"Derived time to first token: {_fmt_metric(after['time_to_first_token_seconds'], 's')}",
        f"Derived end-to-end latency: {_fmt_metric(after['end_to_end_seconds'], 's')}",
        f"Derived throughput: {_fmt_metric(after['throughput'], 'tokens/s')}",
        f"Derived peak CUDA allocated: {_fmt_metric(after['peak_cuda_allocated_bytes'], 'bytes')}",
        "",
        "The same prompt, tokenizer, seed, and generation settings were used for both runs.",
        SEED_NOTE,
        f"Warm-up runs: {before['warmup_runs']}. Measured runs: {before['measured_runs']}.",
        "Validity: finite scaled tensor, source unchanged, reload succeeded, derived inference succeeded.",
        "Capability retention: unavailable. No accuracy score was calculated.",
        f"Only selected tensor changed: {validity['only_selected_tensor_changed']}.",
    ]
    return "\n".join(lines)


def _lineage(source: Path, roots: list[Path] | tuple[Path, ...]) -> list[dict[str, Any]]:
    path = source / "kykeon-provenance.json"
    if not path.exists() and not path.is_symlink():
        return []
    real = within_any(roots, path)
    prior = json.loads(real.read_text(encoding="utf-8"))
    history = list(prior.get("lineage") or [])
    history.append(
        {
            "artifact_id": prior.get("artifact_id"),
            "operation": prior.get("operation"),
            "source_path": (prior.get("source") or {}).get("path"),
        }
    )
    return history


def _measurement_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_token_count": metrics.get("input_token_count"),
        "output_token_count": metrics.get("output_token_count"),
        "time_to_first_token_seconds": metrics.get("time_to_first_token_seconds"),
        "end_to_end_seconds": metrics.get("end_to_end_seconds"),
        "throughput": metrics.get("throughput"),
        "peak_cuda_allocated_bytes": metrics.get("peak_cuda_allocated_bytes"),
        "peak_cuda_reserved_bytes": metrics.get("peak_cuda_reserved_bytes"),
        "warmup_runs": metrics.get("warmup_runs"),
        "measured_runs": metrics.get("measured_runs"),
        "methodology": metrics.get("methodology"),
    }


def tokenize_preflight(directory: Path, roots, inspection: dict[str, Any], prompt: str, max_new_tokens: int, selected) -> dict[str, Any]:
    from transformers import AutoTokenizer

    directory = assert_checkpoint_contained(directory, roots)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    prompt_tokens = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
    config = dict(inspection["config"])
    config["compute_dtype"] = inspection.get("native_dtype")
    return memory_estimate(
        config,
        inspection["tensors"],
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        selected=selected,
    )


def build_plan(
    workflow: dict[str, Any],
    inspection: dict[str, Any],
    resolved: Path,
    warmup: int,
    measured: int,
    roots: list[Path] | tuple[Path, ...],
) -> dict[str, Any]:
    plan = validate_local_workflow(workflow, inspection)
    plan["resolved_path"] = str(resolved)
    plan["inspection"] = inspection
    plan["warmup_runs"] = warmup
    plan["measured_runs"] = measured
    plan["preflight"] = tokenize_preflight(
        resolved,
        roots,
        inspection,
        plan["prompt"],
        int(plan["sampling"]["max_new_tokens"]),
        plan.get("tensor"),
    )
    return plan
