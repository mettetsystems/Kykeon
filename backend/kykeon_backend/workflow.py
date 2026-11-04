from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from typing import Any

from .checkpoints import OPERATION_VERSION, require_eligible
from .errors import ValidationFailed

SAMPLING_DEFAULTS: dict[str, Any] = {
    "temperature": 0.7,
    "topK": 40,
    "topP": 0.9,
    "seed": 42,
    "repetition": 1.1,
    "maxTokens": 80,
    "stop": "",
}
CONTROL_FIELDS = {
    "control-1": "temperature",
    "control-2": "topK",
    "control-3": "topP",
    "control-10": "seed",
    "control-11": "repetition",
    "control-21": "maxTokens",
    "control-23": "stop",
}
INFERENCE_NODES = {
    "model-2": {"inputs": {}, "output": "model"},
    "prompt": {"inputs": {"input": "model"}, "output": "context"},
    "sampling": {"inputs": {"input": "context"}, "output": "context"},
    **{
        node: {"inputs": {"input": "context"}, "output": "context"}
        for node in CONTROL_FIELDS
    },
    "generate": {"inputs": {"input": "context"}, "output": "generation"},
    "compare": {"inputs": {"input": "generation"}, "output": "report"},
}
SURGERY_NODES = {
    "model-2": {"inputs": {}, "output": "model"},
    "operation-4": {"inputs": {"input": "model"}, "output": "model"},
    "evaluate": {"inputs": {"input": "model"}, "output": "model"},
    "export": {"inputs": {"input": "model"}, "output": "model"},
}
OPTIMIZATION_IDS = {
    "runtime-config",
    "workload",
    "benchmark",
    "optimization-report",
    "validation-gate",
    "opt-kv-cache",
    "opt-batching",
    "opt-context",
}
OPERATION_FIELDS = {
    "tensor",
    "scale",
    "prompt",
    "temperature",
    "topK",
    "topP",
    "seed",
    "repetition",
    "maxTokens",
    "stop",
}


def _fail(message: str) -> None:
    raise ValidationFailed(message)


ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")


def ensure_workflow_shape(workflow: Any) -> dict[str, Any]:
    """Reject malformed graphs before any attribute access or path extraction."""
    if not isinstance(workflow, dict):
        _fail("Workflow must be a JSON object.")
    nodes = workflow.get("nodes")
    edges = workflow.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        _fail("Workflow nodes and edges must be lists.")
    for field in ("mode", "execution"):
        if field in workflow and not isinstance(workflow[field], str):
            _fail(f"Workflow {field} must be a string.")
    seen: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict):
            _fail("Each node must be an object. Null or list nodes are rejected.")
        node_id = node.get("id")
        if not isinstance(node_id, str) or not ID_RE.match(node_id):
            _fail("Each node needs an id of letters, numbers, and .:_- up to 80 characters.")
        if node_id in seen:
            _fail(f"Duplicate node id {node_id}.")
        seen.add(node_id)
        data = node.get("data")
        if not isinstance(data, dict):
            _fail(f"Node {node_id} is missing a data object.")
        if not isinstance(data.get("definitionId"), str) or not data.get("definitionId"):
            _fail(f"Node {node_id} is missing its definition.")
        if not isinstance(data.get("config"), dict):
            _fail(f"Node {node_id} configuration must be an object.")
    edge_ids: set[str] = set()
    for edge in edges:
        if not isinstance(edge, dict):
            _fail("Each edge must be an object.")
        edge_id = edge.get("id")
        if not isinstance(edge_id, str) or not ID_RE.match(edge_id) or edge_id in edge_ids:
            _fail("Each edge needs a unique id.")
        edge_ids.add(edge_id)
        for end in ("source", "target"):
            value = edge.get(end)
            if not isinstance(value, str) or value not in seen:
                _fail(f"Edge {edge_id} references a missing {end}.")
        source_handle = edge.get("sourceHandle")
        if source_handle is not None and not isinstance(source_handle, str):
            _fail(f"Edge {edge_id} has an invalid source port.")
        handle = edge.get("targetHandle")
        if handle is not None and not isinstance(handle, str):
            _fail(f"Edge {edge_id} has an invalid target port.")
    return workflow


def graph_order(nodes: list[dict], edges: list[dict]) -> list[dict]:
    by_id = {node["id"]: node for node in nodes}
    incoming = defaultdict(list)
    for edge in edges:
        incoming[edge["target"]].append(edge["source"])
    seen: set[str] = set()
    ordered: list[dict] = []
    while len(ordered) < len(nodes):
        ready = [
            node
            for node in nodes
            if node["id"] not in seen and all(src in seen for src in incoming[node["id"]])
        ]
        if not ready:
            _fail("Connections form a cycle. Remove the loop before running.")
        for node in ready:
            seen.add(node["id"])
            ordered.append(by_id[node["id"]])
    return ordered


def _node_id(node: dict) -> str:
    data = node.get("data") or {}
    value = data.get("definitionId")
    if not isinstance(value, str):
        _fail("A node is missing its definition.")
    return value


def _config(node: dict) -> dict[str, Any]:
    data = node.get("data") or {}
    config = data.get("config")
    if not isinstance(config, dict):
        _fail("A node is missing its configuration.")
    return config


def _reject_known_unsupported(definition_id: str, config: dict[str, Any]) -> None:
    if definition_id == "custom-code":
        _fail(
            "Custom source is stored only and is not executed. "
            "Remove the custom node before local execution."
        )
    if definition_id in OPTIMIZATION_IDS or definition_id in {"optimization", "validation"}:
        _fail(
            "Batching, KV-cache tuning, synthetic benchmarks, and acceptance gates "
            "are not implemented for local execution. Run this workflow as simulation, "
            "or remove those nodes. Planned settings were not ignored."
        )
    if definition_id.startswith("surgery-"):
        _fail(
            f"{definition_id} is simulation-only. Its projected fields are not a local adapter. "
            "The run was rejected instead of ignoring them."
        )
    if definition_id.startswith("operation-") and definition_id != "operation-4":
        _fail(
            f"{definition_id} is a catalog reference without a local adapter. "
            "Only weight rescaling (operation-4) can run."
        )
    if definition_id == "custom":
        _fail(
            "Simulated custom-model records have no weights. Register a local checkpoint instead."
        )
    if any(str(key).startswith("planned_") for key in config):
        _fail(
            "Projected settings are still planned. Local execution will not ignore them. "
            "Remove the planned fields or run the workflow as simulation."
        )


def _number(name: str, value: Any, low: float, high: float, *, integer: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{name} must be a number.")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        _fail(f"{name} must be a number.")
    if not math.isfinite(numeric):
        _fail(f"{name} must be finite.")
    if integer and not numeric.is_integer():
        _fail(f"{name} must be an integer.")
    if numeric < low or numeric > high:
        _fail(f"{name} must be between {low} and {high}.")
    return numeric


def normalize_sampling(settings: dict[str, Any]) -> dict[str, Any]:
    temperature = _number("Temperature", settings.get("temperature"), 0, 2)
    top_k = int(_number("Top-K", settings.get("topK"), 1, 100, integer=True))
    top_p = _number("Top-P", settings.get("topP"), 0.01, 1)
    seed = int(_number("Random seed", settings.get("seed"), 0, 2147483647, integer=True))
    repetition = _number("Repetition penalty", settings.get("repetition"), 1, 2)
    max_tokens = int(_number("Maximum output tokens", settings.get("maxTokens"), 1, 512, integer=True))
    stop = settings.get("stop", "")
    if not isinstance(stop, str):
        _fail("Stop sequences must be a string.")
    if len(stop) > 2000:
        _fail("Stop sequences are limited to 2000 characters.")
    stops = [part.strip() for part in stop.split(",") if part.strip()]
    if len(stops) > 16:
        _fail("Use at most 16 stop sequences.")
    if any(len(part) > 200 for part in stops):
        _fail("Each stop sequence is limited to 200 characters.")
    notes: list[str] = []
    if temperature == 0:
        notes.append(
            "Temperature 0 selects greedy decoding. Top-K and Top-P are not applied."
        )
        if top_k != SAMPLING_DEFAULTS["topK"] or abs(top_p - SAMPLING_DEFAULTS["topP"]) > 1e-9:
            _fail(
                "Temperature 0 uses greedy decoding, which does not apply top-k or top-p. "
                "Reset top-k and top-p to their defaults, or set temperature above 0. "
                "Those controls were not silently ignored."
            )
    return {
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "seed": seed,
        "repetition_penalty": repetition,
        "max_new_tokens": max_tokens,
        "stop_sequences": stops,
        "notes": notes,
    }


def _apply_sampling(base: dict[str, Any], config: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    merged = dict(base)
    for key in keys:
        if key not in config:
            _fail(f"Missing sampling setting {key}.")
        merged[key] = config[key]
    return merged


def _check_prompt(text: Any) -> str:
    if not isinstance(text, str) or not text.strip():
        _fail("Prompt is empty.")
    if len(text) > 8000:
        _fail("Prompt is limited to 8000 characters.")
    return text


def validate_local_workflow(workflow: Any, inspection: dict[str, Any] | None) -> dict[str, Any]:
    workflow = ensure_workflow_shape(workflow)
    if workflow.get("version") != 1:
        _fail("Unsupported workflow version. Expected version 1.")
    if workflow.get("execution") not in {None, "simulation"}:
        _fail(
            "A workflow file cannot request local execution. "
            "Choose Local execution in the workbench. Saved workflows stay simulations until you do."
        )
    mode = workflow.get("mode")
    nodes = workflow.get("nodes")
    edges = workflow.get("edges")
    if mode not in {"inference", "surgery", "optimization", "validation"}:
        _fail("Unknown workflow mode.")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        _fail("Workflow nodes and edges must be lists.")
    if len(nodes) > 500 or len(edges) > 2000:
        _fail("Workflow is too large.")
    for node in nodes:
        _reject_known_unsupported(_node_id(node), _config(node))
    if mode in {"optimization", "validation"}:
        _fail(
            "Inference Optimization and Performance Validation do not have local adapters. "
            "Their batching, cache, and acceptance settings were not executed. Switch to simulation."
        )
    id_set = {_node_id(node) for node in nodes}
    if "operation-4" in id_set:
        allowed = SURGERY_NODES
        kind = "rescale"
    else:
        allowed = INFERENCE_NODES
        kind = "inference"
    for node in nodes:
        definition_id = _node_id(node)
        if definition_id not in allowed:
            _fail(
                f"{definition_id} cannot run in this local workflow. "
                "Unsupported nodes are rejected before a job starts."
            )
    _validate_edges(nodes, edges, allowed)
    ordered = _single_chain(nodes, edges)
    ids = [_node_id(node) for node in ordered]
    if kind == "inference":
        _expect_chain(ids, "generate")
    else:
        if ids != ["model-2", "operation-4", "evaluate", "export"]:
            _fail(
                "The local rescale workflow is one chain: registered model → weight rescaling → "
                "evaluate → export. Remove extra or unused nodes."
            )
    model_node = ordered[0]
    model_config = _config(model_node)
    for key in model_config:
        if key not in {"name", "path", "architecture"}:
            _fail(f"Unrecognized model setting {key}. It was not ignored.")
    path = model_config.get("path")
    if not isinstance(path, str) or not path.strip():
        _fail("The model node needs a local checkpoint path.")
    if inspection is None:
        _fail("Register the checkpoint before local execution.")
    prompt = ""
    settings = dict(SAMPLING_DEFAULTS)
    tensor = None
    scale = 1.0
    for node in ordered:
        definition_id = _node_id(node)
        config = _config(node)
        if definition_id == "prompt":
            if set(config) - {"prompt"}:
                _fail("Unrecognized prompt setting. It was not ignored.")
            prompt = _check_prompt(config.get("prompt"))
        elif definition_id == "sampling":
            unknown = set(config) - set(SAMPLING_DEFAULTS)
            if unknown:
                _fail(
                    "Unrecognized sampling settings: "
                    + ", ".join(sorted(str(key) for key in unknown))
                    + ". They were not ignored."
                )
            settings = _apply_sampling(settings, config, set(SAMPLING_DEFAULTS))
        elif definition_id in CONTROL_FIELDS:
            field = CONTROL_FIELDS[definition_id]
            if set(config) - {field}:
                _fail(f"Unrecognized setting on {definition_id}.")
            settings = _apply_sampling(settings, config, {field})
        elif definition_id == "operation-4":
            unknown = set(config) - OPERATION_FIELDS
            if unknown:
                _fail(
                    "Unrecognized weight-rescaling settings: "
                    + ", ".join(sorted(unknown))
                    + ". They were not ignored."
                )
            prompt = _check_prompt(config.get("prompt"))
            settings = _apply_sampling(settings, config, set(SAMPLING_DEFAULTS))
            tensor_name = config.get("tensor")
            tensor = require_eligible(inspection, "" if tensor_name is None else str(tensor_name))
            scale = _number("Scale factor", config.get("scale"), 0.5, 1.5)
        elif definition_id in {"generate", "compare", "evaluate", "export"} and config:
            _fail(f"{definition_id} has no local settings. Remove unexpected configuration.")
    sampling = normalize_sampling(settings)
    return {
        "kind": kind,
        "operation": {
            "id": "operation-4",
            "version": OPERATION_VERSION,
            "tensor": None if tensor is None else tensor["name"],
            "scale": scale,
        }
        if kind == "rescale"
        else None,
        "tensor": tensor,
        "scale": scale if kind == "rescale" else None,
        "prompt": prompt,
        "sampling": sampling,
        "model_path": path,
        "model_name": str(model_config.get("name") or "registered-checkpoint"),
        "steps": [
            {"node_id": node["id"], "definition_id": _node_id(node)} for node in ordered
        ],
        "order": [_node_id(node) for node in ordered],
        "workflow_document": {
            "version": workflow.get("version"),
            "name": workflow.get("name"),
            "mode": workflow.get("mode"),
            "nodes": [
                {
                    "id": node.get("id"),
                    "definitionId": _node_id(node),
                    "config": _config(node),
                }
                for node in ordered
            ],
        },
        "workflow_sha256": _canonical_hash(workflow),
    }


def _expect_chain(ids: list[str], terminal: str) -> None:
    if len(ids) < 3 or ids[0] != "model-2" or ids[1] != "prompt":
        _fail("Local inference must start with one model connected to one prompt.")
    body = ids[2:]
    if body[-1] == "compare":
        if len(body) < 2 or body[-2] != terminal:
            _fail("Output comparison must follow Generate text and end the chain.")
        controls = body[:-2]
    elif body[-1] == terminal:
        controls = body[:-1]
    else:
        _fail("The chain must end at Generate text, optionally followed by Output comparison.")
    for definition_id in controls:
        if definition_id != "sampling" and definition_id not in CONTROL_FIELDS:
            _fail(
                f"{definition_id} is not part of the local inference chain. "
                "Remove it or reconnect the chain."
            )
    if controls.count("sampling") > 1:
        _fail("Use at most one sampling node.")


def _single_chain(nodes: list[dict], edges: list[dict]) -> list[dict]:
    """Follow the only connected path. Array order must not change the result."""
    by_id = {node["id"]: node for node in nodes}
    outgoing: dict[str, list[dict]] = defaultdict(list)
    incoming: dict[str, list[dict]] = defaultdict(list)
    for edge in edges:
        outgoing[edge["source"]].append(edge)
        incoming[edge["target"]].append(edge)
    starts = [node for node in nodes if not incoming[node["id"]]]
    if len(starts) != 1:
        _fail(
            "Local execution accepts one connected chain. "
            "Remove the extra or disconnected node before running."
        )
    chain: list[dict] = []
    seen: set[str] = set()
    current = starts[0]
    while True:
        if current["id"] in seen:
            _fail("Connections form a cycle. Remove the loop before running.")
        seen.add(current["id"])
        chain.append(current)
        outs = outgoing[current["id"]]
        if not outs:
            break
        if len(outs) != 1:
            _fail(
                f"{_node_id(current)} has more than one outgoing connection. "
                "Remove the unused branch. A dangling control is not ignored, "
                "because it could change generation."
            )
        nxt = by_id.get(outs[0]["target"])
        if nxt is None:
            _fail("An edge references a missing node.")
        current = nxt
    unused = [node for node in nodes if node["id"] not in seen]
    if unused:
        names = ", ".join(_node_id(node) for node in unused)
        _fail(
            f"Remove the unused branch ({names}) before local execution. "
            "A disconnected node is not ignored, because it could change generation."
        )
    return chain


def _expect_counts(ids: list[str], expected: dict[str, int]) -> None:
    for definition_id, count in expected.items():
        if ids.count(definition_id) != count:
            _fail(f"Local execution expects exactly {count} {definition_id} node(s).")


def _validate_edges(nodes: list[dict], edges: list[dict], allowed: dict[str, dict]) -> None:
    by_id = {node["id"]: node for node in nodes}
    if len(by_id) != len(nodes):
        _fail("Duplicate node id.")
    occupied: set[tuple[str, str]] = set()
    for edge in edges:
        source = by_id.get(edge.get("source"))
        target = by_id.get(edge.get("target"))
        if source is None or target is None:
            _fail("An edge references a missing node.")
        if edge.get("sourceHandle") not in {None, "out"}:
            _fail("Invalid source port.")
        handle = edge.get("targetHandle") or ""
        source_def = allowed[_node_id(source)]
        target_def = allowed[_node_id(target)]
        if handle not in target_def["inputs"]:
            _fail("These port types are incompatible.")
        if source_def["output"] != target_def["inputs"][handle]:
            _fail("These port types are incompatible.")
        key = (target["id"], handle)
        if key in occupied:
            _fail("This input already has a connection.")
        occupied.add(key)
    for node in nodes:
        definition = allowed[_node_id(node)]
        for handle in definition["inputs"]:
            if (node["id"], handle) not in occupied:
                _fail(f"{_node_id(node)}: connect the {handle} input.")


def _canonical_hash(workflow: dict[str, Any]) -> str:
    import hashlib

    payload = json.dumps(workflow, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
