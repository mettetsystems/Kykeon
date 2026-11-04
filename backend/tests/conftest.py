from __future__ import annotations

import shutil
import time
from pathlib import Path

import faulthandler

import pytest
from fastapi.testclient import TestClient

from kykeon_backend.app import create_app
from kykeon_backend.config import Settings
from kykeon_backend.tiny_fixture import build_tiny_llama

HEADERS = {"X-Kykeon-Request": "1", "Origin": "http://127.0.0.1:5173"}


@pytest.fixture(autouse=True)
def fail_if_a_test_stalls():
    """Dump a traceback and exit instead of hanging on a wedged worker."""
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture(scope="session")
def tiny_source(tmp_path_factory) -> Path:
    return build_tiny_llama(tmp_path_factory.mktemp("fixture") / "tiny-llama")


@pytest.fixture
def world(tmp_path: Path, tiny_source: Path):
    root = tmp_path / "models"
    root.mkdir()
    model = root / "tiny-llama"
    shutil.copytree(tiny_source, model)
    artifact = tmp_path / "artifacts"
    settings = Settings(
        host="127.0.0.1",
        port=8765,
        model_roots=(root,),
        artifact_root=artifact,
        allowed_origins=("http://127.0.0.1:5173", "http://localhost:5173"),
        cancel_grace_seconds=1,
        kill_grace_seconds=1,
        allow_test_hooks=True,
        job_timeout_seconds=90,
    )
    app = create_app(settings)
    client = TestClient(app, client=("127.0.0.1", 50000))
    return {
        "client": client,
        "app": app,
        "root": root,
        "model": model,
        "artifact": artifact,
        "settings": settings,
    }


def node(node_id: str, definition: str, config: dict) -> dict:
    return {
        "id": node_id,
        "type": "workbench",
        "position": {"x": 0, "y": 0},
        "data": {"definitionId": definition, "config": config},
    }


def chain(nodes: list[dict]) -> list[dict]:
    edges = []
    for index in range(len(nodes) - 1):
        edges.append(
            {
                "id": f"e{index}",
                "source": nodes[index]["id"],
                "target": nodes[index + 1]["id"],
                "sourceHandle": "out",
                "targetHandle": "input",
            }
        )
    return edges


def sampling(**overrides) -> dict:
    values = {
        "temperature": 0.7,
        "topK": 40,
        "topP": 0.9,
        "seed": 7,
        "repetition": 1.1,
        "maxTokens": 4,
        "stop": "",
    }
    values.update(overrides)
    return values


def inference_workflow(path: Path, **overrides) -> dict:
    nodes = [
        node("m", "model-2", {"name": "tiny", "path": str(path), "architecture": "ignored-name"}),
        node("p", "prompt", {"prompt": "hello"}),
        node("s", "sampling", sampling(**overrides)),
        node("g", "generate", {}),
    ]
    return {"version": 1, "name": "infer", "mode": "inference", "nodes": nodes, "edges": chain(nodes)}


def rescale_workflow(path: Path, tensor: str, scale: float, **overrides) -> dict:
    config = {"tensor": tensor, "scale": scale, "prompt": "hello", **sampling(**overrides)}
    nodes = [
        node("m", "model-2", {"name": "tiny", "path": str(path), "architecture": "LlamaForCausalLM"}),
        node("op", "operation-4", config),
        node("ev", "evaluate", {}),
        node("ex", "export", {}),
    ]
    return {"version": 1, "name": "rescale", "mode": "surgery", "nodes": nodes, "edges": chain(nodes)}


def register(world) -> dict:
    response = world["client"].post(
        "/api/v1/models/register",
        headers=HEADERS,
        json={"path": str(world["model"])},
    )
    assert response.status_code == 200, response.text
    return response.json()


def wait_job(client: TestClient, job_id: str, timeout: float = 240) -> dict:
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        last = client.get(f"/api/v1/jobs/{job_id}").json()
        if last["status"] in {"succeeded", "failed", "cancelled"}:
            return last
        time.sleep(0.25)
    raise AssertionError(f"job timed out: {last}")
