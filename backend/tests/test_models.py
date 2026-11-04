import json
import shutil

import pytest

from kykeon_backend.checkpoints import inspect_checkpoint
from kykeon_backend.devices import resolve_device
from kykeon_backend.errors import DeviceUnavailable, UnsupportedModel, ValidationFailed
from kykeon_backend.workflow import validate_local_workflow
from tests.conftest import HEADERS, inference_workflow, register, rescale_workflow


def test_register_reports_llama_tensors_and_not_a_display_name(world):
    record = register(world)
    assert record["architecture"] == "LlamaForCausalLM"
    assert record["simulated"] is False
    assert record["eligible_tensors"]
    assert all("proj.weight" in row["name"] for row in record["eligible_tensors"])
    assert record["memory_estimate"]["uncertainty"] == "high"
    assert any("not a guarantee" in line for line in record["memory_estimate"]["assumptions"])


def test_rejects_other_architectures_quantization_and_pickle(world):
    model = world["model"]
    config_path = model / "config.json"
    config = json.loads(config_path.read_text())
    config["architectures"] = ["Gpt2LMHeadModel"]
    config["model_type"] = "gpt2"
    config_path.write_text(json.dumps(config))
    denied = world["client"].post(
        "/api/v1/models/register", headers=HEADERS, json={"path": str(model)}
    )
    assert denied.status_code == 422
    assert "LlamaForCausalLM" in denied.json()["error"]

    config["architectures"] = ["LlamaForCausalLM"]
    config["model_type"] = "llama"
    config["quantization_config"] = {"quant_method": "gptq"}
    config_path.write_text(json.dumps(config))
    quantized = world["client"].post(
        "/api/v1/models/register", headers=HEADERS, json={"path": str(model)}
    )
    assert quantized.status_code == 422
    assert "Quantized" in quantized.json()["error"]

    del config["quantization_config"]
    config_path.write_text(json.dumps(config))
    (model / "pytorch_model.bin").write_bytes(b"not a real pickle")
    pickled = world["client"].post(
        "/api/v1/models/register", headers=HEADERS, json={"path": str(model)}
    )
    assert pickled.status_code == 422
    assert "pickle" in pickled.json()["error"].lower() or "Refusing" in pickled.json()["error"]


def test_malformed_safetensors_is_reported(world):
    weight = next(world["model"].glob("*.safetensors"))
    weight.write_bytes(b"not safetensors")
    response = world["client"].post(
        "/api/v1/models/register", headers=HEADERS, json={"path": str(world["model"])}
    )
    assert response.status_code == 422
    assert "safetensors" in response.json()["error"].lower() or "Malformed" in response.json()["error"]


def test_shared_and_excluded_tensors_are_rejected(world, tiny_source):
    record = register(world)
    names = {row["name"] for row in record["tensors"]}
    assert any(name.endswith("embed_tokens.weight") for name in names)
    inspection = inspect_checkpoint(world["model"])
    with pytest.raises(ValidationFailed, match="cannot be rescaled|Select one"):
        validate_local_workflow(
            rescale_workflow(world["model"], "model.embed_tokens.weight", 1.0),
            inspection,
        )
    lm_head = "lm_head.weight"
    with pytest.raises(ValidationFailed):
        validate_local_workflow(rescale_workflow(world["model"], lm_head, 1.0), inspection)
    bias = next(name for name in names if name.endswith(".bias")) if any(
        name.endswith(".bias") for name in names
    ) else None
    if bias:
        with pytest.raises(ValidationFailed):
            validate_local_workflow(rescale_workflow(world["model"], bias, 1.0), inspection)

    duplicate = world["model"] / "duplicate.safetensors"
    shutil.copy(next(world["model"].glob("model*.safetensors")), duplicate)
    shared = inspect_checkpoint(world["model"])
    projection = next(row["name"] for row in shared["tensors"] if row["name"].endswith("q_proj.weight"))
    with pytest.raises(ValidationFailed, match="shared|cannot be rescaled"):
        validate_local_workflow(rescale_workflow(world["model"], projection, 1.0), shared)


def test_invalid_scale_and_empty_selection(world):
    inspection = inspect_checkpoint(world["model"])
    tensor = inspection["eligible_tensors"][0]["name"]
    for scale in (0.49, 1.51, float("nan")):
        with pytest.raises(ValidationFailed):
            validate_local_workflow(rescale_workflow(world["model"], tensor, scale), inspection)
    with pytest.raises(ValidationFailed, match="Select one"):
        validate_local_workflow(rescale_workflow(world["model"], "  ", 1.0), inspection)


def test_saved_execution_flag_cannot_request_local(world):
    inspection = inspect_checkpoint(world["model"])
    workflow = inference_workflow(world["model"])
    workflow["execution"] = "local"
    with pytest.raises(ValidationFailed, match="cannot request local"):
        validate_local_workflow(workflow, inspection)


def test_custom_source_projected_fields_and_optimization_are_rejected(world, tmp_path):
    inspection = inspect_checkpoint(world["model"])
    workflow = inference_workflow(world["model"])
    workflow["nodes"].append(
        {
            "id": "custom",
            "type": "workbench",
            "position": {"x": 0, "y": 0},
            "data": {
                "definitionId": "custom-code",
                "config": {
                    "label": "nope",
                    "format": "PyTorch",
                    "source": "open('/tmp/kykeon-custom-executed','w').write('x')",
                    "inputType": "model",
                    "outputType": "model",
                },
            },
        }
    )
    with pytest.raises(ValidationFailed, match="stored only"):
        validate_local_workflow(workflow, inspection)
    assert not (tmp_path / "kykeon-custom-executed").exists()

    projected = inference_workflow(world["model"])
    projected["nodes"][0]["data"]["config"]["planned_mergeMethod"] = "SLERP"
    with pytest.raises(ValidationFailed, match="Projected"):
        validate_local_workflow(projected, inspection)

    optimization = {
        "version": 1,
        "name": "opt",
        "mode": "optimization",
        "nodes": [],
        "edges": [],
    }
    with pytest.raises(ValidationFailed, match="not have local adapters|Batching"):
        validate_local_workflow(optimization, inspection)


def test_gpu_selection_never_falls_back_to_cpu():
    cpu_only = {"devices": [{"id": "cpu", "kind": "cpu", "name": "CPU"}], "arch_list": []}
    with pytest.raises(DeviceUnavailable, match="will not substitute"):
        resolve_device("cuda:0", cpu_only)
    unsupported = {
        "devices": [
            {
                "id": "cuda:1",
                "kind": "cuda",
                "name": "NVIDIA GeForce RTX 5090",
                "architecture": "sm_120",
                "pytorch_supports_architecture": False,
            }
        ],
        "arch_list": ["sm_90"],
    }
    with pytest.raises(DeviceUnavailable, match="not moved to CPU"):
        resolve_device("cuda:1", unsupported)


def test_mutation_without_header_is_rejected(world):
    response = world["client"].post("/api/v1/models/register", json={"path": str(world["model"])})
    assert response.status_code == 403


def test_foreign_origin_is_rejected(world):
    response = world["client"].post(
        "/api/v1/models/register",
        headers={"X-Kykeon-Request": "1", "Origin": "https://evil.example"},
        json={"path": str(world["model"])},
    )
    assert response.status_code == 403
