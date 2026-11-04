import math

import pytest

from kykeon_backend.errors import ValidationFailed
from kykeon_backend.workflow import normalize_sampling, validate_local_workflow
from tests.conftest import chain, inference_workflow, node, rescale_workflow, sampling


def _inspection():
    return {
        "tensors": [
            {
                "name": "model.layers.0.self_attn.q_proj.weight",
                "eligible": True,
                "reasons": [],
                "shape": [2, 2],
                "dtype": "F32",
                "parameters": 4,
            }
        ],
        "eligible_tensors": [
            {
                "name": "model.layers.0.self_attn.q_proj.weight",
                "eligible": True,
                "reasons": [],
                "shape": [2, 2],
                "dtype": "F32",
                "parameters": 4,
            }
        ],
    }


def test_dangling_temperature_control_is_rejected():
    workflow = inference_workflow("/models/tiny")
    dangling = node("dangling", "control-1", {"temperature": 0})
    workflow["nodes"].append(dangling)
    prompt = workflow["nodes"][1]
    workflow["edges"].append(
        {
            "id": "dangling-edge",
            "source": prompt["id"],
            "target": dangling["id"],
            "sourceHandle": "out",
            "targetHandle": "input",
        }
    )
    with pytest.raises(ValidationFailed, match="unused branch|dangling"):
        validate_local_workflow(workflow, _inspection())


def test_node_array_order_does_not_change_the_chain():
    inspection = _inspection()
    forward = inference_workflow("/models/tiny", temperature=0.2)
    reverse = {
        **forward,
        "nodes": list(reversed(forward["nodes"])),
        "edges": list(reversed(forward["edges"])),
    }
    assert validate_local_workflow(forward, inspection)["sampling"]["temperature"] == 0.2
    assert validate_local_workflow(reverse, inspection)["sampling"]["temperature"] == 0.2
    assert [step["definition_id"] for step in validate_local_workflow(reverse, inspection)["steps"]] == [
        "model-2",
        "prompt",
        "sampling",
        "generate",
    ]


def test_repeated_controls_follow_connection_order():
    model = node("m", "model-2", {"name": "tiny", "path": "/models/tiny", "architecture": "LlamaForCausalLM"})
    prompt = node("p", "prompt", {"prompt": "hello"})
    first = node("c-first", "control-1", {"temperature": 0.2})
    second = node("c-second", "control-1", {"temperature": 1.4})
    generate = node("g", "generate", {})
    nodes = [second, model, generate, first, prompt]
    workflow = {
        "version": 1,
        "name": "repeat",
        "mode": "inference",
        "nodes": nodes,
        "edges": chain([model, prompt, first, second, generate]),
    }
    plan = validate_local_workflow(workflow, _inspection())
    assert plan["sampling"]["temperature"] == 1.4
    assert [step["node_id"] for step in plan["steps"]] == ["m", "p", "c-first", "c-second", "g"]


def test_incompatible_ports_cycles_and_disconnected_nodes():
    workflow = inference_workflow("/models/tiny")
    workflow["edges"][0]["target"] = workflow["nodes"][3]["id"]
    workflow["edges"][0]["targetHandle"] = "input"
    with pytest.raises(ValidationFailed, match="incompatible|connect"):
        validate_local_workflow(workflow, _inspection())
    cycled = inference_workflow("/models/tiny")
    cycled["edges"].append(
        {
            "id": "back",
            "source": cycled["nodes"][2]["id"],
            "target": cycled["nodes"][2]["id"],
            "sourceHandle": "out",
            "targetHandle": "input",
        }
    )
    with pytest.raises(ValidationFailed, match="cycle|already|outgoing"):
        validate_local_workflow(cycled, _inspection())
    loose = inference_workflow("/models/tiny")
    loose["nodes"].append(node("alone", "control-2", {"topK": 3}))
    with pytest.raises(ValidationFailed, match="unused branch|disconnected|one connected chain|connect the"):
        validate_local_workflow(loose, _inspection())


def test_malformed_nodes_unknown_fields_and_nonfinite_numbers():
    with pytest.raises(ValidationFailed, match="object"):
        validate_local_workflow({"version": 1, "mode": "inference", "nodes": [None], "edges": []}, _inspection())
    with pytest.raises(ValidationFailed, match="data object"):
        validate_local_workflow(
            {
                "version": 1,
                "mode": "inference",
                "nodes": [{"id": "m", "data": []}],
                "edges": [],
            },
            _inspection(),
        )
    with pytest.raises(ValidationFailed, match="id"):
        validate_local_workflow(
            {
                "version": 1,
                "mode": "inference",
                "nodes": [{"id": "../x", "data": {"definitionId": "model-2", "config": {}}}],
                "edges": [],
            },
            _inspection(),
        )
    workflow = inference_workflow("/models/tiny")
    workflow["nodes"][2]["data"]["config"]["not_a_control"] = 1
    with pytest.raises(ValidationFailed, match="Unrecognized"):
        validate_local_workflow(workflow, _inspection())
    workflow = inference_workflow("/models/tiny")
    workflow["nodes"][2]["data"]["config"]["temperature"] = math.nan
    with pytest.raises(ValidationFailed, match="finite"):
        validate_local_workflow(workflow, _inspection())
    with pytest.raises(ValidationFailed, match="finite"):
        normalize_sampling({**sampling(), "temperature": math.inf})
    rescale = rescale_workflow("/models/tiny", "model.layers.0.self_attn.q_proj.weight", 1)
    assert validate_local_workflow(rescale, _inspection())["kind"] == "rescale"
