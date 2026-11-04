from kykeon_backend.checkpoints import file_record
from tests.conftest import HEADERS, inference_workflow, register, rescale_workflow, wait_job


def test_real_inference_uses_tokenizer_counts_and_hides_missing_cuda_metrics(world):
    register(world)
    workflow = inference_workflow(world["model"], maxTokens=4, stop="XY")
    created = world["client"].post(
        "/api/v1/jobs",
        headers=HEADERS,
        json={"workflow": workflow, "device_id": "cpu", "warmup_runs": 0, "measured_runs": 1},
    )
    assert created.status_code == 200, created.text
    status = wait_job(world["client"], created.json()["id"])
    assert status["status"] == "succeeded", status
    result = world["client"].get(f"/api/v1/jobs/{created.json()['id']}/result").json()
    assert result["execution"] == "local"
    assert result["metrics"]["input_token_count"] > len("hello".split())
    assert result["metrics"]["peak_cuda_allocated_bytes"]["status"] == "unavailable"
    assert result["metrics"]["peak_cuda_allocated_bytes"].get("max_bytes") is None
    assert "bitwise" in result["seed_note"].lower() or "bitwise" in result["report"].lower()
    assert "SIMULATION" not in result["report"]
    assert result["validity"]["capability_retention_score"]["status"] == "unavailable"


def test_rescale_job_exports_reloads_and_keeps_the_source(world):
    record = register(world)
    tensor = record["eligible_tensors"][0]["name"]
    source_weight = world["model"] / "model.safetensors"
    before = file_record(source_weight, world["model"])
    workflow = rescale_workflow(world["model"], tensor, 1.25, maxTokens=4)
    created = world["client"].post(
        "/api/v1/jobs",
        headers=HEADERS,
        json={"workflow": workflow, "device_id": "cpu", "warmup_runs": 0, "measured_runs": 1},
    )
    assert created.status_code == 200, created.text
    status = wait_job(world["client"], created.json()["id"])
    assert status["status"] == "succeeded", status
    result = world["client"].get(f"/api/v1/jobs/{created.json()['id']}/result").json()
    assert file_record(source_weight, world["model"]) == before
    assert result["validity"]["only_selected_tensor_changed"] is True
    assert result["validity"]["reload_succeeded"] is True
    assert result["artifact"]["simulated"] is False
    provenance = result["provenance"]
    assert provenance["simulated"] is False
    assert provenance["operation"]["tensor"] == tensor
    assert provenance["operation"]["scale"] == 1.25
    assert provenance["operation"]["version"] == "1.0.0"
    assert provenance["source"]["files"]
    assert provenance["versions"]["torch"]
    assert provenance["device"]["id"] == "cpu"
    assert provenance["generation"]["seed"] == 7
    artifacts = world["client"].get("/api/v1/artifacts").json()["artifacts"]
    assert artifacts[0]["id"] == result["artifact"]["id"]
    reloaded = world["client"].post(
        "/api/v1/models/register",
        headers=HEADERS,
        json={"path": result["artifact"]["path"]},
    )
    assert reloaded.status_code == 200, reloaded.text
    assert reloaded.json()["architecture"] == "LlamaForCausalLM"
    follow = inference_workflow(result["artifact"]["path"], maxTokens=2)
    second = world["client"].post(
        "/api/v1/jobs",
        headers=HEADERS,
        json={"workflow": follow, "device_id": "cpu", "warmup_runs": 0, "measured_runs": 1},
    )
    assert second.status_code == 200, second.text
    assert wait_job(world["client"], second.json()["id"])["status"] == "succeeded"


def test_default_measurement_counts_are_one_warmup_and_five_runs(world):
    register(world)
    created = world["client"].post(
        "/api/v1/jobs",
        headers=HEADERS,
        json={"workflow": inference_workflow(world["model"], maxTokens=2), "device_id": "cpu"},
    )
    assert created.status_code == 200, created.text
    status = wait_job(world["client"], created.json()["id"])
    assert status["status"] == "succeeded", status
    result = world["client"].get(f"/api/v1/jobs/{created.json()['id']}/result").json()
    assert result["metrics"]["warmup_runs"] == 1
    assert result["metrics"]["measured_runs"] == 5
    assert result["metrics"]["sample_count"] == 5
    assert result["metrics"]["end_to_end_seconds"]["p95"]["label"] == "small-sample estimate"


def test_cancel_and_worker_failure_leave_no_successful_export(world):
    register(world)
    workflow = inference_workflow(world["model"], maxTokens=2)
    paused = world["client"].post(
        "/api/v1/jobs",
        headers=HEADERS,
        json={
            "workflow": workflow,
            "device_id": "cpu",
            "warmup_runs": 0,
            "measured_runs": 1,
            "test_hook": {"pause_seconds": 30},
        },
    )
    assert paused.status_code == 200, paused.text
    world["client"].post(f"/api/v1/jobs/{paused.json()['id']}/cancel", headers=HEADERS)
    cancelled = wait_job(world["client"], paused.json()["id"])
    assert cancelled["status"] == "cancelled"
    assert world["client"].get("/api/v1/artifacts").json()["artifacts"] == []

    failed = world["client"].post(
        "/api/v1/jobs",
        headers=HEADERS,
        json={
            "workflow": workflow,
            "device_id": "cpu",
            "warmup_runs": 0,
            "measured_runs": 1,
            "test_hook": {"fail": "injected worker failure"},
        },
    )
    assert failed.status_code == 200, failed.text
    failure = wait_job(world["client"], failed.json()["id"])
    assert failure["status"] == "failed"
    assert "injected" in failure["error"]
    assert world["client"].get("/api/v1/artifacts").json()["artifacts"] == []


def test_unresponsive_worker_is_terminated(world):
    register(world)
    created = world["client"].post(
        "/api/v1/jobs",
        headers=HEADERS,
        json={
            "workflow": inference_workflow(world["model"], maxTokens=2),
            "device_id": "cpu",
            "test_hook": {"ignore_cancel_seconds": 30},
        },
    )
    assert created.status_code == 200, created.text
    world["client"].post(f"/api/v1/jobs/{created.json()['id']}/cancel", headers=HEADERS)
    finished = wait_job(world["client"], created.json()["id"], timeout=30)
    assert finished["status"] in {"cancelled", "failed"}
    assert world["client"].get("/api/v1/artifacts").json()["artifacts"] == []
    staging = world["artifact"] / "staging"
    assert not staging.exists() or not any(staging.iterdir())


def test_unsupported_real_workflow_is_rejected_before_a_worker_starts(world):
    register(world)
    workflow = inference_workflow(world["model"])
    workflow["nodes"].append(
        {
            "id": "code",
            "type": "workbench",
            "position": {"x": 1, "y": 1},
            "data": {
                "definitionId": "custom-code",
                "config": {"source": "print('KYKEON_CUSTOM_EXECUTED')", "format": "PyTorch"},
            },
        }
    )
    response = world["client"].post(
        "/api/v1/jobs",
        headers=HEADERS,
        json={"workflow": workflow, "device_id": "cpu", "warmup_runs": 0, "measured_runs": 1},
    )
    assert response.status_code == 422
    assert "stored only" in response.json()["error"]
    assert "KYKEON_CUSTOM_EXECUTED" not in response.json()["error"]
    assert world["client"].get("/api/v1/jobs/missing").status_code == 404
