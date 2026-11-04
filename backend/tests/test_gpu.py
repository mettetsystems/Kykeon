import pytest

from kykeon_backend.devices import list_devices
from tests.conftest import HEADERS, inference_workflow, register, wait_job


def test_rtx_5090_tiny_fixture_when_the_device_is_usable(world):
    devices = list_devices()
    match = next(
        (
            device
            for device in devices["devices"]
            if device.get("is_rtx_5090") and device.get("pytorch_supports_architecture")
        ),
        None,
    )
    if match is None:
        pytest.skip(
            "RTX 5090 execution was not run. "
            f"cuda_available={devices['cuda_available']} cuda_error={devices['cuda_error']} "
            f"devices={devices['devices']}"
        )
    register(world)
    created = world["client"].post(
        "/api/v1/jobs",
        headers=HEADERS,
        json={
            "workflow": inference_workflow(world["model"], maxTokens=2),
            "device_id": match["id"],
            "warmup_runs": 0,
            "measured_runs": 1,
        },
    )
    assert created.status_code == 200, created.text
    status = wait_job(world["client"], created.json()["id"])
    assert status["status"] == "succeeded", status
    result = world["client"].get(f"/api/v1/jobs/{created.json()['id']}/result").json()
    assert "5090" in result["device"]["name"]
    assert result["device"]["id"] == match["id"]
    assert isinstance(result["metrics"]["peak_cuda_allocated_bytes"].get("max_bytes"), int)
    assert result["metrics"]["peak_cuda_allocated_bytes"]["max_bytes"] > 0
