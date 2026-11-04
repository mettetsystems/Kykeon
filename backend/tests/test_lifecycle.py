import json

import pytest

from kykeon_backend.errors import ValidationFailed
from kykeon_backend.jobs import read_progress


def test_progress_keeps_an_incomplete_trailing_record_and_reports_corruption():
    data = b'{"event":"node","nodeId":"a"}\n{"event":"node"'
    events, offset = read_progress(data, 0)
    assert events == [{"event": "node", "nodeId": "a"}]
    assert offset == data.index(b"\n") + 1
    again, same = read_progress(data, offset)
    assert again == []
    assert same == offset
    finished = data + b',"nodeId":"b"}\n'
    events, offset = read_progress(finished, offset)
    assert events[-1]["nodeId"] == "b"
    corrupt, offset = read_progress(b'{"event":1}\nnot-json\n', 0)
    assert corrupt[1]["event"] == "error"
    assert offset == len(b'{"event":1}\nnot-json\n')
    with pytest.raises(ValidationFailed, match="offset"):
        read_progress(b"{}\n", 50)


def test_worker_launch_disk_malformed_cancel_and_shutdown(world):
    from pathlib import Path

    from tests.conftest import HEADERS, inference_workflow, register, wait_job

    register(world)
    workflow = inference_workflow(world["model"], maxTokens=2)
    client = world["client"]

    def submit(**hook):
        body = {
            "workflow": workflow,
            "device_id": "cpu",
            "warmup_runs": 0,
            "measured_runs": 1,
            "submission_id": __import__("uuid").uuid4().hex,
        }
        # uuid without hyphens is still accepted by uuid.UUID
        body["submission_id"] = str(__import__("uuid").uuid4())
        if hook:
            body["test_hook"] = hook
        return client.post("/api/v1/jobs", headers=HEADERS, json=body)

    missing = submit(bad_result=True)
    assert missing.status_code == 200, missing.text
    failed = wait_job(client, missing.json()["id"], timeout=30)
    assert failed["status"] == "failed"
    assert "malformed" in failed["error"].lower()
    assert client.get("/api/v1/artifacts").json()["artifacts"] == []

    paused = submit(pause_seconds=30)
    assert paused.status_code == 200, paused.text
    client.post(f"/api/v1/jobs/{paused.json()['id']}/cancel", headers=HEADERS)
    cancelled = wait_job(client, paused.json()["id"], timeout=20)
    assert cancelled["status"] == "cancelled"

    world["app"].state.jobs.shutdown()
    refused = submit()
    assert refused.status_code in {422, 409}
    world["app"].state.jobs._lock_file.close()


def test_launch_and_disk_failures_allow_the_next_job(world, monkeypatch):
    from pathlib import Path

    from kykeon_backend import jobs as jobs_module
    from tests.conftest import HEADERS, inference_workflow, register

    register(world)
    workflow = inference_workflow(world["model"], maxTokens=2)

    def submit():
        return world["client"].post(
            "/api/v1/jobs",
            headers=HEADERS,
            json={
                "workflow": workflow,
                "device_id": "cpu",
                "warmup_runs": 0,
                "measured_runs": 1,
                "submission_id": str(__import__("uuid").uuid4()),
                "test_hook": {"bad_result": True},
            },
        )

    def explode(*_args, **_kwargs):
        raise OSError("worker executable is unavailable")

    monkeypatch.setattr(jobs_module.subprocess, "Popen", explode)
    launched = submit()
    assert launched.status_code == 200, launched.text
    failed = __import__("tests.conftest", fromlist=["wait_job"]).wait_job(
        world["client"], launched.json()["id"], timeout=15
    )
    assert failed["status"] == "failed"
    assert "worker" in failed["error"].lower()
    monkeypatch.undo()

    real_write = Path.write_text

    def fail_spec(self, text, *args, **kwargs):
        if self.name == "spec.json":
            raise OSError("disk full")
        return real_write(self, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_spec)
    denied = world["client"].post(
        "/api/v1/jobs",
        headers=HEADERS,
        json={
            "workflow": workflow,
            "device_id": "cpu",
            "warmup_runs": 0,
            "measured_runs": 1,
            "submission_id": str(__import__("uuid").uuid4()),
        },
    )
    assert denied.status_code == 422
    assert "disk" in denied.json()["error"].lower() or "write" in denied.json()["error"].lower() or "job" in denied.json()["error"].lower()
    monkeypatch.undo()
    recovered = submit()
    assert recovered.status_code == 200, recovered.text
    assert __import__("tests.conftest", fromlist=["wait_job"]).wait_job(
        world["client"], recovered.json()["id"], timeout=20
    )["status"] == "failed"

