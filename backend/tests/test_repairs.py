import json
import uuid
from pathlib import Path
from unittest.mock import Mock

import pytest

from kykeon_backend.errors import PathRejected
from kykeon_backend.jobs import JobManager, list_artifacts
from kykeon_backend.paths import output_path, atomic_json
from kykeon_backend.publication import success_result
from kykeon_backend.surgery import rescale_directory
from tests.conftest import HEADERS, inference_workflow, register, wait_job, node


def submit(world, workflow=None):
    return world['client'].post('/api/v1/jobs', headers=HEADERS, json={
        'workflow': workflow or inference_workflow(world['model'], maxTokens=2),
        'device_id': 'cpu', 'warmup_runs': 0, 'measured_runs': 1,
    })


def test_invalid_submission_releases_slot(world):
    register(world)
    response = submit(world, {'nodes': [], 'edges': [], 'version': 1, 'mode': 'inference'})
    assert response.status_code == 422
    assert world['app'].state.jobs._active is None
    created = submit(world)
    assert wait_job(world['client'], created.json()['id'])['status'] == 'succeeded'


def test_invalid_spec_releases_slot(world, monkeypatch):
    register(world)
    manager = world['app'].state.jobs
    with monkeypatch.context() as patch:
        patch.setattr(manager, '_start_thread', lambda job: None)
        created = submit(world)
    job = manager.jobs[created.json()['id']]
    spec = Path(job['directory']) / 'spec.json'
    spec.write_text('{')
    manager._run(job, spec)
    assert job['status'] == 'failed'
    assert manager._active is None
    following = submit(world)
    assert wait_job(world['client'], following.json()['id'])['status'] == 'succeeded'


@pytest.mark.parametrize('part', ['jobs', 'staging', 'checkpoints'])
def test_output_parents_cannot_escape(tmp_path, part):
    root = tmp_path / 'artifacts'; root.mkdir()
    outside = tmp_path / 'outside'; outside.mkdir()
    (root / part).symlink_to(outside, target_is_directory=True)
    with pytest.raises(PathRejected):
        atomic_json(root, root / part / 'new' / 'file.json', {'value': 1})
    assert list(outside.iterdir()) == []


def test_rescale_rejects_staging_escape_before_writing(tmp_path, tiny_source):
    root = tmp_path / 'artifacts'; root.mkdir()
    outside = tmp_path / 'outside'; outside.mkdir()
    (root / 'staging').symlink_to(outside, target_is_directory=True)
    with pytest.raises(PathRejected):
        rescale_directory(tiny_source, root / 'staging' / 'job', 'unused', 1, lambda: None, (tiny_source,), root)
    assert list(outside.iterdir()) == []


def test_malformed_result_removes_unpublished_artifact(world):
    manager = world['app'].state.jobs
    job_id = str(uuid.uuid4()); artifact_id = str(uuid.uuid4())
    directory = world['artifact'] / 'jobs' / job_id; directory.mkdir(parents=True)
    final = world['artifact'] / 'checkpoints' / artifact_id; final.mkdir()
    (final / 'kykeon-provenance.json').write_text(json.dumps({'job_id': job_id, 'artifact_id': artifact_id, 'simulated': False, 'kind': 'kykeon-real-checkpoint'}))
    (final / 'kykeon-commit.json').write_text(json.dumps({'job_id': job_id}))
    result = directory / 'result.json'; result.write_text('{')
    job = {'id': job_id, 'directory': str(directory), 'result_path': str(result), 'status': 'running'}
    manager._collect(job, 0)
    assert job['status'] == 'failed'
    assert list_artifacts(world['artifact']) == []
    assert not final.exists()


def test_result_without_success_record_is_not_published(tmp_path):
    job_id = str(uuid.uuid4())
    atomic_json(tmp_path, tmp_path / 'jobs' / job_id / 'result.json', {'execution': 'local'})
    assert success_result(tmp_path, job_id) is None
    assert list_artifacts(tmp_path) == []


def test_success_survives_backend_restart_and_result_corruption_is_rejected(world):
    register(world)
    created = submit(world); job_id = created.json()['id']
    assert wait_job(world['client'], job_id)['status'] == 'succeeded'
    original = world['app'].state.jobs
    original.jobs[job_id]['thread'].join(timeout=5)
    original.shutdown(); original._lock_file.close()
    restored = JobManager(world['settings'], original.registry, original.roots)
    try:
        assert restored.by_submission_id(job_id)['status'] == 'succeeded'
        assert restored.result(job_id)['execution'] == 'local'
        result_path = Path(restored.jobs[job_id]['result_path'])
        result_path.write_text('{')
        assert success_result(world['artifact'], job_id) is None
    finally:
        restored.shutdown(); restored._lock_file.close()


@pytest.mark.parametrize('field,value', [('mode', []), ('execution', {}), ('sourceHandle', [])])
def test_malformed_graph_fields_return_422(world, field, value):
    register(world)
    workflow = inference_workflow(world['model'])
    if field == 'sourceHandle': workflow['edges'][0][field] = value
    else: workflow[field] = value
    response = world['client'].post('/api/v1/workflows/validate', headers=HEADERS,
                                  json={'workflow': workflow, 'device_id': 'cpu'})
    assert response.status_code == 422


def test_baseline_uses_selected_seed(world):
    register(world)
    workflow = inference_workflow(world['model'], maxTokens=2, seed=731)
    workflow['nodes'].append(node('compare', 'compare', {}))
    workflow['edges'].append({'id': 'compare-edge', 'source': 'g', 'target': 'compare', 'sourceHandle': 'out', 'targetHandle': 'input'})
    from kykeon_backend.pipeline import _inference_job
    from kykeon_backend.workflow import validate_local_workflow
    plan = validate_local_workflow(workflow, {})
    plan.update(warmup_runs=0, measured_runs=1)
    recorded = []
    def measure(*args, **kwargs):
        recorded.append(args[3]['seed'])
        return {'text': 'output'}
    from unittest.mock import patch
    with patch('kykeon_backend.pipeline.measure_generation', side_effect=measure), patch('kykeon_backend.pipeline._inference_report', return_value='report'), patch('kykeon_backend.pipeline.library_versions', return_value={}):
        _inference_job(plan, {'model': None, 'tokenizer': None}, None, {}, lambda *a: None, lambda: None, '', [])
    assert recorded == [731, 731]


def test_cancel_before_publication_prevents_success_record(world):
    manager = world['app'].state.jobs
    job_id = str(uuid.uuid4())
    directory = world['artifact'] / 'jobs' / job_id; directory.mkdir(parents=True)
    result = directory / 'result.json'; result.write_text(json.dumps({'execution': 'local'}))
    cancel = directory / 'cancel'; cancel.write_text('cancel')
    job = {'id': job_id, 'directory': str(directory), 'result_path': str(result),
           'cancel_path': str(cancel), 'status': 'cancel_requested'}
    manager._collect(job, 0)
    assert job['status'] == 'cancelled'
    assert success_result(world['artifact'], job_id) is None


def test_failed_publication_is_invisible(world, monkeypatch):
    from kykeon_backend.publication import publish_success
    job_id = str(uuid.uuid4())
    result = {'execution': 'local', 'kind': 'inference'}
    atomic_json(world['artifact'], world['artifact'] / 'jobs' / job_id / 'result.json', result)
    def fail(*args): raise OSError('publication disk full')
    monkeypatch.setattr('kykeon_backend.publication.atomic_json', fail)
    with pytest.raises(OSError): publish_success(world['artifact'], job_id, result)
    assert success_result(world['artifact'], job_id) is None


def test_restart_marks_interrupted_job_failed_and_removes_staging(world):
    manager = world['app'].state.jobs
    job_id = str(uuid.uuid4())
    atomic_json(world['artifact'], world['artifact'] / 'jobs' / job_id / 'state.json', {'status': 'running'})
    staging = world['artifact'] / 'staging' / job_id; staging.mkdir(parents=True)
    (staging / 'partial').write_text('incomplete')
    manager.shutdown(); manager._lock_file.close()
    recovered = JobManager(world['settings'], manager.registry, manager.roots)
    try:
        assert recovered.get(job_id)['status'] == 'failed'
        assert not staging.exists()
        assert recovered._active is None
    finally:
        recovered.shutdown(); recovered._lock_file.close()
