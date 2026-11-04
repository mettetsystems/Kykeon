import json
from pathlib import Path

import pytest

from kykeon_backend.checkpoints import inspect_checkpoint
from kykeon_backend.errors import PathRejected
from kykeon_backend.paths import contained


def test_rejects_checkpoint_file_symlink_outside_the_root(tiny_source: Path, tmp_path: Path):
    root = tmp_path / "models"
    root.mkdir()
    model = root / "tiny"
    model.mkdir()
    for item in tiny_source.iterdir():
        if item.name == "config.json":
            continue
        (model / item.name).write_bytes(item.read_bytes())
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "config.json"
    secret.write_text(tiny_source.joinpath("config.json").read_text(), encoding="utf-8")
    (model / "config.json").symlink_to(secret)
    with pytest.raises(PathRejected, match="escapes"):
        inspect_checkpoint(model, (root,))


def test_rejects_index_traversal(tiny_source: Path, tmp_path: Path):
    root = tmp_path / "models"
    model = root / "tiny"
    model.mkdir(parents=True)
    for item in tiny_source.iterdir():
        (model / item.name).write_bytes(item.read_bytes())
    outside = tmp_path / "stolen.safetensors"
    outside.write_bytes(b"not-weights")
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.layers.0.self_attn.q_proj.weight": "../../stolen.safetensors"}}),
        encoding="utf-8",
    )
    with pytest.raises(PathRejected, match="index|escapes|outside"):
        inspect_checkpoint(model, (root,))


def test_rejects_external_tokenizer_reference(tiny_source: Path, tmp_path: Path):
    root = tmp_path / "models"
    model = root / "tiny"
    model.mkdir(parents=True)
    for item in tiny_source.iterdir():
        target = model / item.name
        target.write_bytes(item.read_bytes())
    config_path = model / "tokenizer_config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["vocab_file"] = str(tmp_path / "outside.model")
    (tmp_path / "outside.model").write_bytes(b"external")
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PathRejected, match="escapes"):
        inspect_checkpoint(model, (root,))


def test_internal_symlink_and_plain_checkpoint_are_readable(tiny_source: Path, tmp_path: Path):
    root = tmp_path / "models"
    model = root / "tiny"
    model.mkdir(parents=True)
    for item in tiny_source.iterdir():
        (model / item.name).write_bytes(item.read_bytes())
    real_config = model / "config.json"
    linked = model / "config-link.json"
    linked.write_bytes(real_config.read_bytes())
    before = real_config.read_bytes()
    inspection = inspect_checkpoint(model, (root,))
    assert inspection["architecture"] == "LlamaForCausalLM"
    assert real_config.read_bytes() == before
    assert contained(root, model) == model.resolve()
