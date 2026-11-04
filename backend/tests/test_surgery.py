import shutil

import pytest
import torch
from safetensors import safe_open

from kykeon_backend.checkpoints import file_record, inspect_checkpoint
from kykeon_backend.surgery import assert_rescale_effect, rescale_directory


def _hashes(directory):
    return [
        file_record(path, directory)
        for path in sorted(item for item in directory.rglob("*") if item.is_file())
    ]


def _tensor(directory, name):
    for shard in directory.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            if name in handle.keys():
                return handle.get_tensor(name)
    raise AssertionError(name)


def test_unit_scale_preserves_values_and_source(tmp_path, tiny_source):
    source = tmp_path / "source"
    shutil.copytree(tiny_source, source)
    before = _hashes(source)
    tensor = inspect_checkpoint(source)["eligible_tensors"][0]["name"]
    original = _tensor(source, tensor).clone()
    dest = tmp_path / "same"
    rescale_directory(source, dest, tensor, 1.0, lambda: None, (source,))
    effect = assert_rescale_effect(source, dest, tensor, 1.0)
    assert effect["values_preserved"] is True
    assert torch.equal(original, _tensor(dest, tensor))
    assert _hashes(source) == before


def test_non_unit_scale_changes_only_the_selected_tensor(tmp_path, tiny_source):
    source = tmp_path / "source"
    shutil.copytree(tiny_source, source)
    before = _hashes(source)
    tensor = inspect_checkpoint(source)["eligible_tensors"][0]["name"]
    original = _tensor(source, tensor).clone()
    dest = tmp_path / "scaled"
    rescale_directory(source, dest, tensor, 1.25, lambda: None, (source,))
    assert_rescale_effect(source, dest, tensor, 1.25)
    assert torch.equal(_tensor(dest, tensor), original * 1.25)
    assert _hashes(source) == before
    others = [
        row["name"]
        for row in inspect_checkpoint(source)["tensors"]
        if row["name"] != tensor
    ]
    for name in others:
        assert torch.equal(_tensor(source, name), _tensor(dest, name))


def test_reload_after_rescale_matches_the_scaled_tensor(tmp_path, tiny_source):
    source = tmp_path / "source"
    shutil.copytree(tiny_source, source)
    tensor = inspect_checkpoint(source)["eligible_tensors"][0]["name"]
    dest = tmp_path / "reloaded"
    rescale_directory(source, dest, tensor, 0.5, lambda: None, (source,))
    from transformers import LlamaForCausalLM

    model = LlamaForCausalLM.from_pretrained(
        dest, local_files_only=True, trust_remote_code=False, dtype=torch.float32
    )
    loaded = model.state_dict()[tensor]
    assert torch.allclose(loaded, _tensor(source, tensor).float() * 0.5)


def test_release_slot_drops_the_only_model_reference():
    import gc
    import weakref

    from kykeon_backend.pipeline import release_slot

    class Box:
        pass

    sentinel = Box()
    ref = weakref.ref(sentinel)
    slot = {"model": sentinel, "tokenizer": object()}
    del sentinel
    release_slot(slot, None)
    gc.collect()
    assert slot["model"] is None
    assert ref() is None


def test_release_slot_drops_cuda_storage_when_cuda_is_available():
    import torch

    from kykeon_backend.pipeline import release_slot

    if not torch.cuda.is_available():
        return
    device = torch.device("cuda:0")
    before = torch.cuda.memory_allocated(device)
    slot = {"model": torch.empty(4096, device=device), "tokenizer": None}
    assert torch.cuda.memory_allocated(device) > before
    release_slot(slot, device)
    assert torch.cuda.memory_allocated(device) <= before

