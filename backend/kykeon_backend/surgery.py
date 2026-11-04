from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .checkpoints import copy_sidecar_files
from .errors import MalformedCheckpoint, UnsupportedModel


def _load_tensors(path: Path) -> dict:
    import torch
    from safetensors import safe_open

    tensors = {}
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensors[key] = handle.get_tensor(key)
    except Exception as exc:
        raise MalformedCheckpoint(f"Could not read {path.name}: {exc}") from exc
    for key, value in tensors.items():
        if not torch.isfinite(value).all():
            raise MalformedCheckpoint(f"Source tensor {key} contains non-finite values.")
    return tensors


def rescale_directory(
    source: Path,
    dest: Path,
    tensor_name: str,
    scale: float,
    cancelled: Callable[[], None],
    roots: list[Path] | tuple[Path, ...],
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file

    from .checkpoints import assert_checkpoint_contained
    from .paths import within_any, output_path

    cancelled()
    source = assert_checkpoint_contained(source, roots)
    if dest.exists():
        raise UnsupportedModel(f"Refusing to write into an existing directory: {dest}")
    # Direct callers default to the destination parent; production supplies
    # the configured artifact root to validate all intermediate components.
    output_root = artifact_root if artifact_root is not None else dest.parent
    output_path(output_root, dest)
    dest.mkdir(parents=True)
    copy_sidecar_files(source, dest, roots)
    shards = sorted(path for path in source.glob("*.safetensors"))
    if not shards:
        raise MalformedCheckpoint("No safetensors shards were found to transform.")
    found = False
    changed = []
    unchanged = 0
    for shard in shards:
        cancelled()
        real_shard = within_any(roots, shard)
        tensors = _load_tensors(real_shard)
        if tensor_name in tensors:
            weight = tensors[tensor_name]
            if not torch.is_floating_point(weight):
                raise UnsupportedModel(f"{tensor_name} is not a floating-point tensor.")
            scaled = weight * scale
            if not torch.isfinite(scaled).all():
                raise UnsupportedModel(
                    "The scaled tensor contains non-finite values. The source checkpoint was not modified."
                )
            tensors[tensor_name] = scaled
            found = True
            changed.append(tensor_name)
        unchanged += sum(1 for key in tensors if key != tensor_name)
        save_file(tensors, output_path(output_root, dest / shard.name))
        del tensors
    if not found:
        raise UnsupportedModel(f"{tensor_name} was not present in the safetensors shards.")
    return {"changed": changed, "unchanged_count": unchanged}


def assert_rescale_effect(source: Path, derived: Path, tensor_name: str, scale: float) -> dict[str, Any]:
    """Compare one tensor at a time so both checkpoints are not resident together."""
    import torch
    from safetensors import safe_open

    source_names: set[str] = set()
    derived_names: set[str] = set()
    for shard in sorted(source.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            source_names.update(handle.keys())
    for shard in sorted(derived.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            derived_names.update(handle.keys())
    if source_names != derived_names:
        raise UnsupportedModel("The derived checkpoint does not contain the same tensor names as the source.")
    changed_others: list[str] = []
    found = False
    for shard in sorted(source.glob("*.safetensors")):
        derived_shard = derived / shard.name
        with safe_open(shard, framework="pt", device="cpu") as before, safe_open(
            derived_shard, framework="pt", device="cpu"
        ) as after:
            for name in before.keys():
                source_tensor = before.get_tensor(name)
                derived_tensor = after.get_tensor(name)
                if name == tensor_name:
                    found = True
                    expected = source_tensor * scale
                    same = (
                        torch.equal(source_tensor, derived_tensor)
                        if scale == 1
                        else torch.equal(derived_tensor, expected) or torch.allclose(derived_tensor, expected)
                    )
                    if not same:
                        raise UnsupportedModel("The selected tensor does not match the requested scale.")
                    if not torch.isfinite(derived_tensor).all():
                        raise UnsupportedModel("The derived tensor is not finite.")
                elif not torch.equal(source_tensor, derived_tensor):
                    changed_others.append(name)
                del source_tensor, derived_tensor
    if not found:
        raise UnsupportedModel(f"{tensor_name} was not compared.")
    if changed_others:
        raise UnsupportedModel(
            "Tensors other than the selection changed: " + ", ".join(changed_others[:8])
        )
    return {
        "selected_tensor": tensor_name,
        "scale": scale,
        "only_selected_tensor_changed": True,
        "scaled_tensor_finite": True,
        "values_preserved": scale == 1.0,
    }
