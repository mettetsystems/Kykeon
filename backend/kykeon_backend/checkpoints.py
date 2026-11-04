from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .errors import MalformedCheckpoint, PathRejected, UnsupportedModel, ValidationFailed
from .paths import within_any

FLOAT_DTYPES = {"F32", "F16", "BF16"}
PROJECTION = re.compile(
    r"^model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))\.weight$"
)
EXCLUDED_NAME = re.compile(
    r"(?:^|\.)(?:embed_tokens|lm_head|wte|wpe)(?:\.|$)|\.bias$|\.norm\.|layernorm|rotary",
    re.IGNORECASE,
)
COPY_FILES = {
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "model.safetensors.index.json",
    "chat_template.jinja",
}
PICKLE_OR_OTHER = {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".gguf", ".onnx"}
OPERATION_VERSION = "1.0.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _read_config(directory: Path) -> dict[str, Any]:
    config_path = directory / "config.json"
    if not config_path.is_file():
        raise MalformedCheckpoint("Checkpoint has no config.json.")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MalformedCheckpoint(f"config.json is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise MalformedCheckpoint("config.json must be a JSON object.")
    return payload


def _assert_llama(config: dict[str, Any], directory: Path) -> None:
    if config.get("auto_map"):
        raise UnsupportedModel(
            "This checkpoint references custom model code (auto_map). "
            "Kykeon will not load repository code."
        )
    if "quantization_config" in config or config.get("quant_method"):
        raise UnsupportedModel(
            "Quantized checkpoints are not supported. Register an ordinary unquantized Llama checkpoint."
        )
    architectures = config.get("architectures")
    model_type = config.get("model_type")
    if architectures != ["LlamaForCausalLM"] or model_type != "llama":
        raise UnsupportedModel(
            "Unsupported architecture. Local execution currently accepts only dense "
            f"LlamaForCausalLM checkpoints. This directory reports model_type={model_type!r} "
            f"and architectures={architectures!r}. The directory name is not used as evidence."
        )
    for name in directory.iterdir():
        if name.suffix.lower() in PICKLE_OR_OTHER or name.name.endswith(".safetensors.index.json"):
            if name.suffix.lower() in PICKLE_OR_OTHER:
                raise UnsupportedModel(
                    f"Refusing to load {name.name}. Pickle, GGUF, and non-safetensors weight files are not accepted."
                )


def _tensor_rows(directory: Path) -> list[dict[str, Any]]:
    from safetensors import safe_open

    files = sorted(path for path in directory.glob("*.safetensors") if path.is_file())
    if not files:
        raise UnsupportedModel(
            "No safetensors weights were found. Hugging Face directory checkpoints with safetensors are required."
        )
    rows: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    try:
        for file in files:
            with safe_open(file, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    header = handle.get_slice(key)
                    seen[key] = seen.get(key, 0) + 1
                    shape = [int(dim) for dim in header.get_shape()]
                    dtype = str(header.get_dtype())
                    rows.append(
                        {
                            "name": key,
                            "file": file.name,
                            "shape": shape,
                            "dtype": dtype,
                            "parameters": int(__import__("math").prod(shape)) if shape else 0,
                        }
                    )
    except Exception as exc:
        raise MalformedCheckpoint(f"Could not read safetensors metadata: {exc}") from exc
    for row in rows:
        row["occurrences"] = seen[row["name"]]
    return rows


def tied_names(config: dict[str, Any]) -> set[str]:
    names = {"lm_head.weight"}
    if config.get("tie_word_embeddings", False):
        names.add("model.embed_tokens.weight")
    return names


def classify_tensor(row: dict[str, Any], tied: set[str]) -> dict[str, Any]:
    name = row["name"]
    reasons: list[str] = []
    if row.get("occurrences", 1) != 1:
        reasons.append("The tensor name appears more than once and is treated as shared.")
    if name in tied or EXCLUDED_NAME.search(name):
        reasons.append("Embeddings, output heads, biases, norms, and tied parameters are excluded.")
    if not PROJECTION.match(name):
        reasons.append(
            "Only attention and MLP linear projection weights can be rescaled in this version."
        )
    if row["dtype"] not in FLOAT_DTYPES:
        reasons.append(
            f"Dtype {row['dtype']} is not a supported floating-point weight. Quantized tensors are rejected."
        )
    eligible = not reasons
    return {**row, "eligible": eligible, "reasons": reasons}


def dtype_bytes(dtype: str) -> int | None:
    return {"F32": 4, "F16": 2, "BF16": 2, "F64": 8}.get(dtype)


def memory_estimate(
    config: dict[str, Any],
    tensors: list[dict[str, Any]],
    *,
    prompt_tokens: int | None,
    max_new_tokens: int | None,
    selected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    weight_bytes = 0
    unknown = False
    for row in tensors:
        width = dtype_bytes(row["dtype"])
        if width is None:
            unknown = True
            continue
        weight_bytes += int(row["parameters"]) * width
    layers = int(config.get("num_hidden_layers") or 0)
    heads = int(config.get("num_key_value_heads") or config.get("num_attention_heads") or 0)
    hidden = int(config.get("hidden_size") or 0)
    attn_heads = int(config.get("num_attention_heads") or 1)
    head_dim = hidden // attn_heads if attn_heads else 0
    compute_width = dtype_bytes(str(config.get("compute_dtype") or "")) or None
    if compute_width is None:
        widths = [dtype_bytes(row["dtype"]) for row in tensors if dtype_bytes(row["dtype"])]
        compute_width = max(widths) if widths else None
    sequence = None
    kv_bytes = None
    if (
        prompt_tokens is not None
        and max_new_tokens is not None
        and layers
        and heads
        and head_dim
        and compute_width
    ):
        sequence = int(prompt_tokens) + int(max_new_tokens)
        kv_bytes = 2 * layers * heads * head_dim * sequence * compute_width
    selected_bytes = None
    if selected is not None:
        width = dtype_bytes(selected["dtype"])
        if width is not None:
            selected_bytes = int(selected["parameters"]) * width
    overhead = selected_bytes or 0
    activation = int(weight_bytes * 0.2) if weight_bytes else None
    resident = None
    if weight_bytes and activation is not None:
        resident = weight_bytes + activation + (kv_bytes or 0) + overhead
    return {
        "weight_bytes": weight_bytes,
        "selected_tensor_bytes": selected_bytes,
        "transformation_overhead_bytes": overhead,
        "kv_cache_bytes_upper": kv_bytes,
        "activation_allowance_bytes": activation,
        "estimated_resident_bytes": resident,
        "compute_dtype_bytes": compute_width,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": max_new_tokens,
        "sequence_tokens_upper": sequence,
        "uncertainty": "high",
        "kind": "preflight-estimate",
        "incomplete": unknown or compute_width is None or prompt_tokens is None,
        "assumptions": [
            "This is a preflight estimate, not a measurement and not a guarantee against running out of memory.",
            "Weight bytes count safetensors elements times dtype width, one resident copy.",
            "KV-cache upper bound assumes batch size 1, the model's KV heads, the tokenized prompt, and the requested new tokens.",
            "KV-cache bytes use the checkpoint compute dtype width when it is known.",
            "Activation allowance is a rough 20 percent of weight bytes and can be far from the real peak.",
            "Transformation overhead adds one extra copy of the selected tensor while rescaling.",
            "Allocator overhead and CUDA reserved memory are not fully modeled.",
        ],
    }


PATH_REFERENCE_KEYS = {
    "tokenizer_file",
    "vocab_file",
    "merges_file",
    "model_file",
    "vocab",
    "merges",
}


def _reference_target(directory: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return directory / candidate


def _walk_references(value: Any, directory: Path, roots: list[Path] | tuple[Path, ...], key: str = "") -> None:
    if isinstance(value, str):
        escaped = value.startswith("/") or value.startswith("\\") or ".." in Path(value).parts
        if key in PATH_REFERENCE_KEYS or escaped:
            if key == "name_or_path" and not escaped:
                return
            within_any(roots, _reference_target(directory, value))
        return
    if isinstance(value, dict):
        for child_key, child in value.items():
            _walk_references(child, directory, roots, str(child_key))
    elif isinstance(value, list):
        for child in value:
            _walk_references(child, directory, roots, key)


def assert_checkpoint_contained(directory: Path, roots: list[Path] | tuple[Path, ...]) -> Path:
    """Reject a checkpoint whose readable files leave the configured roots.

    Directory containment is not enough: config, tokenizer, index, and weight
    entries are checked individually, including symlink targets.
    """
    real_directory = within_any(roots, directory)
    if not real_directory.is_dir():
        raise MalformedCheckpoint("A checkpoint must be a Hugging Face model directory.")
    for child in real_directory.iterdir():
        if child.is_symlink() or child.is_file():
            within_any(roots, child)
    index = real_directory / "model.safetensors.index.json"
    if index.is_symlink() or index.exists():
        real_index = within_any(roots, index)
        try:
            payload = json.loads(real_index.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise MalformedCheckpoint(f"Safetensors index is not valid JSON: {exc}") from exc
        weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
        if not isinstance(weight_map, dict):
            raise MalformedCheckpoint("Safetensors index weight_map must be an object.")
        for tensor_name, filename in weight_map.items():
            if (
                not isinstance(filename, str)
                or Path(filename).is_absolute()
                or ".." in Path(filename).parts
            ):
                raise PathRejected(
                    f"Safetensors index entry for {tensor_name} points outside the checkpoint: {filename!r}."
                )
            within_any(roots, real_directory / filename)
    for name in ("config.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json"):
        path = real_directory / name
        if not (path.is_symlink() or path.exists()):
            continue
        real = within_any(roots, path)
        if name == "tokenizer.json":
            continue
        try:
            document = json.loads(real.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        _walk_references(document, real_directory, roots)
    return real_directory


def inspect_checkpoint(directory: Path, roots: list[Path] | tuple[Path, ...] | None = None) -> dict[str, Any]:
    search_roots = tuple(roots) if roots is not None else (directory,)
    directory = assert_checkpoint_contained(directory, search_roots)
    config = _read_config(directory)
    _assert_llama(config, directory)
    rows = _tensor_rows(directory)
    if any(row["dtype"] not in FLOAT_DTYPES | {"F64", "I64", "I32", "U32", "U8", "I8", "BOOL"} for row in rows):
        pass
    quantized = [row for row in rows if row["dtype"] not in FLOAT_DTYPES | {"F64", "I64", "I32", "U32", "BOOL"}]
    # Norms are float. Index tensors may be integer. Quantized integer weight dtypes are rejected
    # when they belong to a weight matrix rather than a small index.
    bad = [
        row
        for row in quantized
        if row["name"].endswith(".weight") or "quant" in row["name"].lower()
    ]
    if bad:
        raise UnsupportedModel(
            "Quantized or non-floating weight tensors are not supported: "
            + ", ".join(f"{row['name']} ({row['dtype']})" for row in bad[:8])
        )
    tied = tied_names(config)
    classified = [classify_tensor(row, tied) for row in rows]
    total_bytes = sum(path.stat().st_size for path in directory.iterdir() if path.is_file())
    parameter_count = sum(row["parameters"] for row in rows)
    return {
        "architecture": "LlamaForCausalLM",
        "model_type": "llama",
        "config": {
            "hidden_size": config.get("hidden_size"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_attention_heads": config.get("num_attention_heads"),
            "num_key_value_heads": config.get("num_key_value_heads"),
            "vocab_size": config.get("vocab_size"),
            "tie_word_embeddings": bool(config.get("tie_word_embeddings", False)),
            "torch_dtype": config.get("torch_dtype"),
        },
        "parameter_count": parameter_count,
        "directory_bytes": total_bytes,
        "tensors": classified,
        "eligible_tensors": [row for row in classified if row["eligible"]],
        "native_dtype": _dominant_dtype(classified),
        "memory_estimate": memory_estimate(config, classified, prompt_tokens=None, max_new_tokens=None),
        "raw_config_keys": sorted(config.keys()),
    }


def _dominant_dtype(rows: list[dict[str, Any]]) -> str | None:
    counts: dict[str, int] = {}
    for row in rows:
        if row["dtype"] in FLOAT_DTYPES and row["name"].endswith(".weight"):
            counts[row["dtype"]] = counts.get(row["dtype"], 0) + row["parameters"]
    if not counts:
        return None
    return max(counts, key=counts.get)


def require_eligible(inspection: dict[str, Any], tensor_name: str) -> dict[str, Any]:
    if not isinstance(tensor_name, str) or not tensor_name.strip():
        raise ValidationFailed(
            "Select one weight tensor. Kykeon does not default to modifying every layer."
        )
    match = next((row for row in inspection["tensors"] if row["name"] == tensor_name), None)
    if match is None:
        raise ValidationFailed(f"Tensor {tensor_name!r} is not in this checkpoint.")
    if not match["eligible"]:
        raise ValidationFailed(
            f"Tensor {tensor_name!r} cannot be rescaled. " + " ".join(match["reasons"])
        )
    return match


def copy_sidecar_files(source: Path, dest: Path, roots: list[Path] | tuple[Path, ...]) -> list[str]:
    copied: list[str] = []
    dest.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.suffix == ".py" or path.name.endswith(".py"):
            raise UnsupportedModel(f"Refusing to copy executable model code: {path.name}.")
        if path.name not in COPY_FILES:
            continue
        real = within_any(roots, path)
        if not real.is_file():
            continue
        (dest / path.name).write_bytes(real.read_bytes())
        copied.append(path.name)
    return copied
