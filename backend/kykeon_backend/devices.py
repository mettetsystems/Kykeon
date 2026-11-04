from __future__ import annotations

import subprocess
from typing import Any

from .errors import DeviceUnavailable


def nvidia_smi_report() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except FileNotFoundError:
        return {
            "available": False,
            "error": "nvidia-smi is not installed, so the NVIDIA driver could not be queried.",
        }
    except subprocess.TimeoutExpired:
        return {"available": False, "error": "nvidia-smi timed out."}
    output = (completed.stdout or "").strip()
    error = (completed.stderr or "").strip()
    if completed.returncode != 0:
        return {
            "available": False,
            "error": error or output or f"nvidia-smi exited {completed.returncode}.",
        }
    return {"available": True, "gpus": [line for line in output.splitlines() if line.strip()]}


def _torch():
    import torch

    return torch


def list_devices() -> dict[str, Any]:
    torch = _torch()
    devices: list[dict[str, Any]] = []
    cuda_error = None
    arch_list: list[str] = []
    if torch.version.cuda:
        try:
            arch_list = list(torch.cuda.get_arch_list())
        except Exception as exc:  # pragma: no cover - depends on the build
            arch_list = []
            cuda_error = f"Could not read this PyTorch build's CUDA architectures: {exc}"
    if not torch.cuda.is_available():
        if torch.version.cuda is None:
            cuda_error = (
                "This PyTorch build has no CUDA runtime. "
                "GPU execution is unavailable and will not be replaced with CPU."
            )
        else:
            cuda_error = cuda_error or (
                "torch.cuda.is_available() is false. The driver may be missing, "
                "blocked, or incompatible with this PyTorch build."
            )
    else:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            name = props.name
            major = int(props.major)
            minor = int(props.minor)
            arch = f"sm_{major}{minor}"
            supported = arch in arch_list if arch_list else False
            devices.append(
                {
                    "id": f"cuda:{index}",
                    "kind": "cuda",
                    "name": name,
                    "uuid": str(getattr(props, "uuid", "")),
                    "total_memory_bytes": int(props.total_memory),
                    "compute_capability": f"{major}.{minor}",
                    "architecture": arch,
                    "pytorch_supports_architecture": supported,
                    "is_rtx_5090": "5090" in name,
                }
            )
    devices.append(
        {
            "id": "cpu",
            "kind": "cpu",
            "name": "CPU",
            "purpose": "small-test-fixtures",
            "note": "Used only when explicitly selected. Never a silent substitute for a GPU.",
        }
    )
    return {
        "devices": devices,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_error": cuda_error,
        "torch_cuda": torch.version.cuda,
        "arch_list": arch_list,
        "selection_required": True,
        "note": (
            "Choose a device by its reported name. GPU index 0 is not assumed, "
            "and a GPU selection is never moved to CPU."
        ),
    }


def resolve_device(device_id: str, catalog: dict[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(device_id, str) or not device_id.strip():
        raise DeviceUnavailable(
            "Select an execution device. Kykeon does not assume GPU index 0 and does not switch to CPU."
        )
    if device_id == "cpu" and catalog is None:
        return {"id": "cpu", "kind": "cpu", "name": "CPU", "purpose": "small-test-fixtures"}
    info = catalog or list_devices()
    match = next((item for item in info["devices"] if item["id"] == device_id), None)
    if match is None:
        if device_id.startswith("cuda"):
            raise DeviceUnavailable(
                f"Device {device_id} is not available. "
                "Kykeon will not substitute CPU or a different GPU."
            )
        raise DeviceUnavailable(f"Unknown device {device_id}.")
    if match["kind"] == "cuda" and not match.get("pytorch_supports_architecture", False):
        raise DeviceUnavailable(
            f"{match['name']} reports {match.get('architecture')}, which this PyTorch "
            f"build does not include ({info.get('arch_list')}). The job was not moved to CPU."
        )
    return match
