"""Remote GPU training diagnostics for GAAM production runs."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

from gaam_graph.production_artifact_bundle import SECRET_TOKENS
from gaam_graph.production_preflight import ProductionPreflightConfig, run_production_preflight
from gaam_graph.utils import write_json


DEFAULT_PACKAGES = [
    "torch",
    "transformers",
    "vllm",
    "ray",
    "flash-attn",
    "openai",
    "pandas",
    "pyarrow",
    "accelerate",
    "peft",
]


@dataclass(frozen=True)
class RemoteTrainingDiagnosticsConfig:
    """Configuration for a remote-readiness diagnostics report."""

    preflight_config: ProductionPreflightConfig
    output_path: Path
    include_nvidia_smi: bool = True
    include_env: bool = True
    env_keys: list[str] = field(default_factory=list)
    package_names: list[str] = field(default_factory=lambda: DEFAULT_PACKAGES.copy())


def collect_remote_training_diagnostics(config: RemoteTrainingDiagnosticsConfig) -> dict[str, Any]:
    """Collect preflight, Python/package, CUDA, and redacted env diagnostics."""
    preflight = run_production_preflight(config.preflight_config)
    diagnostics = {
        "manifest_version": "gaam_remote_training_diagnostics_v1",
        "status": preflight.get("status", "failed"),
        "python": _python_info(),
        "packages": _package_versions(config.package_names),
        "torch": _torch_info(),
        "nvidia_smi": _nvidia_smi_info() if config.include_nvidia_smi else {"status": "skipped"},
        "env": _selected_env(config.env_keys) if config.include_env else {"status": "skipped"},
        "preflight": preflight,
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(config.output_path, diagnostics)
    return diagnostics


def _python_info() -> dict[str, Any]:
    return {
        "executable": sys.executable,
        "version": sys.version,
        "version_info": list(sys.version_info[:3]),
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
    }


def _package_versions(package_names: list[str]) -> dict[str, dict[str, Any]]:
    versions: dict[str, dict[str, Any]] = {}
    for name in package_names:
        try:
            versions[name] = {"installed": True, "version": importlib.metadata.version(name)}
        except importlib.metadata.PackageNotFoundError:
            versions[name] = {"installed": False, "version": None}
        except Exception as exc:
            versions[name] = {"installed": False, "version": None, "error": f"{type(exc).__name__}: {exc}"}
    return versions


def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    devices = []
    for index in range(device_count):
        try:
            devices.append(torch.cuda.get_device_name(index))
        except Exception:
            devices.append(f"cuda:{index}")
    return {
        "available": True,
        "version": getattr(torch, "__version__", None),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": cuda_available,
        "device_count": device_count,
        "device_names": devices,
        "cxx11_abi": bool(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", False)),
    }


def _nvidia_smi_info() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except FileNotFoundError:
        return {"status": "unavailable", "error": "nvidia-smi not found"}
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    return {
        "status": "succeeded" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _selected_env(keys: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for key in keys:
        values[key] = _redact_env_value(key, os.getenv(key, ""))
    return values


def _redact_env_value(key: str, value: str) -> str:
    if any(token in key.upper() for token in SECRET_TOKENS):
        return "<redacted>" if value else ""
    return value
