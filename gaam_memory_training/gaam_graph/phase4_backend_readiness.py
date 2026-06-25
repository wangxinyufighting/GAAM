"""
Phase 4 Milestone 5: Backend Readiness Checker

Validates that required dependencies and resources are available before training.

Key functions:
- check_phase4_backend_readiness: validate backend dependencies
- check_model_paths: validate model files exist
- check_cuda_availability: check GPU availability

Design principle:
Fail early with actionable error messages. Don't start training if critical
dependencies are missing.
"""

import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from gaam_graph.phase4_trainer_schema import Phase4TrainerBackend


class Phase4BackendReadinessReport(BaseModel):
    """Backend readiness check report."""

    backend: Phase4TrainerBackend
    ready: bool
    python_version: str
    torch_available: bool = False
    cuda_available: bool = False
    transformers_available: bool = False
    verl_available: bool = False
    model_paths_valid: bool = True
    output_dir_writable: bool = True
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def check_python_version() -> tuple[bool, str]:
    """Check Python version is acceptable."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    # Require Python 3.9+
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        return False, version_str

    return True, version_str


def check_torch_available() -> bool:
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


def check_cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def check_transformers_available() -> bool:
    """Check if transformers is available."""
    try:
        import transformers
        return True
    except ImportError:
        return False


def check_verl_available() -> bool:
    """Check if verl is available."""
    try:
        import verl
        return True
    except ImportError:
        return False


def check_model_path(model_path: str | None) -> tuple[bool, list[str]]:
    """
    Check if model path exists and contains required files.

    Returns:
        (valid, issues)
    """
    if model_path is None:
        return True, []

    issues = []
    path = Path(model_path)

    if not path.exists():
        issues.append(f"Model path does not exist: {model_path}")
        return False, issues

    if not path.is_dir():
        issues.append(f"Model path is not a directory: {model_path}")
        return False, issues

    # Check for config file
    config_files = [
        "config.json",
        "model_config.json",
        "configuration.json",
    ]

    has_config = any((path / cf).exists() for cf in config_files)
    if not has_config:
        issues.append(f"Model directory missing config file: {model_path}")

    # Check for tokenizer files
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ]

    has_tokenizer = any((path / tf).exists() for tf in tokenizer_files)
    if not has_tokenizer:
        issues.append(f"Model directory missing tokenizer files: {model_path}")

    return len(issues) == 0, issues


def check_output_dir_writable(output_dir: str) -> tuple[bool, list[str]]:
    """Check if output directory is writable."""
    issues = []
    path = Path(output_dir)

    try:
        path.mkdir(parents=True, exist_ok=True)

        # Try writing a test file
        test_file = path / ".write_test"
        test_file.write_text("test")
        test_file.unlink()

        return True, []
    except Exception as e:
        issues.append(f"Output directory not writable: {output_dir} ({e})")
        return False, issues


def check_phase4_backend_readiness(
    *,
    trainer_backend: Phase4TrainerBackend,
    memory_builder_model_path: str | None = None,
    question_agent_model_path: str | None = None,
    answerer_model_path: str | None = None,
    output_dir: str | None = None,
    require_cuda: bool = False,
) -> Phase4BackendReadinessReport:
    """
    Check backend readiness for Phase 4 training.

    Args:
        trainer_backend: Trainer backend to check
        memory_builder_model_path: Memory Builder model path
        question_agent_model_path: Question Agent model path
        answerer_model_path: Answerer model path
        output_dir: Output directory path
        require_cuda: If True, require CUDA availability

    Returns:
        Readiness report with issues and warnings
    """
    issues = []
    warnings = []

    # Check Python version
    python_ok, python_version = check_python_version()
    if not python_ok:
        issues.append(f"Python version {python_version} is too old. Require 3.9+")

    # Check torch
    torch_available = check_torch_available()

    # Check CUDA
    cuda_available = check_cuda_available()

    # Check transformers
    transformers_available = check_transformers_available()

    # Check verl
    verl_available = check_verl_available()

    # Backend-specific checks
    if trainer_backend == Phase4TrainerBackend.DRY_RUN:
        # Dry-run doesn't need any heavy dependencies
        pass

    elif trainer_backend == Phase4TrainerBackend.LOCAL_GRPO:
        if not torch_available:
            issues.append("Local GRPO backend requires PyTorch. Install with: pip install torch")

        if not transformers_available:
            issues.append("Local GRPO backend requires transformers. Install with: pip install transformers")

        if require_cuda and not cuda_available:
            issues.append("CUDA required but not available. Check GPU and CUDA installation.")
        elif not cuda_available:
            warnings.append("CUDA not available. Local GRPO will run on CPU (slow).")

    elif trainer_backend == Phase4TrainerBackend.VERL:
        if not torch_available:
            issues.append("VERL backend requires PyTorch. Install with: pip install torch")

        if not transformers_available:
            issues.append("VERL backend requires transformers. Install with: pip install transformers")

        if not verl_available:
            issues.append("VERL backend requires verl. Install with: pip install verl")

        if not cuda_available:
            issues.append("VERL backend requires CUDA. Check GPU and CUDA installation.")

    # Check model paths
    model_paths_valid = True

    for name, path in [
        ("memory_builder", memory_builder_model_path),
        ("question_agent", question_agent_model_path),
        ("answerer", answerer_model_path),
    ]:
        valid, path_issues = check_model_path(path)
        if not valid:
            model_paths_valid = False
            for issue in path_issues:
                issues.append(f"[{name}] {issue}")

    # Check output directory
    output_dir_writable = True
    if output_dir:
        writable, dir_issues = check_output_dir_writable(output_dir)
        if not writable:
            output_dir_writable = False
            issues.extend(dir_issues)

    # Determine overall readiness
    ready = len(issues) == 0

    return Phase4BackendReadinessReport(
        backend=trainer_backend,
        ready=ready,
        python_version=python_version,
        torch_available=torch_available,
        cuda_available=cuda_available,
        transformers_available=transformers_available,
        verl_available=verl_available,
        model_paths_valid=model_paths_valid,
        output_dir_writable=output_dir_writable,
        issues=issues,
        warnings=warnings,
    )
