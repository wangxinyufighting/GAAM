"""
Phase 4 Milestone 6: Release Package Builder

Build reproducible release packages with configs, reports, and checksums.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()


def write_checksums(
    files: list[Path],
    output_path: Path,
) -> None:
    """Write SHA256 checksums for files."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for file_path in sorted(files):
        if not file_path.exists():
            continue

        checksum = compute_file_sha256(file_path)
        # Use relative path from output_path's directory
        rel_path = file_path.relative_to(output_path.parent)
        lines.append(f"{checksum}  {rel_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_environment_report(
    output_path: Path,
) -> None:
    """Write environment report with Python version and key packages."""
    import platform

    env_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": {},
    }

    # Try to get package versions
    try:
        import importlib.metadata

        packages_to_check = [
            "torch",
            "transformers",
            "numpy",
            "pydantic",
        ]

        for package in packages_to_check:
            try:
                version = importlib.metadata.version(package)
                env_info["packages"][package] = version
            except importlib.metadata.PackageNotFoundError:
                env_info["packages"][package] = "not installed"

    except ImportError:
        pass

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)


def write_artifact_manifest(
    suite_dir: Path,
    release_dir: Path,
    output_path: Path,
) -> None:
    """Write artifact manifest listing all release files."""
    artifacts = {
        "suite_dir": str(suite_dir),
        "release_dir": str(release_dir),
        "files": [],
    }

    # List all files in release directory
    for file_path in sorted(release_dir.rglob("*")):
        if file_path.is_file():
            rel_path = file_path.relative_to(release_dir)
            artifacts["files"].append({
                "path": str(rel_path),
                "size_bytes": file_path.stat().st_size,
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=2)


def write_readme(
    suite_config: dict[str, Any],
    suite_manifest: dict[str, Any],
    output_path: Path,
) -> None:
    """Write README with reproduction instructions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# Benchmark Suite: {suite_config.get('suite_name', 'Unknown')}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"Suite ID: `{suite_manifest.get('suite_id')}`")
    lines.append(f"Status: `{suite_manifest.get('status')}`")
    lines.append(f"Seeds: {suite_config.get('seeds')}")
    lines.append(f"Rounds: {suite_config.get('num_rounds')}")
    lines.append("")

    lines.append("## Dataset")
    lines.append("")
    lines.append(f"Split manifest: `{suite_config.get('split_manifest_path')}`")
    lines.append("")

    lines.append("## Models")
    lines.append("")
    lines.append(f"Memory Builder: `{suite_config.get('memory_builder_model_path', 'N/A')}`")
    lines.append(f"Question Agent: `{suite_config.get('question_agent_model_path', 'N/A')}`")
    lines.append(f"Answerer: `{suite_config.get('answerer_model_path', 'N/A')}`")
    lines.append("")

    lines.append("## Configuration")
    lines.append("")
    lines.append(f"Trainer backend: `{suite_config.get('trainer_backend')}`")
    lines.append(f"Trainer mode: `{suite_config.get('trainer_mode')}`")
    lines.append(f"Train records per round: {suite_config.get('train_records_per_round', 'all')}")
    lines.append(f"Dev records per round: {suite_config.get('dev_records_per_round', 'all')}")
    lines.append(f"Test records limit: {suite_config.get('test_records_limit', 'all')}")
    lines.append("")

    lines.append("## Reproduction")
    lines.append("")
    lines.append("To reproduce this benchmark suite:")
    lines.append("")
    lines.append("```bash")
    lines.append("# Set environment variables")
    lines.append("export PYTHONPATH=gaam_memory_training")
    lines.append("")
    lines.append("# Run benchmark suite")
    lines.append("bash reproduce.sh")
    lines.append("```")
    lines.append("")

    lines.append("## Expected Output")
    lines.append("")
    lines.append("The reproduction script will create:")
    lines.append("")
    lines.append("- `runs/seed_*/` - Per-seed experiment outputs")
    lines.append("- `aggregate/` - Aggregated metrics and reports")
    lines.append("- `suite_manifest.json` - Suite execution manifest")
    lines.append("")

    lines.append("## Known Limitations")
    lines.append("")
    lines.append("- Model checkpoints are not included in this release package")
    lines.append("- Exact reproduction requires the same model paths and dataset")
    lines.append("- Random seeds are fixed but hardware differences may cause minor variations")
    lines.append("")

    lines.append("## Leakage Policy")
    lines.append("")
    lines.append(f"Strict no-leakage: `{suite_config.get('strict_no_leakage', True)}`")
    lines.append("")
    lines.append("Test split was used for final reporting only. No model updates, ")
    lines.append("checkpoint selection, or hyperparameter tuning was performed on test data.")
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_reproduce_script(
    suite_config: dict[str, Any],
    output_path: Path,
) -> None:
    """Write bash script for reproduction."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("#!/usr/bin/env bash")
    lines.append("set -euo pipefail")
    lines.append("")
    lines.append("# Benchmark Suite Reproduction Script")
    lines.append(f"# Suite: {suite_config.get('suite_name')}")
    lines.append("")
    lines.append("echo 'Running benchmark suite reproduction...'")
    lines.append("")
    lines.append("# Run benchmark suite")
    lines.append("python scripts/run_phase4_benchmark_suite.py \\")
    lines.append("  --suite_config suite_config.resolved.json \\")
    lines.append("  --output_dir outputs/reproduced_suite \\")
    lines.append("  --resume")
    lines.append("")
    lines.append("# Aggregate results")
    lines.append("python scripts/aggregate_phase4_benchmark_suite.py \\")
    lines.append("  --suite_dir outputs/reproduced_suite")
    lines.append("")
    lines.append("echo 'Reproduction complete!'")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Make executable
    output_path.chmod(0o755)


def build_phase4_release_package(
    suite_dir: Path,
    *,
    output_dir: Path | None = None,
    include_checkpoints: bool = False,
) -> Path:
    """
    Build release package from benchmark suite.

    Args:
        suite_dir: Root directory of benchmark suite
        output_dir: Output directory for release (default: suite_dir/release)
        include_checkpoints: Include model checkpoints (default: False)

    Returns:
        Path to release directory
    """
    if output_dir is None:
        output_dir = suite_dir / "release"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load suite manifest and config
    manifest_path = suite_dir / "suite_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        suite_manifest = json.load(f)

    config_path = suite_dir / "suite_config.resolved.json"
    with open(config_path, "r", encoding="utf-8") as f:
        suite_config = json.load(f)

    # Copy essential files
    shutil.copy(config_path, output_dir / "suite_config.resolved.json")
    shutil.copy(manifest_path, output_dir / "suite_manifest.json")

    # Copy aggregate outputs
    aggregate_dir = suite_dir / "aggregate"
    if aggregate_dir.exists():
        for file_name in [
            "aggregate_metrics.json",
            "aggregate_metrics.csv",
            "seed_level_metrics.jsonl",
            "paper_table.md",
            "paper_table.tex",
            "paper_table.json",
            "learning_curves.json",
            "figure_data.json",
            "statistical_tests.json",
        ]:
            src = aggregate_dir / file_name
            if src.exists():
                shutil.copy(src, output_dir / file_name)

    # Write README
    write_readme(suite_config, suite_manifest, output_dir / "README.md")

    # Write reproduce script
    write_reproduce_script(suite_config, output_dir / "reproduce.sh")

    # Write environment report
    write_environment_report(output_dir / "environment_report.json")

    # Write artifact manifest
    write_artifact_manifest(suite_dir, output_dir, output_dir / "artifact_manifest.json")

    # Compute checksums
    checksum_files = [
        output_dir / "suite_config.resolved.json",
        output_dir / "suite_manifest.json",
        output_dir / "aggregate_metrics.json",
        output_dir / "paper_table.md",
        output_dir / "paper_table.tex",
    ]

    write_checksums(checksum_files, output_dir / "checksums.sha256")

    return output_dir
