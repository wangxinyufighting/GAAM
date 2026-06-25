"""
Phase 4 Milestone 6: Suite Configuration Loading

Load benchmark suite configs from JSON or YAML, apply CLI overrides,
and write resolved configurations.
"""

import json
from pathlib import Path
from typing import Any

from gaam_graph.phase4_suite_schema import Phase4BenchmarkSuiteConfig


def load_json_config(config_path: Path) -> dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """
    Load configuration from YAML file.

    Raises ImportError if PyYAML is not installed.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is not installed. Install it with: pip install pyyaml"
        ) from exc

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_suite_config(config_path: Path) -> dict[str, Any]:
    """
    Load suite configuration from JSON or YAML file.

    File format detected from extension.
    """
    if config_path.suffix in {".yaml", ".yml"}:
        return load_yaml_config(config_path)
    elif config_path.suffix == ".json":
        return load_json_config(config_path)
    else:
        raise ValueError(
            f"Unsupported config format: {config_path.suffix}. "
            "Expected .json, .yaml, or .yml"
        )


def apply_cli_overrides(
    config_dict: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """
    Apply CLI argument overrides to config dictionary.

    CLI args take precedence over config file values.
    Only non-None override values are applied.
    """
    result = config_dict.copy()

    for key, value in overrides.items():
        if value is not None:
            result[key] = value

    return result


def write_resolved_config(
    config: Phase4BenchmarkSuiteConfig,
    output_path: Path,
) -> None:
    """Write resolved configuration to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config.model_dump(), f, indent=2)


def load_and_resolve_suite_config(
    config_path: Path | None = None,
    config_dict: dict[str, Any] | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> Phase4BenchmarkSuiteConfig:
    """
    Load and resolve suite configuration.

    Priority (highest to lowest):
    1. CLI overrides
    2. Config file

    Args:
        config_path: Path to config file (JSON or YAML)
        config_dict: Pre-loaded config dictionary (alternative to config_path)
        cli_overrides: CLI argument overrides

    Returns:
        Validated Phase4BenchmarkSuiteConfig

    Raises:
        ValueError: If neither config_path nor config_dict is provided
    """
    if config_path is None and config_dict is None:
        raise ValueError("Either config_path or config_dict must be provided")

    # Load base config
    if config_dict is not None:
        base_config = config_dict
    else:
        base_config = load_suite_config(config_path)

    # Apply CLI overrides
    if cli_overrides:
        resolved_dict = apply_cli_overrides(base_config, cli_overrides)
    else:
        resolved_dict = base_config

    # Validate and return
    return Phase4BenchmarkSuiteConfig.model_validate(resolved_dict)
