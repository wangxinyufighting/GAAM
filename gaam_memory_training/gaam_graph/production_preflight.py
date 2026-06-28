"""Production E2E preflight checks for GAAM native VERL training."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


FORBIDDEN_SPLIT_KEYS = {
    "question",
    "answer",
    "gold_answer",
    "benchmark_question",
    "benchmark_answer",
    "target_question",
    "target_answer",
    "oracle_answer",
}


@dataclass
class ProductionPreflightConfig:
    """Inputs required to validate a production GAAM E2E run."""

    input_path: Path
    oracle_graph_dir: Path
    split_manifest: Path
    memory_model_path: Path
    question_model_path: Path
    code_a1_root: Path
    training_output_dir: Path
    eval_output_dir: Path | None = None
    run_evaluation: bool = True
    evaluation_split: str = "test"
    memory_backend: str = "stateful_local_hf"
    answer_backend: str = "api"
    judge_backend: str = "api"
    memory_api_key: str = ""
    answer_api_key: str = ""
    judge_api_key: str = ""
    reward_judge_enabled: bool = False
    reward_judge_api_key: str = ""
    require_cuda: bool = True
    check_imports: bool = True
    require_verl_import: bool = True
    memory_num_gpus: int = 1
    question_num_gpus: int = 1
    train_batch_size: int | None = None
    questions_per_case: int | None = None
    max_reported_issues: int = 30


def run_production_preflight(config: ProductionPreflightConfig) -> dict[str, Any]:
    """Run fail-fast checks before launching the expensive production pipeline."""
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    _check_file(config.input_path, "input_path", errors)
    _check_dir(config.oracle_graph_dir, "oracle_graph_dir", errors)
    _check_file(config.split_manifest, "split_manifest", errors)
    _check_model_dir(config.memory_model_path, "memory_model_path", errors, warnings)
    _check_model_dir(config.question_model_path, "question_model_path", errors, warnings)
    _check_code_a1_root(config.code_a1_root, errors, warnings)

    if config.questions_per_case is not None and config.questions_per_case < 1:
        errors.append(f"questions_per_case must be >= 1, got {config.questions_per_case}.")
    if config.train_batch_size is not None and config.train_batch_size < 1:
        errors.append(f"train_batch_size must be >= 1, got {config.train_batch_size}.")
    if config.memory_num_gpus < 1:
        errors.append(f"memory_num_gpus must be >= 1, got {config.memory_num_gpus}.")
    if config.question_num_gpus < 1:
        errors.append(f"question_num_gpus must be >= 1, got {config.question_num_gpus}.")

    split_report = _inspect_split(config, errors, warnings)
    checks["split"] = split_report

    if config.check_imports:
        checks["imports"] = _check_imports(config, errors, warnings)
    else:
        checks["imports"] = {"status": "skipped"}

    if config.require_cuda:
        checks["cuda"] = _check_cuda(config, errors, warnings)
    else:
        checks["cuda"] = {"status": "skipped", "require_cuda": False}

    checks["api_keys"] = _check_api_keys(config, errors, warnings)

    report = {
        "status": "failed" if errors else "succeeded",
        "input_path": str(config.input_path),
        "oracle_graph_dir": str(config.oracle_graph_dir),
        "split_manifest": str(config.split_manifest),
        "memory_model_path": str(config.memory_model_path),
        "question_model_path": str(config.question_model_path),
        "code_a1_root": str(config.code_a1_root),
        "training_output_dir": str(config.training_output_dir),
        "eval_output_dir": str(config.eval_output_dir) if config.eval_output_dir else None,
        "run_evaluation": config.run_evaluation,
        "evaluation_split": config.evaluation_split,
        "memory_backend": config.memory_backend,
        "answer_backend": config.answer_backend,
        "judge_backend": config.judge_backend,
        "require_cuda": config.require_cuda,
        "check_imports": config.check_imports,
        "require_verl_import": config.require_verl_import,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
    }
    return report


def _check_file(path: Path, label: str, errors: list[str]) -> None:
    if not path.exists() or not path.is_file():
        errors.append(f"{label} does not exist or is not a file: {path}")


def _check_dir(path: Path, label: str, errors: list[str]) -> None:
    if not path.exists() or not path.is_dir():
        errors.append(f"{label} does not exist or is not a directory: {path}")


def _check_model_dir(path: Path, label: str, errors: list[str], warnings: list[str]) -> None:
    _check_dir(path, label, errors)
    if not path.exists() or not path.is_dir():
        return
    if not (path / "config.json").exists():
        warnings.append(f"{label} has no config.json; verify it is a HuggingFace model directory: {path}")
    tokenizer_files = ["tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt"]
    if not any((path / name).exists() for name in tokenizer_files):
        warnings.append(f"{label} has no common tokenizer file; verify tokenizer loading works: {path}")


def _check_code_a1_root(path: Path, errors: list[str], warnings: list[str]) -> None:
    _check_dir(path, "code_a1_root", errors)
    verl_pkg = path / "verl" / "verl"
    if not verl_pkg.exists() or not verl_pkg.is_dir():
        errors.append(f"Code-A1 vendored VERL package is missing: {verl_pkg}")
    main_ppo = path / "verl" / "trainer" / "main_ppo.py"
    if not main_ppo.exists():
        warnings.append(f"Could not find Code-A1 VERL trainer entrypoint: {main_ppo}")


def _inspect_split(
    config: ProductionPreflightConfig,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    report: dict[str, Any] = {"status": "skipped", "num_records": 0, "split_counts": {}}
    if not config.split_manifest.exists():
        return report
    try:
        split_data = json.loads(config.split_manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"Failed to read split_manifest: {type(exc).__name__}: {exc}")
        report["status"] = "failed"
        return report

    forbidden_paths = _find_forbidden_keys(split_data)
    if forbidden_paths:
        errors.append(
            "split_manifest contains forbidden benchmark target fields: "
            + ", ".join(forbidden_paths[: config.max_reported_issues])
        )

    records = _extract_split_records(split_data)
    counts: dict[str, int] = {}
    missing_graphs: list[str] = []
    for item in records:
        split = str(item.get("split", ""))
        record_id = str(item.get("record_id", ""))
        counts[split] = counts.get(split, 0) + 1
        graph_path = item.get("oracle_graph_path")
        candidate = Path(str(graph_path)) if graph_path else config.oracle_graph_dir / f"{record_id}.graph.json"
        if record_id and not candidate.exists():
            missing_graphs.append(record_id)

    if not records:
        errors.append("split_manifest contains no records.")
    if missing_graphs:
        errors.append(
            f"{len(missing_graphs)} split records are missing oracle graphs. "
            f"Examples: {missing_graphs[: config.max_reported_issues]}"
        )
    if counts.get("train", 0) < 1:
        errors.append("split_manifest has no train records.")
    if config.run_evaluation and config.evaluation_split == "test" and counts.get("test", 0) < 1:
        errors.append("RUN_EVALUATION=1 with EVALUATION_SPLIT=test, but split_manifest has no test records.")
    elif config.run_evaluation and config.evaluation_split == "dev" and counts.get("dev", 0) < 1:
        errors.append("RUN_EVALUATION=1 with EVALUATION_SPLIT=dev, but split_manifest has no dev records.")
    if config.train_batch_size is not None:
        train_count = counts.get("train", 0)
        if 0 < train_count < config.train_batch_size:
            warnings.append(
                f"train_batch_size={config.train_batch_size} is larger than train records={train_count}; "
                "native training script will cap it to avoid an empty dataloader."
            )

    report.update(
        {
            "status": "failed" if missing_graphs or not records else "succeeded",
            "num_records": len(records),
            "split_counts": counts,
            "missing_graph_examples": missing_graphs[: config.max_reported_issues],
        }
    )
    return report


def _check_imports(
    config: ProductionPreflightConfig,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    required_modules = ["torch", "transformers", "pandas", "pyarrow", "openai", "ray", "vllm"]
    modules: dict[str, dict[str, Any]] = {}
    for module in required_modules:
        spec = importlib.util.find_spec(module)
        modules[module] = {"available": spec is not None}
        if spec is None:
            errors.append(f"Required Python package is not importable: {module}")

    verl_status = {"available": False}
    if config.require_verl_import:
        sys.path.insert(0, str(config.code_a1_root))
        sys.path.insert(0, str(config.code_a1_root / "verl"))
        try:
            importlib.import_module("verl")
            verl_status["available"] = True
        except Exception as exc:
            errors.append(f"Code-A1/VERL import failed: {type(exc).__name__}: {exc}")
            verl_status["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            for inserted in [str(config.code_a1_root / "verl"), str(config.code_a1_root)]:
                try:
                    sys.path.remove(inserted)
                except ValueError:
                    pass
    else:
        warnings.append("VERL import check skipped; filesystem check still ran.")
        verl_status["status"] = "skipped"

    return {"modules": modules, "verl": verl_status}


def _check_cuda(
    config: ProductionPreflightConfig,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        errors.append(f"CUDA check requires torch import, but torch failed: {type(exc).__name__}: {exc}")
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if available else 0
    required_gpus = max(config.memory_num_gpus, config.question_num_gpus)
    if not available:
        errors.append("REQUIRE_CUDA=1 but torch.cuda.is_available() is False.")
    elif device_count < required_gpus:
        errors.append(f"Need at least {required_gpus} CUDA devices, but torch reports {device_count}.")
    names = []
    if available:
        for index in range(device_count):
            try:
                names.append(torch.cuda.get_device_name(index))
            except Exception:
                names.append(f"cuda:{index}")
    if available and device_count > required_gpus:
        warnings.append(
            f"CUDA has {device_count} devices; this run will request up to {required_gpus}. "
            "Set MEMORY_NUM_GPUS/QUESTION_NUM_GPUS deliberately."
        )
    return {
        "status": "succeeded" if available and device_count >= required_gpus else "failed",
        "available": available,
        "device_count": device_count,
        "required_gpus": required_gpus,
        "device_names": names,
    }


def _check_api_keys(
    config: ProductionPreflightConfig,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    keys = {
        "reward_judge": bool(config.reward_judge_api_key),
        "memory": bool(config.memory_api_key),
        "answer": bool(config.answer_api_key),
        "judge": bool(config.judge_api_key),
    }
    if config.reward_judge_enabled and not config.reward_judge_api_key:
        errors.append("GAAM_REWARD_JUDGE_ENABLED=1 but GAAM_REWARD_JUDGE_API_KEY is empty.")
    if config.run_evaluation and config.memory_backend == "stateful_api" and not config.memory_api_key:
        errors.append("MEMORY_BACKEND=stateful_api requires GAAM_MEMORY_BUILDER_API_KEY.")
    if config.run_evaluation and config.answer_backend == "api" and not config.answer_api_key:
        errors.append("ANSWER_BACKEND=api requires ANSWER_API_KEY or GAAM_ANSWERER_API_KEY.")
    if config.run_evaluation and config.judge_backend == "api" and not config.judge_api_key:
        errors.append("JUDGE_BACKEND=api requires JUDGE_API_KEY or GAAM_EVAL_JUDGE_API_KEY.")
    if config.run_evaluation and config.answer_backend == "no_llm":
        warnings.append("ANSWER_BACKEND=no_llm is for smoke tests, not final production evaluation.")
    if config.run_evaluation and config.judge_backend == "heuristic":
        warnings.append("JUDGE_BACKEND=heuristic is for smoke tests, not final production evaluation.")
    return {"api_key_configured": keys}


def _extract_split_records(split_data: Any) -> list[dict[str, Any]]:
    if isinstance(split_data, dict) and isinstance(split_data.get("records"), list):
        return [item for item in split_data["records"] if isinstance(item, dict)]
    if isinstance(split_data, list):
        return [item for item in split_data if isinstance(item, dict)]
    return []


def _find_forbidden_keys(obj: Any, path: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_path = f"{path}.{key}" if path else key
            if key in FORBIDDEN_SPLIT_KEYS:
                hits.append(child_path)
            hits.extend(_find_forbidden_keys(value, child_path))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            hits.extend(_find_forbidden_keys(value, f"{path}[{index}]"))
    return hits


def truthy(value: str | bool | int | None) -> bool:
    """Parse common shell truthy values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    return str(value or "") in {"1", "true", "True", "yes", "YES"}
