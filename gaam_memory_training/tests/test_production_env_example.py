"""Tests for the production environment template."""

from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def _read_env_template() -> str:
    return (ROOT_DIR / "production.env.example").read_text(encoding="utf-8")


def test_production_env_example_covers_e2e_required_variables():
    text = _read_env_template()
    required_keys = [
        "INPUT_PATH",
        "ORACLE_GRAPH_DIR",
        "CODE_A1_ROOT",
        "MEMORY_MODEL_PATH",
        "QUESTION_MODEL_PATH",
        "SPLIT_MANIFEST",
        "TRAINING_OUTPUT_DIR",
        "EVAL_OUTPUT_DIR",
        "GAAM_REWARD_JUDGE_BASE_URL",
        "GAAM_REWARD_JUDGE_API_KEY",
        "ANSWER_API_KEY",
        "JUDGE_API_KEY",
        "EVALUATION_SPLIT",
        "RUN_FINAL_AUDIT",
        "EXPORT_ARTIFACT_BUNDLE",
        "ARTIFACT_BUNDLE_DIR",
        "RUN_ID",
    ]
    for key in required_keys:
        assert f"{key}=" in text


def test_production_env_example_does_not_contain_real_api_key():
    text = _read_env_template()
    forbidden_fragments = [
        "sk-",
        "Bearer ",
        "api_key=",
        "DEEPSEEK_API_KEY=",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in text
