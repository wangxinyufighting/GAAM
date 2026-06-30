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
        "DEEPSEEK_API_KEY",
        "SPLIT_MANIFEST",
        "TRAINING_OUTPUT_DIR",
        "EVAL_OUTPUT_DIR",
        "GAAM_REWARD_JUDGE_BASE_URL",
        "GAAM_REWARD_JUDGE_API_KEY",
        "ANSWER_API_KEY",
        "JUDGE_API_KEY",
        "EVALUATION_SPLIT",
        "RUN_PREFLIGHT",
        "DIAGNOSTICS_OUTPUT",
        "INSTALL_DEPS_BEFORE_TRAINING",
        "INSTALL_FLASH_ATTN",
        "RUN_DIAGNOSTICS_BEFORE_TRAINING",
        "ALLOW_FAILED_DIAGNOSTICS",
        "RUN_TRAINING",
        "RUN_FINAL_AUDIT",
        "EXPORT_ARTIFACT_BUNDLE",
        "ARTIFACT_BUNDLE_DIR",
        "VERIFY_ARTIFACT_BUNDLE",
        "VERIFY_PRODUCTION_COMPLETION",
        "RUN_ID",
        "NNODES",
        "MEMORY_ROLLOUT_N",
        "QUESTION_ROLLOUT_N",
        "MEMORY_TRAIN_BATCH_SIZE",
        "QUESTION_TRAIN_BATCH_SIZE",
        "MEMORY_MAX_PROMPT_LENGTH",
        "QUESTION_MAX_PROMPT_LENGTH",
        "MEMORY_MAX_RESPONSE_LENGTH",
        "QUESTION_MAX_RESPONSE_LENGTH",
        "MEMORY_SESSION_CHUNK_SIZE",
        "MAX_MEMORY_CHUNK_CHARS",
        "ALLOW_STATIC_INCREMENTAL_SCAFFOLD",
        "DUAL_DRY_RUN",
        "FLASH_ATTN_NO_BUILD_ISOLATION",
        "MEMORY_MODEL",
        "MEMORY_BASE_URL",
        "MEMORY_API_KEY",
        "GAAM_MEMORY_BUILDER_MODEL",
        "GAAM_MEMORY_BUILDER_API_KEY",
        "GAAM_ANSWERER_MODEL",
        "GAAM_ANSWERER_API_KEY",
        "GAAM_EVAL_JUDGE_MODEL",
        "GAAM_EVAL_JUDGE_API_KEY",
        "REQUIRE_RESOLVED_CHECKPOINTS",
    ]
    for key in required_keys:
        assert f"{key}=" in text


def test_production_env_example_does_not_contain_real_api_key():
    text = _read_env_template()
    forbidden_fragments = [
        "sk-",
        "Bearer ",
        "api_key=",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in text
