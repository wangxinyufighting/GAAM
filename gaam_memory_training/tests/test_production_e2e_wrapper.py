"""Shell-level tests for the production E2E wrapper."""

from __future__ import annotations

from pathlib import Path
import os
import subprocess


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_production_e2e_wrapper_splits_trains_and_evaluates(tmp_path: Path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    memory_model = tmp_path / "memory_model"
    question_model = tmp_path / "question_model"
    memory_model.mkdir()
    question_model.mkdir()
    fake_code_a1 = tmp_path / "Code-A1"
    (fake_code_a1 / "verl" / "verl").mkdir(parents=True)
    split_manifest = tmp_path / "split.json"
    training_output = tmp_path / "training_output"
    eval_output = tmp_path / "eval_output"

    fake_python = fake_bin / "python"
    fake_python.write_text(
        f"""#!/bin/bash
if [[ "$1" == "scripts/create_split_from_existing_graphs.py" ]]; then
  touch {marker_dir / "split_created"}
  while [[ "$#" -gt 0 ]]; do
    if [[ "$1" == "--output" ]]; then
      shift
      echo '{{"records":[]}}' > "$1"
    fi
    shift || true
  done
  exit 0
fi
if [[ "$1" == "scripts/check_production_e2e_preflight.py" ]]; then
  echo "preflight args=$*"
  touch {marker_dir / "preflighted"}
  exit 0
fi
if [[ "$1" == "scripts/export_production_artifact_bundle.py" ]]; then
  echo "bundle args=$*"
  touch {marker_dir / "bundled"}
  exit 0
fi
if [[ "$1" == "scripts/verify_production_artifact_bundle.py" ]]; then
  echo "bundle verify args=$*"
  touch {marker_dir / "bundle_verified"}
  exit 0
fi
if [[ "$1" == "scripts/verify_production_e2e_completion.py" ]]; then
  echo "completion verify args=$*"
  touch {marker_dir / "completion_verified"}
  exit 0
fi
if [[ "$1" == "scripts/audit_production_run.py" ]]; then
  echo "audit args=$*"
  touch {marker_dir / "audited"}
  exit 0
fi
exec /usr/bin/env python "$@"
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    fake_bash = fake_bin / "bash"
    fake_bash.write_text(
        f"""#!/bin/bash
if [[ "$1" == "scripts/run_production_gaam_training.sh" ]]; then
  echo "training SPLIT_MANIFEST=$SPLIT_MANIFEST"
  echo "training OUTPUT_DIR=$OUTPUT_DIR"
  echo "training QUESTIONS_PER_CASE=$QUESTIONS_PER_CASE"
  touch {marker_dir / "trained"}
  exit 0
fi
if [[ "$1" == "scripts/run_production_post_training_evaluation.sh" ]]; then
  echo "evaluation TRAINING_OUTPUT_DIR=$TRAINING_OUTPUT_DIR"
  echo "evaluation EVAL_OUTPUT_DIR=$EVAL_OUTPUT_DIR"
  echo "evaluation SPLIT_MANIFEST=$SPLIT_MANIFEST"
  echo "evaluation EVALUATION_SPLIT=$EVALUATION_SPLIT"
  echo "evaluation ANSWER_BACKEND=$ANSWER_BACKEND"
  echo "evaluation REQUIRE_RESOLVED_CHECKPOINTS=$REQUIRE_RESOLVED_CHECKPOINTS"
  touch {marker_dir / "evaluated"}
  exit 0
fi
exec /bin/bash "$@"
""",
        encoding="utf-8",
    )
    fake_bash.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "PYTHON_BIN": "python",
        "MEMORY_MODEL_PATH": str(memory_model),
        "QUESTION_MODEL_PATH": str(question_model),
        "CODE_A1_ROOT": str(fake_code_a1),
        "SPLIT_MANIFEST": str(split_manifest),
        "TRAINING_OUTPUT_DIR": str(training_output),
        "EVAL_OUTPUT_DIR": str(eval_output),
        "QUESTIONS_PER_CASE": "5",
        "MEMORY_API_KEY": "memory-key",
        "ANSWER_BACKEND": "no_llm",
        "JUDGE_BACKEND": "heuristic",
        "VERIFY_TRAINING_OUTPUT": "0",
        "VERIFY_EVALUATION_OUTPUT": "0",
        "REQUIRE_RESOLVED_CHECKPOINTS": "0",
        "REQUIRE_CUDA": "0",
        "CHECK_IMPORTS": "0",
        "REQUIRE_VERL_IMPORT": "0",
    }

    result = subprocess.run(
        ["bash", "scripts/run_production_e2e_training_and_evaluation.sh"],
        cwd=ROOT_DIR,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (marker_dir / "split_created").exists()
    assert (marker_dir / "preflighted").exists()
    assert (marker_dir / "trained").exists()
    assert (marker_dir / "evaluated").exists()
    assert (marker_dir / "audited").exists()
    assert (marker_dir / "bundled").exists()
    assert (marker_dir / "bundle_verified").exists()
    assert (marker_dir / "completion_verified").exists()
    assert f"training SPLIT_MANIFEST={split_manifest}" in result.stdout
    assert f"training OUTPUT_DIR={training_output}" in result.stdout
    assert "training QUESTIONS_PER_CASE=5" in result.stdout
    assert f"evaluation TRAINING_OUTPUT_DIR={training_output}" in result.stdout
    assert f"evaluation EVAL_OUTPUT_DIR={eval_output}" in result.stdout
    assert f"evaluation SPLIT_MANIFEST={split_manifest}" in result.stdout
    assert "evaluation EVALUATION_SPLIT=test" in result.stdout
    assert "evaluation REQUIRE_RESOLVED_CHECKPOINTS=0" in result.stdout
    assert f"--training_output_dir {training_output}" in result.stdout
    assert f"--evaluation_output_dir {eval_output}" in result.stdout
    assert f"--split_manifest {split_manifest}" in result.stdout
    assert "--expected_evaluation_split test" in result.stdout
    assert f"--memory_model_path {memory_model}" in result.stdout
    assert "--memory_api_key memory-key" in result.stdout
    assert "--require_cuda 0" in result.stdout
    assert "--check_imports 0" in result.stdout
    assert "--bundle_output_dir outputs/production_artifact_bundle" in result.stdout
    assert "completion verify args=" in result.stdout


def test_production_e2e_wrapper_can_skip_auto_split_and_evaluation(tmp_path: Path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    memory_model = tmp_path / "memory_model"
    question_model = tmp_path / "question_model"
    memory_model.mkdir()
    question_model.mkdir()
    fake_code_a1 = tmp_path / "Code-A1"
    (fake_code_a1 / "verl" / "verl").mkdir(parents=True)
    split_manifest = tmp_path / "existing_split.json"
    split_manifest.write_text('{"records":[]}', encoding="utf-8")

    fake_python = fake_bin / "python"
    fake_python.write_text(
        f"""#!/bin/bash
if [[ "$1" == "scripts/create_split_from_existing_graphs.py" ]]; then
  touch {marker_dir / "split_should_not_run"}
  exit 9
fi
if [[ "$1" == "scripts/check_production_e2e_preflight.py" ]]; then
  touch {marker_dir / "preflighted"}
  exit 0
fi
if [[ "$1" == "scripts/export_production_artifact_bundle.py" ]]; then
  touch {marker_dir / "bundled"}
  exit 0
fi
if [[ "$1" == "scripts/verify_production_artifact_bundle.py" ]]; then
  echo "bundle verify args=$*"
  touch {marker_dir / "bundle_verified"}
  exit 0
fi
if [[ "$1" == "scripts/verify_production_e2e_completion.py" ]]; then
  touch {marker_dir / "completion_should_not_run"}
  exit 9
fi
if [[ "$1" == "scripts/audit_production_run.py" ]]; then
  touch {marker_dir / "audited"}
  exit 0
fi
exec /usr/bin/env python "$@"
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    fake_bash = fake_bin / "bash"
    fake_bash.write_text(
        f"""#!/bin/bash
if [[ "$1" == "scripts/run_production_gaam_training.sh" ]]; then
  touch {marker_dir / "trained"}
  exit 0
fi
if [[ "$1" == "scripts/run_production_post_training_evaluation.sh" ]]; then
  touch {marker_dir / "eval_should_not_run"}
  exit 9
fi
exec /bin/bash "$@"
""",
        encoding="utf-8",
    )
    fake_bash.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "PYTHON_BIN": "python",
        "MEMORY_MODEL_PATH": str(memory_model),
        "QUESTION_MODEL_PATH": str(question_model),
        "CODE_A1_ROOT": str(fake_code_a1),
        "SPLIT_MANIFEST": str(split_manifest),
        "AUTO_CREATE_SPLIT": "0",
        "RUN_EVALUATION": "0",
        "RUN_FINAL_AUDIT": "0",
        "VERIFY_PRODUCTION_COMPLETION": "0",
        "VERIFY_TRAINING_OUTPUT": "0",
        "REQUIRE_CUDA": "0",
        "CHECK_IMPORTS": "0",
        "REQUIRE_VERL_IMPORT": "0",
    }

    result = subprocess.run(
        ["bash", "scripts/run_production_e2e_training_and_evaluation.sh"],
        cwd=ROOT_DIR,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (marker_dir / "trained").exists()
    assert (marker_dir / "preflighted").exists()
    assert (marker_dir / "bundled").exists()
    assert (marker_dir / "bundle_verified").exists()
    assert not (marker_dir / "split_should_not_run").exists()
    assert not (marker_dir / "eval_should_not_run").exists()
    assert not (marker_dir / "audited").exists()
    assert not (marker_dir / "completion_should_not_run").exists()
