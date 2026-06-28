"""Shell-level tests for post-training evaluation wrapper."""

from __future__ import annotations

from pathlib import Path
import subprocess


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_post_training_evaluation_wrapper_verifies_resolves_and_evaluates(tmp_path: Path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    training_output = tmp_path / "training_output"
    checkpoint = tmp_path / "memory_checkpoint"
    split_manifest = tmp_path / "split.json"
    training_output.mkdir()
    checkpoint.mkdir()
    split_manifest.write_text('{"records":[]}', encoding="utf-8")

    fake_python = fake_bin / "python"
    fake_python.write_text(
        f"""#!/bin/bash
if [[ "$1" == "scripts/verify_native_verl_training_output.py" ]]; then
  touch {marker_dir / "verified"}
  exit 0
fi
if [[ "$1" == "scripts/verify_case_evaluation_output.py" ]]; then
  touch {marker_dir / "eval_verified"}
  exit 0
fi
if [[ "$1" == "scripts/resolve_native_verl_checkpoints.py" ]]; then
  echo "MEMORY_MODEL={checkpoint}"
  echo "MEMORY_MODEL_PATH={checkpoint}"
  echo "QUESTION_MODEL_PATH={checkpoint}"
  echo "MEMORY_BUILDER_CHECKPOINT={checkpoint}"
  echo "QUESTION_AGENT_CHECKPOINT={checkpoint}"
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
if [[ "$1" == "scripts/run_production_case_evaluation.sh" ]]; then
  echo "evaluation MEMORY_MODEL=$MEMORY_MODEL"
  echo "evaluation OUTPUT_DIR=$OUTPUT_DIR"
  echo "evaluation SPLIT_MANIFEST=$SPLIT_MANIFEST"
  echo "evaluation EVALUATION_SPLIT=$EVALUATION_SPLIT"
  touch {marker_dir / "evaluated"}
  exit 0
fi
exec /bin/bash "$@"
""",
        encoding="utf-8",
    )
    fake_bash.chmod(0o755)

    env = {
        **dict(__import__("os").environ),
        "PATH": f"{fake_bin}:{__import__('os').environ['PATH']}",
        "PYTHON_BIN": "python",
        "TRAINING_OUTPUT_DIR": str(training_output),
        "SPLIT_MANIFEST": str(split_manifest),
        "EVAL_OUTPUT_DIR": str(tmp_path / "eval_output"),
        "ANSWER_BACKEND": "no_llm",
        "JUDGE_BACKEND": "heuristic",
    }

    result = subprocess.run(
        ["bash", "scripts/run_production_post_training_evaluation.sh"],
        cwd=ROOT_DIR,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (marker_dir / "verified").exists()
    assert (marker_dir / "evaluated").exists()
    assert (marker_dir / "eval_verified").exists()
    assert f"evaluation MEMORY_MODEL={checkpoint}" in result.stdout
    assert f"evaluation SPLIT_MANIFEST={split_manifest}" in result.stdout
    assert "evaluation EVALUATION_SPLIT=test" in result.stdout
