"""Shell-level tests for production training wrapper behavior."""

from __future__ import annotations

from pathlib import Path
import subprocess


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_production_training_wrapper_verifies_after_native_failure(tmp_path: Path):
    fake_code_a1 = tmp_path / "fake_code_a1"
    (fake_code_a1 / "verl" / "verl").mkdir(parents=True)
    memory_model = tmp_path / "memory_model"
    question_model = tmp_path / "question_model"
    memory_model.mkdir()
    question_model.mkdir()

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_bash = fake_bin / "bash"
    fake_bash.write_text(
        """#!/bin/bash
if [[ "$1" == "scripts/run_native_verl_dual_cotraining.sh" ]]; then
  echo "fake native training failure"
  exit 37
fi
exec /bin/bash "$@"
""",
        encoding="utf-8",
    )
    fake_bash.chmod(0o755)

    output_dir = tmp_path / "out"
    env = {
        **dict(__import__("os").environ),
        "PATH": f"{fake_bin}:{__import__('os').environ['PATH']}",
        "MEMORY_MODEL_PATH": str(memory_model),
        "QUESTION_MODEL_PATH": str(question_model),
        "CODE_A1_ROOT": str(fake_code_a1),
        "SPLIT_MANIFEST": "outputs/splits/e47becba.debug.json",
        "OUTPUT_DIR": str(output_dir),
        "VERIFY_AFTER_TRAINING": "1",
    }

    result = subprocess.run(
        ["bash", "scripts/run_production_gaam_training.sh"],
        cwd=ROOT_DIR,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )

    assert result.returncode == 37
    assert "fake native training failure" in result.stdout
    assert "Verifying native VERL training outputs" in result.stdout
    assert "Production training failed with status 37" in result.stderr
    assert (output_dir / "native_verl_training_verification.json").exists()
