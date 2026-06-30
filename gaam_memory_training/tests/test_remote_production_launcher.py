"""Shell-level tests for the remote production launcher."""

from __future__ import annotations

from pathlib import Path
import os
import subprocess


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_remote_production_launcher_loads_env_runs_diagnostics_then_training(tmp_path: Path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    env_file = tmp_path / "production.env"
    diagnostics_output = tmp_path / "diagnostics.json"
    env_file.write_text(
        "\n".join(
            [
                f"DIAGNOSTICS_OUTPUT={diagnostics_output}",
                "INSTALL_DEPS_BEFORE_TRAINING=1",
                "RUN_DIAGNOSTICS_BEFORE_TRAINING=1",
                "RUN_TRAINING=1",
            ]
        ),
        encoding="utf-8",
    )

    fake_python = fake_bin / "python"
    fake_python.write_text(
        f"""#!/bin/bash
if [[ "$1" == "scripts/collect_remote_training_diagnostics.py" ]]; then
  echo "diagnostics args=$*"
  touch {marker_dir / "diagnosed"}
  while [[ "$#" -gt 0 ]]; do
    if [[ "$1" == "--output" ]]; then
      shift
      echo '{{"status":"succeeded"}}' > "$1"
    fi
    shift || true
  done
  exit 0
fi
if [[ "$1" == "-" ]]; then
  cat >/dev/null
  echo "python: fake"
  echo "executable: fake"
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
if [[ "$1" == "scripts/install_remote_production_deps.sh" ]]; then
  echo "deps installed"
  touch {marker_dir / "deps_installed"}
  exit 0
fi
if [[ "$1" == "scripts/run_production_e2e_training_and_evaluation.sh" ]]; then
  echo "training launched"
  touch {marker_dir / "trained"}
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
        "ENV_FILE": str(env_file),
    }

    result = subprocess.run(
        ["bash", "scripts/run_remote_production_training.sh"],
        cwd=ROOT_DIR,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (marker_dir / "deps_installed").exists()
    assert (marker_dir / "diagnosed").exists()
    assert (marker_dir / "trained").exists()
    assert diagnostics_output.exists()
    assert f"--output {diagnostics_output}" in result.stdout
    assert "training launched" in result.stdout


def test_remote_production_launcher_stops_when_diagnostics_fail(tmp_path: Path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()

    fake_python = fake_bin / "python"
    fake_python.write_text(
        f"""#!/bin/bash
if [[ "$1" == "scripts/collect_remote_training_diagnostics.py" ]]; then
  touch {marker_dir / "diagnosed"}
  exit 7
fi
exec /usr/bin/env python "$@"
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    fake_bash = fake_bin / "bash"
    fake_bash.write_text(
        f"""#!/bin/bash
if [[ "$1" == "scripts/run_production_e2e_training_and_evaluation.sh" ]]; then
  touch {marker_dir / "should_not_train"}
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
        "LOAD_ENV_FILE": "0",
        "RUN_DIAGNOSTICS_BEFORE_TRAINING": "1",
        "ALLOW_FAILED_DIAGNOSTICS": "0",
        "RUN_TRAINING": "1",
    }

    result = subprocess.run(
        ["bash", "scripts/run_remote_production_training.sh"],
        cwd=ROOT_DIR,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )

    assert result.returncode == 7
    assert (marker_dir / "diagnosed").exists()
    assert not (marker_dir / "should_not_train").exists()
    assert "Diagnostics failed" in result.stderr
