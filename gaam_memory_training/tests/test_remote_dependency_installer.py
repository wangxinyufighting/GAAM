"""Shell-level tests for remote dependency installer."""

from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_remote_dependency_installer_runs_selected_pip_steps(tmp_path: Path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    code_a1 = tmp_path / "Code-A1"
    verl_root = code_a1 / "verl"
    verl_root.mkdir(parents=True)
    (verl_root / "requirements.txt").write_text(
        "numpy\nflash-attn==2.8.3\nray[default]\n",
        encoding="utf-8",
    )
    pip_log = tmp_path / "pip.log"

    fake_python = fake_bin / "python"
    fake_python.write_text(
        f"""#!/bin/bash
if [[ "$1" == "-" ]]; then
  shift
  exec {sys.executable} - "$@"
fi
if [[ "$1" == "-m" && "$2" == "pip" ]]; then
  echo "$*" >> {pip_log}
  exit 0
fi
exec {sys.executable} "$@"
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "PYTHON_BIN": "python",
        "CODE_A1_ROOT": str(code_a1),
        "INSTALL_GAAM_DEPS": "1",
        "INSTALL_LOCAL_TRAINING_DEPS": "1",
        "INSTALL_VERL_DEPS": "1",
        "INSTALL_VLLM": "1",
        "INSTALL_FLASH_ATTN": "0",
        "RUN_PIP_CHECK": "1",
    }

    result = subprocess.run(
        ["bash", "scripts/install_remote_production_deps.sh"],
        cwd=ROOT_DIR,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    log = pip_log.read_text(encoding="utf-8")
    assert "-m pip install -r requirements.txt" in log
    assert "-m pip install -r requirements-local-training.txt" in log
    assert "-m pip install vllm==0.8.4" in log
    assert "-m pip check" in log
    assert "flash-attn" not in log


def test_remote_dependency_installer_can_install_flash_attn_when_requested(tmp_path: Path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    pip_log = tmp_path / "pip.log"

    fake_python = fake_bin / "python"
    fake_python.write_text(
        f"""#!/bin/bash
if [[ "$1" == "-" ]]; then
  cat >/dev/null
  echo "torch: fake"
  echo "cuda: fake"
  echo "cuda_available: True"
  echo "cxx11_abi: False"
  exit 0
fi
if [[ "$1" == "-m" && "$2" == "pip" ]]; then
  echo "$*" >> {pip_log}
  exit 0
fi
exec {sys.executable} "$@"
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "PYTHON_BIN": "python",
        "INSTALL_GAAM_DEPS": "0",
        "INSTALL_LOCAL_TRAINING_DEPS": "0",
        "INSTALL_VERL_DEPS": "0",
        "INSTALL_VLLM": "0",
        "INSTALL_FLASH_ATTN": "1",
        "FLASH_ATTN_SPEC": "flash-attn==2.8.3",
        "RUN_PIP_CHECK": "0",
    }

    result = subprocess.run(
        ["bash", "scripts/install_remote_production_deps.sh"],
        cwd=ROOT_DIR,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    log = pip_log.read_text(encoding="utf-8")
    assert "-m pip install flash-attn==2.8.3 --no-build-isolation" in log
