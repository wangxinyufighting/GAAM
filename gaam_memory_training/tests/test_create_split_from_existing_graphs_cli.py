"""Tests for splitting only graph-backed records."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]


def _cli_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["KMP_INIT_AT_FORK"] = "FALSE"
    return env


def _write_records(path: Path, record_ids: list[str]) -> None:
    records = [
        {
            "id": record_id,
            "sessions": [
                {
                    "session_id": f"{record_id}_session",
                    "messages": [{"role": "user", "content": f"memory for {record_id}"}],
                }
            ],
        }
        for record_id in record_ids
    ]
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")


def _write_graph(graph_dir: Path, record_id: str) -> None:
    graph_dir.mkdir(parents=True, exist_ok=True)
    (graph_dir / f"{record_id}.graph.json").write_text(
        json.dumps({"graph": {"record_id": record_id}, "nodes": []}, ensure_ascii=False),
        encoding="utf-8",
    )


def test_create_split_from_existing_graphs_filters_missing_graph_records(tmp_path: Path):
    input_path = tmp_path / "records.json"
    graph_dir = tmp_path / "graphs"
    output_path = tmp_path / "split.json"
    _write_records(
        input_path,
        ["case_a", "case_b", "case_c", "case_d", "case_e", "case_f", "case_g"],
    )
    for record_id in ["case_a", "case_c", "case_e", "case_f", "case_g", "not_in_input"]:
        _write_graph(graph_dir, record_id)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/create_split_from_existing_graphs.py",
            "--input",
            str(input_path),
            "--oracle_graph_dir",
            str(graph_dir),
            "--output",
            str(output_path),
            "--train_ratio",
            "0.6",
            "--dev_ratio",
            "0.2",
            "--test_ratio",
            "0.2",
            "--seed",
            "7",
            "--overwrite",
        ],
        cwd=ROOT_DIR,
        env=_cli_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    split_ids = {item["record_id"] for item in manifest["records"]}
    assert split_ids == {"case_a", "case_c", "case_e", "case_f", "case_g"}
    assert "case_b" not in split_ids
    assert "case_d" not in split_ids
    assert "not_in_input" not in split_ids
    assert all(item["has_benchmark_question"] is False for item in manifest["records"])
    assert all(item["has_benchmark_answer"] is False for item in manifest["records"])


def test_create_split_from_existing_graphs_fails_without_graph_backed_records(tmp_path: Path):
    input_path = tmp_path / "records.json"
    graph_dir = tmp_path / "graphs"
    output_path = tmp_path / "split.json"
    graph_dir.mkdir()
    _write_records(input_path, ["case_a"])

    result = subprocess.run(
        [
            sys.executable,
            "scripts/create_split_from_existing_graphs.py",
            "--input",
            str(input_path),
            "--oracle_graph_dir",
            str(graph_dir),
            "--output",
            str(output_path),
        ],
        cwd=ROOT_DIR,
        env=_cli_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )

    assert result.returncode == 1
    assert "no input records have matching oracle graphs" in result.stderr
    assert not output_path.exists()
