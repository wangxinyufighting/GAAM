"""
CLI smoke tests for build_current_memory.py.

Run with: pytest gaam_memory_training/tests/test_build_current_memory_cli.py
"""

import json
from pathlib import Path
import subprocess
import sys

from gaam_graph.memory_schema import validate_current_memory


def test_build_current_memory_cli_smoke_and_no_leakage(tmp_path):
    """Test CLI output is schema-valid and does not store eval targets."""
    repo_root = Path(__file__).resolve().parents[1]
    input_path = tmp_path / "input.json"
    output_dir = tmp_path / "current_memory"

    input_data = [
        {
            "id": "record_cli_001",
            "question": "LEAK_CLI_QUESTION",
            "target": "LEAK_CLI_TARGET",
            "haystack_question_type": "LEAK_CLI_TYPE",
            "sessions": [
                {
                    "session_id": "sess_01",
                    "messages": [
                        {
                            "role": "user",
                            "content": "I keep a detailed project notebook for memory research experiments.",
                            "timestamp": "2024-01-01T10:00:00",
                        }
                    ],
                }
            ],
        }
    ]
    input_path.write_text(json.dumps(input_data), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_current_memory.py",
            "--input",
            str(input_path),
            "--output_dir",
            str(output_dir),
            "--max_records",
            "1",
            "--overwrite",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    output_path = output_dir / "record_cli_001.current_memory.json"
    assert output_path.exists()

    memory = json.loads(output_path.read_text(encoding="utf-8"))
    validate_current_memory(memory, strict=True)

    serialized = json.dumps(memory, ensure_ascii=False)
    assert "LEAK_CLI_QUESTION" not in serialized
    assert "LEAK_CLI_TARGET" not in serialized
    assert "LEAK_CLI_TYPE" not in serialized
