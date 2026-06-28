#!/usr/bin/env python3
"""Resolve final checkpoints from a GAAM native dual VERL training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gaam_graph.native_verl_training_verifier import resolve_native_verl_checkpoint_paths  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolve Memory Builder and Question Agent checkpoints from native dual VERL outputs."
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--format", choices=["json", "shell"], default="json")
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Do not fail if checkpoint paths are absent or do not exist.",
    )
    args = parser.parse_args()

    resolved = resolve_native_verl_checkpoint_paths(
        args.output_dir,
        require_exists=not args.allow_missing,
    )

    if args.format == "json":
        print(json.dumps(resolved, ensure_ascii=False, indent=2))
    else:
        print(_to_shell_exports(resolved))
    return 0


def _to_shell_exports(resolved: dict) -> str:
    memory = resolved.get("memory_builder_checkpoint") or ""
    question = resolved.get("question_agent_checkpoint") or ""
    lines = [
        f"MEMORY_BUILDER_CHECKPOINT={shlex.quote(memory)}",
        f"QUESTION_AGENT_CHECKPOINT={shlex.quote(question)}",
        f"MEMORY_MODEL={shlex.quote(memory)}",
        f"MEMORY_MODEL_PATH={shlex.quote(memory)}",
        f"QUESTION_MODEL_PATH={shlex.quote(question)}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
