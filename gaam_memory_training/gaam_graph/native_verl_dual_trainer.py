"""
GAAM dual-agent native VERL co-training orchestration.

This trainer intentionally uses Code-A1/VERL's native ``main_ppo`` as the
per-actor GRPO training kernel. It adds the GAAM-specific dual-agent loop around
that kernel: round scheduling, actor ordering, checkpoint handoff, logging, and
run manifests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


ACTOR_MEMORY_BUILDER = "memory_builder"
ACTOR_QUESTION_AGENT = "question_agent"
VALID_ACTORS = {ACTOR_MEMORY_BUILDER, ACTOR_QUESTION_AGENT}


@dataclass(frozen=True)
class NativeVerlActorConfig:
    actor_role: str
    initial_model_path: Path
    num_gpus: int
    rollout_n: int
    train_batch_size: int
    ppo_mini_batch_size: int
    ppo_micro_batch_size_per_gpu: int
    max_prompt_length: int
    max_response_length: int
    lr: str
    rollout_tp_size: int
    gpu_memory_utilization: str

    def validate(self) -> None:
        if self.actor_role not in VALID_ACTORS:
            raise ValueError(f"Unsupported actor_role: {self.actor_role}")
        if not self.initial_model_path.exists():
            raise FileNotFoundError(f"Model path not found for {self.actor_role}: {self.initial_model_path}")
        if self.num_gpus < 1:
            raise ValueError(f"num_gpus must be >= 1 for {self.actor_role}")


@dataclass(frozen=True)
class GAAMDualRayTrainerConfig:
    root_dir: Path
    code_a1_root: Path
    python_bin: str
    input_path: Path
    oracle_graph_dir: Path
    split_manifest: Path | None
    output_dir: Path
    rounds: int
    order: tuple[str, ...]
    memory_builder: NativeVerlActorConfig
    question_agent: NativeVerlActorConfig
    nnodes: int = 1
    total_epochs_per_actor_step: int = 1
    total_training_steps_per_actor_step: int | None = None
    save_freq: int = 1
    test_freq: int = 1
    max_history_chars: int = 16000
    max_oracle_chars: int = 12000
    questions_per_case: int = 8
    memory_input_mode: str = "full"
    memory_session_chunk_size: int = 4
    max_memory_chunk_chars: int | None = None
    max_previous_memory_chars: int = 6000
    allow_static_incremental_scaffold: bool = False
    max_records_per_split: int | None = None
    logger: str = "console"
    save_hf_model: bool = True
    require_hf_checkpoint: bool = True
    dry_run: bool = False

    def validate(self) -> None:
        if self.rounds < 1:
            raise ValueError("rounds must be >= 1")
        if not self.root_dir.exists():
            raise FileNotFoundError(f"root_dir not found: {self.root_dir}")
        if not (self.code_a1_root / "verl" / "verl").exists():
            raise FileNotFoundError(f"vendored VERL not found under: {self.code_a1_root}")
        if not self.input_path.exists():
            raise FileNotFoundError(f"input_path not found: {self.input_path}")
        if not self.oracle_graph_dir.exists():
            raise FileNotFoundError(f"oracle_graph_dir not found: {self.oracle_graph_dir}")
        if self.split_manifest and not self.split_manifest.exists():
            raise FileNotFoundError(f"split_manifest not found: {self.split_manifest}")
        for actor in self.order:
            if actor not in VALID_ACTORS:
                raise ValueError(f"Unsupported actor in order: {actor}")
        self.memory_builder.validate()
        self.question_agent.validate()


class GAAMDualRayTrainer:
    """
    Round-based dual-agent trainer for GAAM.

    Each actor step launches native VERL GRPO. This is not the old local HF
    trainer and not a dataproto handoff: the child process is
    ``python -m verl.trainer.main_ppo algorithm.adv_estimator=grpo``.
    """

    def __init__(self, config: GAAMDualRayTrainerConfig) -> None:
        self.config = config
        self.config.validate()
        self.output_dir = config.output_dir
        self.logs_dir = self.output_dir / "logs"
        self.manifest_path = self.output_dir / "dual_cotraining_manifest.json"
        self.current_model_paths = {
            ACTOR_MEMORY_BUILDER: config.memory_builder.initial_model_path,
            ACTOR_QUESTION_AGENT: config.question_agent.initial_model_path,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def fit(self) -> dict[str, Any]:
        started_at = _utc_now()
        round_records: list[dict[str, Any]] = []
        status = "succeeded"

        for round_index in range(self.config.rounds):
            print(f"\n{'=' * 72}")
            print(f"GAAMDualRayTrainer round {round_index + 1}/{self.config.rounds}")
            print(f"{'=' * 72}")

            actor_records = []
            for actor_role in self.config.order:
                actor_record = self._run_actor_step(round_index, actor_role)
                actor_records.append(actor_record)
                if actor_record["status"] != "succeeded":
                    status = "failed"
                    if not self.config.dry_run:
                        break
                self._maybe_update_current_model_path(actor_role, actor_record)

            round_status = "succeeded" if all(r["status"] == "succeeded" for r in actor_records) else "failed"
            round_records.append(
                {
                    "round_index": round_index,
                    "status": round_status,
                    "actor_records": actor_records,
                }
            )
            if round_status != "succeeded" and not self.config.dry_run:
                break

        manifest = {
            "manifest_version": "gaam_native_verl_dual_cotraining_v1",
            "trainer": "GAAMDualRayTrainer",
            "status": status,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "config": self._serializable_config(),
            "round_records": round_records,
            "final_model_paths": {k: str(v) for k, v in self.current_model_paths.items()},
        }
        self._write_json(self.manifest_path, manifest)
        print(f"\nDual co-training manifest: {self.manifest_path}")
        print(f"Dual co-training status: {status}")
        return manifest

    def _run_actor_step(self, round_index: int, actor_role: str) -> dict[str, Any]:
        actor_config = self._actor_config(actor_role)
        actor_output_dir = self.output_dir / "rounds" / f"round_{round_index:03d}" / actor_role
        dataset_dir = actor_output_dir / "dataset"
        checkpoint_dir = actor_output_dir / "checkpoints"
        log_path = self.logs_dir / f"round_{round_index:03d}.{actor_role}.log"
        model_path = self.current_model_paths[actor_role]

        command, env = self._build_actor_command(
            actor_config=actor_config,
            actor_role=actor_role,
            model_path=model_path,
            dataset_dir=dataset_dir,
            checkpoint_dir=checkpoint_dir,
        )

        print(f"\n[Dual round {round_index}] {actor_role}: launching native VERL GRPO")
        print(f"[Dual round {round_index}] {actor_role}: model_path={model_path}")
        print(f"[Dual round {round_index}] {actor_role}: checkpoint_dir={checkpoint_dir}")
        print(f"[Dual round {round_index}] {actor_role}: log_path={log_path}")

        record = {
            "round_index": round_index,
            "actor_role": actor_role,
            "status": "running",
            "dry_run": self.config.dry_run,
            "model_path": str(model_path),
            "dataset_dir": str(dataset_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "log_path": str(log_path),
            "command": command,
            "env_overrides": env,
            "started_at": _utc_now(),
            "finished_at": None,
            "returncode": None,
            "latest_hf_model_path": None,
            "warnings": [],
            "errors": [],
        }

        if self.config.dry_run:
            record["status"] = "succeeded"
            record["finished_at"] = _utc_now()
            self._write_json(actor_output_dir / "actor_step_manifest.json", record)
            return record

        actor_output_dir.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        returncode = self._run_streaming_subprocess(command, env, log_path)
        record["returncode"] = returncode
        record["finished_at"] = _utc_now()
        if returncode == 0:
            latest_hf_model = find_latest_hf_model_path(checkpoint_dir)
            if latest_hf_model:
                record["status"] = "succeeded"
                record["latest_hf_model_path"] = str(latest_hf_model)
            elif self.config.require_hf_checkpoint:
                record["status"] = "failed"
                record["errors"].append(
                    "Native VERL process exited successfully but no Hugging Face checkpoint directory "
                    f"was found under {checkpoint_dir}. This actor step is treated as failed because "
                    "GAAM full-model co-training requires materialized updated weights."
                )
            else:
                record["status"] = "succeeded"
                record["warnings"].append(
                    "Native VERL step succeeded, but no Hugging Face checkpoint directory was found. "
                    "The next round will reuse the previous model path."
                )
        else:
            record["status"] = "failed"
            record["errors"].append(f"Native VERL process failed with returncode {returncode}")

        self._write_json(actor_output_dir / "actor_step_manifest.json", record)
        return record

    def _build_actor_command(
        self,
        *,
        actor_config: NativeVerlActorConfig,
        actor_role: str,
        model_path: Path,
        dataset_dir: Path,
        checkpoint_dir: Path,
    ) -> tuple[list[str], dict[str, str]]:
        env = {
            "PYTHON_BIN": self.config.python_bin,
            "INPUT_PATH": str(self.config.input_path),
            "ORACLE_GRAPH_DIR": str(self.config.oracle_graph_dir),
            "ACTOR_ROLE": actor_role,
            "MODEL_PATH": str(model_path),
            "DATASET_DIR": str(dataset_dir),
            "CHECKPOINT_DIR": str(checkpoint_dir),
            "CODE_A1_ROOT": str(self.config.code_a1_root),
            "NUM_GPUS": str(actor_config.num_gpus),
            "NNODES": str(self.config.nnodes),
            "ROLLOUT_N": str(actor_config.rollout_n),
            "TRAIN_BATCH_SIZE": str(actor_config.train_batch_size),
            "PPO_MINI_BATCH_SIZE": str(actor_config.ppo_mini_batch_size),
            "PPO_MICRO_BATCH_SIZE_PER_GPU": str(actor_config.ppo_micro_batch_size_per_gpu),
            "MAX_PROMPT_LENGTH": str(actor_config.max_prompt_length),
            "MAX_RESPONSE_LENGTH": str(actor_config.max_response_length),
            "MAX_HISTORY_CHARS": str(self.config.max_history_chars),
            "MAX_ORACLE_CHARS": str(self.config.max_oracle_chars),
            "QUESTIONS_PER_CASE": str(self.config.questions_per_case),
            "MEMORY_INPUT_MODE": self.config.memory_input_mode,
            "MEMORY_SESSION_CHUNK_SIZE": str(self.config.memory_session_chunk_size),
            "MAX_PREVIOUS_MEMORY_CHARS": str(self.config.max_previous_memory_chars),
            "ALLOW_STATIC_INCREMENTAL_SCAFFOLD": "1" if self.config.allow_static_incremental_scaffold else "0",
            "TOTAL_EPOCHS": str(self.config.total_epochs_per_actor_step),
            "SAVE_FREQ": str(self.config.save_freq),
            "TEST_FREQ": str(self.config.test_freq),
            "LR": actor_config.lr,
            "ROLLOUT_TP_SIZE": str(actor_config.rollout_tp_size),
            "GPU_MEMORY_UTILIZATION": actor_config.gpu_memory_utilization,
            "LOGGER": self.config.logger,
            "EXPERIMENT_NAME": f"dual_{actor_role}",
            "SAVE_HF_MODEL": "True" if self.config.save_hf_model else "False",
            "REQUIRE_HF_CHECKPOINT": "1" if self.config.require_hf_checkpoint else "0",
        }
        if self.config.split_manifest:
            env["SPLIT_MANIFEST"] = str(self.config.split_manifest)
        if self.config.max_records_per_split is not None:
            env["MAX_RECORDS_PER_SPLIT"] = str(self.config.max_records_per_split)
        if self.config.max_memory_chunk_chars is not None:
            env["MAX_MEMORY_CHUNK_CHARS"] = str(self.config.max_memory_chunk_chars)
        if self.config.total_training_steps_per_actor_step is not None:
            env["TOTAL_TRAINING_STEPS"] = str(self.config.total_training_steps_per_actor_step)

        command = ["bash", "scripts/run_native_verl_grpo_training.sh"]
        return command, env

    def _run_streaming_subprocess(
        self,
        command: list[str],
        env_overrides: dict[str, str],
        log_path: Path,
    ) -> int:
        env = {**dict(**subprocess_os_environ()), **env_overrides}
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=self.config.root_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                log_file.write(line)
            return process.wait()

    def _maybe_update_current_model_path(self, actor_role: str, actor_record: dict[str, Any]) -> None:
        latest_hf_model_path = actor_record.get("latest_hf_model_path")
        if latest_hf_model_path:
            self.current_model_paths[actor_role] = Path(latest_hf_model_path)

    def _actor_config(self, actor_role: str) -> NativeVerlActorConfig:
        if actor_role == ACTOR_MEMORY_BUILDER:
            return self.config.memory_builder
        if actor_role == ACTOR_QUESTION_AGENT:
            return self.config.question_agent
        raise ValueError(f"Unsupported actor_role: {actor_role}")

    def _serializable_config(self) -> dict[str, Any]:
        config_dict = asdict(self.config)
        return _stringify_paths(config_dict)

    @staticmethod
    def _write_json(path: Path, obj: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def find_latest_hf_model_path(checkpoint_dir: Path) -> Path | None:
    """Find the newest HF model materialized by native VERL checkpoint saving."""
    candidates = [
        path
        for name in ("huggingface", "hf_model")
        for path in checkpoint_dir.rglob(name)
        if path.is_dir()
    ]
    if not candidates:
        return None

    def sort_key(path: Path) -> tuple[float, str]:
        global_step = _extract_global_step(path)
        return (global_step, str(path))

    return sorted(candidates, key=sort_key)[-1]


def _extract_global_step(path: Path) -> float:
    for parent in [path, *path.parents]:
        name = parent.name
        if name.startswith("global_step_"):
            suffix = name.removeprefix("global_step_")
            try:
                return float(int(suffix))
            except ValueError:
                return -1.0
    try:
        return path.stat().st_mtime
    except OSError:
        return -1.0


def _stringify_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _stringify_paths(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_stringify_paths(child) for child in value]
    if isinstance(value, tuple):
        return [_stringify_paths(child) for child in value]
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def subprocess_os_environ() -> dict[str, str]:
    # Kept as a helper to make tests able to monkeypatch process environment.
    import os

    return os.environ.copy()


def require_executable(name_or_path: str) -> None:
    if Path(name_or_path).exists():
        return
    if shutil.which(name_or_path):
        return
    raise FileNotFoundError(f"Executable not found: {name_or_path}")
