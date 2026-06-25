"""
Phase 4 Milestone 4: Round Scheduler

Handles record selection and round plan construction.

Key functions:
- select_round_records: deterministic record selection from split
- build_round_plan: construct round plan with checkpoint carryover

Design principle:
Deterministic scheduling from seed. Same seed + same split -> same schedule.
"""

import random
from pathlib import Path

from gaam_graph.dataset_split_schema import DatasetSplitManifest, DatasetSplitName
from gaam_graph.phase4_cotraining_schema import (
    Phase4CotrainingConfig,
    Phase4RoundPlan,
    Phase4RoundReport,
)


def select_round_records(
    split_manifest: DatasetSplitManifest,
    split: DatasetSplitName,
    *,
    round_id: int,
    max_records: int | None,
    explicit_record_ids: list[str] | None,
    seed: int,
    schedule_policy: str = "deterministic_window",
) -> list[str]:
    """
    Select records for one round.

    Args:
        split_manifest: dataset split manifest
        split: which split (train/dev/test)
        round_id: round number
        max_records: max records per round (None = all)
        explicit_record_ids: explicit list (overrides scheduling)
        seed: random seed
        schedule_policy: "deterministic_window" or "fixed_subset"

    Returns:
        List of record IDs for this round
    """
    # Get available records for this split
    available = [r.record_id for r in split_manifest.records if r.split == split]

    # Explicit record IDs override scheduling
    if explicit_record_ids is not None:
        # Validate all requested records are in split
        invalid = set(explicit_record_ids) - set(available)
        if invalid:
            raise ValueError(
                f"Explicit record IDs not in {split.value} split: {invalid}"
            )
        return explicit_record_ids

    # If no max_records, return all
    if max_records is None:
        return available
    if max_records <= 0:
        return []
    if not available:
        return []

    # Apply scheduling policy
    if schedule_policy == "fixed_subset":
        # Same records every round
        rng = random.Random(seed)
        shuffled = available.copy()
        rng.shuffle(shuffled)
        return shuffled[:max_records]

    elif schedule_policy == "deterministic_window":
        # Sliding window across rounds
        rng = random.Random(seed)
        shuffled = available.copy()
        rng.shuffle(shuffled)

        # Window wraps around
        n = len(shuffled)
        k = max_records
        start = (round_id * k) % n
        end = start + k

        if end <= n:
            return shuffled[start:end]
        else:
            # Wrap around
            return shuffled[start:] + shuffled[: end - n]

    else:
        raise ValueError(f"Unknown schedule_policy: {schedule_policy}")


def build_round_plan(
    config: Phase4CotrainingConfig,
    split_manifest: DatasetSplitManifest,
    round_id: int,
    previous_round: Phase4RoundReport | None,
) -> Phase4RoundPlan:
    """
    Build execution plan for one round.

    Args:
        config: experiment configuration
        split_manifest: dataset split manifest
        round_id: round number
        previous_round: previous round report (for checkpoint carryover)

    Returns:
        Round plan with record IDs and checkpoint IDs
    """
    # Select train records
    train_record_ids = select_round_records(
        split_manifest,
        DatasetSplitName.TRAIN,
        round_id=round_id,
        max_records=config.train_records_per_round,
        explicit_record_ids=config.train_record_ids,
        seed=config.seed,
        schedule_policy=config.record_schedule_policy,
    )

    # Select dev records
    dev_record_ids = select_round_records(
        split_manifest,
        DatasetSplitName.DEV,
        round_id=round_id,
        max_records=config.dev_records_per_round,
        explicit_record_ids=config.dev_record_ids,
        seed=config.seed,
        schedule_policy=config.record_schedule_policy,
    )

    # Determine checkpoint IDs for this round
    memory_builder_checkpoint_id: str | None = None
    question_agent_checkpoint_id: str | None = None
    checkpoint_registry_path: str | None = None

    if previous_round is not None:
        # Carry over selected checkpoints from previous round. If dev selection
        # rejected one actor, keep that actor's previous input checkpoint.
        memory_builder_checkpoint_id = (
            previous_round.selected_memory_builder_checkpoint_id
            or previous_round.input_memory_builder_checkpoint_id
        )
        question_agent_checkpoint_id = (
            previous_round.selected_question_agent_checkpoint_id
            or previous_round.input_question_agent_checkpoint_id
        )
        checkpoint_registry_path = (
            previous_round.metrics.get("checkpoint_registry_path")
            if previous_round.metrics
            else None
        )
    else:
        # Round 0: use initial checkpoints from config
        memory_builder_checkpoint_id = config.initial_memory_builder_checkpoint_id
        question_agent_checkpoint_id = config.initial_question_agent_checkpoint_id
        checkpoint_registry_path = config.checkpoint_registry_path

    # Output directory
    output_dir = str(Path(config.output_dir) / "rounds" / f"round_{round_id:03d}")

    return Phase4RoundPlan(
        round_id=round_id,
        train_record_ids=train_record_ids,
        dev_record_ids=dev_record_ids,
        memory_builder_checkpoint_id=memory_builder_checkpoint_id,
        question_agent_checkpoint_id=question_agent_checkpoint_id,
        checkpoint_registry_path=checkpoint_registry_path,
        output_dir=output_dir,
        seed=config.seed,
    )


def get_previous_round_report(
    round_reports: list[Phase4RoundReport],
    round_id: int,
) -> Phase4RoundReport | None:
    """
    Get previous round report for checkpoint carryover.

    Args:
        round_reports: all round reports so far
        round_id: current round ID

    Returns:
        Previous round report, or None if round 0
    """
    if round_id == 0:
        return None

    # Find most recent completed round before this one
    previous_rounds = [r for r in round_reports if r.round_id < round_id]
    if not previous_rounds:
        return None

    # Return latest completed round
    return max(previous_rounds, key=lambda r: r.round_id)
