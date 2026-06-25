"""
Phase 4 Milestone 1: Dataset Splitter

Core logic for creating deterministic train/dev/test splits from LongMemEval records.

Key principles:
- Deterministic assignment from seed
- Support both random and explicit split modes
- Validate oracle graph existence and record_id matching
- Never store benchmark question or answer text in split manifest
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

from gaam_graph.dataset_split_schema import (
    ISSUE_DUPLICATE_RECORD_ID,
    ISSUE_DUPLICATE_SPLIT_ASSIGNMENT,
    ISSUE_EMPTY_SPLIT,
    ISSUE_INVALID_RATIO,
    ISSUE_MISSING_ORACLE_GRAPH,
    ISSUE_MISSING_RAW_RECORD,
    ISSUE_ORACLE_GRAPH_RECORD_ID_MISMATCH,
    ISSUE_RECORD_ID_NOT_FOUND,
    DatasetSplitIssue,
    DatasetSplitIssueSeverity,
    DatasetSplitManifest,
    DatasetSplitName,
    DatasetSplitRecord,
    compute_split_assignment_hash,
    split_counts,
)
from gaam_graph.lme_loader import LongMemEvalLoader


def load_lme_records_for_split(
    input_path: Path,
    record_id_filter: list[str] | None = None,
) -> list[dict]:
    """Load LongMemEval records for split creation.

    Returns list of dicts with:
    - record_id
    - source_index
    - raw_record (for metadata extraction only, not stored in manifest)
    """
    loader = LongMemEvalLoader(input_path)
    records = loader.load()

    result = []
    for idx, record in enumerate(records):
        if record_id_filter and record.record_id not in record_id_filter:
            continue
        result.append({
            "record_id": record.record_id,
            "source_index": idx,
            "raw_record": record,
        })

    return result


def read_record_id_file(path: Path) -> list[str]:
    """Read record IDs from text file.

    Format: one record_id per line, blank lines and # comments ignored.
    """
    record_ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            record_ids.append(line)
    return record_ids


def validate_oracle_graph_record_id(
    record_id: str,
    oracle_graph_path: Path,
    *,
    missing_severity: DatasetSplitIssueSeverity = DatasetSplitIssueSeverity.ERROR,
) -> list[DatasetSplitIssue]:
    """Validate oracle graph existence and embedded record_id."""
    issues: list[DatasetSplitIssue] = []

    if not oracle_graph_path.exists():
        issues.append(
            DatasetSplitIssue(
                severity=missing_severity,
                code=ISSUE_MISSING_ORACLE_GRAPH,
                message=f"Oracle graph not found: {oracle_graph_path}",
                record_id=record_id,
                path=str(oracle_graph_path),
            )
        )
        return issues

    try:
        with open(oracle_graph_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)
        graph_record_id = graph_data.get("graph", {}).get("record_id")
        if graph_record_id != record_id:
            issues.append(
                DatasetSplitIssue(
                    severity=DatasetSplitIssueSeverity.ERROR,
                    code=ISSUE_ORACLE_GRAPH_RECORD_ID_MISMATCH,
                    message=(
                        "Oracle graph record_id mismatch: "
                        f"expected {record_id}, got {graph_record_id}"
                    ),
                    record_id=record_id,
                    path=str(oracle_graph_path),
                )
            )
    except Exception as e:
        issues.append(
            DatasetSplitIssue(
                severity=DatasetSplitIssueSeverity.WARNING,
                code=ISSUE_ORACLE_GRAPH_RECORD_ID_MISMATCH,
                message=f"Failed to read oracle graph: {e}",
                record_id=record_id,
                path=str(oracle_graph_path),
            )
        )

    return issues


def create_random_split(
    input_path: Path,
    oracle_graph_dir: Path,
    dataset_name: str,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int,
    record_id_filter: list[str] | None = None,
    require_oracle_graph: bool = True,
    debug_single_case: bool = False,
    allow_empty_split: bool = False,
) -> DatasetSplitManifest:
    """Create random train/dev/test split with deterministic seeded assignment.

    Args:
        input_path: Path to LongMemEval input JSON
        oracle_graph_dir: Directory containing oracle graph files
        dataset_name: Name of dataset (e.g. "longmemeval")
        train_ratio: Fraction for train split
        dev_ratio: Fraction for dev split
        test_ratio: Fraction for test split
        seed: Random seed for deterministic assignment
        record_id_filter: Optional list of record IDs to include
        require_oracle_graph: Fail if oracle graph missing
        debug_single_case: Allow single-record train-only split for debugging
        allow_empty_split: Allow empty dev/test splits

    Returns:
        DatasetSplitManifest with assigned records and validation issues
    """
    issues: list[DatasetSplitIssue] = []

    # Load records
    loaded_records = load_lme_records_for_split(input_path, record_id_filter)

    if not loaded_records:
        issues.append(
            DatasetSplitIssue(
                severity=DatasetSplitIssueSeverity.ERROR,
                code=ISSUE_RECORD_ID_NOT_FOUND,
                message="No records loaded from input",
            )
        )
        # Return empty manifest with issues
        return DatasetSplitManifest(
            dataset_name=dataset_name,
            source_records_path=str(input_path),
            oracle_graph_dir=str(oracle_graph_dir),
            seed=seed,
            split_ratios={
                DatasetSplitName.TRAIN: train_ratio,
                DatasetSplitName.DEV: dev_ratio,
                DatasetSplitName.TEST: test_ratio,
            },
            created_at=datetime.utcnow().isoformat(),
            records=[],
            counts={split: 0 for split in DatasetSplitName},
            record_id_hash="",
            issues=issues,
        )

    # Deterministic random assignment
    rng = random.Random(seed)
    record_ids = sorted([r["record_id"] for r in loaded_records])
    shuffled_ids = record_ids.copy()
    rng.shuffle(shuffled_ids)

    n = len(shuffled_ids)
    train_count = int(n * train_ratio)
    dev_count = int(n * dev_ratio)
    test_count = n - train_count - dev_count

    # Debug single-case mode
    if debug_single_case and n == 1:
        train_count = 1
        dev_count = 0
        test_count = 0
    elif not allow_empty_split:
        # Validate non-empty splits for real training
        if n >= 3:
            if train_count == 0 or dev_count == 0 or test_count == 0:
                issues.append(
                    DatasetSplitIssue(
                        severity=DatasetSplitIssueSeverity.ERROR,
                        code=ISSUE_EMPTY_SPLIT,
                        message=f"Split ratios produce empty split with {n} records: train={train_count}, dev={dev_count}, test={test_count}",
                    )
                )
        elif n < 3 and not debug_single_case:
            issues.append(
                DatasetSplitIssue(
                    severity=DatasetSplitIssueSeverity.ERROR,
                    code=ISSUE_EMPTY_SPLIT,
                    message=f"Need at least 3 records for train/dev/test split, got {n}. Use --debug_single_case for single-record debugging.",
                )
            )

    # Assign splits
    split_assignment = {}
    for i, record_id in enumerate(shuffled_ids):
        if i < train_count:
            split_assignment[record_id] = DatasetSplitName.TRAIN
        elif i < train_count + dev_count:
            split_assignment[record_id] = DatasetSplitName.DEV
        else:
            split_assignment[record_id] = DatasetSplitName.TEST

    # Build split records with validation
    split_records = []
    record_lookup = {r["record_id"]: r for r in loaded_records}

    for record_id in sorted(split_assignment.keys()):
        split = split_assignment[record_id]
        loaded = record_lookup[record_id]
        raw_record = loaded["raw_record"]

        oracle_graph_path = oracle_graph_dir / f"{record_id}.graph.json"
        issues.extend(
            validate_oracle_graph_record_id(
                record_id,
                oracle_graph_path,
                missing_severity=(
                    DatasetSplitIssueSeverity.ERROR
                    if require_oracle_graph
                    else DatasetSplitIssueSeverity.WARNING
                ),
            )
        )

        # Build non-sensitive metadata
        metadata = {
            "event_count": len(raw_record.events) if raw_record.events else 0,
        }

        # Check if benchmark question exists (boolean only, not text)
        has_benchmark_question = bool(
            hasattr(raw_record, "target_question") and raw_record.target_question
        )
        has_benchmark_answer = bool(
            hasattr(raw_record, "gold_answer") and raw_record.gold_answer
        )

        split_records.append(
            DatasetSplitRecord(
                record_id=record_id,
                split=split,
                source_index=loaded["source_index"],
                raw_record_path=str(input_path),
                oracle_graph_path=str(oracle_graph_path),
                has_benchmark_question=has_benchmark_question,
                has_benchmark_answer=has_benchmark_answer,
                metadata=metadata,
            )
        )

    # Compute counts and hash
    counts = split_counts(split_records)
    record_id_hash = compute_split_assignment_hash(split_records)

    return DatasetSplitManifest(
        dataset_name=dataset_name,
        source_records_path=str(input_path),
        oracle_graph_dir=str(oracle_graph_dir),
        seed=seed,
        split_ratios={
            DatasetSplitName.TRAIN: train_ratio,
            DatasetSplitName.DEV: dev_ratio,
            DatasetSplitName.TEST: test_ratio,
        },
        created_at=datetime.utcnow().isoformat(),
        records=split_records,
        counts=counts,
        record_id_hash=record_id_hash,
        issues=issues,
    )


def create_explicit_split(
    input_path: Path,
    oracle_graph_dir: Path,
    dataset_name: str,
    train_ids: list[str],
    dev_ids: list[str],
    test_ids: list[str],
    require_oracle_graph: bool = True,
) -> DatasetSplitManifest:
    """Create split from explicit record ID lists.

    Args:
        input_path: Path to LongMemEval input JSON
        oracle_graph_dir: Directory containing oracle graph files
        dataset_name: Name of dataset
        train_ids: Record IDs for train split
        dev_ids: Record IDs for dev split
        test_ids: Record IDs for test split
        require_oracle_graph: Fail if oracle graph missing

    Returns:
        DatasetSplitManifest with validation issues
    """
    issues: list[DatasetSplitIssue] = []

    # Validate no duplicates within splits
    for split_name, ids in [
        ("train", train_ids),
        ("dev", dev_ids),
        ("test", test_ids),
    ]:
        seen = set()
        for record_id in ids:
            if record_id in seen:
                issues.append(
                    DatasetSplitIssue(
                        severity=DatasetSplitIssueSeverity.ERROR,
                        code=ISSUE_DUPLICATE_RECORD_ID,
                        message=f"Duplicate record_id in {split_name} split: {record_id}",
                        record_id=record_id,
                    )
                )
            seen.add(record_id)

    # Validate no cross-split duplicates
    all_ids = set(train_ids) | set(dev_ids) | set(test_ids)
    if len(all_ids) < len(train_ids) + len(dev_ids) + len(test_ids):
        # Find duplicates
        for record_id in all_ids:
            count = (
                (1 if record_id in train_ids else 0)
                + (1 if record_id in dev_ids else 0)
                + (1 if record_id in test_ids else 0)
            )
            if count > 1:
                issues.append(
                    DatasetSplitIssue(
                        severity=DatasetSplitIssueSeverity.ERROR,
                        code=ISSUE_DUPLICATE_SPLIT_ASSIGNMENT,
                        message=f"Record assigned to multiple splits: {record_id}",
                        record_id=record_id,
                    )
                )

    # Load records
    all_record_ids = sorted(all_ids)
    loaded_records = load_lme_records_for_split(input_path, all_record_ids)
    record_lookup = {r["record_id"]: r for r in loaded_records}

    # Check all requested IDs exist
    loaded_ids = set(record_lookup.keys())
    missing_ids = all_ids - loaded_ids
    for record_id in missing_ids:
        issues.append(
            DatasetSplitIssue(
                severity=DatasetSplitIssueSeverity.ERROR,
                code=ISSUE_RECORD_ID_NOT_FOUND,
                message=f"Record ID not found in input: {record_id}",
                record_id=record_id,
            )
        )

    # Build split records
    split_records = []
    split_assignment = {}
    for record_id in train_ids:
        split_assignment[record_id] = DatasetSplitName.TRAIN
    for record_id in dev_ids:
        split_assignment[record_id] = DatasetSplitName.DEV
    for record_id in test_ids:
        split_assignment[record_id] = DatasetSplitName.TEST

    for record_id in sorted(split_assignment.keys()):
        if record_id not in record_lookup:
            continue  # Already reported as missing

        split = split_assignment[record_id]
        loaded = record_lookup[record_id]
        raw_record = loaded["raw_record"]

        oracle_graph_path = oracle_graph_dir / f"{record_id}.graph.json"
        issues.extend(
            validate_oracle_graph_record_id(
                record_id,
                oracle_graph_path,
                missing_severity=(
                    DatasetSplitIssueSeverity.ERROR
                    if require_oracle_graph
                    else DatasetSplitIssueSeverity.WARNING
                ),
            )
        )

        # Build non-sensitive metadata
        metadata = {
            "event_count": len(raw_record.events) if raw_record.events else 0,
        }

        has_benchmark_question = bool(
            hasattr(raw_record, "target_question") and raw_record.target_question
        )
        has_benchmark_answer = bool(
            hasattr(raw_record, "gold_answer") and raw_record.gold_answer
        )

        split_records.append(
            DatasetSplitRecord(
                record_id=record_id,
                split=split,
                source_index=loaded["source_index"],
                raw_record_path=str(input_path),
                oracle_graph_path=str(oracle_graph_path),
                has_benchmark_question=has_benchmark_question,
                has_benchmark_answer=has_benchmark_answer,
                metadata=metadata,
            )
        )

    # Compute counts and hash
    counts = split_counts(split_records)
    record_id_hash = compute_split_assignment_hash(split_records)

    # Compute approximate ratios for display
    total = len(split_records)
    split_ratios = {
        DatasetSplitName.TRAIN: counts[DatasetSplitName.TRAIN] / total if total > 0 else 0.0,
        DatasetSplitName.DEV: counts[DatasetSplitName.DEV] / total if total > 0 else 0.0,
        DatasetSplitName.TEST: counts[DatasetSplitName.TEST] / total if total > 0 else 0.0,
    }

    return DatasetSplitManifest(
        dataset_name=dataset_name,
        source_records_path=str(input_path),
        oracle_graph_dir=str(oracle_graph_dir),
        seed=-1,  # Explicit split mode has no seed
        split_ratios=split_ratios,
        created_at=datetime.utcnow().isoformat(),
        records=split_records,
        counts=counts,
        record_id_hash=record_id_hash,
        issues=issues,
    )


def validate_split_manifest(manifest: DatasetSplitManifest) -> list[DatasetSplitIssue]:
    """Re-validate split manifest by checking referenced paths.

    Returns list of new issues discovered during validation.
    """
    issues: list[DatasetSplitIssue] = []

    # Check source records path
    source_path = Path(manifest.source_records_path)
    if not source_path.exists():
        issues.append(
            DatasetSplitIssue(
                severity=DatasetSplitIssueSeverity.ERROR,
                code=ISSUE_MISSING_RAW_RECORD,
                message=f"Source records path not found: {source_path}",
                path=str(source_path),
            )
        )

    # Check oracle graph dir
    graph_dir = Path(manifest.oracle_graph_dir)
    if not graph_dir.exists():
        issues.append(
            DatasetSplitIssue(
                severity=DatasetSplitIssueSeverity.ERROR,
                code=ISSUE_MISSING_ORACLE_GRAPH,
                message=f"Oracle graph directory not found: {graph_dir}",
                path=str(graph_dir),
            )
        )

    for record in manifest.records:
        graph_path = Path(record.oracle_graph_path)
        issues.extend(validate_oracle_graph_record_id(record.record_id, graph_path))

    # Check for duplicate record IDs
    seen_ids = set()
    for record in manifest.records:
        if record.record_id in seen_ids:
            issues.append(
                DatasetSplitIssue(
                    severity=DatasetSplitIssueSeverity.ERROR,
                    code=ISSUE_DUPLICATE_RECORD_ID,
                    message=f"Duplicate record_id in manifest: {record.record_id}",
                    record_id=record.record_id,
                )
            )
        seen_ids.add(record.record_id)

    return issues


def write_split_manifest(manifest: DatasetSplitManifest, output_path: Path) -> None:
    """Write split manifest to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(mode="json"), f, ensure_ascii=False, indent=2)


def load_dataset_split_manifest(manifest_path: Path) -> DatasetSplitManifest:
    """Load dataset split manifest from JSON file."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return DatasetSplitManifest.model_validate(data)

