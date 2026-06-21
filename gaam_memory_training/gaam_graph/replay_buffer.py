"""
Phase 3 Milestone 5: Replay Buffer

Lightweight append-only replay buffer for trainer-safe rollout items.

Key features:
- JSONL append-only storage
- Iterator-based reading
- Deterministic sampling
- Filtering by record_id and no-leakage status
- Graceful malformed line handling
"""

import json
import logging
from pathlib import Path
from typing import Iterator

from gaam_graph.distributed_reward_schema import (
    ReplayBufferItem,
    ReplayBufferManifestEntry,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Replay Buffer Writer
# ============================================================================


class ReplayBufferWriter:
    """Append-only replay buffer writer."""

    def __init__(self, path: Path):
        """
        Initialize replay buffer writer.

        Args:
            path: Path to replay buffer JSONL file
        """
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, item: ReplayBufferItem) -> None:
        """
        Append a replay item to the buffer.

        Args:
            item: ReplayBufferItem to append
        """
        with open(self.path, "a", encoding="utf-8") as f:
            json_line = json.dumps(item.model_dump(mode="json"))
            f.write(json_line + "\n")

    def write_manifest(self, entries: list[ReplayBufferManifestEntry]) -> None:
        """
        Write manifest file for replay buffer.

        Args:
            entries: List of manifest entries
        """
        manifest_path = self.path.parent / "manifest.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for entry in entries:
                json_line = json.dumps(entry.model_dump(mode="json"))
                f.write(json_line + "\n")


# ============================================================================
# Replay Buffer Reader
# ============================================================================


class ReplayBufferReader:
    """Replay buffer reader with filtering and sampling."""

    def __init__(self, path: Path, strict: bool = False):
        """
        Initialize replay buffer reader.

        Args:
            path: Path to replay buffer JSONL file
            strict: If True, raise on malformed lines; if False, skip with warning
        """
        self.path = path
        self.strict = strict

        if not self.path.exists():
            raise FileNotFoundError(f"Replay buffer not found: {self.path}")

    def iter_items(self) -> Iterator[ReplayBufferItem]:
        """
        Iterate over all replay items.

        Yields:
            ReplayBufferItem instances
        """
        with open(self.path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    item = ReplayBufferItem.model_validate(data)
                    yield item
                except Exception as e:
                    if self.strict:
                        raise ValueError(
                            f"Malformed replay item at line {line_num}: {type(e).__name__}: {e}"
                        ) from e
                    else:
                        logger.warning(
                            f"Skipping malformed replay item at line {line_num}: {type(e).__name__}: {e}"
                        )
                        continue

    def sample(
        self,
        *,
        max_items: int = 100,
        record_id: str | None = None,
        require_no_leakage: bool = True,
        require_selected_for_training: bool = True,
    ) -> list[ReplayBufferItem]:
        """
        Sample replay items with filtering.

        For MVP, sampling is deterministic (first N matching items).
        Weighted sampling and TopVar prioritization can be added later.

        Args:
            max_items: Maximum number of items to return
            record_id: Optional record_id filter
            require_no_leakage: Filter to items that passed no-leakage validation
            require_selected_for_training: Filter to items selected for training

        Returns:
            List of replay items
        """
        items = []

        for item in self.iter_items():
            # Apply filters
            if record_id is not None and item.record_id != record_id:
                continue

            if require_no_leakage and not item.no_leakage_passed:
                continue

            if require_selected_for_training and not item.selected_for_training:
                continue

            items.append(item)

            if len(items) >= max_items:
                break

        return items

    def count(
        self,
        *,
        record_id: str | None = None,
        require_no_leakage: bool = True,
        require_selected_for_training: bool = True,
    ) -> int:
        """
        Count replay items matching filters.

        Args:
            record_id: Optional record_id filter
            require_no_leakage: Filter to items that passed no-leakage validation
            require_selected_for_training: Filter to items selected for training

        Returns:
            Count of matching items
        """
        count = 0

        for item in self.iter_items():
            # Apply filters
            if record_id is not None and item.record_id != record_id:
                continue

            if require_no_leakage and not item.no_leakage_passed:
                continue

            if require_selected_for_training and not item.selected_for_training:
                continue

            count += 1

        return count
