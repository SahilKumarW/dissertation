"""Resumable, parallel execution over a set of records.

Used by the extraction stage and, later, by both experimental conditions, so
checkpointing and resume behave identically everywhere (master spec section 53).

Parallelism is safe here because records are independent: each description is
interpreted on its own, nothing is shared between them, and no record's result
influences another's. Worker count therefore changes wall-clock time and
nothing else about the outcome.

Two things are NOT left to chance:

* **Resume never duplicates or silently overwrites.** Completed record IDs are
  read from the checkpoint before anything is requested, and only missing ones
  are dispatched.
* **The finished artefact is order-stable.** Concurrent workers finish in
  arbitrary order, so the file is rewritten sorted by record ID at the end. An
  artefact whose byte content depends on thread scheduling cannot be hashed
  meaningfully.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

__all__ = ["JsonlCheckpoint", "run_parallel"]


class JsonlCheckpoint:
    """Append-only JSONL checkpoint keyed by a record ID field."""

    def __init__(self, path: str | Path, id_field: str = "benchmark_id"):
        self.path = Path(path)
        self.id_field = id_field
        self._lock = threading.Lock()

    def completed(self) -> dict[str, dict]:
        """Records already written, keyed by ID. Later lines win on duplicates."""
        if not self.path.is_file():
            return {}
        done: dict[str, dict] = {}
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    done[record[self.id_field]] = record
        return done

    def append(self, record: dict) -> None:
        """Write one record. Safe to call from several threads."""
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)
                handle.flush()

    def finalise(self) -> int:
        """Rewrite sorted by ID, keeping the last entry for any duplicate.

        Called once a run completes, so the artefact does not depend on the
        order threads happened to finish in.
        """
        records = self.completed()
        with self._lock:
            with open(self.path, "w", encoding="utf-8", newline="\n") as handle:
                for key in sorted(records):
                    handle.write(json.dumps(records[key], ensure_ascii=False) + "\n")
        return len(records)


def run_parallel(
    items: Iterable,
    work: Callable[[object], dict],
    checkpoint: JsonlCheckpoint,
    workers: int = 1,
    on_result: Callable[[dict, int, int], None] | None = None,
) -> list[dict]:
    """Run ``work`` over ``items`` with bounded concurrency, checkpointing each result.

    A failure inside ``work`` for one record must not abort the run: the whole
    point of per-record status codes is that one bad record costs one record.
    Exceptions are converted into a record carrying the error, which is
    checkpointed like any other so the failure is preserved rather than lost.
    """
    items = list(items)
    total = len(items)
    results: list[dict] = []
    completed = 0
    counter_lock = threading.Lock()

    def _run(item):
        try:
            return work(item)
        except Exception as error:  # noqa: BLE001 -- preserved, not swallowed
            return {
                checkpoint.id_field: getattr(item, checkpoint.id_field, None)
                or (item.get(checkpoint.id_field) if isinstance(item, dict) else None),
                "status": "RUNNER_EXCEPTION",
                "failure_reason": f"{type(error).__name__}: {error}",
            }

    if workers <= 1:
        for item in items:
            record = _run(item)
            checkpoint.append(record)
            results.append(record)
            completed += 1
            if on_result:
                on_result(record, completed, total)
        return results

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run, item): item for item in items}
        for future in as_completed(futures):
            record = future.result()
            checkpoint.append(record)
            with counter_lock:
                completed += 1
                position = completed
            results.append(record)
            if on_result:
                on_result(record, position, total)
    return results
