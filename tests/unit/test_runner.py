"""UNIT TEST -- checkpointing and parallel execution. No network."""

from __future__ import annotations

import json
import threading

import pytest

from georef.pipeline.runner import JsonlCheckpoint, run_parallel


def make_items(count: int) -> list[dict]:
    return [{"benchmark_id": f"ID-{i:03d}", "value": i} for i in range(count)]


class TestCheckpoint:
    def test_round_trips_records(self, tmp_path):
        checkpoint = JsonlCheckpoint(tmp_path / "out.jsonl")
        checkpoint.append({"benchmark_id": "A", "status": "SUCCESS"})
        assert checkpoint.completed()["A"]["status"] == "SUCCESS"

    def test_missing_file_is_an_empty_checkpoint(self, tmp_path):
        assert JsonlCheckpoint(tmp_path / "nothing.jsonl").completed() == {}

    def test_finalise_sorts_by_id(self, tmp_path):
        """Concurrent workers finish out of order; the artefact must not."""
        checkpoint = JsonlCheckpoint(tmp_path / "out.jsonl")
        for key in ("ID-003", "ID-001", "ID-002"):
            checkpoint.append({"benchmark_id": key})
        checkpoint.finalise()
        lines = (tmp_path / "out.jsonl").read_text(encoding="utf-8").strip().split("\n")
        assert [json.loads(line)["benchmark_id"] for line in lines] == [
            "ID-001", "ID-002", "ID-003"
        ]

    def test_finalise_is_idempotent(self, tmp_path):
        checkpoint = JsonlCheckpoint(tmp_path / "out.jsonl")
        checkpoint.append({"benchmark_id": "A"})
        checkpoint.finalise()
        first = (tmp_path / "out.jsonl").read_bytes()
        checkpoint.finalise()
        assert (tmp_path / "out.jsonl").read_bytes() == first

    def test_a_rerun_of_the_same_id_keeps_the_latest(self, tmp_path):
        checkpoint = JsonlCheckpoint(tmp_path / "out.jsonl")
        checkpoint.append({"benchmark_id": "A", "status": "API_FAILURE"})
        checkpoint.append({"benchmark_id": "A", "status": "SUCCESS"})
        checkpoint.finalise()
        assert checkpoint.completed()["A"]["status"] == "SUCCESS"
        assert len(checkpoint.completed()) == 1

    def test_concurrent_appends_do_not_interleave(self, tmp_path):
        """Unlocked writes from many threads produce corrupt half-lines."""
        checkpoint = JsonlCheckpoint(tmp_path / "out.jsonl")
        payload = "x" * 2000

        def writer(index: int):
            for step in range(20):
                checkpoint.append({"benchmark_id": f"{index}-{step}", "pad": payload})

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        lines = (tmp_path / "out.jsonl").read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 160
        for line in lines:
            json.loads(line)  # raises if a write was torn


class TestRunParallel:
    @pytest.mark.parametrize("workers", [1, 4, 16])
    def test_every_item_is_processed_exactly_once(self, tmp_path, workers):
        checkpoint = JsonlCheckpoint(tmp_path / "out.jsonl")
        items = make_items(50)
        run_parallel(items, lambda i: {**i, "status": "SUCCESS"}, checkpoint, workers=workers)
        done = checkpoint.completed()
        assert len(done) == 50
        assert {r["value"] for r in done.values()} == set(range(50))

    def test_results_are_identical_whatever_the_worker_count(self, tmp_path):
        """Records are independent, so concurrency must not change outcomes."""
        outputs = []
        for workers in (1, 8):
            checkpoint = JsonlCheckpoint(tmp_path / f"out{workers}.jsonl")
            run_parallel(make_items(30), lambda i: {**i, "status": "SUCCESS"},
                         checkpoint, workers=workers)
            checkpoint.finalise()
            outputs.append((tmp_path / f"out{workers}.jsonl").read_text())
        assert outputs[0] == outputs[1]

    def test_one_failing_record_does_not_abort_the_run(self, tmp_path):
        def work(item):
            if item["value"] == 3:
                raise RuntimeError("boom")
            return {**item, "status": "SUCCESS"}

        checkpoint = JsonlCheckpoint(tmp_path / "out.jsonl")
        run_parallel(make_items(10), work, checkpoint, workers=4)
        done = checkpoint.completed()
        assert len(done) == 10
        assert done["ID-003"]["status"] == "RUNNER_EXCEPTION"
        assert "boom" in done["ID-003"]["failure_reason"]

    def test_progress_callback_counts_to_the_total(self, tmp_path):
        seen = []
        run_parallel(
            make_items(12),
            lambda i: {**i, "status": "SUCCESS"},
            JsonlCheckpoint(tmp_path / "out.jsonl"),
            workers=4,
            on_result=lambda record, position, total: seen.append((position, total)),
        )
        assert sorted(p for p, _ in seen) == list(range(1, 13))
        assert {t for _, t in seen} == {12}

    def test_workers_actually_run_concurrently(self, tmp_path):
        active, peak = 0, 0
        lock = threading.Lock()
        barrier = threading.Barrier(4, timeout=5)

        def work(item):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                pass
            with lock:
                active -= 1
            return {**item, "status": "SUCCESS"}

        run_parallel(make_items(8), work, JsonlCheckpoint(tmp_path / "out.jsonl"), workers=4)
        assert peak >= 2
