"""Contract tests for pipeline_runner._make_progress_ticker / _iter_with_progress —
the shared item-level progress interpolation used by every sequential test runner
(run_agentic_test, run_rag_test, etc.) so a slow multi-minute test doesn't sit frozen
at its test-start percentage for its entire duration.
"""
import importlib
import sys
import threading
import types
import unittest


def _load_pipeline_runner_module():
    fake_anthropic = sys.modules.get("anthropic")
    if fake_anthropic is None:
        fake_anthropic = types.ModuleType("anthropic")

        class Anthropic:  # pragma: no cover - import stub for tests only
            pass

        fake_anthropic.Anthropic = Anthropic
        sys.modules["anthropic"] = fake_anthropic

    fake_datasets = sys.modules.get("datasets")
    if fake_datasets is None:
        fake_datasets = types.ModuleType("datasets")

        def load_dataset(*args, **kwargs):
            raise RuntimeError("load_dataset should not run in progress contract tests")

        fake_datasets.load_dataset = load_dataset
        sys.modules["datasets"] = fake_datasets

    return importlib.import_module("pipeline_runner")


class _FakeRun:
    def __init__(self):
        self.progress = 0.0


class ProgressTickerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pipeline_cls = _load_pipeline_runner_module().EvaluationPipeline

        # A minimal stand-in exposing only what the progress helpers touch on self,
        # carrying the *real* (unbound) implementations under test as its own methods.
        class _PipelineStub:
            _make_progress_ticker = pipeline_cls._make_progress_ticker
            _iter_with_progress = pipeline_cls._iter_with_progress
            _tqdm_position = pipeline_cls._tqdm_position

            def __init__(self, run, test_idx, total_tests):
                self._run = run
                self._progress_test_idx = test_idx
                self._progress_total_tests = total_tests
                self._tqdm_position_by_thread = threading.local()

        cls.PipelineStub = _PipelineStub

    def test_tick_interpolates_within_current_test_slot(self):
        run = _FakeRun()
        stub = self.PipelineStub(run, test_idx=0, total_tests=2)
        tick = stub._make_progress_ticker(total_items=4)

        tick()
        self.assertAlmostEqual(run.progress, (0 + 1 / 4) / 2)
        tick()
        self.assertAlmostEqual(run.progress, (0 + 2 / 4) / 2)
        tick()
        tick()
        self.assertAlmostEqual(run.progress, (0 + 4 / 4) / 2)

    def test_tick_offsets_by_test_idx(self):
        run = _FakeRun()
        stub = self.PipelineStub(run, test_idx=1, total_tests=2)
        tick = stub._make_progress_ticker(total_items=2)

        tick()
        self.assertAlmostEqual(run.progress, (1 + 1 / 2) / 2)

    def test_no_run_wired_is_noop(self):
        stub = self.PipelineStub(run=None, test_idx=0, total_tests=1)
        tick = stub._make_progress_ticker(total_items=1)
        tick()  # must not raise even though self._run is None

    def test_iter_with_progress_ticks_once_per_item_including_skipped(self):
        run = _FakeRun()
        stub = self.PipelineStub(run, test_idx=0, total_tests=1)
        dataset = ["a", "b", "c"]

        seen = []
        for item in stub._iter_with_progress(dataset, "some_test"):
            seen.append(item)
            if item == "b":
                continue  # simulates a runner skipping an invalid item

        self.assertEqual(seen, dataset)
        self.assertAlmostEqual(run.progress, 1.0)

    def test_iter_with_progress_reaches_full_progress_after_all_items(self):
        run = _FakeRun()
        stub = self.PipelineStub(run, test_idx=0, total_tests=1)
        dataset = list(range(5))

        # Each item's tick fires right after its body finishes (between fetching the
        # current and next item), so the snapshot taken at the *start* of body(item_i)
        # reflects item_{i-1}'s completion — the final item's tick only lands once the
        # `for` statement itself is exhausted, hence it's checked after the loop below.
        progress_snapshots = []
        for _ in stub._iter_with_progress(dataset, "some_test"):
            progress_snapshots.append(run.progress)

        self.assertEqual(len(progress_snapshots), 5)
        self.assertEqual(progress_snapshots, sorted(progress_snapshots))
        self.assertAlmostEqual(run.progress, 1.0)


if __name__ == "__main__":
    unittest.main()
