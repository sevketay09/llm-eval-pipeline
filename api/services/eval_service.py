"""Service for running evaluations with progress tracking."""
from __future__ import annotations
import asyncio
import uuid
import time
from datetime import datetime, timezone
from typing import Any, Optional
from pathlib import Path

from api.config import get_settings
from api.schemas.evaluations import EvalRunRequest, EvalProgress, EvalRunStatus


class EvalRunError(Exception):
    """Structured evaluation failure surfaced to API/UI clients."""

    def __init__(self, code: str, stage: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.stage = stage
        self.detail = detail


class EvalRun:
    """Tracks a single evaluation run."""

    def __init__(self, request: EvalRunRequest):
        self.run_id = str(uuid.uuid4())
        self.request = request
        self.status = "pending"
        self.progress = 0.0
        self.current_model: str | None = None
        self.current_test: str | None = None
        self.message = ""
        self.started_at = datetime.now(timezone.utc)
        self.finished_at: datetime | None = None
        self.report_path: str | None = None
        self.error: str | None = None
        self.error_code: str | None = None
        self.error_stage: str | None = None
        self._task: asyncio.Task | None = None

    def to_status(self) -> EvalRunStatus:
        return EvalRunStatus(
            run_id=self.run_id,
            status=self.status,
            progress=self.progress,
            started_at=self.started_at,
            finished_at=self.finished_at,
            report_path=self.report_path,
            error=self.error,
            error_code=self.error_code,
            error_stage=self.error_stage,
        )

    def to_progress(self) -> EvalProgress:
        elapsed = (datetime.now(timezone.utc) - self.started_at).total_seconds()
        return EvalProgress(
            run_id=self.run_id,
            status=self.status,
            progress=self.progress,
            current_model=self.current_model,
            current_test=self.current_test,
            message=self.message,
            started_at=self.started_at,
            elapsed_seconds=elapsed,
            error_code=self.error_code,
            error_stage=self.error_stage,
        )


class EvalService:
    """Manages evaluation runs with async execution and progress tracking."""

    def __init__(self):
        self._runs: dict[str, EvalRun] = {}
        self._progress_callbacks: dict[str, list] = {}

    @property
    def active_runs(self) -> list[EvalRunStatus]:
        return [r.to_status() for r in self._runs.values() if r.status == "running"]

    def get_run(self, run_id: str) -> EvalRun | None:
        return self._runs.get(run_id)

    def list_runs(self, limit: int = 20) -> list[EvalRunStatus]:
        sorted_runs = sorted(self._runs.values(), key=lambda r: r.started_at, reverse=True)
        return [r.to_status() for r in sorted_runs[:limit]]

    def start_run(self, request: EvalRunRequest) -> EvalRun:
        """Start an evaluation run in background."""
        run = EvalRun(request)
        self._runs[run.run_id] = run
        run._task = asyncio.create_task(self._execute_run(run))
        return run

    def cancel_run(self, run_id: str) -> EvalRunStatus | None:
        run = self._runs.get(run_id)
        if not run or run.status != "running":
            return None
        if run._task:
            run._task.cancel()
        run.status = "cancelled"
        run.message = "Cancelled by user"
        run.finished_at = datetime.now(timezone.utc)
        self._notify_progress(run)
        return run.to_status()

    def subscribe_progress(self, run_id: str, queue: asyncio.Queue):
        """Subscribe to progress updates for a run."""
        if run_id not in self._progress_callbacks:
            self._progress_callbacks[run_id] = []
        self._progress_callbacks[run_id].append(queue)

    def unsubscribe_progress(self, run_id: str, queue: asyncio.Queue):
        if run_id in self._progress_callbacks:
            try:
                self._progress_callbacks[run_id].remove(queue)
            except ValueError:
                pass

    def _notify_progress(self, run: EvalRun):
        """Send progress update to all subscribers."""
        queues = self._progress_callbacks.get(run.run_id, [])
        progress = run.to_progress()
        for q in queues:
            try:
                q.put_nowait(progress)
            except asyncio.QueueFull:
                pass

    async def _execute_run(self, run: EvalRun):
        """Execute evaluation in background thread."""
        import concurrent.futures

        run.status = "running"
        run.message = "Starting evaluation..."
        self._notify_progress(run)

        loop = asyncio.get_event_loop()

        try:
            # Run the blocking pipeline in a thread pool
            result = await loop.run_in_executor(
                None,
                self._run_pipeline_sync,
                run,
            )

            run.status = "completed"
            run.progress = 1.0
            auto_review_count = int(result.get("auto_review_count") or 0)
            queue_warning = result.get("queue_warning")
            if queue_warning:
                run.message = queue_warning
            elif auto_review_count > 0:
                run.message = f"Evaluation completed · {auto_review_count} review candidates queued"
            else:
                run.message = "Evaluation completed"
            run.report_path = result.get("report_path")
            run.finished_at = datetime.now(timezone.utc)

        except asyncio.CancelledError:
            run.status = "cancelled"
            run.message = "Cancelled by user"
            run.finished_at = datetime.now(timezone.utc)
            self._notify_progress(run)
            raise

        except EvalRunError as e:
            run.status = "failed"
            run.error = e.detail
            run.error_code = e.code
            run.error_stage = e.stage
            run.message = f"Failed during {e.stage}: {e.detail}"
            run.finished_at = datetime.now(timezone.utc)

        except Exception as e:
            run.status = "failed"
            run.error = str(e)
            run.error_code = "unexpected_failure"
            run.error_stage = "execution"
            run.message = f"Failed: {e}"
            run.finished_at = datetime.now(timezone.utc)

        self._notify_progress(run)

    def _run_pipeline_sync(self, run: EvalRun) -> dict[str, Any]:
        """Synchronous pipeline execution (runs in thread pool)."""
        from evaluate_api import evaluate
        from datetime import datetime as dt
        from api.services.custom_dataset_service import CustomDatasetService
        from utils.human_annotations import AnnotationManager, create_pending_from_results

        request = run.request
        settings = get_settings()

        # Generate output path
        timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
        output_path = (
            request.output_path.strip()
            if isinstance(request.output_path, str) and request.output_path.strip()
            else f"{settings.reports_dir}/eval_{timestamp}_{run.run_id[:8]}.json"
        )
        custom_dataset_path = None
        custom_dataset_name = None

        if request.custom_dataset_id:
            dataset_service = CustomDatasetService()
            custom_dataset = dataset_service.get_dataset(request.custom_dataset_id)
            if custom_dataset is None:
                raise EvalRunError(
                    "dataset_not_found",
                    "dataset_resolution",
                    f"Generated dataset not found: {request.custom_dataset_id}",
                )
            custom_dataset_path = custom_dataset.path
            custom_dataset_name = custom_dataset.title
            custom_dataset_kind = custom_dataset.dataset_kind
            run.current_test = custom_dataset.title
        else:
            custom_dataset_kind = None

        # Update progress for each model
        total_models = len(request.models)
        for idx, model in enumerate(request.models):
            run.current_model = model
            run.progress = idx / total_models
            if custom_dataset_name:
                run.message = f"Evaluating {model} on {custom_dataset_name} ({idx + 1}/{total_models})"
            else:
                run.message = f"Evaluating {model} ({idx + 1}/{total_models})"
            # Note: Can't await here since we're in sync context
            # Progress will be polled via WebSocket

        try:
            result = evaluate(
                models=request.models,
                suite=request.suite,
                tests=request.tests,
                config_path=settings.models_config_path,
                judge_model=request.judge_model,
                output_path=output_path,
                parallel=request.parallel,
                max_workers=request.max_workers,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                custom_dataset_path=custom_dataset_path,
                custom_dataset_name=custom_dataset_name,
                custom_dataset_kind=custom_dataset_kind,
            )
        except EvalRunError:
            raise
        except Exception as exc:
            raise EvalRunError(
                "evaluation_failed",
                "pipeline_execution",
                str(exc),
            ) from exc

        auto_review_count = 0
        queue_warning = None
        try:
            auto_review_count = create_pending_from_results(
                output_path,
                AnnotationManager(),
                sample_per_test=3,
                run_id=run.run_id,
                disagreement_only=True,
            )
        except Exception as exc:
            # Auto-queue should not fail the completed evaluation path.
            queue_warning = f"Evaluation completed, but review queue generation failed: {exc}"

        return {
            "report_path": output_path,
            "result": result,
            "auto_review_count": auto_review_count,
            "queue_warning": queue_warning,
        }
