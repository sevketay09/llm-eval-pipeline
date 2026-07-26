"""Evaluations API — run, list, cancel."""
from typing import Annotated

from fastapi import APIRouter, HTTPException, Depends

from api.rate_limit import RateLimiter
from api.schemas.evaluations import EvalRunRequest, EvalRunStatus
from api.services.custom_dataset_service import CustomDatasetService
from api.services.eval_service import EvalService

router = APIRouter(prefix="/evaluations", tags=["evaluations"])

_run_rate_limit = RateLimiter("eval_run", limit=10, window_seconds=60)

# Singleton service (survives across requests)
_eval_service = EvalService()


def get_eval_service() -> EvalService:
    return _eval_service


def _raise_eval_error(status_code: int, code: str, detail: str, stage: str = "request_validation") -> None:
    raise HTTPException(
        status_code,
        detail={
            "message": detail,
            "error_code": code,
            "error_stage": stage,
        },
    )


def _validate_requested_tests(request: EvalRunRequest, suites: dict) -> None:
    if request.custom_dataset_id and request.tests:
        _raise_eval_error(
            400,
            "dataset_suite_conflict",
            "Custom datasets cannot be combined with suite test selection",
        )

    if not request.tests:
        return

    suite_tests = []
    suite_config = suites.get(request.suite, {}) if isinstance(suites, dict) else {}
    if isinstance(suite_config, dict):
        suite_tests = suite_config.get("tests", []) or []

    invalid = [test_name for test_name in request.tests if test_name not in suite_tests]
    if invalid:
        _raise_eval_error(
            400,
            "invalid_suite_tests",
            f"Tests {invalid} are not part of suite '{request.suite}'. Available: {suite_tests}",
        )


@router.post(
    "/run",
    response_model=EvalRunStatus,
    status_code=202,
    responses={400: {"description": "Invalid evaluation request"}},
    dependencies=[Depends(_run_rate_limit)],
)
async def start_evaluation(
    request: EvalRunRequest,
    svc: Annotated[EvalService, Depends(get_eval_service)],
):
    # Validate models exist
    from api.services.config_service import ConfigService
    config_svc = ConfigService()
    available_models = config_svc.get_models()
    missing = [m for m in request.models if m not in available_models]
    if missing:
        _raise_eval_error(400, "unknown_models", f"Unknown models: {missing}")

    # Validate suite exists
    suites = config_svc.get_test_suites()
    if suites and request.suite not in suites:
        _raise_eval_error(
            400,
            "unknown_suite",
            f"Unknown suite '{request.suite}'. Available: {list(suites.keys())}",
        )

    _validate_requested_tests(request, suites)

    if request.custom_dataset_id:
        dataset = CustomDatasetService().get_dataset(request.custom_dataset_id)
        if dataset is None:
            _raise_eval_error(
                400,
                "unknown_generated_dataset",
                f"Unknown generated dataset '{request.custom_dataset_id}'",
            )

    run = svc.start_run(request)
    return run.to_status()


@router.get("/runs", response_model=list[EvalRunStatus])
def list_runs(
    svc: Annotated[EvalService, Depends(get_eval_service)],
    limit: int = 20,
):
    return svc.list_runs(limit)


@router.get("/runs/{run_id}", response_model=EvalRunStatus, responses={404: {"description": "Run not found"}})
def get_run(run_id: str, svc: Annotated[EvalService, Depends(get_eval_service)]):
    run = svc.get_run(run_id)
    if not run:
        _raise_eval_error(404, "run_not_found", f"Run '{run_id}' not found", stage="run_lookup")
    return run.to_status()


@router.post(
    "/runs/{run_id}/cancel",
    response_model=EvalRunStatus,
    responses={400: {"description": "Run cannot be cancelled"}},
)
def cancel_run(run_id: str, svc: Annotated[EvalService, Depends(get_eval_service)]):
    run_status = svc.cancel_run(run_id)
    if not run_status:
        _raise_eval_error(
            400,
            "run_not_cancellable",
            f"Cannot cancel run '{run_id}' (not running or not found)",
            stage="run_lifecycle",
        )
    return run_status


@router.get("/suites")
def list_suites():
    from api.services.config_service import ConfigService
    svc = ConfigService()
    suites = svc.get_test_suites()
    detail = {
        name: suite.get("tests", []) if isinstance(suite, dict) else []
        for name, suite in suites.items()
    }
    return {"suites": list(suites.keys()), "detail": detail, "total": len(suites)}
