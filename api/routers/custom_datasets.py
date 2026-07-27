"""Dataset Studio API."""
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from api.schemas.evaluations import (
    CustomDatasetCaseUpdateRequest,
    CustomDatasetDetail,
    CustomDatasetGenerateRequest,
    CustomDatasetImportRequest,
    CustomDatasetReviewStatusUpdateRequest,
    CustomDatasetSummary,
    WorkspaceSourceFile,
)
from api.services.config_service import ConfigService
from api.services.custom_dataset_service import CustomDatasetService

router = APIRouter(prefix="/custom-datasets", tags=["custom-datasets"])


def get_dataset_service() -> CustomDatasetService:
    return CustomDatasetService()


def get_config_service() -> ConfigService:
    return ConfigService()


@router.post(
    "/generate",
    response_model=CustomDatasetDetail,
    status_code=201,
    responses={
        400: {"description": "Unknown generator model"},
        422: {"description": "Dataset generation failed"},
    },
)
def generate_dataset(
    request: CustomDatasetGenerateRequest,
    svc: Annotated[CustomDatasetService, Depends(get_dataset_service)],
    config_svc: Annotated[ConfigService, Depends(get_config_service)],
):
    available_models = config_svc.get_models()
    if request.generator_model not in available_models:
        raise HTTPException(400, f"Unknown generator model: {request.generator_model}")

    try:
        return svc.generate_dataset(request)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post(
    "/import",
    response_model=CustomDatasetDetail,
    status_code=201,
    responses={422: {"description": "Dataset import failed"}},
)
def import_dataset(
    request: CustomDatasetImportRequest,
    svc: Annotated[CustomDatasetService, Depends(get_dataset_service)],
):
    try:
        return svc.import_dataset(request)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("", response_model=list[CustomDatasetSummary])
def list_datasets(
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    svc: Annotated[CustomDatasetService, Depends(get_dataset_service)] = None,
):
    return svc.list_datasets(limit=limit)


@router.get("/workspace-files", response_model=list[WorkspaceSourceFile])
def list_workspace_files(
    limit: Annotated[int, Query(ge=1, le=100)] = 30,
    svc: Annotated[CustomDatasetService, Depends(get_dataset_service)] = None,
):
    return svc.list_workspace_source_files(limit=limit)


@router.get(
    "/{dataset_id}",
    response_model=CustomDatasetDetail,
    responses={404: {"description": "Dataset not found"}},
)
def get_dataset(
    dataset_id: str,
    svc: Annotated[CustomDatasetService, Depends(get_dataset_service)],
):
    dataset = svc.get_dataset(dataset_id)
    if dataset is None:
        raise HTTPException(404, f"Dataset '{dataset_id}' not found")
    return dataset


@router.patch(
    "/{dataset_id}/cases/{case_id}",
    response_model=CustomDatasetDetail,
    responses={404: {"description": "Dataset or case not found"}, 422: {"description": "Invalid case edit"}},
)
def update_dataset_case(
    dataset_id: str,
    case_id: str,
    request: CustomDatasetCaseUpdateRequest,
    svc: Annotated[CustomDatasetService, Depends(get_dataset_service)],
):
    try:
        dataset = svc.update_case(
            dataset_id,
            case_id,
            question=request.question,
            persona=request.persona,
            expected_answer=request.expected_answer,
            expected_outcome=request.expected_outcome,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if dataset is None:
        raise HTTPException(404, f"Dataset '{dataset_id}' or case '{case_id}' not found")
    return dataset


@router.post(
    "/{dataset_id}/review-status",
    response_model=CustomDatasetDetail,
    responses={404: {"description": "Dataset not found"}, 422: {"description": "Invalid review status"}},
)
def update_dataset_review_status(
    dataset_id: str,
    request: CustomDatasetReviewStatusUpdateRequest,
    svc: Annotated[CustomDatasetService, Depends(get_dataset_service)],
):
    try:
        dataset = svc.update_review_status(
            dataset_id,
            request.review_status,
            request.reviewer_role,
            request.reusable_metric_candidate,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if dataset is None:
        raise HTTPException(404, f"Dataset '{dataset_id}' not found")
    return dataset


@router.post(
    "/{dataset_id}/promote-regression",
    response_model=CustomDatasetDetail,
    responses={404: {"description": "Dataset not found"}, 422: {"description": "Dataset cannot be promoted"}},
)
def promote_dataset_to_regression(
    dataset_id: str,
    svc: Annotated[CustomDatasetService, Depends(get_dataset_service)],
):
    try:
        dataset = svc.promote_to_regression(dataset_id)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if dataset is None:
        raise HTTPException(404, f"Dataset '{dataset_id}' not found")
    return dataset