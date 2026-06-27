"""HITL API — pending items, submit annotations, stats, generate."""
from typing import Annotated, Optional

from fastapi import APIRouter, HTTPException, Query

from api.schemas.hitl import (
    PendingItem,
    SubmitAnnotationRequest,
    PendingItemUpdateRequest,
    BatchPendingItemUpdateRequest,
    BatchPendingItemUpdateResponse,
    AnnotationResponse,
    MetricBacklogItem,
    CalibrationInsights,
    CalibrationSampleSetResponse,
    DisagreementExportResponse,
    HitlStats,
    GeneratePendingRequest,
    ExportTrainingResponse,
)
from evaluators.human_feedback_eval import HumanFeedbackEvaluator
from utils.human_annotations import (
    AnnotationManager,
    HumanAnnotation,
    create_pending_from_results,
)
from api.config import get_settings
from datetime import datetime
from pathlib import Path

router = APIRouter(prefix="/hitl", tags=["hitl"])

_manager = AnnotationManager()
_evaluator = HumanFeedbackEvaluator(annotation_manager=_manager)


@router.get("/pending", response_model=list[PendingItem])
async def get_pending_items(
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    category: Optional[str] = None,
    source_report: Optional[str] = None,
    owner: Optional[str] = None,
    status: Optional[str] = None,
):
    """Get items pending human review."""
    items = _manager.get_pending_items(
        limit=limit,
        test_category=category,
        source_report=source_report,
        owner=owner,
        status=status,
    )
    return items


@router.patch(
    "/pending/batch",
    response_model=BatchPendingItemUpdateResponse,
)
async def batch_update_pending_items(req: BatchPendingItemUpdateRequest):
    """Update assignment or workflow status for multiple pending queue items."""
    items, missing_item_ids = _manager.update_pending_items(
        req.item_ids,
        owner=req.owner,
        status=req.status,
    )
    return BatchPendingItemUpdateResponse(
        updated_count=len(items),
        items=items,
        missing_item_ids=missing_item_ids,
    )


@router.patch(
    "/pending/{item_id}",
    response_model=PendingItem,
    responses={404: {"description": "Pending item not found"}},
)
async def update_pending_item(item_id: str, req: PendingItemUpdateRequest):
    """Update assignment or workflow status for a pending queue item."""
    item = _manager.update_pending_item(
        item_id,
        owner=req.owner,
        status=req.status,
    )
    if item is None:
        raise HTTPException(404, f"Pending item not found: {item_id}")
    return item


@router.post(
    "/annotate",
    response_model=AnnotationResponse,
    responses={404: {"description": "Pending item not found"}},
)
async def submit_annotation(req: SubmitAnnotationRequest):
    """Submit a human annotation for a pending item."""
    # Find the pending item
    pending = _manager.get_pending_items()
    target = next((i for i in pending if i["item_id"] == req.item_id), None)

    if target is None:
        raise HTTPException(404, f"Pending item not found: {req.item_id}")

    annotation_metadata = dict(target.get("metadata") or {})
    annotation_metadata["reusable_metric_candidate"] = bool(req.reusable_metric_candidate)
    if req.reusable_metric_candidate:
        annotation_metadata["metric_candidate_source"] = "hitl_review"
        annotation_metadata["metric_candidate_queue_reason"] = target.get("queue_reason")
    if req.policy_decision:
        annotation_metadata["policy_review"] = {
            "decision": req.policy_decision,
            "notes": req.policy_notes,
            "queue_reason": target.get("queue_reason") or "",
            "review_priority": target.get("review_priority") or 0.0,
            "risk_tags": annotation_metadata.get("risk_tags") or [],
            "source": "hitl_review",
        }

    annotation = HumanAnnotation(
        annotation_id=_manager.generate_annotation_id(
            target["test_id"], target["model_name"]
        ),
        test_id=target["test_id"],
        test_category=target["test_category"],
        model_name=target["model_name"],
        question=target["question"],
        model_response=target["model_response"],
        llm_judge_score=target["llm_judge_score"],
        llm_judge_reasoning=target.get("llm_judge_reasoning", ""),
        human_score=req.human_score,
        human_feedback=req.human_feedback,
        correction_type=req.correction_type,
        verdict={
            "label": req.correction_type,
            "resolution": {
                "approve": "accepted",
                "adjust": "corrected",
                "reject": "rejected",
            }[req.correction_type],
            "requires_follow_up": req.correction_type != "approve",
        },
        annotator_id=req.annotator_id,
        timestamp=datetime.now().isoformat(),
        metadata=annotation_metadata,
    )

    _manager.save_annotation(annotation, status="completed")
    _manager.remove_pending_item(req.item_id)

    # Apply annotation back to source report
    settings = get_settings()
    _manager.apply_annotation_to_report(annotation, reports_dir=settings.reports_dir)

    return AnnotationResponse(
        annotation_id=annotation.annotation_id,
        item_id=req.item_id,
        verdict=annotation.verdict,
        policy_review=dict(annotation_metadata.get("policy_review") or {}),
    )


@router.get("/stats", response_model=HitlStats)
async def get_stats():
    """Get HITL annotation statistics."""
    stats = _manager.get_statistics()
    return HitlStats(**stats)


@router.get("/metric-backlog", response_model=list[MetricBacklogItem])
async def get_metric_backlog(
    limit: Annotated[int, Query(ge=1, le=200)] = 20,
):
    """List review-derived metric backlog candidates."""
    return _manager.list_metric_backlog(limit=limit)


@router.get("/calibration", response_model=CalibrationInsights)
async def get_calibration_insights():
    """Get judge calibration metrics and recommendations from reviewed annotations."""
    insights = _evaluator.get_calibration_insights()
    return CalibrationInsights(**insights)


@router.get("/calibration-samples", response_model=CalibrationSampleSetResponse)
async def get_calibration_sample_set(
    sample_size: Annotated[int, Query(ge=3, le=24)] = 12,
):
    """Get a reusable reviewed sample set for judge calibration work."""
    sample_set = _evaluator.build_calibration_sample_set(sample_size=sample_size)
    return CalibrationSampleSetResponse(**sample_set)


@router.post("/export-disagreements", response_model=DisagreementExportResponse)
async def export_disagreement_cases(
    threshold: Annotated[float, Query(ge=0.0, le=1.0)] = 0.3,
):
    """Export high-disagreement judge-vs-human cases for offline review."""
    disagreements = _evaluator.get_disagreement_cases(threshold=threshold)
    output_path = _evaluator.export_disagreement_cases(threshold=threshold)
    return DisagreementExportResponse(
        output_file=output_path,
        threshold=threshold,
        exported_count=len(disagreements),
        reason_taxonomy=_evaluator.summarize_disagreement_taxonomy(disagreements=disagreements),
    )


@router.post(
    "/generate",
    status_code=201,
    responses={
        400: {"description": "Invalid report path"},
        404: {"description": "Report not found"},
    },
)
async def generate_pending(req: GeneratePendingRequest):
    """Generate pending review items from an evaluation report."""
    settings = get_settings()
    report_path = Path(settings.reports_dir) / req.report_filename

    if not report_path.exists():
        raise HTTPException(404, f"Report not found: {req.report_filename}")

    # Security: path traversal check
    if not report_path.resolve().is_relative_to(Path(settings.reports_dir).resolve()):
        raise HTTPException(400, "Invalid report path")

    added = create_pending_from_results(
        str(report_path),
        _manager,
        sample_per_test=req.sample_per_test,
        run_id=req.run_id,
    )

    return {"added_count": added, "report": req.report_filename}


@router.post("/export-training", response_model=ExportTrainingResponse)
async def export_training_data(
    min_agreement: Annotated[float, Query(ge=0.0, le=1.0)] = 0.2,
):
    """Export completed annotations as training data for LLM judge fine-tuning."""
    eligible = _manager.count_training_ready_examples(min_agreement)
    output_path = _manager.export_for_training(min_agreement_threshold=min_agreement)
    return {
        "output_file": output_path,
        "exported_count": eligible,
        "eligible_annotations": eligible,
        "min_agreement": min_agreement,
    }
