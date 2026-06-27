"""Pydantic models for HITL (Human-in-the-Loop) review system."""
from pydantic import BaseModel, Field
from typing import Optional


class PendingItem(BaseModel):
    """A single item awaiting human review."""
    item_id: str
    model_name: str
    test_category: str
    test_id: str
    question: str
    model_response: str
    llm_judge_score: float
    llm_judge_label: str
    llm_judge_reasoning: str
    primary_judge_score: Optional[float] = None
    primary_judge_label: Optional[str] = None
    secondary_judge_score: Optional[float] = None
    secondary_judge_label: Optional[str] = None
    secondary_judge_reasoning: Optional[str] = None
    judge_disagreement: Optional[float] = None
    judge_agreement: Optional[float] = None
    review_priority: float = 0.0
    queue_reason: str = ""
    owner: Optional[str] = None
    status: str = "pending"
    sla_due_at: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class SubmitAnnotationRequest(BaseModel):
    """Human annotation submission."""
    item_id: str
    human_score: float = Field(ge=0.0, le=1.0)
    human_feedback: str = ""
    correction_type: str = Field(pattern=r"^(approve|adjust|reject)$")
    annotator_id: str = "default"
    reusable_metric_candidate: bool = False
    policy_decision: Optional[str] = Field(
        default=None,
        pattern=r"^(confirmed_violation|false_positive|needs_follow_up)$",
    )
    policy_notes: str = ""


class AnnotationVerdict(BaseModel):
    """Standardized verdict schema for a human review decision."""
    label: str = Field(pattern=r"^(approve|adjust|reject)$")
    resolution: str = Field(pattern=r"^(accepted|corrected|rejected)$")
    requires_follow_up: bool = False


class PendingItemUpdateRequest(BaseModel):
    """Assignment or status update for a pending queue item."""
    owner: Optional[str] = None
    status: str = Field(pattern=r"^(pending|in_progress|completed)$")


class BatchPendingItemUpdateRequest(BaseModel):
    """Batch assignment or status update for pending queue items."""
    item_ids: list[str] = Field(min_length=1)
    owner: Optional[str] = None
    status: str = Field(pattern=r"^(pending|in_progress|completed)$")


class BatchPendingItemUpdateResponse(BaseModel):
    """Result of a batch pending-item update."""
    updated_count: int = 0
    items: list[PendingItem] = Field(default_factory=list)
    missing_item_ids: list[str] = Field(default_factory=list)


class AnnotationResponse(BaseModel):
    """Response after annotation is saved."""
    annotation_id: str
    item_id: str
    status: str = "saved"
    verdict: AnnotationVerdict
    policy_review: dict = Field(default_factory=dict)


class MetricBacklogItem(BaseModel):
    """A review-derived candidate for future metric work."""
    entry_id: str
    annotation_id: str
    created_at: str
    status: str = "open"
    source: str = "hitl_review"
    source_report: Optional[str] = None
    model_name: str
    test_category: str
    test_id: str
    question: str
    queue_reason: str = ""
    human_feedback: str = ""
    correction_type: str
    verdict: dict = Field(default_factory=dict)
    llm_judge_score: float
    human_score: float
    score_delta: float


class CalibrationRecommendation(BaseModel):
    """Calibration recommendation generated from human-vs-judge comparisons."""
    issue: str
    recommendation: str


class CalibrationInsights(BaseModel):
    """Calibration summary for the current human annotation corpus."""
    overall_metrics: dict = Field(default_factory=dict)
    recommendations: list[CalibrationRecommendation] = Field(default_factory=list)
    disagreement_taxonomy: dict = Field(default_factory=dict)
    prompt_version_comparison: dict = Field(default_factory=dict)
    training_data_available: int = 0
    ready_for_finetuning: bool = False


class DisagreementExportResponse(BaseModel):
    """Response after exporting disagreement cases."""
    output_file: str
    threshold: float
    exported_count: int = 0
    reason_taxonomy: dict = Field(default_factory=dict)


class CalibrationSampleCase(BaseModel):
    """A reviewed case selected for calibration analysis."""
    bucket: str
    test_id: str
    test_category: str
    model_name: str
    llm_score: float
    human_score: float
    score_difference: float
    correction_type: str
    question: str
    human_feedback: str = ""


class CalibrationSampleSetResponse(BaseModel):
    """A reusable sample set for judge calibration review."""
    total_samples: int = 0
    bucket_counts: dict = Field(default_factory=dict)
    samples: list[CalibrationSampleCase] = Field(default_factory=list)


class HitlStats(BaseModel):
    """Aggregated HITL statistics."""
    total_completed: int = 0
    total_pending: int = 0
    average_agreement: float = 0.0
    panel_review_pending: int = 0
    high_priority_pending: int = 0
    training_ready_examples: int = 0
    metric_candidate_annotations: int = 0
    corrections_by_type: dict = Field(default_factory=dict)
    by_category: dict = Field(default_factory=dict)
    annotators: list[str] = Field(default_factory=list)


class GeneratePendingRequest(BaseModel):
    """Request to generate pending items from a report."""
    report_filename: str
    sample_per_test: int = Field(default=5, ge=1, le=50)
    run_id: Optional[str] = None


class ExportTrainingResponse(BaseModel):
    output_file: str
    exported_count: int = 0
    eligible_annotations: int = 0
    min_agreement: float = 0.2
