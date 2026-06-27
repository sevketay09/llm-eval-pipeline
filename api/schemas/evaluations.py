"""Pydantic schemas for evaluation runs and results."""
from pydantic import BaseModel, Field
from typing import Optional, Any, List, Union
from datetime import datetime


class EvalRunRequest(BaseModel):
    models: list[str] = Field(..., min_length=1, description="Model keys to evaluate")
    suite: str = Field("smoke", description="Test suite name")
    judge_model: Optional[str] = Field(None, description="Judge model override")
    tests: Optional[list[str]] = Field(None, description="Subset of tests to run within the suite")
    output_path: Optional[str] = Field(None, description="Optional output path override for exported report artifacts")
    parallel: bool = False
    max_workers: Optional[int] = Field(None, ge=1, le=16)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(None, ge=256, le=32768)
    custom_dataset_id: Optional[str] = Field(None, description="Generated dataset identifier")


class EvalProgress(BaseModel):
    run_id: str
    status: str  # "running", "completed", "failed", "cancelled"
    progress: float = Field(0.0, ge=0.0, le=1.0)
    current_model: Optional[str] = None
    current_test: Optional[str] = None
    message: str = ""
    started_at: datetime
    elapsed_seconds: float = 0.0
    error_code: Optional[str] = None
    error_stage: Optional[str] = None


class EvalRunStatus(BaseModel):
    run_id: str
    status: str
    progress: float
    started_at: datetime
    finished_at: Optional[datetime] = None
    report_path: Optional[str] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    error_stage: Optional[str] = None


class ReportListItem(BaseModel):
    class ExportLinks(BaseModel):
        raw: str
        markdown: str
        html: str

    filename: str
    path: str
    modified: datetime
    size_kb: int
    model_count: Optional[int] = None
    suite: Optional[str] = None
    export_links: ExportLinks


class TokenEfficiencyPoint(BaseModel):
    model: str
    overall_score: float
    total_tokens: int
    input_tokens: int
    output_tokens: int
    total_requests: int
    total_items: int
    avg_tokens_per_eval: Optional[float] = None
    quality_points: Optional[float] = None
    quality_per_1k_tokens: Optional[float] = None
    tokens_per_quality_point: Optional[float] = None
    frontier: bool = False


class EvaluatorEfficiencyRow(BaseModel):
    provider: str
    metric_count: int = 0
    case_count: int = 0
    model_count: int = 0
    avg_score: Optional[float] = None
    success_rate: Optional[float] = None
    observed_cost: Optional[float] = None
    observed_tokens: int = 0
    cost_per_1k_tokens: Optional[float] = None
    metric_share: float = 0.0


class TokenEfficiencySummary(BaseModel):
    leaderboard: list[TokenEfficiencyPoint] = Field(default_factory=list)
    frontier_models: list[str] = Field(default_factory=list)
    best_quality_yield_model: Optional[str] = None
    leanest_model: Optional[str] = None
    strongest_frontier_model: Optional[str] = None
    evaluator_breakdown: list[EvaluatorEfficiencyRow] = Field(default_factory=list)


class JudgeDisagreementCase(BaseModel):
    model: str
    test_name: str
    test_id: str
    question: str
    llm_judge_score: float
    llm_judge_label: str
    primary_judge_score: Optional[float] = None
    primary_judge_label: Optional[str] = None
    secondary_judge_score: Optional[float] = None
    secondary_judge_label: Optional[str] = None
    judge_disagreement: float
    judge_agreement: Optional[float] = None
    review_priority: float = 0.0
    queue_reason: str = ""


class JudgeDisagreementModelSummary(BaseModel):
    model: str
    panel_case_count: int = 0
    high_disagreement_count: int = 0
    mean_disagreement: float = 0.0


class JudgeDisagreementSummary(BaseModel):
    total_panel_cases: int = 0
    high_disagreement_cases: int = 0
    mean_disagreement: Optional[float] = None
    max_disagreement: Optional[float] = None
    strongest_split_model: Optional[str] = None
    recommended_queue_size: int = 0
    by_model: list[JudgeDisagreementModelSummary] = Field(default_factory=list)
    top_cases: list[JudgeDisagreementCase] = Field(default_factory=list)


class ContinuityModelSummary(BaseModel):
    model: str
    intent_resolution: Optional[float] = None
    unresolved_turn_rate: Optional[float] = None
    unresolved_turns: int = 0
    unresolved_intent_total: int = 0


class ContinuitySummary(BaseModel):
    by_model: list[ContinuityModelSummary] = Field(default_factory=list)
    best_intent_resolution_model: Optional[str] = None
    highest_unresolved_rate_model: Optional[str] = None


class PolicySummaryByType(BaseModel):
    policy_type: str
    total_cases: int = 0
    flagged_cases: int = 0
    high_severity_cases: int = 0
    avg_severity: Optional[float] = None


class PolicySummaryCase(BaseModel):
    model: str
    test_name: str
    test_id: str
    question: str = ""
    policy_type: str
    risk_level: str = "low"
    severity: Optional[float] = None
    flagged: bool = False
    queue_reason: str = ""
    violation_detected: bool = False
    queue_candidate: bool = False


class PolicySummary(BaseModel):
    total_policy_cases: int = 0
    flagged_case_count: int = 0
    high_severity_case_count: int = 0
    avg_severity: Optional[float] = None
    max_severity: Optional[float] = None
    queue_candidate_count: int = 0
    risk_level_counts: dict[str, int] = Field(default_factory=dict)
    by_policy_type: list[PolicySummaryByType] = Field(default_factory=list)
    top_cases: list[PolicySummaryCase] = Field(default_factory=list)


class PolicyAuditReview(BaseModel):
    annotation_id: str
    model: str = ""
    test_name: str = ""
    test_id: str = ""
    question: str = ""
    policy_type: str = "policy_safety"
    decision: str = "confirmed_violation"
    notes: str = ""
    annotator_id: str = ""
    timestamp: Optional[str] = None
    queue_reason: str = ""
    review_priority: float = 0.0
    risk_tags: list[str] = Field(default_factory=list)


class PolicyAuditSummary(BaseModel):
    total_reviews: int = 0
    confirmed_violation_count: int = 0
    false_positive_count: int = 0
    needs_follow_up_count: int = 0
    latest_review_at: Optional[str] = None
    recent_reviews: list[PolicyAuditReview] = Field(default_factory=list)


class ReportCompareEntry(BaseModel):
    model_scores: dict[str, float] = Field(default_factory=dict)
    continuity: ContinuitySummary = Field(default_factory=ContinuitySummary)
    model_comparison: dict[str, Any] = Field(default_factory=dict)
    prompt_version: Optional[str] = None
    schema_version: Optional[str] = None
    metric_version: Optional[str] = None


class GeneratedDatasetCase(BaseModel):
    id: str
    category: str = "custom"
    difficulty: Optional[str] = None
    question: str
    expected_answer: str
    system_prompt: Optional[str] = None
    source_case_id: Optional[str] = None
    mutation_type: Optional[str] = None
    variant_label: Optional[str] = None
    risk_tags: list[str] = Field(default_factory=list)
    mutation_metadata: dict[str, Any] = Field(default_factory=dict)


class GeneratedConversationTurn(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    expected_actions: list[str] = Field(default_factory=list)
    check: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GeneratedConversationCase(BaseModel):
    id: str
    category: str = "conversation"
    difficulty: Optional[str] = None
    persona: Optional[str] = None
    template_id: Optional[str] = None
    variation_type: Optional[str] = None
    expected_outcome: str
    escalation_needed: bool = False
    turn_count: int = 0
    source_case_id: Optional[str] = None
    risk_tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    turns: list[GeneratedConversationTurn] = Field(default_factory=list)


class ConversationCoverageSummary(BaseModel):
    total_conversations: int = 0
    escalation_count: int = 0
    average_user_turns: float = 0.0
    template_counts: dict[str, int] = Field(default_factory=dict)
    variation_counts: dict[str, int] = Field(default_factory=dict)


class FinalizedDiffSummary(BaseModel):
    current_case_count: int = 0
    finalized_case_count: int = 0
    added_count: int = 0
    removed_count: int = 0
    changed_count: int = 0
    unchanged_count: int = 0


class CustomDatasetGenerateRequest(BaseModel):
    title: Optional[str] = Field(None, max_length=120)
    project_description: str = Field(..., min_length=40, max_length=12000)
    sample_count: int = Field(12, ge=3, le=100)
    generator_model: str = Field(..., min_length=1)
    focus_areas: Optional[str] = Field(None, max_length=400)
    dataset_kind: str = Field("single_turn", max_length=40)
    generation_mode: str = Field("generate_from_scratch", max_length=80)
    source_label: Optional[str] = Field(None, max_length=240)
    source_material: Optional[str] = Field(None, max_length=60000)
    source_paths: list[str] = Field(default_factory=list, max_length=50)


class CustomDatasetImportRequest(BaseModel):
    dataset_json: str = Field(..., min_length=2, max_length=2_000_000)
    title: Optional[str] = Field(None, max_length=120)
    project_description: str = Field("Imported dataset", max_length=12000)
    focus_areas: Optional[str] = Field(None, max_length=400)
    source_label: Optional[str] = Field(None, max_length=240)


class CustomDatasetReviewStatusUpdateRequest(BaseModel):
    review_status: str = Field(..., min_length=4, max_length=20)
    reviewer_role: Optional[str] = Field(None, min_length=2, max_length=20)
    reusable_metric_candidate: Optional[bool] = None


class CustomDatasetCaseUpdateRequest(BaseModel):
    question: Optional[str] = Field(None, min_length=1, max_length=12000)
    persona: Optional[str] = Field(None, min_length=1, max_length=240)
    expected_answer: Optional[str] = Field(None, min_length=1, max_length=12000)
    expected_outcome: Optional[str] = Field(None, min_length=1, max_length=12000)


class CustomDatasetSummary(BaseModel):
    dataset_id: str
    title: str
    generator_model: str
    source_type: str = "generated"
    source_label: Optional[str] = None
    dataset_kind: str = "single_turn"
    generation_mode: str = "generate_from_scratch"
    source_attribution: dict[str, Any] = Field(default_factory=dict)
    sample_count: int
    base_case_count: int = 0
    created_at: datetime
    path: str
    mutation_summary: dict[str, int] = Field(default_factory=dict)
    conversation_summary: Optional[ConversationCoverageSummary] = None
    dataset_tags: list[str] = Field(default_factory=list)
    dataset_tag_summary: dict[str, int] = Field(default_factory=dict)
    review_status: str = "draft"
    review_role: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    reusable_metric_candidate: bool = False
    finalized_diff_summary: Optional[FinalizedDiffSummary] = None
    finalized_at: Optional[datetime] = None
    finalized_path: Optional[str] = None
    finalized_case_count: int = 0
    promoted_to_regression_at: Optional[datetime] = None
    regression_dataset_path: Optional[str] = None


class CustomDatasetDetail(CustomDatasetSummary):
    focus_areas: Optional[str] = None
    project_description: str
    preview: List[Union[GeneratedConversationCase, GeneratedDatasetCase]] = Field(default_factory=list)


class WorkspaceSourceFile(BaseModel):
    path: str
    size_kb: float


class ReportSummary(BaseModel):
    filename: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    models: dict[str, Any] = Field(default_factory=dict)
    model_scores: dict[str, float] = Field(default_factory=dict)
    model_comparison: dict[str, Any] = Field(default_factory=dict)
    trends: dict[str, Any] = Field(default_factory=dict)
    continuity: ContinuitySummary = Field(default_factory=ContinuitySummary)
    efficiency: TokenEfficiencySummary = Field(default_factory=TokenEfficiencySummary)
    disagreement: JudgeDisagreementSummary = Field(default_factory=JudgeDisagreementSummary)
    policy: PolicySummary = Field(default_factory=PolicySummary)
    policy_audit: PolicyAuditSummary = Field(default_factory=PolicyAuditSummary)
    statistical_comparison: dict[str, Any] = Field(default_factory=dict)
