const BASE = "/api";

type ErrorPayload = {
  detail?: string | {
    message?: string;
    detail?: string;
    error_code?: string;
    error_stage?: string;
  };
  error_code?: string;
  error_stage?: string;
  message?: string;
};

export class ApiError extends Error {
  errorCode?: string;
  errorStage?: string;

  constructor(message: string, options?: { errorCode?: string; errorStage?: string }) {
    super(message);
    this.name = "ApiError";
    this.errorCode = options?.errorCode;
    this.errorStage = options?.errorStage;
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText })) as ErrorPayload;
    const nestedDetail = typeof err.detail === "object" && err.detail !== null ? err.detail : undefined;
    const message = typeof err.detail === "string"
      ? err.detail
      : nestedDetail?.message || nestedDetail?.detail || err.message || `HTTP ${res.status}`;
    throw new ApiError(message, {
      errorCode: err.error_code || nestedDetail?.error_code,
      errorStage: err.error_stage || nestedDetail?.error_stage,
    });
  }
  return res.json();
}

// ─── Models ──────────────────────────────────────────────────────────────────

export interface ModelConfig {
  provider: string;
  model_name: string;
  api_key?: string;
  base_url?: string;
  api_version?: string;
  max_tokens: number;
  temperature: number;
  supports_function_calling: boolean;
  supports_streaming: boolean;
  supports_response_format?: boolean;
  quirks?: string[];
}

export interface ModelListResponse {
  models: Record<string, ModelConfig>;
  total: number;
}

export const modelsApi = {
  list: () => request<ModelListResponse>("/models"),
  get: (id: string) => request<ModelConfig & { id: string }>(`/models/${id}`),
  create: (id: string, config: Partial<ModelConfig>) =>
    request(`/models/${id}`, { method: "POST", body: JSON.stringify(config) }),
  update: (id: string, config: Partial<ModelConfig>) =>
    request(`/models/${id}`, { method: "PUT", body: JSON.stringify(config) }),
  delete: (id: string) => request(`/models/${id}`, { method: "DELETE" }),
  import: (models: Record<string, Partial<ModelConfig>>, overwrite = false) =>
    request(`/models/import?overwrite=${overwrite}`, {
      method: "POST",
      body: JSON.stringify(models),
    }),
  exportYaml: () =>
    fetch(`${BASE}/models/export/yaml`).then((r) => r.text()),
};

// ─── Evaluations ─────────────────────────────────────────────────────────────

export interface EvalRunRequest {
  models: string[];
  suite: string;
  judge_model?: string;
  tests?: string[];
  output_path?: string;
  parallel?: boolean;
  max_workers?: number;
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  custom_dataset_id?: string;
}

export interface EvalRunStatus {
  run_id: string;
  status: string;
  progress: number;
  started_at: string;
  finished_at?: string;
  report_path?: string;
  error?: string;
  error_code?: string;
  error_stage?: string;
}

export const evaluationsApi = {
  run: (req: EvalRunRequest) =>
    request<EvalRunStatus>("/evaluations/run", {
      method: "POST",
      body: JSON.stringify(req),
    }),
  listRuns: (limit = 20) =>
    request<EvalRunStatus[]>(`/evaluations/runs?limit=${limit}`),
  getRun: (id: string) => request<EvalRunStatus>(`/evaluations/runs/${id}`),
  cancel: (id: string) =>
    request<EvalRunStatus>(`/evaluations/runs/${id}/cancel`, { method: "POST" }),
  listSuites: () =>
    request<{ suites: string[]; detail: Record<string, string[]>; total: number }>("/evaluations/suites"),
};

// ─── Results ─────────────────────────────────────────────────────────────────

export interface ReportListItem {
  export_links: {
    raw: string;
    markdown: string;
    html: string;
  };
  filename: string;
  path: string;
  modified: string;
  size_kb: number;
  model_count?: number;
  suite?: string;
}

export interface StatisticalPerModel {
  weighted_score: number | null;
  mean_test_score: number;
  ci_lower: number;
  ci_upper: number;
  bootstrap_std: number;
  n_tests: number;
  small_sample: boolean;
}

export interface StatisticalPairwise {
  model_a: string;
  model_b: string;
  n_shared_tests: number;
  shared_tests: string[];
  mean_a: number;
  mean_b: number;
  small_sample: boolean;
  is_significant: boolean;
  p_value: number;
  wilcoxon_p_value: number;
  mean_difference: number;
  cohens_dz: number | null;
  effect_size: "negligible" | "small" | "medium" | "large";
  winner: string | null;
  verdict: string;
}

export interface StatisticalComparison {
  alpha: number;
  confidence: number;
  seed: number;
  per_model: Record<string, StatisticalPerModel>;
  pairwise: StatisticalPairwise[];
  warnings: string[];
}

export interface ReportSummary {
  filename: string;
  metadata: Record<string, unknown>;
  models: Record<string, unknown>;
  model_scores: Record<string, number>;
  model_comparison?: Record<string, unknown>;
  trends: Record<string, unknown>;
  continuity: Record<string, unknown>;
  efficiency: TokenEfficiencySummary;
  disagreement: JudgeDisagreementSummary;
  policy: PolicySummary;
  policy_audit: PolicyAuditSummary;
  statistical_comparison?: StatisticalComparison;
}

export interface ReportCompareSummary {
  model_scores: Record<string, number>;
  continuity: Record<string, unknown>;
  model_comparison?: Record<string, unknown>;
  prompt_version?: string | null;
  schema_version?: string | null;
  metric_version?: string | null;
}

export interface JudgeDisagreementCase {
  model: string;
  test_name: string;
  test_id: string;
  question: string;
  llm_judge_score: number;
  llm_judge_label: string;
  primary_judge_score?: number | null;
  primary_judge_label?: string | null;
  secondary_judge_score?: number | null;
  secondary_judge_label?: string | null;
  judge_disagreement: number;
  judge_agreement?: number | null;
  review_priority: number;
  queue_reason: string;
}

export interface PolicySummaryByType {
  policy_type: string;
  total_cases: number;
  flagged_cases: number;
  high_severity_cases: number;
  avg_severity?: number | null;
}

export interface PolicySummaryCase {
  model: string;
  test_name: string;
  test_id: string;
  question: string;
  policy_type: string;
  risk_level: string;
  severity?: number | null;
  flagged: boolean;
  queue_reason: string;
  violation_detected: boolean;
  queue_candidate: boolean;
}

export interface PolicySummary {
  total_policy_cases: number;
  flagged_case_count: number;
  high_severity_case_count: number;
  avg_severity?: number | null;
  max_severity?: number | null;
  queue_candidate_count: number;
  risk_level_counts: Record<string, number>;
  by_policy_type: PolicySummaryByType[];
  top_cases: PolicySummaryCase[];
}

export interface PolicyAuditReview {
  annotation_id: string;
  model: string;
  test_name: string;
  test_id: string;
  question: string;
  policy_type: string;
  decision: "confirmed_violation" | "false_positive" | "needs_follow_up";
  notes: string;
  annotator_id: string;
  timestamp?: string | null;
  queue_reason: string;
  review_priority: number;
  risk_tags: string[];
}

export interface PolicyAuditSummary {
  total_reviews: number;
  confirmed_violation_count: number;
  false_positive_count: number;
  needs_follow_up_count: number;
  latest_review_at?: string | null;
  recent_reviews: PolicyAuditReview[];
}

export interface JudgeDisagreementModelSummary {
  model: string;
  panel_case_count: number;
  high_disagreement_count: number;
  mean_disagreement: number;
}

export interface JudgeDisagreementSummary {
  total_panel_cases: number;
  high_disagreement_cases: number;
  mean_disagreement?: number | null;
  max_disagreement?: number | null;
  strongest_split_model?: string | null;
  recommended_queue_size: number;
  by_model: JudgeDisagreementModelSummary[];
  top_cases: JudgeDisagreementCase[];
}

export interface TokenEfficiencyPoint {
  model: string;
  overall_score: number;
  total_tokens: number;
  input_tokens: number;
  output_tokens: number;
  total_requests: number;
  total_items: number;
  avg_tokens_per_eval?: number | null;
  quality_points?: number | null;
  quality_per_1k_tokens?: number | null;
  tokens_per_quality_point?: number | null;
  frontier: boolean;
}

export interface EvaluatorEfficiencyRow {
  provider: string;
  metric_count: number;
  case_count: number;
  model_count: number;
  avg_score?: number | null;
  success_rate?: number | null;
  observed_cost?: number | null;
  observed_tokens: number;
  cost_per_1k_tokens?: number | null;
  metric_share: number;
}

export interface TokenEfficiencySummary {
  leaderboard: TokenEfficiencyPoint[];
  frontier_models: string[];
  best_quality_yield_model?: string | null;
  leanest_model?: string | null;
  strongest_frontier_model?: string | null;
  evaluator_breakdown: EvaluatorEfficiencyRow[];
}

export interface GeneratedDatasetCase {
  id: string;
  category: string;
  difficulty?: string | null;
  question: string;
  expected_answer: string;
  system_prompt?: string | null;
  source_case_id?: string | null;
  mutation_type?: string | null;
  variant_label?: string | null;
  risk_tags: string[];
  mutation_metadata: Record<string, unknown>;
}

export interface GeneratedConversationTurn {
  role?: string | null;
  content?: string | null;
  expected_actions: string[];
  check?: string | null;
  metadata: Record<string, unknown>;
}

export interface GeneratedConversationCase {
  id: string;
  category: string;
  difficulty?: string | null;
  persona?: string | null;
  template_id?: string | null;
  variation_type?: string | null;
  expected_outcome: string;
  escalation_needed: boolean;
  turn_count: number;
  source_case_id?: string | null;
  risk_tags: string[];
  metadata: Record<string, unknown>;
  turns: GeneratedConversationTurn[];
}

export interface ConversationCoverageSummary {
  total_conversations: number;
  escalation_count: number;
  average_user_turns: number;
  template_counts: Record<string, number>;
  variation_counts: Record<string, number>;
}

export interface FinalizedDiffSummary {
  current_case_count: number;
  finalized_case_count: number;
  added_count: number;
  removed_count: number;
  changed_count: number;
  unchanged_count: number;
}

export type GeneratedDatasetPreviewCase = GeneratedDatasetCase | GeneratedConversationCase;

export interface CustomDatasetGenerateRequest {
  title?: string;
  project_description: string;
  sample_count: number;
  generator_model: string;
  focus_areas?: string;
  dataset_kind?: string;
  generation_mode?: string;
  source_label?: string;
  source_material?: string;
  source_paths?: string[];
}

export interface CustomDatasetImportRequest {
  dataset_json: string;
  title?: string;
  project_description?: string;
  focus_areas?: string;
  source_label?: string;
}

export interface CustomDatasetReviewStatusUpdateRequest {
  review_status: string;
  reviewer_role?: string;
  reusable_metric_candidate?: boolean;
}

export interface CustomDatasetCaseUpdateRequest {
  question?: string;
  persona?: string;
  expected_answer?: string;
  expected_outcome?: string;
}

export interface CustomDatasetSummary {
  dataset_id: string;
  title: string;
  generator_model: string;
  source_type: string;
  source_label?: string | null;
  dataset_kind?: string;
  generation_mode?: string;
  source_attribution?: Record<string, unknown>;
  sample_count: number;
  base_case_count: number;
  created_at: string;
  path: string;
  mutation_summary: Record<string, number>;
  conversation_summary?: ConversationCoverageSummary | null;
  dataset_tags: string[];
  dataset_tag_summary: Record<string, number>;
  review_status: string;
  review_role?: string | null;
  reviewed_at?: string | null;
  reusable_metric_candidate: boolean;
  finalized_diff_summary?: FinalizedDiffSummary | null;
  finalized_at?: string | null;
  finalized_path?: string | null;
  finalized_case_count: number;
  promoted_to_regression_at?: string | null;
  regression_dataset_path?: string | null;
}

export interface CustomDatasetDetail extends CustomDatasetSummary {
  focus_areas?: string | null;
  project_description: string;
  preview: GeneratedDatasetPreviewCase[];
}

export interface WorkspaceSourceFile {
  path: string;
  size_kb: number;
}

export const resultsApi = {
  listReports: (limit = 50) =>
    request<ReportListItem[]>(`/results/reports?limit=${limit}`),
  getReport: (filename: string) =>
    request<ReportSummary>(`/results/reports/${encodeURIComponent(filename)}`),
  getRaw: (filename: string) =>
    request<Record<string, unknown>>(
      `/results/reports/${encodeURIComponent(filename)}/raw`
    ),
  compare: (filenames: string[]) =>
    request<Record<string, ReportCompareSummary>>('/results/compare', {
      method: "POST",
      body: JSON.stringify(filenames),
    }),
};

  export const customDatasetsApi = {
    generate: (req: CustomDatasetGenerateRequest) =>
      request<CustomDatasetDetail>("/custom-datasets/generate", {
        method: "POST",
        body: JSON.stringify(req),
      }),
    importJson: (req: CustomDatasetImportRequest) =>
      request<CustomDatasetDetail>("/custom-datasets/import", {
        method: "POST",
        body: JSON.stringify(req),
      }),
    list: (limit = 20) =>
      request<CustomDatasetSummary[]>(`/custom-datasets?limit=${limit}`),
    listWorkspaceFiles: (limit = 30) =>
      request<WorkspaceSourceFile[]>(`/custom-datasets/workspace-files?limit=${limit}`),
    get: (datasetId: string) =>
      request<CustomDatasetDetail>(`/custom-datasets/${encodeURIComponent(datasetId)}`),
    updateCase: (datasetId: string, caseId: string, req: CustomDatasetCaseUpdateRequest) =>
      request<CustomDatasetDetail>(`/custom-datasets/${encodeURIComponent(datasetId)}/cases/${encodeURIComponent(caseId)}`, {
        method: "PATCH",
        body: JSON.stringify(req),
      }),
    updateReviewStatus: (datasetId: string, req: CustomDatasetReviewStatusUpdateRequest) =>
      request<CustomDatasetDetail>(`/custom-datasets/${encodeURIComponent(datasetId)}/review-status`, {
        method: "POST",
        body: JSON.stringify(req),
      }),
    promoteToRegression: (datasetId: string) =>
      request<CustomDatasetDetail>(`/custom-datasets/${encodeURIComponent(datasetId)}/promote-regression`, {
        method: "POST",
      }),
  };

// ─── Traces ──────────────────────────────────────────────────────────────────

export interface TraceSpan {
  span_id: string;
  parent_span_id: string | null;
  name: string;
  type: string;
  input: unknown;
  output: unknown;
  latency_ms: number;
  start_ts: number;
  metadata: Record<string, unknown>;
}

export interface Trace {
  trace_id: string;
  name: string;
  tags: string[];
  spans: TraceSpan[];
  start_ts: number;
  end_ts: number | null;
  metadata: Record<string, unknown>;
}

export interface TraceDetail {
  trace: Trace;
  span_count: number;
  duration_ms: number | null;
}

export interface TraceListResponse {
  traces: Trace[];
  total: number;
}

export const tracesApi = {
  list: (params?: { run_id?: string; tag?: string; limit?: number }) => {
    const q = new URLSearchParams();
    if (params?.run_id) q.set("run_id", params.run_id);
    if (params?.tag) q.set("tag", params.tag);
    if (params?.limit != null) q.set("limit", String(params.limit));
    const qs = q.toString();
    return request<TraceListResponse>(`/traces${qs ? `?${qs}` : ""}`);
  },
  get: (traceId: string) => request<TraceDetail>(`/traces/${encodeURIComponent(traceId)}`),
  eval: (traceId: string) =>
    request<{ trace_id: string; status: string; span_count: number }>(
      `/traces/${encodeURIComponent(traceId)}/eval`,
      { method: "POST" }
    ),
};

// ─── Health ──────────────────────────────────────────────────────────────────

export const healthApi = {
  check: () => request<{ status: string; version: string }>("/health"),
};

// ─── HITL ────────────────────────────────────────────────────────────────────

export interface PendingItem {
  item_id: string;
  model_name: string;
  test_category: string;
  test_id: string;
  question: string;
  model_response: string;
  llm_judge_score: number;
  llm_judge_label: string;
  llm_judge_reasoning: string;
  primary_judge_score?: number | null;
  primary_judge_label?: string | null;
  secondary_judge_score?: number | null;
  secondary_judge_label?: string | null;
  secondary_judge_reasoning?: string | null;
  judge_disagreement?: number | null;
  judge_agreement?: number | null;
  review_priority: number;
  queue_reason: string;
  owner?: string | null;
  status: string;
  sla_due_at?: string | null;
  metadata: Record<string, unknown>;
}

export type PendingItemStatus = "pending" | "in_progress" | "completed";

export interface PendingItemFilters {
  category?: string;
  owner?: string;
  status?: PendingItemStatus;
}

export interface PendingItemUpdateRequest {
  owner?: string | null;
  status: PendingItemStatus;
}

export interface BatchPendingItemUpdateRequest {
  item_ids: string[];
  owner?: string | null;
  status: PendingItemStatus;
}

export interface BatchPendingItemUpdateResponse {
  updated_count: number;
  items: PendingItem[];
  missing_item_ids: string[];
}

export interface AnnotationVerdict {
  label: "approve" | "adjust" | "reject";
  resolution: "accepted" | "corrected" | "rejected";
  requires_follow_up: boolean;
}

export interface SubmitAnnotationRequest {
  item_id: string;
  human_score: number;
  human_feedback: string;
  correction_type: "approve" | "adjust" | "reject";
  annotator_id?: string;
  reusable_metric_candidate?: boolean;
  policy_decision?: "confirmed_violation" | "false_positive" | "needs_follow_up";
  policy_notes?: string;
}

export interface PolicyReviewResponse {
  decision?: "confirmed_violation" | "false_positive" | "needs_follow_up";
  notes?: string;
  queue_reason?: string;
  review_priority?: number;
  risk_tags?: string[];
  source?: string;
}

export interface AnnotationResponse {
  annotation_id: string;
  item_id: string;
  status: string;
  verdict: AnnotationVerdict;
  policy_review: PolicyReviewResponse;
}

export interface MetricBacklogItem {
  entry_id: string;
  annotation_id: string;
  created_at: string;
  status: string;
  source: string;
  source_report?: string | null;
  model_name: string;
  test_category: string;
  test_id: string;
  question: string;
  queue_reason: string;
  human_feedback: string;
  correction_type: string;
  verdict: AnnotationVerdict;
  llm_judge_score: number;
  human_score: number;
  score_delta: number;
}

export interface HitlStats {
  total_completed: number;
  total_pending: number;
  average_agreement: number;
  panel_review_pending: number;
  high_priority_pending: number;
  training_ready_examples: number;
  metric_candidate_annotations: number;
  corrections_by_type: Record<string, number>;
  by_category: Record<string, { count: number; avg_human_score: number }>;
  annotators: string[];
}

export interface CalibrationRecommendation {
  issue: string;
  recommendation: string;
}

export interface CalibrationInsights {
  overall_metrics: {
    average_agreement?: number;
    mean_absolute_error?: number;
    judge_bias?: number;
  };
  recommendations: CalibrationRecommendation[];
  disagreement_taxonomy?: {
    reason_counts?: Record<string, number>;
    severity_counts?: Record<string, number>;
    direction_counts?: Record<string, number>;
    tag_counts?: Record<string, number>;
    top_reasons?: Array<{ reason: string; label: string; count: number }>;
  };
  prompt_version_comparison?: {
    versions?: Array<{
      prompt_version: string;
      reviewed_cases: number;
      average_agreement: number;
      mean_absolute_error: number;
      judge_bias: number;
    }>;
    known_version_count?: number;
    unknown_count?: number;
    best_agreement_version?: string | null;
    lowest_mae_version?: string | null;
  };
  training_data_available: number;
  ready_for_finetuning: boolean;
}

export interface CalibrationSampleCase {
  bucket: string;
  test_id: string;
  test_category: string;
  model_name: string;
  llm_score: number;
  human_score: number;
  score_difference: number;
  correction_type: string;
  question: string;
  human_feedback: string;
}

export interface CalibrationSampleSetResponse {
  total_samples: number;
  bucket_counts: Record<string, number>;
  samples: CalibrationSampleCase[];
}

export interface ExportTrainingResponse {
  output_file: string;
  exported_count: number;
  eligible_annotations: number;
  min_agreement: number;
}

export interface DisagreementExportResponse {
  output_file: string;
  threshold: number;
  exported_count: number;
  reason_taxonomy?: {
    top_reasons?: Array<{ reason: string; label: string; count: number }>;
  };
}

export const hitlApi = {
  getPending: (limit = 50, filters: PendingItemFilters = {}) => {
    const params = new URLSearchParams({ limit: String(limit) });
    if (filters.category) params.set("category", filters.category);
    if (filters.owner) params.set("owner", filters.owner);
    if (filters.status) params.set("status", filters.status);
    return request<PendingItem[]>(`/hitl/pending?${params}`);
  },
  updatePendingItem: (itemId: string, req: PendingItemUpdateRequest) =>
    request<PendingItem>(`/hitl/pending/${itemId}`, {
      method: "PATCH",
      body: JSON.stringify(req),
    }),
  batchUpdatePendingItems: (req: BatchPendingItemUpdateRequest) =>
    request<BatchPendingItemUpdateResponse>("/hitl/pending/batch", {
      method: "PATCH",
      body: JSON.stringify(req),
    }),
  annotate: (req: SubmitAnnotationRequest) =>
    request<AnnotationResponse>(
      "/hitl/annotate",
      { method: "POST", body: JSON.stringify(req) }
    ),
  getStats: () => request<HitlStats>("/hitl/stats"),
  getMetricBacklog: (limit = 12) => request<MetricBacklogItem[]>(`/hitl/metric-backlog?limit=${limit}`),
  getCalibration: () => request<CalibrationInsights>("/hitl/calibration"),
  getCalibrationSamples: (sampleSize = 12) =>
    request<CalibrationSampleSetResponse>(`/hitl/calibration-samples?sample_size=${sampleSize}`),
  generate: (report_filename: string, sample_per_test = 5) =>
    request<{ added_count: number; report: string }>("/hitl/generate", {
      method: "POST",
      body: JSON.stringify({ report_filename, sample_per_test }),
    }),
  exportDisagreements: (threshold = 0.3) =>
    request<DisagreementExportResponse>(
      `/hitl/export-disagreements?threshold=${threshold}`,
      { method: "POST" }
    ),
  exportTraining: (min_agreement = 0.2) =>
    request<ExportTrainingResponse>(
      `/hitl/export-training?min_agreement=${min_agreement}`,
      { method: "POST" }
    ),
};
