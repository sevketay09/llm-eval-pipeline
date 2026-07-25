import { useState, useEffect, useCallback } from "react";
import {
  AlertTriangle,
  ClipboardCheck,
  ThumbsUp,
  ThumbsDown,
  Minus,
  BarChart3,
  RefreshCw,
  FileDown,
  Filter,
  Sparkles,
} from "lucide-react";
import {
  hitlApi,
  resultsApi,
  tracesApi,
  type CalibrationInsights,
  type CalibrationSampleSetResponse,
  type MetricBacklogItem,
  type PendingItem,
  type PendingItemStatus,
  type HitlStats,
  type ReportListItem,
  type Trace,
} from "../api/client";

type CorrectionType = "approve" | "adjust" | "reject";
type PolicyDecision = "confirmed_violation" | "false_positive" | "needs_follow_up";
type DiffPart = { value: string; kind: "same" | "removed" | "added" };
type FailureCluster = {
  cluster_id: string;
  queue_reason: string;
  test_category: string;
  correction_type: string;
  count: number;
  average_delta: number;
  max_delta: number;
  latest_created_at: string;
  models: string[];
  example_question: string;
  example_feedback: string;
};

type ReviewerPersona = "qa" | "sme" | "pm";

type ReviewSignals = {
  suggestedReviewerPersona: ReviewerPersona;
  casePersona: string | null;
  riskTags: string[];
  escalationNeeded: boolean;
  isHighRisk: boolean;
  promptVersion: string | null;
};

type MetricSuggestion = {
  key: string;
  title: string;
  metricFamily: string;
  candidateName: string;
  rationale: string;
  evidence: string[];
};

function asRecord(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return {};
  }
  return value as Record<string, unknown>;
}

function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is string => typeof item === "string" && item.trim().length > 0);
}

function hasPolicyRiskSignals(riskTags: string[], isHighRisk: boolean): boolean {
  if (isHighRisk) {
    return true;
  }
  return riskTags.some((tag) =>
    ["policy", "safety", "pii", "security", "fraud", "compliance", "legal", "adversarial"].some((token) =>
      tag.toLocaleLowerCase("en-US").includes(token)
    )
  );
}

function firstText(...values: unknown[]): string | null {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return null;
}

function includesAny(value: string, tokens: string[]) {
  return tokens.some((token) => value.includes(token));
}

function slugify(value: string) {
  return value
    .toLocaleLowerCase("en-US")
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 42);
}

function buildReviewSignals(item: PendingItem): ReviewSignals {
  const metadata = asRecord(item.metadata);
  const fullResult = asRecord(metadata.full_result);
  const resultMetadata = asRecord(fullResult.metadata);
  const riskTags = Array.from(
    new Set([
      ...asStringArray(fullResult.risk_tags),
      ...asStringArray(resultMetadata.risk_tags),
      ...asStringArray(metadata.risk_tags),
    ])
  );
  const casePersona = firstText(fullResult.persona, resultMetadata.persona, metadata.persona);
  const promptVersion = firstText(
    metadata.prompt_version,
    metadata.judge_prompt_version,
    resultMetadata.prompt_version,
    fullResult.prompt_version
  );
  const escalationNeeded =
    Boolean(fullResult.escalation_needed) ||
    Boolean(resultMetadata.escalation_needed) ||
    includesAny(`${item.queue_reason} ${item.test_category}`.toLocaleLowerCase("en-US"), ["escalation", "handoff"]);
  const signalHaystack = [
    item.test_category,
    item.queue_reason,
    casePersona ?? "",
    ...riskTags,
  ]
    .join(" ")
    .toLocaleLowerCase("en-US");
  const isHighRisk =
    item.review_priority >= 35 ||
    escalationNeeded ||
    includesAny(signalHaystack, [
      "policy",
      "safety",
      "pii",
      "security",
      "fraud",
      "compliance",
      "legal",
      "adversarial",
      "harm",
      "tool",
      "schema",
      "structured_output",
      "json",
    ]);

  let suggestedReviewerPersona: ReviewerPersona = "qa";
  if (
    escalationNeeded ||
    includesAny(signalHaystack, ["policy", "safety", "pii", "security", "fraud", "compliance", "legal", "adversarial"])
  ) {
    suggestedReviewerPersona = "sme";
  } else if (
    includesAny(signalHaystack, ["persona", "tone", "journey", "retention", "intent", "onboarding", "escalation", "handoff"]) ||
    item.review_priority >= 45
  ) {
    suggestedReviewerPersona = "pm";
  }

  return {
    suggestedReviewerPersona,
    casePersona,
    riskTags,
    escalationNeeded,
    isHighRisk,
    promptVersion,
  };
}

function buildMetricSuggestions(
  item: PendingItem,
  signals: ReviewSignals,
  failureClusters: FailureCluster[]
): MetricSuggestion[] {
  const haystack = [item.test_category, item.queue_reason, signals.casePersona ?? "", ...signals.riskTags]
    .join(" ")
    .toLocaleLowerCase("en-US");
  const relatedCluster =
    failureClusters.find(
      (cluster) => cluster.test_category === item.test_category && cluster.queue_reason === item.queue_reason
    ) ?? failureClusters.find((cluster) => cluster.test_category === item.test_category) ?? null;
  const suggestions: MetricSuggestion[] = [];

  const pushSuggestion = (
    metricFamily: string,
    title: string,
    rationale: string,
    evidence: string[]
  ) => {
    const candidateName = `hitl_${slugify(item.test_category)}_${slugify(metricFamily)}_${slugify(item.queue_reason || title || "candidate")}`;
    if (suggestions.some((entry) => entry.metricFamily === metricFamily || entry.candidateName === candidateName)) {
      return;
    }
    suggestions.push({
      key: `${metricFamily}:${candidateName}`,
      title,
      metricFamily,
      candidateName,
      rationale,
      evidence,
    });
  };

  if (relatedCluster && relatedCluster.count >= 2) {
    pushSuggestion(
      "cluster_regression_gate",
      "Repeatable failure family",
      "The same queue reason and category repeat across multiple reviews; isolating this as a regression gate catches recurring production failures early.",
      [
        `${relatedCluster.count} reviewed cases`,
        `avg delta ${relatedCluster.average_delta.toFixed(2)}`,
        relatedCluster.queue_reason,
      ]
    );
  }

  if (signals.isHighRisk) {
    pushSuggestion(
      "risk_guardrail",
      "High-risk guardrail",
      "This case carries high priority or risk signals; it should be caught automatically with a dedicated guardrail metric before falling into the human queue.",
      [
        `priority ${item.review_priority.toFixed(1)}`,
        ...(signals.riskTags.slice(0, 2).length ? signals.riskTags.slice(0, 2) : [item.queue_reason]),
      ]
    );
  }

  if (includesAny(haystack, ["schema", "json", "structured_output"])) {
    pushSuggestion(
      "schema_reliability",
      "Structured output schema check",
      "The issue looks like structured output or schema reliability; making parse/shape validation a separate metric reduces review burden.",
      [item.test_category, item.queue_reason]
    );
  } else if (includesAny(haystack, ["tool", "function", "retriever"])) {
    pushSuggestion(
      "tool_path_accuracy",
      "Tool path correctness",
      "This case touches tool or retrieval paths; scoring selection correctness and missing tool calls as separate metrics would be more discriminative.",
      [item.test_category, item.queue_reason]
    );
  } else if (includesAny(haystack, ["faithfulness", "ground", "retrieval", "citation", "hallucination"])) {
    pushSuggestion(
      "retrieval_grounding",
      "Retrieval grounding",
      "This failure looks like a retrieval or faithfulness issue; tracking context fidelity as a separate metric is appropriate.",
      [item.test_category, item.queue_reason]
    );
  } else if (includesAny(haystack, ["persona", "intent", "escalation", "conversation", "tone"])) {
    pushSuggestion(
      "conversation_continuity",
      "Conversation continuity",
      "A case touching multi-turn behavior, persona, or escalation flow; a turn-level continuity metric produces a more fitting signal.",
      [item.test_category, signals.casePersona ?? item.queue_reason]
    );
  }

  if ((item.judge_disagreement ?? 0) >= 0.3) {
    pushSuggestion(
      "judge_alignment_rubric",
      "Judge alignment rubric",
      "Judge split is high; adding a clearer rubric or deterministic metric for this failure family would reduce the human queue.",
      [`split ${(item.judge_disagreement ?? 0).toFixed(2)}`, item.llm_judge_label]
    );
  }

  if (!suggestions.length) {
    pushSuggestion(
      "queue_reason_regression",
      "Queue reason regression check",
      "The safest starting point for this case is to build a narrow regression metric around the queue reason.",
      [item.test_category, item.queue_reason]
    );
  }

  return suggestions.slice(0, 3);
}

function tokenizeDiffText(value: string): string[] {
  return value.split(/(\s+)/).filter(Boolean);
}

function buildLcsTable(expectedTokens: string[], actualTokens: string[]): number[][] {
  const table = Array.from({ length: expectedTokens.length + 1 }, () =>
    Array.from({ length: actualTokens.length + 1 }, () => 0)
  );

  for (let expectedIndex = expectedTokens.length - 1; expectedIndex >= 0; expectedIndex -= 1) {
    for (let actualIndex = actualTokens.length - 1; actualIndex >= 0; actualIndex -= 1) {
      const currentRow = table[expectedIndex];
      const nextRow = table[expectedIndex + 1];
      if (!currentRow || !nextRow) {
        continue;
      }

      if (expectedTokens[expectedIndex] === actualTokens[actualIndex]) {
        currentRow[actualIndex] = (nextRow[actualIndex + 1] ?? 0) + 1;
      } else {
        currentRow[actualIndex] = Math.max(nextRow[actualIndex] ?? 0, currentRow[actualIndex + 1] ?? 0);
      }
    }
  }

  return table;
}

function buildDiffParts(expectedText: string, actualText: string): { expected: DiffPart[]; actual: DiffPart[] } {
  const expectedTokens = tokenizeDiffText(expectedText);
  const actualTokens = tokenizeDiffText(actualText);
  const lcsTable = buildLcsTable(expectedTokens, actualTokens);

  const expectedParts: DiffPart[] = [];
  const actualParts: DiffPart[] = [];

  let expectedIndex = 0;
  let actualIndex = 0;

  while (expectedIndex < expectedTokens.length && actualIndex < actualTokens.length) {
    const expectedToken = expectedTokens[expectedIndex];
    const actualToken = actualTokens[actualIndex];
    if (expectedToken === undefined || actualToken === undefined) {
      break;
    }

    if (expectedToken === actualToken) {
      expectedParts.push({ value: expectedToken, kind: "same" });
      actualParts.push({ value: actualToken, kind: "same" });
      expectedIndex += 1;
      actualIndex += 1;
      continue;
    }

    const nextExpectedRow = lcsTable[expectedIndex + 1];
    const currentExpectedRow = lcsTable[expectedIndex];
    const nextExpectedValue = nextExpectedRow?.[actualIndex] ?? 0;
    const nextActualValue = currentExpectedRow?.[actualIndex + 1] ?? 0;

    if (nextExpectedValue >= nextActualValue) {
      expectedParts.push({ value: expectedToken, kind: "removed" });
      expectedIndex += 1;
      continue;
    }

    actualParts.push({ value: actualToken, kind: "added" });
    actualIndex += 1;
  }

  while (expectedIndex < expectedTokens.length) {
    const expectedToken = expectedTokens[expectedIndex];
    if (expectedToken !== undefined) {
      expectedParts.push({ value: expectedToken, kind: "removed" });
    }
    expectedIndex += 1;
  }

  while (actualIndex < actualTokens.length) {
    const actualToken = actualTokens[actualIndex];
    if (actualToken !== undefined) {
      actualParts.push({ value: actualToken, kind: "added" });
    }
    actualIndex += 1;
  }

  return { expected: expectedParts, actual: actualParts };
}

function countChangedTokens(parts: DiffPart[], kind: "removed" | "added"): number {
  return parts.filter((part) => part.kind === kind && part.value.trim()).length;
}

function buildFailureClusters(entries: MetricBacklogItem[]): FailureCluster[] {
  const clusterMap = new Map<
    string,
    {
      queue_reason: string;
      test_category: string;
      correction_type: string;
      count: number;
      total_delta: number;
      max_delta: number;
      latest_created_at: string;
      models: Set<string>;
      example_question: string;
      example_feedback: string;
    }
  >();

  for (const entry of entries) {
    const queueReason = entry.queue_reason?.trim() || "Needs reusable metric follow-up";
    const testCategory = entry.test_category?.trim() || "unknown";
    const correctionType = entry.correction_type?.trim() || "adjust";
    const clusterId = `${queueReason}::${testCategory}::${correctionType}`;
    const existing = clusterMap.get(clusterId);

    if (existing) {
      existing.count += 1;
      existing.total_delta += entry.score_delta;
      existing.max_delta = Math.max(existing.max_delta, entry.score_delta);
      if (entry.created_at > existing.latest_created_at) {
        existing.latest_created_at = entry.created_at;
      }
      if (entry.model_name) {
        existing.models.add(entry.model_name);
      }
      if (!existing.example_feedback && entry.human_feedback) {
        existing.example_feedback = entry.human_feedback;
      }
      continue;
    }

    clusterMap.set(clusterId, {
      queue_reason: queueReason,
      test_category: testCategory,
      correction_type: correctionType,
      count: 1,
      total_delta: entry.score_delta,
      max_delta: entry.score_delta,
      latest_created_at: entry.created_at,
      models: new Set(entry.model_name ? [entry.model_name] : []),
      example_question: entry.question,
      example_feedback: entry.human_feedback,
    });
  }

  return Array.from(clusterMap.entries())
    .map(([cluster_id, cluster]) => ({
      cluster_id,
      queue_reason: cluster.queue_reason,
      test_category: cluster.test_category,
      correction_type: cluster.correction_type,
      count: cluster.count,
      average_delta: cluster.total_delta / cluster.count,
      max_delta: cluster.max_delta,
      latest_created_at: cluster.latest_created_at,
      models: Array.from(cluster.models).sort(),
      example_question: cluster.example_question,
      example_feedback: cluster.example_feedback,
    }))
    .sort((left, right) => {
      if (right.count !== left.count) {
        return right.count - left.count;
      }
      if (right.average_delta !== left.average_delta) {
        return right.average_delta - left.average_delta;
      }
      return right.latest_created_at.localeCompare(left.latest_created_at);
    });
}

function DiffText({ parts, mode }: { parts: DiffPart[]; mode: "expected" | "actual" }) {
  return (
    <p className="review-block-copy whitespace-pre-wrap text-sm leading-6">
      {parts.map((part, index) => {
        let className = "text-[rgba(62,47,31,0.86)]";
        if (part.kind === "removed") {
          className =
            mode === "expected"
              ? "rounded bg-[rgba(199,74,74,0.14)] text-[rgba(130,38,38,0.96)]"
              : "text-[rgba(62,47,31,0.42)]";
        }
        if (part.kind === "added") {
          className =
            mode === "actual"
              ? "rounded bg-[rgba(62,128,96,0.16)] text-[rgba(24,89,61,0.98)]"
              : "text-[rgba(62,47,31,0.42)]";
        }

        return (
          <span key={`${mode}-${index}`} className={className}>
            {part.value}
          </span>
        );
      })}
    </p>
  );
}

export default function HitlReview() {
  const [pending, setPending] = useState<PendingItem[]>([]);
  const [selectedItemIds, setSelectedItemIds] = useState<string[]>([]);
  const [stats, setStats] = useState<HitlStats | null>(null);
  const [calibration, setCalibration] = useState<CalibrationInsights | null>(null);
  const [calibrationSamples, setCalibrationSamples] = useState<CalibrationSampleSetResponse | null>(null);
  const [metricBacklog, setMetricBacklog] = useState<MetricBacklogItem[]>([]);
  const [reports, setReports] = useState<ReportListItem[]>([]);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [claiming, setClaiming] = useState(false);
  const [batchUpdating, setBatchUpdating] = useState(false);
  const [categoryFilter, setCategoryFilter] = useState("");
  const [statusFilter, setStatusFilter] = useState<PendingItemStatus | "">("");
  const [ownerFilter, setOwnerFilter] = useState("");
  const [reviewerPersonaFilter, setReviewerPersonaFilter] = useState<ReviewerPersona | "">("");
  const [disagreementOnly, setDisagreementOnly] = useState(false);
  const [highRiskOnly, setHighRiskOnly] = useState(false);
  const [traceCandidates, setTraceCandidates] = useState<Trace[]>([]);
  const [generating, setGenerating] = useState(false);
  const [genReport, setGenReport] = useState("");
  const [genSample, setGenSample] = useState(5);
  const [disagreementThreshold, setDisagreementThreshold] = useState(0.3);
  const [exportThreshold, setExportThreshold] = useState(0.2);
  const [reviewerId, setReviewerId] = useState("web-reviewer");
  const [exportingDisagreements, setExportingDisagreements] = useState(false);
  const [reusableMetricCandidate, setReusableMetricCandidate] = useState(false);
  const [policyDecision, setPolicyDecision] = useState<PolicyDecision>("confirmed_violation");
  const [policyNotes, setPolicyNotes] = useState("");

  const [humanScore, setHumanScore] = useState(0.5);
  const [feedback, setFeedback] = useState("");
  const [correctionType, setCorrectionType] = useState<CorrectionType>("approve");

  const pendingWithSignals = pending.map((item) => ({ item, signals: buildReviewSignals(item) }));
  const reviewerPersonaCounts = pendingWithSignals.reduce<Record<ReviewerPersona, number>>(
    (acc, entry) => {
      acc[entry.signals.suggestedReviewerPersona] += 1;
      return acc;
    },
    { qa: 0, sme: 0, pm: 0 }
  );
  const highRiskCount = pendingWithSignals.filter((entry) => entry.signals.isHighRisk).length;
  const disagreementCount = pendingWithSignals.filter(
    (entry) => (entry.item.judge_disagreement ?? 0) >= disagreementThreshold
  ).length;
  const filteredPending = pendingWithSignals
    .filter(({ item, signals }) => {
      if (reviewerPersonaFilter && signals.suggestedReviewerPersona !== reviewerPersonaFilter) {
        return false;
      }
      if (disagreementOnly && (item.judge_disagreement ?? 0) < disagreementThreshold) {
        return false;
      }
      if (highRiskOnly && !signals.isHighRisk) {
        return false;
      }
      return true;
    })
    .map((entry) => entry.item);
  const failureClusters = buildFailureClusters(metricBacklog);
  const topFailureCluster = failureClusters[0] ?? null;
  const current = filteredPending[currentIdx] ?? null;
  const currentSignals = current ? buildReviewSignals(current) : null;
  const currentSuggestions = current && currentSignals ? buildMetricSuggestions(current, currentSignals, failureClusters) : [];

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const [items, st, calibrationInsights, calibrationSampleSet, backlogItems, reportItems, sampledTraces] = await Promise.all([
        hitlApi.getPending(100, {
          category: categoryFilter || undefined,
          owner: ownerFilter.trim() || undefined,
          status: statusFilter || undefined,
        }),
        hitlApi.getStats(),
        hitlApi.getCalibration(),
        hitlApi.getCalibrationSamples(),
        hitlApi.getMetricBacklog(),
        resultsApi.listReports(20),
        tracesApi.list({ tag: "eval_sampled", limit: 50 }),
      ]);
      setPending(items);
      setStats(st);
      setCalibration(calibrationInsights);
      setCalibrationSamples(calibrationSampleSet);
      setMetricBacklog(backlogItems);
      setReports(reportItems);
      setTraceCandidates(sampledTraces.traces);
      setCurrentIdx(0);
    } catch (e) {
      console.error("Failed to load HITL data", e);
    } finally {
      setLoading(false);
    }
  }, [categoryFilter, ownerFilter, statusFilter]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  useEffect(() => {
    if (current) {
      setHumanScore(current.llm_judge_score);
      setFeedback("");
      setCorrectionType("approve");
      setReusableMetricCandidate(false);
      setPolicyDecision("confirmed_violation");
      setPolicyNotes("");
    }
  }, [current]);

  useEffect(() => {
    setSelectedItemIds((currentSelected) =>
      currentSelected.filter((itemId) => filteredPending.some((item) => item.item_id === itemId))
    );
  }, [filteredPending]);

  useEffect(() => {
    setCurrentIdx((prev) => {
      if (!filteredPending.length) {
        return 0;
      }
      return Math.min(prev, filteredPending.length - 1);
    });
  }, [filteredPending.length]);

  const refreshQueue = useCallback(
    async (itemIdToKeep?: string) => {
      const [items, refreshedStats, refreshedCalibration, refreshedSamples, refreshedBacklog] = await Promise.all([
        hitlApi.getPending(100, {
          category: categoryFilter || undefined,
          owner: ownerFilter.trim() || undefined,
          status: statusFilter || undefined,
        }),
        hitlApi.getStats(),
        hitlApi.getCalibration(),
        hitlApi.getCalibrationSamples(),
        hitlApi.getMetricBacklog(),
      ]);
      setPending(items);
      setStats(refreshedStats);
      setCalibration(refreshedCalibration);
      setCalibrationSamples(refreshedSamples);
      setMetricBacklog(refreshedBacklog);
      setCurrentIdx((prev) => {
        if (!items.length) return 0;
        if (itemIdToKeep) {
          const matchedIndex = items.findIndex((item) => item.item_id === itemIdToKeep);
          if (matchedIndex >= 0) return matchedIndex;
        }
        return Math.min(prev, Math.max(items.length - 1, 0));
      });
    },
    [categoryFilter, ownerFilter, statusFilter]
  );

  async function handleSubmit() {
    if (!current) return;
    setSubmitting(true);
    try {
      const shouldSendPolicyDecision = hasPolicyRiskSignals(
        currentSignals?.riskTags ?? [],
        Boolean(currentSignals?.isHighRisk)
      );
      await hitlApi.annotate({
        item_id: current.item_id,
        human_score: humanScore,
        human_feedback: feedback,
        correction_type: correctionType,
        annotator_id: reviewerId.trim() || "web-reviewer",
        reusable_metric_candidate: reusableMetricCandidate,
        ...(shouldSendPolicyDecision
          ? {
              policy_decision: policyDecision,
              policy_notes: policyNotes,
            }
          : {}),
      });
      await refreshQueue();
    } catch (e) {
      console.error("Annotation failed", e);
    } finally {
      setSubmitting(false);
    }
  }

  async function handleQueueUpdate(status: PendingItemStatus, owner: string | null) {
    if (!current) return;
    setClaiming(true);
    try {
      await hitlApi.updatePendingItem(current.item_id, { status, owner });
      await refreshQueue(current.item_id);
    } catch (e) {
      console.error("Queue update failed", e);
    } finally {
      setClaiming(false);
    }
  }

  async function handleBatchQueueUpdate(status: PendingItemStatus, owner: string | null) {
    if (!selectedItemIds.length) return;
    setBatchUpdating(true);
    try {
      await hitlApi.batchUpdatePendingItems({
        item_ids: selectedItemIds,
        status,
        owner,
      });
      await refreshQueue(current?.item_id);
      setSelectedItemIds([]);
    } catch (e) {
      console.error("Batch queue update failed", e);
    } finally {
      setBatchUpdating(false);
    }
  }

  function handleToggleItemSelection(itemId: string) {
    setSelectedItemIds((currentSelected) =>
      currentSelected.includes(itemId)
        ? currentSelected.filter((selectedId) => selectedId !== itemId)
        : [...currentSelected, itemId]
    );
  }

  async function handleGenerate() {
    if (!genReport) return;
    setGenerating(true);
    try {
      const res = await hitlApi.generate(genReport, genSample);
      alert(`${res.added_count} items added for review.`);
      loadData();
    } catch (e) {
      console.error("Generate failed", e);
    } finally {
      setGenerating(false);
    }
  }

  async function handleExport() {
    try {
      const res = await hitlApi.exportTraining(exportThreshold);
      alert(`Training data exported: ${res.output_file}\nExamples: ${res.exported_count}`);
    } catch (e) {
      console.error("Export failed", e);
    }
  }

  async function handleDisagreementExport() {
    setExportingDisagreements(true);
    try {
      const res = await hitlApi.exportDisagreements(disagreementThreshold);
      const topReason = res.reason_taxonomy?.top_reasons?.[0]?.label;
      alert(
        `Disagreement report exported: ${res.output_file}\nCases: ${res.exported_count}${topReason ? `\nTop reason: ${topReason}` : ""}`
      );
    } catch (e) {
      console.error("Disagreement export failed", e);
    } finally {
      setExportingDisagreements(false);
    }
  }

  const categories = [...new Set(pending.map((p) => p.test_category))];
  const expectedAnswer = typeof current?.metadata?.expected_answer === "string" ? current.metadata.expected_answer : null;
  const sourceReport = typeof current?.metadata?.source_report === "string" ? current.metadata.source_report : null;
  const reviewerHandle = reviewerId.trim() || "web-reviewer";
  const shouldShowPolicyDecision = hasPolicyRiskSignals(
    currentSignals?.riskTags ?? [],
    Boolean(currentSignals?.isHighRisk)
  );
  const slaDueLabel = current?.sla_due_at ? new Date(current.sla_due_at).toLocaleString() : "—";
  const selectedCount = selectedItemIds.length;
  const queuePreview = filteredPending.slice(0, 12);
  const diffParts = current && expectedAnswer ? buildDiffParts(expectedAnswer, current.model_response || "") : null;
  const removedTokenCount = diffParts ? countChangedTokens(diffParts.expected, "removed") : 0;
  const addedTokenCount = diffParts ? countChangedTokens(diffParts.actual, "added") : 0;

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="loading-orb" />
      </div>
    );
  }

  return (
    <div className="page-shell motion-shell">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <header className="page-header motion-hero">
          <p className="page-kicker">Adjudication</p>
          <div className="flex items-center gap-3">
            <ClipboardCheck className="accent-icon h-6 w-6" />
            <h1 className="page-title">Judge Disagreement Desk</h1>
          </div>
          <p className="page-subtitle">
            Surface primary-vs-secondary judge splits, arbitrate the hardest cases, then turn reviewed calls into training data.
          </p>
        </header>

        <div className="button-row motion-rise motion-delay-1">
          <button onClick={handleExport} className="button-secondary">
            <FileDown className="w-3.5 h-3.5" />
            Export Training
          </button>
          <button onClick={loadData} className="button-secondary">
            <RefreshCw className="w-3.5 h-3.5" />
            Refresh
          </button>
        </div>
      </div>

      {stats && (
        <div className="motion-stagger-grid grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-8">
          <StatCard label="Pending" value={stats.total_pending} />
          <StatCard label="Panel Pending" value={stats.panel_review_pending} />
          <StatCard label="High Priority" value={stats.high_priority_pending} />
          <StatCard label="Completed" value={stats.total_completed} />
          <StatCard label="Agreement" value={`${(stats.average_agreement * 100).toFixed(1)}%`} />
          <StatCard label="Training Ready" value={stats.training_ready_examples} />
          <StatCard label="Metric Candidates" value={stats.metric_candidate_annotations} />
          <StatCard label="Approved" value={stats.corrections_by_type?.approve ?? 0} />
          <StatCard label="Adjusted" value={stats.corrections_by_type?.adjust ?? 0} />
          <StatCard label="Rejected" value={stats.corrections_by_type?.reject ?? 0} />
        </div>
      )}

      {metricBacklog.length > 0 && (
        <div className="panel-surface panel-quiet motion-rise motion-delay-2 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="section-caption mb-2">Metric Backlog</p>
              <h2 className="section-heading">Recent Review-Derived Metric Candidates</h2>
            </div>
            <span className="provider-chip">{metricBacklog.length} recent entries</span>
          </div>
          <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
            {metricBacklog.slice(0, 4).map((entry) => (
              <div key={entry.entry_id} className="rounded-[1rem] hairline px-4 py-3">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="provider-chip">{entry.test_category}</span>
                  <span className="provider-chip">{entry.correction_type}</span>
                  <span className="provider-chip">delta {entry.score_delta.toFixed(2)}</span>
                </div>
                <p className="body-copy mt-3 text-sm line-clamp-2">{entry.question}</p>
                <p className="micro-copy mt-2">{entry.queue_reason}</p>
                {entry.human_feedback && (
                  <p className="micro-copy mt-2 line-clamp-3 whitespace-pre-wrap">{entry.human_feedback}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {failureClusters.length > 0 && (
        <div className="panel-surface panel-roomy motion-rise motion-delay-2 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="section-caption mb-2">Failure Clusters</p>
              <h2 className="section-heading">Reviewed Failure Patterns Ready for Metric Design</h2>
              <p className="page-subtitle mt-2 text-sm">
                Clustered from metric-candidate reviews by queue reason, category and reviewer verdict to surface repeatable failure families.
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <span className="provider-chip">{failureClusters.length} active clusters</span>
              {topFailureCluster && <span className="provider-chip">Top cluster: {topFailureCluster.count} cases</span>}
            </div>
          </div>

          <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
            <div className="rounded-[1rem] hairline px-4 py-3">
              <p className="micro-copy">Largest Cluster</p>
              <p className="body-copy mt-1 font-semibold">{topFailureCluster?.count ?? 0} reviewed failures</p>
              <p className="micro-copy mt-2 line-clamp-2">{topFailureCluster?.queue_reason ?? "No repeated failure cluster yet."}</p>
            </div>
            <div className="rounded-[1rem] hairline px-4 py-3">
              <p className="micro-copy">Distinct Categories</p>
              <p className="body-copy mt-1 font-semibold">{new Set(failureClusters.map((cluster) => cluster.test_category)).size}</p>
              <p className="micro-copy mt-2">Clustered from review-derived metric backlog entries.</p>
            </div>
            <div className="rounded-[1rem] hairline px-4 py-3">
              <p className="micro-copy">Highest Avg Delta</p>
              <p className="body-copy mt-1 font-semibold">
                {failureClusters.length ? Math.max(...failureClusters.map((cluster) => cluster.average_delta)).toFixed(2) : "0.00"}
              </p>
              <p className="micro-copy mt-2">Average judge-human gap in the strongest cluster family.</p>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
            {failureClusters.slice(0, 6).map((cluster) => (
              <div key={cluster.cluster_id} className="rounded-[1rem] hairline px-4 py-4 space-y-3">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="provider-chip">{cluster.test_category}</span>
                  <span className="provider-chip">{cluster.correction_type}</span>
                  <span className="provider-chip">{cluster.count} cases</span>
                </div>

                <div>
                  <p className="body-copy text-sm font-semibold">{cluster.queue_reason}</p>
                  <p className="micro-copy mt-2">
                    Avg delta {cluster.average_delta.toFixed(2)} · max {cluster.max_delta.toFixed(2)} · {cluster.models.length} model
                    {cluster.models.length === 1 ? "" : "s"}
                  </p>
                </div>

                <p className="body-copy text-sm line-clamp-2">{cluster.example_question}</p>

                {cluster.example_feedback && (
                  <p className="micro-copy line-clamp-3 whitespace-pre-wrap">{cluster.example_feedback}</p>
                )}

                <div className="flex flex-wrap gap-2">
                  {cluster.models.slice(0, 3).map((model) => (
                    <span key={`${cluster.cluster_id}-${model}`} className="provider-chip">
                      {model}
                    </span>
                  ))}
                  {cluster.models.length > 3 && (
                    <span className="provider-chip">+{cluster.models.length - 3} more</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="panel-surface panel-roomy motion-rise motion-delay-2 space-y-4">
        <div>
          <p className="section-caption mb-2">Queue Control</p>
          <h2 className="section-heading">Backfill the Review Queue from a Report</h2>
          <p className="page-subtitle mt-2 text-sm">
            New runs now auto-enqueue the strongest judge splits. Use this panel to backfill older reports or widen the queue manually.
          </p>
        </div>
        <div className="flex flex-wrap items-end gap-3">
          <div className="control-group min-w-[200px] flex-1">
            <label className="label">Report</label>
            <select
              value={genReport}
              onChange={(e) => setGenReport(e.target.value)}
              className="control-surface"
            >
              <option value="">Select report...</option>
              {reports.map((r) => (
                <option key={r.filename} value={r.filename}>
                  {r.filename}
                </option>
              ))}
            </select>
          </div>
          <div className="control-group w-24">
            <label className="label">Samples</label>
            <input
              type="number"
              min={1}
              max={50}
              value={genSample}
              onChange={(e) => setGenSample(Number(e.target.value))}
              className="control-surface"
            />
          </div>
          <button
            onClick={handleGenerate}
            disabled={!genReport || generating}
            className="button-primary"
          >
            {generating ? "Generating..." : "Backfill Queue"}
          </button>
        </div>
      </div>

      <div className="panel-surface panel-quiet motion-rise motion-delay-3 space-y-4">
        <div>
          <p className="section-caption mb-2">Training Loop</p>
          <h2 className="section-heading">Export Reviewed Decisions for Judge Tuning</h2>
        </div>
        <div className="flex flex-wrap items-end gap-3">
          <div className="control-group w-32">
            <label className="label">Min Agreement</label>
            <input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={exportThreshold}
              onChange={(e) => setExportThreshold(Number(e.target.value))}
              className="control-surface"
            />
          </div>
          <div className="rounded-[1rem] hairline px-4 py-3">
            <p className="micro-copy">Ready now</p>
            <p className="body-copy mt-1 font-semibold">{stats?.training_ready_examples ?? 0} examples</p>
          </div>
          <button onClick={handleExport} className="button-secondary">
            <FileDown className="w-3.5 h-3.5" />
            Export Training
          </button>
        </div>
      </div>

      {calibration && (
        <div className="panel-surface panel-quiet motion-rise motion-delay-3 space-y-4">
          <div>
            <p className="section-caption mb-2">Calibration</p>
            <h2 className="section-heading">Judge Quality Watch</h2>
            <p className="page-subtitle mt-2 text-sm">
              Reviewed decisions now feed live judge calibration signals and a downloadable disagreement report.
            </p>
          </div>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-4">
            <div className="rounded-[1rem] hairline px-4 py-3">
              <p className="micro-copy">Agreement</p>
              <p className="body-copy mt-1 font-semibold">{((calibration.overall_metrics.average_agreement ?? 0) * 100).toFixed(1)}%</p>
            </div>
            <div className="rounded-[1rem] hairline px-4 py-3">
              <p className="micro-copy">Mean Abs Error</p>
              <p className="body-copy mt-1 font-semibold">{(calibration.overall_metrics.mean_absolute_error ?? 0).toFixed(2)}</p>
            </div>
            <div className="rounded-[1rem] hairline px-4 py-3">
              <p className="micro-copy">Judge Bias</p>
              <p className="body-copy mt-1 font-semibold">{(calibration.overall_metrics.judge_bias ?? 0).toFixed(2)}</p>
            </div>
            <div className="rounded-[1rem] hairline px-4 py-3">
              <p className="micro-copy">Calibration Set</p>
              <p className="body-copy mt-1 font-semibold">{calibration.training_data_available}</p>
            </div>
          </div>
          <div className="flex flex-wrap items-end gap-3">
            <div className="control-group w-36">
              <label className="label">Disagreement Cutoff</label>
              <input
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={disagreementThreshold}
                onChange={(e) => setDisagreementThreshold(Number(e.target.value))}
                className="control-surface"
              />
            </div>
            <div className="rounded-[1rem] hairline px-4 py-3">
              <p className="micro-copy">Fine-tuning Readiness</p>
              <p className="body-copy mt-1 font-semibold">{calibration.ready_for_finetuning ? "Ready" : "Collect More"}</p>
            </div>
            <button onClick={handleDisagreementExport} disabled={exportingDisagreements} className="button-secondary">
              <BarChart3 className="w-3.5 h-3.5" />
              {exportingDisagreements ? "Exporting..." : "Export Disagreements"}
            </button>
          </div>
          <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
            {(calibration.recommendations.length ? calibration.recommendations : [{ issue: "No major drift", recommendation: "Current reviewed sample looks stable enough to keep the current judge prompt." }]).map((item) => (
              <div key={`${item.issue}-${item.recommendation}`} className="rounded-[1rem] hairline px-4 py-3">
                <p className="micro-copy">{item.issue}</p>
                <p className="body-copy mt-2 text-sm">{item.recommendation}</p>
              </div>
            ))}
          </div>
          {calibration.disagreement_taxonomy?.top_reasons && calibration.disagreement_taxonomy.top_reasons.length > 0 && (
            <div className="space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <p className="section-caption">Judge Disagreement Reasons</p>
                  <p className="micro-copy mt-2">Most frequent reviewed split patterns behind judge drift.</p>
                </div>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(calibration.disagreement_taxonomy.direction_counts ?? {}).map(([direction, count]) => (
                    <span key={direction} className="provider-chip">{direction.replace(/_/g, " ")}: {count}</span>
                  ))}
                </div>
              </div>
              <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
                {calibration.disagreement_taxonomy.top_reasons.map((item) => (
                  <div key={item.reason} className="rounded-[1rem] hairline px-4 py-3">
                    <p className="micro-copy">Primary reason</p>
                    <p className="body-copy mt-2 text-sm font-semibold capitalize">{item.label}</p>
                    <p className="micro-copy mt-2">{item.count} reviewed cases</p>
                  </div>
                ))}
              </div>
            </div>
          )}
          {calibration.prompt_version_comparison?.versions && calibration.prompt_version_comparison.versions.length > 0 && (
            <div className="space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <p className="section-caption">Prompt Version Compare</p>
                  <p className="micro-copy mt-2">Human-vs-judge calibration quality grouped by judge prompt version.</p>
                </div>
                <div className="flex flex-wrap gap-2">
                  {calibration.prompt_version_comparison.best_agreement_version && (
                    <span className="provider-chip">Best agreement: {calibration.prompt_version_comparison.best_agreement_version}</span>
                  )}
                  {calibration.prompt_version_comparison.lowest_mae_version && (
                    <span className="provider-chip">Lowest MAE: {calibration.prompt_version_comparison.lowest_mae_version}</span>
                  )}
                </div>
              </div>
              <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
                {calibration.prompt_version_comparison.versions.slice(0, 6).map((version) => (
                  <div key={version.prompt_version} className="rounded-[1rem] hairline px-4 py-3">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="provider-chip">{version.prompt_version}</span>
                      <span className="provider-chip">{version.reviewed_cases} reviewed</span>
                    </div>
                    <div className="mt-3 grid grid-cols-3 gap-2 text-[11px] muted-copy">
                      <div>
                        <p>Agreement</p>
                        <p className="body-copy mt-1 text-sm font-semibold">{(version.average_agreement * 100).toFixed(1)}%</p>
                      </div>
                      <div>
                        <p>MAE</p>
                        <p className="body-copy mt-1 text-sm font-semibold">{version.mean_absolute_error.toFixed(2)}</p>
                      </div>
                      <div>
                        <p>Bias</p>
                        <p className="body-copy mt-1 text-sm font-semibold">{version.judge_bias.toFixed(2)}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          {calibrationSamples && calibrationSamples.samples.length > 0 && (
            <div className="space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <p className="section-caption">Calibration Sample Set</p>
                  <p className="micro-copy mt-2">Balanced reviewed cases for anchor, boundary and disagreement prompt checks.</p>
                </div>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(calibrationSamples.bucket_counts).map(([bucket, count]) => (
                    <span key={bucket} className="provider-chip">{bucket.replace(/_/g, " ")}: {count}</span>
                  ))}
                </div>
              </div>
              <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
                {calibrationSamples.samples.slice(0, 6).map((sample) => (
                  <div key={`${sample.bucket}-${sample.test_id}-${sample.model_name}`} className="rounded-[1rem] hairline px-4 py-3">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="provider-chip">{sample.bucket.replace(/_/g, " ")}</span>
                      <span className="provider-chip">{sample.test_category}</span>
                      <span className="provider-chip">{sample.correction_type}</span>
                    </div>
                    <p className="body-copy mt-3 text-sm">{sample.question}</p>
                    <div className="mt-3 flex flex-wrap gap-3 text-[11px] muted-copy">
                      <span>LLM {sample.llm_score.toFixed(2)}</span>
                      <span>Human {sample.human_score.toFixed(2)}</span>
                      <span>Diff {sample.score_difference.toFixed(2)}</span>
                    </div>
                    {sample.human_feedback && (
                      <p className="micro-copy mt-3 whitespace-pre-wrap">{sample.human_feedback}</p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {categories.length > 0 && (
        <div className="panel-surface panel-quiet motion-rise motion-delay-4 flex flex-wrap items-center gap-3">
          <Filter className="h-4 w-4 muted-copy" />
          <select
            value={categoryFilter}
            onChange={(e) => setCategoryFilter(e.target.value)}
            className="control-surface w-fit min-w-[180px]"
          >
            <option value="">All categories</option>
            {categories.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value as PendingItemStatus | "")}
            className="control-surface w-fit min-w-[180px]"
          >
            <option value="">All statuses</option>
            <option value="pending">pending</option>
            <option value="in_progress">in_progress</option>
            <option value="completed">completed</option>
          </select>
          <input
            value={ownerFilter}
            onChange={(e) => setOwnerFilter(e.target.value)}
            placeholder="Filter by owner"
            className="control-surface w-fit min-w-[180px] text-sm"
          />
          <select
            value={reviewerPersonaFilter}
            onChange={(e) => setReviewerPersonaFilter(e.target.value as ReviewerPersona | "")}
            className="control-surface w-fit min-w-[180px]"
          >
            <option value="">All reviewer lanes</option>
            <option value="qa">QA lane ({reviewerPersonaCounts.qa})</option>
            <option value="sme">SME lane ({reviewerPersonaCounts.sme})</option>
            <option value="pm">PM lane ({reviewerPersonaCounts.pm})</option>
          </select>
          <label className="option-row cursor-pointer">
            <input
              type="checkbox"
              checked={disagreementOnly}
              onChange={(e) => setDisagreementOnly(e.target.checked)}
              className="control-check"
            />
            <span>Disagreement only ({disagreementCount})</span>
          </label>
          <label className="option-row cursor-pointer">
            <input
              type="checkbox"
              checked={highRiskOnly}
              onChange={(e) => setHighRiskOnly(e.target.checked)}
              className="control-check"
            />
            <span>High-risk only ({highRiskCount})</span>
          </label>
          <span className="micro-copy">{filteredPending.length} / {pending.length} items visible</span>
        </div>
      )}

      {filteredPending.length > 0 && (
        <div className="panel-surface panel-quiet motion-rise motion-delay-4 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="section-caption mb-2">Batch Triage</p>
              <h2 className="section-heading">Select a queue slice and update it together</h2>
            </div>
            <span className="micro-copy">{selectedCount} selected</span>
          </div>

          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => setSelectedItemIds(filteredPending.map((item) => item.item_id))}
              disabled={batchUpdating}
              className="button-secondary"
            >
              Select Visible
            </button>
            <button
              onClick={() => setSelectedItemIds([])}
              disabled={batchUpdating || selectedCount === 0}
              className="button-secondary"
            >
              Clear Selection
            </button>
            <button
              onClick={() => handleBatchQueueUpdate("in_progress", reviewerHandle)}
              disabled={batchUpdating || selectedCount === 0}
              className="button-secondary"
            >
              {batchUpdating ? "Updating..." : "Claim Selected"}
            </button>
            <button
              onClick={() => handleBatchQueueUpdate("pending", null)}
              disabled={batchUpdating || selectedCount === 0}
              className="button-secondary"
            >
              Release Selected
            </button>
          </div>

          <div className="grid grid-cols-1 gap-2 xl:grid-cols-2">
            {queuePreview.map((item, index) => {
              const isSelected = selectedItemIds.includes(item.item_id);
              const isFocused = current?.item_id === item.item_id;
              const signals = buildReviewSignals(item);

              return (
                <button
                  key={item.item_id}
                  type="button"
                  onClick={() => setCurrentIdx(index)}
                  className={`flex items-start gap-3 rounded-[1rem] border px-3 py-3 text-left transition ${isFocused ? "border-[rgba(138,229,197,0.55)] bg-[rgba(13,32,28,0.88)]" : "hairline bg-[rgba(13,20,24,0.72)]"}`}
                >
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={() => handleToggleItemSelection(item.item_id)}
                    onClick={(event) => event.stopPropagation()}
                    className="mt-1 h-4 w-4"
                  />
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="provider-chip">{item.model_name}</span>
                      <span className="provider-chip">{item.test_category}</span>
                      <span className="provider-chip">{item.status}</span>
                      <span className="provider-chip">lane {signals.suggestedReviewerPersona.toUpperCase()}</span>
                      {signals.isHighRisk && <span className="provider-chip">high risk</span>}
                      {(item.judge_disagreement ?? 0) >= disagreementThreshold && <span className="provider-chip">split focus</span>}
                    </div>
                    <p className="body-copy mt-2 line-clamp-2 text-sm">{item.question || "(no question)"}</p>
                    <div className="mt-2 flex flex-wrap gap-3 text-[11px] muted-copy">
                      <span>Owner: {item.owner ?? "Unassigned"}</span>
                      <span>Priority: {item.review_priority.toFixed(1)}</span>
                      {signals.casePersona && <span>Case persona: {signals.casePersona}</span>}
                    </div>
                  </div>
                </button>
              );
            })}
          </div>

          {filteredPending.length > queuePreview.length && (
            <p className="micro-copy">Showing first {queuePreview.length} filtered items for batch triage.</p>
          )}
        </div>
      )}

      {current ? (
        <>
          <section className="panel-surface review-command motion-rise motion-delay-5">
            <div className="review-command-grid">
              <div className="space-y-3">
                <div>
                  <p className="section-caption mb-2">Current Focus</p>
                  <h2 className="review-command-title">Arbitrate the split, then feed the training loop.</h2>
                </div>
                <p className="page-subtitle">
                  Prompt, answer, judge panel and final human call are stacked for fast arbitration of the hardest model decisions.
                </p>
              </div>

              <div className="review-meta-row">
                <span className="review-meta-pill">Item {currentIdx + 1} / {filteredPending.length}</span>
                <span className="review-meta-pill review-meta-pill-warm">{current.test_category}</span>
                <span className="review-meta-pill review-meta-pill-cool">{current.model_name}</span>
                {currentSignals && <span className="review-meta-pill">lane {currentSignals.suggestedReviewerPersona.toUpperCase()}</span>}
                {currentSignals?.isHighRisk && <span className="review-meta-pill review-meta-pill-warm">high risk</span>}
                {current.judge_disagreement != null && (
                  <span className="review-meta-pill review-meta-pill-warm">Split {current.judge_disagreement.toFixed(2)}</span>
                )}
              </div>
            </div>
          </section>

          <div className="review-stage motion-rise motion-delay-6">
            <div className="review-main motion-stagger-stack">
              <div className="panel-surface review-block review-block-question space-y-3">
                <div className="flex items-center justify-between gap-3">
                  <span className="section-caption">Question</span>
                  <span className="provider-chip">Prompt</span>
                </div>
                <p className="review-block-copy whitespace-pre-wrap text-sm">
                  {current.question || "(no question)"}
                </p>
              </div>

              <div className="panel-surface review-block review-block-response space-y-3">
                <div className="flex items-center justify-between gap-3">
                  <span className="section-caption">Model Response</span>
                  <span className="provider-chip">{current.model_name}</span>
                </div>
                <div className="review-scroll-box">
                  <p className="review-block-copy whitespace-pre-wrap text-sm">
                    {current.model_response || "(empty)"}
                  </p>
                </div>
              </div>

              {expectedAnswer && (
                <div className="panel-surface review-block space-y-3">
                  <div className="flex items-center justify-between gap-3">
                    <span className="section-caption">Reference</span>
                    <span className="provider-chip">Expected Answer</span>
                  </div>
                  <div className="review-scroll-box">
                    <p className="review-block-copy whitespace-pre-wrap text-sm">{expectedAnswer}</p>
                  </div>
                </div>
              )}

              {expectedAnswer && diffParts && (
                <div className="panel-surface review-block space-y-4">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <span className="section-caption">Case Diff</span>
                      <p className="micro-copy mt-2">
                        Expected vs actual response delta for faster review.
                      </p>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      <span className="provider-chip">Removed {removedTokenCount}</span>
                      <span className="provider-chip">Added {addedTokenCount}</span>
                    </div>
                  </div>
                  <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
                    <div className="rounded-[1.1rem] border border-[rgba(199,74,74,0.18)] bg-[rgba(255,248,248,0.92)] p-4">
                      <p className="micro-copy mb-3">Expected-only changes</p>
                      <div className="review-scroll-box">
                        <DiffText parts={diffParts.expected} mode="expected" />
                      </div>
                    </div>
                    <div className="rounded-[1.1rem] border border-[rgba(62,128,96,0.18)] bg-[rgba(246,252,249,0.92)] p-4">
                      <p className="micro-copy mb-3">Actual-only changes</p>
                      <div className="review-scroll-box">
                        <DiffText parts={diffParts.actual} mode="actual" />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div className="panel-surface review-block review-block-judge space-y-3">
                <div className="flex flex-wrap items-center gap-3">
                  <span className="section-caption">Judge Panel</span>
                  <ScoreBadge score={current.llm_judge_score} />
                  <span className="provider-chip">{current.llm_judge_label}</span>
                </div>
                <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
                  <div className="rounded-[1.1rem] hairline p-4">
                    <div className="flex items-center justify-between gap-3">
                      <p className="section-caption">Primary Judge</p>
                      {current.primary_judge_score != null && <ScoreBadge score={current.primary_judge_score} />}
                    </div>
                    <p className="body-copy mt-3 font-medium">{current.primary_judge_label ?? current.llm_judge_label}</p>
                    <div className="review-judge-note mt-3">
                      <p className="micro-copy whitespace-pre-wrap">{current.llm_judge_reasoning || "(no reasoning)"}</p>
                    </div>
                  </div>

                  <div className="rounded-[1.1rem] hairline p-4">
                    <div className="flex items-center justify-between gap-3">
                      <p className="section-caption">Secondary Judge</p>
                      {current.secondary_judge_score != null ? (
                        <ScoreBadge score={current.secondary_judge_score} />
                      ) : (
                        <span className="provider-chip">Unavailable</span>
                      )}
                    </div>
                    <p className="body-copy mt-3 font-medium">{current.secondary_judge_label ?? "No split signal"}</p>
                    <div className="review-judge-note mt-3">
                      <p className="micro-copy whitespace-pre-wrap">{current.secondary_judge_reasoning || "(no secondary reasoning)"}</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="review-side motion-stagger-stack">
              {/* ── Review Action (top) ── */}
              <div className="panel-surface panel-roomy review-annotation-card space-y-4">
                <div>
                  <p className="section-caption mb-2">Review Action</p>
                  <h3 className="section-heading">Your Assessment</h3>
                </div>
                <div className="control-group">
                  <label className="label">Reviewer ID</label>
                  <input
                    value={reviewerId}
                    onChange={(e) => setReviewerId(e.target.value)}
                    placeholder="web-reviewer"
                    className="control-surface text-sm"
                  />
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <button onClick={() => handleQueueUpdate("in_progress", reviewerHandle)} disabled={claiming} className="button-secondary">
                    {claiming ? "Updating..." : "Claim Item"}
                  </button>
                  <button onClick={() => handleQueueUpdate("pending", null)} disabled={claiming} className="button-secondary">
                    Release
                  </button>
                </div>
                <div>
                  <label className="label">Score: {humanScore.toFixed(2)}</label>
                  <input type="range" min={0} max={1} step={0.05} value={humanScore} onChange={(e) => setHumanScore(Number(e.target.value))} className="control-range" />
                  <div className="mt-2 flex justify-between text-[10px] muted-copy"><span>0</span><span>0.5</span><span>1</span></div>
                </div>
                <div>
                  <label className="label">Verdict</label>
                  <div className="grid grid-cols-3 gap-2">
                    <VerdictButton active={correctionType === "approve"} onClick={() => setCorrectionType("approve")} icon={<ThumbsUp className="w-4 h-4" />} label="Approve" color="green" />
                    <VerdictButton active={correctionType === "adjust"} onClick={() => setCorrectionType("adjust")} icon={<Minus className="w-4 h-4" />} label="Adjust" color="yellow" />
                    <VerdictButton active={correctionType === "reject"} onClick={() => setCorrectionType("reject")} icon={<ThumbsDown className="w-4 h-4" />} label="Reject" color="red" />
                  </div>
                </div>
                <div className="control-group">
                  <label className="label">Feedback (optional)</label>
                  <textarea rows={3} value={feedback} onChange={(e) => setFeedback(e.target.value)} placeholder="Why this score?" className="control-surface resize-none text-sm" />
                </div>
                {shouldShowPolicyDecision && (
                  <>
                    <div className="control-group">
                      <label className="label">Policy Decision</label>
                      <select value={policyDecision} onChange={(e) => setPolicyDecision(e.target.value as PolicyDecision)} className="control-surface text-sm">
                        <option value="confirmed_violation">Confirmed violation</option>
                        <option value="false_positive">False positive</option>
                        <option value="needs_follow_up">Needs policy follow-up</option>
                      </select>
                    </div>
                    <div className="control-group">
                      <label className="label">Policy Notes</label>
                      <textarea rows={2} value={policyNotes} onChange={(e) => setPolicyNotes(e.target.value)} placeholder="Explain why this is a real violation, a false positive, or needs policy follow-up." className="control-surface resize-none text-sm" />
                    </div>
                  </>
                )}
                <label className="option-row cursor-pointer">
                  <input type="checkbox" checked={reusableMetricCandidate} onChange={(e) => setReusableMetricCandidate(e.target.checked)} className="control-check" />
                  <span>Convert to reusable metric candidate</span>
                </label>
                <button onClick={handleSubmit} disabled={submitting} className="button-primary w-full">
                  {submitting ? "Saving..." : "Submit Review"}
                </button>
              </div>

              <div className="panel-surface panel-quiet review-navigation-card flex items-center justify-between gap-3 text-xs">
                <button onClick={() => setCurrentIdx((i) => Math.max(0, i - 1))} disabled={currentIdx === 0} className="button-secondary">← Prev</button>
                <span className="micro-copy">{currentIdx + 1} / {filteredPending.length}</span>
                <button onClick={() => setCurrentIdx((i) => Math.min(filteredPending.length - 1, i + 1))} disabled={currentIdx >= filteredPending.length - 1} className="button-secondary">Next →</button>
              </div>

              {/* ── Queue Status ── */}
              <div className="panel-surface review-queue-card space-y-4">
                <div>
                  <p className="section-caption mb-2">Queue Status</p>
                  <h3 className="section-heading">Review Signal</h3>
                </div>
                <div className="review-queue-grid">
                  <div className="review-queue-mini">
                    <p className="review-queue-label">Position</p>
                    <p className="review-queue-value">{currentIdx + 1}</p>
                  </div>
                  <div className="review-queue-mini">
                    <p className="review-queue-label">Remaining</p>
                    <p className="review-queue-value">{filteredPending.length}</p>
                  </div>
                  <div className="review-queue-mini">
                    <p className="review-queue-label">Judge Score</p>
                    <p className="review-queue-value">{current.llm_judge_score.toFixed(2)}</p>
                  </div>
                  <div className="review-queue-mini">
                    <p className="review-queue-label">Split</p>
                    <p className="review-queue-value">{current.judge_disagreement != null ? current.judge_disagreement.toFixed(2) : "—"}</p>
                  </div>
                  <div className="review-queue-mini">
                    <p className="review-queue-label">Priority</p>
                    <p className="review-queue-value">{current.review_priority.toFixed(1)}</p>
                  </div>
                  <div className="review-queue-mini">
                    <p className="review-queue-label">Agreement</p>
                    <p className="review-queue-value">{current.judge_agreement != null ? `${(current.judge_agreement * 100).toFixed(0)}%` : "—"}</p>
                  </div>
                  <div className="review-queue-mini">
                    <p className="review-queue-label">Status</p>
                    <p className="review-queue-value">{current.status}</p>
                  </div>
                  <div className="review-queue-mini">
                    <p className="review-queue-label">Owner</p>
                    <p className="review-queue-value">{current.owner ?? "Unassigned"}</p>
                  </div>
                  <div className="review-queue-mini">
                    <p className="review-queue-label">SLA</p>
                    <p className="review-queue-value">{slaDueLabel}</p>
                  </div>
                  {currentSignals && (
                    <div className="review-queue-mini">
                      <p className="review-queue-label">Review Lane</p>
                      <p className="review-queue-value">{currentSignals.suggestedReviewerPersona.toUpperCase()}</p>
                    </div>
                  )}
                  {currentSignals?.casePersona && (
                    <div className="review-queue-mini">
                      <p className="review-queue-label">Case Persona</p>
                      <p className="review-queue-value">{currentSignals.casePersona}</p>
                    </div>
                  )}
                  {currentSignals?.promptVersion && (
                    <div className="review-queue-mini">
                      <p className="review-queue-label">Prompt Version</p>
                      <p className="review-queue-value">{currentSignals.promptVersion}</p>
                    </div>
                  )}
                </div>
                <div className="rounded-[1rem] callout-warn px-3 py-3">
                  <div className="flex items-start gap-2">
                    <AlertTriangle className="mt-0.5 h-4 w-4 callout-warn-icon" />
                    <div>
                      <p className="micro-copy">Queue Reason</p>
                      <p className="body-copy mt-1 text-sm">{current.queue_reason}</p>
                      {sourceReport && <p className="micro-copy mt-2">Source: {sourceReport}</p>}
                      {currentSignals?.riskTags && currentSignals.riskTags.length > 0 && (
                        <div className="mt-3 flex flex-wrap gap-2">
                          {currentSignals.riskTags.slice(0, 4).map((tag) => (
                            <span key={`${current.item_id}-${tag}`} className="provider-chip">{tag}</span>
                          ))}
                        </div>
                      )}
                      {currentSignals?.escalationNeeded && <p className="micro-copy mt-2">Escalation path expected for this case.</p>}
                    </div>
                  </div>
                </div>
              </div>

              {currentSignals && currentSuggestions.length > 0 && (
                <div className="panel-surface panel-quiet space-y-4">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="section-caption mb-2">Metric Suggestion</p>
                      <h3 className="section-heading">Case-to-metric next step</h3>
                    </div>
                    <Sparkles className="h-4 w-4 muted-copy" />
                  </div>
                  <div className="rounded-[1rem] hairline px-4 py-3">
                    <div className="flex flex-wrap gap-2">
                      <span className="provider-chip">lane {currentSignals.suggestedReviewerPersona.toUpperCase()}</span>
                      {currentSignals.isHighRisk && <span className="provider-chip">high risk</span>}
                      {(current.judge_disagreement ?? 0) >= disagreementThreshold && <span className="provider-chip">split above cutoff</span>}
                    </div>
                    <p className="micro-copy mt-3">
                      This panel shows the three closest candidate directions to turn the current case into a reusable metric.
                    </p>
                  </div>
                  <div className="space-y-3">
                    {currentSuggestions.map((suggestion) => (
                      <div key={suggestion.key} className="rounded-[1rem] hairline px-4 py-3">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="provider-chip">{suggestion.metricFamily}</span>
                          <span className="provider-chip">{suggestion.candidateName}</span>
                        </div>
                        <p className="body-copy mt-3 text-sm font-semibold">{suggestion.title}</p>
                        <p className="micro-copy mt-2 whitespace-pre-wrap">{suggestion.rationale}</p>
                        <div className="mt-3 flex flex-wrap gap-2">
                          {suggestion.evidence.map((evidence) => (
                            <span key={`${suggestion.key}-${evidence}`} className="provider-chip">{evidence}</span>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

            </div>
          </div>
        </>
      ) : (
        <div className="panel-surface panel-quiet empty-state motion-rise motion-delay-7">
          <BarChart3 className="mb-3 h-12 w-12 opacity-40" />
          <p className="body-copy text-sm">No disagreement items are waiting for review.</p>
          <p className="micro-copy mt-1">
            Auto-queued cases from new runs appear here. Use backfill for older reports.
          </p>
        </div>
      )}

      <div className="panel-surface panel-quiet motion-rise motion-delay-7 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="section-caption mb-2">Trace Queue</p>
            <h2 className="section-heading">Online-Sampled Traces</h2>
            <p className="page-subtitle mt-1 text-sm">
              Traces tagged <code>eval_sampled</code> by the online sampler — ready for human spot-check.
            </p>
          </div>
          <span className="provider-chip">{traceCandidates.length} sampled</span>
        </div>
        {traceCandidates.length === 0 ? (
          <p className="micro-copy">No sampled traces yet. Ingest traces and the online sampler will queue candidates here.</p>
        ) : (
          <div className="space-y-2">
            {traceCandidates.map((trace) => (
              <div
                key={trace.trace_id}
                className="rounded-[1rem] border border-[rgba(124,58,237,0.16)] px-4 py-3"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="min-w-0 flex-1">
                    <span className="block truncate text-sm font-semibold text-[#e5e7eb]">{trace.name}</span>
                    <span className="micro-copy font-mono">{trace.trace_id.slice(0, 20)}…</span>
                  </div>
                  <div className="flex shrink-0 flex-wrap items-center gap-2">
                    <span className="provider-chip">{trace.spans.length} spans</span>
                    {trace.tags.filter((t) => t !== "eval_sampled").map((tag) => (
                      <span key={tag} className="provider-chip">{tag}</span>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="stat-card">
      <p className="stat-label">{label}</p>
      <p className="stat-value">{value}</p>
    </div>
  );
}

function ScoreBadge({ score }: { score: number }) {
  const color =
    score >= 0.8
      ? "score-badge score-badge-good"
      : score >= 0.5
        ? "score-badge score-badge-mid"
        : "score-badge score-badge-low";
  return <span className={color}>{score.toFixed(3)}</span>;
}

function VerdictButton({
  active,
  onClick,
  icon,
  label,
  color,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
  color: "green" | "yellow" | "red";
}) {
  const colors = {
    green: active
      ? "verdict-green-active"
      : "",
    yellow: active
      ? "verdict-yellow-active"
      : "",
    red: active
      ? "verdict-red-active"
      : "",
  };

  return (
    <button
      onClick={onClick}
      className={`verdict-button text-xs ${colors[color]}`.trim()}
    >
      {icon}
      {label}
    </button>
  );
}
