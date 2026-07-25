import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { AlertTriangle, BarChart3, Play, Sparkles, Waypoints, Zap } from "lucide-react";
import { EmptyState } from "@/components";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  resultsApi,
  type EvaluatorEfficiencyRow,
  type ReportCompareSummary,
  type ReportListItem,
  type ReportSummary,
  type TokenEfficiencyPoint,
} from "@/api/client";

function ScoreBadge({
  score,
  variant = "score",
}: {
  score: number;
  variant?: "score" | "disagreement";
}) {
  const tone =
    variant === "disagreement"
      ? score >= 0.8
        ? "score-badge score-badge-low"
        : score >= 0.5
          ? "score-badge score-badge-mid"
          : "score-badge score-badge-good"
      : score >= 0.8
        ? "score-badge score-badge-good"
        : score >= 0.5
          ? "score-badge score-badge-mid"
          : "score-badge score-badge-low";

  return <span className={tone}>{score.toFixed(3)}</span>;
}

function formatCount(value?: number | null) {
  if (value == null || Number.isNaN(value)) return "—";
  return value.toLocaleString("tr-TR");
}

function formatMetric(value?: number | null, digits = 2) {
  if (value == null || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

// 95% CI of the mean per-case judge score (normal approximation for display;
// the CI gate itself uses bootstrap server-side)
function judgeScoreCi(testResult: unknown): [number, number, number] | null {
  const results = (testResult as Record<string, unknown> | null)?.results;
  if (!Array.isArray(results)) return null;
  const scores: number[] = [];
  for (const item of results) {
    const value = ((item as Record<string, unknown>)?.scores as Record<string, unknown> | undefined)?.judge_score;
    if (typeof value === "number" && !Number.isNaN(value)) scores.push(value);
  }
  const n = scores.length;
  if (n < 2) return null;
  const mean = scores.reduce((a, b) => a + b, 0) / n;
  const sd = Math.sqrt(scores.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1));
  const half = (1.96 * sd) / Math.sqrt(n);
  return [Math.max(0, mean - half), Math.min(1, mean + half), n];
}

function formatTimestamp(value?: string | null) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("tr-TR");
}

function humanizePolicyDecision(value?: string | null) {
  if (!value) return "Unknown";
  return value.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function average(values: Array<number | null | undefined>) {
  const valid = values.filter((value): value is number => typeof value === "number" && !Number.isNaN(value));
  if (!valid.length) return null;
  return valid.reduce((sum, value) => sum + value, 0) / valid.length;
}

function getLeanestModel(points: TokenEfficiencyPoint[]) {
  return points.reduce<TokenEfficiencyPoint | null>((leanest, point) => {
    if (point.avg_tokens_per_eval == null) return leanest;
    if (!leanest || leanest.avg_tokens_per_eval == null) return point;
    return point.avg_tokens_per_eval < leanest.avg_tokens_per_eval ? point : leanest;
  }, null);
}

function getStrongestFrontier(points: TokenEfficiencyPoint[]) {
  return points.reduce<TokenEfficiencyPoint | null>((best, point) => {
    if (!point.frontier) return best;
    if (!best) return point;
    if (point.overall_score !== best.overall_score) {
      return point.overall_score > best.overall_score ? point : best;
    }
    return (point.quality_per_1k_tokens ?? 0) > (best.quality_per_1k_tokens ?? 0) ? point : best;
  }, null);
}

function EfficiencyTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: TokenEfficiencyPoint }>;
}) {
  const point = payload?.[0]?.payload;
  if (!active || !point) return null;

  return (
    <div className="panel-surface panel-quiet max-w-xs space-y-2 hairline p-3">
      <p className="body-copy font-semibold">{point.model}</p>
      <p className="micro-copy">Score: {point.overall_score.toFixed(3)}</p>
      <p className="micro-copy">Avg tokens / eval: {formatMetric(point.avg_tokens_per_eval, 1)}</p>
      <p className="micro-copy">Quality / 1K tokens: {formatMetric(point.quality_per_1k_tokens, 2)}</p>
    </div>
  );
}

type ResultsTableTestSummary = {
  summary?: {
    overall_score?: number;
    total_tests?: number;
    avg_scores?: Record<string, number | null | undefined>;
    unresolved_intent_summary?: {
      unresolved_turn_rate?: number | null;
    };
  };
};

type ContinuityModelSummary = {
  model: string;
  intent_resolution?: number | null;
  unresolved_turn_rate?: number | null;
  unresolved_turns?: number | null;
  unresolved_intent_total?: number | null;
};

type ContinuitySummary = {
  by_model: ContinuityModelSummary[];
  best_intent_resolution_model?: string | null;
  highest_unresolved_rate_model?: string | null;
};

type StructuredOutputBreakdownEntry = {
  total_cases?: number;
  valid_cases?: number;
  invalid_cases?: number;
};

type StructuredOutputReliability = {
  total_cases?: number;
  invalid_cases?: number;
  schema_compliance_rate?: number;
  dataset_breakdown?: Record<string, StructuredOutputBreakdownEntry>;
  schema_type_breakdown?: Record<string, StructuredOutputBreakdownEntry>;
  test_breakdown?: Record<string, StructuredOutputBreakdownEntry>;
};

type ModelComparisonEntry = {
  overall_score?: number | null;
  avg_latency?: number | null;
  latency_p95?: number | null;
  total_cost?: number | null;
  total_tokens?: number | null;
  error_rate?: number | null;
  quality_latency_efficiency?: number | null;
  structured_output_reliability?: StructuredOutputReliability | null;
};

type TrendPoint = {
  label: string;
  value: number;
};

type ModelTrendRow = {
  model: string;
  trendLabel: string;
  changePct: number | null;
  historyRuns: number;
  regressions: number;
  points: TrendPoint[];
};

type RawCaseStatus = {
  testName: string;
  caseId: string;
  failed: boolean;
  score: number | null;
  reason: string;
};

type DatasetSignature = {
  name: string | null;
  path: string | null;
  itemCount: number | null;
  labels: string[];
};

type ProviderCostRow = {
  provider: string;
  totalCost: number;
  totalTokens: number;
  modelCount: number;
  modelNames: string[];
  costShare: number;
  avgCostPerModel: number;
  costPer1kTokens: number | null;
};

type ConversationTurnExplorer = {
  turnNumber: number;
  userMessage: string;
  assistantResponse: string;
  expectedCheck: string;
  expectedActions: string[];
  evaluationWindow: string[];
  windowReference: string;
  windowSize: number | null;
  latency: number | null;
  unresolvedIntents: string[];
  unresolvedIntentCount: number;
  hasUnresolvedIntent: boolean;
  retrievalContext: string | null;
  groundednessScore: number | null;
  groundednessReason: string;
  relevancyScore: number | null;
  knowledgeScore: number | null;
  structuredOutputValid: boolean | null;
  jsonCorrectnessScore: number | null;
  promptAlignmentScore: number | null;
  compositeScore: number | null;
  failed: boolean;
};

type ConversationExplorerRow = {
  key: string;
  model: string;
  testName: string;
  caseId: string;
  category: string;
  overallScore: number | null;
  intentResolution: number | null;
  unresolvedTurns: number;
  unresolvedIntentTotal: number;
  avgTurnLatency: number | null;
  turns: ConversationTurnExplorer[];
  noteCandidates: string[];
};

type TraceSpanRow = {
  spanId: string;
  parentSpanId: string | null;
  spanType: string;
  name: string;
  status: string;
  durationMs: number | null;
  inputSummary: string;
  outputSummary: string;
  error: string;
  metricScore: number | null;
  reasoning: string;
  metadata: Record<string, unknown>;
  rawMetrics: unknown[];
  depth: number;
  hasChildren: boolean;
};

type TraceExplorerRow = {
  key: string;
  model: string;
  testName: string;
  caseId: string;
  category: string;
  traceId: string;
  totalSpans: number;
  failedSpans: number;
  durationMs: number | null;
  topReasoning: string;
  spans: TraceSpanRow[];
};

function asRecord(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  return value as Record<string, unknown>;
}

function firstText(...values: unknown[]) {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return "";
}

function extractRawCaseScore(result: Record<string, unknown>) {
  const scores = asRecord(result.scores);
  const candidates = [
    result.overall_score,
    result.score,
    scores.overall_score,
    scores.score,
    result.llm_judge_score,
  ];

  for (const candidate of candidates) {
    if (typeof candidate === "number") return candidate;
  }

  for (const booleanKey of ["passed", "success"] as const) {
    const candidate = result[booleanKey];
    if (typeof candidate === "boolean") return candidate ? 1 : 0;
  }

  return null;
}

function extractRawCaseReason(result: Record<string, unknown>) {
  const details = asRecord(result.details);
  const metadata = asRecord(result.metadata);
  return firstText(
    result.error,
    result.reason,
    result.reasoning,
    result.llm_judge_reasoning,
    details.reason,
    details.reasoning,
    metadata.queue_reason
  );
}

function isFailingRawCase(result: Record<string, unknown>) {
  if (typeof result.error === "string" && result.error.trim()) return true;
  if (typeof result.passed === "boolean") return result.passed === false;
  if (typeof result.success === "boolean") return result.success === false;

  const score = extractRawCaseScore(result);
  return typeof score === "number" ? score < 0.5 : false;
}

function buildRawCaseIndex(rawReport: Record<string, unknown> | null | undefined) {
  const perModel: Record<string, Record<string, RawCaseStatus>> = {};
  const models = asRecord(rawReport?.models);

  for (const [model, modelPayload] of Object.entries(models)) {
    const tests = asRecord(asRecord(modelPayload).tests);
    const caseIndex: Record<string, RawCaseStatus> = {};

    for (const [testName, testPayload] of Object.entries(tests)) {
      const results = asRecord(testPayload).results;
      if (!Array.isArray(results)) continue;

      results.forEach((rawResult, index) => {
        const result = asRecord(rawResult);
        const caseId =
          firstText(result.id, result.test_id, result.case_id) || `${testName}-${index + 1}`;
        const key = `${testName}::${caseId}`;
        caseIndex[key] = {
          testName,
          caseId,
          failed: isFailingRawCase(result),
          score: extractRawCaseScore(result),
          reason: extractRawCaseReason(result),
        };
      });
    }

    perModel[model] = caseIndex;
  }

  return perModel;
}

function extractDatasetSignature(rawReport: Record<string, unknown> | null | undefined): DatasetSignature {
  const runMetadata = asRecord(rawReport?.run_metadata);
  const customDataset = asRecord(runMetadata.custom_dataset);
  const labelSet = new Set<string>();
  const models = asRecord(rawReport?.models);

  for (const modelPayload of Object.values(models)) {
    const tests = asRecord(asRecord(modelPayload).tests);
    for (const testPayload of Object.values(tests)) {
      const metadata = asRecord(asRecord(testPayload).metadata);
      const datasetLabel = metadata.dataset_label;
      if (typeof datasetLabel === "string" && datasetLabel.trim()) {
        labelSet.add(datasetLabel.trim());
      }
    }
  }

  return {
    name: typeof customDataset.name === "string" ? customDataset.name : null,
    path: typeof customDataset.path === "string" ? customDataset.path : null,
    itemCount: typeof customDataset.item_count === "number" ? customDataset.item_count : null,
    labels: Array.from(labelSet).sort(),
  };
}

function tokenizeText(value: string) {
  return Array.from(
    new Set(
      value
        .toLocaleLowerCase("tr-TR")
        .split(/[^\p{L}\p{N}]+/u)
        .filter((token) => token.length > 2)
    )
  );
}

function overlapScore(source: unknown, response: unknown) {
  if (typeof source !== "string" || typeof response !== "string") return null;
  const sourceTokens = tokenizeText(source);
  const responseTokens = new Set(tokenizeText(response));
  if (!sourceTokens.length || !responseTokens.size) return null;
  const matches = sourceTokens.filter((token) => responseTokens.has(token)).length;
  return matches / sourceTokens.length;
}

function extractMetricPayloadScore(value: unknown) {
  const payload = asRecord(value);
  const nested = asRecord(payload.raw_payload);
  const candidates = [payload.normalized_score, payload.score, nested.normalized_score, nested.score];
  for (const candidate of candidates) {
    if (typeof candidate === "number" && !Number.isNaN(candidate)) return candidate;
  }
  return null;
}

function extractRetrievalContextText(value: unknown): string | null {
  if (typeof value === "string" && value.trim()) return value.trim();
  if (Array.isArray(value)) {
    const parts = value
      .map((item) => extractRetrievalContextText(item))
      .filter((item): item is string => typeof item === "string" && Boolean(item));
    return parts.length ? parts.join("\n\n") : null;
  }

  const payload = asRecord(value);
  for (const key of ["content", "text", "chunk", "passage", "context"]) {
    const candidate = payload[key];
    if (typeof candidate === "string" && candidate.trim()) {
      return candidate.trim();
    }
  }

  return null;
}

function extractTurnRetrievalContext(turnPayload: Record<string, unknown>) {
  for (const source of [turnPayload, asRecord(turnPayload.metadata)]) {
    for (const key of [
      "retrieval_context",
      "retrieved_context",
      "retrievalContexts",
      "retrievedContexts",
      "grounding_context",
      "context",
      "contexts",
    ]) {
      const contextText = extractRetrievalContextText(source[key]);
      if (contextText) {
        return contextText;
      }
    }
  }
  return null;
}

function summarizeConversationNotes(turns: ConversationTurnExplorer[]) {
  const notes: string[] = [];
  const unresolvedTurns = turns.filter((turn) => turn.hasUnresolvedIntent).map((turn) => turn.turnNumber);
  const lowFaithfulnessTurns = turns
    .filter((turn) => typeof turn.groundednessScore === "number" && turn.groundednessScore < 0.5)
    .map((turn) => turn.turnNumber);
  const schemaFailureTurns = turns
    .filter((turn) => turn.structuredOutputValid === false)
    .map((turn) => turn.turnNumber);
  const missingContextTurns = turns
    .filter((turn) => !turn.retrievalContext && turn.groundednessScore == null)
    .map((turn) => turn.turnNumber);

  if (unresolvedTurns.length > 0) {
    notes.push(`Open intent remains on turns ${unresolvedTurns.join(", ")}.`);
  }
  if (lowFaithfulnessTurns.length > 0) {
    notes.push(`Faithfulness reasoning is weak on turns ${lowFaithfulnessTurns.join(", ")}.`);
  }
  if (schemaFailureTurns.length > 0) {
    notes.push(`Structured output validation failed on turns ${schemaFailureTurns.join(", ")}.`);
  }
  if (missingContextTurns.length > 0) {
    notes.push(`Turns ${missingContextTurns.join(", ")} have no retrieval context attached for grounded review.`);
  }
  if (!notes.length) {
    notes.push("Conversation is mostly stable; reviewer can focus on transcript quality and escalation tone.");
  }

  return notes.slice(0, 4);
}

function buildConversationExplorerRows(rawReport: Record<string, unknown> | null | undefined) {
  const rows: ConversationExplorerRow[] = [];
  const models = asRecord(rawReport?.models);

  for (const [model, modelPayload] of Object.entries(models)) {
    const tests = asRecord(asRecord(modelPayload).tests);
    for (const [testName, testPayload] of Object.entries(tests)) {
      const results = asRecord(testPayload).results;
      if (!Array.isArray(results)) continue;

      results.forEach((rawResult, index) => {
        const result = asRecord(rawResult);
        const turns = result.turns;
        if (!Array.isArray(turns) || turns.length === 0) return;

        const turnRows = turns.map((rawTurn, turnIndex) => {
          const turn = asRecord(rawTurn);
          const responseText = firstText(turn.assistant_response);
          const userMessage = firstText(turn.user_message);
          const expectedCheck = firstText(turn.expected_check);
          const windowReference = firstText(turn.window_reference);
          const evaluationWindow = Array.isArray(turn.evaluation_window)
            ? turn.evaluation_window
                .map((windowTurn) => {
                  const payload = asRecord(windowTurn);
                  return (
                    firstText(payload.user_message, payload.content, payload.expected_check) ||
                    JSON.stringify(payload)
                  );
                })
                .filter(Boolean)
            : [];
          const unresolvedIntents = Array.isArray(turn.unresolved_intents)
            ? turn.unresolved_intents.filter((item): item is string => typeof item === "string" && item.trim().length > 0)
            : [];
          const groundedness = asRecord(turn.groundedness);
          const groundednessScore =
            typeof groundedness.normalized_score === "number" ? groundedness.normalized_score : null;
          const structuredOutput = asRecord(turn.structured_output);
          const structuredOutputValid =
            typeof structuredOutput.is_valid === "boolean"
              ? structuredOutput.is_valid
              : structuredOutput.parse_error || structuredOutput.schema_error
                ? false
                : null;
          const relevancyScore =
            [
              overlapScore(userMessage, responseText),
              overlapScore(expectedCheck, responseText),
              overlapScore(windowReference, responseText),
            ].reduce<number | null>((best, value) => {
              if (value == null) return best;
              return best == null ? value : Math.max(best, value);
            }, null);
          const knowledgeScore = turnIndex === 0 ? (responseText ? 1 : 0) : overlapScore(windowReference, responseText);
          const jsonCorrectnessScore = extractMetricPayloadScore(turn.json_correctness);
          const promptAlignmentScore = extractMetricPayloadScore(turn.prompt_alignment);
          const compositeParts = [
            relevancyScore,
            knowledgeScore,
            groundednessScore,
            jsonCorrectnessScore,
            promptAlignmentScore,
          ].filter((value): value is number => typeof value === "number" && !Number.isNaN(value));
          const compositeScore = compositeParts.length
            ? compositeParts.reduce((sum, value) => sum + value, 0) / compositeParts.length
            : null;
          const failed =
            unresolvedIntents.length > 0 ||
            !responseText ||
            structuredOutputValid === false ||
            (typeof groundednessScore === "number" && groundednessScore < 0.5) ||
            (typeof compositeScore === "number" && compositeScore < 0.5);

          return {
            turnNumber: typeof turn.turn === "number" ? turn.turn : turnIndex + 1,
            userMessage,
            assistantResponse: responseText,
            expectedCheck,
            expectedActions: Array.isArray(turn.expected_actions)
              ? turn.expected_actions.filter((item): item is string => typeof item === "string" && item.trim().length > 0)
              : [],
            evaluationWindow,
            windowReference,
            windowSize: typeof turn.window_size === "number" ? turn.window_size : null,
            latency: typeof turn.latency === "number" ? turn.latency : null,
            unresolvedIntents,
            unresolvedIntentCount:
              typeof turn.unresolved_intent_count === "number" ? turn.unresolved_intent_count : unresolvedIntents.length,
            hasUnresolvedIntent: turn.has_unresolved_intent === true || unresolvedIntents.length > 0,
            retrievalContext: extractTurnRetrievalContext(turn),
            groundednessScore,
            groundednessReason: firstText(groundedness.reasoning, groundedness.result),
            relevancyScore,
            knowledgeScore,
            structuredOutputValid,
            jsonCorrectnessScore,
            promptAlignmentScore,
            compositeScore,
            failed,
          } satisfies ConversationTurnExplorer;
        });

        const scores = asRecord(result.scores);
        const numericScores = Object.values(scores).filter(
          (value): value is number => typeof value === "number" && !Number.isNaN(value)
        );
        const caseId = firstText(result.id, result.test_id, result.case_id) || `${testName}-${index + 1}`;

        rows.push({
          key: `${model}::${testName}::${caseId}`,
          model,
          testName,
          caseId,
          category: firstText(result.category, result.test_category) || "conversation",
          overallScore: numericScores.length ? numericScores.reduce((sum, value) => sum + value, 0) / numericScores.length : null,
          intentResolution: typeof scores.intent_resolution === "number" ? scores.intent_resolution : null,
          unresolvedTurns:
            typeof asRecord(result.unresolved_intent_summary).unresolved_turns === "number"
              ? (asRecord(result.unresolved_intent_summary).unresolved_turns as number)
              : turnRows.filter((turn) => turn.hasUnresolvedIntent).length,
          unresolvedIntentTotal:
            typeof asRecord(result.unresolved_intent_summary).unresolved_intent_total === "number"
              ? (asRecord(result.unresolved_intent_summary).unresolved_intent_total as number)
              : turnRows.reduce((sum, turn) => sum + turn.unresolvedIntentCount, 0),
          avgTurnLatency:
            typeof result.avg_turn_latency === "number"
              ? result.avg_turn_latency
              : average(turnRows.map((turn) => turn.latency)),
          turns: turnRows,
          noteCandidates: summarizeConversationNotes(turnRows),
        });
      });
    }
  }

  return rows.sort((left, right) => {
    if (right.unresolvedTurns !== left.unresolvedTurns) return right.unresolvedTurns - left.unresolvedTurns;
    return (left.overallScore ?? 1) - (right.overallScore ?? 1);
  });
}

function extractMetricScoreList(value: unknown) {
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => {
      const payload = asRecord(item);
      const candidates = [payload.normalized_value, payload.normalized_score, payload.score, payload.value];
      for (const candidate of candidates) {
        if (typeof candidate === "number" && !Number.isNaN(candidate)) {
          return candidate;
        }
      }
      return null;
    })
    .filter((item): item is number => typeof item === "number" && !Number.isNaN(item));
}

function buildTraceExplorerRows(rawReport: Record<string, unknown> | null | undefined) {
  const rows: TraceExplorerRow[] = [];
  const models = asRecord(rawReport?.models);

  for (const [model, modelPayload] of Object.entries(models)) {
    const tests = asRecord(asRecord(modelPayload).tests);
    for (const [testName, testPayload] of Object.entries(tests)) {
      const results = asRecord(testPayload).results;
      if (!Array.isArray(results)) continue;

      results.forEach((rawResult, index) => {
        const result = asRecord(rawResult);
        const trace = asRecord(result.trace);
        const rawSpans = trace.spans;
        if (!Array.isArray(rawSpans) || rawSpans.length === 0) return;

        const parentToChildren = new Map<string, string[]>();
        rawSpans.forEach((rawSpan) => {
          const span = asRecord(rawSpan);
          const parentId = firstText(span.parent_span_id);
          const spanId = firstText(span.span_id);
          if (!parentId || !spanId) return;
          parentToChildren.set(parentId, [...(parentToChildren.get(parentId) ?? []), spanId]);
        });

        const depthBySpanId = new Map<string, number>();
        const resolveDepth = (spanId: string, parentSpanId: string | null): number => {
          if (depthBySpanId.has(spanId)) {
            return depthBySpanId.get(spanId) ?? 0;
          }
          if (!parentSpanId) {
            depthBySpanId.set(spanId, 0);
            return 0;
          }

          const parentSpan = rawSpans.find((entry) => firstText(asRecord(entry).span_id) === parentSpanId);
          const parentDepth = parentSpan
            ? resolveDepth(parentSpanId, firstText(asRecord(parentSpan).parent_span_id) || null)
            : 0;
          const depth = parentDepth + 1;
          depthBySpanId.set(spanId, depth);
          return depth;
        };

        const spans = rawSpans.map((rawSpan) => {
          const span = asRecord(rawSpan);
          const spanId = firstText(span.span_id) || `span-${index}`;
          const parentSpanId = firstText(span.parent_span_id) || null;
          const metricScores = extractMetricScoreList(span.metric_results);
          const metadata = asRecord(span.metadata);
          const reasoning =
            firstText(
              metadata.primary_reasoning,
              metadata.reasoning,
              ...((Array.isArray(span.metric_results)
                ? span.metric_results.map((item) => asRecord(item).reason)
                : []) as unknown[])
            ) || "";

          return {
            spanId,
            parentSpanId,
            spanType: firstText(span.span_type).toUpperCase() || "UNKNOWN",
            name: firstText(span.name) || spanId,
            status: firstText(span.status) || "unknown",
            durationMs: typeof span.duration_ms === "number" ? span.duration_ms : null,
            inputSummary: firstText(span.input_summary),
            outputSummary: firstText(span.output_summary),
            error: firstText(span.error),
            metricScore: metricScores.length ? metricScores.reduce((sum, value) => sum + value, 0) / metricScores.length : null,
            reasoning,
            metadata,
            rawMetrics: Array.isArray(span.metric_results) ? span.metric_results : [],
            depth: resolveDepth(spanId, parentSpanId),
            hasChildren: parentToChildren.has(spanId),
          } satisfies TraceSpanRow;
        });

        const traceSummary = asRecord(trace.summary);
        const failedSpans =
          typeof traceSummary.failed_spans === "number"
            ? traceSummary.failed_spans
            : spans.filter((span) => span.status === "failed").length;
        const caseId = firstText(result.id, result.test_id, result.case_id) || `${testName}-${index + 1}`;
        const topReasoning =
          spans.find((span) => span.status === "failed" && span.reasoning)?.reasoning ||
          spans.find((span) => span.reasoning)?.reasoning ||
          "No trace reasoning attached.";

        rows.push({
          key: `${model}::${testName}::${caseId}`,
          model,
          testName,
          caseId,
          category: firstText(result.category, result.test_category) || "agentic",
          traceId: firstText(trace.trace_id) || caseId,
          totalSpans:
            typeof traceSummary.total_spans === "number" ? traceSummary.total_spans : spans.length,
          failedSpans,
          durationMs:
            typeof spans[0]?.durationMs === "number"
              ? spans[0].durationMs
              : average(spans.map((span) => span.durationMs)),
          topReasoning,
          spans,
        });
      });
    }
  }

  return rows.sort((left, right) => {
    if (right.failedSpans !== left.failedSpans) return right.failedSpans - left.failedSpans;
    return right.totalSpans - left.totalSpans;
  });
}

function buildProviderCostRows(
  modelsValue: unknown,
  modelComparisons: Record<string, ModelComparisonEntry>
): ProviderCostRow[] {
  const grouped = Object.entries(asRecord(modelsValue)).reduce<
    Record<string, Omit<ProviderCostRow, "costShare" | "avgCostPerModel" | "costPer1kTokens">>
  >((accumulator, [model, payload]) => {
    const providerValue = asRecord(payload).provider;
    const provider = typeof providerValue === "string" && providerValue.trim() ? providerValue : "unknown";
    const comparison = modelComparisons[model];
    const totalCost = comparison?.total_cost;

    if (typeof totalCost !== "number" || Number.isNaN(totalCost)) {
      return accumulator;
    }

    const current = accumulator[provider] ?? {
      provider,
      totalCost: 0,
      totalTokens: 0,
      modelCount: 0,
      modelNames: [],
    };

    current.totalCost += totalCost;
    current.totalTokens += typeof comparison?.total_tokens === "number" ? comparison.total_tokens : 0;
    current.modelCount += 1;
    current.modelNames.push(model);
    accumulator[provider] = current;
    return accumulator;
  }, {});

  const totalProviderCost = Object.values(grouped).reduce((sum, row) => sum + row.totalCost, 0);

  return Object.values(grouped)
    .map<ProviderCostRow>((row) => ({
      ...row,
      costShare: totalProviderCost > 0 ? row.totalCost / totalProviderCost : 0,
      avgCostPerModel: row.modelCount > 0 ? row.totalCost / row.modelCount : 0,
      costPer1kTokens: row.totalTokens > 0 ? (row.totalCost * 1000) / row.totalTokens : null,
    }))
    .sort((left, right) => right.totalCost - left.totalCost);
}

function normalizeTrendRows(value: unknown): ModelTrendRow[] {
  const trends = asRecord(value);

  return Object.entries(trends)
    .map(([model, payload]) => {
      const payloadRecord = asRecord(payload);
      const trendPayload = asRecord(payloadRecord.trend);
      const regressions = payloadRecord.regressions;
      const values = Array.isArray(trendPayload.values)
        ? trendPayload.values.filter((item): item is number => typeof item === "number")
        : [];
      const timestamps = Array.isArray(trendPayload.timestamps)
        ? trendPayload.timestamps.filter((item): item is string => typeof item === "string")
        : [];

      const points = values
        .map((valueItem, index) => ({
          value: valueItem,
          label: timestamps[index] ? timestamps[index].slice(0, 10) : `Run ${index + 1}`,
        }))
        .reverse();

      if (points.length === 0) return null;

      return {
        model,
        trendLabel:
          typeof trendPayload.trend === "string" ? trendPayload.trend : "insufficient_history",
        changePct: typeof trendPayload.change_pct === "number" ? trendPayload.change_pct : null,
        historyRuns: typeof trendPayload.history_runs === "number" ? trendPayload.history_runs : 0,
        regressions: Array.isArray(regressions) ? regressions.length : 0,
        points,
      };
    })
    .filter((row): row is ModelTrendRow => Boolean(row));
}

function normalizeContinuitySummary(value: unknown): ContinuitySummary | null {
  if (!value || typeof value !== "object") return null;
  const payload = value as Record<string, unknown>;
  const byModel = Array.isArray(payload.by_model)
    ? payload.by_model
        .filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object")
        .map((item) => ({
          model: typeof item.model === "string" ? item.model : "unknown",
          intent_resolution:
            typeof item.intent_resolution === "number" ? item.intent_resolution : null,
          unresolved_turn_rate:
            typeof item.unresolved_turn_rate === "number" ? item.unresolved_turn_rate : null,
          unresolved_turns: typeof item.unresolved_turns === "number" ? item.unresolved_turns : null,
          unresolved_intent_total:
            typeof item.unresolved_intent_total === "number" ? item.unresolved_intent_total : null,
        }))
    : [];

  if (byModel.length === 0) return null;

  return {
    by_model: byModel,
    best_intent_resolution_model:
      typeof payload.best_intent_resolution_model === "string"
        ? payload.best_intent_resolution_model
        : null,
    highest_unresolved_rate_model:
      typeof payload.highest_unresolved_rate_model === "string"
        ? payload.highest_unresolved_rate_model
        : null,
  };
}

export default function Results() {
  const [reports, setReports] = useState<ReportListItem[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [report, setReport] = useState<ReportSummary | null>(null);
  const [selectedRawReport, setSelectedRawReport] = useState<Record<string, unknown> | null>(null);
  const [selectedConversationKey, setSelectedConversationKey] = useState("");
  const [selectedTraceKey, setSelectedTraceKey] = useState("");
  const [compareSelection, setCompareSelection] = useState<string[]>([]);
  const [baselineFilename, setBaselineFilename] = useState<string | null>(null);
  const [regressionThreshold, setRegressionThreshold] = useState(0.02);
  const [compareData, setCompareData] = useState<Record<string, ReportCompareSummary> | null>(null);
  const [compareRawReports, setCompareRawReports] = useState<Record<string, Record<string, unknown> | null> | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareError, setCompareError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [reportsLoaded, setReportsLoaded] = useState(false);

  useEffect(() => {
    resultsApi
      .listReports()
      .then(setReports)
      .finally(() => setReportsLoaded(true));
  }, []);

  useEffect(() => {
    if (!selected) {
      setReport(null);
      setSelectedRawReport(null);
      return;
    }
    setLoading(true);
    resultsApi
      .getReport(selected)
      .then(setReport)
      .finally(() => setLoading(false));
  }, [selected]);

  useEffect(() => {
    if (!selected) {
      setSelectedRawReport(null);
      return;
    }

    let cancelled = false;
    resultsApi
      .getRaw(selected)
      .then((raw) => {
        if (!cancelled) {
          setSelectedRawReport(raw);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setSelectedRawReport(null);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [selected]);

  useEffect(() => {
    if (compareSelection.length < 2) {
      setCompareData(null);
      setCompareRawReports(null);
      setCompareLoading(false);
      setCompareError(null);
      return;
    }

    let cancelled = false;
    setCompareLoading(true);
    setCompareError(null);
    Promise.all([
      resultsApi.compare(compareSelection),
      Promise.all(
        compareSelection.map(async (filename) => [
          filename,
          await resultsApi.getRaw(filename).catch(() => null),
        ] as const)
      ),
    ])
      .then(([summary, rawEntries]) => {
        if (!cancelled) {
          setCompareData(summary);
          setCompareRawReports(Object.fromEntries(rawEntries));
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setCompareData(null);
          setCompareRawReports(null);
          setCompareError(error?.message || "Failed to load compare data");
        }
      })
      .finally(() => {
        if (!cancelled) {
          setCompareLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [compareSelection]);

  useEffect(() => {
    if (compareSelection.length < 2) {
      setBaselineFilename(null);
      return;
    }

    setBaselineFilename((current) => {
      if (current && compareSelection.includes(current)) {
        return current;
      }
      return compareSelection[0] ?? null;
    });
  }, [compareSelection]);

  const efficiencyLeaderboard = report?.efficiency?.leaderboard ?? [];
  const evaluatorEfficiencyRows: EvaluatorEfficiencyRow[] = report?.efficiency?.evaluator_breakdown ?? [];
  const leanestModel = getLeanestModel(efficiencyLeaderboard);
  const strongestFrontier = getStrongestFrontier(efficiencyLeaderboard);
  const hasEfficiencyData = efficiencyLeaderboard.some(
    (point) => point.total_tokens > 0 && point.avg_tokens_per_eval != null
  );
  const frontierCount = efficiencyLeaderboard.filter((point) => point.frontier).length;
  const customDataset = (report?.metadata?.custom_dataset as
    | { name?: string; item_count?: number }
    | undefined) ?? null;
  const disagreement = report?.disagreement;
  const policySummary = report?.policy ?? null;
  const policyAudit = report?.policy_audit ?? null;
  const topPolicyCases = policySummary?.top_cases ?? [];
  const recentPolicyReviews = policyAudit?.recent_reviews ?? [];
  const topDisagreementCases = disagreement?.top_cases ?? [];
  const selectedContinuity = normalizeContinuitySummary(report?.continuity);
  const conversationExplorerRows = buildConversationExplorerRows(selectedRawReport);
  const selectedConversation =
    conversationExplorerRows.find((row) => row.key === selectedConversationKey) ?? conversationExplorerRows[0] ?? null;
  const traceExplorerRows = buildTraceExplorerRows(selectedRawReport);
  const selectedTrace = traceExplorerRows.find((row) => row.key === selectedTraceKey) ?? traceExplorerRows[0] ?? null;
  const conversationFailureTurns = conversationExplorerRows.reduce(
    (sum, row) => sum + row.turns.filter((turn) => turn.failed).length,
    0
  );
  const unresolvedConversationTurns = conversationExplorerRows.reduce(
    (sum, row) => sum + row.unresolvedTurns,
    0
  );
  const avgConversationIntentResolution = average(conversationExplorerRows.map((row) => row.intentResolution));
  const failedTraceSpans = traceExplorerRows.reduce((sum, row) => sum + row.failedSpans, 0);
  const toolTraceSpans = traceExplorerRows.reduce(
    (sum, row) => sum + row.spans.filter((span) => span.spanType === "TOOL").length,
    0
  );
  const avgTraceDuration = average(traceExplorerRows.map((row) => row.durationMs));
  const modelTrendRows = normalizeTrendRows(report?.trends);
  const modelComparisonEntries =
    (report?.model_comparison as Record<string, ModelComparisonEntry> | undefined) ?? {};
  const modelEfficiencyRows = Object.entries(
    modelComparisonEntries
  )
    .map(([model, payload]) => ({
      model,
      overallScore: payload?.overall_score ?? null,
      avgLatency: payload?.avg_latency ?? null,
      latencyP95: payload?.latency_p95 ?? null,
      totalCost: payload?.total_cost ?? null,
      errorRate: payload?.error_rate ?? null,
      qualityLatencyEfficiency: payload?.quality_latency_efficiency ?? null,
      qualityCostEfficiency:
        typeof payload?.overall_score === "number" && typeof payload?.total_cost === "number"
          ? payload.overall_score / Math.max(payload.total_cost, 1e-9)
          : null,
    }))
    .filter((row) =>
      [row.avgLatency, row.latencyP95, row.totalCost, row.qualityLatencyEfficiency, row.qualityCostEfficiency].some(
        (value) => typeof value === "number" && !Number.isNaN(value)
      )
    );
  const slowestLatencyModel = modelEfficiencyRows.reduce<(typeof modelEfficiencyRows)[number] | null>(
    (slowest, row) => {
      if (row.avgLatency == null) return slowest;
      if (!slowest || slowest.avgLatency == null) return row;
      return row.avgLatency > slowest.avgLatency ? row : slowest;
    },
    null
  );
  const slowestP95Model = modelEfficiencyRows.reduce<(typeof modelEfficiencyRows)[number] | null>(
    (slowest, row) => {
      if (row.latencyP95 == null) return slowest;
      if (!slowest || slowest.latencyP95 == null) return row;
      return row.latencyP95 > slowest.latencyP95 ? row : slowest;
    },
    null
  );
  const costliestModel = modelEfficiencyRows.reduce<(typeof modelEfficiencyRows)[number] | null>(
    (costliest, row) => {
      if (row.totalCost == null) return costliest;
      if (!costliest || costliest.totalCost == null) return row;
      return row.totalCost > costliest.totalCost ? row : costliest;
    },
    null
  );
  const weakestLatencyYieldModel = modelEfficiencyRows.reduce<(typeof modelEfficiencyRows)[number] | null>(
    (weakest, row) => {
      if (row.qualityLatencyEfficiency == null) return weakest;
      if (!weakest || weakest.qualityLatencyEfficiency == null) return row;
      return row.qualityLatencyEfficiency < weakest.qualityLatencyEfficiency ? row : weakest;
    },
    null
  );
  const strongestCostYieldModel = modelEfficiencyRows.reduce<(typeof modelEfficiencyRows)[number] | null>(
    (strongest, row) => {
      if (row.qualityCostEfficiency == null) return strongest;
      if (!strongest || strongest.qualityCostEfficiency == null) return row;
      return row.qualityCostEfficiency > strongest.qualityCostEfficiency ? row : strongest;
    },
    null
  );
  const topEvaluatorByVolume = evaluatorEfficiencyRows[0] ?? null;
  const topEvaluatorByObservedCost = evaluatorEfficiencyRows.reduce<EvaluatorEfficiencyRow | null>((best, row) => {
    if (row.observed_cost == null) return best;
    if (!best || best.observed_cost == null) return row;
    return row.observed_cost > best.observed_cost ? row : best;
  }, null);
  const bestEvaluatorScore = evaluatorEfficiencyRows.reduce<EvaluatorEfficiencyRow | null>((best, row) => {
    if (row.avg_score == null) return best;
    if (!best || best.avg_score == null) return row;
    return row.avg_score > best.avg_score ? row : best;
  }, null);
  const normalizedProviderCostRows = buildProviderCostRows(report?.models, modelComparisonEntries);
  const totalProviderCost = normalizedProviderCostRows.reduce((sum, row) => sum + row.totalCost, 0);
  const dominantProviderCost = normalizedProviderCostRows[0] ?? null;
  const leanestProviderCost = normalizedProviderCostRows.reduce<ProviderCostRow | null>((best, row) => {
    if (row.costPer1kTokens == null) return best;
    if (!best || best.costPer1kTokens == null) return row;
    return row.costPer1kTokens < best.costPer1kTokens ? row : best;
  }, null);
  const hasRunEfficiencySummary =
    hasEfficiencyData || normalizedProviderCostRows.length > 0 || modelEfficiencyRows.length > 0;
  const structuredOutputRows = Object.entries(
    modelComparisonEntries
  )
    .map(([model, payload]) => {
      const reliability = payload?.structured_output_reliability;
      const schemaRows = Object.entries(reliability?.schema_type_breakdown ?? {}).sort(
        (a, b) => (b[1]?.invalid_cases ?? 0) - (a[1]?.invalid_cases ?? 0)
      );
      const datasetRows = Object.entries(reliability?.dataset_breakdown ?? {}).sort(
        (a, b) => (b[1]?.invalid_cases ?? 0) - (a[1]?.invalid_cases ?? 0)
      );
      const testRows = Object.entries(reliability?.test_breakdown ?? {}).sort(
        (a, b) => (b[1]?.invalid_cases ?? 0) - (a[1]?.invalid_cases ?? 0)
      );

      return {
        model,
        complianceRate: reliability?.schema_compliance_rate ?? null,
        totalCases: reliability?.total_cases ?? 0,
        invalidCases: reliability?.invalid_cases ?? 0,
        topSchema: schemaRows[0] ?? null,
        topDataset: datasetRows[0] ?? null,
        topTest: testRows[0] ?? null,
      };
    })
    .filter((row) => row.totalCases > 0);
  const compareContinuityEntries = Object.entries(compareData ?? {})
    .map(([filename, payload]) => ({
      filename,
      continuity: normalizeContinuitySummary(payload.continuity),
    }))
    .filter((entry) => entry.continuity);
  const compareStructuredOutputEntries = Object.entries(compareData ?? {})
    .map(([filename, payload]) => {
      const rows = Object.entries(
        (payload.model_comparison as Record<string, ModelComparisonEntry> | undefined) ?? {}
      )
        .map(([model, comparison]) => {
          const reliability = comparison?.structured_output_reliability;
          const testRows = Object.entries(reliability?.test_breakdown ?? {}).sort(
            (a, b) => (b[1]?.invalid_cases ?? 0) - (a[1]?.invalid_cases ?? 0)
          );
          const datasetRows = Object.entries(reliability?.dataset_breakdown ?? {}).sort(
            (a, b) => (b[1]?.invalid_cases ?? 0) - (a[1]?.invalid_cases ?? 0)
          );

          return {
            model,
            complianceRate: reliability?.schema_compliance_rate ?? null,
            totalCases: reliability?.total_cases ?? 0,
            invalidCases: reliability?.invalid_cases ?? 0,
            hottestTest: testRows[0] ?? null,
            hottestDataset: datasetRows[0] ?? null,
          };
        })
        .filter((row) => row.totalCases > 0);

      return {
        filename,
        rows,
      };
    })
    .filter((entry) => entry.rows.length > 0);
  const riskLevelEntries = Object.entries(policySummary?.risk_level_counts ?? {});
  const baselineScores = baselineFilename ? compareData?.[baselineFilename]?.model_scores ?? null : null;
  const baselineComparisonModels = baselineFilename
    ? ((compareData?.[baselineFilename]?.model_comparison as Record<string, ModelComparisonEntry> | undefined) ?? {})
    : null;
  const normalizedRegressionThreshold = Math.max(0, regressionThreshold);
  const baselineCaseIndex = baselineFilename
    ? buildRawCaseIndex(compareRawReports?.[baselineFilename] ?? null)
    : null;
  const baselineDeltaEntries = baselineFilename && baselineScores
    ? compareSelection
        .filter((filename) => filename !== baselineFilename)
        .map((filename) => {
          const candidateScores = compareData?.[filename]?.model_scores ?? {};
          const modelNames = Array.from(
            new Set([...Object.keys(baselineScores), ...Object.keys(candidateScores)])
          ).sort();

          const rows = modelNames.map((model) => {
            const baselineScore = baselineScores[model];
            const candidateScore = candidateScores[model];
            const delta =
              typeof baselineScore === "number" && typeof candidateScore === "number"
                ? candidateScore - baselineScore
                : null;

            return {
              model,
              baselineScore,
              candidateScore,
              delta,
            };
          }).sort((left, right) => {
            const leftSeverity = left.delta == null
              ? 2
              : left.delta <= -normalizedRegressionThreshold
                ? 0
                : left.delta >= normalizedRegressionThreshold
                  ? 1
                  : 2;
            const rightSeverity = right.delta == null
              ? 2
              : right.delta <= -normalizedRegressionThreshold
                ? 0
                : right.delta >= normalizedRegressionThreshold
                  ? 1
                  : 2;

            if (leftSeverity !== rightSeverity) {
              return leftSeverity - rightSeverity;
            }
            return (left.delta ?? 0) - (right.delta ?? 0);
          });

          return {
            filename,
            rows,
            improvedCount: rows.filter((row) => (row.delta ?? 0) >= normalizedRegressionThreshold).length,
            regressedCount: rows.filter((row) => (row.delta ?? 0) <= -normalizedRegressionThreshold).length,
            flatCount: rows.filter(
              (row) => row.delta != null && Math.abs(row.delta) < normalizedRegressionThreshold
            ).length,
            promptVersion: compareData?.[filename]?.prompt_version ?? null,
            schemaVersion: compareData?.[filename]?.schema_version ?? null,
            metricVersion: compareData?.[filename]?.metric_version ?? null,
          };
        })
        .filter((entry) => entry.rows.length > 0)
    : [];
  const compareHasScores = compareSelection.length >= 2
    && compareData
    && Object.values(compareData).some(entry => Object.keys(entry.model_scores ?? {}).length > 0);
  const baselineEfficiencyDriftEntries = baselineFilename && baselineComparisonModels
    ? compareSelection
        .filter((filename) => filename !== baselineFilename)
        .map((filename) => {
          const candidateModels =
            ((compareData?.[filename]?.model_comparison as Record<string, ModelComparisonEntry> | undefined) ?? {});
          const modelNames = Array.from(
            new Set([...Object.keys(baselineComparisonModels), ...Object.keys(candidateModels)])
          ).sort();

          const rows = modelNames
            .map((model) => {
              const baselineModel = baselineComparisonModels[model];
              const candidateModel = candidateModels[model];
              const baselineOverallScore = baselineModel?.overall_score ?? null;
              const candidateOverallScore = candidateModel?.overall_score ?? null;
              const baselineAvgLatency = baselineModel?.avg_latency ?? null;
              const candidateAvgLatency = candidateModel?.avg_latency ?? null;
              const baselineCost = baselineModel?.total_cost ?? null;
              const candidateCost = candidateModel?.total_cost ?? null;
              const baselineQualityLatency = baselineModel?.quality_latency_efficiency ?? null;
              const candidateQualityLatency = candidateModel?.quality_latency_efficiency ?? null;
              const baselineQualityCost =
                baselineOverallScore != null && baselineCost != null
                  ? baselineOverallScore / Math.max(baselineCost, 1e-9)
                  : null;
              const candidateQualityCost =
                candidateOverallScore != null && candidateCost != null
                  ? candidateOverallScore / Math.max(candidateCost, 1e-9)
                  : null;

              return {
                model,
                baselineAvgLatency,
                candidateAvgLatency,
                avgLatencyDelta:
                  baselineAvgLatency != null && candidateAvgLatency != null
                    ? candidateAvgLatency - baselineAvgLatency
                    : null,
                baselineCost,
                candidateCost,
                costDelta:
                  baselineCost != null && candidateCost != null ? candidateCost - baselineCost : null,
                baselineQualityLatency,
                candidateQualityLatency,
                qualityLatencyDelta:
                  baselineQualityLatency != null && candidateQualityLatency != null
                    ? candidateQualityLatency - baselineQualityLatency
                    : null,
                baselineQualityCost,
                candidateQualityCost,
                qualityCostDelta:
                  baselineQualityCost != null && candidateQualityCost != null
                    ? candidateQualityCost - baselineQualityCost
                    : null,
              };
            })
            .filter((row) =>
              [
                row.baselineAvgLatency,
                row.candidateAvgLatency,
                row.baselineCost,
                row.candidateCost,
                row.baselineQualityLatency,
                row.candidateQualityLatency,
                row.baselineQualityCost,
                row.candidateQualityCost,
              ].some((value) => typeof value === "number" && !Number.isNaN(value))
            )
            .sort((left, right) => {
              const leftSeverity =
                (left.avgLatencyDelta ?? 0) + (left.costDelta ?? 0) + (-(left.qualityLatencyDelta ?? 0)) + (-(left.qualityCostDelta ?? 0));
              const rightSeverity =
                (right.avgLatencyDelta ?? 0) + (right.costDelta ?? 0) + (-(right.qualityLatencyDelta ?? 0)) + (-(right.qualityCostDelta ?? 0));
              return rightSeverity - leftSeverity;
            });

          return {
            filename,
            rows,
            slowerCount: rows.filter((row) => (row.avgLatencyDelta ?? 0) > 0).length,
            costlierCount: rows.filter((row) => (row.costDelta ?? 0) > 0).length,
            weakerYieldCount: rows.filter((row) => (row.qualityLatencyDelta ?? 0) < 0).length,
            weakerCostYieldCount: rows.filter((row) => (row.qualityCostDelta ?? 0) < 0).length,
          };
        })
        .filter((entry) => entry.rows.length > 0)
    : [];
  const baselineProviderCostRows = baselineFilename && compareRawReports && baselineComparisonModels
    ? buildProviderCostRows(asRecord(compareRawReports[baselineFilename] ?? null).models, baselineComparisonModels)
    : [];
  const baselineProviderDriftEntries = baselineFilename && compareRawReports && baselineProviderCostRows.length > 0
    ? compareSelection
        .filter((filename) => filename !== baselineFilename)
        .map((filename) => {
          const candidateModels =
            ((compareData?.[filename]?.model_comparison as Record<string, ModelComparisonEntry> | undefined) ?? {});
          const candidateProviderRows = buildProviderCostRows(
            asRecord(compareRawReports[filename] ?? null).models,
            candidateModels
          );
          const providerNames = Array.from(
            new Set([
              ...baselineProviderCostRows.map((row) => row.provider),
              ...candidateProviderRows.map((row) => row.provider),
            ])
          ).sort();

          const rows = providerNames
            .map((provider) => {
              const baselineRow = baselineProviderCostRows.find((row) => row.provider === provider) ?? null;
              const candidateRow = candidateProviderRows.find((row) => row.provider === provider) ?? null;
              const baselineTotalCost = baselineRow?.totalCost ?? null;
              const candidateTotalCost = candidateRow?.totalCost ?? null;
              const baselineCostShare = baselineRow?.costShare ?? null;
              const candidateCostShare = candidateRow?.costShare ?? null;
              const baselineCostPer1kTokens = baselineRow?.costPer1kTokens ?? null;
              const candidateCostPer1kTokens = candidateRow?.costPer1kTokens ?? null;

              return {
                provider,
                baselineModelCount: baselineRow?.modelCount ?? 0,
                candidateModelCount: candidateRow?.modelCount ?? 0,
                baselineTotalCost,
                candidateTotalCost,
                totalCostDelta:
                  baselineTotalCost != null && candidateTotalCost != null
                    ? candidateTotalCost - baselineTotalCost
                    : null,
                baselineCostShare,
                candidateCostShare,
                costShareDelta:
                  baselineCostShare != null && candidateCostShare != null
                    ? candidateCostShare - baselineCostShare
                    : null,
                baselineCostPer1kTokens,
                candidateCostPer1kTokens,
                costPer1kTokensDelta:
                  baselineCostPer1kTokens != null && candidateCostPer1kTokens != null
                    ? candidateCostPer1kTokens - baselineCostPer1kTokens
                    : null,
              };
            })
            .filter((row) =>
              [
                row.baselineTotalCost,
                row.candidateTotalCost,
                row.baselineCostShare,
                row.candidateCostShare,
                row.baselineCostPer1kTokens,
                row.candidateCostPer1kTokens,
              ].some((value) => typeof value === "number" && !Number.isNaN(value))
            )
            .sort((left, right) => {
              const leftSeverity = (left.costShareDelta ?? 0) + (left.totalCostDelta ?? 0) + (left.costPer1kTokensDelta ?? 0);
              const rightSeverity = (right.costShareDelta ?? 0) + (right.totalCostDelta ?? 0) + (right.costPer1kTokensDelta ?? 0);
              return rightSeverity - leftSeverity;
            });

          return {
            filename,
            rows,
            higherShareCount: rows.filter((row) => (row.costShareDelta ?? 0) > 0).length,
            higherSpendCount: rows.filter((row) => (row.totalCostDelta ?? 0) > 0).length,
            higherNormalizedCostCount: rows.filter((row) => (row.costPer1kTokensDelta ?? 0) > 0).length,
          };
        })
        .filter((entry) => entry.rows.length > 0)
    : [];
  const introducedFailureEntries = baselineFilename && baselineCaseIndex && compareRawReports
    ? compareSelection
        .filter((filename) => filename !== baselineFilename)
        .map((filename) => {
          const candidateCaseIndex = buildRawCaseIndex(compareRawReports[filename] ?? null);
          const rows = Object.entries(candidateCaseIndex)
            .flatMap(([model, caseIndex]) =>
              Object.entries(caseIndex)
                .filter(([, candidateCase]) => {
                  if (!candidateCase.failed) return false;
                  const baselineCase = baselineCaseIndex[model]?.[`${candidateCase.testName}::${candidateCase.caseId}`];
                  return !baselineCase?.failed;
                })
                .map(([, candidateCase]) => {
                  const baselineCase = baselineCaseIndex[model]?.[`${candidateCase.testName}::${candidateCase.caseId}`];
                  return {
                    model,
                    testName: candidateCase.testName,
                    caseId: candidateCase.caseId,
                    baselineStatus: baselineCase ? (baselineCase.failed ? "failed" : "passed") : "missing",
                    baselineScore: baselineCase?.score ?? null,
                    candidateScore: candidateCase.score,
                    reason: candidateCase.reason,
                  };
                })
            )
            .sort((left, right) => {
              if (left.baselineStatus !== right.baselineStatus) {
                return left.baselineStatus.localeCompare(right.baselineStatus);
              }
              return `${left.model}:${left.testName}:${left.caseId}`.localeCompare(
                `${right.model}:${right.testName}:${right.caseId}`
              );
            });

          return {
            filename,
            rows,
            passedToFailedCount: rows.filter((row) => row.baselineStatus === "passed").length,
            missingBaselineCount: rows.filter((row) => row.baselineStatus === "missing").length,
          };
        })
        .filter((entry) => entry.rows.length > 0)
    : [];
  const baselineDatasetSignature = baselineFilename
    ? extractDatasetSignature(compareRawReports?.[baselineFilename] ?? null)
    : null;
  const datasetDriftEntries = baselineFilename && baselineDatasetSignature && compareRawReports
    ? compareSelection
        .filter((filename) => filename !== baselineFilename)
        .map((filename) => {
          const candidateSignature = extractDatasetSignature(compareRawReports[filename] ?? null);
          const addedLabels = candidateSignature.labels.filter(
            (label) => !baselineDatasetSignature.labels.includes(label)
          );
          const removedLabels = baselineDatasetSignature.labels.filter(
            (label) => !candidateSignature.labels.includes(label)
          );

          return {
            filename,
            baseline: baselineDatasetSignature,
            candidate: candidateSignature,
            changedName: baselineDatasetSignature.name !== candidateSignature.name,
            changedPath: baselineDatasetSignature.path !== candidateSignature.path,
            changedItemCount: baselineDatasetSignature.itemCount !== candidateSignature.itemCount,
            addedLabels,
            removedLabels,
          };
        })
        .filter(
          (entry) =>
            entry.changedName ||
            entry.changedPath ||
            entry.changedItemCount ||
            entry.addedLabels.length > 0 ||
            entry.removedLabels.length > 0
        )
    : [];

  useEffect(() => {
    if (!conversationExplorerRows.length) {
      setSelectedConversationKey("");
      return;
    }

    if (!selectedConversationKey || !conversationExplorerRows.some((row) => row.key === selectedConversationKey)) {
      setSelectedConversationKey(conversationExplorerRows[0]?.key ?? "");
    }
  }, [conversationExplorerRows, selectedConversationKey]);

  useEffect(() => {
    if (!traceExplorerRows.length) {
      setSelectedTraceKey("");
      return;
    }

    if (!selectedTraceKey || !traceExplorerRows.some((row) => row.key === selectedTraceKey)) {
      setSelectedTraceKey(traceExplorerRows[0]?.key ?? "");
    }
  }, [traceExplorerRows, selectedTraceKey]);

  const selectedReportMeta = reports.find((entry) => entry.filename === selected) ?? null;

  return (
    <div className="page-shell motion-shell">
      <header className="page-header motion-hero">
        <p className="page-kicker">Readout</p>
        <h1 className="page-title">Results Explorer</h1>
        <p className="page-subtitle">
          Open each run like a scorecard wall, with averages, model ordering and test-level detail.
        </p>
      </header>

      <section className="panel-surface panel-quiet motion-rise motion-delay-1 max-w-4xl">
        <div className="grid gap-4 lg:grid-cols-2">
          <div>
            <label className="label">Report</label>
            <select
              value={selected ?? ""}
              onChange={(e) => setSelected(e.target.value || null)}
              className="control-surface"
            >
              <option value="">Select a report...</option>
              {reports.map((r) => (
                <option key={r.filename} value={r.filename}>
                  {r.filename} — {r.suite ?? "?"} ({r.model_count ?? "?"} models)
                </option>
              ))}
            </select>
            {selectedReportMeta && (
              <div className="mt-3 space-y-2">
                <p className="micro-copy">Unified export actions</p>
                <div className="button-row">
                  <a
                    href={selectedReportMeta.export_links.raw}
                    target="_blank"
                    rel="noreferrer"
                    className="button-secondary"
                  >
                    JSON
                  </a>
                  <a
                    href={selectedReportMeta.export_links.markdown}
                    target="_blank"
                    rel="noreferrer"
                    className="button-secondary"
                  >
                    Markdown
                  </a>
                  <a
                    href={selectedReportMeta.export_links.html}
                    target="_blank"
                    rel="noreferrer"
                    className="button-secondary"
                  >
                    HTML
                  </a>
                </div>
              </div>
            )}
          </div>

          <div>
            <div className="flex items-center justify-between">
              <label className="label">Continuity Compare</label>
              <div className="flex items-center gap-2">
                <span className="provider-chip">{compareSelection.length} selected</span>
                {compareSelection.length > 0 && (
                  <button
                    type="button"
                    className="button-secondary"
                    onClick={() => setCompareSelection([])}
                  >
                    Clear
                  </button>
                )}
              </div>
            </div>
            <div className="subpanel mt-1 max-h-64 space-y-1 overflow-y-auto rounded-[1rem] p-2">
              {reports.map((r) => (
                <label
                  key={`compare-${r.filename}`}
                  className={`flex cursor-pointer items-start gap-2 rounded-[0.8rem] px-2 py-2 text-left transition ${
                    compareSelection.includes(r.filename) ? "subpanel-selected" : ""
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={compareSelection.includes(r.filename)}
                    onChange={() => {
                      if (compareSelection.includes(r.filename)) {
                        setCompareSelection(compareSelection.filter((f) => f !== r.filename));
                      } else {
                        setCompareSelection([...compareSelection, r.filename]);
                      }
                    }}
                    className="mt-1 flex-shrink-0 accent-[#8ae5c5]"
                  />
                  <div className="flex-1 min-w-0">
                    <div className="body-copy text-sm break-all">{r.filename}</div>
                    <div className="micro-copy">
                      {r.suite ?? "unknown suite"} · {r.model_count ?? "?"} models · {new Date(r.modified).toLocaleString()}
                    </div>
                  </div>
                </label>
              ))}
            </div>
            {compareSelection.length < 2 && (
              <p className="micro-copy mt-2">Select at least two reports — continuity and model score differences are then computed.</p>
            )}
            {compareSelection.length >= 2 && (
              <div className="mt-3">
                <label className="label">Baseline Run</label>
                <select
                  value={baselineFilename ?? ""}
                  onChange={(e) => setBaselineFilename(e.target.value || null)}
                  className="control-surface"
                >
                  {compareSelection.map((filename) => (
                    <option key={`baseline-${filename}`} value={filename}>
                      {filename}
                    </option>
                  ))}
                </select>
                <p className="micro-copy mt-2">Model score delta table is built relative to the selected baseline report.</p>
                <div className="mt-3">
                  <label className="label">Regression Threshold</label>
                  <input
                    type="number"
                    min="0"
                    step="0.005"
                    value={regressionThreshold}
                    onChange={(e) => setRegressionThreshold(Number.parseFloat(e.target.value) || 0)}
                    className="control-surface"
                  />
                  <p className="micro-copy mt-2">A case is flagged as improved or regressed once the absolute delta crosses this threshold.</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </section>

      {reportsLoaded && reports.length === 0 && (
        <div className="motion-rise motion-delay-2">
          <EmptyState
            icon={BarChart3}
            title="No reports to explore yet"
            hint="Once you run an evaluation, its scorecard, model ordering and test-level detail show up here."
            action={
              <Link to="/run" className="button-primary">
                <Play size={14} /> Run Evaluation
              </Link>
            }
          />
        </div>
      )}

      {(loading || compareLoading) && (
        <div className="motion-rise motion-delay-2 flex items-center justify-center py-12">
          <div className="loading-orb" />
        </div>
      )}

      {compareSelection.length >= 2 && !compareLoading && compareError && (
        <section className="motion-rise motion-delay-2">
          <div className="callout-warn rounded-[1rem] px-4 py-3">
            <p className="body-copy">{compareError}</p>
          </div>
        </section>
      )}

      {compareSelection.length >= 2 && !compareLoading && compareData && !compareHasScores && (
        <section className="motion-rise motion-delay-2">
          <div className="callout-warn rounded-[1rem] px-4 py-3">
            <p className="body-copy">Secili raporlarda karsilastirilabilir model skoru yok.</p>
          </div>
        </section>
      )}

      {compareContinuityEntries.length > 0 && !compareLoading && (
        <section className="motion-rise motion-delay-2 space-y-4">
          <div>
            <p className="section-caption mb-2">Continuity Compare</p>
            <h2 className="section-heading">Cross-Report Intent Drift</h2>
            <p className="page-subtitle max-w-3xl text-sm">
              Read multi-turn continuity signals directly across selected reports.
            </p>
          </div>

          <div className="motion-stagger-grid grid grid-cols-1 gap-4 xl:grid-cols-2">
            {compareContinuityEntries.map(({ filename, continuity }) => (
              <div key={filename} className="panel-surface panel-quiet space-y-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <p className="section-caption mb-2">Report</p>
                    <h3 className="section-heading text-lg">{filename}</h3>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {continuity?.best_intent_resolution_model && (
                      <span className="provider-chip">best intent: {continuity.best_intent_resolution_model}</span>
                    )}
                    {continuity?.highest_unresolved_rate_model && (
                      <span className="provider-chip">highest open rate: {continuity.highest_unresolved_rate_model}</span>
                    )}
                  </div>
                </div>

                <div className="table-shell overflow-x-auto">
                  <table>
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>Intent</th>
                        <th>Open Rate</th>
                        <th>Open Turns</th>
                      </tr>
                    </thead>
                    <tbody>
                      {continuity?.by_model.map((item) => (
                        <tr key={`${filename}-${item.model}`}>
                          <td className="body-copy">{item.model}</td>
                          <td>
                            {item.intent_resolution != null ? <ScoreBadge score={item.intent_resolution} /> : "—"}
                          </td>
                          <td className="micro-copy">
                            {item.unresolved_turn_rate != null ? formatMetric(item.unresolved_turn_rate, 2) : "—"}
                          </td>
                          <td className="micro-copy">
                            {item.unresolved_turns != null
                              ? `${formatCount(item.unresolved_turns)} / ${formatCount(item.unresolved_intent_total ?? 0)}`
                              : "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {compareStructuredOutputEntries.length > 0 && !compareLoading && (
        <section className="motion-rise motion-delay-2 space-y-4">
          <div>
            <p className="section-caption mb-2">Structured Output Compare</p>
            <h2 className="section-heading">Cross-Report Reliability Drift</h2>
            <p className="page-subtitle max-w-3xl text-sm">
              Quickly review schema compliance and the most brittle tests or data sources across selected reports.
            </p>
          </div>

          <div className="motion-stagger-grid grid grid-cols-1 gap-4 xl:grid-cols-2">
            {compareStructuredOutputEntries.map(({ filename, rows }) => (
              <div key={`structured-${filename}`} className="panel-surface panel-quiet space-y-4">
                <div>
                  <p className="section-caption mb-2">Report</p>
                  <h3 className="section-heading text-lg">{filename}</h3>
                </div>

                <div className="table-shell overflow-x-auto">
                  <table>
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>Compliance</th>
                        <th>Cases</th>
                        <th>Invalid</th>
                        <th>Top Test</th>
                        <th>Top Dataset</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((row) => (
                        <tr key={`${filename}-${row.model}`}>
                          <td className="body-copy">{row.model}</td>
                          <td>{row.complianceRate != null ? <ScoreBadge score={row.complianceRate} /> : "—"}</td>
                          <td className="micro-copy">{formatCount(row.totalCases)}</td>
                          <td className="micro-copy">{formatCount(row.invalidCases)}</td>
                          <td className="micro-copy">
                            {row.hottestTest ? `${row.hottestTest[0]} (${row.hottestTest[1]?.invalid_cases ?? 0})` : "—"}
                          </td>
                          <td className="micro-copy">
                            {row.hottestDataset ? `${row.hottestDataset[0]} (${row.hottestDataset[1]?.invalid_cases ?? 0})` : "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {baselineDeltaEntries.length > 0 && !compareLoading && baselineFilename && (
        <section className="motion-rise motion-delay-2 space-y-4">
          <div>
            <p className="section-caption mb-2">Baseline Compare</p>
            <h2 className="section-heading">Run Score Delta Wall</h2>
            <p className="page-subtitle max-w-3xl text-sm">
              Quickly review model score differences against the baseline; positive delta indicates improvement, negative indicates regression.
            </p>
          </div>

          <div className="motion-stagger-grid grid grid-cols-1 gap-4 xl:grid-cols-2">
            {baselineDeltaEntries.map((entry) => (
              <div key={`baseline-delta-${entry.filename}`} className="panel-surface panel-quiet space-y-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <p className="section-caption mb-2">Compared To Baseline</p>
                    <h3 className="section-heading text-lg">{entry.filename}</h3>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <span className="provider-chip">baseline: {baselineFilename}</span>
                    <span className="provider-chip">
                      prompt: {(compareData?.[baselineFilename]?.prompt_version as string | undefined) ?? "n/a"} → {entry.promptVersion ?? "n/a"}
                    </span>
                    <span className="provider-chip">
                      schema: {(compareData?.[baselineFilename]?.schema_version as string | undefined) ?? "n/a"} → {entry.schemaVersion ?? "n/a"}
                    </span>
                    <span className="provider-chip">
                      metrics: {(compareData?.[baselineFilename]?.metric_version as string | undefined) ?? "n/a"} → {entry.metricVersion ?? "n/a"}
                    </span>
                    {(
                      (compareData?.[baselineFilename]?.prompt_version ?? null) !== (entry.promptVersion ?? null)
                      || (compareData?.[baselineFilename]?.schema_version ?? null) !== (entry.schemaVersion ?? null)
                      || (compareData?.[baselineFilename]?.metric_version ?? null) !== (entry.metricVersion ?? null)
                    ) && (
                      <span className="provider-chip">version changed</span>
                    )}
                    <span className="provider-chip">threshold: {normalizedRegressionThreshold.toFixed(3)}</span>
                    <span className="provider-chip">improved {entry.improvedCount}</span>
                    <span className="provider-chip">regressed {entry.regressedCount}</span>
                    <span className="provider-chip">flat {entry.flatCount}</span>
                  </div>
                </div>

                <div className="table-shell overflow-x-auto">
                  <table>
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>Baseline</th>
                        <th>Candidate</th>
                        <th>Delta</th>
                      </tr>
                    </thead>
                    <tbody>
                      {entry.rows.map((row) => (
                        <tr key={`${entry.filename}-${row.model}`}>
                          <td className="body-copy">{row.model}</td>
                          <td className="micro-copy">
                            {row.baselineScore != null ? row.baselineScore.toFixed(3) : "—"}
                          </td>
                          <td className="micro-copy">
                            {row.candidateScore != null ? row.candidateScore.toFixed(3) : "—"}
                          </td>
                          <td>
                            {row.delta == null ? (
                              "—"
                            ) : (
                              <span
                                className={`provider-chip ${row.delta <= -normalizedRegressionThreshold ? "score-badge-low" : row.delta >= normalizedRegressionThreshold ? "score-badge-good" : "score-badge-mid"}`}
                              >
                                {row.delta > 0 ? "+" : ""}
                                {row.delta.toFixed(3)}
                              </span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {baselineEfficiencyDriftEntries.length > 0 && !compareLoading && baselineFilename && (
        <section className="motion-rise motion-delay-2 space-y-4">
          <div>
            <p className="section-caption mb-2">Efficiency Drift</p>
            <h2 className="section-heading">Baseline Latency and Cost Drift</h2>
            <p className="page-subtitle max-w-3xl text-sm">
              Quickly review latency, cost, quality-per-latency, and quality-per-cost changes across models against the baseline.
            </p>
          </div>

          <div className="motion-stagger-grid grid grid-cols-1 gap-4 xl:grid-cols-2">
            {baselineEfficiencyDriftEntries.map((entry) => (
              <div key={`baseline-efficiency-${entry.filename}`} className="panel-surface panel-quiet space-y-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <p className="section-caption mb-2">Compared To Baseline</p>
                    <h3 className="section-heading text-lg">{entry.filename}</h3>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <span className="provider-chip">baseline: {baselineFilename}</span>
                    <span className="provider-chip">slower {entry.slowerCount}</span>
                    <span className="provider-chip">costlier {entry.costlierCount}</span>
                    <span className="provider-chip">weaker yield {entry.weakerYieldCount}</span>
                    <span className="provider-chip">weaker cost yield {entry.weakerCostYieldCount}</span>
                  </div>
                </div>

                <div className="table-shell overflow-x-auto">
                  <table>
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>Avg Latency</th>
                        <th>Cost</th>
                        <th>Quality / Cost</th>
                        <th>Quality / Latency</th>
                      </tr>
                    </thead>
                    <tbody>
                      {entry.rows.map((row) => (
                        <tr key={`${entry.filename}-efficiency-${row.model}`}>
                          <td className="body-copy">{row.model}</td>
                          <td className="micro-copy">
                            {formatMetric(row.baselineAvgLatency, 2)} → {formatMetric(row.candidateAvgLatency, 2)}
                            <span className="ml-2">({row.avgLatencyDelta != null ? formatMetric(row.avgLatencyDelta, 2) : "—"})</span>
                          </td>
                          <td className="micro-copy">
                            {formatMetric(row.baselineCost, 4)} → {formatMetric(row.candidateCost, 4)}
                            <span className="ml-2">({row.costDelta != null ? formatMetric(row.costDelta, 4) : "—"})</span>
                          </td>
                          <td className="micro-copy">
                            {formatMetric(row.baselineQualityCost, 4)} → {formatMetric(row.candidateQualityCost, 4)}
                            <span className="ml-2">({row.qualityCostDelta != null ? formatMetric(row.qualityCostDelta, 4) : "—"})</span>
                          </td>
                          <td className="micro-copy">
                            {formatMetric(row.baselineQualityLatency, 4)} → {formatMetric(row.candidateQualityLatency, 4)}
                            <span className="ml-2">({row.qualityLatencyDelta != null ? formatMetric(row.qualityLatencyDelta, 4) : "—"})</span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {baselineProviderDriftEntries.length > 0 && !compareLoading && baselineFilename && (
        <section className="motion-rise motion-delay-2 space-y-4">
          <div>
            <p className="section-caption mb-2">Provider Cost Drift</p>
            <h2 className="section-heading">Baseline Provider Spend Drift</h2>
            <p className="page-subtitle max-w-3xl text-sm">
              Quickly review provider-level spend share and token-normalized cost changes against the baseline.
            </p>
          </div>

          <div className="motion-stagger-grid grid grid-cols-1 gap-4 xl:grid-cols-2">
            {baselineProviderDriftEntries.map((entry) => (
              <div key={`baseline-provider-cost-${entry.filename}`} className="panel-surface panel-quiet space-y-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <p className="section-caption mb-2">Compared To Baseline</p>
                    <h3 className="section-heading text-lg">{entry.filename}</h3>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <span className="provider-chip">baseline: {baselineFilename}</span>
                    <span className="provider-chip">higher share {entry.higherShareCount}</span>
                    <span className="provider-chip">higher spend {entry.higherSpendCount}</span>
                    <span className="provider-chip">higher normalized cost {entry.higherNormalizedCostCount}</span>
                  </div>
                </div>

                <div className="table-shell overflow-x-auto">
                  <table>
                    <thead>
                      <tr>
                        <th>Provider</th>
                        <th>Cost Share</th>
                        <th>Total Cost</th>
                        <th>Cost / 1K Tokens</th>
                        <th>Models</th>
                      </tr>
                    </thead>
                    <tbody>
                      {entry.rows.map((row) => (
                        <tr key={`${entry.filename}-provider-${row.provider}`}>
                          <td className="body-copy">{row.provider}</td>
                          <td className="micro-copy">
                            {row.baselineCostShare != null ? `${formatMetric(row.baselineCostShare * 100, 1)}%` : "—"} → {row.candidateCostShare != null ? `${formatMetric(row.candidateCostShare * 100, 1)}%` : "—"}
                          </td>
                          <td className="micro-copy">
                            {formatMetric(row.baselineTotalCost, 4)} → {formatMetric(row.candidateTotalCost, 4)}
                          </td>
                          <td className="micro-copy">
                            {formatMetric(row.baselineCostPer1kTokens, 4)} → {formatMetric(row.candidateCostPer1kTokens, 4)}
                          </td>
                          <td className="micro-copy">
                            {formatCount(row.baselineModelCount)} → {formatCount(row.candidateModelCount)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {introducedFailureEntries.length > 0 && !compareLoading && baselineFilename && (
        <section className="motion-rise motion-delay-2 space-y-4">
          <div>
            <p className="section-caption mb-2">Regression Desk</p>
            <h2 className="section-heading">New Failures Introduced</h2>
            <p className="page-subtitle max-w-3xl text-sm">
              See cases that passed on the baseline but fail on the candidate report, broken down by model and test.
            </p>
          </div>

          <div className="motion-stagger-grid grid grid-cols-1 gap-4 xl:grid-cols-2">
            {introducedFailureEntries.map((entry) => (
              <div key={`introduced-failures-${entry.filename}`} className="panel-surface panel-quiet space-y-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <p className="section-caption mb-2">Compared To Baseline</p>
                    <h3 className="section-heading text-lg">{entry.filename}</h3>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <span className="provider-chip">baseline: {baselineFilename}</span>
                    <span className="provider-chip">new failures {entry.rows.length}</span>
                    <span className="provider-chip">pass→fail {entry.passedToFailedCount}</span>
                    <span className="provider-chip">baseline missing {entry.missingBaselineCount}</span>
                  </div>
                </div>

                <div className="table-shell overflow-x-auto">
                  <table>
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>Test</th>
                        <th>Case</th>
                        <th>Baseline</th>
                        <th>Candidate</th>
                        <th>Reason</th>
                      </tr>
                    </thead>
                    <tbody>
                      {entry.rows.map((row) => (
                        <tr key={`${entry.filename}-${row.model}-${row.testName}-${row.caseId}`}>
                          <td className="body-copy">{row.model}</td>
                          <td className="micro-copy">{row.testName}</td>
                          <td className="micro-copy">{row.caseId}</td>
                          <td className="micro-copy">
                            {row.baselineStatus === "missing"
                              ? "missing"
                              : `${row.baselineStatus} ${row.baselineScore != null ? `(${row.baselineScore.toFixed(3)})` : ""}`}
                          </td>
                          <td className="micro-copy">
                            {row.candidateScore != null ? `failed (${row.candidateScore.toFixed(3)})` : "failed"}
                          </td>
                          <td className="micro-copy">{row.reason || "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {datasetDriftEntries.length > 0 && !compareLoading && baselineFilename && (
        <section className="motion-rise motion-delay-2 space-y-4">
          <div>
            <p className="section-caption mb-2">Dataset Drift</p>
            <h2 className="section-heading">Baseline Dataset Changes</h2>
            <p className="page-subtitle max-w-3xl text-sm">
              Quickly review dataset identity, scope size, and test-level dataset label changes against the baseline.
            </p>
          </div>

          <div className="motion-stagger-grid grid grid-cols-1 gap-4 xl:grid-cols-2">
            {datasetDriftEntries.map((entry) => (
              <div key={`dataset-drift-${entry.filename}`} className="panel-surface panel-quiet space-y-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <p className="section-caption mb-2">Compared To Baseline</p>
                    <h3 className="section-heading text-lg">{entry.filename}</h3>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <span className="provider-chip">baseline: {baselineFilename}</span>
                    {entry.changedName && <span className="provider-chip">dataset changed</span>}
                    {entry.changedItemCount && <span className="provider-chip">size drift</span>}
                    {(entry.addedLabels.length > 0 || entry.removedLabels.length > 0) && (
                      <span className="provider-chip">label drift</span>
                    )}
                  </div>
                </div>

                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  <div className="rounded-[1.2rem] hairline p-4">
                    <p className="section-caption mb-2">Baseline Dataset</p>
                    <p className="body-copy font-semibold">{entry.baseline.name ?? "—"}</p>
                    <p className="micro-copy mt-2 break-words">{entry.baseline.path ?? "—"}</p>
                    <p className="micro-copy mt-2">items: {formatCount(entry.baseline.itemCount)}</p>
                    <p className="micro-copy mt-2">labels: {entry.baseline.labels.join(", ") || "—"}</p>
                  </div>

                  <div className="rounded-[1.2rem] hairline p-4">
                    <p className="section-caption mb-2">Candidate Dataset</p>
                    <p className="body-copy font-semibold">{entry.candidate.name ?? "—"}</p>
                    <p className="micro-copy mt-2 break-words">{entry.candidate.path ?? "—"}</p>
                    <p className="micro-copy mt-2">items: {formatCount(entry.candidate.itemCount)}</p>
                    <p className="micro-copy mt-2">labels: {entry.candidate.labels.join(", ") || "—"}</p>
                  </div>
                </div>

                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  <div>
                    <p className="micro-copy">Added Labels</p>
                    <p className="body-copy mt-1">{entry.addedLabels.join(", ") || "—"}</p>
                  </div>
                  <div>
                    <p className="micro-copy">Removed Labels</p>
                    <p className="body-copy mt-1">{entry.removedLabels.join(", ") || "—"}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {report && !loading && (
        <div className="space-y-6">
          <section className="panel-surface panel-quiet motion-rise motion-delay-1 space-y-4">
            <div>
              <p className="section-caption mb-2">Metadata</p>
              <h2 className="section-heading">Run Metadata</h2>
            </div>
            <div className="grid grid-cols-2 gap-4 text-sm md:grid-cols-4 xl:grid-cols-6">
              <div>
                <p className="micro-copy">Suite</p>
                <p className="body-copy mt-1">{(report.metadata.test_suite as string) ?? "—"}</p>
              </div>
              <div>
                <p className="micro-copy">Run ID</p>
                <p className="body-copy mt-1 font-mono">{((report.metadata.run_id as string) ?? "").slice(0, 12)}</p>
              </div>
              <div>
                <p className="micro-copy">Timestamp</p>
                <p className="body-copy mt-1">{((report.metadata.timestamp as string) ?? "").slice(0, 19)}</p>
              </div>
              <div>
                <p className="micro-copy">Prompt Version</p>
                <p className="body-copy mt-1">{(report.metadata.prompt_version as string) ?? (report.metadata.judge_prompt_version as string) ?? "—"}</p>
              </div>
              <div>
                <p className="micro-copy">Schema Version</p>
                <p className="body-copy mt-1">{(report.metadata.schema_version as string) ?? "—"}</p>
              </div>
              <div>
                <p className="micro-copy">Metric Bundle</p>
                <p className="body-copy mt-1">{(report.metadata.metric_version as string) ?? "—"}</p>
              </div>
              <div>
                <p className="micro-copy">Models</p>
                <p className="body-copy mt-1">{Object.keys(report.model_scores).length}</p>
              </div>
              {selectedContinuity?.best_intent_resolution_model && (
                <div>
                  <p className="micro-copy">Best Intent Model</p>
                  <p className="body-copy mt-1">{selectedContinuity.best_intent_resolution_model}</p>
                </div>
              )}
              {selectedContinuity?.highest_unresolved_rate_model && (
                <div>
                  <p className="micro-copy">Highest Open Rate</p>
                  <p className="body-copy mt-1">{selectedContinuity.highest_unresolved_rate_model}</p>
                </div>
              )}
              {customDataset?.name && (
                <>
                  <div>
                    <p className="micro-copy">Dataset</p>
                    <p className="body-copy mt-1">{customDataset.name}</p>
                  </div>
                  <div>
                    <p className="micro-copy">Dataset Items</p>
                    <p className="body-copy mt-1">{customDataset.item_count ?? "—"}</p>
                  </div>
                </>
              )}
            </div>
          </section>

          {(() => {
            const commentaryEntries = Object.entries(report.models)
              .map(([modelKey, modelData]) => {
                const data = modelData as Record<string, unknown>;
                const commentary = data?.ai_commentary as string | undefined;
                const judgeKey = data?.ai_commentary_judge as string | undefined;
                const overallScore = (data?.overall_metrics as Record<string, unknown>)?.weighted_score as number | undefined;
                return { modelKey, commentary, judgeKey, overallScore };
              })
              .filter((e) => !!e.commentary);

            if (commentaryEntries.length === 0) return null;

            const bestScore = Math.max(...commentaryEntries.map((e) => e.overallScore ?? 0));

            return (
              <section className="motion-rise motion-delay-1 space-y-4">
                <div>
                  <p className="section-caption mb-2">AI Commentary</p>
                  <h2 className="section-heading">Model Evaluation Commentary</h2>
                  <p className="page-subtitle max-w-3xl text-sm">
                    AI-generated evaluation commentary for each model, automatically produced by the judge.
                  </p>
                </div>
                <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
                  {commentaryEntries.map(({ modelKey, commentary, judgeKey, overallScore }) => {
                    const isBest = typeof overallScore === "number" && overallScore === bestScore && commentaryEntries.length > 1;
                    return (
                      <div
                        key={`commentary-${modelKey}`}
                        className="panel-surface panel-quiet space-y-3"
                      >
                        <div className="flex flex-wrap items-start justify-between gap-2">
                          <div>
                            <p className="section-caption mb-1">Model</p>
                            <h3 className="section-heading text-base">{modelKey}</h3>
                          </div>
                          <div className="flex flex-wrap items-center gap-2">
                            {isBest && (
                              <span className="provider-chip bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300">
                                Best Score
                              </span>
                            )}
                            {typeof overallScore === "number" && (
                              <ScoreBadge score={overallScore} />
                            )}
                          </div>
                        </div>
                        <p className="body-copy leading-relaxed text-sm">{commentary}</p>
                        {judgeKey && (
                          <p className="micro-copy mt-1 text-right opacity-60">
                            ✦ Evaluated by {judgeKey}
                          </p>
                        )}
                      </div>
                    );
                  })}
                </div>
              </section>
            );
          })()}

          {conversationExplorerRows.length > 0 && selectedConversation && (
            <section className="motion-rise motion-delay-1 space-y-4">
              <div>
                <p className="section-caption mb-2">Conversation Explorer</p>
                <h2 className="section-heading">Multi-turn Transcript Diagnostics</h2>
                <p className="page-subtitle max-w-3xl text-sm">
                  Turn transcript, per-turn score, unresolved intent, retrieval context and faithfulness reasoning ayni explorer icinde okunur.
                </p>
              </div>

              <div className="motion-stagger-grid grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
                <div className="panel-surface panel-quiet space-y-2">
                  <p className="metric-label">Conversations</p>
                  <p className="metric-value text-2xl">{formatCount(conversationExplorerRows.length)}</p>
                  <p className="micro-copy">Visible multi-turn case count in the selected report.</p>
                </div>
                <div className="panel-surface panel-quiet space-y-2">
                  <p className="metric-label">Open Turns</p>
                  <p className="metric-value text-2xl">{formatCount(unresolvedConversationTurns)}</p>
                  <p className="micro-copy">Turns still carrying unresolved intent signals.</p>
                </div>
                <div className="panel-surface panel-quiet space-y-2">
                  <p className="metric-label">Flagged Turns</p>
                  <p className="metric-value text-2xl">{formatCount(conversationFailureTurns)}</p>
                  <p className="micro-copy">Turn-level failures from open intent, weak faithfulness or schema issues.</p>
                </div>
                <div className="panel-surface panel-quiet space-y-2">
                  <p className="metric-label">Avg Intent Resolution</p>
                  <p className="metric-value text-2xl">{formatMetric(avgConversationIntentResolution, 3)}</p>
                  <p className="micro-copy">Conversation-level intent closure across visible multi-turn cases.</p>
                </div>
              </div>

              <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(18rem,0.72fr)_minmax(0,1.28fr)]">
                <div className="panel-surface panel-quiet space-y-4">
                  <div>
                    <p className="section-caption mb-2">Case Queue</p>
                    <h3 className="section-heading">Conversation cases</h3>
                  </div>
                  <div className="space-y-3">
                    {conversationExplorerRows.map((row) => {
                      const isSelected = row.key === selectedConversation.key;
                      return (
                        <button
                          key={row.key}
                          type="button"
                          onClick={() => setSelectedConversationKey(row.key)}
                          className={`w-full rounded-[1.2rem] px-4 py-3 text-left transition ${
                            isSelected
                              ? "subpanel-selected"
                              : "subpanel"
                          }`}
                        >
                          <div className="flex flex-wrap items-start justify-between gap-3">
                            <div>
                              <p className="body-copy font-medium">{row.caseId}</p>
                              <p className="micro-copy mt-1">{row.model} · {row.testName}</p>
                            </div>
                            {row.intentResolution != null && <ScoreBadge score={row.intentResolution} />}
                          </div>
                          <div className="mt-3 flex flex-wrap gap-2">
                            <span className="provider-chip">{row.turns.length} turns</span>
                            <span className="provider-chip">{row.unresolvedTurns} open</span>
                            <span className="provider-chip">{row.category}</span>
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="panel-surface panel-quiet space-y-4">
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div>
                        <p className="section-caption mb-2">Selected Conversation</p>
                        <h3 className="section-heading">{selectedConversation.caseId}</h3>
                        <p className="micro-copy mt-2">{selectedConversation.model} · {selectedConversation.testName}</p>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {selectedConversation.intentResolution != null && <span className="provider-chip">intent {formatMetric(selectedConversation.intentResolution, 2)}</span>}
                        <span className="provider-chip">{selectedConversation.unresolvedTurns} open turns</span>
                        {selectedConversation.avgTurnLatency != null && <span className="provider-chip">{formatMetric(selectedConversation.avgTurnLatency, 1)} ms avg latency</span>}
                      </div>
                    </div>

                    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1.25fr)_minmax(18rem,0.75fr)]">
                      <div className="space-y-3">
                        {selectedConversation.turns.map((turn) => (
                          <div
                            key={`${selectedConversation.key}-turn-${turn.turnNumber}`}
                            className={`rounded-[1.2rem] px-4 py-4 ${
                              turn.failed
                                ? "subpanel-danger"
                                : "subpanel"
                            }`}
                          >
                            <div className="flex flex-wrap items-start justify-between gap-3">
                              <div>
                                <p className="body-copy font-medium">Turn {turn.turnNumber}</p>
                                <p className="micro-copy mt-1">Window size {turn.windowSize ?? turn.evaluationWindow.length ?? 0}</p>
                              </div>
                              <div className="flex flex-wrap gap-2">
                                {turn.compositeScore != null && <ScoreBadge score={turn.compositeScore} />}
                                {turn.hasUnresolvedIntent && <span className="provider-chip">open intent</span>}
                                {turn.failed && <span className="provider-chip">flagged</span>}
                              </div>
                            </div>

                            <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
                              <div className="rounded-[1rem] hairline px-4 py-3">
                                <p className="micro-copy">User</p>
                                <p className="body-copy mt-2 whitespace-pre-wrap text-sm">{turn.userMessage || "—"}</p>
                              </div>
                              <div className="rounded-[1rem] hairline px-4 py-3">
                                <p className="micro-copy">Assistant</p>
                                <p className="body-copy mt-2 whitespace-pre-wrap text-sm">{turn.assistantResponse || "No response captured"}</p>
                              </div>
                            </div>

                            <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-5">
                              <div>
                                <p className="micro-copy">Relevancy</p>
                                <p className="body-copy mt-1">{formatMetric(turn.relevancyScore, 2)}</p>
                              </div>
                              <div>
                                <p className="micro-copy">Faithfulness</p>
                                <p className="body-copy mt-1">{formatMetric(turn.groundednessScore, 2)}</p>
                              </div>
                              <div>
                                <p className="micro-copy">Window score</p>
                                <p className="body-copy mt-1">{formatMetric(turn.knowledgeScore, 2)}</p>
                              </div>
                              <div>
                                <p className="micro-copy">Prompt align</p>
                                <p className="body-copy mt-1">{formatMetric(turn.promptAlignmentScore, 2)}</p>
                              </div>
                              <div>
                                <p className="micro-copy">Latency</p>
                                <p className="body-copy mt-1">{turn.latency != null ? `${formatMetric(turn.latency, 1)} ms` : "—"}</p>
                              </div>
                            </div>

                            <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-3">
                              <div>
                                <p className="micro-copy">Expected check</p>
                                <p className="body-copy mt-1 text-sm whitespace-pre-wrap">{turn.expectedCheck || "—"}</p>
                              </div>
                              <div>
                                <p className="micro-copy">Sliding window</p>
                                <p className="body-copy mt-1 text-sm whitespace-pre-wrap">
                                  {turn.evaluationWindow.join(" | ") || turn.windowReference || "—"}
                                </p>
                              </div>
                              <div>
                                <p className="micro-copy">Unresolved intent</p>
                                <p className="body-copy mt-1 text-sm whitespace-pre-wrap">{turn.unresolvedIntents.join(", ") || "resolved"}</p>
                              </div>
                            </div>

                            <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
                              <div>
                                <p className="micro-copy">Retrieval context</p>
                                <p className="body-copy mt-1 text-sm whitespace-pre-wrap">{turn.retrievalContext || "No retrieval context attached"}</p>
                              </div>
                              <div>
                                <p className="micro-copy">Faithfulness reason</p>
                                <p className="body-copy mt-1 text-sm whitespace-pre-wrap">{turn.groundednessReason || "No groundedness reasoning available"}</p>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>

                      <div className="panel-surface panel-quiet space-y-4">
                        <div>
                          <p className="section-caption mb-2">Reviewer Notes</p>
                          <h3 className="section-heading">Suggested focus</h3>
                        </div>
                        <div className="space-y-3">
                          {selectedConversation.noteCandidates.map((note) => (
                            <div key={note} className="rounded-[1rem] hairline px-4 py-3">
                              <p className="body-copy text-sm">{note}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </section>
          )}

          {traceExplorerRows.length > 0 && selectedTrace && (
            <section className="motion-rise motion-delay-1 space-y-4">
              <div>
                <p className="section-caption mb-2">Agent Trace Terminal</p>
                <h2 className="section-heading">Span-first execution trace</h2>
                <p className="page-subtitle max-w-3xl text-sm">
                  View agent execution in a terminal-like but readable tree structure; step type badge, duration, metric score, pass/fail status, and raw payload drawer all in one place.
                </p>
              </div>

              <div className="motion-stagger-grid grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
                <div className="panel-surface panel-quiet space-y-2">
                  <p className="metric-label">Trace Cases</p>
                  <p className="metric-value text-2xl">{formatCount(traceExplorerRows.length)}</p>
                  <p className="micro-copy">Cases carrying span-first trace payloads.</p>
                </div>
                <div className="panel-surface panel-quiet space-y-2">
                  <p className="metric-label">Failed Spans</p>
                  <p className="metric-value text-2xl">{formatCount(failedTraceSpans)}</p>
                  <p className="micro-copy">Span count already marked failed inside the trace.</p>
                </div>
                <div className="panel-surface panel-quiet space-y-2">
                  <p className="metric-label">Tool Steps</p>
                  <p className="metric-value text-2xl">{formatCount(toolTraceSpans)}</p>
                  <p className="micro-copy">Tool executions visible in the current report.</p>
                </div>
                <div className="panel-surface panel-quiet space-y-2">
                  <p className="metric-label">Avg Trace Duration</p>
                  <p className="metric-value text-2xl">{avgTraceDuration != null ? `${formatMetric(avgTraceDuration, 1)} ms` : "—"}</p>
                  <p className="micro-copy">Average root duration across visible traces.</p>
                </div>
              </div>

              <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(18rem,0.72fr)_minmax(0,1.28fr)]">
                <div className="panel-surface panel-quiet space-y-4">
                  <div>
                    <p className="section-caption mb-2">Trace Queue</p>
                    <h3 className="section-heading">Trace-bearing cases</h3>
                  </div>
                  <div className="space-y-3">
                    {traceExplorerRows.map((row) => {
                      const isSelected = row.key === selectedTrace.key;
                      return (
                        <button
                          key={row.key}
                          type="button"
                          onClick={() => setSelectedTraceKey(row.key)}
                          className={`w-full rounded-[1.2rem] px-4 py-3 text-left transition body-copy ${
                            isSelected
                              ? "subpanel-selected"
                              : "subpanel"
                          }`}
                        >
                          <div className="flex flex-wrap items-start justify-between gap-3">
                            <div>
                              <p className="body-copy font-semibold text-[rgba(40,32,18,0.97)]">{row.caseId}</p>
                              <p className="mt-1 text-[0.78rem] font-medium text-[rgba(80,65,40,0.82)]">{row.model} · {row.testName}</p>
                            </div>
                            <span className="provider-chip">{row.failedSpans} failed</span>
                          </div>
                          <div className="mt-3 flex flex-wrap gap-2">
                            <span className="provider-chip">{row.totalSpans} spans</span>
                            <span className="provider-chip font-mono text-[0.7rem]">{row.traceId}</span>
                            {row.durationMs != null && <span className="provider-chip">{formatMetric(row.durationMs, 1)} ms</span>}
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="panel-surface panel-quiet space-y-4">
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div>
                        <p className="section-caption mb-2">Selected Trace</p>
                        <h3 className="section-heading">{selectedTrace.caseId}</h3>
                        <p className="micro-copy mt-2">{selectedTrace.model} · {selectedTrace.testName} · {selectedTrace.traceId}</p>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        <span className="provider-chip">{selectedTrace.totalSpans} spans</span>
                        <span className="provider-chip">{selectedTrace.failedSpans} failed</span>
                        {selectedTrace.durationMs != null && <span className="provider-chip">{formatMetric(selectedTrace.durationMs, 1)} ms</span>}
                      </div>
                    </div>

                    <div className="rounded-[1rem] hairline px-4 py-3">
                      <p className="micro-copy">Top reasoning</p>
                      <p className="body-copy mt-2 text-sm whitespace-pre-wrap">{selectedTrace.topReasoning}</p>
                    </div>

                    <div className="space-y-3 rounded-[1.2rem] hairline bg-[rgba(34,29,24,0.96)] px-4 py-4 text-[0.92rem] text-[rgba(245,238,227,0.92)] shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
                      {selectedTrace.spans.map((span) => {
                        const isFailed = span.status === "failed";
                        const statusTone = isFailed ? "text-[rgba(255,164,146,0.98)]" : span.status === "partial" ? "text-[rgba(255,214,140,0.95)]" : "text-[rgba(181,240,205,0.96)]";

                        return (
                          <details
                            key={span.spanId}
                            className={`rounded-[0.95rem] border px-3 py-3 ${
                              isFailed
                                ? "border-[rgba(174,57,31,0.35)] bg-[rgba(78,28,19,0.55)]"
                                : "border-[rgba(255,255,255,0.08)] bg-[rgba(255,255,255,0.03)]"
                            }`}
                            open={span.depth < 2}
                            style={{ marginLeft: `${span.depth * 18}px` }}
                          >
                            <summary className="cursor-pointer list-none">
                              <div className="flex flex-wrap items-center justify-between gap-3">
                                <div className="flex min-w-0 items-center gap-3">
                                  <span className="font-mono text-[rgba(210,200,185,0.92)]">{span.depth > 0 ? `${"· ".repeat(span.depth)}` : "root"}</span>
                                  <span className="rounded-full border border-[rgba(255,255,255,0.1)] px-2 py-1 text-[0.72rem] font-semibold tracking-[0.08em] text-[rgba(255,240,214,0.95)]">
                                    {span.spanType}
                                  </span>
                                  <span className="truncate font-mono text-[0.88rem]">{span.name}</span>
                                </div>
                                <div className="flex flex-wrap items-center gap-3 text-[0.8rem]">
                                  <span className={statusTone}>{span.status}</span>
                                  <span>{span.durationMs != null ? `${formatMetric(span.durationMs, 1)} ms` : "—"}</span>
                                  <span>{span.metricScore != null ? `score ${formatMetric(span.metricScore, 2)}` : "score —"}</span>
                                </div>
                              </div>
                            </summary>

                            <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(18rem,0.95fr)]">
                              <div className="space-y-3">
                                <div>
                                  <p className="micro-copy text-[rgba(184,176,165,0.9)]">Input</p>
                                  <p className="mt-1 whitespace-pre-wrap font-mono text-[0.82rem] text-[rgba(245,238,227,0.92)]">{span.inputSummary || "—"}</p>
                                </div>
                                <div>
                                  <p className="micro-copy text-[rgba(184,176,165,0.9)]">Output</p>
                                  <p className="mt-1 whitespace-pre-wrap font-mono text-[0.82rem] text-[rgba(245,238,227,0.92)]">{span.outputSummary || "—"}</p>
                                </div>
                                {span.error && (
                                  <div>
                                    <p className="micro-copy text-[rgba(255,177,160,0.92)]">Error</p>
                                    <p className="mt-1 whitespace-pre-wrap font-mono text-[0.82rem] text-[rgba(255,205,192,0.96)]">{span.error}</p>
                                  </div>
                                )}
                                {span.reasoning && (
                                  <div>
                                    <p className="micro-copy text-[rgba(184,176,165,0.9)]">Reasoning</p>
                                    <p className="mt-1 whitespace-pre-wrap text-[0.82rem] text-[rgba(245,238,227,0.92)]">{span.reasoning}</p>
                                  </div>
                                )}
                              </div>

                              <div>
                                <p className="micro-copy text-[rgba(184,176,165,0.9)]">Raw payload drawer</p>
                                <pre className="mt-2 max-h-56 overflow-auto rounded-[0.85rem] border border-[rgba(255,255,255,0.08)] bg-[rgba(10,8,7,0.68)] p-3 font-mono text-[0.75rem] text-[rgba(230,221,207,0.92)]">
{JSON.stringify({ metadata: span.metadata, metric_results: span.rawMetrics }, null, 2)}
                                </pre>
                              </div>
                            </div>
                          </details>
                        );
                      })}
                    </div>
                  </div>
                </div>
              </div>
            </section>
          )}

          {hasRunEfficiencySummary && (
            <section className="motion-rise motion-delay-1 space-y-4">
              <div>
                <p className="section-caption mb-2">Run Summary</p>
                <h2 className="section-heading">Efficiency Pulse</h2>
                <p className="page-subtitle max-w-3xl text-sm">
                  Quickly read token, spend, and latency signals for the selected run at a high summary level.
                </p>
              </div>

              <div className="motion-stagger-grid grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-5">
                <div className="panel-surface panel-quiet space-y-2">
                  <p className="metric-label">Visible Cost</p>
                  <p className="metric-value text-2xl">{formatMetric(totalProviderCost, 4)}</p>
                  <p className="micro-copy">
                    {dominantProviderCost
                      ? `${dominantProviderCost.provider} holds ${formatMetric(dominantProviderCost.costShare * 100, 1)}% cost share`
                      : "Cost telemetry not available"}
                  </p>
                </div>

                <div className="panel-surface panel-quiet space-y-2">
                  <p className="metric-label">Leanest Model</p>
                  <p className="metric-value text-xl">{leanestModel?.model ?? "—"}</p>
                  <p className="micro-copy">
                    {leanestModel?.avg_tokens_per_eval != null
                      ? `${formatMetric(leanestModel.avg_tokens_per_eval, 1)} tokens / eval`
                      : "Token efficiency not available"}
                  </p>
                </div>

                <div className="panel-surface panel-quiet space-y-2">
                  <p className="metric-label">Leanest Provider Spend</p>
                  <p className="metric-value text-xl">{leanestProviderCost?.provider ?? "—"}</p>
                  <p className="micro-copy">
                    {leanestProviderCost?.costPer1kTokens != null
                      ? `${formatMetric(leanestProviderCost.costPer1kTokens, 4)} cost / 1K tokens`
                      : "Provider-normalized cost not available"}
                  </p>
                </div>

                <div className="panel-surface panel-quiet space-y-2">
                  <p className="metric-label">Best Cost Yield</p>
                  <p className="metric-value text-xl">{strongestCostYieldModel?.model ?? "—"}</p>
                  <p className="micro-copy">
                    {strongestCostYieldModel?.qualityCostEfficiency != null
                      ? `${formatMetric(strongestCostYieldModel.qualityCostEfficiency, 4)} quality / cost`
                      : "Quality-per-cost not available"}
                  </p>
                </div>

                <div className="panel-surface panel-quiet space-y-2">
                  <p className="metric-label">Slowest Model</p>
                  <p className="metric-value text-xl">{slowestLatencyModel?.model ?? "—"}</p>
                  <p className="micro-copy">
                    {slowestLatencyModel?.avgLatency != null
                      ? `${formatMetric(slowestLatencyModel.avgLatency, 2)} avg latency`
                      : "Latency telemetry not available"}
                  </p>
                </div>
              </div>
            </section>
          )}

          {disagreement && disagreement.total_panel_cases > 0 && (
            <section className="motion-rise motion-delay-2 space-y-4">
              <div>
                <p className="section-caption mb-2">Judge Desk</p>
                <h2 className="section-heading">Judge Disagreement Radar</h2>
                <p className="page-subtitle max-w-3xl text-sm">
                  See where primary and secondary judges diverge, and which examples produce the strongest signals for the queue.
                </p>
              </div>

              <div className="motion-stagger-grid grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
                <div className="panel-surface panel-quiet space-y-2">
                  <p className="metric-label">Panel Cases</p>
                  <p className="metric-value text-2xl">{formatCount(disagreement.total_panel_cases)}</p>
                  <p className="micro-copy">Cases with secondary judge data</p>
                </div>
                <div className="panel-surface panel-quiet space-y-2">
                  <p className="metric-label">High Splits</p>
                  <p className="metric-value text-2xl">{formatCount(disagreement.high_disagreement_cases)}</p>
                  <p className="micro-copy">Most critical splits for the queue</p>
                </div>
                <div className="panel-surface panel-quiet space-y-2">
                  <p className="metric-label">Strongest Split Model</p>
                  <p className="metric-value text-xl">{disagreement.strongest_split_model ?? "—"}</p>
                  <p className="micro-copy">Leader in mean disagreement</p>
                </div>
                <div className="panel-surface panel-quiet space-y-2">
                  <p className="metric-label">Recommended Queue</p>
                  <p className="metric-value text-2xl">{formatCount(disagreement.recommended_queue_size)}</p>
                  <p className="micro-copy">Signal pool selected for auto-review</p>
                </div>
              </div>

              <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1.05fr)_minmax(0,1.2fr)]">
                <div className="panel-surface panel-quiet space-y-4">
                  <div>
                    <p className="section-caption mb-2">Model Split Table</p>
                    <h3 className="section-heading">Mean Disagreement by Model</h3>
                  </div>
                  <div className="space-y-3">
                    {disagreement.by_model.map((entry) => (
                      <div key={entry.model} className="rounded-[1.2rem] hairline p-4">
                        <div className="flex items-start justify-between gap-4">
                          <div>
                            <p className="body-copy font-semibold">{entry.model}</p>
                            <p className="micro-copy mt-1">{entry.high_disagreement_count} high-split cases</p>
                          </div>
                          <ScoreBadge score={entry.mean_disagreement} variant="disagreement" />
                        </div>
                        <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
                          <div>
                            <p className="micro-copy">Panel Cases</p>
                            <p className="body-copy mt-1">{entry.panel_case_count}</p>
                          </div>
                          <div>
                            <p className="micro-copy">Mean Disagreement</p>
                            <p className="body-copy mt-1">{formatMetric(entry.mean_disagreement, 2)}</p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="panel-surface panel-quiet space-y-4">
                  <div>
                    <p className="section-caption mb-2">Top Queue Candidates</p>
                    <h3 className="section-heading">Most Polarized Cases</h3>
                  </div>
                  <div className="space-y-3">
                    {topDisagreementCases.map((item) => (
                      <div key={`${item.model}-${item.test_name}-${item.test_id}`} className="rounded-[1.2rem] hairline p-4">
                        <div className="flex flex-wrap items-center justify-between gap-3">
                          <div>
                            <p className="section-caption mb-1">{item.model} · {item.test_name}</p>
                            <p className="body-copy font-medium">{item.question || item.test_id}</p>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="provider-chip">Priority {item.review_priority.toFixed(1)}</span>
                            <ScoreBadge score={item.judge_disagreement} variant="disagreement" />
                          </div>
                        </div>
                        <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
                          <div>
                            <p className="micro-copy">Primary</p>
                            <p className="body-copy mt-1">{item.primary_judge_label ?? "—"} {item.primary_judge_score != null ? `· ${item.primary_judge_score.toFixed(2)}` : ""}</p>
                          </div>
                          <div>
                            <p className="micro-copy">Secondary</p>
                            <p className="body-copy mt-1">{item.secondary_judge_label ?? "—"} {item.secondary_judge_score != null ? `· ${item.secondary_judge_score.toFixed(2)}` : ""}</p>
                          </div>
                        </div>
                        <div className="mt-3 flex items-start gap-2 rounded-[1rem] callout-warn px-3 py-2">
                          <AlertTriangle size={16} className="mt-0.5 callout-warn-icon" />
                          <p className="micro-copy">{item.queue_reason}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </section>
          )}

          {policySummary && policySummary.total_policy_cases > 0 && (
            <section className="motion-rise motion-delay-2 space-y-4">
              <div>
                <p className="section-caption mb-2">Policy Desk</p>
                <h2 className="section-heading">Policy-Aware Review Summary</h2>
                <p className="page-subtitle max-w-3xl text-sm">
                  Review safety and policy cases under a unified taxonomy; quickly separate which families generate queue, severity, and review needs.
                </p>
              </div>

              <div className="motion-stagger-grid grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
                <div className="panel-surface panel-quiet space-y-2">
                  <p className="metric-label">Policy Cases</p>
                  <p className="metric-value text-2xl">{formatCount(policySummary.total_policy_cases)}</p>
                  <p className="micro-copy">Safety and policy cases in taxonomy</p>
                </div>
                <div className="panel-surface panel-quiet space-y-2">
                  <p className="metric-label">Flagged Cases</p>
                  <p className="metric-value text-2xl">{formatCount(policySummary.flagged_case_count)}</p>
                  <p className="micro-copy">Cases carrying violation, queue, or guardrail signals</p>
                </div>
                <div className="panel-surface panel-quiet space-y-2">
                  <p className="metric-label">High Severity</p>
                  <p className="metric-value text-2xl">{formatCount(policySummary.high_severity_case_count)}</p>
                  <p className="micro-copy">Cases reaching high or critical risk levels</p>
                </div>
                <div className="panel-surface panel-quiet space-y-2">
                  <p className="metric-label">Queue Candidates</p>
                  <p className="metric-value text-2xl">{formatCount(policySummary.queue_candidate_count)}</p>
                  <p className="micro-copy">
                    {policySummary.avg_severity != null
                      ? `Avg severity ${formatMetric(policySummary.avg_severity, 2)}`
                      : "Severity not available"}
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)]">
                <div className="panel-surface panel-quiet space-y-4">
                  <div>
                    <p className="section-caption mb-2">Policy Taxonomy</p>
                    <h3 className="section-heading">By Policy Type</h3>
                  </div>

                  {riskLevelEntries.length > 0 && (
                    <div className="flex flex-wrap gap-2">
                      {riskLevelEntries.map(([level, count]) => (
                        <span key={level} className="provider-chip">
                          {level}: {count}
                        </span>
                      ))}
                    </div>
                  )}

                  <div className="space-y-3">
                    {policySummary.by_policy_type.map((item) => (
                      <div key={item.policy_type} className="rounded-[1.2rem] hairline p-4">
                        <div className="flex items-start justify-between gap-4">
                          <div>
                            <p className="body-copy font-semibold">{item.policy_type}</p>
                            <p className="micro-copy mt-1">{item.flagged_cases} flagged · {item.high_severity_cases} high severity</p>
                          </div>
                          {item.avg_severity != null ? <ScoreBadge score={item.avg_severity} variant="disagreement" /> : <span className="provider-chip">n/a</span>}
                        </div>
                        <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
                          <div>
                            <p className="micro-copy">Total Cases</p>
                            <p className="body-copy mt-1">{item.total_cases}</p>
                          </div>
                          <div>
                            <p className="micro-copy">Avg Severity</p>
                            <p className="body-copy mt-1">{formatMetric(item.avg_severity, 2)}</p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="panel-surface panel-quiet space-y-4">
                  <div>
                    <p className="section-caption mb-2">Top Queue Candidates</p>
                    <h3 className="section-heading">Policy Review Queue</h3>
                  </div>
                  <div className="space-y-3">
                    {topPolicyCases.map((item) => (
                      <div key={`${item.model}-${item.test_name}-${item.test_id}`} className="rounded-[1.2rem] hairline p-4">
                        <div className="flex flex-wrap items-center justify-between gap-3">
                          <div>
                            <p className="section-caption mb-1">{item.model} · {item.test_name}</p>
                            <p className="body-copy font-medium">{item.question || item.test_id}</p>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="provider-chip">{item.policy_type}</span>
                            <span className="provider-chip">{item.risk_level}</span>
                            {item.severity != null && <ScoreBadge score={item.severity} variant="disagreement" />}
                          </div>
                        </div>
                        <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
                          <div>
                            <p className="micro-copy">Flagged</p>
                            <p className="body-copy mt-1">{item.flagged ? "yes" : "no"}</p>
                          </div>
                          <div>
                            <p className="micro-copy">Violation Detected</p>
                            <p className="body-copy mt-1">{item.violation_detected ? "yes" : "no"}</p>
                          </div>
                        </div>
                        <div className="mt-3 flex items-start gap-2 rounded-[1rem] callout-warn px-3 py-2">
                          <AlertTriangle size={16} className="mt-0.5 callout-warn-icon" />
                          <p className="micro-copy">{item.queue_reason || "Policy-aware export summary captured this case without a queue reason yet."}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {policyAudit && policyAudit.total_reviews > 0 && (
                <div className="panel-surface panel-quiet space-y-4">
                  <div>
                    <p className="section-caption mb-2">Reviewer Workflow</p>
                    <h3 className="section-heading">Policy Review Audit Trail</h3>
                    <p className="micro-copy mt-1">
                      False positive, confirmed violation ve follow-up kararlarini report seviyesinde geriye donuk izle.
                    </p>
                  </div>

                  <div className="motion-stagger-grid grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
                    <div className="rounded-[1.2rem] hairline p-4">
                      <p className="metric-label">Total Reviews</p>
                      <p className="metric-value text-2xl">{formatCount(policyAudit.total_reviews)}</p>
                      <p className="micro-copy">Latest: {formatTimestamp(policyAudit.latest_review_at)}</p>
                    </div>
                    <div className="rounded-[1.2rem] hairline p-4">
                      <p className="metric-label">Confirmed</p>
                      <p className="metric-value text-2xl">{formatCount(policyAudit.confirmed_violation_count)}</p>
                      <p className="micro-copy">Reviewer onayli ihlaller</p>
                    </div>
                    <div className="rounded-[1.2rem] hairline p-4">
                      <p className="metric-label">False Positives</p>
                      <p className="metric-value text-2xl">{formatCount(policyAudit.false_positive_count)}</p>
                      <p className="micro-copy">Haksiz policy/safety flag duzeltmeleri</p>
                    </div>
                    <div className="rounded-[1.2rem] hairline p-4">
                      <p className="metric-label">Needs Follow-Up</p>
                      <p className="metric-value text-2xl">{formatCount(policyAudit.needs_follow_up_count)}</p>
                      <p className="micro-copy">Ek SME veya policy karari gerekenler</p>
                    </div>
                  </div>

                  <div className="space-y-3">
                    {recentPolicyReviews.map((review) => (
                      <div key={review.annotation_id} className="rounded-[1.2rem] hairline p-4">
                        <div className="flex flex-wrap items-center justify-between gap-3">
                          <div>
                            <p className="section-caption mb-1">{review.model} · {review.test_name}</p>
                            <p className="body-copy font-medium">{review.question || review.test_id}</p>
                          </div>
                          <div className="flex flex-wrap items-center gap-2">
                            <span className="provider-chip">{humanizePolicyDecision(review.decision)}</span>
                            <span className="provider-chip">{review.policy_type}</span>
                            {review.risk_tags.slice(0, 2).map((tag) => (
                              <span key={`${review.annotation_id}-${tag}`} className="provider-chip">{tag}</span>
                            ))}
                          </div>
                        </div>

                        <div className="mt-4 grid grid-cols-1 gap-3 text-sm md:grid-cols-3">
                          <div>
                            <p className="micro-copy">Reviewer</p>
                            <p className="body-copy mt-1">{review.annotator_id || "—"}</p>
                          </div>
                          <div>
                            <p className="micro-copy">Reviewed At</p>
                            <p className="body-copy mt-1">{formatTimestamp(review.timestamp)}</p>
                          </div>
                          <div>
                            <p className="micro-copy">Priority</p>
                            <p className="body-copy mt-1">{formatMetric(review.review_priority, 2)}</p>
                          </div>
                        </div>

                        {review.queue_reason && (
                          <div className="mt-3 flex items-start gap-2 rounded-[1rem] callout-warn px-3 py-2">
                            <AlertTriangle size={16} className="mt-0.5 callout-warn-icon" />
                            <p className="micro-copy">{review.queue_reason}</p>
                          </div>
                        )}

                        {review.notes && (
                          <div className="mt-3 rounded-[1rem] subpanel px-3 py-2">
                            <p className="micro-copy">{review.notes}</p>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </section>
          )}

          {(() => {
            const statComp = report?.statistical_comparison;
            const perModelEntries = statComp ? Object.entries(statComp.per_model ?? {}) : [];
            if (!statComp || perModelEntries.length === 0) return null;
            return (
              <section className="motion-rise motion-delay-2 space-y-4">
                <div>
                  <p className="section-caption mb-2">Statistical Analysis</p>
                  <h2 className="section-heading">Statistical Significance</h2>
                  <p className="page-subtitle max-w-3xl text-sm">
                    Model comparison based on bootstrap CI and Wilcoxon tests.
                  </p>
                  <p className="micro-copy mt-1">
                    α={statComp.alpha}, %{(statComp.confidence * 100).toFixed(0)} CI · seed={statComp.seed}
                  </p>
                </div>

                <div className="panel-surface panel-quiet space-y-4">
                  <div>
                    <p className="section-caption mb-2">Model Metrics</p>
                    <h3 className="section-heading">Per-Model CI</h3>
                  </div>
                  <div className="table-shell overflow-x-auto">
                    <table>
                      <thead>
                        <tr>
                          <th>Model</th>
                          <th>Weighted Score</th>
                          <th>Mean</th>
                          <th>%95 CI</th>
                          <th>n</th>
                          <th></th>
                        </tr>
                      </thead>
                      <tbody>
                        {perModelEntries.map(([modelKey, m]) => (
                          <tr key={`stat-model-${modelKey}`}>
                            <td className="body-copy font-medium">{modelKey}</td>
                            <td className="micro-copy">{m.weighted_score != null ? m.weighted_score.toFixed(4) : "—"}</td>
                            <td className="micro-copy">{m.mean_test_score.toFixed(4)}</td>
                            <td className="micro-copy">[{m.ci_lower.toFixed(4)}, {m.ci_upper.toFixed(4)}]</td>
                            <td className="micro-copy">{m.n_tests}</td>
                            <td>{m.small_sample && <span className="provider-chip">small sample</span>}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {statComp.pairwise.length > 0 && (
                  <div className="panel-surface panel-quiet space-y-4">
                    <div>
                      <p className="section-caption mb-2">Pairwise Comparison</p>
                      <h3 className="section-heading">Pairwise Test</h3>
                    </div>
                    <div className="table-shell overflow-x-auto">
                      <table>
                        <thead>
                          <tr>
                            <th>A</th>
                            <th>B</th>
                            <th>Δ</th>
                            <th>p</th>
                            <th>Wilcoxon p</th>
                            <th>Etki</th>
                            <th>Verdict</th>
                          </tr>
                        </thead>
                        <tbody>
                          {statComp.pairwise.map((pw, i) => (
                            <tr key={`stat-pairwise-${i}-${pw.model_a}-${pw.model_b}`}>
                              <td className="body-copy">{pw.model_a}</td>
                              <td className="body-copy">{pw.model_b}</td>
                              <td className="micro-copy">
                                {pw.mean_difference > 0 ? "+" : ""}{pw.mean_difference.toFixed(4)}
                              </td>
                              <td className="micro-copy">{pw.p_value.toFixed(4)}</td>
                              <td className="micro-copy">{pw.wilcoxon_p_value.toFixed(4)}</td>
                              <td className="micro-copy">{pw.effect_size}</td>
                              <td className="micro-copy">
                                {pw.is_significant ? <span>✅ {pw.verdict}</span> : pw.verdict}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {statComp.warnings.length > 0 && (
                  <div className="space-y-2">
                    {statComp.warnings.map((warning, i) => (
                      <div key={`stat-warning-${i}`} className="flex items-start gap-2 rounded-[1rem] callout-warn px-3 py-2">
                        <AlertTriangle size={16} className="mt-0.5 callout-warn-icon" />
                        <p className="micro-copy">{warning}</p>
                      </div>
                    ))}
                  </div>
                )}
              </section>
            );
          })()}

          <section className="motion-rise motion-delay-2 space-y-4">
            <div>
              <p className="section-caption mb-2">Leaderboard</p>
              <h2 className="section-heading">Model Scores (Average)</h2>
            </div>
            <div className="motion-stagger-grid grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {Object.entries(report.model_scores)
                .sort(([, a], [, b]) => b - a)
                .map(([model, score]) => (
                  <div key={model} className="panel-surface panel-quiet flex items-center justify-between gap-4">
                    <p className="body-copy font-medium">{model}</p>
                    <ScoreBadge score={score} />
                  </div>
                ))}
            </div>
          </section>

          {structuredOutputRows.length > 0 && (
            <section className="motion-rise motion-delay-3 space-y-4">
              <div>
                <p className="section-caption mb-2">Structured Output</p>
                <h2 className="section-heading">Reliability Breakdown</h2>
                <p className="page-subtitle max-w-3xl text-sm">
                  Review schema compliance, most brittle tests, and data sources by model in one table.
                </p>
              </div>

              <div className="table-shell overflow-x-auto">
                <table>
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>Compliance</th>
                      <th>Cases</th>
                      <th>Invalid</th>
                      <th>Top Test</th>
                      <th>Top Dataset</th>
                      <th>Top Schema</th>
                    </tr>
                  </thead>
                  <tbody>
                    {structuredOutputRows.map((row) => (
                      <tr key={`structured-${row.model}`}>
                        <td className="body-copy font-medium">{row.model}</td>
                        <td>{row.complianceRate != null ? <ScoreBadge score={row.complianceRate} /> : "—"}</td>
                        <td className="micro-copy">{formatCount(row.totalCases)}</td>
                        <td className="micro-copy">{formatCount(row.invalidCases)}</td>
                        <td className="micro-copy">
                          {row.topTest ? `${row.topTest[0]} (${row.topTest[1]?.invalid_cases ?? 0})` : "—"}
                        </td>
                        <td className="micro-copy">
                          {row.topDataset ? `${row.topDataset[0]} (${row.topDataset[1]?.invalid_cases ?? 0})` : "—"}
                        </td>
                        <td className="micro-copy">
                          {row.topSchema ? `${row.topSchema[0]} (${row.topSchema[1]?.invalid_cases ?? 0})` : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {modelTrendRows.length > 0 && (
            <section className="motion-rise motion-delay-3 space-y-4">
              <div>
                <p className="section-caption mb-2">Trendlines</p>
                <h2 className="section-heading">Overall Score Time Series</h2>
                <p className="page-subtitle max-w-3xl text-sm">
                  Read the overall score time series and trend direction by model for the selected run directly.
                </p>
              </div>

              <div className="motion-stagger-grid grid grid-cols-1 gap-4 xl:grid-cols-2">
                {modelTrendRows.map((row) => (
                  <div key={`trend-${row.model}`} className="panel-surface panel-quiet space-y-4">
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div>
                        <p className="section-caption mb-2">Model</p>
                        <h3 className="section-heading text-lg">{row.model}</h3>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        <span className="provider-chip">trend: {row.trendLabel}</span>
                        <span className="provider-chip">history {row.historyRuns}</span>
                        <span className="provider-chip">regressions {row.regressions}</span>
                        {row.changePct != null && (
                          <span className="provider-chip">
                            {row.changePct > 0 ? "+" : ""}
                            {row.changePct.toFixed(1)}%
                          </span>
                        )}
                      </div>
                    </div>

                    <div className="h-64 w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={row.points} margin={{ top: 12, right: 12, bottom: 8, left: 0 }}>
                          <CartesianGrid stroke="rgba(129,177,166,0.18)" strokeDasharray="4 4" />
                          <XAxis
                            dataKey="label"
                            tick={{ fill: "rgba(84, 63, 38, 0.88)", fontSize: 12 }}
                            stroke="rgba(129,177,166,0.34)"
                          />
                          <YAxis
                            domain={[0, 1]}
                            tick={{ fill: "rgba(84, 63, 38, 0.88)", fontSize: 12 }}
                            stroke="rgba(129,177,166,0.34)"
                            tickFormatter={(value) => formatMetric(Number(value), 2)}
                          />
                          <Tooltip
                            labelFormatter={(label) => `Run: ${label}`}
                          />
                          <Line
                            type="monotone"
                            dataKey="value"
                            stroke="rgba(25, 94, 124, 0.92)"
                            strokeWidth={2.5}
                            dot={{ r: 3, fill: "rgba(168, 106, 42, 0.95)" }}
                            activeDot={{ r: 5 }}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>

                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div>
                        <p className="micro-copy">Earliest</p>
                        <p className="body-copy mt-1">{formatMetric(row.points[0]?.value, 3)}</p>
                      </div>
                      <div>
                        <p className="micro-copy">Latest</p>
                        <p className="body-copy mt-1">{formatMetric(row.points[row.points.length - 1]?.value, 3)}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

            {hasEfficiencyData && (
              <section className="motion-rise motion-delay-3 space-y-4">
                <div>
                  <p className="section-caption mb-2">Efficiency</p>
                  <h2 className="section-heading">Token Efficiency Scoreboard</h2>
                  <p className="page-subtitle max-w-3xl text-sm">
                    Review quality alongside the token cost per model within the same suite.
                    Better efficiency is: lower token on the left, higher quality on top.
                  </p>
                </div>

                <div className="motion-stagger-grid grid grid-cols-1 gap-4 lg:grid-cols-3">
                  <div className="panel-surface panel-quiet space-y-3">
                    <div className="flex items-center gap-3">
                      <div className="metric-emblem">
                        <Sparkles size={18} />
                      </div>
                      <div>
                        <p className="metric-label">Best Quality Yield</p>
                        <p className="metric-value text-xl">{efficiencyLeaderboard[0]?.model ?? "—"}</p>
                      </div>
                    </div>
                    <p className="micro-copy">
                      {efficiencyLeaderboard[0]?.quality_per_1k_tokens != null
                        ? `${formatMetric(efficiencyLeaderboard[0].quality_per_1k_tokens, 2)} quality / 1K tokens`
                        : "Not enough token telemetry"}
                    </p>
                  </div>

                  <div className="panel-surface panel-quiet space-y-3">
                    <div className="flex items-center gap-3">
                      <div className="metric-emblem">
                        <Zap size={18} />
                      </div>
                      <div>
                        <p className="metric-label">Leanest Model</p>
                        <p className="metric-value text-xl">{leanestModel?.model ?? "—"}</p>
                      </div>
                    </div>
                    <p className="micro-copy">
                      {leanestModel?.avg_tokens_per_eval != null
                        ? `${formatMetric(leanestModel.avg_tokens_per_eval, 1)} tokens / eval`
                        : "Not enough token telemetry"}
                    </p>
                  </div>

                  <div className="panel-surface panel-quiet space-y-3">
                    <div className="flex items-center gap-3">
                      <div className="metric-emblem">
                        <Waypoints size={18} />
                      </div>
                      <div>
                        <p className="metric-label">Pareto Frontier</p>
                        <p className="metric-value text-xl">{formatCount(frontierCount)}</p>
                      </div>
                    </div>
                    <p className="micro-copy">
                      {strongestFrontier
                        ? `${strongestFrontier.model} leads the efficient frontier`
                        : "No frontier point available yet"}
                    </p>
                  </div>
                </div>

                {modelEfficiencyRows.length > 0 && (
                  <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
                    <div className="panel-surface panel-quiet space-y-4">
                      <div>
                        <p className="section-caption mb-2">Bottleneck Panel</p>
                        <h3 className="section-heading">Latency and Cost Hotspots</h3>
                        <p className="micro-copy mt-2">Quickly identify the slowest, costliest, and lowest quality-per-latency signals in the run.</p>
                      </div>

                      <div className="space-y-3">
                        <div className="rounded-[1.2rem] hairline p-4">
                          <div className="flex items-center gap-3">
                            <div className="metric-emblem">
                              <AlertTriangle size={18} />
                            </div>
                            <div>
                              <p className="metric-label">Slowest Average Latency</p>
                              <p className="metric-value text-xl">{slowestLatencyModel?.model ?? "—"}</p>
                            </div>
                          </div>
                          <p className="micro-copy mt-3">
                            {slowestLatencyModel?.avgLatency != null
                              ? `${formatMetric(slowestLatencyModel.avgLatency, 2)} avg latency`
                              : "No latency telemetry"}
                          </p>
                        </div>

                        <div className="rounded-[1.2rem] hairline p-4">
                          <div className="flex items-center gap-3">
                            <div className="metric-emblem">
                              <BarChart3 size={18} />
                            </div>
                            <div>
                              <p className="metric-label">Worst Tail Latency</p>
                              <p className="metric-value text-xl">{slowestP95Model?.model ?? "—"}</p>
                            </div>
                          </div>
                          <p className="micro-copy mt-3">
                            {slowestP95Model?.latencyP95 != null
                              ? `${formatMetric(slowestP95Model.latencyP95, 2)} p95 latency`
                              : "No p95 latency telemetry"}
                          </p>
                        </div>

                        <div className="rounded-[1.2rem] hairline p-4">
                          <div className="flex items-center gap-3">
                            <div className="metric-emblem">
                              <Zap size={18} />
                            </div>
                            <div>
                              <p className="metric-label">Weakest Latency Yield</p>
                              <p className="metric-value text-xl">{weakestLatencyYieldModel?.model ?? "—"}</p>
                            </div>
                          </div>
                          <p className="micro-copy mt-3">
                            {weakestLatencyYieldModel?.qualityLatencyEfficiency != null
                              ? `${formatMetric(weakestLatencyYieldModel.qualityLatencyEfficiency, 4)} quality / latency`
                              : "No quality-per-latency telemetry"}
                          </p>
                        </div>

                        <div className="rounded-[1.2rem] hairline p-4">
                          <div className="flex items-center gap-3">
                            <div className="metric-emblem">
                              <Sparkles size={18} />
                            </div>
                            <div>
                              <p className="metric-label">Highest Total Cost</p>
                              <p className="metric-value text-xl">{costliestModel?.model ?? "—"}</p>
                            </div>
                          </div>
                          <p className="micro-copy mt-3">
                            {costliestModel?.totalCost != null
                              ? `${formatMetric(costliestModel.totalCost, 4)} total cost`
                              : "No cost telemetry"}
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="panel-surface panel-quiet space-y-4">
                      <div>
                        <p className="section-caption mb-2">Bottleneck Table</p>
                        <h3 className="section-heading">Model Efficiency Hotspots</h3>
                      </div>

                      <div className="table-shell overflow-x-auto">
                        <table>
                          <thead>
                            <tr>
                              <th>Model</th>
                              <th>Avg Latency</th>
                              <th>P95</th>
                              <th>Cost</th>
                              <th>Quality / Cost</th>
                              <th>Quality / Latency</th>
                              <th>Error Rate</th>
                            </tr>
                          </thead>
                          <tbody>
                            {modelEfficiencyRows
                              .slice()
                              .sort((left, right) => {
                                const latencyDelta = (right.avgLatency ?? -1) - (left.avgLatency ?? -1);
                                if (latencyDelta !== 0) return latencyDelta;
                                const p95Delta = (right.latencyP95 ?? -1) - (left.latencyP95 ?? -1);
                                if (p95Delta !== 0) return p95Delta;
                                return (right.totalCost ?? -1) - (left.totalCost ?? -1);
                              })
                              .map((row) => (
                                <tr key={`efficiency-bottleneck-${row.model}`}>
                                  <td className="body-copy">{row.model}</td>
                                  <td className="micro-copy">{formatMetric(row.avgLatency, 2)}</td>
                                  <td className="micro-copy">{formatMetric(row.latencyP95, 2)}</td>
                                  <td className="micro-copy">{formatMetric(row.totalCost, 4)}</td>
                                  <td className="micro-copy">{formatMetric(row.qualityCostEfficiency, 4)}</td>
                                  <td className="micro-copy">{formatMetric(row.qualityLatencyEfficiency, 4)}</td>
                                  <td>
                                    {row.errorRate != null ? <ScoreBadge score={1 - row.errorRate} /> : "—"}
                                  </td>
                                </tr>
                              ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                )}

                {normalizedProviderCostRows.length > 0 && (
                  <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
                    <div className="panel-surface panel-quiet space-y-4">
                      <div>
                        <p className="section-caption mb-2">Provider Cost</p>
                        <h3 className="section-heading">Normalized Provider Spend</h3>
                        <p className="micro-copy mt-2">Quickly read provider-based cost share and token-normalized spend signals within the selected report.</p>
                      </div>

                      <div className="space-y-3">
                        <div className="rounded-[1.2rem] hairline p-4">
                          <p className="metric-label">Dominant Spend Provider</p>
                          <p className="metric-value mt-2 text-xl">{dominantProviderCost?.provider ?? "—"}</p>
                          <p className="micro-copy mt-2">
                            {dominantProviderCost
                              ? `${formatMetric(dominantProviderCost.costShare * 100, 1)}% of visible cost`
                              : "No cost telemetry"}
                          </p>
                        </div>

                        <div className="rounded-[1.2rem] hairline p-4">
                          <p className="metric-label">Leanest Provider Spend</p>
                          <p className="metric-value mt-2 text-xl">{leanestProviderCost?.provider ?? "—"}</p>
                          <p className="micro-copy mt-2">
                            {leanestProviderCost?.costPer1kTokens != null
                              ? `${formatMetric(leanestProviderCost.costPer1kTokens, 4)} cost / 1K tokens`
                              : "No token-normalized cost telemetry"}
                          </p>
                        </div>

                        <div className="rounded-[1.2rem] hairline p-4">
                          <p className="metric-label">Provider Coverage</p>
                          <p className="metric-value mt-2 text-xl">{formatCount(normalizedProviderCostRows.length)}</p>
                          <p className="micro-copy mt-2">
                            {formatMetric(totalProviderCost, 4)} total visible cost across {formatCount(modelEfficiencyRows.length)} models
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="panel-surface panel-quiet space-y-4">
                      <div>
                        <p className="section-caption mb-2">Provider Breakdown</p>
                        <h3 className="section-heading">Cost Share and Token Normalization</h3>
                      </div>

                      <div className="table-shell overflow-x-auto">
                        <table>
                          <thead>
                            <tr>
                              <th>Provider</th>
                              <th>Cost Share</th>
                              <th>Total Cost</th>
                              <th>Avg / Model</th>
                              <th>Cost / 1K Tokens</th>
                              <th>Models</th>
                            </tr>
                          </thead>
                          <tbody>
                            {normalizedProviderCostRows.map((row) => (
                              <tr key={`provider-cost-${row.provider}`}>
                                <td className="body-copy">{row.provider}</td>
                                <td className="micro-copy">{formatMetric(row.costShare * 100, 1)}%</td>
                                <td className="micro-copy">{formatMetric(row.totalCost, 4)}</td>
                                <td className="micro-copy">{formatMetric(row.avgCostPerModel, 4)}</td>
                                <td className="micro-copy">{formatMetric(row.costPer1kTokens, 4)}</td>
                                <td className="micro-copy">{row.modelNames.join(", ")}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                )}

                {evaluatorEfficiencyRows.length > 0 && (
                  <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
                    <div className="panel-surface panel-quiet space-y-4">
                      <div>
                        <p className="section-caption mb-2">Evaluator Families</p>
                        <h3 className="section-heading">Metric Execution Footprint</h3>
                        <p className="micro-copy mt-2">
                          Read metric provider families' coverage, score, and notable cost signals in one place.
                        </p>
                      </div>

                      <div className="space-y-3">
                        <div className="rounded-[1.2rem] hairline p-4">
                          <p className="metric-label">Highest Metric Volume</p>
                          <p className="metric-value mt-2 text-xl">{topEvaluatorByVolume?.provider ?? "—"}</p>
                          <p className="micro-copy mt-2">
                            {topEvaluatorByVolume
                              ? `${formatCount(topEvaluatorByVolume.metric_count)} metrics across ${formatCount(topEvaluatorByVolume.case_count)} cases`
                              : "No evaluator telemetry"}
                          </p>
                        </div>

                        <div className="rounded-[1.2rem] hairline p-4">
                          <p className="metric-label">Highest Observed Evaluator Cost</p>
                          <p className="metric-value mt-2 text-xl">{topEvaluatorByObservedCost?.provider ?? "—"}</p>
                          <p className="micro-copy mt-2">
                            {topEvaluatorByObservedCost?.observed_cost != null
                              ? `${formatMetric(topEvaluatorByObservedCost.observed_cost, 4)} observed cost`
                              : "Observed evaluator cost not available"}
                          </p>
                        </div>

                        <div className="rounded-[1.2rem] hairline p-4">
                          <p className="metric-label">Best Evaluator Score</p>
                          <p className="metric-value mt-2 text-xl">{bestEvaluatorScore?.provider ?? "—"}</p>
                          <p className="micro-copy mt-2">
                            {bestEvaluatorScore?.avg_score != null
                              ? `${formatMetric(bestEvaluatorScore.avg_score, 3)} avg normalized score`
                              : "No evaluator score signal"}
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="panel-surface panel-quiet space-y-4">
                      <div>
                        <p className="section-caption mb-2">Evaluator Breakdown</p>
                        <h3 className="section-heading">Coverage and Observed Spend</h3>
                      </div>

                      <div className="table-shell overflow-x-auto">
                        <table>
                          <thead>
                            <tr>
                              <th>Provider</th>
                              <th>Metric Share</th>
                              <th>Cases</th>
                              <th>Avg Score</th>
                              <th>Success</th>
                              <th>Observed Cost</th>
                              <th>Cost / 1K Tokens</th>
                            </tr>
                          </thead>
                          <tbody>
                            {evaluatorEfficiencyRows.map((row) => (
                              <tr key={`evaluator-efficiency-${row.provider}`}>
                                <td className="body-copy">{row.provider}</td>
                                <td className="micro-copy">{formatMetric(row.metric_share * 100, 1)}%</td>
                                <td className="micro-copy">{formatCount(row.case_count)}</td>
                                <td>{row.avg_score != null ? <ScoreBadge score={row.avg_score} /> : "—"}</td>
                                <td className="micro-copy">
                                  {row.success_rate != null ? `${formatMetric(row.success_rate * 100, 1)}%` : "—"}
                                </td>
                                <td className="micro-copy">{formatMetric(row.observed_cost, 4)}</td>
                                <td className="micro-copy">{formatMetric(row.cost_per_1k_tokens, 4)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                )}

                <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1.35fr)_minmax(20rem,1fr)]">
                  <div className="panel-surface panel-quiet space-y-4">
                    <div className="flex items-center justify-between gap-4">
                      <div>
                        <p className="section-caption mb-2">Pareto Map</p>
                        <h3 className="section-heading">Quality vs Token Load</h3>
                      </div>
                      <p className="micro-copy">Frontier points are highlighted</p>
                    </div>
                    <div className="h-80 w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart margin={{ top: 12, right: 12, bottom: 18, left: 0 }}>
                          <CartesianGrid stroke="rgba(129,177,166,0.18)" strokeDasharray="4 4" />
                          <XAxis
                            type="number"
                            dataKey="avg_tokens_per_eval"
                            name="Avg Tokens / Eval"
                            tick={{ fill: "rgba(84, 63, 38, 0.88)", fontSize: 12 }}
                            stroke="rgba(129,177,166,0.34)"
                            tickFormatter={(value) => formatMetric(Number(value), 0)}
                          />
                          <YAxis
                            type="number"
                            domain={[0, 1]}
                            dataKey="overall_score"
                            name="Overall Score"
                            tick={{ fill: "rgba(84, 63, 38, 0.88)", fontSize: 12 }}
                            stroke="rgba(129,177,166,0.34)"
                            tickFormatter={(value) => formatMetric(Number(value), 2)}
                          />
                          <Tooltip content={<EfficiencyTooltip />} cursor={{ strokeDasharray: "4 4" }} />
                          <Scatter
                            data={efficiencyLeaderboard.filter((point) => point.avg_tokens_per_eval != null)}
                            fill="rgba(168, 106, 42, 0.45)"
                          />
                          <Scatter
                            data={efficiencyLeaderboard.filter((point) => point.frontier && point.avg_tokens_per_eval != null)}
                            fill="rgba(25, 94, 124, 0.92)"
                          />
                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  <div className="panel-surface panel-quiet space-y-4">
                    <div>
                      <p className="section-caption mb-2">Leaderboard</p>
                      <h3 className="section-heading">Quality Per Token</h3>
                    </div>
                    <div className="space-y-3">
                      {efficiencyLeaderboard.map((point, index) => (
                        <div key={point.model} className="rounded-[1.2rem] hairline p-4">
                          <div className="flex items-start justify-between gap-4">
                            <div>
                              <p className="section-caption mb-1">#{index + 1}</p>
                              <p className="body-copy text-base font-semibold">{point.model}</p>
                            </div>
                            <div className="flex items-center gap-2">
                              <ScoreBadge score={point.overall_score} />
                              {point.frontier && <span className="provider-chip">Pareto</span>}
                            </div>
                          </div>
                          <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
                            <div>
                              <p className="micro-copy">Avg tokens / eval</p>
                              <p className="body-copy mt-1">{formatMetric(point.avg_tokens_per_eval, 1)}</p>
                            </div>
                            <div>
                              <p className="micro-copy">Quality / 1K tokens</p>
                              <p className="body-copy mt-1">{formatMetric(point.quality_per_1k_tokens, 2)}</p>
                            </div>
                            <div>
                              <p className="micro-copy">Tokens / quality point</p>
                              <p className="body-copy mt-1">{formatMetric(point.tokens_per_quality_point, 1)}</p>
                            </div>
                            <div>
                              <p className="micro-copy">Total tokens</p>
                              <p className="body-copy mt-1">{formatCount(point.total_tokens)}</p>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </section>
            )}

            <section className="motion-rise motion-delay-4 space-y-4">
            <div>
              <p className="section-caption mb-2">Breakdown</p>
              <h2 className="section-heading">Detailed Test Results</h2>
            </div>
            <div className="table-shell overflow-x-auto">
              <table>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Test</th>
                    <th>Score</th>
                    <th>95% CI</th>
                    <th>Intent</th>
                    <th>Items</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(report.models).flatMap(([modelKey, modelData]) => {
                    const tests = (modelData as Record<string, unknown>)?.tests as
                      | Record<string, ResultsTableTestSummary>
                      | undefined;
                    if (!tests) return [];
                    return Object.entries(tests).map(([testName, testResult]) => (
                      <tr key={`${modelKey}-${testName}`}>
                        <td className="body-copy">{modelKey}</td>
                        <td>{testName}</td>
                        <td>
                          {testResult.summary?.overall_score != null ? (
                            <ScoreBadge score={testResult.summary.overall_score} />
                          ) : (
                            "—"
                          )}
                        </td>
                        <td>
                          {(() => {
                            const ci = judgeScoreCi(testResult);
                            return ci ? (
                              <span className="micro-copy whitespace-nowrap">
                                {ci[0].toFixed(3)} – {ci[1].toFixed(3)}
                              </span>
                            ) : (
                              "—"
                            );
                          })()}
                        </td>
                        <td>
                          {testResult.summary?.avg_scores?.intent_resolution != null ? (
                            <div className="space-y-1">
                              <ScoreBadge score={testResult.summary.avg_scores.intent_resolution} />
                              {testResult.summary?.unresolved_intent_summary?.unresolved_turn_rate != null && (
                                <p className="micro-copy">
                                  open rate {formatMetric(testResult.summary.unresolved_intent_summary.unresolved_turn_rate, 2)}
                                </p>
                              )}
                            </div>
                          ) : (
                            "—"
                          )}
                        </td>
                        <td className="micro-copy">{testResult.summary?.total_tests ?? "—"}</td>
                      </tr>
                    ));
                  })}
                </tbody>
              </table>
            </div>
          </section>
        </div>
      )}

      {!selected && !loading && (
        <div className="panel-surface panel-quiet empty-state motion-rise motion-delay-2">
          <BarChart3 size={48} className="mb-3 opacity-40" />
          <p className="body-copy">Select a report to view results.</p>
        </div>
      )}
    </div>
  );
}
