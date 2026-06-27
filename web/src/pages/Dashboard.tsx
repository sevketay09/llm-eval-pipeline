import { useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Bot,
  CheckCircle2,
  Clock,
  FileText,
  PauseCircle,
  Timer,
} from "lucide-react";
import {
  modelsApi,
  resultsApi,
  type ModelListResponse,
  type ReportListItem,
  type ReportSummary,
} from "@/api/client";

function MetricCard({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Activity;
  label: string;
  value: string;
}) {
  return (
    <div className="metric-tile">
      <div className="metric-emblem">
        <Icon size={20} />
      </div>
      <div>
        <p className="metric-label">{label}</p>
        <p className="metric-value">{value}</p>
      </div>
    </div>
  );
}

type TrendSnapshot = {
  trend?: string;
  change_pct?: number;
  recent_avg?: number;
};

type ContinuityTrendPayload = {
  continuity?: {
    intent_resolution?: TrendSnapshot;
    unresolved_turn_rate?: TrendSnapshot;
  };
  structured_output?: {
    schema_compliance_rate?: TrendSnapshot;
    schema_fail_rate?: TrendSnapshot;
  };
};

type StructuredOutputBreakdownEntry = {
  total_cases?: number;
  valid_cases?: number;
  invalid_cases?: number;
};

type StructuredOutputReliability = {
  total_cases?: number;
  valid_cases?: number;
  invalid_cases?: number;
  schema_compliance_rate?: number;
  error_type_breakdown?: Record<string, number>;
  dataset_breakdown?: Record<string, StructuredOutputBreakdownEntry>;
  schema_type_breakdown?: Record<string, StructuredOutputBreakdownEntry>;
  test_breakdown?: Record<string, StructuredOutputBreakdownEntry>;
};

type ModelComparisonEntry = {
  total_cost?: number | null;
  schema_compliance_rate?: number | null;
  structured_output_reliability?: StructuredOutputReliability | null;
};

type ReportSnapshot = {
  item: ReportListItem;
  summary: ReportSummary | null;
  promptVersion: string | null;
  overallScore: number | null;
};

type RunCaseRow = {
  model: string;
  testName: string;
  datasetLabel: string | null;
  score: number | null;
  failed: boolean;
  skipped: boolean;
  reason: string;
  latencyMs: number | null;
  structuredTracked: boolean;
  structuredValid: boolean;
};

type MetricTableRow = {
  testName: string;
  total: number;
  passed: number;
  failed: number;
  skipped: number;
  avgScore: number | null;
  avgLatencyMs: number | null;
};

function asRecord(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return {};
  }
  return value as Record<string, unknown>;
}

function firstText(...values: unknown[]) {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return "";
}

function extractPromptVersion(metadata: Record<string, unknown> | null | undefined) {
  return firstText(metadata?.judge_prompt_version, metadata?.prompt_version) || null;
}

function formatMetric(value?: number | null, digits = 2) {
  if (value == null || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

function formatPercent(value?: number | null, digits = 1) {
  if (value == null || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

function average(values: Array<number | null | undefined>) {
  const valid = values.filter((value): value is number => typeof value === "number" && !Number.isNaN(value));
  if (!valid.length) return null;
  return valid.reduce((sum, value) => sum + value, 0) / valid.length;
}

function extractOverallScore(summary: ReportSummary | null) {
  if (!summary) return null;
  return average(Object.values(summary.model_scores));
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
  return (
    firstText(
      result.error,
      result.reason,
      result.reasoning,
      result.llm_judge_reasoning,
      details.reason,
      details.reasoning,
      metadata.queue_reason,
      metadata.error_type
    ) || "Low score without explicit reason"
  );
}

function extractRawLatency(result: Record<string, unknown>) {
  const metadata = asRecord(result.metadata);
  const candidates = [result.latency, result.duration_ms, result.elapsed_ms, metadata.latency];
  for (const candidate of candidates) {
    if (typeof candidate === "number" && !Number.isNaN(candidate)) {
      return candidate;
    }
  }
  return null;
}

function isSkippedRawCase(result: Record<string, unknown>) {
  return result.execution_skipped === true || result.skipped === true || result.status === "skipped";
}

function isFailingRawCase(result: Record<string, unknown>) {
  if (isSkippedRawCase(result)) return false;
  if (typeof result.error === "string" && result.error.trim()) return true;
  if (typeof result.passed === "boolean") return result.passed === false;
  if (typeof result.success === "boolean") return result.success === false;

  const score = extractRawCaseScore(result);
  return typeof score === "number" ? score < 0.5 : false;
}

function buildRunCaseRows(rawReport: Record<string, unknown> | null | undefined) {
  const rows: RunCaseRow[] = [];
  const runMetadata = asRecord(rawReport?.run_metadata);
  const customDataset = asRecord(runMetadata.custom_dataset);
  const fallbackDatasetLabel = firstText(customDataset.name, customDataset.path) || null;
  const models = asRecord(rawReport?.models);

  for (const [model, modelPayload] of Object.entries(models)) {
    const tests = asRecord(asRecord(modelPayload).tests);
    for (const [testName, testPayload] of Object.entries(tests)) {
      const testMetadata = asRecord(asRecord(testPayload).metadata);
      const testDatasetLabel = firstText(testMetadata.dataset_label, testMetadata.dataset_name) || fallbackDatasetLabel;
      const results = asRecord(testPayload).results;
      if (!Array.isArray(results)) continue;

      results.forEach((rawResult) => {
        const result = asRecord(rawResult);
        const resultMetadata = asRecord(result.metadata);
        const structuredOutput = asRecord(result.structured_output);
        const parseError = firstText(result.parse_error, structuredOutput.parse_error);
        const schemaError = firstText(result.schema_error, structuredOutput.schema_error);
        const structuredTracked = Object.keys(structuredOutput).length > 0 || Boolean(parseError) || Boolean(schemaError);
        const structuredValid = structuredTracked ? structuredOutput.is_valid === true || (!parseError && !schemaError) : false;

        rows.push({
          model,
          testName,
          datasetLabel: firstText(resultMetadata.dataset_label, resultMetadata.dataset_name, testDatasetLabel) || null,
          score: extractRawCaseScore(result),
          failed: isFailingRawCase(result),
          skipped: isSkippedRawCase(result),
          reason: extractRawCaseReason(result),
          latencyMs: extractRawLatency(result),
          structuredTracked,
          structuredValid,
        });
      });
    }
  }

  return rows;
}

function buildMetricTableRows(rows: RunCaseRow[]) {
  const grouped = rows.reduce<Record<string, MetricTableRow & { scoreTotal: number; scoreCount: number; latencyTotal: number; latencyCount: number }>>(
    (accumulator, row) => {
      const current = accumulator[row.testName] ?? {
        testName: row.testName,
        total: 0,
        passed: 0,
        failed: 0,
        skipped: 0,
        avgScore: null,
        avgLatencyMs: null,
        scoreTotal: 0,
        scoreCount: 0,
        latencyTotal: 0,
        latencyCount: 0,
      };

      current.total += 1;
      if (row.skipped) {
        current.skipped += 1;
      } else if (row.failed) {
        current.failed += 1;
      } else {
        current.passed += 1;
      }

      if (typeof row.score === "number" && !Number.isNaN(row.score)) {
        current.scoreTotal += row.score;
        current.scoreCount += 1;
      }

      if (typeof row.latencyMs === "number" && !Number.isNaN(row.latencyMs)) {
        current.latencyTotal += row.latencyMs;
        current.latencyCount += 1;
      }

      current.avgScore = current.scoreCount ? current.scoreTotal / current.scoreCount : null;
      current.avgLatencyMs = current.latencyCount ? current.latencyTotal / current.latencyCount : null;
      accumulator[row.testName] = current;
      return accumulator;
    },
    {}
  );

  return Object.values(grouped)
    .map(({ scoreTotal: _scoreTotal, scoreCount: _scoreCount, latencyTotal: _latencyTotal, latencyCount: _latencyCount, ...row }) => row)
    .sort((left, right) => {
      if (right.failed !== left.failed) return right.failed - left.failed;
      return right.total - left.total;
    });
}

export default function Dashboard() {
  const [models, setModels] = useState<ModelListResponse | null>(null);
  const [reports, setReports] = useState<ReportListItem[]>([]);
  const [reportSnapshots, setReportSnapshots] = useState<ReportSnapshot[]>([]);
  const [selectedReportFilename, setSelectedReportFilename] = useState("");
  const [selectedRawReport, setSelectedRawReport] = useState<Record<string, unknown> | null>(null);
  const [promptVersionFilter, setPromptVersionFilter] = useState("");
  const [modelFilter, setModelFilter] = useState("");
  const [testFilter, setTestFilter] = useState("");
  const [datasetFilter, setDatasetFilter] = useState("");
  const [error, setError] = useState<string | null>(null);

  const visibleSnapshots = reportSnapshots.filter((snapshot) => {
    if (!promptVersionFilter) return true;
    return (snapshot.promptVersion ?? "") === promptVersionFilter;
  });

  useEffect(() => {
    let cancelled = false;

    async function loadDashboard() {
      try {
        const [modelsResponse, items] = await Promise.all([modelsApi.list(), resultsApi.listReports(12)]);
        if (cancelled) return;

        setModels(modelsResponse);
        setReports(items);

        const summaries = await Promise.all(items.map((item) => resultsApi.getReport(item.filename).catch(() => null)));
        if (cancelled) return;

        const snapshots = items.map((item, index) => {
          const summary = summaries[index] ?? null;
          return {
            item,
            summary,
            promptVersion: extractPromptVersion(summary?.metadata as Record<string, unknown> | undefined),
            overallScore: extractOverallScore(summary),
          } satisfies ReportSnapshot;
        });

        setReportSnapshots(snapshots);
        setSelectedReportFilename((current) => {
          if (current && snapshots.some((snapshot) => snapshot.item.filename === current)) {
            return current;
          }
          return snapshots[0]?.item.filename ?? "";
        });
      } catch (loadError) {
        const message = loadError instanceof Error ? loadError.message : "Dashboard load failed";
        if (!cancelled) {
          setError(message);
        }
      }
    }

    void loadDashboard();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!visibleSnapshots.length) {
      setSelectedReportFilename("");
      return;
    }

    if (!selectedReportFilename || !visibleSnapshots.some((snapshot) => snapshot.item.filename === selectedReportFilename)) {
      setSelectedReportFilename(visibleSnapshots[0]?.item.filename ?? "");
    }
  }, [selectedReportFilename, visibleSnapshots]);

  useEffect(() => {
    if (!selectedReportFilename) {
      setSelectedRawReport(null);
      return;
    }

    let cancelled = false;
    setSelectedRawReport(null);

    resultsApi
      .getRaw(selectedReportFilename)
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
  }, [selectedReportFilename]);

  const selectedSnapshot = reportSnapshots.find((snapshot) => snapshot.item.filename === selectedReportFilename) ?? null;
  const selectedReport = selectedSnapshot?.summary ?? null;
  const selectedMetadata = asRecord(selectedReport?.metadata);
  const selectedPromptVersion = extractPromptVersion(selectedMetadata);
  const runCaseRows = buildRunCaseRows(selectedRawReport);
  const modelOptions = Array.from(new Set(runCaseRows.map((row) => row.model))).sort();
  const testOptions = Array.from(new Set(runCaseRows.map((row) => row.testName))).sort();
  const datasetOptions = Array.from(new Set(runCaseRows.map((row) => row.datasetLabel).filter(Boolean) as string[])).sort();
  const filteredRunCases = runCaseRows.filter((row) => {
    if (modelFilter && row.model !== modelFilter) return false;
    if (testFilter && row.testName !== testFilter) return false;
    if (datasetFilter && row.datasetLabel !== datasetFilter) return false;
    return true;
  });
  const metricTableRows = buildMetricTableRows(filteredRunCases);
  const passedCount = filteredRunCases.filter((row) => !row.failed && !row.skipped).length;
  const failedCount = filteredRunCases.filter((row) => row.failed).length;
  const skippedCount = filteredRunCases.filter((row) => row.skipped).length;
  const avgScore = average(filteredRunCases.filter((row) => !row.skipped).map((row) => row.score));
  const avgLatency = average(filteredRunCases.map((row) => row.latencyMs));
  const structuredTrackedCases = filteredRunCases.filter((row) => row.structuredTracked);
  const schemaCompliance = structuredTrackedCases.length
    ? structuredTrackedCases.filter((row) => row.structuredValid).length / structuredTrackedCases.length
    : null;
  const failureReasons = Object.entries(
    filteredRunCases
      .filter((row) => row.failed)
      .reduce<Record<string, number>>((accumulator, row) => {
        accumulator[row.reason] = (accumulator[row.reason] ?? 0) + 1;
        return accumulator;
      }, {})
  )
    .sort((left, right) => right[1] - left[1])
    .slice(0, 5);
  const failureByTest = Object.entries(
    filteredRunCases
      .filter((row) => row.failed)
      .reduce<Record<string, number>>((accumulator, row) => {
        accumulator[row.testName] = (accumulator[row.testName] ?? 0) + 1;
        return accumulator;
      }, {})
  )
    .sort((left, right) => right[1] - left[1])
    .slice(0, 6);
  const selectedModels = modelFilter ? [modelFilter] : modelOptions;
  const visibleModelCost = selectedModels.reduce((total, model) => {
    const comparison = asRecord(asRecord(selectedReport?.model_comparison)[model]);
    const totalCost = comparison.total_cost;
    return typeof totalCost === "number" && !Number.isNaN(totalCost) ? total + totalCost : total;
  }, 0);
  const efficiencyLeaders = selectedReport?.efficiency?.leaderboard?.slice(0, 3) ?? [];
  const continuityTrendRows = Object.entries(
    (selectedReport?.trends as Record<string, ContinuityTrendPayload> | undefined) ?? {}
  )
    .map(([model, payload]) => ({
      model,
      intentResolution: payload?.continuity?.intent_resolution,
      unresolvedTurnRate: payload?.continuity?.unresolved_turn_rate,
      schemaComplianceRate: payload?.structured_output?.schema_compliance_rate,
      schemaFailRate: payload?.structured_output?.schema_fail_rate,
    }))
    .filter(
      (row) =>
        row.intentResolution ||
        row.unresolvedTurnRate ||
        row.schemaComplianceRate ||
        row.schemaFailRate
    );
  const structuredOutputRows = Object.entries(
    (selectedReport?.model_comparison as Record<string, ModelComparisonEntry> | undefined) ?? {}
  )
    .map(([model, payload]) => {
      const reliability = payload?.structured_output_reliability;
      const errorBreakdown = Object.entries(reliability?.error_type_breakdown ?? {}).sort((a, b) => b[1] - a[1]);
      const datasetBreakdown = Object.entries(reliability?.dataset_breakdown ?? {}).sort(
        (a, b) => (b[1]?.invalid_cases ?? 0) - (a[1]?.invalid_cases ?? 0)
      );
      const schemaBreakdown = Object.entries(reliability?.schema_type_breakdown ?? {}).sort(
        (a, b) => (b[1]?.invalid_cases ?? 0) - (a[1]?.invalid_cases ?? 0)
      );
      const testBreakdown = Object.entries(reliability?.test_breakdown ?? {}).sort(
        (a, b) => (b[1]?.invalid_cases ?? 0) - (a[1]?.invalid_cases ?? 0)
      );

      return {
        model,
        complianceRate: reliability?.schema_compliance_rate ?? payload?.schema_compliance_rate ?? null,
        totalCases: reliability?.total_cases ?? 0,
        invalidCases: reliability?.invalid_cases ?? 0,
        topError: errorBreakdown[0] ?? null,
        hottestTest: testBreakdown[0] ?? null,
        hottestDataset: datasetBreakdown[0] ?? null,
        hottestSchema: schemaBreakdown[0] ?? null,
      };
    })
    .filter((row) => row.totalCases > 0);

  return (
    <div className="page-shell motion-shell">
      <header className="page-header motion-hero">
        <p className="page-kicker">Run Summary</p>
        <h1 className="page-title">Dashboard</h1>
        <p className="page-subtitle">
          Inspect a single run at first glance with health cards, failure distribution, timeline context and filterable slices.
        </p>
      </header>

      {error && (
        <div className="alert-box alert-danger motion-rise motion-delay-1">
          <p className="font-medium">Connection Error</p>
          <p className="mt-1 text-sm">{error}</p>
          <p className="mt-3 text-xs opacity-80">
            Make sure the API is running: <code>uvicorn api.main:app</code>
          </p>
        </div>
      )}

      {!error && (
        <>
          <section className="panel-surface panel-roomy motion-rise motion-delay-1 space-y-5">
            <div className="flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
              <div>
                <p className="section-caption mb-2">Run Selector</p>
                <h2 className="section-heading">Pick a report, then slice its health signals</h2>
                <p className="page-subtitle mt-2 text-sm">
                  Prompt-version filtering narrows the recent timeline; model, test and dataset filters narrow the selected run itself.
                </p>
              </div>
              <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:min-w-[34rem]">
                <div className="control-group">
                  <label className="label">Prompt Version</label>
                  <select value={promptVersionFilter} onChange={(e) => setPromptVersionFilter(e.target.value)} className="control-surface">
                    <option value="">All prompt versions</option>
                    {Array.from(new Set(reportSnapshots.map((snapshot) => snapshot.promptVersion).filter(Boolean) as string[])).map((version) => (
                      <option key={version} value={version}>
                        {version}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="control-group">
                  <label className="label">Report</label>
                  <select
                    value={selectedReportFilename}
                    onChange={(e) => {
                      setSelectedReportFilename(e.target.value);
                      setModelFilter("");
                      setTestFilter("");
                      setDatasetFilter("");
                    }}
                    className="control-surface"
                    disabled={!visibleSnapshots.length}
                  >
                    <option value="">Select report...</option>
                    {visibleSnapshots.map((snapshot) => (
                      <option key={snapshot.item.filename} value={snapshot.item.filename}>
                        {snapshot.item.filename}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            <div className="flex flex-wrap gap-2">
              <span className="provider-chip">Visible reports {visibleSnapshots.length}</span>
              <span className="provider-chip">Selected suite {selectedSnapshot?.item.suite ?? "—"}</span>
              <span className="provider-chip">Prompt {selectedPromptVersion ?? "n/a"}</span>
              <span className="provider-chip">Timestamp {firstText(selectedMetadata.timestamp, selectedSnapshot?.item.modified) || "n/a"}</span>
              {firstText(asRecord(selectedRawReport?.run_metadata).run_id) && (
                <span className="provider-chip">Run {firstText(asRecord(selectedRawReport?.run_metadata).run_id)}</span>
              )}
            </div>

            <div className="grid grid-cols-1 gap-3 md:grid-cols-3 xl:grid-cols-4">
              <div className="control-group">
                <label className="label">Model Filter</label>
                <select value={modelFilter} onChange={(e) => setModelFilter(e.target.value)} className="control-surface">
                  <option value="">All models</option>
                  {modelOptions.map((model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))}
                </select>
              </div>
              <div className="control-group">
                <label className="label">Test Type Filter</label>
                <select value={testFilter} onChange={(e) => setTestFilter(e.target.value)} className="control-surface">
                  <option value="">All tests</option>
                  {testOptions.map((testName) => (
                    <option key={testName} value={testName}>
                      {testName}
                    </option>
                  ))}
                </select>
              </div>
              <div className="control-group">
                <label className="label">Dataset Filter</label>
                <select value={datasetFilter} onChange={(e) => setDatasetFilter(e.target.value)} className="control-surface">
                  <option value="">All datasets</option>
                  {datasetOptions.map((dataset) => (
                    <option key={dataset} value={dataset}>
                      {dataset}
                    </option>
                  ))}
                </select>
              </div>
              <div className="rounded-[1rem] border border-[rgba(136,109,72,0.16)] px-4 py-3">
                <p className="micro-copy">Visible cases</p>
                <p className="body-copy mt-1 text-lg font-semibold">{filteredRunCases.length}</p>
                <p className="micro-copy mt-2">From {runCaseRows.length} total case results in the selected run.</p>
              </div>
            </div>
          </section>

          <div className="motion-stagger-grid mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <MetricCard icon={FileText} label="Visible Cases" value={String(filteredRunCases.length)} />
            <MetricCard icon={CheckCircle2} label="Passed" value={String(passedCount)} />
            <MetricCard icon={AlertTriangle} label="Failed" value={String(failedCount)} />
            <MetricCard icon={PauseCircle} label="Skipped" value={String(skippedCount)} />
            <MetricCard icon={Activity} label="Avg Score" value={formatMetric(avgScore, 3)} />
            <MetricCard icon={Bot} label="Schema Compliance" value={formatPercent(schemaCompliance)} />
            <MetricCard icon={Clock} label="Visible Cost" value={visibleModelCost > 0 ? visibleModelCost.toFixed(4) : "—"} />
            <MetricCard icon={Timer} label="Avg Latency" value={avgLatency != null ? `${avgLatency.toFixed(1)} ms` : "—"} />
          </div>

          <section className="motion-rise motion-delay-2 mt-6 grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1.2fr)_minmax(20rem,0.8fr)]">
            <div className="panel-surface panel-roomy space-y-4">
              <div>
                <p className="section-caption mb-2">Metric Table</p>
                <h2 className="section-heading">Per-test run health</h2>
              </div>
              <div className="table-shell">
                <table>
                  <thead>
                    <tr>
                      <th>Test</th>
                      <th>Total</th>
                      <th>Passed</th>
                      <th>Failed</th>
                      <th>Skipped</th>
                      <th>Avg Score</th>
                      <th>Avg Latency</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metricTableRows.length > 0 ? (
                      metricTableRows.map((row) => (
                        <tr key={row.testName}>
                          <td>{row.testName}</td>
                          <td>{row.total}</td>
                          <td>{row.passed}</td>
                          <td>{row.failed}</td>
                          <td>{row.skipped}</td>
                          <td>{formatMetric(row.avgScore, 3)}</td>
                          <td>{row.avgLatencyMs != null ? `${row.avgLatencyMs.toFixed(1)} ms` : "—"}</td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan={7} className="micro-copy">No case rows match the current filter set.</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="space-y-4">
              <div className="panel-surface panel-roomy space-y-4">
                <div>
                  <p className="section-caption mb-2">Failure Distribution</p>
                  <h2 className="section-heading">Tests with the most regressions</h2>
                </div>
                <div className="space-y-3">
                  {failureByTest.length > 0 ? (
                    failureByTest.map(([testName, count]) => {
                      const share = failedCount > 0 ? count / failedCount : 0;
                      return (
                        <div key={testName}>
                          <div className="flex items-center justify-between gap-3 text-sm">
                            <span className="body-copy">{testName}</span>
                            <span className="micro-copy">{count} fail · {formatPercent(share)}</span>
                          </div>
                          <div className="mt-2 h-2 rounded-full bg-[rgba(136,109,72,0.12)]">
                            <div className="h-2 rounded-full bg-[rgba(168,106,42,0.72)]" style={{ width: `${Math.max(6, share * 100)}%` }} />
                          </div>
                        </div>
                      );
                    })
                  ) : (
                    <p className="body-copy text-sm">No failed cases in the visible slice.</p>
                  )}
                </div>
              </div>

              <div className="panel-surface panel-roomy space-y-4">
                <div>
                  <p className="section-caption mb-2">Top Failure Reasons</p>
                  <h2 className="section-heading">Most common breakpoints</h2>
                </div>
                <div className="space-y-3">
                  {failureReasons.length > 0 ? (
                    failureReasons.map(([reason, count]) => (
                      <div key={`${reason}-${count}`} className="rounded-[1rem] border border-[rgba(136,109,72,0.16)] px-4 py-3">
                        <div className="flex items-center justify-between gap-3">
                          <p className="body-copy text-sm line-clamp-2">{reason}</p>
                          <span className="provider-chip">{count}</span>
                        </div>
                      </div>
                    ))
                  ) : (
                    <p className="body-copy text-sm">No explicit failure reasons in the visible slice.</p>
                  )}
                </div>
              </div>
            </div>
          </section>

          {visibleSnapshots.length > 0 && (
            <section className="motion-rise motion-delay-2 mt-6 space-y-4">
              <div>
                <p className="section-caption mb-2">Timeline</p>
                <h2 className="section-heading">Recent runs in this prompt lane</h2>
              </div>
              <div className="motion-stagger-grid grid grid-cols-1 gap-4 lg:grid-cols-3 xl:grid-cols-4">
                {visibleSnapshots.map((snapshot) => {
                  const isSelected = snapshot.item.filename === selectedReportFilename;
                  const barWidth = Math.max(8, (snapshot.overallScore ?? 0) * 100);
                  return (
                    <button
                      key={snapshot.item.filename}
                      type="button"
                      onClick={() => {
                        setSelectedReportFilename(snapshot.item.filename);
                        setModelFilter("");
                        setTestFilter("");
                        setDatasetFilter("");
                      }}
                      className={`panel-surface panel-quiet space-y-4 text-left ${isSelected ? "border-[rgba(168,106,42,0.45)] bg-[rgba(250,245,238,0.86)]" : ""}`}
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <p className="section-caption mb-2">{snapshot.item.suite ?? "run"}</p>
                          <p className="body-copy font-semibold line-clamp-2">{snapshot.item.filename}</p>
                        </div>
                        {isSelected && <span className="provider-chip">selected</span>}
                      </div>
                      <div className="h-2 rounded-full bg-[rgba(136,109,72,0.12)]">
                        <div className="h-2 rounded-full bg-[rgba(62,128,96,0.75)]" style={{ width: `${barWidth}%` }} />
                      </div>
                      <div className="grid grid-cols-2 gap-3 text-sm">
                        <div>
                          <p className="micro-copy">Overall score</p>
                          <p className="body-copy mt-1">{formatMetric(snapshot.overallScore, 3)}</p>
                        </div>
                        <div>
                          <p className="micro-copy">Prompt</p>
                          <p className="body-copy mt-1">{snapshot.promptVersion ?? "n/a"}</p>
                        </div>
                      </div>
                      <p className="micro-copy">
                        {new Date(snapshot.item.modified).toLocaleDateString("tr-TR", {
                          day: "2-digit",
                          month: "short",
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                      </p>
                    </button>
                  );
                })}
              </div>
            </section>
          )}

          {efficiencyLeaders.length > 0 && (
            <section className="motion-rise motion-delay-2 mt-6 space-y-4">
              <div>
                <p className="section-caption mb-2">Efficiency Snapshot</p>
                <h2 className="section-heading">Selected run token leaders</h2>
              </div>
              <div className="motion-stagger-grid grid grid-cols-1 gap-4 lg:grid-cols-3">
                {efficiencyLeaders.map((point, index) => (
                  <div key={point.model} className="panel-surface panel-quiet space-y-4">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="section-caption mb-2">#{index + 1} Selected Run</p>
                        <p className="text-base font-semibold">{point.model}</p>
                      </div>
                      {point.frontier && <span className="provider-chip">Pareto</span>}
                    </div>
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div>
                        <p className="micro-copy">Quality / 1K tokens</p>
                        <p className="body-copy mt-1">
                          {point.quality_per_1k_tokens != null ? point.quality_per_1k_tokens.toFixed(2) : "—"}
                        </p>
                      </div>
                      <div>
                        <p className="micro-copy">Avg tokens / eval</p>
                        <p className="body-copy mt-1">
                          {point.avg_tokens_per_eval != null ? point.avg_tokens_per_eval.toFixed(1) : "—"}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {continuityTrendRows.length > 0 && (
            <section className="motion-rise motion-delay-2 mt-6 space-y-4">
              <div>
                <p className="section-caption mb-2">Continuity Drift</p>
                <h2 className="section-heading">Selected run multi-turn trend signals</h2>
              </div>
              <div className="motion-stagger-grid grid grid-cols-1 gap-4 lg:grid-cols-3">
                {continuityTrendRows.map((row) => (
                  <div key={row.model} className="panel-surface panel-quiet space-y-3">
                    <div>
                      <p className="section-caption mb-2">Model</p>
                      <p className="text-base font-semibold">{row.model}</p>
                    </div>
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div>
                        <p className="micro-copy">Intent trend</p>
                        <p className="body-copy mt-1">
                          {row.intentResolution?.recent_avg != null
                            ? `${row.intentResolution.recent_avg.toFixed(2)} · ${row.intentResolution.trend ?? "unknown"}`
                            : "—"}
                        </p>
                        <p className="micro-copy mt-1">
                          {row.intentResolution?.change_pct != null
                            ? `${row.intentResolution.change_pct >= 0 ? "+" : ""}${row.intentResolution.change_pct.toFixed(1)}%`
                            : ""}
                        </p>
                      </div>
                      <div>
                        <p className="micro-copy">Unresolved rate trend</p>
                        <p className="body-copy mt-1">
                          {row.unresolvedTurnRate?.recent_avg != null
                            ? `${row.unresolvedTurnRate.recent_avg.toFixed(2)} · ${row.unresolvedTurnRate.trend ?? "unknown"}`
                            : "—"}
                        </p>
                        <p className="micro-copy mt-1">
                          {row.unresolvedTurnRate?.change_pct != null
                            ? `${row.unresolvedTurnRate.change_pct >= 0 ? "+" : ""}${row.unresolvedTurnRate.change_pct.toFixed(1)}%`
                            : ""}
                        </p>
                      </div>
                      <div>
                        <p className="micro-copy">Schema compliance trend</p>
                        <p className="body-copy mt-1">
                          {row.schemaComplianceRate?.recent_avg != null
                            ? `${row.schemaComplianceRate.recent_avg.toFixed(2)} · ${row.schemaComplianceRate.trend ?? "unknown"}`
                            : "—"}
                        </p>
                        <p className="micro-copy mt-1">
                          {row.schemaComplianceRate?.change_pct != null
                            ? `${row.schemaComplianceRate.change_pct >= 0 ? "+" : ""}${row.schemaComplianceRate.change_pct.toFixed(1)}%`
                            : ""}
                        </p>
                      </div>
                      <div>
                        <p className="micro-copy">Schema fail trend</p>
                        <p className="body-copy mt-1">
                          {row.schemaFailRate?.recent_avg != null
                            ? `${row.schemaFailRate.recent_avg.toFixed(2)} · ${row.schemaFailRate.trend ?? "unknown"}`
                            : "—"}
                        </p>
                        <p className="micro-copy mt-1">
                          {row.schemaFailRate?.change_pct != null
                            ? `${row.schemaFailRate.change_pct >= 0 ? "+" : ""}${row.schemaFailRate.change_pct.toFixed(1)}%`
                            : ""}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {structuredOutputRows.length > 0 && (
            <section className="motion-rise motion-delay-2 mt-6 space-y-4">
              <div>
                <p className="section-caption mb-2">Structured Output</p>
                <h2 className="section-heading">Reliability snapshot</h2>
              </div>
              <div className="motion-stagger-grid grid grid-cols-1 gap-4 lg:grid-cols-3">
                {structuredOutputRows.map((row) => (
                  <div key={row.model} className="panel-surface panel-quiet space-y-4">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="section-caption mb-2">Model</p>
                        <p className="text-base font-semibold">{row.model}</p>
                      </div>
                      <span className="provider-chip">
                        {row.complianceRate != null ? `${(row.complianceRate * 100).toFixed(1)}% compliant` : "Tracked"}
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div>
                        <p className="micro-copy">Structured cases</p>
                        <p className="body-copy mt-1">{row.totalCases}</p>
                      </div>
                      <div>
                        <p className="micro-copy">Invalid cases</p>
                        <p className="body-copy mt-1">{row.invalidCases}</p>
                      </div>
                      <div>
                        <p className="micro-copy">Top failure mode</p>
                        <p className="body-copy mt-1">{row.topError ? `${row.topError[0]} (${row.topError[1]})` : "—"}</p>
                      </div>
                      <div>
                        <p className="micro-copy">Most fragile test</p>
                        <p className="body-copy mt-1">
                          {row.hottestTest ? `${row.hottestTest[0]} (${row.hottestTest[1]?.invalid_cases ?? 0})` : "—"}
                        </p>
                      </div>
                      <div>
                        <p className="micro-copy">Noisiest schema</p>
                        <p className="body-copy mt-1">
                          {row.hottestSchema ? `${row.hottestSchema[0]} (${row.hottestSchema[1]?.invalid_cases ?? 0})` : "—"}
                        </p>
                      </div>
                    </div>
                    <div>
                      <p className="micro-copy">Most fragile dataset</p>
                      <p className="body-copy mt-1">
                        {row.hottestDataset ? `${row.hottestDataset[0]} (${row.hottestDataset[1]?.invalid_cases ?? 0}/${row.hottestDataset[1]?.total_cases ?? 0})` : "—"}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {models && (
            <section className="motion-rise motion-delay-3 mt-6 space-y-4">
              <div>
                <p className="section-caption mb-2">Registry</p>
                <h2 className="section-heading">Registered models</h2>
              </div>
              <div className="motion-stagger-grid grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
                {Object.entries(models.models).map(([id, cfg]) => (
                  <div key={id} className="panel-surface panel-quiet space-y-3">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="section-caption mb-2">Model ID</p>
                        <p className="text-base font-semibold">{id}</p>
                      </div>
                      <span className="provider-chip">{cfg.provider}</span>
                    </div>
                    <p className="body-copy font-mono text-sm">{cfg.model_name}</p>
                    <div className="flex flex-wrap gap-3 text-xs muted-copy">
                      <span>temp: {cfg.temperature}</span>
                      <span>max: {cfg.max_tokens}</span>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {reports.length > 0 && (
            <section className="motion-rise motion-delay-4 mt-6 space-y-4">
              <div>
                <p className="section-caption mb-2">Recent Activity</p>
                <h2 className="section-heading">Recent reports</h2>
              </div>
              <div className="table-shell">
                <table>
                  <thead>
                    <tr>
                      <th>Report</th>
                      <th>Suite</th>
                      <th>Models</th>
                      <th>Date</th>
                      <th>Size</th>
                    </tr>
                  </thead>
                  <tbody>
                    {reports.map((report) => (
                      <tr key={report.filename}>
                        <td className="table-code">{report.filename}</td>
                        <td>{report.suite ?? "—"}</td>
                        <td>{report.model_count ?? "—"}</td>
                        <td className="micro-copy">{new Date(report.modified).toLocaleDateString("tr-TR")}</td>
                        <td className="micro-copy">{report.size_kb} KB</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </>
      )}
    </div>
  );
}
