import { useEffect, useState } from "react";
import { Play, ChevronDown, ChevronRight } from "lucide-react";
import {
  PageHeader,
  Card,
  Button,
  Badge,
  Field,
  Input,
  Select,
  Textarea,
  Skeleton,
  useToast,
} from "@/components";
import { scoreTone } from "@/components";
import { resultsApi, type ReportListItem } from "@/api/client";

const BASE = "/api";

interface ClusterMember {
  model: string;
  test: string;
  case_id: string;
  score: number;
  category: string;
  text: string;
}

interface Cluster {
  cluster_id: number;
  size: number;
  label: string;
  centroid_text: string;
  avg_score: number;
  members: ClusterMember[];
}

interface ClusteringResponse {
  total_failures: number;
  threshold: number;
  clusters: Cluster[];
  model_breakdown: Record<string, number>;
  category_breakdown: Record<string, number>;
}

async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

function ClusterCard({ cluster }: { cluster: Cluster }) {
  const [open, setOpen] = useState(false);
  const pct = Math.round(cluster.avg_score * 100);
  return (
    <Card quiet style={{ padding: 0, marginBottom: "0.5rem" }}>
      <div
        onClick={() => setOpen(v => !v)}
        onKeyDown={e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setOpen(v => !v); } }}
        role="button"
        tabIndex={0}
        aria-expanded={open}
        aria-label={`Cluster: ${cluster.label}, ${cluster.size} failures`}
        className="ds-row-expandable"
        style={{ display: "flex", alignItems: "center", gap: "0.8rem", padding: "0.8rem 0.9rem" }}
      >
        <Badge tone="neutral">{cluster.size} failures</Badge>
        <span className="body-copy" style={{ flex: 1, fontWeight: 600, fontSize: "0.88rem" }}>{cluster.label}</span>
        <span className={`ds-scorebar__value is-${scoreTone(cluster.avg_score)}`}>{pct}% avg</span>
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
      </div>
      {open && (
        <div style={{ borderTop: "1px solid var(--line)", padding: "0.8rem 0.9rem" }}>
          <p className="micro-copy" style={{ margin: "0 0 0.7rem" }}>
            Centroid: <span className="body-copy">{cluster.centroid_text}</span>
          </p>
          <div className="table-shell" style={{ border: "none", boxShadow: "none", borderRadius: 0 }}>
            <table>
              <thead>
                <tr>{["Case ID", "Model", "Category", "Score", "Text"].map(h => <th key={h}>{h}</th>)}</tr>
              </thead>
              <tbody>
                {cluster.members.map(m => (
                  <tr key={m.case_id}>
                    <td className="table-code">{m.case_id}</td>
                    <td>{m.model}</td>
                    <td>{m.category}</td>
                    <td><span className={`ds-scorebar__value is-${scoreTone(m.score)}`}>{Math.round(m.score * 100)}%</span></td>
                    <td style={{ maxWidth: 220, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{m.text}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </Card>
  );
}

function BreakdownBar({ data, label }: { data: Record<string, number>; label: string }) {
  const total = Object.values(data).reduce((a, b) => a + b, 0);
  if (total === 0) return null;
  return (
    <Card>
      <p className="section-caption" style={{ marginBottom: "0.7rem" }}>{label}</p>
      {Object.entries(data).sort((a, b) => b[1] - a[1]).map(([key, count]) => (
        <div key={key} style={{ display: "flex", alignItems: "center", gap: "0.7rem", marginBottom: "0.45rem" }}>
          <span className="micro-copy" style={{ minWidth: 120, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{key}</span>
          <div className="ds-scorebar__track" style={{ flex: 1 }}>
            <div className="ds-scorebar__fill is-low" style={{ width: `${(count / total) * 100}%` }} />
          </div>
          <span className="ds-scorebar__value is-low" style={{ minWidth: "1.4rem" }}>{count}</span>
        </div>
      ))}
    </Card>
  );
}

const SAMPLE_REPORT = {
  models: {
    "gpt-4o": {
      tests: {
        qa: {
          results: [
            { case_id: "c1", question: "Product return policy", scores: { overall_score: 0.3 } },
            { case_id: "c2", question: "Shipping duration", scores: { overall_score: 0.4 }, category: "logistics" },
            { case_id: "c3", question: "Refund timeline", scores: { overall_score: 0.35 }, category: "finance" },
            { case_id: "c4", question: "Return process steps", scores: { overall_score: 0.2 }, category: "support" },
          ],
        },
      },
    },
  },
};

export default function FailureClustering() {
  const [source, setSource] = useState<"report" | "json">("report");
  const [reports, setReports] = useState<ReportListItem[]>([]);
  const [reportsLoading, setReportsLoading] = useState(true);
  const [selectedReport, setSelectedReport] = useState("");
  const [reportText, setReportText] = useState("");
  const [threshold, setThreshold] = useState(0.6);
  const [result, setResult] = useState<ClusteringResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const toast = useToast();

  useEffect(() => {
    let cancelled = false;
    resultsApi
      .listReports()
      .then((items) => {
        if (cancelled) return;
        setReports(items);
        setSelectedReport(items[0]?.filename ?? "");
        if (items.length === 0) setSource("json");
      })
      .catch(() => {
        if (!cancelled) setSource("json");
      })
      .finally(() => {
        if (!cancelled) setReportsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  async function run() {
    let report: unknown;
    setLoading(true);
    try {
      if (source === "report") {
        if (!selectedReport) {
          toast.error("Select a report first.");
          return;
        }
        report = await resultsApi.getRaw(selectedReport);
      } else {
        try {
          report = JSON.parse(reportText);
        } catch {
          toast.error("Invalid JSON in report field.");
          return;
        }
      }
      const r = await apiPost<ClusteringResponse>("/failure-clustering", { report, threshold });
      setResult(r);
      toast.success(`Grouped ${r.total_failures} failures into ${r.clusters.length} clusters.`);
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page-shell motion-stagger-stack">
      <PageHeader
        kicker="Failure Clustering"
        title="Failure Taxonomy"
        subtitle="Turn a flat list of failed cases into grouped, actionable clusters — with model and category breakdowns."
        help={
          <>
            Pick a saved eval report (or paste one as JSON). Cases scoring below
            the <strong>threshold</strong> are treated as failures, then clustered by
            similarity so you see recurring failure modes instead of a flat list.
            Expand a cluster to inspect its member cases.
          </>
        }
      />

      {/* Input */}
      <Card>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: "0.8rem", gap: "1rem", flexWrap: "wrap" }}>
          <div className="tab-strip" role="tablist" aria-label="Report source">
            <button
              type="button"
              role="tab"
              aria-selected={source === "report"}
              className={`tab-button ${source === "report" ? "tab-button-active" : ""}`}
              onClick={() => setSource("report")}
            >
              From report
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={source === "json"}
              className={`tab-button ${source === "json" ? "tab-button-active" : ""}`}
              onClick={() => setSource("json")}
            >
              Paste JSON
            </button>
          </div>
          <Field label="Threshold">
            <Input
              type="number" min={0} max={1} step={0.05}
              value={threshold}
              onChange={e => setThreshold(parseFloat(e.target.value))}
              style={{ width: 90 }}
            />
          </Field>
        </div>

        {source === "report" ? (
          reportsLoading ? (
            <Skeleton height="2.6rem" />
          ) : reports.length === 0 ? (
            <p className="micro-copy" style={{ margin: 0 }}>
              No saved reports found. Switch to “Paste JSON” to cluster an ad-hoc report.
            </p>
          ) : (
            <Field label="Eval report">
              <Select value={selectedReport} onChange={e => setSelectedReport(e.target.value)}>
                {reports.map(r => (
                  <option key={r.filename} value={r.filename}>
                    {r.filename}{r.suite ? ` — ${r.suite}` : ""}
                  </option>
                ))}
              </Select>
            </Field>
          )
        ) : (
          <>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.4rem", gap: "0.6rem", flexWrap: "wrap" }}>
              <span className="ds-field-mini" style={{ marginBottom: 0 }}>Eval report JSON</span>
              <button
                type="button"
                className="ds-icon-button"
                style={{ width: "auto", padding: "0.3rem 0.6rem", fontSize: "0.75rem" }}
                onClick={() => setReportText(JSON.stringify(SAMPLE_REPORT, null, 2))}
              >
                Load example
              </button>
            </div>
            <Textarea
              value={reportText}
              onChange={e => setReportText(e.target.value)}
              rows={10}
              className="font-mono"
              placeholder="Paste an eval report JSON, or click “Load example”…"
              style={{ fontSize: "0.76rem" }}
            />
          </>
        )}

        <div className="button-row" style={{ marginTop: "0.9rem" }}>
          <Button icon={<Play size={14} />} loading={loading} onClick={run}>Cluster Failures</Button>
        </div>
      </Card>

      {/* Results */}
      {result && (
        <>
          <div className="grid gap-3 sm:grid-cols-3">
            <Card className="stat-card" style={{ textAlign: "center" }}>
              <div className="stat-value" style={{ fontSize: "1.8rem", color: "var(--danger)" }}>{result.total_failures}</div>
              <p className="stat-label">total failures</p>
            </Card>
            <Card className="stat-card" style={{ textAlign: "center" }}>
              <div className="stat-value" style={{ fontSize: "1.8rem", color: "var(--warning)" }}>{result.clusters.length}</div>
              <p className="stat-label">clusters</p>
            </Card>
            <Card className="stat-card" style={{ textAlign: "center" }}>
              <div className="stat-value" style={{ fontSize: "1.8rem" }}>{Math.round(result.threshold * 100)}%</div>
              <p className="stat-label">threshold</p>
            </Card>
          </div>

          {result.total_failures === 0 && (
            <div className="alert-box alert-success">No failures found above threshold. All cases pass!</div>
          )}

          {result.clusters.length > 0 && (
            <>
              <div>
                <p className="section-caption" style={{ marginBottom: "0.7rem" }}>Clusters</p>
                {result.clusters.map(c => <ClusterCard key={c.cluster_id} cluster={c} />)}
              </div>
              <div className="grid gap-3 sm:grid-cols-2">
                <BreakdownBar data={result.model_breakdown} label="By model" />
                <BreakdownBar data={result.category_breakdown} label="By category" />
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
