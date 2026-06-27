import { useState } from "react";
import { AlertTriangle, Play, ChevronDown, ChevronRight } from "lucide-react";

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
  const color = cluster.avg_score >= 0.4 ? "#fb923c" : "#f87171";
  return (
    <div style={{ background: "#0d1117", border: "1px solid #1e2736", borderRadius: 6, marginBottom: 8 }}>
      <div onClick={() => setOpen(v => !v)} style={{ display: "flex", alignItems: "center", gap: 12, padding: "12px 14px", cursor: "pointer" }}>
        <span style={{ background: "#1e2736", color: "#9ca3af", fontSize: 11, fontWeight: 700, padding: "2px 7px", borderRadius: 3, minWidth: 60, textAlign: "center" }}>
          {cluster.size} failures
        </span>
        <span style={{ flex: 1, color: "#e2e8f0", fontSize: 13, fontWeight: 600 }}>{cluster.label}</span>
        <span style={{ color, fontSize: 12, fontWeight: 700 }}>{pct}% avg</span>
        {open ? <ChevronDown size={14} color="#6b7280" /> : <ChevronRight size={14} color="#6b7280" />}
      </div>
      {open && (
        <div style={{ borderTop: "1px solid #1e2736", padding: "12px 14px" }}>
          <p style={{ margin: "0 0 10px", fontSize: 11, color: "#6b7280" }}>
            Centroid: <span style={{ color: "#9ca3af" }}>{cluster.centroid_text}</span>
          </p>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr>
                {["Case ID", "Model", "Category", "Score", "Text"].map(h => (
                  <th key={h} style={{ padding: "4px 8px", textAlign: "left", fontSize: 10, fontWeight: 700, color: "#6b7280" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {cluster.members.map(m => (
                <tr key={m.case_id} style={{ borderTop: "1px solid #1e2736" }}>
                  <td style={{ padding: "5px 8px", fontSize: 11, color: "#9ca3af", fontFamily: "monospace" }}>{m.case_id}</td>
                  <td style={{ padding: "5px 8px", fontSize: 11, color: "#9ca3af" }}>{m.model}</td>
                  <td style={{ padding: "5px 8px", fontSize: 11, color: "#9ca3af" }}>{m.category}</td>
                  <td style={{ padding: "5px 8px", fontSize: 11, color: m.score < 0.4 ? "#f87171" : "#fb923c" }}>{(m.score * 100).toFixed(0)}%</td>
                  <td style={{ padding: "5px 8px", fontSize: 11, color: "#94a3b8", maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{m.text}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function BreakdownBar({ data, label }: { data: Record<string, number>; label: string }) {
  const total = Object.values(data).reduce((a, b) => a + b, 0);
  if (total === 0) return null;
  return (
    <div style={{ background: "#111827", border: "1px solid #1e2736", borderRadius: 8, padding: 16 }}>
      <p style={{ margin: "0 0 10px", fontSize: 11, fontWeight: 700, color: "#9ca3af" }}>{label}</p>
      {Object.entries(data).sort((a, b) => b[1] - a[1]).map(([key, count]) => (
        <div key={key} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
          <span style={{ minWidth: 120, fontSize: 12, color: "#9ca3af", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{key}</span>
          <div style={{ flex: 1, height: 6, background: "#1e2736", borderRadius: 3 }}>
            <div style={{ width: `${(count / total) * 100}%`, height: "100%", background: "#f87171", borderRadius: 3 }} />
          </div>
          <span style={{ fontSize: 12, color: "#f87171", fontWeight: 700, minWidth: 20, textAlign: "right" }}>{count}</span>
        </div>
      ))}
    </div>
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
  const [reportText, setReportText] = useState(JSON.stringify(SAMPLE_REPORT, null, 2));
  const [threshold, setThreshold] = useState(0.6);
  const [result, setResult] = useState<ClusteringResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setError(null);
    let report: unknown;
    try {
      report = JSON.parse(reportText);
    } catch {
      setError("Invalid JSON in report field.");
      return;
    }
    setLoading(true);
    try {
      const r = await apiPost<ClusteringResponse>("/failure-clustering", {
        report,
        threshold,
      });
      setResult(r);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ padding: "24px 28px", maxWidth: 960, margin: "0 auto" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 24 }}>
        <AlertTriangle size={22} color="#fbbf24" />
        <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: "#e2e8f0" }}>Failure Clustering</h1>
        <span style={{ fontSize: 12, color: "#6b7280" }}>Auto-taxonomy: turn a failure list into actionable insight</span>
      </div>

      {/* Input */}
      <div style={{ background: "#111827", border: "1px solid #1e2736", borderRadius: 8, padding: 20, marginBottom: 16 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
          <label style={{ fontSize: 12, fontWeight: 700, color: "#9ca3af" }}>EVAL REPORT JSON</label>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <label style={{ fontSize: 12, color: "#9ca3af" }}>
              Threshold:
              <input
                type="number" min={0} max={1} step={0.05}
                value={threshold}
                onChange={e => setThreshold(parseFloat(e.target.value))}
                style={{ marginLeft: 6, width: 60, background: "#0d1117", border: "1px solid #1e2736", borderRadius: 4, color: "#e2e8f0", fontSize: 12, padding: "3px 6px" }}
              />
            </label>
          </div>
        </div>
        <textarea
          value={reportText}
          onChange={e => setReportText(e.target.value)}
          rows={10}
          style={{ width: "100%", boxSizing: "border-box", background: "#0d1117", border: "1px solid #1e2736", borderRadius: 6, color: "#94a3b8", fontSize: 12, padding: "10px 12px", fontFamily: "monospace", resize: "vertical" }}
        />
        <button
          onClick={run}
          disabled={loading}
          style={{ marginTop: 12, display: "flex", alignItems: "center", gap: 6, background: loading ? "#1e2736" : "#d97706", color: "#fff", border: "none", borderRadius: 6, padding: "9px 18px", fontSize: 13, fontWeight: 600, cursor: loading ? "not-allowed" : "pointer" }}
        >
          <Play size={14} /> {loading ? "Clustering…" : "Cluster Failures"}
        </button>
        {error && <p style={{ marginTop: 10, color: "#f87171", fontSize: 13 }}>{error}</p>}
      </div>

      {/* Results */}
      {result && (
        <>
          {/* Summary */}
          <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
            <div style={{ background: "#111827", border: "1px solid #1e2736", borderRadius: 8, padding: "12px 20px", flex: 1, textAlign: "center" }}>
              <div style={{ fontSize: 28, fontWeight: 700, color: "#f87171" }}>{result.total_failures}</div>
              <div style={{ fontSize: 11, color: "#6b7280" }}>total failures</div>
            </div>
            <div style={{ background: "#111827", border: "1px solid #1e2736", borderRadius: 8, padding: "12px 20px", flex: 1, textAlign: "center" }}>
              <div style={{ fontSize: 28, fontWeight: 700, color: "#fbbf24" }}>{result.clusters.length}</div>
              <div style={{ fontSize: 11, color: "#6b7280" }}>clusters</div>
            </div>
            <div style={{ background: "#111827", border: "1px solid #1e2736", borderRadius: 8, padding: "12px 20px", flex: 1, textAlign: "center" }}>
              <div style={{ fontSize: 28, fontWeight: 700, color: "#9ca3af" }}>{(result.threshold * 100).toFixed(0)}%</div>
              <div style={{ fontSize: 11, color: "#6b7280" }}>threshold</div>
            </div>
          </div>

          {result.total_failures === 0 && (
            <div style={{ background: "#14532d22", border: "1px solid #166534", borderRadius: 8, padding: 20, textAlign: "center" }}>
              <p style={{ margin: 0, color: "#4ade80", fontSize: 13 }}>No failures found above threshold. All cases pass!</p>
            </div>
          )}

          {result.clusters.length > 0 && (
            <>
              <div style={{ marginBottom: 16 }}>
                <p style={{ margin: "0 0 10px", fontSize: 12, fontWeight: 700, color: "#9ca3af" }}>CLUSTERS</p>
                {result.clusters.map(c => <ClusterCard key={c.cluster_id} cluster={c} />)}
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <BreakdownBar data={result.model_breakdown} label="BY MODEL" />
                <BreakdownBar data={result.category_breakdown} label="BY CATEGORY" />
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
