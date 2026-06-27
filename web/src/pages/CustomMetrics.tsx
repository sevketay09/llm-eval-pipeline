import { useState } from "react";
import { Sparkles, Play, Plus, Trash2, ChevronDown, ChevronRight } from "lucide-react";

const BASE = "/api";

interface MetricDetail {
  metric_id: string;
  name: string;
  description: string;
  status: string;
  prompt: string;
  created_at: number;
}

interface CaseEvalResult {
  question: string;
  answer: string;
  expected_answer: string;
  score: number | null;
  reasoning: string;
  error: string;
}

interface EvalResponse {
  metric_id: string;
  name: string;
  results: CaseEvalResult[];
  avg_score: number | null;
}

interface EvalCase {
  question: string;
  answer: string;
  expected_answer: string;
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

function ScoreBar({ score }: { score: number | null }) {
  if (score === null) return <span style={{ color: "#6b7280", fontSize: 12 }}>—</span>;
  const pct = Math.round(score * 100);
  const color = score >= 0.7 ? "#4ade80" : score >= 0.4 ? "#fb923c" : "#f87171";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{ flex: 1, height: 6, background: "#1e2736", borderRadius: 3 }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 3 }} />
      </div>
      <span style={{ fontSize: 12, color, fontWeight: 700, minWidth: 32 }}>{pct}%</span>
    </div>
  );
}

function ResultRow({ r }: { r: CaseEvalResult }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <tr onClick={() => setOpen(v => !v)} style={{ cursor: "pointer", borderBottom: "1px solid #1e2736" }}>
        <td style={{ padding: "8px 10px", color: "#e2e8f0", fontSize: 13, maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.question}</td>
        <td style={{ padding: "8px 10px", minWidth: 120 }}><ScoreBar score={r.score} /></td>
        <td style={{ padding: "8px 10px", color: "#9ca3af", fontSize: 11 }}>
          {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </td>
      </tr>
      {open && (
        <tr style={{ background: "#0d1117", borderBottom: "1px solid #1e2736" }}>
          <td colSpan={3} style={{ padding: "12px 16px" }}>
            {r.error && <p style={{ color: "#f87171", fontSize: 12 }}>Error: {r.error}</p>}
            <div style={{ marginBottom: 8 }}>
              <span style={{ color: "#6b7280", fontSize: 11, fontWeight: 600 }}>ANSWER</span>
              <p style={{ margin: "4px 0 0", color: "#e2e8f0", fontSize: 12 }}>{r.answer}</p>
            </div>
            {r.reasoning && (
              <div>
                <span style={{ color: "#6b7280", fontSize: 11, fontWeight: 600 }}>REASONING</span>
                <p style={{ margin: "4px 0 0", color: "#9ca3af", fontSize: 12 }}>{r.reasoning}</p>
              </div>
            )}
          </td>
        </tr>
      )}
    </>
  );
}

export default function CustomMetrics() {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [metric, setMetric] = useState<MetricDetail | null>(null);
  const [cases, setCases] = useState<EvalCase[]>([{ question: "", answer: "", expected_answer: "" }]);
  const [evalResult, setEvalResult] = useState<EvalResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showPrompt, setShowPrompt] = useState(false);

  async function generate() {
    if (!name.trim() || !description.trim()) { setError("Name and description required."); return; }
    setError(null);
    setLoading(true);
    try {
      const m = await apiPost<MetricDetail>("/custom-metrics", { name: name.trim(), description: description.trim() });
      setMetric(m);
      setEvalResult(null);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  async function evaluate() {
    if (!metric) return;
    const validCases = cases.filter(c => c.question.trim() && c.answer.trim());
    if (validCases.length === 0) { setError("Add at least one case with question and answer."); return; }
    setError(null);
    setLoading(true);
    try {
      const r = await apiPost<EvalResponse>(`/custom-metrics/${metric.metric_id}/evaluate`, { cases: validCases });
      setEvalResult(r);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  function updateCase(i: number, field: keyof EvalCase, value: string) {
    setCases(prev => prev.map((c, idx) => idx === i ? { ...c, [field]: value } : c));
  }

  function addCase() {
    setCases(prev => [...prev, { question: "", answer: "", expected_answer: "" }]);
  }

  function removeCase(i: number) {
    setCases(prev => prev.filter((_, idx) => idx !== i));
  }

  return (
    <div style={{ padding: "24px 28px", maxWidth: 900, margin: "0 auto" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 24 }}>
        <Sparkles size={22} color="#a78bfa" />
        <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: "#e2e8f0" }}>Custom Metrics</h1>
        <span style={{ fontSize: 12, color: "#6b7280" }}>Natural language → judge prompt → evaluate</span>
      </div>

      {/* Step 1 — Define metric */}
      <div style={{ background: "#111827", border: "1px solid #1e2736", borderRadius: 8, padding: 20, marginBottom: 16 }}>
        <p style={{ margin: "0 0 12px", fontSize: 12, fontWeight: 700, color: "#9ca3af" }}>1. DEFINE METRIC</p>
        <input
          value={name}
          onChange={e => setName(e.target.value)}
          placeholder="Metric name (e.g. Empathy Score)"
          style={{ width: "100%", boxSizing: "border-box", marginBottom: 10, background: "#0d1117", border: "1px solid #1e2736", borderRadius: 6, color: "#e2e8f0", fontSize: 13, padding: "8px 12px" }}
        />
        <textarea
          value={description}
          onChange={e => setDescription(e.target.value)}
          placeholder="Describe what this metric should measure... (e.g. Rate how empathetic and understanding the response is toward the user's problem, 0=not empathetic, 1=highly empathetic)"
          rows={3}
          style={{ width: "100%", boxSizing: "border-box", background: "#0d1117", border: "1px solid #1e2736", borderRadius: 6, color: "#e2e8f0", fontSize: 13, padding: "10px 12px", resize: "vertical" }}
        />
        <button
          onClick={generate}
          disabled={loading}
          style={{ marginTop: 12, display: "flex", alignItems: "center", gap: 6, background: loading ? "#1e2736" : "#7c3aed", color: "#fff", border: "none", borderRadius: 6, padding: "8px 16px", fontSize: 13, fontWeight: 600, cursor: loading ? "not-allowed" : "pointer" }}
        >
          <Sparkles size={14} /> {loading && !metric ? "Generating…" : "Generate Prompt"}
        </button>
      </div>

      {/* Generated prompt */}
      {metric && (
        <div style={{ background: "#111827", border: "1px solid #1e2736", borderRadius: 8, padding: 20, marginBottom: 16 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
            <p style={{ margin: 0, fontSize: 12, fontWeight: 700, color: "#9ca3af" }}>GENERATED JUDGE PROMPT</p>
            <button onClick={() => setShowPrompt(v => !v)} style={{ background: "none", border: "none", color: "#6b7280", cursor: "pointer", fontSize: 12 }}>
              {showPrompt ? "hide" : "show"}
            </button>
          </div>
          {showPrompt && (
            <pre style={{ margin: 0, color: "#94a3b8", fontSize: 12, whiteSpace: "pre-wrap", fontFamily: "monospace", background: "#0d1117", padding: 12, borderRadius: 6 }}>{metric.prompt}</pre>
          )}
          {!showPrompt && (
            <p style={{ margin: 0, color: "#4ade80", fontSize: 12 }}>Prompt generated for "{metric.name}". Ready to evaluate cases.</p>
          )}
        </div>
      )}

      {/* Step 2 — Add cases */}
      {metric && (
        <div style={{ background: "#111827", border: "1px solid #1e2736", borderRadius: 8, padding: 20, marginBottom: 16 }}>
          <p style={{ margin: "0 0 12px", fontSize: 12, fontWeight: 700, color: "#9ca3af" }}>2. TEST CASES</p>
          {cases.map((c, i) => (
            <div key={i} style={{ marginBottom: 12, padding: 12, background: "#0d1117", borderRadius: 6, position: "relative" }}>
              <div style={{ display: "grid", gap: 6 }}>
                <input value={c.question} onChange={e => updateCase(i, "question", e.target.value)} placeholder="Question" style={{ background: "#111827", border: "1px solid #1e2736", borderRadius: 4, color: "#e2e8f0", fontSize: 12, padding: "6px 10px" }} />
                <input value={c.answer} onChange={e => updateCase(i, "answer", e.target.value)} placeholder="Model answer" style={{ background: "#111827", border: "1px solid #1e2736", borderRadius: 4, color: "#e2e8f0", fontSize: 12, padding: "6px 10px" }} />
                <input value={c.expected_answer} onChange={e => updateCase(i, "expected_answer", e.target.value)} placeholder="Expected answer (optional)" style={{ background: "#111827", border: "1px solid #1e2736", borderRadius: 4, color: "#e2e8f0", fontSize: 12, padding: "6px 10px" }} />
              </div>
              {cases.length > 1 && (
                <button onClick={() => removeCase(i)} style={{ position: "absolute", top: 8, right: 8, background: "none", border: "none", color: "#6b7280", cursor: "pointer" }}>
                  <Trash2 size={14} />
                </button>
              )}
            </div>
          ))}
          <div style={{ display: "flex", gap: 8 }}>
            <button onClick={addCase} style={{ display: "flex", alignItems: "center", gap: 4, background: "#1e2736", color: "#9ca3af", border: "none", borderRadius: 6, padding: "6px 12px", fontSize: 12, cursor: "pointer" }}>
              <Plus size={12} /> Add case
            </button>
            <button onClick={evaluate} disabled={loading} style={{ display: "flex", alignItems: "center", gap: 6, background: loading ? "#1e2736" : "#0284c7", color: "#fff", border: "none", borderRadius: 6, padding: "6px 16px", fontSize: 13, fontWeight: 600, cursor: loading ? "not-allowed" : "pointer" }}>
              <Play size={12} /> {loading && evalResult !== null ? "Evaluating…" : "Evaluate"}
            </button>
          </div>
        </div>
      )}

      {error && <p style={{ color: "#f87171", fontSize: 13, marginBottom: 12 }}>{error}</p>}

      {/* Results */}
      {evalResult && (
        <div style={{ background: "#111827", border: "1px solid #1e2736", borderRadius: 8 }}>
          <div style={{ padding: "12px 16px", borderBottom: "1px solid #1e2736", display: "flex", alignItems: "center", gap: 16 }}>
            <span style={{ fontSize: 13, color: "#9ca3af" }}>{evalResult.results.length} cases</span>
            {evalResult.avg_score !== null && (
              <span style={{ fontSize: 13, color: "#a78bfa", fontWeight: 700 }}>
                Avg: {Math.round(evalResult.avg_score * 100)}%
              </span>
            )}
            {evalResult.avg_score === null && (
              <span style={{ fontSize: 12, color: "#6b7280" }}>Configure a model to get real scores</span>
            )}
          </div>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid #1e2736" }}>
                {["Question", "Score", ""].map(h => (
                  <th key={h} style={{ padding: "8px 10px", textAlign: "left", fontSize: 11, fontWeight: 600, color: "#6b7280" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {evalResult.results.map((r, i) => <ResultRow key={i} r={r} />)}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
