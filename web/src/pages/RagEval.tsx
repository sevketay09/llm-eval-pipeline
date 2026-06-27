import { useState } from "react";
import { BookOpen, Play, Plus, Trash2 } from "lucide-react";

const BASE = "/api";

interface RagEvalResponse {
  question: string;
  context_precision: number;
  context_recall: number;
  faithfulness: number;
  answer_relevance: number;
  fault_component: string;
  overall_score: number;
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

function ScoreRow({ label, score, desc }: { label: string; score: number; desc: string }) {
  const pct = Math.round(score * 100);
  const color = score >= 0.7 ? "#4ade80" : score >= 0.4 ? "#fb923c" : "#f87171";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "10px 0", borderBottom: "1px solid #1e2736" }}>
      <div style={{ minWidth: 160 }}>
        <div style={{ fontSize: 13, color: "#e2e8f0", fontWeight: 600 }}>{label}</div>
        <div style={{ fontSize: 11, color: "#6b7280" }}>{desc}</div>
      </div>
      <div style={{ flex: 1, height: 8, background: "#1e2736", borderRadius: 4 }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 4 }} />
      </div>
      <span style={{ minWidth: 44, textAlign: "right", fontSize: 14, fontWeight: 700, color }}>{pct}%</span>
    </div>
  );
}

function FaultBadge({ fault }: { fault: string }) {
  const styles: Record<string, { bg: string; color: string }> = {
    retriever: { bg: "#431a0022", color: "#fb923c" },
    generator: { bg: "#7f1d1d22", color: "#f87171" },
    both: { bg: "#78350f22", color: "#fbbf24" },
    none: { bg: "#14532d22", color: "#4ade80" },
  };
  const st = styles[fault] ?? styles["none"];
  return (
    <span style={{ background: st.bg, color: st.color, fontSize: 12, fontWeight: 700, padding: "3px 9px", borderRadius: 4 }}>
      fault: {fault}
    </span>
  );
}

export default function RagEval() {
  const [question, setQuestion] = useState("");
  const [contexts, setContexts] = useState([""]);
  const [answer, setAnswer] = useState("");
  const [expected, setExpected] = useState("");
  const [result, setResult] = useState<RagEvalResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function evaluate() {
    if (!question.trim() || !answer.trim()) { setError("Question and answer are required."); return; }
    const validCtx = contexts.filter(c => c.trim());
    if (validCtx.length === 0) { setError("Add at least one context chunk."); return; }
    setError(null);
    setLoading(true);
    try {
      const r = await apiPost<RagEvalResponse>("/rag-eval", {
        question: question.trim(),
        contexts: validCtx.map(t => ({ text: t })),
        answer: answer.trim(),
        expected_answer: expected.trim(),
      });
      setResult(r);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ padding: "24px 28px", maxWidth: 900, margin: "0 auto" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 24 }}>
        <BookOpen size={22} color="#38bdf8" />
        <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: "#e2e8f0" }}>RAG Eval</h1>
        <span style={{ fontSize: 12, color: "#6b7280" }}>Retriever vs generator fault isolation</span>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
        {/* Left — inputs */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div style={{ background: "#111827", border: "1px solid #1e2736", borderRadius: 8, padding: 16 }}>
            <label style={{ fontSize: 11, fontWeight: 700, color: "#9ca3af", display: "block", marginBottom: 6 }}>QUESTION</label>
            <textarea value={question} onChange={e => setQuestion(e.target.value)} rows={2} placeholder="What is the return policy?" style={{ width: "100%", boxSizing: "border-box", background: "#0d1117", border: "1px solid #1e2736", borderRadius: 6, color: "#e2e8f0", fontSize: 13, padding: "8px 10px", resize: "vertical" }} />
          </div>

          <div style={{ background: "#111827", border: "1px solid #1e2736", borderRadius: 8, padding: 16 }}>
            <label style={{ fontSize: 11, fontWeight: 700, color: "#9ca3af", display: "block", marginBottom: 6 }}>CONTEXT CHUNKS</label>
            {contexts.map((ctx, i) => (
              <div key={i} style={{ display: "flex", gap: 6, marginBottom: 6 }}>
                <textarea value={ctx} onChange={e => setContexts(prev => prev.map((c, idx) => idx === i ? e.target.value : c))} rows={2} placeholder={`Chunk ${i + 1}...`} style={{ flex: 1, background: "#0d1117", border: "1px solid #1e2736", borderRadius: 6, color: "#e2e8f0", fontSize: 12, padding: "6px 10px", resize: "vertical" }} />
                {contexts.length > 1 && (
                  <button onClick={() => setContexts(prev => prev.filter((_, idx) => idx !== i))} style={{ background: "none", border: "none", color: "#6b7280", cursor: "pointer" }}>
                    <Trash2 size={14} />
                  </button>
                )}
              </div>
            ))}
            <button onClick={() => setContexts(prev => [...prev, ""])} style={{ display: "flex", alignItems: "center", gap: 4, background: "#1e2736", color: "#9ca3af", border: "none", borderRadius: 6, padding: "4px 10px", fontSize: 12, cursor: "pointer" }}>
              <Plus size={12} /> Add chunk
            </button>
          </div>

          <div style={{ background: "#111827", border: "1px solid #1e2736", borderRadius: 8, padding: 16 }}>
            <label style={{ fontSize: 11, fontWeight: 700, color: "#9ca3af", display: "block", marginBottom: 6 }}>MODEL ANSWER</label>
            <textarea value={answer} onChange={e => setAnswer(e.target.value)} rows={2} placeholder="The response generated by your RAG system..." style={{ width: "100%", boxSizing: "border-box", background: "#0d1117", border: "1px solid #1e2736", borderRadius: 6, color: "#e2e8f0", fontSize: 13, padding: "8px 10px", resize: "vertical" }} />
          </div>

          <div style={{ background: "#111827", border: "1px solid #1e2736", borderRadius: 8, padding: 16 }}>
            <label style={{ fontSize: 11, fontWeight: 700, color: "#9ca3af", display: "block", marginBottom: 6 }}>EXPECTED ANSWER (optional)</label>
            <input value={expected} onChange={e => setExpected(e.target.value)} placeholder="Ground truth answer..." style={{ width: "100%", boxSizing: "border-box", background: "#0d1117", border: "1px solid #1e2736", borderRadius: 6, color: "#e2e8f0", fontSize: 13, padding: "8px 10px" }} />
          </div>

          <button onClick={evaluate} disabled={loading} style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 6, background: loading ? "#1e2736" : "#0284c7", color: "#fff", border: "none", borderRadius: 6, padding: "10px", fontSize: 13, fontWeight: 600, cursor: loading ? "not-allowed" : "pointer" }}>
            <Play size={14} /> {loading ? "Evaluating…" : "Evaluate RAG"}
          </button>

          {error && <p style={{ color: "#f87171", fontSize: 13, margin: 0 }}>{error}</p>}
        </div>

        {/* Right — results */}
        <div>
          {result ? (
            <div style={{ background: "#111827", border: "1px solid #1e2736", borderRadius: 8, padding: 20 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
                <FaultBadge fault={result.fault_component} />
                <span style={{ fontSize: 20, fontWeight: 700, color: result.overall_score >= 0.7 ? "#4ade80" : result.overall_score >= 0.4 ? "#fb923c" : "#f87171" }}>
                  {Math.round(result.overall_score * 100)}%
                </span>
              </div>
              <ScoreRow label="Context Precision" score={result.context_precision} desc="Relevant chunks in context" />
              <ScoreRow label="Context Recall" score={result.context_recall} desc="Answer covered by context" />
              <ScoreRow label="Faithfulness" score={result.faithfulness} desc="Answer grounded in context" />
              <ScoreRow label="Answer Relevance" score={result.answer_relevance} desc="Answer addresses question" />
            </div>
          ) : (
            <div style={{ background: "#111827", border: "1px solid #1e2736", borderRadius: 8, padding: 40, display: "flex", alignItems: "center", justifyContent: "center", height: "100%", minHeight: 300 }}>
              <p style={{ color: "#374151", fontSize: 13, textAlign: "center", margin: 0 }}>
                Fill in the inputs and click Evaluate RAG<br />to see component scores
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
