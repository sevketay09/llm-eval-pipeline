import { useState } from "react";
import { Sparkles, Play, Plus, Trash2, ChevronDown, ChevronRight } from "lucide-react";
import {
  PageHeader,
  Card,
  Button,
  ScoreBar,
  Field,
  Input,
  Textarea,
  useToast,
} from "@/components";

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

function ResultRow({ r }: { r: CaseEvalResult }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <tr
        onClick={() => setOpen(v => !v)}
        onKeyDown={e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setOpen(v => !v); } }}
        role="button"
        tabIndex={0}
        aria-expanded={open}
        aria-label={`Case: ${r.question}`}
        className="ds-row-expandable"
      >
        <td style={{ maxWidth: 220, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.question}</td>
        <td style={{ minWidth: 140 }}><ScoreBar score={r.score} /></td>
        <td style={{ width: 28 }}>{open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}</td>
      </tr>
      {open && (
        <tr className="ds-row-detail">
          <td colSpan={3}>
            {r.error && <p className="alert-box alert-danger" style={{ marginBottom: "0.6rem" }}>Error: {r.error}</p>}
            <div style={{ marginBottom: "0.6rem" }}>
              <span className="ds-field-mini">Answer</span>
              <p className="body-copy" style={{ margin: 0, fontSize: "0.85rem" }}>{r.answer}</p>
            </div>
            {r.reasoning && (
              <div>
                <span className="ds-field-mini">Reasoning</span>
                <p className="muted-copy" style={{ margin: 0, fontSize: "0.85rem" }}>{r.reasoning}</p>
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
  const [showPrompt, setShowPrompt] = useState(false);
  const toast = useToast();

  async function generate() {
    if (!name.trim() || !description.trim()) { toast.error("Name and description required."); return; }
    setLoading(true);
    try {
      const m = await apiPost<MetricDetail>("/custom-metrics", { name: name.trim(), description: description.trim() });
      setMetric(m);
      setEvalResult(null);
      toast.success(`Judge prompt generated for "${m.name}".`);
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  async function evaluate() {
    if (!metric) return;
    const validCases = cases.filter(c => c.question.trim() && c.answer.trim());
    if (validCases.length === 0) { toast.error("Add at least one case with question and answer."); return; }
    setLoading(true);
    try {
      const r = await apiPost<EvalResponse>(`/custom-metrics/${metric.metric_id}/evaluate`, { cases: validCases });
      setEvalResult(r);
      toast.success(`Evaluated ${r.results.length} cases.`);
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  function updateCase(i: number, field: keyof EvalCase, value: string) {
    setCases(prev => prev.map((c, idx) => idx === i ? { ...c, [field]: value } : c));
  }

  return (
    <div className="page-shell motion-stagger-stack">
      <PageHeader
        kicker="Custom Metrics"
        title="Author a Judge"
        subtitle="Describe what you want to measure in natural language, generate a judge prompt, then score test cases against it."
        help={
          <>
            <strong>1)</strong> Name your metric and describe — in plain language —
            what good vs. bad looks like. <strong>2)</strong> Generate turns that into
            a reusable LLM-judge prompt. <strong>3)</strong> Add test cases and
            evaluate to see per-case scores. Without a configured model you get a dry
            run; configure one for real scores.
          </>
        }
      />

      {/* Step 1 — Define metric */}
      <Card>
        <p className="section-caption" style={{ marginBottom: "0.9rem" }}>1 · Define metric</p>
        <div className="flex flex-col gap-3">
          <Field label="Metric name">
            <Input value={name} onChange={e => setName(e.target.value)} placeholder="e.g. Empathy Score" />
          </Field>
          <Field label="Description">
            <Textarea
              value={description}
              onChange={e => setDescription(e.target.value)}
              placeholder="Rate how empathetic the response is toward the user's problem (0 = not empathetic, 1 = highly empathetic)…"
              rows={3}
            />
          </Field>
        </div>
        <div className="button-row" style={{ marginTop: "0.9rem" }}>
          <Button icon={<Sparkles size={14} />} loading={loading && !metric} onClick={generate}>
            Generate Prompt
          </Button>
        </div>
      </Card>

      {/* Generated prompt */}
      {metric && (
        <Card>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.6rem" }}>
            <p className="section-caption" style={{ margin: 0 }}>Generated judge prompt</p>
            <button
              className="ds-icon-button"
              style={{ width: "auto", padding: "0.3rem 0.6rem" }}
              aria-expanded={showPrompt}
              aria-label={showPrompt ? "Hide judge prompt" : "Show judge prompt"}
              onClick={() => setShowPrompt(v => !v)}
            >
              {showPrompt ? "hide" : "show"}
            </button>
          </div>
          {showPrompt ? (
            <pre className="ds-codeblock">{metric.prompt}</pre>
          ) : (
            <p className="body-copy" style={{ margin: 0, fontSize: "0.85rem", color: "var(--success)" }}>
              Prompt generated for "{metric.name}". Ready to evaluate cases.
            </p>
          )}
        </Card>
      )}

      {/* Step 2 — Cases */}
      {metric && (
        <Card>
          <p className="section-caption" style={{ marginBottom: "0.9rem" }}>2 · Test cases</p>
          <div className="flex flex-col gap-3">
            {cases.map((c, i) => (
              <div key={i} className="panel-quiet" style={{ position: "relative", borderRadius: 18, border: "1px solid var(--line)", padding: "0.85rem" }}>
                <div className="flex flex-col gap-2">
                  <Input value={c.question} onChange={e => updateCase(i, "question", e.target.value)} placeholder="Question" />
                  <Input value={c.answer} onChange={e => updateCase(i, "answer", e.target.value)} placeholder="Model answer" />
                  <Input value={c.expected_answer} onChange={e => updateCase(i, "expected_answer", e.target.value)} placeholder="Expected answer (optional)" />
                </div>
                {cases.length > 1 && (
                  <button
                    className="ds-icon-button"
                    aria-label={`Remove case ${i + 1}`}
                    style={{ position: "absolute", top: 8, right: 8 }}
                    onClick={() => setCases(prev => prev.filter((_, idx) => idx !== i))}
                  >
                    <Trash2 size={14} />
                  </button>
                )}
              </div>
            ))}
          </div>
          <div className="button-row" style={{ marginTop: "0.9rem" }}>
            <Button variant="secondary" icon={<Plus size={14} />} onClick={() => setCases(prev => [...prev, { question: "", answer: "", expected_answer: "" }])}>
              Add case
            </Button>
            <Button icon={<Play size={14} />} loading={loading && metric != null && evalResult !== null} onClick={evaluate}>
              Evaluate
            </Button>
          </div>
        </Card>
      )}

      {/* Results */}
      {evalResult && (
        <Card style={{ padding: 0 }}>
          <div style={{ padding: "0.9rem 1.1rem", borderBottom: "1px solid var(--line)", display: "flex", alignItems: "center", gap: "1rem" }}>
            <span className="micro-copy">{evalResult.results.length} cases</span>
            {evalResult.avg_score !== null ? (
              <span className="body-copy" style={{ fontWeight: 700, color: "var(--accent)" }}>
                Avg: {Math.round(evalResult.avg_score * 100)}%
              </span>
            ) : (
              <span className="micro-copy">Configure a model to get real scores</span>
            )}
          </div>
          <div className="table-shell" style={{ border: "none", borderRadius: 0, boxShadow: "none" }}>
            <table>
              <thead>
                <tr>{["Question", "Score", ""].map(h => <th key={h}>{h}</th>)}</tr>
              </thead>
              <tbody>
                {evalResult.results.map((r, i) => <ResultRow key={i} r={r} />)}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}
