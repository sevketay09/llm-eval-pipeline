import { useEffect, useState } from "react";
import { Play, CheckCircle, XCircle, AlertCircle, ChevronDown, ChevronRight } from "lucide-react";
import { PageHeader, Card, Button, Badge, Field, Select, Textarea, ScoreBar, useToast, DecisionModelSelect } from "@/components";
import { modelsApi } from "@/api/client";
import { splitModels } from "@/lib/decision";

const BASE = "/api";

const ALL_CATEGORIES = [
  "prompt_injection",
  "jailbreak",
  "persona_override",
  "boundary_test",
  "role_confusion",
  "tool_result_injection",
  "tool_poisoning",
] as const;

interface AttackResult {
  attack_id: string;
  category: string;
  name: string;
  payload: string;
  response: string;
  passed: boolean;
  reason: string;
  latency_ms: number;
  error: string;
  signals: Record<string, number>;
  needs_review: boolean;
  scorer: string;
}

interface SessionSummary {
  session_id: string;
  attack_count: number;
  status: string;
  passed: number;
  failed: number;
  scorer: string;
  needs_review: number;
}

interface SessionDetail extends SessionSummary {
  error: string;
  results: AttackResult[];
}

async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(typeof err.detail === "string" ? err.detail : `${r.status} ${r.statusText}`);
  }
  return r.json();
}

async function apiGet<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(typeof err.detail === "string" ? err.detail : `${r.status} ${r.statusText}`);
  }
  return r.json();
}

function VerdictBadge({ passed, error, needsReview }: { passed: boolean; error: string; needsReview: boolean }) {
  if (error) return <Badge tone="warning"><AlertCircle size={11} /> error</Badge>;
  if (needsReview) return <Badge tone="warning"><AlertCircle size={11} /> review</Badge>;
  if (passed) return <Badge tone="success"><CheckCircle size={11} /> passed</Badge>;
  return <Badge tone="danger"><XCircle size={11} /> failed</Badge>;
}

function ResultRow({ r }: { r: AttackResult }) {
  const [open, setOpen] = useState(false);
  const signalEntries = Object.entries(r.signals);
  return (
    <>
      <tr
        onClick={() => setOpen(v => !v)}
        onKeyDown={e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setOpen(v => !v); } }}
        role="button"
        tabIndex={0}
        aria-expanded={open}
        aria-label={`Attack ${r.name}, ${r.passed ? "passed" : "failed"}`}
        className="ds-row-expandable"
      >
        <td><Badge tone="neutral" mono>{r.category}</Badge></td>
        <td className="body-copy">{r.name}</td>
        <td><VerdictBadge passed={r.passed} error={r.error} needsReview={r.needs_review} /></td>
        <td className="micro-copy">{r.latency_ms.toFixed(1)} ms</td>
        <td style={{ width: 28 }}>{open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}</td>
      </tr>
      {open && (
        <tr className="ds-row-detail">
          <td colSpan={5}>
            <div style={{ marginBottom: "0.6rem" }}>
              <span className="ds-field-mini">Payload</span>
              <pre className="ds-codeblock">{r.payload}</pre>
            </div>
            <div style={{ marginBottom: "0.6rem" }}>
              <span className="ds-field-mini">Response</span>
              <pre className="ds-codeblock">{r.response || r.error || "—"}</pre>
            </div>
            <div style={{ marginBottom: signalEntries.length ? "0.6rem" : 0 }}>
              <span className="ds-field-mini">Reason</span>
              <p className="muted-copy" style={{ margin: 0, fontSize: "0.85rem" }}>{r.reason || "—"}</p>
            </div>
            {signalEntries.length > 0 && (
              <div>
                <span className="ds-field-mini">Jev signals</span>
                <div className="space-y-1" style={{ marginTop: "0.35rem" }}>
                  {signalEntries.map(([name, value]) => (
                    <div key={name} className="flex items-center gap-3">
                      <span className="micro-copy font-mono" style={{ width: "11rem" }}>{name}</span>
                      <ScoreBar score={value} />
                    </div>
                  ))}
                </div>
              </div>
            )}
          </td>
        </tr>
      )}
    </>
  );
}

export default function RedTeam() {
  const [systemPrompt, setSystemPrompt] = useState("");
  const [categories, setCategories] = useState<string[]>([...ALL_CATEGORIES]);
  const [models, setModels] = useState<string[]>([]);
  const [modelKey, setModelKey] = useState("");
  const [scorer, setScorer] = useState<"heuristic" | "decision">("heuristic");
  const [scorerModel, setScorerModel] = useState("");
  const [loading, setLoading] = useState(false);
  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const toast = useToast();

  useEffect(() => {
    modelsApi
      .list()
      .then(r => {
        const { llm } = splitModels(r.models);
        setModels(llm);
        setModelKey(prev => prev || llm[0] || "");
      })
      .catch(() => setModels([]));
  }, []);

  function toggleCategory(cat: string) {
    setCategories(prev => prev.includes(cat) ? prev.filter(c => c !== cat) : [...prev, cat]);
  }

  async function run() {
    if (!systemPrompt.trim()) { toast.error("System prompt is required."); return; }
    if (categories.length === 0) { toast.error("Select at least one category."); return; }
    if (scorer === "decision" && !scorerModel) { toast.error("Select a decision model for the Jev scorer."); return; }
    setDetail(null);
    setLoading(true);
    try {
      const summary = await apiPost<SessionSummary>("/redteam", {
        system_prompt: systemPrompt.trim(), categories, model_key: modelKey,
        scorer, scorer_model: scorer === "decision" ? scorerModel : "",
      });
      const ran = await apiPost<{ status: string }>(`/redteam/${summary.session_id}/run`, {});
      const d = await apiGet<SessionDetail>(`/redteam/${summary.session_id}/results`);
      if (ran.status === "error") {
        setDetail(d);
        toast.error(d.error || "Red-team run failed.");
        return;
      }
      setDetail(d);
      const reviewSuffix = d.needs_review > 0 ? ` · ${d.needs_review} need review` : "";
      toast.success(`Ran ${d.attack_count} attacks · ${d.failed} got through${reviewSuffix}.`);
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  const passRate = detail && detail.attack_count > 0
    ? Math.round((detail.passed / detail.attack_count) * 100)
    : null;
  const passTone = passRate == null ? "" : passRate >= 80 ? "is-good" : passRate >= 50 ? "is-mid" : "is-low";

  return (
    <div className="page-shell motion-stagger-stack">
      <PageHeader
        kicker="Auto Red-Team"
        title="Adversarial Stress Test"
        subtitle="Stress-test a system prompt against a suite of adversarial attacks across five categories and review what got through."
        help={
          <>
            Paste the system prompt you ship to production, pick attack categories
            (prompt injection, jailbreak, persona override, …), then run. Each attack
            is scored <strong>passed</strong> (prompt held) or <strong>failed</strong>{" "}
            (defense broke). Expand any row to see the exact payload and response.
          </>
        }
      />

      {/* Config */}
      <Card>
        <Field label="System prompt">
          <Textarea
            value={systemPrompt}
            onChange={e => setSystemPrompt(e.target.value)}
            placeholder="You are a helpful assistant…"
            rows={5}
            className="font-mono"
          />
        </Field>

        <Field label="Target model">
          <Select value={modelKey} onChange={e => setModelKey(e.target.value)}>
            {models.length === 0 && <option value="">No models configured — dry run only</option>}
            {models.map(m => <option key={m} value={m}>{m}</option>)}
          </Select>
        </Field>

        <Field label="Scorer">
          <Select value={scorer} onChange={e => setScorer(e.target.value as "heuristic" | "decision")}>
            <option value="heuristic">Heuristic (keywords)</option>
            <option value="decision">Jev (decision)</option>
          </Select>
        </Field>
        {scorer === "decision" && (
          <DecisionModelSelect value={scorerModel} onChange={setScorerModel} label="Scorer model" />
        )}

        <span className="ds-field-mini" style={{ marginTop: "1rem" }}>Attack categories</span>
        <div className="flex flex-wrap gap-2">
          {ALL_CATEGORIES.map(cat => (
            <label key={cat} className="toggle-card" style={{ cursor: "pointer", userSelect: "none" }}>
              <input type="checkbox" className="control-check" checked={categories.includes(cat)} onChange={() => toggleCategory(cat)} />
              <span className="font-mono" style={{ fontSize: "0.78rem" }}>{cat}</span>
            </label>
          ))}
        </div>

        <div className="button-row" style={{ marginTop: "1.1rem" }}>
          <Button variant="danger" icon={<Play size={14} />} loading={loading} onClick={run}>
            {loading ? "Running attacks…" : "Run Red-Team"}
          </Button>
        </div>
      </Card>

      {/* Results */}
      {detail && detail.status === "error" && (
        <div className="alert-box alert-danger">
          Run failed before any attack could complete: {detail.error || "unknown error"}
        </div>
      )}
      {detail && detail.status !== "error" && (
        <Card style={{ padding: 0 }}>
          <div style={{ display: "flex", gap: "1.5rem", padding: "0.9rem 1.1rem", borderBottom: "1px solid var(--line)", alignItems: "center", flexWrap: "wrap" }}>
            <span className="body-copy"><strong>{detail.attack_count}</strong> attacks</span>
            <span style={{ color: "var(--success)" }}><CheckCircle size={13} style={{ verticalAlign: "middle", marginRight: 4 }} /><strong>{detail.passed}</strong> passed</span>
            <span style={{ color: "var(--danger)" }}><XCircle size={13} style={{ verticalAlign: "middle", marginRight: 4 }} /><strong>{detail.failed}</strong> failed</span>
            {detail.needs_review > 0 && (
              <span style={{ color: "var(--warning)" }}><AlertCircle size={13} style={{ verticalAlign: "middle", marginRight: 4 }} /><strong>{detail.needs_review}</strong> need review</span>
            )}
            {passRate !== null && (
              <span className={`ds-scorebar__value ${passTone}`} style={{ marginLeft: "auto", fontSize: "0.9rem" }}>{passRate}% pass rate</span>
            )}
          </div>
          <div className="table-shell" style={{ border: "none", boxShadow: "none", borderRadius: 0 }}>
            <table>
              <thead>
                <tr>{["Category", "Attack", "Verdict", "Latency", ""].map(h => <th key={h}>{h}</th>)}</tr>
              </thead>
              <tbody>
                {detail.results.map(r => <ResultRow key={r.attack_id} r={r} />)}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}
