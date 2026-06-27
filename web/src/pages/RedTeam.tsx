import { useState } from "react";
import { Play, CheckCircle, XCircle, AlertCircle, ChevronDown, ChevronRight } from "lucide-react";
import { PageHeader, Card, Button, Badge, Field, Textarea, useToast } from "@/components";

const BASE = "/api";

const ALL_CATEGORIES = [
  "prompt_injection",
  "jailbreak",
  "persona_override",
  "boundary_test",
  "role_confusion",
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
}

interface SessionSummary {
  session_id: string;
  attack_count: number;
  status: string;
  passed: number;
  failed: number;
}

interface SessionDetail extends SessionSummary {
  results: AttackResult[];
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

async function apiGet<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

function VerdictBadge({ passed, error }: { passed: boolean; error: string }) {
  if (error) return <Badge tone="warning"><AlertCircle size={11} /> error</Badge>;
  if (passed) return <Badge tone="success"><CheckCircle size={11} /> passed</Badge>;
  return <Badge tone="danger"><XCircle size={11} /> failed</Badge>;
}

function ResultRow({ r }: { r: AttackResult }) {
  const [open, setOpen] = useState(false);
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
        <td><VerdictBadge passed={r.passed} error={r.error} /></td>
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
            <div>
              <span className="ds-field-mini">Reason</span>
              <p className="muted-copy" style={{ margin: 0, fontSize: "0.85rem" }}>{r.reason || "—"}</p>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

export default function RedTeam() {
  const [systemPrompt, setSystemPrompt] = useState("");
  const [categories, setCategories] = useState<string[]>([...ALL_CATEGORIES]);
  const [loading, setLoading] = useState(false);
  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const toast = useToast();

  function toggleCategory(cat: string) {
    setCategories(prev => prev.includes(cat) ? prev.filter(c => c !== cat) : [...prev, cat]);
  }

  async function run() {
    if (!systemPrompt.trim()) { toast.error("System prompt is required."); return; }
    if (categories.length === 0) { toast.error("Select at least one category."); return; }
    setDetail(null);
    setLoading(true);
    try {
      const summary = await apiPost<SessionSummary>("/redteam", { system_prompt: systemPrompt.trim(), categories });
      await apiPost(`/redteam/${summary.session_id}/run`, {});
      const d = await apiGet<SessionDetail>(`/redteam/${summary.session_id}/results`);
      setDetail(d);
      toast.success(`Ran ${d.attack_count} attacks · ${d.failed} got through.`);
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
      {detail && (
        <Card style={{ padding: 0 }}>
          <div style={{ display: "flex", gap: "1.5rem", padding: "0.9rem 1.1rem", borderBottom: "1px solid var(--line)", alignItems: "center", flexWrap: "wrap" }}>
            <span className="body-copy"><strong>{detail.attack_count}</strong> attacks</span>
            <span style={{ color: "var(--success)" }}><CheckCircle size={13} style={{ verticalAlign: "middle", marginRight: 4 }} /><strong>{detail.passed}</strong> passed</span>
            <span style={{ color: "var(--danger)" }}><XCircle size={13} style={{ verticalAlign: "middle", marginRight: 4 }} /><strong>{detail.failed}</strong> failed</span>
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
