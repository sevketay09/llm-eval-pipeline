import { useState } from "react";
import { ShieldAlert, Play, CheckCircle, XCircle, AlertCircle, ChevronDown, ChevronRight } from "lucide-react";

const BASE = "/api";

const ALL_CATEGORIES = [
  "prompt_injection",
  "jailbreak",
  "persona_override",
  "boundary_test",
  "role_confusion",
] as const;

// ── Types ─────────────────────────────────────────────────────────────────────

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

// ── API ───────────────────────────────────────────────────────────────────────

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

// ── Verdict badge ─────────────────────────────────────────────────────────────

function VerdictBadge({ passed, error }: { passed: boolean; error: string }) {
  if (error) {
    return (
      <span style={{ background: "#431a0022", color: "#fb923c", fontSize: 11, fontWeight: 700, padding: "2px 7px", borderRadius: 4, display: "inline-flex", alignItems: "center", gap: 3 }}>
        <AlertCircle size={11} /> error
      </span>
    );
  }
  if (passed) {
    return (
      <span style={{ background: "#14532d22", color: "#4ade80", fontSize: 11, fontWeight: 700, padding: "2px 7px", borderRadius: 4, display: "inline-flex", alignItems: "center", gap: 3 }}>
        <CheckCircle size={11} /> passed
      </span>
    );
  }
  return (
    <span style={{ background: "#7f1d1d22", color: "#f87171", fontSize: 11, fontWeight: 700, padding: "2px 7px", borderRadius: 4, display: "inline-flex", alignItems: "center", gap: 3 }}>
      <XCircle size={11} /> failed
    </span>
  );
}

// ── Result row ────────────────────────────────────────────────────────────────

function ResultRow({ r }: { r: AttackResult }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <tr
        onClick={() => setOpen((v) => !v)}
        style={{ cursor: "pointer", borderBottom: "1px solid #1e2736" }}
      >
        <td style={{ padding: "8px 10px", color: "#9ca3af", fontSize: 11 }}>
          <span style={{ background: "#1e2736", borderRadius: 3, padding: "2px 6px", fontFamily: "monospace" }}>{r.category}</span>
        </td>
        <td style={{ padding: "8px 10px", color: "#e2e8f0", fontSize: 13 }}>{r.name}</td>
        <td style={{ padding: "8px 10px" }}><VerdictBadge passed={r.passed} error={r.error} /></td>
        <td style={{ padding: "8px 10px", color: "#9ca3af", fontSize: 11 }}>{r.latency_ms.toFixed(1)} ms</td>
        <td style={{ padding: "8px 10px", color: "#9ca3af", fontSize: 11 }}>
          {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </td>
      </tr>
      {open && (
        <tr style={{ background: "#0d1117", borderBottom: "1px solid #1e2736" }}>
          <td colSpan={5} style={{ padding: "12px 16px" }}>
            <div style={{ marginBottom: 8 }}>
              <span style={{ color: "#6b7280", fontSize: 11, fontWeight: 600 }}>PAYLOAD</span>
              <pre style={{ margin: "4px 0 0", color: "#94a3b8", fontSize: 12, whiteSpace: "pre-wrap", fontFamily: "monospace" }}>{r.payload}</pre>
            </div>
            <div style={{ marginBottom: 8 }}>
              <span style={{ color: "#6b7280", fontSize: 11, fontWeight: 600 }}>RESPONSE</span>
              <pre style={{ margin: "4px 0 0", color: "#e2e8f0", fontSize: 12, whiteSpace: "pre-wrap", fontFamily: "monospace" }}>{r.response || r.error || "—"}</pre>
            </div>
            <div>
              <span style={{ color: "#6b7280", fontSize: 11, fontWeight: 600 }}>REASON</span>
              <p style={{ margin: "4px 0 0", color: "#9ca3af", fontSize: 12 }}>{r.reason || "—"}</p>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function RedTeam() {
  const [systemPrompt, setSystemPrompt] = useState("");
  const [categories, setCategories] = useState<string[]>([...ALL_CATEGORIES]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [detail, setDetail] = useState<SessionDetail | null>(null);

  function toggleCategory(cat: string) {
    setCategories((prev) =>
      prev.includes(cat) ? prev.filter((c) => c !== cat) : [...prev, cat]
    );
  }

  async function run() {
    if (!systemPrompt.trim()) { setError("System prompt is required."); return; }
    if (categories.length === 0) { setError("Select at least one category."); return; }
    setError(null);
    setDetail(null);
    setLoading(true);
    try {
      const summary = await apiPost<SessionSummary>("/redteam", {
        system_prompt: systemPrompt.trim(),
        categories,
      });
      await apiPost(`/redteam/${summary.session_id}/run`, {});
      const d = await apiGet<SessionDetail>(`/redteam/${summary.session_id}/results`);
      setDetail(d);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  const passRate = detail && detail.attack_count > 0
    ? Math.round((detail.passed / detail.attack_count) * 100)
    : null;

  return (
    <div style={{ padding: "24px 28px", maxWidth: 960, margin: "0 auto" }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 24 }}>
        <ShieldAlert size={22} color="#f87171" />
        <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: "#e2e8f0" }}>Auto Red-Team</h1>
        <span style={{ marginLeft: 4, fontSize: 12, color: "#6b7280" }}>Stress-test a system prompt against adversarial attacks</span>
      </div>

      {/* Config panel */}
      <div style={{ background: "#111827", border: "1px solid #1e2736", borderRadius: 8, padding: 20, marginBottom: 20 }}>
        <label style={{ display: "block", marginBottom: 6, fontSize: 12, fontWeight: 600, color: "#9ca3af" }}>SYSTEM PROMPT</label>
        <textarea
          value={systemPrompt}
          onChange={(e) => setSystemPrompt(e.target.value)}
          placeholder="You are a helpful assistant..."
          rows={5}
          style={{
            width: "100%", boxSizing: "border-box",
            background: "#0d1117", border: "1px solid #1e2736", borderRadius: 6,
            color: "#e2e8f0", fontSize: 13, padding: "10px 12px",
            fontFamily: "monospace", resize: "vertical",
          }}
        />

        <label style={{ display: "block", margin: "16px 0 8px", fontSize: 12, fontWeight: 600, color: "#9ca3af" }}>ATTACK CATEGORIES</label>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
          {ALL_CATEGORIES.map((cat) => (
            <label key={cat} style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer", userSelect: "none" }}>
              <input
                type="checkbox"
                checked={categories.includes(cat)}
                onChange={() => toggleCategory(cat)}
                style={{ accentColor: "#f87171" }}
              />
              <span style={{ fontSize: 12, color: categories.includes(cat) ? "#e2e8f0" : "#6b7280", fontFamily: "monospace" }}>{cat}</span>
            </label>
          ))}
        </div>

        <button
          onClick={run}
          disabled={loading}
          style={{
            marginTop: 20, display: "flex", alignItems: "center", gap: 6,
            background: loading ? "#1e2736" : "#dc2626", color: "#fff",
            border: "none", borderRadius: 6, padding: "9px 18px", fontSize: 13,
            fontWeight: 600, cursor: loading ? "not-allowed" : "pointer",
          }}
        >
          <Play size={14} />
          {loading ? "Running attacks…" : "Run Red-Team"}
        </button>

        {error && (
          <p style={{ marginTop: 12, color: "#f87171", fontSize: 13 }}>{error}</p>
        )}
      </div>

      {/* Results */}
      {detail && (
        <div style={{ background: "#111827", border: "1px solid #1e2736", borderRadius: 8 }}>
          {/* Summary bar */}
          <div style={{ display: "flex", gap: 24, padding: "14px 20px", borderBottom: "1px solid #1e2736", alignItems: "center" }}>
            <span style={{ fontSize: 13, color: "#9ca3af" }}>
              <strong style={{ color: "#e2e8f0" }}>{detail.attack_count}</strong> attacks
            </span>
            <span style={{ fontSize: 13, color: "#4ade80" }}>
              <CheckCircle size={13} style={{ verticalAlign: "middle", marginRight: 4 }} />
              <strong>{detail.passed}</strong> passed
            </span>
            <span style={{ fontSize: 13, color: "#f87171" }}>
              <XCircle size={13} style={{ verticalAlign: "middle", marginRight: 4 }} />
              <strong>{detail.failed}</strong> failed
            </span>
            {passRate !== null && (
              <span style={{
                marginLeft: "auto", fontSize: 13, fontWeight: 700,
                color: passRate >= 80 ? "#4ade80" : passRate >= 50 ? "#fb923c" : "#f87171",
              }}>
                {passRate}% pass rate
              </span>
            )}
          </div>

          {/* Table */}
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid #1e2736" }}>
                {["Category", "Attack", "Verdict", "Latency", ""].map((h) => (
                  <th key={h} style={{ padding: "8px 10px", textAlign: "left", fontSize: 11, fontWeight: 600, color: "#6b7280" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {detail.results.map((r) => (
                <ResultRow key={r.attack_id} r={r} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
