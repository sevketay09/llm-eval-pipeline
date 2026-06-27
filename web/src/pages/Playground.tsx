import { useState } from "react";
import {
  FlaskConical,
  Play,
  Plus,
  Trash2,
  TrendingUp,
  TrendingDown,
  Minus,
  AlertCircle,
  ChevronDown,
  ChevronRight,
} from "lucide-react";

const BASE = "/api";

// ── Types ─────────────────────────────────────────────────────────────────────

interface Variant { label: string; system_prompt: string }
interface DatasetCase { case_id: string; input: string; expected: string }

interface CaseDiff {
  case_id: string;
  base_label: string;
  compare_label: string;
  base_score: number;
  compare_score: number;
  base_output: string;
  compare_output: string;
  delta: number;
  verdict: "improved" | "regressed" | "stable" | "missing";
}

interface CompareResponse {
  experiment_id: string;
  base_label: string;
  compare_label: string;
  diffs: CaseDiff[];
  improved: number;
  regressed: number;
  stable: number;
  missing: number;
  avg_delta: number;
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

const VERDICT_STYLE: Record<string, { color: string; bg: string; Icon: React.ElementType }> = {
  improved: { color: "#4ade80", bg: "#14532d22", Icon: TrendingUp },
  regressed: { color: "#f87171", bg: "#7f1d1d22", Icon: TrendingDown },
  stable:    { color: "#9ca3af", bg: "#37415122", Icon: Minus },
  missing:   { color: "#fb923c", bg: "#431a0022", Icon: AlertCircle },
};

function VerdictBadge({ verdict }: { verdict: string }) {
  const st = (VERDICT_STYLE[verdict] ?? VERDICT_STYLE["stable"]) as { color: string; bg: string; Icon: React.ElementType };
  const { color, bg, Icon } = st;
  return (
    <span style={{ background: bg, color, fontSize: 11, fontWeight: 700, padding: "2px 7px", borderRadius: 4, display: "inline-flex", alignItems: "center", gap: 3 }}>
      <Icon size={11} /> {verdict}
    </span>
  );
}

// ── Score bar ─────────────────────────────────────────────────────────────────

function ScoreBar({ score, delta }: { score: number; delta?: number }) {
  const color = score >= 0.8 ? "#4ade80" : score >= 0.5 ? "#facc15" : "#f87171";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <div style={{ width: 60, height: 6, background: "#1f2937", borderRadius: 3, overflow: "hidden" }}>
        <div style={{ width: `${score * 100}%`, height: "100%", background: color, borderRadius: 3 }} />
      </div>
      <span style={{ fontSize: 12, color, fontWeight: 600 }}>{(score * 100).toFixed(0)}%</span>
      {delta != null && (
        <span style={{ fontSize: 11, color: delta > 0 ? "#4ade80" : delta < 0 ? "#f87171" : "#6b7280" }}>
          {delta > 0 ? "+" : ""}{(delta * 100).toFixed(0)}%
        </span>
      )}
    </div>
  );
}

// ── Diff row ──────────────────────────────────────────────────────────────────

function DiffRow({ diff }: { diff: CaseDiff }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <>
      <tr
        onClick={() => setExpanded(p => !p)}
        style={{ cursor: "pointer", borderBottom: "1px solid #1f2937" }}
      >
        <td style={{ padding: "8px 12px", fontFamily: "monospace", fontSize: 12, color: "#9ca3af" }}>
          <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
            {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            {diff.case_id}
          </span>
        </td>
        <td style={{ padding: "8px 12px" }}>
          <ScoreBar score={diff.base_score} />
        </td>
        <td style={{ padding: "8px 12px" }}>
          <ScoreBar score={diff.compare_score} delta={diff.delta} />
        </td>
        <td style={{ padding: "8px 12px" }}>
          <VerdictBadge verdict={diff.verdict} />
        </td>
      </tr>
      {expanded && (
        <tr style={{ background: "#0a0f1a" }}>
          <td colSpan={4} style={{ padding: "10px 16px" }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div>
                <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 4 }}>{diff.base_label}</div>
                <div style={{ fontSize: 12, color: "#d1d5db", background: "#111827", padding: 8, borderRadius: 6, whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
                  {diff.base_output || "(empty)"}
                </div>
              </div>
              <div>
                <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 4 }}>{diff.compare_label}</div>
                <div style={{ fontSize: 12, color: "#d1d5db", background: "#111827", padding: 8, borderRadius: 6, whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
                  {diff.compare_output || "(empty)"}
                </div>
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

const DEFAULT_VARIANTS: Variant[] = [
  { label: "v1", system_prompt: "You are a helpful assistant." },
  { label: "v2", system_prompt: "You are a concise and precise assistant. Answer briefly." },
];

const DEFAULT_CASES: DatasetCase[] = [
  { case_id: "c1", input: "What is the capital of France?", expected: "Paris" },
  { case_id: "c2", input: "What is 2 + 2?", expected: "4" },
  { case_id: "c3", input: "Explain photosynthesis.", expected: "process plants use sunlight" },
];

export default function Playground() {
  const [name, setName] = useState("My Experiment");
  const [modelKey, setModelKey] = useState("");
  const [variants, setVariants] = useState<Variant[]>(DEFAULT_VARIANTS);
  const [cases, setCases] = useState<DatasetCase[]>(DEFAULT_CASES);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [compareResult, setCompareResult] = useState<CompareResponse | null>(null);
  const [activeVariantIdx, setActiveVariantIdx] = useState(0);

  const updateVariant = (idx: number, field: keyof Variant, value: string) => {
    setVariants(vs => vs.map((v, i) => i === idx ? { ...v, [field]: value } : v));
  };

  const addVariant = () => {
    const n = variants.length + 1;
    setVariants(vs => [...vs, { label: `v${n}`, system_prompt: "" }]);
    setActiveVariantIdx(variants.length);
  };

  const removeVariant = (idx: number) => {
    if (variants.length <= 2) return;
    setVariants(vs => vs.filter((_, i) => i !== idx));
    setActiveVariantIdx(Math.max(0, activeVariantIdx - 1));
  };

  const updateCase = (idx: number, field: keyof DatasetCase, value: string) => {
    setCases(cs => cs.map((c, i) => i === idx ? { ...c, [field]: value } : c));
  };

  const addCase = () => {
    const n = cases.length + 1;
    setCases(cs => [...cs, { case_id: `c${n}`, input: "", expected: "" }]);
  };

  const removeCase = (idx: number) => {
    setCases(cs => cs.filter((_, i) => i !== idx));
  };

  const runExperiment = async () => {
    setRunning(true);
    setError(null);
    setCompareResult(null);
    try {
      const created = await apiPost<{ experiment_id: string }>("/experiments", {
        name,
        model_key: modelKey,
        variants,
        dataset: cases,
      });
      await apiPost(`/experiments/${created.experiment_id}/run`, {});
      const result = await apiGet<CompareResponse>(`/experiments/${created.experiment_id}/compare`);
      setCompareResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div style={{ display: "flex", height: "100%", color: "#e5e7eb", overflow: "hidden" }}>
      {/* Left: config panel */}
      <div style={{ width: 420, flexShrink: 0, borderRight: "1px solid #1f2937", overflowY: "auto", padding: 20, display: "flex", flexDirection: "column", gap: 20 }}>

        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <FlaskConical size={18} style={{ color: "#7c3aed" }} />
          <span style={{ fontWeight: 700, fontSize: 15 }}>Prompt Playground</span>
        </div>

        {/* Name + model */}
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <label style={{ fontSize: 12, color: "#6b7280" }}>Experiment name</label>
          <input
            value={name}
            onChange={e => setName(e.target.value)}
            style={inputStyle}
          />
          <label style={{ fontSize: 12, color: "#6b7280" }}>Model key (optional)</label>
          <input
            value={modelKey}
            onChange={e => setModelKey(e.target.value)}
            placeholder="e.g. gpt-4o"
            style={inputStyle}
          />
        </div>

        {/* Variant editor */}
        <div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
            <span style={{ fontSize: 13, fontWeight: 600 }}>Prompt Variants</span>
            <button onClick={addVariant} style={smallBtnStyle}>
              <Plus size={12} /> Add
            </button>
          </div>
          <div style={{ display: "flex", gap: 6, marginBottom: 8, flexWrap: "wrap" }}>
            {variants.map((v, i) => (
              <button
                key={i}
                onClick={() => setActiveVariantIdx(i)}
                style={{
                  ...tabStyle,
                  background: activeVariantIdx === i ? "#7c3aed" : "#1f2937",
                  color: activeVariantIdx === i ? "#fff" : "#9ca3af",
                }}
              >
                {v.label}
                {variants.length > 2 && (
                  <Trash2
                    size={10}
                    style={{ marginLeft: 4 }}
                    onClick={e => { e.stopPropagation(); removeVariant(i); }}
                  />
                )}
              </button>
            ))}
          </div>
          {variants[activeVariantIdx] && (
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <input
                value={variants[activeVariantIdx].label}
                onChange={e => updateVariant(activeVariantIdx, "label", e.target.value)}
                placeholder="Label"
                style={inputStyle}
              />
              <textarea
                value={variants[activeVariantIdx].system_prompt}
                onChange={e => updateVariant(activeVariantIdx, "system_prompt", e.target.value)}
                placeholder="System prompt…"
                rows={6}
                style={{ ...inputStyle, resize: "vertical", fontFamily: "monospace", fontSize: 12 }}
              />
            </div>
          )}
        </div>

        {/* Dataset editor */}
        <div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
            <span style={{ fontSize: 13, fontWeight: 600 }}>Dataset ({cases.length} cases)</span>
            <button onClick={addCase} style={smallBtnStyle}>
              <Plus size={12} /> Add case
            </button>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {cases.map((c, i) => (
              <div key={i} style={{ background: "#111827", borderRadius: 6, padding: 8 }}>
                <div style={{ display: "flex", gap: 6, marginBottom: 6 }}>
                  <input
                    value={c.case_id}
                    onChange={e => updateCase(i, "case_id", e.target.value)}
                    placeholder="ID"
                    style={{ ...inputStyle, width: 80 }}
                  />
                  <button onClick={() => removeCase(i)} style={{ background: "none", border: "none", cursor: "pointer", color: "#4b5563" }}>
                    <Trash2 size={13} />
                  </button>
                </div>
                <input
                  value={c.input}
                  onChange={e => updateCase(i, "input", e.target.value)}
                  placeholder="User input"
                  style={{ ...inputStyle, marginBottom: 4 }}
                />
                <input
                  value={c.expected}
                  onChange={e => updateCase(i, "expected", e.target.value)}
                  placeholder="Expected output (for scoring)"
                  style={inputStyle}
                />
              </div>
            ))}
          </div>
        </div>

        {/* Run button */}
        <button
          onClick={runExperiment}
          disabled={running || cases.length === 0 || variants.length < 2}
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 6,
            padding: "10px 16px",
            background: running ? "#374151" : "#7c3aed",
            color: "#fff",
            border: "none",
            borderRadius: 8,
            cursor: running ? "default" : "pointer",
            fontWeight: 700,
            fontSize: 14,
          }}
        >
          <Play size={15} />
          {running ? "Running…" : "Run Experiment"}
        </button>

        {error && (
          <div style={{ background: "#7f1d1d22", border: "1px solid #ef4444", borderRadius: 6, padding: 10, fontSize: 12, color: "#f87171" }}>
            {error}
          </div>
        )}
      </div>

      {/* Right: diff results */}
      <div style={{ flex: 1, overflowY: "auto", padding: 20 }}>
        {!compareResult ? (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", color: "#374151", gap: 12 }}>
            <FlaskConical size={48} style={{ opacity: 0.3 }} />
            <span style={{ fontSize: 14 }}>Configure variants and dataset, then run the experiment.</span>
          </div>
        ) : (
          <>
            {/* Summary bar */}
            <div style={{ display: "flex", gap: 16, marginBottom: 20, flexWrap: "wrap" }}>
              {[
                { label: "Improved", count: compareResult.improved, color: "#4ade80" },
                { label: "Regressed", count: compareResult.regressed, color: "#f87171" },
                { label: "Stable", count: compareResult.stable, color: "#9ca3af" },
                { label: "Missing", count: compareResult.missing, color: "#fb923c" },
              ].map(({ label, count, color }) => (
                <div key={label} style={{ background: "#111827", borderRadius: 8, padding: "12px 16px", minWidth: 90 }}>
                  <div style={{ fontSize: 22, fontWeight: 700, color }}>{count}</div>
                  <div style={{ fontSize: 11, color: "#6b7280" }}>{label}</div>
                </div>
              ))}
              <div style={{ background: "#111827", borderRadius: 8, padding: "12px 16px", minWidth: 90 }}>
                <div style={{ fontSize: 22, fontWeight: 700, color: compareResult.avg_delta >= 0 ? "#4ade80" : "#f87171" }}>
                  {compareResult.avg_delta >= 0 ? "+" : ""}{(compareResult.avg_delta * 100).toFixed(1)}%
                </div>
                <div style={{ fontSize: 11, color: "#6b7280" }}>Avg Δ</div>
              </div>
            </div>

            {/* Diff table */}
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid #1f2937" }}>
                  <th style={{ padding: "8px 12px", textAlign: "left", fontSize: 12, color: "#6b7280", fontWeight: 600 }}>Case</th>
                  <th style={{ padding: "8px 12px", textAlign: "left", fontSize: 12, color: "#6b7280", fontWeight: 600 }}>{compareResult.base_label}</th>
                  <th style={{ padding: "8px 12px", textAlign: "left", fontSize: 12, color: "#6b7280", fontWeight: 600 }}>{compareResult.compare_label}</th>
                  <th style={{ padding: "8px 12px", textAlign: "left", fontSize: 12, color: "#6b7280", fontWeight: 600 }}>Verdict</th>
                </tr>
              </thead>
              <tbody>
                {compareResult.diffs.map(diff => (
                  <DiffRow key={diff.case_id} diff={diff} />
                ))}
              </tbody>
            </table>
          </>
        )}
      </div>
    </div>
  );
}

// ── Shared styles ─────────────────────────────────────────────────────────────

const inputStyle: React.CSSProperties = {
  background: "#1f2937",
  border: "1px solid #374151",
  borderRadius: 6,
  padding: "6px 10px",
  fontSize: 13,
  color: "#e5e7eb",
  width: "100%",
  outline: "none",
  boxSizing: "border-box",
};

const smallBtnStyle: React.CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  gap: 4,
  padding: "4px 8px",
  background: "#1f2937",
  border: "1px solid #374151",
  borderRadius: 6,
  cursor: "pointer",
  color: "#9ca3af",
  fontSize: 12,
};

const tabStyle: React.CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  padding: "4px 10px",
  borderRadius: 6,
  border: "none",
  cursor: "pointer",
  fontSize: 12,
  fontWeight: 600,
};
