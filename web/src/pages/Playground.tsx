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
import {
  PageHeader,
  Card,
  Button,
  Badge,
  ScoreBar,
  EmptyState,
  Field,
  Input,
  Textarea,
  useToast,
} from "@/components";
import type { BadgeTone } from "@/components";

const BASE = "/api";

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

const VERDICT: Record<string, { tone: BadgeTone; Icon: typeof TrendingUp }> = {
  improved: { tone: "success", Icon: TrendingUp },
  regressed: { tone: "danger", Icon: TrendingDown },
  stable: { tone: "neutral", Icon: Minus },
  missing: { tone: "warning", Icon: AlertCircle },
};

function VerdictBadge({ verdict }: { verdict: string }) {
  const v = VERDICT[verdict] ?? VERDICT["stable"]!;
  const { tone, Icon } = v;
  return <Badge tone={tone}><Icon size={11} /> {verdict}</Badge>;
}

function DiffRow({ diff }: { diff: CaseDiff }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <>
      <tr
        onClick={() => setExpanded(p => !p)}
        onKeyDown={e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setExpanded(p => !p); } }}
        role="button"
        tabIndex={0}
        aria-expanded={expanded}
        aria-label={`Case ${diff.case_id} diff`}
        className="ds-row-expandable"
      >
        <td className="table-code">
          <span style={{ display: "flex", alignItems: "center", gap: "0.3rem" }}>
            {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            {diff.case_id}
          </span>
        </td>
        <td style={{ minWidth: 130 }}><ScoreBar score={diff.base_score} /></td>
        <td style={{ minWidth: 160 }}><ScoreBar score={diff.compare_score} delta={diff.delta} /></td>
        <td><VerdictBadge verdict={diff.verdict} /></td>
      </tr>
      {expanded && (
        <tr className="ds-row-detail">
          <td colSpan={4}>
            <div className="grid gap-3 sm:grid-cols-2">
              <div>
                <div className="ds-field-mini">{diff.base_label}</div>
                <pre className="ds-codeblock">{diff.base_output || "(empty)"}</pre>
              </div>
              <div>
                <div className="ds-field-mini">{diff.compare_label}</div>
                <pre className="ds-codeblock">{diff.compare_output || "(empty)"}</pre>
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

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
  const [compareResult, setCompareResult] = useState<CompareResponse | null>(null);
  const [activeVariantIdx, setActiveVariantIdx] = useState(0);
  const toast = useToast();

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
    setCompareResult(null);
    try {
      const created = await apiPost<{ experiment_id: string }>("/experiments", {
        name, model_key: modelKey, variants, dataset: cases,
      });
      await apiPost(`/experiments/${created.experiment_id}/run`, {});
      const result = await apiGet<CompareResponse>(`/experiments/${created.experiment_id}/compare`);
      setCompareResult(result);
      toast.success("Experiment complete — compare the variants below.");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
    }
  };

  const activeVariant = variants[activeVariantIdx];

  return (
    <div className="page-shell motion-stagger-stack">
      <PageHeader
        kicker="Prompt Playground"
        title="A/B Prompt Lab"
        subtitle="Run two or more system-prompt variants over the same dataset and diff the results case by case."
      />

      <div className="grid gap-4 lg:grid-cols-[420px_1fr] items-start">
        {/* Left: config */}
        <div className="flex flex-col gap-4">
          <Card>
            <div className="flex flex-col gap-3">
              <Field label="Experiment name">
                <Input value={name} onChange={e => setName(e.target.value)} />
              </Field>
              <Field label="Model key (optional)">
                <Input value={modelKey} onChange={e => setModelKey(e.target.value)} placeholder="e.g. gpt-4o" />
              </Field>
            </div>
          </Card>

          {/* Variants */}
          <Card>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.7rem" }}>
              <span className="section-caption">Prompt variants</span>
              <Button variant="secondary" icon={<Plus size={12} />} onClick={addVariant}>Add</Button>
            </div>
            <div className="tab-strip" style={{ flexWrap: "wrap", marginBottom: "0.7rem" }}>
              {variants.map((v, i) => (
                <button
                  key={i}
                  className={`tab-button ${activeVariantIdx === i ? "tab-button-active" : ""}`}
                  onClick={() => setActiveVariantIdx(i)}
                  style={{ display: "inline-flex", alignItems: "center", gap: "0.3rem" }}
                >
                  {v.label}
                  {variants.length > 2 && (
                    <Trash2 size={11} onClick={e => { e.stopPropagation(); removeVariant(i); }} />
                  )}
                </button>
              ))}
            </div>
            {activeVariant && (
              <div className="flex flex-col gap-2">
                <Input value={activeVariant.label} onChange={e => updateVariant(activeVariantIdx, "label", e.target.value)} placeholder="Label" />
                <Textarea
                  value={activeVariant.system_prompt}
                  onChange={e => updateVariant(activeVariantIdx, "system_prompt", e.target.value)}
                  placeholder="System prompt…"
                  rows={6}
                  className="font-mono"
                />
              </div>
            )}
          </Card>

          {/* Dataset */}
          <Card>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.7rem" }}>
              <span className="section-caption">Dataset ({cases.length} cases)</span>
              <Button variant="secondary" icon={<Plus size={12} />} onClick={addCase}>Add case</Button>
            </div>
            <div className="flex flex-col gap-2">
              {cases.map((c, i) => (
                <div key={i} className="panel-quiet" style={{ borderRadius: 16, border: "1px solid var(--line)", padding: "0.6rem" }}>
                  <div style={{ display: "flex", gap: "0.4rem", marginBottom: "0.4rem" }}>
                    <Input value={c.case_id} onChange={e => updateCase(i, "case_id", e.target.value)} placeholder="ID" style={{ width: 90 }} />
                    <button className="ds-icon-button" aria-label={`Remove case ${i + 1}`} onClick={() => removeCase(i)}>
                      <Trash2 size={13} />
                    </button>
                  </div>
                  <Input value={c.input} onChange={e => updateCase(i, "input", e.target.value)} placeholder="User input" style={{ marginBottom: "0.4rem" }} />
                  <Input value={c.expected} onChange={e => updateCase(i, "expected", e.target.value)} placeholder="Expected output (for scoring)" />
                </div>
              ))}
            </div>
          </Card>

          <Button
            icon={<Play size={15} />}
            loading={running}
            disabled={cases.length === 0 || variants.length < 2}
            onClick={runExperiment}
          >
            {running ? "Running…" : "Run Experiment"}
          </Button>
        </div>

        {/* Right: results */}
        <div>
          {!compareResult ? (
            <Card style={{ minHeight: 320, display: "flex", alignItems: "center", justifyContent: "center" }}>
              <EmptyState
                icon={FlaskConical}
                title="No comparison yet"
                hint="Configure variants and dataset, then run the experiment."
              />
            </Card>
          ) : (
            <div className="flex flex-col gap-4">
              <div className="flex flex-wrap gap-3">
                {[
                  { label: "Improved", count: compareResult.improved, color: "var(--success)" },
                  { label: "Regressed", count: compareResult.regressed, color: "var(--danger)" },
                  { label: "Stable", count: compareResult.stable, color: "var(--text-dim)" },
                  { label: "Missing", count: compareResult.missing, color: "var(--warning)" },
                ].map(({ label, count, color }) => (
                  <Card key={label} className="stat-card" style={{ minWidth: 96, textAlign: "center" }}>
                    <div className="stat-value" style={{ fontSize: "1.6rem", color }}>{count}</div>
                    <p className="stat-label">{label}</p>
                  </Card>
                ))}
                <Card className="stat-card" style={{ minWidth: 96, textAlign: "center" }}>
                  <div className="stat-value" style={{ fontSize: "1.6rem", color: compareResult.avg_delta >= 0 ? "var(--success)" : "var(--danger)" }}>
                    {compareResult.avg_delta >= 0 ? "+" : ""}{(compareResult.avg_delta * 100).toFixed(1)}%
                  </div>
                  <p className="stat-label">Avg Δ</p>
                </Card>
              </div>

              <div className="table-shell">
                <table>
                  <thead>
                    <tr>
                      <th>Case</th>
                      <th>{compareResult.base_label}</th>
                      <th>{compareResult.compare_label}</th>
                      <th>Verdict</th>
                    </tr>
                  </thead>
                  <tbody>
                    {compareResult.diffs.map(diff => <DiffRow key={diff.case_id} diff={diff} />)}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
