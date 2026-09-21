import { useEffect, useRef, useState } from "react";
import { Play, Plus, ShieldAlert, Trash2, Upload } from "lucide-react";
import {
  PageHeader,
  Card,
  Button,
  Badge,
  Field,
  Input,
  Select,
  Textarea,
  ScoreBar,
  useToast,
  DecisionModelSelect,
} from "@/components";
import { scoreTone } from "@/components";
import { modelsApi } from "@/api/client";
import { splitModels } from "@/lib/decision";

const BASE = "/api";

interface LintCheck {
  id: string;
  area: "format" | "body" | "structure" | "security";
  severity: "error" | "warning" | "info";
  passed: boolean;
  message: string;
}

interface LintReport {
  score: number;
  checks: LintCheck[];
  summary: {
    total_checks: number;
    failed: number;
    errors: number;
    warnings: number;
    security_flags: number;
    name: string | null;
    body_tokens_estimate: number;
  };
}

interface FitCriterion {
  score: number;
  evidence: string | null;
  reasoning: string;
}

interface FitReport {
  overall: number;
  verdict: "fit" | "partial_fit" | "unfit";
  criteria: Record<string, FitCriterion>;
  missing_criteria: string[];
  gaps: string[];
  suggestions: string[];
}

interface FullReport {
  kind: string;
  timestamp: string;
  judge_model: string;
  combined_score: number;
  combined_basis: "lint+fit" | "lint_only";
  lint: LintReport;
  fit: FitReport | null;
  report_path?: string;
}

interface TriggerPromptResult {
  text: string;
  expected: boolean | "ambiguous";
  predicted: boolean | null;
  trigger_rate: number | null;
  trials: number;
  correct: boolean | null;
  probability?: number | null;
}

interface ThresholdCurvePoint {
  threshold: number;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  false_positive_rate: number | null;
}

interface TriggerSummary {
  precision: number | null;
  recall: number | null;
  f1: number | null;
  accuracy: number | null;
  false_positive_rate: number | null;
  scored: number;
  skipped: number;
  ambiguous_count: number;
  ambiguous_trigger_rate: number | null;
  verdict: "reliable" | "over_triggering" | "under_triggering" | "unreliable" | "insufficient_data";
  mode?: "decision";
  threshold_curve?: ThresholdCurvePoint[];
}

interface TriggerReport {
  skill: { name: string; description: string };
  summary: TriggerSummary;
  results: TriggerPromptResult[];
}

interface RoutingSkillEntry {
  name: string;
  skill_text: string;
}

interface RoutingResult {
  text: string;
  expected: string;
  predicted: string;
  confidence: number;
  ranked: [string, number][];
}

interface RoutingMetrics {
  total: number;
  accuracy?: number | null;
  none_rate?: number | null;
  per_skill?: Record<string, { precision: number | null; recall: number | null }>;
  confusion?: Record<string, Record<string, number>>;
}

interface RoutingReport {
  skills: string[];
  metrics: RoutingMetrics;
  results: RoutingResult[];
  none_label: string;
}

interface ReportSummary {
  filename: string;
  timestamp: string | null;
  judge_model: string | null;
  combined_score: number | null;
  combined_basis: string | null;
  lint_score: number | null;
  fit_overall: number | null;
  verdict: string | null;
  skill_name: string | null;
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

const CRITERION_LABELS: Record<string, string> = {
  scope_coverage: "Scope coverage",
  instruction_clarity: "Instruction clarity",
  completeness: "Completeness",
  convention_alignment: "Convention alignment",
  efficiency_risk: "Efficiency risk",
};

const VERDICT_TONE: Record<string, "success" | "warning" | "danger"> = {
  fit: "success",
  partial_fit: "warning",
  unfit: "danger",
};

const SEVERITY_TONE: Record<string, "danger" | "warning" | "neutral"> = {
  error: "danger",
  warning: "warning",
  info: "neutral",
};

const SAMPLE_SKILL = `---
name: csv-report
description: Generates weekly CSV sales reports with totals per region.
---
# Usage

Load the CSV, group by region, write totals to report.csv.

See [details](./reference.md) for column mapping.
`;

const TRIGGER_VERDICT_TONE: Record<string, "success" | "warning" | "danger" | "neutral"> = {
  reliable: "success",
  over_triggering: "warning",
  under_triggering: "warning",
  unreliable: "danger",
  insufficient_data: "neutral",
};

const SAMPLE_TRIGGER_PROMPTS = [
  "TRUE: Generate the weekly sales CSV report",
  "TRUE: Build the regional sales report from data.csv",
  "FALSE: What is the weather in Ankara?",
  "FALSE: Translate this sentence to German",
  "AMBIGUOUS: Summarize this spreadsheet somehow",
].join("\n");

interface ParsedTriggerPrompt {
  text: string;
  expected: boolean | "ambiguous";
}

function parseTriggerPrompts(raw: string): ParsedTriggerPrompt[] {
  const prompts: ParsedTriggerPrompt[] = [];
  for (const line of raw.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const match = trimmed.match(/^(TRUE|FALSE|AMBIGUOUS)\s*:\s*(.+)$/i);
    if (!match || !match[1] || !match[2]) continue;
    const upper = match[1].toUpperCase();
    const expected: boolean | "ambiguous" = upper === "TRUE" ? true : upper === "FALSE" ? false : "ambiguous";
    prompts.push({ text: match[2].trim(), expected });
  }
  return prompts;
}

interface ParsedRoutingPrompt {
  text: string;
  expected: string;
}

function parseRoutingPrompts(raw: string): ParsedRoutingPrompt[] {
  const prompts: ParsedRoutingPrompt[] = [];
  for (const line of raw.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const idx = trimmed.indexOf(":");
    if (idx <= 0) continue;
    const expected = trimmed.slice(0, idx).trim();
    const text = trimmed.slice(idx + 1).trim();
    if (!expected || !text) continue;
    prompts.push({ text, expected });
  }
  return prompts;
}

function LintPanel({ lint }: { lint: LintReport }) {
  const failed = lint.checks.filter((c) => !c.passed);
  const passed = lint.checks.filter((c) => c.passed);
  return (
    <Card>
      <div style={{ display: "flex", alignItems: "center", gap: "0.8rem", marginBottom: "0.8rem" }}>
        <p className="section-caption" style={{ margin: 0, flex: 1 }}>Static lint</p>
        <span className={`ds-scorebar__value is-${scoreTone(lint.score / 100)}`}>{lint.score}/100</span>
        {lint.summary.security_flags > 0 && (
          <Badge tone="danger">
            <ShieldAlert size={12} style={{ marginRight: 4 }} />
            {lint.summary.security_flags} security
          </Badge>
        )}
      </div>
      {failed.length === 0 ? (
        <div className="alert-box alert-success">All {lint.summary.total_checks} checks pass.</div>
      ) : (
        failed.map((c) => (
          <div key={c.id} style={{ display: "flex", alignItems: "flex-start", gap: "0.6rem", marginBottom: "0.45rem" }}>
            <Badge tone={SEVERITY_TONE[c.severity]}>{c.severity}</Badge>
            <span className="micro-copy" style={{ minWidth: 70 }}>{c.area}</span>
            <span className="body-copy" style={{ fontSize: "0.82rem" }}>{c.message}</span>
          </div>
        ))
      )}
      <p className="micro-copy" style={{ margin: "0.6rem 0 0" }}>
        {passed.length}/{lint.summary.total_checks} checks passed · body ≈{lint.summary.body_tokens_estimate} tokens
      </p>
    </Card>
  );
}

function FitPanel({ fit }: { fit: FitReport }) {
  return (
    <Card>
      <div style={{ display: "flex", alignItems: "center", gap: "0.8rem", marginBottom: "0.8rem" }}>
        <p className="section-caption" style={{ margin: 0, flex: 1 }}>Task fit (judge)</p>
        <span className={`ds-scorebar__value is-${scoreTone(fit.overall)}`}>{Math.round(fit.overall * 100)}%</span>
        <Badge tone={VERDICT_TONE[fit.verdict]}>{fit.verdict.replace("_", " ")}</Badge>
      </div>

      {Object.entries(fit.criteria).map(([name, c]) => (
        <div key={name} style={{ marginBottom: "0.7rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.7rem", marginBottom: "0.25rem" }}>
            <span className="body-copy" style={{ flex: 1, fontSize: "0.85rem", fontWeight: 600 }}>
              {CRITERION_LABELS[name] ?? name}
            </span>
            <div className="ds-scorebar__track" style={{ width: 120 }}>
              <div className={`ds-scorebar__fill is-${scoreTone(c.score)}`} style={{ width: `${c.score * 100}%` }} />
            </div>
            <span className={`ds-scorebar__value is-${scoreTone(c.score)}`} style={{ minWidth: "2.4rem" }}>
              {Math.round(c.score * 100)}%
            </span>
          </div>
          {c.evidence && (
            <p className="micro-copy" style={{ margin: 0, fontStyle: "italic" }}>
              “{c.evidence}”
            </p>
          )}
          {c.reasoning && (
            <p className="micro-copy" style={{ margin: 0 }}>{c.reasoning}</p>
          )}
        </div>
      ))}

      {fit.missing_criteria.length > 0 && (
        <p className="micro-copy">Unscored criteria: {fit.missing_criteria.join(", ")}</p>
      )}

      {fit.gaps.length > 0 && (
        <>
          <p className="section-caption" style={{ margin: "0.8rem 0 0.4rem" }}>Gaps</p>
          {fit.gaps.map((g, i) => (
            <div key={i} className="alert-box alert-warning" style={{ marginBottom: "0.35rem" }}>{g}</div>
          ))}
        </>
      )}
      {fit.suggestions.length > 0 && (
        <>
          <p className="section-caption" style={{ margin: "0.8rem 0 0.4rem" }}>Suggestions</p>
          <ul style={{ margin: 0, paddingLeft: "1.1rem" }}>
            {fit.suggestions.map((s, i) => (
              <li key={i} className="body-copy" style={{ fontSize: "0.82rem" }}>{s}</li>
            ))}
          </ul>
        </>
      )}
    </Card>
  );
}

function TriggerPanel({ report }: { report: TriggerReport }) {
  const { summary } = report;
  const pct = (v: number | null | undefined) => (v == null ? "—" : `${Math.round(v * 100)}%`);
  const decisionMode = summary.mode === "decision";
  return (
    <Card>
      <div style={{ display: "flex", alignItems: "center", gap: "0.8rem", marginBottom: "0.8rem" }}>
        <p className="section-caption" style={{ margin: 0, flex: 1 }}>Trigger simulation</p>
        {decisionMode && <Badge tone="violet">Jev</Badge>}
        <Badge tone={TRIGGER_VERDICT_TONE[summary.verdict] ?? "neutral"}>
          {summary.verdict.replace("_", " ")}
        </Badge>
      </div>

      <div className="grid gap-3 sm:grid-cols-3" style={{ marginBottom: "0.9rem" }}>
        <Card className="stat-card" style={{ textAlign: "center" }}>
          <div className="stat-value" style={{ fontSize: "1.5rem" }}>{pct(summary.precision)}</div>
          <p className="stat-label">precision</p>
        </Card>
        <Card className="stat-card" style={{ textAlign: "center" }}>
          <div className="stat-value" style={{ fontSize: "1.5rem" }}>{pct(summary.recall)}</div>
          <p className="stat-label">recall</p>
        </Card>
        <Card className="stat-card" style={{ textAlign: "center" }}>
          <div className="stat-value" style={{ fontSize: "1.5rem" }}>{pct(summary.f1)}</div>
          <p className="stat-label">F1</p>
        </Card>
      </div>

      <p className="micro-copy" style={{ margin: "0 0 0.6rem" }}>
        {summary.scored} scored · {summary.skipped} skipped (unparseable)
        {summary.ambiguous_count > 0 &&
          ` · ${summary.ambiguous_count} ambiguous (${pct(summary.ambiguous_trigger_rate)} trigger rate)`}
      </p>

      <div className="table-shell" style={{ border: "none", boxShadow: "none", borderRadius: 0 }}>
        <table>
          <thead>
            <tr>
              {["Prompt", "Expected", "Predicted", decisionMode ? "p(trigger)" : "Trials", "Result"].map((h) => (
                <th key={h}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {report.results.map((r, i) => (
              <tr key={i}>
                <td style={{ fontSize: "0.82rem" }}>{r.text}</td>
                <td>{r.expected === "ambiguous" ? "ambiguous" : r.expected ? "true" : "false"}</td>
                <td>{r.predicted == null ? "—" : r.predicted ? "true" : "false"}</td>
                <td>{decisionMode ? <ScoreBar score={r.probability ?? null} /> : r.trials}</td>
                <td>
                  {r.correct == null ? (
                    <Badge tone="neutral">n/a</Badge>
                  ) : r.correct ? (
                    <Badge tone="success">correct</Badge>
                  ) : (
                    <Badge tone="danger">wrong</Badge>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {decisionMode && summary.threshold_curve && (
        <>
          <p className="section-caption" style={{ margin: "1rem 0 0.5rem" }}>Threshold curve</p>
          <div className="table-shell" style={{ border: "none", boxShadow: "none", borderRadius: 0 }}>
            <table>
              <thead>
                <tr>{["Threshold", "Precision", "Recall", "F1", "FP rate"].map((h) => <th key={h}>{h}</th>)}</tr>
              </thead>
              <tbody>
                {summary.threshold_curve.map((p) => (
                  <tr key={p.threshold}>
                    <td>{p.threshold}</td>
                    <td>{pct(p.precision)}</td>
                    <td>{pct(p.recall)}</td>
                    <td>{pct(p.f1)}</td>
                    <td>{pct(p.false_positive_rate)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Card>
  );
}

function RoutingPanel({ report }: { report: RoutingReport }) {
  const { metrics } = report;
  const pct = (v: number | null | undefined) => (v == null ? "—" : `${Math.round(v * 100)}%`);
  return (
    <Card>
      <div style={{ display: "flex", alignItems: "center", gap: "0.8rem", marginBottom: "0.8rem" }}>
        <p className="section-caption" style={{ margin: 0, flex: 1 }}>Multi-skill routing</p>
        <span className="micro-copy">{metrics.total} prompts</span>
      </div>

      <div className="grid gap-3 sm:grid-cols-2" style={{ marginBottom: "0.9rem" }}>
        <Card className="stat-card" style={{ textAlign: "center" }}>
          <div className="stat-value" style={{ fontSize: "1.5rem" }}>{pct(metrics.accuracy)}</div>
          <p className="stat-label">accuracy</p>
        </Card>
        <Card className="stat-card" style={{ textAlign: "center" }}>
          <div className="stat-value" style={{ fontSize: "1.5rem" }}>{pct(metrics.none_rate)}</div>
          <p className="stat-label">"none" rate</p>
        </Card>
      </div>

      {metrics.per_skill && (
        <>
          <p className="section-caption" style={{ margin: "0 0 0.5rem" }}>Per-skill</p>
          <div className="table-shell" style={{ border: "none", boxShadow: "none", borderRadius: 0, marginBottom: "0.9rem" }}>
            <table>
              <thead>
                <tr>{["Skill", "Precision", "Recall"].map((h) => <th key={h}>{h}</th>)}</tr>
              </thead>
              <tbody>
                {Object.entries(metrics.per_skill).map(([name, s]) => (
                  <tr key={name}>
                    <td className="table-code">{name}</td>
                    <td>{pct(s.precision)}</td>
                    <td>{pct(s.recall)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      <div className="table-shell" style={{ border: "none", boxShadow: "none", borderRadius: 0 }}>
        <table>
          <thead>
            <tr>{["Prompt", "Expected", "Predicted", "Confidence"].map((h) => <th key={h}>{h}</th>)}</tr>
          </thead>
          <tbody>
            {report.results.map((r, i) => (
              <tr key={i}>
                <td style={{ fontSize: "0.82rem" }}>{r.text}</td>
                <td className="table-code">{r.expected}</td>
                <td className="table-code">{r.predicted}</td>
                <td>
                  <Badge tone={r.predicted === r.expected ? "success" : "danger"}>
                    {Math.round(r.confidence * 100)}%
                  </Badge>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

export default function SkillLab() {
  const [skillText, setSkillText] = useState("");
  const [task, setTask] = useState("");
  const [fitModels, setFitModels] = useState<string[]>([]);
  const [fitJudgeModel, setFitJudgeModel] = useState("");
  const [triggerModels, setTriggerModels] = useState<string[]>([]);
  const [triggerJudgeModel, setTriggerJudgeModel] = useState("");
  const [lintOnly, setLintOnly] = useState<LintReport | null>(null);
  const [full, setFull] = useState<FullReport | null>(null);
  const [history, setHistory] = useState<ReportSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [triggerPromptsText, setTriggerPromptsText] = useState(SAMPLE_TRIGGER_PROMPTS);
  const [repeats, setRepeats] = useState(1);
  const [triggerReport, setTriggerReport] = useState<TriggerReport | null>(null);
  const [triggerLoading, setTriggerLoading] = useState(false);

  const [routingSkills, setRoutingSkills] = useState<RoutingSkillEntry[]>([
    { name: "", skill_text: "" },
    { name: "", skill_text: "" },
  ]);
  const [routingPromptsText, setRoutingPromptsText] = useState("");
  const [routingDecisionModel, setRoutingDecisionModel] = useState("");
  const [routingReport, setRoutingReport] = useState<RoutingReport | null>(null);
  const [routingLoading, setRoutingLoading] = useState(false);

  const fileRef = useRef<HTMLInputElement>(null);
  const toast = useToast();

  useEffect(() => {
    modelsApi
      .list()
      .then((r) => {
        const { llm, decision } = splitModels(r.models);
        setFitModels(llm);
        setFitJudgeModel((prev) => prev || llm[0] || "");
        const all = [...llm, ...decision];
        setTriggerModels(all);
        setTriggerJudgeModel((prev) => prev || all[0] || "");
      })
      .catch(() => {
        setFitModels([]);
        setTriggerModels([]);
      });
    void refreshHistory();
  }, []);

  async function refreshHistory() {
    try {
      const r = await fetch(`${BASE}/skill-eval/reports`);
      if (r.ok) setHistory((await r.json()).reports ?? []);
    } catch {
      /* history is best-effort */
    }
  }

  function onUpload(file: File | undefined) {
    if (!file) return;
    file.text().then(setSkillText).catch(() => toast.error("Could not read file."));
  }

  async function runLint() {
    if (!skillText.trim()) {
      toast.error("Paste or upload a SKILL.md first.");
      return;
    }
    setLoading(true);
    setFull(null);
    try {
      const r = await apiPost<LintReport>("/skill-eval/lint", { skill_text: skillText });
      setLintOnly(r);
      toast.success(`Lint done — score ${r.score}/100.`);
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  async function runFull() {
    if (!skillText.trim()) {
      toast.error("Paste or upload a SKILL.md first.");
      return;
    }
    if (!task.trim()) {
      toast.error("Describe the task you want the skill to handle.");
      return;
    }
    if (!fitJudgeModel) {
      toast.error("Pick a judge model.");
      return;
    }
    setLoading(true);
    setLintOnly(null);
    try {
      const r = await apiPost<FullReport>("/skill-eval/full", {
        skill_text: skillText,
        task_description: task,
        judge_model: fitJudgeModel,
      });
      setFull(r);
      toast.success(`Combined score ${Math.round(r.combined_score * 100)}% (${r.combined_basis}).`);
      void refreshHistory();
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  async function runTrigger() {
    if (!skillText.trim()) {
      toast.error("Paste or upload a SKILL.md first.");
      return;
    }
    if (!triggerJudgeModel) {
      toast.error("Pick a judge model.");
      return;
    }
    const prompts = parseTriggerPrompts(triggerPromptsText);
    if (prompts.length === 0) {
      toast.error("Add at least one prompt (TRUE:/FALSE:/AMBIGUOUS: prefix).");
      return;
    }
    setTriggerLoading(true);
    try {
      const r = await apiPost<TriggerReport>("/skill-eval/trigger", {
        skill_text: skillText,
        judge_model: triggerJudgeModel,
        prompts,
        repeats,
      });
      setTriggerReport(r);
      toast.success(`Trigger sim done — verdict: ${r.summary.verdict.replace("_", " ")}.`);
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setTriggerLoading(false);
    }
  }

  function updateRoutingSkill(index: number, patch: Partial<RoutingSkillEntry>) {
    setRoutingSkills((prev) => prev.map((s, i) => (i === index ? { ...s, ...patch } : s)));
  }

  function addRoutingSkill() {
    setRoutingSkills((prev) => [...prev, { name: "", skill_text: "" }]);
  }

  function removeRoutingSkill(index: number) {
    setRoutingSkills((prev) => (prev.length > 2 ? prev.filter((_, i) => i !== index) : prev));
  }

  async function runRouting() {
    const skills = routingSkills.filter((s) => s.name.trim() && s.skill_text.trim());
    if (skills.length < 2) {
      toast.error("Add at least two named skills with content.");
      return;
    }
    if (!routingDecisionModel) {
      toast.error("Pick a decision model.");
      return;
    }
    const skillNames = new Set(skills.map((s) => s.name.trim()));
    const prompts = parseRoutingPrompts(routingPromptsText).filter(
      (p) => p.expected === "none" || skillNames.has(p.expected),
    );
    if (prompts.length === 0) {
      toast.error('Add at least one prompt (e.g. "skill-name: prompt text" or "none: prompt text").');
      return;
    }
    setRoutingLoading(true);
    try {
      const r = await apiPost<RoutingReport>("/skill-eval/route", {
        skills: skills.map((s) => ({ name: s.name.trim(), skill_text: s.skill_text })),
        prompts,
        decision_model: routingDecisionModel,
      });
      setRoutingReport(r);
      toast.success(`Routing sim done — ${Math.round((r.metrics.accuracy ?? 0) * 100)}% accuracy.`);
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setRoutingLoading(false);
    }
  }

  const lint = full?.lint ?? lintOnly;

  return (
    <div className="page-shell motion-stagger-stack">
      <PageHeader
        kicker="Skill Lab"
        title="Skill Quality Lab"
        subtitle="Is this SKILL.md good enough for the job you want done? Static lint + task-fit judge with evidence quotes."
        help={
          <>
            Paste a <strong>SKILL.md</strong> and describe your task. <strong>Lint</strong> runs instant
            static checks (format, size, security red flags). <strong>Full evaluation</strong> adds an
            LLM judge scoring five task-fit criteria with verbatim evidence from the skill, and saves
            a report you can revisit below.
          </>
        }
      />

      {/* Input */}
      <Card>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.4rem", gap: "0.6rem", flexWrap: "wrap" }}>
          <span className="ds-field-mini" style={{ marginBottom: 0 }}>SKILL.md content</span>
          <div style={{ display: "flex", gap: "0.5rem" }}>
            <button
              type="button"
              className="ds-icon-button"
              style={{ width: "auto", padding: "0.3rem 0.6rem", fontSize: "0.75rem" }}
              onClick={() => setSkillText(SAMPLE_SKILL)}
            >
              Load example
            </button>
            <button
              type="button"
              className="ds-icon-button"
              style={{ width: "auto", padding: "0.3rem 0.6rem", fontSize: "0.75rem" }}
              onClick={() => fileRef.current?.click()}
            >
              <Upload size={12} style={{ marginRight: 4 }} />
              Upload
            </button>
            <input
              ref={fileRef}
              type="file"
              accept=".md,text/markdown,text/plain"
              style={{ display: "none" }}
              onChange={(e) => onUpload(e.target.files?.[0])}
            />
          </div>
        </div>
        <Textarea
          value={skillText}
          onChange={(e) => setSkillText(e.target.value)}
          rows={10}
          className="font-mono"
          placeholder={"---\nname: my-skill\ndescription: …\n---\n# Instructions…"}
          style={{ fontSize: "0.76rem" }}
        />

        <div className="grid gap-3 sm:grid-cols-2" style={{ marginTop: "0.8rem" }}>
          <Field label="Task description (what you want the skill to do)">
            <Textarea
              value={task}
              onChange={(e) => setTask(e.target.value)}
              rows={3}
              placeholder="e.g. Generate a regional totals report from a weekly sales CSV…"
            />
          </Field>
          <Field label="Judge model">
            <Select value={fitJudgeModel} onChange={(e) => setFitJudgeModel(e.target.value)}>
              {fitModels.length === 0 && <option value="">No models configured</option>}
              {fitModels.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </Select>
          </Field>
        </div>

        <div className="button-row" style={{ marginTop: "0.9rem" }}>
          <Button variant="secondary" loading={loading} onClick={runLint}>
            Lint only
          </Button>
          <Button icon={<Play size={14} />} loading={loading} onClick={runFull}>
            Full evaluation
          </Button>
        </div>
      </Card>

      {/* Combined score */}
      {full && (
        <div className="grid gap-3 sm:grid-cols-3">
          <Card className="stat-card" style={{ textAlign: "center" }}>
            <div className="stat-value" style={{ fontSize: "1.8rem" }}>
              {Math.round(full.combined_score * 100)}%
            </div>
            <p className="stat-label">combined ({full.combined_basis})</p>
          </Card>
          <Card className="stat-card" style={{ textAlign: "center" }}>
            <div className="stat-value" style={{ fontSize: "1.8rem" }}>{full.lint.score}</div>
            <p className="stat-label">lint /100</p>
          </Card>
          <Card className="stat-card" style={{ textAlign: "center" }}>
            <div className="stat-value" style={{ fontSize: "1.8rem" }}>
              {full.fit ? `${Math.round(full.fit.overall * 100)}%` : "—"}
            </div>
            <p className="stat-label">task fit</p>
          </Card>
        </div>
      )}

      {full?.combined_basis === "lint_only" && (
        <div className="alert-box alert-warning">
          Judge output was unusable — combined score is lint-only. Try another judge model.
        </div>
      )}

      {/* Panels */}
      {(lint || full?.fit) && (
        <div className="grid gap-3 lg:grid-cols-2" style={{ alignItems: "start" }}>
          {lint && <LintPanel lint={lint} />}
          {full?.fit && <FitPanel fit={full.fit} />}
        </div>
      )}

      {/* Trigger simulation */}
      <Card>
        <p className="section-caption" style={{ marginBottom: "0.3rem" }}>Trigger simulation</p>
        <p className="micro-copy" style={{ margin: "0 0 0.7rem" }}>
          Probes the judge model with only the skill's <strong>name + description</strong> (never the
          body) against labeled prompts, to see whether routing actually fires when it should.
        </p>
        <Field label={'Labeled prompts — one per line: "TRUE: …", "FALSE: …", or "AMBIGUOUS: …"'}>
          <Textarea
            value={triggerPromptsText}
            onChange={(e) => setTriggerPromptsText(e.target.value)}
            rows={6}
            className="font-mono"
            style={{ fontSize: "0.76rem" }}
          />
        </Field>
        <div className="grid gap-3 sm:grid-cols-2" style={{ marginTop: "0.6rem" }}>
          <Field label="Judge model">
            <Select value={triggerJudgeModel} onChange={(e) => setTriggerJudgeModel(e.target.value)}>
              {triggerModels.length === 0 && <option value="">No models configured</option>}
              {triggerModels.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </Select>
          </Field>
          <Field label="Repeats per prompt (majority vote — ignored for Jev)">
            <Select value={String(repeats)} onChange={(e) => setRepeats(Number(e.target.value))}>
              {[1, 2, 3, 5].map((n) => (
                <option key={n} value={n}>{n}</option>
              ))}
            </Select>
          </Field>
        </div>
        <div className="button-row" style={{ marginTop: "0.9rem" }}>
          <Button icon={<Play size={14} />} loading={triggerLoading} onClick={runTrigger}>
            Run trigger simulation
          </Button>
        </div>
      </Card>

      {triggerReport && <TriggerPanel report={triggerReport} />}

      {/* Multi-skill routing */}
      <Card>
        <p className="section-caption" style={{ marginBottom: "0.3rem" }}>Multi-skill routing</p>
        <p className="micro-copy" style={{ margin: "0 0 0.7rem" }}>
          Given several candidate skills, which one (if any) should handle each prompt? Requires a
          Jev decision model — this is a single-shot choice among all listed skills, not a per-skill
          yes/no probe.
        </p>

        {routingSkills.map((skill, i) => (
          <div key={i} className="panel-surface panel-quiet" style={{ marginBottom: "0.7rem" }}>
            <div style={{ display: "flex", gap: "0.6rem", alignItems: "flex-end", marginBottom: "0.5rem" }}>
              <Field label={`Skill ${i + 1} name`}>
                <Input
                  value={skill.name}
                  onChange={(e) => updateRoutingSkill(i, { name: e.target.value })}
                  placeholder="csv-report"
                />
              </Field>
              {routingSkills.length > 2 && (
                <button
                  type="button"
                  className="ds-icon-button"
                  aria-label={`Remove skill ${i + 1}`}
                  onClick={() => removeRoutingSkill(i)}
                >
                  <Trash2 size={14} />
                </button>
              )}
            </div>
            <Textarea
              value={skill.skill_text}
              onChange={(e) => updateRoutingSkill(i, { skill_text: e.target.value })}
              rows={4}
              className="font-mono"
              style={{ fontSize: "0.76rem" }}
              placeholder={"---\nname: csv-report\ndescription: …\n---\n# Instructions…"}
            />
          </div>
        ))}
        <button
          type="button"
          className="ds-icon-button"
          style={{ width: "auto", padding: "0.3rem 0.6rem", fontSize: "0.75rem", marginBottom: "0.9rem" }}
          onClick={addRoutingSkill}
        >
          <Plus size={12} style={{ marginRight: 4 }} />
          Add skill
        </button>

        <Field label='Labeled prompts — one per line: "skill-name: prompt text" or "none: prompt text"'>
          <Textarea
            value={routingPromptsText}
            onChange={(e) => setRoutingPromptsText(e.target.value)}
            rows={5}
            className="font-mono"
            style={{ fontSize: "0.76rem" }}
            placeholder={"csv-report: Generate the weekly sales CSV report\nnone: What is the weather in Ankara?"}
          />
        </Field>

        <div style={{ marginTop: "0.6rem" }}>
          <DecisionModelSelect value={routingDecisionModel} onChange={setRoutingDecisionModel} label="Decision model" />
        </div>

        <div className="button-row" style={{ marginTop: "0.9rem" }}>
          <Button icon={<Play size={14} />} loading={routingLoading} onClick={runRouting}>
            Run routing simulation
          </Button>
        </div>
      </Card>

      {routingReport && <RoutingPanel report={routingReport} />}

      {/* History */}
      {history.length > 0 && (
        <Card>
          <p className="section-caption" style={{ marginBottom: "0.7rem" }}>Previous evaluations</p>
          <div className="table-shell" style={{ border: "none", boxShadow: "none", borderRadius: 0 }}>
            <table>
              <thead>
                <tr>{["Skill", "When", "Judge", "Lint", "Fit", "Verdict", "Combined"].map((h) => <th key={h}>{h}</th>)}</tr>
              </thead>
              <tbody>
                {history.map((r) => (
                  <tr key={r.filename}>
                    <td className="table-code">{r.skill_name ?? "—"}</td>
                    <td>{r.timestamp ? new Date(r.timestamp).toLocaleString() : "—"}</td>
                    <td>{r.judge_model ?? "—"}</td>
                    <td>{r.lint_score ?? "—"}</td>
                    <td>{r.fit_overall != null ? `${Math.round(r.fit_overall * 100)}%` : "—"}</td>
                    <td>
                      {r.verdict ? (
                        <Badge tone={VERDICT_TONE[r.verdict] ?? "neutral"}>{r.verdict.replace("_", " ")}</Badge>
                      ) : (
                        "—"
                      )}
                    </td>
                    <td>
                      {r.combined_score != null ? (
                        <span className={`ds-scorebar__value is-${scoreTone(r.combined_score)}`}>
                          {Math.round(r.combined_score * 100)}%
                        </span>
                      ) : (
                        "—"
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}
