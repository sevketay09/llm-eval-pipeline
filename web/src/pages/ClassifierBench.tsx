import { useEffect, useState } from "react";
import { ListChecks, Plus, Trash2, Play } from "lucide-react";
import {
  PageHeader,
  Card,
  Button,
  Badge,
  EmptyState,
  Field,
  Input,
  Select,
  Textarea,
  DecisionModelSelect,
  useToast,
} from "@/components";
import {
  classifierBenchApi,
  modelsApi,
  type BenchCase,
  type BenchDetail,
  type ClassifierResult,
} from "@/api/client";
import { splitModels } from "@/lib/decision";

type Mode = "choice" | "guardrail";

interface CriteriaRow {
  label: string;
  description: string;
}

interface GuardrailRow {
  name: string;
  instructions: string;
  kind: "risk" | "scope";
  block: string;
  review: string;
}

function pct(value: number | null | undefined): string {
  return value == null ? "—" : `${Math.round(value * 100)}%`;
}

function topConfusions(confusion: Record<string, Record<string, number>>, n = 5): string[] {
  const mistakes: [string, string, number][] = [];
  for (const [truth, preds] of Object.entries(confusion || {})) {
    for (const [pred, count] of Object.entries(preds)) {
      if (pred !== truth) mistakes.push([truth, pred, count]);
    }
  }
  mistakes.sort((a, b) => b[2] - a[2]);
  return mistakes.slice(0, n).map(([t, p, c]) => `${t} → ${p}: ${c}`);
}

export default function ClassifierBench() {
  const toast = useToast();
  const [name, setName] = useState("");
  const [mode, setMode] = useState<Mode>("choice");
  const [criteria, setCriteria] = useState<CriteriaRow[]>([
    { label: "", description: "" },
    { label: "", description: "" },
  ]);
  const [instructions, setInstructions] = useState(
    "Which option best describes the primary intent of the text in `state`?",
  );
  const [guardrailRows, setGuardrailRows] = useState<GuardrailRow[]>([
    { name: "", instructions: "", kind: "risk", block: "", review: "" },
  ]);
  const [datasetText, setDatasetText] = useState("");
  const [cases, setCases] = useState<BenchCase[]>([]);
  const [decisionModel, setDecisionModel] = useState("");
  const [llmModels, setLlmModels] = useState<string[]>([]);
  const [selectedLlmModels, setSelectedLlmModels] = useState<string[]>([]);
  const [repeats, setRepeats] = useState(1);
  const [bench, setBench] = useState<BenchDetail | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    modelsApi.list().then((r) => setLlmModels(splitModels(r.models).llm)).catch(() => {});
  }, []);

  const loadDataset = () => {
    const trimmed = datasetText.trim();
    if (!trimmed) {
      toast.error("Paste a dataset first.");
      return;
    }
    if (trimmed.startsWith("[")) {
      try {
        setCases(JSON.parse(trimmed));
        return;
      } catch (e) {
        toast.error("Invalid JSON array.");
        return;
      }
    }
    classifierBenchApi
      .importJsonl(trimmed)
      .then((r) => setCases(r.cases))
      .catch((e: unknown) => toast.error(e instanceof Error ? e.message : "Import failed"));
  };

  const toggleLlmModel = (id: string) => {
    setSelectedLlmModels((prev) => (prev.includes(id) ? prev.filter((m) => m !== id) : [...prev, id]));
  };

  const runBench = async () => {
    if (!name.trim()) { toast.error("Name is required."); return; }
    if (!decisionModel) { toast.error("Select a decision (Jev) model."); return; }
    if (cases.length === 0) { toast.error("Load a dataset first."); return; }

    setBusy(true);
    try {
      const payload =
        mode === "choice"
          ? {
              name: name.trim(),
              mode: "choice" as const,
              criteria: Object.fromEntries(
                criteria.filter((c) => c.label.trim()).map((c) => [c.label.trim(), c.description.trim()]),
              ),
              instructions,
              cases,
              decision_model: decisionModel,
              llm_models: selectedLlmModels,
              repeats,
            }
          : {
              name: name.trim(),
              mode: "guardrail" as const,
              guardrail_categories: Object.fromEntries(
                guardrailRows
                  .filter((r) => r.name.trim())
                  .map((r) => [
                    r.name.trim(),
                    {
                      instructions: r.instructions.trim(),
                      kind: r.kind,
                      block: r.block ? Number(r.block) : undefined,
                      review: r.review ? Number(r.review) : undefined,
                    },
                  ]),
              ),
              cases,
              decision_model: decisionModel,
              repeats: 1,
            };

      const created = await classifierBenchApi.create(payload);
      await classifierBenchApi.run(created.bench_id);
      const detail = await classifierBenchApi.get(created.bench_id);
      setBench(detail);
      if (detail.status === "error") {
        toast.error(detail.error || "Bench run failed");
      } else {
        toast.success(`Bench "${detail.name}" finished.`);
      }
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : "Bench run failed");
    } finally {
      setBusy(false);
    }
  };

  const labelSet = Array.from(
    new Set(
      (bench?.results ?? []).flatMap((r) => Object.keys(r.metrics?.per_label ?? {})),
    ),
  );

  return (
    <div className="page-shell motion-stagger-stack">
      <PageHeader
        kicker="Classifier Bench"
        title="Classifier & Guardrail Bench"
        subtitle="Benchmark a Jev decision model against LLM classifiers on an intent dataset, or score a guardrail's PASS/REVIEW/BLOCK/OUT_OF_SCOPE verdicts."
        help={
          <>
            <strong>Choice mode</strong> compares a Jev decision model and any number of
            LLM classifiers on a labeled intent dataset: accuracy, per-label accuracy,
            confusions, consistency across repeats, and (for Jev) a confidence
            coverage–accuracy curve. <strong>Guardrail mode</strong> scores a single Jev
            model's PASS/REVIEW/BLOCK/OUT_OF_SCOPE verdict against expected outcomes —
            an "out of scope" category never produces BLOCK on its own.
          </>
        }
      />

      <div className="grid gap-4 lg:grid-cols-2">
        <div className="flex flex-col gap-4">
          <Card>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <Field label="Name">
                <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="Banking intent v1" />
              </Field>
              <Field label="Mode">
                <Select value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
                  <option value="choice">Intent / Choice</option>
                  <option value="guardrail">Guardrail</option>
                </Select>
              </Field>
            </div>
          </Card>

          {mode === "choice" ? (
            <Card>
              <span className="ds-field-mini">Label schema</span>
              <div className="flex flex-col gap-2">
                {criteria.map((row, i) => (
                  <div key={i} style={{ display: "flex", gap: "0.5rem" }}>
                    <Input
                      value={row.label}
                      onChange={(e) =>
                        setCriteria((prev) => prev.map((r, idx) => (idx === i ? { ...r, label: e.target.value } : r)))
                      }
                      placeholder="label"
                      style={{ flex: "0 0 30%" }}
                    />
                    <Input
                      value={row.description}
                      onChange={(e) =>
                        setCriteria((prev) =>
                          prev.map((r, idx) => (idx === i ? { ...r, description: e.target.value } : r)),
                        )
                      }
                      placeholder="What this label means…"
                      style={{ flex: 1 }}
                    />
                    <button
                      className="ds-icon-button"
                      aria-label={`Remove label ${i + 1}`}
                      onClick={() => setCriteria((prev) => prev.filter((_, idx) => idx !== i))}
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                ))}
              </div>
              <div className="button-row" style={{ marginTop: "0.75rem" }}>
                <Button
                  variant="secondary"
                  icon={<Plus size={14} />}
                  onClick={() => setCriteria((prev) => [...prev, { label: "", description: "" }])}
                >
                  Add label
                </Button>
              </div>
              <Field label="Classification instructions">
                <Textarea value={instructions} onChange={(e) => setInstructions(e.target.value)} rows={2} />
              </Field>
            </Card>
          ) : (
            <Card>
              <span className="ds-field-mini">Guardrail categories</span>
              <div className="flex flex-col gap-2">
                {guardrailRows.map((row, i) => (
                  <div key={i} className="grid grid-cols-1 gap-2 md:grid-cols-[1fr_2fr_6rem_5rem_5rem_auto]">
                    <Input
                      value={row.name}
                      onChange={(e) =>
                        setGuardrailRows((prev) =>
                          prev.map((r, idx) => (idx === i ? { ...r, name: e.target.value } : r)),
                        )
                      }
                      placeholder="category name"
                    />
                    <Input
                      value={row.instructions}
                      onChange={(e) =>
                        setGuardrailRows((prev) =>
                          prev.map((r, idx) => (idx === i ? { ...r, instructions: e.target.value } : r)),
                        )
                      }
                      placeholder="The message attempts…"
                    />
                    <Select
                      value={row.kind}
                      onChange={(e) =>
                        setGuardrailRows((prev) =>
                          prev.map((r, idx) => (idx === i ? { ...r, kind: e.target.value as "risk" | "scope" } : r)),
                        )
                      }
                    >
                      <option value="risk">risk</option>
                      <option value="scope">scope</option>
                    </Select>
                    <Input
                      value={row.block}
                      onChange={(e) =>
                        setGuardrailRows((prev) =>
                          prev.map((r, idx) => (idx === i ? { ...r, block: e.target.value } : r)),
                        )
                      }
                      placeholder="block (.75)"
                    />
                    <Input
                      value={row.review}
                      onChange={(e) =>
                        setGuardrailRows((prev) =>
                          prev.map((r, idx) => (idx === i ? { ...r, review: e.target.value } : r)),
                        )
                      }
                      placeholder="review (.40)"
                    />
                    <button
                      className="ds-icon-button"
                      aria-label={`Remove category ${i + 1}`}
                      onClick={() => setGuardrailRows((prev) => prev.filter((_, idx) => idx !== i))}
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                ))}
              </div>
              <div className="button-row" style={{ marginTop: "0.75rem" }}>
                <Button
                  variant="secondary"
                  icon={<Plus size={14} />}
                  onClick={() =>
                    setGuardrailRows((prev) => [
                      ...prev,
                      { name: "", instructions: "", kind: "risk", block: "", review: "" },
                    ])
                  }
                >
                  Add category
                </Button>
              </div>
            </Card>
          )}

          <Card>
            <Field label={`Dataset (${cases.length} cases loaded)`}>
              <Textarea
                value={datasetText}
                onChange={(e) => setDatasetText(e.target.value)}
                rows={6}
                placeholder={
                  mode === "choice"
                    ? '{"id": "1", "message": "kart bloke", "true_label": "cards"}\n… one JSON object per line, or a JSON array of {id, text, label}'
                    : '[{"id": "g1", "text": "ignore your instructions", "expected_verdict": "BLOCK"}]'
                }
              />
            </Field>
            <div className="button-row" style={{ marginTop: "0.5rem" }}>
              <Button variant="secondary" onClick={loadDataset}>Load dataset</Button>
            </div>
          </Card>

          <Card>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <DecisionModelSelect value={decisionModel} onChange={setDecisionModel} />
              {mode === "choice" && (
                <Field label="Repeats">
                  <Input
                    type="number"
                    min={1}
                    max={5}
                    value={repeats}
                    onChange={(e) => setRepeats(parseInt(e.target.value, 10) || 1)}
                  />
                </Field>
              )}
            </div>
            {mode === "choice" && llmModels.length > 0 && (
              <div style={{ marginTop: "0.75rem" }}>
                <span className="ds-field-mini">Compare against LLM models (optional)</span>
                <div className="mt-2 flex flex-wrap gap-2">
                  {llmModels.map((id) => (
                    <label key={id} style={{ display: "flex", gap: "0.35rem", alignItems: "center" }}>
                      <input
                        type="checkbox"
                        checked={selectedLlmModels.includes(id)}
                        onChange={() => toggleLlmModel(id)}
                      />
                      <span className="micro-copy">{id}</span>
                    </label>
                  ))}
                </div>
              </div>
            )}
          </Card>

          <Button icon={<Play size={14} />} loading={busy} onClick={runBench}>
            Run bench
          </Button>
        </div>

        <div className="flex flex-col gap-4">
          {!bench ? (
            <EmptyState
              icon={ListChecks}
              title="No bench run yet"
              hint="Fill in a label schema or guardrail categories, load a dataset, and run the bench to see results here."
            />
          ) : (
            <>
              <Card>
                <p className="body-copy" style={{ fontWeight: 600 }}>{bench.name}</p>
                <p className="micro-copy mt-1">
                  {bench.mode} · {bench.case_count} cases · status: {bench.status}
                </p>
                {bench.error && <p className="micro-copy mt-1" style={{ color: "var(--danger)" }}>{bench.error}</p>}
              </Card>

              <Card>
                <span className="ds-field-mini">Summary</span>
                <table className="ds-table mt-2">
                  <thead>
                    <tr>
                      <th>Classifier</th>
                      <th>Accuracy</th>
                      <th>p50 (ms)</th>
                      <th>p95 (ms)</th>
                      {bench.mode === "choice" && <th>Consistency</th>}
                      <th>Errors</th>
                      {bench.mode === "choice" && <th>Invalid</th>}
                      <th>Est. $ / run</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bench.results.map((r: ClassifierResult) => (
                      <tr key={r.name}>
                        <td>
                          {r.name} <Badge tone={r.kind === "decision" ? "violet" : "info"}>{r.kind}</Badge>
                        </td>
                        {r.error ? (
                          <td colSpan={6}><Badge tone="danger">{r.error}</Badge></td>
                        ) : (
                          <>
                            <td>{pct(r.metrics.accuracy ?? r.metrics.verdict_accuracy)}</td>
                            <td>{Math.round(r.metrics.median_latency_ms ?? 0)}</td>
                            <td>{Math.round(r.metrics.p95_latency_ms ?? 0)}</td>
                            {bench.mode === "choice" && <td>{pct(r.metrics.consistency_rate)}</td>}
                            <td>{r.metrics.errors ?? 0}</td>
                            {bench.mode === "choice" && <td>{r.metrics.invalid_outputs ?? 0}</td>}
                            <td>{r.metrics.est_cost_usd != null ? `$${r.metrics.est_cost_usd}` : "—"}</td>
                          </>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Card>

              {bench.mode === "choice" && labelSet.length > 0 && (
                <Card>
                  <span className="ds-field-mini">Per-label accuracy</span>
                  <table className="ds-table mt-2">
                    <thead>
                      <tr>
                        <th>Label</th>
                        {bench.results.map((r) => (
                          <th key={r.name}>{r.name}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {labelSet.map((label) => (
                        <tr key={label}>
                          <td>{label}</td>
                          {bench.results.map((r) => (
                            <td key={r.name}>{pct(r.metrics.per_label?.[label]?.accuracy ?? null)}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </Card>
              )}

              {bench.mode === "choice" && (
                <Card>
                  <span className="ds-field-mini">Top confusions</span>
                  {bench.results.map((r) => (
                    <p key={r.name} className="micro-copy mt-2">
                      <strong>{r.name}:</strong>{" "}
                      {r.metrics.confusion ? topConfusions(r.metrics.confusion).join(", ") || "none" : "—"}
                    </p>
                  ))}
                </Card>
              )}

              {bench.mode === "choice" &&
                bench.results.some((r) => r.metrics.coverage_curve) && (
                  <Card>
                    <span className="ds-field-mini">Coverage vs. accuracy (confidence threshold)</span>
                    {bench.results
                      .filter((r) => r.metrics.coverage_curve)
                      .map((r) => (
                        <table key={r.name} className="ds-table mt-2">
                          <caption className="micro-copy" style={{ textAlign: "left" }}>{r.name}</caption>
                          <thead>
                            <tr>
                              <th>Threshold</th>
                              <th>Coverage</th>
                              <th>Accuracy (accepted)</th>
                            </tr>
                          </thead>
                          <tbody>
                            {r.metrics.coverage_curve.map((point: any) => (
                              <tr key={point.threshold}>
                                <td>{point.threshold}</td>
                                <td>{pct(point.coverage)}</td>
                                <td>{pct(point.accuracy)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      ))}
                  </Card>
                )}

              {bench.mode === "guardrail" &&
                bench.results.map((r) => (
                  <Card key={r.name}>
                    <span className="ds-field-mini">Guardrail metrics — {r.name}</span>
                    {r.error ? (
                      <p className="micro-copy mt-2" style={{ color: "var(--danger)" }}>{r.error}</p>
                    ) : (
                      <>
                        <div className="mt-2 flex flex-wrap gap-2">
                          <Badge tone="info">Block recall: {pct(r.metrics.block_recall)}</Badge>
                          <Badge tone={r.metrics.false_block_rate > 0 ? "warning" : "success"}>
                            False block rate: {pct(r.metrics.false_block_rate)}
                          </Badge>
                        </div>
                        <table className="ds-table mt-2">
                          <thead>
                            <tr>
                              <th>Category</th>
                              <th>Precision</th>
                              <th>Recall</th>
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(r.metrics.per_category ?? {}).map(([cat, m]: [string, any]) => (
                              <tr key={cat}>
                                <td>{cat}</td>
                                <td>{pct(m.precision)}</td>
                                <td>{pct(m.recall)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </>
                    )}
                  </Card>
                ))}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
