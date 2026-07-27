import { useEffect, useState } from "react";
import { BookOpen, Play, Plus, Trash2 } from "lucide-react";
import {
  PageHeader,
  Card,
  Button,
  Badge,
  ScoreBar,
  EmptyState,
  Field,
  Input,
  Select,
  Textarea,
  useToast,
} from "@/components";
import type { BadgeTone } from "@/components";
import { modelsApi, type EmbeddingModelConfig } from "@/api/client";

const BASE = "/api";

interface RagEvalResponse {
  question: string;
  context_precision: number;
  context_recall: number;
  context_recall_applicable: boolean;
  faithfulness: number;
  answer_relevance: number;
  fault_component: string;
  overall_score: number;
  scoring_mode: string;
  embedding_model: string | null;
  details: {
    context_precision?: { chunk_scores?: number[] };
  };
}

const CHUNK_RELEVANCE_THRESHOLD = 0.5; // mirrors analysis/rag_eval.py's compute_context_precision

async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

function MetricRow({
  label,
  score,
  desc,
  applicable = true,
}: {
  label: string;
  score: number;
  desc: string;
  applicable?: boolean;
}) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "0.9rem", padding: "0.6rem 0" }}>
      <div style={{ minWidth: 160 }}>
        <div className="body-copy" style={{ fontSize: "0.82rem", fontWeight: 600 }}>{label}</div>
        <div className="micro-copy">
          {applicable ? desc : "No expected answer was given — nothing to compare context against."}
        </div>
      </div>
      <div style={{ flex: 1 }}>
        {applicable ? (
          <ScoreBar score={score} />
        ) : (
          <Badge tone="neutral" mono>N/A</Badge>
        )}
      </div>
    </div>
  );
}

function ChunkPrecisionBreakdown({ contexts, scores }: { contexts: string[]; scores: number[] }) {
  if (contexts.length === 0 || scores.length === 0) return null;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem" }}>
      {contexts.map((text, i) => {
        const score = scores[i];
        if (score === undefined) return null;
        const relevant = score >= CHUNK_RELEVANCE_THRESHOLD;
        return (
          <div
            key={i}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.7rem",
              padding: "0.5rem 0.7rem",
              borderRadius: 8,
              background: "var(--panel-2, rgba(127,127,127,0.08))",
            }}
          >
            <Badge tone={relevant ? "success" : "warning"} mono>
              {score.toFixed(2)}
            </Badge>
            <span className="micro-copy" style={{ flex: 1, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
              {text}
            </span>
          </div>
        );
      })}
    </div>
  );
}

const FAULT_TONE: Record<string, BadgeTone> = {
  retriever: "warning",
  generator: "danger",
  both: "warning",
  none: "success",
};

export default function RagEval() {
  const [question, setQuestion] = useState("");
  const [contexts, setContexts] = useState([""]);
  const [answer, setAnswer] = useState("");
  const [expected, setExpected] = useState("");
  const [embeddingModels, setEmbeddingModels] = useState<Record<string, EmbeddingModelConfig>>({});
  const [embeddingModel, setEmbeddingModel] = useState("");
  const [result, setResult] = useState<RagEvalResponse | null>(null);
  const [evaluatedContexts, setEvaluatedContexts] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const toast = useToast();

  useEffect(() => {
    modelsApi.listEmbeddings().then((r) => setEmbeddingModels(r.models)).catch(() => {
      // Non-fatal: the page still works in token-overlap mode without this.
    });
  }, []);

  async function evaluate() {
    if (!question.trim() || !answer.trim()) { toast.error("Question and answer are required."); return; }
    const validCtx = contexts.filter(c => c.trim());
    if (validCtx.length === 0) { toast.error("Add at least one context chunk."); return; }
    setLoading(true);
    try {
      const r = await apiPost<RagEvalResponse>("/rag-eval", {
        question: question.trim(),
        contexts: validCtx.map(t => ({ text: t })),
        answer: answer.trim(),
        expected_answer: expected.trim(),
        embedding_model: embeddingModel || undefined,
      });
      setResult(r);
      setEvaluatedContexts(validCtx);
      toast.success(`Scored — fault component: ${r.fault_component}.`);
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page-shell motion-stagger-stack">
      <PageHeader
        kicker="RAG Eval"
        title="Retrieval Diagnostics"
        subtitle="Decompose a RAG answer into component scores and isolate whether the retriever or the generator is at fault."
        help={
          <>
            Provide the question, the retrieved context chunks, and the model's answer
            (expected answer is optional). You get four component scores — context
            precision/recall, faithfulness, answer relevance — plus a{" "}
            <strong>fault component</strong> verdict that points at the retriever vs.
            the generator.
          </>
        }
      />

      <div className="grid gap-4 lg:grid-cols-2">
        {/* Left — inputs */}
        <div className="flex flex-col gap-4">
          <Card>
            <Field label="Question">
              <Textarea value={question} onChange={e => setQuestion(e.target.value)} rows={2} placeholder="What is the return policy?" />
            </Field>
          </Card>

          <Card>
            <span className="ds-field-mini">Context chunks</span>
            <div className="flex flex-col gap-2">
              {contexts.map((ctx, i) => (
                <div key={i} style={{ display: "flex", gap: "0.5rem" }}>
                  <Textarea
                    value={ctx}
                    onChange={e => setContexts(prev => prev.map((c, idx) => idx === i ? e.target.value : c))}
                    rows={2}
                    placeholder={`Chunk ${i + 1}…`}
                    style={{ flex: 1 }}
                  />
                  {contexts.length > 1 && (
                    <button
                      className="ds-icon-button"
                      aria-label={`Remove chunk ${i + 1}`}
                      onClick={() => setContexts(prev => prev.filter((_, idx) => idx !== i))}
                    >
                      <Trash2 size={14} />
                    </button>
                  )}
                </div>
              ))}
            </div>
            <div className="button-row" style={{ marginTop: "0.75rem" }}>
              <Button variant="secondary" icon={<Plus size={14} />} onClick={() => setContexts(prev => [...prev, ""])}>
                Add chunk
              </Button>
            </div>
          </Card>

          <Card>
            <Field label="Model answer">
              <Textarea value={answer} onChange={e => setAnswer(e.target.value)} rows={2} placeholder="The response generated by your RAG system…" />
            </Field>
          </Card>

          <Card>
            <Field label="Expected answer (optional)">
              <Input value={expected} onChange={e => setExpected(e.target.value)} placeholder="Ground truth answer…" />
            </Field>
          </Card>

          <Card>
            <Field label="Scoring model">
              <Select value={embeddingModel} onChange={e => setEmbeddingModel(e.target.value)}>
                <option value="">Token overlap (no embedding model)</option>
                {Object.keys(embeddingModels).map((id) => (
                  <option key={id} value={id}>
                    {id}
                  </option>
                ))}
              </Select>
            </Field>
            <p className="micro-copy mt-2">
              Token overlap (default) compares shared words only — a correct answer phrased
              differently can score low. Pick an embedding model to score by meaning instead.
            </p>
          </Card>

          <Button icon={<Play size={14} />} loading={loading} onClick={evaluate}>
            {loading ? "Evaluating…" : "Evaluate RAG"}
          </Button>
        </div>

        {/* Right — results */}
        <div>
          {result ? (
            <Card>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
                <Badge tone={FAULT_TONE[result.fault_component] ?? "neutral"} mono>
                  fault: {result.fault_component}
                </Badge>
                <span style={{ width: 90 }}>
                  <ScoreBar score={result.overall_score} />
                </span>
              </div>
              <MetricRow label="Context Precision" score={result.context_precision} desc="Relevant chunks in context" />
              <MetricRow
                label="Context Recall"
                score={result.context_recall}
                desc="Answer covered by context"
                applicable={result.context_recall_applicable}
              />
              <MetricRow label="Faithfulness" score={result.faithfulness} desc="Answer grounded in context" />
              <MetricRow label="Answer Relevance" score={result.answer_relevance} desc="Answer addresses question" />
              <p className="micro-copy" style={{ marginTop: "0.75rem" }}>
                Scored via{" "}
                {result.scoring_mode === "embedding"
                  ? `embedding similarity (${result.embedding_model})`
                  : "token overlap"}
                .
              </p>
            </Card>
          ) : null}
          {result && result.details.context_precision?.chunk_scores && (
            <Card style={{ marginTop: "1rem" }}>
              <span className="ds-field-mini">Per-chunk precision</span>
              <p className="micro-copy" style={{ marginBottom: "0.6rem" }}>
                Which retrieved chunks the retriever should be credited or blamed for
                (relevant at {CHUNK_RELEVANCE_THRESHOLD.toFixed(1)}+).
              </p>
              <ChunkPrecisionBreakdown
                contexts={evaluatedContexts}
                scores={result.details.context_precision.chunk_scores}
              />
            </Card>
          )}
          {!result && (
            <Card style={{ height: "100%", minHeight: 300, display: "flex", alignItems: "center", justifyContent: "center" }}>
              <EmptyState
                icon={BookOpen}
                title="No evaluation yet"
                hint="Fill in the inputs and click Evaluate RAG to see component scores."
              />
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
