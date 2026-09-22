import { useState, useEffect, useCallback, useRef } from "react";
import {
  Activity,
  RefreshCw,
  ChevronRight,
  ChevronDown,
  Play,
  Tag,
  Clock,
  Layers,
  AlertCircle,
  CheckCircle,
  Code,
  X,
} from "lucide-react";
import { tracesApi, Trace, TraceSpan, TraceDetail } from "@/api/client";
import { PageHeader, Card, Button, Badge, EmptyState, Input, Select, DecisionModelSelect, useToast } from "@/components";
import type { BadgeTone } from "@/components";

const TAG_PRESETS = [
  { value: "", label: "All" },
  { value: "needs_review", label: "Needs review" },
  { value: "risk:pii", label: "risk:pii" },
  { value: "risk:injection", label: "risk:injection" },
  { value: "risk:harmful", label: "risk:harmful" },
  { value: "risk:regulatory", label: "risk:regulatory" },
  { value: "risk:off_topic", label: "risk:off_topic" },
  { value: "quality:low", label: "quality:low" },
];

function tagTone(tag: string): BadgeTone {
  if (tag.startsWith("risk:")) return "danger";
  if (tag === "quality:low") return "warning";
  if (tag === "needs_review") return "warning";
  return "info";
}

const SPAN_TONE: Record<string, { tone: BadgeTone; label: string }> = {
  LLM: { tone: "violet", label: "LLM" },
  TOOL: { tone: "info", label: "TOOL" },
  RETRIEVER: { tone: "success", label: "RETR" },
  AGENT: { tone: "warning", label: "AGNT" },
  GENERIC: { tone: "neutral", label: "GEN" },
};

function spanTone(type: string) {
  return SPAN_TONE[type.toUpperCase()] ?? SPAN_TONE["GENERIC"]!;
}

function fmt(ms: number) {
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

function spanFailed(span: TraceSpan): boolean {
  return (
    typeof span.output === "object" &&
    span.output !== null &&
    "error" in (span.output as Record<string, unknown>)
  );
}

function SpanRow({ span, depth, onShowRaw }: { span: TraceSpan; depth: number; onShowRaw: (s: TraceSpan) => void }) {
  const [expanded, setExpanded] = useState(false);
  const st = spanTone(span.type);
  const hasFailed = spanFailed(span);

  return (
    <div style={{ marginLeft: depth * 20 }}>
      <div
        className={`ds-row-expandable ${hasFailed ? "ds-row-failed" : ""}`}
        style={{ display: "flex", alignItems: "center", gap: "0.5rem", padding: "0.35rem 0.5rem", borderRadius: 8 }}
        onClick={() => setExpanded(p => !p)}
        onKeyDown={e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setExpanded(p => !p); } }}
        role="button"
        tabIndex={0}
        aria-expanded={expanded}
        aria-label={`Span ${span.name}`}
      >
        <span style={{ color: "var(--text-dim)", width: 14, flexShrink: 0 }}>
          {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </span>
        <Badge tone={st.tone} mono>{st.label}</Badge>
        <span className="body-copy" style={{ flex: 1, fontSize: "0.85rem", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {span.name}
        </span>
        {hasFailed && <AlertCircle size={14} style={{ color: "var(--danger)", flexShrink: 0 }} />}
        <span className="micro-copy" style={{ flexShrink: 0 }}>{fmt(span.latency_ms)}</span>
        <button className="ds-icon-button" aria-label="View raw" onClick={e => { e.stopPropagation(); onShowRaw(span); }}>
          <Code size={13} />
        </button>
      </div>

      {expanded && (
        <div style={{ marginLeft: 22, marginBottom: 4 }}>
          {span.input != null && (
            <div style={{ marginBottom: "0.4rem" }}>
              <span className="ds-field-mini" style={{ display: "inline", marginRight: "0.4rem" }}>Input</span>
              <span className="micro-copy">{JSON.stringify(span.input).slice(0, 300)}</span>
            </div>
          )}
          {span.output != null && (
            <div>
              <span className="ds-field-mini" style={{ display: "inline", marginRight: "0.4rem" }}>Output</span>
              <span className="micro-copy" style={{ color: hasFailed ? "var(--danger)" : undefined }}>
                {JSON.stringify(span.output).slice(0, 300)}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function buildTree(spans: TraceSpan[]): Array<{ span: TraceSpan; depth: number }> {
  const depthMap = new Map<string, number>();
  function getDepth(span: TraceSpan): number {
    if (depthMap.has(span.span_id)) return depthMap.get(span.span_id)!;
    if (!span.parent_span_id) { depthMap.set(span.span_id, 0); return 0; }
    const parent = spans.find(s => s.span_id === span.parent_span_id);
    const d = parent ? getDepth(parent) + 1 : 0;
    depthMap.set(span.span_id, d);
    return d;
  }
  return spans.slice().sort((a, b) => a.start_ts - b.start_ts).map(span => ({ span, depth: getDepth(span) }));
}

function DetailPanel({ detail, onEval, evalStatus, onSendToHitl }: {
  detail: TraceDetail;
  onEval: (id: string) => void;
  evalStatus: Record<string, "queued" | "error">;
  onSendToHitl: (id: string) => void;
}) {
  const [rawSpan, setRawSpan] = useState<TraceSpan | null>(null);
  const { trace, span_count, duration_ms } = detail;
  const tree = buildTree(trace.spans);
  const queued = evalStatus[trace.trace_id] === "queued";
  const decisions = (trace.metadata?.decisions ?? null) as Record<string, number | string> | null;
  const needsReview = trace.tags.includes("needs_review");

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", position: "relative" }}>
      <div style={{ padding: "0.9rem 1.1rem", borderBottom: "1px solid var(--line)", display: "flex", alignItems: "center", gap: "0.7rem" }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div className="body-copy" style={{ fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{trace.name}</div>
          <div className="micro-copy font-mono">{trace.trace_id}</div>
        </div>
        <div style={{ display: "flex", gap: "0.5rem", flexShrink: 0, alignItems: "center", flexWrap: "wrap" }}>
          <span className="micro-copy" style={{ display: "flex", alignItems: "center", gap: 3 }}><Layers size={12} />{span_count}</span>
          {duration_ms != null && <span className="micro-copy" style={{ display: "flex", alignItems: "center", gap: 3 }}><Clock size={12} />{fmt(duration_ms)}</span>}
          {trace.tags.map(tag => <Badge key={tag} tone={tagTone(tag)}>{tag}</Badge>)}
          <Button
            icon={queued ? <CheckCircle size={12} /> : <Play size={12} />}
            disabled={queued}
            onClick={() => onEval(trace.trace_id)}
          >
            {queued ? "Queued" : "Eval"}
          </Button>
          {needsReview && (
            <Button variant="secondary" onClick={() => onSendToHitl(trace.trace_id)}>Send to HITL</Button>
          )}
        </div>
      </div>

      {decisions && (
        <div style={{ padding: "0.6rem 1.1rem", borderBottom: "1px solid var(--line)" }}>
          <span className="ds-field-mini">Signals (Jev)</span>
          <div className="mt-1 flex flex-wrap gap-2">
            {Object.entries(decisions).map(([name, value]) => (
              <span key={name} className="micro-copy">
                {name}: {typeof value === "number" ? value.toFixed(2) : String(value)}
              </span>
            ))}
          </div>
        </div>
      )}

      <div style={{ flex: 1, overflowY: "auto", padding: "0.5rem 0.75rem" }}>
        {tree.length === 0 ? (
          <EmptyState icon={Layers} title="No spans recorded" />
        ) : (
          tree.map(({ span, depth }) => <SpanRow key={span.span_id} span={span} depth={depth} onShowRaw={setRawSpan} />)
        )}
      </div>

      {rawSpan && (
        <div style={{ position: "absolute", inset: 0, background: "rgba(0,0,0,0.55)", zIndex: 50, display: "flex", alignItems: "flex-end" }} onClick={() => setRawSpan(null)}>
          <div className="panel-surface" style={{ width: "100%", maxHeight: "60%", borderRadius: "18px 18px 0 0", overflowY: "auto" }} onClick={e => e.stopPropagation()}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.7rem", alignItems: "center" }}>
              <span className="body-copy" style={{ fontWeight: 600 }}>Raw — {rawSpan.name}</span>
              <button className="ds-icon-button" aria-label="Close" onClick={() => setRawSpan(null)}><X size={16} /></button>
            </div>
            <pre className="ds-codeblock">{JSON.stringify(rawSpan, null, 2)}</pre>
          </div>
        </div>
      )}
    </div>
  );
}

function TraceListItem({ trace, selected, onClick }: { trace: Trace; selected: boolean; onClick: () => void }) {
  const failedSpans = trace.spans.filter(spanFailed).length;
  const duration = trace.end_ts != null ? (trace.end_ts - trace.start_ts) * 1000 : null;
  return (
    <div
      onClick={onClick}
      onKeyDown={e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onClick(); } }}
      role="button"
      tabIndex={0}
      aria-pressed={selected}
      aria-label={`Trace ${trace.name ?? trace.trace_id}`}
      className="ds-row-expandable"
      style={{
        padding: "0.65rem 0.9rem",
        borderBottom: "1px solid var(--line)",
        background: selected ? "rgba(255,255,255,0.04)" : "transparent",
        borderLeft: selected ? "2px solid var(--accent)" : "2px solid transparent",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <span className="body-copy" style={{ fontWeight: 600, fontSize: "0.85rem", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1 }}>{trace.name}</span>
        {failedSpans > 0 && (
          <span style={{ fontSize: "0.72rem", color: "var(--danger)", flexShrink: 0, marginLeft: 6, display: "flex", alignItems: "center", gap: 2 }}>
            <AlertCircle size={11} /> {failedSpans}
          </span>
        )}
      </div>
      <div className="micro-copy font-mono" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{trace.trace_id.slice(0, 16)}…</div>
      <div style={{ display: "flex", gap: "0.6rem", marginTop: "0.3rem", alignItems: "center", flexWrap: "wrap" }}>
        <span className="micro-copy" style={{ display: "flex", alignItems: "center", gap: 2 }}><Layers size={11} /> {trace.spans.length}</span>
        {duration != null && <span className="micro-copy" style={{ display: "flex", alignItems: "center", gap: 2 }}><Clock size={11} /> {fmt(duration)}</span>}
        {trace.tags.slice(0, 2).map(tag => (
          <span
            key={tag}
            className="micro-copy"
            style={{ display: "flex", alignItems: "center", gap: 2, color: tag.startsWith("risk:") || tag === "needs_review" || tag === "quality:low" ? "var(--danger)" : "var(--accent-cool)" }}
          >
            <Tag size={10} /> {tag}
          </span>
        ))}
      </div>
    </div>
  );
}

export default function Traces() {
  const [traces, setTraces] = useState<Trace[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<TraceDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [tagFilter, setTagFilter] = useState("");
  const [evalStatus, setEvalStatus] = useState<Record<string, "queued" | "error">>({});
  const [decisionModel, setDecisionModel] = useState("");
  const [runningJev, setRunningJev] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const toast = useToast();

  const loadTraces = useCallback(async () => {
    try {
      const res = await tracesApi.list({ limit: 100, tag: tagFilter || undefined });
      setTraces(res.traces);
    } catch {
      // silent — list stays stale
    }
  }, [tagFilter]);

  useEffect(() => {
    setLoading(true);
    loadTraces().finally(() => setLoading(false));
  }, [loadTraces]);

  useEffect(() => {
    if (!autoRefresh) {
      if (intervalRef.current) clearInterval(intervalRef.current);
      return;
    }
    intervalRef.current = setInterval(loadTraces, 4000);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [autoRefresh, loadTraces]);

  useEffect(() => {
    if (!selectedId) { setDetail(null); return; }
    tracesApi.get(selectedId).then(setDetail).catch(() => setDetail(null));
  }, [selectedId]);

  const handleEval = async (traceId: string) => {
    try {
      await tracesApi.eval(traceId);
      setEvalStatus(prev => ({ ...prev, [traceId]: "queued" }));
    } catch {
      setEvalStatus(prev => ({ ...prev, [traceId]: "error" }));
    }
  };

  const handleSendToHitl = async (traceId: string) => {
    try {
      await tracesApi.toHitl(traceId);
      toast.success("Sent to HITL review queue.");
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : "Failed to send to HITL");
    }
  };

  const handleRunJevOnVisible = async () => {
    if (!decisionModel) { toast.error("Select a Jev model first."); return; }
    setRunningJev(true);
    try {
      const ids = traces.map(t => t.trace_id);
      const result = await tracesApi.decideBatch(decisionModel, ids);
      toast.success(`Jev decided ${result.decided} trace(s), flagged ${result.flagged}.`);
      await loadTraces();
      if (selectedId) tracesApi.get(selectedId).then(setDetail).catch(() => {});
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : "Jev run failed");
    } finally {
      setRunningJev(false);
    }
  };

  return (
    <div className="page-shell">
      <PageHeader
        kicker="Live Traces"
        title="Trace Observatory"
        subtitle="Stream live evaluation traces, inspect the span tree, and queue any trace for scoring."
        actions={
          <>
            <Select value={tagFilter} onChange={e => setTagFilter(e.target.value)} style={{ width: 150 }}>
              {TAG_PRESETS.map(p => <option key={p.value} value={p.value}>{p.label}</option>)}
            </Select>
            <Input placeholder="or filter by tag…" value={tagFilter} onChange={e => setTagFilter(e.target.value)} style={{ width: 150 }} />
            <span className="micro-copy">{traces.length} traces</span>
            <button className="ds-icon-button" aria-label="Refresh" onClick={loadTraces}><RefreshCw size={14} /></button>
            <Button variant={autoRefresh ? "primary" : "secondary"} onClick={() => setAutoRefresh(p => !p)}>
              {autoRefresh ? "● LIVE" : "○ PAUSED"}
            </Button>
          </>
        }
      />

      <Card>
        <div style={{ display: "flex", gap: "0.75rem", alignItems: "flex-end", flexWrap: "wrap" }}>
          <div style={{ minWidth: 220 }}>
            <DecisionModelSelect value={decisionModel} onChange={setDecisionModel} label="Jev model" />
          </div>
          <Button loading={runningJev} onClick={handleRunJevOnVisible}>
            Run Jev on visible traces
          </Button>
        </div>
      </Card>

      <div className="grid gap-4 lg:grid-cols-[300px_1fr] items-start">
        {/* List */}
        <Card style={{ padding: 0, overflow: "hidden", maxHeight: "72vh", display: "flex", flexDirection: "column" }}>
          <div style={{ overflowY: "auto" }}>
            {loading && traces.length === 0 ? (
              <div style={{ padding: "1.5rem" }}><EmptyState title="Loading…" /></div>
            ) : traces.length === 0 ? (
              <div style={{ padding: "1.5rem" }}>
                <EmptyState icon={Activity} title="No traces yet" hint={<>Instrument your app with <code className="font-mono">@eval.trace</code></>} />
              </div>
            ) : (
              traces.map(t => (
                <TraceListItem key={t.trace_id} trace={t} selected={t.trace_id === selectedId} onClick={() => setSelectedId(t.trace_id)} />
              ))
            )}
          </div>
        </Card>

        {/* Detail */}
        <Card style={{ padding: 0, overflow: "hidden", height: "72vh" }}>
          {detail ? (
            <DetailPanel detail={detail} onEval={handleEval} evalStatus={evalStatus} onSendToHitl={handleSendToHitl} />
          ) : (
            <div style={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <EmptyState icon={Activity} title="Select a trace to inspect spans" />
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
