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
import { tracesApi, Trace, TraceSpan, TraceDetail } from "../api/client";

// ── Span type color/label map ─────────────────────────────────────────────────

const SPAN_TYPE_STYLES: Record<string, { bg: string; text: string; label: string }> = {
  LLM:       { bg: "#7c3aed22", text: "#a78bfa", label: "LLM" },
  TOOL:      { bg: "#0369a122", text: "#38bdf8", label: "TOOL" },
  RETRIEVER: { bg: "#06645322", text: "#34d399", label: "RETR" },
  AGENT:     { bg: "#b4500622", text: "#fb923c", label: "AGNT" },
  GENERIC:   { bg: "#37415122", text: "#9ca3af", label: "GEN" },
};

function spanStyle(type: string) {
  return SPAN_TYPE_STYLES[type.toUpperCase()] ?? SPAN_TYPE_STYLES.GENERIC;
}

function fmt(ms: number) {
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

// ── Span row ──────────────────────────────────────────────────────────────────

interface SpanRowProps {
  span: TraceSpan;
  depth: number;
  isLast: boolean;
  onShowRaw: (span: TraceSpan) => void;
}

function SpanRow({ span, depth, isLast: _isLast, onShowRaw }: SpanRowProps) {
  const [expanded, setExpanded] = useState(false);
  const st = spanStyle(span.type);
  const hasFailed =
    typeof span.output === "object" &&
    span.output !== null &&
    "error" in (span.output as Record<string, unknown>);

  return (
    <div style={{ marginLeft: depth * 20 }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "6px 8px",
          borderRadius: 6,
          background: hasFailed ? "#7f1d1d22" : "transparent",
          borderLeft: hasFailed ? "2px solid #ef4444" : "2px solid transparent",
          cursor: "pointer",
          userSelect: "none",
        }}
        onClick={() => setExpanded((p) => !p)}
      >
        <span style={{ color: "#6b7280", width: 14, flexShrink: 0 }}>
          {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </span>

        {/* Type badge */}
        <span
          style={{
            background: st.bg,
            color: st.text,
            fontSize: 10,
            fontWeight: 700,
            padding: "1px 6px",
            borderRadius: 4,
            fontFamily: "monospace",
            flexShrink: 0,
          }}
        >
          {st.label}
        </span>

        <span style={{ flex: 1, fontSize: 13, color: "#e5e7eb", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {span.name}
        </span>

        {hasFailed && <AlertCircle size={14} style={{ color: "#ef4444", flexShrink: 0 }} />}

        <span style={{ fontSize: 11, color: "#6b7280", flexShrink: 0 }}>
          {fmt(span.latency_ms)}
        </span>

        <button
          onClick={(e) => { e.stopPropagation(); onShowRaw(span); }}
          style={{ background: "none", border: "none", cursor: "pointer", color: "#4b5563", padding: 0 }}
          title="View raw"
        >
          <Code size={13} />
        </button>
      </div>

      {expanded && (
        <div
          style={{
            marginLeft: 22,
            marginBottom: 4,
            padding: "8px 10px",
            background: "#111827",
            borderRadius: 6,
            fontSize: 12,
            color: "#9ca3af",
          }}
        >
          {span.input != null && (
            <div style={{ marginBottom: 6 }}>
              <span style={{ color: "#6b7280", fontSize: 11 }}>INPUT </span>
              <span style={{ color: "#d1d5db" }}>{JSON.stringify(span.input).slice(0, 300)}</span>
            </div>
          )}
          {span.output != null && (
            <div>
              <span style={{ color: "#6b7280", fontSize: 11 }}>OUTPUT </span>
              <span style={{ color: hasFailed ? "#fca5a5" : "#d1d5db" }}>
                {JSON.stringify(span.output).slice(0, 300)}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Span tree builder ─────────────────────────────────────────────────────────

function buildTree(spans: TraceSpan[]): Array<{ span: TraceSpan; depth: number }> {
  const depthMap = new Map<string, number>();

  function getDepth(span: TraceSpan): number {
    if (depthMap.has(span.span_id)) return depthMap.get(span.span_id)!;
    if (!span.parent_span_id) {
      depthMap.set(span.span_id, 0);
      return 0;
    }
    const parent = spans.find((s) => s.span_id === span.parent_span_id);
    const d = parent ? getDepth(parent) + 1 : 0;
    depthMap.set(span.span_id, d);
    return d;
  }

  return spans
    .slice()
    .sort((a, b) => a.start_ts - b.start_ts)
    .map((span) => ({ span, depth: getDepth(span) }));
}

// ── Trace detail panel ────────────────────────────────────────────────────────

interface DetailPanelProps {
  detail: TraceDetail;
  onEval: (traceId: string) => void;
  evalStatus: Record<string, "queued" | "error">;
}

function DetailPanel({ detail, onEval, evalStatus }: DetailPanelProps) {
  const [rawSpan, setRawSpan] = useState<TraceSpan | null>(null);
  const { trace, span_count, duration_ms } = detail;
  const tree = buildTree(trace.spans);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", gap: 0 }}>
      {/* Header */}
      <div
        style={{
          padding: "12px 16px",
          borderBottom: "1px solid #1f2937",
          display: "flex",
          alignItems: "center",
          gap: 10,
          flexShrink: 0,
        }}
      >
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: "#f9fafb", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {trace.name}
          </div>
          <div style={{ fontSize: 11, color: "#6b7280", fontFamily: "monospace", marginTop: 2 }}>
            {trace.trace_id}
          </div>
        </div>

        <div style={{ display: "flex", gap: 6, flexShrink: 0, alignItems: "center" }}>
          <span style={{ fontSize: 11, color: "#6b7280", display: "flex", alignItems: "center", gap: 3 }}>
            <Layers size={12} />{span_count}
          </span>
          {duration_ms != null && (
            <span style={{ fontSize: 11, color: "#6b7280", display: "flex", alignItems: "center", gap: 3 }}>
              <Clock size={12} />{fmt(duration_ms)}
            </span>
          )}
          {trace.tags.length > 0 && trace.tags.map((tag) => (
            <span
              key={tag}
              style={{
                fontSize: 10,
                padding: "1px 6px",
                background: "#1e3a5f",
                color: "#60a5fa",
                borderRadius: 4,
              }}
            >
              {tag}
            </span>
          ))}

          <button
            onClick={() => onEval(trace.trace_id)}
            disabled={evalStatus[trace.trace_id] === "queued"}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 4,
              padding: "4px 10px",
              background: evalStatus[trace.trace_id] === "queued" ? "#166534" : "#7c3aed",
              color: "#fff",
              border: "none",
              borderRadius: 6,
              fontSize: 12,
              cursor: evalStatus[trace.trace_id] === "queued" ? "default" : "pointer",
              fontWeight: 600,
            }}
          >
            {evalStatus[trace.trace_id] === "queued" ? (
              <><CheckCircle size={12} /> Queued</>
            ) : (
              <><Play size={12} /> Eval</>
            )}
          </button>
        </div>
      </div>

      {/* Span tree */}
      <div style={{ flex: 1, overflowY: "auto", padding: "8px 12px" }}>
        {tree.length === 0 ? (
          <div style={{ color: "#4b5563", fontSize: 13, textAlign: "center", marginTop: 40 }}>
            No spans recorded
          </div>
        ) : (
          tree.map(({ span, depth }, idx) => (
            <SpanRow
              key={span.span_id}
              span={span}
              depth={depth}
              isLast={idx === tree.length - 1}
              onShowRaw={setRawSpan}
            />
          ))
        )}
      </div>

      {/* Raw payload drawer */}
      {rawSpan && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            background: "#00000088",
            zIndex: 50,
            display: "flex",
            alignItems: "flex-end",
          }}
          onClick={() => setRawSpan(null)}
        >
          <div
            style={{
              background: "#111827",
              width: "100%",
              maxHeight: "60%",
              borderTopLeftRadius: 12,
              borderTopRightRadius: 12,
              padding: 16,
              overflowY: "auto",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10, alignItems: "center" }}>
              <span style={{ fontSize: 13, fontWeight: 600, color: "#e5e7eb" }}>
                Raw — {rawSpan.name}
              </span>
              <button
                onClick={() => setRawSpan(null)}
                style={{ background: "none", border: "none", cursor: "pointer", color: "#6b7280" }}
              >
                <X size={16} />
              </button>
            </div>
            <pre
              style={{
                fontSize: 11,
                color: "#9ca3af",
                overflowX: "auto",
                whiteSpace: "pre-wrap",
                wordBreak: "break-word",
              }}
            >
              {JSON.stringify(rawSpan, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Trace list item ───────────────────────────────────────────────────────────

function TraceListItem({
  trace,
  selected,
  onClick,
}: {
  trace: Trace;
  selected: boolean;
  onClick: () => void;
}) {
  const failedSpans = trace.spans.filter(
    (s) =>
      typeof s.output === "object" &&
      s.output !== null &&
      "error" in (s.output as Record<string, unknown>)
  ).length;

  const duration =
    trace.end_ts != null ? (trace.end_ts - trace.start_ts) * 1000 : null;

  return (
    <div
      onClick={onClick}
      style={{
        padding: "10px 14px",
        borderBottom: "1px solid #1f2937",
        cursor: "pointer",
        background: selected ? "#1e293b" : "transparent",
        borderLeft: selected ? "2px solid #7c3aed" : "2px solid transparent",
        transition: "background 0.1s",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <span
          style={{
            fontSize: 13,
            fontWeight: 600,
            color: "#e5e7eb",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            flex: 1,
          }}
        >
          {trace.name}
        </span>
        {failedSpans > 0 && (
          <span style={{ fontSize: 11, color: "#ef4444", flexShrink: 0, marginLeft: 6, display: "flex", alignItems: "center", gap: 2 }}>
            <AlertCircle size={11} /> {failedSpans}
          </span>
        )}
      </div>

      <div style={{ fontSize: 11, color: "#6b7280", fontFamily: "monospace", marginTop: 2, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
        {trace.trace_id.slice(0, 16)}…
      </div>

      <div style={{ display: "flex", gap: 8, marginTop: 4, alignItems: "center", flexWrap: "wrap" }}>
        <span style={{ fontSize: 11, color: "#4b5563", display: "flex", alignItems: "center", gap: 2 }}>
          <Layers size={11} /> {trace.spans.length}
        </span>
        {duration != null && (
          <span style={{ fontSize: 11, color: "#4b5563", display: "flex", alignItems: "center", gap: 2 }}>
            <Clock size={11} /> {fmt(duration)}
          </span>
        )}
        {trace.tags.slice(0, 2).map((tag) => (
          <span key={tag} style={{ fontSize: 10, color: "#60a5fa", display: "flex", alignItems: "center", gap: 2 }}>
            <Tag size={10} /> {tag}
          </span>
        ))}
      </div>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function Traces() {
  const [traces, setTraces] = useState<Trace[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<TraceDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [tagFilter, setTagFilter] = useState("");
  const [evalStatus, setEvalStatus] = useState<Record<string, "queued" | "error">>({});
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const loadTraces = useCallback(async () => {
    try {
      const res = await tracesApi.list({ limit: 100, tag: tagFilter || undefined });
      setTraces(res.traces);
    } catch {
      // silent — list will stay stale
    }
  }, [tagFilter]);

  // initial load + tag change
  useEffect(() => {
    setLoading(true);
    loadTraces().finally(() => setLoading(false));
  }, [loadTraces]);

  // auto-refresh
  useEffect(() => {
    if (!autoRefresh) {
      if (intervalRef.current) clearInterval(intervalRef.current);
      return;
    }
    intervalRef.current = setInterval(loadTraces, 4000);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [autoRefresh, loadTraces]);

  // load detail when selection changes
  useEffect(() => {
    if (!selectedId) { setDetail(null); return; }
    tracesApi.get(selectedId).then(setDetail).catch(() => setDetail(null));
  }, [selectedId]);

  const handleEval = async (traceId: string) => {
    try {
      await tracesApi.eval(traceId);
      setEvalStatus((prev) => ({ ...prev, [traceId]: "queued" }));
    } catch {
      setEvalStatus((prev) => ({ ...prev, [traceId]: "error" }));
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", color: "#e5e7eb" }}>
      {/* Toolbar */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          padding: "12px 16px",
          borderBottom: "1px solid #1f2937",
          flexShrink: 0,
        }}
      >
        <Activity size={18} style={{ color: "#7c3aed" }} />
        <span style={{ fontWeight: 700, fontSize: 15 }}>Live Traces</span>

        <input
          placeholder="filter by tag…"
          value={tagFilter}
          onChange={(e) => setTagFilter(e.target.value)}
          style={{
            background: "#111827",
            border: "1px solid #374151",
            borderRadius: 6,
            padding: "4px 10px",
            fontSize: 12,
            color: "#e5e7eb",
            width: 160,
            outline: "none",
          }}
        />

        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 12, color: "#4b5563" }}>{traces.length} traces</span>

          <button
            onClick={loadTraces}
            style={{
              background: "#1f2937",
              border: "1px solid #374151",
              borderRadius: 6,
              padding: "4px 8px",
              cursor: "pointer",
              color: "#9ca3af",
              display: "flex",
              alignItems: "center",
              gap: 4,
              fontSize: 12,
            }}
          >
            <RefreshCw size={13} />
          </button>

          <button
            onClick={() => setAutoRefresh((p) => !p)}
            style={{
              background: autoRefresh ? "#166534" : "#1f2937",
              border: `1px solid ${autoRefresh ? "#15803d" : "#374151"}`,
              borderRadius: 6,
              padding: "4px 10px",
              cursor: "pointer",
              color: autoRefresh ? "#4ade80" : "#6b7280",
              fontSize: 12,
              fontWeight: 600,
            }}
          >
            {autoRefresh ? "● LIVE" : "○ PAUSED"}
          </button>
        </div>
      </div>

      {/* Body: list + detail */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
        {/* Trace list */}
        <div
          style={{
            width: 280,
            flexShrink: 0,
            borderRight: "1px solid #1f2937",
            overflowY: "auto",
          }}
        >
          {loading && traces.length === 0 ? (
            <div style={{ padding: 24, color: "#4b5563", fontSize: 13, textAlign: "center" }}>
              Loading…
            </div>
          ) : traces.length === 0 ? (
            <div style={{ padding: 24, color: "#4b5563", fontSize: 13, textAlign: "center" }}>
              <Activity size={32} style={{ margin: "0 auto 12px", opacity: 0.3 }} />
              No traces yet.
              <br />
              Instrument your app with
              <br />
              <code style={{ fontSize: 11 }}>@eval.trace</code>
            </div>
          ) : (
            traces.map((t) => (
              <TraceListItem
                key={t.trace_id}
                trace={t}
                selected={t.trace_id === selectedId}
                onClick={() => setSelectedId(t.trace_id)}
              />
            ))
          )}
        </div>

        {/* Detail panel */}
        <div style={{ flex: 1, overflow: "hidden", position: "relative" }}>
          {detail ? (
            <DetailPanel detail={detail} onEval={handleEval} evalStatus={evalStatus} />
          ) : (
            <div
              style={{
                height: "100%",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "#374151",
                flexDirection: "column",
                gap: 10,
              }}
            >
              <Activity size={40} style={{ opacity: 0.3 }} />
              <span style={{ fontSize: 13 }}>Select a trace to inspect spans</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
