import { useEffect, useRef, useState, useCallback } from "react";

export interface EvalProgress {
  run_id: string;
  status: string;
  progress: number;
  current_model?: string;
  current_test?: string;
  message: string;
  started_at: string;
  elapsed_seconds: number;
  error_code?: string;
  error_stage?: string;
}

export function useEvalProgress(runId: string | null) {
  const [progress, setProgress] = useState<EvalProgress | null>(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!runId) return;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/progress/${runId}`);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as EvalProgress;
        setProgress(data);
        if (data.status === "completed" || data.status === "failed" || data.status === "cancelled") {
          ws.close();
        }
      } catch {
        // ignore parse errors
      }
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [runId]);

  const disconnect = useCallback(() => {
    wsRef.current?.close();
  }, []);

  return { progress, connected, disconnect };
}
