import clsx from "clsx";
import { scoreTone } from "./tone";

interface ScoreBarProps {
  // 0..1 score, or null for "not scored"
  score: number | null;
  // optional delta (0..1 fraction) shown after the value
  delta?: number;
}

export default function ScoreBar({ score, delta }: ScoreBarProps) {
  if (score === null || Number.isNaN(score)) {
    return <span className="micro-copy">—</span>;
  }
  const pct = Math.round(score * 100);
  const tone = scoreTone(score);
  return (
    <div className="ds-scorebar">
      <div className="ds-scorebar__track">
        <div className={clsx("ds-scorebar__fill", `is-${tone}`)} style={{ width: `${pct}%` }} />
      </div>
      <span className={clsx("ds-scorebar__value", `is-${tone}`)}>{pct}%</span>
      {delta != null && (
        <span
          className="ds-scorebar__value"
          style={{
            color:
              delta > 0
                ? "var(--success)"
                : delta < 0
                  ? "var(--danger)"
                  : "var(--text-dim)",
          }}
        >
          {delta > 0 ? "+" : ""}
          {Math.round(delta * 100)}%
        </span>
      )}
    </div>
  );
}
