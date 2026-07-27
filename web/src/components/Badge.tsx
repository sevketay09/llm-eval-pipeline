import type { ReactNode } from "react";
import clsx from "clsx";

export type BadgeTone =
  | "success"
  | "warning"
  | "danger"
  | "info"
  | "violet"
  | "neutral";

interface BadgeProps {
  tone?: BadgeTone;
  mono?: boolean;
  children: ReactNode;
}

export default function Badge({ tone = "neutral", mono, children }: BadgeProps) {
  return (
    <span className={clsx("ds-badge", `ds-badge--${tone}`, mono && "ds-badge--mono")}>
      {children}
    </span>
  );
}
