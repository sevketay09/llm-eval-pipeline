import type { ReactNode } from "react";
import type { LucideIcon } from "lucide-react";

interface EmptyStateProps {
  icon?: LucideIcon;
  title: string;
  hint?: ReactNode;
  action?: ReactNode;
}

export default function EmptyState({ icon: Icon, title, hint, action }: EmptyStateProps) {
  return (
    <div className="empty-state">
      {Icon && (
        <div className="metric-emblem" style={{ marginBottom: "0.9rem" }}>
          <Icon size={22} />
        </div>
      )}
      <p className="body-copy" style={{ fontWeight: 600 }}>
        {title}
      </p>
      {hint && (
        <p className="micro-copy" style={{ marginTop: "0.4rem", maxWidth: "24rem" }}>
          {hint}
        </p>
      )}
      {action && <div style={{ marginTop: "1rem" }}>{action}</div>}
    </div>
  );
}
