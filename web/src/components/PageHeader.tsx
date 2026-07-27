import type { ReactNode } from "react";
import HelpHint from "./HelpHint";

interface PageHeaderProps {
  kicker?: string;
  title: string;
  subtitle?: string;
  actions?: ReactNode;
  help?: ReactNode;
}

export default function PageHeader({ kicker, title, subtitle, actions, help }: PageHeaderProps) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4">
      <div className="page-header">
        {kicker && <p className="page-kicker">{kicker}</p>}
        <h1 className="page-title">
          {title}
          {help && <HelpHint>{help}</HelpHint>}
        </h1>
        {subtitle && <p className="page-subtitle">{subtitle}</p>}
      </div>
      {actions && <div className="button-row">{actions}</div>}
    </div>
  );
}
