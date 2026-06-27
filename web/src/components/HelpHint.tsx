import { useEffect, useRef, useState, type ReactNode } from "react";
import { HelpCircle } from "lucide-react";

interface HelpHintProps {
  children: ReactNode;
  label?: string;
}

export default function HelpHint({ children, label = "What is this?" }: HelpHintProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onDown(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div className="help-hint" ref={ref}>
      <button
        type="button"
        className="ds-icon-button help-hint-trigger"
        aria-label={label}
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        <HelpCircle size={16} />
      </button>
      {open && <div className="help-hint-popover">{children}</div>}
    </div>
  );
}
