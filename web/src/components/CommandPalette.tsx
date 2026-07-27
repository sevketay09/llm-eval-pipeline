import { useEffect, useMemo, useRef, useState, type KeyboardEvent as ReactKeyboardEvent } from "react";
import { useNavigate } from "react-router-dom";
import { Play, Search, BarChart3, CornerDownLeft } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { navItems } from "../nav";

interface Command {
  id: string;
  label: string;
  hint: string;
  icon: LucideIcon;
  run: () => void;
}

export default function CommandPalette() {
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const commands = useMemo<Command[]>(() => {
    const navCommands: Command[] = navItems.map((item) => ({
      id: `nav:${item.to}`,
      label: item.label,
      hint: "Go to page",
      icon: item.icon,
      run: () => navigate(item.to),
    }));
    const actions: Command[] = [
      {
        id: "action:run",
        label: "Run new evaluation",
        hint: "Action",
        icon: Play,
        run: () => navigate("/run"),
      },
      {
        id: "action:results",
        label: "Open latest report",
        hint: "Action",
        icon: BarChart3,
        run: () => navigate("/results"),
      },
    ];
    return [...actions, ...navCommands];
  }, [navigate]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return commands;
    return commands.filter(
      (c) => c.label.toLowerCase().includes(q) || c.hint.toLowerCase().includes(q)
    );
  }, [commands, query]);

  // Global Cmd/Ctrl+K toggles the palette.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((v) => !v);
      } else if (e.key === "Escape") {
        setOpen(false);
      }
    }
    function onOpen() {
      setOpen(true);
    }
    window.addEventListener("keydown", onKey);
    window.addEventListener("open-command-palette", onOpen);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("open-command-palette", onOpen);
    };
  }, []);

  // Reset transient state whenever the palette opens.
  useEffect(() => {
    if (open) {
      setQuery("");
      setActive(0);
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  useEffect(() => {
    setActive(0);
  }, [query]);

  if (!open) return null;

  function choose(index: number) {
    const cmd = filtered[index];
    if (!cmd) return;
    setOpen(false);
    cmd.run();
  }

  function onInputKey(e: ReactKeyboardEvent<HTMLInputElement>) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((a) => Math.min(a + 1, filtered.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((a) => Math.max(a - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      choose(active);
    }
  }

  return (
    <div className="cmdk-backdrop" onClick={() => setOpen(false)}>
      <div
        className="cmdk-panel"
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="cmdk-search">
          <Search size={16} className="cmdk-search-icon" />
          <input
            ref={inputRef}
            className="cmdk-input"
            placeholder="Jump to a page or action…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onInputKey}
          />
          <kbd className="cmdk-kbd">esc</kbd>
        </div>
        <div className="cmdk-list">
          {filtered.length === 0 && (
            <p className="cmdk-empty">No matches for “{query}”.</p>
          )}
          {filtered.map((cmd, index) => {
            const Icon = cmd.icon;
            return (
              <button
                key={cmd.id}
                type="button"
                className={`cmdk-item ${index === active ? "is-active" : ""}`}
                onMouseEnter={() => setActive(index)}
                onClick={() => choose(index)}
              >
                <Icon size={16} className="cmdk-item-icon" />
                <span className="cmdk-item-label">{cmd.label}</span>
                <span className="cmdk-item-hint">{cmd.hint}</span>
                {index === active && <CornerDownLeft size={13} className="cmdk-item-enter" />}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
