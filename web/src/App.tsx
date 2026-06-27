import { useState, useEffect } from "react";
import { Routes, Route, NavLink, useLocation } from "react-router-dom";
import clsx from "clsx";
import { Menu, X, Search } from "lucide-react";
import { navGroups } from "./nav";
import { CommandPalette } from "./components";
import Dashboard from "./pages/Dashboard";
import RunEvaluation from "./pages/RunEvaluation";
import Results from "./pages/Results";
import Models from "./pages/Models";
import HitlReview from "./pages/HitlReview";
import DatasetStudio from "./pages/DatasetStudio";
import Traces from "./pages/Traces";
import Playground from "./pages/Playground";
import RedTeam from "./pages/RedTeam";
import CustomMetrics from "./pages/CustomMetrics";
import RagEval from "./pages/RagEval";
import FailureClustering from "./pages/FailureClustering";

export default function App() {
  const [navOpen, setNavOpen] = useState(false);
  const location = useLocation();

  // Close the mobile drawer on route change.
  useEffect(() => {
    setNavOpen(false);
  }, [location.pathname]);

  return (
    <div className="app-shell">
      <CommandPalette />
      <header className="mobile-topbar">
        <button
          type="button"
          className="ds-icon-button"
          aria-label="Open navigation"
          onClick={() => setNavOpen(true)}
        >
          <Menu size={20} />
        </button>
        <span className="brand-wordmark">LLM Eval</span>
      </header>

      {navOpen && <div className="nav-backdrop" onClick={() => setNavOpen(false)} />}

      <aside className={clsx("chrome-rail", navOpen && "is-open")}>
        <div className="brand-block">
          <button
            type="button"
            className="ds-icon-button md:hidden"
            aria-label="Close navigation"
            onClick={() => setNavOpen(false)}
            style={{ position: "absolute", top: "0.5rem", right: "0.5rem" }}
          >
            <X size={16} />
          </button>
          <span className="brand-kicker">Signal Lab</span>
          <span className="brand-wordmark">LLM Eval</span>
          <span className="brand-caption">
            Pipeline observatory for runs, scorecards, model registry and human review.
          </span>
        </div>

        <button
          type="button"
          className="cmdk-trigger"
          onClick={() => window.dispatchEvent(new Event("open-command-palette"))}
        >
          <Search size={15} />
          <span className="nav-text">Quick search</span>
          <kbd className="cmdk-kbd nav-text">⌘K</kbd>
        </button>

        <nav className="chrome-nav">
          {navGroups.map((group) => (
            <div key={group.label} className="nav-group">
              <span className="nav-group-label">{group.label}</span>
              {group.items.map(({ to, icon: Icon, label }) => (
                <NavLink
                  key={to}
                  to={to}
                  end={to === "/"}
                  title={label}
                  aria-label={label}
                  className={({ isActive }) =>
                    `nav-link ${isActive ? "nav-link-active" : ""}`.trim()
                  }
                >
                  <Icon size={18} />
                  <span className="nav-text text-sm">{label}</span>
                </NavLink>
              ))}
            </div>
          ))}
        </nav>
      </aside>

      <main className="page-viewport">
        <div className="page-stage">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/run" element={<RunEvaluation />} />
            <Route path="/datasets" element={<DatasetStudio />} />
            <Route path="/results" element={<Results />} />
            <Route path="/traces" element={<Traces />} />
            <Route path="/playground" element={<Playground />} />
            <Route path="/redteam" element={<RedTeam />} />
            <Route path="/custom-metrics" element={<CustomMetrics />} />
            <Route path="/rag-eval" element={<RagEval />} />
            <Route path="/failures" element={<FailureClustering />} />
            <Route path="/models" element={<Models />} />
            <Route path="/review" element={<HitlReview />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}
