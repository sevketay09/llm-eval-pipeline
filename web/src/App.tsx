import { Routes, Route, NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  Play,
  BarChart3,
  Settings,
  MessageSquare,
  Database,
  Activity,
} from "lucide-react";
import Dashboard from "./pages/Dashboard";
import RunEvaluation from "./pages/RunEvaluation";
import Results from "./pages/Results";
import Models from "./pages/Models";
import HitlReview from "./pages/HitlReview";
import DatasetStudio from "./pages/DatasetStudio";
import Traces from "./pages/Traces";

const navItems = [
  { to: "/", icon: LayoutDashboard, label: "Dashboard" },
  { to: "/run", icon: Play, label: "Run" },
  { to: "/datasets", icon: Database, label: "Datasets" },
  { to: "/results", icon: BarChart3, label: "Results" },
  { to: "/traces", icon: Activity, label: "Traces" },
  { to: "/models", icon: Settings, label: "Models" },
  { to: "/review", icon: MessageSquare, label: "Review Desk" },
];

export default function App() {
  return (
    <div className="app-shell">
      <aside className="chrome-rail">
        <div className="brand-block">
          <span className="brand-kicker hidden lg:block">Signal Lab</span>
          <span className="brand-wordmark hidden lg:block">LLM Eval</span>
          <span className="brand-caption hidden lg:block">
            Pipeline observatory for runs, scorecards, model registry and human review.
          </span>
        </div>

        <nav className="chrome-nav">
          {navItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                `nav-link ${isActive ? "nav-link-active" : ""}`.trim()
              }
            >
              <Icon size={18} />
              <span className="hidden text-sm lg:block">{label}</span>
            </NavLink>
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
            <Route path="/models" element={<Models />} />
            <Route path="/review" element={<HitlReview />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}
