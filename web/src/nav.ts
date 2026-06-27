import {
  LayoutDashboard,
  Play,
  BarChart3,
  Settings,
  MessageSquare,
  Database,
  Activity,
  FlaskConical,
  ShieldAlert,
  Sparkles,
  BookOpen,
  AlertTriangle,
  type LucideIcon,
} from "lucide-react";

export interface NavItem {
  to: string;
  icon: LucideIcon;
  label: string;
}

export interface NavGroup {
  label: string;
  items: NavItem[];
}

export const navGroups: NavGroup[] = [
  {
    label: "Evaluate",
    items: [
      { to: "/", icon: LayoutDashboard, label: "Dashboard" },
      { to: "/run", icon: Play, label: "Run" },
      { to: "/datasets", icon: Database, label: "Datasets" },
      { to: "/playground", icon: FlaskConical, label: "Playground" },
    ],
  },
  {
    label: "Analyze",
    items: [
      { to: "/results", icon: BarChart3, label: "Results" },
      { to: "/traces", icon: Activity, label: "Traces" },
      { to: "/failures", icon: AlertTriangle, label: "Failures" },
      { to: "/rag-eval", icon: BookOpen, label: "RAG Eval" },
      { to: "/custom-metrics", icon: Sparkles, label: "Metrics" },
      { to: "/redteam", icon: ShieldAlert, label: "Red-Team" },
    ],
  },
  {
    label: "Configure",
    items: [
      { to: "/models", icon: Settings, label: "Models" },
      { to: "/review", icon: MessageSquare, label: "Review Desk" },
    ],
  },
];

export const navItems: NavItem[] = navGroups.flatMap((group) => group.items);
