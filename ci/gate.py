"""CI gate core.

Evaluates an evaluation report against a gate configuration and (optionally)
a baseline report, producing a structured pass/fail decision plus renderers
for CI consumption.

Report contract (produced by ``pipeline_runner`` / ``main.py``):
    report["models"][model_key]["overall_metrics"]["weighted_score"]
    report["models"][model_key]["overall_metrics"]["latency_p95"]
    report["models"][model_key]["tests"][test]["summary"]["overall_score"]
    report["models"][model_key]["tests"][test]["error"]   # present on failure

Gate config (``config/ci_gate.yaml`` by default)::

    gate:
      weighted_score_min: 0.70
      max_latency_p95: 10.0
      fail_on_test_error: true
      tests:
        function_calling: 0.80
        turkish_grammar: 0.75
      regression:
        max_weighted_drop: 0.03
        max_test_drop: 0.05

CLI::

    python -m ci.gate REPORT.json [--config config/ci_gate.yaml]
        [--baseline BASE.json] [--format text|json|markdown|badge]
        [--output FILE]

Exit code 0 = gate passed, 1 = gate failed, 2 = usage/IO error.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a project dependency
    yaml = None


DEFAULT_CONFIG_PATH = "config/ci_gate.yaml"

# Check kinds
KIND_MIN = "min"          # value must be >= threshold
KIND_MAX = "max"          # value must be <= threshold
KIND_ERROR = "error"      # test execution error (boolean fail)
KIND_REGRESSION = "regression"  # drop vs baseline must not exceed threshold


@dataclass
class CheckResult:
    """Single gate check for a model."""
    name: str                       # e.g. "weighted_score", "test:function_calling"
    kind: str                       # KIND_*
    value: Optional[float]
    threshold: Optional[float]
    passed: bool
    baseline_value: Optional[float] = None
    delta: Optional[float] = None
    detail: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelGateResult:
    model_key: str
    passed: bool = True
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
        }


@dataclass
class GateReport:
    passed: bool
    models: list[ModelGateResult] = field(default_factory=list)
    baseline_used: Optional[str] = None
    report_path: Optional[str] = None

    @property
    def passed_models(self) -> list[str]:
        return [m.model_key for m in self.models if m.passed]

    @property
    def failed_models(self) -> list[str]:
        return [m.model_key for m in self.models if not m.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "report_path": self.report_path,
            "baseline_used": self.baseline_used,
            "passed_models": self.passed_models,
            "failed_models": self.failed_models,
            "models": [m.to_dict() for m in self.models],
        }


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_report(path: str | Path) -> dict[str, Any]:
    """Load an evaluation report JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_gate_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load gate config. Returns the ``gate`` sub-dict (or sensible defaults).

    Falls back to ``thresholds`` in ``config/tests.yaml`` for backward
    compatibility when no dedicated gate config exists.
    """
    path = Path(path)
    if yaml is not None and path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        gate = data.get("gate", data)
        if isinstance(gate, dict):
            return _normalize_gate_config(gate)

    # Backward-compatible fallback: legacy thresholds in tests.yaml
    legacy = Path("config/tests.yaml")
    if yaml is not None and legacy.exists():
        with open(legacy, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        thresholds = data.get("thresholds", {})
        return _normalize_gate_config(
            {
                "weighted_score_min": thresholds.get("overall_score", 0.70),
                "max_latency_p95": thresholds.get("max_latency_p95", 10.0),
                "fail_on_test_error": True,
            }
        )

    return _normalize_gate_config({})


def _normalize_gate_config(gate: dict[str, Any]) -> dict[str, Any]:
    """Fill defaults so downstream code can rely on keys existing."""
    normalized = {
        "weighted_score_min": gate.get("weighted_score_min"),
        "max_latency_p95": gate.get("max_latency_p95"),
        "fail_on_test_error": gate.get("fail_on_test_error", True),
        "tests": dict(gate.get("tests", {}) or {}),
        "regression": dict(gate.get("regression", {}) or {}),
    }
    return normalized


# --------------------------------------------------------------------------- #
# Report field accessors (tolerant to shape drift)
# --------------------------------------------------------------------------- #

def _models(report: dict[str, Any]) -> dict[str, Any]:
    return report.get("models", {}) or {}


def _weighted_score(model_data: dict[str, Any]) -> Optional[float]:
    val = (model_data.get("overall_metrics", {}) or {}).get("weighted_score")
    return float(val) if isinstance(val, (int, float)) else None


def _latency_p95(model_data: dict[str, Any]) -> Optional[float]:
    val = (model_data.get("overall_metrics", {}) or {}).get("latency_p95")
    return float(val) if isinstance(val, (int, float)) else None


def _tests(model_data: dict[str, Any]) -> dict[str, Any]:
    return model_data.get("tests", {}) or {}


def _test_overall_score(test_data: dict[str, Any]) -> Optional[float]:
    val = (test_data.get("summary", {}) or {}).get("overall_score")
    return float(val) if isinstance(val, (int, float)) else None


def _test_has_error(test_data: dict[str, Any]) -> bool:
    return bool(isinstance(test_data, dict) and test_data.get("error"))


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate_gate(
    report: dict[str, Any],
    config: dict[str, Any],
    baseline: Optional[dict[str, Any]] = None,
    report_path: Optional[str] = None,
    baseline_path: Optional[str] = None,
) -> GateReport:
    """Apply the gate config (+ optional baseline) to a report."""
    config = _normalize_gate_config(config)
    regression = config.get("regression", {})
    baseline_models = _models(baseline) if baseline else {}

    gate = GateReport(passed=True, report_path=report_path, baseline_used=baseline_path)

    for model_key, model_data in _models(report).items():
        result = ModelGateResult(model_key=model_key)
        base_model = baseline_models.get(model_key, {})

        # 1. Test execution errors
        if config.get("fail_on_test_error", True):
            errored = [t for t, td in _tests(model_data).items() if _test_has_error(td)]
            if errored:
                result.checks.append(
                    CheckResult(
                        name="test_errors",
                        kind=KIND_ERROR,
                        value=float(len(errored)),
                        threshold=0.0,
                        passed=False,
                        detail=f"tests with errors: {', '.join(sorted(errored))}",
                    )
                )

        # 2. weighted_score minimum
        wmin = config.get("weighted_score_min")
        if wmin is not None:
            val = _weighted_score(model_data)
            base_val = _weighted_score(base_model) if base_model else None
            passed = val is not None and val >= wmin
            result.checks.append(
                CheckResult(
                    name="weighted_score",
                    kind=KIND_MIN,
                    value=val,
                    threshold=float(wmin),
                    passed=passed,
                    baseline_value=base_val,
                    delta=(val - base_val) if (val is not None and base_val is not None) else None,
                    detail=None if val is not None else "weighted_score missing",
                )
            )

        # 3. latency p95 maximum
        lmax = config.get("max_latency_p95")
        if lmax is not None:
            val = _latency_p95(model_data)
            if val is not None:
                result.checks.append(
                    CheckResult(
                        name="latency_p95",
                        kind=KIND_MAX,
                        value=val,
                        threshold=float(lmax),
                        passed=val <= lmax,
                    )
                )

        # 4. per-test overall_score minimums
        for test_name, test_min in (config.get("tests") or {}).items():
            test_data = _tests(model_data).get(test_name)
            if not isinstance(test_data, dict):
                continue
            val = _test_overall_score(test_data)
            base_val = _test_overall_score(base_model.get("tests", {}).get(test_name, {})) \
                if base_model.get("tests", {}).get(test_name) else None
            passed = val is not None and val >= test_min
            result.checks.append(
                CheckResult(
                    name=f"test:{test_name}",
                    kind=KIND_MIN,
                    value=val,
                    threshold=float(test_min),
                    passed=passed,
                    baseline_value=base_val,
                    delta=(val - base_val) if (val is not None and base_val is not None) else None,
                    detail=None if val is not None else "overall_score missing",
                )
            )

        # 5. regression vs baseline
        if base_model:
            max_w_drop = regression.get("max_weighted_drop")
            if max_w_drop is not None:
                cur, base = _weighted_score(model_data), _weighted_score(base_model)
                if cur is not None and base is not None:
                    delta = cur - base
                    result.checks.append(
                        CheckResult(
                            name="regression:weighted_score",
                            kind=KIND_REGRESSION,
                            value=cur,
                            threshold=float(max_w_drop),
                            passed=delta >= -max_w_drop,
                            baseline_value=base,
                            delta=delta,
                            detail=None if delta >= -max_w_drop
                            else f"dropped {abs(delta):.4f} (allowed {max_w_drop})",
                        )
                    )

            max_t_drop = regression.get("max_test_drop")
            if max_t_drop is not None:
                base_tests = base_model.get("tests", {}) or {}
                for test_name, test_data in _tests(model_data).items():
                    base_td = base_tests.get(test_name)
                    if not isinstance(base_td, dict):
                        continue
                    cur, base = _test_overall_score(test_data), _test_overall_score(base_td)
                    if cur is None or base is None:
                        continue
                    delta = cur - base
                    if delta < -max_t_drop:
                        result.checks.append(
                            CheckResult(
                                name=f"regression:test:{test_name}",
                                kind=KIND_REGRESSION,
                                value=cur,
                                threshold=float(max_t_drop),
                                passed=False,
                                baseline_value=base,
                                delta=delta,
                                detail=f"dropped {abs(delta):.4f} (allowed {max_t_drop})",
                            )
                        )

        result.passed = all(c.passed for c in result.checks)
        if not result.passed:
            gate.passed = False
        gate.models.append(result)

    # An empty report should not silently pass
    if not gate.models:
        gate.passed = False

    return gate


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #

def render_text(gate: GateReport) -> str:
    lines = ["", "=" * 72, "LLM EVAL — CI GATE", "=" * 72, ""]
    for m in gate.models:
        status = "PASS" if m.passed else "FAIL"
        lines.append(f"[{status}] {m.model_key}")
        for c in m.checks:
            mark = "  ok " if c.passed else "  XX "
            val = "n/a" if c.value is None else f"{c.value:.4f}"
            thr = "" if c.threshold is None else f" (thr {c.threshold})"
            delta = "" if c.delta is None else f" Δ{c.delta:+.4f}"
            detail = f" — {c.detail}" if c.detail else ""
            lines.append(f"{mark}{c.name}: {val}{thr}{delta}{detail}")
        lines.append("")
    lines.append("-" * 72)
    if gate.passed_models:
        lines.append(f"PASSED: {', '.join(gate.passed_models)}")
    if gate.failed_models:
        lines.append(f"FAILED: {', '.join(gate.failed_models)}")
    lines.append(f"\nGATE: {'PASSED ✅' if gate.passed else 'FAILED ❌'}")
    return "\n".join(lines)


def render_json(gate: GateReport) -> str:
    return json.dumps(gate.to_dict(), indent=2, ensure_ascii=False)


def render_markdown(gate: GateReport) -> str:
    """Markdown suitable for a PR comment or GITHUB_STEP_SUMMARY."""
    icon = "✅" if gate.passed else "❌"
    out = [f"## {icon} LLM Eval — CI Gate: {'PASSED' if gate.passed else 'FAILED'}", ""]
    if gate.baseline_used:
        out.append(f"_Baseline: `{gate.baseline_used}`_\n")
    for m in gate.models:
        status = "✅ PASS" if m.passed else "❌ FAIL"
        out.append(f"### {m.model_key} — {status}")
        out.append("")
        out.append("| Check | Value | Threshold | Δ vs baseline | Result |")
        out.append("|---|---|---|---|---|")
        for c in m.checks:
            val = "n/a" if c.value is None else f"{c.value:.4f}"
            thr = "—" if c.threshold is None else f"{c.threshold}"
            delta = "—" if c.delta is None else f"{c.delta:+.4f}"
            res = "✅" if c.passed else "❌"
            label = c.name + (f" ({c.detail})" if c.detail and not c.passed else "")
            out.append(f"| {label} | {val} | {thr} | {delta} | {res} |")
        out.append("")
    return "\n".join(out)


def render_badge(gate: GateReport) -> str:
    """shields.io endpoint JSON (https://shields.io/endpoint)."""
    if gate.passed:
        message, color = "passing", "brightgreen"
    else:
        n = len(gate.failed_models)
        message, color = f"{n} model(s) failing", "red"
    return json.dumps(
        {"schemaVersion": 1, "label": "llm eval", "message": message, "color": color},
        ensure_ascii=False,
    )


_RENDERERS = {
    "text": render_text,
    "json": render_json,
    "markdown": render_markdown,
    "badge": render_badge,
}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ci.gate",
        description="Gate an evaluation report for CI (thresholds + regression).",
    )
    parser.add_argument("report", help="Path to the evaluation report JSON")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Gate config YAML")
    parser.add_argument("--baseline", default=None, help="Baseline report JSON for regression checks")
    parser.add_argument("--format", default="text", choices=list(_RENDERERS.keys()))
    parser.add_argument("--output", default=None, help="Write render output to this file instead of stdout")
    args = parser.parse_args(argv)

    if not Path(args.report).exists():
        print(f"Error: report not found: {args.report}", file=sys.stderr)
        return 2

    try:
        report = load_report(args.report)
        config = load_gate_config(args.config)
        baseline = load_report(args.baseline) if args.baseline else None
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    gate = evaluate_gate(
        report, config, baseline=baseline,
        report_path=args.report, baseline_path=args.baseline,
    )

    rendered = _RENDERERS[args.format](gate)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)

    return 0 if gate.passed else 1


if __name__ == "__main__":
    sys.exit(main())
