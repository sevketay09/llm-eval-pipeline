"""Statistical significance + confidence intervals for model comparison.

Most eval reports state "model A scored 0.88, model B scored 0.85" with no
notion of whether that 0.03 gap is real. This module adds the missing rigor by
treating each test's ``overall_score`` as one observation in a battery shared
across models:

* per-model **bootstrap confidence interval** over its test scores, and
* **pairwise paired tests** (paired t + Wilcoxon signed-rank) over the tests
  both models ran, with effect size (Cohen's d_z) and a small-sample warning.

It reuses the primitives in ``metrics.StatisticalMetrics`` and ``scipy.stats``;
the bootstrap is seeded, so output is deterministic.

Report contract::

    report["models"][model_key]["tests"][test]["summary"]["overall_score"]
    report["models"][model_key]["tests"][test]["error"]            # skip if set
    report["models"][model_key]["overall_metrics"]["weighted_score"]

CLI::

    python -m analysis.significance REPORT.json
        [--alpha 0.05] [--confidence 0.95] [--seed 42]
        [--format text|json|markdown] [--output FILE]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy import stats

from metrics import StatisticalMetrics

DEFAULT_ALPHA = 0.05
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 42
# Below this many shared/observed tests, paired stats are unreliable.
SMALL_SAMPLE_N = 8


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        return f if not np.isnan(f) else None
    return None


def extract_test_scores(report: dict[str, Any]) -> dict[str, dict[str, float]]:
    """{model_key: {test_name: overall_score}} for non-errored, numeric tests."""
    out: dict[str, dict[str, float]] = {}
    for model_key, model_data in (report.get("models", {}) or {}).items():
        scores: dict[str, float] = {}
        for test_name, test_data in (model_data.get("tests", {}) or {}).items():
            if not isinstance(test_data, dict) or test_data.get("error"):
                continue
            val = _as_float((test_data.get("summary", {}) or {}).get("overall_score"))
            if val is not None:
                scores[test_name] = val
        out[model_key] = scores
    return out


def _weighted_score(report: dict[str, Any], model_key: str) -> Optional[float]:
    model = (report.get("models", {}) or {}).get(model_key, {})
    return _as_float((model.get("overall_metrics", {}) or {}).get("weighted_score"))


def _effect_label(abs_d: float) -> str:
    if abs_d < 0.2:
        return "negligible"
    if abs_d < 0.5:
        return "small"
    if abs_d < 0.8:
        return "medium"
    return "large"


# --------------------------------------------------------------------------- #
# Computation
# --------------------------------------------------------------------------- #

def per_model_intervals(
    report: dict[str, Any],
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, Any]]:
    """Bootstrap CI over each model's test scores + its weighted_score."""
    test_scores = extract_test_scores(report)
    result: dict[str, dict[str, Any]] = {}
    for model_key, scores in test_scores.items():
        values = list(scores.values())
        ci = StatisticalMetrics.bootstrap_confidence_interval(
            values, confidence=confidence, random_seed=seed
        )
        result[model_key] = {
            "weighted_score": _weighted_score(report, model_key),
            "mean_test_score": ci["mean"],
            "ci_lower": ci["ci_lower"],
            "ci_upper": ci["ci_upper"],
            "bootstrap_std": ci["bootstrap_std"],
            "n_tests": len(values),
            "small_sample": len(values) < SMALL_SAMPLE_N,
        }
    return result


def _paired_test(diffs: list[float], alpha: float) -> dict[str, Any]:
    """Paired t-test + Wilcoxon signed-rank on per-test differences (a - b)."""
    n = len(diffs)
    arr = np.asarray(diffs, dtype=float)
    mean_diff = float(np.mean(arr)) if n else 0.0
    std = float(np.std(arr, ddof=1)) if n >= 2 else 0.0

    # Paired t-test (vs 0). A perfectly consistent non-zero difference has zero
    # variance: that is the *most* significant case, so handle it explicitly
    # (scipy would emit inf/nan) and keep the output JSON-safe.
    if n < 2:
        t_stat, t_p, dz = None, 1.0, 0.0
    elif std == 0:
        if mean_diff == 0:
            t_stat, t_p, dz = 0.0, 1.0, 0.0
        else:
            t_stat, t_p, dz = None, 0.0, None  # consistent, unambiguous difference
    else:
        t_stat_raw, t_p_raw = stats.ttest_1samp(arr, 0.0)
        t_stat = float(t_stat_raw) if not np.isnan(t_stat_raw) else None
        t_p = float(t_p_raw) if not np.isnan(t_p_raw) else 1.0
        dz = mean_diff / std

    # Wilcoxon signed-rank (non-parametric); undefined if all diffs are zero
    try:
        if n >= 1 and np.any(arr != 0):
            _, w_p_raw = stats.wilcoxon(arr)
            w_p = float(w_p_raw) if not np.isnan(w_p_raw) else 1.0
        else:
            w_p = 1.0
    except ValueError:
        w_p = 1.0

    effect_dz = abs(dz) if dz is not None else float("inf")

    return {
        "t_statistic": t_stat,
        "p_value": t_p,
        "wilcoxon_p_value": w_p,
        "mean_difference": mean_diff,
        "cohens_dz": dz,
        "effect_size": _effect_label(effect_dz),
        "is_significant": bool(t_p < alpha),
    }


def pairwise_comparisons(
    report: dict[str, Any],
    alpha: float = DEFAULT_ALPHA,
) -> list[dict[str, Any]]:
    """Paired model-vs-model comparison over the tests both models ran."""
    test_scores = extract_test_scores(report)
    models = list(test_scores.keys())
    out: list[dict[str, Any]] = []

    for model_a, model_b in itertools.combinations(models, 2):
        a_scores, b_scores = test_scores[model_a], test_scores[model_b]
        shared = sorted(set(a_scores) & set(b_scores))
        diffs = [a_scores[t] - b_scores[t] for t in shared]
        mean_a = float(np.mean([a_scores[t] for t in shared])) if shared else 0.0
        mean_b = float(np.mean([b_scores[t] for t in shared])) if shared else 0.0

        entry: dict[str, Any] = {
            "model_a": model_a,
            "model_b": model_b,
            "n_shared_tests": len(shared),
            "shared_tests": shared,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "small_sample": len(shared) < SMALL_SAMPLE_N,
        }

        if len(shared) < 2:
            entry.update(
                {
                    "is_significant": False,
                    "p_value": 1.0,
                    "wilcoxon_p_value": 1.0,
                    "mean_difference": mean_a - mean_b,
                    "cohens_dz": 0.0,
                    "effect_size": "negligible",
                    "winner": None,
                    "verdict": "insufficient data (need >= 2 shared tests)",
                }
            )
            out.append(entry)
            continue

        test = _paired_test(diffs, alpha)
        entry.update(test)

        if test["is_significant"]:
            winner = model_a if test["mean_difference"] > 0 else model_b
            verdict = (
                f"{winner} significantly better "
                f"(p={test['p_value']:.4f}, {test['effect_size']} effect)"
            )
        else:
            winner = None
            verdict = (
                f"no significant difference (p={test['p_value']:.4f}, "
                f"Δ={test['mean_difference']:+.4f})"
            )
        entry["winner"] = winner
        entry["verdict"] = verdict
        if entry["small_sample"]:
            entry["verdict"] += f" — small sample (n={len(shared)})"
        out.append(entry)

    return out


def compute_significance(
    report: dict[str, Any],
    alpha: float = DEFAULT_ALPHA,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Full statistical comparison surface for a report."""
    intervals = per_model_intervals(report, confidence=confidence, seed=seed)
    pairwise = pairwise_comparisons(report, alpha=alpha)
    warnings: list[str] = []
    if any(m["small_sample"] for m in intervals.values()):
        warnings.append(
            f"Some models have < {SMALL_SAMPLE_N} tests; confidence intervals are wide."
        )
    if any(p["small_sample"] for p in pairwise):
        warnings.append(
            f"Some pairs share < {SMALL_SAMPLE_N} tests; significance is underpowered."
        )
    return {
        "alpha": alpha,
        "confidence": confidence,
        "seed": seed,
        "per_model": intervals,
        "pairwise": pairwise,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #

def render_json(result: dict[str, Any]) -> str:
    return json.dumps(result, indent=2, ensure_ascii=False)


def render_text(result: dict[str, Any]) -> str:
    lines = ["", "=" * 72, "STATISTICAL COMPARISON", "=" * 72, ""]
    lines.append(f"alpha={result['alpha']}  confidence={result['confidence']}  seed={result['seed']}")
    lines.append("")
    lines.append(f"Per-model {int(result['confidence']*100)}% CI (over test scores):")
    for model, m in result["per_model"].items():
        ws = "n/a" if m["weighted_score"] is None else f"{m['weighted_score']:.4f}"
        flag = "  [small sample]" if m["small_sample"] else ""
        lines.append(
            f"  {model}: weighted={ws}  mean={m['mean_test_score']:.4f} "
            f"[{m['ci_lower']:.4f}, {m['ci_upper']:.4f}]  n={m['n_tests']}{flag}"
        )
    lines.append("")
    lines.append("Pairwise (paired over shared tests):")
    for p in result["pairwise"]:
        lines.append(f"  {p['model_a']} vs {p['model_b']}: {p['verdict']}")
    if result["warnings"]:
        lines.append("")
        lines.append("Warnings:")
        for w in result["warnings"]:
            lines.append(f"  - {w}")
    return "\n".join(lines)


def render_markdown(result: dict[str, Any]) -> str:
    conf = int(result["confidence"] * 100)
    out = ["## 📊 İstatistiksel Karşılaştırma", ""]
    out.append(f"_alpha={result['alpha']}, {conf}% CI, seed={result['seed']}_")
    out.append("")
    out.append(f"### Model bazlı %{conf} güven aralığı (test skorları üzerinden)")
    out.append("")
    out.append("| Model | weighted_score | Ortalama | CI | n | Not |")
    out.append("|---|---|---|---|---|---|")
    for model, m in result["per_model"].items():
        ws = "—" if m["weighted_score"] is None else f"{m['weighted_score']:.4f}"
        note = "⚠️ küçük örneklem" if m["small_sample"] else ""
        out.append(
            f"| {model} | {ws} | {m['mean_test_score']:.4f} "
            f"| [{m['ci_lower']:.4f}, {m['ci_upper']:.4f}] | {m['n_tests']} | {note} |"
        )
    out.append("")
    out.append("### İkili karşılaştırma (ortak testler üzerinde eşleştirilmiş)")
    out.append("")
    out.append("| A | B | Δ ortalama | p (paired t) | Wilcoxon p | Etki | Sonuç |")
    out.append("|---|---|---|---|---|---|---|")
    for p in result["pairwise"]:
        sig = "✅" if p.get("is_significant") else "—"
        out.append(
            f"| {p['model_a']} | {p['model_b']} | {p.get('mean_difference', 0.0):+.4f} "
            f"| {p.get('p_value', 1.0):.4f} | {p.get('wilcoxon_p_value', 1.0):.4f} "
            f"| {p.get('effect_size', 'n/a')} | {sig} {p['verdict']} |"
        )
    if result["warnings"]:
        out.append("")
        for w in result["warnings"]:
            out.append(f"> ⚠️ {w}")
    return "\n".join(out)


_RENDERERS = {"text": render_text, "json": render_json, "markdown": render_markdown}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.significance",
        description="Confidence intervals + pairwise significance for a report.",
    )
    parser.add_argument("report", help="Path to the evaluation report JSON")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--format", default="text", choices=list(_RENDERERS.keys()))
    parser.add_argument("--output", default=None, help="Write output to FILE instead of stdout")
    args = parser.parse_args(argv)

    if not Path(args.report).exists():
        print(f"Error: report not found: {args.report}", file=sys.stderr)
        return 2
    try:
        with open(args.report, "r", encoding="utf-8") as f:
            report = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    result = compute_significance(
        report, alpha=args.alpha, confidence=args.confidence, seed=args.seed
    )
    rendered = _RENDERERS[args.format](result)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
