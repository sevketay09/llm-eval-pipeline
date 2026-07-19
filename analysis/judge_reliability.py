"""
analysis/judge_reliability.py — Judge ↔ human agreement statistics.

Pure Python (no scipy/numpy): safe under the conftest scipy mock and in
any runtime. Follows the analysis/ standalone-module pattern: no imports
from api/, utils/, adapters/.

Metrics:
- Spearman rho (rank correlation, average ranks for ties)
- Cohen's kappa on 3-bucket labels (low/mid/high) — chance-corrected agreement
- mean bias (judge - human), MAE, simple agreement rate
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "spearman_rho",
    "cohens_kappa",
    "score_bucket",
    "compute_judge_reliability",
]

# 0-1 scores → categorical buckets aligned with the categorical judge design
# (YANLIS ~0, KISMEN_DOGRU ~0.5, TAM_DOGRU ~1)
_BUCKET_LOW_MAX = 0.25
_BUCKET_MID_MAX = 0.75

MIN_PAIRS = 5


def score_bucket(score: float) -> str:
    if score <= _BUCKET_LOW_MAX:
        return "low"
    if score <= _BUCKET_MID_MAX:
        return "mid"
    return "high"


def _average_ranks(values: Sequence[float]) -> List[float]:
    """Ranks starting at 1; ties get the average of their positions."""
    indexed = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    return cov / (var_x ** 0.5 * var_y ** 0.5)


def spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation with average ranks for ties. None if undefined."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson(_average_ranks(xs), _average_ranks(ys))


def cohens_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> Optional[float]:
    """Unweighted Cohen's kappa. None if undefined (e.g. both raters constant)."""
    if len(labels_a) != len(labels_b) or not labels_a:
        return None
    n = len(labels_a)
    categories = sorted(set(labels_a) | set(labels_b))

    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n

    expected = 0.0
    for cat in categories:
        p_a = sum(1 for a in labels_a if a == cat) / n
        p_b = sum(1 for b in labels_b if b == cat) / n
        expected += p_a * p_b

    if expected >= 1.0:
        # Both raters constant on the same single category: kappa undefined
        return None
    return (observed - expected) / (1.0 - expected)


def _kappa_interpretation(kappa: Optional[float]) -> str:
    """Landis & Koch (1977) bands."""
    if kappa is None:
        return "undefined"
    if kappa < 0:
        return "poor"
    if kappa < 0.2:
        return "slight"
    if kappa < 0.4:
        return "fair"
    if kappa < 0.6:
        return "moderate"
    if kappa < 0.8:
        return "substantial"
    return "almost_perfect"


def compute_judge_reliability(pairs: Sequence[Tuple[float, float]]) -> Dict[str, Any]:
    """Compute judge↔human reliability metrics from (llm_score, human_score) pairs.

    Scores are expected on a 0-1 scale. Returns a dict that is always
    JSON-serializable; metrics are None when undefined.
    """
    n = len(pairs)
    if n < MIN_PAIRS:
        return {
            "n": n,
            "spearman_rho": None,
            "cohens_kappa": None,
            "kappa_interpretation": "undefined",
            "mean_bias": None,
            "mean_absolute_error": None,
            "agreement_rate": None,
            "verdict": "insufficient_data",
        }

    llm = [float(p[0]) for p in pairs]
    human = [float(p[1]) for p in pairs]

    rho = spearman_rho(llm, human)
    kappa = cohens_kappa([score_bucket(s) for s in llm], [score_bucket(s) for s in human])
    mean_bias = sum(l - h for l, h in zip(llm, human)) / n
    mae = sum(abs(l - h) for l, h in zip(llm, human)) / n
    agreement_rate = sum(1.0 - min(1.0, abs(l - h)) for l, h in zip(llm, human)) / n

    interpretation = _kappa_interpretation(kappa)
    if kappa is not None and kappa >= 0.6 and mae <= 0.15:
        verdict = "reliable"
    elif kappa is not None and kappa >= 0.4:
        verdict = "acceptable"
    else:
        verdict = "needs_calibration"

    return {
        "n": n,
        "spearman_rho": round(rho, 4) if rho is not None else None,
        "cohens_kappa": round(kappa, 4) if kappa is not None else None,
        "kappa_interpretation": interpretation,
        "mean_bias": round(mean_bias, 4),
        "mean_absolute_error": round(mae, 4),
        "agreement_rate": round(agreement_rate, 4),
        "verdict": verdict,
    }
