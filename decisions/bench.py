"""Classifier & Guardrail Bench core — decision (Jev) and LLM intent
classifiers benchmarked side by side, plus Jev-scored guardrail verdicts.
Standalone: no api/, utils/, adapters/, evaluators/ imports.
"""
from __future__ import annotations

import re
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

from decisions.client import DecisionClient
from decisions.types import ChoiceQ, NoulQ

ERROR = "ERROR"      # classify_fn raised on every retry
INVALID = "INVALID"  # LLM output did not resolve to exactly one label

ClassifyFn = Callable[[str], Tuple[str, float, Dict[str, Any]]]
GuardrailDecideFn = Callable[[str], Tuple[Dict[str, float], float]]

_COVERAGE_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
_DEFAULT_BLOCK_THRESHOLD = 0.75
_DEFAULT_REVIEW_THRESHOLD = 0.40


def make_decision_classifier(
    client: DecisionClient, criteria: Dict[str, str], instructions: str
) -> ClassifyFn:
    """ClassifyFn backed by a single Jev ChoiceQ call per text."""
    question = ChoiceQ(instructions=instructions, criteria=criteria)

    def classify(text: str) -> Tuple[str, float, Dict[str, Any]]:
        start = time.perf_counter()
        answer = client.decide(text, {"label": question}).answers["label"]
        elapsed_ms = (time.perf_counter() - start) * 1000
        return str(answer.value), elapsed_ms, {"confidence": answer.confidence}

    classify.est_cost_usd_fn = lambda: client.stats.snapshot()["est_cost_usd"]  # type: ignore[attr-defined]
    return classify


def make_llm_classifier(
    generate_fn: Callable[[List[Dict[str, str]]], Dict[str, Any]],
    criteria: Dict[str, str],
    instructions: str,
    cost_per_mtok_input: Optional[float] = None,
) -> ClassifyFn:
    """ClassifyFn backed by a generic chat-completion `generate_fn` (the
    shape of UnifiedLLMAdapter.generate: dict with content/usage/error).
    """
    system_prompt = (
        f"{instructions} Classify the input into exactly one of these categories:\n\n"
        + "\n".join(f"- {label}: {desc}" for label, desc in criteria.items())
        + "\n\nRespond with ONLY the category name exactly as written above, nothing else."
    )
    labels = list(criteria.keys())
    total_tokens = [0]

    def classify(text: str) -> Tuple[str, float, Dict[str, Any]]:
        start = time.perf_counter()
        result = generate_fn(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ]
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        if result.get("error"):
            raise RuntimeError(result["error"])
        raw = result.get("content") or ""
        extra: Dict[str, Any] = {"raw": raw}
        tokens = (result.get("usage") or {}).get("total_tokens")
        if tokens is not None:
            extra["tokens"] = tokens
            total_tokens[0] += tokens
        return parse_label(raw, labels), elapsed_ms, extra

    classify.est_cost_usd_fn = lambda: (  # type: ignore[attr-defined]
        round(total_tokens[0] / 1_000_000 * cost_per_mtok_input, 6)
        if cost_per_mtok_input is not None
        else None
    )
    return classify


def make_decision_guardrail_fn(
    client: DecisionClient, guardrail_categories: Dict[str, str]
) -> GuardrailDecideFn:
    """GuardrailDecideFn backed by one Jev call per text: a NoulQ per category."""
    questions = {name: NoulQ(instructions) for name, instructions in guardrail_categories.items()}

    def decide(text: str) -> Tuple[Dict[str, float], float]:
        start = time.perf_counter()
        answers = client.decide(text, questions).answers
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {name: answers[name].value for name in questions}, elapsed_ms

    return decide


def parse_label(text: str, labels: List[str]) -> str:
    """LLM output -> exact label, or INVALID if it can't be resolved to one."""
    label_by_lower = {label.lower(): label for label in labels}
    cleaned = text.strip().strip("`'\".*: \n").lower()
    if cleaned in label_by_lower:
        return label_by_lower[cleaned]
    found = {
        label
        for label in labels
        if re.search(rf"(?<![a-z]){re.escape(label.lower())}(?![a-z])", text.lower())
    }
    return found.pop() if len(found) == 1 else INVALID


def run_choice_bench(
    name: str,
    classify_fn: ClassifyFn,
    cases: List[Dict[str, Any]],
    repeats: int = 1,
    max_workers: int = 1,
) -> Dict[str, Any]:
    """Run one classifier over `cases` (`{id, text, label}`) `repeats` times.

    A raised exception counts as a wrong answer (`ERROR`), so a flaky
    classifier gains no advantage from failing instead of guessing.
    `max_workers` > 1 runs the (case, repeat) pairs concurrently — latency is
    measured per call inside `classify_fn`, so metrics are unaffected.
    """
    predictions: Dict[str, List[str]] = defaultdict(list)
    details: List[Dict[str, Any]] = []
    latencies: List[float] = []
    per_label: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    confusion: Dict[str, Counter] = defaultdict(Counter)
    correct = errors = invalid = 0

    def _run_one(task: Tuple[Dict[str, Any], int]) -> Tuple[Dict[str, Any], int, str, Optional[float], Dict[str, Any]]:
        case, rep = task
        try:
            predicted, latency_ms, extra = classify_fn(case["text"])
            return case, rep, predicted, latency_ms, extra
        except Exception:  # noqa: BLE001 — any classifier failure is a wrong answer
            return case, rep, ERROR, None, {}

    tasks = [(case, rep) for case in cases for rep in range(repeats)]
    if max_workers > 1 and tasks:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            outcomes = list(pool.map(_run_one, tasks))
    else:
        outcomes = [_run_one(t) for t in tasks]

    for case, rep, predicted, latency_ms, extra in outcomes:
        if predicted == ERROR:
            errors += 1
        else:
            latencies.append(latency_ms)
        if predicted == INVALID:
            invalid += 1
        is_correct = predicted == case["label"]
        correct += is_correct
        per_label[case["label"]]["total"] += 1
        per_label[case["label"]]["correct"] += is_correct
        confusion[case["label"]][predicted] += 1
        predictions[case["id"]].append(predicted)
        details.append(
            {
                "case_id": case["id"],
                "repeat": rep,
                "true_label": case["label"],
                "predicted": predicted,
                "correct": is_correct,
                "latency_ms": latency_ms,
                **extra,
            }
        )

    total = len(cases) * repeats
    consistent_cases = sum(1 for preds in predictions.values() if len(set(preds)) == 1)

    result: Dict[str, Any] = {
        "name": name,
        "total_decisions": total,
        "correct": correct,
        "accuracy": correct / total if total else 0,
        "errors": errors,
        "invalid_outputs": invalid,
        "median_latency_ms": statistics.median(latencies) if latencies else 0,
        "p95_latency_ms": _percentile(latencies, 95),
        "consistent_cases": consistent_cases,
        "total_cases": len(cases),
        "consistency_rate": consistent_cases / len(cases) if cases else 0,
        "per_label": {
            label: {**v, "accuracy": v["correct"] / v["total"] if v["total"] else 0}
            for label, v in per_label.items()
        },
        "confusion": {label: dict(c) for label, c in confusion.items()},
        "details": details,
    }

    if any("confidence" in d for d in details):
        result["coverage_curve"] = _coverage_curve(details)

    cost_fn = getattr(classify_fn, "est_cost_usd_fn", None)
    result["est_cost_usd"] = cost_fn() if cost_fn else None

    return result


def _coverage_curve(details: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    scored = [d for d in details if d.get("confidence") is not None]
    curve = []
    for threshold in _COVERAGE_THRESHOLDS:
        accepted = [d for d in scored if d["confidence"] >= threshold]
        coverage = len(accepted) / len(details) if details else 0.0
        accuracy = sum(1 for d in accepted if d["correct"]) / len(accepted) if accepted else None
        curve.append(
            {
                "threshold": threshold,
                "coverage": round(coverage, 4),
                "accuracy": round(accuracy, 4) if accuracy is not None else None,
            }
        )
    return curve


def run_guardrail_bench(
    name: str,
    decide_fn: GuardrailDecideFn,
    cases: List[Dict[str, Any]],
    guardrail_categories: Dict[str, Dict[str, Any]],
    thresholds: Optional[Dict[str, Dict[str, float]]] = None,
    max_workers: int = 1,
) -> Dict[str, Any]:
    """Run one guardrail over `cases` (`{id, text, expected, labels?}`).

    `guardrail_categories`: {category: {"kind": "risk"|"scope", ...}}. A
    "scope" category above threshold yields OUT_OF_SCOPE, never BLOCK — a
    message being off-topic is not itself a security risk. `max_workers` > 1
    runs cases concurrently, as in `run_choice_bench`.
    """
    thresholds = thresholds or {}
    risk_cats = [c for c, meta in guardrail_categories.items() if meta.get("kind") != "scope"]
    scope_cats = [c for c, meta in guardrail_categories.items() if meta.get("kind") == "scope"]

    details: List[Dict[str, Any]] = []
    latencies: List[float] = []
    errors = 0
    verdict_confusion: Dict[str, Counter] = defaultdict(Counter)
    per_category: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    correct = 0

    def _run_one(case: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], Optional[float], bool]:
        try:
            scores, latency_ms = decide_fn(case["text"])
            return case, scores, latency_ms, False
        except Exception:  # noqa: BLE001 — a failed guardrail call is a recorded error, not a crash
            return case, {}, None, True

    if max_workers > 1 and cases:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            outcomes = list(pool.map(_run_one, cases))
    else:
        outcomes = [_run_one(case) for case in cases]

    for case, scores, latency_ms, failed in outcomes:
        if failed:
            errors += 1
        else:
            latencies.append(latency_ms)

        verdict, triggered = _guardrail_verdict(scores, risk_cats, scope_cats, thresholds)
        is_correct = verdict == case["expected"]
        correct += is_correct
        verdict_confusion[case["expected"]][verdict] += 1

        expected_labels = set(case.get("labels") or [])
        for cat in guardrail_categories:
            predicted_flag = cat in triggered
            actual_flag = cat in expected_labels
            key = "tp" if predicted_flag and actual_flag else "fp" if predicted_flag else "fn" if actual_flag else "tn"
            per_category[cat][key] += 1

        details.append(
            {
                "case_id": case["id"],
                "text": case["text"],
                "expected": case["expected"],
                "verdict": verdict,
                "scores": scores,
                "triggered": sorted(triggered),
                "correct": is_correct,
                "latency_ms": latency_ms,
            }
        )

    total = len(cases)
    block_cases = [d for d in details if d["expected"] == "BLOCK"]
    non_block_cases = [d for d in details if d["expected"] != "BLOCK"]
    block_recall = (
        sum(1 for d in block_cases if d["verdict"] == "BLOCK") / len(block_cases)
        if block_cases
        else None
    )
    false_block_rate = (
        sum(1 for d in non_block_cases if d["verdict"] == "BLOCK") / len(non_block_cases)
        if non_block_cases
        else None
    )

    per_category_metrics = {}
    for cat, counts in per_category.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        per_category_metrics[cat] = {
            "precision": tp / (tp + fp) if (tp + fp) else None,
            "recall": tp / (tp + fn) if (tp + fn) else None,
            **counts,
        }

    return {
        "name": name,
        "total_cases": total,
        "correct": correct,
        "verdict_accuracy": correct / total if total else 0,
        "errors": errors,
        "median_latency_ms": statistics.median(latencies) if latencies else 0,
        "p95_latency_ms": _percentile(latencies, 95),
        "block_recall": block_recall,
        "false_block_rate": false_block_rate,
        "verdict_confusion": {k: dict(v) for k, v in verdict_confusion.items()},
        "per_category": per_category_metrics,
        "details": details,
    }


def _guardrail_verdict(
    scores: Dict[str, float],
    risk_cats: List[str],
    scope_cats: List[str],
    thresholds: Dict[str, Dict[str, float]],
) -> Tuple[str, set]:
    triggered: set = set()
    block = False
    review = False
    for cat in risk_cats:
        score = scores.get(cat, 0.0)
        block_t = thresholds.get(cat, {}).get("block", _DEFAULT_BLOCK_THRESHOLD)
        review_t = thresholds.get(cat, {}).get("review", _DEFAULT_REVIEW_THRESHOLD)
        if score >= review_t:
            triggered.add(cat)
            review = True
        if score >= block_t:
            block = True

    if block:
        return "BLOCK", triggered

    for cat in scope_cats:
        score = scores.get(cat, 0.0)
        scope_t = thresholds.get(cat, {}).get("block", _DEFAULT_BLOCK_THRESHOLD)
        if score >= scope_t:
            triggered.add(cat)
            return "OUT_OF_SCOPE", triggered

    return ("REVIEW" if review else "PASS"), triggered


def _percentile(values: List[float], pct: int) -> float:
    if not values:
        return 0.0
    if len(values) < 2:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[pct - 1]
