"""
Skill Trigger Simulation — does the skill's description route correctly?

Given a SKILL.md and a labeled prompt set (should-trigger / should-not /
ambiguous), asks a model per prompt whether it would invoke the skill, then
scores routing quality: precision, recall, F1, false-positive rate
(philschmid / Promptfoo skill-testing methodology). Ambiguous prompts are
tracked separately as a trigger rate, never counted in precision/recall.

Each prompt can be probed `repeats` times (distribution over trials, not a
single sample); the majority vote becomes the prediction. Unparseable model
replies are skipped, never guessed.
"""
from typing import Any, Dict, List, Optional

import yaml

from analysis.skill_lint import split_frontmatter
from utils.logger import get_logger

logger = get_logger(__name__)

MIN_SCORED_PROMPTS = 4
PRECISION_TARGET = 0.8
RECALL_TARGET = 0.8

PROBE_SYSTEM_PROMPT = (
    "Bir agent'sın ve elinde şu skill var:\n"
    "İSİM: {name}\n"
    "AÇIKLAMA: {description}\n\n"
    "Kullanıcı isteği için bu skill'i tetikler miydin? Skill'in açıklamasına "
    "göre karar ver; alakasız isteklerde tetikleme. SADECE şu JSON ile yanıt "
    'ver: {{"trigger": true}} veya {{"trigger": false}}'
)

EXPECTED_VALUES = (True, False, "ambiguous")


def extract_skill_meta(skill_text: str) -> Dict[str, str]:
    """Pull name/description out of SKILL.md frontmatter (empty strings if absent)."""
    raw_fm, _ = split_frontmatter(skill_text or "")
    meta = {"name": "", "description": ""}
    if raw_fm is None:
        return meta
    try:
        data = yaml.safe_load(raw_fm)
    except yaml.YAMLError:
        return meta
    if isinstance(data, dict):
        meta["name"] = str(data.get("name") or "").strip()
        meta["description"] = str(data.get("description") or "").strip()
    return meta


def routing_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Precision/recall/F1 over labeled (non-ambiguous) prompt results."""
    tp = fp = fn = tn = 0
    for item in results:
        if item.get("predicted") is None or item.get("expected") == "ambiguous":
            continue
        expected, predicted = bool(item["expected"]), bool(item["predicted"])
        if expected and predicted:
            tp += 1
        elif not expected and predicted:
            fp += 1
        elif expected and not predicted:
            fn += 1
        else:
            tn += 1
    scored = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "scored": scored,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "accuracy": round((tp + tn) / scored, 4) if scored else None,
        "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) else None,
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = routing_metrics(results)
    ambiguous = [r for r in results if r.get("expected") == "ambiguous" and r.get("predicted") is not None]
    ambiguous_rate = (
        round(sum(1 for r in ambiguous if r["predicted"]) / len(ambiguous), 4) if ambiguous else None
    )
    skipped = sum(1 for r in results if r.get("predicted") is None)

    if metrics["scored"] < MIN_SCORED_PROMPTS:
        verdict = "insufficient_data"
    else:
        precision_ok = metrics["precision"] is None or metrics["precision"] >= PRECISION_TARGET
        recall_ok = metrics["recall"] is None or metrics["recall"] >= RECALL_TARGET
        if precision_ok and recall_ok:
            verdict = "reliable"
        elif recall_ok:
            verdict = "over_triggering"  # fires on prompts it shouldn't
        elif precision_ok:
            verdict = "under_triggering"  # misses prompts it should catch
        else:
            verdict = "unreliable"

    return {
        **metrics,
        "ambiguous_count": len(ambiguous),
        "ambiguous_trigger_rate": ambiguous_rate,
        "skipped": skipped,
        "verdict": verdict,
    }


THRESHOLD_CURVE_POINTS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


class SkillTriggerChecker:
    """Probes a model with labeled prompts to measure skill-routing quality.

    Two backends: an LLM `adapter` (JSON-probed, majority vote over `repeats`
    trials), or a Jev `decision_client` (single Noul call per prompt, no
    repeats — Jev is deterministic). Supplying both prefers decision.
    """

    def __init__(
        self,
        adapter: Any = None,
        repeats: int = 1,
        decision_client: Any = None,
        threshold: float = 0.5,
    ):
        if adapter is None and decision_client is None:
            raise ValueError("SkillTriggerChecker requires an adapter or a decision_client")
        self.adapter = adapter
        self.repeats = max(1, int(repeats))
        self.decision_client = decision_client
        self.threshold = threshold

    def _probe_decision(self, meta: Dict[str, str], prompt_text: str) -> Optional[float]:
        from decisions.skill_routing import trigger_probability
        from decisions.types import DecisionError

        try:
            return trigger_probability(self.decision_client, meta, prompt_text)
        except DecisionError as e:
            logger.warning(f"[skill_trigger] decision probe failed for {prompt_text[:60]!r}: {e}")
            return None

    def _probe_once(self, meta: Dict[str, str], prompt_text: str) -> Optional[bool]:
        # Lazy import: evaluators/__init__ drags in scipy-dependent modules,
        # which must not be a requirement for lint-only / import-time paths.
        from evaluators.judge_utils import request_judge_json

        messages = [
            {
                "role": "system",
                "content": PROBE_SYSTEM_PROMPT.format(
                    name=meta["name"] or "(isimsiz)",
                    description=meta["description"] or "(açıklama yok)",
                ),
            },
            {"role": "user", "content": prompt_text},
        ]
        parsed = request_judge_json(self.adapter, messages, tag="skill_trigger")
        if parsed is None or not isinstance(parsed.get("trigger"), bool):
            return None
        return parsed["trigger"]

    def probe_prompt(self, meta: Dict[str, str], prompt: Dict[str, Any]) -> Dict[str, Any]:
        expected = prompt["expected"]

        if self.decision_client is not None:
            probability = self._probe_decision(meta, prompt["text"])
            trigger_rate = probability
            predicted = (probability >= self.threshold) if probability is not None else None
            trials = 1 if probability is not None else 0
            result = {
                "text": prompt["text"],
                "expected": expected,
                "predicted": predicted,
                "trigger_rate": trigger_rate,
                "trials": trials,
                "correct": (predicted == bool(expected))
                if predicted is not None and expected != "ambiguous"
                else None,
                "probability": probability,
            }
            return result

        trials = []
        for _ in range(self.repeats):
            outcome = self._probe_once(meta, prompt["text"])
            if outcome is not None:
                trials.append(outcome)
        trigger_rate = round(sum(trials) / len(trials), 4) if trials else None
        predicted = (trigger_rate >= 0.5) if trigger_rate is not None else None
        return {
            "text": prompt["text"],
            "expected": expected,
            "predicted": predicted,
            "trigger_rate": trigger_rate,
            "trials": len(trials),
            "correct": (predicted == bool(expected))
            if predicted is not None and expected != "ambiguous"
            else None,
        }

    def run(self, skill_text: str, prompts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """prompts: [{"text": str, "expected": True | False | "ambiguous"}, ...]"""
        meta = extract_skill_meta(skill_text)
        results = []
        for prompt in prompts:
            text = str(prompt.get("text") or "").strip()
            expected = prompt.get("expected")
            if not text or expected not in EXPECTED_VALUES:
                logger.warning(f"[skill_trigger] skipping malformed prompt: {prompt!r}")
                continue
            try:
                results.append(self.probe_prompt(meta, {"text": text, "expected": expected}))
            except Exception as e:  # network/adapter faults must not kill the run
                logger.warning(f"[skill_trigger] probe failed for {text[:60]!r}: {e}")

        summary = summarize(results)
        if self.decision_client is not None:
            summary["mode"] = "decision"
            summary["threshold_curve"] = _threshold_curve(results)

        return {"skill": meta, "summary": summary, "results": results}


def _threshold_curve(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Precision/recall/F1 at each candidate threshold, from stored probabilities."""
    curve = []
    for t in THRESHOLD_CURVE_POINTS:
        reclassified = [
            {
                "expected": r["expected"],
                "predicted": (r["probability"] >= t) if r.get("probability") is not None else None,
            }
            for r in results
        ]
        metrics = routing_metrics(reclassified)
        curve.append(
            {
                "threshold": t,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "false_positive_rate": metrics["false_positive_rate"],
            }
        )
    return curve
