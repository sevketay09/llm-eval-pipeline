"""
Skill Fit Judge — LLM-as-judge scoring of how well a SKILL.md fits a task.

Answers: "is this skill good enough for the job I want done?" — the gap the
generic skill benchmarkers leave open (they score skills in isolation).

Rubric (5 criteria, each 0-1 + a verbatim evidence quote from the skill,
Skill Bench style):
  - scope_coverage:       does the skill's scope cover the task?
  - instruction_clarity:  are the instructions unambiguous and actionable?
  - completeness:         are steps/edge cases the task needs missing?
  - convention_alignment: does it pin versions/APIs/formats correctly?
  - efficiency_risk:      will it waste context/tokens or cause retries?

Parse failures return None (never a fabricated 0.0) — same contract as the
other judges via judge_utils.request_judge_json.
"""
import json
from typing import Any, Dict, List, Optional

from evaluators.judge_utils import request_judge_json
from utils.logger import get_logger

logger = get_logger(__name__)

CRITERIA = [
    "scope_coverage",
    "instruction_clarity",
    "completeness",
    "convention_alignment",
    "efficiency_risk",
]

FIT_THRESHOLD = 0.75
PARTIAL_FIT_THRESHOLD = 0.5

SYSTEM_PROMPT = (
    "Sen bir agent skill (SKILL.md) denetçisisin. Sana bir SKILL.md içeriği ve "
    "kullanıcının yaptırmak istediği görev tanımı verilecek. Skill'in bu görev "
    "için yeterliliğini beş kriterde 0.0-1.0 arası puanla: "
    "scope_coverage (kapsam görevi örtüyor mu), "
    "instruction_clarity (talimatlar net ve uygulanabilir mi), "
    "completeness (görevin gerektirdiği adımlar/uç durumlar eksik mi), "
    "convention_alignment (sürüm/API/format konvansiyonları doğru mu), "
    "efficiency_risk (context/token israfı veya gereksiz retry riski; 1.0 = risk yok). "
    "Her kriter için skill metninden BIREBIR bir alıntıyı evidence alanına koy; "
    "ilgili metin yoksa evidence null olsun. "
    "SADECE şu JSON şemasıyla yanıt ver: "
    '{"criteria": {"<kriter>": {"score": <float>, "evidence": <str|null>, '
    '"reasoning": <str>}}, "gaps": [<str>], "suggestions": [<str>]}'
)


def _clamp(value: Any) -> Optional[float]:
    try:
        return round(min(1.0, max(0.0, float(value))), 4)
    except (TypeError, ValueError):
        return None


def _verdict(overall: float) -> str:
    if overall >= FIT_THRESHOLD:
        return "fit"
    if overall >= PARTIAL_FIT_THRESHOLD:
        return "partial_fit"
    return "unfit"


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


class SkillFitJudge:
    """Scores a SKILL.md against a task description using a judge model."""

    def __init__(self, judge_adapter: Any):
        self.judge_adapter = judge_adapter

    def _build_messages(self, skill_text: str, task_description: str) -> List[Dict[str, str]]:
        user_content = (
            f"## GÖREV TANIMI\n{task_description.strip()}\n\n"
            f"## SKILL.MD\n{skill_text.strip()}"
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def evaluate(self, skill_text: str, task_description: str) -> Optional[Dict[str, Any]]:
        """Return the fit report, or None when the judge output is unusable."""
        if not (skill_text or "").strip() or not (task_description or "").strip():
            logger.warning("[skill_fit] empty skill or task input")
            return None

        parsed = request_judge_json(
            self.judge_adapter,
            self._build_messages(skill_text, task_description),
            tag="skill_fit",
        )
        if parsed is None:
            return None

        raw_criteria = parsed.get("criteria")
        if not isinstance(raw_criteria, dict):
            logger.warning("[skill_fit] verdict missing 'criteria' object")
            return None

        criteria: Dict[str, Dict[str, Any]] = {}
        for name in CRITERIA:
            entry = raw_criteria.get(name)
            if not isinstance(entry, dict):
                continue
            score = _clamp(entry.get("score"))
            if score is None:
                # Non-numeric score: drop the criterion rather than fake it.
                logger.warning(f"[skill_fit] non-numeric score for {name!r}")
                continue
            evidence = entry.get("evidence")
            criteria[name] = {
                "score": score,
                "evidence": str(evidence) if evidence else None,
                "reasoning": str(entry.get("reasoning") or ""),
            }

        if not criteria:
            logger.warning("[skill_fit] no usable criterion scores")
            return None

        overall = round(sum(c["score"] for c in criteria.values()) / len(criteria), 4)
        return {
            "overall": overall,
            "verdict": _verdict(overall),
            "criteria": criteria,
            "missing_criteria": [name for name in CRITERIA if name not in criteria],
            "gaps": _string_list(parsed.get("gaps")),
            "suggestions": _string_list(parsed.get("suggestions")),
        }
