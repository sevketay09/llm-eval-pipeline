"""
G-Eval Implementation - Dynamic Criteria Evaluation with Chain-of-Thought.

Based on: "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment"
(Liu et al., 2023)

Key idea: Given a custom evaluation criteria, the LLM first generates a
detailed chain-of-thought evaluation plan, then scores the response step by step.
This produces more aligned, explainable scores than single-prompt rubrics.
"""
import json
import re
from typing import Dict, Any, List, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


# Default criteria templates (users can override with custom criteria)
BUILTIN_CRITERIA = {
    "coherence": {
        "name": "Tutarlılık (Coherence)",
        "description": (
            "Yanıtın mantıksal tutarlılığını değerlendir. "
            "Cümleler birbirini destekliyor mu? Çelişki var mı? "
            "Bilgiler düzgün bir akış içinde sunuluyor mu?"
        ),
        "scale": "1-5",
    },
    "fluency": {
        "name": "Akıcılık (Fluency)",
        "description": (
            "Yanıtın dil kalitesini değerlendir. "
            "Gramer doğru mu? Cümleler doğal mı? "
            "Türkçe dil yapısına uygun mu?"
        ),
        "scale": "1-5",
    },
    "relevance": {
        "name": "İlgililik (Relevance)",
        "description": (
            "Yanıtın soruyla ne kadar ilgili olduğunu değerlendir. "
            "Tüm bilgiler soruyu yanıtlamaya yönelik mi? "
            "Gereksiz/konu dışı bilgi var mı?"
        ),
        "scale": "1-5",
    },
    "correctness": {
        "name": "Doğruluk (Correctness)",
        "description": (
            "Yanıtın faktüel doğruluğunu referans cevapla karşılaştırarak değerlendir. "
            "Ana fikir doğru mu? Detaylar tutarlı mı? "
            "Hallüsinasyon var mı?"
        ),
        "scale": "1-5",
    },
    "completeness": {
        "name": "Tamlık (Completeness)",
        "description": (
            "Yanıtın soruyu ne kadar eksiksiz yanıtladığını değerlendir. "
            "Sorunun tüm kısımları cevaplanmış mı? "
            "Önemli bir bilgi atlanmış mı?"
        ),
        "scale": "1-5",
    },
}


class GEvalEvaluator:
    """G-Eval: Dynamic criteria evaluation using LLM-generated CoT plan.
    
    Usage:
        geval = GEvalEvaluator(judge_adapter)
        
        # Use built-in criteria
        result = geval.evaluate("coherence", query="...", response="...")
        
        # Use custom criteria
        result = geval.evaluate_custom(
            criteria_name="domain_accuracy",
            criteria_description="Evaluate financial accuracy...",
            query="...",
            response="...",
            reference="..."
        )
    """

    PLAN_GENERATION_PROMPT = """Sen bir değerlendirme uzmanısın. Aşağıdaki kritere göre bir metni puanlamak için adım adım bir değerlendirme planı oluştur.

Kriter: {criteria_name}
Açıklama: {criteria_description}
Puan Skalası: {scale}

Değerlendirme planını oluştur. Her adım, metni incelerken nelere bakılacağını net olarak belirtmeli.
Yanıtını JSON formatında ver:

{{"evaluation_steps": ["Adım 1: ...", "Adım 2: ...", "Adım 3: ..."]}}"""

    SCORING_PROMPT = """Aşağıdaki metni verilen değerlendirme planına göre puanla.

Kriter: {criteria_name} ({criteria_description})
Puan Skalası: {scale}

Soru/Görev: {query}

Değerlendirilecek Yanıt:
{response}

{reference_section}

Değerlendirme Planı:
{plan_steps}

Her adımı uygula ve sonunda bir puan ver.
Yanıtını JSON formatında ver:

{{"step_evaluations": ["Adım 1 sonucu: ...", "Adım 2 sonucu: ..."], "score": <puan>, "reasoning": "<genel değerlendirme>"}}"""

    def __init__(self, judge_adapter, cache_plans: bool = True):
        """Initialize G-Eval evaluator.
        
        Args:
            judge_adapter: LLM adapter for judge model (UnifiedLLMAdapter instance).
            cache_plans: If True, caches generated evaluation plans per criteria.
        """
        self.judge = judge_adapter
        self._cache_plans = cache_plans
        self._plan_cache: Dict[str, List[str]] = {}

    def _generate_plan(self, criteria_name: str, criteria_description: str, scale: str) -> List[str]:
        """Generate evaluation plan (CoT steps) for given criteria."""
        cache_key = f"{criteria_name}:{criteria_description}:{scale}"
        if self._cache_plans and cache_key in self._plan_cache:
            return self._plan_cache[cache_key]

        prompt = self.PLAN_GENERATION_PROMPT.format(
            criteria_name=criteria_name,
            criteria_description=criteria_description,
            scale=scale,
        )

        messages = [
            {"role": "system", "content": "Sen bir NLG değerlendirme uzmanısın."},
            {"role": "user", "content": prompt},
        ]

        logger.debug(f"[geval] Generating plan for criteria='{criteria_name}'")
        result = self.judge.generate(messages)
        content = result.get("content") or ""
        if not content:
            logger.warning(f"[geval] plan empty response for '{criteria_name}' | error: {result.get('error')} | model: {result.get('model')}")

        try:
            parsed = self._parse_json(content)
            steps = parsed.get("evaluation_steps", [])
            logger.debug(f"[geval] plan ready for '{criteria_name}': {len(steps)} steps")
        except (json.JSONDecodeError, ValueError, AttributeError):
            logger.warning(f"[geval] plan parse failed for '{criteria_name}' | raw: {content[:500]!r}")
            steps = [f"Kriteri '{criteria_name}' doğrudan değerlendir."]

        if self._cache_plans:
            self._plan_cache[cache_key] = steps
        return steps

    def _score_with_plan(
        self,
        criteria_name: str,
        criteria_description: str,
        scale: str,
        plan_steps: List[str],
        query: str,
        response: str,
        reference: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Score a response using the generated evaluation plan."""
        reference_section = ""
        if reference:
            reference_section = f"Referans/Beklenen Yanıt:\n{reference}"

        numbered_steps = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan_steps))

        prompt = self.SCORING_PROMPT.format(
            criteria_name=criteria_name,
            criteria_description=criteria_description,
            scale=scale,
            query=query,
            response=response,
            reference_section=reference_section,
            plan_steps=numbered_steps,
        )

        messages = [
            {"role": "system", "content": "Sen bir NLG değerlendirme uzmanısın. Puanlamayı dikkatli ve adım adım yap."},
            {"role": "user", "content": prompt},
        ]

        logger.debug(f"[geval] Scoring criteria='{criteria_name}' | query_len={len(query)} response_len={len(response)}")
        result = self.judge.generate(messages)
        content = result.get("content") or ""
        if not content:
            logger.warning(f"[geval] scoring empty response for '{criteria_name}' | error: {result.get('error')} | full: {result}")

        try:
            parsed = self._parse_json(content)
            score = float(parsed.get("score", 0))
            reasoning = parsed.get("reasoning", "")
            step_evaluations = parsed.get("step_evaluations", [])
            logger.info(f"[geval] '{criteria_name}' → score={score:.2f}")
        except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
            logger.warning(f"[geval] scoring parse failed for '{criteria_name}' | raw: {content[:500]!r}")
            score = 0.0
            reasoning = "Parse error"
            step_evaluations = []

        return {
            "criteria": criteria_name,
            "score": score,
            "normalized_score": score / 5.0 if "5" in scale else score / 10.0,
            "reasoning": reasoning,
            "step_evaluations": step_evaluations,
            "plan_steps": plan_steps,
        }

    def evaluate(
        self,
        criteria_key: str,
        query: str,
        response: str,
        reference: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate using a built-in criteria key.
        
        Args:
            criteria_key: One of 'coherence', 'fluency', 'relevance', 'correctness', 'completeness'
            query: The user question/task.
            response: The model-generated response.
            reference: Optional expected answer for comparison.
            
        Returns:
            Dict with score, normalized_score, reasoning, step_evaluations, plan_steps
        """
        if criteria_key not in BUILTIN_CRITERIA:
            raise ValueError(
                f"Unknown criteria '{criteria_key}'. "
                f"Available: {list(BUILTIN_CRITERIA.keys())}. "
                "Use evaluate_custom() for custom criteria."
            )
        
        criteria = BUILTIN_CRITERIA[criteria_key]
        plan = self._generate_plan(criteria["name"], criteria["description"], criteria["scale"])
        return self._score_with_plan(
            criteria["name"], criteria["description"], criteria["scale"],
            plan, query, response, reference
        )

    def evaluate_custom(
        self,
        criteria_name: str,
        criteria_description: str,
        query: str,
        response: str,
        reference: Optional[str] = None,
        scale: str = "1-5",
    ) -> Dict[str, Any]:
        """Evaluate using custom user-defined criteria.
        
        Args:
            criteria_name: Name of the evaluation dimension.
            criteria_description: Detailed description of what to evaluate.
            query: The user question/task.
            response: The model-generated response.
            reference: Optional expected answer for comparison.
            scale: Score scale (e.g., "1-5", "1-10").
            
        Returns:
            Dict with score, normalized_score, reasoning, step_evaluations, plan_steps
        """
        plan = self._generate_plan(criteria_name, criteria_description, scale)
        return self._score_with_plan(
            criteria_name, criteria_description, scale,
            plan, query, response, reference
        )

    def evaluate_multi(
        self,
        criteria_keys: List[str],
        query: str,
        response: str,
        reference: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate across multiple built-in criteria.
        
        Returns:
            Dict mapping criteria_key -> evaluation result
        """
        results = {}
        for key in criteria_keys:
            results[key] = self.evaluate(key, query, response, reference)
        return results

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Extract and parse JSON from LLM response."""
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Find outermost JSON object by tracking brace depth
        start = text.find("{")
        if start != -1:
            depth = 0
            for i, ch in enumerate(text[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            break

        raise ValueError(f"No valid JSON found in: {text[:200]}")
