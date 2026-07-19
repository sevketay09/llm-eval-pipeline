"""
Quality Judge Evaluator — replaces azure_quality.py.
LLM-as-judge for coherence, fluency, relevance, groundedness.
No Azure dependency. Uses UnifiedLLMAdapter (same judge model as pipeline).
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional
from adapters.unified_adapter import UnifiedLLMAdapter
from utils.logger import get_logger

logger = get_logger(__name__)


class QualityJudgeEvaluator:
    """LLM-as-judge replacement for Azure AI quality evaluators.

    Evaluates: coherence, fluency, relevance, groundedness (1-5 → normalized 0-1).
    Output keys match azure_quality.py so pipeline_runner.py call sites are unchanged.
    """

    _PROMPTS: Dict[str, str] = {
        "coherence": """Aşağıdaki yanıtın mantıksal tutarlılığını değerlendir.

Soru: {query}
Yanıt: {response}

Değerlendirme kriterleri:
- Cümleler birbirini mantıksal olarak takip ediyor mu?
- Çelişkili ifadeler var mı?
- Düşünceler net bir akış içinde sunuluyor mu?

Puan skalası: 1 (tamamen tutarsız) → 5 (mükemmel tutarlı)

Yanıtını JSON formatında ver:
{{"score": <1-5>, "reasoning": "<kısa gerekçe>"}}""",

        "fluency": """Aşağıdaki yanıtın dil kalitesini değerlendir.

Soru: {query}
Yanıt: {response}

Değerlendirme kriterleri:
- Gramer ve yazım doğruluğu
- Cümlelerin doğallığı ve akıcılığı
- Dil yapısına uygunluk

Puan skalası: 1 (çok kötü dil kalitesi) → 5 (mükemmel akıcılık)

Yanıtını JSON formatında ver:
{{"score": <1-5>, "reasoning": "<kısa gerekçe>"}}""",

        "relevance": """Aşağıdaki yanıtın soruyla ne kadar ilgili olduğunu değerlendir.

Soru: {query}
Yanıt: {response}

Değerlendirme kriterleri:
- Yanıt soruyu doğrudan ele alıyor mu?
- Konu dışı bilgi var mı?
- Sorunun tüm kısımları yanıtlanmış mı?

Puan skalası: 1 (tamamen ilgisiz) → 5 (tam anlamıyla ilgili)

Yanıtını JSON formatında ver:
{{"score": <1-5>, "reasoning": "<kısa gerekçe>"}}""",

        "groundedness": """Aşağıdaki yanıtın verilen bağlama ne kadar dayandığını değerlendir.

Soru: {query}
Bağlam/Kaynak: {context}
Yanıt: {response}

Değerlendirme kriterleri:
- Yanıttaki iddialar bağlamda destekleniyor mu?
- Bağlamda olmayan bilgiler ekleniyor mu?
- Olgusal tutarlılık var mı?

Puan skalası: 1 (tamamen bağlam dışı) → 5 (tamamen bağlama dayalı)

Yanıtını JSON formatında ver:
{{"score": <1-5>, "reasoning": "<kısa gerekçe>"}}""",
    }

    def __init__(self, judge_adapter: UnifiedLLMAdapter):
        self.judge = judge_adapter

    def _run(self, prompt: str, metric: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": "Sen bir metin kalitesi değerlendirme uzmanısın. Verilen kriterlere göre yanıtları objektif şekilde puanla. Sadece JSON formatında yanıt ver."},
            {"role": "user", "content": prompt},
        ]
        result = self.judge.generate(messages)
        content = (result.get("content") or "").strip()

        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            parsed = json.loads(content)
            raw_score = float(parsed.get("score", 0))
            score = max(0.0, min(5.0, raw_score))
            reasoning = parsed.get("reasoning", "")
            logger.debug(f"[quality_judge] {metric} → score={score:.1f}")
            return {"score": score, "normalized": round(score / 5.0, 4), "reasoning": reasoning}
        except Exception as e:
            logger.warning(f"[quality_judge] {metric} parse failed: {e} | raw: {content[:200]!r}")
            return {"score": 0.0, "normalized": 0.0, "reasoning": "parse error"}

    def evaluate_coherence(self, query: str, response: str) -> Dict[str, Any]:
        prompt = self._PROMPTS["coherence"].format(query=query, response=response)
        return self._run(prompt, "coherence")

    def evaluate_fluency(self, query: str, response: str) -> Dict[str, Any]:
        prompt = self._PROMPTS["fluency"].format(query=query, response=response)
        return self._run(prompt, "fluency")

    def evaluate_relevance(self, query: str, response: str) -> Dict[str, Any]:
        prompt = self._PROMPTS["relevance"].format(query=query, response=response)
        return self._run(prompt, "relevance")

    def evaluate_groundedness(
        self, query: str, response: str, context: str
    ) -> Dict[str, Any]:
        prompt = self._PROMPTS["groundedness"].format(
            query=query, response=response, context=context
        )
        return self._run(prompt, "groundedness")

    def evaluate_all(
        self,
        query: str,
        response: str,
        context: Optional[str] = None,
    ) -> Dict[str, float]:
        """Run all applicable quality evaluations.

        Returns raw 1-5 scores keyed by metric name — matches azure_quality.py
        contract so pipeline_runner.py divides by 5.0 as before.
        """
        tasks = {
            "coherence": lambda: self.evaluate_coherence(query, response)["score"],
            "fluency":   lambda: self.evaluate_fluency(query, response)["score"],
            "relevance": lambda: self.evaluate_relevance(query, response)["score"],
        }
        if context:
            tasks["groundedness"] = lambda: self.evaluate_groundedness(query, response, context)["score"]

        scores: Dict[str, float] = {}
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = {executor.submit(fn): name for name, fn in tasks.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    scores[name] = future.result()
                except Exception as e:
                    logger.warning(f"[quality_judge] {name} failed: {e}")
                    scores[name] = 0.0

        logger.info(f"[quality_judge] evaluate_all → {scores}")
        return scores


def is_quality_available() -> bool:
    """Always True — no external credentials needed."""
    return True
