"""
Quality Judge Evaluator — replaces azure_quality.py.
LLM-as-judge for coherence, fluency, relevance, groundedness.
No Azure dependency. Uses UnifiedLLMAdapter (same judge model as pipeline).
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional
from adapters.unified_adapter import UnifiedLLMAdapter
from evaluators.judge_utils import request_judge_json, extract_score
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
        parsed = request_judge_json(self.judge, messages, f"quality_judge:{metric}")
        raw_score = extract_score(parsed, f"quality_judge:{metric}")
        if raw_score is None:
            # None (not 0.0) so callers can drop the metric instead of
            # polluting averages with a fake zero.
            return {"score": None, "normalized": None, "reasoning": "parse error"}

        score = max(0.0, min(5.0, raw_score))
        reasoning = parsed.get("reasoning", "")
        logger.debug(f"[quality_judge] {metric} → score={score:.1f}")
        return {"score": score, "normalized": round(score / 5.0, 4), "reasoning": reasoning}

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
        failed = []
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = {executor.submit(fn): name for name, fn in tasks.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    value = future.result()
                except Exception as e:
                    logger.warning(f"[quality_judge] {name} failed: {e}")
                    value = None
                # Failed metrics are omitted entirely — downstream mapping
                # builders treat a missing key as "not scored".
                if isinstance(value, (int, float)):
                    scores[name] = value
                else:
                    failed.append(name)

        logger.info(f"[quality_judge] evaluate_all → {scores}" + (f" | failed: {failed}" if failed else ""))
        return scores


def is_quality_available() -> bool:
    """Always True — no external credentials needed."""
    return True
