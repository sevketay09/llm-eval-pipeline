"""
Groundedness Judge Evaluator — replaces faithfulness.py.
LLM-as-judge for RAG faithfulness / context grounding.
No Azure dependency. Uses UnifiedLLMAdapter.
Output schema identical to FaithfulnessEvaluator for backward compatibility.
"""
from typing import Dict, Any, Optional
from adapters.unified_adapter import UnifiedLLMAdapter
from evaluators.judge_utils import request_judge_json, extract_score
from utils.logger import get_logger

logger = get_logger(__name__)


class GroundednessJudgeEvaluator:
    """LLM-as-judge replacement for Azure GroundednessEvaluator.

    Evaluates whether every claim in the response is supported by the given context.
    Part of the RAG Triad (faithfulness dimension).

    Output schema matches FaithfulnessEvaluator exactly.
    """

    _PROMPT_WITH_QUERY = """Aşağıdaki yanıtın verilen bağlama ne kadar dayandığını değerlendir.

Kullanıcı sorusu: {query}

Bağlam (kaynak belgeler):
{context}

Model yanıtı:
{response}

Değerlendirme kriterleri:
- Yanıttaki her iddia bağlamda destekleniyor mu?
- Bağlamda olmayan bilgiler eklenmiş mi (hallüsinasyon)?
- Yanıt yalnızca bağlamdaki bilgilere mi dayanıyor?

Puan skalası:
1 - Yanıt büyük ölçüde bağlam dışı bilgi içeriyor
2 - Yanıtın çoğu bağlamla desteklenmiyor
3 - Kısmen destekleniyor, bazı bağlam dışı ifadeler var
4 - Büyük ölçüde bağlama dayalı, küçük sapmalar var
5 - Yanıttaki her iddia tamamen bağlamla destekleniyor

Sadece JSON formatında yanıt ver:
{{"score": <1-5>, "reasoning": "<hangi iddiaların desteklenip desteklenmediğini belirt>", "result": "pass" veya "fail"}}"""

    _PROMPT_WITHOUT_QUERY = """Aşağıdaki yanıtın verilen bağlama ne kadar dayandığını değerlendir.

Bağlam (kaynak belgeler):
{context}

Model yanıtı:
{response}

Değerlendirme kriterleri:
- Yanıttaki her iddia bağlamda destekleniyor mu?
- Bağlamda olmayan bilgiler eklenmiş mi (hallüsinasyon)?
- Yanıt yalnızca bağlamdaki bilgilere mi dayanıyor?

Puan skalası:
1 - Yanıt büyük ölçüde bağlam dışı bilgi içeriyor
2 - Yanıtın çoğu bağlamla desteklenmiyor
3 - Kısmen destekleniyor, bazı bağlam dışı ifadeler var
4 - Büyük ölçüde bağlama dayalı, küçük sapmalar var
5 - Yanıttaki her iddia tamamen bağlamla destekleniyor

Sadece JSON formatında yanıt ver:
{{"score": <1-5>, "reasoning": "<hangi iddiaların desteklenip desteklenmediğini belirt>", "result": "pass" veya "fail"}}"""

    def __init__(self, judge_adapter: UnifiedLLMAdapter, threshold: float = 3.0):
        """
        Args:
            judge_adapter: UnifiedLLMAdapter instance (same judge model as pipeline).
            threshold: Raw 1-5 score threshold for is_grounded. Default 3.0.
        """
        self.judge = judge_adapter
        self._threshold = threshold

    def evaluate(
        self,
        response: str,
        context: str,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate faithfulness of response against context.

        Args:
            response: Model-generated answer to evaluate.
            context: Retrieved context/documents to ground against.
            query: Optional original query.

        Returns:
            Dict with score (1-5), normalized_score (0-1), is_faithful, reasoning, result.
            Matches FaithfulnessEvaluator output schema exactly.
        """
        if query:
            prompt = self._PROMPT_WITH_QUERY.format(
                query=query, context=context, response=response
            )
        else:
            prompt = self._PROMPT_WITHOUT_QUERY.format(
                context=context, response=response
            )

        messages = [
            {"role": "system", "content": "Sen bir RAG değerlendirme uzmanısın. Yanıtların verilen bağlama dayanıp dayanmadığını objektif şekilde değerlendir. Sadece JSON formatında yanıt ver."},
            {"role": "user", "content": prompt},
        ]

        parsed = request_judge_json(self.judge, messages, "groundedness_judge")
        raw_score = extract_score(parsed, "groundedness_judge")
        if raw_score is None:
            # Parse/score failure: score stays None so aggregations exclude
            # this item instead of counting it as a genuine 0.
            return {
                "score": None,
                "normalized_score": None,
                "is_faithful": None,
                "reasoning": "parse error",
                "result": "error",
                "raw": parsed or {},
            }

        score = max(1.0, min(5.0, raw_score))
        reasoning = parsed.get("reasoning", "")
        res = parsed.get("result", "pass" if score >= self._threshold else "fail")
        is_faithful = score >= self._threshold
        logger.debug(f"[groundedness_judge] score={score:.1f} is_faithful={is_faithful}")
        return {
            "score": score,
            "normalized_score": round(score / 5.0, 4),
            "is_faithful": is_faithful,
            "reasoning": reasoning,
            "result": res,
            "raw": parsed,
        }

    def evaluate_batch(self, items: list) -> list:
        """Evaluate faithfulness for a batch of items.

        Args:
            items: List of dicts with keys: response, context, query (optional).

        Returns:
            List of evaluation result dicts.
        """
        results = []
        for item in items:
            results.append(self.evaluate(
                response=item["response"],
                context=item["context"],
                query=item.get("query"),
            ))
        return results


def is_faithfulness_available() -> bool:
    """Always True — no external credentials needed."""
    return True
