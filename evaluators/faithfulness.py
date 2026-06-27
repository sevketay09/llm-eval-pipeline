"""
RAG Faithfulness Evaluator - GroundednessEvaluator wrapper for RAG Triad.

Extends the existing RAGEvaluator (judge-based) with Azure AI SDK's
GroundednessEvaluator for more rigorous context-grounding checks.

RAG Triad components:
1. Context Relevance: Is the retrieved context relevant to the query? (existing)
2. Answer Relevance: Does the answer address the query? (existing via judge)
3. Faithfulness/Groundedness: Is every claim in the answer supported by context? (THIS MODULE)
"""
import os
from typing import Dict, Any, Optional
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    from azure.ai.evaluation import GroundednessEvaluator
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False
    logger.warning("azure-ai-evaluation not installed. Faithfulness evaluator unavailable.")


def _get_model_config() -> Optional[Dict[str, str]]:
    """Build model_config from environment variables."""
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    key = os.environ.get("AZURE_OPENAI_KEY")
    deployment = os.environ.get(
        "AZURE_OPENAI_DEPLOYMENT_NAME_PTU",
        os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME_PR", "")
    )
    if endpoint and key and deployment:
        return {
            "azure_endpoint": endpoint,
            "api_key": key,
            "azure_deployment": deployment,
            "api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01"),
        }
    return None


def is_faithfulness_available() -> bool:
    """Check if GroundednessEvaluator can be used."""
    return _SDK_AVAILABLE and _get_model_config() is not None


class FaithfulnessEvaluator:
    """Evaluates faithfulness/groundedness of answers against provided context.

    Uses Azure AI GroundednessEvaluator to check if claims in the response
    are substantiated by the given context. Part of the RAG Triad.

    Usage:
        evaluator = FaithfulnessEvaluator()
        result = evaluator.evaluate(
            query="What is the interest rate?",
            context="The current interest rate is 4.5% as of January 2024.",
            response="The interest rate is 4.5%."
        )
        # result = {"score": 5.0, "reasoning": "...", "is_faithful": True, ...}
    """

    def __init__(self, model_config: Optional[Dict[str, str]] = None, threshold: float = 3.0):
        """Initialize GroundednessEvaluator.

        Args:
            model_config: Azure OpenAI config dict. If None, reads from env vars.
            threshold: Score threshold for passing (1-5 scale). Default 3.0.
        """
        if not _SDK_AVAILABLE:
            raise ImportError("azure-ai-evaluation package required.")

        config = model_config or _get_model_config()
        if not config:
            raise ValueError(
                "Azure OpenAI config required. Set AZURE_OPENAI_ENDPOINT, "
                "AZURE_OPENAI_KEY, and Azure deployment env vars."
            )

        self._evaluator = GroundednessEvaluator(model_config=config, threshold=threshold)
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
            query: Optional original query (uses different prompt template if provided).

        Returns:
            Dict with:
                - score: float (1-5 scale)
                - normalized_score: float (0-1 scale)
                - is_faithful: bool (score >= threshold)
                - reasoning: str
                - result: str ("pass"/"fail")
        """
        try:
            if query:
                result = self._evaluator(
                    response=response,
                    context=context,
                    query=query,
                )
            else:
                result = self._evaluator(
                    response=response,
                    context=context,
                )

            score = result.get("groundedness", 0.0)
            return {
                "score": score,
                "normalized_score": score / 5.0,
                "is_faithful": score >= self._threshold,
                "reasoning": result.get("groundedness_reason", ""),
                "result": result.get("groundedness_result", ""),
                "raw": result,
            }
        except Exception as e:
            logger.warning(f"Faithfulness evaluation failed: {e}")
            return {
                "score": 0.0,
                "normalized_score": 0.0,
                "is_faithful": False,
                "reasoning": f"Error: {e}",
                "result": "error",
                "raw": {},
            }

    def evaluate_batch(
        self,
        items: list,
    ) -> list:
        """Evaluate faithfulness for a batch of items.

        Args:
            items: List of dicts with keys: response, context, query (optional).

        Returns:
            List of evaluation result dicts.
        """
        results = []
        for item in items:
            result = self.evaluate(
                response=item["response"],
                context=item["context"],
                query=item.get("query"),
            )
            results.append(result)
        return results
