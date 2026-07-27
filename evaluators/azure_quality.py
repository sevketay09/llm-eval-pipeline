"""
Azure Quality Evaluators - Wraps Azure AI Evaluation SDK quality metrics.
Uses existing Azure OpenAI deployment (same credentials as judge model).
Provides: Coherence, Fluency, Relevance, Groundedness evaluations.
"""
import os
from typing import Dict, Any, Optional
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    from azure.ai.evaluation import (
        CoherenceEvaluator,
        FluencyEvaluator,
        RelevanceEvaluator,
        GroundednessEvaluator,
    )
    _QUALITY_SDK_AVAILABLE = True
except ImportError:
    _QUALITY_SDK_AVAILABLE = False
    logger.warning("azure-ai-evaluation not installed. Quality evaluators unavailable.")


def _build_model_config() -> Optional[Dict[str, str]]:
    """Build model_config dict from environment variables (Azure OpenAI)."""
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    key = os.environ.get("AZURE_OPENAI_KEY")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME_PTU") or os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME_PR")

    if not all([endpoint, key, deployment]):
        logger.debug("Azure OpenAI env vars not fully set; quality evaluators disabled.")
        return None

    return {
        "azure_endpoint": endpoint,
        "api_key": key,
        "azure_deployment": deployment,
        "api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01"),
    }


class AzureQualityEvaluator:
    """Azure AI quality evaluators leveraging existing Azure OpenAI deployment.
    
    Evaluates: coherence, fluency, relevance, groundedness.
    All metrics return a 1-5 score (GPT-based assessment).
    """

    def __init__(self, model_config: Optional[Dict[str, str]] = None):
        if not _QUALITY_SDK_AVAILABLE:
            raise ImportError(
                "azure-ai-evaluation package required. "
                "Install with: pip install azure-ai-evaluation"
            )
        
        self._model_config = model_config or _build_model_config()
        if not self._model_config:
            raise ValueError(
                "Azure OpenAI credentials not available. "
                "Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, and Azure deployment env vars."
            )

        self._coherence = CoherenceEvaluator(model_config=self._model_config)
        self._fluency = FluencyEvaluator(model_config=self._model_config)
        self._relevance = RelevanceEvaluator(model_config=self._model_config)
        self._groundedness = GroundednessEvaluator(model_config=self._model_config)

    def evaluate_coherence(self, query: str, response: str) -> Dict[str, Any]:
        """Evaluate response coherence (1-5 scale)."""
        try:
            return self._coherence(query=query, response=response)
        except Exception as e:
            logger.warning(f"Coherence evaluation failed: {e}")
            return {"gpt_coherence": 0.0}

    def evaluate_fluency(self, query: str, response: str) -> Dict[str, Any]:
        """Evaluate response fluency (1-5 scale)."""
        try:
            return self._fluency(query=query, response=response)
        except Exception as e:
            logger.warning(f"Fluency evaluation failed: {e}")
            return {"gpt_fluency": 0.0}

    def evaluate_relevance(self, query: str, response: str) -> Dict[str, Any]:
        """Evaluate response relevance to query (1-5 scale)."""
        try:
            return self._relevance(query=query, response=response)
        except Exception as e:
            logger.warning(f"Relevance evaluation failed: {e}")
            return {"gpt_relevance": 0.0}

    def evaluate_groundedness(self, query: str, response: str, context: str) -> Dict[str, Any]:
        """Evaluate response groundedness against provided context (1-5 scale)."""
        try:
            return self._groundedness(query=query, response=response, context=context)
        except Exception as e:
            logger.warning(f"Groundedness evaluation failed: {e}")
            return {"gpt_groundedness": 0.0}

    def evaluate_all(self, query: str, response: str, context: Optional[str] = None) -> Dict[str, float]:
        """Run all applicable quality evaluations.
        
        Args:
            query: The user question/prompt.
            response: Model-generated response.
            context: Optional reference context (needed for groundedness).
            
        Returns:
            Dict with keys: coherence, fluency, relevance, [groundedness]
        """
        scores = {}
        
        coherence_result = self.evaluate_coherence(query, response)
        scores["coherence"] = coherence_result.get("gpt_coherence", 0.0)

        fluency_result = self.evaluate_fluency(query, response)
        scores["fluency"] = fluency_result.get("gpt_fluency", 0.0)

        relevance_result = self.evaluate_relevance(query, response)
        scores["relevance"] = relevance_result.get("gpt_relevance", 0.0)

        if context:
            groundedness_result = self.evaluate_groundedness(query, response, context)
            scores["groundedness"] = groundedness_result.get("gpt_groundedness", 0.0)

        return scores


def is_quality_available() -> bool:
    """Check if Azure quality evaluators can be initialized."""
    if not _QUALITY_SDK_AVAILABLE:
        return False
    config = _build_model_config()
    return config is not None
