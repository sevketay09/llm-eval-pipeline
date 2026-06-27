"""
NLP Metrics Evaluator - Azure AI Evaluation SDK wrapper
Runs BLEU, ROUGE, METEOR, GLEU, and F1 metrics fully offline (no Azure subscription required).
"""
from typing import Dict, Any, Optional
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    from azure.ai.evaluation import (
        BleuScoreEvaluator,
        RougeScoreEvaluator,
        RougeType,
        MeteorScoreEvaluator,
        GleuScoreEvaluator,
        F1ScoreEvaluator,
    )
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False
    logger.warning("azure-ai-evaluation not installed. NLP metrics will be unavailable.")


class NLPMetricsEvaluator:
    """Wraps Azure AI Evaluation NLP metrics for local use (zero Azure dependency)."""

    def __init__(self):
        if not _SDK_AVAILABLE:
            raise ImportError(
                "azure-ai-evaluation package is required. "
                "Install with: pip install azure-ai-evaluation"
            )
        self._bleu = BleuScoreEvaluator()
        self._rouge = RougeScoreEvaluator(rouge_type=RougeType.ROUGE_L)
        self._meteor = MeteorScoreEvaluator()
        self._gleu = GleuScoreEvaluator()
        self._f1 = F1ScoreEvaluator()

    def evaluate(self, response: str, ground_truth: str) -> Dict[str, float]:
        """Compute all NLP metrics for a single response/ground_truth pair.

        Args:
            response: Model-generated text.
            ground_truth: Reference/expected text.

        Returns:
            Dict with keys: bleu, rouge_l, meteor, gleu, f1
        """
        if not response or not ground_truth:
            return {"bleu": 0.0, "rouge_l": 0.0, "meteor": 0.0, "gleu": 0.0, "f1": 0.0}

        scores: Dict[str, float] = {}
        try:
            scores["bleu"] = self._bleu(response=response, ground_truth=ground_truth).get("bleu_score", 0.0)
        except Exception as e:
            logger.debug(f"BLEU computation failed: {e}")
            scores["bleu"] = 0.0

        try:
            rouge_result = self._rouge(response=response, ground_truth=ground_truth)
            scores["rouge_l"] = rouge_result.get("rouge_f1_score", 0.0)
        except Exception as e:
            logger.debug(f"ROUGE computation failed: {e}")
            scores["rouge_l"] = 0.0

        try:
            scores["meteor"] = self._meteor(response=response, ground_truth=ground_truth).get("meteor_score", 0.0)
        except Exception as e:
            logger.debug(f"METEOR computation failed: {e}")
            scores["meteor"] = 0.0

        try:
            scores["gleu"] = self._gleu(response=response, ground_truth=ground_truth).get("gleu_score", 0.0)
        except Exception as e:
            logger.debug(f"GLEU computation failed: {e}")
            scores["gleu"] = 0.0

        try:
            scores["f1"] = self._f1(response=response, ground_truth=ground_truth).get("f1_score", 0.0)
        except Exception as e:
            logger.debug(f"F1 computation failed: {e}")
            scores["f1"] = 0.0

        return scores

    def evaluate_batch(self, pairs: list) -> list:
        """Evaluate a list of (response, ground_truth) tuples.

        Args:
            pairs: List of (response, ground_truth) tuples.

        Returns:
            List of score dicts.
        """
        return [self.evaluate(resp, gt) for resp, gt in pairs]


def is_available() -> bool:
    """Check if the NLP metrics SDK is available."""
    return _SDK_AVAILABLE
