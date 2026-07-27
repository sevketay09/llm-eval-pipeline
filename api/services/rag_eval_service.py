"""RAG eval service — wraps analysis/rag_eval."""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

import yaml

from api.schemas.rag_eval import RagContext, RagEvalResponse
from analysis.rag_eval import evaluate_rag_case


def _default_embedding_adapter_factory(model_key: str, config_path: str) -> Any:
    """Build a UnifiedEmbeddingAdapter for `model_key` with ${ENV_VAR} expansion.

    Mirrors SkillEvalService's _default_adapter_factory (same whole-config
    dump/substitute/reload approach) — api/services/config_service.py's loader
    does not expand ${ENV_VAR} placeholders, so reading straight from it would
    hand the adapter a literal "${OPENROUTER_API_KEY}" string as its key.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config_str = yaml.dump(config)
    for key, value in os.environ.items():
        config_str = config_str.replace(f"${{{key}}}", value)
    config = yaml.safe_load(config_str)
    if model_key not in config.get("embedding_models", {}):
        raise ValueError(f"Embedding model '{model_key}' not found in config")
    from adapters.embedding_adapter import UnifiedEmbeddingAdapter  # heavy import kept lazy

    return UnifiedEmbeddingAdapter(dict(config["embedding_models"][model_key]), model_key=model_key)


class RagEvalService:
    def __init__(
        self,
        config_path: str = "config/models.yaml",
        embedding_adapter_factory: Optional[Callable[[str, str], Any]] = None,
    ):
        self.config_path = config_path
        self.embedding_adapter_factory = embedding_adapter_factory or _default_embedding_adapter_factory
        # Embedding adapters are expensive to build (HuggingFace providers load
        # real model weights) and this service is a process-lifetime singleton
        # (see api/routers/rag_eval.py), so cache by model_key instead of
        # rebuilding per request.
        self._embedding_adapters: Dict[str, Any] = {}

    def _get_embedding_adapter(self, model_key: str) -> Any:
        if model_key not in self._embedding_adapters:
            self._embedding_adapters[model_key] = self.embedding_adapter_factory(model_key, self.config_path)
        return self._embedding_adapters[model_key]

    def evaluate(
        self,
        question: str,
        contexts: List[RagContext],
        answer: str,
        expected_answer: str = "",
        embedding_model: Optional[str] = None,
    ) -> RagEvalResponse:
        context_texts = [c.text for c in contexts]
        case = {
            "question": question,
            "contexts": context_texts,
            "answer": answer,
            "expected_answer": expected_answer,
        }

        embed_fn = None
        if embedding_model:
            adapter = self._get_embedding_adapter(embedding_model)

            def embed_fn(texts: List[str]):
                return adapter.encode(texts, normalize=True)["embeddings"]

        result = evaluate_rag_case(case, embed_fn=embed_fn)

        # evaluate_rag_case() returns metrics at the top level (context_precision,
        # context_recall, faithfulness, answer_relevance, fault_isolation), not
        # nested under a "scores" key — reading via result.get("scores", {})
        # always got an empty dict, so every field below silently fell back to
        # its 0.0/"none" default regardless of the real computed values (which
        # were still correct, just discarded). context_recall's "recall" is
        # None (not 0.0) when no expected_answer was given — `or 0.0` handles
        # both that and a genuine 0.0 score identically.
        cp = result.get("context_precision", {}).get("precision") or 0.0
        cr = result.get("context_recall", {}).get("recall") or 0.0
        fa = result.get("faithfulness", {}).get("faithfulness") or 0.0
        ar = result.get("answer_relevance", {}).get("answer_relevance") or 0.0
        fault = result.get("fault_isolation", {}).get("fault", "none")
        overall = round(result.get("overall_rag_score", 0.0), 4)

        return RagEvalResponse(
            question=question,
            context_precision=round(cp, 4),
            context_recall=round(cr, 4),
            faithfulness=round(fa, 4),
            answer_relevance=round(ar, 4),
            fault_component=fault,
            overall_score=overall,
            scoring_mode="embedding" if embedding_model else "token_overlap",
            embedding_model=embedding_model,
            details=result,
        )
