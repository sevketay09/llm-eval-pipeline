"""RAG component-level evaluation — retriever vs generator fault isolation."""
import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

__all__ = [
    "compute_context_precision",
    "compute_context_recall",
    "compute_faithfulness",
    "compute_answer_relevance",
    "isolate_fault",
    "evaluate_rag_case",
    "evaluate_rag_report",
]

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "ve", "bir", "bu", "şu", "o", "de", "da", "ki", "ile", "için",
    "ama", "fakat", "ancak", "ya", "veya", "hem", "ne", "mi", "mı",
    "mu", "mü", "in", "on", "at", "to", "of", "and", "or", "but", "not",
}


def _tokenize(text: str) -> list:
    """Tokenize text: lowercase, split, filter stopwords and length < 2."""
    text = text.lower()
    tokens = re.findall(r'\b[a-z0-9]+\b', text)
    return [t for t in tokens if t not in _STOPWORDS and len(t) >= 2]


def _overlap(tokens_a: set, tokens_b: set) -> float:
    """Overlap coefficient: |a ∩ b| / min(|a|, |b|). Better than Jaccard for short queries."""
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / min(len(tokens_a), len(tokens_b))


def _jaccard(tokens_a: set, tokens_b: set) -> float:
    """Jaccard similarity: |a ∩ b| / |a ∪ b|."""
    if not tokens_a and not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


def _cosine(v1: list, v2: list) -> float:
    """Cosine similarity between two vectors."""
    if not v1 or not v2:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def compute_context_precision(question: str, contexts: list, *, embed_fn=None) -> dict:
    """Measure: how many retrieved chunks are relevant to the question?"""
    if not contexts:
        return {
            "precision": 0.0,
            "relevant_count": 0,
            "total_chunks": 0,
            "chunk_scores": [],
        }

    question_tokens = set(_tokenize(question))
    chunk_scores = []

    for context in contexts:
        context_tokens = set(_tokenize(context))

        if embed_fn is None:
            # Overlap coefficient (better than Jaccard for short queries)
            score = _overlap(question_tokens, context_tokens)
        else:
            # Embedding similarity: cosine
            embeddings = embed_fn([question, context])
            score = _cosine(embeddings[0], embeddings[1])

        chunk_scores.append(score)

    relevant_count = sum(1 for s in chunk_scores if s >= 0.5)
    precision = relevant_count / len(contexts) if contexts else 0.0

    return {
        "precision": precision,
        "relevant_count": relevant_count,
        "total_chunks": len(contexts),
        "chunk_scores": chunk_scores,
    }


def compute_context_recall(question: str, contexts: list, expected_answer: str, *, embed_fn=None) -> dict:
    """Measure: does context contain enough information to answer the question?"""
    if not expected_answer:
        return {
            "recall": 0.0,
            "covered_tokens": 0,
            "total_tokens": 0,
        }

    expected_tokens = set(_tokenize(expected_answer))
    if not expected_tokens:
        return {
            "recall": 0.0,
            "covered_tokens": 0,
            "total_tokens": 0,
        }

    combined_context = " ".join(contexts) if contexts else ""
    context_tokens = set(_tokenize(combined_context))

    if embed_fn is None:
        # Token overlap approach
        covered = len(expected_tokens & context_tokens)
        total = len(expected_tokens)
        recall = covered / total if total > 0 else 0.0
        return {
            "recall": recall,
            "covered_tokens": covered,
            "total_tokens": total,
        }
    else:
        # Embedding approach: max similarity of expected_answer to any context
        texts = [expected_answer] + contexts
        embeddings = embed_fn(texts)
        expected_emb = embeddings[0]

        max_similarity = 0.0
        for context_emb in embeddings[1:]:
            sim = _cosine(expected_emb, context_emb)
            max_similarity = max(max_similarity, sim)

        return {
            "recall": max_similarity,
            "covered_tokens": None,
            "total_tokens": None,
        }


def compute_faithfulness(answer: str, contexts: list, *, embed_fn=None) -> dict:
    """Measure: is the answer grounded in the provided contexts?"""
    answer_tokens = set(_tokenize(answer))
    if not answer_tokens:
        return {
            "faithfulness": 0.0,
            "grounded_tokens": 0,
            "total_answer_tokens": 0,
        }

    combined_context = " ".join(contexts) if contexts else ""
    context_tokens = set(_tokenize(combined_context))

    if embed_fn is None:
        # Token overlap approach
        grounded = len(answer_tokens & context_tokens)
        total = len(answer_tokens)
        faithfulness = grounded / total if total > 0 else 0.0
        return {
            "faithfulness": faithfulness,
            "grounded_tokens": grounded,
            "total_answer_tokens": total,
        }
    else:
        # Embedding approach: max similarity of answer to any context
        texts = [answer] + contexts
        embeddings = embed_fn(texts)
        answer_emb = embeddings[0]

        max_similarity = 0.0
        for context_emb in embeddings[1:]:
            sim = _cosine(answer_emb, context_emb)
            max_similarity = max(max_similarity, sim)

        return {
            "faithfulness": max_similarity,
            "grounded_tokens": None,
            "total_answer_tokens": None,
        }


def compute_answer_relevance(question: str, answer: str, *, embed_fn=None) -> dict:
    """Measure: does the answer address the question?"""
    question_tokens = set(_tokenize(question))
    answer_tokens = set(_tokenize(answer))

    if embed_fn is None:
        # Token overlap
        overlap = len(question_tokens & answer_tokens)
        total = len(question_tokens)
        relevance = overlap / total if total > 0 else 0.0
        return {"answer_relevance": relevance}
    else:
        # Embedding similarity
        embeddings = embed_fn([question, answer])
        relevance = _cosine(embeddings[0], embeddings[1])
        return {"answer_relevance": relevance}


def isolate_fault(case_result: dict) -> dict:
    """Determine where the fault is: retriever, generator, mixed, or none."""
    cp = case_result.get("context_precision", {}).get("precision", 0.5)
    cr = case_result.get("context_recall", {}).get("recall")
    faith = case_result.get("faithfulness", {}).get("faithfulness", 0.5)
    ar = case_result.get("answer_relevance", {}).get("answer_relevance", 0.5)

    # Severity logic - handle None from context_recall
    metrics_for_min = [cp, faith, ar]
    if cr is not None:
        metrics_for_min.append(cr)
    min_metric = min(metrics_for_min) if metrics_for_min else 0.5
    if min_metric < 0.3:
        severity = "high"
    elif min_metric < 0.5:
        severity = "medium"
    else:
        severity = "low"

    # Fault logic - handle None cr
    if cp < 0.5 and (cr is None or cr < 0.5):
        fault = "retriever"
        reason = "Low context precision and recall indicate poor retrieval quality."
    elif faith < 0.5:
        fault = "generator"
        reason = "Low faithfulness suggests the answer contains hallucinations."
    elif ar < 0.5:
        fault = "generator"
        reason = "Low answer relevance suggests the answer does not address the question."
    elif cp >= 0.5 and faith >= 0.5 and ar >= 0.5:
        fault = "none"
        reason = "All metrics are strong; no apparent fault detected."
    elif cp < 0.5 and faith >= 0.5:
        fault = "retriever"
        reason = "Low precision but answer is faithful; retriever needs improvement."
    else:
        fault = "mixed"
        reason = "Unclear fault pattern; multiple components may need review."

    return {
        "fault": fault,
        "reason": reason,
        "severity": severity,
    }


def evaluate_rag_case(case: dict, *, embed_fn=None) -> dict:
    """Evaluate a single RAG case."""
    question = case.get("question", "")
    contexts = case.get("contexts", [])
    answer = case.get("answer", "")
    expected_answer = case.get("expected_answer")

    # Compute metrics
    cp_result = compute_context_precision(question, contexts, embed_fn=embed_fn)
    faith_result = compute_faithfulness(answer, contexts, embed_fn=embed_fn)
    ar_result = compute_answer_relevance(question, answer, embed_fn=embed_fn)

    # Context recall only if expected_answer is provided
    if expected_answer:
        cr_result = compute_context_recall(question, contexts, expected_answer, embed_fn=embed_fn)
    else:
        cr_result = {"recall": None, "covered_tokens": None, "total_tokens": None}

    # Fault isolation
    fault_result = isolate_fault({
        "context_precision": cp_result,
        "context_recall": cr_result,
        "faithfulness": faith_result,
        "answer_relevance": ar_result,
    })

    # Compute overall RAG score (weighted average)
    metrics = [
        (0.25, cp_result["precision"]),
        (0.30, faith_result["faithfulness"]),
        (0.20, ar_result["answer_relevance"]),
    ]

    if cr_result["recall"] is not None:
        metrics.append((0.25, cr_result["recall"]))

    # Renormalize weights if recall is None
    total_weight = sum(w for w, _ in metrics)
    if total_weight == 0:
        overall_score = 0.0
    else:
        overall_score = sum(w * v for w, v in metrics) / total_weight

    return {
        "question": question,
        "context_precision": cp_result,
        "context_recall": cr_result,
        "faithfulness": faith_result,
        "answer_relevance": ar_result,
        "fault_isolation": fault_result,
        "overall_rag_score": overall_score,
    }


def evaluate_rag_report(report: dict, *, embed_fn=None) -> dict:
    """Evaluate entire report: walk models → tests → cases."""
    models_section = report.get("models", {})
    all_rag_evaluations = []
    model_results = {}

    for model_key, model_data in models_section.items():
        tests_section = model_data.get("tests", {})
        model_rag_cases = []

        for test_name, test_data in tests_section.items():
            results = test_data.get("results", [])

            for case in results:
                # Detect RAG case: has contexts or related fields
                if any(k in case for k in ["contexts", "retrieved_chunks", "context"]):
                    # Normalize field names
                    if "retrieved_chunks" in case and "contexts" not in case:
                        case["contexts"] = case["retrieved_chunks"]
                    if "context" in case and "contexts" not in case:
                        case["contexts"] = [case["context"]]

                    if "contexts" in case:
                        rag_eval = evaluate_rag_case(case, embed_fn=embed_fn)
                        model_rag_cases.append(rag_eval)
                        all_rag_evaluations.append(rag_eval)

        if model_rag_cases:
            # Compute model-level aggregates
            precisions = [e["context_precision"]["precision"] for e in model_rag_cases]
            recalls = [e["context_recall"]["recall"] for e in model_rag_cases if e["context_recall"]["recall"] is not None]
            faiths = [e["faithfulness"]["faithfulness"] for e in model_rag_cases]
            rels = [e["answer_relevance"]["answer_relevance"] for e in model_rag_cases]
            scores = [e["overall_rag_score"] for e in model_rag_cases]

            fault_counts = defaultdict(int)
            for e in model_rag_cases:
                fault_counts[e["fault_isolation"]["fault"]] += 1

            model_results[model_key] = {
                "avg_context_precision": sum(precisions) / len(precisions) if precisions else 0.0,
                "avg_context_recall": sum(recalls) / len(recalls) if recalls else None,
                "avg_faithfulness": sum(faiths) / len(faiths) if faiths else 0.0,
                "avg_answer_relevance": sum(rels) / len(rels) if rels else 0.0,
                "avg_overall_rag_score": sum(scores) / len(scores) if scores else 0.0,
                "fault_distribution": dict(fault_counts),
                "rag_case_count": len(model_rag_cases),
            }

    # Overall aggregates
    overall_fault_counts = defaultdict(int)
    for e in all_rag_evaluations:
        overall_fault_counts[e["fault_isolation"]["fault"]] += 1

    return {
        "total_rag_cases": len(all_rag_evaluations),
        "models": model_results,
        "overall_fault_distribution": dict(overall_fault_counts),
    }


def format_rag_text(result: dict) -> str:
    """Format RAG result as plain text table."""
    lines = []
    lines.append("=" * 80)
    lines.append("RAG EVALUATION SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Total RAG Cases: {result['total_rag_cases']}\n")

    models = result.get("models", {})
    if models:
        lines.append("MODEL RESULTS:")
        lines.append("-" * 80)
        for model_key, metrics in models.items():
            lines.append(f"\n{model_key}:")
            lines.append(f"  Context Precision:     {metrics['avg_context_precision']:.3f}")
            lines.append(f"  Context Recall:        {metrics['avg_context_recall'] or 'N/A'}")
            lines.append(f"  Faithfulness:          {metrics['avg_faithfulness']:.3f}")
            lines.append(f"  Answer Relevance:      {metrics['avg_answer_relevance']:.3f}")
            lines.append(f"  Overall RAG Score:     {metrics['avg_overall_rag_score']:.3f}")
            lines.append(f"  RAG Case Count:        {metrics['rag_case_count']}")
            lines.append(f"  Fault Distribution:    {metrics['fault_distribution']}")

    lines.append("\n" + "=" * 80)
    lines.append("OVERALL FAULT DISTRIBUTION")
    lines.append("=" * 80)
    for fault_type, count in result.get("overall_fault_distribution", {}).items():
        lines.append(f"  {fault_type:12s}: {count}")

    return "\n".join(lines)


def format_rag_markdown(result: dict) -> str:
    """Format RAG result as markdown tables."""
    lines = []
    lines.append("# RAG Evaluation Report\n")
    lines.append(f"**Total RAG Cases:** {result['total_rag_cases']}\n")

    models = result.get("models", {})
    if models:
        lines.append("## Model Results\n")
        lines.append("| Model | Precision | Recall | Faithfulness | Relevance | Score | Cases |")
        lines.append("|-------|-----------|--------|--------------|-----------|-------|-------|")
        for model_key, metrics in models.items():
            recall_str = f"{metrics['avg_context_recall']:.3f}" if metrics['avg_context_recall'] is not None else "N/A"
            lines.append(
                f"| {model_key} | {metrics['avg_context_precision']:.3f} | {recall_str} | "
                f"{metrics['avg_faithfulness']:.3f} | {metrics['avg_answer_relevance']:.3f} | "
                f"{metrics['avg_overall_rag_score']:.3f} | {metrics['rag_case_count']} |"
            )

    lines.append("\n## Fault Distribution\n")
    lines.append("| Fault Type | Count |")
    lines.append("|------------|-------|")
    for fault_type, count in result.get("overall_fault_distribution", {}).items():
        lines.append(f"| {fault_type} | {count} |")

    return "\n".join(lines)


def format_rag_json(result: dict) -> str:
    """Format RAG result as JSON."""
    return json.dumps(result, indent=2, ensure_ascii=False)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="RAG component-level evaluation")
    parser.add_argument("report", help="Path to eval report JSON file")
    parser.add_argument("--format", choices=["text", "json", "markdown"], default="text",
                        help="Output format")
    parser.add_argument("--output", help="Output file (default: stdout)")

    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"Error: Report file not found: {args.report}", file=sys.stderr)
        sys.exit(1)

    with open(report_path) as f:
        report = json.load(f)

    result = evaluate_rag_report(report)

    if args.format == "text":
        output = format_rag_text(result)
    elif args.format == "markdown":
        output = format_rag_markdown(result)
    else:  # json
        output = format_rag_json(result)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Wrote to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
