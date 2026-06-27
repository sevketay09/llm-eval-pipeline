"""Failure clustering and taxonomy generation for evaluation reports.

Extracts failed/low-scoring cases from an eval report, clusters them by
semantic similarity, and produces a taxonomy: "37% of failures are numerical
reasoning errors". Turns a raw failure list into actionable insight.

Core flow: extract failures below threshold or with errors → cluster by
text similarity → label clusters with keyword extraction.

The ``embed_fn`` and ``label_fn`` parameters are injectable for testing
without real embeddings or LLM calls.

Report contract::

    report["models"][model_key]["tests"][test_name]["results"]
        → list of cases, each:
            {
              "case_id": str,
              "scores": {"overall_score": float, ...},   # or flat dict
              "latency": float,
              "category": str,   # optional
              "question": str,   # optional, may also be "input_text" or "prompt"
              "expected_output": str,  # optional
              "actual_output": str,    # optional
              "error": str,      # optional, if case failed with error
            }

CLI::

    python -m analysis.failure_clustering REPORT.json
        [--threshold 0.6]
        [--n-clusters N]
        [--format text|json|markdown]
        [--output FILE]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# Stopwords for keyword extraction (Turkish + English)
STOPWORDS = {
    "ve", "bir", "bu", "ile", "için", "da", "de", "mi", "mu", "mü", "mı", "ne",
    "şu", "o", "the", "a", "an", "is", "are", "was", "were", "in", "of", "to",
    "for", "on", "at", "or", "and", "that", "this", "it", "be", "have", "has",
    "been", "being", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "can", "shall", "just", "can't", "won't", "don't",
    "doesn't", "shouldn't", "wouldn't", "couldn't", "mightn't", "mustn't",
    "as", "with", "from", "by", "up", "about", "into", "through", "during",
    "before", "after", "above", "below", "between", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "only", "same", "so", "than", "too", "very", "s", "t", "can",
    "he", "she", "we", "you", "they", "him", "her", "us", "them", "me", "my",
    "his", "her", "our", "your", "their", "its"
}


def extract_failures(
    report: dict[str, Any],
    threshold: float = 0.6
) -> list[dict[str, Any]]:
    """Extract failed/low-scoring cases from report.

    Walks report["models"][model_key]["tests"] for every model+test.
    Extracts cases where overall_score < threshold OR case has "error".

    Args:
        report: eval report dict
        threshold: score threshold; cases with score < threshold or with "error"
                   are failures

    Returns:
        List of failure dicts with keys: model, test, case_id, score, category,
        text, error
    """
    failures = []

    models = report.get("models", {})
    for model_key, model_data in models.items():
        tests = model_data.get("tests", {})
        for test_name, test_data in tests.items():
            results = test_data.get("results", [])
            for case in results:
                # Check if error or low score
                has_error = "error" in case and case["error"]

                # Extract overall_score from nested or flat dict
                score = 1.0
                if isinstance(case.get("scores"), dict):
                    score = case["scores"].get("overall_score", 1.0)
                else:
                    # Flat dict format
                    score = case.get("overall_score", 1.0)

                if has_error or score < threshold:
                    # Extract text with fallback chain: question→input_text→prompt→case_id
                    text = (
                        case.get("question") or
                        case.get("input_text") or
                        case.get("prompt") or
                        case.get("case_id", "unknown")
                    )

                    failure_dict = {
                        "model": model_key,
                        "test": test_name,
                        "case_id": case.get("case_id", "unknown"),
                        "score": 0.0 if has_error else score,
                        "category": case.get("category", "unknown"),
                        "text": text,
                        "error": case.get("error") if has_error else None,
                    }
                    failures.append(failure_dict)

    return failures


def _tfidf_embed(texts: list[str]) -> np.ndarray:
    """Default embedding function using TF-IDF vectorization.

    Args:
        texts: list of text strings

    Returns:
        (n_samples, n_features) array of TF-IDF vectors
    """
    if not texts:
        return np.array([]).reshape(0, 0)

    vectorizer = TfidfVectorizer(max_features=100, stop_words=None)
    return vectorizer.fit_transform(texts).toarray()


def cluster_failures(
    failures: list[dict[str, Any]],
    n_clusters: Optional[int] = None,
    embed_fn: Optional[Callable[[list[str]], np.ndarray]] = None
) -> list[dict[str, Any]]:
    """Cluster failures by semantic similarity.

    Uses KMeans with injected embedding function. Auto-selects cluster count
    if n_clusters is None: min(max(2, len(failures)//3), 8).

    Args:
        failures: list of failure dicts
        n_clusters: number of clusters (auto-select if None)
        embed_fn: Callable[[list[str]], ndarray]; defaults to TF-IDF

    Returns:
        List of cluster dicts with keys: cluster_id, size, label, members,
        centroid_text, avg_score. Empty list if < 2 failures.
    """
    if len(failures) < 2:
        return []

    if embed_fn is None:
        embed_fn = _tfidf_embed

    texts = [f["text"] for f in failures]
    embeddings = embed_fn(texts)

    # Auto-select n_clusters
    if n_clusters is None:
        n_clusters = min(max(2, len(failures) // 3), 8)

    n_clusters = max(2, min(n_clusters, len(failures)))

    # KMeans clustering (seeded for determinism)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Group failures by cluster
    clusters_dict: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for idx, label in enumerate(labels):
        if label not in clusters_dict:
            clusters_dict[label] = []
        clusters_dict[label].append((idx, failures[idx]))

    # Build cluster dicts
    clusters = []
    for cluster_id, members_with_idx in sorted(clusters_dict.items()):
        member_indices = [idx for idx, _ in members_with_idx]
        members = [f for _, f in members_with_idx]

        # Find member closest to centroid
        centroid = kmeans.cluster_centers_[cluster_id]
        dists = [np.linalg.norm(embeddings[idx] - centroid) for idx in member_indices]
        closest_idx = member_indices[np.argmin(dists)]
        centroid_text = failures[closest_idx]["text"]

        avg_score = float(np.mean([f["score"] for f in members]))

        cluster_dict = {
            "cluster_id": int(cluster_id),
            "size": len(members),
            "label": f"Cluster {cluster_id}",  # overwritten by label_clusters
            "members": members,
            "centroid_text": centroid_text,
            "avg_score": avg_score,
        }
        clusters.append(cluster_dict)

    return clusters


def _default_label_fn(texts: list[str]) -> str:
    """Generate label from most common words in texts.

    Excludes stopwords, returns top 3 words joined by space.

    Args:
        texts: list of text strings

    Returns:
        Label string
    """
    words = []
    for text in texts:
        # Simple tokenization: extract word tokens
        tokens = re.findall(r'\b\w+\b', text.lower())
        words.extend(tokens)

    # Count words, exclude stopwords
    word_counts = Counter(w for w in words if w not in STOPWORDS)
    if not word_counts:
        return "unlabeled"

    top_words = [w for w, _ in word_counts.most_common(3)]
    return " ".join(top_words)


def label_clusters(
    clusters: list[dict[str, Any]],
    label_fn: Optional[Callable[[list[str]], str]] = None
) -> list[dict[str, Any]]:
    """Label clusters using provided or default function.

    Args:
        clusters: list of cluster dicts
        label_fn: Callable[[list[str]], str]; if None, uses keyword extraction

    Returns:
        Clusters with "label" field updated in-place
    """
    if label_fn is None:
        label_fn = _default_label_fn

    for cluster in clusters:
        # Get up to 5 sample texts from members
        sample_texts = [m["text"] for m in cluster["members"][:5]]
        label = label_fn(sample_texts)
        cluster["label"] = label

    return clusters


def compute_failure_summary(
    report: dict[str, Any],
    threshold: float = 0.6,
    n_clusters: Optional[int] = None,
    embed_fn: Optional[Callable[[list[str]], np.ndarray]] = None,
    label_fn: Optional[Callable[[list[str]], str]] = None
) -> dict[str, Any]:
    """Orchestrate failure extraction, clustering, and labeling.

    Args:
        report: eval report dict
        threshold: score threshold
        n_clusters: number of clusters
        embed_fn: embedding function
        label_fn: labeling function

    Returns:
        Summary dict with: total_failures, threshold, clusters, model_breakdown,
        category_breakdown
    """
    failures = extract_failures(report, threshold)

    if not failures:
        return {
            "total_failures": 0,
            "threshold": threshold,
            "clusters": [],
            "model_breakdown": {},
            "category_breakdown": {},
        }

    clusters = cluster_failures(failures, n_clusters, embed_fn)
    clusters = label_clusters(clusters, label_fn)

    # Model breakdown
    model_breakdown = Counter(f["model"] for f in failures)

    # Category breakdown
    category_breakdown = Counter(f["category"] for f in failures)

    return {
        "total_failures": len(failures),
        "threshold": threshold,
        "clusters": clusters,
        "model_breakdown": dict(model_breakdown),
        "category_breakdown": dict(category_breakdown),
    }


def _format_text(summary: dict[str, Any]) -> str:
    """Format summary as plain text."""
    lines = [
        f"Failure Clustering — {summary['total_failures']} failures "
        f"(threshold: {summary['threshold']})",
        ""
    ]

    for cluster in summary["clusters"]:
        lines.append(
            f"Cluster {cluster['cluster_id']} ({cluster['size']} failures, "
            f"avg score: {cluster['avg_score']:.2f}) — \"{cluster['label']}\""
        )

        # Model breakdown for this cluster
        model_counts = Counter(m["model"] for m in cluster["members"])
        models_str = ", ".join(
            f"{m} ({c})" for m, c in model_counts.most_common()
        )
        lines.append(f"  Models: {models_str}")

        # Sample texts
        sample_texts = [m["text"][:50] for m in cluster["members"][:3]]
        examples = ", ".join(f'"{t}"' for t in sample_texts)
        lines.append(f"  Examples: {examples}")
        lines.append("")

    return "\n".join(lines)


def _format_markdown(summary: dict[str, Any]) -> str:
    """Format summary as markdown."""
    lines = [
        "# Failure Clustering\n",
        f"**Total failures:** {summary['total_failures']} "
        f"(threshold: {summary['threshold']})\n",
    ]

    for cluster in summary["clusters"]:
        lines.append(
            f"## Cluster {cluster['cluster_id']}: {cluster['label']}\n"
        )
        lines.append(f"- **Size:** {cluster['size']} failures\n")
        lines.append(f"- **Avg Score:** {cluster['avg_score']:.2f}\n")

        # Model breakdown
        model_counts = Counter(m["model"] for m in cluster["members"])
        lines.append("- **Models:**\n")
        for model, count in model_counts.most_common():
            lines.append(f"  - {model}: {count}\n")

        # Examples
        lines.append("- **Examples:**\n")
        for member in cluster["members"][:3]:
            lines.append(f"  - {member['text'][:60]}\n")
        lines.append("\n")

    # Breakdowns
    lines.append("## Model Breakdown\n")
    for model, count in sorted(summary["model_breakdown"].items()):
        lines.append(f"- {model}: {count}\n")

    lines.append("\n## Category Breakdown\n")
    for category, count in sorted(summary["category_breakdown"].items()):
        lines.append(f"- {category}: {count}\n")

    return "".join(lines)


def _format_json(summary: dict[str, Any]) -> str:
    """Format summary as JSON."""
    return json.dumps(summary, indent=2)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Cluster evaluation failures")
    parser.add_argument("report", help="Path to eval report JSON")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Score threshold for failure extraction (default: 0.6)"
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Number of clusters (auto-select if omitted)"
    )
    parser.add_argument(
        "--format",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format (default: text)"
    )
    parser.add_argument("--output", help="Output file (default: stdout)")

    args = parser.parse_args()

    # Load report
    with open(args.report) as f:
        report = json.load(f)

    # Compute summary
    summary = compute_failure_summary(report, args.threshold, args.n_clusters)

    # Format
    if args.format == "text":
        output = _format_text(summary)
    elif args.format == "markdown":
        output = _format_markdown(summary)
    else:  # json
        output = _format_json(summary)

    # Write
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
