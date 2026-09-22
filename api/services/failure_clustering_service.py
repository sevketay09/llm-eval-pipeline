"""Failure clustering service — wraps analysis/failure_clustering."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from api.schemas.failure_clustering import (
    ClusterMemberSchema,
    ClusterSchema,
    FailureClusteringResponse,
)
from analysis.failure_clustering import compute_failure_summary


class FailureClusteringService:
    def __init__(self, decision_client_factory: Optional[Callable[[str, str], Any]] = None) -> None:
        from decisions.config import build_decision_client

        self.decision_client_factory = decision_client_factory or build_decision_client
        self.config_path = "config/models.yaml"

    def cluster(
        self,
        report: Dict[str, Any],
        threshold: float = 0.6,
        n_clusters: Optional[int] = None,
        labeling: str = "keywords",
        decision_model: Optional[str] = None,
        taxonomy: Optional[Dict[str, str]] = None,
    ) -> FailureClusteringResponse:
        classify_fn = None
        if labeling == "decision":
            if not decision_model:
                raise ValueError("decision_model is required when labeling='decision'")
            from decisions.failure_taxonomy import make_classify_fn

            client = self.decision_client_factory(decision_model, self.config_path)
            classify_fn = make_classify_fn(client, taxonomy=taxonomy)

        summary = compute_failure_summary(
            report,
            threshold=threshold,
            n_clusters=n_clusters,
            classify_fn=classify_fn,
        )

        clusters = []
        for c in summary.get("clusters", []):
            members = [
                ClusterMemberSchema(
                    model=m.get("model", ""),
                    test=m.get("test", ""),
                    case_id=m.get("case_id", ""),
                    score=m.get("score", 0.0),
                    category=m.get("category", ""),
                    text=m.get("text", ""),
                    taxonomy_label=m.get("taxonomy_label"),
                )
                for m in c.get("members", [])
            ]
            clusters.append(
                ClusterSchema(
                    cluster_id=c["cluster_id"],
                    size=c["size"],
                    label=c.get("label", f"Cluster {c['cluster_id']}"),
                    centroid_text=c.get("centroid_text", ""),
                    avg_score=c.get("avg_score", 0.0),
                    members=members,
                    taxonomy_label=c.get("taxonomy_label"),
                    label_distribution=c.get("label_distribution", {}),
                )
            )

        return FailureClusteringResponse(
            total_failures=summary["total_failures"],
            threshold=summary["threshold"],
            clusters=clusters,
            model_breakdown=summary.get("model_breakdown", {}),
            category_breakdown=summary.get("category_breakdown", {}),
            taxonomy_breakdown=summary.get("taxonomy_breakdown", {}),
        )
