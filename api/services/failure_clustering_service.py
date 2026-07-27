"""Failure clustering service — wraps analysis/failure_clustering."""
from __future__ import annotations

from typing import Any, Dict, Optional

from api.schemas.failure_clustering import (
    ClusterMemberSchema,
    ClusterSchema,
    FailureClusteringResponse,
)
from analysis.failure_clustering import compute_failure_summary


class FailureClusteringService:
    def cluster(
        self,
        report: Dict[str, Any],
        threshold: float = 0.6,
        n_clusters: Optional[int] = None,
    ) -> FailureClusteringResponse:
        summary = compute_failure_summary(
            report,
            threshold=threshold,
            n_clusters=n_clusters,
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
                )
            )

        return FailureClusteringResponse(
            total_failures=summary["total_failures"],
            threshold=summary["threshold"],
            clusters=clusters,
            model_breakdown=summary.get("model_breakdown", {}),
            category_breakdown=summary.get("category_breakdown", {}),
        )
