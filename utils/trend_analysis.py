"""
Trend Analysis Utility
Historical results comparison and regression detection
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
from utils.evaluation_store import get_runs, DEFAULT_STORE_PATH


class TrendAnalyzer:
    """Analyze trends across multiple evaluation runs"""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
    
    def load_historical_results(
        self,
        model_key: str,
        limit: int = 10,
        suite_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load historical evaluation results for a model
        
        Args:
            model_key: Model identifier
            limit: Maximum number of historical results to load
        
        Returns:
            List of historical results sorted by timestamp (newest first)
        """
        historical = []

        # Prefer unified store (single json)
        try:
            runs = get_runs(DEFAULT_STORE_PATH)
            for run in runs:
                report_suite = run.get("run_metadata", {}).get("test_suite")
                if suite_filter and report_suite != suite_filter:
                    continue

                if model_key in run.get("models", {}):
                    historical.append({
                        "timestamp": run.get("timestamp"),
                        "file": run.get("run_metadata", {}).get("source_file", DEFAULT_STORE_PATH),
                        "suite": report_suite,
                        "results": run["models"][model_key]
                    })

                if len(historical) >= limit:
                    break
        except Exception:
            historical = []

        if historical:
            return historical

        if not self.reports_dir.exists():
            return []
        
        for report_file in sorted(self.reports_dir.glob("eval_*.json"), reverse=True):
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                report_suite = data.get("run_metadata", {}).get("test_suite")
                if suite_filter:
                    if report_suite != suite_filter:
                        continue
                
                if model_key in data.get("models", {}):
                    historical.append({
                        "timestamp": data.get("timestamp"),
                        "file": str(report_file),
                        "suite": report_suite,
                        "results": data["models"][model_key]
                    })
                
                if len(historical) >= limit:
                    break
            except Exception:
                continue
        
        return historical
    
    def calculate_trend(
        self,
        historical: List[Dict[str, Any]],
        metric_path: str
    ) -> Dict[str, Any]:
        """
        Calculate trend for a specific metric
        
        Args:
            historical: List of historical results
            metric_path: Path to metric (e.g., "overall_metrics.weighted_score")
        
        Returns:
            {
                "values": List[float],
                "timestamps": List[str],
                "trend": str ("improving", "declining", "stable"),
                "change_pct": float
            }
        """
        if not historical:
            return {
                "values": [],
                "timestamps": [],
                "trend": "unknown",
                "change_pct": 0
            }
        
        values = []
        timestamps = []
        
        for item in reversed(historical):  # Oldest first for trend
            value = self._get_nested_value(item["results"], metric_path)
            if value is not None:
                values.append(value)
                timestamps.append(item["timestamp"])
        
        if len(values) < 2:
            return {
                "values": values,
                "timestamps": timestamps,
                "trend": "unknown",
                "change_pct": 0
            }
        
        # Calculate trend
        recent_avg = np.mean(values[-3:])  # Last 3
        older_avg = np.mean(values[:3])     # First 3
        
        change_pct = ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
        
        if change_pct > 5:
            trend = "improving"
        elif change_pct < -5:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "values": values,
            "timestamps": timestamps,
            "trend": trend,
            "change_pct": float(change_pct),
            "recent_avg": float(recent_avg),
            "older_avg": float(older_avg)
        }
    
    def detect_regressions(
        self,
        current_results: Dict[str, Any],
        historical: List[Dict[str, Any]],
        threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Detect performance regressions
        
        Args:
            current_results: Current evaluation results
            historical: Historical results
            threshold: Regression threshold (e.g., 0.1 = 10% drop)
        
        Returns:
            List of detected regressions
        """
        if not historical:
            return []
        
        regressions = []
        
        # Get baseline (average of last 3 runs)
        baseline_scores = []
        for item in historical[:3]:
            score = item["results"]["overall_metrics"].get("weighted_score", 0)
            baseline_scores.append(score)
        
        baseline = np.mean(baseline_scores) if baseline_scores else 0
        current_score = current_results["overall_metrics"].get("weighted_score", 0)
        
        # Check for regression
        if baseline > 0:
            drop_pct = (baseline - current_score) / baseline
            if drop_pct > threshold:
                regressions.append({
                    "metric": "overall_score",
                    "baseline": float(baseline),
                    "current": float(current_score),
                    "drop_percentage": float(drop_pct * 100),
                    "severity": "high" if drop_pct > 0.2 else "medium"
                })
        
        # Check per-test regressions
        for test_name, test_data in current_results.get("tests", {}).items():
            if "summary" not in test_data:
                continue
            
            current_test_score = test_data["summary"].get("overall_score", 0)
            
            historical_test_scores = []
            for item in historical[:3]:
                if test_name in item["results"].get("tests", {}):
                    hist_test_data = item["results"]["tests"][test_name]
                    # Skip if historical test has no summary (e.g., had error)
                    if "summary" not in hist_test_data:
                        continue
                    hist_score = hist_test_data["summary"].get("overall_score", 0)
                    historical_test_scores.append(hist_score)
            
            if historical_test_scores:
                test_baseline = np.mean(historical_test_scores)
                if test_baseline > 0:
                    drop_pct = (test_baseline - current_test_score) / test_baseline
                    if drop_pct > threshold:
                        regressions.append({
                            "metric": f"{test_name}_score",
                            "baseline": float(test_baseline),
                            "current": float(current_test_score),
                            "drop_percentage": float(drop_pct * 100),
                            "severity": "high" if drop_pct > 0.2 else "medium"
                        })
        
        return regressions
    
    def generate_comparison_report(
        self,
        model_keys: List[str],
        current_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive comparison report
        
        Returns:
            {
                "model_rankings": List[Dict],
                "trends": Dict,
                "regressions": Dict,
                "recommendations": List[str]
            }
        """
        model_rankings = []
        trends = {}
        regressions = {}
        
        # Rank models
        for model_key in model_keys:
            if model_key not in current_results["models"]:
                continue
            
            model_data = current_results["models"][model_key]
            overall_score = model_data["overall_metrics"].get("weighted_score", 0)
            
            model_rankings.append({
                "model": model_key,
                "overall_score": overall_score,
                "avg_latency": model_data["overall_metrics"].get("latency_avg", 0)
            })
        
        # Sort by score
        model_rankings.sort(key=lambda x: x["overall_score"], reverse=True)
        
        # Analyze trends for each model
        for model_key in model_keys:
            if model_key not in current_results["models"]:
                continue
            
            historical = self.load_historical_results(model_key)
            
            if historical:
                trends[model_key] = self.calculate_trend(
                    historical,
                    "overall_metrics.weighted_score"
                )
                
                regressions[model_key] = self.detect_regressions(
                    current_results["models"][model_key],
                    historical
                )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            model_rankings,
            trends,
            regressions
        )
        
        return {
            "model_rankings": model_rankings,
            "trends": trends,
            "regressions": regressions,
            "recommendations": recommendations
        }
    
    def _get_nested_value(self, data: Dict, path: str) -> Optional[Any]:
        """Get value from nested dict using dot notation"""
        keys = path.split(".")
        value = data
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def _generate_recommendations(
        self,
        rankings: List[Dict],
        trends: Dict,
        regressions: Dict
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if rankings:
            best = rankings[0]
            recommendations.append(
                f"🏆 En iyi performans: {best['model']} (score: {best['overall_score']:.3f})"
            )
        
        # Check for regressions
        for model_key, model_regressions in regressions.items():
            if model_regressions:
                high_severity = [r for r in model_regressions if r['severity'] == 'high']
                if high_severity:
                    recommendations.append(
                        f"⚠️ {model_key}: Ciddi regresyon tespit edildi! {len(high_severity)} kritik düşüş."
                    )
        
        # Check trends
        for model_key, trend_data in trends.items():
            if trend_data["trend"] == "improving":
                recommendations.append(
                    f"📈 {model_key}: Performans artış trendinde (+{trend_data['change_pct']:.1f}%)"
                )
            elif trend_data["trend"] == "declining":
                recommendations.append(
                    f"📉 {model_key}: Performans azalış trendinde ({trend_data['change_pct']:.1f}%)"
                )
        
        return recommendations
