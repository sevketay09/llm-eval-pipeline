"""Contract tests for analysis.run_diff.

Tests case-level comparison between two evaluation reports.
Uses synthetic report payloads so the diff logic is verified without
real evaluation data. All tests are offline and deterministic.
"""
import json

import pytest

from analysis.run_diff import (
    extract_case_scores,
    diff_reports,
    format_diff_text,
    format_diff_json,
    format_diff_markdown,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_case(case_id, score, category="support", question=None, error=None):
    """Create a minimal case dict."""
    case = {
        "case_id": case_id,
        "scores": {"overall_score": score},
        "category": category,
    }
    if question is not None:
        case["question"] = question
    else:
        case["question"] = f"Question {case_id}?"
    if error:
        case["error"] = error
    return case


def _make_report(models_data, weighted_score=0.8, test_score=0.8):
    """Build a report dict from models_data.

    models_data = {model_key: {test_name: [case_dicts]}}
    weighted_score: default model weighted_score
    test_score: default test overall_score
    """
    report = {"models": {}}
    for model, tests in models_data.items():
        report["models"][model] = {
            "overall_metrics": {"weighted_score": weighted_score},
            "tests": {}
        }
        for test, cases in tests.items():
            report["models"][model]["tests"][test] = {
                "summary": {"overall_score": test_score},
                "results": cases
            }
    return report


# --------------------------------------------------------------------------- #
# ExtractCaseScoresContractTests
# --------------------------------------------------------------------------- #


class TestExtractCaseScoresContractTests:
    """Extract case scores from report."""

    def test_extracts_scores_correctly(self):
        """Scores extracted per model+test+case."""
        report = _make_report({
            "gpt-4o": {
                "turkish_qa": [
                    _make_case("case_001", 0.9),
                    _make_case("case_002", 0.7),
                ]
            },
            "claude-3": {
                "turkish_qa": [
                    _make_case("case_001", 0.85),
                ]
            }
        })

        result = extract_case_scores(report)

        assert "gpt-4o" in result
        assert "turkish_qa" in result["gpt-4o"]
        assert result["gpt-4o"]["turkish_qa"]["case_001"]["score"] == 0.9
        assert result["gpt-4o"]["turkish_qa"]["case_002"]["score"] == 0.7

        assert "claude-3" in result
        assert result["claude-3"]["turkish_qa"]["case_001"]["score"] == 0.85

    def test_error_case_score_is_zero(self):
        """Case with 'error' key → score = 0.0."""
        report = _make_report({
            "gpt-4o": {
                "test_a": [
                    _make_case("case_001", 0.9),
                    _make_case("case_002", 0.8, error="timeout"),
                ]
            }
        })

        result = extract_case_scores(report)

        assert result["gpt-4o"]["test_a"]["case_001"]["score"] == 0.9
        assert result["gpt-4o"]["test_a"]["case_002"]["score"] == 0.0

    def test_text_fallback_chain(self):
        """Text extracted: question → input_text → prompt → case_id."""
        # Create cases with and without question
        case_001 = _make_case("case_001", 0.9)
        case_001["question"] = "Hangisi?"
        case_002 = _make_case("case_002", 0.8)
        # Remove question to test fallback
        del case_002["question"]
        case_002["input_text"] = "İnsan mı?"

        report = _make_report({
            "gpt-4o": {
                "test": [case_001, case_002]
            }
        })

        result = extract_case_scores(report)

        assert result["gpt-4o"]["test"]["case_001"]["text"] == "Hangisi?"
        assert result["gpt-4o"]["test"]["case_002"]["text"] == "İnsan mı?"


# --------------------------------------------------------------------------- #
# DiffReportsContractTests
# --------------------------------------------------------------------------- #


class TestDiffReportsContractTests:
    """Case-level diff detection."""

    def test_improved_case_detected(self):
        """Score goes up by ≥ threshold → 'improved'."""
        report_a = _make_report({
            "gpt-4o": {
                "test": [_make_case("case_001", 0.5)]
            }
        })
        report_b = _make_report({
            "gpt-4o": {
                "test": [_make_case("case_001", 0.6)]
            }
        })

        diff = diff_reports(
            report_a,
            report_b,
            improvement_threshold=0.05
        )

        cases = diff["models"]["gpt-4o"]["tests"]["test"]["cases"]
        assert len(cases) == 1
        assert cases[0]["status"] == "improved"
        assert cases[0]["delta"] == pytest.approx(0.1)

    def test_regressed_case_detected(self):
        """Score goes down by ≥ threshold → 'regressed'."""
        report_a = _make_report({
            "gpt-4o": {
                "test": [_make_case("case_001", 0.9)]
            }
        })
        report_b = _make_report({
            "gpt-4o": {
                "test": [_make_case("case_001", 0.75)]
            }
        })

        diff = diff_reports(
            report_a,
            report_b,
            regression_threshold=0.05
        )

        cases = diff["models"]["gpt-4o"]["tests"]["test"]["cases"]
        assert len(cases) == 1
        assert cases[0]["status"] == "regressed"
        assert cases[0]["delta"] == pytest.approx(-0.15)

    def test_unchanged_case_detected(self):
        """Small delta → 'unchanged'."""
        report_a = _make_report({
            "gpt-4o": {
                "test": [_make_case("case_001", 0.80)]
            }
        })
        report_b = _make_report({
            "gpt-4o": {
                "test": [_make_case("case_001", 0.81)]
            }
        })

        diff = diff_reports(
            report_a,
            report_b,
            improvement_threshold=0.05,
            regression_threshold=0.05
        )

        cases = diff["models"]["gpt-4o"]["tests"]["test"]["cases"]
        assert len(cases) == 1
        assert cases[0]["status"] == "unchanged"
        assert cases[0]["delta"] == pytest.approx(0.01)

    def test_added_case_detected(self):
        """Case in B only → 'added'."""
        report_a = _make_report({
            "gpt-4o": {
                "test": []
            }
        })
        report_b = _make_report({
            "gpt-4o": {
                "test": [_make_case("case_new", 0.9)]
            }
        })

        diff = diff_reports(report_a, report_b)

        cases = diff["models"]["gpt-4o"]["tests"]["test"]["cases"]
        assert len(cases) == 1
        assert cases[0]["status"] == "added"
        assert cases[0]["case_id"] == "case_new"
        assert cases[0]["delta"] is None

    def test_removed_case_detected(self):
        """Case in A only → 'removed'."""
        report_a = _make_report({
            "gpt-4o": {
                "test": [_make_case("case_old", 0.8)]
            }
        })
        report_b = _make_report({
            "gpt-4o": {
                "test": []
            }
        })

        diff = diff_reports(report_a, report_b)

        cases = diff["models"]["gpt-4o"]["tests"]["test"]["cases"]
        assert len(cases) == 1
        assert cases[0]["status"] == "removed"
        assert cases[0]["case_id"] == "case_old"
        assert cases[0]["delta"] is None

    def test_summary_counts_correct(self):
        """Mixed cases, verify all summary counters."""
        report_a = _make_report({
            "gpt-4o": {
                "test": [
                    _make_case("case_001", 0.8),   # will regress
                    _make_case("case_002", 0.5),   # will improve
                    _make_case("case_003", 0.7),   # will be unchanged
                    _make_case("case_004", 0.6),   # will be removed
                ]
            }
        })
        report_b = _make_report({
            "gpt-4o": {
                "test": [
                    _make_case("case_001", 0.7),   # regressed
                    _make_case("case_002", 0.65),  # improved
                    _make_case("case_003", 0.71),  # unchanged
                    _make_case("case_005", 0.8),   # added
                ]
            }
        })

        diff = diff_reports(
            report_a,
            report_b,
            improvement_threshold=0.1,
            regression_threshold=0.05
        )

        summary = diff["summary"]
        assert summary["improved"] == 1      # case_002
        assert summary["regressed"] == 1     # case_001
        assert summary["unchanged"] == 1     # case_003
        assert summary["added"] == 1         # case_005
        assert summary["removed"] == 1       # case_004


# --------------------------------------------------------------------------- #
# FormatDiffContractTests
# --------------------------------------------------------------------------- #


class TestFormatDiffContractTests:
    """Output formatting."""

    def test_text_format_contains_summary_line(self):
        """Text output includes summary with 'improved' and 'regressed'."""
        report_a = _make_report({
            "gpt-4o": {
                "test": [
                    _make_case("case_001", 0.9),
                    _make_case("case_002", 0.5),
                ]
            }
        })
        report_b = _make_report({
            "gpt-4o": {
                "test": [
                    _make_case("case_001", 0.7),  # regressed
                    _make_case("case_002", 0.7),  # improved
                ]
            }
        })

        diff = diff_reports(
            report_a,
            report_b,
            improvement_threshold=0.1,
            regression_threshold=0.05
        )

        text = format_diff_text(diff)

        assert "Summary" in text
        assert "improved" in text.lower()
        assert "regressed" in text.lower()

    def test_json_format_is_valid_json(self):
        """JSON output parses successfully."""
        report_a = _make_report({
            "gpt-4o": {
                "test": [_make_case("case_001", 0.8)]
            }
        })
        report_b = _make_report({
            "gpt-4o": {
                "test": [_make_case("case_001", 0.85)]
            }
        })

        diff = diff_reports(report_a, report_b)
        json_text = format_diff_json(diff)

        # Should not raise
        parsed = json.loads(json_text)
        assert "models" in parsed
        assert "summary" in parsed

    def test_markdown_format_has_headers(self):
        """Markdown output contains ## headers."""
        report_a = _make_report({
            "gpt-4o": {
                "test": [_make_case("case_001", 0.8)]
            }
        })
        report_b = _make_report({
            "gpt-4o": {
                "test": [_make_case("case_001", 0.85)]
            }
        })

        diff = diff_reports(report_a, report_b)
        markdown = format_diff_markdown(diff)

        assert "##" in markdown
        assert "Summary" in markdown


# --------------------------------------------------------------------------- #
# DiffReportsEdgeCaseContractTests
# --------------------------------------------------------------------------- #


class TestDiffReportsEdgeCaseContractTests:
    """Edge cases."""

    def test_empty_reports_return_zero_summary(self):
        """Empty reports → all summary counts are zero."""
        report_a = {"models": {}}
        report_b = {"models": {}}

        diff = diff_reports(report_a, report_b)

        summary = diff["summary"]
        assert summary["improved"] == 0
        assert summary["regressed"] == 0
        assert summary["unchanged"] == 0
        assert summary["added"] == 0
        assert summary["removed"] == 0

    def test_model_only_in_one_report_included(self):
        """Model in A only or B only is included in diff."""
        report_a = _make_report({
            "gpt-4o": {
                "test": [_make_case("case_001", 0.8)]
            }
        }, weighted_score=0.8)
        report_b = _make_report({
            "claude-3": {
                "test": [_make_case("case_001", 0.85)]
            }
        }, weighted_score=0.85)

        diff = diff_reports(report_a, report_b)

        assert "gpt-4o" in diff["models"]
        assert "claude-3" in diff["models"]
        assert diff["models"]["gpt-4o"]["score_a"] == 0.8
        assert diff["models"]["gpt-4o"]["score_b"] is None
        assert diff["models"]["claude-3"]["score_a"] is None
        assert diff["models"]["claude-3"]["score_b"] == 0.85
