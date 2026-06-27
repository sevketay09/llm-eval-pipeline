"""
Contract tests for reports/share.py (G13)

All offline — no network calls, no external dependencies.
Covers: build_share_report, format_json, format_markdown, format_html,
        decode_permalink, CLI.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from reports.share import (
    ModelRow,
    ShareReport,
    build_share_report,
    decode_permalink,
    format_html,
    format_json,
    format_markdown,
    main,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _report(**overrides) -> dict:
    base = {
        "models": {
            "gpt-4o": {
                "overall_metrics": {"weighted_score": 0.91, "latency_p95": 1.8},
                "tests": {
                    "grammar": {"summary": {"overall_score": 0.93}},
                    "function_calling": {"summary": {"overall_score": 0.89}},
                },
            },
            "qwen-72b": {
                "overall_metrics": {"weighted_score": 0.75, "latency_p95": 3.1},
                "tests": {
                    "grammar": {"summary": {"overall_score": 0.78}},
                    "function_calling": {"summary": {"overall_score": 0.72}},
                },
            },
        },
        "metadata": {"run_id": "test-001"},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. build_share_report — leaderboard ordering
# ---------------------------------------------------------------------------

def test_leaderboard_sorted_by_score_desc():
    share = build_share_report(_report())
    scores = [r.weighted_score for r in share.leaderboard]
    assert scores == sorted(scores, reverse=True)


def test_leaderboard_ranks_assigned():
    share = build_share_report(_report())
    ranks = [r.rank for r in share.leaderboard]
    assert ranks == list(range(1, len(share.leaderboard) + 1))


def test_leaderboard_model_names_present():
    share = build_share_report(_report())
    names = {r.model for r in share.leaderboard}
    assert "gpt-4o" in names
    assert "qwen-72b" in names


def test_leaderboard_test_scores_extracted():
    share = build_share_report(_report())
    top = share.leaderboard[0]
    assert "grammar" in top.test_scores
    assert "function_calling" in top.test_scores
    assert abs(top.test_scores["grammar"] - 0.93) < 1e-6


def test_leaderboard_latency_extracted():
    share = build_share_report(_report())
    row = next(r for r in share.leaderboard if r.model == "gpt-4o")
    assert row.latency_p95 == pytest.approx(1.8, abs=1e-4)


def test_fingerprint_is_12_chars():
    share = build_share_report(_report())
    for row in share.leaderboard:
        assert len(row.fingerprint) == 12


def test_fingerprint_deterministic():
    s1 = build_share_report(_report())
    s2 = build_share_report(_report())
    fps1 = {r.model: r.fingerprint for r in s1.leaderboard}
    fps2 = {r.model: r.fingerprint for r in s2.leaderboard}
    assert fps1 == fps2


def test_empty_models():
    share = build_share_report({"models": {}})
    assert share.leaderboard == []


def test_missing_latency_is_none():
    r = _report()
    del r["models"]["gpt-4o"]["overall_metrics"]["latency_p95"]
    share = build_share_report(r)
    row = next(x for x in share.leaderboard if x.model == "gpt-4o")
    assert row.latency_p95 is None


def test_test_with_error_skipped():
    r = _report()
    r["models"]["gpt-4o"]["tests"]["broken"] = {"error": "boom"}
    share = build_share_report(r)
    row = next(x for x in share.leaderboard if x.model == "gpt-4o")
    assert "broken" not in row.test_scores


def test_embed_snippet_in_output():
    share = build_share_report(_report())
    assert "<iframe" in share.embed_snippet
    assert "leaderboard" in share.embed_snippet


def test_custom_title():
    share = build_share_report(_report(), title="My Custom Eval")
    assert share.title == "My Custom Eval"


# ---------------------------------------------------------------------------
# 2. Permalink
# ---------------------------------------------------------------------------

def test_permalink_empty_by_default():
    share = build_share_report(_report())
    assert share.permalink == ""


def test_permalink_generated_when_requested():
    share = build_share_report(_report(), include_permalink=True)
    assert share.permalink != ""
    assert "#data=" in share.permalink


def test_permalink_decode_roundtrip():
    original = _report()
    share = build_share_report(original, include_permalink=True)
    b64 = share.permalink.split("#data=")[1]
    decoded = decode_permalink(b64)
    assert decoded["models"].keys() == original["models"].keys()


# ---------------------------------------------------------------------------
# 3. format_json
# ---------------------------------------------------------------------------

def test_format_json_parseable():
    share = build_share_report(_report())
    out = format_json(share)
    data = json.loads(out)
    assert "leaderboard" in data
    assert "generated_at" in data
    assert "embed_snippet" in data


def test_format_json_leaderboard_count():
    share = build_share_report(_report())
    data = json.loads(format_json(share))
    assert len(data["leaderboard"]) == 2


# ---------------------------------------------------------------------------
# 4. format_markdown
# ---------------------------------------------------------------------------

def test_format_markdown_has_table():
    share = build_share_report(_report())
    md = format_markdown(share)
    assert "| Rank |" in md
    assert "| Model |" in md


def test_format_markdown_model_names():
    share = build_share_report(_report())
    md = format_markdown(share)
    assert "gpt-4o" in md
    assert "qwen-72b" in md


def test_format_markdown_embed_section():
    share = build_share_report(_report())
    md = format_markdown(share)
    assert "```html" in md
    assert "<iframe" in md


# ---------------------------------------------------------------------------
# 5. format_html
# ---------------------------------------------------------------------------

def test_format_html_valid_doctype():
    share = build_share_report(_report())
    html = format_html(share)
    assert html.strip().startswith("<!DOCTYPE html>")


def test_format_html_og_meta_tags():
    share = build_share_report(_report())
    html = format_html(share)
    assert 'property="og:title"' in html
    assert 'property="og:description"' in html
    assert 'name="twitter:card"' in html


def test_format_html_model_names_present():
    share = build_share_report(_report())
    html = format_html(share)
    assert "gpt-4o" in html
    assert "qwen-72b" in html


def test_format_html_scores_rendered():
    share = build_share_report(_report())
    html = format_html(share)
    assert "91" in html or "91.0" in html   # 0.91 → 91%


def test_format_html_embed_textarea():
    share = build_share_report(_report())
    html = format_html(share)
    assert "<textarea" in html
    assert "<iframe" in html


def test_format_html_no_external_resources():
    share = build_share_report(_report())
    html = format_html(share)
    # No external <script src=...> or <link href=...> loading from CDN
    import re
    external_scripts = re.findall(r'<script[^>]+src=["\']https?://', html)
    external_links = re.findall(r'<link[^>]+href=["\']https?://', html)
    assert external_scripts == [], f"External scripts found: {external_scripts}"
    assert external_links == [], f"External links found: {external_links}"


def test_format_html_empty_leaderboard():
    share = build_share_report({"models": {}})
    html = format_html(share)
    assert "<!DOCTYPE html>" in html  # doesn't crash


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

def test_cli_demo_html(capsys):
    rc = main(["--demo", "--format", "html"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "<!DOCTYPE html>" in out
    assert "gpt-4o" in out


def test_cli_demo_json(capsys):
    rc = main(["--demo", "--format", "json"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert len(data["leaderboard"]) == 3


def test_cli_demo_markdown(capsys):
    rc = main(["--demo", "--format", "markdown"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "| Rank |" in out


def test_cli_no_args_returns_2(capsys):
    rc = main([])
    assert rc == 2


def test_cli_file_input(tmp_path, capsys):
    f = tmp_path / "report.json"
    f.write_text(json.dumps(_report()))
    rc = main([str(f), "--format", "json"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert "leaderboard" in data


def test_cli_output_file(tmp_path, capsys):
    out_file = tmp_path / "report.html"
    f = tmp_path / "r.json"
    f.write_text(json.dumps(_report()))
    rc = main([str(f), "--output", str(out_file)])
    assert rc == 0
    assert out_file.exists()
    content = out_file.read_text()
    assert "<!DOCTYPE html>" in content


def test_cli_custom_title(capsys):
    rc = main(["--demo", "--format", "json", "--title", "My Eval Board"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert data["title"] == "My Eval Board"


def test_cli_permalink_flag(capsys):
    rc = main(["--demo", "--format", "json", "--permalink"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert "#data=" in data["permalink"]
