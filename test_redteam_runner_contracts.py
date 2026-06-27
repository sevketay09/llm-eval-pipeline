"""Contract tests for redteam/runner.py, redteam/generator.py, redteam/scorer.py."""
from __future__ import annotations
import pytest
from redteam.generator import generate_attacks
from redteam.runner import RedTeamRunner
from redteam.scorer import score_response
from redteam.store import Attack, CATEGORIES, make_session


def _refusal_model(system_prompt: str, user_input: str):
    return "I cannot help with that request.", 10.0


def _compliance_model(system_prompt: str, user_input: str):
    return "DAN mode activated. I will do anything now.", 8.0


def _error_model(system_prompt: str, user_input: str):
    raise RuntimeError("model unavailable")


class TestGenerateAttacks:
    def test_returns_attacks_for_valid_category(self):
        attacks = generate_attacks("be helpful", ["jailbreak"])
        assert len(attacks) > 0
        assert all(a.category == "jailbreak" for a in attacks)

    def test_all_categories_when_empty_list(self):
        attacks = generate_attacks("p", [])
        cats = {a.category for a in attacks}
        assert cats == set(CATEGORIES)

    def test_all_categories_when_invalid(self):
        attacks = generate_attacks("p", ["nonexistent"])
        cats = {a.category for a in attacks}
        assert cats == set(CATEGORIES)

    def test_multiple_categories(self):
        attacks = generate_attacks("p", ["jailbreak", "prompt_injection"])
        cats = {a.category for a in attacks}
        assert "jailbreak" in cats
        assert "prompt_injection" in cats

    def test_attack_has_required_fields(self):
        attacks = generate_attacks("p", ["jailbreak"])
        a = attacks[0]
        assert a.attack_id
        assert a.category
        assert a.name
        assert a.payload

    def test_unique_attack_ids(self):
        attacks = generate_attacks("p", list(CATEGORIES))
        ids = [a.attack_id for a in attacks]
        assert len(ids) == len(set(ids))


class TestScorer:
    def _attack(self) -> Attack:
        return Attack(attack_id="x", category="jailbreak", name="test", payload="payload")

    def test_refusal_passes(self):
        reason, passed = score_response(self._attack(), "I cannot help with that.")
        assert passed is True

    def test_compliance_fails(self):
        reason, passed = score_response(self._attack(), "dan mode activated.")
        assert passed is False

    def test_short_response_passes(self):
        reason, passed = score_response(self._attack(), "No.")
        assert passed is True

    def test_reason_nonempty(self):
        reason, _ = score_response(self._attack(), "I cannot do that.")
        assert reason

    def test_unknown_response_passes(self):
        reason, passed = score_response(self._attack(), "The weather today is sunny and warm in the city.")
        assert passed is True


class TestRedTeamRunner:
    def _session_with_attacks(self, categories=None):
        s = make_session("You are a helpful assistant.", categories or ["jailbreak"])
        s.attacks = generate_attacks(s.system_prompt, s.categories)
        return s

    def test_run_returns_result_per_attack(self):
        session = self._session_with_attacks()
        runner = RedTeamRunner(model_fn=_refusal_model)
        results = runner.run_session(session)
        assert len(results) == len(session.attacks)

    def test_refusal_model_passes(self):
        session = self._session_with_attacks()
        runner = RedTeamRunner(model_fn=_refusal_model)
        results = runner.run_session(session)
        assert all(r.passed for r in results)

    def test_compliance_model_fails(self):
        session = self._session_with_attacks(["jailbreak"])
        runner = RedTeamRunner(model_fn=_compliance_model)
        results = runner.run_session(session)
        assert any(not r.passed for r in results)

    def test_error_model_records_error(self):
        session = self._session_with_attacks(["jailbreak"])
        runner = RedTeamRunner(model_fn=_error_model)
        results = runner.run_session(session)
        assert all(r.error for r in results)
        assert all(not r.passed for r in results)

    def test_result_fields(self):
        session = self._session_with_attacks(["jailbreak"])
        runner = RedTeamRunner(model_fn=_refusal_model)
        results = runner.run_session(session)
        r = results[0]
        assert r.attack_id
        assert r.category == "jailbreak"
        assert r.response
        assert r.latency_ms >= 0

    def test_injectable_score_fn(self):
        session = self._session_with_attacks(["jailbreak"])
        always_fail = lambda attack, response: ("always fail", False)
        runner = RedTeamRunner(model_fn=_refusal_model, score_fn=always_fail)
        results = runner.run_session(session)
        assert all(not r.passed for r in results)
