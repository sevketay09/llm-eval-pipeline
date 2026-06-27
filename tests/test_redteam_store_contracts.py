"""Contract tests for redteam/store.py."""
from __future__ import annotations
import pytest
from redteam.store import Attack, AttackResult, RedTeamSession, RedTeamStore, CATEGORIES, make_session


def _make_attack(idx: int = 0) -> Attack:
    return Attack(attack_id=f"aid{idx}", category="jailbreak", name=f"attack{idx}", payload="payload")


def _make_result(attack: Attack, passed: bool = True) -> AttackResult:
    return AttackResult(
        attack_id=attack.attack_id,
        category=attack.category,
        name=attack.name,
        payload=attack.payload,
        response="I cannot do that",
        passed=passed,
        reason="refused",
        latency_ms=5.0,
    )


class TestMakeSession:
    def test_creates_with_id(self):
        s = make_session("be helpful", ["jailbreak"])
        assert s.session_id
        assert s.system_prompt == "be helpful"
        assert s.categories == ["jailbreak"]
        assert s.status == "pending"

    def test_custom_id(self):
        s = make_session("p", ["jailbreak"], session_id="custom123")
        assert s.session_id == "custom123"

    def test_empty_attacks_by_default(self):
        s = make_session("p", ["jailbreak"])
        assert s.attacks == []
        assert s.results == []


class TestCategories:
    def test_all_categories_present(self):
        expected = {"prompt_injection", "jailbreak", "persona_override", "boundary_test", "role_confusion"}
        assert set(CATEGORIES) == expected


class TestRedTeamStore:
    def test_create_and_get(self):
        store = RedTeamStore()
        s = make_session("p", ["jailbreak"])
        store.create(s)
        assert store.get(s.session_id) is s

    def test_get_missing_returns_none(self):
        store = RedTeamStore()
        assert store.get("nonexistent") is None

    def test_list_empty(self):
        store = RedTeamStore()
        assert store.list() == []

    def test_list_multiple(self):
        store = RedTeamStore()
        s1 = make_session("p1", ["jailbreak"])
        s2 = make_session("p2", ["jailbreak"])
        store.create(s1)
        store.create(s2)
        lst = store.list()
        assert len(lst) == 2

    def test_update(self):
        store = RedTeamStore()
        s = make_session("p", ["jailbreak"])
        store.create(s)
        s.status = "done"
        store.update(s)
        assert store.get(s.session_id).status == "done"

    def test_count(self):
        store = RedTeamStore()
        store.create(make_session("p1", ["jailbreak"]))
        store.create(make_session("p2", ["jailbreak"]))
        assert store.count() == 2

    def test_fifo_eviction(self):
        from redteam.store import _MAX_SESSIONS
        store = RedTeamStore()
        ids = []
        for i in range(_MAX_SESSIONS + 2):
            s = make_session(f"p{i}", ["jailbreak"])
            store.create(s)
            ids.append(s.session_id)
        assert store.count() == _MAX_SESSIONS
        assert store.get(ids[0]) is None
        assert store.get(ids[1]) is None
        assert store.get(ids[-1]) is not None

    def test_to_dict(self):
        s = make_session("p", ["jailbreak"])
        s.attacks = [_make_attack()]
        s.results = [_make_result(s.attacks[0])]
        d = s.to_dict()
        assert d["system_prompt"] == "p"
        assert len(d["attacks"]) == 1
        assert len(d["results"]) == 1


class TestAttackToDict:
    def test_keys(self):
        a = _make_attack()
        d = a.to_dict()
        assert set(d.keys()) == {"attack_id", "category", "name", "payload"}


class TestAttackResultToDict:
    def test_keys(self):
        a = _make_attack()
        r = _make_result(a)
        d = r.to_dict()
        assert "passed" in d
        assert "reason" in d
        assert "latency_ms" in d
        assert "error" in d
