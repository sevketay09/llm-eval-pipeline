"""Regression tests: the mock calculator tool must not be an eval()-based
code-execution gadget, while still doing normal arithmetic correctly."""
import pytest

from utils.mock_tools import MockToolEnvironment, safe_calculate


@pytest.fixture
def executor():
    return MockToolEnvironment()


def test_basic_arithmetic(executor):
    result = executor._mock_calculate("2 + 3 * 4")
    assert result["result"] == 14


def test_parentheses_and_division(executor):
    result = executor._mock_calculate("(200 - 50) / 3", precision=2)
    assert result["result"] == round(150 / 3, 2)


def test_negative_and_power(executor):
    result = executor._mock_calculate("-2 ** 3")
    assert result["result"] == -8


@pytest.mark.parametrize("malicious", [
    "__import__('os').system('id')",
    "().__class__.__bases__[0].__subclasses__()",
    "open('/etc/passwd').read()",
    "[x for x in range(1)]",
    "'a' + 'b'",
])
def test_non_numeric_expressions_are_rejected(executor, malicious):
    result = executor._mock_calculate(malicious)
    assert "error" in result
    assert "result" not in result


def test_safe_calculate_matches_python_semantics():
    assert safe_calculate("3 + 4 * 2") == 11
    assert safe_calculate("(1 + 2) * (3 + 4)") == 21
