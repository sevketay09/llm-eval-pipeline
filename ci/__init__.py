"""CI gating package for the LLM Evaluation Pipeline.

Turns evaluation reports into pass/fail CI decisions: threshold checks,
regression comparison against a baseline report, and machine/human readable
renderers (JSON, Markdown PR comment, shields.io badge endpoint, text).

Import the core from ``ci.gate`` and the pytest helpers from
``ci.pytest_plugin`` (kept out of this ``__init__`` so that
``python -m ci.gate`` does not re-import the module).
"""

__all__ = ["gate", "pytest_plugin"]
