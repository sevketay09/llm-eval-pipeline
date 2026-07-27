"""Statistical analysis surfaces for evaluation reports.

``analysis.significance`` turns a report's per-test scores into per-model
bootstrap confidence intervals and pairwise model-vs-model significance tests,
so claims like "model A beats B by 3%" can be qualified as significant or not.

Kept out of this ``__init__`` (no eager imports) so ``python -m
analysis.significance`` does not re-import the module.
"""

__all__ = ["significance"]
