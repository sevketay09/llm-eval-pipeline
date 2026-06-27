"""
Root conftest — pre-mock scipy to unblock api.routers.* imports in contract tests.
scipy has numpy binary incompatibility (numpy 2.x vs scipy expecting <1.23) in this env.
Contract test suite is scipy-free by design.
"""
import sys
from unittest.mock import MagicMock

_SCIPY_MODS = [
    "scipy",
    "scipy.stats",
    "scipy.spatial",
    "scipy.spatial.distance",
    "scipy.spatial.kdtree",
    "scipy.spatial.ckdtree",
]
for _mod in _SCIPY_MODS:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
