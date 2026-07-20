"""
Root conftest — pre-mock scipy + sklearn to unblock contract test imports.
scipy/sklearn have numpy 2.x binary incompatibility in this env.
Contract tests are scipy/sklearn-free by design (use injectable fns).
"""
import sys
from types import ModuleType
from unittest.mock import MagicMock

# ── scipy mocks ───────────────────────────────────────────────────────────────

_SCIPY_MODS = [
    "scipy",
    "scipy.stats",
    "scipy.spatial",
    "scipy.spatial.distance",
    "scipy.spatial.kdtree",
    "scipy.spatial.ckdtree",
    "scipy.sparse",
    "scipy.sparse.linalg",
    "scipy.sparse.csgraph",
]
for _mod in _SCIPY_MODS:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# ── sklearn mock — pure-numpy KMeans so injectable-fn tests pass ──────────────

def _make_sklearn_mock() -> ModuleType:
    import numpy as np

    class _KMeans:
        def __init__(self, n_clusters=8, random_state=None, n_init=10, **kw):
            self.n_clusters = n_clusters
            self.random_state = random_state
            self.cluster_centers_ = None

        def fit_predict(self, X):
            rng = np.random.RandomState(self.random_state or 0)
            X = np.asarray(X)
            n = len(X)
            # Init centroids by picking random rows
            idx = rng.choice(n, size=self.n_clusters, replace=False)
            centers = X[idx].copy()
            labels = np.zeros(n, dtype=int)
            # 10 iterations of Lloyd's
            for _ in range(10):
                dists = np.array([[np.linalg.norm(x - c) for c in centers] for x in X])
                labels = np.argmin(dists, axis=1)
                for k in range(self.n_clusters):
                    members = X[labels == k]
                    if len(members) > 0:
                        centers[k] = members.mean(axis=0)
            self.cluster_centers_ = centers
            return labels

    cluster_mod = MagicMock()
    cluster_mod.KMeans = _KMeans

    class _TfidfVectorizer:
        def __init__(self, **kw):
            self._vocab = {}

        def fit_transform(self, texts):
            words = set(w for t in texts for w in t.lower().split())
            self._vocab = {w: i for i, w in enumerate(sorted(words))}
            mat = np.zeros((len(texts), max(len(self._vocab), 1)))
            for i, t in enumerate(texts):
                for w in t.lower().split():
                    if w in self._vocab:
                        mat[i, self._vocab[w]] += 1.0

            class _Sparse:
                def __init__(self, arr):
                    self._arr = arr
                def toarray(self):
                    return self._arr

            return _Sparse(mat)

    text_mod = MagicMock()
    text_mod.TfidfVectorizer = _TfidfVectorizer

    sklearn_mod = MagicMock()
    sklearn_mod.cluster = cluster_mod
    sklearn_mod.feature_extraction = MagicMock()
    sklearn_mod.feature_extraction.text = text_mod

    # Sub-module registrations needed for `from sklearn.X import Y`
    sys.modules["sklearn"] = sklearn_mod
    sys.modules["sklearn.cluster"] = cluster_mod
    sys.modules["sklearn.feature_extraction"] = sklearn_mod.feature_extraction
    sys.modules["sklearn.feature_extraction.text"] = text_mod
    metrics_mod = MagicMock()
    sklearn_mod.metrics = metrics_mod
    sys.modules["sklearn.metrics"] = metrics_mod
    sys.modules["sklearn.metrics.pairwise"] = metrics_mod.pairwise
    sys.modules["sklearn.utils"] = MagicMock()
    sys.modules["sklearn.utils.murmurhash"] = MagicMock()
    sys.modules["sklearn.base"] = MagicMock()

    return sklearn_mod


if "sklearn" not in sys.modules:
    _make_sklearn_mock()
