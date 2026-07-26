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


def _pure_ttest_1samp(arr, popmean):
    """Pure-Python one-sample t-test (normal-approx p) for the scipy.stats mock."""
    import math

    xs = [float(x) - popmean for x in arr]
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    sd = var ** 0.5
    if sd == 0:
        return (float("inf") if mean else 0.0, 0.0 if mean else 1.0)
    t = mean / (sd / n ** 0.5)
    p = math.erfc(abs(t) / math.sqrt(2))  # two-sided, normal approximation
    return (t, p)


if isinstance(sys.modules["scipy.stats"], MagicMock):
    sys.modules["scipy.stats"].ttest_1samp = _pure_ttest_1samp
    sys.modules["scipy.stats"].wilcoxon = lambda arr, **kw: (0.0, 1.0)
    if isinstance(sys.modules["scipy"], MagicMock):
        sys.modules["scipy"].stats = sys.modules["scipy.stats"]

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

    def _cosine_similarity(X, Y=None):
        X = np.asarray(X, dtype=float)
        Y = X if Y is None else np.asarray(Y, dtype=float)
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
        return X_norm @ Y_norm.T

    def _average_precision_score(y_true, y_score):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score, dtype=float)
        n_pos = int(np.sum(y_true == 1))
        if n_pos == 0:
            return 0.0
        order = np.argsort(-y_score, kind="mergesort")
        y_true_sorted = y_true[order]
        tp_cumsum = np.cumsum(y_true_sorted == 1)
        precision_at_k = tp_cumsum / np.arange(1, len(y_true_sorted) + 1)
        return float(np.sum(precision_at_k[y_true_sorted == 1]) / n_pos)

    def _silhouette_score(X, labels, metric="euclidean", **kw):
        X = np.asarray(X, dtype=float)
        labels = np.asarray(labels)
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return 0.0
        if metric == "cosine":
            dist = 1.0 - _cosine_similarity(X, X)
        else:
            dist = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
        scores = []
        for i in range(len(X)):
            own = labels[i]
            same_mask = (labels == own)
            same_mask[i] = False
            a = dist[i, same_mask].mean() if same_mask.any() else 0.0
            b_candidates = []
            for other in unique_labels:
                if other == own:
                    continue
                other_mask = labels == other
                if other_mask.any():
                    b_candidates.append(dist[i, other_mask].mean())
            b = min(b_candidates) if b_candidates else 0.0
            scores.append(0.0 if max(a, b) == 0 else (b - a) / max(a, b))
        return float(np.mean(scores))

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
    metrics_mod.average_precision_score = _average_precision_score
    metrics_mod.silhouette_score = _silhouette_score
    pairwise_mod = MagicMock()
    pairwise_mod.cosine_similarity = _cosine_similarity
    metrics_mod.pairwise = pairwise_mod
    sklearn_mod.metrics = metrics_mod
    sys.modules["sklearn.metrics"] = metrics_mod
    sys.modules["sklearn.metrics.pairwise"] = pairwise_mod
    sys.modules["sklearn.utils"] = MagicMock()
    sys.modules["sklearn.utils.murmurhash"] = MagicMock()
    sys.modules["sklearn.base"] = MagicMock()

    return sklearn_mod


if "sklearn" not in sys.modules:
    _make_sklearn_mock()


# ── event-loop guard ──────────────────────────────────────────────────────────
# Python 3.9: asyncio.Lock() at service __init__ needs a current event loop.
# Tests that call asyncio.run() leave none behind, breaking later router tests.
import asyncio

import pytest


@pytest.fixture(autouse=True)
def _ensure_event_loop():
    try:
        loop = asyncio.get_event_loop_policy().get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    yield
