"""Decision tree classifier built from scratch on entropy and information gain.

Only NumPy is used for the learning algorithm. The tree grows binary splits of
the form ``x[feature] <= threshold`` and picks, at every node, the split with
the largest information gain::

    H(S)      = - sum_k p_k * log2(p_k)
    IG(S, s)  = H(S) - |S_left|/|S| * H(S_left) - |S_right|/|S| * H(S_right)

Nodes are expanded best-first (highest gain first). Without ``max_leaf_nodes``
this produces exactly the same tree as the classic recursive depth-first
algorithm; with it, the tree keeps only the most informative splits.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ["DecisionTreeClassifier", "Node", "entropy", "information_gain"]


def entropy(counts) -> float:
    """Shannon entropy (base 2) of a vector of class counts."""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum()) + 0.0  # + 0.0 turns -0.0 into 0.0


def information_gain(parent_counts, left_counts, right_counts) -> float:
    """Information gain of splitting ``parent`` into ``left`` and ``right``."""
    n_left = float(np.sum(left_counts))
    n_right = float(np.sum(right_counts))
    n = n_left + n_right
    if n == 0:
        return 0.0
    children = (n_left * entropy(left_counts) + n_right * entropy(right_counts)) / n
    return entropy(parent_counts) - children


def _row_entropy(counts: np.ndarray, totals: np.ndarray) -> np.ndarray:
    """Entropy of every row of a (m, K) count matrix, vectorised."""
    with np.errstate(divide="ignore", invalid="ignore"):
        p = counts / totals[:, None]
        logp = np.where(p > 0, np.log2(p), 0.0)
    return -(p * logp).sum(axis=1)


@dataclass
class Node:
    """One node of the tree. Leaves have ``feature is None``."""

    value: np.ndarray  # class counts of the training samples reaching the node
    n_samples: int
    entropy: float
    depth: int
    feature: Optional[int] = None
    threshold: Optional[float] = None
    gain: float = 0.0
    left: Optional["Node"] = None  # samples with x[feature] <= threshold
    right: Optional["Node"] = None
    node_id: int = 0
    _indices: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def is_leaf(self) -> bool:
        return self.feature is None

    @property
    def prediction(self) -> int:
        """Index (into ``classes_``) of the majority class."""
        return int(np.argmax(self.value))


class DecisionTreeClassifier:
    """Entropy / information-gain decision tree classifier.

    Parameters
    ----------
    max_depth : int or None
        Maximum depth of the tree (root has depth 0). ``None`` = unlimited.
    min_samples_split : int
        Minimum number of samples a node needs before it may be split.
    min_samples_leaf : int
        Minimum number of samples each child of a split must contain.
    min_information_gain : float
        A split is only made if its information gain is strictly greater than
        this value (in bits). ``0.0`` means "any gain at all".
    max_leaf_nodes : int or None
        Maximum number of leaves. Splits are made in order of decreasing
        information gain until the budget is used up.
    max_features : int, float, "sqrt", "log2" or None
        Number of features examined at each node (random subset). ``None``
        examines every feature, which makes the tree deterministic.
    random_state : int or None
        Seed for the feature subsampling.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_information_gain: float = 0.0,
        max_leaf_nodes: Optional[int] = None,
        max_features=None,
        random_state: Optional[int] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.max_leaf_nodes = max_leaf_nodes
        self.max_features = max_features
        self.random_state = random_state

    # ------------------------------------------------------------------ params
    def get_params(self, deep: bool = True) -> dict:
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_information_gain": self.min_information_gain,
            "max_leaf_nodes": self.max_leaf_nodes,
            "max_features": self.max_features,
            "random_state": self.random_state,
        }

    def set_params(self, **params) -> "DecisionTreeClassifier":
        for key, value in params.items():
            if key not in self.get_params():
                raise ValueError(f"Invalid parameter {key!r}")
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        defaults = DecisionTreeClassifier().get_params()
        changed = {k: v for k, v in self.get_params().items() if v != defaults[k]}
        args = ", ".join(f"{k}={v!r}" for k, v in changed.items())
        return f"DecisionTreeClassifier({args})"

    def _validate_params(self) -> None:
        if self.max_depth is not None and (not isinstance(self.max_depth, (int, np.integer)) or self.max_depth < 0):
            raise ValueError("max_depth must be None or a non-negative integer")
        if not isinstance(self.min_samples_split, (int, np.integer)) or self.min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer >= 2")
        if not isinstance(self.min_samples_leaf, (int, np.integer)) or self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be an integer >= 1")
        if self.min_information_gain < 0:
            raise ValueError("min_information_gain must be >= 0")
        if self.max_leaf_nodes is not None and (
            not isinstance(self.max_leaf_nodes, (int, np.integer)) or self.max_leaf_nodes < 1
        ):
            raise ValueError("max_leaf_nodes must be None or an integer >= 1")

    def _n_features_per_node(self, n_features: int) -> int:
        mf = self.max_features
        if mf is None:
            return n_features
        if mf == "sqrt":
            k = int(np.sqrt(n_features))
        elif mf == "log2":
            k = int(np.log2(n_features))
        elif isinstance(mf, (float, np.floating)):
            if not 0.0 < mf <= 1.0:
                raise ValueError("float max_features must be in (0, 1]")
            k = int(mf * n_features)
        elif isinstance(mf, (int, np.integer)):
            if not 1 <= mf <= n_features:
                raise ValueError("int max_features must be in [1, n_features]")
            k = int(mf)
        else:
            raise ValueError(f"Invalid max_features: {mf!r}")
        return max(1, k)

    # --------------------------------------------------------------------- fit
    def fit(self, X, y) -> "DecisionTreeClassifier":
        self._validate_params()
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        X = self._check_X(X, fitting=True)
        y = np.asarray(y).ravel()
        if len(y) != X.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but y has {len(y)} labels")
        if len(y) == 0:
            raise ValueError("Cannot fit on an empty dataset")

        self.classes_, y_enc = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        self._X, self._y = X, y_enc
        self._rng = np.random.default_rng(self.random_state)
        self._k_features = self._n_features_per_node(self.n_features_in_)
        self._importances = np.zeros(self.n_features_in_)
        ids = itertools.count()

        root = self._make_node(np.arange(len(y_enc)), depth=0, node_id=next(ids))
        self.root_ = root
        n_leaves = 1
        heap: list = []
        tie = itertools.count()  # keeps heap order stable when gains are equal
        self._push_split(heap, tie, root)

        while heap:
            if self.max_leaf_nodes is not None and n_leaves >= self.max_leaf_nodes:
                break
            _, _, node, (feat, thr, gain, left_idx, right_idx) = heapq.heappop(heap)
            node.feature, node.threshold, node.gain = feat, thr, gain
            node.left = self._make_node(left_idx, node.depth + 1, next(ids))
            node.right = self._make_node(right_idx, node.depth + 1, next(ids))
            node._indices = None
            n_leaves += 1
            self._importances[feat] += node.n_samples / len(y_enc) * gain
            self._push_split(heap, tie, node.left)
            self._push_split(heap, tie, node.right)

        # free memory held by unexpanded leaves
        for _, _, node, _ in heap:
            node._indices = None
        self._clear_indices(root)
        del self._X, self._y
        return self

    def _clear_indices(self, node: Node) -> None:
        stack = [node]
        while stack:
            n = stack.pop()
            n._indices = None
            if not n.is_leaf:
                stack.extend((n.left, n.right))

    def _make_node(self, indices: np.ndarray, depth: int, node_id: int) -> Node:
        counts = np.bincount(self._y[indices], minlength=self.n_classes_)
        return Node(
            value=counts,
            n_samples=len(indices),
            entropy=entropy(counts),
            depth=depth,
            node_id=node_id,
            _indices=indices,
        )

    def _push_split(self, heap, tie, node: Node) -> None:
        """Find the best split of ``node`` and queue it if allowed."""
        if self.max_depth is not None and node.depth >= self.max_depth:
            return
        if node.n_samples < self.min_samples_split or node.n_samples < 2 * self.min_samples_leaf:
            return
        if node.entropy == 0.0:  # pure node
            return
        split = self._best_split(node)
        if split is None:
            return
        gain = split[2]
        if gain <= self.min_information_gain or gain <= 1e-12:
            return
        heapq.heappush(heap, (-gain, next(tie), node, split))

    def _best_split(self, node: Node):
        idx = node._indices
        X, y = self._X[idx], self._y[idx]
        n = len(idx)
        msl = self.min_samples_leaf

        features = np.arange(self.n_features_in_)
        if self._k_features < self.n_features_in_:
            features = self._rng.choice(features, size=self._k_features, replace=False)

        one_hot = np.eye(self.n_classes_)[y]
        total = node.value.astype(float)
        n_left = np.arange(1, n, dtype=float)
        n_right = n - n_left
        # a split position i puts sorted rows [0..i] on the left
        size_ok = (n_left >= msl) & (n_right >= msl)

        best = None
        best_gain = -np.inf
        for f in features:
            order = np.argsort(X[:, f], kind="stable")
            xs = X[order, f]
            left_counts = np.cumsum(one_hot[order], axis=0)[:-1]
            right_counts = total - left_counts
            # can only cut between two different values
            valid = size_ok & (xs[:-1] < xs[1:])
            if not valid.any():
                continue
            h_left = _row_entropy(left_counts, n_left)
            h_right = _row_entropy(right_counts, n_right)
            gains = node.entropy - (n_left * h_left + n_right * h_right) / n
            gains = np.where(valid, gains, -np.inf)
            i = int(np.argmax(gains))
            if gains[i] > best_gain + 1e-12:
                thr = (xs[i] + xs[i + 1]) / 2.0
                if thr >= xs[i + 1]:  # float rounding on very close values
                    thr = xs[i]
                best_gain = float(gains[i])
                best = (int(f), float(thr), i)

        if best is None:
            return None
        f, thr, _ = best
        go_left = X[:, f] <= thr
        return f, thr, best_gain, idx[go_left], idx[~go_left]

    # ----------------------------------------------------------------- predict
    def _check_is_fitted(self) -> None:
        if not hasattr(self, "root_"):
            raise RuntimeError("This DecisionTreeClassifier is not fitted yet; call fit() first.")

    def _check_X(self, X, fitting: bool = False) -> np.ndarray:
        try:
            X = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("X must contain only numeric values") from exc
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if np.isnan(X).any():
            raise ValueError("X contains NaN values")
        if not fitting and X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, the tree was fitted with {self.n_features_in_}")
        return X

    def apply(self, X) -> np.ndarray:
        """Return the ``node_id`` of the leaf each sample ends up in."""
        self._check_is_fitted()
        return np.array([leaf.node_id for leaf in self._leaves_for(self._check_X(X))])

    def _leaves_for(self, X: np.ndarray) -> list:
        out = [None] * X.shape[0]
        stack = [(self.root_, np.arange(X.shape[0]))]
        while stack:
            node, rows = stack.pop()
            if node.is_leaf:
                for r in rows:
                    out[r] = node
                continue
            go_left = X[rows, node.feature] <= node.threshold
            stack.append((node.left, rows[go_left]))
            stack.append((node.right, rows[~go_left]))
        return out

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities = class frequencies in the leaf."""
        self._check_is_fitted()
        leaves = self._leaves_for(self._check_X(X))
        counts = np.array([leaf.value for leaf in leaves], dtype=float).reshape(-1, self.n_classes_)
        return counts / counts.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        self._check_is_fitted()
        leaves = self._leaves_for(self._check_X(X))
        return self.classes_[np.array([leaf.prediction for leaf in leaves], dtype=int)]

    def score(self, X, y) -> float:
        """Mean accuracy."""
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))

    # ------------------------------------------------------------- inspection
    @property
    def feature_importances_(self) -> np.ndarray:
        """Total information gain contributed by each feature, normalised to 1."""
        self._check_is_fitted()
        total = self._importances.sum()
        return self._importances / total if total > 0 else self._importances.copy()

    def nodes(self) -> list:
        """All nodes in depth-first (pre-order) order."""
        self._check_is_fitted()
        out, stack = [], [self.root_]
        while stack:
            node = stack.pop()
            out.append(node)
            if not node.is_leaf:
                stack.extend((node.right, node.left))
        return out

    def get_depth(self) -> int:
        return max(node.depth for node in self.nodes())

    def get_n_leaves(self) -> int:
        return sum(node.is_leaf for node in self.nodes())

    @property
    def node_count(self) -> int:
        return len(self.nodes())

    def export_text(self, **kwargs) -> str:
        """Shortcut for :func:`entropy_tree.export_text`."""
        from .display import export_text

        return export_text(self, **kwargs)

    def plot(self, **kwargs):
        """Shortcut for :func:`entropy_tree.plot_tree`."""
        from .display import plot_tree

        return plot_tree(self, **kwargs)
