import numpy as np


def entropy(counts):
    counts = np.asarray(counts, dtype=float)
    total = np.sum(counts)
    if total <= 0:
        return 0.0
    probs = counts[counts > 0] / total
    h = -np.sum(probs * np.log2(probs))
    return float(max(0.0, h))


def information_gain(parent_counts, left_counts, right_counts):
    n_left = np.sum(left_counts)
    n_right = np.sum(right_counts)
    total = n_left + n_right
    if total == 0 or n_left == 0 or n_right == 0:
        return 0.0
    h_children = (n_left / total) * entropy(left_counts) + (n_right / total) * entropy(right_counts)
    return float(entropy(parent_counts) - h_children)


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, entropy=0.0, gain=0.0, n_samples=0, depth=0, node_id=0):
        self.feature = feature        # feature index used to split
        self.threshold = threshold    # threshold value (x <= threshold goes left)
        self.left = left              # left child node
        self.right = right            # right child node
        self.value = value            # class counts at this node
        self.entropy = entropy        # node entropy
        self.gain = gain              # information gain from this split
        self.n_samples = n_samples    # number of samples at this node
        self.depth = depth            # depth in the tree (root = 0)
        self.node_id = node_id        # identifier for visualization

    @property
    def is_leaf(self):
        return self.feature is None

    @property
    def prediction(self):
        return int(np.argmax(self.value))


class DecisionTreeClassifier:
    """A clean, simple binary decision tree classifier using entropy and information gain."""

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_information_gain=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.root_ = None

    def fit(self, X, y):
        if self.max_depth is not None and self.max_depth < 0:
            raise ValueError("max_depth must be non-negative or None")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        if self.min_information_gain < 0:
            raise ValueError("min_information_gain must be >= 0")

        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array([str(c) for c in X.columns], dtype=object)

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if len(X) == 0:
            raise ValueError("Cannot fit on empty dataset")
        if np.isnan(X).any():
            raise ValueError("X contains NaN values")

        y = np.asarray(y).ravel()
        if len(y) != len(X):
            raise ValueError("X and y must have the same number of rows")

        self.classes_, y_enc = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        self._n_total_samples = len(y_enc)
        self._importances = np.zeros(self.n_features_in_)
        self._node_id_counter = 0

        self.root_ = self._build_tree(X, y_enc, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        counts = np.bincount(y, minlength=self.n_classes_)
        node_entropy = entropy(counts)

        node = Node(
            value=counts,
            entropy=node_entropy,
            n_samples=n_samples,
            depth=depth,
            node_id=self._node_id_counter,
        )
        self._node_id_counter += 1

        # Check stopping conditions: max depth, too few samples to split, or pure node
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           node_entropy == 0.0:
            return node

        best_gain = 0.0
        best_feat = None
        best_thresh = None

        # Try every feature and candidate threshold
        for feat in range(n_features):
            vals = np.sort(np.unique(X[:, feat]))
            if len(vals) <= 1:
                continue

            thresholds = (vals[:-1] + vals[1:]) / 2.0
            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                n_left = np.sum(left_mask)
                n_right = n_samples - n_left

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                left_counts = np.bincount(y[left_mask], minlength=self.n_classes_)
                right_counts = counts - left_counts

                gain = information_gain(counts, left_counts, right_counts)
                if gain > best_gain and gain > self.min_information_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        if best_feat is None:
            return node

        node.feature = best_feat
        node.threshold = best_thresh
        node.gain = best_gain

        self._importances[best_feat] += (n_samples / self._n_total_samples) * best_gain

        left_mask = X[:, best_feat] <= best_thresh
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def _check_is_fitted(self):
        if self.root_ is None:
            raise RuntimeError("This tree is not fitted yet. Call fit() first.")

    def _check_X(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        return X

    def _predict_one(self, node, x):
        if node.is_leaf:
            return self.classes_[node.prediction]
        if x[node.feature] <= node.threshold:
            return self._predict_one(node.left, x)
        return self._predict_one(node.right, x)

    def _predict_proba_one(self, node, x):
        if node.is_leaf:
            total = np.sum(node.value)
            return node.value / total if total > 0 else np.zeros(self.n_classes_)
        if x[node.feature] <= node.threshold:
            return self._predict_proba_one(node.left, x)
        return self._predict_proba_one(node.right, x)

    def predict(self, X):
        self._check_is_fitted()
        X = self._check_X(X)
        return np.array([self._predict_one(self.root_, row) for row in X])

    def predict_proba(self, X):
        self._check_is_fitted()
        X = self._check_X(X)
        return np.array([self._predict_proba_one(self.root_, row) for row in X])

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))

    @property
    def feature_importances_(self):
        self._check_is_fitted()
        total = np.sum(self._importances)
        if total > 0:
            return self._importances / total
        return self._importances.copy()

    def nodes(self):
        self._check_is_fitted()
        result = []
        def traverse(node):
            if node is not None:
                result.append(node)
                traverse(node.left)
                traverse(node.right)
        traverse(self.root_)
        return result

    def get_depth(self):
        return max(n.depth for n in self.nodes())

    def get_n_leaves(self):
        return sum(1 for n in self.nodes() if n.is_leaf)

    @property
    def node_count(self):
        return len(self.nodes())

    def export_text(self, **kwargs):
        from .display import export_text
        return export_text(self, **kwargs)

    def plot(self, **kwargs):
        from .display import plot_tree
        return plot_tree(self, **kwargs)
