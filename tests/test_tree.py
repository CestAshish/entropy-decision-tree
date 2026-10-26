import numpy as np
import pytest

from entropy_tree import (
    DecisionTreeClassifier,
    entropy,
    export_graphviz,
    export_text,
    information_gain,
)


# --------------------------------------------------------------- the maths
def test_entropy_known_values():
    assert entropy([5, 5]) == pytest.approx(1.0)
    assert entropy([10, 0]) == 0.0
    assert entropy([1, 1, 1, 1]) == pytest.approx(2.0)
    assert entropy([]) == 0.0
    # play-tennis dataset: 9 yes / 5 no
    assert entropy([9, 5]) == pytest.approx(0.940, abs=1e-3)


def test_information_gain_play_tennis_wind():
    # Wind: weak -> 6 yes / 2 no, strong -> 3 yes / 3 no  (Mitchell, ch. 3)
    assert information_gain([9, 5], [6, 2], [3, 3]) == pytest.approx(0.048, abs=1e-3)


def test_perfect_split_gain_equals_parent_entropy():
    assert information_gain([4, 4], [4, 0], [0, 4]) == pytest.approx(1.0)


# ---------------------------------------------------------- basic fitting
def test_fits_simple_threshold():
    X = np.array([[1], [2], [3], [10], [11], [12]])
    y = np.array([0, 0, 0, 1, 1, 1])
    tree = DecisionTreeClassifier().fit(X, y)
    assert tree.root_.feature == 0
    assert tree.root_.threshold == pytest.approx(6.5)
    assert tree.root_.gain == pytest.approx(1.0)
    assert tree.get_n_leaves() == 2
    assert tree.predict([[0], [6], [7], [100]]).tolist() == [0, 0, 1, 1]


def test_string_labels_and_proba():
    X = [[0, 0], [0, 1], [1, 0], [1, 1], [1, 1]]
    y = ["no", "no", "no", "yes", "yes"]
    tree = DecisionTreeClassifier().fit(X, y)
    assert tree.predict([[1, 1], [0, 0]]).tolist() == ["yes", "no"]
    proba = tree.predict_proba([[1, 1]])
    assert proba.shape == (1, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_memorises_training_data_when_unlimited():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = rng.integers(0, 3, size=200)
    tree = DecisionTreeClassifier().fit(X, y)
    assert tree.score(X, y) == 1.0


def test_single_class_gives_single_leaf():
    tree = DecisionTreeClassifier().fit([[1], [2], [3]], [7, 7, 7])
    assert tree.root_.is_leaf
    assert tree.predict([[100]]).tolist() == [7]


def test_xor_has_no_first_split_gain():
    # Greedy information gain cannot see XOR: every single split has IG = 0.
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    tree = DecisionTreeClassifier().fit(X, y)
    assert tree.root_.is_leaf


def test_pandas_feature_names():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"age": [20, 25, 60, 65], "income": [1, 2, 1, 2]})
    tree = DecisionTreeClassifier().fit(df, [0, 0, 1, 1])
    assert "age <=" in export_text(tree)


# -------------------------------------------------------- hyperparameters
@pytest.fixture
def noisy():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(400, 5))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.8, size=400) > 0).astype(int)
    return X, y


@pytest.mark.parametrize("depth", [0, 1, 2, 3, 5])
def test_max_depth(noisy, depth):
    tree = DecisionTreeClassifier(max_depth=depth).fit(*noisy)
    assert tree.get_depth() <= depth


@pytest.mark.parametrize("leaves", [1, 2, 5, 10])
def test_max_leaf_nodes(noisy, leaves):
    tree = DecisionTreeClassifier(max_leaf_nodes=leaves).fit(*noisy)
    assert tree.get_n_leaves() == leaves


def test_min_samples_leaf(noisy):
    tree = DecisionTreeClassifier(min_samples_leaf=25).fit(*noisy)
    assert min(n.n_samples for n in tree.nodes() if n.is_leaf) >= 25


def test_min_samples_split(noisy):
    tree = DecisionTreeClassifier(min_samples_split=50).fit(*noisy)
    for node in tree.nodes():
        if not node.is_leaf:
            assert node.n_samples >= 50


def test_min_information_gain(noisy):
    tree = DecisionTreeClassifier(min_information_gain=0.05).fit(*noisy)
    for node in tree.nodes():
        if not node.is_leaf:
            assert node.gain > 0.05


def test_size_shrinks_with_regularisation(noisy):
    full = DecisionTreeClassifier().fit(*noisy).node_count
    small = DecisionTreeClassifier(max_depth=3).fit(*noisy).node_count
    assert small < full


def test_max_features_is_reproducible(noisy):
    a = DecisionTreeClassifier(max_features="sqrt", random_state=1).fit(*noisy)
    b = DecisionTreeClassifier(max_features="sqrt", random_state=1).fit(*noisy)
    assert export_text(a) == export_text(b)


def test_best_first_keeps_most_informative_splits(noisy):
    # the root split of a 2-leaf tree is the same as the root of the full tree
    full = DecisionTreeClassifier().fit(*noisy)
    stump = DecisionTreeClassifier(max_leaf_nodes=2).fit(*noisy)
    assert stump.root_.feature == full.root_.feature
    assert stump.root_.threshold == full.root_.threshold


@pytest.mark.parametrize(
    "params",
    [
        {"max_depth": -1},
        {"min_samples_split": 1},
        {"min_samples_leaf": 0},
        {"min_information_gain": -0.1},
        {"max_leaf_nodes": 0},
        {"max_features": "bad"},
    ],
)
def test_invalid_params(params):
    with pytest.raises(ValueError):
        DecisionTreeClassifier(**params).fit([[0], [1]], [0, 1])


def test_not_fitted():
    with pytest.raises(RuntimeError):
        DecisionTreeClassifier().predict([[0]])


def test_wrong_feature_count():
    tree = DecisionTreeClassifier().fit([[0, 1], [1, 0]], [0, 1])
    with pytest.raises(ValueError):
        tree.predict([[0, 1, 2]])


def test_set_params():
    tree = DecisionTreeClassifier().set_params(max_depth=4)
    assert tree.max_depth == 4
    with pytest.raises(ValueError):
        tree.set_params(nope=1)


# ------------------------------------------------- inspection and display
def test_feature_importances(noisy):
    tree = DecisionTreeClassifier(max_depth=4).fit(*noisy)
    imp = tree.feature_importances_
    assert imp.sum() == pytest.approx(1.0)
    assert np.argmax(imp) == 0  # feature 0 drives the label


def test_export_text_and_dot(noisy):
    tree = DecisionTreeClassifier(max_depth=2).fit(*noisy)
    text = export_text(tree, feature_names=list("abcde"), class_names=["neg", "pos"])
    assert text.splitlines()[0].startswith("[a <=")
    assert "|-- yes:" in text and "`-- no:" in text
    dot = export_graphviz(tree)
    assert dot.startswith("digraph Tree {") and dot.rstrip().endswith("}")
    assert dot.count("->") == tree.node_count - 1


def test_plot_tree(noisy):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    tree = DecisionTreeClassifier(max_depth=3).fit(*noisy)
    ax = tree.plot()
    assert len(ax.texts) >= tree.node_count


# ------------------------------------------------------- vs scikit-learn
def test_matches_sklearn_on_iris():
    datasets = pytest.importorskip("sklearn.datasets")
    from sklearn.tree import DecisionTreeClassifier as SkTree

    X, y = datasets.load_iris(return_X_y=True)
    ours = DecisionTreeClassifier(max_depth=1).fit(X, y)
    sk = SkTree(criterion="entropy", max_depth=1, random_state=0).fit(X, y)
    # root entropy and information gain are identical (sklearn uses log2 too)
    sk_root_h = sk.tree_.impurity[0]
    sk_children_h = (
        sk.tree_.n_node_samples[1] * sk.tree_.impurity[1] + sk.tree_.n_node_samples[2] * sk.tree_.impurity[2]
    ) / sk.tree_.n_node_samples[0]
    assert ours.root_.entropy == pytest.approx(sk_root_h)
    assert ours.root_.gain == pytest.approx(sk_root_h - sk_children_h)
    assert ours.score(X, y) == pytest.approx(sk.score(X, y))
