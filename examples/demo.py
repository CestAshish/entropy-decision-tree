"""Demo: fit on Iris / Breast Cancer, compare with scikit-learn, draw the tree.

Run from the repo root:  python examples/demo.py
Needs the dev extras:    pip install -e .[dev]
"""

from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as SkTree

from entropy_tree import DecisionTreeClassifier, export_text

DOCS = Path(__file__).resolve().parent.parent / "docs"
DOCS.mkdir(exist_ok=True)


def compare(name, X, y, **params):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    t0 = time.perf_counter()
    ours = DecisionTreeClassifier(**params).fit(X_tr, y_tr)
    t_ours = time.perf_counter() - t0
    sk_params = {k: v for k, v in params.items() if k != "min_information_gain"}
    t0 = time.perf_counter()
    sk = SkTree(criterion="entropy", random_state=0, **sk_params).fit(X_tr, y_tr)
    t_sk = time.perf_counter() - t0
    print(
        f"{name:<14} {str(params):<22} ours: test={ours.score(X_te, y_te):.3f} "
        f"depth={ours.get_depth()} leaves={ours.get_n_leaves():<3} ({t_ours * 1000:.0f} ms) | "
        f"sklearn: test={sk.score(X_te, y_te):.3f} depth={sk.get_depth()} "
        f"leaves={sk.get_n_leaves()} ({t_sk * 1000:.1f} ms)"
    )


def depth_sweep(X, y, title, path):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    depths = range(0, 11)
    train, test = [], []
    for d in depths:
        tree = DecisionTreeClassifier(max_depth=d).fit(X_tr, y_tr)
        train.append(tree.score(X_tr, y_tr))
        test.append(tree.score(X_te, y_te))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(depths, train, marker="o", label="train")
    ax.plot(depths, test, marker="o", label="test")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"\n{title}")
    for d, a, b in zip(depths, train, test):
        print(f"  max_depth={d:<2} train={a:.3f} test={b:.3f}")


def main():
    iris = load_iris()
    cancer = load_breast_cancer()

    print("=== ours vs sklearn (criterion='entropy'), 70/30 split ===")
    for params in ({}, {"max_depth": 2}, {"max_depth": 3}, {"min_samples_leaf": 5}):
        compare("iris", iris.data, iris.target, **params)
    for params in ({}, {"max_depth": 3}, {"max_leaf_nodes": 8}, {"min_samples_leaf": 5}):
        compare("breast_cancer", cancer.data, cancer.target, **params)

    tree = DecisionTreeClassifier(max_depth=3).fit(iris.data, iris.target)
    names = [n.replace(" (cm)", "") for n in iris.feature_names]
    print("\n=== Iris, max_depth=3 ===")
    print(export_text(tree, feature_names=names, class_names=list(iris.target_names)))
    print("feature importances:", dict(zip(names, tree.feature_importances_.round(3).tolist())))

    ax = tree.plot(feature_names=names, class_names=list(iris.target_names))
    ax.set_title("Iris, max_depth=3 (entropy / information gain)")
    ax.figure.savefig(DOCS / "iris_tree.png", dpi=130, bbox_inches="tight")
    plt.close(ax.figure)

    depth_sweep(cancer.data, cancer.target, "Breast cancer: accuracy vs max_depth", DOCS / "depth_sweep.png")


if __name__ == "__main__":
    main()
