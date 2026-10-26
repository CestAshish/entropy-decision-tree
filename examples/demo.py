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


def run_comparison(name, X, y, **params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    t0 = time.perf_counter()
    ours = DecisionTreeClassifier(**params).fit(X_train, y_train)
    t_ours = time.perf_counter() - t0

    sk_params = {k: v for k, v in params.items() if k != "min_information_gain"}
    t0 = time.perf_counter()
    sk = SkTree(criterion="entropy", random_state=0, **sk_params).fit(X_train, y_train)
    t_sk = time.perf_counter() - t0

    print(
        f"{name:<14} {str(params):<22} ours: test={ours.score(X_test, y_test):.3f} "
        f"depth={ours.get_depth()} leaves={ours.get_n_leaves():<3} ({t_ours * 1000:.0f} ms) | "
        f"sklearn: test={sk.score(X_test, y_test):.3f} depth={sk.get_depth()} "
        f"leaves={sk.get_n_leaves()} ({t_sk * 1000:.1f} ms)"
    )


def plot_depth_curve(X, y, title, save_path):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    depths = range(0, 11)
    train_scores, test_scores = [], []

    for d in depths:
        tree = DecisionTreeClassifier(max_depth=d).fit(X_train, y_train)
        train_scores.append(tree.score(X_train, y_train))
        test_scores.append(tree.score(X_test, y_test))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(depths, train_scores, marker="o", label="train")
    ax.plot(depths, test_scores, marker="o", label="test")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)

    print(f"\n{title}")
    for d, tr, te in zip(depths, train_scores, test_scores):
        print(f"  max_depth={d:<2} train={tr:.3f} test={te:.3f}")


def main():
    iris = load_iris()
    cancer = load_breast_cancer()

    print("=== Comparison with sklearn (criterion='entropy'), 70/30 split ===")
    for params in ({}, {"max_depth": 2}, {"max_depth": 3}, {"min_samples_leaf": 5}):
        run_comparison("iris", iris.data, iris.target, **params)
    for params in ({}, {"max_depth": 3}, {"min_samples_split": 10}, {"min_samples_leaf": 5}):
        run_comparison("breast_cancer", cancer.data, cancer.target, **params)

    tree = DecisionTreeClassifier(max_depth=3).fit(iris.data, iris.target)
    feature_names = [name.replace(" (cm)", "") for name in iris.feature_names]

    print("\n=== Iris Decision Tree (max_depth=3) ===")
    print(export_text(tree, feature_names=feature_names, class_names=list(iris.target_names)))
    print("\nFeature importances:", dict(zip(feature_names, tree.feature_importances_.round(3).tolist())))

    ax = tree.plot(feature_names=feature_names, class_names=list(iris.target_names))
    ax.set_title("Iris, max_depth=3 (entropy / information gain)")
    ax.figure.savefig(DOCS / "iris_tree.png", dpi=130, bbox_inches="tight")
    plt.close(ax.figure)

    plot_depth_curve(cancer.data, cancer.target, "Breast cancer: accuracy vs max_depth", DOCS / "depth_sweep.png")


if __name__ == "__main__":
    main()
