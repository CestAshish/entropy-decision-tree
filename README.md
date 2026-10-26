# entropy-decision-tree

A decision tree classifier written from scratch with NumPy. It splits on **entropy** and **information gain**, lets you set how big the tree can grow, and can draw the tree as text, Graphviz or a matplotlib figure.

```bash
pip install entropy-decision-tree          # core (numpy only)
pip install entropy-decision-tree[plot]    # + matplotlib for plot_tree
```

## Quick start

```python
from sklearn.datasets import load_iris
from entropy_tree import DecisionTreeClassifier, export_text

iris = load_iris()
tree = DecisionTreeClassifier(max_depth=3).fit(iris.data, iris.target)

print(tree.score(iris.data, iris.target))
print(export_text(tree, feature_names=iris.feature_names,
                  class_names=list(iris.target_names)))
tree.plot(feature_names=iris.feature_names)   # matplotlib Axes
```

```
[petal length (cm) <= 2.450]  H=1.585 IG=0.918 n=150
|-- yes: setosa  n=50 H=0.000 [50, 0, 0]
`-- no:  [petal width (cm) <= 1.750]  H=1.000 IG=0.690 n=100
    |-- yes: [petal length (cm) <= 4.950]  H=0.445 IG=0.213 n=54
    ...
```

![Iris tree](https://raw.githubusercontent.com/cestashish/entropy-decision-tree/main/docs/iris_tree.png)

## Hyperparameters (tree size)

| parameter | default | effect |
|---|---|---|
| `max_depth` | `None` | maximum depth (root = depth 0) |
| `min_samples_split` | `2` | a node needs at least this many samples to be split |
| `min_samples_leaf` | `1` | each child of a split must keep at least this many samples |
| `min_information_gain` | `0.0` | a split must gain strictly more than this (bits) |
| `max_leaf_nodes` | `None` | leaf budget; splits are taken best-gain first |
| `max_features` | `None` | features tried per node: int, float, `"sqrt"`, `"log2"` |
| `random_state` | `None` | seed for `max_features` sampling |

## API

- `fit(X, y)`, `predict(X)`, `predict_proba(X)`, `score(X, y)`, `apply(X)`
- `feature_importances_`, `get_depth()`, `get_n_leaves()`, `node_count`, `nodes()`, `root_`
- `export_text(tree)`, `export_graphviz(tree)`, `plot_tree(tree)`
- `entropy(counts)`, `information_gain(parent, left, right)`

`X` must be numeric (NumPy array, list or pandas DataFrame, whose column names are reused). Labels can be any type.

## Development

```bash
pip install -e .[dev]
pytest
python examples/demo.py
```

MIT licence.
