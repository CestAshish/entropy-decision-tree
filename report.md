# Report: decision tree from scratch (entropy and information gain)

- **GitHub:** https://github.com/cestashish/entropy-decision-tree
- **PyPI:** https://pypi.org/project/entropy-decision-tree/ (`pip install entropy-decision-tree`)

## 1. What was asked and what was done

| requirement | status | where |
|---|---|---|
| Decision tree from scratch, using entropy and information gain | done (NumPy only, no scikit-learn in the algorithm) | `src/entropy_tree/tree.py` |
| Hyperparameters for the size of the tree | done: `max_depth`, `min_samples_split`, `min_samples_leaf`, `min_information_gain`, `max_leaf_nodes` (plus `max_features`) | `DecisionTreeClassifier.__init__` |
| Bonus: display the tree | done: ASCII text, Graphviz DOT, matplotlib figure | `src/entropy_tree/display.py` |
| Bonus: package on PyPI | packaged as `entropy-decision-tree`, builds and passes `twine check` | `pyproject.toml` |

## 2. How the algorithm works

**Entropy** of a set of samples $S$ with class proportions $p_k$:

$$H(S) = -\sum_k p_k \log_2 p_k$$

It is 0 for a pure node and $\log_2 K$ when $K$ classes are equally mixed (1.585 bits for the 3 Iris classes).

**Information gain** of splitting $S$ into $S_L$ and $S_R$:

$$IG = H(S) - \frac{|S_L|}{|S|} H(S_L) - \frac{|S_R|}{|S|} H(S_R)$$

**Finding the best split at a node.** For every feature:

1. Sort the samples by that feature.
2. Take a running sum (cumulative sum) of one-hot labels. Row $i$ then holds the class counts of the left child if we cut after sample $i$; the right child is the total minus that.
3. Compute the entropy of all left and right candidates at once (vectorised), and the gain for each cut.
4. Keep only cuts between two *different* feature values that leave at least `min_samples_leaf` samples on each side.
5. The threshold is the midpoint between the two neighbouring values, and the rule is `x[feature] <= threshold → left`.

This is $O(d \cdot n \log n)$ per node instead of the naive $O(d \cdot n^2)$ that recomputes entropy for each threshold.

**Growing the tree.** Candidate splits go in a priority queue ordered by information gain, and the best one is always expanded first. A node becomes a leaf when:

- it is pure ($H = 0$),
- it reached `max_depth`,
- it has fewer than `min_samples_split` samples,
- no split respects `min_samples_leaf`,
- the best gain is not greater than `min_information_gain`, or
- the `max_leaf_nodes` budget is used up.

Without `max_leaf_nodes` the result is identical to the classic recursive ID3/C4.5-style algorithm, since every allowed split gets made. With a leaf budget, best-first order means the tree keeps the most informative splits rather than whichever came first in a depth-first walk.

**Prediction.** A sample goes down the tree until it reaches a leaf. `predict` returns the majority class of that leaf, and `predict_proba` returns the class frequencies in it.

**Feature importance.** This is the sum of $\frac{n_{node}}{N}\cdot IG$ over every split that uses the feature, normalised to 1.

## 3. Hyperparameters controlling tree size

| parameter | what it limits |
|---|---|
| `max_depth` | the longest path from root to leaf |
| `min_samples_split` | the smallest node allowed to split |
| `min_samples_leaf` | the smallest allowed leaf |
| `min_information_gain` | splits that do not reduce entropy enough (pre-pruning) |
| `max_leaf_nodes` | the total number of leaves (best-first) |
| `max_features` | not a size control: random feature subset per node (useful to build a random forest) |

The effect of `max_depth` on the Breast Cancer dataset (70/30 split) is the usual bias–variance trade-off. Test accuracy peaks at depth 4–5; deeper trees reach 100 % training accuracy but test accuracy drops a little.

| max_depth | 0 | 1 | 2 | 3 | 4 | 5 | 6+ |
|---|---|---|---|---|---|---|---|
| train | 0.628 | 0.932 | 0.932 | 0.977 | 0.987 | 0.997 | 1.000 |
| test | 0.626 | 0.889 | 0.883 | 0.912 | **0.930** | **0.930** | 0.924 |

![accuracy vs depth](docs/depth_sweep.png)

## 4. Validation

**Unit tests:** `pytest`, 37 tests, all passing. They cover:

- entropy and gain values computed by hand, including the textbook *Play Tennis* example ($H = 0.940$, $IG(\text{Wind}) = 0.048$);
- each hyperparameter actually limiting the tree;
- input validation, string labels and pandas column names;
- the display functions.

**Comparison with scikit-learn** (`criterion="entropy"`, same data split):

| dataset | params | ours (test acc / depth / leaves) | scikit-learn |
|---|---|---|---|
| Iris | `max_depth=2` | 0.956 / 2 / 3 | 0.956 / 2 / 3 |
| Iris | `max_depth=3` | 0.956 / 3 / 5 | 0.956 / 3 / 5 |
| Iris | `min_samples_leaf=5` | 0.911 / 4 / 6 | 0.911 / 4 / 6 |
| Breast cancer | `max_depth=3` | 0.912 / 3 / 7 | 0.912 / 3 / 7 |
| Breast cancer | `min_samples_leaf=5` | 0.936 / 5 / 11 | 0.936 / 5 / 11 |
| Breast cancer | `max_leaf_nodes=8` | 0.912 / 6 / 8 | 0.906 / 5 / 8 |
| Iris | unlimited | 0.889 / 7 / 10 | 0.956 / 7 / 10 |

With size limits the trees match scikit-learn. A test checks that the root entropy and information gain on Iris are equal to scikit-learn's values. The fully grown trees have the same depth and number of leaves but can differ a little in accuracy. Deep in the tree, several splits often tie on gain, and scikit-learn breaks ties by visiting features in random order, while this implementation takes the first feature. That changes which tied split is used, not the quality of the split.

**Speed.** This implementation is about 3–5× slower than scikit-learn's Cython code (for example 28 ms vs 6 ms on Breast Cancer), which is expected for pure NumPy.

## 5. Displaying the tree

```python
from entropy_tree import DecisionTreeClassifier, export_text, export_graphviz
tree = DecisionTreeClassifier(max_depth=3).fit(X, y)
print(export_text(tree, feature_names=..., class_names=...))
tree.plot(feature_names=..., class_names=...)        # matplotlib
open("tree.dot", "w").write(export_graphviz(tree))   # dot -Tpng tree.dot
```

```
[petal length <= 2.450]  H=1.585 IG=0.918 n=150
|-- yes: setosa  n=50 H=0.000 [50, 0, 0]
`-- no:  [petal width <= 1.750]  H=1.000 IG=0.690 n=100
    |-- yes: [petal length <= 4.950]  H=0.445 IG=0.213 n=54
    |   |-- yes: versicolor  n=48 H=0.146 [0, 47, 1]
    |   `-- no:  virginica  n=6 H=0.918 [0, 2, 4]
    `-- no:  [petal length <= 4.850]  H=0.151 IG=0.091 n=46
        |-- yes: virginica  n=3 H=0.918 [0, 1, 2]
        `-- no:  virginica  n=43 H=0.000 [0, 0, 43]
```

In the plot, the box colour is the majority class and the colour saturation shows purity: faded means high entropy.

![Iris tree](docs/iris_tree.png)

## 6. Packaging

- The layout is `src/` with `pyproject.toml` and the hatchling build backend.
- The only runtime dependency is `numpy`. matplotlib is an optional extra: `pip install entropy-decision-tree[plot]`.
- `python -m build` produces a wheel and an sdist, and both pass `twine check`. The wheel was installed in a clean virtual environment and smoke-tested.
- To publish:

```bash
python -m build
python -m twine upload dist/*
```

## 7. Limitations and possible extensions

- Features must be numeric. Categorical features need to be one-hot or ordinal encoded first.
- Missing values are not handled.
- Growing is greedy, so a pattern like XOR, where no single split has any gain, is not found. This is covered by a test.
- Possible extensions: gain ratio (C4.5) to reduce the bias towards many-valued features, cost-complexity post-pruning, and a regression tree using variance reduction.

## 8. Repository layout

```
src/entropy_tree/tree.py      algorithm (entropy, IG, split search, growth, predict)
src/entropy_tree/display.py   export_text, export_graphviz, plot_tree
tests/test_tree.py            37 pytest tests
examples/demo.py              benchmarks vs scikit-learn, generates docs/*.png
pyproject.toml                packaging metadata
```
