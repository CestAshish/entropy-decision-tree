# Binary Decision Tree Classifier

A simple and clean binary decision tree classifier built from scratch using NumPy. It uses Shannon entropy and information gain to find the best binary split at each node.

## Quick Start

```python
from sklearn.datasets import load_iris
from entropy_tree import DecisionTreeClassifier, export_text

iris = load_iris()
tree = DecisionTreeClassifier(max_depth=3).fit(iris.data, iris.target)

print("Accuracy:", tree.score(iris.data, iris.target))
print(export_text(tree, feature_names=iris.feature_names, class_names=list(iris.target_names)))
```

Example text output:
```text
[petal length (cm) <= 2.450] (n=150, H=1.585, IG=0.918)
|-- yes: setosa (n=50, H=0.000) [50, 0, 0]
`-- no:  [petal width (cm) <= 1.750] (n=100, H=1.000, IG=0.690)
    |-- yes: [petal length (cm) <= 4.950] (n=54, H=0.445, IG=0.213)
    |   |-- yes: versicolor (n=48, H=0.146) [0, 47, 1]
    |   `-- no:  virginica (n=6, H=0.918) [0, 2, 4]
    `-- no:  [petal length (cm) <= 4.850] (n=46, H=0.151, IG=0.091)
        |-- yes: virginica (n=3, H=0.918) [0, 1, 2]
        `-- no:  virginica (n=43, H=0.000) [0, 0, 43]
```

## Hyperparameters (Controlling Tree Size)

- `max_depth`: maximum depth of the tree (default: `None`).
- `min_samples_split`: minimum samples required to split a node (default: `2`).
- `min_samples_leaf`: minimum samples required in each leaf node (default: `1`).
- `min_information_gain`: minimum information gain required to accept a split (default: `0.0`).

## How It Works

1. **Entropy**: $H(S) = -\sum p_k \log_2(p_k)$
2. **Information Gain**: $IG(S, S_L, S_R) = H(S) - \frac{|S_L|}{|S|} H(S_L) - \frac{|S_R|}{|S|} H(S_R)$
3. **Splitting**: At each node, it tests candidate thresholds (midpoints between consecutive sorted unique values) across all features and selects the binary split that maximizes information gain.
4. **Recursion**: Recursively splits into left (`x <= threshold`) and right (`x > threshold`) children until stopping conditions are met.

## Testing

```bash
pytest
```
