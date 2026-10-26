import numpy as np
from .tree import DecisionTreeClassifier, Node


def _get_feature_and_class_names(tree, feature_names, class_names):
    tree._check_is_fitted()
    if feature_names is None:
        feature_names = getattr(tree, "feature_names_in_", [f"feature_{i}" for i in range(tree.n_features_in_)])
    if class_names is None:
        class_names = [str(c) for c in tree.classes_]
    return list(feature_names), list(class_names)


def export_text(tree, feature_names=None, class_names=None, decimals=3):
    """Print the decision tree in simple indented text format."""
    fnames, cnames = _get_feature_and_class_names(tree, feature_names, class_names)
    lines = []

    def print_node(node, prefix="", is_left=None):
        indent = prefix
        if is_left is True:
            marker = "|-- yes: "
            child_prefix = prefix + "|   "
        elif is_left is False:
            marker = "`-- no:  "
            child_prefix = prefix + "    "
        else:
            marker = ""
            child_prefix = ""

        if node.is_leaf:
            majority = cnames[node.prediction]
            lines.append(f"{indent}{marker}{majority} (n={node.n_samples}, H={node.entropy:.{decimals}f}) {node.value.tolist()}")
        else:
            feat_name = fnames[node.feature]
            thresh = f"{node.threshold:.{decimals}f}"
            lines.append(f"{indent}{marker}[{feat_name} <= {thresh}] (n={node.n_samples}, H={node.entropy:.{decimals}f}, IG={node.gain:.{decimals}f})")
            print_node(node.left, child_prefix, is_left=True)
            print_node(node.right, child_prefix, is_left=False)

    print_node(tree.root_)
    return "\n".join(lines)


def export_graphviz(tree, feature_names=None, class_names=None, decimals=3):
    """Generate simple Graphviz DOT string."""
    fnames, cnames = _get_feature_and_class_names(tree, feature_names, class_names)
    lines = ["digraph DecisionTree {", 'node [shape=box, style="rounded,filled", fillcolor="#eef2f7"];']

    for node in tree.nodes():
        if node.is_leaf:
            label = f"{cnames[node.prediction]}\\nn={node.n_samples}\\nH={node.entropy:.{decimals}f}"
        else:
            label = f"{fnames[node.feature]} <= {node.threshold:.{decimals}f}\\nIG={node.gain:.{decimals}f}\\nn={node.n_samples}"
        lines.append(f'  {node.node_id} [label="{label}"];')
        if not node.is_leaf:
            lines.append(f'  {node.node_id} -> {node.left.node_id} [label="<= {node.threshold:.{decimals}f}"];')
            lines.append(f'  {node.node_id} -> {node.right.node_id} [label="> {node.threshold:.{decimals}f}"];')

    lines.append("}")
    return "\n".join(lines)


def plot_tree(tree, feature_names=None, class_names=None, decimals=2, ax=None, figsize=(10, 6)):
    """Draw the tree with matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is needed for plotting.")

    fnames, cnames = _get_feature_and_class_names(tree, feature_names, class_names)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    depth = tree.get_depth()
    n_leaves = tree.get_n_leaves()

    # Simple coordinate calculation
    positions = {}
    leaf_idx = [0]

    def calc_pos(node):
        if node.is_leaf:
            x = leaf_idx[0]
            leaf_idx[0] += 1
        else:
            x_left = calc_pos(node.left)
            x_right = calc_pos(node.right)
            x = (x_left + x_right) / 2.0
        positions[node.node_id] = (x, -node.depth)
        return x

    calc_pos(tree.root_)

    for node in tree.nodes():
        x, y = positions[node.node_id]
        if node.is_leaf:
            text = f"{cnames[node.prediction]}\nn={node.n_samples}\nH={node.entropy:.{decimals}f}"
            bg = "#d4edda"
        else:
            text = f"{fnames[node.feature]} <= {node.threshold:.{decimals}f}\nIG={node.gain:.{decimals}f}\nn={node.n_samples}"
            bg = "#d1ecf1"

            # Draw lines to children
            lx, ly = positions[node.left.node_id]
            rx, ry = positions[node.right.node_id]
            ax.plot([x, lx], [y, ly], color="gray", lw=1)
            ax.plot([x, rx], [y, ry], color="gray", lw=1)

        ax.text(
            x, y, text,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=bg, edgecolor="gray"),
            fontsize=8
        )

    ax.set_xlim(-1, n_leaves)
    ax.set_ylim(-depth - 1, 1)
    ax.axis("off")
    return ax
