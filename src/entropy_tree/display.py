"""Ways to look at a fitted tree: plain text, Graphviz DOT, and matplotlib."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .tree import DecisionTreeClassifier, Node

__all__ = ["export_text", "export_graphviz", "plot_tree"]


def _names(tree: DecisionTreeClassifier, feature_names, class_names):
    tree._check_is_fitted()
    if feature_names is None:
        feature_names = getattr(tree, "feature_names_in_", None)
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(tree.n_features_in_)]
    if len(feature_names) != tree.n_features_in_:
        raise ValueError("feature_names must have one entry per feature")
    if class_names is None:
        class_names = [str(c) for c in tree.classes_]
    if len(class_names) != tree.n_classes_:
        raise ValueError("class_names must have one entry per class")
    return list(map(str, feature_names)), list(map(str, class_names))


def _fmt(x: float, decimals: int) -> str:
    return f"{x:.{decimals}f}"


def export_text(
    tree: DecisionTreeClassifier,
    feature_names: Optional[Sequence[str]] = None,
    class_names: Optional[Sequence[str]] = None,
    decimals: int = 3,
    show_counts: bool = True,
) -> str:
    """Render the tree as an indented ASCII diagram.

    Example::

        [petal length <= 2.450]  H=1.585 IG=0.918 n=150
        |-- yes: setosa  n=50 H=0.000 [50, 0, 0]
        `-- no:  [petal width <= 1.750]  H=1.000 IG=0.690 n=100
            ...
    """
    fnames, cnames = _names(tree, feature_names, class_names)

    def describe(node: Node) -> str:
        if node.is_leaf:
            text = f"{cnames[node.prediction]}  n={node.n_samples} H={_fmt(node.entropy, decimals)}"
            if show_counts:
                text += f" {node.value.astype(int).tolist()}"
            return text
        return (
            f"[{fnames[node.feature]} <= {_fmt(node.threshold, decimals)}]  "
            f"H={_fmt(node.entropy, decimals)} IG={_fmt(node.gain, decimals)} n={node.n_samples}"
        )

    lines = [describe(tree.root_)]

    def walk(node: Node, prefix: str) -> None:
        children = [("yes: ", node.left), ("no:  ", node.right)]
        for i, (label, child) in enumerate(children):
            last = i == 1
            lines.append(prefix + ("`-- " if last else "|-- ") + label + describe(child))
            if not child.is_leaf:
                walk(child, prefix + ("    " if last else "|   "))

    if not tree.root_.is_leaf:
        walk(tree.root_, "")
    return "\n".join(lines)


def export_graphviz(
    tree: DecisionTreeClassifier,
    feature_names: Optional[Sequence[str]] = None,
    class_names: Optional[Sequence[str]] = None,
    decimals: int = 3,
) -> str:
    """Return the tree in Graphviz DOT format (render with ``dot -Tpng``)."""
    fnames, cnames = _names(tree, feature_names, class_names)
    out = ["digraph Tree {", 'node [shape=box, style="rounded,filled", fontname="Helvetica"];']
    for node in tree.nodes():
        color = _hex(_node_color(node, tree.n_classes_))
        if node.is_leaf:
            label = f"{cnames[node.prediction]}\\nn = {node.n_samples}\\nH = {_fmt(node.entropy, decimals)}"
        else:
            label = (
                f"{fnames[node.feature]} <= {_fmt(node.threshold, decimals)}\\n"
                f"IG = {_fmt(node.gain, decimals)}\\nH = {_fmt(node.entropy, decimals)}\\n"
                f"n = {node.n_samples}\\nclass = {cnames[node.prediction]}"
            )
        label = label.replace('"', '\\"')
        out.append(f'{node.node_id} [label="{label}", fillcolor="{color}"];')
        if not node.is_leaf:
            out.append(f'{node.node_id} -> {node.left.node_id} [label="yes"];')
            out.append(f'{node.node_id} -> {node.right.node_id} [label="no"];')
    out.append("}")
    return "\n".join(out)


_PALETTE = [
    (0.122, 0.467, 0.706),
    (1.000, 0.498, 0.055),
    (0.173, 0.627, 0.173),
    (0.839, 0.153, 0.157),
    (0.580, 0.404, 0.741),
    (0.549, 0.337, 0.294),
    (0.890, 0.467, 0.761),
    (0.498, 0.498, 0.498),
    (0.737, 0.741, 0.133),
    (0.090, 0.745, 0.812),
]


def _node_color(node: Node, n_classes: int):
    """Majority-class colour, faded towards white as entropy grows."""
    base = np.array(_PALETTE[node.prediction % len(_PALETTE)])
    max_h = np.log2(n_classes) if n_classes > 1 else 1.0
    purity = 1.0 - node.entropy / max_h
    alpha = 0.15 + 0.75 * purity
    return tuple(alpha * base + (1 - alpha) * np.ones(3))


def _hex(rgb) -> str:
    return "#" + "".join(f"{int(round(c * 255)):02x}" for c in rgb)


def _layout(root: Node):
    """x = in-order leaf slot (internal nodes centred over children), y = -depth."""
    pos = {}
    next_x = [0]

    def place(node: Node) -> float:
        if node.is_leaf:
            x = float(next_x[0])
            next_x[0] += 1
        else:
            x = (place(node.left) + place(node.right)) / 2.0
        pos[node.node_id] = (x, -node.depth)
        return x

    place(root)
    return pos, next_x[0]


def plot_tree(
    tree: DecisionTreeClassifier,
    feature_names: Optional[Sequence[str]] = None,
    class_names: Optional[Sequence[str]] = None,
    decimals: int = 2,
    ax=None,
    fontsize: Optional[float] = None,
    figsize=None,
):
    """Draw the tree with matplotlib and return the ``Axes``.

    Each box shows the split (or predicted class), entropy, information gain
    and sample count. Colour = majority class, saturation = purity.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plot_tree needs matplotlib: pip install entropy-decision-tree[plot]") from exc

    fnames, cnames = _names(tree, feature_names, class_names)
    pos, n_leaves = _layout(tree.root_)
    depth = tree.get_depth()

    if ax is None:
        if figsize is None:
            figsize = (max(6.0, 1.9 * n_leaves), max(3.0, 1.9 * (depth + 1)))
        _, ax = plt.subplots(figsize=figsize)
    if fontsize is None:
        fontsize = max(5.0, min(10.0, 90.0 / max(n_leaves, 1)))

    for node in tree.nodes():
        x, y = pos[node.node_id]
        if not node.is_leaf:
            for child, label in ((node.left, "yes"), (node.right, "no")):
                cx, cy = pos[child.node_id]
                ax.annotate(
                    "",
                    xy=(cx, cy + 0.18),
                    xytext=(x, y - 0.18),
                    arrowprops=dict(arrowstyle="->", color="0.4", lw=1),
                )
                ax.text((x + cx) / 2, (y + cy) / 2, label, ha="center", va="center",
                        fontsize=fontsize * 0.85, color="0.3",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
            text = (
                f"{fnames[node.feature]} <= {_fmt(node.threshold, decimals)}\n"
                f"IG = {_fmt(node.gain, 3)}   H = {_fmt(node.entropy, 3)}\n"
                f"n = {node.n_samples}"
            )
        else:
            text = (
                f"{cnames[node.prediction]}\n"
                f"H = {_fmt(node.entropy, 3)}\n"
                f"n = {node.n_samples} {node.value.astype(int).tolist()}"
            )
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                bbox=dict(boxstyle="round,pad=0.4", fc=_node_color(node, tree.n_classes_), ec="0.3"))

    ax.set_xlim(-0.7, n_leaves - 0.3)
    ax.set_ylim(-depth - 0.6, 0.6)
    ax.axis("off")
    return ax
