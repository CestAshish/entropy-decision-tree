"""entropy_tree: a from-scratch decision tree using entropy and information gain."""

from .display import export_graphviz, export_text, plot_tree
from .tree import DecisionTreeClassifier, Node, entropy, information_gain

__version__ = "0.1.0"

__all__ = [
    "DecisionTreeClassifier",
    "Node",
    "entropy",
    "information_gain",
    "export_text",
    "export_graphviz",
    "plot_tree",
]
