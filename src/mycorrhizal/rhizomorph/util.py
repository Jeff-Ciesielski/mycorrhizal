#!/usr/bin/env python3
"""
Rhizomorph Behavior Tree Utilities

Utility functions for behavior tree operations like Mermaid diagram generation.
"""
from typing import Any
from types import SimpleNamespace


def to_mermaid(tree: SimpleNamespace) -> str:
    """
    Generate Mermaid diagram from a behavior tree namespace.

    This is a utility wrapper that accesses the tree's internal to_mermaid method.
    Use this when you have a tree namespace and want to generate a diagram.

    Args:
        tree: The behavior tree namespace (created with @bt.tree decorator)

    Returns:
        Mermaid diagram as a string

    Example:
        ```python
        from mycorrhizal.rhizomorph.core import bt
        from mycorrhizal.rhizomorph.util import to_mermaid

        @bt.tree
        def MyTree():
            # ... define tree ...
            pass

        # Generate diagram
        mermaid_diagram = to_mermaid(MyTree)
        print(mermaid_diagram)
        ```
    """
    # The tree namespace has a to_mermaid lambda attached by @bt.tree
    if not hasattr(tree, "to_mermaid"):
        raise ValueError(
            "Tree namespace must have a 'to_mermaid' attribute. "
            "Did you create this tree with @bt.tree decorator?"
        )
    return tree.to_mermaid()
