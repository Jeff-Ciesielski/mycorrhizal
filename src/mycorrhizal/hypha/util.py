#!/usr/bin/env python3
"""
Hypha Petri Net Utilities

Utility functions for Petri net operations like Mermaid diagram generation.
"""
from .core.specs import NetSpec


def to_mermaid(spec: NetSpec) -> str:
    """
    Generate Mermaid diagram from a NetSpec.

    This is a utility wrapper around NetSpec.to_mermaid() for convenience.
    Use this when you have a spec object and want to generate a diagram.

    Args:
        spec: The NetSpec to generate a diagram for

    Returns:
        Mermaid diagram as a string

    Example:
        ```python
        from mycorrhizal.hypha.core import pn
        from mycorrhizal.hypha.util import to_mermaid

        @pn.net
        def MyNet(builder):
            # ... define net ...
            pass

        # Generate diagram
        mermaid_diagram = to_mermaid(MyNet._spec)
        print(mermaid_diagram)
        ```
    """
    return spec.to_mermaid()
