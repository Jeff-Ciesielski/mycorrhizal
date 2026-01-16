#!/usr/bin/env python3
"""
Shared Mermaid diagram formatting utilities.

This module provides common formatting functions for generating Mermaid diagrams
across Mycorrhizal systems (Enoki, Mycelium, etc.). Using these utilities ensures:
- Consistent visual styling across all systems
- Single source of truth for formatting logic
- Bug fixes apply to all systems

Usage:
    from mycorrhizal.common.mermaid import format_state_node, format_transition

    node = format_state_node("state1", "IdleState", is_initial=True)
    edge = format_transition("state1", "state2", "START")
"""

from __future__ import annotations

from typing import Optional


def format_state_node(node_id: str, state_name: str, is_initial: bool = False) -> str:
    """
    Format a state node for Mermaid diagrams.

    Args:
        node_id: The unique identifier for this node (e.g., "fsm1_IdleState")
        state_name: The display name for the state (e.g., "IdleState")
        is_initial: Whether this is the initial/entry state

    Returns:
        Mermaid node definition as a string

    Example:
        >>> format_state_node("state1", "IdleState", is_initial=True)
        '    state1(["IdleState (initial)"])'
        >>> format_state_node("state2", "Running", is_initial=False)
        '    state2(["Running"])'
    """
    display_name = state_name
    if is_initial:
        display_name = f"{state_name} (initial)"

    # Use stadium shape (rounded rectangle) for states: node_id(["label"])
    return f'    {node_id}(["[{display_name}]"])'


def format_transition(
    from_id: str,
    to_id: str,
    label: Optional[str] = None,
    style: Optional[str] = None
) -> str:
    """
    Format a transition edge for Mermaid diagrams.

    Args:
        from_id: Source node ID
        to_id: Target node ID
        label: Optional transition label (e.g., event name)
        style: Optional edge style (e.g., ".thick" for thick lines)

    Returns:
        Mermaid edge definition as a string

    Example:
        >>> format_transition("state1", "state2", "START")
        '    state1 -->|START| state2'
        >>> format_transition("state1", "state2")
        '    state1 --> state2'
    """
    edge = f"    {from_id} -->"
    if label:
        edge = f"{edge}|{label}|"
    edge = f"{edge} {to_id}"

    if style:
        edge = f"{edge}{style}"

    return edge


def format_subgraph(subgraph_id: str, title: str, content: list[str]) -> str:
    """
    Format a subgraph for grouping related nodes.

    Args:
        subgraph_id: Unique identifier for the subgraph
        title: Display title for the subgraph
        content: List of lines containing the subgraph content

    Returns:
        Mermaid subgraph definition as a string

    Example:
        >>> content = [
        ...     '    state1(["Idle"])',
        ...     '    state2(["Running"])'
        ... ]
        >>> format_subgraph("FSM1", "My FSM", content)
        '    subgraph FSM1 ["My FSM"]\\n    state1(["Idle"])\\n    state2(["Running"])\\n    end'
    """
    lines = [f'    subgraph {subgraph_id} ["{title}"]']
    lines.extend(content)
    lines.append("    end")
    return "\n".join(lines)


def format_comment(text: str) -> str:
    """
    Format a comment for Mermaid diagrams.

    Args:
        text: Comment text

    Returns:
        Mermaid comment as a string

    Example:
        >>> format_comment("This is a comment")
        '    %% This is a comment'
    """
    return f"    %% {text}"
