#!/usr/bin/env python3
"""
Tree specification - metadata about a Mycelium tree.

This module defines the data structures that hold tree metadata,
including FSM/BT integrations, BT nodes, and their relationships.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Callable


@dataclass
class FSMIntegration:
    """Metadata about an FSM integrated into a BT action."""

    initial_state: Type  # StateSpec or callable returning StateSpec
    action_name: str  # The action that uses this FSM

    def __post_init__(self):
        """Validate FSM integration."""
        if not self.action_name:
            raise ValueError("Action name cannot be empty")


@dataclass
class BTIntegration:
    """Metadata about a BT integrated into an FSM state."""

    bt_tree: Any  # BT namespace from bt.tree()
    state_name: str  # The state that uses this BT (fully qualified)
    state_func: Callable  # The on_state function

    def __post_init__(self):
        """Validate BT integration."""
        if not self.state_name:
            raise ValueError("State name cannot be empty")


@dataclass
class NodeDefinition:
    """
    Metadata about a BT node defined in a tree.

    Now includes optional FSM integration for actions.
    """

    name: str
    node_type: str  # "action", "condition", "sequence", "selector", "parallel"
    func: Optional[Callable] = None
    children: List[NodeDefinition] = field(default_factory=list)
    is_root: bool = False
    fsm_integration: Optional[FSMIntegration] = None  # FSM for actions

    def __post_init__(self):
        """Validate node definition."""
        if not self.name:
            raise ValueError("Node name cannot be empty")


@dataclass
class MyceliumTreeSpec:
    """
    Complete specification of a Mycelium tree.

    Contains all metadata about FSM/BT integrations, BT nodes, and their relationships.
    This is what gets built by the @tree decorator and used by TreeInstance.
    """

    name: str
    tree_func: Optional[Callable] = None  # The original @tree decorated function (for reference)
    bt_namespace: Optional[Any] = None  # The BT namespace returned by bt.tree() (set after build)
    actions: Dict[str, NodeDefinition] = field(default_factory=dict)
    conditions: Dict[str, NodeDefinition] = field(default_factory=dict)
    root_node: Optional[NodeDefinition] = None
    # FSM integrations keyed by action name
    fsm_integrations: Dict[str, FSMIntegration] = field(default_factory=dict)
    # BT integrations keyed by state name (fully qualified)
    bt_integrations: Dict[str, BTIntegration] = field(default_factory=dict)

    def get_action(self, name: str) -> Optional[NodeDefinition]:
        """Get action definition by name."""
        return self.actions.get(name)

    def get_condition(self, name: str) -> Optional[NodeDefinition]:
        """Get condition definition by name."""
        return self.conditions.get(name)

    def get_fsm_integration(self, action_name: str) -> Optional[FSMIntegration]:
        """Get FSM integration for an action."""
        return self.fsm_integrations.get(action_name)

    def get_bt_integration(self, state_name: str) -> Optional[BTIntegration]:
        """Get BT integration for a state."""
        return self.bt_integrations.get(state_name)

    def validate(self) -> List[str]:
        """
        Validate the tree specification.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Check root node exists
        if self.root_node is None:
            errors.append("Tree has no root node")

        # Check action names don't collide with condition names
        all_names = list(self.actions.keys()) + list(self.conditions.keys())
        duplicates = [name for name in set(all_names) if all_names.count(name) > 1]
        if duplicates:
            errors.append(f"Duplicate names: {duplicates}")

        return errors
