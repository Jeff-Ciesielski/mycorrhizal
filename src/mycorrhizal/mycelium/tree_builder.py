#!/usr/bin/env python3
"""
Tree builder - collects tree metadata during decoration.

This module provides the MyceliumTreeBuilder which is responsible for
collecting FSM/BT integrations, BT nodes, and other metadata during the
@tree decoration phase.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from .tree_spec import MyceliumTreeSpec, FSMIntegration, BTIntegration, NodeDefinition
from .exceptions import TreeDefinitionError


class MyceliumTreeBuilder:
    """
    Builder that collects tree metadata during decoration.

    This is used internally by the @tree decorator to gather all
    FSM/BT integrations, BT nodes, and other metadata before creating
    the final MyceliumTreeSpec.
    """

    def __init__(self, name: str, tree_func: Callable):
        self.name = name
        self.tree_func = tree_func
        self.actions: Dict[str, NodeDefinition] = {}
        self.conditions: Dict[str, NodeDefinition] = {}
        self.root_node: Optional[NodeDefinition] = None
        self.fsm_integrations: Dict[str, FSMIntegration] = {}
        self.bt_integrations: Dict[str, BTIntegration] = {}

    def add_action(self, node_def: NodeDefinition) -> None:
        """Add an action node definition."""
        # Allow replacing (func() may be called multiple times)
        self.actions[node_def.name] = node_def

    def add_condition(self, node_def: NodeDefinition) -> None:
        """Add a condition node definition."""
        # Allow replacing (func() may be called multiple times)
        self.conditions[node_def.name] = node_def

    def set_root(self, node_def: NodeDefinition) -> None:
        """Set the root node."""
        # Allow replacing (func() may be called multiple times)
        self.root_node = node_def

    def add_fsm_integration(self, integration: FSMIntegration) -> None:
        """Add an FSM integration for an action."""
        self.fsm_integrations[integration.action_name] = integration

    def add_bt_integration(self, integration: BTIntegration) -> None:
        """Add a BT integration for a state."""
        self.bt_integrations[integration.state_name] = integration

    def build(self) -> MyceliumTreeSpec:
        """
        Build the final MyceliumTreeSpec.

        Returns:
            Complete tree specification

        Raises:
            TreeDefinitionError: If validation fails
        """
        # Create tree spec
        spec = MyceliumTreeSpec(
            name=self.name,
            tree_func=self.tree_func,
            actions=self.actions,
            conditions=self.conditions,
            root_node=self.root_node,
            fsm_integrations=self.fsm_integrations,
            bt_integrations=self.bt_integrations,
        )

        # Validate
        errors = spec.validate()
        if errors:
            raise TreeDefinitionError(
                "Tree validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        return spec


# Context variable to track current tree builder
_current_builder: Optional[MyceliumTreeBuilder] = None


def get_current_builder() -> Optional[MyceliumTreeBuilder]:
    """Get the currently active tree builder (if any)."""
    return _current_builder


def set_current_builder(builder: Optional[MyceliumTreeBuilder]) -> None:
    """Set the current tree builder."""
    global _current_builder
    _current_builder = builder
