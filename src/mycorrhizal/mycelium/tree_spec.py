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
class PNIntegration:
    """Metadata about a BT or FSM integrated into a PN transition."""

    transition_name: str  # The transition that uses this integration
    integration_type: str = "bt"  # "bt" or "fsm"
    mode: str = "token"  # "token" or "batch" (BT only)

    # BT integration fields
    bt_tree: Any = None  # BT namespace from bt.tree()

    # FSM integration fields
    initial_state: Any = None  # Initial state for FSM (StateSpec or callable)

    # Output destinations
    output_places: Optional[List[Any]] = None  # List of PlaceRef objects (output destinations)

    def __post_init__(self):
        """Validate PN integration."""
        if not self.transition_name:
            raise ValueError("Transition name cannot be empty")

        # Validate integration type
        if self.integration_type not in ("bt", "fsm"):
            raise ValueError(f"Invalid integration_type: {self.integration_type}. Must be 'bt' or 'fsm'")

        # Validate BT-specific fields
        if self.integration_type == "bt":
            if self.bt_tree is None:
                raise ValueError("BT integration requires bt_tree parameter")
            if self.mode not in ("token", "batch"):
                raise ValueError(f"Invalid mode: {self.mode}. Must be 'token' or 'batch'")

        # Validate FSM-specific fields
        if self.integration_type == "fsm":
            if self.initial_state is None:
                raise ValueError("FSM integration requires initial_state parameter")

        # Initialize output_places if None
        if self.output_places is None:
            self.output_places = []

    @property
    def fsm_state_name(self) -> str:
        """Get the name of the FSM initial state for display purposes."""
        if self.initial_state is None:
            return "Unknown"

        # If it's a StateSpec, extract the name
        if hasattr(self.initial_state, 'name'):
            name = self.initial_state.name
            # Extract just the simple name (last component after dot)
            if '.' in name:
                return name.split('.')[-1]
            return name

        # Try to get the name from the state
        if hasattr(self.initial_state, '__name__'):
            return self.initial_state.__name__

        return str(self.initial_state)


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
    # PN integrations keyed by transition name (fully qualified)
    pn_integrations: Dict[str, PNIntegration] = field(default_factory=dict)

    @property
    def fsms(self) -> Dict[str, FSMIntegration]:
        """
        Access FSM integrations (alias for fsm_integrations).

        This provides convenient access like:
            spec.fsms['robot']
        """
        return self.fsm_integrations

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

    def get_pn_integration(self, transition_name: str) -> Optional["PNIntegration"]:
        """Get PN integration for a transition."""
        return self.pn_integrations.get(transition_name)

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
