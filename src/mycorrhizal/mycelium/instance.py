#!/usr/bin/env python3
"""
Tree instance - runtime instance of a Mycelium tree.

This module provides the TreeInstance class which manages the runtime
behavior of a tree, including FSM instances for FSM-integrated actions
and execution of BT nodes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING
from pydantic import BaseModel

from ..septum.core import StateMachine, StateSpec
from ..rhizomorph.core import Status, Runner as BTRunner
from .tree_spec import MyceliumTreeSpec, FSMIntegration
from .exceptions import FSMInstantiationError

if TYPE_CHECKING:
    pass


class FSMRunner:
    """
    Wrapper around an FSM instance for use in FSM-integrated actions.

    Attributes:
        fsm: The StateMachine instance
        name: Action name this FSM is associated with
        _initialized: Whether the FSM has been initialized
    """

    def __init__(self, name: str, fsm: StateMachine):
        self.name = name
        self.fsm = fsm
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure the FSM is initialized (lazy initialization)."""
        if not self._initialized:
            await self.fsm.initialize()
            self._initialized = True

    @property
    def current_state(self) -> Optional[StateSpec]:
        """Get the current state of the FSM."""
        return self.fsm.current_state

    def send_message(self, message: Any) -> None:
        """Send a message to the FSM."""
        self.fsm.send_message(message)

    async def tick(self, timeout: Optional[float] = None) -> Any:
        """Tick the FSM once."""
        await self._ensure_initialized()
        return await self.fsm.tick(timeout=timeout)

    async def run(self, timeout: Optional[float] = None) -> Any:
        """Run the FSM to completion."""
        await self._ensure_initialized()
        return await self.fsm.run(timeout=timeout)


class FSMAccessor:
    """
    Provides convenient access to FSM runners.

    Allows both attribute-style and dict-style access:
        accessor.robot
        accessor['robot']
    """

    def __init__(self, fsm_runners: Dict[str, "FSMRunner"]):
        self._fsm_runners = fsm_runners

    def __getattr__(self, name: str) -> "FSMRunner":
        if name in self._fsm_runners:
            return self._fsm_runners[name]
        raise AttributeError(f"No FSM named '{name}'")

    def __getitem__(self, name: str) -> "FSMRunner":
        if name not in self._fsm_runners:
            raise KeyError(f"No FSM named '{name}'")
        return self._fsm_runners[name]

    def __contains__(self, name: str) -> bool:
        return name in self._fsm_runners

    def __iter__(self):
        return iter(self._fsm_runners)

    def __len__(self) -> int:
        return len(self._fsm_runners)


class TreeInstance:
    """
    Runtime instance of a Mycelium tree.

    Manages FSM instances for FSM-integrated actions, executes BT nodes,
    and coordinates execution.

    Attributes:
        spec: The tree specification
        bb: The blackboard (shared state)
        tb: The timebase (shared timing)
        bt_runner: The underlying BT runner
    """

    def __init__(
        self,
        spec: MyceliumTreeSpec,
        bb: BaseModel,
        tb: Any,
    ):
        self.spec = spec
        self.bb = bb
        self.tb = tb
        self._fsm_runners: Dict[str, FSMRunner] = {}
        self.bt_runner: Optional[BTRunner] = None

        # Instantiate FSMs for FSM-integrated actions
        self._instantiate_fsms()

    @property
    def fsms(self) -> Any:
        """
        Access FSM instances.

        Provides attribute and dict-style access:
            tree_instance.fsms.robot
            tree_instance.fsms['robot']
        """
        return FSMAccessor(self._fsm_runners)

    def _instantiate_fsms(self) -> None:
        """Instantiate FSMs for all FSM-integrated actions."""
        for action_name, fsm_integration in self.spec.fsm_integrations.items():
            try:
                # Get the initial state spec
                initial = fsm_integration.initial_state

                # Handle callable states
                if callable(initial) and not isinstance(initial, type):
                    initial = initial()

                # Create common data from blackboard
                common_data = self.bb.model_dump() if hasattr(self.bb, 'model_dump') else dict(self.bb)

                # Create FSM instance
                fsm = StateMachine(initial_state=initial, common_data=common_data)  # type: ignore[arg-type]

                # Store FSM runner (will be initialized lazily)
                runner = FSMRunner(
                    name=action_name,
                    fsm=fsm,
                )
                self._fsm_runners[action_name] = runner

            except Exception as e:
                raise FSMInstantiationError(
                    f"Failed to instantiate FSM for action '{action_name}': {e}"
                ) from e

    def get_fsm_runner(self, action_name: str) -> Optional[FSMRunner]:
        """Get FSM runner for an action."""
        return self._fsm_runners.get(action_name)

    async def tick(self) -> Status:
        """
        Execute one tick of the tree.

        This runs the BT tree. For FSM-integrated actions, the tree instance
        is injected onto the blackboard so actions can access their FSM runners.

        Returns:
            Status from the BT tick
        """
        # Get BT namespace
        tree_namespace = self.spec.bt_namespace

        # Create BT runner if needed
        if self.bt_runner is None:
            from ..rhizomorph.core import Runner as BTRunner
            if tree_namespace is None:
                raise RuntimeError("BT namespace is None, cannot create BT runner")
            self.bt_runner = BTRunner(tree_namespace, bb=self.bb, tb=self.tb)

        # Inject tree instance onto blackboard for FSM-integrated actions
        self.bb._mycelium_tree_instance = self  # type: ignore[attr-defined]

        try:
            # Run BT tree
            result = await self.bt_runner.tick()
        finally:
            # Clean up
            if hasattr(self.bb, '_mycelium_tree_instance'):
                delattr(self.bb, '_mycelium_tree_instance')

        return result

    def to_mermaid(self) -> str:
        """
        Generate a unified Mermaid diagram showing BT and FSM integrations.

        Returns:
            Mermaid diagram code
        """
        lines = []
        lines.append("%% Mycelium Unified Diagram")
        lines.append("%% Shows Behavior Tree (BT) and FSM integrations")
        lines.append("graph TD")

        # Add the BT tree structure and track node ID mappings
        bt_diagram = self._generate_bt_mermaid()
        node_id_to_action = {}  # Map N1, N2, etc. to action names

        if bt_diagram:
            lines.append("")
            lines.append("    %% Behavior Tree Structure")
            # Extract just the node definitions and edges
            for line in bt_diagram.split("\n"):
                stripped = line.strip()
                if stripped and not stripped.startswith("TD") and not stripped.startswith("LR"):
                    lines.append(f"    {stripped}")

                    # Parse action nodes to map IDs to names
                    if "ACTION" in stripped and "<br/>" in stripped:
                        parts = stripped.split("((ACTION<br/>")
                        if len(parts) == 2:
                            node_id = parts[0].strip()
                            action_name = parts[1].rstrip("))").strip()
                            node_id_to_action[node_id] = action_name

        # Add FSM integrations as subgraphs
        if self.spec.fsm_integrations:
            lines.append("")
            lines.append("    %% FSM Integrations")

            for action_name, fsm_integration in self.spec.fsm_integrations.items():
                lines.append("")
                lines.append(f"    %% FSM for action: {action_name}")
                lines.append(f"    subgraph FSM_{action_name} [\"FSM: {action_name}\"]")

                # Generate state diagram for this FSM
                fsm_diagram = self._generate_fsm_mermaid(action_name, fsm_integration)
                for line in fsm_diagram.split("\n"):
                    if line.strip():
                        lines.append(f"    {line}")

                lines.append("    end")

        # Add connections between BT actions and their FSMs
        if self.spec.fsm_integrations and node_id_to_action:
            lines.append("")
            lines.append("    %% BT-FSM Connections")

            # Find which BT nodes have FSM integrations
            action_to_node_id = {v: k for k, v in node_id_to_action.items()}

            for action_name in self.spec.fsm_integrations.keys():
                if action_name in action_to_node_id:
                    node_id = action_to_node_id[action_name]
                    lines.append(f"    {node_id} -.->|ticks| FSM_{action_name}")

        return "\n".join(lines)

    def _generate_bt_mermaid(self) -> str:
        """Generate BT structure from the namespace."""
        try:
            # Use the BT's to_mermaid if available
            from ..rhizomorph.util import to_mermaid as bt_to_mermaid
            bt_namespace = self.spec.bt_namespace
            if bt_namespace is None:
                return "    Root[\"No BT namespace\"]"

            diagram = bt_to_mermaid(bt_namespace)

            # Remove the ```mermaid wrapper if present
            if diagram.startswith("```"):
                lines = diagram.split("\n")
                if lines[0].startswith("```"):
                    diagram = "\n".join(lines[1:-1])

            # Replace flowchart with graph for consistency
            diagram = diagram.replace("flowchart TD", "TD").replace("flowchart LR", "LR")

            return diagram.strip()
        except Exception:
            # Fallback: simple representation
            return f"    Root[\"{self.spec.name}\"]"

    def _generate_fsm_mermaid(self, action_name: str, fsm_integration: FSMIntegration) -> str:
        """Generate FSM state diagram using shared formatting utilities."""
        from ..common.mermaid import format_state_node, format_transition

        lines = []

        # Get the initial state
        initial = fsm_integration.initial_state
        if callable(initial) and not isinstance(initial, type):
            initial = initial()

        # Try to discover states by following transitions
        from ..septum.core import StateSpec

        discovered: Dict[str, StateSpec] = {}
        to_visit: List[Any] = [initial]  # Can be StateSpec or callable returning StateSpec
        visited: set[str] = set()

        while to_visit:
            state = to_visit.pop(0)

            if callable(state) and not isinstance(state, type):
                try:
                    state = state()
                except Exception:
                    continue

            if not isinstance(state, StateSpec):
                continue

            state_name = state.name

            if state_name in visited:
                continue

            visited.add(state_name)
            discovered[state_name] = state

            # Get transitions and add target states
            try:
                transitions = state.get_transitions()
                for transition in transitions:
                    target_state = None

                    if hasattr(transition, 'transition'):  # type: ignore[attr-defined]
                        target_state = transition.transition  # type: ignore[attr-defined]
                    elif hasattr(transition, 'states'):  # Push
                        target_state = transition  # type: ignore[attr-defined]
                    else:
                        continue

                    # Skip if no target state
                    if target_state is None:
                        continue

                    # Skip special transitions
                    class_name = target_state.__class__.__name__
                    if class_name in ('Again', 'Unhandled', 'Retry', 'Restart', 'Repeat', 'Pop'):
                        continue

                    # Handle Push
                    if class_name == 'Push':
                        for push_state in target_state.states:  # type: ignore[attr-defined]
                            # Get the name to check if visited
                            if hasattr(push_state, 'name'):
                                push_name = push_state.name  # type: ignore[attr-defined]
                            elif callable(push_state):
                                # Will be resolved when popped from to_visit
                                to_visit.append(push_state)
                                continue
                            else:
                                continue
                            if push_name not in visited:
                                to_visit.append(push_state)
                    else:
                        # Regular state transition - only add if it's a StateSpec or callable
                        if isinstance(target_state, StateSpec) or callable(target_state):
                            # Get the name to check if visited
                            if hasattr(target_state, 'name'):
                                target_name = target_state.name  # type: ignore[attr-defined]
                            elif callable(target_state):
                                # Will be resolved when popped from to_visit
                                to_visit.append(target_state)
                                continue
                            else:
                                continue
                            if target_name not in visited:
                                to_visit.append(target_state)

            except Exception:
                pass

        # Add states
        if not discovered:
            return f"    {action_name}_Empty[\"No states discovered\"]"

        added_states = set()

        for state_name, state_spec in discovered.items():
            short_name = state_name.split(".")[-1]
            state_id = f"{action_name}_{short_name}"

            if state_id not in added_states:
                is_initial = state_spec == initial
                lines.append(format_state_node(state_id, short_name, is_initial))
                added_states.add(state_id)

            # Get transitions from this state
            try:
                transitions = state_spec.get_transitions()
                for transition in transitions:
                    target_state = None
                    label = ""

                    if hasattr(transition, 'transition'):
                        target_state = transition.transition  # type: ignore[attr-defined]
                        if hasattr(transition, 'label'):
                            label = transition.label.name  # type: ignore[attr-defined]
                    elif hasattr(transition, 'name'):
                        target_state = transition
                        label = "transition"

                    if target_state is None:
                        continue

                    class_name = target_state.__class__.__name__
                    if class_name in ('Again', 'Unhandled', 'Retry', 'Restart', 'Repeat'):
                        continue
                    if class_name == 'Pop':
                        continue

                    if class_name == 'Push':
                        for push_state in target_state.states:  # type: ignore[attr-defined]
                            push_name = push_state.name if hasattr(push_state, 'name') else push_state.__name__
                            target_short = push_name.split(".")[-1]
                            target_id = f"{action_name}_{target_short}"
                            lines.append(format_transition(state_id, target_id, "Push"))
                    else:
                        target_name = target_state.name if hasattr(target_state, 'name') else str(target_state)  # type: ignore[attr-defined]
                        target_short = target_name.split(".")[-1]
                        target_id = f"{action_name}_{target_short}"
                        lines.append(format_transition(state_id, target_id, label))

            except Exception:
                pass

        return "\n".join(lines)
