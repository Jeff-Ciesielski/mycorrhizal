#!/usr/bin/env python3
"""
Septum State Machine Utilities

Utility functions for FSM operations like Mermaid diagram generation.
"""
from typing import Dict, List, Set, Optional, Union
from dataclasses import dataclass


@dataclass
class _TransitionInfo:
    """Information about a transition for mermaid diagram"""
    label: str
    target: str
    transition_type: str  # "normal", "push", "pop", "again", "retry", etc.


def _short_state_name(state_name: str) -> str:
    """
    Extract short state name from full qualified name.

    Converts 'examples.septum.network_protocol_fsm.IdleState' to 'IdleState'.
    Handles edge cases like '<string>.IdleState' gracefully.

    Args:
        state_name: Full state name with module path

    Returns:
        Short state name (just the class name)
    """
    if "." in state_name:
        return state_name.rsplit(".", 1)[-1]
    return state_name


def to_mermaid(fsm) -> str:
    """
    Generate Mermaid diagram from a StateMachine instance.

    This creates a flowchart showing all states and their transitions,
    including special handling for Push/Pop transitions.

    Args:
        fsm: A StateMachine instance (from mycorrhizal.septum.core)

    Returns:
        Mermaid diagram as a string

    Example:
        ```python
        from mycorrhizal.septum.core import StateMachine
        from mycorrhizal.septum.util import to_mermaid

        fsm = StateMachine(initial_state=MyState)
        await fsm.initialize()

        # Generate diagram
        mermaid_diagram = to_mermaid(fsm)
        print(mermaid_diagram)
        ```
    """
    # Import here to avoid circular imports
    from mycorrhizal.septum.core import (
        StateSpec,
        StateRef,
        LabeledTransition,
        Push,
        Pop,
        Again,
        Retry,
        Unhandled,
        Repeat,
        Restart,
    )

    lines: List[str] = ["flowchart TD"]
    node_ids: Dict[str, str] = {}
    counter = 0

    # Get all validated states from the registry
    validation_result = fsm.registry.validate_all_states(
        fsm.initial_state, fsm.error_state
    )
    state_names: Set[str] = validation_result.discovered_states

    def nid(state_name: str) -> str:
        """Get or create a node ID for a state"""
        nonlocal counter
        if state_name in node_ids:
            return node_ids[state_name]
        counter += 1
        node_ids[state_name] = f"S{counter}"
        return node_ids[state_name]

    # Add initial state marker
    initial_name = fsm.initial_state.name if isinstance(fsm.initial_state, StateSpec) else str(fsm.initial_state)
    lines.append(f'    start((start)) --> {nid(initial_name)}')

    # Add error state if exists
    if fsm.error_state:
        error_name = fsm.error_state.name if isinstance(fsm.error_state, StateSpec) else str(fsm.error_state)
        lines.append(f'    {nid(error_name)}{{"ERROR"}}')

    # Collect all transitions
    transitions: Dict[str, List[_TransitionInfo]] = {}

    for state_name in state_names:
        state = fsm.registry.resolve_state(state_name)
        if not state:
            continue

        transitions[state_name] = []

        # Get the transition list
        raw_transitions = state.get_transitions()

        # Handle single transition
        if isinstance(raw_transitions, (Again, Retry, Unhandled, Repeat, Restart, Pop)):
            raw_transitions = [raw_transitions]
        elif not isinstance(raw_transitions, list):
            raw_transitions = list(raw_transitions) if raw_transitions else []

        for trans in raw_transitions:
            match trans:
                case LabeledTransition(label, target):
                    match target:
                        case Again():
                            transitions[state_name].append(
                                _TransitionInfo(label.name, state_name, "again")
                            )
                        case Retry():
                            transitions[state_name].append(
                                _TransitionInfo(label.name, state_name, "retry")
                            )
                        case Unhandled():
                            transitions[state_name].append(
                                _TransitionInfo(label.name, state_name, "unhandled")
                            )
                        case Repeat():
                            transitions[state_name].append(
                                _TransitionInfo(label.name, state_name, "repeat")
                            )
                        case Restart():
                            transitions[state_name].append(
                                _TransitionInfo(label.name, state_name, "restart")
                            )
                        case Pop():
                            transitions[state_name].append(
                                _TransitionInfo(label.name, "__pop__", "pop")
                            )
                        case Push() as p:
                            # Push transitions - show all states being pushed
                            for push_state in p.push_states:
                                push_name = push_state.name if isinstance(push_state, StateSpec) else str(push_state)
                                # Use -> instead of unicode arrow for better compatibility
                                transitions[state_name].append(
                                    _TransitionInfo(f"{label.name}->push", push_name, "push")
                                )
                        case StateSpec() as target_state:
                            transitions[state_name].append(
                                _TransitionInfo(label.name, target_state.name, "normal")
                            )
                        case StateRef() as ref:
                            transitions[state_name].append(
                                _TransitionInfo(label.name, ref.state, "normal")
                            )
                        case _:
                            # Unknown transition type
                            pass

                case Again():
                    transitions[state_name].append(
                        _TransitionInfo("", state_name, "again")
                    )
                case Retry():
                    transitions[state_name].append(
                        _TransitionInfo("", state_name, "retry")
                    )
                case Unhandled():
                    transitions[state_name].append(
                        _TransitionInfo("", state_name, "unhandled")
                    )
                case Repeat():
                    transitions[state_name].append(
                        _TransitionInfo("", state_name, "repeat")
                    )
                case Restart():
                    transitions[state_name].append(
                        _TransitionInfo("", state_name, "restart")
                    )
                case Pop():
                    transitions[state_name].append(
                        _TransitionInfo("", "__pop__", "pop")
                    )

    # Generate nodes and edges
    for state_name in state_names:
        state = fsm.registry.resolve_state(state_name)
        if not state:
            continue

        state_id = nid(state_name)
        short_name = _short_state_name(state_name)

        # Determine node shape based on state properties
        # All nodes use quotes to handle special characters
        if state.config.terminal:
            # Terminal state: use double parentheses and note terminal status
            lines.append(f'    {state_id}[["{short_name}<br/>terminal"]]')
        elif state_name == initial_name:
            # Initial state: normal shape
            lines.append(f'    {state_id}["{short_name}"]')
        elif fsm.error_state and state_name == (fsm.error_state.name if isinstance(fsm.error_state, StateSpec) else str(fsm.error_state)):
            # Error state: use rhombus shape
            lines.append(f'    {state_id}{{"{short_name}"}}')
        else:
            # Normal state
            lines.append(f'    {state_id}["{short_name}"]')

        # Add transitions
        for trans in transitions.get(state_name, []):
            if trans.transition_type in ("again", "retry", "repeat", "restart", "unhandled"):
                # Self-loop with transition type
                lines.append(f'    {state_id} -->|"{trans.transition_type}"| {state_id}')
            elif trans.transition_type == "pop":
                # Pop goes to a special pop node
                label = trans.label if trans.label else "pop"
                lines.append(f'    {state_id} -->|"{label}"| pop((pop))')
            elif trans.transition_type == "push":
                # Push transition
                target_id = nid(trans.target)
                lines.append(f'    {state_id} -->|"{trans.label}"| {target_id}')
            else:
                # Normal transition
                target_id = nid(trans.target)
                lines.append(f'    {state_id} -->|"{trans.label}"| {target_id}')

    # Add pop node if any pop transitions exist
    if any(t.transition_type == "pop" for transs in transitions.values() for t in transs):
        lines.append('    pop((pop))')

    return "\n".join(lines)
