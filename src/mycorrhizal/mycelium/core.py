#!/usr/bin/env python3
"""
Core Mycelium decorators - @tree, @Action, @Condition, @on_state

This module provides the main user-facing decorators for creating
unified Mycelium trees with FSMs and BTs.

It also mirrors the Septum API (state, events, on_state, transitions) for
convenience and to support BT integration in FSM states.
"""

from __future__ import annotations

__all__ = [
    # BT decorators
    "tree",
    "Action",
    "Condition",
    "Sequence",
    "Selector",
    "Parallel",
    "root",
    # Mirrored Septum API
    "state",
    "events",
    "on_state",
    "transitions",
    # Septum types (re-exported)
    "LabeledTransition",
    "StateConfiguration",
    "Push",
    "Pop",
    "Again",
    "Unhandled",
    "Retry",
    "Restart",
    "Repeat",
]

import inspect
import functools
from typing import Callable, Type, TYPE_CHECKING
from enum import Enum

from .tree_spec import FSMIntegration, BTIntegration, NodeDefinition
from .tree_builder import MyceliumTreeBuilder, get_current_builder, set_current_builder
from .exceptions import TreeDefinitionError

# Re-export BT decorators for convenience
from ..rhizomorph.core import bt, Status

# Import Septum decorators for mirroring
if TYPE_CHECKING:
    from ..septum.core import _SeptumDecoratorAPI as SeptumDecoratorAPI
else:
    from ..septum.core import _SeptumDecoratorAPI as SeptumDecoratorAPI

# Import Septum types for re-export
from ..septum.core import (
    LabeledTransition,
    StateConfiguration,
    Push,
    Pop,
    Again,
    Unhandled,
    Retry,
    Restart,
    Repeat,
)

# Create a singleton SeptumDecoratorAPI instance for accessing decorators
_septum = SeptumDecoratorAPI()


# ======================================================================================
# @tree decorator
# ======================================================================================


def tree(func: Callable) -> Callable:
    """
    Decorator to define a Mycelium tree.

    A tree can contain BT actions (@Action), BT conditions (@Condition),
    BT composites (@Sequence, @Selector), and FSM/BT integrations.

    Example:
        >>> @tree
        >>> def RobotController():
        >>>     @Action(fsm=TaskFSM)
        >>>     async def process(bb, tb, fsm_state):
        >>>         if "Idle" in fsm_state.name:
        >>>             return Status.SUCCESS
        >>>         return Status.RUNNING
        >>>
        >>>     @root
        >>>     @Sequence
        >>>     def main():
        >>>         yield process
    """

    # Build the tree spec builder
    builder = MyceliumTreeBuilder(
        name=func.__name__,
        tree_func=func,
    )

    # Set as current builder so @Action, @Condition, @root can register
    set_current_builder(builder)

    # Use bt.tree to create the BT namespace
    # This calls func() which registers all BT nodes with the builder
    try:
        bt_namespace = bt.tree(func)
    finally:
        # Clear current builder
        set_current_builder(None)

    # Now build the Mycelium spec with all the registered nodes and integrations
    spec = builder.build()

    # Update spec with BT namespace
    spec.bt_namespace = bt_namespace
    spec.tree_func = func

    # Attach spec to the namespace
    bt_namespace._mycelium_spec = spec

    # Also attach to original function for direct access
    func._mycelium_spec = spec

    return bt_namespace


# ======================================================================================
# @Action decorator with FSM integration
# ======================================================================================


def Action(func=None, *, fsm=None):
    """
    Decorator to define a BT action within a tree.

    Actions can optionally include an FSM integration via the `fsm` parameter.
    When `fsm` is provided, the FSM will auto-tick each time the action runs,
    and the resulting state will be passed to the action.

    Examples:
        >>> # Vanilla action (portable)
        >>> @Action
        >>> async def my_action(bb, tb):
        >>>     return Status.SUCCESS
        >>>
        >>> # Action with FSM integration
        >>> @Action(fsm=TaskFSM)
        >>> async def process(bb, tb, fsm_state):
        >>>     # fsm_state is where the FSM came to rest after ticking
        >>>     if "Idle" in fsm_state.name:
        >>>         return Status.SUCCESS
        >>>     return Status.RUNNING
    """
    def decorator(f: Callable) -> Callable:
        builder = get_current_builder()
        if builder is None:
            raise TreeDefinitionError(
                f"@Action decorator used outside of @tree. "
                f"Action '{f.__name__}' must be defined inside a @tree."
            )

        # Check if function has FSM integration
        has_fsm = fsm is not None

        # Determine signature
        sig = inspect.signature(f)
        params = list(sig.parameters.keys())

        # Validate signature based on whether FSM is integrated
        if has_fsm:
            # FSM actions should have (bb, tb, fsm_runner) signature
            if len(params) < 3:
                raise TreeDefinitionError(
                    f"Action '{f.__name__}' has FSM integration but incorrect signature. "
                    f"Expected (bb, tb, fsm_runner), got {params}"
                )

        # Create FSM integration if needed
        fsm_integration = None
        if has_fsm:
            fsm_integration = FSMIntegration(
                initial_state=fsm,
                action_name=f.__name__,
            )
            builder.add_fsm_integration(fsm_integration)

        # Store action name for runtime FSM lookup
        action_name_for_fsm = f.__name__ if has_fsm else None

        # Create BT action that wraps the original function
        async def bt_action(bb, tb):
            # FSM integration: tick FSM and pass result
            if has_fsm:
                # Get tree instance from blackboard (injected by TreeInstance)
                tree_instance = getattr(bb, '_mycelium_tree_instance', None)
                if tree_instance is None:
                    raise RuntimeError(
                        f"Action '{f.__name__}' has FSM integration but no tree instance available"
                    )

                # Get FSM runner for this action
                fsm_runner = tree_instance.get_fsm_runner(action_name_for_fsm)
                if fsm_runner is None:
                    raise RuntimeError(
                        f"Action '{f.__name__}' has FSM integration but no FSM runner found"
                    )

                # Tick the FSM with timeout=0 (non-blocking)
                _ = await fsm_runner.tick(timeout=0)

                # Call the original function with FSM runner (has current_state + send_message)
                return await f(bb, tb, fsm_runner)
            else:
                # Vanilla action - just call the function
                return await f(bb, tb)

        # Set name BEFORE applying @bt.action decorator
        bt_action.__name__ = f.__name__
        bt_action.__qualname__ = f.__qualname__

        # Now apply the BT decorator
        bt_action = bt.action(bt_action)

        # Create node definition with FSM integration
        node_def = NodeDefinition(
            name=f.__name__,
            node_type="action",
            func=f,  # Store original function
            fsm_integration=fsm_integration,
        )

        # Store metadata for later
        bt_action._mycelium_node_def = node_def
        bt_action._original_func = f
        bt_action._has_fsm_integration = has_fsm
        f._bt_action = bt_action

        # Register with builder
        builder.add_action(node_def)

        return bt_action

    # Support both @Action and @Action(fsm=...) syntax
    if func is None:
        # Called with arguments: @Action(fsm=...)
        return decorator
    else:
        # Called without arguments: @Action
        return decorator(func)


# ======================================================================================
# @Condition decorator
# ======================================================================================


def Condition(func: Callable) -> Callable:
    """
    Decorator to define a BT condition within a tree.

    Conditions are like actions but return bool instead of Status.

    Example:
        >>> @Condition
        >>> def has_tasks(bb):
        >>>     return len(bb.task_queue) > 0
    """

    builder = get_current_builder()
    if builder is None:
        raise TreeDefinitionError(
            f"@Condition decorator used outside of @tree. "
            f"Condition '{func.__name__}' must be defined inside a @tree."
        )

    # Create node definition
    node_def = NodeDefinition(
        name=func.__name__,
        node_type="condition",
        func=func,
    )

    # Also register with BT system
    def bt_condition(bb):
        return func(bb)

    # Set name BEFORE applying @bt.condition decorator
    bt_condition.__name__ = func.__name__
    bt_condition.__qualname__ = func.__qualname__

    # Apply the BT decorator
    bt_condition = bt.condition(bt_condition)

    # Store metadata
    bt_condition._mycelium_node_def = node_def
    bt_condition._original_func = func
    func._bt_condition = bt_condition

    # Register with builder
    builder.add_condition(node_def)

    return bt_condition


# ======================================================================================
# BT Composite decorators
# ======================================================================================


def Sequence(func: Callable) -> Callable:
    """
    Decorator to define a sequence composite.

    Children are executed in order until one fails.
    """
    return bt.sequence(func)


def Selector(func: Callable) -> Callable:
    """
    Decorator to define a selector composite.

    Children are executed in order until one succeeds.
    """
    return bt.selector(func)


def Parallel(func: Callable) -> Callable:
    """
    Decorator to define a parallel composite.

    All children are executed simultaneously.
    """
    return bt.parallel(func)


# ======================================================================================
# @root decorator
# ======================================================================================


def root(func: Callable) -> Callable:
    """
    Decorator to mark the root node of a tree.

    Must be used once per tree.

    Example:
        >>> @root
        >>> @Sequence
        >>> def main():
        >>>     yield action1
        >>>     yield action2
    """

    builder = get_current_builder()
    if builder is None:
        raise TreeDefinitionError(
            f"@root decorator used outside of @tree. "
            f"Root '{func.__name__}' must be defined inside a @tree."
        )

    # Mark as root in the BT system
    bt_root = bt.root(func)

    # Create node definition
    node_def = NodeDefinition(
        name=func.__name__,
        node_type="root",
        func=func,
        is_root=True,
    )

    # Store metadata
    bt_root._mycelium_node_def = node_def

    # Register with builder
    builder.set_root(node_def)

    return bt_root


# ======================================================================================
# Mirrored Septum API - state, events, on_state, transitions
# ======================================================================================


# Global registry to track BT integrations for states
_state_bt_integrations: dict[str, BTIntegration] = {}


def state(bt=None, **kwargs):
    """
    Decorator to define a state (mirrors @septum.state).

    Supports optional BT integration via the `bt` parameter.
    When `bt` is provided, the state's on_state handler will automatically
    receive the BT result as a parameter.

    Examples:
        >>> # Standard state (no BT integration)
        >>> @state()
        >>> def MyState():
        >>>     @events
        >>>     class Events(Enum):
        >>>         DONE = auto()
        >>>
        >>>     @on_state
        >>>     async def handler(ctx):
        >>>         return Events.DONE
        >>>
        >>>     @transitions
        >>>     def transitions():
        >>>         return [LabeledTransition(Events.DONE, OtherState)]

        >>> # State with BT integration
        >>> @state(bt=DecideBT)
        >>> def MyState():
        >>>     @events
        >>>     class Events(Enum):
        >>>         DONE = auto()
        >>>
        >>>     @on_state
        >>>     async def handler(ctx, bt_result):
        >>>         if bt_result == Status.SUCCESS:
        >>>             return Events.DONE
        >>>         return None

        >>> # State with config
        >>> @state(config=StateConfiguration(can_dwell=True))
        >>> def MyState():
        >>>     @events
        >>>     class Events(Enum):
        >>>         DONE = auto()

    Args:
        bt: Optional BT tree to integrate into this state
        **kwargs: Additional arguments passed to septum.state() (e.g., config)

    Returns:
        State function decorated with septum.state()
    """
    def decorator(func: Callable) -> Callable:
        # Store BT integration metadata if provided
        if bt is not None:
            integration = BTIntegration(
                bt_tree=bt,
                state_name=func.__qualname__,
                state_func=func,
            )
            _state_bt_integrations[func.__qualname__] = integration

        # Apply septum.state decorator with additional kwargs
        return _septum.state(**kwargs)(func)

    return decorator


def events(cls: Type[Enum]) -> Type[Enum]:
    """
    Decorator to define state events (mirrors @septum.events).

    Example:
        >>> @state()
        >>> def MyState():
        >>>     @events
        >>>     class Events(Enum):
        >>>         START = auto()
        >>>         STOP = auto()
    """
    return _septum.events(cls)


# Make on_state work without bt parameter for standard usage
# It will check for BT integration from the @state decorator
def on_state(func: Callable = None, *, bt_tree=None) -> Callable:
    """
    Decorator for state handlers (mirrors @septum.on_state).

    When used with @state(bt=SomeBT), this decorator automatically
    wraps the handler to receive the BT result.

    Examples:
        >>> # Standard usage (no BT integration)
        >>> @state()
        >>> def MyState():
        >>>     @on_state
        >>>     async def handler(ctx):
        >>>         return Events.DONE
        >>>
        >>>     @transitions
        >>>     def transitions():
        >>>         return [LabeledTransition(Events.DONE, OtherState)]

        >>> # With BT integration (BT from @state decorator)
        >>> @state(bt=DecideBT)
        >>> def MyState():
        >>>     @on_state  # Automatically wraps to receive bt_result
        >>>     async def handler(ctx, bt_result):
        >>>         if bt_result == Status.SUCCESS:
        >>>             return Events.DONE

    Args:
        func: The state handler function
        bt_tree: (Deprecated) Use @state(bt=...) instead

    Returns:
        Decorated state handler function
    """
    def decorator(f: Callable) -> Callable:
        # Check if this state has BT integration registered
        # Extract state name from qualname (handles nested functions like MyState.<locals>.handler)
        qualname = f.__qualname__
        if '.<locals>.' in qualname:
            # Nested function - extract parent state name
            state_name = qualname.split('.<locals>.')[0]
        else:
            state_name = qualname

        bt_integration = _state_bt_integrations.get(state_name)

        if bt_integration is not None:
            # Wrap the function to inject BT result
            sig = inspect.signature(f)
            params = list(sig.parameters.keys())
            expects_bt_result = len(params) >= 2 and params[1] == 'bt_result'

            @functools.wraps(f)
            async def wrapped(ctx, *args, **kwargs):
                # Get BT runner from context
                bt_runner = getattr(ctx, '_mycelium_bt_runner', None)

                if bt_runner is None:
                    # No BT runner - call original function
                    if expects_bt_result:
                        return await f(ctx, None, *args, **kwargs)
                    else:
                        return await f(ctx, *args, **kwargs)

                # Run the BT to completion
                bt_result = Status.RUNNING
                max_ticks = 10
                ticks = 0
                while bt_result == Status.RUNNING and ticks < max_ticks:
                    bt_result = await bt_runner.tick()
                    ticks += 1

                # Call the original function with BT result
                return await f(ctx, bt_result, *args, **kwargs)

            # Mark as septum on_state for validation (use func itself, not True)
            wrapped._septum_on_state = wrapped

            # Add to tracking stack if available
            if _septum._tracking_stack:
                _septum._tracking_stack[-1].append((f.__name__, wrapped))

            return wrapped
        else:
            # No BT integration - just use septum.on_state
            return _septum.on_state(f)

    # Support both @on_state and @on_state(bt_tree=...) syntax
    if func is None:
        # Called with arguments: @on_state(...)
        if bt_tree is not None:
            # Legacy syntax - user provided bt_tree here
            # We could handle this, but recommend @state(bt=...) instead
            import warnings
            warnings.warn(
                "Using @on_state(bt_tree=...) is deprecated. "
                "Use @state(bt=...) instead.",
                DeprecationWarning,
                stacklevel=2
            )
        return decorator
    else:
        # Called without arguments: @on_state
        return decorator(func)


def transitions(func: Callable) -> Callable:
    """
    Decorator to define state transitions (mirrors @septum.transitions).

    Example:
        >>> @state()
        >>> def MyState():
        >>>     @transitions
        >>>     def transitions():
        >>>         return [LabeledTransition(Events.DONE, OtherState)]
    """
    return _septum.transitions(func)
