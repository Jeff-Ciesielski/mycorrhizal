#!/usr/bin/env python3
"""
Mycelium - Unified Orchestration Layer for Mycorrhizal

Mycelium provides a unified API for composing Enoki FSMs and Rhizomorph behavior trees
into coherent systems with bidirectional FSM-BT integration.

Key Features:
    - Unified @tree decorator for composing FSMs and BTs
    - Mirrored Enoki API: @state, @events, @on_state, @transitions (with optional BT integration)
    - FSM-integrated actions: @Action(fsm=SomeFSM) for BT-driven FSM control
    - BT-integrated states: @state(bt=SomeBT) for FSM-driven BT control
    - Unified mermaid diagrams showing full system

Example:
    >>> from mycorrhizal.mycelium import tree, Action, Sequence, state, events, on_state, transitions
    >>> from mycorrhizal.mycelium import LabeledTransition
    >>> from mycorrhizal.rhizomorph.core import bt, Status
    >>>
    >>> # Define states using mirrored Enoki API
    >>> @state()
    >>> def Idle():
    >>>     @events
    >>>     class Events(Enum):
    >>>         START = auto()
    >>>
    >>>     @on_state
    >>>     async def on_state(ctx):
    >>>         return Events.START
    >>>
    >>>     @transitions
    >>>     def transitions():
    >>>         return [LabeledTransition(Events.START, Working)]
    >>>
    >>> # Define BT tree for FSM integration
    >>> @bt.tree
    >>> def DecideNext():
    >>>     @bt.action
    >>>     async def check(bb):
    >>>         return Status.SUCCESS if bb.ready else Status.FAILURE
    >>>
    >>>     @bt.root
    >>>     @Sequence
    >>>     def main():
    >>>         yield check
    >>>
    >>> # Define state with BT integration
    >>> @state(bt=DecideNext)
    >>> def Working():
    >>>     @events
    >>>     class Events(Enum):
    >>>         DONE = auto()
    >>>
    >>>     @on_state
    >>>     async def on_state_handler(ctx, bt_result):
    >>>         if bt_result == Status.SUCCESS:
    >>>             return Events.DONE
    >>>         return None
    >>>
    >>>     @transitions
    >>>     def transitions():
    >>>         return [LabeledTransition(Events.DONE, Idle)]
    >>>
    >>> # Compose in Mycelium tree
    >>> @tree
    >>> def RobotController():
    >>>     @Action(fsm=Idle)
    >>>     async def run(bb, tb, fsm_runner):
    >>>         if "Working" in fsm_runner.current_state.name:
    >>>             return Status.SUCCESS
    >>>         fsm_runner.send_message(Idle.Events.START)
    >>>         return Status.RUNNING
    >>>
    >>>     @root
    >>>     @Sequence
    >>>     def main():
    >>>         yield run
    >>>
    >>> # Use it
    >>> runner = TreeRunner(RobotController, bb=bb, tb=tb)
    >>> await runner.tick()
    >>> print(runner.to_mermaid())
"""

# Public API exports
from .core import (
    tree,
    Action,
    Condition,
    Sequence,
    Selector,
    Parallel,
    root,
    # Mirrored Enoki API
    state,
    events,
    on_state,
    transitions,
    # Enoki types for re-export
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
from .runner import TreeRunner, TreeInstance
from .exceptions import (
    MyceliumError,
    TreeDefinitionError,
    FSMInstantiationError,
)

__all__ = [
    # Core decorators
    "tree",
    "Action",
    "Condition",
    "Sequence",
    "Selector",
    "Parallel",
    "root",

    # Mirrored Enoki API
    "state",
    "events",
    "on_state",
    "transitions",

    # Enoki types
    "LabeledTransition",
    "StateConfiguration",
    "Push",
    "Pop",
    "Again",
    "Unhandled",
    "Retry",
    "Restart",
    "Repeat",

    # Tree runner and instance
    "TreeRunner",
    "TreeInstance",

    # Exceptions
    "MyceliumError",
    "TreeDefinitionError",
    "FSMInstantiationError",
]

# Version info
__version__ = "0.2.0"
