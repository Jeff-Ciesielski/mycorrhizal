#!/usr/bin/env python3
"""
Tree runner - public API for running Mycelium trees.

This module provides the TreeRunner class which is the main public API
for creating and running Mycelium trees.
"""

from __future__ import annotations

from typing import Any, Union
from pydantic import BaseModel

from ..rhizomorph.core import Status
from .tree_spec import MyceliumTreeSpec
from .instance import TreeInstance


def _get_spec(tree_input: Union[callable, MyceliumTreeSpec]) -> MyceliumTreeSpec:
    """Extract MyceliumTreeSpec from various input types."""
    if isinstance(tree_input, MyceliumTreeSpec):
        return tree_input

    # Assume it's a decorated tree function
    if hasattr(tree_input, '_mycelium_spec'):
        return tree_input._mycelium_spec

    raise TypeError(
        f"Expected @tree decorated function or MyceliumTreeSpec, got {type(tree_input)}"
    )


class TreeRunner:
    """
    Public API for running a Mycelium tree.

    The runner creates a TreeInstance with FSM instances and provides
    methods for ticking, running, and inspecting the tree.

    Example:
        >>> @tree
        >>> def RobotController():
        >>>     @fsm(initial=Idle)
        >>>     def robot():
        >>>         pass
        >>>
        >>>     @Action
        >>>     async def run(bb, tb, tree):
        >>>         await tree.fsms.robot.send_message(Events.START)
        >>>         return Status.SUCCESS
        >>>
        >>>     @root
        >>>     @Sequence
        >>>     def main():
        >>>         yield run
        >>>
        >>> # Use it
        >>> bb = RobotBlackboard(task_queue=["task1"])
        >>> tb = MonotonicClock()
        >>> runner = TreeRunner(RobotController, bb=bb, tb=tb)
        >>>
        >>> # Run ticks
        >>> result = await runner.tick()
        >>>
        >>> # Or run to completion
        >>> await runner.run(max_ticks=10)
        >>>
        >>> # Get FSM
        >>> fsm = runner.fsms.robot
        >>>
        >>> # Generate diagram
        >>> print(runner.to_mermaid())
    """

    def __init__(
        self,
        tree: Union[callable, MyceliumTreeSpec],
        bb: BaseModel,
        tb: Any,
    ):
        """
        Create a new TreeRunner.

        Args:
            tree: @tree decorated function or MyceliumTreeSpec
            bb: Blackboard (shared state)
            tb: Timebase (shared timing)
        """
        self.spec = _get_spec(tree)
        self.bb = bb
        self.tb = tb

        # Create the tree instance
        self.instance = TreeInstance(
            spec=self.spec,
            bb=self.bb,
            tb=self.tb,
        )

    @property
    def fsms(self):
        """
        Access FSM instances.

        Provides attribute and dict-style access:
            runner.fsms.robot
            runner.fsms['robot']
        """
        return self.instance.fsms

    async def tick(self) -> Status:
        """
        Execute one tick of the tree.

        This will auto-tick all FSMs with auto_tick=True, then
        execute the BT tree once.

        Returns:
            Status from the tree tick
        """
        return await self.instance.tick()

    async def run(self, max_ticks: int = 100) -> Status:
        """
        Run the tree until completion or max_ticks reached.

        Args:
            max_ticks: Maximum number of ticks to execute

        Returns:
            Final status
        """
        for _ in range(max_ticks):
            result = await self.tick()
            if result in (Status.SUCCESS, Status.FAILURE, Status.ERROR):
                return result

        return Status.RUNNING

    def reset(self) -> None:
        """
        Reset the tree instance.

        Creates new FSM instances and clears any state.
        """
        # Create new instance
        self.instance = TreeInstance(
            spec=self.spec,
            bb=self.bb,
            tb=self.tb,
        )

    def to_mermaid(self) -> str:
        """
        Generate a unified Mermaid diagram.

        Returns:
            Mermaid diagram code showing BT and all FSMs
        """
        diagram = self.instance.to_mermaid()
        # Wrap in mermaid code block
        return f"```mermaid\n{diagram}\n```"

    def __repr__(self) -> str:
        return f"TreeRunner(tree={self.spec.name}, fsms={list(self.spec.fsms.keys())})"
