#!/usr/bin/env python3
"""
Robot Controller Demo - Mycelium FSM-in-BT Integration

This example demonstrates FSM-integrated actions where a BT action
has an FSM that auto-ticks each time the action runs.

The robot:
1. Processes tasks one at a time
2. Battery depletes with each task
3. Returns to charging when battery is low
4. Continues until all tasks are complete
"""

import sys
sys.path.insert(0, "src")

import asyncio
from enum import Enum, auto
from pydantic import BaseModel

from mycorrhizal.septum.core import septum, LabeledTransition, StateConfiguration
from mycorrhizal.mycelium import (
    tree,
    Action,
    Sequence,
    root,
    TreeRunner,
)
from mycorrhizal.rhizomorph.core import Status
from mycorrhizal.common.timebase import MonotonicClock


# ======================================================================================
# State Definitions
# ======================================================================================


@septum.state(config=StateConfiguration(can_dwell=True))
def IdleState():
    """Robot is idle, waiting for tasks."""

    @septum.events
    class Events(Enum):
        START = auto()

    @septum.on_state
    async def on_state(ctx):
        # Check if we have tasks and battery
        task_count = ctx.common.get("task_count", 0)
        battery = ctx.common.get("battery", 100)

        if task_count > 0 and battery > 0:
            return Events.START
        return None

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.START, ProcessingState),
        ]


@septum.state()
def ProcessingState():
    """Robot is actively processing a task."""

    @septum.events
    class Events(Enum):
        COMPLETE = auto()
        LOW_BATTERY = auto()

    @septum.on_state
    async def on_state(ctx):
        # Check battery BEFORE processing
        battery = ctx.common.get("battery", 100)

        if battery <= 20:
            # Battery too low, must charge first
            return Events.LOW_BATTERY

        # Process one task
        task_count = ctx.common.get("task_count", 0)

        if task_count > 0:
            ctx.common["task_count"] = task_count - 1
            # Use 20 battery per task (5 tasks before recharge)
            ctx.common["battery"] = battery - 20
            return Events.COMPLETE

        return Events.COMPLETE

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.COMPLETE, IdleState),
            LabeledTransition(Events.LOW_BATTERY, ChargingState),
        ]


@septum.state(config=StateConfiguration(can_dwell=True))
def ChargingState():
    """Robot is charging battery."""

    @septum.events
    class Events(Enum):
        CHARGED = auto()

    @septum.on_state
    async def on_state(ctx):
        battery = ctx.common.get("battery", 0)

        # Charge to 100%
        if battery >= 100:
            return Events.CHARGED

        # Charge 30% per tick
        ctx.common["battery"] = min(100, battery + 30)
        return None

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.CHARGED, IdleState),
        ]


# ======================================================================================
# Blackboard
# ======================================================================================


class RobotBlackboard(BaseModel):
    """Shared state for the robot controller."""

    task_count: int = 10
    battery: int = 100


# ======================================================================================
# Mycelium Tree
# ======================================================================================


@tree
def RobotController():
    """
    Robot controller with FSM-integrated action.

    The FSM handles the robot's operational state (idle/processing/charging)
    and automatically transitions between them based on battery and tasks.
    """

    @Action(fsm=IdleState)
    async def control_robot(bb, tb, fsm_runner):
        """
        Control the robot using the integrated FSM.

        The FSM auto-ticks each time this action runs, handling:
        - Task processing
        - Battery management
        - Automatic charging when low
        """
        state_name = fsm_runner.current_state.name if fsm_runner.current_state else "Unknown"

        # Sync FSM state back to blackboard
        if "Idle" in state_name or "Processing" in state_name or "Charging" in state_name:
            bb.battery = fsm_runner.fsm.context.common.get("battery", 100)
            bb.task_count = fsm_runner.fsm.context.common.get("task_count", 0)

        # Report what's happening
        if "Processing" in state_name:
            print(f"  [Processing] Task in progress... Battery: {bb.battery}%")

        elif "Charging" in state_name:
            print(f"  [Charging] Recharging... Battery: {bb.battery}%")

        elif "Idle" in state_name:
            # Check if all work is done
            if bb.task_count == 0:
                print(f"  [Idle] All tasks complete! Final battery: {bb.battery}%")
                return Status.FAILURE  # Signal completion

            # More work to do, keep the FSM running
            print(f"  [Idle] Ready for next task. Tasks remaining: {bb.task_count}, Battery: {bb.battery}%")
            return Status.RUNNING

        # Keep running
        return Status.RUNNING

    @root
    @Sequence
    def main():
        """Main behavior tree."""
        yield control_robot


# ======================================================================================
# Main Execution
# ======================================================================================


async def main():
    """Run the robot controller demo."""

    print("=" * 70)
    print("Mycelium FSM-in-BT Demo: Robot Controller")
    print("=" * 70)
    print()
    print("This demo shows a robot that:")
    print("  - Processes tasks (uses 20% battery per task)")
    print("  - Goes to charging when battery <= 20%")
    print("  - Charges to 100% before resuming work")
    print("  - Uses FSM-integrated action with @Action(fsm=IdleState)")
    print()

    # Create blackboard
    bb = RobotBlackboard(task_count=10, battery=100)
    tb = MonotonicClock()

    # Create tree runner
    runner = TreeRunner(RobotController, bb=bb, tb=tb)

    print("Initial state:")
    print(f"  Tasks: {bb.task_count}")
    print(f"  Battery: {bb.battery}%")
    print()

    # Run until all tasks are done
    max_ticks = 30
    print(f"Running robot (max {max_ticks} ticks)...")
    print("-" * 70)

    tick_num = 0
    for i in range(max_ticks):
        tick_num += 1
        print(f"Tick {tick_num}:")

        result = await runner.tick()

        # Check if we're done
        if result == Status.FAILURE:
            print()
            print("All tasks complete!")
            break

        # Safety check
        if bb.task_count == 0:
            print()
            print("All tasks complete!")
            break

    print("-" * 70)
    print()
    print("Final state:")
    print(f"  Tasks completed: {10 - bb.task_count} / 10")
    print(f"  Battery: {bb.battery}%")
    print()

    # Generate Mermaid diagram
    print("=" * 70)
    print("Mycelium Unified Diagram")
    print("=" * 70)
    print()
    print(runner.to_mermaid())
    print()
    print("Paste the above into https://mermaid.live to visualize!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    asyncio.run(main())
