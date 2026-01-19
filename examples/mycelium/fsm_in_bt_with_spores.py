#!/usr/bin/env python3
"""
FSM-in-BT with Spores Logging

This example demonstrates:
1. Using MyceliumAdapter to log FSM-in-BT patterns
2. Logging FSM state transitions
3. Logging action execution
4. Object lifecycle tracking
5. Event relationships to objects

The scenario: A robot controller with FSM states that processes tasks.
We log all state transitions, action executions, and object snapshots.
"""

import sys
sys.path.insert(0, "src")

import asyncio
import tempfile
from enum import Enum, auto
from pathlib import Path
from pydantic import BaseModel
from typing import Annotated

from mycorrhizal.septum.core import septum, LabeledTransition, StateConfiguration
from mycorrhizal.mycelium import (
    TreeSporesAdapter,
    tree,
    Action,
    Sequence,
    root,
    TreeRunner,
)
from mycorrhizal.rhizomorph.core import Status
from mycorrhizal.common.timebase import MonotonicClock
from mycorrhizal.spores import configure
from mycorrhizal.spores.models import EventAttr, ObjectRef, ObjectScope
from mycorrhizal.spores.transport import AsyncFileTransport


# ======================================================================================
# Domain Models with Spores Annotations
# ======================================================================================


class Robot(BaseModel):
    """Robot object with logged attributes."""
    id: str
    name: str
    battery: int  # Logged via SporesAttr


# ======================================================================================
# Blackboard with Spores Annotations
# ======================================================================================


class RobotContext(BaseModel):
    """
    Blackboard with Spores annotations for automatic logging.

    EventAttr: Mark fields to extract as event attributes
    ObjectRef: Mark fields as OCEL objects with relationships
    """
    # Event attributes (auto-extracted)
    task_count: Annotated[int, EventAttr]
    current_task: Annotated[str, EventAttr]

    # Object reference (auto-logged with relationship)
    robot: Annotated[Robot, ObjectRef(qualifier="actor", scope=ObjectScope.GLOBAL)]


# ======================================================================================
# FSM States
# ======================================================================================


@septum.state(config=StateConfiguration(can_dwell=True))
def IdleState():
    """Robot is idle, waiting for tasks."""

    @septum.events
    class Events(Enum):
        START = auto()

    @septum.on_state
    async def on_state(ctx):
        task_count = ctx.common.get("task_count", 0)

        if task_count > 0:
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
        battery = ctx.common.get("battery", 100)

        if battery <= 20:
            return Events.LOW_BATTERY

        task_count = ctx.common.get("task_count", 0)

        if task_count > 0:
            ctx.common["task_count"] = task_count - 1
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

        if battery >= 100:
            return Events.CHARGED

        ctx.common["battery"] = min(100, battery + 30)
        return None

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.CHARGED, IdleState),
        ]


# ======================================================================================
# Mycelium Tree with Spores Logging
# ======================================================================================


# Create spores adapter
mycelium_adapter = TreeSporesAdapter(tree_name="RobotController")


@tree
def RobotController():
    """
    Robot controller with comprehensive Spores logging.

    Every action execution is logged with:
    - FSM state
    - Action result
    - Blackboard attributes (with EventAttr annotations)
    - Object relationships (with ObjectRef annotations)
    """

    @Action(fsm=IdleState)
    @mycelium_adapter.log_action(
        event_type="robot_control_tick",
        log_fsm_state=True,
        log_status=True,
        log_blackboard=True,
    )
    async def control_robot(bb, tb, fsm_runner):
        """
        Control the robot with FSM integration.

        This action is logged with:
        - FSM state (which state the FSM is in)
        - Action status (SUCCESS, RUNNING, etc.)
        - Blackboard attributes (task_count, current_task, robot)
        - Object relationships (robot object linked as "actor")
        """
        state_name = fsm_runner.current_state.name if fsm_runner.current_state else "Unknown"

        # Sync FSM state back to blackboard
        if "Idle" in state_name or "Processing" in state_name or "Charging" in state_name:
            bb.robot.battery = fsm_runner.fsm.context.common.get("battery", 100)
            bb.task_count = fsm_runner.fsm.context.common.get("task_count", 0)

        # Update current task description
        if "Processing" in state_name:
            bb.current_task = f"Processing task {10 - bb.task_count}"
            print(f"  [Processing] {bb.current_task}... Battery: {bb.robot.battery}%")

        elif "Charging" in state_name:
            bb.current_task = "Charging"
            print(f"  [Charging] Recharging... Battery: {bb.robot.battery}%")

        elif "Idle" in state_name:
            if bb.task_count == 0:
                bb.current_task = "All tasks complete"
                print(f"  [Idle] All tasks complete! Final battery: {bb.robot.battery}%")
                return Status.FAILURE  # Signal completion

            bb.current_task = "Ready for next task"
            print(f"  [Idle] Ready. Tasks remaining: {bb.task_count}, Battery: {bb.robot.battery}%")
            return Status.RUNNING

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
    """Run the robot controller with Spores logging."""

    print("=" * 70)
    print("Mycelium + Spores: FSM-in-BT with Event Logging")
    print("=" * 70)
    print()
    print("This demo shows:")
    print("  - FSM-in-BT pattern with Mycelium")
    print("  - Automatic event logging via MyceliumAdapter")
    print("  - FSM state tracking in events")
    print("  - Object lifecycle logging (robot)")
    print("  - Event-object relationships")
    print()

    # Create temporary file for spores output
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        log_file = f.name

    print(f"Logging to: {log_file}")
    print()

    # Configure spores
    configure(transport=AsyncFileTransport(log_file))

    # Create blackboard with robot
    robot = Robot(id="robot-1", name="RoboX", battery=100)
    bb = RobotContext(
        task_count=5,
        current_task="Initializing",
        robot=robot,
    )
    tb = MonotonicClock()

    # Create tree runner
    runner = TreeRunner(RobotController, bb=bb, tb=tb)

    print("Initial state:")
    print(f"  Robot: {bb.robot.name} (ID: {bb.robot.id})")
    print(f"  Tasks: {bb.task_count}")
    print(f"  Battery: {bb.robot.battery}%")
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

        if result == Status.FAILURE:
            print()
            print("All tasks complete!")
            break

        if bb.task_count == 0:
            print()
            print("All tasks complete!")
            break

    print("-" * 70)
    print()
    print("Final state:")
    print(f"  Tasks completed: {5 - bb.task_count} / 5")
    print(f"  Battery: {bb.robot.battery}%")
    print()

    # Show log file contents
    print("=" * 70)
    print("Spores Event Log (OCEL format)")
    print("=" * 70)
    print()

    with open(log_file, 'r') as f:
        for i, line in enumerate(f, 1):
            print(f"Record {i}: {line.rstrip()}")

    print()
    print("=" * 70)
    print("What was logged:")
    print("  - Events: robot_control_tick (per action execution)")
    print("  - Objects: Robot (with battery attribute)")
    print("  - Relationships: event --[actor]--> robot-1")
    print("  - Event attributes: task_count, current_task, fsm_state, status")
    print()

    # Clean up
    Path(log_file).unlink()


if __name__ == "__main__":
    asyncio.run(main())
