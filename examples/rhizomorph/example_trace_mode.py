#!/usr/bin/env python3
"""
Example demonstrating trace mode for Rhizomorph behavior trees.

Trace mode allows you to log every action and condition execution
with their fully qualified names and return status.
"""

import asyncio
import logging
from dataclasses import dataclass

from mycorrhizal.rhizomorph.core import bt, Runner, Status


# Setup logging to see trace output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)-8s %(name)s: %(message)s'
)

# Create a dedicated trace logger
trace_logger = logging.getLogger("bt.trace")
trace_logger.setLevel(logging.DEBUG)


@dataclass
class RobotBlackboard:
    """Blackboard for robot control example."""
    battery_level: float = 100.0
    distance_to_target: float = 10.0
    task_completed: bool = False


@bt.tree
def RobotControlTree():
    """Behavior tree for robot navigation."""

    @bt.condition
    def has_battery(bb: RobotBlackboard) -> bool:
        """Check if robot has enough battery."""
        return bb.battery_level > 20.0

    @bt.condition
    def at_target(bb: RobotBlackboard) -> bool:
        """Check if robot reached target."""
        return bb.distance_to_target <= 0.5

    @bt.action
    def move_to_target(bb: RobotBlackboard) -> Status:
        """Move robot toward target."""
        bb.distance_to_target -= 2.0
        bb.battery_level -= 5.0
        print(f"  Moving... distance: {bb.distance_to_target:.1f}m, battery: {bb.battery_level:.1f}%")
        if bb.distance_to_target <= 0.5:
            return Status.SUCCESS
        return Status.RUNNING

    @bt.action
    def complete_task(bb: RobotBlackboard) -> Status:
        """Mark task as completed."""
        bb.task_completed = True
        print(f"  Task completed! Final battery: {bb.battery_level:.1f}%")
        return Status.SUCCESS

    @bt.action
    def return_to_charger(bb: RobotBlackboard) -> Status:
        """Return to charging station."""
        print("  Low battery! Returning to charger...")
        bb.battery_level = 0.0
        return Status.SUCCESS

    @bt.root
    @bt.sequence
    def root():
        """Main control sequence."""
        # Check if we have battery
        yield bt.selector(
            has_battery,
            return_to_charger  # If no battery, return to charger
        )
        # Move to target or complete if already there
        yield bt.selector(
            complete_task,  # If at target, complete task
            bt.sequence(
                at_target,  # Check if at target
                move_to_target  # If not, move towards it
            )
        )


async def main():
    """Run the behavior tree with trace logging."""

    print("=" * 60)
    print("Example 1: Running WITH trace logging")
    print("=" * 60)

    bb = RobotBlackboard()
    runner = Runner(
        RobotControlTree,
        bb=bb,
        trace=trace_logger  # Enable trace logging
    )

    print("\nInitial state:")
    print(f"  Battery: {bb.battery_level:.1f}%")
    print(f"  Distance: {bb.distance_to_target:.1f}m")

    print("\nExecuting behavior tree:")
    result = await runner.tick_until_complete()

    print(f"\nFinal state:")
    print(f"  Battery: {bb.battery_level:.1f}%")
    print(f"  Distance: {bb.distance_to_target:.1f}m")
    print(f"  Task completed: {bb.task_completed}")
    print(f"  Result: {result.name}")

    print("\n" + "=" * 60)
    print("Example 2: Running WITHOUT trace logging (no output)")
    print("=" * 60)

    bb2 = RobotBlackboard()
    runner2 = Runner(
        RobotControlTree,
        bb=bb2,
        trace=None  # Disable trace logging
    )

    print("\nNotice: No trace logs appear below!\n")
    result2 = await runner2.tick_until_complete()
    print(f"Result: {result2.name}")


if __name__ == "__main__":
    asyncio.run(main())
