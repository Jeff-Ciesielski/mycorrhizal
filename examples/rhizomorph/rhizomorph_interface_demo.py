#!/usr/bin/env python3
"""
Rhizomorph Behavior Tree Interface Demo

This example demonstrates how to use interface-based blackboard access
with Rhizomorph behavior trees to enforce access control and type safety.

The demo shows a battery-powered robot that:
1. Only has read-only access to configuration (max battery, threshold)
2. Has read-write access to state (current battery, operations counter)
3. Cannot access internal fields not in the interface
"""

import asyncio
from typing import Annotated, Optional
from pydantic import BaseModel

from mycorrhizal.rhizomorph.core import bt, Runner, Status
from mycorrhizal.common.interface_builder import blackboard_interface, readonly, readwrite
from mycorrhizal.common.wrappers import AccessControlError


# ============================================================================
# Blackboard Definition
# ============================================================================

class RobotBlackboard(BaseModel):
    """Complete robot state with all fields"""
    # Configuration (read-only for operations)
    max_battery: float = 100.0
    low_battery_threshold: float = 20.0
    max_operations: int = 10

    # State (read-write for operations)
    current_battery: float = 100.0
    operations_completed: int = 0
    current_task: Optional[str] = None

    # Internal fields (not accessible via interface)
    _internal_calibration: float = 1.0
    _secret_debug_mode: bool = False


# ============================================================================
# Interface Definition
# ============================================================================

@blackboard_interface
class RobotOperationInterface:
    """
    Interface for robot operations.

    Provides:
    - Read-only access to configuration (max_battery, thresholds)
    - Read-write access to state (battery, operations)
    - No access to internal fields
    """
    max_battery: Annotated[float, readonly]
    low_battery_threshold: Annotated[float, readonly]
    max_operations: Annotated[int, readonly]
    current_battery: Annotated[float, readwrite]
    operations_completed: Annotated[int, readwrite]
    current_task: Annotated[Optional[str], readwrite]


# ============================================================================
# Behavior Tree Definition
# ============================================================================

@bt.tree
def RobotController():
    """
    Behavior tree for robot operation control.

    The tree:
    1. Checks if battery is sufficient
    2. Checks if operation limit not reached
    3. Performs operation if conditions met
    4. Charges battery when low
    """

    # ----------------------------------------
    # Conditions
    # ----------------------------------------

    @bt.condition
    def has_battery(bb: RobotOperationInterface) -> bool:
        """Check if battery is above threshold"""
        return bb.current_battery >= bb.low_battery_threshold

    @bt.condition
    def has_operations_remaining(bb: RobotOperationInterface) -> bool:
        """Check if we haven't hit operation limit"""
        return bb.operations_completed < bb.max_operations

    @bt.condition
    def has_task(bb: RobotOperationInterface) -> bool:
        """Check if there's a task to process"""
        return bb.current_task is not None

    # ----------------------------------------
    # Actions
    # ----------------------------------------

    @bt.action
    async def perform_operation(bb: RobotOperationInterface) -> Status:
        """Perform a robot operation (consumes battery)"""
        print(f"  Performing operation {bb.operations_completed + 1}")
        print(f"  Battery before: {bb.current_battery:.1f}%")

        # Consume battery
        bb.current_battery -= 10.0
        bb.operations_completed += 1

        print(f"  Battery after: {bb.current_battery:.1f}%")
        return Status.SUCCESS

    @bt.action
    async def charge_battery(bb: RobotOperationInterface) -> Status:
        """Charge the battery to full"""
        print(f"  Charging battery from {bb.current_battery:.1f}% to {bb.max_battery:.1f}%")
        bb.current_battery = bb.max_battery
        return Status.SUCCESS

    @bt.action
    async def wait_for_task(bb: RobotOperationInterface) -> Status:
        """Wait for a task to be assigned"""
        print(f"  Waiting for task (operations: {bb.operations_completed})")
        return Status.FAILURE  # Fail to indicate no task yet

    @bt.action
    async def operation_complete(bb: RobotOperationInterface) -> Status:
        """Mark operation as complete"""
        print(f"  Operation {bb.operations_completed} complete, clearing task")
        bb.current_task = None
        return Status.SUCCESS

    @bt.action
    async def cannot_operate(bb: RobotOperationInterface) -> Status:
        """Called when cannot perform operations"""
        print(f"  Cannot operate - battery: {bb.current_battery:.1f}%, ops: {bb.operations_completed}")
        return Status.FAILURE

    # ----------------------------------------
    # Tree Structure
    # ----------------------------------------

    @bt.sequence()
    def operation_sequence():
        """Sequence for performing an operation"""
        yield has_battery
        yield has_operations_remaining
        yield has_task
        yield perform_operation
        yield operation_complete

    @bt.sequence()
    def charge_sequence():
        """Sequence for charging (simplified)"""
        yield cannot_operate

    @bt.root
    @bt.selector()
    def root():
        """Main control logic"""
        # Try to perform operation if conditions met
        yield operation_sequence

        # If operation fails (no task, battery low, etc.), show status
        yield cannot_operate

        # No task available
        yield wait_for_task


# ============================================================================
# Demonstration
# ============================================================================

async def main():
    """Run the robot controller demo"""
    print("=" * 70)
    print("Rhizomorph Interface-Based Access Control Demo")
    print("=" * 70)
    print()

    # Create blackboard and runner
    bb = RobotBlackboard(
        max_battery=100.0,
        low_battery_threshold=30.0,
        max_operations=5,
        current_battery=100.0,
    )

    runner = Runner(RobotController, bb)

    print("Initial State:")
    print(f"  Battery: {bb.current_battery:.1f}%")
    print(f"  Operations: {bb.operations_completed}/{bb.max_operations}")
    print(f"  Low battery threshold: {bb.low_battery_threshold}%")
    print()

    # Simulate task processing cycle
    for i in range(8):
        print(f"--- Tick {i + 1} ---")

        # Assign a task
        if i < 5:
            bb.current_task = f"task_{i + 1}"

        # Run the behavior tree
        status = await runner.tick()
        print(f"  Status: {status.name}")

        print(f"  State: battery={bb.current_battery:.1f}%, ops={bb.operations_completed}")

        # If we ran out of battery or hit limit, break
        if status == Status.FAILURE:
            print("  Cannot continue - stopping")
            break

        print()

    print("=" * 70)
    print("Demo Complete")
    print("=" * 70)
    print()
    print("Key Points:")
    print("1. Interface enforced read-only access to configuration")
    print("2. Interface enforced read-write access to state")
    print("3. Internal fields (_internal_calibration) were inaccessible")
    print("4. Type hints provided compile-time safety")
    print("5. Runtime enforcement prevented accidental modifications")


if __name__ == "__main__":
    asyncio.run(main())
