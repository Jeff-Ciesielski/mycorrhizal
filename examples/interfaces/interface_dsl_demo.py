#!/usr/bin/env python3
"""
Simple demonstration of the @blackboard_interface DSL itself.

This shows how to define and use interfaces before integrating with the three DSLs.
"""

import sys
sys.path.insert(0, "src")

from typing import Annotated
from pydantic import BaseModel
from mycorrhizal.common.interface_builder import blackboard_interface, readonly, readwrite
from mycorrhizal.common.wrappers import create_view_from_protocol, AccessControlError


# ============================================================================
# Example 1: Basic Interface Definition
# ============================================================================

print("="*60)
print("Example 1: Basic Interface Definition")
print("="*60)

class SensorData(BaseModel):
    """Blackboard with sensor data"""
    temperature: float = 20.0
    humidity: float = 50.0
    pressure: float = 1013.25

    # Internal calibration data (not exposed via interface)
    # Note: using regular names but hiding via interface
    calibration_offset: float = 0.0
    last_calibration: str = "factory"


@blackboard_interface
class ReadOnlySensor:
    """Read-only interface for sensor data"""
    temperature: Annotated[float, readonly]
    humidity: Annotated[float, readonly]
    pressure: Annotated[float, readonly]


@blackboard_interface
class CalibrationInterface:
    """Interface for calibration (can read and write calibration data)"""
    temperature: Annotated[float, readonly]
    humidity: Annotated[float, readonly]
    pressure: Annotated[float, readonly]
    calibration_offset: Annotated[float, readwrite]
    last_calibration: Annotated[str, readwrite]


# Create blackboard
sensor = SensorData()
print(f"\nInitial sensor data: temp={sensor.temperature}, humidity={sensor.humidity}")

# Create read-only view
readonly_view = create_view_from_protocol(sensor, ReadOnlySensor)
print(f"Read-only view can read temperature: {readonly_view.temperature}")

# Try to modify through read-only view
try:
    readonly_view.temperature = 999.0
    print("ERROR: Should not have been able to modify readonly field!")
except AccessControlError as e:
    print(f"✓ Readonly protected: {type(e).__name__}")

# Try to access internal field through read-only view
try:
    _ = readonly_view.calibration_offset
    print("ERROR: Should not have been able to access internal field!")
except (AccessControlError, AttributeError) as e:
    print(f"✓ Internal field hidden: {type(e).__name__}")


# ============================================================================
# Example 2: ReadWrite Fields
# ============================================================================

print("\n" + "="*60)
print("Example 2: ReadWrite Fields")
print("="*60)

class TaskState(BaseModel):
    """Task tracking state"""
    max_tasks: int = 10
    completed_tasks: int = 0
    failed_tasks: int = 0

    # Internal tracking (hidden from interface)
    internal_counter: int = 0


@blackboard_interface
class TaskInterface:
    """Interface for task management"""
    max_tasks: Annotated[int, readonly]
    completed_tasks: Annotated[int, readwrite]
    failed_tasks: Annotated[int, readwrite]


task_state = TaskState()
task_view = create_view_from_protocol(task_state, TaskInterface)

print(f"\nInitial state: completed={task_view.completed_tasks}, max={task_view.max_tasks}")

# Can modify readwrite field
task_view.completed_tasks = 5
print(f"✓ Modified completed_tasks: {task_view.completed_tasks}")

# Cannot modify readonly field
try:
    task_view.max_tasks = 999
    print("ERROR: Should not have been able to modify max_tasks!")
except AccessControlError as e:
    print(f"✓ Readonly max_tasks protected: {type(e).__name__}")

# Cannot access internal field
try:
    _ = task_view.internal_counter
    print("ERROR: Should not have been able to access internal_counter!")
except (AccessControlError, AttributeError) as e:
    print(f"✓ Internal field hidden: {type(e).__name__}")


# ============================================================================
# Example 3: Type Safety and Introspection
# ============================================================================

print("\n" + "="*60)
print("Example 3: Type Safety and Introspection")
print("="*60)

print(f"\nInterface metadata:")
print(f"  ReadOnlySensor._readonly_fields: {ReadOnlySensor._readonly_fields}")
print(f"  ReadOnlySensor._readwrite_fields: {ReadOnlySensor._readwrite_fields}")

print(f"\nTaskInterface metadata:")
print(f"  TaskInterface._readonly_fields: {TaskInterface._readonly_fields}")
print(f"  TaskInterface._readwrite_fields: {TaskInterface._readwrite_fields}")

# Check if a class has interface metadata
print(f"\nDoes ReadOnlySensor have interface metadata? {hasattr(ReadOnlySensor, '_readonly_fields')}")
print(f"Does SensorData have interface metadata? {hasattr(SensorData, '_readonly_fields')}")


# ============================================================================
# Example 4: Multiple Interfaces for Different Access Levels
# ============================================================================

print("\n" + "="*60)
print("Example 4: Multiple Interfaces for Different Access Levels")
print("="*60)

class RobotState(BaseModel):
    """Robot state with multiple access levels"""
    # Safety-critical config (readonly for most)
    max_velocity: float = 1.0
    emergency_stop_limit: float = 2.0

    # Operational state (readwrite for operators)
    current_velocity: float = 0.0
    battery_level: float = 100.0

    # Maintenance only (special interface)
    # Note: Pydantic doesn't allow field names starting with underscore without special config
    # So we use regular names but hide them via interface
    maintenance_log: list = []
    service_count: int = 0


@blackboard_interface
class OperatorInterface:
    """Interface for normal robot operation"""
    max_velocity: Annotated[float, readonly]
    emergency_stop_limit: Annotated[float, readonly]
    battery_level: Annotated[float, readonly]
    current_velocity: Annotated[float, readwrite]


@blackboard_interface
class MaintenanceInterface:
    """Interface for maintenance personnel"""
    max_velocity: Annotated[float, readonly]
    emergency_stop_limit: Annotated[float, readonly]
    current_velocity: Annotated[float, readwrite]
    battery_level: Annotated[float, readwrite]
    maintenance_log: Annotated[list, readwrite]
    service_count: Annotated[int, readwrite]


robot = RobotState()

# Operator view - limited access
operator_view = create_view_from_protocol(robot, OperatorInterface)
print(f"\nOperator can see max_velocity: {operator_view.max_velocity}")
print(f"Operator can change velocity: {operator_view.current_velocity}")
operator_view.current_velocity = 0.5
print(f"✓ Operator changed velocity to: {operator_view.current_velocity}")

# Operator cannot access maintenance data
try:
    _ = operator_view.maintenance_log
    print("ERROR: Operator should not access maintenance log!")
except (AccessControlError, AttributeError) as e:
    print(f"✓ Maintenance log hidden from operator: {type(e).__name__}")

# Maintenance view - broader access
maintenance_view = create_view_from_protocol(robot, MaintenanceInterface)
maintenance_view.service_count += 1
print(f"✓ Maintenance incremented service count to: {maintenance_view.service_count}")


# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("\nThe @blackboard_interface decorator provides:")
print("  1. ✓ Declaration of readonly vs readwrite fields")
print("  2. ✓ Runtime enforcement of access constraints")
print("  3. ✓ Hiding of internal/private fields")
print("  4. ✓ Type-safe interface definitions")
print("  5. ✓ Multiple interfaces per blackboard for different access levels")
print("\nThis enables safe composition across the three DSLs:")
print("  - Rhizomorph (behavior trees)")
print("  - Hypha (Petri nets)")
print("  - Enoki (state machines)")
print()
