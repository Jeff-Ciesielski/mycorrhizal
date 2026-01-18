#!/usr/bin/env python3
"""
Septum State Machine with Spores Logging Example

Demonstrates how to use the Spores SeptumAdapter to log events from a state machine.
Shows state execution, lifecycle events, and transition logging.
"""

import sys
import asyncio
sys.path.insert(0, "src")

from pydantic import BaseModel
from typing import Annotated, Optional
from enum import Enum, auto

from mycorrhizal.septum.core import (
    septum, StateMachine, StateConfiguration,
    SharedContext, Again, Unhandled, LabeledTransition,
)
from mycorrhizal.common.timebase import WallClock
from mycorrhizal.spores import configure, spore, EventAttr, ObjectRef, ObjectScope
from mycorrhizal.spores.dsl import SeptumAdapter
from mycorrhizal.spores.transport import Transport


# ============================================================================
# Data Models
# ============================================================================

class TrafficLightState(Enum):
    """Traffic light states."""
    RED = auto()
    YELLOW = auto()
    GREEN = auto()


class Vehicle(BaseModel):
    """A vehicle at the intersection."""
    id: str
    type: str
    speed: float = 0.0


class Intersection(BaseModel):
    """Traffic intersection state."""
    intersection_id: Annotated[str, EventAttr]
    cycle_count: Annotated[int, EventAttr]
    total_vehicles: Annotated[int, EventAttr]
    current_vehicle: Annotated[Optional[Vehicle], ObjectRef(qualifier="vehicle", scope=ObjectScope.EVENT)]


# ============================================================================
# Mock Transport for Testing
# ============================================================================

class ConsoleTransport(Transport):
    """Transport that prints to console for demonstration."""

    def __init__(self):
        self.count = 0

    async def send(self, data: bytes, content_type: str) -> None:
        """Print log records to console."""
        import json
        self.count += 1
        record = json.loads(data.decode('utf-8'))

        if "event" in record:
            evt = record["event"]
            print(f"[EVENT #{self.count}] {evt['type']}")
            print(f"  ID: {evt['id']}")
            print(f"  Time: {evt['time']}")
            print(f"  Attributes:")
            for attr in evt.get("attributes", []):
                print(f"    - {attr['name']}: {attr['value']}")
            if evt.get("relationships"):
                print(f"  Relationships: {len(evt['relationships'])} object(s)")
            print()

        if "object" in record:
            obj = record["object"]
            print(f"[OBJECT #{self.count}] {obj['type']}")
            print(f"  ID: {obj['id']}")
            print(f"  Attributes:")
            for attr in obj.get("attributes", []):
                print(f"    - {attr['name']}: {attr['value']}")
            print()

    def is_async(self) -> bool:
        return False

    def close(self) -> None:
        print(f"\nTotal records sent: {self.count}")


# ============================================================================
# State Machine Definition
# ============================================================================

def create_traffic_light_fsm(adapter: SeptumAdapter):
    """Create a traffic light state machine with spores logging."""

    @septum.state(config=StateConfiguration(can_dwell=True))
    def RedState():
        """Red light state - stop all traffic."""
        class Events(Enum):
            ADVANCE = auto()
            EMERGENCY = auto()

        @septum.on_state
        @adapter.log_state(
            event_type="red_light_active",
            log_state_name=True,
            log_transition=True,
        )
        async def on_state(ctx: SharedContext):
            """
            Red light is active.

            Spores logs:
            - state_name: "RedState"
            - transition: result of this state
            - intersection_id, cycle_count from ctx.common
            """
            # Only print when we get the advance signal
            if ctx.msg == "advance":
                print("RED - Stopping traffic")
                return Events.ADVANCE

            # Check for emergency vehicles
            if ctx.msg and isinstance(ctx.msg, dict) and ctx.msg.get("emergency"):
                print("  → Emergency vehicle detected!")
                return Events.EMERGENCY

            return None

        @septum.on_enter
        @adapter.log_state_lifecycle(event_type="state_enter")
        async def on_enter(ctx: SharedContext):
            """Called when entering red state."""
            print("  → Entering RED state")

        @septum.transitions
        def transitions():
            return [
                LabeledTransition(Events.ADVANCE, GreenState),
                LabeledTransition(Events.EMERGENCY, GreenState),
            ]

    @septum.state(config=StateConfiguration(can_dwell=True))
    def YellowState():
        """Yellow light state - prepare to stop."""

        class Events(Enum):
            ADVANCE = auto()

        @septum.on_state
        @adapter.log_state(event_type="yellow_light_active")
        async def on_state(ctx: SharedContext):
            """
            Yellow light is active.

            Spores logs:
            - state_name: "YellowState"
            - cycle_count from ctx.common
            """
            # Only print when we get the advance signal
            if ctx.msg == "advance":
                print("YELLOW - Prepare to stop")
                return Events.ADVANCE

            return None

        @septum.on_enter
        @adapter.log_state_lifecycle(event_type="state_enter")
        async def on_enter(ctx: SharedContext):
            """Called when entering yellow state."""
            print("  → Entering YELLOW state")

        @septum.transitions
        def transitions():
            return [
                LabeledTransition(Events.ADVANCE, RedState),
            ]

    @septum.state(config=StateConfiguration(can_dwell=True))
    def GreenState():
        """Green light state - allow traffic flow."""

        class Events(Enum):
            ADVANCE = auto()
            FORCE_RED = auto()

        @septum.on_state
        @adapter.log_state(event_type="green_light_active")
        async def on_state(ctx: SharedContext):
            """
            Green light is active.

            Spores logs:
            - state_name: "GreenState"
            - Vehicle object if present in context
            """
            # Check for force red message
            if ctx.msg and isinstance(ctx.msg, dict) and ctx.msg.get("force_red"):
                print("  → Force red requested!")
                return Events.FORCE_RED

            # Only print when we get the advance signal
            if ctx.msg == "advance":
                print("GREEN - Traffic flowing")
                return Events.ADVANCE

            return None

        @septum.on_enter
        @adapter.log_state_lifecycle(event_type="state_enter")
        async def on_enter(ctx: SharedContext):
            """Called when entering green state."""
            print("  → Entering GREEN state")

        @septum.transitions
        def transitions():
            return [
                LabeledTransition(Events.ADVANCE, YellowState),
                LabeledTransition(Events.FORCE_RED, RedState),
            ]

    @septum.state()
    def ErrorState():
        """Error state - something went wrong."""

        @septum.on_state
        @adapter.log_state(event_type="error_state")
        async def on_state(ctx: SharedContext):
            """Error state handler."""
            print("ERROR - Malfunction!")
            return Unhandled()

    return RedState


# ============================================================================
# Main Execution
# ============================================================================

async def main():
    """Run the traffic light state machine with spores logging."""

    # Configure spores
    print("=" * 60)
    print("Septum State Machine with Spores Logging")
    print("=" * 60)
    print()

    transport = ConsoleTransport()
    configure(
        enabled=True,
        object_cache_size=10,
        transport=transport,
    )

    # Mark object types
    @spore.object(object_type="Vehicle")
    class _Vehicle(Vehicle):
        pass

    # Create adapter
    adapter = SeptumAdapter()

    # Create intersection state
    intersection = Intersection(
        intersection_id="INT-MAIN-ST-5TH",
        cycle_count=0,
        total_vehicles=0,
        current_vehicle=None,
    )

    # Create state machine
    initial_state = create_traffic_light_fsm(adapter)
    fsm = StateMachine(
        initial_state=initial_state,
        error_state=None,
        common_data=intersection,
    )

    # Initialize FSM
    await fsm.initialize()

    # Suppress FSM debug output for cleaner example output
    fsm.log = lambda msg: None

    print("Starting traffic light state machine...")
    print()
    print(f"Intersection: {intersection.intersection_id}")
    print(f"Initial state: {initial_state.base_name}")
    print()
    print("-" * 60)
    print()

    # Run through a few cycles
    cycles = 3

    for cycle in range(cycles):
        print(f"\n--- Cycle {cycle + 1} ---")

        # Increment cycle count
        intersection.cycle_count += 1

        # Run the state machine for this cycle
        for i in range(3):  # Run through 3 states
            print(f"  Tick {i+1}")
            fsm.send_message("advance")
            await fsm.tick(timeout=1.0)
            print(f"    Current state: {fsm.current_state.base_name if fsm.current_state else 'None'}")

    print()
    print("-" * 60)
    print()
    print(f"Completed {cycles} cycles")
    print(f"Final cycle count: {intersection.cycle_count}")
    transport.close()
    print()


if __name__ == "__main__":
    asyncio.run(main())
