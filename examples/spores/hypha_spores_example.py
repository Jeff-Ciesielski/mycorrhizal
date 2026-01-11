#!/usr/bin/env python3
"""
Hypha Petri Net with Spores Logging Example

Demonstrates how to use the Spores HyphaAdapter to log events from a Petri net.
Shows transition logging with token relationships and blackboard attributes.
"""

import sys
import asyncio
sys.path.insert(0, "src")

from pydantic import BaseModel
from typing import Annotated

from mycorrhizal.hypha.core import pn, PlaceType, Runner as PNRunner
from mycorrhizal.common.timebase import MonotonicClock
from mycorrhizal.spores import configure, spore, EventAttr, ObjectRef, ObjectScope
from mycorrhizal.spores.dsl import HyphaAdapter
from mycorrhizal.spores.transport import Transport


# ============================================================================
# Data Models
# ============================================================================

class WorkItem(BaseModel):
    """A work item in the processing system."""
    data: str


class Robot(BaseModel):
    """A robot that performs work."""
    id: str
    name: str
    status: str = "active"


class ProcessingBlackboard(BaseModel):
    """Shared state for the processing system."""
    batch_id: Annotated[str, EventAttr]
    operator: Annotated[str, EventAttr]
    robot: Annotated[Robot, ObjectRef(qualifier="actor", scope=ObjectScope.GLOBAL)]


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
# Petri Net Definition
# ============================================================================

@pn.net
def ProcessingNet(builder):
    """Simple processing net with spores logging."""
    import mycorrhizal.spores.dsl.hypha
    adapter = mycorrhizal.spores.dsl.hypha.HyphaAdapter()

    # Places
    processed = builder.place("processed", type=PlaceType.QUEUE)

    # IO input source
    @builder.io_input_place()
    async def source(bb, timebase):
        """Generate work items."""
        items = ["ITEM-001", "ITEM-002", "ITEM-003"]
        for item_id in items:
            await timebase.sleep(0.2)
            token = WorkItem(data=item_id)
            print(f"[Source] Produced: {item_id}")
            yield token

    # Transition: Process item
    @builder.transition()
    @adapter.log_transition(
        event_type="process_item",
        log_inputs=True,
    )
    async def process(consumed, bb, timebase):
        """
        Process a work item.

        Spores automatically logs:
        - token_count: number of consumed tokens
        - transition_name: "process"
        - Relationships to consumed tokens
        - Attributes from bb (batch_id, operator)
        - Robot object (global scope)
        """
        for token in consumed:
            print(f"[Process] Processing {token.data}")
            await timebase.sleep(0.1)
            yield {processed: token}

    # Wiring: source → process → processed
    builder.arc(source, process).arc(processed)


# ============================================================================
# Main Execution
# ============================================================================

async def main():
    """Run the processing Petri net with spores logging."""

    # Configure spores
    print("=" * 60)
    print("Hypha Petri Net with Spores Logging")
    print("=" * 60)
    print()

    transport = ConsoleTransport()
    configure(
        enabled=True,
        object_cache_size=10,
        transport=transport,
    )

    # Mark object types
    @spore.object(object_type="WorkItem")
    class _WorkItem(WorkItem):
        pass

    @spore.object(object_type="Robot")
    class _Robot(Robot):
        pass

    # Create blackboard
    robot = Robot(id="robot-1", name="Processor-01")

    bb = ProcessingBlackboard(
        batch_id="BATCH-2024-001",
        operator="Alice",
        robot=robot,
    )

    # Create and start runner
    runner = PNRunner(ProcessingNet, bb)
    timebase = MonotonicClock()

    print("Starting Petri net...")
    print()
    print(f"Batch ID: {bb.batch_id}")
    print(f"Operator: {bb.operator}")
    print(f"Robot: {robot.name}")
    print()
    print("-" * 60)
    print()

    await runner.start(timebase)

    # Wait for processing to complete
    await asyncio.sleep(3)

    await runner.stop()
    transport.close()

    print("-" * 60)
    print("Petri net stopped.")
    print()


if __name__ == "__main__":
    asyncio.run(main())
