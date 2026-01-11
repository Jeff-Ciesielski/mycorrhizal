#!/usr/bin/env python3
"""
Rhizomorph Behavior Tree with Spores Logging Example

Demonstrates how to use the Spores RhizomorphAdapter to log events from a behavior tree.
Shows node execution logging with status results and blackboard attributes.
"""

import sys
import asyncio
sys.path.insert(0, "src")

from pydantic import BaseModel
from typing import Annotated, List
from enum import Enum

from mycorrhizal.rhizomorph.core import bt, Runner as BTRunner, Status
from mycorrhizal.common.timebase import WallClock
from mycorrhizal.spores import configure, spore, EventAttr, ObjectRef, ObjectScope
from mycorrhizal.spores.dsl import RhizomorphAdapter
from mycorrhizal.spores.transport import Transport


# ============================================================================
# Data Models
# ============================================================================

class ThreatType(Enum):
    """Types of threats."""
    UNKNOWN = "unknown"
    MISSILE = "missile"
    AIRCRAFT = "aircraft"
    GROUND = "ground"


class Threat(BaseModel):
    """A detected threat."""
    id: str
    type: ThreatType
    distance: float
    speed: float
    bearing: float
    hostility: int = 50  # 0-100 scale


class Sensor(BaseModel):
    """A sensor platform."""
    id: str
    name: str
    status: str = "active"
    range_km: float = 100.0


class Blackboard(BaseModel):
    """Blackboard for threat assessment system."""
    mission_id: Annotated[str, EventAttr]
    alert_level: Annotated[str, EventAttr]
    sensor: Annotated[Sensor, ObjectRef(qualifier="sensor", scope=ObjectScope.GLOBAL)]
    current_threat: Annotated[Threat, ObjectRef(qualifier="target", scope=ObjectScope.EVENT)]


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
# Behavior Tree Definition
# ============================================================================

def create_threat_assessment_tree(adapter: RhizomorphAdapter):
    """Create a behavior tree for threat assessment with spores logging."""

    @bt.tree
    def ThreatAssessmentTree():
        # Condition: Check if threat is hostile
        @bt.condition
        @adapter.log_node(event_type="check_hostility", log_status=True)
        def is_hostile(bb: Blackboard) -> bool:
            """
            Check if threat is hostile.

            Spores logs:
            - node_name: "is_hostile"
            - status: SUCCESS or FAILURE
            - mission_id, alert_level from bb
            """
            return bb.current_threat.hostility >= 70

        # Condition: Check if threat is in range
        @bt.condition
        @adapter.log_node(event_type="check_range", log_status=True)
        def in_range(bb: Blackboard) -> bool:
            """
            Check if threat is in sensor range.

            Spores logs:
            - node_name: "in_range"
            - status: SUCCESS or FAILURE
            - Sensor object (global scope)
            - Threat object (event scope)
            """
            return bb.current_threat.distance <= bb.sensor.range_km

        # Action: Classify threat type
        @bt.action
        @adapter.log_node(
            event_type="classify_threat",
            log_status=False,  # Don't include status in attributes
        )
        async def classify(bb: Blackboard) -> Status:
            """
            Classify the threat type.

            Spores logs:
            - node_name: "classify"
            - Threat object with updated classification
            """
            threat = bb.current_threat

            if threat.speed > 800 and threat.distance > 50:
                threat.type = ThreatType.MISSILE
            elif threat.speed > 200:
                threat.type = ThreatType.AIRCRAFT
            else:
                threat.type = ThreatType.GROUND

            print(f"  Classified as: {threat.type.value}")
            return Status.SUCCESS

        # Action: Issue alert
        @bt.action
        @adapter.log_node(event_type="issue_alert")
        async def issue_alert(bb: Blackboard) -> Status:
            """
            Issue high-priority alert.

            Spores logs:
            - node_name: "issue_alert"
            - alert_level attribute
            """
            bb.alert_level = "HIGH"
            print(f"  ALERT ISSUED: {bb.current_threat.id}")
            return Status.SUCCESS

        # Action: Log threat
        @bt.action
        @adapter.log_node(event_type="log_threat")
        async def log_threat(bb: Blackboard) -> Status:
            """
            Log threat for monitoring.

            Spores logs:
            - node_name: "log_threat"
            """
            print(f"  Threat logged: {bb.current_threat.id} (hostility: {bb.current_threat.hostility})")
            return Status.SUCCESS

        # Root sequence
        @bt.root
        @bt.sequence(memory=False)
        def root():
            """Main threat assessment sequence."""
            yield is_hostile
            yield in_range
            yield classify

            # Selector for response actions
            @bt.selector(memory=False)
            def response():
                """Choose response based on classification."""
                yield issue_alert
                yield log_threat

            yield response

    return ThreatAssessmentTree


# ============================================================================
# Main Execution
# ============================================================================

async def main():
    """Run the threat assessment behavior tree with spores logging."""

    # Configure spores
    print("=" * 60)
    print("Rhizomorph Behavior Tree with Spores Logging")
    print("=" * 60)
    print()

    transport = ConsoleTransport()
    configure(
        enabled=True,
        object_cache_size=10,
        transport=transport,
    )

    # Mark object types
    @spore.object(object_type="Threat")
    class _Threat(Threat):
        pass

    @spore.object(object_type="Sensor")
    class _Sensor(Sensor):
        pass

    # Create adapter
    adapter = RhizomorphAdapter()

    # Create blackboard
    sensor = Sensor(id="sensor-1", name="Radar-01", range_km=150.0)

    bb = Blackboard(
        mission_id="MISSION-ALPHA",
        alert_level="NORMAL",
        sensor=sensor,
        current_threat=Threat(
            id="THREAT-001",
            type=ThreatType.UNKNOWN,
            distance=80.0,
            speed=850.0,
            bearing=45.0,
            hostility=85,
        ),
    )

    # Create behavior tree
    tree_func = create_threat_assessment_tree(adapter)

    # Create and start runner
    from mycorrhizal.rhizomorph.core import Runner
    runner = Runner(
        tree_func,
        bb=bb,
    )

    print("Starting behavior tree...")
    print()
    print(f"Mission ID: {bb.mission_id}")
    print(f"Alert Level: {bb.alert_level}")
    print(f"Sensor: {sensor.name} (range: {sensor.range_km} km)")
    print(f"Current Threat: {bb.current_threat.id}")
    print(f"  - Type: {bb.current_threat.type.value}")
    print(f"  - Distance: {bb.current_threat.distance} km")
    print(f"  - Speed: {bb.current_threat.speed} m/s")
    print(f"  - Hostility: {bb.current_threat.hostility}%")
    print()
    print("-" * 60)
    print()

    # Run the behavior tree
    tb = WallClock()
    status = await runner.tick_until_complete()

    print()
    print("-" * 60)
    print()
    print(f"Tree completed with status: {status.name}")
    print(f"Final alert level: {bb.alert_level}")
    print(f"Classified threat type: {bb.current_threat.type.value}")
    transport.close()
    print()


if __name__ == "__main__":
    asyncio.run(main())
