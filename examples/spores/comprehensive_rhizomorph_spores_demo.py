#!/usr/bin/env python3
"""
Mars Rover Sample Processing with Behavior Trees and OCEL Logging

Demonstrates:
- Blackboard interfaces with automatic view creation
- Spores OCEL event logging with object relationships
- Behavior tree decision-making
"""

import sys
sys.path.insert(0, "src")

import asyncio
from datetime import datetime
from typing import Annotated
from pydantic import BaseModel

from mycorrhizal.rhizomorph.core import bt, Runner, Status
from mycorrhizal.common.timebase import MonotonicClock
from mycorrhizal.spores import configure, get_spore_async, SporesAttr, ObjectRef, ObjectScope, EventAttr
from mycorrhizal.spores.transport import AsyncFileTransport
from mycorrhizal.common.interface_builder import blackboard_interface, readonly, readwrite


# ============================================================================
# Domain Models
# ============================================================================

class RoverState(BaseModel):
    id: str
    name: Annotated[str, SporesAttr]
    battery_level: Annotated[int, SporesAttr]
    temperature: Annotated[float, SporesAttr]
    status: Annotated[str, SporesAttr]


class Sample(BaseModel):
    id: str
    location: Annotated[str, SporesAttr]
    composition_type: Annotated[str, SporesAttr]
    mass: Annotated[float, SporesAttr]
    collection_time: Annotated[datetime, SporesAttr]


class AnalysisResult(BaseModel):
    id: str
    sample_id: Annotated[str, SporesAttr]
    mineral_content: Annotated[float, SporesAttr]
    organic_compounds: Annotated[bool, SporesAttr]
    confidence: Annotated[float, SporesAttr]
    analysis_duration: Annotated[float, SporesAttr]


# ============================================================================
# Blackboard
# ============================================================================

class MissionBlackboard(BaseModel):
    rover: Annotated[RoverState, ObjectRef(qualifier="actor", scope=ObjectScope.GLOBAL)]
    current_sample: Annotated[Sample, ObjectRef(qualifier="target", scope=ObjectScope.EVENT)] | None = None
    current_analysis: Annotated[AnalysisResult, ObjectRef(qualifier="result", scope=ObjectScope.EVENT)] | None = None
    mission_id: Annotated[str, EventAttr]
    priority: Annotated[int, EventAttr]
    processing_queue: list[Sample] = []
    completed_analyses: list[AnalysisResult] = []


# ============================================================================
# Interfaces
# ============================================================================

@blackboard_interface
class ReadOnlyConfig:
    mission_id: Annotated[str, readonly]
    priority: Annotated[int, readonly]


@blackboard_interface
class RoverStatus:
    rover: Annotated[RoverState, readonly]


@blackboard_interface
class AnalysisAccess:
    processing_queue: Annotated[list[Sample], readwrite]
    current_sample: Annotated[Sample | None, readwrite]
    current_analysis: Annotated[AnalysisResult | None, readwrite]
    completed_analyses: list[AnalysisResult]
    rover: Annotated[RoverState, readonly]
    mission_id: Annotated[str, readonly]


@blackboard_interface
class RoverControl:
    rover: Annotated[RoverState, readwrite]


# ============================================================================
# Behavior Tree
# ============================================================================

@bt.tree
def SampleProcessingTree():
    """Behavior tree for processing Mars samples with interface-based access control."""

    spore = get_spore_async(__name__)

    @bt.condition
    def has_samples(bb: AnalysisAccess) -> bool:
        # Only processing_queue and related fields are accessible through this interface
        return len(bb.processing_queue) > 0

    @bt.condition
    def has_battery(bb: RoverStatus) -> bool:
        # Only rover state is readable through this interface
        return bb.rover.battery_level > 20

    @bt.action
    @spore.log_event(
        event_type="SampleAnalyzed",
        relationships={
            "actor": ("bb", "RoverState"),
            "sample": ("bb", "Sample"),
            "result": ("bb", "AnalysisResult"),
        },
        attributes={
            "analysis_duration": lambda bb: bb.current_analysis.analysis_duration if bb.current_analysis else 0.0,
        }
    )
    async def analyze_sample(bb: AnalysisAccess) -> Status:
        # This function only sees fields defined in AnalysisAccess
        # Cannot access bb.priority or modify bb.rover (readonly)
        bb.current_sample = bb.processing_queue.pop(0)

        print(f"  Analyzing {bb.current_sample.id} from {bb.current_sample.location}")
        print(f"  Mission: {bb.mission_id} (readonly)")
        print(f"  Rover battery: {bb.rover.battery_level}% (readonly)")

        # Simulate analysis
        await asyncio.sleep(0.3)

        analysis = AnalysisResult(
            id=f"analysis-{bb.current_sample.id}",
            sample_id=bb.current_sample.id,
            mineral_content=70.0,
            organic_compounds=True,
            confidence=0.95,
            analysis_duration=1.0
        )
        bb.current_analysis = analysis
        bb.completed_analyses.append(analysis)

        print(f"  Analysis complete: {analysis.mineral_content}% minerals")
        return Status.SUCCESS

    @bt.action
    @spore.log_event(
        event_type="RoverBatteryConsumed",
        relationships={
            "actor": ("bb", "RoverState"),
        },
        attributes={
            "battery_consumed": lambda bb: 5,
        }
    )
    async def consume_battery(bb: RoverControl) -> Status:
        # This interface has write access to rover but cannot see mission_id or samples
        bb.rover.battery_level = max(0, bb.rover.battery_level - 5)
        bb.rover.temperature += 2

        print(f"  Rover battery: {bb.rover.battery_level}%")
        return Status.SUCCESS

    @bt.root
    @bt.sequence
    def root(N):
        yield N.has_samples
        yield N.has_battery
        yield N.analyze_sample
        yield N.consume_battery


# ============================================================================
# Main
# ============================================================================

async def main():
    # Configure Spores for OCEL logging
    configure(
        transport=AsyncFileTransport("logs/rover_sample_processing.jsonl"),
        enabled=True
    )

    # Create mission blackboard
    bb = MissionBlackboard(
        rover=RoverState(
            id="rover-1",
            name="Curiosity",
            battery_level=100,
            temperature=20.0,
            status="operational"
        ),
        mission_id="MARS-2024-001",
        priority=1,
        processing_queue=[
            Sample(
                id=f"sample-{i}",
                location=f"coordinates-{100+i*10},200",
                composition_type="sedimentary",
                mass=50.0 + i * 10.0,
                collection_time=datetime.now()
            )
            for i in range(1, 4)
        ]
    )

    print(f"Mission {bb.mission_id}")
    print(f"Rover: {bb.rover.name} (battery: {bb.rover.battery_level}%)")
    print(f"Samples in queue: {len(bb.processing_queue)}")
    print(f"OCEL output: logs/rover_sample_processing.jsonl\n")

    # Create and run behavior tree
    timebase = MonotonicClock()
    runner = Runner(SampleProcessingTree, bb=bb, tb=timebase)

    # Process samples
    tick_count = 0
    while len(bb.processing_queue) > 0 and bb.rover.battery_level > 20 and tick_count < 10:
        print(f"\n--- Tick {tick_count + 1} ---")
        status = await runner.tick()
        tick_count += 1

        if status not in (Status.RUNNING, Status.SUCCESS):
            break

    await asyncio.sleep(0.5)

    print(f"\n{'='*60}")
    print(f"Mission complete")
    print(f"  Samples processed: {len(bb.completed_analyses)}")
    print(f"  Remaining in queue: {len(bb.processing_queue)}")
    print(f"  Final battery: {bb.rover.battery_level}%")


if __name__ == "__main__":
    asyncio.run(main())
