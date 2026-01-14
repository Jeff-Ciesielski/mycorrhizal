#!/usr/bin/env python3
"""
Comprehensive Rhizomorph + Spores Integration Demo

This demo showcases the full integration of:
1. Blackboard Interfaces - Type-safe, validated shared state
2. Spores OCEL Logging - Object-Centric Event Log generation
3. Relationships - Linking events to multiple objects
4. Behavior Trees - Decision-making and control flow
5. SporesAttr - Auto-detection of loggable object attributes

Scenario: A Mars rover mission where:
- Rover processes samples
- Each sample has analysis results
- Events are logged with OCEL format for process mining
- Multiple objects are related to events (rover, sample, analysis)

This generates OCEL logs that can be analyzed with process mining tools
to understand bottlenecks, optimize workflows, and audit operations.
"""

import sys
sys.path.insert(0, "src")

import asyncio
from datetime import datetime
from typing import Annotated, List
from pydantic import BaseModel
from enum import Enum

# Mycorrhizal imports
from mycorrhizal.rhizomorph.core import bt, Runner, Status
from mycorrhizal.common.timebase import MonotonicClock
from mycorrhizal.spores import (
    configure, get_spore_async,
    SporesAttr, ObjectRef, ObjectScope, EventAttr
)
from mycorrhizal.spores.transport import AsyncFileTransport


# ============================================================================
# Domain Models with SporesAttr Annotations
# ============================================================================

class RoverState(BaseModel):
    """Rover state with OCEL-logged attributes."""
    id: str
    name: Annotated[str, SporesAttr]
    battery_level: Annotated[int, SporesAttr]
    temperature: Annotated[float, SporesAttr]
    status: Annotated[str, SporesAttr]


class Sample(BaseModel):
    """Mars sample with OCEL-logged attributes."""
    id: str
    location: Annotated[str, SporesAttr]
    composition_type: Annotated[str, SporesAttr]
    mass: Annotated[float, SporesAttr]
    collection_time: Annotated[datetime, SporesAttr]


class AnalysisResult(BaseModel):
    """Sample analysis results with OCEL-logged attributes."""
    id: str
    sample_id: Annotated[str, SporesAttr]
    mineral_content: Annotated[float, SporesAttr]  # percentage
    organic_compounds: Annotated[bool, SporesAttr]
    confidence: Annotated[float, SporesAttr]
    analysis_duration: Annotated[float, SporesAttr]  # seconds


# ============================================================================
# Blackboard with Spores Annotations
# ============================================================================

class MissionBlackboard(BaseModel):
    """
    Mission blackboard with OCEL annotations.

    - Global scope: rover is included in all events
    - Event scope: current_sample and analysis only in specific events
    - EventAttr: mission_id and priority logged as event attributes
    """

    # Global objects - included in every event
    rover: Annotated[RoverState, ObjectRef(qualifier="actor", scope=ObjectScope.GLOBAL)]

    # Event-scoped objects - only when explicitly referenced (optional)
    current_sample: Annotated[Sample, ObjectRef(qualifier="target", scope=ObjectScope.EVENT)] | None = None
    current_analysis: Annotated[AnalysisResult, ObjectRef(qualifier="result", scope=ObjectScope.EVENT)] | None = None

    # Event attributes - logged as event attributes (not relationships)
    mission_id: Annotated[str, EventAttr]
    priority: Annotated[int, EventAttr]

    # Non-logged fields
    processing_queue: List[Sample] = []
    completed_analyses: List[AnalysisResult] = []


# ============================================================================
# Behavior Tree Definition with Spores Logging
# ============================================================================

@bt.tree
def SampleProcessingTree():
    """Behavior tree for processing Mars samples with full OCEL logging."""

    spore = get_spore_async(__name__)

    # ------------------------------------------------------------------------
    # Action: Check if we have samples to process
    # ------------------------------------------------------------------------
    @bt.action
    @spore.log_event(
        event_type="QueueChecked",
        relationships={
            "actor": ("bb", "RoverState"),  # Log the rover
        },
        attributes={
            "queue_size": lambda bb: len(bb.processing_queue),
            "available_capacity": lambda bb: 10 - len(bb.processing_queue),
        }
    )
    async def check_queue(bb: MissionBlackboard) -> Status:
        """Check if there are samples in the processing queue."""
        if len(bb.processing_queue) == 0:
            print(f"  [{bb.mission_id}] Queue is empty")
            return Status.FAILURE
        print(f"  [{bb.mission_id}] Queue has {len(bb.processing_queue)} samples")
        return Status.SUCCESS

    # ------------------------------------------------------------------------
    # Action: Check battery level
    # ------------------------------------------------------------------------
    @bt.action
    @spore.log_event(
        event_type="BatteryChecked",
        relationships={
            "actor": ("bb", "RoverState"),
        },
        attributes={
            "battery_level": lambda bb: bb.rover.battery_level,
            "status": lambda bb: "adequate" if bb.rover.battery_level > 20 else "low",
        }
    )
    async def check_battery(bb: MissionBlackboard) -> Status:
        """Check if rover has enough battery."""
        if bb.rover.battery_level < 20:
            print(f"  [{bb.mission_id}] Battery low: {bb.rover.battery_level}%")
            return Status.FAILURE
        print(f"  [{bb.mission_id}] Battery OK: {bb.rover.battery_level}%")
        return Status.SUCCESS

    # ------------------------------------------------------------------------
    # Action: Dequeue next sample
    # ------------------------------------------------------------------------
    @bt.action
    @spore.log_event(
        event_type="SampleDequeued",
        relationships={
            "actor": ("bb", "RoverState"),
            "sample": ("bb", "Sample"),  # Event-scoped, auto-logged
        },
        attributes={
            "queue_size_after": lambda bb: len(bb.processing_queue),
        }
    )
    async def dequeue_sample(bb: MissionBlackboard) -> Status:
        """Dequeue the next sample for processing."""
        bb.current_sample = bb.processing_queue.pop(0)
        print(f"  [{bb.mission_id}] Dequeued sample {bb.current_sample.id} from {bb.current_sample.location}")
        return Status.SUCCESS

    # ------------------------------------------------------------------------
    # Action: Analyze sample
    # ------------------------------------------------------------------------
    @bt.action
    @spore.log_event(
        event_type="SampleAnalyzed",
        relationships={
            "actor": ("bb", "RoverState"),
            "sample": ("bb", "Sample"),
            "result": ("bb", "AnalysisResult"),  # From bb.current_analysis (set after analysis)
        },
        attributes={
            "analysis_duration": lambda bb: bb.current_analysis.analysis_duration if bb.current_analysis else 0.0,
            "success_rate": lambda bb: bb.current_analysis.confidence if bb.current_analysis else 0.0,
        }
    )
    async def analyze_sample(bb: MissionBlackboard) -> Status:
        """Perform detailed analysis of the sample."""
        import random
        import time

        print(f"  [{bb.mission_id}] Analyzing sample {bb.current_sample.id}...")

        # Simulate analysis work
        duration = random.uniform(0.5, 2.0)
        await asyncio.sleep(duration)

        # Create analysis result
        analysis = AnalysisResult(
            id=f"analysis-{bb.current_sample.id}",
            sample_id=bb.current_sample.id,
            mineral_content=random.uniform(40.0, 95.0),
            organic_compounds=random.choice([True, False]),
            confidence=random.uniform(0.8, 0.99),
            analysis_duration=duration
        )

        bb.current_analysis = analysis
        bb.completed_analyses.append(analysis)

        print(f"  [{bb.mission_id}] Analysis complete: {analysis.mineral_content:.1f}% minerals, "
              f"organics={analysis.organic_compounds}, confidence={analysis.confidence:.2f}")

        return Status.SUCCESS

    # ------------------------------------------------------------------------
    # Action: Update rover state after processing
    # ------------------------------------------------------------------------
    @bt.action
    @spore.log_event(
        event_type="RoverStateUpdated",
        relationships={
            "actor": ("bb", "RoverState"),
        },
        attributes={
            "battery_consumed": lambda bb: 5,  # Fixed amount
            "temperature": lambda bb: bb.rover.temperature,
        }
    )
    async def update_rover_state(bb: MissionBlackboard) -> Status:
        """Update rover state after processing."""
        # Consume battery
        old_battery = bb.rover.battery_level
        bb.rover.battery_level = max(0, bb.rover.battery_level - 5)
        bb.rover.temperature = max(0, bb.rover.temperature + 2)

        print(f"  [{bb.mission_id}] Rover state updated: battery {old_battery}% -> {bb.rover.battery_level}%, "
              f"temp {bb.rover.temperature}°C")

        return Status.SUCCESS

    # ------------------------------------------------------------------------
    # Sequence: Complete processing workflow
    # ------------------------------------------------------------------------
    @bt.root
    @bt.sequence
    def root(N):
        """Main processing sequence."""
        yield N.check_queue
        yield N.check_battery
        yield N.dequeue_sample
        yield N.analyze_sample
        yield N.update_rover_state


# ============================================================================
# Main Demo
# ============================================================================

async def main():
    """Run the comprehensive Rhizomorph + Spores demo."""

    print("\n" + "="*80)
    print("Comprehensive Rhizomorph + Spores Integration Demo")
    print("="*80)
    print("\nScenario: Mars Rover Sample Processing with OCEL Logging")
    print("-" * 80)

    # Configure Spores for OCEL logging
    configure(
        transport=AsyncFileTransport("logs/mars_mission_ocel.jsonl"),
        enabled=True
    )

    # Create mission blackboard
    bb = MissionBlackboard(
        rover=RoverState(
            id="rover-curiosity",
            name="Curiosity",
            battery_level=100,
            temperature=20.0,
            status="operational"
        ),
        mission_id="MARS-2024-001",
        priority=1,
        processing_queue=[],
        completed_analyses=[]
    )

    # Create samples to process
    samples = [
        Sample(
            id=f"sample-{i}",
            location=f"coordinates-{100+i*10},200",
            composition_type="sedimentary" if i % 2 == 0 else "igneous",
            mass=50.0 + i * 10.0,
            collection_time=datetime.now()
        )
        for i in range(1, 6)
    ]

    bb.processing_queue = samples

    print(f"\nMission {bb.mission_id} initialized:")
    print(f"  Rover: {bb.rover.name} (battery: {bb.rover.battery_level}%, temp: {bb.rover.temperature}°C)")
    print(f"  Samples queued: {len(bb.processing_queue)}")
    print(f"  OCEL output: logs/mars_mission_ocel.jsonl")
    print("\n" + "-" * 80)
    print("Starting sample processing...")
    print("-" * 80 + "\n")

    # Create behavior tree runner
    timebase = MonotonicClock()
    runner = Runner(SampleProcessingTree, bb=bb, tb=timebase)

    # Process samples until queue is empty or battery is low
    tick_count = 0
    while len(bb.processing_queue) > 0 and bb.rover.battery_level > 20:
        print(f"\n--- Tick {tick_count + 1} ---")
        status = await runner.tick()
        print(f"Tree status: {status}")

        if status != Status.RUNNING and status != Status.SUCCESS:
            print(f"\nProcessing stopped with status: {status}")
            break

        tick_count += 1
        if tick_count > 20:  # Safety limit
            print("\nReached maximum tick count")
            break

        # Small delay between ticks
        await asyncio.sleep(0.1)

    # Wait for async logging to complete
    await asyncio.sleep(0.5)

    # Print summary
    print("\n" + "=" * 80)
    print("Mission Summary")
    print("=" * 80)
    print(f"  Samples processed: {len(bb.completed_analyses)}")
    print(f"  Remaining in queue: {len(bb.processing_queue)}")
    print(f"  Final battery level: {bb.rover.battery_level}%")
    print(f"  Final temperature: {bb.rover.temperature}°C")
    print(f"\n  OCEL Events logged to: logs/mars_mission_ocel.jsonl")
    print(f"  Each event includes:")
    print(f"    - Event attributes (mission_id, priority, computed metrics)")
    print(f"    - Relationships to objects (rover, sample, analysis)")
    print(f"    - Object attributes (SporesAttr-marked fields)")
    print("\n" + "=" * 80)
    print("\nYou can analyze the OCEL log with process mining tools to:")
    print("  - Identify bottlenecks in the workflow")
    print("  - Track object lifecycles (rover, samples, analyses)")
    print("  - Audit all operations with full provenance")
    print("  - Optimize resource allocation (battery management)")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
