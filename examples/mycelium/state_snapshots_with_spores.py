#!/usr/bin/env python3
"""
State Snapshots with Spores Logging

This example demonstrates:
1. Taking periodic state snapshots of Mycelium trees
2. Logging tree-level events
3. Snapshotting blackboard state
4. Snapshotting FSM states
5. Analyzing system evolution over time

The scenario: A circuit breaker that transitions between states
based on failure rates. We take snapshots at key points to analyze
the system's behavior.
"""

import sys
sys.path.insert(0, "src")

import asyncio
import tempfile
from enum import Enum, auto
from pathlib import Path
from pydantic import BaseModel
from typing import Annotated

from mycorrhizal.septum.core import septum, LabeledTransition
from mycorrhizal.mycelium import (
    tree,
    Action,
    Sequence,
    root,
    TreeRunner,
    TreeSporesAdapter,
    log_tree_event,
)
from mycorrhizal.rhizomorph.core import Status
from mycorrhizal.common.timebase import MonotonicClock
from mycorrhizal.spores import configure
from mycorrhizal.spores.models import EventAttr
from mycorrhizal.spores.transport import AsyncFileTransport


# ======================================================================================
# Blackboard with Spores Annotations
# ======================================================================================


class CircuitBreakerState(BaseModel):
    """
    Circuit breaker state with annotated fields for logging.
    """
    # Event attributes (auto-extracted)
    state: Annotated[str, EventAttr]
    failure_count: Annotated[int, EventAttr]
    success_count: Annotated[int, EventAttr]
    failure_threshold: Annotated[int, EventAttr]
    last_failure_time: Annotated[str, EventAttr]


# ======================================================================================
# FSM States for Circuit Breaker
# ======================================================================================


@septum.state()
def ClosedState():
    """
    Closed state: Requests pass through normally.
    Transition to Open if too many failures.
    """

    @septum.events
    class Events(Enum):
        SUCCESS = auto()
        FAILURE = auto()
        THRESHOLD_EXCEEDED = auto()

    @septum.on_state
    async def on_state(ctx):
        failure_count = ctx.common.get("failure_count", 0)
        threshold = ctx.common.get("failure_threshold", 3)

        if failure_count >= threshold:
            return Events.THRESHOLD_EXCEEDED

        # Simulate request (80% success rate)
        import random
        if random.random() < 0.8:
            ctx.common["success_count"] = ctx.common.get("success_count", 0) + 1
            return Events.SUCCESS
        else:
            ctx.common["failure_count"] = failure_count + 1
            from datetime import datetime
            ctx.common["last_failure_time"] = datetime.now().isoformat()
            return Events.FAILURE

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.SUCCESS, ClosedState),
            LabeledTransition(Events.FAILURE, ClosedState),
            LabeledTransition(Events.THRESHOLD_EXCEEDED, OpenState),
        ]


@septum.state()
def OpenState():
    """
    Open state: Requests are blocked.
    Wait for timeout, then transition to HalfOpen.
    """

    @septum.events
    class Events(Enum):
        TIMEOUT = auto()

    @septum.on_state
    async def on_state(ctx):
        # Simulate timeout (after 1 tick, go to half-open)
        return Events.TIMEOUT

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.TIMEOUT, HalfOpenState),
        ]


@septum.state()
def HalfOpenState():
    """
    Half-open state: Allow one request through.
    If success, go to Closed. If failure, go back to Open.
    """

    @septum.events
    class Events(Enum):
        SUCCESS = auto()
        FAILURE = auto()

    @septum.on_state
    async def on_state(ctx):
        # Simulate test request (50% success rate in half-open)
        import random
        if random.random() < 0.5:
            # Success: reset failure count and go to closed
            ctx.common["failure_count"] = 0
            ctx.common["success_count"] = ctx.common.get("success_count", 0) + 1
            return Events.SUCCESS
        else:
            # Failure: back to open
            ctx.common["failure_count"] = ctx.common.get("failure_threshold", 3)
            from datetime import datetime
            ctx.common["last_failure_time"] = datetime.now().isoformat()
            return Events.FAILURE

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.SUCCESS, ClosedState),
            LabeledTransition(Events.FAILURE, OpenState),
        ]


# ======================================================================================
# Mycelium Tree with Spores Logging
# ======================================================================================


# Create spores adapter
mycelium_adapter = TreeSporesAdapter(tree_name="CircuitBreaker")


@tree
def CircuitBreakerTree():
    """
    Circuit breaker with state snapshot logging.

    Takes snapshots at key points to track system evolution.
    """

    @Action(fsm=ClosedState)
    @mycelium_adapter.log_action(
        event_type="circuit_breaker_tick",
        log_fsm_state=True,
        log_status=True,
        log_blackboard=True,
    )
    async def monitor_circuit(bb, tb, fsm_runner):
        """
        Monitor circuit breaker state.

        Each tick logs:
        - FSM state (Closed, Open, HalfOpen)
        - Failure/success counts
        - Blackboard snapshot
        """
        state_name = fsm_runner.current_state.name if fsm_runner.current_state else "Unknown"

        # Sync FSM state to blackboard
        if "Closed" in state_name:
            bb.state = "Closed"
        elif "Open" in state_name:
            bb.state = "Open"
        elif "HalfOpen" in state_name:
            bb.state = "HalfOpen"

        # Sync counters
        bb.failure_count = fsm_runner.fsm.context.common.get("failure_count", 0)
        bb.success_count = fsm_runner.fsm.context.common.get("success_count", 0)
        bb.failure_threshold = fsm_runner.fsm.context.common.get("failure_threshold", 3)
        bb.last_failure_time = fsm_runner.fsm.context.common.get("last_failure_time", "N/A")

        print(f"  State: {bb.state} | Failures: {bb.failure_count}/{bb.failure_threshold} | Successes: {bb.success_count}")

        # Keep running
        return Status.RUNNING

    @root
    @Sequence
    def main():
        """Main behavior tree."""
        yield monitor_circuit


# ======================================================================================
# Custom Wrapper with Tree-Level Logging
# ======================================================================================


class LoggedTreeRunner(TreeRunner):
    """
    Tree runner with automatic snapshot logging.

    Takes snapshots every N ticks and on state transitions.
    """

    def __init__(self, *args, snapshot_interval=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.snapshot_interval = snapshot_interval
        self.tick_count = 0
        self.last_state = None

    async def tick(self):
        """Tick the tree and take periodic snapshots."""
        result = await super().tick()

        self.tick_count += 1

        # Check for state transition
        current_state = self.bb.state
        state_changed = (self.last_state != current_state)
        self.last_state = current_state

        # Take snapshot on state change or at interval
        if state_changed or (self.tick_count % self.snapshot_interval == 0):
            snapshot_type = "state_change" if state_changed else "periodic"
            await self._take_snapshot(snapshot_type)

        return result

    async def _take_snapshot(self, snapshot_type):
        """Take a state snapshot."""
        from mycorrhizal.mycelium.spores_integration import _log_state_snapshot

        await _log_state_snapshot(
            self.instance,
            f"CircuitBreaker_{snapshot_type}",
            include_blackboard=True,
            include_fsms=True,
        )

        print(f"  [Snapshot] {snapshot_type} snapshot taken at tick {self.tick_count}")


# ======================================================================================
# Tree-Level Event Logging
# ======================================================================================


@log_tree_event(
    event_type="circuit_breaker_started",
    tree_name="CircuitBreaker",
    attributes={
        "description": "Circuit breaker monitoring started",
        "failure_threshold": lambda runner: runner.bb.failure_threshold if hasattr(runner, 'bb') else 3,
    }
)
async def run_circuit_breaker(runner: LoggedTreeRunner):
    """Run the circuit breaker with tree-level logging."""
    await runner.run(max_ticks=20)


# ======================================================================================
# Main Execution
# ======================================================================================


async def main():
    """Run the circuit breaker with state snapshots."""

    print("=" * 70)
    print("Mycelium + Spores: State Snapshots and Tree-Level Events")
    print("=" * 70)
    print()
    print("This demo shows:")
    print("  - State snapshots at key points")
    print("  - Tree-level event logging")
    print("  - FSM state transition tracking")
    print("  - Blackboard evolution over time")
    print("  - Object-type logs (snapshots)")
    print()

    # Create temporary file for spores output
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        log_file = f.name

    print(f"Logging to: {log_file}")
    print()

    # Configure spores
    configure(transport=AsyncFileTransport(log_file))

    # Create blackboard
    bb = CircuitBreakerState(
        state="Closed",
        failure_count=0,
        success_count=0,
        failure_threshold=3,
        last_failure_time="N/A",
    )
    tb = MonotonicClock()

    # Create tree runner with snapshot logging
    runner = LoggedTreeRunner(
        CircuitBreakerTree,
        bb=bb,
        tb=tb,
        snapshot_interval=5,  # Snapshot every 5 ticks
    )

    print("Initial state:")
    print(f"  State: {bb.state}")
    print(f"  Failure threshold: {bb.failure_threshold}")
    print()

    # Run the circuit breaker
    print("Running circuit breaker...")
    print("-" * 70)

    await run_circuit_breaker(runner)

    print("-" * 70)
    print()
    print("Final state:")
    print(f"  Total ticks: {runner.tick_count}")
    print(f"  State: {bb.state}")
    print(f"  Failures: {bb.failure_count}")
    print(f"  Successes: {bb.success_count}")
    print()

    # Show log file contents
    print("=" * 70)
    print("Spores Event Log (OCEL format)")
    print("=" * 70)
    print()

    with open(log_file, 'r') as f:
        lines = f.readlines()
        event_count = sum(1 for line in lines if '"event"' in line)
        object_count = sum(1 for line in lines if '"object"' in line)
        snapshot_count = sum(1 for line in lines if '"type": "StateSnapshot' in line)

        print(f"Total records: {len(lines)}")
        print(f"  Events: {event_count}")
        print(f"  Objects: {object_count}")
        print(f"  Snapshots: {snapshot_count}")
        print()

        # Show first few records
        print("First 5 records:")
        for i, line in enumerate(lines[:5], 1):
            print(f"Record {i}: {line.rstrip()}")

        if len(lines) > 5:
            print(f"... ({len(lines) - 5} more records)")

    print()
    print("=" * 70)
    print("What was logged:")
    print("  - Events:")
    print("    - circuit_breaker_started (tree-level)")
    print("    - circuit_breaker_tick (per action)")
    print("  - Objects (State Snapshots):")
    print("    - CircuitBreaker_state_change (on FSM transition)")
    print("    - CircuitBreaker_periodic (every N ticks)")
    print("  - Event attributes:")
    print("    - FSM state, failure/success counts")
    print("    - Blackboard snapshot in state snapshots")
    print()

    # Clean up
    Path(log_file).unlink()


if __name__ == "__main__":
    asyncio.run(main())
