# Your First Mycelium Tree

This tutorial guides you through building your first Mycelium tree with FSM-in-BT integration. You'll create a simple task processor where a behavior tree controls an FSM that manages processing states.

## What You'll Build

A task processing system with:

- An FSM with two states: `Idle` and `Processing`
- A behavior tree that decides when to start/stop processing
- The BT controls FSM state transitions
- A single Mermaid diagram showing the integrated system

This demonstrates the core Mycelium pattern: BT provides the decision-making logic, FSM provides state management.

## Prerequisites

- Python 3.10 or higher
- Mycorrhizal installed: `pip install mycorrhizal`
- Basic familiarity with async/await in Python
- (Optional) Familiarity with behavior trees and FSMs

## Step 1: Define Your FSM

First, define the FSM states using Mycelium's mirrored Septum API. Import from `mycorrhizal.mycelium` for a unified experience:

```python
from mycorrhizal.mycelium import state, events, on_state, transitions, LabeledTransition
from enum import Enum, auto

@state()
def IdleState():
    """Waiting for tasks to process."""

    @events
    class Events(Enum):
        START = auto()  # Transition to Processing

    @on_state
    async def on_state(ctx, common):
        # Idle state logic
        print("  FSM: Idling, waiting for work...")
        return None  # Stay in idle until BT sends START event

    @transitions
    def transitions():
        return [
            LabeledTransition(IdleState.Events.START, ProcessingState),
        ]

@state()
def ProcessingState():
    """Processing a task."""

    @events
    class Events(Enum):
        PAUSE = auto()  # Return to Idle
        DONE = auto()   # Task complete

    @on_state
    async def on_state(ctx, common):
        # Processing state logic
        print("  FSM: Processing task...")
        common.tasks_processed += 1
        return None  # Wait for BT to send PAUSE or DONE

    @transitions
    def transitions():
        return [
            LabeledTransition(ProcessingState.Events.PAUSE, IdleState),
            LabeledTransition(ProcessingState.Events.DONE, IdleState),
        ]
```

Key points:

- `@state()` decorator defines FSM states
- `@events` class defines transition events
- `@on_state` handler runs when state is active
- `@transitions` defines state graph

## Step 2: Create Mycelium Tree with FSM Integration

Now create a behavior tree that uses the FSM. The `@Action(fsm=...)` parameter integrates the FSM:

```python
from mycorrhizal.mycelium import tree, Action, Sequence, root

@tree
def TaskProcessor():
    """BT that controls the FSM."""

    @Action(fsm=IdleState)
    async def check_for_work(bb, tb, fsm_runner):
        """
        FSM auto-ticks before this action runs.
        fsm_runner provides access to FSM state and messaging.
        """
        # Check current FSM state
        state_name = fsm_runner.current_state.name
        print(f"BT: FSM is in {state_name}")

        # Decide if we should start processing
        if bb.tasks_processed >= bb.max_tasks:
            print("BT: All tasks complete!")
            return Status.FAILURE  # Signal BT to stop

        if bb.has_work:
            print("BT: Starting work...")
            fsm_runner.send_message(IdleState.Events.START)
            bb.has_work = False  # Consumed the work

        return Status.SUCCESS

    @Action(fsm=ProcessingState)
    async def monitor_processing(bb, tb, fsm_runner):
        """Monitor the processing state."""
        state_name = fsm_runner.current_state.name
        print(f"BT: FSM is in {state_name}")

        # After processing, return to idle
        fsm_runner.send_message(ProcessingState.Events.PAUSE)
        return Status.SUCCESS

    @root
    @Sequence
    def control_loop(N):
        """Main BT sequence."""
        yield N.check_for_work
        yield N.monitor_processing
```

Key points:

- `@Action(fsm=IdleState)` integrates FSM into action
- FSM auto-ticks before action runs
- `fsm_runner` parameter provides FSM access
- BT controls when FSM transitions via `send_message()`

## Step 3: Define Blackboard

Define the shared state (blackboard) that both BT and FSM access:

```python
from pydantic import BaseModel

class TaskBlackboard(BaseModel):
    """Shared state for task processing."""
    tasks_processed: int = 0
    max_tasks: int = 3
    has_work: bool = True
```

## Step 4: Run and Visualize

Create a runner and execute the tree:

```python
import asyncio
from mycorrhizal.mycelium import TreeRunner

async def main():
    # Initialize blackboard
    bb = TaskBlackboard()

    # Create tree runner
    runner = TreeRunner(TaskProcessor, bb=bb)

    print("=" * 60)
    print("Task Processor - Mycelium FSM-in-BT Demo")
    print("=" * 60)
    print()

    # Run the BT for several ticks
    for tick in range(6):
        print(f"Tick {tick + 1}:")
        result = await runner.tick()
        await asyncio.sleep(0.1)  # Small delay between ticks
        print()

        # Stop if BT completes
        if result != Status.RUNNING:
            break

    print("=" * 60)
    print("Final Statistics:")
    print(f"  Tasks processed: {bb.tasks_processed}")
    print("=" * 60)
    print()

    # Generate and print Mermaid diagram
    print("System Visualization:")
    print("-" * 60)
    print(runner.tree.to_mermaid())

if __name__ == "__main__":
    asyncio.run(main())
```

## Step 5: See It Run

Put it all together and run:

```bash
python your_first_mycelium.py
```

Expected output:

```
============================================================
Task Processor - Mycelium FSM-in-BT Demo
============================================================

Tick 1:
BT: FSM is in IdleState
  FSM: Idling, waiting for work...
BT: Starting work...
BT: FSM is in ProcessingState
  FSM: Processing task...

Tick 2:
BT: FSM is in IdleState
  FSM: Idling, waiting for work...
BT: FSM is in ProcessingState
  FSM: Processing task...

Tick 3:
BT: FSM is in IdleState
  FSM: Idling, waiting for work...
BT: All tasks complete!
============================================================
Final Statistics:
  Tasks processed: 3
============================================================

System Visualization:
------------------------------------------------------------
graph TD
    Root[root] --> S1[Sequence]
    S1 --> A1[check_for_work<br/>FSM: IdleState]
    S1 --> A2[monitor_processing<br/>FSM: ProcessingState]
```

The Mermaid diagram shows the BT structure with FSM states labeled directly in the action nodes - a single, integrated view of both systems.

## What's Happening?

Here's the execution flow:

1. **Tick 1:**
   - BT runs `check_for_work` action
   - FSM auto-ticks, entering `IdleState`
   - BT sees FSM is idle, sends `START` event
   - FSM transitions to `ProcessingState`
   - BT runs `monitor_processing`, sends `PAUSE`
   - FSM returns to `IdleState`

2. **Tick 2-3:**
   - Cycle repeats for remaining tasks

3. **Tick 4:**
   - All tasks processed
   - BT returns `FAILURE` to signal completion

The integration is seamless:

- **FSM auto-ticks** before each BT action runs
- **BT controls** when FSM transitions occur
- **Single diagram** shows both systems together

## Full Example

Here's the complete, runnable script:

```python
#!/usr/bin/env python3
"""Your First Mycelium Tree - FSM-in-BT Tutorial"""

import asyncio
from enum import Enum, auto
from pydantic import BaseModel

from mycorrhizal.mycelium import (
    tree, Action, Sequence, root,
    state, events, on_state, transitions, LabeledTransition,
    TreeRunner,
)
from mycorrhizal.rhizomorph.core import Status

# ==================================================================================
# FSM States
# ==================================================================================

@state()
def IdleState():
    """Waiting for tasks to process."""

    @events
    class Events(Enum):
        START = auto()

    @on_state
    async def on_state(ctx, bb):
        print("  FSM: Idling, waiting for work...")
        return None

    @transitions
    def transitions():
        return [
            LabeledTransition(IdleState.Events.START, ProcessingState),
        ]

@state()
def ProcessingState():
    """Processing a task."""

    @events
    class Events(Enum):
        PAUSE = auto()
        DONE = auto()

    @on_state
    async def on_state(ctx, bb):
        print("  FSM: Processing task...")
        bb.tasks_processed += 1
        return None

    @transitions
    def transitions():
        return [
            LabeledTransition(ProcessingState.Events.PAUSE, IdleState),
            LabeledTransition(ProcessingState.Events.DONE, IdleState),
        ]

# ==================================================================================
# Behavior Tree
# ==================================================================================

@tree
def TaskProcessor():
    """BT that controls the FSM."""

    @Action(fsm=IdleState)
    async def check_for_work(bb, tb, fsm_runner):
        """Check if we should start processing."""
        state_name = fsm_runner.current_state.name
        print(f"BT: FSM is in {state_name}")

        if bb.tasks_processed >= bb.max_tasks:
            print("BT: All tasks complete!")
            return Status.FAILURE

        if bb.has_work:
            print("BT: Starting work...")
            fsm_runner.send_message(IdleState.Events.START)
            bb.has_work = False

        return Status.SUCCESS

    @Action(fsm=ProcessingState)
    async def monitor_processing(bb, tb, fsm_runner):
        """Monitor the processing state."""
        state_name = fsm_runner.current_state.name
        print(f"BT: FSM is in {state_name}")
        fsm_runner.send_message(ProcessingState.Events.PAUSE)
        return Status.SUCCESS

    @root
    @Sequence
    def control_loop(N):
        yield N.check_for_work
        yield N.monitor_processing

# ==================================================================================
# Blackboard
# ==================================================================================

class TaskBlackboard(BaseModel):
    """Shared state for task processing."""
    tasks_processed: int = 0
    max_tasks: int = 3
    has_work: bool = True

# ==================================================================================
# Main
# ==================================================================================

async def main():
    bb = TaskBlackboard()
    runner = TreeRunner(TaskProcessor, bb=bb)

    print("=" * 60)
    print("Task Processor - Mycelium FSM-in-BT Demo")
    print("=" * 60)
    print()

    for tick in range(6):
        print(f"Tick {tick + 1}:")
        result = await runner.tick()
        await asyncio.sleep(0.1)
        print()

        if result != Status.RUNNING:
            break

    print("=" * 60)
    print(f"Tasks processed: {bb.tasks_processed}")
    print("=" * 60)
    print()
    print(runner.tree.to_mermaid())

if __name__ == "__main__":
    asyncio.run(main())
```

## Next Steps

Now that you've built your first Mycelium tree:

- **Deep dive:** Learn the [FSM-in-BT pattern](../mycelium/fsm-in-bt.md) in detail
- **Other patterns:** Explore [BT-in-FSM](../mycelium/bt-in-fsm.md) and [BT-in-PN](../mycelium/bt-in-pn.md)
- **Advanced:** Read about [complex nesting strategies](../mycelium/advanced-patterns.md)
- **Visualizations:** See more examples of [seamless diagrams](../mycelium/visualizations.md)
