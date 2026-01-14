# Your First Enoki FSM

This tutorial will walk you through creating a simple Finite State Machine (FSM) using Enoki.

## What You'll Build

You'll create a simple traffic light controller with three states: Red, Yellow, and Green.

## Step 1: Import Enoki

```python
from mycorrhizal.enoki.core import enoki, StateMachine
from enum import Enum, auto
```

## Step 2: Define Your States

```python
@enoki.state
class RedState:
    class Events(Enum):
        TIMER = auto()

    @enoki.on_state
    async def on_state(ctx):
        print("🔴 Red light - Stop!")
        await asyncio.sleep(2)  # Wait 2 seconds
        return RedState.Events.TIMER

    @enoki.transitions
    def transitions():
        from mycorrhizal.enoki.core import LabeledTransition
        return [
            LabeledTransition(RedState.Events.TIMER, GreenState),
        ]
```

```python
@enoki.state
class GreenState:
    class Events(Enum):
        TIMER = auto()

    @enoki.on_state
    async def on_state(ctx):
        print("🟢 Green light - Go!")
        await asyncio.sleep(2)
        return GreenState.Events.TIMER

    @enoki.transitions
    def transitions():
        from mycorrhizal.enoki.core import LabeledTransition
        return [
            LabeledTransition(GreenState.Events.TIMER, YellowState),
        ]
```

```python
@enoki.state
class YellowState:
    class Events(Enum):
        TIMER = auto()

    @enoki.on_state
    async def on_state(ctx):
        print("🟡 Yellow light - Caution!")
        await asyncio.sleep(1)
        return YellowState.Events.TIMER

    @enoki.transitions
    def transitions():
        from mycorrhizal.enoki.core import LabeledTransition
        return [
            LabeledTransition(YellowState.Events.TIMER, RedState),
        ]
```

## Step 3: Visualize Your FSM (Before Running!)

One of Enoki's powerful features is the ability to **visualize your state machine before execution**. This helps you verify the structure and catch errors early:

```python
# Create the FSM
fsm = StateMachine(initial_state=RedState)
await fsm.initialize()

# Export to Mermaid diagram
mermaid = fsm.generate_mermaid_flowchart()
print(mermaid)
```

This outputs a Mermaid diagram you can view in the [Mermaid Live Editor](https://mermaid.live/):

```mermaid
%%{init: {'layout':'elk'}}%%
flowchart TD
    start((start)) --> S1
    S1[RedState]
    S1 -->|"TIMER"| S3
    S3[GreenState]
    S3 -->|"TIMER"| S2
    S2[YellowState]
    S2 -->|"TIMER"| S1
```

You can verify:

- All states are reachable from the initial state
- Transitions form a valid cycle
- No states are orphaned or disconnected

This static analysis happens **before you run the code**, so you can catch structural issues immediately!

## Step 4: Create and Run the FSM

```python
import asyncio

async def main():
    fsm = StateMachine(initial_state=RedState)
    await fsm.initialize()

    # Run for 3 cycles
    for _ in range(3):
        await fsm.run()

asyncio.run(main())
```

## Expected Output

```
🔴 Red light - Stop!
🟢 Green light - Go!
🟡 Yellow light - Caution!
🔴 Red light - Stop!
...
```

## Full Example

See the full example in the repository:
```bash
python examples/enoki_decorator_basic.py
```

## Next Steps

- Learn about [transitions](../../api/enoki.md) in the API reference
- Understand [timebases](../guides/timebases.md) for timing control
- Explore [state hierarchies](../guides/composition.md) with push/pop
