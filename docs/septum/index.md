# Septum FSM

Septum is a decorator-based Finite State Machine (FSM) DSL with asyncio support.

## Overview

Septum FSMs provide:

- **Decorator-based syntax** - Define states as classes with decorators
- **Enum-based transitions** - Type-safe, statically analyzable state transitions
- **Asyncio-native** - Built-in timeout support and async/await throughout
- **Hierarchical states** - Push/pop stack for nested state machines
- **Comprehensive validation** - Construction-time validation of all reachable states

## Quick Example

```python
from mycorrhizal.septum.core import septum, StateMachine
from enum import Enum, auto

@septum.state
class IdleState:
    class Events(Enum):
        START = auto()
        QUIT = auto()

    @septum.on_state
    async def on_state(ctx):
        print("Idling...")
        await asyncio.sleep(1)
        return IdleState.Events.START

    @septum.transitions
    def transitions():
        from mycorrhizal.septum.core import LabeledTransition
        return [
            LabeledTransition(IdleState.Events.START, WorkingState),
            LabeledTransition(IdleState.Events.QUIT, None),
        ]

@septum.state
class WorkingState:
    @septum.on_state
    async def on_state(ctx):
        print("Working...")
        await asyncio.sleep(2)
        return WorkingState.Events.DONE

    @septum.transitions
    def transitions():
        from mycorrhizal.septum.core import LabeledTransition
        return [
            LabeledTransition(WorkingState.Events.DONE, IdleState),
        ]

# Create and run
fsm = StateMachine(initial_state=IdleState)
await fsm.initialize()
await fsm.run()
```

## Key Concepts

### States

States are defined using the `@septum.state` decorator:

```python
@septum.state
class MyState:
    # Event enum for transitions
    class Events(Enum):
        NEXT = auto()

    @septum.on_state
    async def on_state(ctx):
        # Main state logic
        return MyState.Events.NEXT

    @septum.on_enter
    async def on_enter(ctx):
        # Called when entering state
        pass

    @septum.on_leave
    async def on_leave(ctx):
        # Called when exiting state
        pass

    @septum.on_timeout
    async def on_timeout(ctx):
        # Handle timeout if configured
        return Events.ERROR

    @septum.transitions
    def transitions():
        # Define state transitions
        pass
```

### Transitions

Transitions define how the FSM moves between states:

```python
from mycorrhizal.septum.core import LabeledTransition, Again, Unhandled, Retry, Restart, Repeat, Push, Pop

@septum.transitions
def transitions():
    return [
        LabeledTransition(Events.DONE, NextState),  # Go to NextState
        LabeledTransition(Events.RETRY, Retry),     # Re-enter with retry counter
        LabeledTransition(Events.WAIT, Unhandled), # Wait for next message
        LabeledTransition(Events.LOOP, Again),     # Re-execute immediately
        LabeledTransition(Events.RESET, Restart),  # Reset and wait for message
        LabeledTransition(Events.SUB, Push(SubState)),  # Push to stack
        LabeledTransition(Events.POP, Pop),        # Pop from stack
    ]
```

### Timeouts

States can have timeouts:

```python
from mycorrhizal.septum.core import StateConfiguration

@septum.state(config=StateConfiguration(timeout=5.0))
class TimedState:
    @septum.on_timeout
    async def on_timeout(ctx):
        print("Timeout occurred!")
        return Events.ERROR
```

### Hierarchical States

Use push/pop for nested state machines:

```python
@septum.state
class MainMenu:
    @septum.transitions
    def transitions():
        from mycorrhizal.septum.core import Push
        return [
            LabeledTransition(Events.START_GAME, Push(GamePlay, PauseMenu)),
        ]

@septum.state
class GamePlay:
    @septum.transitions
    def transitions():
        from mycorrhizal.septum.core import Push, Pop
        return [
            LabeledTransition(Events.PAUSE, Push(PauseMenu)),
            LabeledTransition(Events.QUIT, Pop),
        ]
```

## Examples

- [Basic FSM](../../examples/septum_decorator_basic.py) - Simple state machine
- [Timeout Handling](../../examples/septum_decorator_timeout.py) - Timeouts and retries
- [Blended Demo](../../examples/blended_demo.py) - FSM with other DSLs

## Documentation

- [API Reference](../api/septum.md) - Complete API documentation
- [Getting Started](../getting-started/your-first-septum.md) - Tutorial
- [PDA Guide](../guides/septum-pda-guide.md) - Hierarchical state machines with Push/Pop
- [Production Guide](production.md) - Deployment, monitoring, and performance
- [Troubleshooting](troubleshooting.md) - Common issues and debugging
- [Composition](../guides/composition.md) - Combining DSLs

## Mermaid Export

### Visualize Before You Run

A key feature of Septum is the ability to **export your state machine to a Mermaid diagram before execution**. This enables static analysis and verification of your FSM structure:

```python
from mycorrhizal.septum.util import to_mermaid

fsm = StateMachine(initial_state=MyState)
await fsm.initialize()

mermaid = to_mermaid(fsm)
print(mermaid)
```

Paste the output into the [Mermaid Live Editor](https://mermaid.live/) to visualize your state machine.

**Benefits:**
- Verify all states are reachable from the initial state
- Check for unreachable states or disconnected components
- Validate transition logic and event flows
- Document your state machine architecture automatically
- Catch structural bugs **before runtime**

### Example: Idle/Processing/Done Workflow

This state machine shows a simple three-state workflow with idle, processing, and terminal states:

```mermaid
flowchart TD
    start((start)) --> S1
    S1[IdleState]
    S1 -->|"START"| S2
    S1 -->|"QUIT"| S3
    S2[ProcessingState]
    S2 -->|"DONE"| S1
    S3[DoneState (**terminal**)]
```

**Key features shown:**
- Initial state transitions to multiple destinations
- Event-labeled transitions (START, QUIT, DONE)
- Bidirectional state flow (idle <-> processing)
- Terminal state marking workflow completion
- Clean start node visualization

See the [Septum Basic Example](../../examples/septum/septum_decorator_basic.py) for the complete executable example.

## See Also

- [Rhizomorph](../rhizomorph/) - Behavior Trees for decision-making
- [Hypha](../hypha/) - Petri Nets for workflow
