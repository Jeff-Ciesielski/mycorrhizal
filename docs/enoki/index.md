# Enoki FSM

Enoki is a decorator-based Finite State Machine (FSM) DSL with asyncio support.

## Overview

Enoki FSMs provide:

- **Decorator-based syntax** - Define states as classes with decorators
- **Enum-based transitions** - Type-safe, statically analyzable state transitions
- **Asyncio-native** - Built-in timeout support and async/await throughout
- **Hierarchical states** - Push/pop stack for nested state machines
- **Comprehensive validation** - Construction-time validation of all reachable states

## Quick Example

```python
from mycorrhizal.enoki.core import enoki, StateMachine
from enum import Enum, auto

@enoki.state
class IdleState:
    class Events(Enum):
        START = auto()
        QUIT = auto()

    @enoki.on_state
    async def on_state(ctx):
        print("Idling...")
        await asyncio.sleep(1)
        return IdleState.Events.START

    @enoki.transitions
    def transitions():
        from mycorrhizal.enoki.core import LabeledTransition
        return [
            LabeledTransition(IdleState.Events.START, WorkingState),
            LabeledTransition(IdleState.Events.QUIT, None),
        ]

@enoki.state
class WorkingState:
    @enoki.on_state
    async def on_state(ctx):
        print("Working...")
        await asyncio.sleep(2)
        return WorkingState.Events.DONE

    @enoki.transitions
    def transitions():
        from mycorrhizal.enoki.core import LabeledTransition
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

States are defined using the `@enoki.state` decorator:

```python
@enoki.state
class MyState:
    # Event enum for transitions
    class Events(Enum):
        NEXT = auto()

    @enoki.on_state
    async def on_state(ctx):
        # Main state logic
        return MyState.Events.NEXT

    @enoki.on_enter
    async def on_enter(ctx):
        # Called when entering state
        pass

    @enoki.on_exit
    async def on_exit(ctx):
        # Called when exiting state
        pass

    @enoki.on_timeout
    async def on_timeout(ctx):
        # Handle timeout if configured
        return Events.ERROR

    @enoki.transitions
    def transitions():
        # Define state transitions
        pass
```

### Transitions

Transitions define how the FSM moves between states:

```python
from mycorrhizal.enoki.core import LabeledTransition, Again, Unhandled, Retry, Restart, Repeat, Push, Pop

@enoki.transitions
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
from mycorrhizal.enoki.core import StateConfiguration

@enoki.state(config=StateConfiguration(timeout=5.0))
class TimedState:
    @enoki.on_timeout
    async def on_timeout(ctx):
        print("Timeout occurred!")
        return Events.ERROR
```

### Hierarchical States

Use push/pop for nested state machines:

```python
@enoki.state
class MainMenu:
    @enoki.transitions
    def transitions():
        from mycorrhizal.enoki.core import Push
        return [
            LabeledTransition(Events.START_GAME, Push(GamePlay, PauseMenu)),
        ]

@enoki.state
class GamePlay:
    @enoki.transitions
    def transitions():
        from mycorrhizal.enoki.core import Push, Pop
        return [
            LabeledTransition(Events.PAUSE, Push(PauseMenu)),
            LabeledTransition(Events.QUIT, Pop),
        ]
```

## Examples

- [Basic FSM](../../examples/enoki_decorator_basic.py) - Simple state machine
- [Timeout Handling](../../examples/enoki_decorator_timeout.py) - Timeouts and retries
- [Blended Demo](../../examples/blended_demo.py) - FSM with other DSLs

## Documentation

- [API Reference](../api/enoki.md) - Complete API documentation
- [Getting Started](../getting-started/your-first-enoki.md) - Tutorial
- [Composition](../guides/composition.md) - Hierarchical state patterns

## Mermaid Export

### Visualize Before You Run

A key feature of Enoki is the ability to **export your state machine to a Mermaid diagram before execution**. This enables static analysis and verification of your FSM structure:

```python
fsm = StateMachine(initial_state=MyState)
await fsm.initialize()

mermaid = fsm.generate_mermaid_flowchart()
print(mermaid)
```

Paste the output into the [Mermaid Live Editor](https://mermaid.live/) to visualize your state machine.

**Benefits:**
- Verify all states are reachable from the initial state
- Check for unreachable states or disconnected components
- Validate transition logic and event flows
- Document your state machine architecture automatically
- Catch structural bugs **before runtime**

## See Also

- [Rhizomorph](../rhizomorph/) - Behavior Trees for decision-making
- [Hypha](../hypha/) - Petri Nets for workflow
