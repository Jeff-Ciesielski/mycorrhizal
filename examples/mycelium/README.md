# Mycelium Examples

This directory contains examples demonstrating the Mycelium unified orchestration layer for Mycorrhizal.

## Mirrored Enoki API

Mycelium provides a mirrored Enoki API that allows you to import all FSM-related decorators from `mycorrhizal.mycelium` instead of `mycorrhizal.enoki.core`. This provides a unified import experience when working with FSM-BT integration.

**Imports from Mycelium:**
```python
from mycorrhizal.mycelium import (
    # BT decorators
    tree, Action, Condition, Sequence, Selector, Parallel, root,
    # Mirrored Enoki API
    state, events, on_state, transitions,
    # Enoki types
    LabeledTransition, StateConfiguration,
    # BT runners
    TreeRunner, TreeInstance,
)
from mycorrhizal.rhizomorph.core import bt, Status
```

## Unified API Pattern

Mycorrhizal provides a unified API for integrating Enoki FSMs and Rhizomorph behavior trees in both directions:

### FSM-in-BT: Actions with Integrated FSMs

```python
from mycorrhizal.mycelium import tree, Action, Sequence, root, state, events, transitions, LabeledTransition
from mycorrhizal.rhizomorph.core import bt, Status
from enum import Enum, auto

# Define FSM states using mirrored Enoki API
@state()
def MyState():
    @events
    class Events(Enum):
        START = auto()
        DONE = auto()

    @on_state
    async def on_state(ctx):
        return Events.START

    @transitions
    def transitions():
        return [LabeledTransition(Events.START, MyState)]

# Define BT tree with FSM-integrated action
@tree
def MyTree():
    @Action(fsm=MyState)
    async def control_fsm(bb, tb, fsm_runner):
        # FSM auto-ticks each time this action runs
        state_name = fsm_runner.current_state.name
        fsm_runner.send_message(MyState.Events.START)
        return Status.SUCCESS

    @root
    @Sequence
    def main():
        yield control_fsm
```

### BT-in-FSM: States with Integrated BTs

```python
from mycorrhizal.mycelium import state, events, on_state as mycelium_on_state, transitions as mycelium_transitions, LabeledTransition, StateConfiguration
from mycorrhizal.rhizomorph.core import bt, Status

# Define BT subtree
@bt.tree
def DecideNext():
    @bt.action
    async def check(bb):
        return Status.SUCCESS if bb.ready else Status.FAILURE

    @bt.root
    @Sequence
    def main():
        yield check

# Define FSM state with BT integration
@state(bt=DecideNext, config=StateConfiguration(can_dwell=True))
def MyState():
    @events
    class Events(Enum):
        DONE = auto()

    @mycelium_on_state
    async def on_state(ctx, bt_result):
        # BT auto-runs, bt_result is Status
        if bt_result == Status.SUCCESS:
            return Events.DONE
        return None

    @mycelium_transitions
    def transitions():
        return []
```

**Note:** The mirrored `on_state` decorator is imported as `mycelium_on_state` to avoid naming conflicts with the `on_state` function name inside the state definition.

## Examples

### `robot_controller.py` - FSM-in-BT Integration

A robot controller that demonstrates:
- FSM-integrated actions with `@Action(fsm=IdleState)`
- Automatic FSM ticking before action execution
- Complex state management (idle/processing/charging)
- Battery management and automatic recharging
- Mermaid diagrams showing BT+FSM connections

**Key features:**
- Robot processes tasks using 20% battery each
- Automatically goes to charging when battery <= 20%
- Charges to 100% before resuming work
- FSM handles all state transitions automatically

Run it with:
```bash
uv run python examples/mycelium/robot_controller.py
```

### `ci_cd_orchestrator.py` - BT-in-FSM Integration

A CI/CD deployment orchestrator that demonstrates:
- FSM managing deployment workflow (Build → Test → Deploy → Monitor)
- BT subtrees for intelligent error recovery decisions
- `@state(bt=...)` for BT integration in FSM states
- Context-aware decision making based on environment and error type

**Key features:**
- FSM drives high-level deployment pipeline
- BTs decide recovery strategies:
  - **Build failures**: Clear cache, reinstall dependencies, retry, or alert
  - **Test failures**: Detect flaky tests, fix environment, ignore in dev, or mark for review
  - **Deployment failures**: Extend health timeout, scale resources, rollback, or alert on-call
  - **Monitoring issues**: Continue monitoring, rollback, or alert humans
- Each state uses a different BT strategy tailored to its needs

Run it with:
```bash
uv run python examples/mycelium/ci_cd_orchestrator.py
```

## Key Benefits

1. **Mirrored Enoki API** - Import all FSM decorators from `mycorrhizal.mycelium`
2. **Bidirectional Integration** - FSM-in-BT and BT-in-FSM patterns both supported
3. **Explicit Declaration** - Integration declared where it's used via decorator parameters
4. **No Boilerplate** - No manual registry access or tree-level declarations
5. **Type-Safe** - FSM runner provides typed access to current state and messaging
6. **Unified Visualization** - Single Mermaid diagram shows entire system
7. **Backward Compatible** - Vanilla BT actions and FSM states still work

## When to Use Each Pattern

### Use FSM-in-BT (`@Action(fsm=...)`) when:
- You have a BT driving high-level behavior
- An action needs persistent state (mode, configuration, etc.)
- The FSM represents a subsystem or component (robot, vehicle, character)
- You want the BT to orchestrate multiple FSMs

### Use BT-in-FSM (`@state(bt=...)`) when:
- An FSM state needs complex decision-making logic
- You want to use BT's compositional pattern for evaluating options
- The decision logic should be reusable across multiple states
- The FSM drives the high-level flow, but delegates decisions to BTs

## Naming Conventions

When using the mirrored Enoki API with BT-in-FSM integration, you'll need to import decorators with aliases to avoid naming conflicts:

```python
from mycorrhizal.mycelium import (
    state, events,
    on_state as mycelium_on_state,      # Alias to avoid conflict
    transitions as mycelium_transitions,  # Alias to avoid conflict
    LabeledTransition, StateConfiguration,
)
```

This is necessary because:
1. The `@state` decorator immediately executes the state function to extract its contents
2. Inside the state function, you define an `async def on_state(...)` function
3. Python treats `on_state` as a local variable throughout the function scope
4. Using `@mycelium_on_state` avoids the name collision
