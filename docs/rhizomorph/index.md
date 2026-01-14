# Rhizomorph Behavior Trees

Rhizomorph is an async-first Behavior Tree DSL for decision-making and control logic.

## Overview

Rhizomorph Behavior Trees provide:

- **Decorator-based syntax** - Define trees, nodes, and composites with decorators
- **Async-first design** - Native asyncio support throughout
- **Type-safe references** - Owner-aware composition with `N.member` syntax
- **Modular subtrees** - Reusable tree components with `bt.subtree()` and `bt.bind()`
- **Rich composites** - Sequence, selector, parallel, and more

## Quick Example

```python
from mycorrhizal.rhizomorph.core import bt, Runner as BTRunner, Status
from pydantic import BaseModel

class RobotContext(BaseModel):
    battery_level: int = 100
    has_task: bool = True

@bt.tree
class RobotAI:
    @bt.condition
    def has_battery(bb: RobotContext) -> bool:
        return bb.battery_level > 20

    @bt.action
    async def recharge(bb: RobotContext) -> Status:
        print("Recharging...")
        bb.battery_level = 100
        return Status.SUCCESS

    @bt.action
    async def do_task(bb: RobotContext) -> Status:
        if not bb.has_task:
            return Status.FAILURE
        print("Doing task!")
        bb.battery_level -= 10
        return Status.SUCCESS

    @bt.action
    async def idle(bb: RobotContext) -> Status:
        print("Idling...")
        return Status.SUCCESS

    @bt.root
    @bt.selector
    def root(N):
        """Try battery check, fall through to tasks, then idle."""
        yield N.has_battery
        yield N.do_task
        yield N.idle

    @bt.sequence
    def charging_sequence(N):
        """Battery must be low AND recharge succeeds."""
        yield N.has_battery  # Actually: check if LOW battery (negated)
        yield N.recharge
```

## Key Concepts

### Nodes

Behavior trees have three types of leaf nodes:

#### Actions

Perform operations and return status:

```python
@bt.action
async def my_action(bb: Blackboard) -> Status:
    # Do work
    result = do_something(bb)
    return Status.SUCCESS if result else Status.FAILURE
```

#### Conditions

Return boolean or status:

```python
@bt.condition
def my_condition(bb: Blackboard) -> bool:
    return bb.some_value > 10

# OR return Status directly
@bt.condition
def my_condition(bb: Blackboard) -> Status:
    return Status.SUCCESS if bb.ready else Status.FAILURE
```

#### Decorators

Modify node behavior:

```python
@bt.decorator
async def retry_decorator(child, bb, executor):
    """Retry child node up to 3 times."""
    for attempt in range(3):
        status = await executor(child)
        if status == Status.SUCCESS:
            return Status.SUCCESS
    return Status.FAILURE
```

### Composites

Composites control flow among child nodes:

#### Sequence

Execute children in order, fail fast:

```python
@bt.sequence
def my_sequence(N):
    """All children must succeed."""
    yield N.step_1  # Runs first
    yield N.step_2  # Runs only if step_1 succeeds
    yield N.step_3  # Runs only if step_2 succeeds
```

#### Selector

Execute children in order, succeed fast:

```python
@bt.selector
def my_selector(N):
    """Try each until one succeeds."""
    yield N.option_a  # Runs first
    yield N.option_b  # Runs only if option_a fails
    yield N.option_c  # Runs only if option_b fails
```

#### Parallel

Execute all children simultaneously:

```python
@bt.parallel
def my_parallel(N):
    """All children run concurrently."""
    yield N.task_a  # All run in parallel
    yield N.task_b
    yield N.task_c
```

### Root Node

Every tree must have a root:

```python
@bt.root
@bt.sequence
def root(N):
    """Entry point for the tree."""
    yield N.initialize
    yield N.process
    yield N.cleanup
```

## Status Values

Nodes return one of three statuses:

- **SUCCESS** - Node completed successfully
- **FAILURE** - Node failed (expected failure)
- **ERROR** - Unexpected error (aborts tree execution)

## Subtrees

Create reusable tree components:

```python
@bt.tree
class NavigationSubtree:
    """Reusable navigation behavior."""
    @bt.action
    async def move_to_target(bb):
        print(f"Moving to {bb.target}")
        return Status.SUCCESS

    @bt.action
    async def avoid_obstacles(bb):
        print("Avoiding obstacles")
        return Status.SUCCESS

    @bt.root
    @bt.sequence
    def root(N):
        yield N.avoid_obstacles
        yield N.move_to_target

# Use in main tree
@bt.tree
class MainRobotAI:
    @bt.root
    @bt.sequence
    def root(N):
        nav = bt.subtree("navigation")  # Subtree placeholder
        yield N.check_battery
        yield nav.move_to_target  # Reference subtree node
```

Bind subtrees at runtime:

```python
main_tree = MainRobotAI()
nav_tree = NavigationSubtree()

bound_tree = bt.bind(main_tree, {
    "navigation": nav_tree
})

runner = BTRunner(tree=bound_tree, blackboard=bb)
```

## Blackboard Integration

Access shared state:

```python
from pydantic import BaseModel
from typing import Annotated

class GameContext(BaseModel):
    player_health: int = 100
    has_key: bool = False
    enemies_visible: int = 0

@bt.tree
class GameAI:
    @bt.condition
    def is_low_health(bb: GameContext) -> bool:
        return bb.player_health < 30

    @bt.action
    async def heal(bb: GameContext) -> Status:
        bb.player_health = min(100, bb.player_health + 20)
        return Status.SUCCESS
```

## Running Trees

Create a runner and tick the tree:

```python
# Create runner
bb = RobotContext()
tree = RobotAI()
runner = BTRunner(tree=tree, blackboard=bb)

# Tick the tree (one evaluation)
await runner.tick()

# Run multiple ticks
for _ in range(10):
    await runner.tick()
    await asyncio.sleep(0.1)
```

## Examples

- [Basic Example](../../examples/rhizomorph_example.py) - Simple behavior tree
- [Blended Demo](../../examples/blended_demo.py) - Behavior tree + Petri net

## Documentation

- [API Reference](../api/rhizomorph.md) - Complete API documentation
- [Getting Started](../getting-started/your-first-rhizomorph.md) - Tutorial
- [Composition](../guides/composition.md) - Subtree patterns

## Mermaid Export

### Visualize Before You Run

Rhizomorph behavior trees support **Mermaid diagram export for static verification**:

```python
tree = MyTree()
mermaid = tree.to_mermaid()
print(mermaid)
```

View in [Mermaid Live Editor](https://mermaid.live/) to visualize your decision logic.

**Benefits:**
- Verify all nodes are reachable from the root
- Understand the control flow (sequences, selectors, parallels)
- Identify potential logic errors in tree structure
- Validate composite node behavior
- Document AI/decision logic automatically

**Review and validate your decision trees before ticking them!**

## See Also

- [Enoki](../enoki/) - State Machines for stateful behavior
- [Hypha](../hypha/) - Petri Nets for workflow orchestration
