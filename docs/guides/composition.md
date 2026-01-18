# Composition Patterns

Learn how to combine DSLs and build modular, reusable systems with Mycorrhizal.

## Combining Multiple DSLs

Mycorrhizal's four DSLs (Septum, Hypha, Rhizomorph, Spores) are designed to work together.

### Hypha + Rhizomorph Integration

A common pattern is using a Petri net to manage workflow and behavior trees for decision making:

```python
from mycorrhizal.hypha.core import pn, Runner as PNRunner
from mycorrhizal.rhizomorph.core import bt, Runner as BTRunner, Status
from pydantic import BaseModel

class SharedContext(BaseModel):
    current_task: str = ""
    task_queue: list = []

# Petri net generates tasks
@pn.net
class TaskScheduler:
    @pn.place(type=pn.PlaceType.QUEUE)
    def pending_tasks(bb):
        return []

    @pn.transition()
    async def generate_task(consumed, bb, timebase):
        new_task = f"task-{len(bb.task_queue)}"
        bb.task_queue.append(new_task)
        print(f"Generated: {new_task}")
        return [pn.Token(value=new_task)]

# Behavior tree processes tasks
@bt.tree
class TaskProcessor:
    @bt.condition
    def has_task(bb):
        return len(bb.task_queue) > 0

    @bt.action
    async def process_task(bb):
        task = bb.task_queue.pop(0)
        bb.current_task = task
        print(f"Processing: {task}")
        return Status.SUCCESS

    @bt.root
    @bt.sequence
    def root():
        yield has_task
        yield process_task
```

### Spores Observability

Add logging to any DSL:

```python
from mycorrhizal.spores import spore
from mycorrhizal.spores.models import EventAttr

class MissionContext(BaseModel):
    mission_id: Annotated[str, EventAttr]

@bt.tree
@spore.object(object_type="MissionControl")
class LoggedBehaviorTree:
    @bt.action
    async def execute_mission(bb: MissionContext) -> Status:
        print(f"Executing mission {bb.mission_id}")
        return Status.SUCCESS
```

## Modular Design with Subtrees

### Defining Subtrees

Create reusable behavior tree components:

```python
@bt.tree
class NavigationSubtree:
    """Reusable navigation behavior."""
    @bt.action
    async def move_to_target(bb):
        print(f"Moving to {bb.target_location}")
        return Status.SUCCESS

    @bt.action
    async def avoid_obstacles(bb):
        print("Avoiding obstacles")
        return Status.SUCCESS

    @bt.root
    @bt.sequence
    def root():
        yield avoid_obstacles
        yield move_to_target
```

### Binding Subtrees

Use `bt.subtree()` and `bt.bind()` to compose trees:

```python
@bt.tree
class MainRobotAI:
    @bt.action
    async def patrol(bb):
        print("Patrolling")
        return Status.SUCCESS

    @bt.root
    @bt.sequence
    def root():
        # Subtree placeholder
        nav = bt.subtree("navigation")
        yield patrol
        yield nav.move_to_target

# Bind the subtree
ai_tree = MainRobotAI()
nav_tree = NavigationSubtree()

bound_tree = bt.bind(ai_tree, {
    "navigation": nav_tree
})
```

## Modular Design with Subnets

### Defining Subnets

Create reusable Petri net components:

```python
@pn.net
class DataValidationNet:
    """Reusable validation subnet."""
    @pn.place(type=pn.PlaceType.QUEUE)
    def input_data(bb):
        return []

    @pn.place(type=pn.PlaceType.QUEUE)
    def validated_data(bb):
        return []

    @pn.transition()
    async def validate(consumed, bb, timebase):
        data = consumed[0].value
        # Validate data
        if is_valid(data):
            return [pn.Token(value=data)]
        return []
```

### Composing Nets

```python
@pn.net
class ProcessingPipeline:
    @pn.place(type=pn.PlaceType.QUEUE)
    def raw_input(bb):
        return []

    @pn.place(type=pn.PlaceType.QUEUE)
    def final_output(bb):
        return []

    # Use subnet as a component
    validator = DataValidationNet()

    @pn.transition()
    async def process(consumed, bb, timebase):
        data = consumed[0].value
        # Process validated data
        result = transform(data)
        return [pn.Token(value=result)]
```

## State Machine Hierarchies

### Push/Pop for Sub-states

Use `Push` and `Pop` for hierarchical state machines:

```python
from mycorrhizal.septum.core import Push, Pop

@septum.state
class MainMenu:
    @septum.on_state
    async def on_state(ctx):
        print("Main Menu")
        return MainMenu.Events.START_GAME

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(
                MainMenu.Events.START_GAME,
                Push(GamePlay, PauseMenu)  # Push game state, pause is above it
            ),
        ]

@septum.state
class GamePlay:
    @septum.on_state
    async def on_state(ctx):
        print("Playing...")
        await asyncio.sleep(1)
        return GamePlay.Events.PAUSE

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(GamePlay.Events.PAUSE, Push(PauseMenu)),
        ]

@septum.state
class PauseMenu:
    @septum.on_state
    async def on_state(ctx):
        print("Paused")

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(PauseMenu.Events.RESUME, Pop),  # Return to GamePlay
            LabeledTransition(PauseMenu.Events.QUIT, Pop(2)),  # Pop 2 levels to MainMenu
        ]
```

## Parallel Execution

### Rhizomorph Parallel Nodes

```python
@bt.tree
class ParallelTasks:
    @bt.action
    async def task_a(bb):
        print("Task A")
        return Status.SUCCESS

    @bt.action
    async def task_b(bb):
        print("Task B")
        return Status.SUCCESS

    @bt.action
    async def task_c(bb):
        print("Task C")
        return Status.SUCCESS

    @bt.root
    @bt.parallel
    def root():
        # All tasks run concurrently
        yield task_a
        yield task_b
        yield task_c
```

### Hypha Concurrent Transitions

Multiple transitions can fire simultaneously if they have tokens:

```python
@pn.net
class ConcurrentProcessing:
    @pn.transition()
    async def worker_a(consumed, bb, timebase):
        print("Worker A processing")
        return []

    @pn.transition()
    async def worker_b(consumed, bb, timebase):
        print("Worker B processing")
        return []

    # Both can fire if they have input tokens
```

## Best Practices

1. **Keep modules focused** - Each subtree/subnet should do one thing well
2. **Use clear interfaces** - Define what data flows between components
3. **Test components in isolation** - Verify subtrees work independently
4. **Document composition** - Explain how components fit together
5. **Avoid tight coupling** - Minimize dependencies between components
6. **Use type hints** - Make blackboard schemas explicit

## Common Patterns

### Pipeline Pattern

Process data through multiple stages:

```python
@pn.net
class Pipeline:
    # Stage 1: Validate
    @pn.transition()
    async def validate(consumed, bb, timebase):
        return [pn.Token(value=validate_data(consumed[0].value))]

    # Stage 2: Transform
    @pn.transition()
    async def transform(consumed, bb, timebase):
        return [pn.Token(value=transform_data(consumed[0].value))]

    # Stage 3: Save
    @pn.transition()
    async def save(consumed, bb, timebase):
        save_data(consumed[0].value)
        return []
```

### Supervisor Pattern

Monitor and manage multiple systems:

```python
@bt.tree
class Supervisor:
    @bt.sequence
    def system_a():
        yield check_system_a
        yield run_system_a

    @bt.sequence
    def system_b():
        yield check_system_b
        yield run_system_b

    @bt.root
    @bt.parallel
    def root():
        yield system_a
        yield system_b
```

### State Recovery Pattern

```python
@septum.state
class ErrorRecovery:
    @septum.on_state
    async def on_state(ctx):
        print("Attempting recovery...")
        await attempt_recovery(ctx)
        return ErrorRecovery.Events.RETRY

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(ErrorRecovery.Events.RETRY, Retry(PreviousState)),
            LabeledTransition(ErrorRecovery.Events.FAILED, ErrorState),
        ]
```

## See Also

- [Blackboards](blackboards.md) - Shared state across components
- [Timebases](timebases.md) - Time management for composed systems
- [Septum API](../api/septum.md) - State machine push/pop
- [Rhizomorph API](../api/rhizomorph.md) - Subtree documentation
- [Hypha API](../api/hypha.md) - Subnet documentation
