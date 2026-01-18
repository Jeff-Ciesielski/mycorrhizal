# Best Practices

Recommended patterns and anti-patterns for building robust systems with Mycorrhizal.

## Design Principles

### 1. Keep Components Small

**Good**: Small, focused nodes/transitions/states

```python
@bt.action
async def check_door(bb) -> Status:
    """Check if door is open."""
    return Status.SUCCESS if bb.door_open else Status.FAILURE
```

**Bad**: Large, multi-purpose nodes

```python
@bt.action
async def handle_everything(bb) -> Status:
    """Check door, move robot, grab item, return..."""
    # 50 lines of code doing multiple things
```

### 2. Use Descriptive Names

**Good**: Clear, intent-revealing names

```python
@bt.action
async def validate_user_credentials(bb) -> Status:
    pass
```

**Bad**: Vague or abbreviated names

```python
@bt.action
async def chk_usr(bb) -> Status:
    pass
```

### 3. Favor Composition Over Complexity

**Good**: Combine simple components

```python
@bt.sequence
def login_flow():
    yield validate_credentials
    yield check_account_status
    yield create_session
```

**Bad**: One giant complex component

```python
@bt.action
async def login_and_validate_and_check(bb) -> Status:
    # Everything in one function
```

## Rhizomorph Best Practices

### Use Condition Nodes for Guards

```python
@bt.condition
def is_authenticated(bb) -> bool:
    return bb.user is not None

@bt.action
async def access_resource(bb) -> Status:
    # Safe to assume user is authenticated here
    return Status.SUCCESS
```

### Return Appropriate Status

```python
@bt.action
async def fetch_data(bb) -> Status:
    try:
        bb.data = await api_call()
        return Status.SUCCESS  # Work completed
    except NotFoundError:
        return Status.FAILURE  # Expected failure
    except CriticalError:
        return Status.ERROR  # Unexpected error, abort tree
```

### Avoid Side Effects in Conditions

**Bad**:

```python
@bt.condition
def check_and_consume(bb) -> bool:
    item = bb.queue.pop(0)  # Side effect!
    return item is not None
```

**Good**:

```python
@bt.condition
def has_item(bb) -> bool:
    return len(bb.queue) > 0

@bt.action
async def consume_item(bb) -> Status:
    bb.queue.pop(0)
    return Status.SUCCESS
```

## Septum Best Practices

### Use Timeouts Appropriately

```python
@septum.state(config=StateConfiguration(timeout=30.0))
class WaitingForResponse:
    @septum.on_timeout
    async def on_timeout(ctx):
        # Handle timeout gracefully
        ctx.timeout_occurred = True
        return WaitingForResponse.Events.RETRY
```

### Keep Transition Logic Clear

**Good**: Explicit transition mapping

```python
@septum.transitions
def transitions():
    return [
        LabeledTransition(Events.SUCCESS, CompletedState),
        LabeledTransition(Events.FAILURE, RetryState),
        LabeledTransition(Events.FATAL, ErrorState),
    ]
```

### Use on_enter/on_exit for Setup/Teardown

```python
@septum.state
class ProcessingState:
    @septum.on_enter
    async def on_enter(ctx):
        ctx.start_time = time.time()
        ctx.processing = True

    @septum.on_exit
    async def on_exit(ctx):
        ctx.processing = False
        ctx.elapsed = time.time() - ctx.start_time
```

## Hypha Best Practices

### Model Places Clearly

```python
# Use QUEUE for ordered items
@pn.place(type=PlaceType.QUEUE)
def work_queue(bb):
    return []

# Use SET for unique items
@pn.place(type=PlaceType.SET)
def unique_tasks(bb):
    return set()

# Use BOOLEAN for flags
@pn.place(type=PlaceType.BOOLEAN)
def is_processing(bb):
    return False
```

### Keep Transitions Focused

```python
@pn.transition()
async def validate_data(consumed, bb, timebase):
    """Validate incoming data."""
    data = consumed[0].value

    if not is_valid(data):
        return []  # No output tokens

    # Produce validated token
    return [pn.Token(value=validated_data)]
```

### Document Arcs

```python
@pn.arc
def input_to_validator():
    """Route raw input to validation transition."""
    return pn.Arc(input_place, validator_transition)
```

## Blackboard Best Practices

### Use Type Hints

```python
from pydantic import BaseModel
from typing import Annotated

class GameContext(BaseModel):
    player_health: int  # Clear type
    score: float
    inventory: list[str]
```

### Provide Defaults

```python
class SystemConfig(BaseModel):
    max_retries: int = 3
    timeout: float = 30.0
    debug_mode: bool = False
```

### Document Important Fields

```python
class RobotContext(BaseModel):
    """Shared state for robot control system."""

    battery_level: int = 100
    """Current battery percentage (0-100)."""

    current_location: str = "home"
    """Robot's current location identifier."""
```

## Anti-Patterns

### Don't Overuse Async

**Bad**: Unnecessary async

```python
@bt.action
async def simple_check(bb) -> Status:
    return Status.SUCCESS  # No I/O, why async?
```

**Good**: Sync for simple operations

```python
@bt.action
def simple_check(bb) -> Status:
    return Status.SUCCESS
```

### Don't Create God Objects

**Bad**: Everything in one blackboard

```python
class Universe(BaseModel):
    everything_a: str
    everything_b: int
    everything_c: list
    # 100 more fields...
```

**Good**: Focused contexts

```python
class UserContext(BaseModel):
    user_id: str
    permissions: list[str]

class SessionContext(BaseModel):
    session_id: str
    started_at: float
```

### Don't Mix Concerns

**Bad**: State machine doing HTTP and DB

```python
@septum.state
class MixedConcernsState:
    @septum.on_state
    async def on_state(ctx):
        # HTTP call
        response = await http_get(url)
        # DB call
        result = await db_query(query)
        # Business logic
        processed = calculate(response, result)
```

**Good**: Separate layers

```python
# State machine orchestrates
@septum.state
class OrchestratorState:
    @septum.on_state
    async def on_state(ctx):
        # Delegates to services
        data = await data_service.fetch(ctx.params)
        ctx.result = process(data)
```

### Don't Ignore Errors

**Bad**: Silent failures

```python
@bt.action
async def risky_operation(bb) -> Status:
    try:
        do_work()
    except Exception:
        pass  # Swallowed error
    return Status.SUCCESS
```

**Good**: Explicit error handling

```python
@bt.action
async def risky_operation(bb) -> Status:
    try:
        do_work()
        return Status.SUCCESS
    except RecoverableError as e:
        logger.warning(f"Recoverable: {e}")
        return Status.FAILURE
    except Exception as e:
        logger.error(f"Unexpected: {e}")
        bb.last_error = str(e)
        return Status.ERROR
```

## Testing Best Practices

### Test Components in Isolation

```python
@pytest.mark.asyncio
async def test_action_node():
    bb = TestContext(value=10)
    node = MyAction()
    result = await node.execute(bb)
    assert result == Status.SUCCESS
    assert bb.value == 20
```

### Use Fixtures for Setup

```python
@pytest.fixture
def game_context():
    return GameContext(
        player_health=100,
        score=0
    )

@pytest.mark.asyncio
async def test_gameplay(game_context):
    # Test with pre-configured context
    pass
```

### Test Edge Cases

```python
@pytest.mark.parametrize("input,expected", [
    (0, Status.FAILURE),
    (10, Status.SUCCESS),
    (100, Status.SUCCESS),
])
async def test_boundary_conditions(input, expected):
    bb = Context(value=input)
    result = await node.execute(bb)
    assert result == expected
```

## Performance Tips

### Profile Hot Paths

```python
import time

@bt.action
async def monitored_action(bb) -> Status:
    start = time.perf_counter()
    result = await do_work(bb)
    elapsed = time.perf_counter() - start
    if elapsed > 0.1:  # Log slow operations
        logger.warning(f"Slow action: {elapsed:.3f}s")
    return result
```

### Avoid Unnecessary Copies

**Bad**:

```python
@bt.action
async def process(bb):
    data = bb.data.copy()  # Unnecessary copy
    data["processed"] = True
    bb.data = data
```

**Good**: Mutate when safe

```python
@bt.action
async def process(bb):
    bb.data["processed"] = True
```

### Use Efficient Data Structures

```python
from collections import deque

class TaskContext(BaseModel):
    # O(1) append/popleft vs O(n) for list
    task_queue: deque = deque()
```

## Documentation Tips

### Document Intent

```python
@bt.tree
class EnemyAI:
    """
    Behavior tree for enemy AI in combat.

    Prioritizes:
    1. Survival (low health -> flee)
    2. Combat (has target -> attack)
    3. Patrol (no target -> patrol)
    """
    pass
```

### Document Complex Logic

```python
@pn.transition()
async def complex_transition(consumed, bb, timebase):
    """
    Validates and transforms input data.

    Validation rules:
    - Must have non-empty 'name' field
    - 'value' must be between 0 and 100
    - 'timestamp' must be within last hour

    Returns:
        List of tokens with validated data, or empty list if invalid
    """
    pass
```

## See Also

- [Composition](composition.md) - Building modular systems
- [Observability](observability.md) - Monitoring and debugging
- [Blackboards](blackboards.md) - State management patterns
