# Septum FSM Troubleshooting

This guide covers common issues, error messages, and debugging techniques for Septum finite state machines.

## Common Errors

### BlockedInUntimedState

**Error message:**
```
mycorrhizal.septum.core.BlockedInUntimedState: FSM blocked in untimed state 'MyState'
```

**Cause:**
The FSM is in a state without a timeout configured, and no message is available to process.

**Solutions:**

1. **Configure a timeout:**
```python
from mycorrhizal.septum.core import StateConfiguration

@septum.state(config=StateConfiguration(timeout=5.0))
class WaitingState:
    # ... state implementation
```

2. **Send a message:**
```python
# In your code
fsm.send_message("process_now")
```

3. **Use `can_dwell=True` for intentionally blocking states:**
```python
@septum.state(config=StateConfiguration(can_dwell=True))
class WaitingForUserInput:
    # ... state implementation
```

### PopFromEmptyStack

**Error message:**
```
mycorrhizal.septum.core.PopFromEmptyStack: Cannot pop from empty state stack
```

**Cause:**
Attempting to `Pop` when no states have been pushed onto the stack.

**Solutions:**

1. **Ensure `Push` before `Pop`:**
```python
@septum.state
class MainMenu:
    @septum.transitions
    def transitions():
        return [
            # Must push before popping later
            LabeledTransition(Events.SETTINGS, Push(SettingsMenu, MainMenu)),
        ]

@septum.state
class SettingsMenu:
    @septum.transitions
    def transitions():
        return [
            # Now pop is safe
            LabeledTransition(Events.BACK, Pop),
        ]
```

2. **Track stack depth in state logic:**
```python
@septum.state
class SafeState:
    @septum.on_state
    async def on_state(ctx):
        stack_depth = ctx.common.get("stack_depth", 0)
        if stack_depth > 0:
            return Events.POP_BACK
        else:
            return Events.GO_HOME

    @septum.on_enter
    async def on_enter(ctx):
        # Track stack depth
        ctx.common["stack_depth"] = ctx.common.get("stack_depth", 0)

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.POP_BACK, Pop),
            LabeledTransition(Events.GO_HOME, HomeState),
        ]
```

### Timeout Errors

**Error message:**
```
State timeout occurred in 'MyState' after 30.0 seconds
```

**Cause:**
State exceeded its configured timeout duration.

**Solutions:**

1. **Increase timeout:**
```python
@septum.state(config=StateConfiguration(timeout=60.0))
class SlowOperationState:
    # ... state implementation
```

2. **Handle timeout in `on_timeout`:**
```python
@septum.state(config=StateConfiguration(timeout=30.0))
class ExternalAPIState:
    @septum.on_timeout
    async def on_timeout(ctx):
        logger.warning("API call timed out, using fallback")
        return Events.USE_FALLBACK

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.USE_FALLBACK, FallbackState),
        ]
```

3. **Make state operation faster:**
```python
@septum.state
class OptimizedState:
    @septum.on_state
    async def on_state(ctx):
        # Use async I/O, avoid blocking operations
        result = await async_api_call()  # Not: sync_api_call()
        return Events.DONE
```

### Retry Exhaustion

**Error message:**
```
State 'RetryState' exceeded maximum retry count (3)
```

**Cause:**
State with retry configuration has exhausted all retry attempts.

**Solutions:**

1. **Handle retry exhaustion:**
```python
@septum.state(config=StateConfiguration(retries=3))
class RetryState:
    @septum.on_fail
    async def on_fail(ctx):
        logger.error(f"Operation failed after {ctx.retry_count} retries")
        # Transition to error state
        return Events.FAILED

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.FAILED, ErrorState),
        ]
```

2. **Increase retry count:**
```python
@septum.state(config=StateConfiguration(retries=10))
class RetryState:
    # ... state implementation
```

3. **Fix underlying issue:**
```python
@septum.state(config=StateConfiguration(retries=3))
class DatabaseState:
    @septum.on_state
    async def on_state(ctx):
        try:
            # Fix the actual issue causing retries
            result = await db_connection.execute(query)
            return Events.SUCCESS
        except ConnectionError:
            # Retry only for transient errors
            return Events.RETRY
```

### Validation Errors

**Error message:**
```
ValidationError: State 'InvalidState' has unreachable transition
```

**Cause:**
FSM construction detected structural issues (unreachable states, invalid transitions, etc.).

**Solutions:**

1. **Check state references:**
```python
# BAD: Typo in state name
LabeledTransition(Events.DONE, NexState)  # Typo!

# GOOD: Correct state reference
LabeledTransition(Events.DONE, NextState)
```

2. **Ensure all states are reachable:**
```python
# BAD: OrphanedState is never reached
@septum.state
class OrphanedState:
    # ... no transitions to this state

# GOOD: State is reachable
@septum.state
class ConnectedState:
    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.CONNECT, OrphanedState),
        ]
```

3. **Handle all event enum values:**
```python
# BAD: Missing transition for Events.ERROR
@septum.state
class MyState:
    class Events(Enum):
        SUCCESS = auto()
        ERROR = auto()  # Not handled!

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.SUCCESS, NextState),
            # ERROR not handled!
        ]

# GOOD: All events handled
@septum.transitions
def transitions():
    return [
        LabeledTransition(Events.SUCCESS, NextState),
        LabeledTransition(Events.ERROR, ErrorState),
    ]
```

## Debugging Techniques

### Enable Debug Logging

```python
import logging

# Enable detailed FSM logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

logger = logging.getLogger("mycorrhizal.septum")
logger.setLevel(logging.DEBUG)
```

**Debug log output:**
```
[FSM] Transitioned to MyState
[FSM]   [DEBUG] common object id: 140234567890
[FSM]   [DEBUG] on_enter is None? False
[FSM]   [DEBUG] After on_enter, common: {'counter': 1}
[FSM] Executing on_state handler
[FSM] Transitioned to NextState
```

### Inspect FSM State at Runtime

```python
async def debug_fsm(fsm: StateMachine):
    """Print detailed FSM state."""
    print(f"Current state: {fsm.current_state.name}")
    print(f"Stack depth: {len(fsm.state_stack)}")
    print(f"Stack contents:")
    for i, state in enumerate(fsm.state_stack):
        print(f"  {i}: {state.name}")
    print(f"Common context: {fsm.context.common}")
    print(f"Message queue depth: {fsm.message_queue.qsize()}")
```

### Visualize FSM Structure

Export FSM to Mermaid diagram for visual debugging:

```python
from mycorrhizal.septum.util import to_mermaid

fsm = StateMachine(initial_state=MyState)
await fsm.initialize()

# Export to Mermaid
mermaid = to_mermaid(fsm)
print(mermaid)

# Copy output to https://mermaid.live/ for visualization
```

### Add Breakpoints in State Handlers

```python
@septum.state
class DebugState:
    @septum.on_enter
    async def on_enter(ctx):
        print(f"[DEBUG] Entering {DebugState.__name__}")
        print(f"[DEBUG] Context: {ctx.common}")
        # Add breakpoint here in debugger
        import pdb; pdb.set_trace()

    @septum.on_state
    async def on_state(ctx):
        print(f"[DEBUG] Executing {DebugState.__name__}")
        result = await some_operation()
        print(f"[DEBUG] Result: {result}")
        return Events.DONE

    @septum.on_leave
    async def on_leave(ctx):
        print(f"[DEBUG] Exiting {DebugState.__name__}")
```

### Trace State Transitions

```python
class TracingFSM(StateMachine):
    """FSM with transition tracing."""

    async def transition_to(self, target_state):
        """Override to trace transitions."""
        from_state = self.current_state.name if self.current_state else "None"
        to_state = target_state.name

        print(f"[TRACE] Transition: {from_state} -> {to_state}")
        print(f"[TRACE] Stack: {[s.name for s in self.state_stack]}")
        print(f"[TRACE] Context: {self.context.common}")

        # Call parent implementation
        result = await super().transition_to(target_state)

        print(f"[TRACE] Transition complete")
        return result
```

## Common Issues

### FSM Hangs

**Symptoms:**
- FSM stops responding
- No state transitions occurring
- Messages not being processed

**Diagnosis:**

```python
async def diagnose_hang(fsm: StateMachine):
    """Diagnose why FSM is hanging."""
    print(f"Current state: {fsm.current_state.name}")
    print(f"State has timeout: {fsm.current_state.config.timeout}")
    print(f"State can dwell: {fsm.current_state.config.can_dwell}")
    print(f"Messages in queue: {fsm.message_queue.qsize()}")
    print(f"Stack depth: {len(fsm.state_stack)}")
```

**Solutions:**

1. **Check for blocking operations:**
```python
# BAD: Blocking call
@septum.on_state
async def on_state(ctx):
    result = requests.get(url)  # Blocks!
    return Events.DONE

# GOOD: Async call
@septum.on_state
async def on_state(ctx):
    result = await aiohttp.get(url)  # Non-blocking
    return Events.DONE
```

2. **Ensure state returns event:**
```python
# BAD: No return value
@septum.on_state
async def on_state(ctx):
    print("Processing...")
    # Missing return!

# GOOD: Returns event
@septum.on_state
async def on_state(ctx):
    print("Processing...")
    return Events.DONE
```

3. **Add timeout:**
```python
@septum.state(config=StateConfiguration(timeout=10.0))
class PotentiallySlowState:
    # ... state implementation
```

### Unexpected State Transitions

**Symptoms:**
- FSM transitions to wrong state
- States skipped or repeated
- Transition order incorrect

**Diagnosis:**

```python
@septum.state
class DebugTransitionState:
    @septum.on_state
    async def on_state(ctx):
        # Log decision-making
        logger.debug(f"Current context: {ctx.common}")

        if ctx.common.get("should_retry"):
            logger.debug("Deciding: RETRY")
            return Events.RETRY
        else:
            logger.debug("Deciding: DONE")
            return Events.DONE

    @septum.transitions
    def transitions():
        logger.debug("Available transitions:")
        for t in [
            LabeledTransition(Events.RETRY, RetryState),
            LabeledTransition(Events.DONE, DoneState),
        ]:
            logger.debug(f"  {t.event} -> {t.target}")
        return [
            LabeledTransition(Events.RETRY, RetryState),
            LabeledTransition(Events.DONE, DoneState),
        ]
```

**Solutions:**

1. **Check transition logic:**
```python
# Verify transition mappings
@septum.transitions
def transitions():
    return [
        # Ensure events map to correct states
        LabeledTransition(Events.SUCCESS, NextState),  # Correct
        LabeledTransition(Events.FAILURE, NextState),  # Bug! Should be ErrorState
    ]
```

2. **Check for event name conflicts:**
```python
# BAD: Event name conflicts
@septum.state
class State1:
    class Events(Enum):
        NEXT = auto()

@septum.state
class State2:
    class Events(Enum):
        NEXT = auto()  # Same name, different enum!

# GOOD: Unique event names
@septum.state
class State1:
    class Events(Enum):
        STATE1_NEXT = auto()

@septum.state
class State2:
    class Events(Enum):
        STATE2_NEXT = auto()
```

### Memory Leaks

**Symptoms:**
- Memory usage grows over time
- FSM instance count increasing
- Context data accumulating

**Diagnosis:**

```python
import tracemalloc
import gc

async def check_memory_usage():
    """Check for memory leaks."""
    gc.collect()
    snapshot1 = tracemalloc.take_snapshot()

    # Run FSM for a while
    await run_fsm_extended()

    gc.collect()
    snapshot2 = tracemalloc.take_snapshot()

    # Compare snapshots
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    for stat in top_stats[:10]:
        print(stat)
```

**Solutions:**

1. **Clean up context data:**
```python
@septum.state
class CleanState:
    @septum.on_leave
    async def on_leave(ctx):
        # Clean up large data structures
        if "large_data" in ctx.common:
            del ctx.common["large_data"]
```

2. **Avoid accumulating data:**
```python
# BAD: Unbounded growth
@septum.state
class AccumulatingState:
    @septum.on_state
    async def on_state(ctx):
        # List grows forever
        ctx.common.setdefault("history", []).append(data)
        return Events.DONE

# GOOD: Bounded size
@septum.state
class BoundedState:
    @septum.on_state
    async def on_state(ctx):
        history = ctx.common.setdefault("history", [])
        history.append(data)
        # Keep only last 100 items
        if len(history) > 100:
            history.pop(0)
        return Events.DONE
```

3. **Reuse FSM instances:**
```python
# BAD: Creating new FSM for each request
async def handle_request(request):
    fsm = StateMachine(initial_state=ProcessState)
    await fsm.initialize()
    await fsm.run()
    # FSM discarded

# GOOD: Pool of FSMs
class FSMPool:
    def __init__(self, state_class, size=10):
        self.pool = asyncio.Queue(maxsize=size)
        self.state_class = state_class

    async def initialize(self):
        for _ in range(size):
            fsm = StateMachine(initial_state=self.state_class)
            await fsm.initialize()
            await self.pool.put(fsm)

    async def acquire(self):
        return await self.pool.get()

    async def release(self, fsm):
        # Reset context if needed
        await self.pool.put(fsm)
```

### Performance Issues

**Symptoms:**
- Slow state transitions
- High CPU usage
- Poor throughput

**Diagnosis:**

```python
import time

@septum.state
class ProfiledState:
    @septum.on_state
    async def on_state(ctx):
        start = time.perf_counter()

        # State logic
        result = await expensive_operation()

        elapsed = time.perf_counter() - start
        if elapsed > 0.1:  # Log if > 100ms
            logger.warning(f"Slow state: {elapsed:.3f}s")

        return Events.DONE
```

**Solutions:**

1. **Profile and optimize hot paths:**
```python
# Use profiler
import cProfile

async def profile_fsm():
    fsm = StateMachine(initial_state=MyState)
    await fsm.initialize()

    pr = cProfile.Profile()
    pr.enable()

    for _ in range(1000):
        await fsm.run()

    pr.disable()
    pr.print_stats(sort='cumtime')
```

2. **Avoid unnecessary work:**
```python
# BAD: Expensive operation every tick
@septum.on_state
async def on_state(ctx):
    # Recalculated every time
    result = expensive_computation(input_data)
    return Events.DONE

# GOOD: Cache when possible
@septum.on_enter
async def on_enter(ctx):
    # Calculate once on entry
    ctx.common["cached_result"] = expensive_computation(ctx.common["input_data"])

@septum.on_state
async def on_state(ctx):
    # Use cached result
    result = ctx.common["cached_result"]
    return Events.DONE
```

3. **Use async operations:**
```python
# BAD: Blocking I/O
@septum.on_state
async def on_state(ctx):
    with open(filename, 'r') as f:
        data = f.read()  # Blocks!
    return Events.DONE

# GOOD: Async I/O
@septum.on_state
async def on_state(ctx):
    async with aiofiles.open(filename, 'r') as f:
        data = await f.read()  # Non-blocking
    return Events.DONE
```

## Getting Help

If you're still stuck after trying these solutions:

1. **Check the examples:**
   ```bash
   ls examples/septum/
   ```

2. **Review the API reference:**
   - [Septum API](../api/septum.md)

3. **Enable debug logging:**
   ```python
   logging.basicConfig(level=logging.DEBUG)
   ```

4. **Create a minimal reproduction:**
   ```python
   @septum.state
   class MinimalState:
       # Simplified version of your problematic state
       pass
   ```

5. **Report issues:**
   - Include error messages
   - Share relevant code snippets
   - Describe expected vs actual behavior
   - Include FSM structure (Mermaid export)

## See Also

- [Production Guide](production.md) - Deployment and performance
- [API Reference](../api/septum.md) - Complete API documentation
- [Best Practices](../guides/best-practices.md) - Design patterns
- [PDA Guide](../guides/septum-pda-guide.md) - Hierarchical state machines
