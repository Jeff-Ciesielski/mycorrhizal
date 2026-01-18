# Septum Transition Types - Complete Reference

This document provides a comprehensive reference for all transition types in the Septum FSM system.

## Overview

In Septum, state transitions determine how the state machine moves between states. Each transition type has specific semantics for lifecycle methods (`on_enter`, `on_leave`, `on_state`) and return values that control execution flow.

## Return Value Meaning

In the original implementation, `_process_transition` returns a boolean:
- **True**: Check the queue (continue the loop to wait for/process next message)
- **False**: Don't check queue (stop the loop, return to caller)

## Transition Types Table

| Transition | Type | Retry Counter | on_leave | on_enter | Next Step | Returns | Notes |
|------------|------|---------------|----------|----------|-----------|---------|-------|
| **StateSpec** | Transition | No change | ✅ called | ✅ called | Continue processing | False | Normal state transition |
| **Again** | Continuation | No change | ❌ not called | ❌ not called | Re-execute on_state immediately | False + background task | Originally: `asyncio.create_task(self.tick(timeout=0))` |
| **Unhandled** | Continuation | No change | ❌ not called | ❌ not called | Wait for message | True* | Only if can_dwell=True or has timeout, else raises BlockedInUntimedState |
| **Repeat** | Renewal | Reset | ✅ called | ✅ called | Continue processing | False | Re-enters state from on_enter, resets retry counter |
| **Restart** | Renewal | Reset | ✅ called | ✅ called | Wait for message | True | Re-enters state from on_enter, resets retry counter, then waits |
| **Retry** | Continuation | Increment | ✅ called | ✅ called | Continue processing | False | Decrements retry counter, fails if exceeded |
| **Push(states...)** | Transition | No change | ✅ called | ✅ called | Continue processing | False | Pushes states to stack, enters first state |
| **Pop** | Transition | No change | ✅ called | ✅ called | Continue processing | False | Pops from stack, enters popped state |
| **None** | - | No change | ❌ not called | ❌ not called | Wait for message | True* | Same as Unhandled |

## Key Behaviors

### Continuation (stay in same state)

**Again**: Stay in state, re-execute on_state immediately (no on_leave/on_enter)
- Use for: Immediate re-execution without waiting for events
- Does NOT call lifecycle methods
- Creates background task for continuous execution

**Unhandled**: Stay in state, wait for next message
- Use for: Waiting for events/messages
- Does NOT call lifecycle methods
- Requires `can_dwell=True` or timeout, otherwise raises `BlockedInUntimedState`

**Retry**: Stay in state, re-enter from on_enter, increment retry counter
- Use for: Retry logic with failure handling
- Calls lifecycle methods
- Decrements retry counter, fails if exceeded

### Renewal (re-enter current state)

**Repeat**: Re-enter from on_enter, reset retry counter, continue immediately
- Use for: Reset state and continue processing
- Calls lifecycle methods
- Resets retry counter
- Returns False (continues processing)

**Restart**: Re-enter from on_enter, reset retry counter, then wait for message
- Use for: Reset state and wait for next event
- Calls lifecycle methods
- Resets retry counter
- Returns True (waits for message)

### State Change

**StateSpec**: Leave current state, enter new state
- Calls `on_leave` of current state
- Transitions to new state
- Calls `on_enter` of new state
- Returns False (doesn't check queue)

**Push**: Stack states, enter first pushed state
- Pushes states onto stack
- Transitions to first pushed state (calls on_leave/on_enter)
- Returns False (continues processing)

**Pop**: Return to previous stacked state
- Pops state from stack
- Transitions to popped state (calls on_leave/on_enter)
- Returns False (continues processing)

## The Again Problem

### Original Implementation

```python
elif target == Again:
    # Schedule another tick without blocking
    asyncio.create_task(self.tick(timeout=0))
    return False  # Always returns False
```

**Issue:** The background task approach works for `run()` but causes issues with manual `tick()` calls.

### Current Decorator-Based Implementation Issue

- Manual `tick(timeout=0)` expects one state execution per call
- But `Again` means "keep going immediately"
- These are conflicting requirements!

## Proposed Solution

The `Again` transition needs different behavior based on context:

1. **When called from `run()`**: Keep the state machine running continuously
2. **When called from manual `tick()`**: Execute once and return

The key insight is that `Again` should return `True` (continue loop) when we want continuous execution, but return `False` (stop loop) when we want manual control.

The `block` parameter indicates the mode:
- `block=True` (automatic/run mode): Again returns True, keeps looping
- `block=False` (manual/tick mode): Again returns False, stops after one execution

## Implementation Guide

For each transition type, the implementation should:

### 1. StateSpec

```python
# Call on_leave of current state
await current_state.on_leave(ctx)
# Transition to new state
self.current_state = new_state
# Call on_enter of new state
await new_state.on_enter(ctx)
# Return False (don't check queue)
return False
```

### 2. Again

```python
# Do NOT call on_leave/on_enter
# Re-execute on_state immediately
if block:
    # Run mode: keep looping
    return True
else:
    # Manual mode: stop after this execution
    return False
```

### 3. Unhandled/None

```python
# Do NOT call on_leave/on_enter
if can_dwell or has_timeout:
    # Wait for message
    return True
else:
    # Can't wait in untimed state
    raise BlockedInUntimedState()
```

### 4. Repeat

```python
# Call on_leave
await current_state.on_leave(ctx)
# Reset retry counter
self.retry_count = 0
# Call on_enter
await current_state.on_enter(ctx)
# Return False (continue processing)
return False
```

### 5. Restart

```python
# Call on_leave
await current_state.on_leave(ctx)
# Reset retry counter
self.retry_count = 0
# Call on_enter
await current_state.on_enter(ctx)
# Return True (wait for message)
return True
```

### 6. Retry

```python
# Increment retry counter
self.retry_count += 1
# Check if exceeded
if self.retry_count > self.max_retries:
    # Call on_fail with its transition
    return await self.on_fail(ctx)
# Else: Call on_leave, call on_enter
await current_state.on_leave(ctx)
await current_state.on_enter(ctx)
# Return False (continue processing)
return False
```

### 7. Push

```python
# Push states to stack
for state in reversed(states):
    self.state_stack.append(state)
# Transition to first pushed state (calls on_leave/on_enter)
new_state = states[0]
await current_state.on_leave(ctx)
self.current_state = new_state
await new_state.on_enter(ctx)
# Return False (continue processing)
return False
```

### 8. Pop

```python
# Pop state from stack
new_state = self.state_stack.pop()
# Transition to popped state (calls on_leave/on_enter)
await current_state.on_leave(ctx)
self.current_state = new_state
await new_state.on_enter(ctx)
# Return False (continue processing)
return False
```

## Usage Examples

### Basic State Transition

```python
@septum.state
def StateA():
    class Events(Enum):
        GO_TO_B = auto()

    @septum.on_state
    async def on_state(ctx):
        return Events.GO_TO_B

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.GO_TO_B, StateB),
        ]
```

### Retry with Counter

```python
@septum.state(config=StateConfiguration(max_retries=3))
def AttemptOperation():
    class Events(Enum):
        SUCCESS = auto()
        RETRY = auto()
        FAIL = auto()

    @septum.on_state
    async def on_state(ctx):
        try:
            # Attempt operation
            result = await risky_operation()
            return Events.SUCCESS
        except Exception:
            if ctx.retry_count < 3:
                return Events.RETRY
            else:
                return Events.FAIL

    @septum.on_fail
    async def on_fail(ctx):
        # Called when retries exceeded
        logger.error("Operation failed after retries")
        return StateShutdown

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.SUCCESS, NextState),
            LabeledTransition(Events.RETRY, Retry),
            LabeledTransition(Events.FAIL, ErrorHandler),
        ]
```

### Push/Pop for Sub-states

```python
@septum.state
def MainMenu():
    class Events(Enum):
        START_SETTINGS = auto()
        EXIT = auto()

    @septum.on_state
    async def on_state(ctx):
        # Push SettingsMenu state
        return Push(SettingsMenu)

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.START_SETTINGS, Push(SettingsMenu)),
            LabeledTransition(Events.EXIT, Pop),
        ]
```

## Best Practices

1. **Use StateSpec for normal transitions** - Most transitions should be direct state-to-state
2. **Use Retry for failure recovery** - Leverage retry counter for transient failures
3. **Use Push/Pop for hierarchical navigation** - Stack-based navigation for menus/wizards
4. **Use Repeat/Restart sparingly** - Only when you need to reset state
5. **Avoid Again in most cases** - Use proper state transitions instead

## Common Patterns

### Pattern 1: Timeout with Retry

```python
@septum.state(config=StateConfiguration(timeout=5.0, max_retries=3))
def WaitForResponse():
    class Events(Enum):
        RECEIVED = auto()
        TIMEOUT = auto()

    @septum.on_timeout
    async def on_timeout(ctx):
        return Events.TIMEOUT

    @septum.on_state
    async def on_state(ctx):
        # Wait for response message
        return Events.RECEIVED

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.RECEIVED, ProcessResponse),
            LabeledTransition(Events.TIMEOUT, Retry),
        ]
```

### Pattern 2: Hierarchical Menu

```python
@septum.state
def MainMenu():
    @septum.on_state
    async def on_state(ctx):
        # Show menu options
        return Events.SHOW_SETTINGS

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.SHOW_SETTINGS, Push(SettingsMenu)),
            LabeledTransition(Events.BACK, Pop),
        ]
```

### Pattern 3: Error Recovery with Restart

```python
@septum.state
def ProcessingState():
    class Events(Enum):
        ERROR = auto()

    @septum.on_state
    async def on_state(ctx):
        try:
            await process()
            return Events.DONE
        except Exception:
            return Events.ERROR

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.ERROR, Restart),
        ]
```

## References

- Septum library documentation
- State machine design patterns
- Asyncio best practices
