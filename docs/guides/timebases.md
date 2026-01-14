# Timebases

Timebases provide a time abstraction layer for Mycorrhizal, enabling simulation, testing, and deterministic execution.

## What is a Timebase?

A **timebase** is an object that provides the current time to your system. Different timebase implementations allow you to:

- Use real wall-clock time in production
- Use monotonic time for accurate intervals
- Use simulated time for testing
- Use manually controlled time for deterministic execution

## Available Timebases

### WallClock

Real wall-clock time. Use this for production systems.

```python
from mycorrhizal.common.timebase import WallClock

clock = WallClock()
print(clock.now())  # Current actual time
```

### MonotonicClock

Monotonic system time (guaranteed to always increase). Use for intervals and timeouts.

```python
from mycorrhizal.common.timebase import MonotonicClock

clock = MonotonicClock()
start = clock.now()
# ... do work ...
elapsed = clock.now() - start
```

### UTCClock

UTC time zone aware wall-clock time.

```python
from mycorrhizal.common.timebase import UTCClock

clock = UTCClock()
print(clock.now())  # Current UTC time
```

### CycleClock

Stepped time for simulation and testing. Time only advances when you explicitly step it.

```python
from mycorrhizal.common.timebase import CycleClock

clock = CycleClock(initial=0.0)

print(clock.now())  # 0.0
clock.advance(amount=1.0)
print(clock.now())  # 1.0
clock.advance(amount=1.0)
print(clock.now())  # 2.0
```

### DictatedClock

Programmatically controlled time. You set the time explicitly.

```python
from mycorrhizal.common.timebase import DictatedClock

clock = DictatedClock(initial=0.0)

print(clock.now())  # 0.0
clock.set_time(100.0)
print(clock.now())  # 100.0
clock.set_time(200.0)
print(clock.now())  # 200.0
```

## Using Timebases with DSLs

### Rhizomorph (Behavior Trees)

```python
from mycorrhizal.rhizomorph.core import bt, Runner as BTRunner
from mycorrhizal.common.timebase import CycleClock

clock = CycleClock(initial=0.0)
runner = BTRunner(tree=tree, blackboard=bb, timebase=clock)

# Run for several simulation cycles
for _ in range(10):
    await runner.tick()
    clock.advance(amount=1.0)
```

### Hypha (Petri Nets)

```python
from mycorrhizal.hypha.core import Runner as PNRunner
from mycorrhizal.common.timebase import MonotonicClock

clock = MonotonicClock()
runner = PNRunner(net=net, blackboard=bb, timebase=clock)
```

### Enoki (State Machines)

```python
from mycorrhizal.enoki.core import StateMachine, StateConfiguration
from mycorrhizal.common.timebase import WallClock

@enoki.state(config=StateConfiguration(timeout=5.0))
class MyState:
    # Timeout uses the timebase
    pass
```

## Simulation Example

Simulate a day in the life of a system:

```python
from mycorrhizal.common.timebase import CycleClock

clock = CycleClock(initial=0.0)
runner = BTRunner(tree=tree, blackboard=bb, timebase=clock)

# Simulate 24 hours, advancing by 1 hour each tick
for hour in range(24):
    print(f"Hour {hour}: {clock.now()}")
    await runner.tick()
    clock.advance(amount=3600.0)  # 1 hour in seconds
```

## Testing Example

Test timeout behavior without waiting:

```python
import pytest
from mycorrhizal.common.timebase import CycleClock

@pytest.mark.asyncio
async def test_timeout():
    clock = CycleClock(initial=0.0)

    @enoki.state(config=StateConfiguration(timeout=5.0))
    class TimeoutState:
        @enoki.on_timeout
        async def on_timeout(ctx):
            ctx.timed_out = True

    fsm = StateMachine(initial_state=TimeoutState, timebase=clock)
    await fsm.initialize()

    # Initially not timed out
    assert not fsm.blackboard.timed_out

    # Advance time past timeout
    clock.advance(amount=6.0)
    await fsm.run()

    # Now timed out!
    assert fsm.blackboard.timed_out
```

## Deterministic Execution

For reproducible tests and simulations:

```python
from mycorrhizal.common.timebase import DictatedClock

clock = DictatedClock(initial=0.0)

# Every run produces identical results
for step in [0.0, 1.5, 3.0, 5.0]:
    clock.set_time(step)
    result = await runner.tick()
    print(f"At {step}: {result}")
```

## Time-Based Transitions

Create time-based behavior:

```python
@bt.tree
class TimedBehavior:
    @bt.condition
    def is_morning(bb):
        return bb.timebase.hour() >= 6 and bb.timebase.hour() < 12

    @bt.condition
    def is_afternoon(bb):
        return bb.timebase.hour() >= 12 and bb.timebase.hour() < 18

    @bt.action
    async def morning_routine(bb):
        print("Good morning!")
        return Status.SUCCESS

    @bt.action
    async def afternoon_routine(bb):
        print("Good afternoon!")
        return Status.SUCCESS
```

## Best Practices

1. **Use WallClock for production** - Real-world systems need real time
2. **Use CycleClock for testing** - Enables fast, deterministic tests
3. **Pass timebase explicitly** - Don't rely on global state
4. **Document time assumptions** - Note if your code assumes specific time units
5. **Handle time wraps** - Be careful with code that assumes time always increases

## Common Patterns

### Time-Restricted Actions

```python
@bt.condition
def is_business_hours(bb):
    hour = bb.timebase.now().hour
    return 9 <= hour < 17
```

### Rate Limiting

```python
class RateLimiter:
    def __init__(self, max_calls: int, period: float, timebase):
        self.max_calls = max_calls
        self.period = period
        self.timebase = timebase
        self.calls = []

    def can_proceed(self) -> bool:
        now = self.timebase.now()
        # Remove old calls outside the period
        self.calls = [c for c in self.calls if now - c < self.period]
        return len(self.calls) < self.max_calls

    def record_call(self):
        self.calls.append(self.timebase.now())
```

## See Also

- [Blackboards](blackboards.md) - State management
- [Composition](composition.md) - Combining systems
- [API Reference](../api/common.md) - Timebase interface
