# Septum FSM in Production

This guide covers deploying Septum finite state machines in production environments, including performance characteristics, monitoring, error handling, and scaling considerations.

## Performance Characteristics

### Benchmarks

Based on production benchmarks (see `tests/septum/run_benchmarks.py`):

**State Transition Overhead:**
- Approximately 6-8 µs per state transition
- Throughput: ~125,000-156,000 transitions/second
- Includes event handling, lifecycle methods, and state transitions

**Memory Usage:**
- Approximately 4.28 KB per FSM instance
- 10,000 concurrent FSMs use ~42 MB
- Memory scales linearly with FSM count

**FSM Creation:**
- Approximately 0.025 ms per FSM instance
- Can create ~40,000 FSMs/second
- Includes validation and initialization

**Concurrency:**
- Tested with 100+ concurrent FSMs
- Each FSM runs independently
- Asyncio-based concurrency model

### Performance Considerations

**State Transition Speed:**
- State transitions are lightweight (microsecond-scale)
- Bottlenecks typically in state logic, not FSM framework
- Profile state handlers for hot paths

**Memory Scaling:**
- Memory per FSM is minimal (~4 KB)
- Consider pooling FSMs for high-count scenarios
- Monitor memory in production for FSM count vs usage

**Message Throughput:**
- Message sending: ~100,000+ messages/second
- Message processing: depends on state logic
- Use message batching for high-throughput scenarios

## Deployment Patterns

### Single FSM Instance

Simplest deployment pattern for standalone applications:

```python
import asyncio
from mycorrhizal.septum.core import StateMachine, septum, LabeledTransition
from enum import Enum, auto

@septum.state
class MyState:
    class Events(Enum):
        DONE = auto()

    @septum.on_state
    async def on_state(ctx):
        # State logic
        return Events.DONE

    @septum.transitions
    def transitions():
        return [LabeledTransition(Events.DONE, MyState)]

async def main():
    fsm = StateMachine(initial_state=MyState)
    await fsm.initialize()

    # Run indefinitely
    while True:
        await fsm.run()

if __name__ == "__main__":
    asyncio.run(main())
```

### Multiple FSM Instances

For managing multiple independent state machines:

```python
async def run_multiple_fsms():
    """Run multiple FSM instances concurrently."""
    fsms = [
        StateMachine(initial_state=MyState)
        for _ in range(10)
    ]

    # Initialize all
    for fsm in fsms:
        await fsm.initialize()

    # Run concurrently
    tasks = [fsm.run() for fsm in fsms]
    await asyncio.gather(*tasks)
```

### FSM Pool Pattern

For high-throughput scenarios with FSM reuse:

```python
class FSMPool:
    """Pool of pre-initialized FSMs."""

    def __init__(self, state_class, pool_size=10):
        self.pool = asyncio.Queue(maxsize=pool_size)
        self.state_class = state_class
        self.pool_size = pool_size

    async def initialize(self):
        """Pre-create and initialize FSMs."""
        for _ in range(self.pool_size):
            fsm = StateMachine(initial_state=self.state_class)
            await fsm.initialize()
            await self.pool.put(fsm)

    async def acquire(self):
        """Get an FSM from the pool."""
        return await self.pool.get()

    async def release(self, fsm):
        """Return an FSM to the pool."""
        await self.pool.put(fsm)

async def with_fsm_pool():
    pool = FSMPool(MyState, pool_size=10)
    await pool.initialize()

    # Use FSM from pool
    fsm = await pool.acquire()
    await fsm.run()
    await pool.release(fsm)
```

## Monitoring and Observability

### Logging

Septum provides built-in logging for state transitions:

```python
import logging

# Enable FSM logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mycorrhizal.septum")
```

**Log levels:**
- `INFO`: State transitions, FSM lifecycle events
- `DEBUG`: Detailed execution trace, message handling
- `WARNING`: Validation warnings, potential issues
- `ERROR`: Runtime errors, exceptions

### Custom Metrics

Add custom metrics to track FSM behavior:

```python
import time
from collections import Counter

class MetricsFSM:
    """FSM with custom metrics."""

    def __init__(self):
        self.state_counts = Counter()
        self.transition_times = []
        self.start_time = None

    @septum.state
    class MyState:
        @septum.on_enter
        async def on_enter(ctx):
            # Track state entry
            metrics = ctx.common.get("metrics")
            if metrics:
                metrics.state_counts[MyState.__name__] += 1
                metrics.start_time = time.time()

        @septum.on_leave
        async def on_leave(ctx):
            # Track time in state
            metrics = ctx.common.get("metrics")
            if metrics and metrics.start_time:
                elapsed = time.time() - metrics.start_time
                metrics.transition_times.append(elapsed)

    @classmethod
    def report_metrics(cls, metrics):
        """Generate metrics report."""
        print(f"State visits: {dict(metrics.state_counts)}")
        if metrics.transition_times:
            avg_time = sum(metrics.transition_times) / len(metrics.transition_times)
            print(f"Avg transition time: {avg_time:.4f}s")
```

### Health Checks

Implement health checks for FSM-based services:

```python
class FSMHealthChecker:
    """Health checking for FSM instances."""

    def __init__(self, fsm):
        self.fsm = fsm
        self.last_activity = time.time()
        self.timeout = 30.0  # 30 seconds

    async def check_health(self):
        """Check if FSM is healthy."""
        elapsed = time.time() - self.last_activity

        if elapsed > self.timeout:
            return {
                "status": "unhealthy",
                "reason": f"No activity for {elapsed:.1f}s"
            }

        return {
            "status": "healthy",
            "current_state": self.fsm.current_state.name,
            "stack_depth": len(self.fsm.state_stack)
        }

    def record_activity(self):
        """Record FSM activity."""
        self.last_activity = time.time()
```

## Error Handling

### Timeout Handling

States should handle timeouts gracefully:

```python
from mycorrhizal.septum.core import StateConfiguration

@septum.state(config=StateConfiguration(timeout=30.0))
class ExternalAPIState:
    """State with timeout for external API calls."""

    class Events(Enum):
        SUCCESS = auto()
        TIMEOUT = auto()
        RETRY = auto()

    @septum.on_state
    async def on_state(ctx):
        try:
            # External API call with timeout
            result = await call_external_api(timeout=25.0)
            return Events.SUCCESS
        except TimeoutError:
            return Events.TIMEOUT

    @septum.on_timeout
    async def on_timeout(ctx):
        """Called when state exceeds configured timeout."""
        return Events.TIMEOUT

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.SUCCESS, CompletedState),
            LabeledTransition(Events.TIMEOUT, RetryState),
        ]
```

### Retry Logic

Implement retry patterns for transient failures:

```python
@septum.state(config=StateConfiguration(retries=3))
class RetryState:
    """State with retry logic."""

    class Events(Enum):
        SUCCESS = auto()
        FAILURE = auto()
        MAX_RETRIES = auto()

    @septum.on_state
    async def on_state(ctx):
        attempt = ctx.common.get("retry_attempt", 0)
        ctx.common["retry_attempt"] = attempt + 1

        try:
            result = await flaky_operation()
            return Events.SUCCESS
        except TemporaryError:
            return Events.FAILURE

    @septum.on_fail
    async def on_fail(ctx):
        """Called after max retries exhausted."""
        logger.error(f"Operation failed after {ctx.retry_count} attempts")
        return Events.MAX_RETRIES

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.SUCCESS, CompletedState),
            LabeledTransition(Events.FAILURE, Retry),  # Automatic retry
            LabeledTransition(Events.MAX_RETRIES, ErrorState),
        ]
```

### Exception Propagation

Handle exceptions in state handlers:

```python
@septum.state
class SafeState:
    """State with exception handling."""

    class Events(Enum):
        SUCCESS = auto()
        ERROR = auto()

    @septum.on_state
    async def on_state(ctx):
        try:
            result = await risky_operation()
            return Events.SUCCESS
        except ValueError as e:
            # Log and transition to error state
            logger.error(f"Value error: {e}")
            ctx.common["last_error"] = str(e)
            return Events.ERROR
        except Exception as e:
            # Unexpected error
            logger.exception(f"Unexpected error in SafeState: {e}")
            return Events.ERROR

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.SUCCESS, NextState),
            LabeledTransition(Events.ERROR, ErrorState),
        ]
```

## Thread Safety and Concurrency

### Asyncio Concurrency

Septum FSMs are designed for asyncio concurrency:

```python
async def run_concurrent_fsms():
    """Run multiple FSMs concurrently using asyncio."""

    async def run_single_fsm(id):
        fsm = StateMachine(initial_state=MyState)
        await fsm.initialize()
        logger.info(f"FSM {id} started")

        for _ in range(10):
            await fsm.run()

        logger.info(f"FSM {id} completed")
        return id

    # Run 100 FSMs concurrently
    tasks = [run_single_fsm(i) for i in range(100)]
    results = await asyncio.gather(*tasks)

    logger.info(f"All {len(results)} FSMs completed")
```

**Key points:**
- Each FSM instance is independent
- No shared state between FSMs (unless explicitly designed)
- Asyncio handles concurrency automatically
- Scale to thousands of concurrent FSMs

### Thread Safety Considerations

Septum FSMs are **not thread-safe** and should not be shared across threads:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

# BAD: Sharing FSM across threads
fsm = StateMachine(initial_state=MyState)
await fsm.initialize()

with ThreadPoolExecutor() as executor:
    # This will cause issues!
    executor.submit(lambda: asyncio.run(fsm.run()))

# GOOD: Each thread creates its own FSM
def run_in_thread():
    fsm = StateMachine(initial_state=MyState)
    asyncio.run(fsm.initialize())
    asyncio.run(fsm.run())

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(run_in_thread) for _ in range(4)]
```

**Guidelines:**
- Keep FSMs in the asyncio event loop that created them
- Don't share FSM instances across threads
- Use message passing for inter-FSM communication
- Consider separate processes for CPU-bound work

## Resource Management

### Memory Management

Monitor and control memory usage:

```python
import gc
import tracemalloc

class MemoryAwareFSM:
    """FSM with memory monitoring."""

    async def run_with_memory_limit(self, memory_limit_mb=100):
        """Run FSM with memory limit."""
        tracemalloc.start()

        try:
            while True:
                # Check memory before each tick
                current, peak = tracemalloc.get_traced_memory()
                if current > memory_limit_mb * 1024 * 1024:
                    logger.warning(f"Memory limit exceeded: {current / 1024 / 1024:.1f} MB")
                    break

                await self.fsm.run()

                # Periodic GC
                if self.tick_count % 1000 == 0:
                    gc.collect()

        finally:
            tracemalloc.stop()
```

### Connection Pooling

Manage external resources efficiently:

```python
class ConnectionPoolFSM:
    """FSM with connection pooling."""

    def __init__(self, pool_size=10):
        self.db_pool = None
        self.pool_size = pool_size

    async def initialize(self):
        """Initialize connection pool."""
        self.db_pool = await create_db_pool(max_size=self.pool_size)

    @septum.state
    class QueryState:
        @septum.on_state
        async def on_state(ctx):
            # Get connection from pool
            pool = ctx.common["db_pool"]
            async with pool.acquire() as conn:
                result = await conn.fetch_query("SELECT * FROM data")
                ctx.common["query_result"] = result
            return Events.DONE

    async def cleanup(self):
        """Clean up resources."""
        if self.db_pool:
            await self.db_pool.close()
```

## Scaling Strategies

### Vertical Scaling

Optimize single FSM instance:

1. **Profile State Handlers:** Identify slow operations
2. **Async I/O:** Use async libraries for I/O operations
3. **Caching:** Cache expensive computations
4. **Batch Processing:** Batch similar operations

### Horizontal Scaling

Distribute FSMs across processes/machines:

```python
# Process-based scaling
import multiprocessing

def run_fsm_process(id):
    """Run FSM in separate process."""
    fsm = StateMachine(initial_state=MyState)
    asyncio.run(fsm.initialize())
    asyncio.run(fsm.run())

if __name__ == "__main__":
    processes = []
    for i in range(4):  # 4 processes
        p = multiprocessing.Process(target=run_fsm_process, args=(i,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
```

### Message-Based Distribution

Use message queues for distributed FSMs:

```python
import asyncio
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer

async def distributed_fsm_worker():
    """Worker processing FSM commands from Kafka."""

    # Consumer for FSM commands
    consumer = AIOKafkaConsumer(
        'fsm-commands',
        bootstrap_servers='localhost:9092'
    )
    await consumer.start()

    # Producer for FSM events
    producer = AIOKafkaProducer(
        bootstrap_servers='localhost:9092'
    )
    await producer.start()

    # Process messages
    async for msg in consumer:
        command = deserialize(msg.value)

        fsm = StateMachine(initial_state=MyState)
        await fsm.initialize()
        await fsm.run()

        # Publish result
        await producer.send_and_wait(
            'fsm-events',
            serialize(fsm.current_state)
        )
```

## Testing in Production-Like Environments

### Load Testing

```python
import asyncio
import time
from statistics import mean

async def load_test_fsm(num_concurrent=100, num_ticks=10):
    """Load test FSM performance."""

    async def run_fsm(id):
        start = time.time()
        fsm = StateMachine(initial_state=MyState)
        await fsm.initialize()

        for _ in range(num_ticks):
            await fsm.run()

        elapsed = time.time() - start
        return elapsed

    # Run load test
    start_time = time.time()
    results = await asyncio.gather(*[
        run_fsm(i) for i in range(num_concurrent)
    ])
    total_time = time.time() - start_time

    # Report results
    print(f"Load test: {num_concurrent} FSMs × {num_ticks} ticks")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average per FSM: {mean(results):.4f}s")
    print(f"Throughput: {num_concurrent * num_ticks / total_time:.1f} ticks/sec")
```

### Chaos Testing

```python
async def chaos_test_fsm():
    """Test FSM resilience under adverse conditions."""

    async def run_with_random_delays():
        fsm = StateMachine(initial_state=MyState)
        await fsm.initialize()

        for _ in range(100):
            # Random delays to simulate network issues
            if random.random() < 0.1:  # 10% chance
                await asyncio.sleep(random.uniform(0.01, 0.1))

            await fsm.run()

    # Run with chaos
    tasks = [run_with_random_delays() for _ in range(10)]
    await asyncio.gather(*tasks)
```

## Best Practices Summary

**Performance:**
- Profile state handlers before optimization
- Use async I/O for network/disk operations
- Monitor memory usage in production
- Test load with expected concurrency levels

**Reliability:**
- Implement timeout handling for external operations
- Use retry logic for transient failures
- Log state transitions for debugging
- Implement health checks

**Scalability:**
- Each FSM instance is independent
- Use asyncio for single-process concurrency
- Use multiprocessing/messaging for multi-host scaling
- Monitor resource usage per FSM

**Monitoring:**
- Track state transition rates
- Monitor error rates and types
- Measure time-in-state for bottlenecks
- Set up alerts for anomalies

## See Also

- [API Reference](../api/septum.md) - Complete API documentation
- [Best Practices](../guides/best-practices.md) - Design patterns and anti-patterns
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
- [PDA Guide](../guides/septum-pda-guide.md) - Hierarchical state machines
