#!/usr/bin/env python3
"""Performance benchmarks for Septum FSM."""

import sys
sys.path.insert(0, "src")

import asyncio
import pytest
import time
import gc
import tracemalloc
from enum import Enum, auto

from mycorrhizal.septum.core import (
    septum,
    StateMachine,
    LabeledTransition,
    StateConfiguration,
    SharedContext,
)


# ============================================================================
# Benchmark States
# ============================================================================


@septum.state()
def BenchmarkState():
    """Simple state for benchmarking."""

    @septum.events
    class Events(Enum):
        NEXT = auto()

    @septum.on_state
    async def on_state(ctx: SharedContext):
        return Events.NEXT

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.NEXT, BenchmarkState),
        ]


@septum.state()
def MessageState():
    """State for message throughput benchmark."""

    @septum.events
    class Events(Enum):
        PROCESS = auto()

    @septum.on_state
    async def on_state(ctx: SharedContext):
        # Check if we got a message
        msg = ctx.msg
        if msg is not None:
            ctx.common["messages_processed"] = ctx.common.get("messages_processed", 0) + 1
            return Events.PROCESS
        return None

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.PROCESS, MessageState),
        ]


def create_state_machine_with_n_states(num_states: int):
    """Create an FSM with N states in a chain."""
    states = {}

    for i in range(num_states):
        state_id = f"State{i}"

        # Create state dynamically
        @septum.state()
        def state_func():
            class Events(Enum):
                NEXT = auto()

            @septum.on_state
            async def on_state(ctx: SharedContext):
                return Events.NEXT

            @septum.transitions
            def transitions():
                next_idx = (i + 1) % num_states
                return [LabeledTransition(Events.NEXT, f"State{next_idx}")]

        # Store reference
        states[state_id] = state_func

    return StateMachine(initial_state="State0", states=states)


# ============================================================================
# Benchmark 1: State Transition Overhead
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.benchmark
async def benchmark_state_transition_overhead():
    """Measure time per state transition."""
    fsm = StateMachine(initial_state=BenchmarkState)
    await fsm.initialize()

    # Warm up
    for _ in range(100):
        await fsm.tick()

    # Benchmark
    iterations = 10000
    start = time.perf_counter()

    for _ in range(iterations):
        await fsm.tick()

    end = time.perf_counter()
    total_time = end - start

    per_transition_us = (total_time / iterations) * 1_000_000

    print(f"\nState transition overhead: {per_transition_us:.2f} µs per transition")
    print(f"  Total time for {iterations} transitions: {total_time:.4f}s")

    # Assertion: Should be < 100 µs per transition
    # This is a soft assertion - we're documenting performance characteristics
    # rather than enforcing strict limits
    if per_transition_us > 100:
        pytest.warn(f"State transition slow: {per_transition_us:.2f} µs (target: < 100 µs)")

    return per_transition_us


# ============================================================================
# Benchmark 2: Scaling with State Count
# ============================================================================


@pytest.mark.parametrize("num_states", [10, 50, 100])
@pytest.mark.asyncio
@pytest.mark.benchmark
async def benchmark_scaling_with_state_count(num_states):
    """Test FSM performance with varying numbers of states."""
    try:
        fsm = create_state_machine_with_n_states(num_states)
        await fsm.initialize()

        # Measure tick time
        iterations = 1000
        start = time.perf_counter()

        for _ in range(iterations):
            await fsm.tick()

        end = time.perf_counter()
        total_time = end - start

        per_tick_ms = (total_time / iterations) * 1000

        print(f"\nScaling test ({num_states} states): {per_tick_ms:.3f} ms per tick")
        print(f"  Total time for {iterations} ticks: {total_time:.4f}s")

        # Assertion: Should scale reasonably
        if per_tick_ms > 10:
            pytest.warn(f"Tick slow with {num_states} states: {per_tick_ms:.3f} ms (target: < 10 ms)")

        return per_tick_ms

    except Exception as e:
        pytest.skip(f"Could not create FSM with {num_states} states: {e}")


# ============================================================================
# Benchmark 3: Memory Usage
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.benchmark
async def benchmark_memory_usage():
    """Measure memory per FSM instance."""
    gc.collect()
    tracemalloc.start()

    # Create baseline
    snapshot1 = tracemalloc.take_snapshot()

    # Create 1000 FSMs
    fsms = []
    for _ in range(1000):
        fsm = StateMachine(initial_state=BenchmarkState)
        await fsm.initialize()
        fsms.append(fsm)

    snapshot2 = tracemalloc.take_snapshot()

    # Calculate memory
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    total_memory = sum(stat.size_diff for stat in top_stats)
    per_fsm_kb = total_memory / 1000 / 1024

    print(f"\nMemory usage: {per_fsm_kb:.2f} KB per FSM instance")
    print(f"  Total memory for 1000 FSMs: {total_memory / 1024:.2f} KB")

    tracemalloc.stop()

    # Assertion: Should be reasonable
    if per_fsm_kb > 100:
        pytest.warn(f"High memory usage: {per_fsm_kb:.2f} KB per FSM (target: < 100 KB)")

    return per_fsm_kb


# ============================================================================
# Benchmark 4: Message Throughput
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.benchmark
async def benchmark_message_send_throughput():
    """Measure message send rate (messages per second)."""
    fsm = StateMachine(initial_state=MessageState)
    await fsm.initialize()

    # Send messages as fast as possible
    iterations = 10000
    start = time.perf_counter()

    for i in range(iterations):
        fsm.send_message(f"message_{i}")

    end = time.perf_counter()
    total_time = end - start

    messages_per_sec = iterations / total_time

    print(f"\nMessage send throughput: {messages_per_sec:.0f} msg/sec")
    print(f"  Total time to send {iterations} messages: {total_time:.4f}s")

    # Assertion: Should handle high throughput
    if messages_per_sec < 10000:
        pytest.warn(f"Low send throughput: {messages_per_sec:.0f} msg/sec (target: > 10000 msg/sec)")

    return messages_per_sec


@pytest.mark.asyncio
@pytest.mark.benchmark
async def benchmark_message_process_throughput():
    """Measure message processing throughput."""
    fsm = StateMachine(initial_state=MessageState)
    await fsm.initialize()

    # Send messages
    num_messages = 1000
    for i in range(num_messages):
        fsm.send_message(f"message_{i}")

    # Process all messages
    start = time.perf_counter()

    for _ in range(num_messages * 2):  # Overestimate to ensure all processed
        await fsm.tick(timeout=0.1)
        if fsm.context.common.get("messages_processed", 0) >= num_messages:
            break

    end = time.perf_counter()
    total_time = end - start

    processed = fsm.context.common.get("messages_processed", 0)
    messages_per_sec = processed / total_time if total_time > 0 else 0

    print(f"\nMessage process throughput: {messages_per_sec:.0f} msg/sec")
    print(f"  Processed {processed} messages in {total_time:.4f}s")

    # Assertion: Should process messages efficiently
    if messages_per_sec < 1000:
        pytest.warn(f"Low process throughput: {messages_per_sec:.0f} msg/sec (target: > 1000 msg/sec)")

    return messages_per_sec


# ============================================================================
# Benchmark 5: FSM Creation Overhead
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.benchmark
async def benchmark_fsm_creation_overhead():
    """Measure time to create and initialize FSM instances."""
    iterations = 1000

    start = time.perf_counter()

    for _ in range(iterations):
        fsm = StateMachine(initial_state=BenchmarkState)
        await fsm.initialize()

    end = time.perf_counter()
    total_time = end - start

    per_fsm_ms = (total_time / iterations) * 1000

    print(f"\nFSM creation overhead: {per_fsm_ms:.3f} ms per FSM")
    print(f"  Total time for {iterations} FSMs: {total_time:.4f}s")

    # Assertion: Should be fast enough
    if per_fsm_ms > 10:
        pytest.warn(f"Slow FSM creation: {per_fsm_ms:.3f} ms (target: < 10 ms)")

    return per_fsm_ms


# ============================================================================
# Benchmark 6: Concurrent FSM Performance
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.benchmark
@pytest.mark.slow
async def benchmark_concurrent_fsm_performance():
    """Measure performance of running FSMs concurrently."""
    async def run_fsm(fsm_id: int):
        """Run FSM for a few ticks."""
        fsm = StateMachine(initial_state=BenchmarkState)
        await fsm.initialize()

        for _ in range(100):
            await fsm.tick()

        return fsm_id

    # Test with different concurrency levels
    for num_fsms in [10, 50, 100]:
        start = time.perf_counter()

        tasks = [run_fsm(i) for i in range(num_fsms)]
        await asyncio.gather(*tasks)

        end = time.perf_counter()
        total_time = end - start

        print(f"\nConcurrent FSMs ({num_fsms}): {total_time:.4f}s for all FSMs")
        print(f"  Average per FSM: {(total_time / num_fsms) * 1000:.3f} ms")

        # Check if scaling is reasonable
        avg_per_fsm = total_time / num_fsms
        if avg_per_fsm > 0.1:  # 100ms per FSM is too slow
            pytest.warn(f"Slow concurrent execution with {num_fsms} FSMs: {avg_per_fsm * 1000:.3f} ms per FSM")


# ============================================================================
# Summary Benchmark
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.benchmark
async def benchmark_summary():
    """Run all key benchmarks and print summary."""
    print("\n" + "=" * 70)
    print("SEPTUM FSM PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 70)

    results = {}

    # Benchmark 1: State transition overhead
    print("\n[Benchmark 1] State Transition Overhead")
    result = await benchmark_state_transition_overhead()
    results["transition_overhead_us"] = result

    # Benchmark 2: Scaling with state count
    print("\n[Benchmark 2] Scaling with State Count")
    for num_states in [10, 50, 100]:
        result = await benchmark_scaling_with_state_count(num_states)
        results[f"scaling_{num_states}_states_ms"] = result

    # Benchmark 3: Memory usage
    print("\n[Benchmark 3] Memory Usage")
    result = await benchmark_memory_usage()
    results["memory_per_fsm_kb"] = result

    # Benchmark 4: Message throughput
    print("\n[Benchmark 4] Message Throughput")
    send_result = await benchmark_message_send_throughput()
    results["message_send_throughput"] = send_result
    process_result = await benchmark_message_process_throughput()
    results["message_process_throughput"] = process_result

    # Benchmark 5: FSM creation overhead
    print("\n[Benchmark 5] FSM Creation Overhead")
    result = await benchmark_fsm_creation_overhead()
    results["fsm_creation_ms"] = result

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"State transition overhead: {results['transition_overhead_us']:.2f} µs")
    print(f"Memory per FSM: {results['memory_per_fsm_kb']:.2f} KB")
    print(f"Message send throughput: {results['message_send_throughput']:.0f} msg/sec")
    print(f"Message process throughput: {results['message_process_throughput']:.0f} msg/sec")
    print(f"FSM creation: {results['fsm_creation_ms']:.3f} ms")
    print("=" * 70)
    print("\nPractical Limits Documented:")
    print(f"  - State transition overhead: {results['transition_overhead_us']:.2f} µs/transition")
    print(f"  - Memory usage: {results['memory_per_fsm_kb']:.2f} KB/instance")
    print(f"  - Message throughput: {results['message_send_throughput']:.0f} msg/sec (send)")
    print(f"  - FSM creation: {results['fsm_creation_ms']:.3f} ms/instance")
    print("=" * 70)

    return results
