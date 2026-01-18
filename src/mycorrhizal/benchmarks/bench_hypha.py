"""
Throughput benchmarks for Hypha (Petri net DSL)

Measures tokens processed per second across various scenarios:
- Simple single-transition processing
- Multi-transition routing
- Concurrent worker transitions
- Nested subnet structures
"""

import asyncio
import pytest
from dataclasses import dataclass
from typing import Any

from mycorrhizal.hypha.core import pn, PlaceType, Runner
from mycorrhizal.hypha.core.builder import NetBuilder
from mycorrhizal.common.timebase import MonotonicClock, CycleClock


# ============================================================================
# Fixtures and Helpers
# ============================================================================

class SimpleBlackboard:
    """Blackboard for benchmarks"""
    def __init__(self):
        self.processed_count = 0


@dataclass
class Token:
    """Simple token for benchmarks"""
    id: int
    value: int


# ============================================================================
# Benchmark Nets
# ============================================================================

@pn.net
def SimpleProcessingNet(builder: NetBuilder):
    """Simple net: input -> transition -> output

    Measures baseline token processing throughput.
    """

    input_place = builder.place("input", type=PlaceType.BAG)
    output_place = builder.place("output", type=PlaceType.BAG)

    @builder.transition()
    async def process(consumed, bb: SimpleBlackboard, timebase):
        """Simple identity transition"""
        for token in consumed:
            bb.processed_count += 1
            yield {output_place: token}

    builder.arc(input_place, process)
    builder.arc(process, output_place)


@pn.net
def RoutingNet(builder: NetBuilder):
    """Routing net: input -> transition -> [output_a, output_b]

    Measures throughput with conditional routing.
    """

    input_place = builder.place("input", type=PlaceType.BAG)
    output_a = builder.place("output_a", type=PlaceType.BAG)
    output_b = builder.place("output_b", type=PlaceType.BAG)

    @builder.transition()
    async def route(consumed, bb: SimpleBlackboard, timebase):
        """Route even tokens to output_a, odd tokens to output_b"""
        even_tokens = [t for t in consumed if t.id % 2 == 0]
        odd_tokens = [t for t in consumed if t.id % 2 == 1]

        for token in even_tokens:
            bb.processed_count += 1
            yield {output_a: token}
        for token in odd_tokens:
            bb.processed_count += 1
            yield {output_b: token}

    builder.arc(input_place, route)
    builder.arc(route, output_a)
    builder.arc(route, output_b)


@pn.net
def MultiStageNet(builder: NetBuilder):
    """Multi-stage pipeline: stage1 -> stage2 -> stage3 -> output

    Measures throughput through a processing pipeline.
    """

    input_q = builder.place("input", type=PlaceType.QUEUE)
    stage1 = builder.place("stage1", type=PlaceType.BAG)
    stage2 = builder.place("stage2", type=PlaceType.BAG)
    stage3 = builder.place("stage3", type=PlaceType.BAG)
    output = builder.place("output", type=PlaceType.BAG)

    @builder.transition()
    async def stage1_process(consumed, bb: SimpleBlackboard, timebase):
        for t in consumed:
            bb.processed_count += 1
            yield {stage1: t}

    @builder.transition()
    async def stage2_process(consumed, bb: SimpleBlackboard, timebase):
        for t in consumed:
            bb.processed_count += 1
            yield {stage2: t}

    @builder.transition()
    async def stage3_process(consumed, bb: SimpleBlackboard, timebase):
        for t in consumed:
            bb.processed_count += 1
            yield {stage3: t}

    @builder.transition()
    async def output_process(consumed, bb: SimpleBlackboard, timebase):
        for t in consumed:
            bb.processed_count += 1
            yield {output: t}

    builder.arc(input_q, stage1_process)
    builder.arc(stage1, stage2_process)
    builder.arc(stage2, stage3_process)
    builder.arc(stage3, output_process)
    builder.arc(output_process, output)


@pn.net
def WorkerPoolNet(builder: NetBuilder):
    """Worker pool: input -> [worker1, worker2, worker3] -> output

    Measures throughput with concurrent processing transitions.
    """

    input_place = builder.place("input", type=PlaceType.QUEUE)
    output_place = builder.place("output", type=PlaceType.BAG)

    @builder.transition()
    async def worker1(consumed, bb: SimpleBlackboard, timebase):
        for t in consumed:
            bb.processed_count += 1
            yield {output_place: t}

    @builder.transition()
    async def worker2(consumed, bb: SimpleBlackboard, timebase):
        for t in consumed:
            bb.processed_count += 1
            yield {output_place: t}

    @builder.transition()
    async def worker3(consumed, bb: SimpleBlackboard, timebase):
        for t in consumed:
            bb.processed_count += 1
            yield {output_place: t}

    builder.arc(input_place, worker1)
    builder.arc(input_place, worker2)
    builder.arc(input_place, worker3)



# ============================================================================
# Benchmarks
# ============================================================================

@pytest.mark.hypha
@pytest.mark.benchmark(group="hypha-simple")
def test_simple_token_throughput(benchmark):
    """Benchmark simple token processing (single transition)

    Measures: tokens processed per second through basic transition
    Scenario: 100 tokens through input -> process -> output
    Expected: Baseline throughput metric
    """
    NUM_TOKENS = 100

    def run_net():
        async def _run():
            bb = SimpleBlackboard()
            timebase = MonotonicClock()

            runner = Runner(SimpleProcessingNet, bb)
            await runner.start(timebase)

            # Add tokens gradually to allow transition to fire multiple times
            input_place = runner.runtime.places[("SimpleProcessingNet", "input")]
            for i in range(NUM_TOKENS):
                input_place.add_token(Token(id=i, value=i))
                await asyncio.sleep(0.001)  # Small sleep to let transition process

            # Wait for all tokens to be processed (checking blackboard)
            timeout_count = 0
            max_timeout = 200  # 2 seconds should be enough
            while bb.processed_count < NUM_TOKENS and timeout_count < max_timeout:
                await asyncio.sleep(0.01)
                timeout_count += 1

            await runner.stop(timeout=5)

            return bb.processed_count

        return asyncio.run(_run())

    result = benchmark(run_net)
    assert result == NUM_TOKENS


@pytest.mark.hypha
@pytest.mark.benchmark(group="hypha-routing")
def test_routing_throughput(benchmark):
    """Benchmark token routing (multiple transitions)

    Measures: tokens processed per second through conditional routing
    Scenario: 100 tokens split across two processing paths
    Expected: ~50% throughput reduction vs simple case
    """
    NUM_TOKENS = 100

    def run_net():
        async def _run():
            bb = SimpleBlackboard()
            timebase = MonotonicClock()

            runner = Runner(RoutingNet, bb)
            await runner.start(timebase)

            # Add tokens gradually to allow transition to fire multiple times
            input_place = runner.runtime.places[("RoutingNet", "input")]
            for i in range(NUM_TOKENS):
                input_place.add_token(Token(id=i, value=i))
                await asyncio.sleep(0.001)  # Small sleep to let transition process

            # Wait for all tokens to be processed (checking blackboard)
            timeout_count = 0
            max_timeout = 200  # 2 seconds should be enough
            while bb.processed_count < NUM_TOKENS and timeout_count < max_timeout:
                await asyncio.sleep(0.01)
                timeout_count += 1

            await runner.stop(timeout=5)

            return bb.processed_count

        return asyncio.run(_run())

    result = benchmark(run_net)
    assert result == NUM_TOKENS


@pytest.mark.hypha
@pytest.mark.benchmark(group="hypha-pipeline")
def test_pipeline_throughput(benchmark):
    """Benchmark multi-stage pipeline

    Measures: tokens processed per second through 4-stage pipeline
    Scenario: 100 tokens through stage1 -> stage2 -> stage3 -> output
    Expected: Each token processed 4 times (count = 400)
    """
    NUM_TOKENS = 100

    def run_net():
        async def _run():
            bb = SimpleBlackboard()
            timebase = MonotonicClock()

            runner = Runner(MultiStageNet, bb)
            await runner.start(timebase)

            # Add tokens gradually to allow transition to fire multiple times
            input_place = runner.runtime.places[("MultiStageNet", "input")]
            for i in range(NUM_TOKENS):
                input_place.add_token(Token(id=i, value=i))

            # Wait for all tokens to be processed through all stages
            timeout_count = 0
            max_timeout = 200  # 2 seconds for 4-stage pipeline
            while bb.processed_count < NUM_TOKENS * 4 and timeout_count < max_timeout:
                await asyncio.sleep(0.01)
                timeout_count += 1

            await runner.stop(timeout=5)

            return bb.processed_count

        return asyncio.run(_run())

    result = benchmark(run_net)
    assert result == NUM_TOKENS * 4  # Each token goes through 4 stages


@pytest.mark.hypha
@pytest.mark.benchmark(group="hypha-workers")
def test_worker_pool_throughput(benchmark):
    """Benchmark worker pool (concurrent transitions)

    Measures: tokens processed per second with multiple worker transitions
    Scenario: 100 tokens across 3 competing worker transitions
    Expected: Similar or better throughput than simple case
    """
    NUM_TOKENS = 100

    def run_net():
        async def _run():
            bb = SimpleBlackboard()
            timebase = MonotonicClock()

            runner = Runner(WorkerPoolNet, bb)
            await runner.start(timebase)

            # Add tokens gradually to allow transition to fire multiple times
            input_place = runner.runtime.places[("WorkerPoolNet", "input")]
            for i in range(NUM_TOKENS):
                input_place.add_token(Token(id=i, value=i))

            # Wait for all tokens to be processed (checking blackboard)
            timeout_count = 0
            max_timeout = 200  # 2 seconds should be enough
            while bb.processed_count < NUM_TOKENS and timeout_count < max_timeout:
                await asyncio.sleep(0.01)
                timeout_count += 1

            await runner.stop(timeout=5)

            return bb.processed_count

        return asyncio.run(_run())

    result = benchmark(run_net)
    assert result == NUM_TOKENS


@pytest.mark.hypha
@pytest.mark.benchmark(group="hypha-scalability")
@pytest.mark.parametrize("num_tokens", [10, 50, 100, 500])
def test_scalability_different_loads(benchmark, num_tokens):
    """Benchmark scalability with different token loads

    Measures: how throughput scales with token count
    Scenario: Varying numbers of tokens through simple net
    Expected: Linear scaling (2x tokens = 2x time)
    """
    def run_net():
        async def _run():
            bb = SimpleBlackboard()
            timebase = MonotonicClock()

            runner = Runner(SimpleProcessingNet, bb)
            await runner.start(timebase)

            # Add tokens gradually to allow transition to fire multiple times
            input_place = runner.runtime.places[("SimpleProcessingNet", "input")]
            for i in range(num_tokens):
                input_place.add_token(Token(id=i, value=i))

            # Wait for all tokens to be processed (checking blackboard)
            timeout_count = 0
            max_timeout = 100 + (num_tokens // 10)  # Scale timeout with token count
            while bb.processed_count < num_tokens and timeout_count < max_timeout:
                await asyncio.sleep(0.01)
                timeout_count += 1

            await runner.stop(timeout=5)

            return bb.processed_count

        return asyncio.run(_run())

    result = benchmark(run_net)
    assert result == num_tokens
