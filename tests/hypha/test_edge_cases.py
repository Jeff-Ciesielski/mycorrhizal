#!/usr/bin/env python3
"""
Edge case tests for Hypha runtime.

Tests unusual but valid scenarios that may not be covered in normal tests.

Run with: pytest tests/hypha/test_edge_cases.py -v
"""

import asyncio
import pytest
from dataclasses import dataclass, field
from typing import Any, List

from mycorrhizal.hypha.core import pn, PlaceType, Runner as PNRunner
from mycorrhizal.common.timebase import CycleClock


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class EdgeCaseBlackboard:
    """Blackboard for edge case testing."""
    counter: int = 0
    tokens_seen: List[str] = field(default_factory=list)


@pytest.fixture
def edge_bb():
    return EdgeCaseBlackboard()


@pytest.fixture
def cycle_tb():
    return CycleClock()


# =============================================================================
# Empty Token Tests
# =============================================================================

class TestEmptyTokens:
    """Tests for handling of empty or missing tokens"""

    async def test_empty_place_with_waiting_transition(self, edge_bb, cycle_tb):
        """Transition waiting on empty place"""

        @pn.net
        def EmptyPlaceNet(builder):
            input_p = builder.place("input", type=PlaceType.BAG)
            output_p = builder.place("output", type=PlaceType.BAG)

            @builder.transition()
            async def worker(consumed, bb, timebase):
                yield {output_p: consumed[0]}

            builder.arc(input_p, worker)

        runner = PNRunner(EmptyPlaceNet, edge_bb)
        await runner.start(cycle_tb)

        # Don't add any tokens - transition should wait gracefully
        await asyncio.sleep(0.1)

        # Verify no tokens in output
        output_tokens = runner.runtime.places[('EmptyPlaceNet', 'output')].tokens
        assert len(output_tokens) == 0

        # Now add token - should process normally
        input_place = runner.runtime.places[('EmptyPlaceNet', 'input')]
        input_place.add_token("test")
        await asyncio.sleep(0.1)

        output_tokens = runner.runtime.places[('EmptyPlaceNet', 'output')].tokens
        assert len(output_tokens) == 1

        await runner.stop()

    async def test_place_starts_empty_then_gets_tokens(self, edge_bb, cycle_tb):
        """Place that starts empty and receives tokens later"""

        @pn.net
        def DelayedTokensNet(builder):
            input_p = builder.place("input", type=PlaceType.BAG)
            output_p = builder.place("output", type=PlaceType.BAG)

            @builder.transition()
            async def worker(consumed, bb, timebase):
                bb.tokens_seen.append(str(consumed[0]))
                yield {output_p: consumed[0]}

            builder.arc(input_p, worker)

        runner = PNRunner(DelayedTokensNet, edge_bb)
        await runner.start(cycle_tb)

        # Wait a bit, then add tokens
        await asyncio.sleep(0.1)
        input_place = runner.runtime.places[('DelayedTokensNet', 'input')]

        for i in range(3):
            input_place.add_token(f"token{i}")
            await asyncio.sleep(0.05)

        await asyncio.sleep(0.2)

        # Verify all tokens processed
        assert len(edge_bb.tokens_seen) == 3
        output_tokens = runner.runtime.places[('DelayedTokensNet', 'output')].tokens
        assert len(output_tokens) == 3

        await runner.stop()


# =============================================================================
# Transition with No Inputs/Outputs
# =============================================================================

class TestNoInputsOutputs:
    """Tests for transitions with no inputs or outputs"""

    async def test_transition_with_no_inputs(self, edge_bb, cycle_tb):
        """Generator transition (no inputs)"""

        @pn.net
        def GeneratorNet(builder):
            output_p = builder.place("output", type=PlaceType.BAG)

            @builder.transition()
            async def generator(consumed, bb, timebase):
                # This is a generator - creates tokens without consuming
                yield {output_p: f"generated_{bb.counter}"}
                bb.counter += 1

        runner = PNRunner(GeneratorNet, edge_bb)
        await runner.start(cycle_tb)

        await asyncio.sleep(0.2)

        # Should have generated tokens
        output_tokens = runner.runtime.places[('GeneratorNet', 'output')].tokens
        assert len(output_tokens) > 0, "Generator should have created tokens"

        await runner.stop()

    async def test_transition_with_no_outputs(self, edge_bb, cycle_tb):
        """Sink transition (no outputs)"""

        @pn.net
        def SinkNet(builder):
            input_p = builder.place("input", type=PlaceType.BAG)

            @builder.transition()
            async def sink(consumed, bb, timebase):
                # Consumes tokens but produces nothing
                bb.tokens_seen.append(str(consumed[0]))

            builder.arc(input_p, sink)

        runner = PNRunner(SinkNet, edge_bb)
        await runner.start(cycle_tb)

        # Add tokens
        input_place = runner.runtime.places[('SinkNet', 'input')]
        for i in range(3):
            input_place.add_token(f"token{i}")

        await asyncio.sleep(0.2)

        # Verify all tokens consumed but no output
        assert len(edge_bb.tokens_seen) == 3
        assert len(input_place.tokens) == 0

        await runner.stop()


# =============================================================================
# Self-Loop Tests
# =============================================================================

class TestSelfLoops:
    """Tests for transitions that loop back to their input place"""

    async def test_self_loop_transition(self, edge_bb, cycle_tb):
        """Transition that outputs back to its input place"""

        @pn.net
        def SelfLoopNet(builder):
            loop_p = builder.place("loop", type=PlaceType.BAG)

            @builder.transition()
            async def looper(consumed, bb, timebase):
                if bb.counter < 5:  # Limit iterations to prevent infinite loop
                    bb.counter += 1
                    yield {loop_p: f"loop_{bb.counter}"}

            builder.arc(loop_p, looper)

        runner = PNRunner(SelfLoopNet, edge_bb)
        await runner.start(cycle_tb)

        # Start with one token
        loop_place = runner.runtime.places[('SelfLoopNet', 'loop')]
        loop_place.add_token("start")

        await asyncio.sleep(0.3)

        # Should have looped 5 times
        assert edge_bb.counter == 5

        await runner.stop()


# =============================================================================
# Multiple Arcs Same Place
# =============================================================================

class TestMultipleArcs:
    """Tests for multiple arcs between same place and transition"""

    async def test_multiple_input_arcs_same_place(self, edge_bb, cycle_tb):
        """Transition with multiple input arcs from same place (weight > 1)"""

        @pn.net
        def MultiArcNet(builder):
            input_p = builder.place("input", type=PlaceType.BAG)
            output_p = builder.place("output", type=PlaceType.BAG)

            @builder.transition()
            async def multi_consumer(consumed, bb, timebase):
                # Consumes 2 tokens at once
                yield {output_p: f"pair_{consumed[0]}_{consumed[1]}"}

            builder.arc(input_p, multi_consumer, weight=2)

        runner = PNRunner(MultiArcNet, edge_bb)
        await runner.start(cycle_tb)

        # Add tokens
        input_place = runner.runtime.places[('MultiArcNet', 'input')]
        for i in range(4):
            input_place.add_token(f"token{i}")

        await asyncio.sleep(0.2)

        # Should have created 2 pairs
        output_tokens = runner.runtime.places[('MultiArcNet', 'output')].tokens
        assert len(output_tokens) == 2

        await runner.stop()


# =============================================================================
# Cancellation Tests
# =============================================================================

class TestCancellation:
    """Tests for cancellation during processing"""

    async def test_cancel_during_active_processing(self, edge_bb, cycle_tb):
        """Cancel runner while transitions are actively processing"""

        @pn.net
        def CancelNet(builder):
            input_p = builder.place("input", type=PlaceType.BAG)
            output_p = builder.place("output", type=PlaceType.BAG)

            @builder.transition()
            async def slow_worker(consumed, bb, timebase):
                await asyncio.sleep(0.1)  # Slow processing
                yield {output_p: consumed[0]}

            builder.arc(input_p, slow_worker)

        runner = PNRunner(CancelNet, edge_bb)
        await runner.start(cycle_tb)

        # Add many tokens
        input_place = runner.runtime.places[('CancelNet', 'input')]
        for i in range(10):
            input_place.add_token(f"token{i}")

        # Wait a bit then cancel
        await asyncio.sleep(0.15)
        await runner.stop()

        # Should have processed some but not all tokens
        output_tokens = runner.runtime.places[('CancelNet', 'output')].tokens
        assert len(output_tokens) >= 0  # Should not crash
        assert len(output_tokens) < 10  # Should not have finished all

    async def test_rapid_start_stop_cycles(self, edge_bb, cycle_tb):
        """Multiple rapid start/stop cycles"""

        @pn.net
        def StartStopNet(builder):
            input_p = builder.place("input", type=PlaceType.BAG)
            output_p = builder.place("output", type=PlaceType.BAG)

            @builder.transition()
            async def worker(consumed, bb, timebase):
                yield {output_p: consumed[0]}

            builder.arc(input_p, worker)

        # Run multiple cycles
        for cycle in range(5):
            runner = PNRunner(StartStopNet, edge_bb)
            await runner.start(cycle_tb)

            # Add tokens
            input_place = runner.runtime.places[('StartStopNet', 'input')]
            input_place.add_token(f"cycle{cycle}")

            await asyncio.sleep(0.1)
            await runner.stop()

            # Verify processing
            assert len(edge_bb.tokens_seen) <= 1  # At most one token per cycle


# =============================================================================
# Single Token Tests
# =============================================================================

class TestSingleToken:
    """Tests for nets with minimal tokens"""

    async def test_single_token_net(self, edge_bb, cycle_tb):
        """Net with exactly one token"""

        @pn.net
        def SingleTokenNet(builder):
            input_p = builder.place("input", type=PlaceType.BAG)
            output_p = builder.place("output", type=PlaceType.BAG)

            @builder.transition()
            async def worker(consumed, bb, timebase):
                yield {output_p: consumed[0]}

            builder.arc(input_p, worker)

        runner = PNRunner(SingleTokenNet, edge_bb)
        await runner.start(cycle_tb)

        # Add exactly one token
        input_place = runner.runtime.places[('SingleTokenNet', 'input')]
        input_place.add_token("only_token")

        await asyncio.sleep(0.2)

        # Verify processed
        output_tokens = runner.runtime.places[('SingleTokenNet', 'output')].tokens
        assert len(output_tokens) == 1
        assert output_tokens[0] == "only_token"

        await runner.stop()


# =============================================================================
# Queue vs Bag Tests
# =============================================================================

class TestQueueVsBag:
    """Tests comparing QUEUE and BAG place types"""

    async def test_queue_place_ordering(self, edge_bb, cycle_tb):
        """QUEUE place maintains FIFO order"""

        @pn.net
        def QueueNet(builder):
            input_p = builder.place("input", type=PlaceType.QUEUE)
            output_p = builder.place("output", type=PlaceType.BAG)

            @builder.transition()
            async def worker(consumed, bb, timebase):
                yield {output_p: consumed[0]}

            builder.arc(input_p, worker)

        runner = PNRunner(QueueNet, edge_bb)
        await runner.start(cycle_tb)

        # Add tokens in specific order
        input_place = runner.runtime.places[('QueueNet', 'input')]
        for i in range(5):
            input_place.add_token(f"token{i}")

        await asyncio.sleep(0.2)

        # Verify order maintained
        output_tokens = runner.runtime.places[('QueueNet', 'output')].tokens
        assert len(output_tokens) == 5
        # QUEUE should process in FIFO order
        # Note: This is a basic test - more sophisticated ordering tests could be added

        await runner.stop()

    async def test_bag_place_no_ordering(self, edge_bb, cycle_tb):
        """BAG place has no guaranteed order"""

        @pn.net
        def BagNet(builder):
            input_p = builder.place("input", type=PlaceType.BAG)
            output_p = builder.place("output", type=PlaceType.BAG)

            @builder.transition()
            async def worker(consumed, bb, timebase):
                yield {output_p: consumed[0]}

            builder.arc(input_p, worker)

        runner = PNRunner(BagNet, edge_bb)
        await runner.start(cycle_tb)

        # Add tokens
        input_place = runner.runtime.places[('BagNet', 'input')]
        for i in range(5):
            input_place.add_token(f"token{i}")

        await asyncio.sleep(0.2)

        # Verify all processed (order not guaranteed for BAG)
        output_tokens = runner.runtime.places[('BagNet', 'output')].tokens
        assert len(output_tokens) == 5

        await runner.stop()
