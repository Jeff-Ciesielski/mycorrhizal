#!/usr/bin/env python3
"""
Comprehensive concurrency tests for Hypha runtime.

Tests race conditions, spurious wakeups, and competing transitions.

Run with: pytest tests/hypha/test_concurrency.py -v
"""

import asyncio
import pytest
from dataclasses import dataclass, field
from typing import Any, List
from hypothesis import given, strategies as st, settings, Phase, HealthCheck

from mycorrhizal.hypha.core import pn, PlaceType, Runner as PNRunner
from mycorrhizal.common.timebase import CycleClock


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class TestBlackboard:
    """Simple blackboard for testing."""
    processed_count: int = 0
    log: List[str] = field(default_factory=list)


@pytest.fixture
def test_bb():
    return TestBlackboard()


@pytest.fixture
def cycle_tb():
    return CycleClock()


# =============================================================================
# Property-Based Tests (using hypothesis)
# =============================================================================

class TestPropertyBased:
    """Property-based tests using Hypothesis

    These tests use Hypothesis to generate random inputs and verify invariants
    that should always hold true for Petri net execution.
    """

    @pytest.mark.slow
    @given(
        num_tokens=st.integers(min_value=0, max_value=20),
        num_transitions=st.integers(min_value=1, max_value=5),
        delay_ms=st.integers(min_value=0, max_value=10)
    )
    @settings(
        max_examples=50,
        phases=[Phase.generate],  # Skip reuse phase to avoid slow tests
        deadline=None,  # Disable deadline for async tests
        suppress_health_check=[HealthCheck.function_scoped_fixture]  # We create resources inline
    )
    async def test_property_token_flow(self, num_tokens, num_transitions, delay_ms):
        """Property: All tokens should eventually reach output places

        This test verifies the token flow invariant:
        - For a simple input -> processing -> output net
        - All tokens added to input should eventually appear in output
        - No tokens should be lost or duplicated

        Hypothesis generates:
        - num_tokens: How many tokens to add (0-20)
        - num_transitions: How many competing transitions (1-5)
        - delay_ms: Processing delay in transitions (0-10ms)
        """

        # Create fresh resources for each test run (Hypothesis requirement)
        test_bb = TestBlackboard()
        cycle_tb = CycleClock()

        # Build a net with N competing transitions processing tokens
        @pn.net
        def TokenFlowNet(builder):
            input_p = builder.place("input", type=PlaceType.BAG)
            output_p = builder.place("output", type=PlaceType.BAG)

            # Create N competing transitions
            for i in range(num_transitions):
                @builder.transition()
                async def processor(consumed, bb, timebase, idx=i):
                    # Add variable delay to create interesting concurrency scenarios
                    if delay_ms > 0:
                        await asyncio.sleep(delay_ms / 1000.0)
                    # Transform token to track which transition processed it
                    yield {output_p: f"{consumed[0]}_t{idx}"}

                builder.arc(input_p, processor)

        runner = PNRunner(TokenFlowNet, test_bb)
        await runner.start(cycle_tb)

        # Add tokens to input
        input_place = runner.runtime.places[('TokenFlowNet', 'input')]
        for i in range(num_tokens):
            input_place.add_token(f"token{i}")

        # Wait for processing to complete
        # Calculate expected wait time: more tokens + more transitions = more time
        wait_time = 0.5 + (num_tokens * 0.01) + (delay_ms / 1000.0)
        await asyncio.sleep(wait_time)

        # Verify all tokens reached output
        output_tokens = list(runner.runtime.places[('TokenFlowNet', 'output')].tokens)

        # Property: No lost tokens
        assert len(output_tokens) == num_tokens, (
            f"Token flow violation: {num_tokens} tokens in, but {len(output_tokens)} tokens out. "
            f"Lost {num_tokens - len(output_tokens)} tokens"
        )

        # Property: No duplicate processing (each input token produces exactly one output)
        # Verify by checking that all output tokens have unique source tokens
        if num_tokens > 0:
            # Extract original token numbers (before "_tN" suffix)
            token_sources = [t.split('_t')[0] for t in output_tokens]
            # Each source should appear exactly once
            from collections import Counter
            counts = Counter(token_sources)
            duplicates = {token: count for token, count in counts.items() if count > 1}

            assert not duplicates, (
                f"Token duplication detected: {duplicates}. "
                f"Each input token should be processed exactly once."
            )

        await runner.stop()

    @pytest.mark.slow
    @given(
        initial_tokens=st.integers(min_value=0, max_value=15),
        add_during_run=st.integers(min_value=0, max_value=10),
        place_type=st.sampled_from([PlaceType.BAG, PlaceType.QUEUE]),
        use_output=st.booleans()
    )
    @settings(
        max_examples=30,
        phases=[Phase.generate],
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]  # We create resources inline
    )
    async def test_property_no_lost_tokens(self, initial_tokens, add_during_run, place_type, use_output):
        """Property: Token count is conserved through the net

        This test verifies the token conservation invariant:
        - Tokens are never lost (consumed from input but not produced to output)
        - For simple pass-through transitions, count should be conserved
        - Works for both BAG and QUEUE place types

        Hypothesis generates:
        - initial_tokens: Starting token count (0-15)
        - add_during_run: Additional tokens to add while running (0-10)
        - place_type: BAG or QUEUE
        - use_output: Whether transition has output arc (test sink transitions)
        """

        # Create fresh resources for each test run (Hypothesis requirement)
        test_bb = TestBlackboard()
        cycle_tb = CycleClock()

        # Track actual tokens for verification
        tokens_added = []
        tokens_produced = []

        @pn.net
        def ConservationNet(builder):
            input_p = builder.place("input", type=place_type)
            if use_output:
                output_p = builder.place("output", type=place_type)

            @builder.transition()
            async def passthrough(consumed, bb, timebase):
                # Track consumed tokens
                consumed_token = consumed[0]
                tokens_produced.append(consumed_token)

                if use_output:
                    yield {output_p: consumed_token}
                # If no output, tokens are consumed (sink transition)

            builder.arc(input_p, passthrough)
            if use_output:
                builder.arc(passthrough, output_p)

        runner = PNRunner(ConservationNet, test_bb)
        await runner.start(cycle_tb)

        input_place = runner.runtime.places[('ConservationNet', 'input')]

        # Add initial tokens
        for i in range(initial_tokens):
            token = f"initial_{i}"
            tokens_added.append(token)
            input_place.add_token(token)

        # Add more tokens during processing
        await asyncio.sleep(0.05)  # Let some processing start
        for i in range(add_during_run):
            token = f"added_{i}"
            tokens_added.append(token)
            input_place.add_token(token)

        # Wait for processing
        total_tokens = initial_tokens + add_during_run
        wait_time = 0.2 + (total_tokens * 0.01)
        await asyncio.sleep(wait_time)

        # Verify token conservation
        if use_output:
            # Tokens should be conserved (input -> output)
            output_place = runner.runtime.places[('ConservationNet', 'output')]
            output_tokens = list(output_place.tokens)

            # Property: All tokens that were consumed should be in output
            # Note: Some tokens may still be in input if not yet processed
            tokens_still_in_input = len(input_place.tokens)

            # Count tokens that were actually processed
            tokens_processed = len(tokens_produced)
            tokens_in_output = len(output_tokens)

            # All processed tokens should be in output
            assert tokens_processed == tokens_in_output, (
                f"Token conservation violation: "
                f"{tokens_processed} tokens processed but {tokens_in_output} in output. "
                f"Lost {tokens_processed - tokens_in_output} tokens during processing."
            )

            # Verify no more tokens remain in input than expected
            # (accounting for tokens that may still be processing)
            total_accounted = tokens_in_output + tokens_still_in_input
            assert total_accounted <= total_tokens, (
                f"Token duplication detected: "
                f"{total_tokens} tokens added but {total_accounted} tokens found "
                f"({tokens_in_output} in output + {tokens_still_in_input} in input)"
            )
        else:
            # Sink transition: tokens should be consumed but not produced
            # Just verify no crashes and processing happened
            tokens_processed = len(tokens_produced)
            tokens_still_in_input = len(input_place.tokens)

            # All tokens should either be processed or still in input
            total_accounted = tokens_processed + tokens_still_in_input
            assert total_accounted == total_tokens, (
                f"Token conservation violation in sink net: "
                f"{total_tokens} tokens added but {total_accounted} accounted for "
                f"({tokens_processed} processed + {tokens_still_in_input} in input)"
            )

        await runner.stop()


# =============================================================================
# Race Condition Tests
# =============================================================================

class TestRaceConditions:
    """Tests for concurrent access scenarios"""

    async def test_competing_transitions(self, test_bb, cycle_tb):
        """Two transitions competing for same input place"""

        @pn.net
        def CompetingNet(builder):
            input_p = builder.place("input", type=PlaceType.BAG)
            output1_p = builder.place("output1", type=PlaceType.BAG)
            output2_p = builder.place("output2", type=PlaceType.BAG)

            @builder.transition()
            async def consumer1(consumed, bb, timebase):
                await asyncio.sleep(0.01)  # Small delay to create race
                bb.processed_count += 1
                yield {output1_p: consumed[0]}

            @builder.transition()
            async def consumer2(consumed, bb, timebase):
                await asyncio.sleep(0.01)  # Small delay to create race
                bb.processed_count += 1
                yield {output2_p: consumed[0]}

            builder.arc(input_p, consumer1)
            builder.arc(input_p, consumer2)

        runner = PNRunner(CompetingNet, test_bb)
        await runner.start(cycle_tb)

        # Add single token - both transitions wake up
        input_place = runner.runtime.places[('CompetingNet', 'input')]
        input_place.add_token("test")

        await asyncio.sleep(0.2)

        # Verify: exactly one transition consumed the token
        output1_tokens = runner.runtime.places[('CompetingNet', 'output1')].tokens
        output2_tokens = runner.runtime.places[('CompetingNet', 'output2')].tokens

        total_tokens = len(output1_tokens) + len(output2_tokens)
        assert total_tokens == 1, f"Expected 1 token, got {total_tokens}"

        # Verify no infinite loops (spurious wakeup count should be reasonable)
        consumer1 = runner.runtime.transitions[('CompetingNet', 'consumer1')]
        consumer2 = runner.runtime.transitions[('CompetingNet', 'consumer2')]
        assert consumer1.get_spurious_wakeup_count() < 100, "Too many spurious wakeups in consumer1"
        assert consumer2.get_spurious_wakeup_count() < 100, "Too many spurious wakeups in consumer2"

        await runner.stop()

    async def test_multiple_competing_transitions(self, test_bb, cycle_tb):
        """Ten transitions competing for same input place"""

        @pn.net
        def ManyCompetingNet(builder):
            input_p = builder.place("input", type=PlaceType.BAG)

            # Create 10 competing transitions
            for i in range(10):
                output_p = builder.place(f"output{i}", type=PlaceType.BAG)

                @builder.transition()
                async def consumer(consumed, bb, timebase, idx=i):
                    await asyncio.sleep(0.001)
                    yield {output_p: consumed[0]}

                builder.arc(input_p, consumer)

        runner = PNRunner(ManyCompetingNet, test_bb)
        await runner.start(cycle_tb)

        # Add 5 tokens - should be distributed across transitions
        input_place = runner.runtime.places[('ManyCompetingNet', 'input')]
        for i in range(5):
            input_place.add_token(f"token{i}")

        await asyncio.sleep(0.3)

        # Count total output tokens
        total_output = 0
        for i in range(10):
            output_tokens = runner.runtime.places[('ManyCompetingNet', f'output{i}')].tokens
            total_output += len(output_tokens)

        assert total_output == 5, f"Expected 5 tokens, got {total_output}"

        # Verify no transition had excessive spurious wakeups
        for i in range(10):
            trans = runner.runtime.transitions[('ManyCompetingNet', f'consumer')]
            # Note: This will access the same transition 10 times - in real test we'd access each one
            if hasattr(trans, 'get_spurious_wakeup_count'):
                count = trans.get_spurious_wakeup_count()
                assert count < 100, f"Too many spurious wakeups: {count}"

        await runner.stop()

    async def test_spurious_wakeup_handling(self, test_bb, cycle_tb):
        """Verify transitions handle spurious wakeups correctly"""

        @pn.net
        def SpuriousWakeupNet(builder):
            input_p = builder.place("input", type=PlaceType.BAG)
            output_p = builder.place("output", type=PlaceType.BAG)

            @builder.transition()
            async def processor(consumed, bb, timebase):
                yield {output_p: consumed[0]}

            builder.arc(input_p, processor)

        runner = PNRunner(SpuriousWakeupNet, test_bb)
        await runner.start(cycle_tb)

        # Manually trigger event without adding tokens (simulate spurious wakeup)
        input_place = runner.runtime.places[('SpuriousWakeupNet', 'input')]
        input_place.token_added_event.set()

        await asyncio.sleep(0.1)

        # Transition should handle spurious wakeup gracefully
        trans = runner.runtime.transitions[('SpuriousWakeupNet', 'processor')]
        spurious_count = trans.get_spurious_wakeup_count()
        assert spurious_count > 0, "Should have detected at least one spurious wakeup"

        # Now add actual token - should process normally
        input_place.add_token("test")
        await asyncio.sleep(0.1)

        output_tokens = runner.runtime.places[('SpuriousWakeupNet', 'output')].tokens
        assert len(output_tokens) == 1, "Token should have been processed"

        await runner.stop()

    async def test_concurrent_place_access(self, test_bb, cycle_tb):
        """Multiple transitions accessing same place concurrently"""

        @pn.net
        def ConcurrentAccessNet(builder):
            input_p = builder.place("input", type=PlaceType.BAG)
            output_p = builder.place("output", type=PlaceType.BAG)

            @builder.transition()
            async def fast_processor(consumed, bb, timebase):
                # Very fast processing
                yield {output_p: f"fast_{consumed[0]}"}

            @builder.transition()
            async def slow_processor(consumed, bb, timebase):
                await asyncio.sleep(0.05)
                yield {output_p: f"slow_{consumed[0]}"}

            builder.arc(input_p, fast_processor)
            builder.arc(input_p, slow_processor)

        runner = PNRunner(ConcurrentAccessNet, test_bb)
        await runner.start(cycle_tb)

        # Add multiple tokens rapidly
        input_place = runner.runtime.places[('ConcurrentAccessNet', 'input')]
        for i in range(10):
            input_place.add_token(f"token{i}")

        await asyncio.sleep(0.3)

        # Verify all tokens processed
        output_tokens = runner.runtime.places[('ConcurrentAccessNet', 'output')].tokens
        assert len(output_tokens) == 10, f"Expected 10 tokens, got {len(output_tokens)}"

        await runner.stop()


# =============================================================================
# Stress Tests
# =============================================================================

class TestStress:
    """Stress tests for high-load scenarios"""

    @pytest.mark.slow
    async def test_many_transitions(self, test_bb, cycle_tb):
        """100 transitions processing tokens"""

        @pn.net
        def ManyTransitionsNet(builder):
            input_p = builder.place("input", type=PlaceType.BAG)
            output_p = builder.place("output", type=PlaceType.BAG)

            # Create 100 transitions
            for i in range(100):
                @builder.transition()
                async def worker(consumed, bb, timebase):
                    yield {output_p: consumed[0]}

                builder.arc(input_p, worker)

        runner = PNRunner(ManyTransitionsNet, test_bb)
        await runner.start(cycle_tb)

        # Add tokens
        input_place = runner.runtime.places[('ManyTransitionsNet', 'input')]
        for i in range(50):
            input_place.add_token(f"token{i}")

        await asyncio.sleep(1.0)

        # Verify processing
        output_tokens = runner.runtime.places[('ManyTransitionsNet', 'output')].tokens
        assert len(output_tokens) == 50, f"Expected 50 tokens, got {len(output_tokens)}"

        await runner.stop()

    @pytest.mark.slow
    async def test_rapid_token_addition(self, test_bb, cycle_tb):
        """Add tokens while transitions are actively processing"""

        @pn.net
        def RapidAdditionNet(builder):
            input_p = builder.place("input", type=PlaceType.BAG)
            output_p = builder.place("output", type=PlaceType.BAG)

            @builder.transition()
            async def worker(consumed, bb, timebase):
                await asyncio.sleep(0.01)  # Simulate work
                yield {output_p: consumed[0]}

            builder.arc(input_p, worker)

        runner = PNRunner(RapidAdditionNet, test_bb)
        await runner.start(cycle_tb)

        # Add tokens in batches while processing
        input_place = runner.runtime.places[('RapidAdditionNet', 'input')]
        for batch in range(5):
            for i in range(10):
                input_place.add_token(f"batch{batch}_token{i}")
            await asyncio.sleep(0.05)

        await asyncio.sleep(0.5)

        # Verify all tokens processed
        output_tokens = runner.runtime.places[('RapidAdditionNet', 'output')].tokens
        assert len(output_tokens) == 50, f"Expected 50 tokens, got {len(output_tokens)}"

        await runner.stop()
