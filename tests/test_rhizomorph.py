#!/usr/bin/env python3
"""
Comprehensive pytest test suite for rhizomorph behavior tree library.

Run with: pytest test_rhizomorph.py -v
"""

import asyncio
import pytest
from dataclasses import dataclass, field
from typing import Any, List, Optional

from mycorrhizal.rhizomorph.core import (
    bt,
    Status,
    ExceptionPolicy,
    Runner,
    RecursionError,
)
from mycorrhizal.common.timebase import Timebase, MonotonicClock


# =============================================================================
# Test Fixtures
# =============================================================================


class MockTimebase(Timebase):
    """Controllable timebase for testing time-dependent behavior."""

    def __init__(self, start: float = 0.0):
        self._time = start

    def now(self) -> float:
        return self._time

    def advance(self, delta: float = 0.1) -> None:
        self._time += delta

    def set_time(self, t: float) -> None:
        self._time = t
        
    async def sleep(self, duration: float) -> None:
        self._time += duration


@dataclass
class SimpleBlackboard:
    """Simple blackboard for basic tests."""
    value: int = 0
    log: List[str] = field(default_factory=list)
    should_succeed: bool = True
    tick_count: int = 0


@dataclass
class ActionBlackboard:
    """Blackboard for action dispatch tests."""
    current_action: Any = None
    handled_by: str = ""
    log: List[str] = field(default_factory=list)


@pytest.fixture
def simple_bb():
    return SimpleBlackboard()


@pytest.fixture
def action_bb():
    return ActionBlackboard()


@pytest.fixture
def mock_tb():
    return MockTimebase()


# =============================================================================
# Action Types for Match Tests
# =============================================================================


@dataclass
class ImageAction:
    pod_id: int
    image_type: str = "standard"


@dataclass
class MoveAction:
    x: float
    y: float


@dataclass
class CalibrateAction:
    sensor: str


@dataclass
class PriorityAction:
    priority: int
    message: str


# =============================================================================
# Test: Basic Action and Condition Nodes
# =============================================================================


class TestActionNode:
    """Tests for basic action nodes."""

    @pytest.mark.asyncio
    async def test_action_returns_success(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def succeed(bb: SimpleBlackboard):
                bb.log.append("executed")
                return Status.SUCCESS

            @bt.root
            @bt.sequence()
            def root(N):
                yield succeed

        runner = Runner(Tree, simple_bb, tb=mock_tb)
        status = await runner.tick()

        assert status == Status.SUCCESS
        assert simple_bb.log == ["executed"]

    @pytest.mark.asyncio
    async def test_action_returns_failure(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def fail(bb: SimpleBlackboard):
                return Status.FAILURE

            @bt.root
            @bt.sequence()
            def root(N):
                yield fail

        runner = Runner(Tree, simple_bb, tb=mock_tb)
        status = await runner.tick()

        assert status == Status.FAILURE

    @pytest.mark.asyncio
    async def test_action_returns_bool_true(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def bool_action(bb: SimpleBlackboard):
                return True

            @bt.root
            @bt.sequence()
            def root(N):
                yield bool_action

        runner = Runner(Tree, simple_bb, tb=mock_tb)
        status = await runner.tick()

        assert status == Status.SUCCESS

    @pytest.mark.asyncio
    async def test_action_returns_bool_false(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def bool_action(bb: SimpleBlackboard):
                return False

            @bt.root
            @bt.sequence()
            def root(N):
                yield bool_action

        runner = Runner(Tree, simple_bb, tb=mock_tb)
        status = await runner.tick()

        assert status == Status.FAILURE

    @pytest.mark.asyncio
    async def test_action_returns_none_is_success(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def none_action(bb: SimpleBlackboard):
                bb.log.append("ran")
                # implicit None return

            @bt.root
            @bt.sequence()
            def root(N):
                yield none_action

        runner = Runner(Tree, simple_bb, tb=mock_tb)
        status = await runner.tick()

        assert status == Status.SUCCESS
        assert simple_bb.log == ["ran"]

    @pytest.mark.asyncio
    async def test_async_action(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            async def async_action(bb: SimpleBlackboard):
                await asyncio.sleep(0)
                bb.log.append("async executed")
                return Status.SUCCESS

            @bt.root
            @bt.sequence()
            def root(N):
                yield async_action

        runner = Runner(Tree, simple_bb, tb=mock_tb)
        status = await runner.tick()

        assert status == Status.SUCCESS
        assert simple_bb.log == ["async executed"]

    @pytest.mark.asyncio
    async def test_action_with_timebase(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def timed_action(bb: SimpleBlackboard, tb: Timebase):
                bb.log.append(f"time={tb.now()}")
                return Status.SUCCESS

            @bt.root
            @bt.sequence()
            def root(N):
                yield timed_action

        mock_tb.set_time(42.0)
        runner = Runner(Tree, simple_bb, tb=mock_tb)
        status = await runner.tick()

        assert status == Status.SUCCESS
        assert simple_bb.log == ["time=42.0"]


class TestConditionNode:
    """Tests for condition nodes."""

    @pytest.mark.asyncio
    async def test_condition_true(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.condition
            def is_true(bb: SimpleBlackboard):
                return True

            @bt.root
            @bt.sequence()
            def root(N):
                yield is_true

        runner = Runner(Tree, simple_bb, tb=mock_tb)
        status = await runner.tick()

        assert status == Status.SUCCESS

    @pytest.mark.asyncio
    async def test_condition_false(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.condition
            def is_false(bb: SimpleBlackboard):
                return False

            @bt.root
            @bt.sequence()
            def root(N):
                yield is_false

        runner = Runner(Tree, simple_bb, tb=mock_tb)
        status = await runner.tick()

        assert status == Status.FAILURE

    @pytest.mark.asyncio
    async def test_condition_with_blackboard_state(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.condition
            def check_value(bb: SimpleBlackboard):
                return bb.value > 10

            @bt.root
            @bt.sequence()
            def root(N):
                yield check_value

        runner = Runner(Tree, simple_bb, tb=mock_tb)

        simple_bb.value = 5
        status = await runner.tick()
        assert status == Status.FAILURE

        runner.root.reset()
        simple_bb.value = 15
        status = await runner.tick()
        assert status == Status.SUCCESS


# =============================================================================
# Test: Composite Nodes
# =============================================================================


class TestSequence:
    """Tests for sequence composite nodes."""

    @pytest.mark.asyncio
    async def test_sequence_all_success(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def action1(bb: SimpleBlackboard):
                bb.log.append("a1")
                return Status.SUCCESS

            @bt.action
            def action2(bb: SimpleBlackboard):
                bb.log.append("a2")
                return Status.SUCCESS

            @bt.action
            def action3(bb: SimpleBlackboard):
                bb.log.append("a3")
                return Status.SUCCESS

            @bt.root
            @bt.sequence()
            def root(N):
                yield action1
                yield action2
                yield action3

        runner = Runner(Tree, simple_bb, tb=mock_tb)
        status = await runner.tick()

        assert status == Status.SUCCESS
        assert simple_bb.log == ["a1", "a2", "a3"]

    @pytest.mark.asyncio
    async def test_sequence_fails_fast(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def action1(bb: SimpleBlackboard):
                bb.log.append("a1")
                return Status.SUCCESS

            @bt.action
            def action2(bb: SimpleBlackboard):
                bb.log.append("a2")
                return Status.FAILURE

            @bt.action
            def action3(bb: SimpleBlackboard):
                bb.log.append("a3")
                return Status.SUCCESS

            @bt.root
            @bt.sequence()
            def root(N):
                yield action1
                yield action2
                yield action3

        runner = Runner(Tree, simple_bb, tb=mock_tb)
        status = await runner.tick()

        assert status == Status.FAILURE
        assert simple_bb.log == ["a1", "a2"]  # a3 never runs

    @pytest.mark.asyncio
    async def test_sequence_with_memory(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def action1(bb: SimpleBlackboard):
                bb.log.append("a1")
                return Status.SUCCESS

            @bt.action
            def running_then_success(bb: SimpleBlackboard):
                bb.tick_count += 1
                bb.log.append(f"a2-tick{bb.tick_count}")
                if bb.tick_count < 2:
                    return Status.RUNNING
                return Status.SUCCESS

            @bt.action
            def action3(bb: SimpleBlackboard):
                bb.log.append("a3")
                return Status.SUCCESS

            @bt.root
            @bt.sequence(memory=True)
            def root(N):
                yield action1
                yield running_then_success
                yield action3

        runner = Runner(Tree, simple_bb, tb=mock_tb)

        status = await runner.tick()
        assert status == Status.RUNNING
        assert simple_bb.log == ["a1", "a2-tick1"]

        status = await runner.tick()
        assert status == Status.SUCCESS
        assert simple_bb.log == ["a1", "a2-tick1", "a2-tick2", "a3"]

    @pytest.mark.asyncio
    async def test_sequence_without_memory(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def action1(bb: SimpleBlackboard):
                bb.log.append("a1")
                return Status.SUCCESS

            @bt.action
            def running_then_success(bb: SimpleBlackboard):
                bb.tick_count += 1
                bb.log.append(f"a2-tick{bb.tick_count}")
                if bb.tick_count < 2:
                    return Status.RUNNING
                return Status.SUCCESS

            @bt.root
            @bt.sequence(memory=False)
            def root(N):
                yield action1
                yield running_then_success

        runner = Runner(Tree, simple_bb, tb=mock_tb)

        status = await runner.tick()
        assert status == Status.RUNNING

        status = await runner.tick()
        assert status == Status.SUCCESS
        # Without memory, a1 runs again on second tick
        assert simple_bb.log == ["a1", "a2-tick1", "a1", "a2-tick2"]


class TestSelector:
    """Tests for selector composite nodes."""

    @pytest.mark.asyncio
    async def test_selector_first_succeeds(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def action1(bb: SimpleBlackboard):
                bb.log.append("a1")
                return Status.SUCCESS

            @bt.action
            def action2(bb: SimpleBlackboard):
                bb.log.append("a2")
                return Status.SUCCESS

            @bt.root
            @bt.selector()
            def root(N):
                yield action1
                yield action2

        runner = Runner(Tree, simple_bb, tb=mock_tb)
        status = await runner.tick()

        assert status == Status.SUCCESS
        assert simple_bb.log == ["a1"]  # a2 never runs

    @pytest.mark.asyncio
    async def test_selector_tries_until_success(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def fail1(bb: SimpleBlackboard):
                bb.log.append("f1")
                return Status.FAILURE

            @bt.action
            def fail2(bb: SimpleBlackboard):
                bb.log.append("f2")
                return Status.FAILURE

            @bt.action
            def succeed(bb: SimpleBlackboard):
                bb.log.append("s")
                return Status.SUCCESS

            @bt.root
            @bt.selector()
            def root(N):
                yield fail1
                yield fail2
                yield succeed

        runner = Runner(Tree, simple_bb, tb=mock_tb)
        status = await runner.tick()

        assert status == Status.SUCCESS
        assert simple_bb.log == ["f1", "f2", "s"]

    @pytest.mark.asyncio
    async def test_selector_all_fail(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def fail1(bb: SimpleBlackboard):
                bb.log.append("f1")
                return Status.FAILURE

            @bt.action
            def fail2(bb: SimpleBlackboard):
                bb.log.append("f2")
                return Status.FAILURE

            @bt.root
            @bt.selector()
            def root(N):
                yield fail1
                yield fail2

        runner = Runner(Tree, simple_bb, tb=mock_tb)
        status = await runner.tick()

        assert status == Status.FAILURE
        assert simple_bb.log == ["f1", "f2"]


class TestParallel:
    """Tests for parallel composite nodes."""

    @pytest.mark.asyncio
    async def test_parallel_success_threshold(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def succeed1(bb: SimpleBlackboard):
                bb.log.append("s1")
                return Status.SUCCESS

            @bt.action
            def succeed2(bb: SimpleBlackboard):
                bb.log.append("s2")
                return Status.SUCCESS

            @bt.action
            def fail1(bb: SimpleBlackboard):
                bb.log.append("f1")
                return Status.FAILURE

            @bt.root
            @bt.parallel(success_threshold=2)
            def root(N):
                yield succeed1
                yield succeed2
                yield fail1

        runner = Runner(Tree, simple_bb, tb=mock_tb)
        status = await runner.tick()

        assert status == Status.SUCCESS
        assert set(simple_bb.log) == {"s1", "s2", "f1"}

    @pytest.mark.asyncio
    async def test_parallel_failure_threshold(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def succeed1(bb: SimpleBlackboard):
                bb.log.append("s1")
                return Status.SUCCESS

            @bt.action
            def fail1(bb: SimpleBlackboard):
                bb.log.append("f1")
                return Status.FAILURE

            @bt.action
            def fail2(bb: SimpleBlackboard):
                bb.log.append("f2")
                return Status.FAILURE

            @bt.root
            @bt.parallel(success_threshold=2, failure_threshold=2)
            def root(N):
                yield succeed1
                yield fail1
                yield fail2

        runner = Runner(Tree, simple_bb, tb=mock_tb)
        status = await runner.tick()

        assert status == Status.FAILURE


# =============================================================================
# Test: Decorator Nodes
# =============================================================================


class TestInverter:
    """Tests for inverter decorator."""

    @pytest.mark.asyncio
    async def test_inverter_success_to_failure(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def succeed(bb: SimpleBlackboard):
                return Status.SUCCESS

            @bt.root
            @bt.sequence()
            def root(N):
                yield bt.inverter()(succeed)

        runner = Runner(Tree, simple_bb, tb=mock_tb)
        status = await runner.tick()

        assert status == Status.FAILURE

    @pytest.mark.asyncio
    async def test_inverter_failure_to_success(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def fail(bb: SimpleBlackboard):
                return Status.FAILURE

            @bt.root
            @bt.sequence()
            def root(N):
                yield bt.inverter()(fail)

        runner = Runner(Tree, simple_bb, tb=mock_tb)
        status = await runner.tick()

        assert status == Status.SUCCESS

    @pytest.mark.asyncio
    async def test_inverter_running_unchanged(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def running(bb: SimpleBlackboard):
                return Status.RUNNING

            @bt.root
            @bt.sequence()
            def root(N):
                yield bt.inverter()(running)

        runner = Runner(Tree, simple_bb, tb=mock_tb)
        status = await runner.tick()

        assert status == Status.RUNNING


class TestSucceeder:
    """Tests for succeeder decorator."""

    @pytest.mark.asyncio
    async def test_succeeder_converts_failure(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def fail(bb: SimpleBlackboard):
                return Status.FAILURE

            @bt.root
            @bt.sequence()
            def root(N):
                yield bt.succeeder()(fail)

        runner = Runner(Tree, simple_bb, tb=mock_tb)
        status = await runner.tick()

        assert status == Status.SUCCESS

    @pytest.mark.asyncio
    async def test_succeeder_preserves_running(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def running(bb: SimpleBlackboard):
                return Status.RUNNING

            @bt.root
            @bt.sequence()
            def root(N):
                yield bt.succeeder()(running)

        runner = Runner(Tree, simple_bb, tb=mock_tb)
        status = await runner.tick()

        assert status == Status.RUNNING


class TestFailer:
    """Tests for failer decorator."""

    @pytest.mark.asyncio
    async def test_failer_converts_success(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def succeed(bb: SimpleBlackboard):
                return Status.SUCCESS

            @bt.root
            @bt.selector()
            def root(N):
                yield bt.failer()(succeed)

        runner = Runner(Tree, simple_bb, tb=mock_tb)
        status = await runner.tick()

        assert status == Status.FAILURE


class TestRetry:
    """Tests for retry decorator."""

    @pytest.mark.asyncio
    async def test_retry_eventually_succeeds(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def fail_twice_then_succeed(bb: SimpleBlackboard):
                bb.tick_count += 1
                bb.log.append(f"attempt{bb.tick_count}")
                if bb.tick_count < 3:
                    return Status.FAILURE
                return Status.SUCCESS

            @bt.root
            @bt.sequence()
            def root(N):
                yield bt.retry(max_attempts=5)(fail_twice_then_succeed)

        runner = Runner(Tree, simple_bb, tb=mock_tb)

        # First attempt fails, returns RUNNING to retry
        status = await runner.tick()
        assert status == Status.RUNNING

        # Second attempt fails, returns RUNNING to retry
        status = await runner.tick()
        assert status == Status.RUNNING

        # Third attempt succeeds
        status = await runner.tick()
        assert status == Status.SUCCESS
        assert simple_bb.log == ["attempt1", "attempt2", "attempt3"]

    @pytest.mark.asyncio
    async def test_retry_exhausted(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def always_fail(bb: SimpleBlackboard):
                bb.tick_count += 1
                return Status.FAILURE

            @bt.root
            @bt.sequence()
            def root(N):
                yield bt.retry(max_attempts=3)(always_fail)

        runner = Runner(Tree, simple_bb, tb=mock_tb)

        status = await runner.tick()
        assert status == Status.RUNNING

        status = await runner.tick()
        assert status == Status.RUNNING

        status = await runner.tick()
        assert status == Status.FAILURE
        assert simple_bb.tick_count == 3


class TestTimeout:
    """Tests for timeout decorator."""

    @pytest.mark.asyncio
    async def test_timeout_completes_in_time(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def quick_action(bb: SimpleBlackboard):
                return Status.SUCCESS

            @bt.root
            @bt.sequence()
            def root(N):
                yield bt.timeout(5.0)(quick_action)

        runner = Runner(Tree, simple_bb, tb=mock_tb)
        status = await runner.tick()

        assert status == Status.SUCCESS

    @pytest.mark.asyncio
    async def test_timeout_expires(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def slow_action(bb: SimpleBlackboard):
                return Status.RUNNING

            @bt.root
            @bt.sequence()
            def root(N):
                yield bt.timeout(1.0)(slow_action)

        runner = Runner(Tree, simple_bb, tb=mock_tb)

        mock_tb.set_time(0.0)
        status = await runner.tick()
        assert status == Status.RUNNING

        mock_tb.set_time(0.5)
        status = await runner.tick()
        assert status == Status.RUNNING

        mock_tb.set_time(1.5)
        status = await runner.tick()
        assert status == Status.FAILURE


class TestGate:
    """Tests for gate decorator."""

    @pytest.mark.asyncio
    async def test_gate_allows_when_condition_true(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.condition
            def is_allowed(bb: SimpleBlackboard):
                return bb.should_succeed

            @bt.action
            def guarded_action(bb: SimpleBlackboard):
                bb.log.append("executed")
                return Status.SUCCESS

            @bt.root
            @bt.selector()
            def root(N):
                yield bt.gate(is_allowed)(guarded_action)

        simple_bb.should_succeed = True
        runner = Runner(Tree, simple_bb, tb=mock_tb)
        status = await runner.tick()

        assert status == Status.SUCCESS
        assert simple_bb.log == ["executed"]

    @pytest.mark.asyncio
    async def test_gate_blocks_when_condition_false(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.condition
            def is_allowed(bb: SimpleBlackboard):
                return bb.should_succeed

            @bt.action
            def guarded_action(bb: SimpleBlackboard):
                bb.log.append("executed")
                return Status.SUCCESS

            @bt.root
            @bt.selector()
            def root(N):
                yield bt.gate(is_allowed)(guarded_action)

        simple_bb.should_succeed = False
        runner = Runner(Tree, simple_bb, tb=mock_tb)
        status = await runner.tick()

        assert status == Status.FAILURE
        assert simple_bb.log == []


class TestRateLimit:
    """Tests for rate limit decorator."""

    @pytest.mark.asyncio
    async def test_ratelimit_throttles(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def action(bb: SimpleBlackboard):
                bb.tick_count += 1
                return Status.SUCCESS

            @bt.root
            @bt.sequence()
            def root(N):
                yield bt.ratelimit(period=1.0)(action)

        runner = Runner(Tree, simple_bb, tb=mock_tb)

        mock_tb.set_time(0.0)
        status = await runner.tick()
        assert status == Status.SUCCESS
        assert simple_bb.tick_count == 1

        # Too soon - should be throttled (don't reset, just tick again)
        mock_tb.set_time(0.5)
        status = await runner.tick()
        assert status == Status.RUNNING
        assert simple_bb.tick_count == 1

        # Enough time passed
        mock_tb.set_time(1.5)
        status = await runner.tick()
        assert status == Status.SUCCESS
        assert simple_bb.tick_count == 2

    @pytest.mark.asyncio
    async def test_ratelimit_hz(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def action(bb: SimpleBlackboard):
                bb.tick_count += 1
                return Status.SUCCESS

            @bt.root
            @bt.sequence()
            def root(N):
                yield bt.ratelimit(hz=2.0)(action)  # 0.5 second period

        runner = Runner(Tree, simple_bb, tb=mock_tb)

        mock_tb.set_time(0.0)
        status = await runner.tick()
        assert status == Status.SUCCESS

        mock_tb.set_time(0.6)
        runner.root.reset()
        status = await runner.tick()
        assert status == Status.SUCCESS
        assert simple_bb.tick_count == 2


# =============================================================================
# Test: Decorator Chaining
# =============================================================================


class TestDecoratorChaining:
    """Tests for chaining multiple decorators."""

    @pytest.mark.asyncio
    async def test_chain_gate_and_retry(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.condition
            def is_enabled(bb: SimpleBlackboard):
                return bb.should_succeed

            @bt.action
            def flaky_action(bb: SimpleBlackboard):
                bb.tick_count += 1
                if bb.tick_count < 3:
                    return Status.FAILURE
                return Status.SUCCESS

            @bt.root
            @bt.selector()
            def root(N):
                yield bt.gate(is_enabled).retry(max_attempts=5)(flaky_action)

        simple_bb.should_succeed = True
        runner = Runner(Tree, simple_bb, tb=mock_tb)

        status = await runner.tick()
        assert status == Status.RUNNING

        status = await runner.tick()
        assert status == Status.RUNNING

        status = await runner.tick()
        assert status == Status.SUCCESS

    @pytest.mark.asyncio
    async def test_chain_succeeder_and_inverter(self, simple_bb, mock_tb):
        """
        Test decorator chaining. Chain is applied left-to-right as outer-to-inner.
        bt.succeeder().inverter()(fail) means:
          - fail returns FAILURE
          - inverter (innermost) converts FAILURE -> SUCCESS  
          - succeeder (outermost) keeps SUCCESS -> SUCCESS
        """

        @bt.tree
        def Tree():
            @bt.action
            def fail(bb: SimpleBlackboard):
                return Status.FAILURE

            @bt.root
            @bt.sequence()
            def root(N):
                yield bt.succeeder().inverter()(fail)

        runner = Runner(Tree, simple_bb, tb=mock_tb)
        status = await runner.tick()

        assert status == Status.SUCCESS

    @pytest.mark.asyncio
    async def test_chain_inverter_then_succeeder(self, simple_bb, mock_tb):
        """
        bt.inverter().succeeder()(succeed) means:
          - succeed returns SUCCESS
          - succeeder (innermost) keeps SUCCESS -> SUCCESS
          - inverter (outermost) converts SUCCESS -> FAILURE
        """

        @bt.tree
        def Tree():
            @bt.action
            def succeed(bb: SimpleBlackboard):
                return Status.SUCCESS

            @bt.root
            @bt.sequence()
            def root(N):
                yield bt.inverter().succeeder()(succeed)

        runner = Runner(Tree, simple_bb, tb=mock_tb)
        status = await runner.tick()

        assert status == Status.FAILURE


# =============================================================================
# Test: Match Node
# =============================================================================


class TestMatch:
    """Tests for the match pattern-matching node."""

    @pytest.mark.asyncio
    async def test_match_by_type(self, action_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def handle_image(bb: ActionBlackboard):
                bb.handled_by = "handle_image"
                return Status.SUCCESS

            @bt.action
            def handle_move(bb: ActionBlackboard):
                bb.handled_by = "handle_move"
                return Status.SUCCESS

            @bt.action
            def handle_calibrate(bb: ActionBlackboard):
                bb.handled_by = "handle_calibrate"
                return Status.SUCCESS

            @bt.root
            @bt.sequence()
            def root(N):
                yield bt.match(lambda bb: bb.current_action)(
                    bt.case(ImageAction)(handle_image),
                    bt.case(MoveAction)(handle_move),
                    bt.case(CalibrateAction)(handle_calibrate),
                )

        # Test ImageAction
        action_bb.current_action = ImageAction(pod_id=42)
        runner = Runner(Tree, action_bb, tb=mock_tb)
        status = await runner.tick()
        assert status == Status.SUCCESS
        assert action_bb.handled_by == "handle_image"

        # Test MoveAction
        action_bb.current_action = MoveAction(x=1.0, y=2.0)
        action_bb.handled_by = ""
        runner = Runner(Tree, action_bb, tb=mock_tb)
        status = await runner.tick()
        assert status == Status.SUCCESS
        assert action_bb.handled_by == "handle_move"

        # Test CalibrateAction
        action_bb.current_action = CalibrateAction(sensor="gyro")
        action_bb.handled_by = ""
        runner = Runner(Tree, action_bb, tb=mock_tb)
        status = await runner.tick()
        assert status == Status.SUCCESS
        assert action_bb.handled_by == "handle_calibrate"

    @pytest.mark.asyncio
    async def test_match_by_predicate(self, action_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def handle_urgent(bb: ActionBlackboard):
                bb.handled_by = "urgent"
                return Status.SUCCESS

            @bt.action
            def handle_normal(bb: ActionBlackboard):
                bb.handled_by = "normal"
                return Status.SUCCESS

            @bt.root
            @bt.sequence()
            def root(N):
                yield bt.match(lambda bb: bb.current_action)(
                    bt.case(lambda a: a.priority > 5)(handle_urgent),
                    bt.case(lambda a: a.priority <= 5)(handle_normal),
                )

        action_bb.current_action = PriorityAction(priority=10, message="urgent!")
        runner = Runner(Tree, action_bb, tb=mock_tb)
        status = await runner.tick()
        assert status == Status.SUCCESS
        assert action_bb.handled_by == "urgent"

        action_bb.current_action = PriorityAction(priority=3, message="normal")
        action_bb.handled_by = ""
        runner = Runner(Tree, action_bb, tb=mock_tb)
        status = await runner.tick()
        assert status == Status.SUCCESS
        assert action_bb.handled_by == "normal"

    @pytest.mark.asyncio
    async def test_match_by_value(self, action_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def handle_start(bb: ActionBlackboard):
                bb.handled_by = "start"
                return Status.SUCCESS

            @bt.action
            def handle_stop(bb: ActionBlackboard):
                bb.handled_by = "stop"
                return Status.SUCCESS

            @bt.root
            @bt.sequence()
            def root(N):
                yield bt.match(lambda bb: bb.current_action)(
                    bt.case("start")(handle_start),
                    bt.case("stop")(handle_stop),
                )

        action_bb.current_action = "start"
        runner = Runner(Tree, action_bb, tb=mock_tb)
        status = await runner.tick()
        assert status == Status.SUCCESS
        assert action_bb.handled_by == "start"

        action_bb.current_action = "stop"
        action_bb.handled_by = ""
        runner = Runner(Tree, action_bb, tb=mock_tb)
        status = await runner.tick()
        assert status == Status.SUCCESS
        assert action_bb.handled_by == "stop"

    @pytest.mark.asyncio
    async def test_match_default_case(self, action_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def handle_image(bb: ActionBlackboard):
                bb.handled_by = "image"
                return Status.SUCCESS

            @bt.action
            def handle_unknown(bb: ActionBlackboard):
                bb.handled_by = "unknown"
                return Status.FAILURE

            @bt.root
            @bt.sequence()
            def root(N):
                yield bt.match(lambda bb: bb.current_action)(
                    bt.case(ImageAction)(handle_image),
                    bt.defaultcase(handle_unknown),
                )

        action_bb.current_action = "something else entirely"
        runner = Runner(Tree, action_bb, tb=mock_tb)
        status = await runner.tick()
        assert status == Status.FAILURE
        assert action_bb.handled_by == "unknown"

    @pytest.mark.asyncio
    async def test_match_no_match_returns_failure(self, action_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def handle_image(bb: ActionBlackboard):
                bb.handled_by = "image"
                return Status.SUCCESS

            @bt.root
            @bt.sequence()
            def root(N):
                yield bt.match(lambda bb: bb.current_action)(
                    bt.case(ImageAction)(handle_image),
                )

        action_bb.current_action = MoveAction(x=0, y=0)
        runner = Runner(Tree, action_bb, tb=mock_tb)
        status = await runner.tick()
        assert status == Status.FAILURE
        assert action_bb.handled_by == ""

    @pytest.mark.asyncio
    async def test_match_remembers_running_case(self, action_bb, mock_tb):
        """Once a case matches and returns RUNNING, subsequent ticks continue on that case."""
        tick_count = [0]

        @bt.tree
        def Tree():
            @bt.action
            def multi_tick_handler(bb: ActionBlackboard):
                tick_count[0] += 1
                bb.log.append(f"tick{tick_count[0]}")
                if tick_count[0] < 3:
                    return Status.RUNNING
                return Status.SUCCESS

            @bt.action
            def should_not_run(bb: ActionBlackboard):
                bb.log.append("wrong!")
                return Status.SUCCESS

            @bt.root
            @bt.sequence()
            def root(N):
                yield bt.match(lambda bb: bb.current_action)(
                    bt.case(ImageAction)(multi_tick_handler),
                    bt.case(MoveAction)(should_not_run),
                )

        action_bb.current_action = ImageAction(pod_id=1)
        runner = Runner(Tree, action_bb, tb=mock_tb)

        status = await runner.tick()
        assert status == Status.RUNNING

        # Change action type - but we're committed to ImageAction handler
        action_bb.current_action = MoveAction(x=0, y=0)

        status = await runner.tick()
        assert status == Status.RUNNING

        status = await runner.tick()
        assert status == Status.SUCCESS
        assert action_bb.log == ["tick1", "tick2", "tick3"]
        assert "wrong!" not in action_bb.log

    @pytest.mark.asyncio
    async def test_match_first_matching_case_wins(self, action_bb, mock_tb):
        """Cases are checked in order; first match wins."""

        @bt.tree
        def Tree():
            @bt.action
            def handle_high_priority(bb: ActionBlackboard):
                bb.handled_by = "high"
                return Status.SUCCESS

            @bt.action
            def handle_any_priority(bb: ActionBlackboard):
                bb.handled_by = "any"
                return Status.SUCCESS

            @bt.root
            @bt.sequence()
            def root(N):
                yield bt.match(lambda bb: bb.current_action)(
                    bt.case(lambda a: a.priority > 5)(handle_high_priority),
                    bt.case(PriorityAction)(handle_any_priority),  # Would also match
                )

        action_bb.current_action = PriorityAction(priority=10, message="test")
        runner = Runner(Tree, action_bb, tb=mock_tb)
        status = await runner.tick()
        assert status == Status.SUCCESS
        assert action_bb.handled_by == "high"  # First case matched


# =============================================================================
# Test: DoWhile Loop
# =============================================================================


class TestDoWhile:
    """Tests for the do_while loop decorator."""

    @pytest.mark.asyncio
    async def test_do_while_loops_until_condition_false(self, simple_bb, mock_tb):
        """Loop runs while condition is true, succeeds when condition becomes false."""

        @bt.tree
        def Tree():
            @bt.condition
            def has_items(bb: SimpleBlackboard):
                return bb.value > 0

            @bt.action
            def process_item(bb: SimpleBlackboard):
                bb.value -= 1
                bb.log.append(f"processed, remaining={bb.value}")
                return Status.SUCCESS

            @bt.root
            @bt.sequence()
            def root(N):
                yield bt.do_while(has_items)(process_item)

        simple_bb.value = 3
        runner = Runner(Tree, simple_bb, tb=mock_tb)

        # First tick: condition true, child succeeds, returns RUNNING
        status = await runner.tick()
        assert status == Status.RUNNING
        assert simple_bb.value == 2

        # Second tick: condition true, child succeeds, returns RUNNING
        status = await runner.tick()
        assert status == Status.RUNNING
        assert simple_bb.value == 1

        # Third tick: condition true, child succeeds, returns RUNNING
        status = await runner.tick()
        assert status == Status.RUNNING
        assert simple_bb.value == 0

        # Fourth tick: condition false, returns SUCCESS
        status = await runner.tick()
        assert status == Status.SUCCESS
        assert simple_bb.log == [
            "processed, remaining=2",
            "processed, remaining=1",
            "processed, remaining=0",
        ]

    @pytest.mark.asyncio
    async def test_do_while_condition_false_immediately(self, simple_bb, mock_tb):
        """If condition is false from the start, loop succeeds immediately."""

        @bt.tree
        def Tree():
            @bt.condition
            def has_items(bb: SimpleBlackboard):
                return bb.value > 0

            @bt.action
            def process_item(bb: SimpleBlackboard):
                bb.log.append("should not run")
                return Status.SUCCESS

            @bt.root
            @bt.sequence()
            def root(N):
                yield bt.do_while(has_items)(process_item)

        simple_bb.value = 0
        runner = Runner(Tree, simple_bb, tb=mock_tb)

        status = await runner.tick()
        assert status == Status.SUCCESS
        assert simple_bb.log == []

    @pytest.mark.asyncio
    async def test_do_while_child_failure_aborts(self, simple_bb, mock_tb):
        """If child fails, loop aborts with FAILURE."""

        @bt.tree
        def Tree():
            @bt.condition
            def always_true(bb: SimpleBlackboard):
                return True

            @bt.action
            def fail_on_second(bb: SimpleBlackboard):
                bb.tick_count += 1
                bb.log.append(f"tick {bb.tick_count}")
                if bb.tick_count >= 2:
                    return Status.FAILURE
                return Status.SUCCESS

            @bt.root
            @bt.sequence()
            def root(N):
                yield bt.do_while(always_true)(fail_on_second)

        runner = Runner(Tree, simple_bb, tb=mock_tb)

        # First tick: child succeeds, loop continues
        status = await runner.tick()
        assert status == Status.RUNNING

        # Second tick: child fails, loop aborts
        status = await runner.tick()
        assert status == Status.FAILURE
        assert simple_bb.log == ["tick 1", "tick 2"]

    @pytest.mark.asyncio
    async def test_do_while_child_running(self, simple_bb, mock_tb):
        """If child returns RUNNING, loop waits for it to complete."""

        @bt.tree
        def Tree():
            @bt.condition
            def has_items(bb: SimpleBlackboard):
                return bb.value > 0

            @bt.action
            def slow_process(bb: SimpleBlackboard):
                bb.tick_count += 1
                bb.log.append(f"tick {bb.tick_count}")
                # Takes 2 ticks to process each item
                if bb.tick_count % 2 == 1:
                    return Status.RUNNING
                bb.value -= 1
                return Status.SUCCESS

            @bt.root
            @bt.sequence()
            def root(N):
                yield bt.do_while(has_items)(slow_process)

        simple_bb.value = 2
        runner = Runner(Tree, simple_bb, tb=mock_tb)

        # Tick 1: condition true, child RUNNING
        status = await runner.tick()
        assert status == Status.RUNNING
        assert simple_bb.value == 2

        # Tick 2: child completes (SUCCESS), value decremented, loop continues
        status = await runner.tick()
        assert status == Status.RUNNING
        assert simple_bb.value == 1

        # Tick 3: condition true, child RUNNING
        status = await runner.tick()
        assert status == Status.RUNNING
        assert simple_bb.value == 1

        # Tick 4: child completes (SUCCESS), value decremented, loop continues
        status = await runner.tick()
        assert status == Status.RUNNING
        assert simple_bb.value == 0

        # Tick 5: condition false, loop complete
        status = await runner.tick()
        assert status == Status.SUCCESS

    @pytest.mark.asyncio
    async def test_do_while_with_sequence_body(self, simple_bb, mock_tb):
        """do_while can wrap a sequence as its body."""

        @bt.tree
        def Tree():
            @bt.condition
            def samples_remain(bb: SimpleBlackboard):
                return bb.value < 3

            @bt.action
            def step_one(bb: SimpleBlackboard):
                bb.log.append(f"step1-{bb.value}")
                return Status.SUCCESS

            @bt.action
            def step_two(bb: SimpleBlackboard):
                bb.log.append(f"step2-{bb.value}")
                bb.value += 1
                return Status.SUCCESS

            @bt.sequence()
            def process_sample(N):
                yield step_one
                yield step_two

            @bt.root
            @bt.sequence()
            def root(N):
                yield bt.do_while(samples_remain)(process_sample)

        simple_bb.value = 0
        runner = Runner(Tree, simple_bb, tb=mock_tb)

        # Each iteration: RUNNING after child success
        status = await runner.tick()
        assert status == Status.RUNNING
        assert simple_bb.value == 1

        status = await runner.tick()
        assert status == Status.RUNNING
        assert simple_bb.value == 2

        status = await runner.tick()
        assert status == Status.RUNNING
        assert simple_bb.value == 3

        # Condition now false
        status = await runner.tick()
        assert status == Status.SUCCESS
        assert simple_bb.log == [
            "step1-0", "step2-0",
            "step1-1", "step2-1",
            "step1-2", "step2-2",
        ]

    @pytest.mark.asyncio
    async def test_do_while_with_inverter_for_until(self, simple_bb, mock_tb):
        """Use inverter on condition to get 'until' behavior."""

        @bt.tree
        def Tree():
            @bt.condition
            def is_done(bb: SimpleBlackboard):
                return bb.value >= 3

            @bt.action
            def increment(bb: SimpleBlackboard):
                bb.value += 1
                bb.log.append(f"value={bb.value}")
                return Status.SUCCESS

            @bt.root
            @bt.sequence()
            def root(N):
                # Loop until is_done is true (i.e., while NOT is_done)
                yield bt.do_while(bt.inverter()(is_done))(increment)

        simple_bb.value = 0
        runner = Runner(Tree, simple_bb, tb=mock_tb)

        status = await runner.tick()
        assert status == Status.RUNNING

        status = await runner.tick()
        assert status == Status.RUNNING

        status = await runner.tick()
        assert status == Status.RUNNING

        # Now is_done returns True, inverter makes it False, loop exits
        status = await runner.tick()
        assert status == Status.SUCCESS
        assert simple_bb.value == 3


# =============================================================================
# Test: Subtrees
# =============================================================================


class TestSubtree:
    """Tests for subtree composition."""

    @pytest.mark.asyncio
    async def test_subtree_basic(self, simple_bb, mock_tb):
        @bt.tree
        def SubTree():
            @bt.action
            def sub_action(bb: SimpleBlackboard):
                bb.log.append("sub")
                return Status.SUCCESS

            @bt.root
            @bt.sequence()
            def root(N):
                yield sub_action

        @bt.tree
        def MainTree():
            @bt.action
            def main_action(bb: SimpleBlackboard):
                bb.log.append("main")
                return Status.SUCCESS

            @bt.root
            @bt.sequence()
            def root(N):
                yield main_action
                yield bt.subtree(SubTree)

        runner = Runner(MainTree, simple_bb, tb=mock_tb)
        status = await runner.tick()

        assert status == Status.SUCCESS
        assert simple_bb.log == ["main", "sub"]

    @pytest.mark.asyncio
    async def test_subtree_failure_propagates(self, simple_bb, mock_tb):
        @bt.tree
        def FailingSubTree():
            @bt.action
            def fail(bb: SimpleBlackboard):
                bb.log.append("fail")
                return Status.FAILURE

            @bt.root
            @bt.sequence()
            def root(N):
                yield fail

        @bt.tree
        def MainTree():
            @bt.action
            def before(bb: SimpleBlackboard):
                bb.log.append("before")
                return Status.SUCCESS

            @bt.action
            def after(bb: SimpleBlackboard):
                bb.log.append("after")
                return Status.SUCCESS

            @bt.root
            @bt.sequence()
            def root(N):
                yield before
                yield bt.subtree(FailingSubTree)
                yield after

        runner = Runner(MainTree, simple_bb, tb=mock_tb)
        status = await runner.tick()

        assert status == Status.FAILURE
        assert simple_bb.log == ["before", "fail"]


# =============================================================================
# Test: Exception Handling
# =============================================================================


class TestExceptionHandling:
    """Tests for exception handling policies."""

    @pytest.mark.asyncio
    async def test_exception_log_and_continue(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def raise_error(bb: SimpleBlackboard):
                raise ValueError("test error")

            @bt.root
            @bt.sequence()
            def root(N):
                yield raise_error

        runner = Runner(
            Tree, simple_bb, tb=mock_tb, exception_policy=ExceptionPolicy.LOG_AND_CONTINUE
        )
        status = await runner.tick()

        assert status == Status.ERROR

    @pytest.mark.asyncio
    async def test_exception_propagate(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def raise_error(bb: SimpleBlackboard):
                raise ValueError("test error")

            @bt.root
            @bt.sequence()
            def root(N):
                yield raise_error

        runner = Runner(
            Tree, simple_bb, tb=mock_tb, exception_policy=ExceptionPolicy.PROPAGATE
        )

        with pytest.raises(ValueError, match="test error"):
            await runner.tick()


# =============================================================================
# Test: Mermaid Generation
# =============================================================================


class TestMermaidGeneration:
    """Tests for mermaid diagram generation."""

    def test_mermaid_basic_tree(self):
        @bt.tree
        def Tree():
            @bt.action
            def action1(bb):
                return Status.SUCCESS

            @bt.action
            def action2(bb):
                return Status.SUCCESS

            @bt.root
            @bt.sequence()
            def root(N):
                yield action1
                yield action2

        mermaid = Tree.to_mermaid()

        assert "flowchart TD" in mermaid
        assert "Sequence" in mermaid
        assert "action1" in mermaid
        assert "action2" in mermaid
        assert "-->" in mermaid

    def test_mermaid_selector(self):
        @bt.tree
        def Tree():
            @bt.action
            def action1(bb):
                return Status.SUCCESS

            @bt.root
            @bt.selector()
            def root(N):
                yield action1

        mermaid = Tree.to_mermaid()
        assert "Selector" in mermaid

    def test_mermaid_match_with_labeled_edges(self):
        @bt.tree
        def Tree():
            @bt.action
            def handle_image(bb):
                return Status.SUCCESS

            @bt.action
            def handle_move(bb):
                return Status.SUCCESS

            @bt.action
            def handle_default(bb):
                return Status.SUCCESS

            @bt.root
            @bt.sequence()
            def root(N):
                yield bt.match(lambda bb: bb.action)(
                    bt.case(ImageAction)(handle_image),
                    bt.case(MoveAction)(handle_move),
                    bt.defaultcase(handle_default),
                )

        mermaid = Tree.to_mermaid()

        assert "Match" in mermaid
        assert "ImageAction" in mermaid
        assert "MoveAction" in mermaid
        assert "default" in mermaid
        assert '-->|"' in mermaid  # Labeled edges

    def test_mermaid_subtree(self):
        @bt.tree
        def SubTree():
            @bt.action
            def sub_action(bb):
                return Status.SUCCESS

            @bt.root
            @bt.sequence()
            def root(N):
                yield sub_action

        @bt.tree
        def MainTree():
            @bt.action
            def main_action(bb):
                return Status.SUCCESS

            @bt.root
            @bt.sequence()
            def root(N):
                yield main_action
                yield bt.subtree(SubTree)

        mermaid = MainTree.to_mermaid()
        assert "Subtree" in mermaid

    def test_mermaid_do_while(self):
        @bt.tree
        def Tree():
            @bt.condition
            def has_items(bb):
                return bb.value > 0

            @bt.action
            def process(bb):
                return Status.SUCCESS

            @bt.root
            @bt.sequence()
            def root(N):
                yield bt.do_while(has_items)(process)

        mermaid = Tree.to_mermaid()
        
        assert "DoWhile" in mermaid
        assert "has_items" in mermaid
        assert "process" in mermaid
        assert '-->|"condition"|' in mermaid
        assert '-->|"body"|' in mermaid


# =============================================================================
# Test: Recursion Detection
# =============================================================================


class TestRecursionDetection:
    """Tests for recursion detection in tree building."""

    def test_direct_recursion_detected(self):
        with pytest.raises(RecursionError):

            @bt.tree
            def RecursiveTree():
                @bt.sequence()
                def recursive(N):
                    yield recursive  # Direct recursion

                @bt.root
                @bt.sequence()
                def root(N):
                    yield recursive

            # Force expansion by building or generating mermaid
            RecursiveTree.to_mermaid()


# =============================================================================
# Test: Runner
# =============================================================================


class TestRunner:
    """Tests for the Runner class."""

    @pytest.mark.asyncio
    async def test_runner_tick_until_complete(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def three_tick_action(bb: SimpleBlackboard):
                bb.tick_count += 1
                if bb.tick_count < 3:
                    return Status.RUNNING
                return Status.SUCCESS

            @bt.root
            @bt.sequence()
            def root(N):
                yield three_tick_action

        runner = Runner(Tree, simple_bb, tb=mock_tb)
        status = await runner.tick_until_complete()

        assert status == Status.SUCCESS
        assert simple_bb.tick_count == 3

    @pytest.mark.asyncio
    async def test_runner_tick_until_complete_with_timeout(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.action
            def never_completes(bb: SimpleBlackboard):
                bb.tick_count += 1
                return Status.RUNNING

            @bt.root
            @bt.sequence()
            def root(N):
                yield never_completes

        runner = Runner(Tree, simple_bb, tb=mock_tb)

        # Use a mock timebase that advances on each tick
        original_advance = mock_tb.advance

        def advancing_tick():
            original_advance(1.0)

        mock_tb.advance = advancing_tick

        status = await runner.tick_until_complete(timeout=5.0)

        assert status == Status.CANCELLED
        assert simple_bb.tick_count >= 1


# =============================================================================
# Test: Lambda Actions
# =============================================================================


class TestLambdaActions:
    """Tests for inline lambda actions."""

    @pytest.mark.asyncio
    async def test_lambda_action(self, simple_bb, mock_tb):
        @bt.tree
        def Tree():
            @bt.root
            @bt.sequence()
            def root(N):
                yield bt.action(lambda bb: Status.SUCCESS)

        runner = Runner(Tree, simple_bb, tb=mock_tb)
        status = await runner.tick()

        assert status == Status.SUCCESS

    @pytest.mark.asyncio
    async def test_lambda_action_modifies_blackboard(self, simple_bb, mock_tb):
        def set_value(bb):
            bb.value = 42
            return Status.SUCCESS

        @bt.tree
        def Tree():
            @bt.root
            @bt.sequence()
            def root(N):
                yield bt.action(set_value)

        runner = Runner(Tree, simple_bb, tb=mock_tb)
        await runner.tick()

        assert simple_bb.value == 42


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_tree_without_root_raises(self, mock_tb):
        @bt.tree
        def TreeWithoutRoot():
            @bt.action
            def some_action(bb):
                return Status.SUCCESS

        with pytest.raises(ValueError, match="root"):
            Runner(TreeWithoutRoot, SimpleBlackboard(), tb=mock_tb)

    def test_subtree_without_root_raises(self):
        @bt.tree
        def InvalidSubTree():
            @bt.action
            def action(bb):
                return Status.SUCCESS
            # No @bt.root

        with pytest.raises(ValueError, match="root"):
            bt.subtree(InvalidSubTree)

    def test_match_requires_cases(self):
        with pytest.raises(ValueError, match="at least one case"):
            bt.match(lambda bb: bb.action)()

    def test_match_requires_case_specs(self):
        @bt.action
        def some_action(bb):
            return Status.SUCCESS

        with pytest.raises(TypeError, match="CaseSpec"):
            bt.match(lambda bb: bb.action)(some_action)