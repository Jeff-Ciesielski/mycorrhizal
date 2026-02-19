#!/usr/bin/env python3
"""Tests for Mycelium BT-to-PN Compiler."""

import sys
sys.path.insert(0, "src")

import pytest
import asyncio
from pydantic import BaseModel
from typing import List

from mycorrhizal.rhizomorph.core import bt, Status, Runner as BTRunner
from mycorrhizal.mycelium import compile_bt_to_pn, BTtoPNCompiler, PNRunner
from mycorrhizal.common.timebase import MonotonicClock


# =============================================================================
# Test Blackboard
# =============================================================================


class TestBlackboard(BaseModel):
    """Blackboard for testing."""
    battery_level: float = 50.0
    action_taken: str = ""
    attempts: int = 0
    events: List[str] = []
    current_event: object = None
    command: str = ""


# =============================================================================
# Test Trees
# =============================================================================


@bt.tree
def SimpleActionBT():
    """Simple tree with single action."""

    @bt.action
    async def my_action(bb: TestBlackboard) -> Status:
        bb.action_taken = "done"
        return Status.SUCCESS

    @bt.root
    @bt.sequence
    def root():
        yield my_action


@bt.tree
def SequenceBT():
    """Tree with sequence of actions."""

    @bt.action
    async def action1(bb: TestBlackboard) -> Status:
        bb.events.append("action1")
        return Status.SUCCESS

    @bt.action
    async def action2(bb: TestBlackboard) -> Status:
        bb.events.append("action2")
        return Status.SUCCESS

    @bt.root
    @bt.sequence
    def root():
        yield action1
        yield action2


@bt.tree
def SelectorBT():
    """Tree with selector."""

    @bt.condition
    def check_high(bb: TestBlackboard) -> bool:
        bb.events.append("check_high")
        return bb.battery_level > 75

    @bt.condition
    def check_low(bb: TestBlackboard) -> bool:
        bb.events.append("check_low")
        return bb.battery_level > 25

    @bt.action
    async def fallback(bb: TestBlackboard) -> Status:
        bb.events.append("fallback")
        bb.action_taken = "fallback"
        return Status.SUCCESS

    @bt.root
    @bt.selector
    def root():
        yield check_high
        yield check_low
        yield fallback


@bt.tree
def ParallelBT():
    """Tree with parallel execution."""

    @bt.action
    async def task1(bb: TestBlackboard) -> Status:
        bb.events.append("task1")
        return Status.SUCCESS

    @bt.action
    async def task2(bb: TestBlackboard) -> Status:
        bb.events.append("task2")
        return Status.SUCCESS

    @bt.action
    async def task3(bb: TestBlackboard) -> Status:
        bb.events.append("task3")
        return Status.FAILURE

    @bt.root
    @bt.parallel(success_threshold=2)
    def root():
        yield task1
        yield task2
        yield task3


@bt.tree
def RetryBT():
    """Tree with retry decorator."""

    @bt.action
    async def flaky_action(bb: TestBlackboard) -> Status:
        bb.attempts += 1
        bb.events.append(f"attempt_{bb.attempts}")
        if bb.attempts < 3:
            return Status.FAILURE
        bb.action_taken = "succeeded"
        return Status.SUCCESS

    @bt.root
    @bt.sequence
    def root():
        yield bt.retry(max_attempts=3)(flaky_action)


@bt.tree
def InverterBT():
    """Tree with inverter decorator."""

    @bt.condition
    def is_low(bb: TestBlackboard) -> bool:
        return bb.battery_level < 20

    @bt.action
    async def alert(bb: TestBlackboard) -> Status:
        bb.action_taken = "alerted"
        return Status.SUCCESS

    @bt.root
    @bt.sequence
    def root():
        yield bt.inverter()(is_low)
        yield alert


@bt.tree
def TimeoutBT():
    """Tree with timeout decorator."""

    @bt.action
    async def slow_action(bb: TestBlackboard) -> Status:
        bb.attempts += 1
        if bb.attempts < 2:
            return Status.RUNNING
        bb.action_taken = "completed"
        return Status.SUCCESS

    @bt.root
    @bt.sequence
    def root():
        yield bt.timeout(seconds=1.0)(slow_action)


@bt.tree
def DoWhileBT():
    """Tree with do-while decorator."""

    @bt.condition
    def count_below_3(bb: TestBlackboard) -> bool:
        return bb.attempts < 3

    @bt.action
    async def increment(bb: TestBlackboard) -> Status:
        bb.attempts += 1
        bb.events.append(f"count_{bb.attempts}")
        return Status.SUCCESS

    @bt.root
    @bt.sequence
    def root():
        yield bt.do_while(count_below_3)(increment)


@bt.tree
def GateBT():
    """Tree with gate decorator."""

    @bt.condition
    def is_enabled(bb: TestBlackboard) -> bool:
        return bb.battery_level > 50

    @bt.action
    async def gated_action(bb: TestBlackboard) -> Status:
        bb.action_taken = "executed"
        return Status.SUCCESS

    @bt.root
    @bt.sequence
    def root():
        yield bt.gate(is_enabled)(gated_action)


@bt.tree
def WhenBT():
    """Tree with when decorator."""

    @bt.condition
    def feature_enabled(bb: TestBlackboard) -> bool:
        return bb.battery_level > 50

    @bt.action
    async def optional_action(bb: TestBlackboard) -> Status:
        bb.action_taken = "optional"
        return Status.SUCCESS

    @bt.action
    async def next_action(bb: TestBlackboard) -> Status:
        bb.action_taken = "next"
        return Status.SUCCESS

    @bt.root
    @bt.sequence
    def root():
        yield bt.when(feature_enabled)(optional_action)
        yield next_action


@bt.tree
def RateLimitBT():
    """Tree with rate limit decorator."""

    @bt.action
    async def limited_action(bb: TestBlackboard) -> Status:
        bb.attempts += 1
        bb.action_taken = f"call_{bb.attempts}"
        return Status.SUCCESS

    @bt.root
    @bt.sequence
    def root():
        yield bt.ratelimit(hz=2.0)(limited_action)


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.asyncio
async def test_compile_simple_action():
    """Test compiling a simple action tree."""
    pn_spec = compile_bt_to_pn(SimpleActionBT)

    assert pn_spec is not None
    assert pn_spec.name == "CompiledBT"
    assert len(pn_spec.places) > 0
    assert len(pn_spec.transitions) > 0


@pytest.mark.asyncio
async def test_compile_with_custom_name():
    """Test compiling with a custom net name."""
    pn_spec = compile_bt_to_pn(SimpleActionBT, name="MyCustomNet")

    assert pn_spec.name == "MyCustomNet"


@pytest.mark.asyncio
async def test_compile_with_custom_delays():
    """Test compiling with custom delay settings."""
    compiler = BTtoPNCompiler(
        name="CustomDelays",
        running_delay=0.05,
        retry_delay=0.1
    )
    pn_spec = compiler.compile(SimpleActionBT.root)

    assert pn_spec is not None


@pytest.mark.asyncio
async def test_run_compiled_simple_action():
    """Test running a compiled simple action tree."""
    bb = TestBlackboard()
    tb = MonotonicClock()

    # Compile
    pn_spec = compile_bt_to_pn(SimpleActionBT)

    # Create a wrapper function with _spec attribute (expected by PNRunner)
    def net_func():
        pass
    net_func._spec = pn_spec

    # Run
    runner = PNRunner(net_func, blackboard=bb)
    await runner.start(tb)

    # Inject token
    entry_key = (pn_spec.name, "entry")
    runner.runtime.places[entry_key].add_token("start")

    # Wait for completion
    await asyncio.sleep(0.2)

    # Stop
    await runner.stop(timeout=2)

    # Check result
    assert bb.action_taken == "done"


@pytest.mark.asyncio
async def test_run_compiled_sequence():
    """Test running a compiled sequence tree."""
    bb = TestBlackboard()
    tb = MonotonicClock()

    pn_spec = compile_bt_to_pn(SequenceBT)

    def net_func():
        pass
    net_func._spec = pn_spec

    runner = PNRunner(net_func, blackboard=bb)
    await runner.start(tb)

    entry_key = (pn_spec.name, "entry")
    runner.runtime.places[entry_key].add_token("start")

    await asyncio.sleep(0.3)
    await runner.stop(timeout=2)

    assert len(bb.events) == 2
    assert "action1" in bb.events
    assert "action2" in bb.events


@pytest.mark.asyncio
async def test_run_compiled_selector():
    """Test running a compiled selector tree."""
    # Test with low battery (should take fallback)
    bb = TestBlackboard(battery_level=10.0)
    tb = MonotonicClock()

    pn_spec = compile_bt_to_pn(SelectorBT)

    def net_func():
        pass
    net_func._spec = pn_spec

    runner = PNRunner(net_func, blackboard=bb)
    await runner.start(tb)

    entry_key = (pn_spec.name, "entry")
    runner.runtime.places[entry_key].add_token("start")

    await asyncio.sleep(0.3)
    await runner.stop(timeout=2)

    assert bb.action_taken == "fallback"
    assert "check_high" in bb.events
    assert "check_low" in bb.events


@pytest.mark.asyncio
async def test_run_compiled_selector_first_match():
    """Test selector where first condition matches."""
    bb = TestBlackboard(battery_level=80.0)
    tb = MonotonicClock()

    pn_spec = compile_bt_to_pn(SelectorBT)

    def net_func():
        pass
    net_func._spec = pn_spec

    runner = PNRunner(net_func, blackboard=bb)
    await runner.start(tb)

    entry_key = (pn_spec.name, "entry")
    runner.runtime.places[entry_key].add_token("start")

    await asyncio.sleep(0.3)
    await runner.stop(timeout=2)

    # Should succeed on first condition
    assert "check_high" in bb.events
    # Should not check second condition
    assert "check_low" not in bb.events


@pytest.mark.asyncio
async def test_run_compiled_parallel():
    """Test running a compiled parallel tree."""
    bb = TestBlackboard()
    tb = MonotonicClock()

    pn_spec = compile_bt_to_pn(ParallelBT)

    def net_func():
        pass
    net_func._spec = pn_spec

    runner = PNRunner(net_func, blackboard=bb)
    await runner.start(tb)

    entry_key = (pn_spec.name, "entry")
    runner.runtime.places[entry_key].add_token("start")

    await asyncio.sleep(0.3)
    await runner.stop(timeout=2)

    # All tasks should execute
    assert "task1" in bb.events
    assert "task2" in bb.events
    assert "task3" in bb.events


@pytest.mark.asyncio
async def test_run_compiled_retry():
    """Test running a compiled retry tree."""
    bb = TestBlackboard(attempts=0)
    tb = MonotonicClock()

    pn_spec = compile_bt_to_pn(RetryBT, retry_delay=0.01)

    def net_func():
        pass
    net_func._spec = pn_spec

    runner = PNRunner(net_func, blackboard=bb)
    await runner.start(tb)

    entry_key = (pn_spec.name, "entry")
    runner.runtime.places[entry_key].add_token("start")

    await asyncio.sleep(0.5)
    await runner.stop(timeout=2)

    assert bb.attempts == 3
    assert bb.action_taken == "succeeded"


@pytest.mark.asyncio
async def test_run_compiled_inverter():
    """Test running a compiled inverter tree."""
    bb = TestBlackboard(battery_level=80.0)
    tb = MonotonicClock()

    pn_spec = compile_bt_to_pn(InverterBT)

    def net_func():
        pass
    net_func._spec = pn_spec

    runner = PNRunner(net_func, blackboard=bb)
    await runner.start(tb)

    entry_key = (pn_spec.name, "entry")
    runner.runtime.places[entry_key].add_token("start")

    await asyncio.sleep(0.3)
    await runner.stop(timeout=2)

    assert bb.action_taken == "alerted"


@pytest.mark.asyncio
async def test_run_compiled_timeout():
    """Test running a compiled timeout tree."""
    bb = TestBlackboard(attempts=0)
    tb = MonotonicClock()

    # 1 second timeout, action completes in 2 RUNNING ticks
    pn_spec = compile_bt_to_pn(TimeoutBT, running_delay=0.1)

    def net_func():
        pass
    net_func._spec = pn_spec

    runner = PNRunner(net_func, blackboard=bb)
    await runner.start(tb)

    entry_key = (pn_spec.name, "entry")
    runner.runtime.places[entry_key].add_token("start")

    await asyncio.sleep(0.5)
    await runner.stop(timeout=2)

    assert bb.action_taken == "completed"


@pytest.mark.asyncio
async def test_run_compiled_do_while():
    """Test running a compiled do-while tree."""
    bb = TestBlackboard(attempts=0)
    tb = MonotonicClock()

    pn_spec = compile_bt_to_pn(DoWhileBT, running_delay=0.05)

    def net_func():
        pass
    net_func._spec = pn_spec

    runner = PNRunner(net_func, blackboard=bb)
    await runner.start(tb)

    entry_key = (pn_spec.name, "entry")
    runner.runtime.places[entry_key].add_token("start")

    await asyncio.sleep(0.5)
    await runner.stop(timeout=2)

    assert bb.attempts == 3
    assert len(bb.events) == 3


@pytest.mark.asyncio
async def test_run_compiled_gate_open():
    """Test gate when condition is true."""
    bb = TestBlackboard(battery_level=75.0)
    tb = MonotonicClock()

    pn_spec = compile_bt_to_pn(GateBT)

    def net_func():
        pass
    net_func._spec = pn_spec

    runner = PNRunner(net_func, blackboard=bb)
    await runner.start(tb)

    entry_key = (pn_spec.name, "entry")
    runner.runtime.places[entry_key].add_token("start")

    await asyncio.sleep(0.3)
    await runner.stop(timeout=2)

    assert bb.action_taken == "executed"


@pytest.mark.asyncio
async def test_run_compiled_gate_closed():
    """Test gate when condition is false."""
    bb = TestBlackboard(battery_level=25.0)
    tb = MonotonicClock()

    pn_spec = compile_bt_to_pn(GateBT)

    def net_func():
        pass
    net_func._spec = pn_spec

    runner = PNRunner(net_func, blackboard=bb)
    await runner.start(tb)

    entry_key = (pn_spec.name, "entry")
    runner.runtime.places[entry_key].add_token("start")

    await asyncio.sleep(0.3)
    await runner.stop(timeout=2)

    assert bb.action_taken == ""


@pytest.mark.asyncio
async def test_run_compiled_when_true():
    """Test when when condition is true."""
    bb = TestBlackboard(battery_level=75.0)
    tb = MonotonicClock()

    pn_spec = compile_bt_to_pn(WhenBT)

    def net_func():
        pass
    net_func._spec = pn_spec

    runner = PNRunner(net_func, blackboard=bb)
    await runner.start(tb)

    entry_key = (pn_spec.name, "entry")
    runner.runtime.places[entry_key].add_token("start")

    await asyncio.sleep(0.3)
    await runner.stop(timeout=2)

    assert bb.action_taken == "next"


@pytest.mark.asyncio
async def test_run_compiled_when_false():
    """Test when when condition is false."""
    bb = TestBlackboard(battery_level=25.0)
    tb = MonotonicClock()

    pn_spec = compile_bt_to_pn(WhenBT)

    def net_func():
        pass
    net_func._spec = pn_spec

    runner = PNRunner(net_func, blackboard=bb)
    await runner.start(tb)

    entry_key = (pn_spec.name, "entry")
    runner.runtime.places[entry_key].add_token("start")

    await asyncio.sleep(0.3)
    await runner.stop(timeout=2)

    # Should skip optional action but still run next
    assert bb.action_taken == "next"


@pytest.mark.asyncio
async def test_compiler_creates_valid_structure():
    """Test that compiler creates valid PN structure."""
    pn_spec = compile_bt_to_pn(SimpleActionBT)

    # Should have entry, success, failure, running places
    place_names = set(pn_spec.places.keys())
    assert "entry" in place_names
    assert "success" in place_names
    assert "failure" in place_names
    assert "running" in place_names

    # Should have at least some transitions
    assert len(pn_spec.transitions) > 0


@pytest.mark.asyncio
async def test_compiler_no_orphan_places():
    """Test that compiler creates no orphan places (all places connected to transitions)."""
    from mycorrhizal.rhizomorph.core import bt, Status

    @bt.tree
    def ComplexBT():
        """Tree with sequences and selectors to test for orphan places."""

        @bt.action
        async def action1(bb):
            return Status.SUCCESS

        @bt.action
        async def action2(bb):
            return Status.SUCCESS

        @bt.sequence
        def seq1():
            yield action1
            yield action2

        @bt.selector
        def sel1():
            yield seq1
            yield action2

        @bt.root
        @bt.selector
        def root():
            yield seq1
            yield sel1

    pn_spec = compile_bt_to_pn(ComplexBT)

    # Build a set of all places that are sources or targets of arcs
    connected_places = set()
    for arc in pn_spec.arcs:
        # Get the local name from the place
        if hasattr(arc.source, 'local_name'):
            connected_places.add(arc.source.local_name)
        if hasattr(arc.target, 'local_name'):
            connected_places.add(arc.target.local_name)

    # All declared places should be connected (no orphans)
    for place_name in pn_spec.places.keys():
        assert place_name in connected_places, f"Orphan place found: {place_name} (not connected to any arc)"


@pytest.mark.asyncio
async def test_compiler_export_status_token():
    """Test that StatusToken is exported."""
    from mycorrhizal.mycelium import StatusToken

    token = StatusToken(status=Status.SUCCESS, data="test")
    assert token.status == Status.SUCCESS
    assert token.data == "test"


@pytest.mark.asyncio
async def test_compiler_export_match_token():
    """Test that MatchToken is exported."""
    from mycorrhizal.mycelium import MatchToken, StatusToken

    base = StatusToken(status=Status.SUCCESS)
    token = MatchToken(base=base, matched_case_idx=1, key_value="test")
    assert token.matched_case_idx == 1
    assert token.key_value == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
