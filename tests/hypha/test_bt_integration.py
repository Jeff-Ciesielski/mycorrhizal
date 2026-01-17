#!/usr/bin/env python3
"""
Tests for BT-in-PN (Behavior Tree in Petri Net) integration.
"""

import pytest
import asyncio
from enum import Enum, auto
from typing import Any, List
from pydantic import BaseModel

from mycorrhizal.mycelium import pn, PNRunner as Runner, PlaceType
from mycorrhizal.rhizomorph.core import bt, Status
from mycorrhizal.common.timebase import MonotonicClock, CycleClock


# ==================================================================================
# Test Fixtures
# ==================================================================================


class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class Task(BaseModel):
    task_id: str
    priority: TaskPriority
    payload: str
    valid: bool = True


class Blackboard(BaseModel):
    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}
    tasks_processed: int = 0
    tokens_routed: int = 0


# ==================================================================================
# Basic Routing Tests
# ==================================================================================


@pytest.mark.asyncio
async def test_bt_basic_routing():
    """Test basic BT-in-PN routing with token mode."""

    @bt.tree
    def SimpleRouter():
        @bt.action
        async def route_high(bb):
            token = bb.pn_ctx.current_token
            pn_ctx = bb.pn_ctx
            print("[BT] route_high called with token: " + str(token))
            if isinstance(token, Task) and token.priority == TaskPriority.HIGH:
                print("[BT] Routing HIGH task to high_q")
                pn_ctx.route_to(pn_ctx.places.high_q)
                return Status.SUCCESS
            return Status.FAILURE

        @bt.action
        async def route_low(bb):
            token = bb.pn_ctx.current_token
            pn_ctx = bb.pn_ctx
            print("[BT] route_low called with token: " + str(token))
            if isinstance(token, Task) and token.priority == TaskPriority.LOW:
                print("[BT] Routing LOW task to low_q")
                pn_ctx.route_to(pn_ctx.places.low_q)
                return Status.SUCCESS
            return Status.FAILURE

        @bt.action
        async def reject(bb):
            pn_ctx = bb.pn_ctx
            print("[BT] reject called")
            pn_ctx.reject_token("Unknown priority")
            return Status.SUCCESS

        @bt.root
        @bt.selector
        def root():
            yield route_high
            yield route_low
            yield reject

    @pn.net
    def TestNet(builder):
        input_q = builder.place("input_q", type=PlaceType.QUEUE)
        high_q = builder.place("high_q", type=PlaceType.QUEUE)
        low_q = builder.place("low_q", type=PlaceType.QUEUE)

        @builder.transition(bt=SimpleRouter, bt_mode="token", outputs=[high_q, low_q])
        async def route(consumed, bb, timebase):
            pass

        builder.arc(input_q, route)
        builder.arc(route, high_q)
        builder.arc(route, low_q)

    bb = Blackboard()
    timebase = CycleClock()
    runner = Runner(TestNet, bb)
    await runner.start(timebase)

    # Add test tokens
    runtime = runner.runtime
    # Debug: print actual keys
    print(f"Available place keys: {list(runtime.places.keys())}")

    # Use the actual key from the runtime
    input_place_key = None
    for key in runtime.places.keys():
        if "input_q" in key:
            input_place_key = key
            break

    if input_place_key is None:
        raise KeyError(f"Could not find input_q in {list(runtime.places.keys())}")

    input_place = runtime.places[input_place_key]

    high_task = Task(task_id="high-1", priority=TaskPriority.HIGH, payload="High priority")
    low_task = Task(task_id="low-1", priority=TaskPriority.LOW, payload="Low priority")

    input_place.add_token(high_task)
    input_place.add_token(low_task)

    await asyncio.sleep(3.0)

    # Check routing - find the actual keys
    high_place_key = None
    low_place_key = None
    for key in runtime.places.keys():
        if "high_q" in key:
            high_place_key = key
        if "low_q" in key:
            low_place_key = key

    high_place = runtime.places[high_place_key]
    low_place = runtime.places[low_place_key]

    print(f"[TEST] high_place has {len(high_place.tokens)} tokens")
    print(f"[TEST] low_place has {len(low_place.tokens)} tokens")
    print(f"[TEST] input_place still has {len(input_place.tokens)} tokens")

    # At least one token should be routed
    total_routed = len(high_place.tokens) + len(low_place.tokens)
    assert total_routed >= 1, f"Expected at least 1 token to be routed, got {total_routed}"

    if len(high_place.tokens) > 0:
        assert list(high_place.tokens)[0].task_id == "high-1"
    if len(low_place.tokens) > 0:
        assert list(low_place.tokens)[0].task_id == "low-1"

    await runner.stop(timeout=1)


@pytest.mark.asyncio
async def test_bt_token_rejection():
    """Test BT rejecting tokens (returning to input)."""

    @bt.tree
    def Validator():
        @bt.action
        async def validate(bb):
            token = bb.pn_ctx.current_token
            pn_ctx = bb.pn_ctx
            if isinstance(token, Task) and token.valid:
                pn_ctx.route_to(pn_ctx.places.validated_q)
                return Status.SUCCESS
            else:
                pn_ctx.reject_token("Invalid task")
                return Status.SUCCESS

        @bt.root
        @bt.selector
        def root():
            yield validate

    @pn.net
    def TestNet(builder):
        input_q = builder.place("input_q", type=PlaceType.QUEUE)
        validated_q = builder.place("validated_q", type=PlaceType.QUEUE)

        @builder.transition(bt=Validator, bt_mode="token", outputs=[validated_q])
        async def validate(consumed, bb, timebase):
            pass

        builder.arc(input_q, validate)
        builder.arc(validate, validated_q)

    bb = Blackboard()
    timebase = CycleClock()
    runner = Runner(TestNet, bb)
    await runner.start(timebase)

    runtime = runner.runtime
    input_place = runtime.places[("TestNet", "input_q")]

    valid_task = Task(task_id="valid-1", priority=TaskPriority.HIGH, payload="Valid", valid=True)
    invalid_task = Task(task_id="invalid-1", priority=TaskPriority.LOW, payload="Invalid", valid=False)

    input_place.add_token(valid_task)
    input_place.add_token(invalid_task)

    await asyncio.sleep(1.0)

    # Valid token should be in validated queue
    validated_place = runtime.places[("TestNet", "validated_q")]
    assert len(validated_place.tokens) == 1
    assert list(validated_place.tokens)[0].task_id == "valid-1"

    # Invalid token should be back in input queue
    # (This tests the rejection logic)
    assert len(input_place.tokens) >= 1

    await runner.stop(timeout=1)


@pytest.mark.asyncio
async def test_bt_enrichment():
    """Test BT enriching tokens during routing."""

    class EnrichedTask(BaseModel):
        task_id: str
        original: Task
        metadata: str

    @bt.tree
    def Enricher():
        @bt.action
        async def enrich(bb):
            token = bb.pn_ctx.current_token
            pn_ctx = bb.pn_ctx
            if isinstance(token, Task):
                enriched = EnrichedTask(
                    task_id=token.task_id,
                    original=token,
                    metadata=f"Enriched at {pn_ctx.timebase.now():.2f}"
                )
                pn_ctx.route_to(pn_ctx.places.enriched_q, token=enriched)
                return Status.SUCCESS
            return Status.FAILURE

        @bt.root
        @bt.selector
        def root():
            yield enrich

    @pn.net
    def TestNet(builder):
        input_q = builder.place("input_q", type=PlaceType.QUEUE)
        enriched_q = builder.place("enriched_q", type=PlaceType.QUEUE)

        @builder.transition(bt=Enricher, bt_mode="token", outputs=[enriched_q])
        async def enrich(consumed, bb, timebase):
            pass

        builder.arc(input_q, enrich)
        builder.arc(enrich, enriched_q)

    bb = Blackboard()
    timebase = CycleClock()
    runner = Runner(TestNet, bb)
    await runner.start(timebase)

    runtime = runner.runtime
    input_place = runtime.places[("TestNet", "input_q")]

    task = Task(task_id="task-1", priority=TaskPriority.HIGH, payload="Test")
    input_place.add_token(task)

    await asyncio.sleep(1.0)

    # Check enrichment
    enriched_place = runtime.places[("TestNet", "enriched_q")]
    assert len(enriched_place.tokens) == 1

    enriched_token = list(enriched_place.tokens)[0]
    assert isinstance(enriched_token, EnrichedTask)
    assert enriched_token.task_id == "task-1"
    assert enriched_token.original.task_id == "task-1"
    assert "Enriched at" in enriched_token.metadata

    await runner.stop(timeout=1)


@pytest.mark.asyncio
async def test_bt_with_side_effects():
    """Test BT performing side effects (blackboard updates)."""

    @bt.tree
    def Counter():
        @bt.action
        async def count_and_route(bb):
            pn_ctx = bb.pn_ctx
            bb.tasks_processed += 1
            pn_ctx.route_to(pn_ctx.places.output_q)
            return Status.SUCCESS

        @bt.root
        @bt.selector
        def root():
            yield count_and_route

    @pn.net
    def TestNet(builder):
        input_q = builder.place("input_q", type=PlaceType.QUEUE)
        output_q = builder.place("output_q", type=PlaceType.QUEUE)

        @builder.transition(bt=Counter, bt_mode="token", outputs=[output_q])
        async def count(consumed, bb, timebase):
            pass

        builder.arc(input_q, count)
        builder.arc(count, output_q)

    bb = Blackboard()
    timebase = CycleClock()
    runner = Runner(TestNet, bb)
    await runner.start(timebase)

    runtime = runner.runtime
    input_place = runtime.places[("TestNet", "input_q")]

    # Add multiple tasks
    for i in range(3):
        task = Task(task_id=f"task-{i}", priority=TaskPriority.LOW, payload=f"Task {i}")
        input_place.add_token(task)

    await asyncio.sleep(1.0)

    # Check counter
    assert bb.tasks_processed == 3

    output_place = runtime.places[("TestNet", "output_q")]
    assert len(output_place.tokens) == 3

    await runner.stop(timeout=1)


@pytest.mark.asyncio
async def test_backward_compatibility():
    """Test that vanilla transitions still work (no BT)."""

    @pn.net
    def TestNet(builder):
        input_q = builder.place("input_q", type=PlaceType.QUEUE)
        output_q = builder.place("output_q", type=PlaceType.QUEUE)

        @builder.transition()
        async def vanilla_transition(consumed, bb, timebase):
            for token in consumed:
                bb.tasks_processed += 1
                yield {output_q: token}

        builder.arc(input_q, vanilla_transition)
        builder.arc(vanilla_transition, output_q)

    bb = Blackboard()
    timebase = CycleClock()
    runner = Runner(TestNet, bb)
    await runner.start(timebase)

    runtime = runner.runtime
    input_place = runtime.places[("TestNet", "input_q")]

    task = Task(task_id="task-1", priority=TaskPriority.HIGH, payload="Test")
    input_place.add_token(task)

    await asyncio.sleep(1.0)

    # Check vanilla transition worked
    assert bb.tasks_processed == 1
    output_place = runtime.places[("TestNet", "output_q")]
    assert len(output_place.tokens) == 1

    await runner.stop(timeout=1)


@pytest.mark.asyncio
async def test_bt_with_composite_nodes():
    """Test BT-in-PN with composite nodes (sequence, selector)."""

    @bt.tree
    def ComplexRouter():
        @bt.condition
        def is_critical(bb):
            return isinstance(bb.pn_ctx.current_token, Task) and bb.pn_ctx.current_token.priority == TaskPriority.CRITICAL

        @bt.action
        async def route_critical(bb):
            pn_ctx = bb.pn_ctx
            pn_ctx.route_to(pn_ctx.places.critical_q)
            return Status.SUCCESS

        @bt.action
        async def route_normal(bb):
            pn_ctx = bb.pn_ctx
            pn_ctx.route_to(pn_ctx.places.normal_q)
            return Status.SUCCESS

        # Define sequence using proper syntax
        @bt.sequence
        def critical_flow():
            yield is_critical
            yield route_critical

        @bt.root
        @bt.selector
        def root():
            yield critical_flow
            yield route_normal

    @pn.net
    def TestNet(builder):
        input_q = builder.place("input_q", type=PlaceType.QUEUE)
        critical_q = builder.place("critical_q", type=PlaceType.QUEUE)
        normal_q = builder.place("normal_q", type=PlaceType.QUEUE)

        @builder.transition(bt=ComplexRouter, bt_mode="token", outputs=[critical_q, normal_q])
        async def route(consumed, bb, timebase):
            pass

        builder.arc(input_q, route)
        builder.arc(route, critical_q)
        builder.arc(route, normal_q)

    bb = Blackboard()
    timebase = CycleClock()
    runner = Runner(TestNet, bb)
    await runner.start(timebase)

    runtime = runner.runtime
    input_place = runtime.places[("TestNet", "input_q")]

    critical_task = Task(task_id="critical-1", priority=TaskPriority.CRITICAL, payload="Critical")
    normal_task = Task(task_id="normal-1", priority=TaskPriority.LOW, payload="Normal")

    input_place.add_token(critical_task)
    input_place.add_token(normal_task)

    await asyncio.sleep(1.0)

    critical_place = runtime.places[("TestNet", "critical_q")]
    normal_place = runtime.places[("TestNet", "normal_q")]

    assert len(critical_place.tokens) == 1
    assert len(normal_place.tokens) == 1

    await runner.stop(timeout=1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
