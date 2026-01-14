#!/usr/bin/env python3
"""Tests for decorator syntax variations (with and without parentheses)"""

import sys
sys.path.insert(0, "src")

import pytest
from mycorrhizal.rhizomorph.core import bt, Status, Runner, NodeSpecKind
from pydantic import BaseModel


class TestBB(BaseModel):
    value: int = 0


# Test @bt.sequence without parens
@bt.tree
def SequenceNoParens():
    @bt.action
    async def increment(bb: TestBB) -> Status:
        bb.value += 1
        return Status.SUCCESS

    @bt.root
    @bt.sequence
    def root():
        yield increment


# Test @bt.sequence() with parens but no args
@bt.tree
def SequenceWithEmptyParens():
    @bt.action
    async def increment(bb: TestBB) -> Status:
        bb.value += 1
        return Status.SUCCESS

    @bt.root
    @bt.sequence()
    def root():
        yield increment


# Test @bt.sequence(memory=False) with args
@bt.tree
def SequenceWithArgs():
    @bt.action
    async def increment(bb: TestBB) -> Status:
        bb.value += 1
        return Status.SUCCESS

    @bt.root
    @bt.sequence(memory=False)
    def root():
        yield increment


# Test @bt.selector without parens
@bt.tree
def SelectorNoParens():
    @bt.action
    async def failing_action(bb: TestBB) -> Status:
        return Status.FAILURE

    @bt.action
    async def success_action(bb: TestBB) -> Status:
        return Status.SUCCESS

    @bt.root
    @bt.selector
    def root():
        yield failing_action
        yield success_action


# Test @bt.selector() with parens but no args
@bt.tree
def SelectorWithEmptyParens():
    @bt.action
    async def failing_action(bb: TestBB) -> Status:
        return Status.FAILURE

    @bt.action
    async def success_action(bb: TestBB) -> Status:
        return Status.SUCCESS

    @bt.root
    @bt.selector()
    def root():
        yield failing_action
        yield success_action


# Test @bt.selector(reactive=True) with args
@bt.tree
def SelectorWithArgs():
    @bt.action
    async def failing_action(bb: TestBB) -> Status:
        return Status.FAILURE

    @bt.action
    async def success_action(bb: TestBB) -> Status:
        return Status.SUCCESS

    @bt.root
    @bt.selector(reactive=True)
    def root():
        yield failing_action
        yield success_action


@pytest.mark.asyncio
async def test_sequence_no_parens():
    """Test @bt.sequence without parentheses"""
    bb = TestBB()
    runner = Runner(SequenceNoParens, bb)
    result = await runner.tick_until_complete(timeout=1.0)
    assert result == Status.SUCCESS
    assert bb.value == 1


@pytest.mark.asyncio
async def test_sequence_with_empty_parens():
    """Test @bt.sequence() with empty parentheses"""
    bb = TestBB()
    runner = Runner(SequenceWithEmptyParens, bb)
    result = await runner.tick_until_complete(timeout=1.0)
    assert result == Status.SUCCESS
    assert bb.value == 1


@pytest.mark.asyncio
async def test_sequence_with_args():
    """Test @bt.sequence(memory=False) with arguments"""
    bb = TestBB()
    runner = Runner(SequenceWithArgs, bb)
    result = await runner.tick_until_complete(timeout=1.0)
    assert result == Status.SUCCESS
    assert bb.value == 1


@pytest.mark.asyncio
async def test_selector_no_parens():
    """Test @bt.selector without parentheses"""
    bb = TestBB()
    runner = Runner(SelectorNoParens, bb)
    result = await runner.tick_until_complete(timeout=1.0)
    # Should succeed because second child succeeds
    assert result == Status.SUCCESS


@pytest.mark.asyncio
async def test_selector_with_empty_parens():
    """Test @bt.selector() with empty parentheses"""
    bb = TestBB()
    runner = Runner(SelectorWithEmptyParens, bb)
    result = await runner.tick_until_complete(timeout=1.0)
    assert result == Status.SUCCESS


@pytest.mark.asyncio
async def test_selector_with_args():
    """Test @bt.selector(reactive=True) with arguments"""
    bb = TestBB()
    runner = Runner(SelectorWithArgs, bb)
    result = await runner.tick_until_complete(timeout=1.0)
    assert result == Status.SUCCESS


def test_all_trees_have_root_spec():
    """Verify that all trees have root node specs with correct kinds"""
    # @bt.root converts the function to a NodeSpec, so we check the spec directly
    assert SequenceNoParens.root.kind == NodeSpecKind.SEQUENCE
    assert SequenceWithEmptyParens.root.kind == NodeSpecKind.SEQUENCE
    assert SequenceWithArgs.root.kind == NodeSpecKind.SEQUENCE

    assert SelectorNoParens.root.kind == NodeSpecKind.SELECTOR
    assert SelectorWithEmptyParens.root.kind == NodeSpecKind.SELECTOR
    assert SelectorWithArgs.root.kind == NodeSpecKind.SELECTOR


def test_node_spec_memory_flags():
    """Verify that memory flags are set correctly"""
    # SequenceWithArgs uses memory=False
    assert SequenceWithArgs.root.payload['memory'] is False
    # Others use default (True)
    assert SequenceNoParens.root.payload['memory'] is True
    assert SequenceWithEmptyParens.root.payload['memory'] is True


def test_node_spec_reactive_flags():
    """Verify that reactive flags are set correctly"""
    # SelectorWithArgs uses reactive=True
    assert SelectorWithArgs.root.payload['reactive'] is True
    # Others use default (False)
    assert SelectorNoParens.root.payload['reactive'] is False
    assert SelectorWithEmptyParens.root.payload['reactive'] is False
