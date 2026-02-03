#!/usr/bin/env python3
"""
Tests for TreeBuilder Mermaid export and serialization.

Run with: pytest tests/rhizomorph/test_tree_builder_features.py -v
"""

import asyncio
import json

from mycorrhizal.rhizomorph import TreeBuilder, Runner, Status


# ============================================================================
# Test Fixtures
# ============================================================================


class SimpleBlackboard:
    """Simple blackboard for tests"""
    value: int = 0


# ============================================================================
# Mermaid Export Tests
# ============================================================================


def test_simple_tree_mermaid_export():
    """Test Mermaid export for a simple tree"""
    builder = TreeBuilder("SimpleTree")

    async def action_a(bb):
        return Status.SUCCESS

    a = builder.action("action_a", action_a)
    tree = builder.build(a)

    mermaid = tree.to_mermaid()

    assert "flowchart TD" in mermaid
    assert "action_a" in mermaid
    assert "ACTION" in mermaid


def test_sequence_mermaid_export():
    """Test Mermaid export for a sequence"""
    builder = TreeBuilder("SequenceTree")

    async def action_a(bb):
        return Status.SUCCESS

    async def action_b(bb):
        return Status.SUCCESS

    a = builder.action("action_a", action_a)
    b = builder.action("action_b", action_b)
    seq = builder.sequence(a, b, memory=True)
    tree = builder.build(seq)

    mermaid = tree.to_mermaid()

    assert "Sequence" in mermaid
    assert "action_a" in mermaid
    assert "action_b" in mermaid


def test_selector_mermaid_export():
    """Test Mermaid export for a selector"""
    builder = TreeBuilder("SelectorTree")

    async def action_a(bb):
        return Status.SUCCESS

    async def action_b(bb):
        return Status.FAILURE

    a = builder.action("action_a", action_a)
    b = builder.action("action_b", action_b)
    sel = builder.selector(a, b, memory=False)
    tree = builder.build(sel)

    mermaid = tree.to_mermaid()

    assert "Selector" in mermaid
    assert "action_a" in mermaid
    assert "action_b" in mermaid


# ============================================================================
# Serialization Tests
# ============================================================================


def test_simple_action_serialization():
    """Test serializing a tree with a single action"""
    builder = TreeBuilder("SingleAction")

    async def my_action(bb):
        return Status.SUCCESS

    action = builder.action("my_action", my_action)
    tree = builder.build(action)

    tree_dict = tree.to_dict()

    assert tree_dict["name"] == "SingleAction"
    assert tree_dict["root"]["kind"] == "action"
    assert tree_dict["root"]["name"] == "my_action"
    assert "<function:my_action>" in tree_dict["root"]["payload"]


def test_condition_serialization():
    """Test serializing a tree with a condition"""
    builder = TreeBuilder("ConditionOnly")

    def my_condition(bb):
        return True

    cond = builder.condition("my_condition", my_condition)
    tree = builder.build(cond)

    tree_dict = tree.to_dict()

    assert tree_dict["name"] == "ConditionOnly"
    assert tree_dict["root"]["kind"] == "condition"
    assert tree_dict["root"]["name"] == "my_condition"
    assert "<function:my_condition>" in tree_dict["root"]["payload"]


def test_sequence_serialization():
    """Test serializing a sequence tree"""
    builder = TreeBuilder("SequenceTree")

    async def action_a(bb):
        return Status.SUCCESS

    async def action_b(bb):
        return Status.SUCCESS

    a = builder.action("action_a", action_a)
    b = builder.action("action_b", action_b)
    seq = builder.sequence(a, b, memory=True)
    tree = builder.build(seq)

    tree_dict = tree.to_dict()

    assert tree_dict["name"] == "SequenceTree"
    assert tree_dict["root"]["kind"] == "sequence"
    assert "action_a" in tree_dict["root"]["name"]
    assert tree_dict["root"]["payload"]["memory"] == True


def test_selector_serialization():
    """Test serializing a selector tree"""
    builder = TreeBuilder("SelectorTree")

    async def action_a(bb):
        return Status.SUCCESS

    async def action_b(bb):
        return Status.FAILURE

    a = builder.action("action_a", action_a)
    b = builder.action("action_b", action_b)
    sel = builder.selector(a, b, memory=False)
    tree = builder.build(sel)

    tree_dict = tree.to_dict()

    assert tree_dict["name"] == "SelectorTree"
    assert tree_dict["root"]["kind"] == "selector"
    assert tree_dict["root"]["payload"]["memory"] == False


def test_nested_composite_serialization():
    """Test serializing nested composites"""
    builder = TreeBuilder("NestedTree")

    async def action_a(bb):
        return Status.SUCCESS

    async def action_b(bb):
        return Status.FAILURE

    async def action_c(bb):
        return Status.SUCCESS

    a = builder.action("action_a", action_a)
    b = builder.action("action_b", action_b)
    c = builder.action("action_c", action_c)

    seq = builder.sequence(a, b, memory=True)
    sel = builder.selector(seq, c, memory=False)
    tree = builder.build(sel)

    tree_dict = tree.to_dict()

    assert tree_dict["name"] == "NestedTree"
    assert tree_dict["root"]["kind"] == "selector"
    assert tree_dict["root"]["payload"]["memory"] == False


# ============================================================================
# Integration Tests - Mermaid and Serialization
# ============================================================================


def test_mermaid_matches_structure():
    """Test that Mermaid output reflects tree structure"""
    builder = TreeBuilder("StructureTest")

    async def action_a(bb):
        return Status.SUCCESS

    async def action_b(bb):
        return Status.SUCCESS

    a = builder.action("action_a", action_a)
    b = builder.action("action_b", action_b)
    tree = builder.build(builder.sequence(a, b, memory=True))

    mermaid = tree.to_mermaid()
    tree_dict = tree.to_dict()

    # Both should contain action_a and action_b
    assert "action_a" in mermaid
    assert "action_b" in mermaid
    assert "action_a" in tree_dict["root"]["name"]
    assert "action_b" in tree_dict["root"]["name"]


def test_serialization_is_json_compatible():
    """Test that serialized tree can be converted to JSON"""
    builder = TreeBuilder("JSONTest")

    async def my_action(bb):
        return Status.SUCCESS

    tree = builder.build(builder.action("my_action", my_action))
    tree_dict = tree.to_dict()

    # Should not raise when converting to JSON
    json_str = json.dumps(tree_dict)

    assert "my_action" in json_str
    assert "action" in json_str


def test_multiple_trees_have_unique_factory_names():
    """Test that different trees get unique factory names"""
    builder1 = TreeBuilder("Tree1")
    builder2 = TreeBuilder("Tree2")

    async def action_a(bb):
        return Status.SUCCESS

    async def action_b(bb):
        return Status.SUCCESS

    a1 = builder1.action("a", action_a)
    a2 = builder2.action("b", action_b)

    tree1 = builder1.build(builder1.sequence(a1, memory=True))
    tree2 = builder2.build(builder2.sequence(a2, memory=True))

    dict1 = tree1.to_dict()
    dict2 = tree2.to_dict()

    # Factory names should be different
    factory1 = dict1["root"]["payload"]["factory"]
    factory2 = dict2["root"]["payload"]["factory"]

    assert factory1 != factory2


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_to_dict_without_root_raises():
    """Test that to_dict() raises when tree has no root"""
    builder = TreeBuilder("EmptyTree")

    try:
        builder.to_dict()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "no root" in str(e)


def test_to_mermaid_without_root_raises():
    """Test that to_mermaid() raises when tree has no root"""
    builder = TreeBuilder("EmptyTree")

    try:
        builder.to_mermaid()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "no root" in str(e)
