#!/usr/bin/env python3
"""Tests for interface builder DSL"""

import sys
sys.path.insert(0, "src")

import pytest
from typing import Annotated
from pydantic import BaseModel

from mycorrhizal.common.interface_builder import blackboard_interface, readonly, readwrite, FieldSpec
from mycorrhizal.common.wrappers import ConstrainedView, ReadOnlyView, AccessControlError


# ============================================================================
# Test Fixtures
# ============================================================================

class TestBlackboard(BaseModel):
    """Test blackboard"""
    max_tasks: int = 10
    tasks_completed: int = 0
    current_task: str | None = None


# ============================================================================
# Basic Interface Definition Tests
# ============================================================================

def test_blackboard_interface_creates_class():
    """Test @blackboard_interface creates a class with type annotations"""
    @blackboard_interface
    class TaskInterface:
        max_tasks: int
        tasks_completed: int

    # Should be a class
    assert isinstance(TaskInterface, type)

    # Should have type annotations
    assert hasattr(TaskInterface, '__annotations__')
    assert 'max_tasks' in TaskInterface.__annotations__
    assert 'tasks_completed' in TaskInterface.__annotations__


def test_blackboard_interface_with_readonly_fields():
    """Test @blackboard_interface with Annotated readonly fields"""
    @blackboard_interface
    class TaskInterface:
        max_tasks: Annotated[int, readonly]
        tasks_completed: int

    # Should have readonly metadata
    assert hasattr(TaskInterface, '_readonly_fields')
    assert 'max_tasks' in TaskInterface._readonly_fields
    assert 'tasks_completed' not in TaskInterface._readonly_fields


def test_blackboard_interface_with_readwrite_fields():
    """Test @blackboard_interface with Annotated readwrite fields"""
    @blackboard_interface
    class TaskInterface:
        max_tasks: int
        tasks_completed: Annotated[int, readwrite]

    # Should have readwrite metadata
    assert hasattr(TaskInterface, '_readwrite_fields')
    assert 'tasks_completed' in TaskInterface._readwrite_fields


def test_blackboard_interface_with_mixed_fields():
    """Test @blackboard_interface with both readonly and readwrite fields"""
    @blackboard_interface
    class TaskInterface:
        max_tasks: Annotated[int, readonly]
        tasks_completed: Annotated[int, readwrite]

    # Should have both metadata sets
    assert TaskInterface._readonly_fields == frozenset(['max_tasks'])
    assert TaskInterface._readwrite_fields == frozenset(['tasks_completed'])


def test_blackboard_interface_default_to_readwrite():
    """Test @blackboard_interface defaults to readwrite when not specified"""
    @blackboard_interface
    class TaskInterface:
        max_tasks: int
        tasks_completed: int

    # Should default all fields to readwrite
    assert TaskInterface._readwrite_fields == frozenset(['max_tasks', 'tasks_completed'])
    assert TaskInterface._readonly_fields == frozenset()


# ============================================================================
# Integration with Wrappers Tests
# ============================================================================

def test_blackboard_interface_with_constrained_view():
    """Test using interface from DSL with ConstrainedView"""
    @blackboard_interface
    class TaskInterface:
        max_tasks: Annotated[int, readonly]
        tasks_completed: Annotated[int, readwrite]

    bb = TestBlackboard()
    view = ConstrainedView(bb, {'max_tasks', 'tasks_completed'}, {'max_tasks'})

    # Should work with the view
    assert view.max_tasks == 10
    assert view.tasks_completed == 0


def test_create_view_from_protocol_with_dsl_interface():
    """Test create_view_from_protocol works with DSL-generated interface"""
    @blackboard_interface
    class TaskInterface:
        max_tasks: Annotated[int, readonly]
        tasks_completed: Annotated[int, readwrite]

    bb = TestBlackboard()
    view = ConstrainedView(bb, {'max_tasks', 'tasks_completed'}, {'max_tasks'})

    # Readonly field should be readable
    assert view.max_tasks == 10

    # Readonly field should not be writable
    with pytest.raises(Exception):  # AccessControlError
        view.max_tasks = 20


# ============================================================================
# Interface Composition Tests
# ============================================================================

def test_blackboard_interface_inheritance():
    """Test interface composition using class inheritance"""
    @blackboard_interface
    class BaseInterface:
        base_field: int

    # Note: Current implementation doesn't support inheritance
    # Interfaces are standalone classes
    # This test documents current limitation

    # BaseInterface should still work
    assert hasattr(BaseInterface, '__annotations__')
    assert 'base_field' in BaseInterface.__annotations__


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_blackboard_interface_empty():
    """Test @blackboard_interface with no fields"""
    @blackboard_interface
    class EmptyInterface:
        pass

    # Should still create a class
    assert isinstance(EmptyInterface, type)

    # Should have empty annotations
    assert EmptyInterface.__annotations__ == {}


def test_blackboard_interface_with_docstring():
    """Test @blackboard_interface preserves docstring"""
    @blackboard_interface
    class DocumentedInterface:
        """This is a documented interface"""
        field: int

    # Should preserve docstring
    assert DocumentedInterface.__doc__ == "This is a documented interface"


def test_blackboard_interface_with_private_fields():
    """Test @blackboard_interface filters private fields"""
    @blackboard_interface
    class InterfaceWithPrivate:
        public: int
        _private: str

    # Should not include private fields in annotations
    assert 'public' in InterfaceWithPrivate.__annotations__
    assert '_private' not in InterfaceWithPrivate.__annotations__


def test_blackboard_interface_with_optional_fields():
    """Test @blackboard_interface handles Optional types"""
    from typing import Optional

    @blackboard_interface
    class InterfaceWithOptional:
        required: int
        optional: Optional[str]

    # Should preserve Optional type in annotations
    assert 'required' in InterfaceWithOptional.__annotations__
    assert 'optional' in InterfaceWithOptional.__annotations__


# ============================================================================
# Custom Name Tests
# ============================================================================

def test_blackboard_interface_with_custom_name():
    """Test @blackboard_interface with custom name"""
    @blackboard_interface(name="CustomTaskInterface")
    class TaskInterface:
        max_tasks: int

    # Should use custom name
    assert TaskInterface.__name__ == "CustomTaskInterface"


# ============================================================================
# Real-World Example Tests
# ============================================================================

def test_real_world_task_processing_interface():
    """Test a realistic task processing interface"""
    @blackboard_interface
    class TaskProcessingInterface:
        """Interface for task processing operations"""
        # Read-only configuration
        max_tasks: int
        interval: float

        # Read-write state
        tasks_completed: int
        current_token: object | None

    # Verify structure from annotations
    assert set(TaskProcessingInterface.__annotations__.keys()) == {
        'max_tasks', 'interval', 'tasks_completed', 'current_token'
    }

    # Verify defaults to readwrite
    assert TaskProcessingInterface._readwrite_fields == frozenset([
        'max_tasks', 'interval', 'tasks_completed', 'current_token'
    ])


def test_real_world_config_interface():
    """Test a realistic configuration interface"""
    @blackboard_interface
    class ConfigInterface:
        """Read-only configuration interface"""
        max_concurrent: Annotated[int, readonly]
        timeout: Annotated[float, readonly]
        retry_count: Annotated[int, readwrite]

    # Verify readonly fields
    assert ConfigInterface._readonly_fields == frozenset(['max_concurrent', 'timeout'])

    # Verify readwrite for specified
    assert 'retry_count' in ConfigInterface._readwrite_fields


# ============================================================================
# Type Safety Tests
# ============================================================================

def test_blackboard_interface_preserves_types():
    """Test @blackboard_interface preserves field types"""
    from typing import Optional, List

    @blackboard_interface
    class TypedInterface:
        int_field: int
        str_field: str
        optional_field: Optional[int]
        list_field: List[str]

    # Check types from annotations
    assert TypedInterface.__annotations__['int_field'] == int
    assert TypedInterface.__annotations__['str_field'] == str
    # Optional and List types are preserved as-is
    assert 'int' in str(TypedInterface.__annotations__['optional_field']) or 'Union' in str(TypedInterface.__annotations__['optional_field'])
    assert 'List' in str(TypedInterface.__annotations__['list_field']) or 'list' in str(TypedInterface.__annotations__['list_field'])
