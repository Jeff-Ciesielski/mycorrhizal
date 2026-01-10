#!/usr/bin/env python3
"""Tests for blackboard wrapper classes"""

import sys
sys.path.insert(0, "src")

import pytest
from typing import Protocol
from pydantic import BaseModel

from mycorrhizal.common.wrappers import (
    AccessControlError,
    ReadOnlyView,
    WriteOnlyView,
    ConstrainedView,
    CompositeView,
    create_readonly_view,
    create_constrained_view,
    create_view_from_protocol,
    View,
)


# ============================================================================
# Test Fixtures
# ============================================================================

class SimpleBlackboard:
    """Simple test blackboard"""
    def __init__(self):
        self.config_value = 42
        self.mutable_state = 0
        self.private_data = "secret"


class TaskBlackboard(BaseModel):
    """Pydantic blackboard for testing"""
    max_tasks: int = 10
    tasks_completed: int = 0
    current_task: str | None = None
    error_count: int = 0


class TaskInterface(Protocol):
    """Protocol for task processing"""
    max_tasks: int
    tasks_completed: int


# ============================================================================
# ReadOnlyView Tests
# ============================================================================

def test_readonly_view_allows_reads():
    """Test ReadOnlyView allows reading specified fields"""
    bb = SimpleBlackboard()
    view = ReadOnlyView(bb, {'config_value'})

    assert view.config_value == 42


def test_readonly_view_denies_writes():
    """Test ReadOnlyView denies writing to any field"""
    bb = SimpleBlackboard()
    view = ReadOnlyView(bb, {'config_value'})

    with pytest.raises(AccessControlError) as exc_info:
        view.config_value = 100

    assert exc_info.value.field_name == 'config_value'
    assert exc_info.value.operation == 'write'
    assert 'read-only' in str(exc_info.value).lower()


def test_readonly_view_denies_access_to_unspecified_fields():
    """Test ReadOnlyView denies access to fields not in allowed set"""
    bb = SimpleBlackboard()
    view = ReadOnlyView(bb, {'config_value'})

    with pytest.raises(AttributeError, match='not accessible'):
        _ = view.mutable_state


def test_readonly_view_dir():
    """Test ReadOnlyView dir() shows only allowed fields"""
    bb = SimpleBlackboard()
    view = ReadOnlyView(bb, {'config_value', 'mutable_state'})

    attrs = dir(view)
    assert 'config_value' in attrs
    assert 'mutable_state' in attrs
    assert 'private_data' not in attrs


def test_readonly_view_repr():
    """Test ReadOnlyView has useful repr"""
    bb = SimpleBlackboard()
    view = ReadOnlyView(bb, {'config_value'})

    r = repr(view)
    assert 'ReadOnlyView' in r
    assert 'config_value' in r


def test_readonly_view_with_pydantic_model():
    """Test ReadOnlyView works with Pydantic models"""
    bb = TaskBlackboard()
    view = ReadOnlyView(bb, {'max_tasks', 'tasks_completed'})

    assert view.max_tasks == 10
    assert view.tasks_completed == 0


def test_readonly_view_private_attributes_allowed():
    """Test that wrapper can set its own private attributes"""
    bb = SimpleBlackboard()
    view = ReadOnlyView(bb, {'config_value'})

    # Should not raise - these are wrapper internals
    assert hasattr(view, '_bb')
    assert hasattr(view, '_allowed_fields')


# ============================================================================
# WriteOnlyView Tests
# ============================================================================

def test_writeonly_view_allows_writes():
    """Test WriteOnlyView allows writing to specified fields"""
    bb = SimpleBlackboard()
    view = WriteOnlyView(bb, {'mutable_state'})

    view.mutable_state = 100
    assert bb.mutable_state == 100


def test_writeonly_view_denies_reads():
    """Test WriteOnlyView denies reading any field"""
    bb = SimpleBlackboard()
    view = WriteOnlyView(bb, {'mutable_state'})

    with pytest.raises(AccessControlError) as exc_info:
        _ = view.mutable_state

    assert exc_info.value.field_name == 'mutable_state'
    assert exc_info.value.operation == 'read'
    assert 'write-only' in str(exc_info.value).lower()


def test_writeonly_view_denies_access_to_unspecified_fields():
    """Test WriteOnlyView denies access to fields not in allowed set"""
    bb = SimpleBlackboard()
    view = WriteOnlyView(bb, {'mutable_state'})

    with pytest.raises(AttributeError, match='not accessible'):
        view.config_value = 100


def test_writeonly_view_dir():
    """Test WriteOnlyView dir() shows allowed fields"""
    bb = SimpleBlackboard()
    view = WriteOnlyView(bb, {'mutable_state'})

    attrs = dir(view)
    assert 'mutable_state' in attrs


# ============================================================================
# ConstrainedView Tests
# ============================================================================

def test_constrained_view_allows_read_on_readonly_fields():
    """Test ConstrainedView allows reading read-only fields"""
    bb = TaskBlackboard()
    view = ConstrainedView(
        bb,
        allowed_fields={'max_tasks', 'tasks_completed'},
        readonly_fields={'max_tasks'}
    )

    assert view.max_tasks == 10


def test_constrained_view_denies_write_on_readonly_fields():
    """Test ConstrainedView denies writing to read-only fields"""
    bb = TaskBlackboard()
    view = ConstrainedView(
        bb,
        allowed_fields={'max_tasks', 'tasks_completed'},
        readonly_fields={'max_tasks'}
    )

    with pytest.raises(AccessControlError) as exc_info:
        view.max_tasks = 20

    assert exc_info.value.field_name == 'max_tasks'
    assert exc_info.value.operation == 'write'


def test_constrained_view_allows_readwrite_on_readwrite_fields():
    """Test ConstrainedView allows both reading and writing read-write fields"""
    bb = TaskBlackboard()
    view = ConstrainedView(
        bb,
        allowed_fields={'max_tasks', 'tasks_completed'},
        readonly_fields={'max_tasks'}
    )

    # Read works
    assert view.tasks_completed == 0

    # Write works
    view.tasks_completed = 5
    assert bb.tasks_completed == 5
    assert view.tasks_completed == 5


def test_constrained_view_denies_access_to_unspecified_fields():
    """Test ConstrainedView denies access to unspecified fields"""
    bb = TaskBlackboard()
    view = ConstrainedView(
        bb,
        allowed_fields={'max_tasks', 'tasks_completed'},
        readonly_fields={'max_tasks'}
    )

    with pytest.raises(AttributeError, match='not accessible'):
        _ = view.current_task


def test_constrained_view_with_empty_readonly():
    """Test ConstrainedView works with empty readonly set"""
    bb = TaskBlackboard()
    view = ConstrainedView(
        bb,
        allowed_fields={'tasks_completed'},
        readonly_fields=set()
    )

    # Both read and write should work
    assert view.tasks_completed == 0
    view.tasks_completed = 10
    assert view.tasks_completed == 10


def test_constrained_view_with_all_readonly():
    """Test ConstrainedView with all fields read-only"""
    bb = TaskBlackboard()
    view = ConstrainedView(
        bb,
        allowed_fields={'max_tasks', 'tasks_completed'},
        readonly_fields={'max_tasks', 'tasks_completed'}
    )

    # Read works
    assert view.max_tasks == 10
    assert view.tasks_completed == 0

    # Write fails on both
    with pytest.raises(AccessControlError):
        view.max_tasks = 20

    with pytest.raises(AccessControlError):
        view.tasks_completed = 5


# ============================================================================
# CompositeView Tests
# ============================================================================

def test_composite_view_combine():
    """Test CompositeView.combine creates composite from multiple views"""
    bb = TaskBlackboard()
    readonly = ReadOnlyView(bb, {'max_tasks'})
    readwrite = ConstrainedView(bb, {'tasks_completed'}, readonly_fields=set())

    composite = CompositeView.combine([readonly, readwrite])

    assert composite.max_tasks == 10  # Read from readonly
    assert composite.tasks_completed == 0  # Read from readwrite


def test_composite_view_enforces_constraints():
    """Test CompositeView enforces constraints from composed views"""
    bb = TaskBlackboard()
    readonly = ReadOnlyView(bb, {'max_tasks'})
    readwrite = ConstrainedView(bb, {'tasks_completed'}, readonly_fields=set())

    composite = CompositeView.combine([readonly, readwrite])

    # Should inherit readonly constraint
    with pytest.raises(AccessControlError):
        composite.max_tasks = 20

    # Should allow write to readwrite field
    composite.tasks_completed = 5
    assert bb.tasks_completed == 5


def test_composite_view_denies_unspecified_fields():
    """Test CompositeView denies access to fields not in any composed view"""
    bb = TaskBlackboard()
    readonly = ReadOnlyView(bb, {'max_tasks'})
    readwrite = ConstrainedView(bb, {'tasks_completed'}, readonly_fields=set())

    composite = CompositeView.combine([readonly, readwrite])

    with pytest.raises(AttributeError, match='not accessible'):
        _ = composite.current_task


def test_composite_view_requires_same_blackboard():
    """Test CompositeView.combine raises if views reference different blackboards"""
    bb1 = TaskBlackboard()
    bb2 = TaskBlackboard()
    view1 = ReadOnlyView(bb1, {'max_tasks'})
    view2 = ReadOnlyView(bb2, {'tasks_completed'})

    with pytest.raises(ValueError, match='same blackboard'):
        CompositeView.combine([view1, view2])


def test_composite_view_empty_list_raises():
    """Test CompositeView.combine raises ValueError for empty list"""
    with pytest.raises(ValueError, match='empty list'):
        CompositeView.combine([])


# ============================================================================
# Factory Function Tests
# ============================================================================

def test_create_readonly_view():
    """Test create_readonly_view factory function"""
    bb = TaskBlackboard()
    view = create_readonly_view(bb, ('max_tasks', 'tasks_completed'))

    assert isinstance(view, ReadOnlyView)
    assert view.max_tasks == 10
    assert view.tasks_completed == 0

    with pytest.raises(AccessControlError):
        view.max_tasks = 20


def test_create_readonly_view_multiple_calls():
    """Test create_readonly_view creates new instances on each call"""
    bb = TaskBlackboard()

    view1 = create_readonly_view(bb, ('max_tasks',))
    view2 = create_readonly_view(bb, ('max_tasks',))

    # Should be different instances (no caching)
    assert view1 is not view2
    # But both should work
    assert view1.max_tasks == 10
    assert view2.max_tasks == 10


def test_create_constrained_view():
    """Test create_constrained_view factory function"""
    bb = TaskBlackboard()
    view = create_constrained_view(
        bb,
        ('max_tasks', 'tasks_completed'),
        ('max_tasks',)
    )

    assert isinstance(view, ConstrainedView)
    assert view.max_tasks == 10
    view.tasks_completed = 5
    assert bb.tasks_completed == 5


def test_create_constrained_view_multiple_calls():
    """Test create_constrained_view creates new instances on each call"""
    bb = TaskBlackboard()

    view1 = create_constrained_view(bb, ('max_tasks',), ('max_tasks',))
    view2 = create_constrained_view(bb, ('max_tasks',), ('max_tasks',))

    # Should be different instances (no caching)
    assert view1 is not view2
    # But both should work
    assert view1.max_tasks == 10
    assert view2.max_tasks == 10


def test_create_view_from_protocol():
    """Test create_view_from_protocol extracts fields from protocol"""
    bb = TaskBlackboard()
    view = create_view_from_protocol(bb, TaskInterface)

    # Should have fields from protocol
    assert view.max_tasks == 10
    assert view.tasks_completed == 0

    # Should not have fields not in protocol
    with pytest.raises(AttributeError):
        _ = view.current_task


def test_create_view_from_protocol_with_readonly():
    """Test create_view_from_protocol respects readonly_fields parameter"""
    bb = TaskBlackboard()
    view = create_view_from_protocol(
        bb,
        TaskInterface,
        readonly_fields={'max_tasks'}
    )

    # max_tasks should be read-only
    with pytest.raises(AccessControlError):
        view.max_tasks = 20

    # tasks_completed should be read-write
    view.tasks_completed = 5
    assert bb.tasks_completed == 5


# ============================================================================
# View Factory Function Tests
# ============================================================================

def test_view_factory():
    """Test View factory function creates typed views"""
    bb = TaskBlackboard()
    view = View(bb, TaskInterface)

    # Should work as a ConstrainedView
    assert isinstance(view, ConstrainedView)
    assert view.max_tasks == 10
    assert view.tasks_completed == 0


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_wrapper_with_no_fields():
    """Test wrappers work with empty field set"""
    bb = SimpleBlackboard()
    view = ReadOnlyView(bb, set())

    # Should deny all access
    with pytest.raises(AttributeError):
        _ = view.config_value


def test_wrapper_with_missing_field():
    """Test wrappers handle missing fields gracefully"""
    bb = SimpleBlackboard()
    view = ReadOnlyView(bb, {'nonexistent_field'})

    # Should raise AttributeError indicating field doesn't exist
    with pytest.raises(AttributeError, match='does not exist'):
        _ = view.nonexistent_field


def test_wrapper_preserves_underlying_object():
    """Test wrappers don't copy the underlying object"""
    bb = SimpleBlackboard()
    view = ReadOnlyView(bb, {'config_value'})

    # Modify underlying object directly
    bb.config_value = 100

    # View should see the change
    assert view.config_value == 100


def test_wrapper_with_attribute_with_underscore():
    """Test wrappers handle underscore-prefixed attributes"""
    bb = SimpleBlackboard()
    view = ConstrainedView(bb, {'config_value'}, readonly_fields=set())

    # Should be able to access wrapper's private attributes
    assert hasattr(view, '_bb')
    assert hasattr(view, '_allowed_fields')


def test_constrained_view_initial_readonly():
    """Test ConstrainedView with readonly_fields from protocol"""
    class TestInterface(Protocol):
        value: int
        _readonly_fields = {'value'}

    bb = SimpleBlackboard()
    bb.value = 42

    # This tests the protocol metadata extraction
    # (The actual readonly extraction would be done by create_view_from_protocol)
    view = ConstrainedView(bb, {'value'}, readonly_fields={'value'})

    assert view.value == 42
    with pytest.raises(AccessControlError):
        view.value = 100


# ============================================================================
# Integration Tests
# ============================================================================

def test_multiple_views_same_blackboard():
    """Test multiple views can access same blackboard independently"""
    bb = TaskBlackboard()

    readonly_view = ReadOnlyView(bb, {'max_tasks'})
    readwrite_view = ConstrainedView(bb, {'tasks_completed'}, readonly_fields=set())

    # Both views work independently
    assert readonly_view.max_tasks == 10
    assert readwrite_view.tasks_completed == 0

    # Modifying through readwrite_view
    readwrite_view.tasks_completed = 5

    # readonly_view sees the change
    assert bb.tasks_completed == 5


def test_view_chaining():
    """Test creating a view from another view"""
    bb = TaskBlackboard()

    view1 = ConstrainedView(bb, {'max_tasks', 'tasks_completed'}, readonly_fields={'max_tasks'})

    # Can't directly create a view from a view (would need special handling)
    # But we can create a composite view
    view2 = ReadOnlyView(bb, {'error_count'})

    composite = CompositeView.combine([view1, view2])

    # Should have access to all fields
    assert composite.max_tasks == 10
    assert composite.tasks_completed == 0
    assert composite.error_count == 0


def test_wrapper_with_none_value():
    """Test wrappers handle None field values correctly"""
    bb = TaskBlackboard()
    view = ReadOnlyView(bb, {'current_task'})

    # None is a valid value
    assert view.current_task is None


def test_wrapper_type_compatibility():
    """Test wrappers maintain type compatibility"""
    bb = TaskBlackboard()

    # Create typed view
    view: TaskInterface = View(bb, TaskInterface)

    # Should pass type checking
    assert view.max_tasks == 10
    assert view.tasks_completed == 0
