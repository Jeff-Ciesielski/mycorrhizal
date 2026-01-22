#!/usr/bin/env python3
"""Tests for blackboard interface protocols"""

import sys
sys.path.insert(0, "src")

from typing import Protocol
from pydantic import BaseModel
import pytest

from mycorrhizal.common.interfaces import (
    Readable,
    Writable,
    FieldAccessor,
    BlackboardProtocol,
    FieldMetadata,
    InterfaceMetadata,
    get_interface_fields,
    create_interface_from_model,
    _extract_interface_metadata,
)


# ============================================================================
# Test Fixtures
# ============================================================================

class IntCounter:
    """Test class with get_value method"""
    def __init__(self, value: int = 0):
        self._value = value

    def get_value(self) -> int:
        return self._value


class IntStorage:
    """Test class with set_value method"""
    def __init__(self):
        self._value = None

    def set_value(self, value: int) -> None:
        self._value = value


class IntCounterFull:
    """Test class with both get and set methods"""
    def __init__(self, value: int = 0):
        self._value = value

    def get_value(self) -> int:
        return self._value

    def set_value(self, value: int) -> None:
        self._value = value


class ValidBlackboard:
    """Test class implementing BlackboardProtocol"""
    def __init__(self):
        self.data = 42

    def validate(self) -> bool:
        return self.data is not None

    def to_dict(self) -> dict:
        return {"data": self.data}


class InvalidBlackboard:
    """Test class not implementing BlackboardProtocol"""
    def __init__(self):
        self.data = 42


class TaskModel(BaseModel):
    """Pydantic model for testing"""
    max_tasks: int = 10
    tasks_completed: int = 0
    current_task: str | None = None


# ============================================================================
# Protocol Tests
# ============================================================================

def test_readable_protocol_with_valid_class():
    """Test Readable protocol recognizes valid classes"""
    counter = IntCounter(5)
    assert isinstance(counter, Readable)
    assert counter.get_value() == 5


def test_readable_protocol_with_invalid_class():
    """Test Readable protocol rejects invalid classes"""
    storage = IntStorage()
    assert not isinstance(storage, Readable)


def test_writable_protocol_with_valid_class():
    """Test Writable protocol recognizes valid classes"""
    storage = IntStorage()
    assert isinstance(storage, Writable)
    storage.set_value(10)


def test_writable_protocol_with_invalid_class():
    """Test Writable protocol rejects invalid classes"""
    counter = IntCounter(5)
    assert not isinstance(counter, Writable)


def test_field_accessor_protocol_with_valid_class():
    """Test FieldAccessor protocol recognizes valid classes"""
    counter = IntCounterFull(5)
    assert isinstance(counter, FieldAccessor)
    assert isinstance(counter, Readable)
    assert isinstance(counter, Writable)


def test_field_accessor_protocol_with_partial_class():
    """Test FieldAccessor protocol rejects partial implementations"""
    counter = IntCounter(5)
    assert not isinstance(counter, FieldAccessor)
    # But it should be Readable
    assert isinstance(counter, Readable)


def test_blackboard_protocol_with_valid_class():
    """Test BlackboardProtocol recognizes valid implementations"""
    bb = ValidBlackboard()
    assert isinstance(bb, BlackboardProtocol)
    assert bb.validate() is True
    assert bb.to_dict() == {"data": 42}


def test_blackboard_protocol_with_invalid_class():
    """Test BlackboardProtocol rejects invalid implementations"""
    bb = InvalidBlackboard()
    assert not isinstance(bb, BlackboardProtocol)


# ============================================================================
# Helper Function Tests
# ============================================================================

def test_get_interface_fields_from_protocol():
    """Test get_interface_fields extracts field annotations"""
    class TestInterface(Protocol):
        counter: int
        name: str
        _private: float  # Should be filtered out

    fields = get_interface_fields(TestInterface)
    assert "counter" in fields
    assert "name" in fields
    assert "_private" not in fields
    assert fields["counter"] == int
    assert fields["name"] == str


def test_create_interface_from_pydantic_model():
    """Test creating interface from Pydantic model"""
    TaskInterface = create_interface_from_model(TaskModel)

    fields = get_interface_fields(TaskInterface)
    assert "max_tasks" in fields
    assert "tasks_completed" in fields
    assert "current_task" in fields


def test_create_interface_from_model_with_readonly():
    """Test creating interface with readonly fields"""
    TaskInterface = create_interface_from_model(
        TaskModel,
        readonly_fields={"max_tasks", "tasks_completed"}
    )

    # Check that readonly metadata is stored
    assert hasattr(TaskInterface, '_readonly_fields')
    assert TaskInterface._readonly_fields == {"max_tasks", "tasks_completed"}


# ============================================================================
# Metadata Tests
# ============================================================================

def test_field_metadata_creation():
    """Test FieldMetadata dataclass"""
    meta = FieldMetadata(
        name="test_field",
        type=int,
        readonly=True,
        required=True
    )

    assert meta.name == "test_field"
    assert meta.type == int
    assert meta.readonly is True
    assert meta.required is True


def test_interface_metadata_creation():
    """Test InterfaceMetadata dataclass"""
    fields = {
        "readonly_field": FieldMetadata("readonly_field", int, readonly=True),
        "readwrite_field": FieldMetadata("readwrite_field", str, readonly=False)
    }

    meta = InterfaceMetadata(
        name="TestInterface",
        fields=fields,
        description="Test interface"
    )

    assert meta.name == "TestInterface"
    assert len(meta.fields) == 2
    assert meta.get_readonly_fields() == {"readonly_field"}
    assert meta.get_readwrite_fields() == {"readwrite_field"}


def test_extract_interface_metadata():
    """Test _extract_interface_metadata function"""
    class TestInterface(Protocol):
        counter: int
        name: str

    readonly = {"counter"}
    metadata = _extract_interface_metadata(TestInterface, readonly_fields=readonly)

    assert metadata.name == "TestInterface"
    assert "counter" in metadata.fields
    assert "name" in metadata.fields
    assert metadata.fields["counter"].readonly is True
    assert metadata.fields["name"].readonly is False


# ============================================================================
# Integration Tests
# ============================================================================

def test_protocol_with_multiple_implementations():
    """Test that protocols work with different class implementations"""
    class CounterA:
        def get_value(self) -> int:
            return 1

    class CounterB:
        def get_value(self) -> int:
            return 2

    a = CounterA()
    b = CounterB()

    # Both should be Readable
    assert isinstance(a, Readable)
    assert isinstance(b, Readable)

    # But different types
    assert type(a) is not type(b)


def test_protocol_duck_typing():
    """Test that protocols use structural subtyping (duck typing)"""
    # These classes don't inherit from Readable but satisfy it
    class DuckCounter1:
        def get_value(self) -> int:
            return 1

    class DuckCounter2:
        def get_value(self) -> int:  # Different signature but compatible
            return 2

    duck1 = DuckCounter1()
    duck2 = DuckCounter2()

    # Both should be Readable due to structural subtyping
    assert isinstance(duck1, Readable)
    assert isinstance(duck2, Readable)


def test_blackboard_protocol_multiple_implementations():
    """Test BlackboardProtocol with different implementations"""
    class BB1:
        def validate(self) -> bool:
            return True
        def to_dict(self) -> dict:
            return {}

    class BB2:
        def validate(self) -> bool:
            return False
        def to_dict(self) -> dict:
            return {"error": True}

    bb1 = BB1()
    bb2 = BB2()

    assert isinstance(bb1, BlackboardProtocol)
    assert isinstance(bb2, BlackboardProtocol)

    assert bb1.validate() is True
    assert bb2.validate() is False


def test_interface_inheritance():
    """Test that protocols support inheritance"""
    class BaseInterface(Protocol):
        base_field: int

    class DerivedInterface(BaseInterface, Protocol):
        derived_field: str

    # Derived should have both fields
    fields = get_interface_fields(DerivedInterface)
    assert "base_field" in fields
    assert "derived_field" in fields


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_protocol_with_no_annotations():
    """Test protocol with no type annotations"""
    class EmptyInterface(Protocol):
        pass

    fields = get_interface_fields(EmptyInterface)
    assert len(fields) == 0


def test_create_interface_from_empty_model():
    """Test creating interface from model with no fields"""
    class EmptyModel(BaseModel):
        pass

    EmptyInterface = create_interface_from_model(EmptyModel)
    fields = get_interface_fields(EmptyInterface)
    assert len(fields) == 0


def test_get_interface_fields_with_private_attributes():
    """Test that private attributes are filtered out"""
    class TestInterface(Protocol):
        public: int
        _private: str
        __dunder: float

    fields = get_interface_fields(TestInterface)
    assert "public" in fields
    assert "_private" not in fields
    assert "__dunder" not in fields
