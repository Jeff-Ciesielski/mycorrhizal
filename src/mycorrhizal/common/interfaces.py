"""
Blackboard Interface System

This module provides Protocol-based interfaces for type-safe, constrained access
to blackboard state across Mycorrhizal's three DSL systems (Hypha, Rhizomorph, Enoki).

Key concepts:
- Protocols: Structural subtyping for interface definitions
- Wrappers: Runtime enforcement of access constraints
- Composition: Combine multiple interfaces into larger ones
"""

from typing import Protocol, TypeVar, Generic, Any, runtime_checkable, runtime_checkable
from typing import Optional, get_type_hints, Union
from dataclasses import dataclass, field
from functools import wraps
import inspect
import types


# ============================================================================
# Type Variables
# ============================================================================

T = TypeVar('T')
P = Protocol


# ============================================================================
# Core Protocol Definitions
# ============================================================================

@runtime_checkable
class Readable(Protocol[T]):
    """
    Protocol for read-only access to a typed value.

    This protocol provides structural subtyping - any class with a matching
    `get_value()` method will satisfy this protocol.

    Type Parameters:
        T: The type of value being read

    Example:
        >>> class Config:
        ...     def get_value(self) -> int:
        ...         return 42
        >>> config: Readable[int] = Config()
        >>> isinstance(config, Readable)
        True
    """
    def get_value(self) -> T:
        """Get the current value"""
        ...


@runtime_checkable
class Writable(Protocol[T]):
    """
    Protocol for write access to a typed value.

    This protocol provides structural subtyping - any class with a matching
    `set_value()` method will satisfy this protocol.

    Type Parameters:
        T: The type of value being written

    Example:
        >>> class Storage:
        ...     def set_value(self, value: int) -> None:
        ...         self._value = value
        >>> storage: Writable[int] = Storage()
        >>> isinstance(storage, Writable)
        True
    """
    def set_value(self, value: T) -> None:
        """Set a new value"""
        ...


@runtime_checkable
class FieldAccessor(Readable[T], Writable[T], Protocol[T]):
    """
    Protocol for read/write access to a typed value.

    Combines both Readable and Writable protocols for full access.

    Type Parameters:
        T: The type of value being accessed

    Example:
        >>> class Counter:
        ...     def get_value(self) -> int:
        ...         return self._count
        ...     def set_value(self, value: int) -> None:
        ...         self._count = value
        >>> counter: FieldAccessor[int] = Counter()
        >>> isinstance(counter, FieldAccessor)
        True
    """
    pass


@runtime_checkable
class BlackboardProtocol(Protocol):
    """
    Base blackboard protocol that all blackboards should implement.

    This protocol provides structural subtyping for blackboard-like objects.
    Any class with matching methods will satisfy this protocol, no inheritance
    required.

    Required Methods:
        validate(): Check if blackboard state is valid
        to_dict(): Convert blackboard to dictionary representation

    Example:
        >>> class MyBlackboard:
        ...     def validate(self) -> bool:
        ...         return True
        ...     def to_dict(self) -> dict[str, Any]:
        ...         return {"data": 42}
        >>> bb: BlackboardProtocol = MyBlackboard()
        >>> isinstance(bb, BlackboardProtocol)
        True
    """

    def validate(self) -> bool:
        """
        Validate the blackboard state.

        Returns:
            True if blackboard state is valid, False otherwise
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """
        Convert blackboard to dictionary representation.

        Returns:
            Dictionary representation of blackboard state
        """
        ...


# ============================================================================
# Interface Metadata
# ============================================================================

@dataclass
class FieldMetadata:
    """
    Metadata for blackboard interface fields.

    Attributes:
        name: Field name
        type: Field type annotation
        readonly: Whether field is read-only
        required: Whether field is required (not Optional)
    """
    name: str
    type: type
    readonly: bool = False
    required: bool = True


@dataclass
class InterfaceMetadata:
    """
    Metadata for a blackboard interface.

    Attributes:
        name: Interface name
        fields: Dictionary of field metadata by field name
        description: Optional description of the interface
        bases: Base interfaces this interface extends
    """
    name: str
    fields: dict[str, FieldMetadata] = field(default_factory=dict)
    description: Optional[str] = None
    bases: tuple[type, ...] = field(default_factory=tuple)

    def get_readonly_fields(self) -> set[str]:
        """Get set of read-only field names"""
        return {name for name, meta in self.fields.items() if meta.readonly}

    def get_readwrite_fields(self) -> set[str]:
        """Get set of read-write field names"""
        return {name for name, meta in self.fields.items() if not meta.readonly}


def _extract_interface_metadata(
    interface_class: type,
    readonly_fields: set[str] | None = None,
    readwrite_fields: set[str] | None = None
) -> InterfaceMetadata:
    """
    Extract metadata from an interface class.

    Args:
        interface_class: The interface class to analyze
        readonly_fields: Set of field names that are read-only
        readwrite_fields: Set of field names that are read-write

    Returns:
        InterfaceMetadata with extracted information
    """
    readonly_fields = readonly_fields or set()
    readwrite_fields = readwrite_fields or set()

    # Get type hints from the class
    try:
        hints = get_type_hints(interface_class)
    except Exception:
        hints = {}

    # Build field metadata
    fields = {}
    for field_name, field_type in hints.items():
        # Skip private attributes and methods
        if field_name.startswith('_'):
            continue

        # Check if field is readonly or readwrite
        is_readonly = field_name in readonly_fields

        # Check if field is required (not Optional)
        # Handle Union types (Optional[T] is Union[T, None])
        is_required = True
        if hasattr(field_type, '__origin__'):
            # Check for Union (which includes Optional)
            if field_type.__origin__ is Union:
                # Union[X, None] means Optional[X]
                args = getattr(field_type, '__args__', ())
                is_required = type(None) not in args
        elif str(field_type).startswith('typing.Union[') or 'Optional[' in str(field_type):
            # String-based fallback for edge cases
            is_required = 'None' not in str(field_type)

        fields[field_name] = FieldMetadata(
            name=field_name,
            type=field_type,
            readonly=is_readonly,
            required=is_required
        )

    return InterfaceMetadata(
        name=interface_class.__name__,
        fields=fields,
        description=interface_class.__doc__,
        bases=interface_class.__bases__
    )


# ============================================================================
# Protocol Helper Functions
# ============================================================================

def validate_implements(cls: type, protocol: type) -> bool:
    """
    Validate that a class implements a protocol.

    Uses structural subtyping to check if the class has all required
    methods and attributes defined by the protocol.

    Args:
        cls: The class to check
        protocol: The protocol to check against

    Returns:
        True if cls implements protocol, False otherwise

    Example:
        >>> class MyBlackboard:
        ...     def validate(self) -> bool:
        ...         return True
        ...     def to_dict(self) -> dict:
        ...         return {}
        >>> validate_implements(MyBlackboard, BlackboardProtocol)
        True
        >>> validate_implements(str, BlackboardProtocol)
        False
    """
    if not isinstance(protocol, type) or not hasattr(protocol, '__protocol_attrs__'):
        # Not a protocol, try anyway with isinstance
        try:
            return issubclass(cls, protocol) if isinstance(cls, type) else isinstance(cls, protocol)
        except TypeError:
            return False

    # Check if protocol is runtime_checkable
    if not hasattr(protocol, '__is_protocol__'):
        return False

    # Use isinstance if we have an instance, otherwise check structure
    try:
        # For protocols, we need to check if the class has all protocol attributes
        for attr in getattr(protocol, '__protocol_attrs__', []):
            if not hasattr(cls, attr):
                return False
        return True
    except Exception:
        return False


def get_interface_fields(interface: type) -> dict[str, type]:
    """
    Extract field type annotations from an interface.

    Args:
        interface: The interface class to extract fields from

    Returns:
        Dictionary mapping field names to their types

    Example:
        >>> class MyInterface(Protocol):
        ...     counter: int
        ...     name: str
        >>> get_interface_fields(MyInterface)
        {'counter': <class 'int'>, 'name': <class 'str'>}
    """
    try:
        hints = get_type_hints(interface)
        # Filter out private attributes
        return {
            name: typ for name, typ in hints.items()
            if not name.startswith('_')
        }
    except Exception:
        return {}


def create_interface_from_model(model_class: type, readonly_fields: set[str] | None = None) -> type[Protocol]:
    """
    Create a Protocol interface from a Pydantic model or dataclass.

    This is a utility function for automatically generating interfaces from
    existing model classes, supporting gradual migration.

    Args:
        model_class: The model class to create interface from
        readonly_fields: Optional set of field names that should be read-only

    Returns:
        A Protocol class with the same fields as the model

    Example:
        >>> from pydantic import BaseModel
        >>> class TaskModel(BaseModel):
        ...     max_tasks: int
        ...     tasks_completed: int
        >>> TaskInterface = create_interface_from_model(
        ...     TaskModel,
        ...     readonly_fields={'max_tasks'}
        ... )
        >>> get_interface_fields(TaskInterface)
        {'max_tasks': <class 'int'>, 'tasks_completed': <class 'int'>}
    """
    readonly_fields = readonly_fields or set()

    # Get field annotations from the model
    try:
        hints = get_type_hints(model_class)
    except Exception:
        hints = {}

    # Create protocol namespace
    namespace = {
        '__annotations__': {},
        '__doc__': f"Auto-generated interface from {model_class.__name__}",
        '_readonly_fields': readonly_fields,
    }

    # Add field annotations
    for field_name, field_type in hints.items():
        if not field_name.startswith('_'):
            namespace['__annotations__'][field_name] = field_type

    # Create the Protocol class
    interface_name = f"{model_class.__name__}Interface"
    interface_class = type(interface_name, (Protocol,), namespace)

    return interface_class


# ============================================================================
# Public API Exports
# ============================================================================

__all__ = [
    # Type Variables
    "T",
    "P",

    # Core Protocols
    "Readable",
    "Writable",
    "FieldAccessor",
    "BlackboardProtocol",

    # Metadata
    "FieldMetadata",
    "InterfaceMetadata",

    # Helper Functions
    "validate_implements",
    "get_interface_fields",
    "create_interface_from_model",
    "_extract_interface_metadata",
]
