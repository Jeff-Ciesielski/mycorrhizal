"""
Interface Builder DSL for Blackboard Interfaces

This module provides a declarative DSL for defining blackboard interfaces
using PEP 593 Annotated types and access control markers.

Key concepts:
- @blackboard_interface - Decorator for defining interfaces
- readonly/readwrite markers for field access control
- Annotated types for clean, Pythonic field definitions
"""

from typing import Protocol, TypeVar, Generic, Type, Any, Optional, get_type_hints, Annotated, get_args
from dataclasses import dataclass
import inspect


# ============================================================================
# Type Variables
# ============================================================================

T = TypeVar('T')


# ============================================================================
# Access Control Markers
# ============================================================================

class _AccessMarker:
    """Base class for access control markers"""
    pass


class readonly(_AccessMarker):
    """
    Marker for readonly fields in blackboard interfaces.

    Fields marked as readonly can be read but not modified through the interface.

    Example:
        ```python
        @blackboard_interface
        class SensorInterface:
            temperature: Annotated[float, readonly] = 20.0
        ```
    """
    pass


class readwrite(_AccessMarker):
    """
    Marker for readwrite fields in blackboard interfaces.

    Fields marked as readwrite can be both read and modified through the interface.

    Example:
        ```python
        @blackboard_interface
        class StateInterface:
            counter: Annotated[int, readwrite] = 0
        ```
    """
    pass


# ============================================================================
# Field Metadata
# ============================================================================

@dataclass
class FieldSpec:
    """Specification for a field in an interface"""
    name: str
    type: type
    readonly: bool = False
    computed: bool = False
    computation: Optional[callable] = None


# ============================================================================
# Decorator: blackboard_interface
# ============================================================================

def blackboard_interface(cls: Type[T] | None = None, *, name: str | None = None) -> Any:
    """
    Decorator for defining blackboard interfaces using Annotated types.

    Processes class annotations looking for Annotated types with readonly or
    readwrite markers, and generates a Protocol class with access control metadata.

    Args:
        cls: The class being decorated
        name: Optional name for the generated interface

    Returns:
        A Protocol class with type hints and metadata

    Example:
        ```python
        from typing import Annotated
        from mycorrhizal.common.interface_builder import blackboard_interface, readonly, readwrite

        @blackboard_interface
        class TaskInterface:
            max_tasks: Annotated[int, readonly] = 10
            tasks_completed: Annotated[int, readwrite] = 0
        ```

    Fields without Annotated markers are treated as readwrite by default.
    """
    def process_class(source_class: Type) -> Type:
        """Process the source class and generate a Protocol-like class"""
        # Get the interface name
        interface_name = name or source_class.__name__

        # Extract type hints
        try:
            hints = get_type_hints(source_class, include_extras=True)
        except Exception:
            hints = {}

        # Determine field metadata from Annotated types
        readonly_fields = set()
        readwrite_fields = set()

        for field_name, field_type in hints.items():
            if field_name.startswith('_'):
                continue

            # Check if this is an Annotated type with access marker
            if str(field_type).startswith('typing.Annotated['):
                # Extract metadata from Annotated[type, *markers]
                args = get_args(field_type)
                if len(args) >= 2:
                    actual_type = args[0]
                    # Check metadata for readonly or readwrite marker
                    metadata = args[1:]
                    found_marker = False
                    for marker in metadata:
                        if isinstance(marker, type) and issubclass(marker, _AccessMarker):
                            if marker is readonly:
                                readonly_fields.add(field_name)
                                found_marker = True
                                break
                            elif marker is readwrite:
                                readwrite_fields.add(field_name)
                                found_marker = True
                                break

                    if not found_marker:
                        # No access marker found, default to readwrite
                        readwrite_fields.add(field_name)
                else:
                    # Annotated with only type, no metadata
                    readwrite_fields.add(field_name)
            else:
                # Not Annotated, default to readwrite
                readwrite_fields.add(field_name)

        # Create a new class with Protocol-like behavior
        namespace = {
            '__annotations__': {},
            '_readonly_fields': frozenset(readonly_fields),
            '_readwrite_fields': frozenset(readwrite_fields),
            '__doc__': source_class.__doc__ or f"Interface: {interface_name}",
            '__module__': source_class.__module__,
            '__protocol_attrs__': frozenset(),  # For runtime_checkable
        }

        # Copy field annotations (use actual types from Annotated)
        for field_name, field_type in hints.items():
            if not field_name.startswith('_'):
                # Extract actual type from Annotated if present
                if hasattr(field_type, '__origin__') and field_type.__origin__ is Annotated:
                    actual_type = field_type.__args__[0]
                else:
                    actual_type = field_type
                namespace['__annotations__'][field_name] = actual_type

        # Create the interface class
        interface = type(interface_name, (), namespace)

        return interface

    # Support @blackboard_interface and @blackboard_interface()
    if cls is None:
        # Called with parentheses
        return lambda c: process_class(c)
    else:
        # Called without parentheses
        return process_class(cls)


# ============================================================================
# Public API Exports
# ============================================================================

__all__ = [
    "blackboard_interface",
    "readonly",
    "readwrite",
    "FieldSpec",
]
