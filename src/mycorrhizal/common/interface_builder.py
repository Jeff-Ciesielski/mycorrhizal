"""
Interface Builder DSL for Blackboard Interfaces

This module provides a declarative DSL for defining blackboard interfaces
using decorators. The DSL auto-generates Protocol classes with proper type hints
and metadata.

Key concepts:
- @blackboard_interface - Decorator for defining interfaces
- Field metadata stored in class attributes
- Auto-generation of Protocol classes
"""

from typing import Protocol, TypeVar, Generic, Type, Any, Optional, get_type_hints
from dataclasses import dataclass
import inspect


# ============================================================================
# Type Variables
# ============================================================================

T = TypeVar('T')


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
    Decorator for defining blackboard interfaces.

    Processes class annotations and generates a Protocol class with
    access control metadata.

    Args:
        cls: The class being decorated
        name: Optional name for the generated interface

    Returns:
        A Protocol class with type hints and metadata

    Example:
        ```python
        @blackboard_interface
        class TaskInterface:
            max_tasks: int  # Read-write by default
            tasks_completed: int  # Read-write by default
        ```
    """
    def process_class(source_class: Type) -> Type:
        """Process the source class and generate a Protocol-like class"""
        # Get the interface name
        interface_name = name or source_class.__name__

        # Extract type hints
        try:
            hints = get_type_hints(source_class)
        except Exception:
            hints = {}

        # Determine field metadata
        readonly_fields = set()
        readwrite_fields = set()

        # Check for _readonly_fields or _readwrite_fields attributes
        if hasattr(source_class, '_readonly_fields'):
            readonly_fields = set(source_class._readonly_fields)
        if hasattr(source_class, '_readwrite_fields'):
            readwrite_fields = set(source_class._readwrite_fields)

        # Default: if neither specified, all fields are read-write
        # If only readonly_fields specified, remaining fields are read-write
        if not readonly_fields and not readwrite_fields:
            readwrite_fields = set(hints.keys())
        elif readonly_fields and not readwrite_fields:
            # Remaining fields (not in readonly) are read-write
            readwrite_fields = set(hints.keys()) - readonly_fields

        # Create a new class with Protocol-like behavior
        # We don't inherit from Protocol to avoid the typing error
        namespace = {
            '__annotations__': {},
            '_readonly_fields': frozenset(readonly_fields),
            '_readwrite_fields': frozenset(readwrite_fields),
            '__doc__': source_class.__doc__ or f"Interface: {interface_name}",
            '__module__': source_class.__module__,
            '__protocol_attrs__': frozenset(),  # For runtime_checkable
        }

        # Copy field annotations
        for field_name, field_type in hints.items():
            if not field_name.startswith('_'):
                namespace['__annotations__'][field_name] = field_type

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
    "FieldSpec",
]
