"""
Blackboard Wrapper Classes for Constrained Access

This module provides wrapper classes that enforce interface constraints at runtime,
enabling type-safe, constrained access to blackboard state.

Key concepts:
- ReadOnlyView: Enforces read-only access to specified fields
- WriteOnlyView: Enforces write-only access to specified fields
- ConstrainedView: Provides access to subset of fields with permissions
- CompositeView: Combines multiple views into single interface
"""

from typing import TypeVar, Generic, Type, Protocol, Any, Set, Dict, Optional
from dataclasses import dataclass, field

from mycorrhizal.common.interfaces import (
    InterfaceMetadata,
    _extract_interface_metadata,
    get_interface_fields,
)


# ============================================================================
# Type Variables
# ============================================================================

T = TypeVar('T')
P = TypeVar('P')


# ============================================================================
# Exceptions
# ============================================================================

class AccessControlError(Exception):
    """Raised when access control is violated"""

    def __init__(self, message: str, field_name: str, operation: str):
        self.field_name = field_name
        self.operation = operation
        super().__init__(message)


# ============================================================================
# Base Wrapper Class
# ============================================================================

class BaseWrapper:
    """
    Base class for all wrapper classes.

    Provides common functionality for attribute access interception
    and error handling.
    """

    __slots__ = ('_bb', '_allowed_fields', '_private_attrs')

    def __init__(
        self,
        bb: Any,
        allowed_fields: Set[str],
        private_attrs: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the wrapper.

        Args:
            bb: The underlying blackboard object
            allowed_fields: Set of field names that can be accessed
            private_attrs: Optional dict of private attributes for the wrapper itself
        """
        object.__setattr__(self, '_bb', bb)
        object.__setattr__(self, '_allowed_fields', allowed_fields)
        object.__setattr__(self, '_private_attrs', private_attrs or {})

    def __getattr__(self, name: str) -> Any:
        """Get attribute from underlying blackboard"""
        if name in object.__getattribute__(self, '_allowed_fields'):
            bb = object.__getattribute__(self, '_bb')
            try:
                return getattr(bb, name)
            except AttributeError:
                # Re-raise with more helpful message
                raise AttributeError(
                    f"Field '{name}' does not exist on the underlying blackboard"
                ) from None
        raise AttributeError(
            f"Field '{name}' not accessible through this interface"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute on underlying blackboard (to be overridden by subclasses)"""
        # Default implementation allows all subclasses to override
        if name.startswith('_') or name in ('_bb', '_allowed_fields', '_private_attrs'):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(
                f"Cannot set field '{name}' through this interface"
            )

    def __dir__(self) -> list[str]:
        """Return list of accessible attributes"""
        allowed = object.__getattribute__(self, '_allowed_fields')
        return sorted(list(allowed))

    def __repr__(self) -> str:
        """Return string representation"""
        cls_name = self.__class__.__name__
        allowed = object.__getattribute__(self, '_allowed_fields')
        return f"<{cls_name} fields={sorted(allowed)}>"


# ============================================================================
# Read-Only View
# ============================================================================

class ReadOnlyView(BaseWrapper, Generic[T]):
    """
    Read-only view of specific blackboard fields.

    Provides read access to specified fields while preventing any modifications.
    Useful for creating safe, immutable views of shared state.

    Type Parameters:
        T: The type of the interface being viewed

    Example:
        >>> class Blackboard:
        ...     def __init__(self):
        ...         self.config_value = 42
        ...         self.mutable_state = 0
        >>> bb = Blackboard()
        >>> view = ReadOnlyView(bb, {'config_value'})
        >>> view.config_value  # Read works
        42
        >>> view.config_value = 100  # Write raises
        AttributeError: Field 'config_value' is read-only
    """

    __slots__ = ()

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent all modifications"""
        if name.startswith('_') or name in ('_bb', '_allowed_fields', '_private_attrs'):
            object.__setattr__(self, name, value)
        else:
            raise AccessControlError(
                f"Field '{name}' is read-only and cannot be modified",
                field_name=name,
                operation='write'
            )


# ============================================================================
# Write-Only View
# ============================================================================

class WriteOnlyView(BaseWrapper, Generic[T]):
    """
    Write-only view of specific blackboard fields.

    Provides write access to specified fields while preventing reads.
    Useful for output interfaces and sinks.

    Type Parameters:
        T: The type of the interface being viewed

    Example:
        >>> class Blackboard:
        ...     def __init__(self):
        ...         self.output_value = None
        >>> bb = Blackboard()
        >>> view = WriteOnlyView(bb, {'output_value'})
        >>> view.output_value = 42  # Write works
        >>> value = view.output_value  # Read raises
        AttributeError: Field 'output_value' is write-only (cannot read)
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        """Prevent reads"""
        allowed = object.__getattribute__(self, '_allowed_fields')
        if name in allowed:
            raise AccessControlError(
                f"Field '{name}' is write-only and cannot be read",
                field_name=name,
                operation='read'
            )
        raise AttributeError(
            f"Field '{name}' not accessible through this interface"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Allow writes to specified fields"""
        if name.startswith('_') or name in ('_bb', '_allowed_fields', '_private_attrs'):
            object.__setattr__(self, name, value)
        elif name in object.__getattribute__(self, '_allowed_fields'):
            bb = object.__getattribute__(self, '_bb')
            setattr(bb, name, value)
        else:
            raise AttributeError(
                f"Field '{name}' not accessible through this interface"
            )


# ============================================================================
# Constrained View
# ============================================================================

class ConstrainedView(BaseWrapper, Generic[T, P]):
    """
    Constrained access view for interface-based access.

    Provides access to a subset of fields with specified permissions.
    Read-only fields cannot be modified, read-write fields can be both read and written.

    Type Parameters:
        T: The type of the interface being viewed
        P: The Protocol type that defines the interface

    Example:
        >>> from mycorrhizal.common.wrappers import ConstrainedView
        >>> class Blackboard:
        ...     def __init__(self):
        ...         self.max_tasks = 10
        ...         self.tasks_completed = 0
        >>> bb = Blackboard()
        >>> view = ConstrainedView(
        ...     bb,
        ...     allowed_fields={'max_tasks', 'tasks_completed'},
        ...     readonly_fields={'max_tasks'}
        ... )
        >>> view.max_tasks  # Read works
        10
        >>> view.max_tasks = 20  # Write raises (read-only)
        AttributeError: Field 'max_tasks' is read-only
        >>> view.tasks_completed = 5  # Write works (read-write)
        >>> view.tasks_completed  # Read works
        5
    """

    __slots__ = ('_readonly_fields',)

    def __init__(
        self,
        bb: Any,
        allowed_fields: Set[str],
        readonly_fields: Set[str] | None = None,
        private_attrs: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the constrained view.

        Args:
            bb: The underlying blackboard object
            allowed_fields: Set of field names that can be accessed
            readonly_fields: Set of field names that are read-only
            private_attrs: Optional dict of private attributes for the wrapper itself
        """
        readonly_fields = readonly_fields or set()
        super().__init__(bb, allowed_fields, private_attrs)
        object.__setattr__(self, '_readonly_fields', readonly_fields)

    def __getattr__(self, name: str) -> Any:
        """Get attribute from underlying blackboard"""
        allowed = object.__getattribute__(self, '_allowed_fields')
        if name in allowed:
            bb = object.__getattribute__(self, '_bb')
            return getattr(bb, name)
        raise AttributeError(
            f"Field '{name}' not accessible through this interface"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute with read-only enforcement"""
        if name.startswith('_') or name in ('_bb', '_allowed_fields', '_readonly_fields', '_private_attrs'):
            object.__setattr__(self, name, value)
            return

        allowed = object.__getattribute__(self, '_allowed_fields')
        if name not in allowed:
            raise AttributeError(
                f"Field '{name}' not accessible through this interface"
            )

        readonly = object.__getattribute__(self, '_readonly_fields')
        if name in readonly:
            raise AccessControlError(
                f"Field '{name}' is read-only and cannot be modified",
                field_name=name,
                operation='write'
            )

        bb = object.__getattribute__(self, '_bb')
        setattr(bb, name, value)


# ============================================================================
# Composite View
# ============================================================================

class CompositeView(BaseWrapper, Generic[T]):
    """
    Combines multiple views into a single interface.

    Allows composing multiple constrained views into one, enabling
    complex access patterns by combining simpler views.

    Type Parameters:
        T: The type of the composite interface

    Example:
        >>> config_view = ReadOnlyView(bb, {'max_tasks'})
        >>> state_view = ConstrainedView(bb, {'tasks_completed'}, readonly_fields=set())
        >>> composite = CompositeView.combine([config_view, state_view])
        >>> composite.max_tasks  # Read from config_view
        10
        >>> composite.tasks_completed = 5  # Write through state_view
        >>> composite.max_tasks = 20  # Raises (read-only)
        AttributeError: Field 'max_tasks' is read-only
    """

    __slots__ = ('_views',)

    def __init__(self, bb: Any, views: list[BaseWrapper]):
        """
        Initialize composite view from multiple views.

        Args:
            bb: The underlying blackboard object
            views: List of wrapper objects to compose
        """
        # Collect all allowed fields from all views
        allowed_fields: Set[str] = set()
        private_attrs: Dict[str, Any] = {'_views': views}

        for view in views:
            allowed_fields.update(view._allowed_fields)

        super().__init__(bb, allowed_fields, private_attrs)
        object.__setattr__(self, '_views', views)

    @classmethod
    def combine(cls, views: list[BaseWrapper]) -> 'CompositeView[T]':
        """
        Combine multiple views into a composite view.

        Args:
            views: List of wrapper objects to combine

        Returns:
            A new CompositeView that combines all input views

        Raises:
            ValueError: If views reference different blackboards
        """
        if not views:
            raise ValueError("Cannot combine empty list of views")

        # Get the blackboard from the first view
        bb = views[0]._bb

        # Verify all views reference the same blackboard
        for view in views:
            if view._bb is not bb:
                raise ValueError("All views must reference the same blackboard")

        return cls(bb, views)

    def __getattr__(self, name: str) -> Any:
        """Delegate read to first view that has this field"""
        views = object.__getattribute__(self, '_views')
        for view in views:
            if name in view._allowed_fields:
                return getattr(view, name)

        raise AttributeError(
            f"Field '{name}' not accessible through any composed view"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Delegate write to first view that has this field"""
        if name.startswith('_') or name in ('_bb', '_allowed_fields', '_private_attrs', '_views'):
            object.__setattr__(self, name, value)
            return

        views = object.__getattribute__(self, '_views')
        for view in views:
            if name in view._allowed_fields:
                setattr(view, name, value)
                return

        raise AttributeError(
            f"Field '{name}' not accessible through any composed view"
        )


# ============================================================================
# Factory Functions
# ============================================================================

def create_readonly_view(bb: Any, fields: tuple[str, ...]) -> ReadOnlyView:
    """
    Create a read-only view.

    Args:
        bb: The blackboard to wrap
        fields: Tuple of field names for read-only access

    Returns:
        A ReadOnlyView instance

    Example:
        >>> view = create_readonly_view(bb, ('config', 'max_tasks'))
    """
    return ReadOnlyView(bb, set(fields))


def create_constrained_view(
    bb: Any,
    allowed_fields: tuple[str, ...],
    readonly_fields: tuple[str, ...] = ()
) -> ConstrainedView:
    """
    Create a constrained view.

    Args:
        bb: The blackboard to wrap
        allowed_fields: Tuple of field names that can be accessed
        readonly_fields: Tuple of field names that are read-only

    Returns:
        A ConstrainedView instance

    Example:
        >>> view = create_constrained_view(
        ...     bb,
        ...     ('max_tasks', 'tasks_completed'),
        ...     ('max_tasks',)
        ... )
    """
    return ConstrainedView(bb, set(allowed_fields), set(readonly_fields))


def create_view_from_protocol(
    bb: Any,
    protocol: Type[P],
    readonly_fields: Set[str] | None = None
) -> ConstrainedView:
    """
    Create a constrained view from a Protocol interface.

    Extracts field information from the Protocol and creates a view
    that enforces the interface constraints.

    Args:
        bb: The blackboard to wrap
        protocol: The Protocol class defining the interface
        readonly_fields: Optional set of field names that should be read-only.
                       If not provided, will try to extract from protocol metadata.

    Returns:
        A ConstrainedView instance

    Example:
        >>> class TaskInterface(Protocol):
        ...     max_tasks: int
        ...     tasks_completed: int
        >>> view = create_view_from_protocol(bb, TaskInterface, {'max_tasks'})
    """
    # Extract fields from protocol
    fields = get_interface_fields(protocol)
    allowed_fields = set(fields.keys())

    # If readonly_fields not provided, try to extract from protocol
    if readonly_fields is None:
        readonly_fields = getattr(protocol, '_readonly_fields', set())

    return ConstrainedView(bb, allowed_fields, readonly_fields)


# ============================================================================
# View Factory for Protocol-based Views
# ============================================================================

def View(bb: Any, protocol: Type[P]) -> Any:
    """
    Factory function for creating typed views from protocols.

    This provides a clean, type-safe API for creating constrained views
    with full type hint support.

    Args:
        bb: The blackboard to wrap
        protocol: The Protocol class defining the interface

    Returns:
        A wrapper that implements the protocol

    Example:
        >>> class ConfigInterface(Protocol):
        ...     max_tasks: int
        ...     interval: float
        >>> view: ConfigInterface = View(bb, ConfigInterface)
        >>> # Type checker knows view has max_tasks and interval
    """
    return create_view_from_protocol(bb, protocol)


# ============================================================================
# Public API Exports
# ============================================================================

__all__ = [
    # Exceptions
    "AccessControlError",

    # Wrapper Classes
    "BaseWrapper",
    "ReadOnlyView",
    "WriteOnlyView",
    "ConstrainedView",
    "CompositeView",

    # Factory Functions
    "create_readonly_view",
    "create_constrained_view",
    "create_view_from_protocol",
    "View",
]
