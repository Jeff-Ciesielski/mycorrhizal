"""
Automatic interface view creation for Mycorrhizal DSLs.

This module provides utilities to automatically detect interface types
in function signatures and create constrained views, ensuring type-safe
access control without requiring manual view creation.
"""
from typing import Any, Callable, Type, TypeVar, get_type_hints
from inspect import signature, Parameter
from functools import lru_cache

from mycorrhizal.common.wrappers import create_view_from_protocol


# Cache for interface detection to avoid repeated introspection
_interface_cache: dict[Type, bool] = {}


def is_interface_type(t: Type) -> bool:
    """
    Check if a type is a blackboard interface.

    Interfaces are marked by the presence of _readonly_fields and
    _readwrite_fields attributes added by @blackboard_interface.

    Args:
        t: Type to check

    Returns:
        True if t is a blackboard interface, False otherwise
    """
    if t in _interface_cache:
        return _interface_cache[t]

    result = (
        hasattr(t, '_readonly_fields') and
        hasattr(t, '_readwrite_fields') and
        callable(t)  # Interfaces are also protocols
    )
    _interface_cache[t] = result
    return result


def get_interface_parameters(func: Callable) -> dict[str, Type]:
    """
    Extract parameters with interface types from a function signature.

    Args:
        func: Function to inspect

    Returns:
        Dictionary mapping parameter names to their interface types
    """
    interface_params = {}

    try:
        type_hints = get_type_hints(func)
        sig = signature(func)

        for param_name, param in sig.parameters.items():
            if param_name == 'bb':
                # Skip the bb parameter itself (will be replaced)
                continue

            if param_name in type_hints:
                param_type = type_hints[param_name]
                if is_interface_type(param_type):
                    interface_params[param_name] = param_type
    except (ValueError, TypeError):
        # get_type_hints can fail for some functions
        pass

    return interface_params


def create_interface_views(
    bb: Any,
    func: Callable,
    view_cache: dict[int, dict[Type, Any]] | None = None
) -> dict[str, tuple[Any, Type]]:
    """
    Create interface views for a function based on its signature.

    Inspects the function's type hints and automatically creates
    constrained views for any parameters typed as interfaces.

    Args:
        bb: The blackboard to wrap
        func: The function whose signature will be inspected
        view_cache: Optional cache mapping (bb_id, interface_type) -> view

    Returns:
        Dictionary mapping parameter names to (view, interface_type) tuples
    """
    interface_views = {}

    if view_cache is None:
        view_cache = {}

    bb_id = id(bb)

    interface_params = get_interface_parameters(func)

    for param_name, interface_type in interface_params.items():
        # Check cache first
        cache_key = (bb_id, interface_type)
        if cache_key in view_cache:
            view = view_cache[cache_key]
        else:
            # Create new view
            readonly_fields = getattr(interface_type, '_readonly_fields', set())
            view = create_view_from_protocol(
                bb,
                interface_type,
                readonly_fields=readonly_fields
            )
            view_cache[cache_key] = view

        interface_views[param_name] = (view, interface_type)

    return interface_views


def prepare_function_args(
    bb: Any,
    func: Callable,
    view_cache: dict[int, dict[Type, Any]] | None = None
) -> dict[str, Any]:
    """
    Prepare function arguments, creating interface views as needed.

    This function:
    1. Inspects the function signature for interface-typed parameters
    2. Creates constrained views for those parameters
    3. Returns a dict of arguments ready to pass to the function

    Args:
        bb: The blackboard
        func: The function to call
        view_cache: Optional cache for created views

    Returns:
        Dictionary of parameter names to values (views or original values)
    """
    interface_views = create_interface_views(bb, func, view_cache)
    return {name: view for name, (view, _) in interface_views.items()}


def should_wrap_with_interface(func: Callable) -> bool:
    """
    Check if a function has any interface-typed parameters.

    Args:
        func: Function to check

    Returns:
        True if function has interface parameters, False otherwise
    """
    return len(get_interface_parameters(func)) > 0
