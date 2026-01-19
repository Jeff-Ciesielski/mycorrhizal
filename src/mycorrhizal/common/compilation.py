#!/usr/bin/env python3
"""
Common Compilation Cache for DSL Systems

This module provides a shared compilation cache for all Mycorrhizal DSL systems
(Rhizomorph, Septum, Hypha) to eliminate redundant runtime type inspection overhead.

The compilation cache stores pre-computed interface metadata for functions, including:
- Interface type detection (Protocol-based)
- Readonly and readwrite field extraction
- Parameter type analysis

This module uses type-based blackboard detection instead of parameter name matching,
making it more robust and type-safe.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Set,
    Type,
    get_type_hints,
)


# ======================================================================================
# Compiled Metadata
# ======================================================================================


@dataclass(frozen=True)
class CompiledMetadata:
    """
    Pre-computed interface metadata for a function.

    This dataclass stores the results of type inspection so that we don't need
    to inspect function signatures at runtime. Metadata is compiled lazily on
    first use and then cached globally.

    Attributes:
        has_interface: Whether the function has an interface type hint
        interface_type: The interface Protocol type (if any)
        readonly_fields: Set of readonly field names from the interface
        readwrite_fields: Set of readwrite field names from the interface
        bb_param_index: Index of the blackboard parameter in function signature
    """
    has_interface: bool
    interface_type: Optional[Type]
    readonly_fields: Optional[Set[str]] = None
    readwrite_fields: Optional[Set[str]] = None
    bb_param_index: int = 0


# ======================================================================================
# Global Compilation Cache
# ======================================================================================

_compilation_cache: Dict[int, CompiledMetadata] = {}


def _clear_compilation_cache() -> None:
    """Clear the global compilation cache. Useful for testing."""
    global _compilation_cache
    _compilation_cache.clear()


def _get_compilation_cache_size() -> int:
    """Get the current size of the compilation cache (for debugging)."""
    return len(_compilation_cache)


# ======================================================================================
# Compilation Functions
# ======================================================================================


def _compile_function_metadata(
    func: Callable,
    bb_param_name: str = "bb",
    ctx_param_name: str = "ctx",
) -> CompiledMetadata:
    """
    Compile function metadata by inspecting type hints and extracting interface information.

    This function performs the expensive type inspection operations once, so that
    the compiled metadata can be cached and reused without repeated inspection.

    The function uses type-based blackboard detection:
    - Checks if the parameter type is a Protocol with interface metadata
    - Does NOT rely on parameter names (more type-safe)

    Args:
        func: The function to inspect
        bb_param_name: Expected blackboard parameter name (for Rhizomorph/Hypha)
        ctx_param_name: Expected context parameter name (for Septum)

    Returns:
        CompiledMetadata with pre-extracted interface information

    Raises:
        TypeError: If the function is not callable
        NameError: If type hints reference undefined types
        AttributeError: If type hints reference undefined attributes
    """
    # Validate input
    if not callable(func):
        raise TypeError(f"Expected callable, got {type(func).__name__}")

    # Get function name for error messages (handle callables without __name__)
    func_name = getattr(func, '__name__', repr(func))

    try:
        sig = inspect.signature(func)

        # Try to get type hints, but handle callable objects that don't support it
        try:
            hints = get_type_hints(func)
        except TypeError:
            # Callable objects (like TransitionRuntimeWrapper) may not support get_type_hints
            # Return metadata without interface in this case
            return CompiledMetadata(
                has_interface=False,
                interface_type=None,
                readonly_fields=None,
                readwrite_fields=None,
                bb_param_index=0,
            )

        params = list(sig.parameters.values())

        has_interface = False
        interface_type = None
        readonly_fields = None
        readwrite_fields = None
        bb_param_index = 0

        # Find the blackboard/context parameter by TYPE, not by NAME
        for idx, param in enumerate(params):
            param_type = hints.get(param.name)

            if param_type is None:
                continue

            # Check for Protocol with interface metadata
            # This works for both direct Protocol types and generic types like SharedContext[Interface]
            if _has_interface_metadata(param_type):
                has_interface = True
                interface_type = param_type
                bb_param_index = idx

                # Extract interface fields
                # For generic types, extract the type argument
                actual_type = _extract_generic_type_arg(param_type)
                if actual_type and _has_interface_metadata(actual_type):
                    # For generic types, use the extracted type for the interface
                    interface_type = actual_type
                    readonly_fields = getattr(actual_type, '_readonly_fields', set())
                    readwrite_fields = getattr(actual_type, '_readwrite_fields', set())
                else:
                    # For direct Protocol types
                    readonly_fields = getattr(param_type, '_readonly_fields', set())
                    readwrite_fields = getattr(param_type, '_readwrite_fields', set())

                break

        return CompiledMetadata(
            has_interface=has_interface,
            interface_type=interface_type,
            readonly_fields=readonly_fields,
            readwrite_fields=readwrite_fields,
            bb_param_index=bb_param_index,
        )

    except (TypeError, NameError, AttributeError) as e:
        # Re-raise expected exceptions with better context
        raise type(e)(
            f"Failed to compile metadata for {func_name}: {e}"
        ) from e


def _has_interface_metadata(type_hint: Any) -> bool:
    """
    Check if a type hint has interface metadata (readonly/readwrite fields).

    Args:
        type_hint: A type hint from get_type_hints()

    Returns:
        True if the type has _readonly_fields attribute (interface metadata)
    """
    # For generic types (like SharedContext[Interface]), extract the type argument
    actual_type = _extract_generic_type_arg(type_hint)
    if actual_type:
        return hasattr(actual_type, '_readonly_fields')

    # For direct Protocol types
    return hasattr(type_hint, '_readonly_fields')


def _extract_generic_type_arg(type_hint: Any) -> Optional[Type]:
    """
    Extract the type argument from a generic type.

    For example:
    - SharedContext[MyInterface] -> MyInterface
    - list[MyType] -> MyType
    - dict[str, int] -> None (not a single-type generic)

    Args:
        type_hint: A type hint from get_type_hints()

    Returns:
        The extracted type argument, or None if not a single-type generic
    """
    # Check if type has __args__ (generic type)
    if not hasattr(type_hint, '__args__'):
        return None

    args = type_hint.__args__
    if not args or len(args) != 1:
        return None

    return args[0]


def _get_compiled_metadata(func: Callable) -> CompiledMetadata:
    """
    Get compiled metadata for a function, using write-through cache with EAFP pattern.

    This function checks the global compilation cache first using try/except
    (EAFP - Easier to Ask for Forgiveness than Permission), which is faster
    than inclusion checks in Python.

    If the function's metadata hasn't been compiled yet, it compiles it and
    writes it to the cache.

    Args:
        func: The function to get metadata for

    Returns:
        CompiledMetadata for the function

    Raises:
        TypeError: If func is not callable
        ValueError: If type hints are malformed
        AttributeError: If type hints reference undefined types
    """
    func_id = id(func)

    # EAFP pattern: try to get from cache, handle miss
    try:
        return _compilation_cache[func_id]
    except KeyError:
        # Cache miss - compile and write through
        metadata = _compile_function_metadata(func)
        _compilation_cache[func_id] = metadata
        return metadata
