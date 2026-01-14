#!/usr/bin/env python3
"""
Spores Core API

Main spores module with configuration and logger functionality.
"""

from __future__ import annotations

import asyncio
import inspect
import functools
import logging
import threading
from datetime import datetime
from typing import (
    Any, Callable, Optional, Union, Dict, List,
    ParamSpec, TypeVar, overload, get_origin, get_args, Annotated
)
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .models import (
    Event, Object, LogRecord, Relationship,
    EventAttributeValue, ObjectAttributeValue,
    ObjectScope, ObjectRef, EventAttr, SporesAttr,
    generate_event_id, generate_object_id,
    attribute_value_from_python, object_attribute_from_python
)
from .cache import ObjectLRUCache
from .encoder import Encoder, JSONEncoder
from .transport import Transport
from . import extraction


logger = logging.getLogger(__name__)

# Type variables for decorators
P = ParamSpec('P')
R = TypeVar('R')


# ============================================================================
# Global Configuration
# ============================================================================

@dataclass
class SporesConfig:
    """
    Global configuration for the Spores logging system.

    Attributes:
        enabled: Whether spores logging is enabled
        object_cache_size: Maximum objects in LRU cache
        encoder: Encoder instance to use
        transport: Transport instance to use (required for logging to work)
    """
    enabled: bool = True
    object_cache_size: int = 128
    encoder: Optional[Encoder] = None
    transport: Optional[Transport] = None

    def __post_init__(self):
        """Set default encoder if not provided."""
        if self.encoder is None:
            self.encoder = JSONEncoder()


# Global state
_config: Optional[SporesConfig] = None
_object_cache: Optional[ObjectLRUCache[str, Object]] = None


def configure(
    enabled: bool = True,
    object_cache_size: int = 128,
    encoder: Optional[Encoder] = None,
    transport: Optional[Transport] = None
) -> None:
    """
    Configure the Spores logging system.

    This should be called once at application startup.

    Args:
        enabled: Whether spores logging is enabled (default: True)
        object_cache_size: Maximum objects in LRU cache (default: 128)
        encoder: Encoder instance (defaults to JSONEncoder)
        transport: Transport instance (required for logging to work)

    Example:
        ```python
        from mycorrhizal.spores.transport import FileTransport

        spore.configure(
            transport=FileTransport("logs/ocel.jsonl"),
            object_cache_size=256
        )
        ```
    """
    global _config, _object_cache

    _config = SporesConfig(
        enabled=enabled,
        object_cache_size=object_cache_size,
        encoder=encoder,
        transport=transport
    )

    # Create object cache with eviction callback
    def on_evict(object_id: str, obj: Object):
        """Send object when evicted from cache."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(_send_log_record(LogRecord(object=obj)))
            else:
                # No running loop, schedule the coroutine
                asyncio.ensure_future(_send_log_record(LogRecord(object=obj)))
        except RuntimeError:
            # No event loop at all, ignore for now
            pass

    def on_first_sight(object_id: str, obj: Object):
        """Send object when first seen."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(_send_log_record(LogRecord(object=obj)))
            else:
                # No running loop, schedule the coroutine
                asyncio.ensure_future(_send_log_record(LogRecord(object=obj)))
        except RuntimeError:
            # No event loop at all, ignore for now
            pass

    _object_cache = ObjectLRUCache(
        maxsize=object_cache_size,
        on_evict=on_evict,
        on_first_sight=on_first_sight
    )

    logger.info(f"Spores configured: enabled={enabled}, cache_size={object_cache_size}")


def get_config() -> SporesConfig:
    """Get the current spores configuration."""
    global _config
    if _config is None:
        # Use default configuration
        _config = SporesConfig()
    return _config


def get_object_cache() -> ObjectLRUCache[str, Object]:
    """Get the object cache."""
    global _object_cache
    if _object_cache is None:
        configure()  # Initialize with defaults
    return _object_cache


async def _send_log_record(record: LogRecord) -> None:
    """
    Send a log record via the configured transport (async).

    Args:
        record: The LogRecord to send
    """
    config = get_config()

    if not config.enabled:
        return

    if config.transport is None or config.encoder is None:
        logger.warning("Spores not properly configured (missing transport or encoder)")
        return

    try:
        # Encode the record
        data = config.encoder.encode(record)
        content_type = config.encoder.content_type()

        # Send via transport (async)
        await config.transport.send(data, content_type)

    except Exception as e:
        logger.error(f"Failed to send log record: {e}")


def _send_log_record_sync(record: LogRecord) -> None:
    """
    Send a log record via the configured transport (sync).

    This is a synchronous version that uses blocking I/O.
    Used by SyncEventLogger in daemon threads.

    Args:
        record: The LogRecord to send
    """
    config = get_config()

    if not config.enabled:
        return

    if config.transport is None or config.encoder is None:
        logger.warning("Spores not properly configured (missing transport or encoder)")
        return

    try:
        # Encode the record
        data = config.encoder.encode(record)
        content_type = config.encoder.content_type()

        # Send via transport (blocking call)
        config.transport.send(data, content_type)

    except Exception as e:
        logger.error(f"Failed to send log record: {e}")


# ============================================================================
# Logger Interface
# ============================================================================

class EventLogger(ABC):
    """
    Abstract base class for event loggers.

    Implementations can be sync or async, but the interface is the same.
    """

    @abstractmethod
    def event(self, event_type: str, **kwargs):
        """Log an event."""
        pass

    @abstractmethod
    def log_object(self, obj_type: str, obj_id: str, **kwargs):
        """Log an object."""
        pass


class AsyncEventLogger(EventLogger):
    """
    Async event logger for use in async contexts.
    """

    def __init__(self, name: str):
        self.name = name

    async def event(self, event_type: str, relationships: Dict[str, Relationship] | None = None, **kwargs) -> None:
        """
        Log an event asynchronously.

        Args:
            event_type: The type of event
            relationships: Optional dict of qualifier -> Relationship for OCEL object relationships
            **kwargs: Event attributes
        """
        config = get_config()
        if not config.enabled:
            return

        timestamp = datetime.now()

        attr_values = {}
        for key, value in kwargs.items():
            attr_values[key] = attribute_value_from_python(value)

        event = Event(
            id=generate_event_id(),
            type=event_type,
            time=timestamp,
            attributes=attr_values,
            relationships=relationships or {}
        )

        record = LogRecord(event=event)
        await _send_log_record(record)

    async def log_object(self, obj_type: str, obj_id: str, **kwargs) -> None:
        """Log an object asynchronously."""
        config = get_config()
        if not config.enabled:
            return

        timestamp = datetime.now()

        attr_values = {}
        for key, value in kwargs.items():
            attr_values[key] = object_attribute_from_python(value, time=timestamp)

        obj = Object(
            id=obj_id,
            type=obj_type,
            attributes=attr_values
        )

        record = LogRecord(object=obj)
        await _send_log_record(record)

    def log_event(self, event_type: str, relationships: Dict[str, tuple] | None = None, attributes: Dict[str, Any] | None = None):
        """
        Decorator factory that logs events with auto-logged object relationships (async version).

        Args:
            event_type: The type of event to log
            relationships: Dict mapping qualifiers to relationship specs:
                - qualifier: Relationship qualifier (e.g., "order", "customer")
                - spec: 2-tuple (source, obj_type) or 3-tuple (source, obj_type, attrs)
                    - source: Where to get object ("return", "ret", param name, "self")
                    - obj_type: OCEL object type
                    - attrs: Optional list of attribute names to log (omitted = auto-detect SporesAttr)
            attributes: Dict mapping event attribute names to values or callables

        Returns:
            Decorator function
        """
        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @functools.wraps(func)
            async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                # Execute the function
                result = await func(*args, **kwargs)

                # Build context for resolving objects and attributes
                context = self._build_context(func, args, kwargs, result)

                # Process relationships - log objects and build event relationships
                event_relationships = {}
                if relationships:
                    for qualifier, rel_spec in relationships.items():
                        # Handle both 2-tuple and 3-tuple formats
                        if len(rel_spec) == 2:
                            source, obj_type = rel_spec
                            attrs = None  # Auto-detect SporesAttr
                        else:
                            source, obj_type, attrs = rel_spec

                        # Resolve object from source
                        obj = self._resolve_source(source, context, obj_type)
                        if obj is None:
                            continue

                        # Extract object ID
                        obj_id = self._get_object_id(obj)
                        if obj_id is None:
                            continue

                        # Extract and log object attributes
                        obj_attrs = self._extract_object_attrs(obj, attrs)
                        await self.log_object(obj_type, obj_id, **obj_attrs)

                        # Add to event relationships
                        event_relationships[qualifier] = Relationship(object_id=obj_id, qualifier=qualifier)

                # Extract event attributes
                event_attrs = {}
                if attributes:
                    for attr_name, attr_value in attributes.items():
                        event_attrs[attr_name] = self._evaluate_expression(attr_value, context)

                # Log the event with relationships
                await self.event(event_type, relationships=event_relationships, **event_attrs)

                return result

            return wrapper  # type: ignore
        return decorator

    def _build_context(self, func: Callable, args: tuple, kwargs: dict, result: Any) -> Dict[str, Any]:
        """Build context dict for async version."""
        context = {
            'return': result,
            'ret': result,
        }

        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        for param_name, param_value in bound.arguments.items():
            context[param_name] = param_value

        return context

    def _resolve_source(self, source: str, context: Dict[str, Any], obj_type: str | None = None) -> Any:
        """Resolve source for async version.

        If source is "bb" (blackboard) and obj_type is provided, extract the field
        of that type from the blackboard instead of returning the entire blackboard.
        """
        if source == "return" or source == "ret":
            return context.get("return")
        elif source == "self":
            return context.get("self")
        else:
            obj = context.get(source)

            # If source is blackboard and we need a specific type, extract that field
            if obj_type and source == "bb" and hasattr(obj, '__annotations__'):
                return self._find_object_by_type(obj, obj_type)

            return obj

    def _find_object_by_type(self, blackboard: Any, obj_type: str) -> Any:
        """Find a field in the blackboard that matches the requested object type.

        Scans the blackboard's fields and returns the first field whose type
        annotation matches obj_type.
        """
        # Get the class to check annotations
        obj_class = blackboard if isinstance(blackboard, type) else type(blackboard)

        if not hasattr(obj_class, '__annotations__'):
            return blackboard

        # Check each field's type annotation
        for field_name, field_type in obj_class.__annotations__.items():
            # Handle Annotated types
            if get_origin(field_type) is Annotated:
                args = get_args(field_type)
                if args:
                    actual_type = args[0]
                    # Check if type name matches (handle both str and type)
                    type_name = actual_type if isinstance(actual_type, str) else actual_type.__name__
                    if type_name == obj_type:
                        field_value = getattr(blackboard, field_name, None)
                        # Return the actual value, not None
                        if field_value is not None:
                            return field_value
            # Handle Union types (e.g., Sample | None)
            elif get_origin(field_type) is Union:
                args = get_args(field_type)
                for arg in args:
                    # Skip None
                    if arg is type(None):
                        continue
                    # Check if this arg matches our target type
                    if get_origin(arg) is Annotated:
                        annotated_args = get_args(arg)
                        if annotated_args:
                            actual_type = annotated_args[0]
                            type_name = actual_type if isinstance(actual_type, str) else actual_type.__name__
                            if type_name == obj_type:
                                field_value = getattr(blackboard, field_name, None)
                                if field_value is not None:
                                    return field_value
                    else:
                        type_name = arg if isinstance(arg, str) else arg.__name__
                        if type_name == obj_type:
                            field_value = getattr(blackboard, field_name, None)
                            if field_value is not None:
                                return field_value
            else:
                # Check if type name matches
                type_name = field_type if isinstance(field_type, str) else field_type.__name__
                if type_name == obj_type:
                    field_value = getattr(blackboard, field_name, None)
                    if field_value is not None:
                        return field_value

        # Fallback: return blackboard if no matching field found
        return blackboard

    def _get_object_id(self, obj: Any) -> str | None:
        """Extract object ID for async version."""
        if obj is None:
            return None

        if hasattr(obj, 'id'):
            return str(getattr(obj, 'id'))
        else:
            return str(obj)

    def _extract_object_attrs(self, obj: Any, attrs_spec: list | dict) -> Dict[str, Any]:
        """Extract attributes for async version."""
        if attrs_spec is None:
            return self._extract_spores_attrs(obj)

        if isinstance(attrs_spec, list):
            result = {}
            for attr_name in attrs_spec:
                if hasattr(obj, attr_name):
                    result[attr_name] = getattr(obj, attr_name)
            return result

        elif isinstance(attrs_spec, dict):
            result = {}
            for attr_name, expr in attrs_spec.items():
                if isinstance(expr, str) and expr.startswith(("return.", "ret.", "self.")):
                    parts = expr.split(".", 1)
                    source = self._resolve_source(parts[0], {})
                    if source and len(parts) > 1:
                        result[attr_name] = getattr(source, parts[1])
                elif isinstance(expr, str) and hasattr(obj, expr):
                    result[attr_name] = getattr(obj, expr)
                elif callable(expr):
                    result[attr_name] = expr(obj)
                else:
                    result[attr_name] = expr
            return result

        return {}

    def _extract_spores_attrs(self, obj: Any) -> Dict[str, Any]:
        """Extract SporesAttr-marked fields for async version."""
        result = {}
        obj_class = obj if isinstance(obj, type) else type(obj)

        if hasattr(obj_class, '__annotations__'):
            for field_name, field_type in obj_class.__annotations__.items():
                if get_origin(field_type) is Annotated:
                    args = get_args(field_type)
                    for arg in args:
                        if arg is SporesAttr:
                            if hasattr(obj, field_name):
                                value = getattr(obj, field_name)
                                result[field_name] = value
                            break

        return result

    def _evaluate_expression(self, expr: Any, context: Dict[str, Any]) -> Any:
        """Evaluate expression for async version."""
        if callable(expr):
            sig = inspect.signature(expr)
            params = sig.parameters

            if len(params) == 1 and list(params.values())[0].kind == inspect.Parameter.VAR_POSITIONAL:
                return expr(**context)
            elif len(params) == 0:
                return expr()
            else:
                kwargs = {}
                for param_name in params:
                    if param_name in context:
                        kwargs[param_name] = context[param_name]
                return expr(**kwargs)
        elif isinstance(expr, str):
            # Check if it's a simple parameter reference
            if expr in context:
                return context[expr]
            # Check if it's a dotted attribute access
            elif "." in expr:
                parts = expr.split(".", 1)
                if parts[0] in context:
                    obj = context[parts[0]]
                    if hasattr(obj, parts[1]):
                        return getattr(obj, parts[1])
        return expr


class SyncEventLogger(EventLogger):
    """
    Sync event logger for use in synchronous contexts.

    Uses daemon threads for fire-and-forget logging - business logic never blocks.
    Logs are written via sync transport's blocking send() (no event loop needed).
    """

    def __init__(self, name: str):
        self.name = name

    def event(self, event_type: str, relationships: Dict[str, Relationship] | None = None, **kwargs) -> None:
        """
        Log an event in background daemon thread (fire-and-forget).

        Args:
            event_type: The type of event
            relationships: Optional dict of qualifier -> Relationship for OCEL object relationships
            **kwargs: Event attributes
        """
        config = get_config()
        if not config.enabled:
            return

        def log_in_thread():
            timestamp = datetime.now()

            attr_values = {}
            for key, value in kwargs.items():
                attr_values[key] = attribute_value_from_python(value)

            event = Event(
                id=generate_event_id(),
                type=event_type,
                time=timestamp,
                attributes=attr_values,
                relationships=relationships or {}
            )

            record = LogRecord(event=event)
            # Blocking send - no event loop needed
            _send_log_record_sync(record)

        thread = threading.Thread(target=log_in_thread, daemon=True)
        thread.start()

    def log_object(self, obj_type: str, obj_id: str, **kwargs) -> None:
        """Log an object in background daemon thread (fire-and-forget)."""
        config = get_config()
        if not config.enabled:
            return

        def log_in_thread():
            timestamp = datetime.now()

            attr_values = {}
            for key, value in kwargs.items():
                attr_values[key] = object_attribute_from_python(value, time=timestamp)

            obj = Object(
                id=obj_id,
                type=obj_type,
                attributes=attr_values
            )

            record = LogRecord(object=obj)
            # Blocking send - no event loop needed
            _send_log_record_sync(record)

        thread = threading.Thread(target=log_in_thread, daemon=True)
        thread.start()

    def log_event(self, event_type: str, relationships: Dict[str, tuple] | None = None, attributes: Dict[str, Any] | None = None):
        """
        Decorator factory that logs events with auto-logged object relationships.

        Args:
            event_type: The type of event to log
            relationships: Dict mapping qualifiers to relationship specs:
                - qualifier: Relationship qualifier (e.g., "order", "customer")
                - spec: 2-tuple (source, obj_type) or 3-tuple (source, obj_type, attrs)
                    - source: Where to get object ("return", "ret", param name, "self")
                    - obj_type: OCEL object type
                    - attrs: Optional list of attribute names to log (omitted = auto-detect SporesAttr)
            attributes: Dict mapping event attribute names to values or callables

        Returns:
            Decorator function

        Example:
            ```python
            @spore.log_event(
                event_type="OrderCreated",
                relationships={
                    "order": ("return", "Order"),  # 2-tuple: auto-detect SporesAttr
                    "customer": ("customer", "Customer"),  # 2-tuple: auto-detect SporesAttr
                },
                attributes={
                    "item_count": lambda items: len(items),
                },
            )
            def create_order(customer: Customer, items: list) -> Order:
                order = Order(...)
                return order
            ```
        """
        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @functools.wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                # Execute the function
                result = func(*args, **kwargs)

                # Build context for resolving objects and attributes
                context = self._build_context(func, args, kwargs, result)

                # Process relationships - log objects and build event relationships
                event_relationships = {}
                if relationships:
                    for qualifier, rel_spec in relationships.items():
                        # Handle both 2-tuple and 3-tuple formats
                        if len(rel_spec) == 2:
                            source, obj_type = rel_spec
                            attrs = None  # Auto-detect SporesAttr
                        else:
                            source, obj_type, attrs = rel_spec

                        # Resolve object from source
                        obj = self._resolve_source(source, context, obj_type)
                        if obj is None:
                            continue

                        # Extract object ID
                        obj_id = self._get_object_id(obj)
                        if obj_id is None:
                            continue

                        # Extract and log object attributes
                        obj_attrs = self._extract_object_attrs(obj, attrs)
                        self.log_object(obj_type, obj_id, **obj_attrs)

                        # Add to event relationships
                        event_relationships[qualifier] = Relationship(object_id=obj_id, qualifier=qualifier)

                # Extract event attributes
                event_attrs = {}
                if attributes:
                    for attr_name, attr_value in attributes.items():
                        event_attrs[attr_name] = self._evaluate_expression(attr_value, context)

                # Log the event with relationships
                self.event(event_type, relationships=event_relationships, **event_attrs)

                return result

            return wrapper  # type: ignore
        return decorator

    def _build_context(self, func: Callable, args: tuple, kwargs: dict, result: Any) -> Dict[str, Any]:
        """
        Build a context dict for resolving sources and evaluating expressions.

        Returns dict with:
            - 'return' and 'ret': The return value
            - 'self': For methods, the self parameter
            - All function parameters by name
        """
        context = {
            'return': result,
            'ret': result,
        }

        # Bind arguments to parameter names
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        for param_name, param_value in bound.arguments.items():
            context[param_name] = param_value

        return context

    def _resolve_source(self, source: str, context: Dict[str, Any], obj_type: str | None = None) -> Any:
        """
        Resolve an object from a source expression.

        Sources:
            - "return" or "ret": Return value
            - "self": For methods
            - Any other string: Parameter name from context
            - If source is "bb" (blackboard) and obj_type is provided, extract the field of that type
        """
        if source == "return" or source == "ret":
            return context.get("return")
        elif source == "self":
            return context.get("self")
        else:
            obj = context.get(source)

            # If source is blackboard and we need a specific type, extract that field
            if obj_type and source == "bb" and hasattr(obj, '__annotations__'):
                return self._find_object_by_type(obj, obj_type)

            return obj

    def _find_object_by_type(self, blackboard: Any, obj_type: str) -> Any:
        """Find a field in the blackboard that matches the requested object type.

        Scans the blackboard's fields and returns the first field whose type
        annotation matches obj_type.
        """
        # Get the class to check annotations
        obj_class = blackboard if isinstance(blackboard, type) else type(blackboard)

        if not hasattr(obj_class, '__annotations__'):
            return blackboard

        # Check each field's type annotation
        for field_name, field_type in obj_class.__annotations__.items():
            # Handle Annotated types
            if get_origin(field_type) is Annotated:
                args = get_args(field_type)
                if args:
                    actual_type = args[0]
                    # Check if type name matches (handle both str and type)
                    type_name = actual_type if isinstance(actual_type, str) else actual_type.__name__
                    if type_name == obj_type:
                        field_value = getattr(blackboard, field_name, None)
                        # Return the actual value, not None
                        if field_value is not None:
                            return field_value
            # Handle Union types (e.g., Sample | None)
            elif get_origin(field_type) is Union:
                args = get_args(field_type)
                for arg in args:
                    # Skip None
                    if arg is type(None):
                        continue
                    # Check if this arg matches our target type
                    if get_origin(arg) is Annotated:
                        annotated_args = get_args(arg)
                        if annotated_args:
                            actual_type = annotated_args[0]
                            type_name = actual_type if isinstance(actual_type, str) else actual_type.__name__
                            if type_name == obj_type:
                                field_value = getattr(blackboard, field_name, None)
                                if field_value is not None:
                                    return field_value
                    else:
                        type_name = arg if isinstance(arg, str) else arg.__name__
                        if type_name == obj_type:
                            field_value = getattr(blackboard, field_name, None)
                            if field_value is not None:
                                return field_value
            else:
                # Check if type name matches
                type_name = field_type if isinstance(field_type, str) else field_type.__name__
                if type_name == obj_type:
                    field_value = getattr(blackboard, field_name, None)
                    if field_value is not None:
                        return field_value

        # Fallback: return blackboard if no matching field found
        return blackboard

    def _get_object_id(self, obj: Any) -> str | None:
        """
        Extract object ID from an object.

        Tries:
            1. obj.id attribute
            2. str(obj)
        """
        if obj is None:
            return None

        if hasattr(obj, 'id'):
            return str(getattr(obj, 'id'))
        else:
            return str(obj)

    def _extract_object_attrs(self, obj: Any, attrs_spec: list | dict) -> Dict[str, Any]:
        """
        Extract attributes from an object for logging.

        Args:
            obj: The object to extract from
            attrs_spec: Either a list of attribute names, or a dict of {name: expression}

        Returns:
            Dict of attribute name -> value
        """
        if attrs_spec is None:
            # Check for SporesAttr annotations on the object's class
            return self._extract_spores_attrs(obj)

        if isinstance(attrs_spec, list):
            # Simple list of attribute names
            result = {}
            for attr_name in attrs_spec:
                if hasattr(obj, attr_name):
                    result[attr_name] = getattr(obj, attr_name)
            return result

        elif isinstance(attrs_spec, dict):
            # Dict with custom names or callables
            result = {}
            for attr_name, expr in attrs_spec.items():
                if isinstance(expr, str) and expr.startswith(("return.", "ret.", "self.")):
                    # Special handling for return/ret/self references
                    parts = expr.split(".", 1)
                    source = self._resolve_source(parts[0], {})
                    if source and len(parts) > 1:
                        result[attr_name] = getattr(source, parts[1])
                elif isinstance(expr, str) and hasattr(obj, expr):
                    # Simple attribute access
                    result[attr_name] = getattr(obj, expr)
                elif callable(expr):
                    # Callable expression
                    result[attr_name] = expr(obj)
                else:
                    # Static value
                    result[attr_name] = expr
            return result

        return {}

    def _extract_spores_attrs(self, obj: Any) -> Dict[str, Any]:
        """
        Extract attributes marked with SporesAttr from a Pydantic model.

        Checks the object's class __annotations__ for Annotated[type, SporesAttr] fields.
        """
        result = {}

        # Get the class (handle both instances and classes)
        obj_class = obj if isinstance(obj, type) else type(obj)

        if hasattr(obj_class, '__annotations__'):
            for field_name, field_type in obj_class.__annotations__.items():
                # Check if this is an Annotated type with SporesAttr
                if get_origin(field_type) is Annotated:
                    args = get_args(field_type)
                    for arg in args:
                        if arg is SporesAttr:
                            # Found a SporesAttr marker - log this field
                            if hasattr(obj, field_name):
                                value = getattr(obj, field_name)
                                result[field_name] = value
                            break

        return result

    def _evaluate_expression(self, expr: Any, context: Dict[str, Any]) -> Any:
        """
        Evaluate an expression for event or object attributes.

        Supports:
            - Static values (strings, numbers, etc.)
            - Callables (called with context)
            - Strings that are parameter references or attribute accesses
        """
        if callable(expr):
            # Callable - try to detect what params it wants
            sig = inspect.signature(expr)
            params = sig.parameters

            if len(params) == 1 and list(params.values())[0].kind == inspect.Parameter.VAR_POSITIONAL:
                # **kwargs style
                return expr(**context)
            elif len(params) == 0:
                # No params
                return expr()
            else:
                # Named params - pass relevant context
                kwargs = {}
                for param_name in params:
                    if param_name in context:
                        kwargs[param_name] = context[param_name]
                return expr(**kwargs)
        elif isinstance(expr, str):
            # Check if it's a simple parameter reference
            if expr in context:
                return context[expr]
            # Check if it's an attribute access like "order.id"
            elif "." in expr:
                parts = expr.split(".", 1)
                if parts[0] in context:
                    obj = context[parts[0]]
                    if hasattr(obj, parts[1]):
                        return getattr(obj, parts[1])
        return expr


def get_spore_sync(name: str) -> SyncEventLogger:
    """
    Get a synchronous spore logger.

    Use this in synchronous code. The logger uses daemon threads for
    fire-and-forget logging - business logic never blocks.

    Args:
        name: Spore name (typically __module__ or __name__)

    Returns:
        A SyncEventLogger instance

    Example:
        ```python
        from mycorrhizal.spores import configure, get_spore_sync
        from mycorrhizal.spores.transport import SyncFileTransport

        configure(transport=SyncFileTransport("logs/ocel.jsonl"))
        spore = get_spore_sync(__name__)

        @spore.log_event(event_type="OrderCreated", order_id="order.id")
        def create_order(order: Order) -> Order:
            return order
        ```
    """
    return SyncEventLogger(name)


def get_spore_async(name: str) -> AsyncEventLogger:
    """
    Get an asynchronous spore logger.

    Use this in asynchronous code. The logger uses async I/O.

    Args:
        name: Spore name (typically __module__ or __name__)

    Returns:
        An AsyncEventLogger instance

    Example:
        ```python
        from mycorrhizal.spores import configure, get_spore_async
        from mycorrhizal.spores.transport import AsyncFileTransport

        configure(transport=AsyncFileTransport("logs/ocel.jsonl"))
        spore = get_spore_async(__name__)

        @spore.log_event(event_type="OrderCreated", order_id="order.id")
        async def create_order(order: Order) -> Order:
            return order
        ```
    """
    return AsyncEventLogger(name)


# ============================================================================
# Legacy API (deprecated)
# ============================================================================

class SporeDecorator:
    """
    Main decorator interface for spores logging.

    Usage:
        ```python
        @spore.event(event_type="process_item")
        async def process(bb: Context) -> Status:
            return Status.SUCCESS
        ```
    """

    def event(
        self,
        event_type: str,
        attributes: Optional[Union[Dict[str, Any], List[str]]] = None,
        objects: Optional[List[str]] = None
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """
        Decorator to log events when decorated function is called.

        Args:
            event_type: The type/category of event
            attributes: Event attributes (dict or list of param names)
            objects: List of object fields to include (from interface)

        Returns:
            Decorator function
        """
        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                # Call the original function
                result = await func(*args, **kwargs)

                # Log the event
                await self._log_event(
                    func, args, kwargs, event_type, attributes, objects
                )

                return result

            @functools.wraps(func)
            def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                # Call the original function
                result = func(*args, **kwargs)

                # Log in background thread since we're not in async context
                import threading
                def log_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        new_loop.run_until_complete(self._log_event(
                            func, args, kwargs, event_type, attributes, objects
                        ))
                    finally:
                        new_loop.close()

                thread = threading.Thread(target=log_in_thread, daemon=True)
                thread.start()

                return result

            # Return appropriate wrapper based on whether function is async
            if asyncio.iscoroutinefunction(func):
                return async_wrapper  # type: ignore
            else:
                return sync_wrapper  # type: ignore

            return decorator

        return decorator

    async def _log_event(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        event_type: str,
        attributes: Optional[Union[Dict[str, Any], List[str]]],
        objects: Optional[List[str]]
    ) -> None:
        """Log an event with extracted context."""
        config = get_config()

        if not config.enabled:
            return

        try:
            timestamp = datetime.now()

            # Extract attributes from multiple sources
            event_attrs = {}

            # 1. Extract from function parameters
            param_attrs = extraction.extract_attributes_from_params(
                func, args, kwargs, timestamp
            )
            event_attrs.update(param_attrs)

            # 2. Extract from blackboard (if present)
            bb = self._find_blackboard(args, kwargs)
            if bb is not None:
                bb_attrs = extraction.extract_attributes_from_blackboard(bb, timestamp)
                event_attrs.update(bb_attrs)

            # 3. Static/computed attributes from decorator
            if isinstance(attributes, dict):
                computed_attrs = extraction.evaluate_computed_attributes(
                    attributes, bb, timestamp
                )
                event_attrs.update(computed_attrs)

            # Extract objects
            event_objects = []

            if objects is not None and bb is not None:
                # Extract objects from interface using proper Annotated type handling
                for obj_field in objects:
                    if hasattr(bb, obj_field):
                        obj = getattr(bb, obj_field)

                        # Get the field type from annotations
                        field_type = type(bb).__annotations__.get(obj_field)
                        if field_type is not None:
                            # Check if it's an Annotated type
                            if get_origin(field_type) is Annotated:
                                args = get_args(field_type)
                                if len(args) >= 2:
                                    # args[0] is the actual type, args[1:] are metadata
                                    for metadata in args[1:]:
                                        # Check if it's an ObjectRef
                                        if isinstance(metadata, ObjectRef):
                                            rel = Relationship(
                                                object_id=obj.id,
                                                qualifier=metadata.qualifier
                                            )
                                            event_objects.append((obj, rel))
                                            break

            # Create event
            event = Event(
                id=generate_event_id(),
                type=event_type,
                time=timestamp,
                attributes={
                    k: attribute_value_from_python(v)
                    for k, v in event_attrs.items()
                },
                relationships={
                    r.qualifier: r
                    for _, r in event_objects
                }
            )

            # Send event
            await _send_log_record(LogRecord(event=event))

            # Send object records (convert Pydantic models to OCEL Objects)
            for obj, _ in event_objects:
                ocel_obj = extraction.convert_to_ocel_object(obj)
                if ocel_obj:
                    obj_record = LogRecord(object=ocel_obj)
                    await _send_log_record(obj_record)

        except Exception as e:
            logger.error(f"Failed to log event: {e}")

    def _find_blackboard(self, args: tuple, kwargs: dict) -> Optional[Any]:
        """Find blackboard in arguments.

        Looks for objects marked with _is_blackboard, or falls back to
        the first argument that looks like a Pydantic model (has __annotations__).
        """
        # First, look for explicitly marked blackboard
        for arg in args:
            if hasattr(arg, '_is_blackboard'):
                return arg

        # Fallback: look for Pydantic BaseModel or dataclass with annotations
        for arg in args:
            if hasattr(arg, '__annotations__') and hasattr(arg, '__dict__'):
                # Looks like a data container (Pydantic model or dataclass)
                return arg

        return None

    def object(self, object_type: str) -> Callable[[type], type]:
        """
        Decorator to mark a class as an OCEL object type.

        Args:
            object_type: The OCEL object type name

        Returns:
            Class decorator

        Example:
            ```python
            @spore.object(object_type="WorkItem")
            class WorkItem(BaseModel):
                id: str
                status: str
            ```
        """
        def decorator(cls: type) -> type:
            # Store metadata on the class
            cls._spores_object_type = object_type
            # Store in global registry
            if not hasattr(self, '_object_types'):
                self._object_types = {}
            self._object_types[object_type] = cls
            return cls

        return decorator


# Create global spore instance for backward compatibility
spore = SporeDecorator()


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Configuration
    'configure',
    'get_config',
    'get_object_cache',

    # Spore getters (explicit)
    'get_spore_sync',
    'get_spore_async',

    # Logger types
    'EventLogger',
    'AsyncEventLogger',
    'SyncEventLogger',

    # Legacy (deprecated)
    'spore',
]
