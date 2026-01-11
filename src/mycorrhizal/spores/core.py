#!/usr/bin/env python3
"""
Spores Core API

Main spores module with configuration and decorator functionality.
"""

from __future__ import annotations

import asyncio
import inspect
import functools
import logging
from datetime import datetime
from typing import (
    Any, Callable, Optional, Union, Dict, List,
    ParamSpec, TypeVar, overload
)
from dataclasses import dataclass

from .models import (
    Event, Object, LogRecord, Relationship,
    EventAttributeValue, ObjectAttributeValue,
    ObjectScope, ObjectRef, EventAttr,
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
        transport: Transport instance to use (required for logging)
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
    Send a log record via the configured transport.

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

        # Send via transport
        await config.transport.send(data, content_type)

    except Exception as e:
        logger.error(f"Failed to send log record: {e}")


# ============================================================================
# Decorators
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

                # Log the event asynchronously
                asyncio.create_task(self._log_event(
                    func, args, kwargs, event_type, attributes, objects
                ))

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

            # 1. Extract from blackboard with ObjectRef metadata
            if bb is not None:
                bb_objects = extraction.extract_objects_from_blackboard(bb, objects)
                event_objects.extend(bb_objects)

            # 2. Manual object specification
            if isinstance(attributes, dict):
                # Check for object specifications (tuples with qualifier)
                manual_objects = extraction.extract_objects_from_spec(
                    attributes, timestamp
                )
                event_objects.extend(manual_objects)

            # Build relationships
            relationships = {}
            for obj in event_objects:
                rel = Relationship(object_id=obj.id, qualifier="related")
                relationships[obj.id] = rel

            # Build event
            event = Event(
                id=generate_event_id(),
                type=event_type,
                time=timestamp,
                attributes=event_attrs,
                relationships=relationships
            )

            # Send event
            await _send_log_record(LogRecord(event=event))

            # Send objects to cache
            cache = get_object_cache()
            for obj in event_objects:
                cache.contains_or_add(obj.id, obj)

        except Exception as e:
            logger.error(f"Failed to log event: {e}")

    def _find_blackboard(self, args: tuple, kwargs: dict) -> Optional[Any]:
        """Find blackboard in function arguments."""
        # Try common parameter names
        for param_name in ('bb', 'blackboard', 'ctx', 'context'):
            if param_name in kwargs:
                return kwargs[param_name]

        # Try positional arguments
        for i, arg in enumerate(args):
            # Skip self
            if i == 0 and inspect.ismethod(args[0] if args else None):
                continue

            # Check for common blackboard patterns
            type_name = type(arg).__name__.lower()
            if 'blackboard' in type_name or 'context' in type_name:
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
            return cls

        return decorator


# Global spores decorator instance
spore = SporeDecorator()


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'configure',
    'get_config',
    'get_object_cache',
    'spore',
    'SporesConfig',
    # Models
    'Event',
    'Object',
    'LogRecord',
    'Relationship',
    'EventAttributeValue',
    'ObjectAttributeValue',
    'ObjectRef',
    'ObjectScope',
    'EventAttr',
    'generate_event_id',
    'generate_object_id',
    # Cache
    'ObjectLRUCache',
    # Encoder
    'Encoder',
    'JSONEncoder',
    # Transport
    'Transport',
    'HTTPTransport',
]
