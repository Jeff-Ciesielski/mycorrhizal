#!/usr/bin/env python3
"""
Spores Adapter for Enoki (State Machines)

Provides logging integration for Enoki state machines.
Extracts context information and automatically creates event/object logs.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from ...spores import (
    get_config,
    Event,
    LogRecord,
    Relationship,
    EventAttributeValue,
    generate_event_id,
    attribute_value_from_python,
)
from ...spores.extraction import (
    extract_attributes_from_blackboard,
    extract_objects_from_blackboard,
)
from ...spores.core import _send_log_record, get_object_cache
from ...enoki.core import SharedContext, TransitionType


logger = logging.getLogger(__name__)


class EnokiAdapter:
    """
    Adapter for Enoki state machine logging.

    Provides decorators and helpers for logging:
    - State entry/exit/timeout events
    - State execution events
    - Transition events
    - Message/context object lifecycle tracking

    Usage:
        ```python
        from mycorrhizal.spores.dsl import EnokiAdapter

        adapter = EnokiAdapter()

        @enoki.state()
        def MyState():
            @enoki.on_state
            @adapter.log_state(event_type="state_execute")
            async def on_state(ctx: SharedContext):
                # Event automatically logged with:
                # - state_name attribute
                # - attributes from ctx.common (with EventAttr annotations)
                # - relationships to objects (with ObjectRef annotations)
                return Events.DONE

            @enoki.on_enter
            @adapter.log_state_lifecycle(event_type="state_enter")
            async def on_enter(ctx: SharedContext):
                pass
        ```
    """

    def __init__(self):
        """Initialize the Enoki adapter."""
        self._enabled = True

    def enable(self):
        """Enable logging for this adapter."""
        self._enabled = True

    def disable(self):
        """Disable logging for this adapter."""
        self._enabled = False

    def log_state(
        self,
        event_type: str,
        attributes: Optional[Dict[str, Any]] = None,
        log_state_name: bool = True,
        log_transition: bool = True,
    ) -> Callable:
        """
        Decorator to log Enoki state execution.

        Automatically captures:
        - State name
        - Transition result (if log_transition=True)
        - Attributes from ctx.common (with EventAttr annotations)
        - Objects from ctx.common (with ObjectRef annotations)
        - Message information (if present in ctx)

        Args:
            event_type: Type of event to log
            attributes: Static attributes to include
            log_state_name: Whether to include state name in attributes
            log_transition: Whether to include transition result

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(ctx: SharedContext):
                # Call original state handler
                result = await func(ctx)

                # Log event
                await _log_state_event(
                    func, ctx, event_type, attributes, log_state_name, log_transition, result
                )

                return result

            @functools.wraps(func)
            def sync_wrapper(ctx: SharedContext):
                # Call original state handler
                result = func(ctx)

                # Schedule logging
                asyncio.create_task(_log_state_event(
                    func, ctx, event_type, attributes, log_state_name, log_transition, result
                ))

                return result

            if asyncio.iscoroutinefunction(func):
                return async_wrapper  # type: ignore
            else:
                return sync_wrapper  # type: ignore

        return decorator

    def log_state_lifecycle(
        self,
        event_type: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """
        Decorator to log state lifecycle events (on_enter, on_leave, on_timeout).

        Use this for logging lifecycle hooks rather than main state execution.

        Args:
            event_type: Type of event to log (e.g., "state_enter", "state_leave")
            attributes: Static attributes to include

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(ctx: SharedContext):
                # Log before execution
                config = get_config()
                if config.enabled:
                    await _log_lifecycle_event(func, ctx, event_type, attributes, "enter")

                # Call original lifecycle handler
                result = await func(ctx)

                return result

            @functools.wraps(func)
            def sync_wrapper(ctx: SharedContext):
                # Schedule logging
                config = get_config()
                if config.enabled:
                    asyncio.create_task(_log_lifecycle_event(
                        func, ctx, event_type, attributes, "enter"
                    ))

                return func(ctx)

            if asyncio.iscoroutinefunction(func):
                return async_wrapper  # type: ignore
            else:
                return sync_wrapper  # type: ignore

        return decorator


async def _log_state_event(
    func: Callable,
    ctx: SharedContext,
    event_type: str,
    attributes: Optional[Dict[str, Any]],
    log_state_name: bool,
    log_transition: bool,
    result: Any,
) -> None:
    """Log a state execution event."""
    config = get_config()
    if not config.enabled:
        return

    try:
        timestamp = datetime.now()

        # Build event attributes
        event_attrs = {}

        # Add state name if requested
        if log_state_name:
            # Try to get state name from function or context
            state_name = func.__name__
            if hasattr(ctx, 'current_state') and ctx.current_state:
                state_name = getattr(ctx.current_state, 'name', state_name)

            event_attrs["state_name"] = attribute_value_from_python(state_name)

        # Add transition result if requested
        if log_transition and isinstance(result, TransitionType):
            event_attrs["transition"] = attribute_value_from_python(result.name)

        # Extract from ctx.common (blackboard)
        if hasattr(ctx, 'common') and ctx.common:
            bb_attrs = extract_attributes_from_blackboard(ctx.common, timestamp)
            event_attrs.update(bb_attrs)

        # Add message if present
        if hasattr(ctx, 'msg') and ctx.msg is not None:
            event_attrs["message_type"] = attribute_value_from_python(type(ctx.msg).__name__)

        # Extract objects from ctx.common
        relationships = {}
        event_objects = []

        if hasattr(ctx, 'common') and ctx.common:
            bb_objects = extract_objects_from_blackboard(ctx.common)
            event_objects.extend(bb_objects)

            for obj in bb_objects:
                relationships[obj.id] = Relationship(
                    object_id=obj.id,
                    qualifier="context"
                )

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
        logger.error(f"Failed to log state event: {e}")


async def _log_lifecycle_event(
    func: Callable,
    ctx: SharedContext,
    event_type: str,
    attributes: Optional[Dict[str, Any]],
    phase: str,
) -> None:
    """Log a state lifecycle event."""
    config = get_config()
    if not config.enabled:
        return

    try:
        timestamp = datetime.now()

        # Build event attributes
        event_attrs = {
            "lifecycle_method": attribute_value_from_python(func.__name__),
            "phase": attribute_value_from_python(phase)
        }

        # Add state name
        state_name = func.__name__
        if hasattr(ctx, 'current_state') and ctx.current_state:
            state_name = getattr(ctx.current_state, 'name', state_name)

        event_attrs["state_name"] = attribute_value_from_python(state_name)

        # Add static attributes
        if attributes:
            for key, value in attributes.items():
                if callable(value):
                    if hasattr(ctx, 'common') and ctx.common:
                        result = value(ctx.common)
                        event_attrs[key] = attribute_value_from_python(str(result))
                else:
                    event_attrs[key] = attribute_value_from_python(str(value))

        # Extract from ctx.common
        if hasattr(ctx, 'common') and ctx.common:
            bb_attrs = extract_attributes_from_blackboard(ctx.common, timestamp)
            event_attrs.update(bb_attrs)

        # Extract objects from ctx.common
        relationships = {}
        event_objects = []

        if hasattr(ctx, 'common') and ctx.common:
            bb_objects = extract_objects_from_blackboard(ctx.common)
            event_objects.extend(bb_objects)

            for obj in bb_objects:
                relationships[obj.id] = Relationship(
                    object_id=obj.id,
                    qualifier="context"
                )

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
        logger.error(f"Failed to log lifecycle event: {e}")


def log_message(
    event_type: str = "message_received",
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Decorator to log message handling events.

    Use this for logging when messages are processed by states.

    Args:
        event_type: Type of event to log
        attributes: Static attributes to include

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(ctx: SharedContext):
            # Log message
            config = get_config()
            if config.enabled and hasattr(ctx, 'msg') and ctx.msg is not None:
                await _log_message_event(func, ctx, event_type, attributes)

            return await func(ctx)

        @functools.wraps(func)
        def sync_wrapper(ctx: SharedContext):
            # Schedule logging
            config = get_config()
            if config.enabled and hasattr(ctx, 'msg') and ctx.msg is not None:
                asyncio.create_task(_log_message_event(
                    func, ctx, event_type, attributes
                ))

            return func(ctx)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore

    return decorator


async def _log_message_event(
    func: Callable,
    ctx: SharedContext,
    event_type: str,
    attributes: Optional[Dict[str, Any]],
) -> None:
    """Log a message event."""
    config = get_config()
    if not config.enabled:
        return

    try:
        timestamp = datetime.now()

        # Build event attributes
        event_attrs = {
            "message_type": attribute_value_from_python(type(ctx.msg).__name__),
            "handler": attribute_value_from_python(func.__name__)
        }

        # Add state name
        if hasattr(ctx, 'current_state') and ctx.current_state:
            state_name = getattr(ctx.current_state, 'name', 'unknown')
            event_attrs["state_name"] = attribute_value_from_python(state_name)

        # Add static attributes
        if attributes:
            for key, value in attributes.items():
                if callable(value):
                    if hasattr(ctx, 'common') and ctx.common:
                        result = value(ctx.common)
                        event_attrs[key] = attribute_value_from_python(str(result))
                else:
                    event_attrs[key] = attribute_value_from_python(str(value))

        # Extract from ctx.common
        if hasattr(ctx, 'common') and ctx.common:
            bb_attrs = extract_attributes_from_blackboard(ctx.common, timestamp)
            event_attrs.update(bb_attrs)

        # Extract objects from ctx.common
        relationships = {}
        event_objects = []

        if hasattr(ctx, 'common') and ctx.common:
            bb_objects = extract_objects_from_blackboard(ctx.common)
            event_objects.extend(bb_objects)

            for obj in bb_objects:
                relationships[obj.id] = Relationship(
                    object_id=obj.id,
                    qualifier="context"
                )

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
        logger.error(f"Failed to log message event: {e}")
