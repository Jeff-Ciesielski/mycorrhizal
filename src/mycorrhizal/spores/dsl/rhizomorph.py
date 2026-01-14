#!/usr/bin/env python3
"""
Spores Adapter for Rhizomorph (Behavior Trees)

Provides logging integration for Rhizomorph nodes and trees.
Extracts blackboard information and automatically creates event/object logs.
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
    attribute_value_from_python,
    generate_event_id,
)
from ...spores.extraction import (
    extract_attributes_from_blackboard,
    extract_objects_from_blackboard,
)
from ...spores.core import _send_log_record, get_object_cache
from ...rhizomorph.core import Status


logger = logging.getLogger(__name__)


def _supports_timebase(func: Callable) -> bool:
    """Check if function accepts a timebase parameter."""
    try:
        sig = inspect.signature(func)
        return 'tb' in sig.parameters
    except (ValueError, TypeError):
        return False


class RhizomorphAdapter:
    """
    Adapter for Rhizomorph behavior tree logging.

    Provides decorators and helpers for logging:
    - Node tick execution with status results
    - Tree-level events
    - Blackboard object lifecycle tracking

    Usage:
        ```python
        from mycorrhizal.spores.dsl import RhizomorphAdapter

        adapter = RhizomorphAdapter()

        @bt.tree
        def MyTree():
            @bt.action
            @adapter.log_node(event_type="check_threat")
            async def check_threat(bb: MissionContext) -> Status:
                # Event automatically logged with:
                # - status result attribute
                # - attributes from bb (with EventAttr annotations)
                # - relationships to objects (with ObjectRef annotations)
                return Status.SUCCESS
        ```
    """

    def __init__(self):
        """Initialize the Rhizomorph adapter."""
        self._enabled = True

    def enable(self):
        """Enable logging for this adapter."""
        self._enabled = True

    def disable(self):
        """Disable logging for this adapter."""
        self._enabled = False

    def log_node(
        self,
        event_type: str,
        attributes: Optional[Union[Dict[str, Any], List[str]]] = None,
        log_status: bool = True,
    ) -> Callable:
        """
        Decorator to log Rhizomorph node execution.

        Automatically captures:
        - Node execution status (SUCCESS, FAILURE, RUNNING, etc.)
        - Attributes from blackboard (with EventAttr annotations)
        - Objects from blackboard (with ObjectRef annotations)
        - Node name

        Args:
            event_type: Type of event to log
            attributes: Static attributes or param names to extract
            log_status: Whether to include status in event attributes

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(bb: Any, tb: Any = None):
                # Call original node function
                if _supports_timebase(func):
                    result = await func(bb=bb, tb=tb)
                else:
                    result = await func(bb=bb)

                # Log event
                await _log_node_event(
                    func, bb, tb, event_type, attributes, log_status, result
                )

                return result

            @functools.wraps(func)
            def sync_wrapper(bb: Any, tb: Any = None):
                # Call original node function
                if _supports_timebase(func):
                    result = func(bb=bb, tb=tb)
                else:
                    result = func(bb=bb)

                # Schedule logging
                asyncio.create_task(_log_node_event(
                    func, bb, tb, event_type, attributes, log_status, result
                ))

                return result

            # Return appropriate wrapper
            if asyncio.iscoroutinefunction(func):
                return async_wrapper  # type: ignore
            else:
                return sync_wrapper  # type: ignore

        return decorator


async def _log_node_event(
    func: Callable,
    bb: Any,
    tb: Any,
    event_type: str,
    attributes: Optional[Union[Dict[str, Any], List[str]]],
    log_status: bool,
    result: Any,
) -> None:
    """Log a node tick event."""
    config = get_config()
    if not config.enabled:
        return

    try:
        timestamp = datetime.now()

        # Build event attributes
        event_attrs = {}

        # Add node name
        event_attrs["node_name"] = attribute_value_from_python(func.__name__)

        # Add status if requested and result is a Status
        if log_status and isinstance(result, Status):
            event_attrs["status"] = attribute_value_from_python(result.name)

        # Extract from blackboard
        bb_attrs = extract_attributes_from_blackboard(bb, timestamp)
        event_attrs.update(bb_attrs)

        # Extract objects from blackboard
        bb_objects = extract_objects_from_blackboard(bb)

        # Build relationships
        relationships = {}
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
        for obj in bb_objects:
            cache.contains_or_add(obj.id, obj)

    except Exception as e:
        logger.error(f"Failed to log node event: {e}")


def log_tree_event(
    event_type: str,
    tree_name: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Decorator to log tree-level events.

    Use this for logging events at the tree level rather than individual nodes.

    Args:
        event_type: Type of event to log
        tree_name: Name of the tree
        attributes: Static attributes to include

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)

            # Log event
            config = get_config()
            if config.enabled:
                await _log_tree_event(func, tree_name, event_type, attributes, args, kwargs)

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            # Schedule logging
            config = get_config()
            if config.enabled:
                asyncio.create_task(_log_tree_event(
                    func, tree_name, event_type, attributes, args, kwargs
                ))

            return result

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore

    return decorator


async def _log_tree_event(
    func: Callable,
    tree_name: str,
    event_type: str,
    attributes: Optional[Dict[str, Any]],
    args: tuple,
    kwargs: dict,
) -> None:
    """Log a tree-level event."""
    config = get_config()
    if not config.enabled:
        return

    try:
        timestamp = datetime.now()

        # Build event attributes
        event_attrs = {
            "tree_name": EventAttributeValue(
                name="tree_name",
                value=tree_name,
                time=timestamp
            )
        }

        # Add static attributes
        if attributes:
            for key, value in attributes.items():
                if callable(value):
                    # Extract bb from kwargs or args
                    bb = kwargs.get('bb') or (args[0] if args else None)
                    if bb:
                        result = value(bb)
                        event_attrs[key] = EventAttributeValue(
                            name=key,
                            value=str(result),
                            time=timestamp
                        )
                else:
                    event_attrs[key] = EventAttributeValue(
                        name=key,
                        value=str(value),
                        time=timestamp
                    )

        # Extract bb for object logging
        bb = kwargs.get('bb') or (args[0] if args else None)

        # Extract objects from blackboard
        relationships = {}
        event_objects = []

        if bb:
            bb_objects = extract_objects_from_blackboard(bb)
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
        logger.error(f"Failed to log tree event: {e}")
