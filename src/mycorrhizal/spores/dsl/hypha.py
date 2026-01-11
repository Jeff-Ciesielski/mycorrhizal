#!/usr/bin/env python3
"""
Spores Adapter for Hypha (Petri Nets)

Provides logging integration for Hypha transitions and places.
Extracts token information and automatically creates event/object logs.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from ...spores import (
    spore,
    get_config,
    Event,
    Object,
    LogRecord,
    Relationship,
    EventAttributeValue,
    ObjectAttributeValue,
    generate_event_id,
    generate_object_id,
    attribute_value_from_python,
    object_attribute_from_python,
)
from ...spores.extraction import (
    extract_attributes_from_blackboard,
    extract_objects_from_blackboard,
    convert_to_ocel_object,
)
from ...spores.core import _send_log_record, get_object_cache


logger = logging.getLogger(__name__)


class HyphaAdapter:
    """
    Adapter for Hypha Petri net logging.

    Provides decorators and helpers for logging:
    - Transition execution with consumed/produced tokens
    - Place token arrivals/departures
    - Token object lifecycle tracking

    Usage:
        ```python
        from mycorrhizal.spores.dsl import HyphaAdapter

        adapter = HyphaAdapter()

        @pn.net
        def MyNet(builder):
            @builder.transition()
            @adapter.log_transition(event_type="process_item")
            async def process(consumed, bb, timebase):
                # Event automatically logged with:
                # - token_count attribute
                # - relationships to consumed tokens
                yield {output: consumed[0]}

            @builder.place(type=PlaceType.QUEUE)
            @adapter.log_place(event_type="item_arrived")
            def input_place(bb):
                return None
        ```
    """

    def __init__(self):
        """Initialize the Hypha adapter."""
        self._enabled = True

    def enable(self):
        """Enable logging for this adapter."""
        self._enabled = True

    def disable(self):
        """Disable logging for this adapter."""
        self._enabled = False

    def log_transition(
        self,
        event_type: str,
        attributes: Optional[Union[Dict[str, Any], List[str]]] = None,
        log_inputs: bool = True,
        log_outputs: bool = False,
    ) -> Callable:
        """
        Decorator to log Hypha transition execution.

        Automatically captures:
        - Token count from consumed tokens
        - Relationships to consumed/produced tokens
        - Attributes from blackboard (if specified)
        - Objects from blackboard with ObjectRef metadata

        Args:
            event_type: Type of event to log
            attributes: Static attributes or param names to extract
            log_inputs: Whether to log input tokens as related objects
            log_outputs: Whether to log output tokens as related objects

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(consumed: List[Any], bb: Any, timebase: Any, state: Any = None):
                # Call original transition
                if state is not None:
                    result = func(consumed, bb, timebase, state)
                else:
                    result = func(consumed, bb, timebase)

                # Handle async generator
                if inspect.isasyncgen(result):
                    # Log before processing
                    await _log_transition_event(
                        func, consumed, bb, timebase, event_type, attributes, log_inputs, None
                    )

                    # Collect outputs
                    outputs = []
                    async for yielded in result:
                        outputs.append(yielded)
                        yield yielded

                    # Log outputs if requested
                    if log_outputs:
                        await _log_outputs(func, outputs, bb, timebase, event_type)
                else:
                    # Handle coroutine
                    result = await result

                    # Log event
                    await _log_transition_event(
                        func, consumed, bb, timebase, event_type, attributes, log_inputs, result
                    )

                    # Yield the result if it's not None
                    if result is not None:
                        yield result

            @functools.wraps(func)
            def sync_wrapper(consumed: List[Any], bb: Any, timebase: Any, state: Any = None):
                # For sync transitions, log asynchronously
                if state is not None:
                    result = func(consumed, bb, timebase, state)
                else:
                    result = func(consumed, bb, timebase)

                # Schedule logging
                asyncio.create_task(_log_transition_event(
                    func, consumed, bb, timebase, event_type, attributes, log_inputs, result
                ))

                return result

            # Return appropriate wrapper
            if asyncio.iscoroutinefunction(func):
                return async_wrapper  # type: ignore
            else:
                return sync_wrapper  # type: ignore

        return decorator

    def log_place(self, event_type: str = "token_arrived") -> Callable:
        """
        Decorator to log place token arrivals.

        Note: This requires instrumenting the place handler or using
        a custom place runtime wrapper. For most cases, transition
        logging provides sufficient coverage.

        Args:
            event_type: Type of event to log

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(bb: Any, timebase: Any):
                result = await func(bb, timebase)

                # Log token arrival
                config = get_config()
                if config.enabled:
                    await _log_place_event(func, bb, timebase, event_type)

                return result

            @functools.wraps(func)
            def sync_wrapper(bb: Any, timebase: Any):
                result = func(bb, timebase)

                # Schedule logging
                config = get_config()
                if config.enabled:
                    asyncio.create_task(_log_place_event(func, bb, timebase, event_type))

                return result

            if asyncio.iscoroutinefunction(func):
                return async_wrapper  # type: ignore
            else:
                return sync_wrapper  # type: ignore

        return decorator


async def _log_transition_event(
    func: Callable,
    consumed: List[Any],
    bb: Any,
    timebase: Any,
    event_type: str,
    attributes: Optional[Union[Dict[str, Any], List[str]]],
    log_inputs: bool,
    result: Any,
) -> None:
    """Log a transition execution event."""
    config = get_config()
    if not config.enabled:
        return

    try:
        timestamp = datetime.now()

        # Build event attributes
        event_attrs = {}

        # Add token count
        event_attrs["token_count"] = EventAttributeValue(
            name="token_count",
            value=str(len(consumed)),
            time=timestamp
        )

        # Add transition name
        event_attrs["transition_name"] = EventAttributeValue(
            name="transition_name",
            value=func.__name__,
            time=timestamp
        )

        # Extract from blackboard
        bb_attrs = extract_attributes_from_blackboard(bb, timestamp)
        event_attrs.update(bb_attrs)

        # Build relationships from tokens
        relationships = {}
        event_objects = []

        if log_inputs:
            for token in consumed:
                if token is not None:
                    obj = convert_to_ocel_object(token, qualifier="input")
                    if obj:
                        event_objects.append(obj)
                        relationships[obj.id] = Relationship(
                            object_id=obj.id,
                            qualifier="input"
                        )

        # Extract objects from blackboard
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
        logger.error(f"Failed to log transition event: {e}")


async def _log_outputs(
    func: Callable,
    outputs: List[Any],
    bb: Any,
    timebase: Any,
    event_type: str,
) -> None:
    """Log output tokens as related objects."""
    config = get_config()
    if not config.enabled:
        return

    try:
        timestamp = datetime.now()
        event_objects = []

        # Process outputs
        for output in outputs:
            if isinstance(output, dict):
                # Dict of place_ref -> token
                for place_ref, token in output.items():
                    if place_ref != '*' and token is not None:
                        obj = convert_to_ocel_object(token, qualifier="output")
                        if obj:
                            event_objects.append(obj)
            elif isinstance(output, tuple) and len(output) == 2:
                # (place_ref, token) tuple
                place_ref, token = output
                if token is not None:
                    obj = convert_to_ocel_object(token, qualifier="output")
                    if obj:
                        event_objects.append(obj)

        # Send objects to cache
        cache = get_object_cache()
        for obj in event_objects:
            cache.contains_or_add(obj.id, obj)

    except Exception as e:
        logger.error(f"Failed to log outputs: {e}")


async def _log_place_event(
    func: Callable,
    bb: Any,
    timebase: Any,
    event_type: str,
) -> None:
    """Log a place token event."""
    config = get_config()
    if not config.enabled:
        return

    try:
        timestamp = datetime.now()

        # Build event attributes
        event_attrs = {
            "place_name": EventAttributeValue(
                name="place_name",
                value=func.__name__,
                time=timestamp
            )
        }

        # Extract from blackboard
        bb_attrs = extract_attributes_from_blackboard(bb, timestamp)
        event_attrs.update(bb_attrs)

        # Extract objects from blackboard
        bb_objects = extract_objects_from_blackboard(bb)
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
        logger.error(f"Failed to log place event: {e}")
