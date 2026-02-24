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
    get_config,
    Event,
    LogRecord,
    Relationship,
    generate_event_id,
    attribute_value_from_python,
)
from ...spores.extraction import (
    extract_attributes_from_blackboard,
    extract_objects_from_blackboard,
    convert_to_ocel_object,
)
from ...spores.core import _send_log_record, get_object_cache


logger = logging.getLogger(__name__)


def _is_async(func: Callable) -> bool:
    """
    Check whether a transition function needs the async wrapper path.

    A transition can be a plain coroutine function (returns a single
    dict/None) or an async generator function (yields multiple outputs,
    the shape used throughout this codebase and its examples).
    asyncio.iscoroutinefunction() alone returns False for async
    generator functions, so checking only that would route every
    generator-shaped transition to the sync wrapper - which never
    inspects or awaits the transition's own execution, just fires a
    log immediately and hands back the transition's raw, unwrapped
    generator. Both async shapes need the async wrapper.

    Behavior-changing bug fix: async-generator transitions previously
    bypassed the wrapper - logging was fire-and-forget and log_outputs
    was inert. They now log like every other shape, awaited before
    their first yield, and log_outputs works.
    """
    return asyncio.iscoroutinefunction(func) or inspect.isasyncgenfunction(func)


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

            @builder.place()
            @adapter.log_place(event_type="item_arrived")
            def input_place(bb):
                return None
        ```
    """

    def __init__(self, enter_exit: bool = False):
        """
        Initialize the Hypha adapter.

        Args:
            enter_exit: Adapter-wide default for enter/exit event pairs
                (see log_transition). Defaults to off, matching today's
                single post-hoc event. Individual log_transition() calls
                can override this default.
        """
        self._enabled = True
        self._enter_exit = enter_exit

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
        enter_exit: Optional[bool] = None,
    ) -> Callable:
        """
        Decorator to log Hypha transition execution.

        Automatically captures:
        - Token count from consumed tokens
        - Relationships to consumed/produced tokens
        - Attributes from blackboard (if specified)
        - Objects from blackboard with ObjectRef metadata

        By default, a single event is logged per transition call (today's
        behavior, unchanged: before token processing for async-generator
        transitions, after completion for coroutine transitions). Set
        enter_exit=True (or construct the adapter with enter_exit=True)
        to instead log a pair of events for every transition call: one at
        entry (type f"{event_type}_start") and one at exit (type
        f"{event_type}_stop"), whether the transition completed
        normally, raised, or was abandoned before completion. The
        transition_name and token_count attributes are identical on both
        events. Blackboard-derived attributes and object/token
        relationships are recomputed independently at each phase - entry
        reflects pre-execution state, exit reflects post-execution state
        - so a blackboard mutation made during the call is visible as a
        before/after difference between the two events; this is the
        intended enter/exit semantics, not an inconsistency. The
        f"{event_type}_start"/f"{event_type}_stop" naming is the
        convention this enter/exit extension establishes (applied
        identically by both HyphaAdapter and RhizomorphAdapter), not a
        pre-existing convention. The exit event additionally carries the
        outcome status: "SUCCESS", "EXCEPTION" (the transition raised),
        or "ABANDONED" (an async-generator transition was closed - via
        `aclose()` or the consuming `async for` breaking out - before it
        finished producing output). Exactly one exit event fires per
        transition call, with an explicit status, for every outcome.

        Bug fix note: async-generator transitions previously bypassed
        this wrapper entirely (see _is_async docstring); as a result,
        log_outputs and enter_exit logging were both inert for the
        dominant Hypha transition shape. Both now behave like every
        other transition shape.

        Args:
            event_type: Type of event to log
            attributes: Static attributes or param names to extract
            log_inputs: Whether to log input tokens as related objects
            log_outputs: Whether to log output tokens as related objects
            enter_exit: Override the adapter's enter_exit default for
                this transition. None inherits the adapter's constructor
                setting.

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(consumed: List[Any], bb: Any, timebase: Any, state: Any = None):
                resolved_enter_exit = (
                    self._enter_exit if enter_exit is None else enter_exit
                )

                if not resolved_enter_exit:
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
                    return

                # Enter/exit mode: entry is emitted before the
                # transition's own first await; exit is emitted from a
                # finally so every outcome - success, a raised
                # exception, or abandonment - still produces the exit
                # event before it propagates. This holds for both the
                # async-generator and coroutine transition shapes.
                await _log_transition_lifecycle_event(
                    func, consumed, bb, timebase, event_type, log_inputs, "start", None
                )
                status_name = None
                try:
                    if state is not None:
                        result = func(consumed, bb, timebase, state)
                    else:
                        result = func(consumed, bb, timebase)

                    if inspect.isasyncgen(result):
                        outputs = []
                        async for yielded in result:
                            outputs.append(yielded)
                            yield yielded

                        if log_outputs:
                            await _log_outputs(func, outputs, bb, timebase, event_type)
                    else:
                        result = await result
                        if result is not None:
                            yield result
                except GeneratorExit:
                    # This wrapper is itself an async generator (it
                    # yields transition outputs); closing it early -
                    # `aclose()`, or the consumer's `async for` breaking
                    # out - throws GeneratorExit in here rather than a
                    # normal Exception. Classify it distinctly from a
                    # transition-raised error and always re-raise
                    # unchanged so the close actually completes.
                    status_name = "ABANDONED"
                    raise
                except BaseException:
                    status_name = "EXCEPTION"
                    raise
                else:
                    status_name = "SUCCESS"
                finally:
                    await _log_transition_lifecycle_event(
                        func, consumed, bb, timebase, event_type, log_inputs, "stop", status_name
                    )

            @functools.wraps(func)
            def sync_wrapper(consumed: List[Any], bb: Any, timebase: Any, state: Any = None):
                resolved_enter_exit = (
                    self._enter_exit if enter_exit is None else enter_exit
                )

                if not resolved_enter_exit:
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

                # Enter/exit mode (fire-and-forget logging, same
                # scheduling pattern as the default sync path above).
                asyncio.create_task(_log_transition_lifecycle_event(
                    func, consumed, bb, timebase, event_type, log_inputs, "start", None
                ))
                status_name = None
                try:
                    if state is not None:
                        result = func(consumed, bb, timebase, state)
                    else:
                        result = func(consumed, bb, timebase)
                except BaseException:
                    status_name = "EXCEPTION"
                    raise
                else:
                    status_name = "SUCCESS"
                    return result
                finally:
                    asyncio.create_task(_log_transition_lifecycle_event(
                        func, consumed, bb, timebase, event_type, log_inputs, "stop", status_name
                    ))

            # Return appropriate wrapper. A transition can be a plain
            # coroutine function or an async generator function (the
            # multi-output shape used throughout this codebase); both
            # need async_wrapper; only a genuinely synchronous function
            # (no async keyword at all) goes through sync_wrapper.
            if _is_async(func):
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
        event_attrs["token_count"] = attribute_value_from_python(len(consumed))

        # Add transition name
        event_attrs["transition_name"] = attribute_value_from_python(func.__name__)

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


async def _log_transition_lifecycle_event(
    func: Callable,
    consumed: List[Any],
    bb: Any,
    timebase: Any,
    event_type: str,
    log_inputs: bool,
    phase: str,
    status_name: Optional[str],
) -> None:
    """
    Log an entry or exit event for a Hypha transition call.

    The transition_name and token_count attributes are identical
    between the entry and exit event for a given call. Blackboard
    attributes and object/token relationships are recomputed
    independently at each call to this function, from whatever state
    consumed/bb hold at that phase - entry reflects pre-execution
    state, exit reflects post-execution state - so they are not
    guaranteed to match between the two events; that divergence is the
    intended enter/exit semantics, since it is what lets a downstream
    consumer see what the transition changed. The emitted event type is
    tagged with a start/stop suffix (f"{event_type}_{phase}") so the
    wrapped duration is recoverable downstream. Called once at
    transition entry (phase="start", status_name=None) and once at
    transition exit (phase="stop") for every outcome - success,
    exception, or abandonment - never more than once per phase per
    call.

    Args:
        func: The decorated transition function
        consumed: Tokens consumed by the transition
        bb: Blackboard passed to the transition
        timebase: Timebase passed to the transition
        event_type: Base event type; the emitted type is
            f"{event_type}_{phase}"
        log_inputs: Whether to log input tokens as related objects
        phase: "start" or "stop"
        status_name: Outcome name to attach on exit ("SUCCESS",
            "EXCEPTION", or "ABANDONED" - an async-generator transition
            closed before finishing), or None at entry
    """
    config = get_config()
    if not config.enabled:
        return

    try:
        timestamp = datetime.now()

        # Build event attributes
        event_attrs = {}

        # Add token count
        event_attrs["token_count"] = attribute_value_from_python(len(consumed))

        # Add transition name
        event_attrs["transition_name"] = attribute_value_from_python(func.__name__)

        # Add outcome status (exit events only; entry passes None)
        if status_name is not None:
            event_attrs["status"] = attribute_value_from_python(status_name)

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
            type=f"{event_type}_{phase}",
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
        logger.error(f"Failed to log transition {phase} event: {e}")


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
            "place_name": attribute_value_from_python(func.__name__)
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
