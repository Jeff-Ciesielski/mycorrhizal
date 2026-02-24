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
import threading
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


class _RunRegistry:
    """Open node runs, so a node is logged once per RUN rather than once per tick.

    A behavior tree calls its nodes at the tick rate, not at the rate the process
    changes. A node that reports RUNNING for forty ticks is one activity that
    lasted forty ticks; logging each tick writes forty events with the same name,
    which a process miner reads as a forty-iteration self loop and which inflates
    every count and duration derived from it.

    A run opens on the first tick that finds the node inactive and closes when the
    node returns a terminal status (anything other than RUNNING, including a raise
    and including a condition returning a bool). Ticks in between are counted and
    not logged.

    State is keyed by (node, case identity), never by node alone. One node function
    is ticked against many blackboards when a tree serves several entities
    concurrently, and a key of node alone would let one entity's run suppress
    another entity's start.
    """

    def __init__(self):
        self._open: Dict[Any, int] = {}
        self._lock = threading.Lock()

    @staticmethod
    def key(func: Callable, bb: Any) -> Any:
        try:
            objects = extract_objects_from_blackboard(bb)
            identity = tuple(sorted(obj.id for obj in objects))
        except Exception:
            identity = ()
        if not identity:
            # No global-scope object on the blackboard. Fall back to the
            # blackboard's own identity so concurrent blackboards still do not
            # share run state.
            identity = (id(bb),)
        return (getattr(func, "__qualname__", func.__name__), identity)

    def tick(self, key: Any) -> int:
        """Count this tick against the run. A return of 1 opened the run."""
        with self._lock:
            count = self._open.get(key, 0) + 1
            self._open[key] = count
            return count

    def close(self, key: Any) -> int:
        """End the run and return how many ticks it lasted."""
        with self._lock:
            return self._open.pop(key, 0)


def _is_running(result: Any) -> bool:
    """Is the node still running, or has this tick finished its run?

    Only an explicit RUNNING status continues a run. A condition returning a
    bool has always finished, and so has a node that raised.
    """
    return isinstance(result, Status) and result is Status.RUNNING


_RUNS = _RunRegistry()


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

    def __init__(self, enter_exit: bool = False):
        """
        Initialize the Rhizomorph adapter.

        Args:
            enter_exit: Adapter-wide default for enter/exit event pairs
                (see log_node). Defaults to off, matching today's single
                post-hoc event. Individual log_node() calls can override
                this default.
        """
        self._enabled = True
        self._enter_exit = enter_exit

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
        enter_exit: Optional[bool] = None,
    ) -> Callable:
        """
        Decorator to log Rhizomorph node execution.

        Automatically captures:
        - Node execution status (SUCCESS, FAILURE, RUNNING, etc.)
        - Attributes from blackboard (with EventAttr annotations)
        - Objects from blackboard (with ObjectRef annotations)
        - Node name

        By default, a single event is logged after the node returns
        (today's behavior, unchanged). Set enter_exit=True (or construct
        the adapter with enter_exit=True) to instead log a pair of events:
        one at node entry (type f"{event_type}_start") and one at node
        exit (type f"{event_type}_stop"), whether the node returned
        normally or raised. The node_name attribute is identical on both
        events. Blackboard-derived attributes and object relationships
        are recomputed independently at each phase - entry reflects
        pre-execution state, exit reflects post-execution state - so a
        blackboard mutation made during the tick is visible as a
        before/after difference between the two events; this is the
        intended enter/exit semantics, not an inconsistency. The
        f"{event_type}_start"/f"{event_type}_stop" naming is the
        convention this enter/exit extension establishes (applied
        identically by both RhizomorphAdapter and HyphaAdapter), not a
        pre-existing convention. The exit event additionally carries the
        outcome status. Exactly one exit event fires per node tick, with
        an explicit status, for every outcome.

        Args:
            event_type: Type of event to log
            attributes: Static attributes or param names to extract
            log_status: Whether to include status in event attributes
            enter_exit: Override the adapter's enter_exit default for this
                node. None inherits the adapter's constructor setting.

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(bb: Any, tb: Any = None):
                resolved_enter_exit = (
                    self._enter_exit if enter_exit is None else enter_exit
                )

                # One run may span many ticks. `opening` is true only on the
                # tick that started it, and the run closes on the first tick
                # that does not return RUNNING.
                run_key = _RunRegistry.key(func, bb)
                opening = _RUNS.tick(run_key) == 1

                if not resolved_enter_exit:
                    # Call original node function
                    if _supports_timebase(func):
                        result = await func(bb=bb, tb=tb)
                    else:
                        result = await func(bb=bb)

                    if _is_running(result):
                        # The run continues. One activity, so no second event.
                        return result

                    # Log event
                    await _log_node_event(
                        func, bb, tb, event_type, attributes, log_status, result,
                        _RUNS.close(run_key)
                    )

                    return result

                # Enter/exit mode: entry is emitted on the tick that opens the
                # run, before the node's own first await; exit is emitted from
                # a finally on the tick that closes it, so a raised exception
                # still produces the exit event before it propagates.
                # BaseException is caught (not just Exception) so
                # cancellation-style outcomes are still classified and logged
                # rather than silently skipping the status attribute. A tick
                # that leaves the node RUNNING emits neither.
                if opening:
                    await _log_node_lifecycle_event(
                        func, bb, tb, event_type, log_status, "start", None
                    )
                status_name = None
                still_running = False
                try:
                    if _supports_timebase(func):
                        result = await func(bb=bb, tb=tb)
                    else:
                        result = await func(bb=bb)
                except BaseException:
                    status_name = "EXCEPTION"
                    raise
                else:
                    if isinstance(result, Status):
                        status_name = result.name
                    still_running = _is_running(result)
                    return result
                finally:
                    if not still_running:
                        await _log_node_lifecycle_event(
                            func, bb, tb, event_type, log_status, "stop",
                            status_name, _RUNS.close(run_key)
                        )

            @functools.wraps(func)
            def sync_wrapper(bb: Any, tb: Any = None):
                resolved_enter_exit = (
                    self._enter_exit if enter_exit is None else enter_exit
                )

                run_key = _RunRegistry.key(func, bb)
                opening = _RUNS.tick(run_key) == 1

                if not resolved_enter_exit:
                    # Call original node function
                    if _supports_timebase(func):
                        result = func(bb=bb, tb=tb)
                    else:
                        result = func(bb=bb)

                    if _is_running(result):
                        # The run continues. One activity, so no second event.
                        return result

                    # Schedule logging
                    asyncio.create_task(_log_node_event(
                        func, bb, tb, event_type, attributes, log_status, result,
                        _RUNS.close(run_key)
                    ))

                    return result

                # Enter/exit mode (fire-and-forget logging, same scheduling
                # pattern as the default sync path above). A tick that leaves
                # the node RUNNING emits neither event.
                if opening:
                    asyncio.create_task(_log_node_lifecycle_event(
                        func, bb, tb, event_type, log_status, "start", None
                    ))
                status_name = None
                still_running = False
                try:
                    if _supports_timebase(func):
                        result = func(bb=bb, tb=tb)
                    else:
                        result = func(bb=bb)
                except BaseException:
                    status_name = "EXCEPTION"
                    raise
                else:
                    if isinstance(result, Status):
                        status_name = result.name
                    still_running = _is_running(result)
                    return result
                finally:
                    if not still_running:
                        asyncio.create_task(_log_node_lifecycle_event(
                            func, bb, tb, event_type, log_status, "stop",
                            status_name, _RUNS.close(run_key)
                        ))

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
    ticks: int = 1,
) -> None:
    """Log one completed node RUN.

    `ticks` is how many ticks the run lasted. It is accepted and not emitted:
    the payload of the default path is unchanged, and duration is recoverable
    from the entry and exit timestamps in enter_exit mode.
    """
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


async def _log_node_lifecycle_event(
    func: Callable,
    bb: Any,
    tb: Any,
    event_type: str,
    log_status: bool,
    phase: str,
    status_name: Optional[str],
    ticks: int = 1,
) -> None:
    """
    Log an entry or exit event for a Rhizomorph node tick.

    The node_name attribute is identical between the entry and exit
    event for a given tick. Blackboard attributes and object
    relationships are recomputed independently at each call to this
    function, from whatever state bb holds at that phase - entry
    reflects pre-execution state, exit reflects post-execution state -
    so they are not guaranteed to match between the two events; that
    divergence is the intended enter/exit semantics, since it is what
    lets a downstream consumer see what the tick changed. The emitted
    event type is tagged with a start/stop suffix
    (f"{event_type}_{phase}") so the wrapped duration is recoverable
    downstream. Called once at node entry (phase="start",
    status_name=None) and once at node exit (phase="stop"), whether the
    node returned normally or raised.

    Args:
        func: The decorated node function
        bb: Blackboard passed to the node
        tb: Timebase passed to the node (if any)
        event_type: Base event type; the emitted type is
            f"{event_type}_{phase}"
        log_status: Whether to include the status attribute for a
            normal (non-exception) outcome. An "EXCEPTION" outcome is
            always included regardless of this flag.
        phase: "start" or "stop"
        status_name: Outcome name to attach on exit ("SUCCESS",
            "FAILURE", "RUNNING", or "EXCEPTION"), or None at entry
    """
    config = get_config()
    if not config.enabled:
        return

    try:
        timestamp = datetime.now()

        # Build event attributes
        event_attrs = {}

        # Add node name
        event_attrs["node_name"] = attribute_value_from_python(func.__name__)

        # Add status: normal outcomes respect log_status, exceptions are
        # always surfaced since suppressing them would defeat the point
        # of an exit event
        if status_name is not None and (log_status or status_name == "EXCEPTION"):
            event_attrs["status"] = attribute_value_from_python(status_name)

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
            type=f"{event_type}_{phase}",
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
        logger.error(f"Failed to log node {phase} event: {e}")


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
            "tree_name": attribute_value_from_python(tree_name)
        }

        # Add static attributes
        if attributes:
            for key, value in attributes.items():
                if callable(value):
                    # Extract bb from kwargs or args
                    bb = kwargs.get('bb') or (args[0] if args else None)
                    if bb:
                        result = value(bb)
                        event_attrs[key] = attribute_value_from_python(str(result))
                else:
                    event_attrs[key] = attribute_value_from_python(str(value))

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
