#!/usr/bin/env python3
"""
Spores Integration for Mycelium

This module provides spores logging integration for Mycelium trees.
It wraps the existing spores DSL adapters (HyphaAdapter, RhizomorphAdapter,
SeptumAdapter) to provide logging for Mycelium-specific patterns.

Key principle: Mycelium wraps base modules. Base modules do NOT know about Mycelium.

This integration layer:
- Uses existing spores adapters for Hypha, Rhizomorph, and Septum
- Provides Mycelium-specific logging via wrapper classes
- Does NOT modify the spores module
"""

from __future__ import annotations

import asyncio
import functools
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from ..spores import (
    get_config,
    Event,
    LogRecord,
    Relationship,
    EventAttributeValue,
    ObjectAttributeValue,
    generate_event_id,
)
from ..spores.extraction import (
    extract_attributes_from_blackboard,
    extract_objects_from_blackboard,
)
from ..spores.core import _send_log_record, get_object_cache
from ..spores.dsl import RhizomorphAdapter
from ..rhizomorph.core import Status


logger = logging.getLogger(__name__)


# ======================================================================================
# Mycelium-Specific Spores Integration
# ======================================================================================


class TreeSporesAdapter:
    """
    Adapter for logging Mycelium tree execution with spores.

    This wraps the existing spores adapters and provides Mycelium-specific
    logging functionality without modifying the spores module.

    Usage:
        ```python
        from mycorrhizal.mycelium.spores_integration import TreeSporesAdapter

        adapter = TreeSporesAdapter(tree_name="MyTree")

        @tree
        def MyTree():
            @Action(fsm=MyFSM)
            @adapter.log_action(event_type="control_robot")
            async def control(bb, tb, fsm_runner):
                return Status.SUCCESS
        ```
    """

    def __init__(self, tree_name: Optional[str] = None):
        """
        Initialize the tree spores adapter.

        Args:
            tree_name: Optional tree name for logging
        """
        self._enabled = True
        self._tree_name = tree_name

        # Create base adapters for reuse
        self._bt_adapter = RhizomorphAdapter()

    def enable(self):
        """Enable logging for this adapter."""
        self._enabled = True
        self._bt_adapter.enable()

    def disable(self):
        """Disable logging for this adapter."""
        self._enabled = False
        self._bt_adapter.disable()

    def log_action(
        self,
        event_type: str,
        attributes: Optional[Union[Dict[str, Any], List[str]]] = None,
        log_fsm_state: bool = True,
        log_status: bool = True,
        log_blackboard: bool = True,
    ) -> Callable:
        """
        Decorator to log Mycelium action execution.

        Automatically captures:
        - FSM state (if action has FSM integration)
        - Action execution status
        - Attributes from blackboard (with EventAttr annotations)
        - Objects from blackboard (with ObjectRef annotations)
        - Action name

        Args:
            event_type: Type of event to log
            attributes: Static attributes or param names to extract
            log_fsm_state: Whether to include FSM state in event attributes
            log_status: Whether to include status in event attributes
            log_blackboard: Whether to extract attributes from blackboard

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(bb: Any, tb: Any, fsm_runner: Any = None):
                # Call original action function
                result = await func(bb, tb, fsm_runner)

                # Log event with full context
                await _log_action_event(
                    func, bb, tb, fsm_runner, event_type,
                    attributes, log_fsm_state, log_status, log_blackboard, result,
                    self._tree_name,
                )

                return result

            return async_wrapper  # type: ignore

        return decorator

    def log_tree_tick(
        self,
        event_type: str = "tree_tick",
        log_blackboard: bool = True,
        log_fsm_states: bool = True,
    ) -> Callable:
        """
        Decorator to log tree tick events.

        Use this to wrap the TreeRunner.tick() method.

        Args:
            event_type: Type of event to log
            log_blackboard: Whether to snapshot blackboard state
            log_fsm_states: Whether to include all FSM states

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(self, *args, **kwargs):
                # Call original tick method
                result = await func(*args, **kwargs)

                # Log event
                await _log_tree_tick_event(
                    self, event_type, log_blackboard, log_fsm_states, result,
                    self._tree_name,
                )

                return result

            return async_wrapper  # type: ignore

        return decorator

    def log_state_snapshot(
        self,
        snapshot_name: str,
        include_blackboard: bool = True,
        include_fsms: bool = True,
    ) -> Callable:
        """
        Decorator to log state snapshots.

        Creates object-type log entries capturing the complete
        state of the tree at a point in time.

        Args:
            snapshot_name: Name for this snapshot type
            include_blackboard: Whether to include blackboard state
            include_fsms: Whether to include FSM states

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Execute function
                result = await func(*args, **kwargs)

                # Log snapshot - find tree_instance from args
                tree_instance = None
                for arg in args:
                    if hasattr(arg, 'spec') and hasattr(arg, 'bb'):
                        tree_instance = arg
                        break

                if tree_instance:
                    await _log_state_snapshot(
                        tree_instance, snapshot_name,
                        include_blackboard, include_fsms,
                    )

                return result

            return async_wrapper  # type: ignore

        return decorator


# ======================================================================================
# Internal Logging Functions
# ======================================================================================


async def _log_action_event(
    func: Callable,
    bb: Any,
    tb: Any,
    fsm_runner: Any,
    event_type: str,
    attributes: Optional[Union[Dict[str, Any], List[str]]],
    log_fsm_state: bool,
    log_status: bool,
    log_blackboard: bool,
    result: Any,
    tree_name: Optional[str],
) -> None:
    """Log an action execution event."""
    config = get_config()
    if not config.enabled:
        return

    try:
        timestamp = datetime.now()

        # Build event attributes
        event_attrs = {}

        # Add tree name if available
        if tree_name:
            event_attrs["tree_name"] = EventAttributeValue(
                name="tree_name",
                value=tree_name,
                type="string"
            )

        # Add action name
        event_attrs["action_name"] = EventAttributeValue(
            name="action_name",
            value=func.__name__,
            type="string"
        )

        # Add status if requested and result is a Status
        if log_status and isinstance(result, Status):
            event_attrs["status"] = EventAttributeValue(
                name="status",
                value=result.name,
                type="string"
            )

        # Add FSM state if available and requested
        if log_fsm_state and fsm_runner is not None:
            try:
                current_state = fsm_runner.current_state
                if current_state is not None:
                    state_name = current_state.name
                    event_attrs["fsm_state"] = EventAttributeValue(
                        name="fsm_state",
                        value=state_name,
                        type="string"
                    )
            except Exception as e:
                logger.debug(f"Could not extract FSM state: {e}")

        # Extract from blackboard
        if log_blackboard:
            bb_attrs = extract_attributes_from_blackboard(bb, timestamp)
            # Convert EventAttributeValue to dict format for merging
            for attr_name, attr_value in bb_attrs.items():
                event_attrs[attr_name] = attr_value

        # Extract objects from blackboard
        bb_objects = extract_objects_from_blackboard(bb)

        # Build relationships
        relationships = {}
        for obj in bb_objects:
            # Determine qualifier from ObjectRef if available
            qualifier = "context"
            if hasattr(obj, '__dict__') and hasattr(type(obj), '__annotations__'):
                for field_name, field_type in type(obj).__annotations__.items():
                    # Check for ObjectRef metadata
                    from ..spores.models import ObjectRef
                    if hasattr(field_type, '__metadata__'):
                        for meta in field_type.__metadata__:
                            if isinstance(meta, ObjectRef):
                                qualifier = meta.qualifier
                                break

            relationships[obj.id] = Relationship(
                object_id=obj.id,
                qualifier=qualifier
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
        logger.error(f"Failed to log action event: {e}")


async def _log_tree_tick_event(
    tree_runner: Any,
    event_type: str,
    log_blackboard: bool,
    log_fsm_states: bool,
    result: Status,
    tree_name: Optional[str],
) -> None:
    """Log a tree tick event."""
    config = get_config()
    if not config.enabled:
        return

    try:
        timestamp = datetime.now()

        # Build event attributes
        event_attrs = {}

        # Add tree name
        if tree_name:
            event_attrs["tree_name"] = EventAttributeValue(
                name="tree_name",
                value=tree_name,
                type="string"
            )
        elif hasattr(tree_runner, 'spec') and tree_runner.spec:
            event_attrs["tree_name"] = EventAttributeValue(
                name="tree_name",
                value=tree_runner.spec.name,
                type="string"
            )

        # Add tick result
        event_attrs["tick_result"] = EventAttributeValue(
            name="tick_result",
            value=result.name,
            type="string"
        )

        # Add blackboard snapshot
        if log_blackboard and hasattr(tree_runner, 'bb'):
            bb = tree_runner.bb
            bb_attrs = extract_attributes_from_blackboard(bb, timestamp)
            for attr_name, attr_value in bb_attrs.items():
                event_attrs[attr_name] = attr_value

        # Add FSM states
        if log_fsm_states and hasattr(tree_runner, 'instance'):
            try:
                fsm_states = {}
                for fsm_name, fsm_runner in tree_runner.instance._fsm_runners.items():
                    current_state = fsm_runner.current_state
                    if current_state is not None:
                        fsm_states[fsm_name] = current_state.name

                if fsm_states:
                    event_attrs["fsm_states"] = EventAttributeValue(
                        name="fsm_states",
                        value=str(fsm_states),
                        type="string"
                    )
            except Exception as e:
                logger.debug(f"Could not extract FSM states: {e}")

        # Extract objects from blackboard
        bb_objects = []
        if hasattr(tree_runner, 'bb'):
            bb_objects = extract_objects_from_blackboard(tree_runner.bb)

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
        logger.error(f"Failed to log tree tick event: {e}")


async def _log_state_snapshot(
    tree_instance: Any,
    snapshot_name: str,
    include_blackboard: bool,
    include_fsms: bool,
) -> None:
    """Log a state snapshot as an object."""
    config = get_config()
    if not config.enabled:
        return

    try:
        timestamp = datetime.now()

        # Build snapshot attributes
        snapshot_attrs = {}

        # Add tree name
        if hasattr(tree_instance, 'spec') and tree_instance.spec:
            snapshot_attrs["tree_name"] = ObjectAttributeValue(
                name="tree_name",
                value=tree_instance.spec.name,
                type="string",
                time=timestamp
            )

        # Add blackboard state
        if include_blackboard and hasattr(tree_instance, 'bb'):
            bb = tree_instance.bb
            if hasattr(bb, 'model_dump'):
                snapshot_attrs["blackboard"] = ObjectAttributeValue(
                    name="blackboard",
                    value=str(bb.model_dump()),
                    type="string",
                    time=timestamp
                )
            elif hasattr(bb, '__dict__'):
                snapshot_attrs["blackboard"] = ObjectAttributeValue(
                    name="blackboard",
                    value=str(bb.__dict__),
                    type="string",
                    time=timestamp
                )

        # Add FSM states
        if include_fsms and hasattr(tree_instance, '_fsm_runners'):
            fsm_states = {}
            for fsm_name, fsm_runner in tree_instance._fsm_runners.items():
                current_state = fsm_runner.current_state
                if current_state is not None:
                    fsm_states[fsm_name] = current_state.name

            snapshot_attrs["fsm_states"] = ObjectAttributeValue(
                name="fsm_states",
                value=str(fsm_states),
                type="string",
                time=timestamp
            )

        # Create snapshot object
        from ..spores.models import Object as SporesObject
        import uuid

        snapshot_obj = SporesObject(
            id=str(uuid.uuid4()),  # Generate unique ID for snapshot
            type=f"StateSnapshot:{snapshot_name}",
            attributes=snapshot_attrs
        )

        # Send object - type: ignore because pyright confuses Object kwarg with built-in
        await _send_log_record(LogRecord(object=snapshot_obj))  # type: ignore[arg-type]

    except Exception as e:
        logger.error(f"Failed to log state snapshot: {e}")


def log_tree_event(
    event_type: str,
    tree_name: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Decorator to log tree-level events.

    Use this for logging custom events at the tree level.

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
                type="string"
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
                            type="string"
                        )
                else:
                    event_attrs[key] = EventAttributeValue(
                        name=key,
                        value=str(value),
                        type="string"
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


__all__ = [
    'TreeSporesAdapter',
    'log_tree_event',
]
