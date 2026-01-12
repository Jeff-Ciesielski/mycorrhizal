#!/usr/bin/env python3
"""
Spores - Event and Object Logging for Observability

OCEL (Object-Centric Event Log) compatible logging system for tracking events
and objects in Mycorrhizal systems. Provides automatic extraction of events
and objects from DSL execution.

Usage:
    from mycorrhizal.spores import configure, get_spore_sync, get_spore_async
    from mycorrhizal.spores.transport import SyncFileTransport, AsyncFileTransport

    # Sync code
    configure(transport=SyncFileTransport("logs/ocel.jsonl"))
    spore = get_spore_sync(__name__)

    @spore.log_event(event_type="OrderCreated", order_id="order.id")
    def create_order(order: Order) -> Order:
        return order

    # Async code
    configure(transport=AsyncFileTransport("logs/ocel.jsonl"))
    aspore = get_spore_async(__name__)

    @aspore.log_event(event_type="OrderCreated", order_id="order.id")
    async def create_order_async(order: Order) -> Order:
        return order

DSL Adapters:
    HyphaAdapter - Log Petri net transitions with token relationships
    RhizomorphAdapter - Log behavior tree node execution with status
    EnokiAdapter - Log state machine execution and lifecycle events

Key Concepts:
    Event - Something that happens at a point in time with attributes
    Object - An entity with a type and attributes
    Relationship - Link between events and objects (e.g., "actor", "target")
    EventAttr - Mark blackboard fields for automatic event attribute extraction
    ObjectRef - Mark blackboard fields as object references with scope
    SporesAttr - Mark model fields for automatic object attribute logging

Object Scopes:
    EVENT - Object exists only for this event (default)
    GLOBAL - Object exists across all events (deduplicated by ID)
"""

from .core import (
    configure,
    get_config,
    get_object_cache,
    get_spore_sync,
    get_spore_async,
    spore,
    SporesConfig,
    EventLogger,
    AsyncEventLogger,
    SyncEventLogger,
)

from .models import (
    Event,
    Object,
    LogRecord,
    Relationship,
    EventAttributeValue,
    ObjectAttributeValue,
    ObjectRef,
    ObjectScope,
    EventAttr,
    SporesAttr,
    generate_event_id,
    generate_object_id,
    attribute_value_from_python,
    object_attribute_from_python,
)

from .cache import ObjectLRUCache

from .encoder import Encoder, JSONEncoder
from .transport import SyncTransport, AsyncTransport, Transport, SyncFileTransport, AsyncFileTransport

# DSL adapters
from .dsl import (
    HyphaAdapter,
    RhizomorphAdapter,
    EnokiAdapter,
)


__version__ = "0.1.0"


__all__ = [
    # Core API
    'configure',
    'get_config',
    'get_object_cache',
    'get_spore_sync',
    'get_spore_async',
    'EventLogger',
    'AsyncEventLogger',
    'SyncEventLogger',
    'spore',  # Legacy decorator
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
    'SporesAttr',
    'generate_event_id',
    'generate_object_id',
    'attribute_value_from_python',
    'object_attribute_from_python',

    # Cache
    'ObjectLRUCache',

    # Encoder
    'Encoder',
    'JSONEncoder',

    # Transport
    'SyncTransport',
    'AsyncTransport',
    'Transport',  # Alias for AsyncTransport (backward compatibility)
    'SyncFileTransport',
    'AsyncFileTransport',

    # DSL Adapters
    'HyphaAdapter',
    'RhizomorphAdapter',
    'EnokiAdapter',
]
