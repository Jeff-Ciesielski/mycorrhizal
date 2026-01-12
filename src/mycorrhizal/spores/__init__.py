#!/usr/bin/env python3
"""
Spores - Event and Object Logging for Observability

OCEL (Object-Centric Event Log) compatible logging system for tracking events
and objects in Mycorrhizal systems. Provides automatic extraction of events
and objects from DSL execution.

Usage (Standalone Event and Object Logging):
    from mycorrhizal.spores import configure, get_spore_sync, get_spore_async
    from mycorrhizal.spores.transport import SyncFileTransport, AsyncFileTransport
    from mycorrhizal.spores.models import SporesAttr
    from pydantic import BaseModel
    from typing import Annotated

    # Mark domain model fields for object attribute logging
    class Order(BaseModel):
        id: str
        status: Annotated[str, SporesAttr]
        total: Annotated[float, SporesAttr]

    # Sync code
    configure(transport=SyncFileTransport("logs/ocel.jsonl"))
    spore = get_spore_sync(__name__)

    @spore.log_event(
        event_type="OrderCreated",
        relationships={"order": ("return", "Order")},
    )
    def create_order(customer: Customer, items: list) -> Order:
        return Order(...)

    # Async code
    configure(transport=AsyncFileTransport("logs/ocel.jsonl"))
    aspore = get_spore_async(__name__)

    @aspore.log_event(
        event_type="OrderCreated",
        relationships={"order": ("return", "Order")},
    )
    async def create_order_async(customer: Customer, items: list) -> Order:
        return Order(...)

Usage (DSL Adapters):
    from mycorrhizal.spores import configure, spore
    from mycorrhizal.spores.models import EventAttr, ObjectRef, ObjectScope
    from pydantic import BaseModel
    from typing import Annotated

    # Mark blackboard fields for event attribute extraction
    class MissionContext(BaseModel):
        mission_id: Annotated[str, EventAttr]
        robot: Annotated[Robot, ObjectRef(qualifier="actor", scope=ObjectScope.GLOBAL)]

    # Mark object types for logging
    @spore.object(object_type="Robot")
    class Robot(BaseModel):
        id: str
        name: str

DSL Adapters:
    HyphaAdapter - Log Petri net transitions with token relationships
    RhizomorphAdapter - Log behavior tree node execution with status
    EnokiAdapter - Log state machine execution and lifecycle events

Annotation Types:
    EventAttr - Mark blackboard fields for automatic event attribute extraction (DSL adapters)
    SporesAttr - Mark domain model fields for automatic object attribute logging (log_event)
    ObjectRef - Mark blackboard fields as object references with scope

Key Concepts:
    Event - Something that happens at a point in time with attributes
    Object - An entity with a type and attributes
    Relationship - Link between events and objects (e.g., "actor", "target")

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


try:
    from mycorrhizal._version import version as __version__
except ImportError:
    __version__ = "0.0.0+dev"


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
    'spore',  # For DSL adapters and @spore.object() decorator
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
