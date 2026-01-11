#!/usr/bin/env python3
"""
Spores - Event and Object Logging for Observability

OCEL (Object-Centric Event Log) compatible logging system for tracking events
and objects in Mycorrhizal systems. Provides automatic extraction of events
and objects from DSL execution.

Usage:
    from mycorrhizal.spores import configure, spore, EventAttr, ObjectRef

    # Configure logging
    configure(
        enabled=True,
        object_cache_size=100,
        transport=FileTransport("logs/ocel.jsonl"),
    )

    # Mark model fields for automatic extraction
    class MissionContext(BaseModel):
        mission_id: Annotated[str, EventAttr]
        robot: Annotated[Robot, ObjectRef(qualifier="actor", scope=ObjectScope.GLOBAL)]

    # Mark object types
    @spore.object(object_type="Robot")
    class Robot(BaseModel):
        id: str
        name: str

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

Object Scopes:
    EVENT - Object exists only for this event (default)
    GLOBAL - Object exists across all events (deduplicated by ID)
"""

from .core import (
    configure,
    get_config,
    get_object_cache,
    spore,
    SporesConfig,
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
    generate_event_id,
    generate_object_id,
    attribute_value_from_python,
    object_attribute_from_python,
)

from .cache import ObjectLRUCache

from .encoder import Encoder, JSONEncoder
from .transport import Transport

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
    'attribute_value_from_python',
    'object_attribute_from_python',

    # Cache
    'ObjectLRUCache',

    # Encoder
    'Encoder',
    'JSONEncoder',

    # Transport
    'Transport',

    # DSL Adapters
    'HyphaAdapter',
    'RhizomorphAdapter',
    'EnokiAdapter',
]
