#!/usr/bin/env python3
"""
Spores - Event and Object Logging for Mycorrhizal

Spores provides decorator-based event and object logging for OCPM integration.
Generates OCEL-compatible logs for consumption by OCEL systems.

Example:
    ```python
    from mycorrhizal.spores import spore
    from mycorrhizal.spores.transport import FileTransport

    # Configure spores
    spore.configure(transport=FileTransport("logs/ocel.jsonl"))

    # Mark objects
    @spore.object(object_type="Robot")
    class Robot(BaseModel):
        id: str
        name: str

    # Log events
    @spore.event(event_type="navigate")
    async def navigate(bb: MissionContext):
        return Status.SUCCESS
    ```
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
