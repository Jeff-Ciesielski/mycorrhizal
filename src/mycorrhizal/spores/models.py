#!/usr/bin/env python3
"""
Spores Data Models

OCEL-compatible data models for event and object logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Union, Any
from enum import Enum


# ============================================================================
# OCEL Data Models
# ============================================================================

@dataclass(frozen=True)
class Relationship:
    """
    A relationship from an event to an object.

    Attributes:
        object_id: The ID of the related object
        qualifier: How the object relates to the event (e.g., "input", "output", "actor")
    """
    object_id: str
    qualifier: str


@dataclass(frozen=True)
class EventAttributeValue:
    """
    An attribute value for an event.

    OCEL attributes include timestamps for versioning.

    Attributes:
        name: The attribute name
        value: The string representation of the value
        time: When this attribute value was set
    """
    name: str
    value: str
    time: datetime


@dataclass(frozen=True)
class ObjectAttributeValue:
    """
    An attribute value for an object.

    OCEL attributes include timestamps for versioning.

    Attributes:
        name: The attribute name
        value: The string representation of the value
        time: When this attribute value was set
    """
    name: str
    value: str
    time: datetime


@dataclass
class Event:
    """
    An OCEL event representing something that happened.

    Attributes:
        id: Unique event identifier
        type: Event type/category
        time: When the event occurred
        attributes: Event attributes (name -> EventAttributeValue)
        relationships: Objects related to this event (qualifier -> Relationship)
    """
    id: str
    type: str
    time: datetime
    attributes: Dict[str, EventAttributeValue] = field(default_factory=dict)
    relationships: Dict[str, Relationship] = field(default_factory=dict)


@dataclass
class Object:
    """
    An OCEL object representing an entity in the system.

    Attributes:
        id: Unique object identifier
        type: Object type/category
        attributes: Object attributes (name -> ObjectAttributeValue)
        relationships: Other objects related to this object (qualifier -> Relationship)
    """
    id: str
    type: str
    attributes: Dict[str, ObjectAttributeValue] = field(default_factory=dict)
    relationships: Dict[str, Relationship] = field(default_factory=dict)


@dataclass
class LogRecord:
    """
    A log record containing either an event or an object (not both).

    This is the union type used for streaming to OCEL consumers.

    Attributes:
        event: An event record (if logging an event)
        object: An object record (if logging an object)

    Note:
        Exactly one of event or object must be set.
    """
    event: Optional[Event] = None
    object: Optional[Object] = None

    def __post_init__(self):
        """Validate that exactly one of event or object is set."""
        if self.event is None and self.object is None:
            raise ValueError("LogRecord must have either event or object set")
        if self.event is not None and self.object is not None:
            raise ValueError("LogRecord cannot have both event and object set")


# ============================================================================
# Spores-Specific Metadata
# ============================================================================

class ObjectScope(Enum):
    """Scope for object associations with events."""
    GLOBAL = "global"  # Included in every event
    EVENT = "event"    # Only when explicitly requested
    AUTO = "auto"      # Automatically detected from context


@dataclass(frozen=True)
class ObjectRef:
    """
    Metadata annotation for object references in blackboard interfaces.

    Used with typing.Annotated to mark fields as OCEL objects:

        robot: Annotated[Robot, ObjectRef(qualifier="actor", scope=ObjectScope.GLOBAL)]

    Attributes:
        qualifier: How the object relates to events ("input", "output", "actor", etc.)
        scope: When to include this object in events
    """
    qualifier: str
    scope: Union[ObjectScope, str] = ObjectScope.EVENT

    def __post_init__(self):
        """Convert string scope to ObjectScope enum if needed."""
        if isinstance(self.scope, str):
            object.__setattr__(self, 'scope', ObjectScope(self.scope))


@dataclass(frozen=True)
class EventAttr:
    """
    Metadata annotation for event attributes in blackboard interfaces.

    Used with typing.Annotated to mark fields that should be logged as event attributes:

        mission_id: Annotated[str, EventAttr]
        battery_level: Annotated[int, EventAttr]

    Attributes:
        name: Optional custom name for the attribute (defaults to field name)
    """
    name: Optional[str] = None


# ============================================================================
# Attribute Value Conversion
# ============================================================================

def attribute_value_from_python(value: Any, time: datetime) -> EventAttributeValue:
    """
    Convert a Python value to an OCEL EventAttributeValue.

    All values are converted to strings per OCEL specification.

    Args:
        value: The Python value to convert
        time: The timestamp for this attribute value

    Returns:
        An EventAttributeValue with string representation
    """
    # Handle None
    if value is None:
        str_value = "null"

    # Handle enums (use name)
    elif isinstance(value, Enum):
        str_value = value.name

    # Handle datetime (ISO format)
    elif isinstance(value, datetime):
        str_value = value.isoformat()

    # Handle bool (lowercase true/false)
    elif isinstance(value, bool):
        str_value = "true" if value else "false"

    # Handle primitives
    elif isinstance(value, (str, int, float)):
        str_value = str(value)

    # Everything else: convert to string
    else:
        str_value = str(value)

    return EventAttributeValue(
        name="",  # Name set by caller
        value=str_value,
        time=time
    )


def object_attribute_from_python(value: Any, time: datetime) -> ObjectAttributeValue:
    """
    Convert a Python value to an OCEL ObjectAttributeValue.

    Same conversion rules as attribute_value_from_python but for objects.

    Args:
        value: The Python value to convert
        time: The timestamp for this attribute value

    Returns:
        An ObjectAttributeValue with string representation
    """
    # Use same conversion logic
    event_attr = attribute_value_from_python(value, time)

    return ObjectAttributeValue(
        name=event_attr.name,
        value=event_attr.value,
        time=event_attr.time
    )


# ============================================================================
# Event ID Generation
# ============================================================================

def generate_event_id() -> str:
    """
    Generate a unique event ID.

    Simple implementation using counter. Could be enhanced with UUIDs
    or more sophisticated schemes.

    Returns:
        A unique event identifier
    """
    import uuid
    return f"evt-{uuid.uuid4()}"


def generate_object_id(obj: Any) -> str:
    """
    Generate an object ID from an object.

    Tries to use obj.id if available, otherwise generates a UUID.

    Args:
        obj: The object to generate an ID for

    Returns:
        A unique object identifier
    """
    # Try to get id attribute
    if hasattr(obj, 'id'):
        return str(obj.id)

    # Try Pydantic model's field
    if hasattr(obj, '__dict__') and 'id' in obj.__dict__:
        return str(obj.__dict__['id'])

    # Generate UUID-based ID
    import uuid
    return f"obj-{uuid.uuid4()}"
