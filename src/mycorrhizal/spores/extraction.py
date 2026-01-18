#!/usr/bin/env python3
"""
Spores Data Extraction

Extract attributes and objects from function context for event logging.
"""

from __future__ import annotations

import inspect
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable, get_type_hints, Annotated, get_origin, get_args
from dataclasses import fields

from .models import (
    Event, Relationship, EventAttributeValue, Object, ObjectAttributeValue,
    ObjectRef, ObjectScope, EventAttr,
    attribute_value_from_python, object_attribute_from_python,
    generate_object_id
)


logger = logging.getLogger(__name__)


# ============================================================================
# Attribute Extraction
# ============================================================================

def extract_attributes_from_params(
    func: Callable,
    args: tuple,
    kwargs: dict,
    timestamp: datetime
) -> Dict[str, EventAttributeValue]:
    """
    Extract event attributes from function parameters.

    Args:
        func: The decorated function
        args: Positional arguments passed to function
        kwargs: Keyword arguments passed to function
        timestamp: Event timestamp (not used for attribute timestamps)

    Returns:
        Dictionary of attribute name -> EventAttributeValue
    """
    attributes = {}

    try:
        # Get function signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Extract from each parameter
        for param_name, param_value in bound_args.arguments.items():
            # Skip certain parameter names
            if param_name in ('self', 'cls', 'bb', 'ctx', 'timebase', 'consumed'):
                continue

            # Extract attribute value (time=None for event attributes)
            attr = attribute_value_from_python(param_value)
            attributes[param_name] = attr

    except Exception as e:
        logger.error(f"Failed to extract attributes from params: {e}")

    return attributes


def extract_attributes_from_dict(
    data: Dict[str, Any],
    timestamp: datetime
) -> Dict[str, EventAttributeValue]:
    """
    Extract event attributes from a dictionary.

    Args:
        data: Dictionary with attribute values
        timestamp: Event timestamp (not used for attribute timestamps)

    Returns:
        Dictionary of attribute name -> EventAttributeValue
    """
    attributes = {}

    for key, value in data.items():
        attr = attribute_value_from_python(value)
        attributes[key] = attr

    return attributes


def extract_attributes_from_blackboard(
    bb: Any,
    timestamp: datetime
) -> Dict[str, EventAttributeValue]:
    """
    Extract attributes from blackboard fields marked with EventAttr.

    Args:
        bb: Blackboard object
        timestamp: Event timestamp (not used for attribute timestamps)

    Returns:
        Dictionary of attribute name -> EventAttributeValue
    """
    attributes = {}

    try:
        # Get the class type hints - try __annotations__ first for Pydantic models
        cls = type(bb)

        # Pydantic models preserve Annotated in __annotations__
        if hasattr(cls, '__annotations__'):
            hints = cls.__annotations__
        else:
            hints = get_type_hints(cls)

        for field_name, field_type in hints.items():
            # Check if this is Annotated
            if get_origin(field_type) is Annotated:
                args = get_args(field_type)
                if len(args) >= 2:
                    # args[0] is the actual type, args[1:] are metadata
                    for metadata in args[1:]:
                        # Check if it's an EventAttr (class or instance)
                        if metadata is EventAttr or isinstance(metadata, EventAttr):
                            # Get the value from blackboard
                            if hasattr(bb, field_name):
                                value = getattr(bb, field_name)
                                # EventAttr might be a class or instance
                                if isinstance(metadata, EventAttr):
                                    attr_name = metadata.name or field_name
                                else:
                                    attr_name = field_name
                                attr = attribute_value_from_python(value)
                                attributes[attr_name] = attr

    except Exception as e:
        logger.error(f"Failed to extract attributes from blackboard: {e}")

    return attributes


def evaluate_computed_attributes(
    attr_spec: Dict[str, Any],
    bb: Any,
    timestamp: datetime
) -> Dict[str, EventAttributeValue]:
    """
    Evaluate computed/callable attributes.

    Args:
        attr_spec: Attribute specification (can contain callables)
        bb: Blackboard object for evaluation context
        timestamp: Event timestamp

    Returns:
        Dictionary of evaluated attributes
    """
    attributes = {}

    for key, value in attr_spec.items():
        if callable(value):
            # Call the function with blackboard
            try:
                result = value(bb)
                attr = attribute_value_from_python(result)
                attributes[key] = attr
            except Exception as e:
                logger.error(f"Failed to compute attribute {key}: {e}")
        else:
            # Static value
            attr = attribute_value_from_python(value)
            attributes[key] = attr

    return attributes


# ============================================================================
# Object Extraction
# ============================================================================

def extract_objects_from_blackboard(
    bb: Any,
    object_names: Optional[List[str]] = None
) -> List[Object]:
    """
    Extract OCEL objects from blackboard fields marked with ObjectRef.

    Args:
        bb: Blackboard object
        object_names: Optional list of specific field names to extract.
                     If None, extracts all global-scope objects.

    Returns:
        List of OCEL Object instances
    """
    objects = []

    try:
        # Get the class type hints - try __annotations__ first for Pydantic models
        cls = type(bb)

        # Pydantic models preserve Annotated in __annotations__
        if hasattr(cls, '__annotations__'):
            hints = cls.__annotations__
        else:
            hints = get_type_hints(cls)

        for field_name, field_type in hints.items():
            # Check if this is Annotated
            if get_origin(field_type) is Annotated:
                args = get_args(field_type)
                if len(args) >= 2:
                    # args[0] is the actual type, args[1:] are metadata
                    for metadata in args[1:]:
                        # Check if it's an ObjectRef (instance of frozen dataclass)
                        if isinstance(metadata, ObjectRef):
                            # Check if we should extract this object
                            if object_names is not None and field_name not in object_names:
                                # Only extract specified objects
                                if metadata.scope != ObjectScope.GLOBAL:
                                    continue

                            if isinstance(metadata.scope, str):
                                scope = ObjectScope(metadata.scope)
                            else:
                                scope = metadata.scope

                            if scope == ObjectScope.EVENT and field_name not in (object_names or []):
                                # Event-scope but not requested
                                continue

                            # Get the object value
                            if hasattr(bb, field_name):
                                obj_value = getattr(bb, field_name)
                                if obj_value is None:
                                    continue

                                # Convert to OCEL Object
                                ocel_obj = convert_to_ocel_object(
                                    obj_value,
                                    metadata.qualifier
                                )
                                if ocel_obj:
                                    objects.append(ocel_obj)

    except Exception as e:
        logger.error(f"Failed to extract objects from blackboard: {e}")

    return objects


def convert_to_ocel_object(
    obj: Any,
    qualifier: str = "related"
) -> Optional[Object]:
    """
    Convert a Python object to an OCEL Object.

    Args:
        obj: The Python object to convert
        qualifier: Relationship qualifier for this object

    Returns:
        OCEL Object instance, or None if conversion fails
    """
    try:
        timestamp = datetime.now()

        # Generate object ID
        obj_id = generate_object_id(obj)

        # Get object type
        obj_type = getattr(obj, '_spores_object_type', type(obj).__name__)

        # Extract attributes
        attributes = {}

        if hasattr(obj, '__dict__'):
            # Extract from instance dict
            for key, value in obj.__dict__.items():
                if not key.startswith('_'):
                    attr = object_attribute_from_python(value, timestamp)
                    attr = attr.__class__(
                        name=key,
                        value=attr.value,
                        type=attr.type,
                        time=attr.time
                    )
                    attributes[key] = attr

        elif hasattr(obj, '__dataclass_fields__'):
            # Extract dataclass fields
            for field_info in fields(obj):
                key = field_info.name
                value = getattr(obj, key)
                attr = object_attribute_from_python(value, timestamp)
                attr = attr.__class__(
                    name=key,
                    value=attr.value,
                    type=attr.type,
                    time=attr.time
                )
                attributes[key] = attr

        # Create OCEL Object
        return Object(
            id=obj_id,
            type=obj_type,
            attributes=attributes,
            relationships={}
        )

    except Exception as e:
        logger.error(f"Failed to convert object to OCEL: {e}")
        return None


def extract_objects_from_spec(
    object_spec: Dict[str, tuple],
    timestamp: datetime
) -> List[Object]:
    """
    Extract objects from a manual specification.

    Args:
        object_spec: Dict of {name: (qualifier, object)} or {name: object}
        timestamp: Timestamp for object attributes

    Returns:
        List of OCEL Object instances
    """
    objects = []

    for name, spec in object_spec.items():
        try:
            if isinstance(spec, tuple) and len(spec) == 2:
                qualifier, obj_value = spec
            else:
                qualifier = "related"
                obj_value = spec

            if obj_value is None:
                continue

            # Convert to OCEL Object
            obj_id = generate_object_id(obj_value)
            obj_type = getattr(obj_value, '_spores_object_type', type(obj_value).__name__)

            # Extract attributes
            attributes = {}
            if hasattr(obj_value, '__dict__'):
                for key, value in obj_value.__dict__.items():
                    if not key.startswith('_'):
                        attr = object_attribute_from_python(value, timestamp)
                        attr = attr.__class__(
                            name=key,
                            value=attr.value,
                            type=attr.type,
                            time=attr.time
                        )
                        attributes[key] = attr

            ocel_obj = Object(
                id=obj_id,
                type=obj_type,
                attributes=attributes,
                relationships={}
            )

            objects.append(ocel_obj)

        except Exception as e:
            logger.error(f"Failed to extract object {name}: {e}")

    return objects


# ============================================================================
# DSL-Specific Extraction
# ============================================================================

def extract_from_hypha_context(
    consumed: list,
    bb: Any,
    timebase: Any,
    timestamp: datetime
) -> tuple[Dict[str, EventAttributeValue], List[Object]]:
    """
    Extract attributes and objects from Hypha transition context.

    Args:
        consumed: List of consumed tokens
        bb: Blackboard
        timebase: Timebase
        timestamp: Event timestamp

    Returns:
        Tuple of (attributes, objects)
    """
    attributes = {}
    objects = []

    # Add token count
    attributes["token_count"] = EventAttributeValue(
        name="token_count",
        value=str(len(consumed)),
        time=timestamp
    )

    # Extract from blackboard
    bb_attrs = extract_attributes_from_blackboard(bb, timestamp)
    attributes.update(bb_attrs)

    bb_objects = extract_objects_from_blackboard(bb)
    objects.extend(bb_objects)

    # Extract objects from tokens
    for token in consumed:
        obj = convert_to_ocel_object(token, qualifier="input")
        if obj:
            objects.append(obj)

    return attributes, objects


def extract_from_rhizomorph_context(
    bb: Any,
    timestamp: datetime
) -> tuple[Dict[str, EventAttributeValue], List[Object]]:
    """
    Extract attributes and objects from Rhizomorph action context.

    Args:
        bb: Blackboard
        timestamp: Event timestamp

    Returns:
        Tuple of (attributes, objects)
    """
    # Extract from blackboard
    attributes = extract_attributes_from_blackboard(bb, timestamp)
    objects = extract_objects_from_blackboard(bb)

    return attributes, objects


def extract_from_septum_context(
    ctx: Any,
    timestamp: datetime
) -> tuple[Dict[str, EventAttributeValue], List[Object]]:
    """
    Extract attributes and objects from Septum state context.

    Args:
        ctx: SharedContext
        timestamp: Event timestamp

    Returns:
        Tuple of (attributes, objects)
    """
    attributes = {}
    objects = []

    # Add state name if available
    if hasattr(ctx, 'current_state') and ctx.current_state:
        state_name = getattr(ctx.current_state, 'name', str(ctx.current_state))
        attributes["state_name"] = EventAttributeValue(
            name="state_name",
            value=state_name,
            time=timestamp
        )

    # Extract from common (blackboard)
    if hasattr(ctx, 'common'):
        bb_attrs = extract_attributes_from_blackboard(ctx.common, timestamp)
        attributes.update(bb_attrs)

        bb_objects = extract_objects_from_blackboard(ctx.common)
        objects.extend(bb_objects)

    return attributes, objects
