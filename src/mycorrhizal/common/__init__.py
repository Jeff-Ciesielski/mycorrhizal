"""
Common utilities and interfaces for Mycorrhizal.

This module provides shared components used across all three DSL systems:
- Hypha (Petri Nets)
- Rhizomorph (Behavior Trees)
- Enoki (State Machines)
"""

# Import interfaces for easy access
from mycorrhizal.common.interfaces import (
    Readable,
    Writable,
    FieldAccessor,
    BlackboardProtocol,
    FieldMetadata,
    InterfaceMetadata,
    validate_implements,
    get_interface_fields,
    create_interface_from_model,
)

# Import wrappers for easy access
from mycorrhizal.common.wrappers import (
    AccessControlError,
    ReadOnlyView,
    WriteOnlyView,
    ConstrainedView,
    CompositeView,
    create_readonly_view,
    create_constrained_view,
    create_view_from_protocol,
    View,
)

# Import interface builder for easy access
from mycorrhizal.common.interface_builder import (
    blackboard_interface,
    FieldSpec,
)

__all__ = [
    # Interfaces
    "Readable",
    "Writable",
    "FieldAccessor",
    "BlackboardProtocol",
    "FieldMetadata",
    "InterfaceMetadata",
    "validate_implements",
    "get_interface_fields",
    "create_interface_from_model",

    # Wrappers
    "AccessControlError",
    "ReadOnlyView",
    "WriteOnlyView",
    "ConstrainedView",
    "CompositeView",
    "create_readonly_view",
    "create_constrained_view",
    "create_view_from_protocol",
    "View",

    # Interface Builder
    "blackboard_interface",
    "FieldSpec",
]
