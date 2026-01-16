#!/usr/bin/env python3
"""
Mycelium-specific exceptions.

All Mycelium exceptions inherit from MyceliumError for easy catching.
"""


class MyceliumError(Exception):
    """Base exception for all Mycelium errors."""


class TreeDefinitionError(MyceliumError):
    """Error in tree definition or structure."""


class FSMDiscoveryError(MyceliumError):
    """Error discovering or validating FSM state graph."""


class FSMInstantiationError(MyceliumError):
    """Error creating or initializing FSM instances."""


class StateNotFoundError(FSMDiscoveryError):
    """A required state could not be found in the FSM graph."""


class CircularStateError(FSMDiscoveryError):
    """FSM state graph has a problematic cycle (note: normal cycles are OK)."""
