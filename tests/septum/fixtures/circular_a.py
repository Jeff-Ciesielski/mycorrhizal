"""Test fixture for circular import testing - Module A.

This module imports Module B directly and transitions to its state.
Module B will reference this module via string to avoid circular import.
"""

from mycorrhizal.septum.core import septum, LabeledTransition, StateConfiguration
from tests.septum.fixtures.circular_b import StateB
from enum import Enum, auto


@septum.state(config=StateConfiguration(can_dwell=True))
def StateA():
    """State in module A that transitions to StateB directly."""

    class Events(Enum):
        GO = auto()
        BACK = auto()

    @septum.on_state
    async def on_state(ctx):
        if ctx.msg == "go":
            return Events.GO
        elif ctx.msg == "back":
            return Events.BACK
        return None

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.GO, StateB),  # Direct reference - OK!
            LabeledTransition(Events.BACK, StateA),  # Self-reference also OK
        ]
