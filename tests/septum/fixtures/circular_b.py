"""Test fixture for circular import testing - Module B.

This module CANNOT import Module A (would cause circular import).
Instead, it uses a string-based reference to StateA.

This proves that the global registry enables string references
to break circular dependencies.
"""

from mycorrhizal.septum.core import septum, LabeledTransition, StateRef, StateConfiguration
from enum import Enum, auto


@septum.state(config=StateConfiguration(can_dwell=True))
def StateB():
    """State in module B that transitions to StateA via string reference."""

    class Events(Enum):
        BACK = auto()

    @septum.on_state
    async def on_state(ctx):
        if ctx.msg == "back":
            return Events.BACK
        return None

    @septum.transitions
    def transitions():
        return [
            # String reference to avoid circular import!
            LabeledTransition(Events.BACK, StateRef("tests.septum.fixtures.circular_a.StateA")),
        ]
