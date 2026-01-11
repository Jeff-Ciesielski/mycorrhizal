#!/usr/bin/env python3
"""
Basic Enoki Decorator Example

This example demonstrates the simplest possible state machine using
the decorator-based Enoki API.
"""

import asyncio
from mycorrhizal.enoki.core import (
    enoki,
    StateMachine,
    StateConfiguration,
    SharedContext,
    LabeledTransition,
    Again,
    StateMachineComplete,
)
from enum import Enum, auto


# ============================================================================
# Define States using Decorators
# ============================================================================

@enoki.state(config=StateConfiguration(can_dwell=True))
def IdleState():
    """Initial idle state waiting for work"""

    class Events(Enum):
        START = auto()
        QUIT = auto()

    @enoki.on_state
    async def on_state(ctx: SharedContext):
        msg = ctx.msg
        if msg == "start":
            print("Idle: Received start command")
            return Events.START
        elif msg == "quit":
            print("Idle: Received quit command")
            return Events.QUIT
        return None  # Wait for message

    @enoki.transitions
    def transitions():
        return [
            LabeledTransition(Events.START, ProcessingState),
            LabeledTransition(Events.QUIT, DoneState),
        ]


@enoki.state()
def ProcessingState():
    """Process data and return to idle when done"""

    class Events(Enum):
        DONE = auto()

    @enoki.on_enter
    async def on_enter(ctx: SharedContext):
        print("Processing: Starting work")
        ctx.common["process_count"] = ctx.common.get("process_count", 0) + 1

    @enoki.on_state
    async def on_state(ctx: SharedContext):
        print("Processing: Doing work...")
        await asyncio.sleep(0.1)  # Simulate work
        print("Processing: Work complete")
        return Events.DONE

    @enoki.on_leave
    async def on_leave(ctx: SharedContext):
        print("Processing: Cleaning up")

    @enoki.transitions
    def transitions():
        return [
            LabeledTransition(Events.DONE, IdleState),
        ]


@enoki.state(config=StateConfiguration(terminal=True))
def DoneState():
    """Terminal state - FSM will exit when reaching this state"""

    @enoki.on_state
    async def on_state(ctx: SharedContext):
        print(f"Done: Processed {ctx.common.get('process_count', 0)} items")
        print("Done: State machine complete")


# ============================================================================
# Main
# ============================================================================

async def main():
    print("=== Basic Enoki Decorator Example ===\n")

    # Create state machine
    fsm = StateMachine(
        initial_state=IdleState,
        common_data={},
    )

    # Initialize
    await fsm.initialize()
    print("State machine initialized\n")

    # Send some messages
    print("Sending: start")
    fsm.send_message("start")

    print("Sending: start")
    fsm.send_message("start")

    print("Sending: start")
    fsm.send_message("start")

    print("Sending: quit\n")
    fsm.send_message("quit")

    # Run the state machine (will exit when DoneState is reached)
    try:
        await fsm.run()
    except StateMachineComplete:
        pass  # Expected when reaching terminal state
    print("\nState machine finished!")


if __name__ == "__main__":
    asyncio.run(main())
