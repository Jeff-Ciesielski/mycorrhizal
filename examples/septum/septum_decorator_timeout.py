#!/usr/bin/env python3
"""
Septum Timeout Example

This example demonstrates how to use timeouts with states.
A state with a timeout will call on_timeout if no message is
received within the specified time period.
"""

import asyncio
from mycorrhizal.septum.core import (
    septum,
    StateMachine,
    StateConfiguration,
    SharedContext,
    LabeledTransition,
    StateMachineComplete,
)
from enum import Enum, auto


# ============================================================================
# Define States with Timeouts
# ============================================================================

@septum.state(config=StateConfiguration(timeout=2.0))
def AwaitingResponseState():
    """
    Wait for a response with a 2-second timeout.
    If no response arrives within 2 seconds, on_timeout will be called.
    """

    class Events(Enum):
        RESPONSE = auto()
        TIMEOUT = auto()

    @septum.on_enter
    async def on_enter(ctx: SharedContext):
        print("AwaitingResponse: Request sent, waiting for response...")

    @septum.on_state
    async def on_state(ctx: SharedContext):
        # If we have a message, process it
        if ctx.msg is not None:
            print(f"AwaitingResponse: Received response: {ctx.msg}")
            return Events.RESPONSE
        # No message yet - return None to wait (timeout will trigger if no message)
        return None

    @septum.on_timeout
    async def on_timeout(ctx: SharedContext):
        print("AwaitingResponse: Timeout! No response received.")
        # Transition to retry state
        return Events.TIMEOUT

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.RESPONSE, SuccessState),
            LabeledTransition(Events.TIMEOUT, RetryState),
        ]



@septum.state()
def RetryState():
    """Retry the request"""

    class Events(Enum):
        RETRY = auto()
        GIVE_UP = auto()

    @septum.on_state
    async def on_state(ctx: SharedContext):
        retry_count = ctx.common.get("retry_count", 0)
        retry_count += 1
        ctx.common["retry_count"] = retry_count

        if retry_count > 2:
            print("Retry: Max retries exceeded, giving up")
            return Events.GIVE_UP
        else:
            print(f"Retry: Attempt {retry_count}/3")
            # Send another request by transitioning back to awaiting state
            return Events.RETRY

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.RETRY, AwaitingResponseState),
            LabeledTransition(Events.GIVE_UP, FailureState),
        ]



@septum.state(config=StateConfiguration(terminal=True))
def SuccessState():
    """Request succeeded"""

    @septum.on_state
    async def on_state(ctx: SharedContext):
        print("Success: Request completed successfully!")



@septum.state(config=StateConfiguration(terminal=True))
def FailureState():
    """Request failed after retries"""

    @septum.on_state
    async def on_state(ctx: SharedContext):
        print("Failure: Request failed after max retries")



# ============================================================================
# Main - Demonstrate timeout behavior
# ============================================================================

async def demo_with_response():
    """Demo where response arrives in time"""
    print("\n=== Demo 1: Response arrives in time ===\n")

    fsm = StateMachine(
        initial_state=AwaitingResponseState,
        common_data={},
    )

    await fsm.initialize()

    # Send a response before timeout
    await asyncio.sleep(0.5)
    print("Demo: Sending response...")
    fsm.send_message("OK")

    try:
        await fsm.run()
    except StateMachineComplete:
        pass  # Expected when reaching terminal state


async def demo_timeout_once():
    """Demo where timeout occurs once, then response arrives"""
    print("\n=== Demo 2: Timeout once, then response ===\n")

    fsm = StateMachine(
        initial_state=AwaitingResponseState,
        common_data={},
    )

    await fsm.initialize()

    # Wait for timeout (2 seconds), then send response
    await asyncio.sleep(2.5)
    print("Demo: Sending response after timeout...")
    fsm.send_message("OK")

    try:
        await fsm.run()
    except StateMachineComplete:
        pass  # Expected when reaching terminal state


async def demo_max_retries():
    """Demo where timeout occurs multiple times"""
    print("\n=== Demo 3: Multiple timeouts (max retries) ===\n")

    fsm = StateMachine(
        initial_state=AwaitingResponseState,
        common_data={},
    )

    await fsm.initialize()

    # Just wait - no response ever arrives
    # Will timeout 3 times then give up
    try:
        await fsm.run()
    except StateMachineComplete:
        pass  # Expected when reaching terminal state


async def main():
    print("=== Septum Timeout Example ===")
    print("States have timeouts to prevent infinite waiting\n")

    # Demo 1: Response arrives in time
    await demo_with_response()

    # Demo 2: Timeout occurs once
    await demo_timeout_once()

    # Demo 3: Multiple timeouts
    await demo_max_retries()

    print("\n=== All demos complete ===")


if __name__ == "__main__":
    asyncio.run(main())
