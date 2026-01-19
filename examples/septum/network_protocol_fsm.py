#!/usr/bin/env python3
"""
Network Protocol State Machine

This example demonstrates a more complex Septum FSM that models a network
protocol connection lifecycle with multiple states, error handling, retries,
and push/pop for nested protocol states.

Protocol States:
- Idle: Initial state, waiting for connection request
- Connecting: Establishing connection (with timeout and retry)
- Authenticating: Performing authentication (pushed onto stack)
- Connected: Normal operation state
- Transferring: Data transfer sub-state (pushed onto stack)
- ErrorHandling: Centralized error state
- Disconnecting: Graceful shutdown
- Disconnected: Terminal state

This demonstrates:
- 8 states with realistic use case
- Timeouts on connection, authentication, and transfer
- Retry logic with counter
- Push/pop for nested protocol states (authentication)
- Error handling and recovery
- Graceful shutdown sequence
- Self-loop transitions (heartbeat)
- Message-based state transitions
"""

import asyncio
import random
from mycorrhizal.septum.core import (
    septum,
    StateMachine,
    StateConfiguration,
    SharedContext,
    LabeledTransition,
    Push,
    Pop,
    Again,
    Retry,
    StateMachineComplete,
)
from mycorrhizal.septum.util import to_mermaid
from enum import Enum, auto


# ============================================================================
# Protocol State Machine
# ============================================================================

@septum.state(config=StateConfiguration(can_dwell=True))
def IdleState():
    """
    Initial idle state waiting for connection request.

    This state can dwell indefinitely (no timeout) as it's waiting for
    an external trigger.
    """

    class Events(Enum):
        CONNECT = auto()
        SHUTDOWN = auto()

    @septum.on_state
    async def on_state(ctx: SharedContext):
        msg = ctx.msg
        if msg == "connect":
            print("Idle: Connection request received")
            return Events.CONNECT
        elif msg == "shutdown":
            print("Idle: Shutdown request, exiting")
            return Events.SHUTDOWN
        return None  # Wait for message

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.CONNECT, ConnectingState),
            LabeledTransition(Events.SHUTDOWN, DisconnectedState),
        ]


@septum.state(config=StateConfiguration(timeout=3.0, retries=3))
def ConnectingState():
    """
    Establish network connection with timeout and retry logic.

    This state demonstrates:
    - Timeout configuration (3 seconds)
    - Retry configuration (3 attempts)
    - Transition to push-based nested state (authentication)
    """

    class Events(Enum):
        CONNECTED = auto()
        TIMEOUT = auto()
        MAX_RETRIES = auto()
        FAIL = auto()

    @septum.on_enter
    async def on_enter(ctx: SharedContext):
        retry_count = ctx.common.get("connect_retries", 0)
        print(f"Connecting: Attempting connection (attempt {retry_count + 1}/3)")
        ctx.common["connect_retries"] = retry_count + 1

    @septum.on_state
    async def on_state(ctx: SharedContext):
        # Simulate connection attempt with random success/failure
        await asyncio.sleep(0.5)

        # Simulate 70% success rate
        if random.random() < 0.7:
            print("Connecting: Connection established")
            ctx.common["connect_retries"] = 0  # Reset on success
            return Events.CONNECTED
        else:
            print("Connecting: Connection failed")
            return Events.FAIL

    @septum.on_timeout
    async def on_timeout(ctx: SharedContext):
        retry_count = ctx.common.get("connect_retries", 0)
        print(f"Connecting: Timeout (attempt {retry_count}/3)")
        if retry_count >= 3:
            return Events.MAX_RETRIES
        return Events.TIMEOUT

    @septum.on_fail
    async def on_fail(ctx: SharedContext):
        print("Connecting: Max retries exceeded, going to error state")
        return Events.MAX_RETRIES

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.CONNECTED, Push(AuthenticatingState, ConnectedState)),
            LabeledTransition(Events.TIMEOUT, Retry()),
            LabeledTransition(Events.MAX_RETRIES, ErrorHandlingState),
            LabeledTransition(Events.FAIL, Retry()),
        ]


@septum.state(config=StateConfiguration(timeout=2.0))
def AuthenticatingState():
    """
    Authenticate with the server.

    This state is pushed onto the stack by ConnectingState.
    When authentication completes, it pops back to the previous state.
    """

    class Events(Enum):
        SUCCESS = auto()
        FAILURE = auto()
        TIMEOUT = auto()

    @septum.on_enter
    async def on_enter(ctx: SharedContext):
        print("Authenticating: Starting authentication...")

    @septum.on_state
    async def on_state(ctx: SharedContext):
        # Simulate authentication
        await asyncio.sleep(0.3)

        # Simulate 80% success rate
        if random.random() < 0.8:
            print("Authenticating: Authentication successful")
            return Events.SUCCESS
        else:
            print("Authenticating: Authentication failed")
            return Events.FAILURE

    @septum.on_timeout
    async def on_timeout(ctx: SharedContext):
        print("Authenticating: Timeout during authentication")
        return Events.TIMEOUT

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.SUCCESS, Pop()),
            LabeledTransition(Events.FAILURE, ErrorHandlingState),
            LabeledTransition(Events.TIMEOUT, ErrorHandlingState),
        ]


@septum.state(config=StateConfiguration(can_dwell=True))
def ConnectedState():
    """
    Main connected state, ready for data transfer or commands.

    This state can dwell indefinitely, waiting for commands or data transfer requests.
    """

    class Events(Enum):
        TRANSFER = auto()
        DISCONNECT = auto()
        HEARTBEAT = auto()

    @septum.on_enter
    async def on_enter(ctx: SharedContext):
        print("Connected: Session established and ready")
        ctx.common["transfer_count"] = 0

    @septum.on_state
    async def on_state(ctx: SharedContext):
        msg = ctx.msg
        if msg == "transfer":
            return Events.TRANSFER
        elif msg == "disconnect":
            return Events.DISCONNECT
        elif msg == "heartbeat":
            return Events.HEARTBEAT
        return None

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.TRANSFER, TransferringState),
            LabeledTransition(Events.DISCONNECT, DisconnectingState),
            LabeledTransition(Events.HEARTBEAT, Again()),
        ]


@septum.state(config=StateConfiguration(timeout=5.0))
def TransferringState():
    """
    Data transfer state with timeout support.

    Returns to ConnectedState when complete or on error.
    """

    class Events(Enum):
        COMPLETE = auto()
        TIMEOUT = auto()
        ERROR = auto()

    @septum.on_enter
    async def on_enter(ctx: SharedContext):
        count = ctx.common.get("transfer_count", 0)
        ctx.common["transfer_count"] = count + 1
        print(f"Transferring: Starting data transfer #{count + 1}")

    @septum.on_state
    async def on_state(ctx: SharedContext):
        # Simulate data transfer
        await asyncio.sleep(0.5)

        # Simulate 90% success rate
        if random.random() < 0.9:
            print("Transferring: Transfer complete")
            return Events.COMPLETE
        else:
            print("Transferring: Transfer error")
            return Events.ERROR

    @septum.on_timeout
    async def on_timeout(ctx: SharedContext):
        print("Transferring: Timeout during transfer")
        return Events.TIMEOUT

    @septum.on_leave
    async def on_leave(ctx: SharedContext):
        print("Transferring: Cleanup complete")

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.COMPLETE, ConnectedState),
            LabeledTransition(Events.ERROR, ErrorHandlingState),
            LabeledTransition(Events.TIMEOUT, ErrorHandlingState),
        ]


@septum.state()
def ErrorHandlingState():
    """
    Centralized error handling state.

    Attempts recovery or escalates to disconnection.
    """

    class Events(Enum):
        RETRY = auto()
        FATAL = auto()
        RECOVERED = auto()

    @septum.on_state
    async def on_state(ctx: SharedContext):
        error_count = ctx.common.get("error_count", 0)
        error_count += 1
        ctx.common["error_count"] = error_count

        print(f"ErrorHandling: Handling error #{error_count}")

        # Simulate error handling logic
        if error_count < 3:
            print("ErrorHandling: Attempting recovery")
            await asyncio.sleep(0.2)
            return Events.RETRY
        else:
            print("ErrorHandling: Too many errors, fatal failure")
            return Events.FATAL

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.RETRY, IdleState),
            LabeledTransition(Events.FATAL, DisconnectedState),
        ]


@septum.state()
def DisconnectingState():
    """
    Graceful shutdown state.

    Performs cleanup before final disconnection.
    """

    class Events(Enum):
        DONE = auto()

    @septum.on_enter
    async def on_enter(ctx: SharedContext):
        print("Disconnecting: Initiating graceful shutdown...")

    @septum.on_state
    async def on_state(ctx: SharedContext):
        # Simulate graceful shutdown
        await asyncio.sleep(0.3)
        print("Disconnecting: Shutdown complete")
        return Events.DONE

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.DONE, DisconnectedState),
        ]


@septum.state(config=StateConfiguration(terminal=True))
def DisconnectedState():
    """
    Terminal state - connection is closed.

    Reaching this state completes the FSM.
    """

    @septum.on_state
    async def on_state(ctx: SharedContext):
        error_count = ctx.common.get("error_count", 0)
        transfer_count = ctx.common.get("transfer_count", 0)
        print(f"\nDisconnected: Session ended")
        print(f"  Transfers completed: {transfer_count}")
        print(f"  Errors encountered: {error_count}")


# ============================================================================
# Demo Functions
# ============================================================================

async def demo_successful_connection():
    """Demo: Successful connection, authentication, and data transfer"""
    print("\n" + "="*70)
    print("DEMO 1: Successful Connection and Transfer")
    print("="*70 + "\n")

    fsm = StateMachine(
        initial_state=IdleState,
        common_data={},
    )

    await fsm.initialize()

    # Start connection
    fsm.send_message("connect")

    # Wait for connection and auth
    await asyncio.sleep(0.5)

    # Initiate data transfer
    fsm.send_message("transfer")

    # Wait for transfer
    await asyncio.sleep(0.7)

    # Another transfer
    fsm.send_message("transfer")
    await asyncio.sleep(0.7)

    # Disconnect
    fsm.send_message("disconnect")

    # Run until terminal state
    try:
        await fsm.run()
    except StateMachineComplete:
        pass


async def demo_connection_timeout():
    """Demo: Connection timeout and retry"""
    print("\n" + "="*70)
    print("DEMO 2: Connection Timeout with Retry")
    print("="*70 + "\n")

    # Force connection failures by manipulating random seed
    random.seed(42)  # This seed causes connection failures

    fsm = StateMachine(
        initial_state=IdleState,
        common_data={},
    )

    await fsm.initialize()
    fsm.send_message("connect")

    try:
        await fsm.run()
    except StateMachineComplete:
        pass


async def demo_data_transfer_error():
    """Demo: Error during data transfer with recovery"""
    print("\n" + "="*70)
    print("DEMO 3: Data Transfer Error Recovery")
    print("="*70 + "\n")

    # Force transfer error
    random.seed(123)  # This seed causes transfer failures

    fsm = StateMachine(
        initial_state=IdleState,
        common_data={},
    )

    await fsm.initialize()
    fsm.send_message("connect")
    await asyncio.sleep(0.3)
    fsm.send_message("transfer")
    await asyncio.sleep(0.7)
    fsm.send_message("transfer")  # This will fail
    await asyncio.sleep(0.7)

    try:
        await fsm.run()
    except StateMachineComplete:
        pass


async def demo_mermaid_diagram():
    """Print the Mermaid diagram for the protocol FSM"""
    print("\n" + "="*70)
    print("MERMAID DIAGRAM")
    print("="*70 + "\n")

    fsm = StateMachine(initial_state=IdleState, common_data={})
    await fsm.initialize()

    diagram = to_mermaid(fsm)
    print(diagram)
    print("\n" + "="*70)
    print("Copy the diagram above into a Mermaid-compatible renderer")
    print("See https://mermaid.live/ or use pymdownx.superfences in MkDocs")
    print("="*70 + "\n")


# ============================================================================
# Main
# ============================================================================

async def main():
    print("="*70)
    print("Network Protocol State Machine Demo")
    print("="*70)
    print("\nThis demo shows a realistic network protocol FSM with:")
    print("  - Connection establishment with timeout and retry")
    print("  - Authentication (pushed state)")
    print("  - Data transfer (pushed state)")
    print("  - Error handling and recovery")
    print("  - Graceful shutdown")

    # Show the diagram first
    await demo_mermaid_diagram()

    # Run demos
    # Uncomment the demos you want to see:

    await demo_successful_connection()
    # await demo_connection_timeout()
    # await demo_data_transfer_error()

    print("\n" + "="*70)
    print("All demos complete!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
