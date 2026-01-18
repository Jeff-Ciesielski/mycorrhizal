#!/usr/bin/env python3
"""
Simple FSM-in-PN Example with Mermaid Diagram Generation

This example demonstrates the new FSM-in-PN integration feature
that allows Finite State Machines to be embedded in Petri net transitions,
with automatic Mermaid diagram generation showing the FSM state names.

Concept:
    - Input tokens flow through a Petri net
    - A transition with an FSM processes each token
    - The FSM state is shown in the Mermaid diagram: "FSM: StateName"
    - Users can visualize which transitions have FSMs embedded

This is a simplified example showing the diagram generation feature.
"""

import asyncio
from enum import Enum, auto
from pydantic import BaseModel

from mycorrhizal.mycelium import pn, PNRunner, PlaceType
from mycorrhizal.septum.core import septum, LabeledTransition, StateConfiguration
from mycorrhizal.common.timebase import MonotonicClock


# ==================================================================================
# Domain Models
# ==================================================================================


class Request(BaseModel):
    """A simple request to process."""
    request_id: str
    value: int


class Blackboard(BaseModel):
    """Shared state for the system."""
    model_config = {"extra": "allow"}
    requests_processed: int = 0
    failures: int = 0


# ==================================================================================
# FSM Definition
# ==================================================================================


@septum.state()
def Validating():
    """
    FSM state for validating requests.

    This is the initial state where we validate incoming requests.
    """
    @septum.events
    class Events(Enum):
        VALID = auto()     # Request is valid, proceed to processing
        INVALID = auto()   # Request is invalid, reject

    @septum.on_state
    async def on_state(ctx):
        bb = ctx.common
        # Simple validation: value must be positive
        if bb.current_request.value > 0:
            return Validating.Events.VALID
        return Validating.Events.INVALID

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.VALID, Processing),
            LabeledTransition(Events.INVALID, Rejected),
        ]


@septum.state()
def Processing():
    """
    FSM state for processing validated requests.

    This state processes the request and determines success or failure.
    """
    @septum.events
    class Events(Enum):
        SUCCESS = auto()
        FAILURE = auto()

    @septum.on_state
    async def on_state(ctx):
        bb = ctx.common
        # Simple logic: succeed if value > 0, fail otherwise
        if bb.current_request.value > 0:
            return Processing.Events.SUCCESS
        return Processing.Events.FAILURE

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.SUCCESS, Completed),
            LabeledTransition(Events.FAILURE, Failed),
        ]


@septum.state()
def Completed():
    """
    FSM state for completed requests.

    Terminal state - request was successfully processed.
    """
    @septum.events
    class Events(Enum):
        DONE = auto()

    @septum.on_state
    async def on_state(ctx):
        return Completed.Events.DONE

    @septum.transitions
    def transitions():
        return []


@septum.state()
def Failed():
    """
    FSM state for failed requests.

    Terminal state - request processing failed.
    """
    @septum.events
    class Events(Enum):
        DONE = auto()

    @septum.on_state
    async def on_state(ctx):
        return Failed.Events.DONE

    @septum.transitions
    def transitions():
        return []


@septum.state()
def Rejected():
    """
    FSM state for rejected requests.

    Terminal state - request was rejected during validation.
    """
    @septum.events
    class Events(Enum):
        DONE = auto()

    @septum.on_state
    async def on_state(ctx):
        return Rejected.Events.DONE

    @septum.transitions
    def transitions():
        return []


# ==================================================================================
# Petri Net Definition
# ==================================================================================


@pn.net
def SimpleFSMNet(builder):
    """
    Petri net with FSM-in-PN integration.

    The transition 'process_with_fsm' has an FSM embedded,
    which will be shown in the Mermaid diagram as a subgraph
    containing all FSM states and transitions.
    """

    # Places
    input_q = builder.place("input_q", type=PlaceType.QUEUE)
    success_q = builder.place("success_q", type=PlaceType.QUEUE)
    failure_q = builder.place("failure_q", type=PlaceType.QUEUE)

    # Transition with FSM integration
    # Note: The fsm parameter takes the initial state of the FSM
    @builder.transition(fsm=Validating, outputs=[success_q, failure_q])
    async def process_with_fsm(consumed, bb, timebase):
        """
        Process requests using an FSM.

        The FSM (starting with Validating state) will be executed for each token.
        The FSM has multiple states: Validating -> Processing -> [Completed|Failed|Rejected]

        Future versions will support true FSM-in-PN execution similar to BT-in-PN.
        """
        # For now, manually implement the FSM logic
        # In production, this will be replaced by automatic FSM execution
        for request in consumed:
            bb.current_request = request

            # Simulate FSM state execution
            if request.value > 0:
                bb.requests_processed += 1
                print(f"[FSM] Request {request.request_id}: SUCCESS (value={request.value})")
                yield {success_q: request}
            else:
                bb.failures += 1
                print(f"[FSM] Request {request.request_id}: FAILURE (value={request.value})")
                yield {failure_q: request}

    # Wire the net
    builder.arc(input_q, process_with_fsm)
    builder.arc(process_with_fsm, success_q)
    builder.arc(process_with_fsm, failure_q)


# ==================================================================================
# Demo Execution
# ==================================================================================


async def main():
    """Run the FSM-in-PN demo."""

    print("=" * 80)
    print("Simple FSM-in-PN Example with Mermaid Diagram Generation")
    print("=" * 80)
    print()
    print("This demo shows:")
    print("  - FSM-in-PN integration: FSMs embedded in Petri net transitions")
    print("  - Mermaid diagram generation: Full FSM state diagram embedded in transition")
    print("  - Clear visualization: All FSM states and transitions shown as subgraph")
    print()

    # Create blackboard
    bb = Blackboard()
    timebase = MonotonicClock()

    # Create runner
    runner = PNRunner(SimpleFSMNet, bb)

    # Generate and display Mermaid diagram
    print("\n" + "=" * 80)
    print("Mermaid Diagram")
    print("=" * 80)
    mermaid = runner.to_mermaid()
    print(mermaid)
    print("=" * 80)
    print()
    print("Note that the FSM is embedded as a subgraph within the transition!")
    print("The subgraph shows all 5 FSM states and their transitions:")
    print("  - Validating → Processing → Completed (success path)")
    print("  - Validating → Rejected (validation failure)")
    print("  - Processing → Failed (processing failure)")
    print()

    # Start the runner
    await runner.start(timebase)

    # Get runtime to access places
    runtime = runner.runtime

    # Find input queue
    if runtime.places is None:
        print("ERROR: Runtime places is None!")
        return

    input_place = runtime.places.get(("SimpleFSMNet", "input_q"))
    if not input_place:
        print("ERROR: Could not find input queue!")
        return

    # Submit requests
    print("Submitting requests...")
    requests = [
        Request(request_id="req-1", value=10),
        Request(request_id="req-2", value=-5),  # Will fail
        Request(request_id="req-3", value=20),
        Request(request_id="req-4", value=0),   # Will fail
        Request(request_id="req-5", value=30),
    ]

    for req in requests:
        input_place.add_token(req)
        print(f"  Submitted: {req.request_id} (value={req.value})")

    print()
    print("Processing requests...")
    print("-" * 80)

    # Let the system process
    await asyncio.sleep(1.0)

    print("-" * 80)
    print()
    print("Final state:")

    # Print queue states
    if runtime.places is None:
        print("ERROR: Runtime places is None!")
        return

    success_place = runtime.places.get(("SimpleFSMNet", "success_q"))
    failure_place = runtime.places.get(("SimpleFSMNet", "failure_q"))

    if success_place is None or failure_place is None:
        print("ERROR: Could not find output queues!")
        return

    print(f"\nSuccess queue: {len(success_place.tokens)} requests")
    for req in success_place.tokens:
        print(f"  {req.request_id} (value={req.value})")

    print(f"\nFailure queue: {len(failure_place.tokens)} requests")
    for req in failure_place.tokens:
        print(f"  {req.request_id} (value={req.value})")

    print(f"\nSystem statistics:")
    print(f"  Requests processed: {bb.requests_processed}")
    print(f"  Failures: {bb.failures}")

    print()
    print("Stopping...")
    await runner.stop(timeout=1)

    print()
    print("=" * 80)
    print("Demo complete!")
    print("=" * 80)
    print()
    print("Key takeaways:")
    print("  1. FSM-in-PN integration allows embedding FSMs in Petri net transitions")
    print("  2. Mermaid diagrams show the complete FSM state machine as a subgraph")
    print("  3. All FSM states and transitions are visible within the transition node")
    print("  4. This makes it easy to visualize complex systems with nested state machines")
    print()
    print("The Mermaid diagram shows:")
    print("  - Petri net structure (places and transitions)")
    print("  - FSM embedded in the 'process_with_fsm' transition")
    print("  - All 5 FSM states: Validating, Processing, Completed, Failed, Rejected")
    print("  - State transitions: VALID, INVALID, SUCCESS, FAILURE")
    print("  - Start node pointing to initial state (Validating)")
    print()


if __name__ == "__main__":
    asyncio.run(main())
