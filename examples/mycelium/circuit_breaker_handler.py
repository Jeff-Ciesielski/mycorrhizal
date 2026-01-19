#!/usr/bin/env python3
"""
Circuit Breaker with Request Processing - FSM-in-PN Integration Example

This example demonstrates an API request handler that combines
Finite State Machines (FSMs) with Petri nets for circuit breaker pattern implementation.

Real-world use case:
- API gateway or service proxy protecting backend services from cascading failures
- Circuit breaker FSM tracks service health (CLOSED, OPEN, HALF_OPEN)
- Request validation FSM checks and enriches incoming requests
- Petri net orchestrates request flow through FSMs with proper error handling
- Dead letter queue captures failed requests for later analysis

The FSM-in-PN integration allows us to:
1. Use FSMs for stateful logic (circuit breaker state, request validation)
2. Use Petri nets for workflow orchestration (token flow, parallel processing)
3. Handle failures gracefully without losing requests
4. Keep FSMs focused and testable (single responsibility)

Architecture:
    Input Requests → [Circuit Breaker FSM] → [Validator FSM] → [Success|Fail|Retry]
                                                ↓
                                          Circuit Open Rejections

This is a classic distributed systems pattern used in:
- Netflix Hystrix
- AWS Resilience Hub
- Kubernetes Service Mesh
- Enterprise API gateways
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
from pydantic import BaseModel, ConfigDict

from mycorrhizal.mycelium import (
    pn, PNRunner, PlaceType,
)
from mycorrhizal.septum.core import (
    septum, LabeledTransition, StateConfiguration,
)
from mycorrhizal.common.timebase import MonotonicClock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ==================================================================================
# Domain Models
# ==================================================================================


class CircuitBreakerState(Enum):
    """States of the circuit breaker FSM."""
    CLOSED = auto()      # Normal operation, requests pass through
    OPEN = auto()        # Circuit is open, requests are rejected
    HALF_OPEN = auto()   # Testing if service has recovered


class RequestPriority(Enum):
    """Request priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3


@dataclass
class APIRequest:
    """An incoming API request."""
    request_id: str
    endpoint: str
    priority: RequestPriority
    payload: dict
    timestamp: float = field(default_factory=time.time)

    def __repr__(self):
        return f"APIRequest({self.request_id}, {self.endpoint})"


@dataclass
class APIResponse:
    """Response from circuit breaker processing."""
    request: APIRequest
    status: str  # "success", "rejected", "failed", "retry"
    message: str = ""
    circuit_state: Optional[CircuitBreakerState] = None


class Blackboard(BaseModel):
    """Shared state for the circuit breaker system."""
    model_config = ConfigDict(extra='allow')  # Allow pn_ctx to be injected

    # Circuit breaker configuration
    failure_threshold: int = 5          # Open circuit after N failures
    success_threshold: int = 3          # Close circuit after N successes
    timeout: float = 10.0               # Half-open timeout (seconds)

    # Circuit breaker state tracking
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    circuit_state: CircuitBreakerState = CircuitBreakerState.CLOSED

    # Statistics
    requests_processed: int = 0
    requests_rejected: int = 0
    requests_failed: int = 0
    requests_succeeded: int = 0


# ==================================================================================
# Circuit Breaker FSM
# ==================================================================================


@septum.state()
def CircuitBreakerClosed():
    """
    Circuit breaker is CLOSED - normal operation.

    In this state:
    - Requests are allowed to pass through
    - We track failures and successes
    - Transition to OPEN when failure threshold is reached
    """

    @septum.events
    class Events(Enum):
        FAILURE = auto()   # Request failed, increment failure count
        SUCCESS = auto()   # Request succeeded, may reset failures
        OPEN_CIRCUIT = auto()  # Failure threshold reached, open circuit

    @septum.on_state
    async def on_state(ctx):
        """Handle events in CLOSED state."""
        bb = ctx.common

        # Access Events through the state function's nested class
        Events = CircuitBreakerClosed.Events
        if ctx.msg == Events.FAILURE:
            bb.failure_count += 1
            bb.requests_failed += 1
            logger.info(f"[CircuitBreaker-CLOSED] Failure #{bb.failure_count}/{bb.failure_threshold}")

            # Check if we should open the circuit
            if bb.failure_count >= bb.failure_threshold:
                bb.last_failure_time = time.time()
                logger.warning("[CircuitBreaker] FAILURE_THRESHOLD REACHED: Opening circuit")
                return Events.OPEN_CIRCUIT

        elif ctx.msg == Events.SUCCESS:
            bb.failure_count = 0  # Reset failure count on success
            bb.requests_succeeded += 1
            logger.info("[CircuitBreaker-CLOSED] Success, failures reset to 0")

        return None  # Stay in CLOSED state

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.OPEN_CIRCUIT, CircuitBreakerOpen),
        ]


@septum.state(config=StateConfiguration(timeout=5.0))
def CircuitBreakerOpen():
    """
    Circuit breaker is OPEN - blocking requests.

    In this state:
    - All requests are rejected immediately
    - After timeout, transition to HALF_OPEN to test service recovery
    - Track time since opening for timeout calculation
    """

    @septum.events
    class Events(Enum):
        TEST_RECOVERY = auto()  # Timeout elapsed, test if service recovered
        CLOSE_CIRCUIT = auto()  # Service recovered, close circuit immediately

    @septum.on_state
    async def on_state(ctx):
        """Handle events in OPEN state."""
        bb = ctx.common

        # Access Events through the state function's nested class
        Events = CircuitBreakerOpen.Events
        if ctx.msg == Events.TEST_RECOVERY:
            logger.info("[CircuitBreaker-OPEN] Testing recovery, moving to HALF_OPEN")
            return Events.TEST_RECOVERY

        return None

    @septum.on_timeout
    async def on_timeout(ctx):
        """Timeout reached, test if service has recovered."""
        bb = ctx.common
        logger.info("[CircuitBreaker-OPEN] Timeout reached, testing recovery")
        Events = CircuitBreakerOpen.Events
        return Events.TEST_RECOVERY

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.TEST_RECOVERY, CircuitBreakerHalfOpen),
        ]


@septum.state(config=StateConfiguration(timeout=5.0))
def CircuitBreakerHalfOpen():
    """
    Circuit breaker is HALF_OPEN - testing service recovery.

    In this state:
    - Allow limited requests through to test service
    - If enough requests succeed, close the circuit
    - If any request fails, open the circuit again
    - This prevents thundering herd problem
    """

    @septum.events
    class Events(Enum):
        SUCCESS = auto()   # Test request succeeded
        FAILURE = auto()   # Test request failed, reopen circuit
        CLOSE_CIRCUIT = auto()  # Success threshold reached, close circuit
        REOPEN_CIRCUIT = auto()  # Test failed, reopen circuit

    @septum.on_state
    async def on_state(ctx):
        """Handle events in HALF_OPEN state."""
        bb = ctx.common

        # Access Events through the state function's nested class
        Events = CircuitBreakerHalfOpen.Events
        if ctx.msg == Events.FAILURE:
            bb.failure_count = 0
            bb.requests_failed += 1
            logger.warning("[CircuitBreaker-HALF_OPEN] Test request FAILED, reopening circuit")
            return Events.REOPEN_CIRCUIT

        elif ctx.msg == Events.SUCCESS:
            bb.success_count += 1
            bb.requests_succeeded += 1
            logger.info(f"[CircuitBreaker-HALF_OPEN] Test request succeeded ({bb.success_count}/{bb.success_threshold})")

            # Check if we should close the circuit
            if bb.success_count >= bb.success_threshold:
                logger.info("[CircuitBreaker] SUCCESS_THRESHOLD REACHED: Closing circuit")
                return Events.CLOSE_CIRCUIT

        return None

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.CLOSE_CIRCUIT, CircuitBreakerClosed),
            LabeledTransition(Events.REOPEN_CIRCUIT, CircuitBreakerOpen),
        ]


# NOTE: For this simplified example, we're using inline validation logic
# in the Petri net transition instead of a separate Validator FSM.
# In a production system, you could create a full FSM for validation
# with multiple states (ValidateRequest, RequestValid, RequestRejected).


# ==================================================================================
# Petri Net with FSM-in-PN Integration
# ==================================================================================


@pn.net
def CircuitBreakerOrchestrator(builder):
    """
    Petri net that orchestrates API requests through FSMs.

    The PN structure:
    1. Input queue receives API requests
    2. Validator FSM checks and filters requests
    3. Circuit breaker FSM protects backend service
    4. Output queues for success, failure, retry

    The FSMs handle all stateful logic:
    - Circuit breaker tracks service health over time
    - Validator enforces request quality rules

    The PN handles workflow:
    - Token flow between stages
    - Parallel processing
    - Error handling and routing
    """

    # Places
    input_queue = builder.place("input_queue", type=PlaceType.QUEUE)
    validated_requests = builder.place("validated_requests", type=PlaceType.QUEUE)
    rejected_requests = builder.place("rejected_requests", type=PlaceType.QUEUE)
    success_responses = builder.place("success_responses", type=PlaceType.QUEUE)
    failed_requests = builder.place("failed_requests", type=PlaceType.QUEUE)
    circuit_open_rejections = builder.place("circuit_open_rejections", type=PlaceType.QUEUE)

    # Transition 1: Request validation (simple filter, no FSM)
    @builder.transition(outputs=[validated_requests, rejected_requests])
    async def validate_request(consumed, bb, timebase):
        """Validate requests before passing to circuit breaker."""
        for request in consumed:
            bb.requests_processed += 1

            # Simple validation: reject low priority unless explicitly allowed
            if request.priority == RequestPriority.LOW and not bb.model_extra.get("allow_low_priority", False):
                logger.warning(f"[Validator] Rejecting low-priority request {request.request_id}")
                yield {rejected_requests: request}
            else:
                logger.info(f"[Validator] Request {request.request_id} validated")
                yield {validated_requests: request}

    # Transition 2: Circuit breaker state machine (FSM-in-PN)
    # This transition runs the circuit breaker FSM for each request
    # Note: fsm=CircuitBreakerClosed registers the FSM for diagram visualization
    # The actual FSM logic is implemented manually below ( FSM-in-PN execution
    # not yet implemented, but visualization works)
    @builder.transition(fsm=CircuitBreakerClosed, outputs=[success_responses, failed_requests, circuit_open_rejections])
    async def route_through_circuit_breaker(consumed, bb, timebase):
        """
        Route requests through circuit breaker FSM.

        This demonstrates FSM-in-PN integration:
        - FSM states (CLOSED, OPEN, HALF_OPEN) are defined using @septum.state()
        - The fsm=CircuitBreakerClosed parameter registers the FSM for visualization
        - The Mermaid diagram will show the full FSM embedded in the transition
        - FSM state is tracked in bb.circuit_state (shared state)
        - FSM transitions are implemented manually below (execution not yet automated)

        The FSM-in-PN integration provides:
        1. Clear visualization: FSM states shown as subgraph in Mermaid diagram
        2. Structured definition: FSM states defined declaratively
        3. Manual execution: Transition implements FSM logic (future: automated)

        This is similar to how early BT-in-PN worked: visualization first, execution later.
        """
        import random

        # Simulate backend degradation: first 3 requests in phase 2 will fail
        # This is controlled state for demo purposes
        phase2_failing_requests = {"req-4", "req-5", "req-6"}

        for request in consumed:
            logger.info(f"[CircuitBreaker] Processing request {request.request_id}")

            # Check current circuit state
            current_state = bb.circuit_state

            if current_state == CircuitBreakerState.OPEN:
                # Circuit is open, reject immediately
                bb.requests_rejected += 1
                response = APIResponse(
                    request=request,
                    status="rejected",
                    message="Circuit breaker is OPEN, rejecting request",
                    circuit_state=current_state,
                )
                logger.warning(f"[CircuitBreaker] OPEN: Rejecting {request.request_id}")
                yield {circuit_open_rejections: response}

            elif current_state in [CircuitBreakerState.CLOSED, CircuitBreakerState.HALF_OPEN]:
                # Circuit is closed or half-open, try to process request
                # Simulate backend service call (controlled for demo)
                will_fail = request.request_id in phase2_failing_requests
                success_rate = 0.9 if current_state == CircuitBreakerState.CLOSED else 0.7

                # Add some randomness for more interesting behavior
                if not will_fail and random.random() < success_rate:
                    # Request succeeded
                    bb.requests_succeeded += 1
                    response = APIResponse(
                        request=request,
                        status="success",
                        message="Request processed successfully",
                        circuit_state=current_state,
                    )
                    logger.info(f"[CircuitBreaker] {current_state.name}: Success for {request.request_id}")

                    # Update circuit state based on FSM rules
                    if current_state == CircuitBreakerState.CLOSED:
                        # Success in CLOSED state: decrement failure count
                        bb.failure_count = max(0, bb.failure_count - 1)
                    elif current_state == CircuitBreakerState.HALF_OPEN:
                        # Success in HALF_OPEN: track successes
                        bb.success_count += 1
                        if bb.success_count >= bb.success_threshold:
                            # Success threshold reached, close circuit
                            bb.circuit_state = CircuitBreakerState.CLOSED
                            bb.success_count = 0
                            logger.info("[CircuitBreaker] CLOSED: Circuit recovered!")

                    yield {success_responses: response}
                else:
                    # Request failed
                    bb.requests_failed += 1
                    bb.failure_count += 1
                    response = APIResponse(
                        request=request,
                        status="failed",
                        message="Backend service error",
                        circuit_state=current_state,
                    )
                    logger.error(f"[CircuitBreaker] {current_state.name}: Failure for {request.request_id}")

                    # Update circuit state based on FSM rules
                    if current_state == CircuitBreakerState.CLOSED:
                        # Check if we should open circuit
                        if bb.failure_count >= bb.failure_threshold:
                            bb.circuit_state = CircuitBreakerState.OPEN
                            bb.last_failure_time = time.time()
                            logger.error(f"[CircuitBreaker] OPEN: Threshold {bb.failure_threshold} reached!")

                    elif current_state == CircuitBreakerState.HALF_OPEN:
                        # In HALF_OPEN, any failure reopens circuit immediately
                        bb.circuit_state = CircuitBreakerState.OPEN
                        bb.failure_count = 0
                        logger.error("[CircuitBreaker] OPEN: Half-open test failed!")

                    yield {failed_requests: response}

    # Wire the net
    builder.arc(input_queue, validate_request)
    builder.arc(validate_request, validated_requests)
    builder.arc(validate_request, rejected_requests)

    builder.arc(validated_requests, route_through_circuit_breaker)
    builder.arc(route_through_circuit_breaker, success_responses)
    builder.arc(route_through_circuit_breaker, failed_requests)
    builder.arc(route_through_circuit_breaker, circuit_open_rejections)


# ==================================================================================
# Demo Execution
# ==================================================================================


async def main():
    """Run the circuit breaker demo."""

    print("=" * 80)
    print("Circuit Breaker - FSM-in-PN Integration Demo")
    print("=" * 80)
    print()
    print("This demo shows an example API gateway with:")
    print("  - Circuit breaker FSM tracking service health (CLOSED → OPEN → HALF_OPEN)")
    print("  - Request validation FSM filtering bad requests")
    print("  - Petri net orchestrating request flow through FSMs")
    print("  - Graceful handling of service failures with automatic recovery")
    print()
    print("Key distributed systems patterns:")
    print("  - Circuit breaker prevents cascading failures")
    print("  - State machines manage service health tracking")
    print("  - Petri nets handle workflow orchestration")
    print("  - Dead letter queue for failed requests")
    print()

    # Create blackboard with circuit breaker configuration
    bb = Blackboard(
        failure_threshold=3,  # Open circuit after 3 failures
        success_threshold=2,  # Close circuit after 2 successes
        timeout=5.0,          # Test recovery after 5 seconds
    )
    timebase = MonotonicClock()

    # Start PN runner
    runner = PNRunner(CircuitBreakerOrchestrator, bb)
    await runner.start(timebase)

    # Get runtime to access places
    runtime = runner.runtime

    # Find input queue
    input_place = None
    if runtime.places:
        for key, place in runtime.places.items():
            if "input_queue" in key:
                input_place = place
                break

    if not input_place:
        print("ERROR: Could not find input queue!")
        return

    # Submit requests (simulating real API traffic)
    print("\n" + "=" * 80)
    print("Phase 1: Normal operation (CLOSED)")
    print("=" * 80)
    print()

    requests = [
        APIRequest(request_id="req-1", endpoint="/api/users", priority=RequestPriority.HIGH,
                   payload={"action": "get_users"}),
        APIRequest(request_id="req-2", endpoint="/api/posts", priority=RequestPriority.NORMAL,
                   payload={"action": "get_posts"}),
        APIRequest(request_id="req-3", endpoint="/api/data", priority=RequestPriority.NORMAL,
                   payload={"action": "get_data"}),
    ]

    for req in requests:
        input_place.add_token(req)
        print(f"Submitted: {req}")
        await asyncio.sleep(0.5)

    await asyncio.sleep(2.0)

    print("\n" + "=" * 80)
    print("Phase 2: Service degradation (failures trigger circuit open)")
    print("=" * 80)
    print()

    # Submit requests that will fail (simulating backend failure)
    # These failures will trigger the circuit to open
    failing_requests = [
        APIRequest(request_id="req-4", endpoint="/api/fail1", priority=RequestPriority.HIGH,
                   payload={"action": "fail"}),
        APIRequest(request_id="req-5", endpoint="/api/fail2", priority=RequestPriority.HIGH,
                   payload={"action": "fail"}),
        APIRequest(request_id="req-6", endpoint="/api/fail3", priority=RequestPriority.HIGH,
                   payload={"action": "fail"}),
    ]

    for req in failing_requests:
        input_place.add_token(req)
        print(f"Submitted: {req}")
        await asyncio.sleep(0.5)

    await asyncio.sleep(2.0)

    # Check circuit state
    print(f"\nCircuit state after failures: {bb.circuit_state.name}")
    print(f"Failure count: {bb.failure_count}/{bb.failure_threshold}")

    # If circuit is open, simulate timeout to HALF_OPEN transition
    if bb.circuit_state == CircuitBreakerState.OPEN:
        print("\n(Circuit timeout simulated - transitioning to HALF_OPEN)")
        bb.circuit_state = CircuitBreakerState.HALF_OPEN
        bb.failure_count = 0

    print("\n" + "=" * 80)
    print("Phase 3: Circuit OPEN (requests rejected)")
    print("=" * 80)
    print()

    # These should be rejected immediately
    more_requests = [
        APIRequest(request_id="req-7", endpoint="/api/users", priority=RequestPriority.HIGH,
                   payload={"action": "get_users"}),
        APIRequest(request_id="req-8", endpoint="/api/posts", priority=RequestPriority.NORMAL,
                   payload={"action": "get_posts"}),
    ]

    for req in more_requests:
        input_place.add_token(req)
        print(f"Submitted: {req}")
        await asyncio.sleep(0.5)

    await asyncio.sleep(2.0)

    print("\n" + "=" * 80)
    print("Phase 4: Recovery testing (waiting for timeout...)")
    print("=" * 80)
    print()

    # Wait for circuit to transition to HALF_OPEN
    print("Waiting 6 seconds for circuit timeout...")
    await asyncio.sleep(6.0)

    print(f"\nCircuit state after timeout: {bb.circuit_state.name}")

    print("\n" + "=" * 80)
    print("Phase 5: Recovery testing (HALF_OPEN)")
    print("=" * 80)
    print()

    # Submit test requests during HALF_OPEN
    test_requests = [
        APIRequest(request_id="req-9", endpoint="/api/test1", priority=RequestPriority.HIGH,
                   payload={"action": "test"}),
        APIRequest(request_id="req-10", endpoint="/api/test2", priority=RequestPriority.HIGH,
                   payload={"action": "test"}),
    ]

    for req in test_requests:
        input_place.add_token(req)
        print(f"Submitted: {req}")
        await asyncio.sleep(0.5)

    await asyncio.sleep(2.0)

    print(f"\nCircuit state after recovery test: {bb.circuit_state.name}")
    print(f"Success count in half-open: {bb.success_count}/{bb.success_threshold}")

    print("\n" + "=" * 80)
    print("Final Statistics")
    print("=" * 80)
    print()

    # Print queue states
    print("\nQueue states:")
    if runtime.places:
        for name_parts, place in runtime.places.items():
            place_name = name_parts[-1]
            if place.tokens:
                # Convert to list to handle both deque and list
                token_list = list(place.tokens)
                print(f"\n{place_name.replace('_', ' ').title()}: {len(token_list)} items")
                for item in token_list[:5]:  # Show first 5
                    print(f"  {item}")
                if len(token_list) > 5:
                    print(f"  ... and {len(token_list) - 5} more")

    print("\nSystem statistics:")
    print(f"  Requests processed: {bb.requests_processed}")
    print(f"  Requests succeeded: {bb.requests_succeeded}")
    print(f"  Requests failed: {bb.requests_failed}")
    print(f"  Requests rejected (circuit open): {bb.requests_rejected}")
    print(f"\nFinal circuit state: {bb.circuit_state.name}")
    print(f"  Failure count: {bb.failure_count}")
    print(f"  Success count: {bb.success_count}")
    print(f"  Last failure time: {bb.last_failure_time}")

    print("\nStopping circuit breaker...")
    await runner.stop(timeout=1)

    print()
    print("=" * 80)
    print("Demo complete!")
    print("=" * 80)
    print()
    print("Key takeaways:")
    print("  1. Circuit breaker FSM tracks service health across multiple requests")
    print("  2. State transitions prevent cascading failures")
    print("  3. Automatic recovery through HALF_OPEN testing")
    print("  4. Petri net orchestrates request flow through FSM states")
    print("  5. FSMs manage stateful logic, PN manages workflow")
    print()
    print("This pattern is used in:")
    print("  - Netflix Hystrix")
    print("  - AWS Resilience Hub")
    print("  - Kubernetes Service Mesh (Istio, Linkerd)")
    print("  - Enterprise API gateways (Kong, APISIX)")
    print()
    print("=" * 80)
    print("Circuit Breaker Architecture Diagram")
    print("=" * 80)
    print()
    print("Generating Mermaid diagram of the Petri net structure...")
    print()

    # Generate and display the Mermaid diagram
    diagram = runner.to_mermaid()
    print(diagram)
    print()
    print("Diagram explanation:")
    print("  - Places (circles) hold requests at different stages")
    print("  - Transitions (rectangles) process requests and move them between places")
    print("  - The circuit breaker FSM state is managed in the blackboard")
    print("  - Requests flow from input_queue through validation and circuit breaking")
    print()
    print("=" * 80)
    print("Circuit Breaker FSM State Diagram")
    print("=" * 80)
    print()
    print("The FSM defined in this example has these states:")
    print()
    print("  CLOSED (normal operation)")
    print("    - Requests allowed through")
    print("    - Track failures and successes")
    print("    - Transition to OPEN when failure_threshold reached")
    print()
    print("  OPEN (circuit is open)")
    print("    - All requests rejected immediately")
    print("    - After timeout, transition to HALF_OPEN")
    print("    - Protects backend from further load")
    print()
    print("  HALF_OPEN (testing recovery)")
    print("    - Limited requests allowed to test backend")
    print("    - If success_threshold reached → CLOSED")
    print("    - If any failure → OPEN")
    print()
    print("This example demonstrates FSM-in-PN integration:")
    print("  - FSM states are defined declaratively using @septum.state()")
    print("  - The fsm= parameter registers FSM for diagram visualization")
    print("  - Mermaid diagrams show the FSM embedded in the transition")
    print("  - FSM execution is currently manual (future: automated like BT-in-PN)")
    print()


if __name__ == "__main__":
    asyncio.run(main())
