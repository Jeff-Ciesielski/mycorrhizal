# Circuit Breaker Example - Implementation Summary

## Overview

Created a comprehensive, production-quality example demonstrating **FSM-in-PN integration** using the circuit breaker pattern - a classic distributed systems resilience pattern.

## Files Created

1. **`examples/mycelium/circuit_breaker_handler.py`** (644 lines)
   - Complete, runnable example
   - Well-documented with clear comments
   - Demonstrates FSM logic + Petri net orchestration

2. **`examples/mycelium/README_circuit_breaker.md`**
   - Comprehensive documentation
   - Explains the circuit breaker pattern
   - Shows real-world use cases
   - Provides learning resources

## What the Example Shows

### FSM States (Septum)

Three FSM states modeling circuit breaker behavior:

1. **`CircuitBreakerClosed`** - Normal operation
   - Requests allowed through
   - Track failures/successes
   - Open circuit when threshold reached

2. **`CircuitBreakerOpen`** - Circuit is open
   - All requests rejected
   - Timeout-based recovery testing
   - Protects backend from load

3. **`CircuitBreakerHalfOpen`** - Testing recovery
   - Limited test requests
   - Close circuit if successful
   - Reopen if any failure

### Petri Net (Hypha)

Orchestrates request flow through FSM states:

```
input_queue → validate_request → validated_requests
                                           ↓
                              route_through_circuit_breaker
                                           ↓
         ┌─────────────────┬──────────────┴───────────────┐
         ↓                 ↓                              ↓
success_responses   failed_requests        circuit_open_rejections
```

## Key Features

### Real-World Relevance

- **Netflix Hystrix**: Same pattern used for microservice resilience
- **AWS Resilience Hub**: Automated failure isolation
- **Kubernetes Service Mesh**: Circuit breaking for service-to-service
- **API Gateways**: Kong, APISIX use this pattern

### Engineering Best Practices

1. **Separation of Concerns**
   - FSM manages state logic (circuit health)
   - Petri net manages workflow (request routing)

2. **Explicit State**
   - Circuit states are explicit (CLOSED, OPEN, HALF_OPEN)
   - Transitions are clearly defined
   - No implicit state hidden in conditionals

3. **Observable**
   - Clear logging of state transitions
   - Statistics tracking (failures, successes, rejections)
   - Multiple output queues for different outcomes

4. **Resilient**
   - Graceful degradation (circuit opens on failures)
   - Automatic recovery (HALF_OPEN testing)
   - Protects backend from overload

## Technical Implementation

### FSM-in-PN Concept

The example demonstrates **how FSMs can integrate with Petri nets**:

```python
# FSM defines state behavior
@septum.state()
def CircuitBreakerClosed():
    @septum.on_state
    async def on_state(ctx):
        # Handle events in CLOSED state
        if failure_threshold_reached:
            return Events.OPEN_CIRCUIT

# PN transition applies FSM logic to tokens
@builder.transition()
async def route_through_circuit_breaker(consumed, bb, timebase):
    current_state = bb.circuit_state

    if current_state == CircuitBreakerState.OPEN:
        # Reject immediately
        yield {circuit_open_rejections: response}
    else:
        # Try processing, update FSM state
        # ...
```

### Current Limitations

This example uses **manual FSM integration** (FSM state tracked in blackboard, PN transition applies FSM rules).

**Future**: True FSM-in-PN integration like BT-in-PN:

```python
# Future API (not yet implemented)
@builder.transition(fsm=CircuitBreakerFSM, outputs=[...])
async def route(consumed, bb, timebase):
    pass  # FSM auto-ticks, state persists across requests
```

## Demo Execution

The example runs through 5 phases:

1. **Normal Operation** - Requests succeed in CLOSED state
2. **Service Degradation** - Failures trigger circuit to OPEN
3. **Circuit OPEN** - Requests rejected immediately
4. **Recovery Testing** - Timeout to HALF_OPEN
5. **Final State** - Circuit remains OPEN (test failed)

## Testing

Verified:
- ✅ Example runs successfully
- ✅ Circuit breaker state transitions work correctly
- ✅ Requests routed to appropriate outputs
- ✅ All existing tests still pass
- ✅ Clear, informative output

## Code Quality

- **Type hints**: Full type annotations
- **Documentation**: Extensive docstrings
- **Comments**: Explains "why" not just "what"
- **Error handling**: Proper exception handling
- **Logging**: Clear, structured log messages
- **Production-ready**: Follows engineering best practices

## Educational Value

This example teaches:

1. **Circuit Breaker Pattern**
   - When and why to use it
   - How to implement it correctly
   - Real-world applications

2. **FSM Design**
   - Modeling stateful behavior
   - Explicit state transitions
   - Time-based transitions (timeouts)

3. **Petri Net Workflow**
   - Declarative request routing
   - Token flow visualization
   - Parallel processing

4. **Integration Patterns**
   - How FSMs and PNs complement each other
   - Separation of state vs. workflow
   - Future of FSM-in-PN in Mycelium

## Next Steps

To extend this example:

1. **Add true FSM-in-PN integration** - Implement `@builder.transition(fsm=...)`
2. **Multiple circuit breakers** - One per backend service
3. **Metrics collection** - Track state transitions, failure rates
4. **Distributed state** - Share circuit state across instances
5. **Configuration** - Make thresholds configurable per service

## References

- Example: `examples/mycelium/circuit_breaker_handler.py`
- Documentation: `examples/mycelium/README_circuit_breaker.md`
- BT-in-PN: `examples/mycelium/job_queue_processor.py`
- FSM-in-BT: `examples/mycelium/robot_controller.py`
