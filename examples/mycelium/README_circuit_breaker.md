# Circuit Breaker Handler - FSM-in-PN Integration

This example demonstrates a **circuit breaker pattern** implementation combining Finite State Machines (FSMs) with Petri Nets for robust API request handling.

## What is a Circuit Breaker?

The circuit breaker pattern is a **distributed systems resilience pattern** used to:

- **Prevent cascading failures**: Stop calling a failing service before it brings down your entire system
- **Enable automatic recovery**: Test if the service has recovered without overwhelming it
- **Protect backend services**: Give failing services time to recover instead of hammering them with requests

## Real-World Use Cases

This pattern is used in production by:

- **Netflix Hystrix**: Circuit breakers for microservice communication
- **AWS Resilience Hub**: Automated failure isolation and recovery
- **Kubernetes Service Mesh** (Istio, Linkerd): Circuit breaking for service-to-service calls
- **API Gateways** (Kong, APISIX): Protecting backend services from overload

## Architecture

```
Incoming Requests
       ↓
   [Validator] ← Simple validation filter
       ↓
[Circuit Breaker FSM] ← State machine tracking service health
   ↓     ↓     ↓
Success  Failure  Rejected
```

### Circuit Breaker States

**CLOSED** (Normal Operation)
- Requests pass through to backend service
- Track failures and successes
- Transition to OPEN when failure_threshold is reached

**OPEN** (Circuit is Open)
- All requests rejected immediately
- Protects backend from further load
- After timeout, transition to HALF_OPEN to test recovery

**HALF_OPEN** (Testing Recovery)
- Limited requests allowed to test backend
- If success_threshold reached → CLOSED (service recovered)
- If any request fails → OPEN (service still failing)

## How It Works

### FSM (Finite State Machine)

The FSM defines the circuit breaker states and transitions:

```python
@septum.state()
def CircuitBreakerClosed():
    """Normal operation - requests allowed"""

@septum.state(config=StateConfiguration(timeout=5.0))
def CircuitBreakerOpen():
    """Circuit open - requests rejected"""

@septum.state(config=StateConfiguration(timeout=5.0))
def CircuitBreakerHalfOpen():
    """Testing recovery - limited requests"""
```

### Petri Net (Orchestration)

The Petri net orchestrates request flow:

```python
@pn.net
def CircuitBreakerOrchestrator(builder):
    # Input queue
    input_queue = builder.place("input_queue", type=PlaceType.QUEUE)

    # Validation transition
    @builder.transition()
    async def validate_request(consumed, bb, timebase):
        # Filter invalid requests

    # Circuit breaker transition
    @builder.transition()
    async def route_through_circuit_breaker(consumed, bb, timebase):
        # Apply FSM state to each request
        # Route to: success, failed, or rejected
```

## Running the Example

```bash
cd /home/jeff/workspace/mycorrhizal
uv run python examples/mycelium/circuit_breaker_handler.py
```

### Output Phases

**Phase 1: Normal Operation (CLOSED)**
- Requests 1-3 succeed
- Circuit breaker allows all requests through

**Phase 2: Service Degradation**
- Requests 4-6 fail (simulating backend failure)
- After 3 failures, circuit opens
- All subsequent requests rejected

**Phase 3: Circuit OPEN**
- Requests 7-8 rejected immediately
- Circuit protects backend from load

**Phase 4: Recovery Testing**
- Timeout expires, circuit transitions to HALF_OPEN
- Test request (req-7) fails
- Circuit reopens immediately

**Phase 5: Final State**
- Circuit remains OPEN
- Requests 9-10 rejected
- System protected from cascading failure

## Key Concepts

### FSM-in-PN Integration

This example demonstrates the **concept** of FSM-in-PN integration:

1. **FSM defines state logic**: CircuitBreakerClosed/Open/HalfOpen states define behavior
2. **PN orchestrates workflow**: Petri net routes requests through FSM states
3. **Shared state**: Blackboard tracks circuit state across requests

### Future: True FSM-in-PN

In future Mycelium versions, this will use true FSM-in-PN integration:

```python
# Future API (not yet implemented)
@builder.transition(fsm=CircuitBreakerFSM, outputs=[...])
async def route_through_circuit_breaker(consumed, bb, timebase):
    # FSM auto-ticks for each request
    # FSM state persists across requests
    pass
```

This will mirror the existing BT-in-PN pattern:

```python
# Current BT-in-PN API
@builder.transition(bt=MyBehaviorTree, bt_mode="token", outputs=[...])
async def process(consumed, bb, timebase):
    pass  # BT handles all logic
```

## Why FSM + Petri Net?

### FSM Advantages

- **Stateful logic**: Track service health across multiple requests
- **Explicit transitions**: CLEAR rules for when to change states
- **Time-based transitions**: Timeouts for automatic recovery testing
- **Easy to test**: Isolated FSM logic, deterministic state machine

### Petri Net Advantages

- **Workflow orchestration**: Declarative request routing
- **Token flow**: Visual representation of request paths
- **Concurrency**: Handle multiple requests in parallel
- **Error handling**: Separate queues for success, failure, retry

### Combined Benefits

- **Separation of concerns**: FSM = state logic, PN = workflow
- **Composability**: Multiple FSMs can coexist in same PN
- **Observability**: Clear visualizations of both FSM states and PN flow
- **Maintainability**: Each component has single responsibility

## Learning Resources

- **Petri Nets**: `examples/hypha_demo.py`
- **Behavior Trees**: `examples/rhizomorph_example.py`
- **FSMs**: `examples/enoki_decorator_basic.py`
- **BT-in-PN**: `examples/mycelium/job_queue_processor.py`
- **FSM-in-BT**: `examples/mycelium/robot_controller.py`

## Production Considerations

In a production system, you would:

1. **Persist FSM state**: Store circuit state in Redis/database
2. **Distributed coordination**: Share circuit state across instances
3. **Metrics and monitoring**: Track circuit state transitions
4. **Configurable thresholds**: Adjust failure/success thresholds per service
5. **Multiple circuit breakers**: One per backend service
6. **Alerting**: Notify when circuits open/close

## Related Patterns

- **Retry Pattern**: Combine with circuit breaker for transient failures
- **Bulkhead Pattern**: Isolate different parts of the system
- **Timeout Pattern**: Prevent hanging requests
- **Fallback Pattern**: Provide degraded service when circuit is open
