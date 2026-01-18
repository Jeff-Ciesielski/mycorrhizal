"""
Throughput benchmarks for Septum (Finite state machine DSL)

Measures state transitions per second across various scenarios:
- Simple two-state machines
- Multi-state linear chains
- Message passing scenarios
- State machines with timeouts
"""

import asyncio
import pytest
from enum import Enum, auto

from mycorrhizal.septum.core import (
    septum,
    StateMachine,
    StateConfiguration,
    SharedContext,
    LabeledTransition,
    Again,
)


# ============================================================================
# Benchmark State Machines
# ============================================================================

@septum.state(config=StateConfiguration(can_dwell=True))
def StateA():
    """First state in simple two-state machine"""

    class Events(Enum):
        NEXT = auto()
        BACK = auto()

    @septum.on_state
    async def on_state(ctx: SharedContext):
        msg = ctx.msg
        if msg == "next":
            return Events.NEXT
        elif msg == "back":
            return Events.BACK
        return None

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.NEXT, StateB),
            LabeledTransition(Events.BACK, StateA),  # Self-loop
        ]


@septum.state(config=StateConfiguration(can_dwell=True))
def StateB():
    """Second state in simple two-state machine"""

    class Events(Enum):
        NEXT = auto()
        BACK = auto()

    @septum.on_state
    async def on_state(ctx: SharedContext):
        msg = ctx.msg
        if msg == "next":
            return Events.NEXT
        elif msg == "back":
            return Events.BACK
        return None

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.NEXT, StateB),  # Self-loop
            LabeledTransition(Events.BACK, StateA),
        ]


@septum.state(config=StateConfiguration(can_dwell=True))
def LinearState1():
    """State 1 in linear chain"""

    class Events(Enum):
        FORWARD = auto()

    @septum.on_state
    async def on_state(ctx: SharedContext):
        if ctx.msg == "forward":
            return Events.FORWARD
        return None

    @septum.transitions
    def transitions():
        return [LabeledTransition(Events.FORWARD, LinearState2)]


@septum.state(config=StateConfiguration(can_dwell=True))
def LinearState2():
    """State 2 in linear chain"""

    class Events(Enum):
        FORWARD = auto()

    @septum.on_state
    async def on_state(ctx: SharedContext):
        if ctx.msg == "forward":
            return Events.FORWARD
        return None

    @septum.transitions
    def transitions():
        return [LabeledTransition(Events.FORWARD, LinearState3)]


@septum.state(config=StateConfiguration(can_dwell=True))
def LinearState3():
    """State 3 in linear chain"""

    class Events(Enum):
        FORWARD = auto()

    @septum.on_state
    async def on_state(ctx: SharedContext):
        if ctx.msg == "forward":
            return Events.FORWARD
        return None

    @septum.transitions
    def transitions():
        return [LabeledTransition(Events.FORWARD, LinearState4)]


@septum.state(config=StateConfiguration(can_dwell=True))
def LinearState4():
    """State 4 in linear chain"""

    class Events(Enum):
        FORWARD = auto()

    @septum.on_state
    async def on_state(ctx: SharedContext):
        if ctx.msg == "forward":
            return Events.FORWARD
        return None

    @septum.transitions
    def transitions():
        return [LabeledTransition(Events.FORWARD, LinearState5)]


@septum.state()
def LinearState5():
    """State 5 in linear chain (terminal)"""

    @septum.on_state
    async def on_state(ctx: SharedContext):
        # Terminal state - no transitions
        pass


@septum.state(config=StateConfiguration(can_dwell=True))
def ProcessingState():
    """State that processes messages rapidly"""

    class Events(Enum):
        PROCESS = auto()
        DONE = auto()

    @septum.on_state
    async def on_state(ctx: SharedContext):
        msg = ctx.msg
        if msg == "process":
            return Events.PROCESS  # Self-loop with Again
        elif msg == "done":
            return Events.DONE
        return None

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.PROCESS, ProcessingState),  # Stay in same state
            LabeledTransition(Events.DONE, DoneState),
        ]


@septum.state(config=StateConfiguration(terminal=True))
def DoneState():
    """Terminal state"""

    @septum.on_state
    async def on_state(ctx: SharedContext):
        pass


@septum.state(config=StateConfiguration(timeout=0.001))
def TimeoutState1():
    """State with timeout that transitions to next state"""

    class Events(Enum):
        MANUAL = auto()

    @septum.on_state
    async def on_state(ctx: SharedContext):
        if ctx.msg == "manual":
            return Events.MANUAL
        return None

    @septum.on_timeout
    async def on_timeout(ctx: SharedContext):
        # Auto-transition on timeout
        ctx.common["timeout_count"] = ctx.common.get("timeout_count", 0) + 1
        return TimeoutState2

    @septum.transitions
    def transitions():
        return [LabeledTransition(Events.MANUAL, TimeoutState2)]


@septum.state(config=StateConfiguration(timeout=0.001))
def TimeoutState2():
    """Second state with timeout"""

    class Events(Enum):
        MANUAL = auto()

    @septum.on_state
    async def on_state(ctx: SharedContext):
        if ctx.msg == "manual":
            return Events.MANUAL
        return None

    @septum.on_timeout
    async def on_timeout(ctx: SharedContext):
        # Auto-transition on timeout
        ctx.common["timeout_count"] = ctx.common.get("timeout_count", 0) + 1
        return TimeoutState3

    @septum.transitions
    def transitions():
        return [LabeledTransition(Events.MANUAL, TimeoutState3)]


@septum.state()
def TimeoutState3():
    """Terminal state for timeout chain"""

    @septum.on_state
    async def on_state(ctx: SharedContext):
        pass


# ============================================================================
# Benchmarks
# ============================================================================

@pytest.mark.septum
@pytest.mark.benchmark(group="septum-simple")
def test_simple_two_state_transitions(benchmark):
    """Benchmark simple two-state machine

    Measures: state transitions per second between two states
    Scenario: 100 transitions back and forth between StateA and StateB
    Expected: Baseline transition throughput metric
    """
    NUM_TRANSITIONS = 100

    def run_fsm():
        async def _run():
            common_data = {"transition_count": 0}
            fsm = StateMachine(initial_state=StateA, common_data=common_data)
            await fsm.initialize()

            # Alternate between states
            for i in range(NUM_TRANSITIONS):
                if i % 2 == 0:
                    fsm.send_message("next")  # A -> B or B -> B
                else:
                    fsm.send_message("back")  # B -> A or A -> A
                await fsm.tick()

            return common_data["transition_count"]

        return asyncio.run(_run())

    result = benchmark(run_fsm)
    # Result depends on FSM implementation


@pytest.mark.septum
@pytest.mark.benchmark(group="septum-linear")
def test_linear_state_chain(benchmark):
    """Benchmark linear state chain

    Measures: state transitions per second through 5-state linear chain
    Scenario: 20 complete traversals through 5-state chain (100 transitions)
    Expected: Each traversal goes through 5 states
    """
    NUM_TRAVERSALS = 20

    def run_fsm():
        async def _run():
            common_data = {"state_visits": 0}

            @septum.state(config=StateConfiguration(can_dwell=True))
            def CountingState1():
                class Events(Enum):
                    FORWARD = auto()

                @septum.on_state
                async def on_state(ctx: SharedContext):
                    if ctx.msg == "forward":
                        ctx.common["state_visits"] += 1
                        return Events.FORWARD
                    return None

                @septum.transitions
                def transitions():
                    return [LabeledTransition(Events.FORWARD, CountingState2)]

            @septum.state(config=StateConfiguration(can_dwell=True))
            def CountingState2():
                class Events(Enum):
                    FORWARD = auto()

                @septum.on_state
                async def on_state(ctx: SharedContext):
                    if ctx.msg == "forward":
                        ctx.common["state_visits"] += 1
                        return Events.FORWARD
                    return None

                @septum.transitions
                def transitions():
                    return [LabeledTransition(Events.FORWARD, CountingState3)]

            @septum.state(config=StateConfiguration(can_dwell=True))
            def CountingState3():
                class Events(Enum):
                    FORWARD = auto()

                @septum.on_state
                async def on_state(ctx: SharedContext):
                    if ctx.msg == "forward":
                        ctx.common["state_visits"] += 1
                        return Events.FORWARD
                    return None

                @septum.transitions
                def transitions():
                    return [LabeledTransition(Events.FORWARD, CountingState4)]

            @septum.state(config=StateConfiguration(can_dwell=True))
            def CountingState4():
                class Events(Enum):
                    FORWARD = auto()

                @septum.on_state
                async def on_state(ctx: SharedContext):
                    if ctx.msg == "forward":
                        ctx.common["state_visits"] += 1
                        return Events.FORWARD
                    return None

                @septum.transitions
                def transitions():
                    return [LabeledTransition(Events.FORWARD, CountingState5)]

            @septum.state(config=StateConfiguration(can_dwell=True))
            def CountingState5():
                class Events(Enum):
                    RESTART = auto()

                @septum.on_state
                async def on_state(ctx: SharedContext):
                    ctx.common["state_visits"] += 1
                    return Events.RESTART

                @septum.transitions
                def transitions():
                    return [LabeledTransition(Events.RESTART, CountingState1)]

            fsm = StateMachine(initial_state=CountingState1, common_data=common_data)
            await fsm.initialize()

            # Run through the chain multiple times
            for _ in range(NUM_TRAVERSALS):
                fsm.send_message("forward")
                await fsm.tick()
                fsm.send_message("forward")
                await fsm.tick()
                fsm.send_message("forward")
                await fsm.tick()
                fsm.send_message("forward")
                await fsm.tick()
                fsm.send_message("forward")
                await fsm.tick()

            return common_data["state_visits"]

        return asyncio.run(_run())

    result = benchmark(run_fsm)
    assert result == NUM_TRAVERSALS * 5  # 5 states visited per traversal


@pytest.mark.septum
@pytest.mark.benchmark(group="septum-message")
def test_high_frequency_message_passing(benchmark):
    """Benchmark high-frequency message processing

    Measures: messages processed per second in single state
    Scenario: 100 messages processed with self-transitions
    Expected: High message processing throughput
    """
    NUM_MESSAGES = 100

    def run_fsm():
        async def _run():
            common_data = {"processed": 0}

            @septum.state(config=StateConfiguration(can_dwell=True))
            def MessageProcessor():
                class Events(Enum):
                    PROCESS = auto()

                @septum.on_state
                async def on_state(ctx: SharedContext):
                    if ctx.msg == "process":
                        ctx.common["processed"] += 1
                        return Events.PROCESS
                    return None

                @septum.transitions
                def transitions():
                    return [LabeledTransition(Events.PROCESS, MessageProcessor)]

            fsm = StateMachine(initial_state=MessageProcessor, common_data=common_data)
            await fsm.initialize()

            # Send all messages
            for _ in range(NUM_MESSAGES):
                fsm.send_message("process")
                await fsm.tick()

            return common_data["processed"]

        return asyncio.run(_run())

    result = benchmark(run_fsm)
    assert result == NUM_MESSAGES


@pytest.mark.septum
@pytest.mark.benchmark(group="septum-complex")
def test_message_driven_state_machine(benchmark):
    """Benchmark message-driven state machine with processing

    Measures: state transitions per second with message-driven processing
    Scenario: 50 process messages followed by done
    Expected: Each message causes a self-transition
    """
    NUM_PROCESSES = 50

    def run_fsm():
        async def _run():
            common_data = {"process_count": 0}
            fsm = StateMachine(initial_state=ProcessingState, common_data=common_data)
            await fsm.initialize()

            # Send process messages
            for _ in range(NUM_PROCESSES):
                fsm.send_message("process")
                await fsm.tick()

            # Send done message
            fsm.send_message("done")
            await fsm.tick()

            return common_data["process_count"]

        return asyncio.run(_run())

    result = benchmark(run_fsm)


@pytest.mark.septum
@pytest.mark.benchmark(group="septum-scalability")
@pytest.mark.parametrize("num_transitions", [10, 50, 100, 200])
def test_scalability_different_transitions(benchmark, num_transitions):
    """Benchmark scalability with different transition counts

    Measures: how throughput scales with transition count
    Scenario: Varying numbers of state transitions
    Expected: Linear scaling (2x transitions = 2x time)
    """
    def run_fsm():
        async def _run():
            fsm = StateMachine(initial_state=StateA, common_data={})
            await fsm.initialize()

            # Alternate transitions
            for i in range(num_transitions):
                if i % 2 == 0:
                    fsm.send_message("next")
                else:
                    fsm.send_message("back")
                await fsm.tick()

            return num_transitions

        return asyncio.run(_run())

    result = benchmark(run_fsm)
