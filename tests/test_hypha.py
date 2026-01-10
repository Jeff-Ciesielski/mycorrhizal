#!/usr/bin/env python3
"""
Comprehensive pytest test suite for Hypha Petri Net library.

Run with: pytest tests/test_hypha.py -v

Test Organization:
- Fixtures: Timebase, blackboard, token classes
- Specs Tests: PlaceSpec, TransitionSpec, GuardSpec, ArcSpec, NetSpec, Reference types
- Builder Tests: NetBuilder, IO places, Subnets, Helper methods, @pn.net decorator
- Runtime Tests: PlaceRuntime, TransitionRuntime, NetRuntime, Runner
- Integration Tests: End-to-end scenarios
"""

import asyncio
import pytest
from dataclasses import dataclass, field
from typing import Any, List, Optional
from collections import deque

from mycorrhizal.hypha.core import pn, PlaceType, Runner as PNRunner
from mycorrhizal.hypha.core.specs import (
    PlaceSpec,
    TransitionSpec,
    GuardSpec,
    ArcSpec,
    NetSpec,
    PlaceRef,
    TransitionRef,
    SubnetRef,
)
from mycorrhizal.hypha.core.builder import NetBuilder, ArcChain
from mycorrhizal.hypha.core.runtime import PlaceRuntime, TransitionRuntime, NetRuntime
from mycorrhizal.common.timebase import Timebase, MonotonicClock


# =============================================================================
# Test Fixtures
# =============================================================================


class MockTimebase(Timebase):
    """Controllable timebase for testing time-dependent behavior."""

    def __init__(self, start: float = 0.0):
        self._time = start

    def now(self) -> float:
        return self._time

    def advance(self, delta: float = 0.1) -> None:
        self._time += delta

    def set_time(self, t: float) -> None:
        self._time = t

    async def sleep(self, duration: float) -> None:
        self._time += duration


@dataclass
class SimpleBlackboard:
    """Simple blackboard for basic tests."""
    value: int = 0
    log: List[str] = field(default_factory=list)
    processed_count: int = 0


@dataclass
class Token:
    """Basic token for testing."""
    data: Any


@dataclass
class TaskToken:
    """Task token for workflow testing."""
    task_id: str
    status: str = "pending"


@dataclass
class ErrorToken:
    """Error token for testing error handling."""
    error: str


@pytest.fixture
def simple_bb():
    return SimpleBlackboard()


@pytest.fixture
def mock_tb():
    return MockTimebase()


@pytest.fixture
def mono_tb():
    return MonotonicClock()


# =============================================================================
# Helper Functions
# =============================================================================


def assert_token_count(place: PlaceRuntime, expected: int):
    """Assert place has expected number of tokens."""
    actual = len(place.tokens)
    assert actual == expected, f"Expected {expected} tokens, got {actual}"


def assert_no_tokens(place: PlaceRuntime):
    """Assert place has no tokens."""
    assert_token_count(place, 0)


async def await_tokens(place: PlaceRuntime, count: int, timeout: float = 1.0):
    """Wait for place to have at least count tokens."""
    start = MockTimebase().now()
    while len(place.tokens) < count:
        await asyncio.sleep(0.01)
        if MockTimebase().now() - start > timeout:
            raise TimeoutError(f"Timeout waiting for {count} tokens")


# =============================================================================
# PlaceType Tests
# =============================================================================


def test_placetype_enum():
    """Test PlaceType enum values."""
    assert PlaceType.BAG.value == 1
    assert PlaceType.QUEUE.value == 2


# =============================================================================
# PlaceSpec Tests
# =============================================================================


def test_placespec_creation():
    """Test PlaceSpec creation with name and type."""
    spec = PlaceSpec("test_place", PlaceType.BAG)
    assert spec.name == "test_place"
    assert spec.place_type == PlaceType.BAG
    assert spec.handler is None
    assert spec.state_factory is None
    assert spec.is_io_input is False
    assert spec.is_io_output is False


def test_placespec_with_state_factory():
    """Test PlaceSpec with state_factory."""
    def factory():
        return {"count": 0}

    spec = PlaceSpec("counter", PlaceType.BAG, state_factory=factory)
    assert spec.state_factory == factory


def test_placespec_io_flags():
    """Test PlaceSpec IO flags."""
    input_spec = PlaceSpec("input", PlaceType.QUEUE, is_io_input=True)
    assert input_spec.is_io_input is True
    assert input_spec.is_io_output is False

    output_spec = PlaceSpec("output", PlaceType.QUEUE, is_io_output=True)
    assert output_spec.is_io_output is True
    assert output_spec.is_io_input is False


# =============================================================================
# TransitionSpec Tests
# =============================================================================


async def simple_handler(consumed, bb, timebase):
    """Simple handler for testing."""
    yield {}


def test_transition_spec_creation():
    """Test TransitionSpec creation."""
    spec = TransitionSpec("test_trans", simple_handler)
    assert spec.name == "test_trans"
    assert spec.handler == simple_handler
    assert spec.guard is None
    assert spec.state_factory is None


def test_transition_spec_with_guard():
    """Test TransitionSpec with guard."""
    def guard_func(tokens, bb, timebase):
        yield tokens[0]

    guard = GuardSpec(guard_func)
    spec = TransitionSpec("test_trans", simple_handler, guard=guard)
    assert spec.guard == guard


def test_transition_spec_with_state_factory():
    """Test TransitionSpec with state_factory."""
    def factory():
        return {"index": 0}

    spec = TransitionSpec("test_trans", simple_handler, state_factory=factory)
    assert spec.state_factory == factory


# =============================================================================
# GuardSpec Tests
# =============================================================================


def test_guard_spec_creation():
    """Test GuardSpec creation."""
    def guard_func(tokens, bb, timebase):
        yield tokens[0]

    guard = GuardSpec(guard_func)
    assert guard.func == guard_func


# =============================================================================
# ArcSpec Tests
# =============================================================================


def test_arc_spec_creation():
    """Test ArcSpec creation with PlaceRef to TransitionRef."""
    net = NetSpec("test_net")
    place = PlaceRef("input", net)
    trans = TransitionRef("process", net)

    arc = ArcSpec(place, trans, weight=1, name="test_arc")
    assert arc.source == place
    assert arc.target == trans
    assert arc.weight == 1
    assert arc.name == "test_arc"
    assert arc.source_parts == ("test_net", "input")
    assert arc.target_parts == ("test_net", "process")


def test_arc_spec_custom_weight():
    """Test ArcSpec with custom weight."""
    net = NetSpec("test_net")
    place = PlaceRef("input", net)
    trans = TransitionRef("process", net)

    arc = ArcSpec(place, trans, weight=3)
    assert arc.weight == 3


# =============================================================================
# PlaceRef Tests
# =============================================================================


def test_place_ref_fqn():
    """Test PlaceRef.get_fqn() for top-level place."""
    parent = NetSpec("parent")
    place = PlaceRef("my_place", parent)
    assert place.get_fqn() == "parent.my_place"
    assert place.fqn == "parent.my_place"  # backward compat


def test_place_ref_nested():
    """Test PlaceRef.get_fqn() for nested place."""
    root = NetSpec("root")
    child = NetSpec("child", parent=root)
    place = PlaceRef("nested_place", child)
    assert place.get_fqn() == "root.child.nested_place"


def test_place_ref_get_parts():
    """Test PlaceRef.get_parts()."""
    parent = NetSpec("parent")
    place = PlaceRef("my_place", parent)
    assert place.get_parts() == ["parent", "my_place"]


# =============================================================================
# TransitionRef Tests
# =============================================================================


def test_transition_ref_fqn():
    """Test TransitionRef.get_fqn()."""
    parent = NetSpec("parent")
    trans = TransitionRef("my_transition", parent)
    assert trans.get_fqn() == "parent.my_transition"


def test_transition_ref_get_parts():
    """Test TransitionRef.get_parts()."""
    parent = NetSpec("parent")
    trans = TransitionRef("my_transition", parent)
    assert trans.get_parts() == ["parent", "my_transition"]


# =============================================================================
# SubnetRef Tests
# =============================================================================


def test_subnet_ref_attribute_access_place():
    """Test SubnetRef.__getattr__ for place access."""
    net = NetSpec("test_net")
    net.places["input"] = PlaceSpec("input", PlaceType.QUEUE)

    subnet_ref = SubnetRef(net)
    place_ref = subnet_ref.input

    assert isinstance(place_ref, PlaceRef)
    assert place_ref.local_name == "input"


def test_subnet_ref_attribute_access_transition():
    """Test SubnetRef.__getattr__ for transition access."""
    net = NetSpec("test_net")
    net.transitions["process"] = TransitionSpec("process", simple_handler)

    subnet_ref = SubnetRef(net)
    trans_ref = subnet_ref.process

    assert isinstance(trans_ref, TransitionRef)
    assert trans_ref.local_name == "process"


def test_subnet_ref_attribute_unknown():
    """Test SubnetRef raises AttributeError for unknown member."""
    net = NetSpec("test_net")
    subnet_ref = SubnetRef(net)

    with pytest.raises(AttributeError):
        _ = subnet_ref.nonexistent


# =============================================================================
# NetSpec Tests
# =============================================================================


def test_netspec_creation():
    """Test NetSpec creation."""
    net = NetSpec("test_net")
    assert net.name == "test_net"
    assert net.parent is None
    assert net.places == {}
    assert net.transitions == {}
    assert net.arcs == []
    assert net.subnets == {}


def test_netspec_with_parent():
    """Test NetSpec with parent."""
    parent = NetSpec("parent")
    child = NetSpec("child", parent=parent)
    assert child.parent == parent


def test_netspec_fqn():
    """Test NetSpec.get_fqn()."""
    parent = NetSpec("parent")
    child = NetSpec("child", parent=parent)
    assert child.get_fqn() == "parent.child"


def test_netspec_get_parts():
    """Test NetSpec.get_parts()."""
    net = NetSpec("test_net")
    assert net.get_parts() == ["test_net"]

    parent = NetSpec("parent")
    child = NetSpec("child", parent=parent)
    assert child.get_parts("member") == ["parent", "child", "member"]


def test_netspec_to_mermaid():
    """Test NetSpec.to_mermaid() generates valid Mermaid syntax."""
    net = NetSpec("test_net")
    net.places["input"] = PlaceSpec("input", PlaceType.QUEUE)
    net.transitions["process"] = TransitionSpec("process", simple_handler)

    mermaid = net.to_mermaid()
    assert "graph TD" in mermaid
    assert "test_net.input" in mermaid
    assert "test_net.process" in mermaid


def test_netspec_to_mermaid_with_arcs():
    """Test NetSpec.to_mermaid() includes arcs."""
    net = NetSpec("test_net")
    input_place = PlaceRef("input", net)
    trans = TransitionRef("process", net)
    output_place = PlaceRef("output", net)

    net.places["input"] = PlaceSpec("input", PlaceType.QUEUE)
    net.places["output"] = PlaceSpec("output", PlaceType.QUEUE)
    net.transitions["process"] = TransitionSpec("process", simple_handler)
    net.arcs.append(ArcSpec(input_place, trans))
    net.arcs.append(ArcSpec(trans, output_place))

    mermaid = net.to_mermaid()
    assert "-->" in mermaid  # Arc connections


def test_netspec_to_mermaid_with_io_places():
    """Test NetSpec.to_mermaid() includes IO place labels."""
    net = NetSpec("test_net")
    net.places["input"] = PlaceSpec("input", PlaceType.QUEUE, is_io_input=True)
    net.places["output"] = PlaceSpec("output", PlaceType.QUEUE, is_io_output=True)

    mermaid = net.to_mermaid()
    # IO places should have special labels
    assert "test_net.input" in mermaid
    assert "test_net.output" in mermaid


# =============================================================================
# NetBuilder Tests
# =============================================================================


def test_builder_place_bag():
    """Test builder.place() creates BAG type place."""
    builder = NetBuilder("test_net")
    place_ref = builder.place("my_place", type=PlaceType.BAG)

    assert isinstance(place_ref, PlaceRef)
    assert "my_place" in builder.spec.places
    assert builder.spec.places["my_place"].place_type == PlaceType.BAG


def test_builder_place_queue():
    """Test builder.place() creates QUEUE type place."""
    builder = NetBuilder("test_net")
    place_ref = builder.place("my_place", type=PlaceType.QUEUE)

    assert isinstance(place_ref, PlaceRef)
    assert builder.spec.places["my_place"].place_type == PlaceType.QUEUE


def test_builder_place_with_state_factory():
    """Test builder.place() with state_factory."""
    def factory():
        return {"count": 0}

    builder = NetBuilder("test_net")
    place_ref = builder.place("counter", type=PlaceType.BAG, state_factory=factory)

    assert "counter" in builder.spec.places
    assert builder.spec.places["counter"].state_factory == factory


def test_builder_transition():
    """Test builder.transition() creates transition."""
    builder = NetBuilder("test_net")

    async def handler_func(consumed, bb, timebase):
        yield {}

    # The decorator returns a TransitionRef
    handler_ref = builder.transition()(handler_func)

    assert "handler_func" in builder.spec.transitions
    trans_spec = builder.spec.transitions["handler_func"]
    assert trans_spec.handler == handler_func
    assert isinstance(handler_ref, TransitionRef)


def test_builder_guard():
    """Test builder.guard() creates guard."""
    builder = NetBuilder("test_net")

    def guard_func(tokens, bb, timebase):
        yield tokens[0]

    guard = builder.guard(guard_func)
    assert isinstance(guard, GuardSpec)
    assert guard.func == guard_func


def test_builder_arc():
    """Test builder.arc() creates arc and returns ArcChain."""
    builder = NetBuilder("test_net")
    input_place = builder.place("input", PlaceType.QUEUE)

    @builder.transition()
    async def process(consumed, bb, timebase):
        yield {}

    chain = builder.arc(input_place, process)

    assert isinstance(chain, ArcChain)
    assert len(builder.spec.arcs) == 1
    assert builder.spec.arcs[0].source == input_place
    # process is a TransitionRef
    assert builder.spec.arcs[0].target == process


def test_builder_arc_chain():
    """Test ArcChain can chain multiple arcs."""
    builder = NetBuilder("test_net")
    input_place = builder.place("input", PlaceType.QUEUE)
    middle_place = builder.place("middle", PlaceType.QUEUE)
    output_place = builder.place("output", PlaceType.QUEUE)

    @builder.transition()
    async def trans1(consumed, bb, timebase):
        yield {}

    @builder.transition()
    async def trans2(consumed, bb, timebase):
        yield {}

    # Chain: input -> trans1 -> middle -> trans2 -> output
    builder.arc(input_place, trans1).arc(middle_place).arc(trans2).arc(output_place)

    assert len(builder.spec.arcs) == 4


# =============================================================================
# IO Place Tests
# =============================================================================


@pytest.mark.asyncio
async def test_io_input_place():
    """Test @builder.io_input_place() decorator."""
    builder = NetBuilder("test_net")

    @builder.io_input_place()
    async def source(bb, timebase):
        yield Token(1)
        yield Token(2)

    assert "source" in builder.spec.places
    place_spec = builder.spec.places["source"]
    assert place_spec.is_io_input is True
    assert place_spec.is_io_output is False
    assert place_spec.place_type == PlaceType.QUEUE
    assert callable(place_spec.handler)


@pytest.mark.asyncio
async def test_io_input_place_no_params():
    """Test io_input_place with no parameters."""
    builder = NetBuilder("test_net")

    @builder.io_input_place()
    async def source():
        yield Token(1)

    assert "source" in builder.spec.places
    assert callable(builder.spec.places["source"].handler)


@pytest.mark.asyncio
async def test_io_output_place():
    """Test @builder.io_output_place() decorator."""
    builder = NetBuilder("test_net")

    results = []

    @builder.io_output_place()
    async def sink(token, bb, timebase):
        results.append(token)

    assert "sink" in builder.spec.places
    place_spec = builder.spec.places["sink"]
    assert place_spec.is_io_output is True
    assert place_spec.is_io_input is False
    assert callable(place_spec.handler)


@pytest.mark.asyncio
async def test_io_output_place_token_only():
    """Test io_output_place with only token parameter."""
    builder = NetBuilder("test_net")

    @builder.io_output_place()
    async def sink(token):
        pass

    assert "sink" in builder.spec.places
    assert callable(builder.spec.places["sink"].handler)


# =============================================================================
# @pn.net Decorator Tests
# =============================================================================


def test_pn_net_decorator():
    """Test @pn.net decorator creates NetSpec."""
    @pn.net
    def SimpleNet(builder):
        p = builder.place("test", PlaceType.BAG)

    assert hasattr(SimpleNet, "_spec")
    assert isinstance(SimpleNet._spec, NetSpec)
    assert SimpleNet._spec.name == "SimpleNet"


def test_pn_net_decorator_with_to_mermaid():
    """Test @pn.net decorator attaches to_mermaid() method."""
    @pn.net
    def SimpleNet(builder):
        p = builder.place("test", PlaceType.BAG)

    assert hasattr(SimpleNet, "to_mermaid")
    assert callable(SimpleNet.to_mermaid)

    mermaid = SimpleNet.to_mermaid()
    assert "graph TD" in mermaid
    assert "SimpleNet.test" in mermaid


def test_multiple_nets():
    """Test multiple @pn.net decorators work independently."""
    @pn.net
    def Net1(builder):
        builder.place("place1", PlaceType.BAG)

    @pn.net
    def Net2(builder):
        builder.place("place2", PlaceType.BAG)

    assert Net1._spec.name == "Net1"
    assert Net2._spec.name == "Net2"
    assert "place1" in Net1._spec.places
    assert "place2" in Net2._spec.places


# =============================================================================
# Builder Helper Methods Tests
# =============================================================================


def test_forward_helper():
    """Test builder.forward() creates pass-through transition."""
    builder = NetBuilder("test")
    input_place = builder.place("input", PlaceType.QUEUE)
    output_place = builder.place("output", PlaceType.QUEUE)

    builder.forward(input_place, output_place, name="my_forward")

    # Check transition was created
    assert "my_forward" in builder.spec.transitions

    # Check arcs were created
    assert len(builder.spec.arcs) == 2
    # First arc: input -> transition
    assert builder.spec.arcs[0].source == input_place
    # Second arc: transition -> output
    assert builder.spec.arcs[1].target == output_place


def test_fork_helper():
    """Test builder.fork() creates transition with multiple outputs."""
    builder = NetBuilder("test")
    input_place = builder.place("input", PlaceType.QUEUE)
    out1 = builder.place("out1", PlaceType.QUEUE)
    out2 = builder.place("out2", PlaceType.QUEUE)
    out3 = builder.place("out3", PlaceType.QUEUE)

    builder.fork(input_place, [out1, out2, out3], name="my_fork")

    # Check transition was created
    assert "my_fork" in builder.spec.transitions

    # Check arcs: 1 input + 3 outputs = 4 arcs
    assert len(builder.spec.arcs) == 4

    # Count input/output arcs by checking source/target types
    # Arcs from place to transition (input arcs)
    input_arcs = [a for a in builder.spec.arcs if isinstance(a.source, PlaceRef) and a.source == input_place]
    # Arcs from transition to place (output arcs) - check that targets are the output places
    output_arcs = [a for a in builder.spec.arcs if a.target in (out1, out2, out3)]

    assert len(input_arcs) == 1
    assert len(output_arcs) == 3


def test_join_helper():
    """Test builder.join() creates transition with multiple inputs."""
    builder = NetBuilder("test")
    in1 = builder.place("in1", PlaceType.QUEUE)
    in2 = builder.place("in2", PlaceType.QUEUE)
    in3 = builder.place("in3", PlaceType.QUEUE)
    output_place = builder.place("output", PlaceType.QUEUE)

    builder.join([in1, in2, in3], output_place, name="my_join")

    # Check transition was created
    assert "my_join" in builder.spec.transitions

    # Check arcs: 3 inputs + 1 output = 4 arcs
    assert len(builder.spec.arcs) == 4

    # Count arcs by checking if source is one of the input places
    input_arcs = [a for a in builder.spec.arcs if a.source in (in1, in2, in3)]
    # Output arc goes to output_place
    output_arcs = [a for a in builder.spec.arcs if a.target == output_place]

    assert len(input_arcs) == 3
    assert len(output_arcs) == 1


def test_merge_helper():
    """Test builder.merge() creates multiple transitions."""
    builder = NetBuilder("test")
    in1 = builder.place("in1", PlaceType.QUEUE)
    in2 = builder.place("in2", PlaceType.QUEUE)
    output_place = builder.place("output", PlaceType.QUEUE)

    builder.merge([in1, in2], output_place)

    # Should create 2 transitions (one per input)
    assert len(builder.spec.transitions) == 2

    # Should create 4 arcs (2 inputs -> 2 transitions -> 1 output)
    assert len(builder.spec.arcs) == 4


def test_round_robin_helper():
    """Test builder.round_robin() creates stateful distributor."""
    builder = NetBuilder("test")
    input_place = builder.place("input", PlaceType.QUEUE)
    out1 = builder.place("out1", PlaceType.QUEUE)
    out2 = builder.place("out2", PlaceType.QUEUE)

    builder.round_robin(input_place, [out1, out2], name="rr")

    # Check transition was created
    assert "rr" in builder.spec.transitions

    # Check it has state_factory
    trans_spec = builder.spec.transitions["rr"]
    assert trans_spec.state_factory is not None

    # Initialize state to verify factory works
    state = trans_spec.state_factory()
    assert state == {"index": 0}


def test_route_helper():
    """Test builder.route() creates type-based router."""
    builder = NetBuilder("test")
    input_place = builder.place("input", PlaceType.QUEUE)
    str_out = builder.place("str_out", PlaceType.QUEUE)
    int_out = builder.place("int_out", PlaceType.QUEUE)

    type_map = {str: str_out, int: int_out}
    builder.route(input_place, type_map, name="router")

    # Check transition was created
    assert "router" in builder.spec.transitions

    # Check arcs
    assert len(builder.spec.arcs) >= 2  # At least input arc + some outputs


# =============================================================================
# Subnet Tests
# =============================================================================


def test_subnet_instantiation():
    """Test builder.subnet() creates nested net."""
    @pn.net
    def WorkerNet(builder):
        input_place = builder.place("input", PlaceType.QUEUE)
        output_place = builder.place("output", PlaceType.QUEUE)

        @builder.transition()
        async def work(consumed, bb, timebase):
            for token in consumed:
                yield {output_place: Token(f"processed_{token.data}")}

        builder.arc(input_place, work).arc(output_place)

    builder = NetBuilder("main")
    worker1 = builder.subnet(WorkerNet, "worker1")
    worker2 = builder.subnet(WorkerNet, "worker2")

    # Check subnets are in spec
    assert "worker1" in builder.spec.subnets
    assert "worker2" in builder.spec.subnets

    # Check FQNs are unique
    assert worker1.spec.get_fqn() == "main.worker1"
    assert worker2.spec.get_fqn() == "main.worker2"

    # Check places are accessible via SubnetRef
    assert isinstance(worker1.input, PlaceRef)
    assert worker1.input.get_fqn() == "main.worker1.input"
    assert worker1.output.get_fqn() == "main.worker1.output"


def test_subnet_remaps_arcs():
    """Test subnet arcs are remapped to instance."""
    @pn.net
    def WorkerNet(builder):
        input_place = builder.place("input", PlaceType.QUEUE)
        output_place = builder.place("output", PlaceType.QUEUE)

        @builder.transition()
        async def work(consumed, bb, timebase):
            yield {}

        builder.arc(input_place, work).arc(output_place)

    builder = NetBuilder("main")
    worker = builder.subnet(WorkerNet, "worker1")

    # Check that subnet has arcs
    assert len(worker.spec.arcs) == 2

    # Check arc references point to subnet, not original
    for arc in worker.spec.arcs:
        # Arc source/target should have worker1 in their FQN
        if hasattr(arc.source, 'get_fqn'):
            assert "worker1" in arc.source.get_fqn()
        if hasattr(arc.target, 'get_fqn'):
            assert "worker1" in arc.target.get_fqn()


def test_subnet_non_decorated_function_raises():
    """Test subnet() raises ValueError for non-@pn.net function."""
    def not_a_net(builder):
        pass

    builder = NetBuilder("main")

    with pytest.raises(ValueError, match="not_a_net is not a valid net"):
        builder.subnet(not_a_net, "instance")


def test_nested_subnets():
    """Test subnets within subnets."""
    @pn.net
    def InnerWorker(builder):
        builder.place("inner_input", PlaceType.QUEUE)

    @pn.net
    def OuterWorker(builder):
        builder.place("outer_input", PlaceType.QUEUE)
        inner = builder.subnet(InnerWorker, "inner")

    builder = NetBuilder("main")
    outer = builder.subnet(OuterWorker, "outer")

    # Check nested subnet exists
    assert "inner" in outer.spec.subnets
    assert "outer" in builder.spec.subnets

    # Check FQN hierarchy
    inner_fqn = outer.spec.subnets["inner"].get_fqn()
    assert "outer.inner" in inner_fqn


# =============================================================================
# PlaceRuntime Tests
# =============================================================================


def test_place_runtime_bag():
    """Test PlaceRuntime with BAG type creates list."""
    spec = PlaceSpec("test", PlaceType.BAG)
    place = PlaceRuntime(spec, None, None, "test")

    assert isinstance(place.tokens, list)
    assert len(place.tokens) == 0


def test_place_runtime_queue():
    """Test PlaceRuntime with QUEUE type creates deque."""
    spec = PlaceSpec("test", PlaceType.QUEUE)
    place = PlaceRuntime(spec, None, None, "test")

    assert isinstance(place.tokens, deque)


def test_place_runtime_add_token():
    """Test PlaceRuntime.add_token()."""
    spec = PlaceSpec("test", PlaceType.BAG)
    place = PlaceRuntime(spec, None, None, "test")

    place.add_token(Token(1))
    place.add_token(Token(2))

    assert len(place.tokens) == 2
    assert Token(1) in place.tokens
    assert Token(2) in place.tokens


def test_place_runtime_remove_tokens():
    """Test PlaceRuntime.remove_tokens()."""
    spec = PlaceSpec("test", PlaceType.BAG)
    place = PlaceRuntime(spec, None, None, "test")

    token1 = Token(1)
    token2 = Token(2)
    place.add_token(token1)
    place.add_token(token2)

    place.remove_tokens([token1])

    assert len(place.tokens) == 1
    assert token1 not in place.tokens
    assert token2 in place.tokens


def test_place_runtime_peek_tokens():
    """Test PlaceRuntime.peek_tokens()."""
    spec = PlaceSpec("test", PlaceType.QUEUE)
    place = PlaceRuntime(spec, None, None, "test")

    place.add_token(Token(1))
    place.add_token(Token(2))
    place.add_token(Token(3))

    # Peek 2 tokens
    tokens = place.peek_tokens(2)
    assert len(tokens) == 2
    assert tokens[0].data == 1
    assert tokens[1].data == 2

    # Original tokens should remain
    assert len(place.tokens) == 3


def test_place_runtime_state_factory():
    """Test PlaceRuntime calls state_factory."""
    def factory():
        return {"count": 0}

    spec = PlaceSpec("counter", PlaceType.BAG, state_factory=factory)
    place = PlaceRuntime(spec, None, None, "counter")

    assert place.state == {"count": 0}


# =============================================================================
# NetRuntime Tests
# =============================================================================


def test_netruntime_flatten_simple(simple_bb, mock_tb):
    """Test NetRuntime._flatten_spec() with simple net."""
    @pn.net
    def SimpleNet(builder):
        builder.place("input", PlaceType.QUEUE)
        builder.place("output", PlaceType.BAG)

    runtime = NetRuntime(SimpleNet._spec, simple_bb, mock_tb)

    # Check places
    assert ("SimpleNet", "input") in runtime.places
    assert ("SimpleNet", "output") in runtime.places


def test_netruntime_build(simple_bb, mock_tb):
    """Test NetRuntime._build_runtime() creates runtime objects."""
    @pn.net
    def SimpleNet(builder):
        input_place = builder.place("input", PlaceType.QUEUE)
        output_place = builder.place("output", PlaceType.QUEUE)

        @builder.transition()
        async def process(consumed, bb, timebase):
            yield {}

        builder.arc(input_place, process).arc(output_place)

    runtime = NetRuntime(SimpleNet._spec, simple_bb, mock_tb)

    # Check place runtimes
    input_runtime = runtime.places[("SimpleNet", "input")]
    assert isinstance(input_runtime, PlaceRuntime)
    assert input_runtime.fqn == "SimpleNet.input"

    # Check transition runtimes
    trans_runtime = runtime.transitions[("SimpleNet", "process")]
    assert isinstance(trans_runtime, TransitionRuntime)
    assert trans_runtime.fqn == "SimpleNet.process"


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.asyncio
async def test_simple_pipeline(simple_bb, mono_tb):
    """Test end-to-end token flow through simple pipeline."""
    @pn.net
    def Pipeline(builder):
        input_place = builder.place("input", PlaceType.QUEUE)
        output_place = builder.place("output", PlaceType.QUEUE)

        @builder.transition()
        async def process(consumed, bb, timebase):
            for token in consumed:
                processed = Token(f"processed_{token.data}")
                yield {output_place: processed}

        builder.arc(input_place, process).arc(output_place)

    runner = PNRunner(Pipeline, simple_bb)
    await runner.start(mono_tb)

    # Add tokens to input place with delay
    runtime = runner.runtime
    runtime.places[("Pipeline", "input")].add_token(Token("token1"))
    await asyncio.sleep(0.1)
    runtime.places[("Pipeline", "input")].add_token(Token("token2"))

    # Give time for processing
    await asyncio.sleep(0.5)

    # Check output
    output = runtime.places[("Pipeline", "output")]
    assert len(output.tokens) == 2
    assert "processed_token1" in [t.data for t in output.tokens]
    assert "processed_token2" in [t.data for t in output.tokens]

    await runner.stop()


@pytest.mark.asyncio
async def test_pn_net_with_runner(simple_bb, mono_tb):
    """Test complete workflow with @pn.net and Runner."""
    @pn.net
    def ProcessingNet(builder):
        @builder.io_input_place()
        async def source(bb, timebase):
            for i in range(3):
                await timebase.sleep(0.05)
                yield TaskToken(f"task_{i}")

        done_place = builder.place("done", PlaceType.BAG)

        @builder.transition()
        async def process(consumed, bb, timebase):
            for token in consumed:
                if isinstance(token, TaskToken):
                    token.status = "done"
                    yield {done_place: token}

        builder.arc(source, process).arc(done_place)

    runner = PNRunner(ProcessingNet, simple_bb)
    await runner.start(mono_tb)

    # Wait for tokens to be generated and processed
    await asyncio.sleep(1.0)

    runtime = runner.runtime
    done_place = runtime.places[("ProcessingNet", "done")]
    assert len(done_place.tokens) >= 1

    await runner.stop()


@pytest.mark.asyncio
async def test_subnet_integration(simple_bb, mono_tb):
    """Test end-to-end flow with subnets."""
    @pn.net
    def Worker(builder):
        input_place = builder.place("input", PlaceType.QUEUE)
        output_place = builder.place("output", PlaceType.QUEUE)

        @builder.transition()
        async def work(consumed, bb, timebase):
            for token in consumed:
                result = Token(f"{token.data}_worked")
                yield {output_place: result}

        builder.arc(input_place, work).arc(output_place)

    @pn.net
    def Main(builder):
        input_place = builder.place("input", PlaceType.QUEUE)
        output_place = builder.place("output", PlaceType.QUEUE)

        worker1 = builder.subnet(Worker, "worker1")
        worker2 = builder.subnet(Worker, "worker2")

        builder.forward(input_place, worker1.input)
        builder.forward(input_place, worker2.input)
        builder.merge([worker1.output, worker2.output], output_place)

    runner = PNRunner(Main, simple_bb)
    await runner.start(mono_tb)

    # Add tokens with delay
    runtime = runner.runtime
    runtime.places[("Main", "input")].add_token(Token("job1"))
    await asyncio.sleep(0.1)
    runtime.places[("Main", "input")].add_token(Token("job2"))

    await asyncio.sleep(1.0)

    # Check both workers processed
    output = runtime.places[("Main", "output")]
    # Since both transitions compete for tokens from the same queue,
    # we expect 2 tokens total (one per job, each processed by one worker)
    assert len(output.tokens) == 2

    await runner.stop()



@pytest.mark.asyncio
async def test_forward_helper_integration(simple_bb, mono_tb):
    """Test forward helper in actual net execution."""
    @pn.net
    def ForwardNet(builder):
        input_place = builder.place("input", PlaceType.QUEUE)
        output_place = builder.place("output", PlaceType.QUEUE)

        builder.forward(input_place, output_place)

    runner = PNRunner(ForwardNet, simple_bb)
    await runner.start(mono_tb)

    # Add token
    runtime = runner.runtime
    runtime.places[("ForwardNet", "input")].add_token(Token("test"))

    await asyncio.sleep(0.3)

    # Check token moved through
    output = runtime.places[("ForwardNet", "output")]
    assert len(output.tokens) == 1
    assert output.tokens[0].data == "test"

    await runner.stop()


@pytest.mark.asyncio
async def test_fork_helper_integration(simple_bb, mono_tb):
    """Test fork helper distributes tokens to multiple outputs."""
    @pn.net
    def ForkNet(builder):
        input_place = builder.place("input", PlaceType.QUEUE)
        out1 = builder.place("out1", PlaceType.BAG)
        out2 = builder.place("out2", PlaceType.BAG)
        out3 = builder.place("out3", PlaceType.BAG)

        builder.fork(input_place, [out1, out2, out3])

    runner = PNRunner(ForkNet, simple_bb)
    await runner.start(mono_tb)

    # Add token
    runtime = runner.runtime
    runtime.places[("ForkNet", "input")].add_token(Token("test"))

    await asyncio.sleep(0.3)

    # Check token was forked to all outputs
    assert len(runtime.places[("ForkNet", "out1")].tokens) == 1
    assert len(runtime.places[("ForkNet", "out2")].tokens) == 1
    assert len(runtime.places[("ForkNet", "out3")].tokens) == 1

    await runner.stop()


@pytest.mark.asyncio
async def test_round_robin_helper_integration(simple_bb, mono_tb):
    """Test round_robin helper distributes tokens sequentially."""
    @pn.net
    def RRNet(builder):
        input_place = builder.place("input", PlaceType.QUEUE)
        out1 = builder.place("out1", PlaceType.BAG)
        out2 = builder.place("out2", PlaceType.BAG)

        builder.round_robin(input_place, [out1, out2])

    runner = PNRunner(RRNet, simple_bb)
    await runner.start(mono_tb)

    runtime = runner.runtime
    input_place = runtime.places[("RRNet", "input")]

    # Add multiple tokens with delay
    for i in range(4):
        input_place.add_token(Token(f"token_{i}"))
        await asyncio.sleep(0.05)

    await asyncio.sleep(0.5)

    # Check round-robin distribution (2 tokens each)
    out1_tokens = runtime.places[("RRNet", "out1")].tokens
    out2_tokens = runtime.places[("RRNet", "out2")].tokens

    assert len(out1_tokens) == 2
    assert len(out2_tokens) == 2

    await runner.stop()

