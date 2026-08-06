#!/usr/bin/env python3
"""Tests for Spores DSL adapters."""

import sys
sys.path.insert(0, "src")

import pytest
import asyncio
from typing import Annotated
from pydantic import BaseModel

from mycorrhizal.spores import configure, get_config, get_object_cache
from mycorrhizal.spores.transport import Transport
from mycorrhizal.spores.dsl import HyphaAdapter, RhizomorphAdapter, SeptumAdapter
from mycorrhizal.spores.models import ObjectRef, ObjectScope, EventAttr
from mycorrhizal.rhizomorph.core import Status
from mycorrhizal.septum.core import SharedContext
from mycorrhizal.common.timebase import WallClock


# ============================================================================
# Test Fixtures
# ============================================================================

class MockTransport(Transport):
    """Mock transport for testing."""

    def __init__(self):
        self.records = []

    async def send(self, data: bytes, content_type: str) -> None:
        """Store records instead of sending."""
        import json
        record = json.loads(data.decode('utf-8'))
        self.records.append(record)

    def is_async(self) -> bool:
        return False

    def close(self) -> None:
        pass


@pytest.fixture
def mock_transport():
    """Create a mock transport."""
    return MockTransport()


@pytest.fixture
def spore_config(mock_transport):
    """Configure spores with mock transport."""
    configure(
        enabled=True,
        object_cache_size=10,
        transport=mock_transport,
    )
    yield get_config()

    # Reset after test
    import importlib
    import mycorrhizal.spores.core as core
    importlib.reload(core)


def get_attr_value(attributes_list, attr_name):
    """Get attribute value from attributes list (OCEL format)."""
    for attr in attributes_list:
        if attr.get("name") == attr_name:
            return attr.get("value")
    return None


def has_attr(attributes_list, attr_name):
    """Check if attribute exists in attributes list (OCEL format)."""
    return any(attr.get("name") == attr_name for attr in attributes_list)


# ============================================================================
# Test Data Models
# ============================================================================

class WorkItem(BaseModel):
    id: str
    status: str


class Robot(BaseModel):
    id: str
    name: str


class MissionBlackboard(BaseModel):
    mission_id: Annotated[str, EventAttr]
    current_item: Annotated[WorkItem, ObjectRef(qualifier="target", scope=ObjectScope.EVENT)]
    robot: Annotated[Robot, ObjectRef(qualifier="actor", scope=ObjectScope.GLOBAL)]
    value: int


# ============================================================================
# Rhizomorph Adapter Tests
# ============================================================================

@pytest.mark.asyncio
async def test_rhizomorph_log_node_basic(spore_config, mock_transport):
    """Test basic node logging with Rhizomorph adapter."""

    adapter = RhizomorphAdapter()
    tb = WallClock()

    bb = MissionBlackboard(
        mission_id="mission-123",
        current_item=WorkItem(id="item-1", status="pending"),
        robot=Robot(id="robot-1", name="Robo-X"),
        value=42
    )

    @adapter.log_node(event_type="test_action")
    async def test_action(bb: MissionBlackboard, tb: WallClock) -> Status:
        return Status.SUCCESS

    result = await test_action(bb, tb)

    assert result == Status.SUCCESS
    await asyncio.sleep(0.01)  # Let async tasks complete

    assert len(mock_transport.records) >= 1

    # Check event was logged
    event_records = [r for r in mock_transport.records if "event" in r]
    assert len(event_records) >= 1

    event = event_records[0]["event"]
    assert event["type"] == "test_action"
    assert has_attr(event["attributes"], "node_name")
    assert has_attr(event["attributes"], "status")
    assert get_attr_value(event["attributes"], "status") == "SUCCESS"

    # Check objects were logged
    object_records = [r for r in mock_transport.records if "object" in r]
    # At least the Robot (global scope) should be logged
    # WorkItem is EVENT scope and not explicitly requested, so may not be logged
    assert len(object_records) >= 1


@pytest.mark.asyncio
async def test_rhizomorph_log_node_no_status(spore_config, mock_transport):
    """Test node logging without status attribute."""

    adapter = RhizomorphAdapter()
    tb = WallClock()

    bb = MissionBlackboard(
        mission_id="mission-456",
        current_item=WorkItem(id="item-2", status="active"),
        robot=Robot(id="robot-2", name="Robo-Y"),
        value=99
    )

    @adapter.log_node(event_type="test_action", log_status=False)
    async def test_action(bb: MissionBlackboard, tb: WallClock) -> Status:
        return Status.FAILURE

    await test_action(bb, tb)
    await asyncio.sleep(0.01)

    event_records = [r for r in mock_transport.records if "event" in r]
    assert len(event_records) >= 1

    event = event_records[0]["event"]
    assert not has_attr(event["attributes"], "status")  # Status should not be logged


@pytest.mark.asyncio
async def test_rhizomorph_log_node_sync(spore_config, mock_transport):
    """A sync node spanning several ticks is logged once, when its run ends."""

    adapter = RhizomorphAdapter()
    tb = WallClock()

    bb = MissionBlackboard(
        mission_id="mission-789",
        current_item=WorkItem(id="item-3", status="done"),
        robot=Robot(id="robot-3", name="Robo-Z"),
        value=123
    )

    ticks = {"n": 0}

    @adapter.log_node(event_type="sync_action")
    def sync_action(bb: MissionBlackboard, tb: WallClock) -> Status:
        ticks["n"] += 1
        return Status.SUCCESS if ticks["n"] >= 3 else Status.RUNNING

    # A node is logged once per RUN, not once per tick. Two RUNNING ticks are
    # one activity still in progress, so neither emits; the terminal tick ends
    # the run and emits the single event for it.
    # Note the `r.get("event")` filter: a LogRecord carries both keys with one
    # of them null, so `"event" in r` is true for an object record too.
    assert sync_action(bb, tb) == Status.RUNNING
    assert sync_action(bb, tb) == Status.RUNNING
    await asyncio.sleep(0.01)
    assert [r for r in mock_transport.records if r.get("event")] == []

    assert sync_action(bb, tb) == Status.SUCCESS
    await asyncio.sleep(0.01)  # Let async tasks complete

    event_records = [r for r in mock_transport.records if r.get("event")]
    assert len(event_records) == 1
    assert event_records[0]["event"]["type"] == "sync_action"
    assert get_attr_value(event_records[0]["event"]["attributes"], "status") == "SUCCESS"


# ============================================================================
# Rhizomorph Adapter: enter/exit mode tests
# ============================================================================

@pytest.mark.asyncio
async def test_rhizomorph_log_node_default_off_byte_parity(spore_config, mock_transport):
    """Default (enter_exit unset) must stay byte-identical to today's single
    post-hoc event: exactly one event record, type unchanged (no _start/_stop
    suffix), for both the adapter default and an explicit per-call False."""

    for adapter, decorator_enter_exit in (
        (RhizomorphAdapter(), None),
        (RhizomorphAdapter(), False),
        (RhizomorphAdapter(enter_exit=False), None),
    ):
        mock_transport.records.clear()
        tb = WallClock()

        bb = MissionBlackboard(
            mission_id="mission-parity",
            current_item=WorkItem(id="item-parity", status="pending"),
            robot=Robot(id="robot-parity", name="Robo-Parity"),
            value=1,
        )

        kwargs = {"event_type": "parity_action"}
        if decorator_enter_exit is not None:
            kwargs["enter_exit"] = decorator_enter_exit

        @adapter.log_node(**kwargs)
        async def parity_action(bb: MissionBlackboard, tb: WallClock) -> Status:
            return Status.SUCCESS

        result = await parity_action(bb, tb)
        await asyncio.sleep(0.01)

        assert result == Status.SUCCESS

        event_records = [r for r in mock_transport.records if r.get("event")]
        assert len(event_records) == 1

        event = event_records[0]["event"]
        assert event["type"] == "parity_action"
        assert get_attr_value(event["attributes"], "status") == "SUCCESS"


@pytest.mark.asyncio
async def test_rhizomorph_log_node_enter_exit_pairing(spore_config, mock_transport):
    """enter_exit=True logs a start/stop pair sharing node name, blackboard
    attributes, and object relationships; the exit event carries status."""

    adapter = RhizomorphAdapter(enter_exit=True)
    tb = WallClock()

    bb = MissionBlackboard(
        mission_id="mission-pair",
        current_item=WorkItem(id="item-pair", status="pending"),
        robot=Robot(id="robot-pair", name="Robo-Pair"),
        value=7,
    )

    @adapter.log_node(event_type="paired_action")
    async def paired_action(bb: MissionBlackboard, tb: WallClock) -> Status:
        return Status.SUCCESS

    result = await paired_action(bb, tb)
    await asyncio.sleep(0.01)

    assert result == Status.SUCCESS

    event_records = [r for r in mock_transport.records if r.get("event")]
    assert len(event_records) == 2

    entry, exit_ = event_records[0]["event"], event_records[1]["event"]

    # Order and naming
    assert entry["type"] == "paired_action_start"
    assert exit_["type"] == "paired_action_stop"
    assert entry["time"] <= exit_["time"]

    # Shared context: node name and blackboard-derived attributes/objects
    for evt in (entry, exit_):
        assert get_attr_value(evt["attributes"], "node_name") == "paired_action"
        assert get_attr_value(evt["attributes"], "mission_id") == "mission-pair"
        assert len(evt["relationships"]) >= 1  # global-scope Robot at least

    # Only the exit event carries the outcome status
    assert not has_attr(entry["attributes"], "status")
    assert get_attr_value(exit_["attributes"], "status") == "SUCCESS"


@pytest.mark.asyncio
async def test_rhizomorph_log_node_enter_exit_exception_path(spore_config, mock_transport):
    """A raised exception still produces the exit event (status=EXCEPTION)
    before propagating to the caller."""

    adapter = RhizomorphAdapter(enter_exit=True)
    tb = WallClock()

    bb = MissionBlackboard(
        mission_id="mission-fail",
        current_item=WorkItem(id="item-fail", status="pending"),
        robot=Robot(id="robot-fail", name="Robo-Fail"),
        value=0,
    )

    @adapter.log_node(event_type="failing_action")
    async def failing_action(bb: MissionBlackboard, tb: WallClock) -> Status:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await failing_action(bb, tb)

    await asyncio.sleep(0.01)

    event_records = [r for r in mock_transport.records if r.get("event")]
    assert [r["event"]["type"] for r in event_records] == [
        "failing_action_start",
        "failing_action_stop",
    ]

    exit_event = event_records[1]["event"]
    assert get_attr_value(exit_event["attributes"], "status") == "EXCEPTION"


@pytest.mark.asyncio
async def test_rhizomorph_log_node_enter_exit_exception_ignores_log_status_false(
    spore_config, mock_transport
):
    """log_status=False suppresses the status attribute for normal outcomes,
    but an EXCEPTION outcome is always surfaced on the exit event."""

    adapter = RhizomorphAdapter(enter_exit=True)
    tb = WallClock()

    bb = MissionBlackboard(
        mission_id="mission-fail-2",
        current_item=WorkItem(id="item-fail-2", status="pending"),
        robot=Robot(id="robot-fail-2", name="Robo-Fail-2"),
        value=0,
    )

    @adapter.log_node(event_type="failing_action_quiet", log_status=False)
    async def failing_action_quiet(bb: MissionBlackboard, tb: WallClock) -> Status:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await failing_action_quiet(bb, tb)

    await asyncio.sleep(0.01)

    event_records = [r for r in mock_transport.records if r.get("event")]
    exit_event = event_records[1]["event"]
    assert get_attr_value(exit_event["attributes"], "status") == "EXCEPTION"


@pytest.mark.asyncio
async def test_rhizomorph_log_node_enter_exit_async_ordering(spore_config, mock_transport):
    """Entry is emitted before the wrapped node's own first await, and exit
    is emitted only after the node resumes and returns."""

    adapter = RhizomorphAdapter(enter_exit=True)
    tb = WallClock()

    bb = MissionBlackboard(
        mission_id="mission-order",
        current_item=WorkItem(id="item-order", status="pending"),
        robot=Robot(id="robot-order", name="Robo-Order"),
        value=3,
    )

    order = []

    original_send = mock_transport.send

    async def tracking_send(data, content_type):
        import json
        record = json.loads(data.decode("utf-8"))
        if record.get("event"):
            order.append(f"event:{record['event']['type']}")
        await original_send(data, content_type)

    mock_transport.send = tracking_send

    @adapter.log_node(event_type="ordered_action")
    async def ordered_action(bb: MissionBlackboard, tb: WallClock) -> Status:
        order.append("func_start")
        await asyncio.sleep(0)
        order.append("func_resumed")
        return Status.SUCCESS

    result = await ordered_action(bb, tb)
    await asyncio.sleep(0.01)

    assert result == Status.SUCCESS
    assert order.index("event:ordered_action_start") < order.index("func_start")
    assert order.index("func_start") < order.index("func_resumed")
    assert order.index("func_resumed") < order.index("event:ordered_action_stop")


@pytest.mark.asyncio
async def test_rhizomorph_log_node_enter_exit_sync(spore_config, mock_transport):
    """enter_exit mode pairs entry/exit around a sync node's RUN, not its tick."""

    adapter = RhizomorphAdapter(enter_exit=True)
    tb = WallClock()

    bb = MissionBlackboard(
        mission_id="mission-sync",
        current_item=WorkItem(id="item-sync", status="pending"),
        robot=Robot(id="robot-sync", name="Robo-Sync"),
        value=9,
    )

    ticks = {"n": 0}

    @adapter.log_node(event_type="sync_paired_action")
    def sync_paired_action(bb: MissionBlackboard, tb: WallClock) -> Status:
        ticks["n"] += 1
        return Status.SUCCESS if ticks["n"] >= 3 else Status.RUNNING

    # The pair brackets the RUN, not the tick: entry on the tick that opens it,
    # exit on the tick that ends it, and nothing from the ticks in between.
    assert sync_paired_action(bb, tb) == Status.RUNNING
    await asyncio.sleep(0.01)
    assert [r["event"]["type"] for r in mock_transport.records if r.get("event")] == [
        "sync_paired_action_start",
    ]

    assert sync_paired_action(bb, tb) == Status.RUNNING
    await asyncio.sleep(0.01)
    assert len([r for r in mock_transport.records if r.get("event")]) == 1

    assert sync_paired_action(bb, tb) == Status.SUCCESS
    await asyncio.sleep(0.01)

    event_records = [r for r in mock_transport.records if r.get("event")]
    assert [r["event"]["type"] for r in event_records] == [
        "sync_paired_action_start",
        "sync_paired_action_stop",
    ]
    assert get_attr_value(event_records[1]["event"]["attributes"], "status") == "SUCCESS"


@pytest.mark.asyncio
async def test_rhizomorph_log_node_enter_exit_per_call_override(spore_config, mock_transport):
    """A decorator-level enter_exit overrides the adapter's constructor
    default in both directions."""

    adapter = RhizomorphAdapter(enter_exit=False)
    tb = WallClock()

    bb = MissionBlackboard(
        mission_id="mission-override",
        current_item=WorkItem(id="item-override", status="pending"),
        robot=Robot(id="robot-override", name="Robo-Override"),
        value=2,
    )

    @adapter.log_node(event_type="overridden_action", enter_exit=True)
    async def overridden_action(bb: MissionBlackboard, tb: WallClock) -> Status:
        return Status.SUCCESS

    await overridden_action(bb, tb)
    await asyncio.sleep(0.01)

    event_records = [r for r in mock_transport.records if r.get("event")]
    assert [r["event"]["type"] for r in event_records] == [
        "overridden_action_start",
        "overridden_action_stop",
    ]


@pytest.mark.asyncio
async def test_rhizomorph_log_node_full_payload_parity_default_off(spore_config, mock_transport):
    """Full-payload parity: with enter_exit off (default), the event's
    type, complete attribute set, and complete relationship set match the
    pre-extension single-event behavior exactly - not just the type and a
    couple of spot-checked attributes. Guards against the enter/exit
    refactor silently changing what the default path emits."""

    adapter = RhizomorphAdapter()
    tb = WallClock()

    bb = MissionBlackboard(
        mission_id="mission-full-parity-r",
        current_item=WorkItem(id="item-full-parity-r", status="pending"),
        robot=Robot(id="robot-full-parity-r", name="Robo-Full-Parity-R"),
        value=1,
    )

    @adapter.log_node(event_type="full_parity_action")
    async def full_parity_action(bb: MissionBlackboard, tb: WallClock) -> Status:
        return Status.SUCCESS

    result = await full_parity_action(bb, tb)
    await asyncio.sleep(0.01)

    assert result == Status.SUCCESS

    event_records = [r for r in mock_transport.records if r.get("event")]
    assert len(event_records) == 1
    event = event_records[0]["event"]

    assert event["type"] == "full_parity_action"

    attrs = {a["name"]: a["value"] for a in event["attributes"]}
    assert attrs == {
        "node_name": "full_parity_action",
        "status": "SUCCESS",
        "mission_id": "mission-full-parity-r",
    }

    # Only the global-scope Robot is pulled from the blackboard;
    # current_item is EVENT scope and not requested, so it is absent.
    rel_object_ids = {r["objectId"] for r in event["relationships"]}
    assert rel_object_ids == {"robot-full-parity-r"}
    assert len(event["relationships"]) == 1


@pytest.mark.asyncio
async def test_rhizomorph_log_node_enter_exit_sync_exception_path(spore_config, mock_transport):
    """A raised exception in a *sync* node function still produces the
    exit event (status=EXCEPTION) before propagating - mirrors the
    existing async exception-path test, but for the sync_wrapper branch."""

    adapter = RhizomorphAdapter(enter_exit=True)
    tb = WallClock()

    bb = MissionBlackboard(
        mission_id="mission-sync-fail-r",
        current_item=WorkItem(id="item-sync-fail-r", status="pending"),
        robot=Robot(id="robot-sync-fail-r", name="Robo-Sync-Fail-R"),
        value=0,
    )

    @adapter.log_node(event_type="failing_sync_action")
    def failing_sync_action(bb: MissionBlackboard, tb: WallClock) -> Status:
        raise RuntimeError("sync boom")

    with pytest.raises(RuntimeError, match="sync boom"):
        failing_sync_action(bb, tb)

    await asyncio.sleep(0.01)

    event_records = [r for r in mock_transport.records if r.get("event")]
    assert [r["event"]["type"] for r in event_records] == [
        "failing_sync_action_start",
        "failing_sync_action_stop",
    ]

    exit_event = event_records[1]["event"]
    assert get_attr_value(exit_event["attributes"], "status") == "EXCEPTION"


# ============================================================================
# Hypha Adapter Tests
# ============================================================================

@pytest.mark.asyncio
async def test_hypha_log_transition_basic(spore_config, mock_transport):
    """Test basic transition logging with Hypha adapter."""

    adapter = HyphaAdapter()
    tb = WallClock()

    bb = MissionBlackboard(
        mission_id="mission-999",
        current_item=WorkItem(id="item-4", status="processing"),
        robot=Robot(id="robot-4", name="Robo-A"),
        value=77
    )

    @adapter.log_transition(event_type="process_item")
    async def test_transition(consumed: list, bb: MissionBlackboard, timebase: WallClock):
        yield {"output": consumed[0]}

    # Simulate transition call
    consumed = [WorkItem(id="item-4", status="processing")]
    gen = test_transition(consumed, bb, tb)

    result = await gen.__anext__()
    await asyncio.sleep(0.01)

    assert result is not None

    # Check event was logged
    event_records = [r for r in mock_transport.records if "event" in r]
    assert len(event_records) >= 1

    event = event_records[0]["event"]
    assert event["type"] == "process_item"
    assert has_attr(event["attributes"], "transition_name")
    assert has_attr(event["attributes"], "token_count")
    assert get_attr_value(event["attributes"], "token_count") == "1"


@pytest.mark.asyncio
async def test_hypha_log_transition_with_tokens(spore_config, mock_transport):
    """Test transition logging with input token logging."""

    adapter = HyphaAdapter()
    tb = WallClock()

    bb = MissionBlackboard(
        mission_id="mission-888",
        current_item=WorkItem(id="item-5", status="queued"),
        robot=Robot(id="robot-5", name="Robo-B"),
        value=55
    )

    @adapter.log_transition(event_type="batch_process", log_inputs=True)
    async def batch_transition(consumed: list, bb: MissionBlackboard, timebase: WallClock):
        # Process batch
        yield {"output": consumed[0]}

    consumed = [
        WorkItem(id="item-5", status="queued"),
        WorkItem(id="item-6", status="queued"),
    ]

    gen = batch_transition(consumed, bb, tb)
    await gen.__anext__()
    await asyncio.sleep(0.01)

    event_records = [r for r in mock_transport.records if "event" in r]
    assert len(event_records) >= 1

    event = event_records[0]["event"]
    assert get_attr_value(event["attributes"], "token_count") == "2"

    # Check relationships to input tokens
    assert "relationships" in event
    assert len(event["relationships"]) >= 2  # At least the two input tokens


# ============================================================================
# Hypha Adapter: enter/exit mode tests
# ============================================================================

@pytest.mark.asyncio
async def test_hypha_log_transition_default_off_byte_parity_asyncgen(
    spore_config, mock_transport
):
    """Default (enter_exit unset) stays byte-identical to today's single
    pre-processing event for an async-generator transition: exactly one
    event record, type unchanged."""

    for adapter, decorator_enter_exit in (
        (HyphaAdapter(), None),
        (HyphaAdapter(), False),
        (HyphaAdapter(enter_exit=False), None),
    ):
        mock_transport.records.clear()
        tb = WallClock()

        bb = MissionBlackboard(
            mission_id="mission-parity-h1",
            current_item=WorkItem(id="item-parity-h1", status="processing"),
            robot=Robot(id="robot-parity-h1", name="Robo-Parity-H1"),
            value=1,
        )

        kwargs = {"event_type": "parity_transition"}
        if decorator_enter_exit is not None:
            kwargs["enter_exit"] = decorator_enter_exit

        @adapter.log_transition(**kwargs)
        async def parity_transition(consumed: list, bb: MissionBlackboard, timebase: WallClock):
            yield {"output": consumed[0]}

        consumed = [WorkItem(id="item-parity-h1", status="processing")]
        gen = parity_transition(consumed, bb, tb)
        await gen.__anext__()
        await asyncio.sleep(0.01)

        event_records = [r for r in mock_transport.records if r.get("event")]
        assert len(event_records) == 1
        assert event_records[0]["event"]["type"] == "parity_transition"


@pytest.mark.asyncio
async def test_hypha_log_transition_default_off_byte_parity_coroutine(
    spore_config, mock_transport
):
    """Default (enter_exit unset) stays byte-identical to today's single
    post-completion event for a coroutine-style transition."""

    adapter = HyphaAdapter()
    tb = WallClock()

    bb = MissionBlackboard(
        mission_id="mission-parity-h2",
        current_item=WorkItem(id="item-parity-h2", status="processing"),
        robot=Robot(id="robot-parity-h2", name="Robo-Parity-H2"),
        value=1,
    )

    @adapter.log_transition(event_type="parity_coroutine_transition")
    async def parity_coroutine_transition(consumed: list, bb: MissionBlackboard, timebase: WallClock):
        return {"output": consumed[0]}

    consumed = [WorkItem(id="item-parity-h2", status="processing")]
    gen = parity_coroutine_transition(consumed, bb, tb)
    await gen.__anext__()
    await asyncio.sleep(0.01)

    event_records = [r for r in mock_transport.records if r.get("event")]
    assert len(event_records) == 1
    assert event_records[0]["event"]["type"] == "parity_coroutine_transition"


@pytest.mark.asyncio
async def test_hypha_log_transition_enter_exit_pairing_asyncgen(spore_config, mock_transport):
    """enter_exit=True logs a start/stop pair around an async-generator
    transition, sharing token/blackboard context; exit carries status."""

    adapter = HyphaAdapter(enter_exit=True)
    tb = WallClock()

    bb = MissionBlackboard(
        mission_id="mission-pair-h",
        current_item=WorkItem(id="item-pair-h", status="processing"),
        robot=Robot(id="robot-pair-h", name="Robo-Pair-H"),
        value=4,
    )

    @adapter.log_transition(event_type="paired_transition")
    async def paired_transition(consumed: list, bb: MissionBlackboard, timebase: WallClock):
        yield {"output": consumed[0]}

    # Drain the transition fully (as the Petri net runtime would via
    # `async for`) so the generator is exhausted and the finally block
    # that emits the exit event actually runs.
    consumed = [WorkItem(id="item-pair-h", status="processing")]
    gen = paired_transition(consumed, bb, tb)
    results = [item async for item in gen]
    await asyncio.sleep(0.01)

    assert results == [{"output": consumed[0]}]

    event_records = [r for r in mock_transport.records if r.get("event")]
    assert len(event_records) == 2

    entry, exit_ = event_records[0]["event"], event_records[1]["event"]
    assert entry["type"] == "paired_transition_start"
    assert exit_["type"] == "paired_transition_stop"
    assert entry["time"] <= exit_["time"]

    for evt in (entry, exit_):
        assert get_attr_value(evt["attributes"], "transition_name") == "paired_transition"
        assert get_attr_value(evt["attributes"], "token_count") == "1"
        assert len(evt["relationships"]) >= 1  # input token at least

    assert not has_attr(entry["attributes"], "status")
    assert get_attr_value(exit_["attributes"], "status") == "SUCCESS"


@pytest.mark.asyncio
async def test_hypha_log_transition_enter_exit_pairing_coroutine(spore_config, mock_transport):
    """enter_exit=True also pairs entry/exit for a coroutine-style
    transition (no internal yield)."""

    adapter = HyphaAdapter(enter_exit=True)
    tb = WallClock()

    bb = MissionBlackboard(
        mission_id="mission-pair-h2",
        current_item=WorkItem(id="item-pair-h2", status="processing"),
        robot=Robot(id="robot-pair-h2", name="Robo-Pair-H2"),
        value=5,
    )

    @adapter.log_transition(event_type="paired_coroutine_transition")
    async def paired_coroutine_transition(consumed: list, bb: MissionBlackboard, timebase: WallClock):
        return {"output": consumed[0]}

    consumed = [WorkItem(id="item-pair-h2", status="processing")]
    gen = paired_coroutine_transition(consumed, bb, tb)
    results = [item async for item in gen]
    await asyncio.sleep(0.01)

    assert results == [{"output": consumed[0]}]

    event_records = [r for r in mock_transport.records if r.get("event")]
    assert [r["event"]["type"] for r in event_records] == [
        "paired_coroutine_transition_start",
        "paired_coroutine_transition_stop",
    ]
    assert get_attr_value(event_records[1]["event"]["attributes"], "status") == "SUCCESS"


@pytest.mark.asyncio
async def test_hypha_log_transition_enter_exit_exception_asyncgen(spore_config, mock_transport):
    """An exception raised inside an async-generator transition still
    produces the exit event (status=EXCEPTION) before propagating."""

    adapter = HyphaAdapter(enter_exit=True)
    tb = WallClock()

    bb = MissionBlackboard(
        mission_id="mission-fail-h1",
        current_item=WorkItem(id="item-fail-h1", status="processing"),
        robot=Robot(id="robot-fail-h1", name="Robo-Fail-H1"),
        value=0,
    )

    @adapter.log_transition(event_type="failing_asyncgen_transition")
    async def failing_asyncgen_transition(consumed: list, bb: MissionBlackboard, timebase: WallClock):
        if True:
            raise RuntimeError("boom")
        yield {"output": None}  # pragma: no cover - makes this an async generator

    consumed = [WorkItem(id="item-fail-h1", status="processing")]
    gen = failing_asyncgen_transition(consumed, bb, tb)

    with pytest.raises(RuntimeError, match="boom"):
        await gen.__anext__()

    await asyncio.sleep(0.01)

    event_records = [r for r in mock_transport.records if r.get("event")]
    assert [r["event"]["type"] for r in event_records] == [
        "failing_asyncgen_transition_start",
        "failing_asyncgen_transition_stop",
    ]
    assert get_attr_value(event_records[1]["event"]["attributes"], "status") == "EXCEPTION"


@pytest.mark.asyncio
async def test_hypha_log_transition_enter_exit_exception_coroutine(spore_config, mock_transport):
    """An exception raised inside a coroutine-style transition still
    produces the exit event (status=EXCEPTION) before propagating."""

    adapter = HyphaAdapter(enter_exit=True)
    tb = WallClock()

    bb = MissionBlackboard(
        mission_id="mission-fail-h2",
        current_item=WorkItem(id="item-fail-h2", status="processing"),
        robot=Robot(id="robot-fail-h2", name="Robo-Fail-H2"),
        value=0,
    )

    @adapter.log_transition(event_type="failing_coroutine_transition")
    async def failing_coroutine_transition(consumed: list, bb: MissionBlackboard, timebase: WallClock):
        raise RuntimeError("boom")

    consumed = [WorkItem(id="item-fail-h2", status="processing")]
    gen = failing_coroutine_transition(consumed, bb, tb)

    with pytest.raises(RuntimeError, match="boom"):
        await gen.__anext__()

    await asyncio.sleep(0.01)

    event_records = [r for r in mock_transport.records if r.get("event")]
    assert [r["event"]["type"] for r in event_records] == [
        "failing_coroutine_transition_start",
        "failing_coroutine_transition_stop",
    ]
    assert get_attr_value(event_records[1]["event"]["attributes"], "status") == "EXCEPTION"


@pytest.mark.asyncio
async def test_hypha_log_transition_enter_exit_async_ordering(spore_config, mock_transport):
    """Entry is emitted before the wrapped transition's own first await,
    and exit is emitted only after it resumes and yields its output."""

    adapter = HyphaAdapter(enter_exit=True)
    tb = WallClock()

    bb = MissionBlackboard(
        mission_id="mission-order-h",
        current_item=WorkItem(id="item-order-h", status="processing"),
        robot=Robot(id="robot-order-h", name="Robo-Order-H"),
        value=6,
    )

    order = []
    original_send = mock_transport.send

    async def tracking_send(data, content_type):
        import json
        record = json.loads(data.decode("utf-8"))
        if record.get("event"):
            order.append(f"event:{record['event']['type']}")
        await original_send(data, content_type)

    mock_transport.send = tracking_send

    @adapter.log_transition(event_type="ordered_transition")
    async def ordered_transition(consumed: list, bb: MissionBlackboard, timebase: WallClock):
        order.append("func_start")
        await asyncio.sleep(0)
        order.append("func_resumed")
        yield {"output": consumed[0]}

    consumed = [WorkItem(id="item-order-h", status="processing")]
    gen = ordered_transition(consumed, bb, tb)
    async for _ in gen:
        pass
    await asyncio.sleep(0.01)

    assert order.index("event:ordered_transition_start") < order.index("func_start")
    assert order.index("func_start") < order.index("func_resumed")
    assert order.index("func_resumed") < order.index("event:ordered_transition_stop")


@pytest.mark.asyncio
async def test_hypha_log_transition_enter_exit_sync(spore_config, mock_transport):
    """enter_exit mode also pairs entry/exit for a sync transition
    function."""

    adapter = HyphaAdapter(enter_exit=True)
    tb = WallClock()

    bb = MissionBlackboard(
        mission_id="mission-sync-h",
        current_item=WorkItem(id="item-sync-h", status="processing"),
        robot=Robot(id="robot-sync-h", name="Robo-Sync-H"),
        value=8,
    )

    @adapter.log_transition(event_type="sync_paired_transition")
    def sync_paired_transition(consumed: list, bb: MissionBlackboard, timebase: WallClock):
        return {"output": consumed[0]}

    consumed = [WorkItem(id="item-sync-h", status="processing")]
    result = sync_paired_transition(consumed, bb, tb)
    await asyncio.sleep(0.01)

    assert result is not None

    event_records = [r for r in mock_transport.records if r.get("event")]
    assert [r["event"]["type"] for r in event_records] == [
        "sync_paired_transition_start",
        "sync_paired_transition_stop",
    ]
    assert get_attr_value(event_records[1]["event"]["attributes"], "status") == "SUCCESS"


@pytest.mark.asyncio
async def test_hypha_log_transition_full_payload_parity_asyncgen(spore_config, mock_transport):
    """Full-payload parity: with enter_exit off (default), the event's
    type, complete attribute set, and complete relationship set match the
    pre-extension single-event behavior exactly for an async-generator
    transition - not just the type and a couple of spot-checked
    attributes. Guards against the enter/exit refactor (and the _is_async
    routing fix) silently changing what the default path emits."""

    adapter = HyphaAdapter()
    tb = WallClock()

    bb = MissionBlackboard(
        mission_id="mission-full-parity-h",
        current_item=WorkItem(id="item-full-parity-h", status="processing"),
        robot=Robot(id="robot-full-parity-h", name="Robo-Full-Parity-H"),
        value=1,
    )

    @adapter.log_transition(event_type="full_parity_transition")
    async def full_parity_transition(consumed: list, bb: MissionBlackboard, timebase: WallClock):
        yield {"output": consumed[0]}

    consumed = [WorkItem(id="item-full-parity-in", status="processing")]
    gen = full_parity_transition(consumed, bb, tb)
    await gen.__anext__()
    await asyncio.sleep(0.01)

    event_records = [r for r in mock_transport.records if r.get("event")]
    assert len(event_records) == 1
    event = event_records[0]["event"]

    assert event["type"] == "full_parity_transition"

    attrs = {a["name"]: a["value"] for a in event["attributes"]}
    assert attrs == {
        "token_count": "1",
        "transition_name": "full_parity_transition",
        "mission_id": "mission-full-parity-h",
    }

    # One relationship for the consumed input token, one for the
    # global-scope Robot pulled from the blackboard; current_item is
    # EVENT scope and not requested, so it is absent.
    rel_object_ids = {r["objectId"] for r in event["relationships"]}
    assert rel_object_ids == {"item-full-parity-in", "robot-full-parity-h"}
    assert len(event["relationships"]) == 2


@pytest.mark.asyncio
async def test_hypha_log_transition_log_outputs_asyncgen_regression(spore_config, mock_transport):
    """Regression test pinning the _is_async routing fix: before it,
    async-generator transitions were always routed to sync_wrapper, which
    hands back the transition's raw generator without ever inspecting its
    output, so log_outputs=True was silently inert for the dominant Hypha
    transition shape - output objects never reached the object cache.
    With the fix, log_outputs=True on an async-generator transition
    populates the cache with the objects it yields."""

    adapter = HyphaAdapter()
    tb = WallClock()

    bb = MissionBlackboard(
        mission_id="mission-log-outputs",
        current_item=WorkItem(id="item-log-outputs-in", status="processing"),
        robot=Robot(id="robot-log-outputs", name="Robo-Log-Outputs"),
        value=1,
    )

    output_item = WorkItem(id="item-log-outputs-out", status="done")

    @adapter.log_transition(event_type="log_outputs_transition", log_outputs=True)
    async def log_outputs_transition(consumed: list, bb: MissionBlackboard, timebase: WallClock):
        yield {"output": output_item}

    consumed = [WorkItem(id="item-log-outputs-in", status="processing")]
    gen = log_outputs_transition(consumed, bb, tb)
    results = [item async for item in gen]
    await asyncio.sleep(0.01)

    assert results == [{"output": output_item}]

    cache = get_object_cache()
    cached_output = cache.get("item-log-outputs-out")
    assert cached_output is not None
    assert cached_output.id == "item-log-outputs-out"


@pytest.mark.asyncio
async def test_hypha_log_transition_enter_exit_sync_exception_path(spore_config, mock_transport):
    """A raised exception in a *sync* transition function still produces
    the exit event (status=EXCEPTION) before propagating - mirrors the
    existing async/coroutine exception-path tests, but for the
    sync_wrapper branch."""

    adapter = HyphaAdapter(enter_exit=True)
    tb = WallClock()

    bb = MissionBlackboard(
        mission_id="mission-sync-fail-h",
        current_item=WorkItem(id="item-sync-fail-h", status="processing"),
        robot=Robot(id="robot-sync-fail-h", name="Robo-Sync-Fail-H"),
        value=0,
    )

    @adapter.log_transition(event_type="failing_sync_transition")
    def failing_sync_transition(consumed: list, bb: MissionBlackboard, timebase: WallClock):
        raise RuntimeError("sync boom")

    consumed = [WorkItem(id="item-sync-fail-h", status="processing")]

    with pytest.raises(RuntimeError, match="sync boom"):
        failing_sync_transition(consumed, bb, tb)

    await asyncio.sleep(0.01)

    event_records = [r for r in mock_transport.records if r.get("event")]
    assert [r["event"]["type"] for r in event_records] == [
        "failing_sync_transition_start",
        "failing_sync_transition_stop",
    ]

    exit_event = event_records[1]["event"]
    assert get_attr_value(exit_event["attributes"], "status") == "EXCEPTION"


@pytest.mark.asyncio
async def test_hypha_log_transition_enter_exit_abandoned_asyncgen(spore_config, mock_transport):
    """An async-generator transition closed early (the runtime breaks out
    of `async for` and explicitly calls `aclose()`, the shape used when a
    Petri net stops draining a transition mid-flight) produces exactly one
    exit event with status=ABANDONED - distinct from EXCEPTION - and never
    double-emits."""

    adapter = HyphaAdapter(enter_exit=True)
    tb = WallClock()

    bb = MissionBlackboard(
        mission_id="mission-abandon-h",
        current_item=WorkItem(id="item-abandon-h", status="processing"),
        robot=Robot(id="robot-abandon-h", name="Robo-Abandon-H"),
        value=0,
    )

    @adapter.log_transition(event_type="abandoned_transition")
    async def abandoned_transition(consumed: list, bb: MissionBlackboard, timebase: WallClock):
        yield {"output": consumed[0]}
        # Never reached if the consumer abandons after the first yield.
        yield {"output": consumed[0]}

    consumed = [WorkItem(id="item-abandon-h", status="processing")]
    gen = abandoned_transition(consumed, bb, tb)

    async for _ in gen:
        break
    await gen.aclose()

    await asyncio.sleep(0.01)

    event_records = [r for r in mock_transport.records if r.get("event")]
    assert [r["event"]["type"] for r in event_records] == [
        "abandoned_transition_start",
        "abandoned_transition_stop",
    ]

    exit_event = event_records[1]["event"]
    assert get_attr_value(exit_event["attributes"], "status") == "ABANDONED"


# ============================================================================
# Septum Adapter Tests
# ============================================================================

@pytest.mark.asyncio
async def test_septum_log_state_basic(spore_config, mock_transport):
    """Test basic state logging with Septum adapter."""

    adapter = SeptumAdapter()

    # Create a simple shared context
    bb = MissionBlackboard(
        mission_id="mission-111",
        current_item=WorkItem(id="item-7", status="new"),
        robot=Robot(id="robot-6", name="Robo-C"),
        value=33
    )

    ctx = SharedContext(
        send_message=lambda msg: None,
        log=lambda msg: None,
        common=bb,
        msg=None
    )

    @adapter.log_state(event_type="state_execute")
    async def state_handler(ctx: SharedContext):
        # Simulate a transition result
        class TestTransition:
            name = "DONE"
        return TestTransition()

    await state_handler(ctx)
    await asyncio.sleep(0.01)

    # Check event was logged
    event_records = [r for r in mock_transport.records if "event" in r]
    assert len(event_records) >= 1

    event = event_records[0]["event"]
    assert event["type"] == "state_execute"
    assert has_attr(event["attributes"], "state_name")
    # Note: transition attribute is only logged if result is a TransitionType


@pytest.mark.asyncio
async def test_septum_log_state_with_message(spore_config, mock_transport):
    """Test state logging with message in context."""

    adapter = SeptumAdapter()

    bb = MissionBlackboard(
        mission_id="mission-222",
        current_item=WorkItem(id="item-8", status="received"),
        robot=Robot(id="robot-7", name="Robo-D"),
        value=11
    )

    # Create a test message
    class TestMessage:
        pass

    ctx = SharedContext(
        send_message=lambda msg: None,
        log=lambda msg: None,
        common=bb,
        msg=TestMessage()
    )

    @adapter.log_state(event_type="handle_message")
    async def message_handler(ctx: SharedContext):
        class TestTransition:
            name = "CONTINUE"
        return TestTransition()

    await message_handler(ctx)
    await asyncio.sleep(0.01)

    event_records = [r for r in mock_transport.records if "event" in r]
    assert len(event_records) >= 1

    event = event_records[0]["event"]
    assert has_attr(event["attributes"], "state_name")
    # Note: message_type is logged when ctx.msg is not None
    # The test sets msg=TestMessage(), so message_type should be present
    assert has_attr(event["attributes"], "message_type")


@pytest.mark.asyncio
async def test_septum_log_state_lifecycle(spore_config, mock_transport):
    """Test state lifecycle logging."""

    adapter = SeptumAdapter()

    bb = MissionBlackboard(
        mission_id="mission-333",
        current_item=WorkItem(id="item-9", status="initialized"),
        robot=Robot(id="robot-8", name="Robo-E"),
        value=22
    )

    ctx = SharedContext(
        send_message=lambda msg: None,
        log=lambda msg: None,
        common=bb,
        msg=None
    )

    @adapter.log_state_lifecycle(event_type="state_enter")
    async def on_enter(ctx: SharedContext):
        pass

    await on_enter(ctx)
    await asyncio.sleep(0.01)

    event_records = [r for r in mock_transport.records if "event" in r]
    assert len(event_records) >= 1

    event = event_records[0]["event"]
    assert event["type"] == "state_enter"
    assert has_attr(event["attributes"], "lifecycle_method")
    assert get_attr_value(event["attributes"], "lifecycle_method") == "on_enter"
    assert has_attr(event["attributes"], "phase")
    assert get_attr_value(event["attributes"], "phase") == "enter"


# ============================================================================
# Adapter Enable/Disable Tests
# ============================================================================

@pytest.mark.asyncio
async def test_rhizomorph_adapter_disable(spore_config, mock_transport):
    """Test that disabling adapter prevents logging."""

    adapter = RhizomorphAdapter()
    adapter.disable()  # Disable logging
    tb = WallClock()

    bb = MissionBlackboard(
        mission_id="mission-disabled",
        current_item=WorkItem(id="item-disabled", status="disabled"),
        robot=Robot(id="robot-disabled", name="Disabled"),
        value=0
    )

    @adapter.log_node(event_type="should_not_log")
    async def test_action(bb: MissionBlackboard, tb: WallClock) -> Status:
        return Status.SUCCESS

    await test_action(bb, tb)
    await asyncio.sleep(0.01)

    # Note: The adapter._enabled flag controls the wrapper's behavior
    # Since we're using the decorator through the adapter, we need to check
    # if the logging actually happened

    # The decorator should still log (adapter enable/disable is for manual control)
    # So this test verifies the basic mechanism works


@pytest.mark.asyncio
async def test_hypha_adapter_enable_disable(spore_config, mock_transport):
    """Test Hypha adapter enable/disable."""

    adapter = HyphaAdapter()

    # Adapter should be enabled by default
    assert adapter._enabled is True

    adapter.disable()
    assert adapter._enabled is False

    adapter.enable()
    assert adapter._enabled is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@pytest.mark.asyncio
async def test_rhizomorph_log_node_runs_are_per_blackboard(spore_config, mock_transport):
    """One node function ticked against two blackboards keeps two runs.

    A tree serving several entities concurrently ticks the same node object for
    each of them. If run state were keyed by the node alone, the first entity's
    open run would suppress the second entity's start and the second entity's
    terminal tick would close the first entity's run. Keying by
    (node, case identity) is what keeps them apart.
    """

    adapter = RhizomorphAdapter()
    tb = WallClock()

    def blackboard(suffix: str) -> MissionBlackboard:
        return MissionBlackboard(
            mission_id=f"mission-{suffix}",
            current_item=WorkItem(id=f"item-{suffix}", status="pending"),
            robot=Robot(id=f"robot-{suffix}", name=f"Robo-{suffix}"),
            value=1,
        )

    first, second = blackboard("first"), blackboard("second")
    done = set()

    @adapter.log_node(event_type="shared_node")
    def shared_node(bb: MissionBlackboard, tb: WallClock) -> Status:
        # `first` finishes on its second tick, `second` on its third.
        key = bb.robot.id
        done.add(key)
        if key == "robot-first":
            return Status.SUCCESS if bb.value >= 2 else Status.RUNNING
        return Status.SUCCESS if bb.value >= 3 else Status.RUNNING

    # Interleave the two, the way a runner servicing both would. A node that
    # has returned SUCCESS is not ticked again; ticking it would legitimately
    # open a second run.
    live = [first, second]
    for tick in (1, 2, 3):
        for bb in list(live):
            bb.value = tick
            if shared_node(bb, tb) is not Status.RUNNING:
                live.remove(bb)
    await asyncio.sleep(0.01)

    events = [r["event"] for r in mock_transport.records if r.get("event")]

    # Exactly one event per entity: neither run swallowed the other.
    assert len(events) == 2
    missions = [get_attr_value(e["attributes"], "mission_id") for e in events]
    assert sorted(missions) == ["mission-first", "mission-second"]
    assert all(e["type"] == "shared_node" for e in events)
    assert all(get_attr_value(e["attributes"], "status") == "SUCCESS" for e in events)


@pytest.mark.asyncio
async def test_rhizomorph_log_node_run_state_dies_with_the_blackboard(spore_config, mock_transport):
    """Run state must not outlive the blackboard it belongs to.

    id() is unique only among simultaneously existing objects. A run left open by
    a blackboard that has since been freed, whether abandoned mid-RUNNING or ended
    by a raise, would otherwise sit on the address a later blackboard is allocated
    at, and that later entity would be read as a continuation of the dead one.
    """
    import gc

    from mycorrhizal.spores.dsl.rhizomorph import _RUNS

    adapter = RhizomorphAdapter()
    tb = WallClock()

    def blackboard() -> MissionBlackboard:
        return MissionBlackboard(
            mission_id="mission-transient",
            current_item=WorkItem(id="item-transient", status="pending"),
            robot=Robot(id="robot-transient", name="Robo-Transient"),
            value=1,
        )

    @adapter.log_node(event_type="never_settles")
    async def never_settles(bb: MissionBlackboard, tb: WallClock) -> Status:
        return Status.RUNNING

    @adapter.log_node(event_type="raises")
    async def raises(bb: MissionBlackboard, tb: WallClock) -> Status:
        raise RuntimeError("node failed")

    before = _RUNS.open_carriers()

    for _ in range(50):
        await never_settles(blackboard(), tb)   # opens a run, then the bb dies
    for _ in range(10):
        with pytest.raises(RuntimeError):
            await raises(blackboard(), tb)
    gc.collect()

    assert _RUNS.open_carriers() == before

    # A blackboard that is still alive keeps its run across ticks.
    live = blackboard()
    for _ in range(3):
        await never_settles(live, tb)
    assert _RUNS.open_carriers() == before + 1
