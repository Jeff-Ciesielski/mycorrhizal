#!/usr/bin/env python3
"""
Tests for FSM-in-PN (Finite State Machine in Petri Net) Mermaid diagram generation.
"""

import pytest
from enum import Enum, auto
from pydantic import BaseModel

from mycorrhizal.mycelium import pn, PNRunner, PlaceType
from mycorrhizal.mycelium.core import state, events, on_state, transitions, LabeledTransition


# ==================================================================================
# Test Fixtures
# ==================================================================================


class Blackboard(BaseModel):
    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}
    state: str = "initial"


# ==================================================================================
# FSM Definitions
# ==================================================================================


@state()
def ClosedState():
    @events
    class Events(Enum):
        OPEN = auto()
        SUCCESS = auto()

    @on_state
    async def handle_state(ctx):
        return ClosedState.Events.SUCCESS

    @transitions
    def state_transitions():
        return [
            LabeledTransition(Events.OPEN, None),  # Will be replaced
            LabeledTransition(Events.SUCCESS, None),  # Will be replaced
        ]


@state()
def OpenState():
    @events
    class Events(Enum):
        CLOSE = auto()

    @on_state
    async def handle_state(ctx):
        return OpenState.Events.CLOSE

    @transitions
    def state_transitions():
        return [
            LabeledTransition(Events.CLOSE, None),  # Will be replaced
        ]


# ==================================================================================
# Mermaid Diagram Tests
# ==================================================================================


def test_fsm_in_pn_mermaid_diagram():
    """Test that FSM-in-PN transitions generate proper Mermaid diagrams."""

    @pn.net
    def SimpleFSMNet(builder):
        input_q = builder.place("input_q", type=PlaceType.QUEUE)
        success_q = builder.place("success_q", type=PlaceType.QUEUE)
        failure_q = builder.place("failure_q", type=PlaceType.QUEUE)

        # Transition with FSM integration
        @builder.transition(fsm=ClosedState, outputs=[success_q, failure_q])
        async def process_with_fsm(consumed, bb, timebase):
            pass

        builder.arc(input_q, process_with_fsm)
        builder.arc(process_with_fsm, success_q)
        builder.arc(process_with_fsm, failure_q)

    # Create runner
    bb = Blackboard()
    runner = PNRunner(SimpleFSMNet, bb)

    # Generate Mermaid diagram
    mermaid = runner.to_mermaid()

    print("\n" + "=" * 80)
    print("FSM-in-PN Mermaid Diagram")
    print("=" * 80)
    print(mermaid)
    print("=" * 80)

    # Verify the diagram contains key elements
    assert "graph TD" in mermaid, "Mermaid diagram should start with graph TD"
    assert "SimpleFSMNet" in mermaid, "Diagram should contain the net name"
    assert "process_with_fsm" in mermaid, "Diagram should contain the transition name"
    assert "FSM:" in mermaid, "Diagram should indicate FSM integration"
    assert "ClosedState" in mermaid, "Diagram should show the FSM state name"

    # Verify structure
    lines = mermaid.split("\n")

    # Find the transition line with FSM label
    transition_lines = [l for l in lines if "process_with_fsm" in l]
    assert len(transition_lines) > 0, "Should have at least one transition line"

    # Check that the FSM label is properly formatted
    fsm_line = [l for l in transition_lines if "FSM:" in l]
    assert len(fsm_line) > 0, "Transition should have FSM label"


def test_mixed_bt_and_fsm_in_pn_mermaid():
    """Test that a PN with both BT and FSM integrations generates proper Mermaid diagrams."""

    from mycorrhizal.rhizomorph.core import bt, Status

    @bt.tree
    def SimpleBT():
        @bt.action
        async def check(bb):
            return Status.SUCCESS

        @bt.root
        @bt.sequence
        def root():
            yield check

    @pn.net
    def MixedNet(builder):
        input_q = builder.place("input_q", type=PlaceType.QUEUE)
        output_q = builder.place("output_q", type=PlaceType.QUEUE)

        # Transition with BT integration
        @builder.transition(bt=SimpleBT, bt_mode="token", outputs=[output_q])
        async def bt_transition(consumed, bb, timebase):
            pass

        # Transition with FSM integration
        @builder.transition(fsm=ClosedState, outputs=[output_q])
        async def fsm_transition(consumed, bb, timebase):
            pass

        # Vanilla transition
        @builder.transition()
        async def vanilla_transition(consumed, bb, timebase):
            yield {output_q: consumed[0]}

        builder.arc(input_q, bt_transition)
        builder.arc(bt_transition, output_q)
        builder.arc(input_q, fsm_transition)
        builder.arc(fsm_transition, output_q)
        builder.arc(input_q, vanilla_transition)
        builder.arc(vanilla_transition, output_q)

    # Create runner
    bb = Blackboard()
    runner = PNRunner(MixedNet, bb)

    # Generate Mermaid diagram
    mermaid = runner.to_mermaid()

    print("\n" + "=" * 80)
    print("Mixed BT+FSM-in-PN Mermaid Diagram")
    print("=" * 80)
    print(mermaid)
    print("=" * 80)

    # Verify the diagram contains both BT and FSM labels
    assert "BT:" in mermaid, "Diagram should indicate BT integration"
    assert "FSM:" in mermaid, "Diagram should indicate FSM integration"

    # Verify both integrations are shown
    assert "SimpleBT" in mermaid, "Diagram should show BT tree name"
    assert "ClosedState" in mermaid, "Diagram should show FSM state name"


def test_vanilla_pn_mermaid_still_works():
    """Test that vanilla PN (without integrations) still generates diagrams correctly."""

    @pn.net
    def VanillaNet(builder):
        input_q = builder.place("input_q", type=PlaceType.QUEUE)
        output_q = builder.place("output_q", type=PlaceType.QUEUE)

        @builder.transition()
        async def process(consumed, bb, timebase):
            yield {output_q: consumed[0]}

        builder.arc(input_q, process)
        builder.arc(process, output_q)

    # Create runner
    bb = Blackboard()
    runner = PNRunner(VanillaNet, bb)

    # Generate Mermaid diagram
    mermaid = runner.to_mermaid()

    print("\n" + "=" * 80)
    print("Vanilla PN Mermaid Diagram")
    print("=" * 80)
    print(mermaid)
    print("=" * 80)

    # Verify basic structure
    assert "graph TD" in mermaid
    assert "process" in mermaid
    assert "input_q" in mermaid
    assert "output_q" in mermaid

    # Verify no integration labels
    assert "BT:" not in mermaid
    assert "FSM:" not in mermaid


def test_fsm_subgraph_is_connected_to_pn():
    """
    Test that FSM subgraph is properly connected to the PN structure.

    This test verifies the fix for the "disconnected FSM" issue where
    the FSM subgraph was created but arcs pointed to the transition node
    instead of the subgraph, resulting in a disconnected diagram.

    The correct behavior is:
    - Input places should connect to the subgraph (not the transition node)
    - The subgraph should connect to output places (not the transition node)
    - Arcs should use the subgraph ID (transition_name + "_fsm" suffix)
    """

    @state()
    def Validating():
        @events
        class Events(Enum):
            VALID = auto()
            INVALID = auto()

        @on_state
        async def handle_state(ctx):
            return Validating.Events.VALID

        @transitions
        def state_transitions():
            return [
                LabeledTransition(Events.VALID, Processing),
                LabeledTransition(Events.INVALID, Rejected),
            ]

    @state()
    def Processing():
        @events
        class Events(Enum):
            SUCCESS = auto()
            FAILURE = auto()

        @on_state
        async def handle_state(ctx):
            return Processing.Events.SUCCESS

        @transitions
        def state_transitions():
            return [
                LabeledTransition(Events.SUCCESS, Completed),
                LabeledTransition(Events.FAILURE, Failed),
            ]

    @state()
    def Completed():
        @events
        class Events(Enum):
            DONE = auto()

        @on_state
        async def handle_state(ctx):
            return Completed.Events.DONE

        @transitions
        def state_transitions():
            return []

    @state()
    def Failed():
        @events
        class Events(Enum):
            DONE = auto()

        @on_state
        async def handle_state(ctx):
            return Failed.Events.DONE

        @transitions
        def state_transitions():
            return []

    @state()
    def Rejected():
        @events
        class Events(Enum):
            DONE = auto()

        @on_state
        async def handle_state(ctx):
            return Rejected.Events.DONE

        @transitions
        def state_transitions():
            return []

    @pn.net
    def ConnectedFSMNet(builder):
        input_q = builder.place("input_q", type=PlaceType.QUEUE)
        success_q = builder.place("success_q", type=PlaceType.QUEUE)
        failure_q = builder.place("failure_q", type=PlaceType.QUEUE)

        # Transition with FSM integration
        @builder.transition(fsm=Validating, outputs=[success_q, failure_q])
        async def process_with_fsm(consumed, bb, timebase):
            pass

        builder.arc(input_q, process_with_fsm)
        builder.arc(process_with_fsm, success_q)
        builder.arc(process_with_fsm, failure_q)

    # Create runner
    bb = Blackboard()
    runner = PNRunner(ConnectedFSMNet, bb)

    # Generate Mermaid diagram
    mermaid = runner.to_mermaid()

    print("\n" + "=" * 80)
    print("Connected FSM-in-PN Mermaid Diagram")
    print("=" * 80)
    print(mermaid)
    print("=" * 80)

    # Parse the diagram into lines
    lines = mermaid.split("\n")

    # Find the subgraph definition
    subgraph_lines = [l for l in lines if l.strip().startswith("subgraph")]
    assert len(subgraph_lines) == 1, "Should have exactly one FSM subgraph"

    # Extract the subgraph ID
    subgraph_line = subgraph_lines[0]
    # Format: subgraph ConnectedFSMNet.process_with_fsm_fsm["..."]
    subgraph_id = subgraph_line.split()[1].split("[")[0]
    assert subgraph_id == "ConnectedFSMNet.process_with_fsm_fsm", \
        f"Subgraph ID should be 'ConnectedFSMNet.process_with_fsm_fsm', got '{subgraph_id}'"

    # Find arc lines
    arc_lines = [l for l in lines if "-->" in l]

    # Verify input arc connects to subgraph (not transition node)
    input_arcs = [l for l in arc_lines if "input_q -->" in l]
    assert len(input_arcs) == 1, "Should have exactly one input arc"
    input_arc = input_arcs[0]
    # Input arc should point to subgraph ID
    assert f"{subgraph_id}" in input_arc, \
        f"Input arc should point to subgraph '{subgraph_id}', got: {input_arc}"
    # Input arc should NOT point to transition node (without _fsm suffix)
    assert "process_with_fsm -->" not in input_arc, \
        f"Input arc should NOT point to transition node, got: {input_arc}"

    # Verify output arcs connect from subgraph (not transition node)
    output_arcs = [l for l in arc_lines if f"{subgraph_id} -->" in l]
    assert len(output_arcs) == 2, "Should have exactly two output arcs (success and failure)"

    # Both output places should be reachable from the subgraph
    output_place_targets = [l.split("-->")[-1].strip() for l in output_arcs]
    assert any("success_q" in t for t in output_place_targets), \
        "Success queue should be reachable from subgraph"
    assert any("failure_q" in t for t in output_place_targets), \
        "Failure queue should be reachable from subgraph"

    # Verify no arcs point to the transition node (without _fsm suffix)
    # This is the key check - the FSM should be connected, not disconnected
    bad_arcs = [l for l in arc_lines if "process_with_fsm -->" in l and "_fsm" not in l]
    assert len(bad_arcs) == 0, \
        f"No arcs should point to transition node (without _fsm suffix), found: {bad_arcs}"

    # Verify all FSM states are in the subgraph
    assert "Validating" in mermaid, "FSM state 'Validating' should be in diagram"
    assert "Processing" in mermaid, "FSM state 'Processing' should be in diagram"
    assert "Completed" in mermaid, "FSM state 'Completed' should be in diagram"
    assert "Failed" in mermaid, "FSM state 'Failed' should be in diagram"
    assert "Rejected" in mermaid, "FSM state 'Rejected' should be in diagram"

    # Verify state transitions
    assert "VALID" in mermaid, "FSM transition 'VALID' should be in diagram"
    assert "INVALID" in mermaid, "FSM transition 'INVALID' should be in diagram"
    assert "SUCCESS" in mermaid, "FSM transition 'SUCCESS' should be in diagram"
    assert "FAILURE" in mermaid, "FSM transition 'FAILURE' should be in diagram"
