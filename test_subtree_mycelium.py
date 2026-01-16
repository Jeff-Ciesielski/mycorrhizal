#!/usr/bin/env python3
"""Test if subtrees work in Mycelium"""

import sys
sys.path.insert(0, "src")

import asyncio
from enum import Enum, auto
from pydantic import BaseModel

from mycorrhizal.enoki.core import enoki, LabeledTransition, StateConfiguration
from mycorrhizal.mycelium import tree, fsm, Action, Sequence, Selector, root, TreeRunner
from mycorrhizal.rhizomorph.core import bt, Status, Runner
from mycorrhizal.common.timebase import MonotonicClock


# Define a standalone BT tree for subtree use
@bt.tree
def SubTreeTask():
    @bt.action
    async def do_work(bb):
        print("[Subtree] Doing work...")
        return Status.SUCCESS

    @bt.root
    @Sequence
    def main():
        yield do_work


# Define states
@enoki.state(config=StateConfiguration(can_dwell=True))
def IdleState():
    @enoki.events
    class Events(Enum):
        START = auto()

    @enoki.on_state
    async def on_state(ctx):
        if ctx.common.get("start", False):
            return Events.START
        return None

    @enoki.transitions
    def transitions():
        return [LabeledTransition(Events.START, IdleState)]


# Blackboard
class TestBB(BaseModel):
    start: bool = False


# Mycelium tree that uses the subtree
@tree
def MyceliumWithSubtree():
    @fsm(initial=IdleState, auto_tick=False)
    def my_fsm():
        pass

    @Action
    async def trigger_fsm(bb, tb, tree):
        tree.fsms.my_fsm.send_message(IdleState.Events.START)
        return Status.SUCCESS

    @Action
    async def check_fsm(bb, tb, tree):
        state = tree.fsms.my_fsm.current_state
        print(f"FSM state: {state.name if state else 'None'}")
        return Status.SUCCESS

    @root
    @Sequence
    def main():
        yield trigger_fsm
        yield check_fsm
        yield bt.subtree(SubTreeTask)  # Use the BT subtree


async def main():
    bb = TestBB()
    tb = MonotonicClock()

    runner = TreeRunner(MyceliumWithSubtree, bb=bb, tb=tb)

    print("Running Mycelium tree with BT subtree...")
    result = await runner.tick()
    print(f"Result: {result}")

    print("\nMermaid diagram:")
    print(runner.to_mermaid())


if __name__ == "__main__":
    asyncio.run(main())
