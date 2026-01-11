#!/usr/bin/env python3
"""
Classic "search robot" example for Rhizomorph
"""

import asyncio

from mycorrhizal.rhizomorph.core import bt, Runner, Status, ExceptionPolicy
from mycorrhizal.common.timebase import Timebase

# ======================================================================================
# Demo
# ======================================================================================

class DemoBlackboard:
    """Typed blackboard for demo trees."""
    def __init__(self):
        self.battery: int = 50
        self.counter: int = 0
        self.goto_progress: int = 0
        self.scan_attempts: int = 0
        self.telemetry_count: int = 0
        self.engage_steps: int = 0

@bt.tree
def Engage():
    @bt.action
    async def engage(bb: DemoBlackboard) -> Status:
        bb.engage_steps += 1
        return Status.SUCCESS if bb.engage_steps >= 10 else Status.RUNNING

    @bt.condition
    def threat_detected(bb: DemoBlackboard) -> bool:
        return bb.counter % 3 == 0

    @bt.condition
    def battery_ok(bb: DemoBlackboard) -> bool:
        return bb.battery > 20

    @bt.root
    @bt.sequence(memory=False)
    def engage_threat():
        yield threat_detected
        yield bt.failer().gate(battery_ok).timeout(0.12)(engage)

@bt.tree
def Demo():
    @bt.condition
    def has_waypoints(bb: DemoBlackboard) -> bool:
        return (bb.counter % 2) == 1


    @bt.action
    async def go_to_next(bb: DemoBlackboard) -> Status:
        bb.goto_progress += 1
        return Status.SUCCESS if bb.goto_progress >= 3 else Status.RUNNING

    @bt.action
    async def scan_area(bb: DemoBlackboard) -> Status:
        bb.scan_attempts += 1
        return Status.SUCCESS if bb.scan_attempts >= 2 else Status.FAILURE

    @bt.action
    async def telemetry_push(bb: DemoBlackboard) -> Status:
        bb.telemetry_count += 1
        return Status.SUCCESS

    @bt.sequence(memory=True)
    def patrol():
        yield has_waypoints
        yield go_to_next
        yield bt.succeeder().retry(3).timeout(1.0)(scan_area)

    @bt.root
    @bt.selector(memory=True, reactive=True)
    def root():
        yield bt.subtree(Engage)
        yield patrol
        yield bt.failer().ratelimit(hz=5.0)(telemetry_push)


async def _demo():
    bb = DemoBlackboard()
    runner = Runner(
        Demo,
        bb=bb,
        exception_policy=ExceptionPolicy.LOG_AND_CONTINUE
    )

    for i in range(1, 10):
        bb.counter = i
        st = await runner.tick()
        print(
            f"[tick {i}] status={st.name} "
            f"(goto={bb.goto_progress}, scan={bb.scan_attempts}, "
            f"eng={bb.engage_steps}, tel={bb.telemetry_count})"
        )
        await asyncio.sleep(0)

    from mycorrhizal.rhizomorph.util import to_mermaid

    print("\n--- Mermaid ---")
    print(to_mermaid(Demo))

asyncio.run(_demo())
