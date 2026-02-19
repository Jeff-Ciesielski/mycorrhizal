#!/usr/bin/env python3
"""
BT-to-PN Compiler Demo - Mycelium Integration

This example demonstrates the BT-to-PN compiler integrated into Mycelium.
The compiler transforms Rhizomorph behavior trees into Hypha Petri nets
that can be executed within the Mycelium unified orchestration layer.

Example usage:
    uv run python examples/mycelium/bt_compiler_demo.py
"""

import asyncio
from pydantic import BaseModel
from typing import List

from mycorrhizal.rhizomorph.core import bt, Status, Runner as BTRunner
from mycorrhizal.mycelium import compile_bt_to_pn, PNRunner
from mycorrhizal.common.timebase import MonotonicClock


# ============================================================================
# Blackboard
# ============================================================================


class DemoBlackboard(BaseModel):
    """Blackboard for the demo."""
    battery_level: float = 100.0
    action_taken: str = ""
    events: List[str] = []
    retry_count: int = 0


# ============================================================================
# Behavior Tree Definition
# =============================================================================


@bt.tree
def RobotControllerBT():
    """
    A behavior tree that selects an action based on battery level.

    Structure:
    - Selector (try in order, stop at first success)
        - Sequence: battery > 75% -> full_speed
        - Sequence: battery > 25% -> slow_speed
        - emergency_stop (fallback)
    """

    @bt.condition
    def battery_above_75(bb: DemoBlackboard) -> bool:
        """Check if battery is above 75%."""
        bb.events.append("check_75%")
        return bb.battery_level > 75

    @bt.action
    async def full_speed(bb: DemoBlackboard) -> Status:
        """Run at full speed."""
        bb.action_taken = "full_speed"
        bb.events.append("full_speed")
        return Status.SUCCESS

    @bt.condition
    def battery_above_25(bb: DemoBlackboard) -> bool:
        """Check if battery is above 25%."""
        bb.events.append("check_25%")
        return bb.battery_level > 25

    @bt.action
    async def slow_speed(bb: DemoBlackboard) -> Status:
        """Run at slow speed."""
        bb.action_taken = "slow_speed"
        bb.events.append("slow_speed")
        return Status.SUCCESS

    @bt.action
    async def emergency_stop(bb: DemoBlackboard) -> Status:
        """Emergency stop."""
        bb.action_taken = "emergency_stop"
        bb.events.append("emergency_stop")
        return Status.SUCCESS

    @bt.sequence
    def aggressive_mode():
        """High battery mode."""
        yield battery_above_75
        yield full_speed

    @bt.sequence
    def conservative_mode():
        """Medium battery mode."""
        yield battery_above_25
        yield slow_speed

    @bt.root
    @bt.selector
    def root():
        """Select mode based on battery level."""
        yield aggressive_mode
        yield conservative_mode
        yield emergency_stop


# ============================================================================
# Demo Functions
# =============================================================================


async def run_bt():
    """Run the behavior tree directly."""
    print("\n" + "=" * 60)
    print("Running Behavior Tree (Native)")
    print("=" * 60)

    bb = DemoBlackboard(battery_level=80.0)
    tb = MonotonicClock()
    runner = BTRunner(tree=RobotControllerBT, bb=bb, tb=tb)

    result = await runner.tick()

    print(f"  Result: {result}")
    print(f"  Action taken: {bb.action_taken}")
    print(f"  Events: {' -> '.join(bb.events)}")


async def run_compiled_pn():
    """Run the compiled Petri net."""
    print("\n" + "=" * 60)
    print("Running Compiled Petri Net")
    print("=" * 60)

    bb = DemoBlackboard(battery_level=80.0)
    tb = MonotonicClock()

    # Compile the behavior tree to a Petri net
    pn_spec = compile_bt_to_pn(RobotControllerBT, name="RobotController")

    # Show the mermaid diagram for visualization
    print("\n  Compiled Petri Net Diagram (Mermaid):")
    print(pn_spec.to_mermaid())
    print("---------------------------------")

    # Create a wrapper function with _spec attribute
    def net_func():
        pass
    net_func._spec = pn_spec

    # Run with PNRunner
    runner = PNRunner(net_func, blackboard=bb)
    await runner.start(tb)

    # Inject a token to start execution
    entry_key = (pn_spec.name, "entry")
    runner.runtime.places[entry_key].add_token("start")

    # Wait for completion
    await asyncio.sleep(0.2)

    # Stop
    await runner.stop(timeout=2)

    print(f"  Action taken: {bb.action_taken}")
    print(f"  Events: {' -> '.join(bb.events)}")


async def test_different_battery_levels():
    """Test both BT and PN with different battery levels."""
    print("\n" + "=" * 60)
    print("Comparison Test: BT vs PN")
    print("=" * 60)

    test_cases = [
        (80.0, "full_speed"),
        (50.0, "slow_speed"),
        (10.0, "emergency_stop"),
    ]

    for battery, expected_action in test_cases:
        print(f"\n  Battery: {battery}%")

        # Run BT
        bb_bt = DemoBlackboard(battery_level=battery)
        tb_bt = MonotonicClock()
        runner_bt = BTRunner(tree=RobotControllerBT, bb=bb_bt, tb=tb_bt)
        await runner_bt.tick()

        # Run PN
        bb_pn = DemoBlackboard(battery_level=battery)
        tb_pn = MonotonicClock()

        pn_spec = compile_bt_to_pn(RobotControllerBT)

        def net_func():
            pass
        net_func._spec = pn_spec

        runner_pn = PNRunner(net_func, blackboard=bb_pn)
        await runner_pn.start(tb_pn)

        entry_key = (pn_spec.name, "entry")
        runner_pn.runtime.places[entry_key].add_token("start")

        await asyncio.sleep(0.2)
        await runner_pn.stop(timeout=2)

        # Compare
        match = bb_bt.action_taken == bb_pn.action_taken == expected_action
        status = "OK" if match else "FAIL"
        print(f"    BT: {bb_bt.action_taken}")
        print(f"    PN: {bb_pn.action_taken}")
        print(f"    Expected: {expected_action} [{status}]")


async def main():
    """Main demo function."""
    print("\n" + "=" * 60)
    print("BT-to-PN Compiler - Mycelium Integration Demo")
    print("=" * 60)

    try:
        await run_bt()
        await run_compiled_pn()
        await test_different_battery_levels()

        print("\n" + "=" * 60)
        print("Demo Complete")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
