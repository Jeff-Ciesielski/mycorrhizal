#!/usr/bin/env python3
"""
Example: Configuration-Driven Behavior Trees

This demonstrates building behavior trees from configuration files or dictionaries.
Useful for:
- Loading behavior trees from JSON/YAML config files
- A/B testing different tree structures
- Runtime tree modification
- Game AI with editable behavior configs
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


from mycorrhizal.rhizomorph.core import TreeBuilder, Runner, Status


# ============================================================================
# Configuration Structure
# ============================================================================

# Example config that could be loaded from JSON/YAML
ROBOT_CONFIG = {
    "name": "MiningRobot",
    "actions": {
        "move_to_ore": {
            "type": "action",
            "battery_cost": 10,
            "timeout": 5.0,
        },
        "mine_ore": {
            "type": "action",
            "battery_cost": 20,
            "timeout": 10.0,
        },
        "return_to_base": {
            "type": "action",
            "battery_cost": 15,
            "timeout": 8.0,
        },
        "recharge": {
            "type": "action",
            "timeout": 3.0,
        },
        "scan_for_ore": {
            "type": "action",
            "battery_cost": 5,
            "timeout": 5.0,
        },
    },
    "conditions": {
        "has_battery": {
            "type": "condition",
            "threshold": 20,
        },
        "has_ore_target": {
            "type": "condition",
            "check": "target_visible",
        },
        "cargo_full": {
            "type": "condition",
            "threshold": 100,
        },
    },
    "behaviors": {
        "mining_cycle": {
            "type": "sequence",
            "children": ["has_battery", "scan_for_ore", "move_to_ore", "mine_ore"],
        },
        "return_cycle": {
            "type": "sequence",
            "children": ["cargo_full", "return_to_base"],
        },
        "main_behavior": {
            "type": "selector",
            "children": ["mining_cycle", "return_cycle", "recharge"],
        },
    },
}


@dataclass
class RobotBlackboard:
    """State for the mining robot"""
    battery: int = 100
    cargo: int = 0
    target_visible: bool = False
    scans_done: int = 0
    ore_collected: int = 0
    at_base: bool = True
    log: list = field(default_factory=list)


# ============================================================================
# Config-to-Tree Builder
# ============================================================================

class ConfigTreeBuilder:
    """
    Builds behavior trees from configuration dictionaries.

    This pattern enables loading tree definitions from external config files
    (JSON, YAML, database, etc.) and building trees at runtime.
    """

    def __init__(self, config: Dict[str, Any], blackboard_type=RobotBlackboard):
        self.config = config
        self.bb_type = blackboard_type
        self.builder = TreeBuilder(config["name"])

    def build_all_actions(self) -> Dict[str, Any]:
        """Build all action nodes from config"""
        actions = {}

        for name, action_config in self.config.get("actions", {}).items():
            # Create action function
            def make_action(config_name: str, config_dict: Dict):
                async def action(bb: RobotBlackboard) -> Status:
                    cost = config_dict.get("battery_cost", 0)
                    if bb.battery < cost:
                        bb.log.append(f"[{config_name}] Insufficient battery ({bb.battery} < {cost})")
                        return Status.FAILURE

                    bb.battery -= cost
                    bb.log.append(f"[{config_name}] Executing (battery now: {bb.battery})")

                    # Simulate work
                    match config_name:
                        case "move_to_ore":
                            bb.at_base = False
                        case "mine_ore":
                            bb.ore_collected += 10
                            bb.cargo += 10
                        case "return_to_base":
                            bb.at_base = True
                            bb.cargo = 0  # Unload cargo

                    return Status.SUCCESS

                return action

            action_func = make_action(name, action_config)
            action_node = self.builder.action(name, action_func)

            # Apply timeout if configured
            timeout = action_config.get("timeout")
            if timeout:
                action_node = action_node.timeout(timeout)

            actions[name] = action_node

        return actions

    def build_all_conditions(self) -> Dict[str, Any]:
        """Build all condition nodes from config"""
        conditions = {}

        for name, cond_config in self.config.get("conditions", {}).items():
            # Create condition function based on config
            def make_condition(config_name: str, config_dict: Dict):
                if "threshold" in config_dict:
                    threshold = config_dict["threshold"]
                    if config_name == "has_battery":
                        return lambda bb: bb.battery >= threshold
                    if config_name == "cargo_full":
                        return lambda bb: bb.cargo >= threshold

                if "check" in config_dict:
                    check = config_dict["check"]
                    if check == "target_visible":
                        return lambda bb: bb.target_visible

                raise ValueError(f"Unknown condition config: {config_dict}")

            condition_func = make_condition(name, cond_config)
            conditions[name] = self.builder.condition(name, condition_func)

        return conditions

    def build_tree(self) -> Any:
        """
        Build complete tree from config.

        Parses behavior definitions and constructs the tree.
        """
        actions = self.build_all_actions()
        conditions = self.build_all_conditions()

        def build_behavior(name: str) -> Any:
            """Recursively build behavior nodes"""
            if name in actions:
                return actions[name]
            if name in conditions:
                return conditions[name]

            behavior_config = self.config["behaviors"][name]
            behavior_type = behavior_config["type"]

            children = [build_behavior(child) for child in behavior_config.get("children", [])]

            if behavior_type == "sequence":
                return self.builder.sequence(*children, memory=True)
            elif behavior_type == "selector":
                return self.builder.selector(*children, memory=False)
            else:
                raise ValueError(f"Unknown behavior type: {behavior_type}")

        # Build root behavior
        root_behavior = build_behavior("main_behavior")
        return self.builder.build(root_behavior)


# ============================================================================
# Demo Functions
# ============================================================================

async def demo_config_driven_tree():
    """Demonstrate building and running a config-driven tree"""
    print("=" * 60)
    print("Demo: Configuration-Driven Behavior Tree")
    print("=" * 60)

    # Show the config
    print("\nRobot Config:")
    print(json.dumps(ROBOT_CONFIG, indent=2))

    # Build tree from config
    tree_builder = ConfigTreeBuilder(ROBOT_CONFIG)
    tree = tree_builder.build_tree()

    # Initial state
    bb = RobotBlackboard(battery=100, cargo=0)

    print("\nInitial state:")
    print(f"  Battery: {bb.battery}%")
    print(f"  Cargo: {bb.cargo}/100")
    print(f"  At base: {bb.at_base}")
    print("\nRunning behavior tree:\n")

    # Run the tree
    runner = Runner(tree, bb=bb)

    # Simulate multiple mining cycles
    for cycle in range(1, 4):
        print(f"--- Cycle {cycle} ---")

        # Simulate finding ore after scanning
        if bb.scans_done >= 1:
            bb.target_visible = True

        result = await runner.tick_until_complete()

        print(f"\nAfter cycle {cycle}:")
        print(f"  Battery: {bb.battery}%")
        print(f"  Cargo: {bb.cargo}/100")
        print(f"  Ore collected: {bb.ore_collected}")
        print(f"  Result: {result}")

        # Recharge if needed
        if bb.battery < 20:
            print("\n  [SYSTEM] Recharging...")
            bb.battery = 100
            bb.at_base = True
            bb.target_visible = False

        if bb.battery < 30:
            print("\n  [SYSTEM] Low battery, ending demo")
            break

    print(f"\nFinal state:")
    print(f"  Battery: {bb.battery}%")
    print(f"  Total ore collected: {bb.ore_collected}")
    print("\nExecution log:")
    for entry in bb.log:
        print(f"  {entry}")


# ============================================================================
# Dynamic Policy Modification Example
# ============================================================================

async def demo_policy_modification():
    """
    Demonstrate dynamically modifying tree behavior based on policy.

    This simulates a simple RL scenario where policy affects tree structure.
    """
    print("\n" + "=" * 60)
    print("Demo: Dynamic Policy Modification")
    print("=" * 60)

    # Policy represents learned behavior preferences
    # Higher values = more likely to include that behavior
    policy = {
        "aggressive_mining": 0.3,  # Low initially
        "careful_planning": 0.7,  # High initially
    }

    def build_tree_from_policy(current_policy: Dict[str, float]) -> Any:
        """Build tree based on current policy values"""
        builder = TreeBuilder("PolicyDriven")

        # Base actions
        async def quick_mine(bb: RobotBlackboard) -> Status:
            bb.log.append("Quick mining (high yield, high risk)")
            bb.ore_collected += 15
            bb.battery -= 25
            return Status.SUCCESS

        async def careful_mine(bb: RobotBlackboard) -> Status:
            bb.log.append("Careful mining (low yield, low risk)")
            bb.ore_collected += 8
            bb.battery -= 12
            return Status.SUCCESS

        async def plan_action(bb: RobotBlackboard) -> Status:
            bb.log.append("Planning action")
            return Status.SUCCESS

        quick = builder.action("quick_mine", quick_mine)
        careful = builder.action("careful_mine", careful_mine)
        plan = builder.action("plan", plan_action)

        # Build tree based on policy thresholds
        children = []
        if current_policy["careful_planning"] > 0.5:
            children.append(plan)
        if current_policy["aggressive_mining"] > 0.5:
            children.append(quick)
        else:
            children.append(careful)

        # Always include careful option as fallback
        if careful not in children:
            children.append(careful)

        root = builder.selector(*children, memory=False)
        return builder.build(root)

    print("\nInitial policy:")
    print(f"  Aggressive mining: {policy['aggressive_mining']}")
    print(f"  Careful planning: {policy['careful_planning']}")

    # Simulate policy update based on "learning"
    print("\nSimulating learning... (increasing aggressive_mining)")
    policy["aggressive_mining"] = 0.8

    print("\nUpdated policy:")
    print(f"  Aggressive mining: {policy['aggressive_mining']}")
    print(f"  Careful planning: {policy['careful_planning']}")

    tree = build_tree_from_policy(policy)
    bb = RobotBlackboard(battery=100)

    print("\nRunning tree:\n")
    runner = Runner(tree, bb=bb)
    result = await runner.tick_until_complete()

    print(f"\nResult: {result}")
    print("\nExecution log:")
    for entry in bb.log:
        print(f"  {entry}")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all demos"""
    await demo_config_driven_tree()
    await demo_policy_modification()

    print("\n" + "=" * 60)
    print("All demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
