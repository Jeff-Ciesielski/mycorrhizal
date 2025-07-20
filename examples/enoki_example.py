#!/usr/bin/env python3
"""
Enoki Usage Examples

This file demonstrates how to use the Enoki state machine library
with various patterns including timeouts, retries, and state grouping.
"""

from dataclasses import dataclass
import gevent
from gevent import sleep, spawn
from random import randint
from mycorrhizal.enoki import (
    State,
    StateMachine,
    TransitionName,
    GlobalTransitions,
    StateConfiguration,
    SharedContext,
    TimeoutMessage,
    Push,
    Pop,
)


# Example 1: Basic Network Request State Machine
class NetworkTransitions(TransitionName):
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"


class NetworkRequestState(State):
    CONFIG = StateConfiguration(timeout=5.0, retries=3)

    @classmethod
    def transitions(cls):
        return {
            NetworkTransitions.SUCCESS: "enoki_example.ProcessingState",
            NetworkTransitions.FAILURE: "enoki_example.ErrorState",
        }

    @classmethod
    def on_enter(cls, ctx: SharedContext):
        print("Starting network request...")
        # In real implementation, would initiate async network request here
        ctx.common.request_id = f"req_{randint}"
        spawn(cls._send_msg, ctx)

    @classmethod
    def _send_msg(cls, ctx):
        sleep(0.2)
        print("sending message")
        ctx.send_message({'type': 'network_response', 'success': True})
        

    @classmethod
    def on_state(cls, ctx: SharedContext) -> TransitionName:
        if ctx.msg and ctx.msg.get("type") == "network_response":
            if ctx.msg.get("success"):
                return NetworkTransitions.SUCCESS
            else:
                return NetworkTransitions.FAILURE

    @classmethod
    def on_timeout(cls, ctx: SharedContext) -> TransitionName:
        print("Network request timed out")
        return GlobalTransitions.RETRY

    @classmethod
    def on_fail(cls, ctx: SharedContext) -> TransitionName:
        print("Network request failed after all retries")
        return NetworkTransitions.FAILURE


class ProcessingState(State):
    CONFIG = StateConfiguration(timeout=2.0, terminal=True)

    @classmethod
    def on_state(cls, ctx: SharedContext) -> TransitionName:
        print("Processing complete!")


class ErrorState(State):
    CONFIG = StateConfiguration(terminal=True)

    @classmethod
    def on_state(cls, ctx: SharedContext) -> TransitionName:
        print("Error state reached")


# Example 2: State Grouping for Imaging System
class ImagingState(State):
    """Base class for imaging-related states"""

    pass


class ImagingTransitions(TransitionName):
    LIGHTING_SET = "lighting_set"
    PICTURE_TAKEN = "picture_taken"
    FAILED = "failed"


class SetImageLighting(ImagingState):
    CONFIG = StateConfiguration(timeout=10.0, retries=3)

    @classmethod
    def transitions(cls):
        return {
            ImagingTransitions.LIGHTING_SET: TakePicture,
            ImagingTransitions.FAILED: ErrorState,
            GlobalTransitions.TIMEOUT: GlobalTransitions.RETRY,
            GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED,
        }

    @classmethod
    def on_enter(cls, ctx: SharedContext):
        print("Setting up lighting...")
        # Simulate async lighting setup
        spawn(cls._simulate_lighting_setup, ctx)

    @classmethod
    def _simulate_lighting_setup(cls, ctx: SharedContext):
        """Simulate async lighting setup"""
        sleep(2)  # Simulate lighting adjustment time
        ctx.send_message({"type": "lighting_ready"})

    @classmethod
    def on_state(cls, ctx: SharedContext) -> TransitionName:
        if isinstance(ctx.msg, TimeoutMessage):
            return GlobalTransitions.TIMEOUT
        elif ctx.msg and ctx.msg.get("type") == "lighting_ready":
            return ImagingTransitions.LIGHTING_SET
        return GlobalTransitions.UNHANDLED

    @classmethod
    def on_timeout(cls, ctx: SharedContext) -> TransitionName:
        print("Lighting setup timed out")
        return GlobalTransitions.RETRY

    @classmethod
    def on_fail(cls, ctx: SharedContext) -> TransitionName:
        print("Lighting setup failed after all retries")
        return ImagingTransitions.FAILED


class TakePicture(ImagingState):
    CONFIG = StateConfiguration(timeout=15.0, retries=2)

    @classmethod
    def transitions(cls):
        return {
            ImagingTransitions.PICTURE_TAKEN: ProcessingState,
            ImagingTransitions.FAILED: ErrorState,
            GlobalTransitions.TIMEOUT: GlobalTransitions.RETRY,
            GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED,
        }

    @classmethod
    def on_enter(cls, ctx: SharedContext):
        print("Taking picture...")
        spawn(cls._simulate_picture_capture, ctx)

    @classmethod
    def _simulate_picture_capture(cls, ctx: SharedContext):
        """Simulate async picture capture"""
        sleep(3)  # Simulate capture time
        ctx.send_message({"type": "picture_complete", "success": True})

    @classmethod
    def on_state(cls, ctx: SharedContext) -> TransitionName:
        if isinstance(ctx.msg, TimeoutMessage):
            return GlobalTransitions.TIMEOUT
        elif ctx.msg and ctx.msg.get("type") == "picture_complete":
            if ctx.msg.get("success"):
                return ImagingTransitions.PICTURE_TAKEN
            else:
                return ImagingTransitions.FAILED
        return GlobalTransitions.UNHANDLED

    @classmethod
    def on_timeout(cls, ctx: SharedContext) -> TransitionName:
        print("Picture capture timed out")
        return GlobalTransitions.RETRY

    @classmethod
    def on_fail(cls, ctx: SharedContext) -> TransitionName:
        print("Picture capture failed after all retries")
        return ImagingTransitions.FAILED


# Example 3: Push/Pop State Stack Example
class MenuTransitions(TransitionName):
    ENTER_SUBMENU = "enter_submenu"
    BACK = "back"
    EXIT = "exit"


class MainMenu(State):
    CONFIG = StateConfiguration(can_dwell=True)

    @classmethod
    def transitions(cls):
        return {
            MenuTransitions.ENTER_SUBMENU: Push(SubMenu, MainMenu),
            MenuTransitions.EXIT: ProcessingState,
            GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED,
        }

    @classmethod
    def on_state(cls, ctx: SharedContext) -> TransitionName:
        if ctx.msg and ctx.msg.get("action") == "enter_submenu":
            return MenuTransitions.ENTER_SUBMENU
        elif ctx.msg and ctx.msg.get("action") == "exit":
            return MenuTransitions.EXIT
        return GlobalTransitions.UNHANDLED


class SubMenu(State):
    CONFIG = StateConfiguration(can_dwell=True)

    @classmethod
    def transitions(cls):
        return {
            MenuTransitions.BACK: Pop,
            MenuTransitions.ENTER_SUBMENU: Push(SubMenuItem, SubMenu),
            GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED,
        }

    @classmethod
    def on_state(cls, ctx: SharedContext) -> TransitionName:
        if ctx.msg and ctx.msg.get("action") == "back":
            return MenuTransitions.BACK
        elif ctx.msg and ctx.msg.get("action") == "enter_item":
            return MenuTransitions.ENTER_SUBMENU
        return GlobalTransitions.UNHANDLED


class SubMenuItem(State):
    CONFIG = StateConfiguration(can_dwell=True)

    @classmethod
    def transitions(cls):
        return {
            MenuTransitions.BACK: Pop,
            GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED,
        }

    @classmethod
    def on_state(cls, ctx: SharedContext) -> TransitionName:
        if ctx.msg and ctx.msg.get("action") == "back":
            return MenuTransitions.BACK
        return GlobalTransitions.UNHANDLED


# Example usage and testingv
def example_network_fsm():
    """Example 1: Network request with timeout and retry"""
    print("=" * 50)
    print("Example 1: Network Request FSM")
    print("=" * 50)

    @dataclass
    class CommonData:
        request_id: str = ""

    # Create the state machine
    fsm = StateMachine(
        initial_state=NetworkRequestState,
        error_state=ErrorState,
        common_data=CommonData(),
    )

    print(f"✓ FSM created with {len(fsm.discovered_states)} states")

    # Generate flowchart
    flowchart = fsm.generate_mermaid_flowchart()
    print("\nMermaid Flowchart:")
    print(flowchart)

    # Simulate network response
    def simulate_network_response():
        sleep(2)  # Simulate network delay
        fsm.send_message({"type": "network_response", "success": True})

    # Start simulation
    spawn(simulate_network_response)

    # Run FSM
    fsm.run(max_iterations=10)

    print(f"Final state: {fsm.current_state.name()}")


def example_imaging_fsm():
    """Example 2: Imaging system with state grouping"""
    print("\n" + "=" * 50)
    print("Example 2: Imaging System FSM")
    print("=" * 50)

    # Create the state machine
    fsm = StateMachine(
        initial_state=SetImageLighting,
        error_state=ErrorState,
        common_data={"images_captured": 0},
    )

    print(f"✓ FSM created with {len(fsm.discovered_states)} states")
    print(f"State groups: {list(fsm.state_groups.keys())}")

    # Generate flowchart with grouping
    flowchart = fsm.generate_mermaid_flowchart(include_groups=True)
    print("\nMermaid Flowchart with Grouping:")
    print(flowchart)

    # Generate subgraph from SetImageLighting
    subgraph = fsm.generate_mermaid_flowchart(
        from_state=SetImageLighting, max_depth=2, include_groups=False
    )
    print("\nSubgraph (depth 2, no groups):")
    print(subgraph)

    # Show reachable states
    reachable = fsm.get_reachable_states(SetImageLighting)
    print(f"\nStates reachable from SetImageLighting: {reachable}")

    # Run FSM
    fsm.run(max_iterations=10)

    print(f"Final state: {fsm.current_state.name()}")


def example_menu_fsm():
    """Example 3: Menu system with push/pop"""
    print("\n" + "=" * 50)
    print("Example 3: Menu System FSM")
    print("=" * 50)

    # Create the state machine
    fsm = StateMachine(
        initial_state=MainMenu,
        error_state=ErrorState,
        common_data={"navigation_history": []},
    )

    print(f"✓ FSM created with {len(fsm.discovered_states)} states")

    # Generate flowchart
    flowchart = fsm.generate_mermaid_flowchart()
    print("\nMermaid Flowchart:")
    print(flowchart)

    # Simulate menu navigation
    def simulate_menu_navigation():
        sleep(1)
        fsm.send_message({"action": "enter_submenu"})
        sleep(1)
        fsm.send_message({"action": "enter_item"})
        sleep(1)
        fsm.send_message({"action": "back"})
        sleep(1)
        fsm.send_message({"action": "back"})
        sleep(1)
        fsm.send_message({"action": "exit"})

    # Start simulation
    spawn(simulate_menu_navigation)

    fsm.print_push_pop_analysis()

    # Run FSM with menu navigation
    print("\nRunning menu navigation simulation...")

    def simulate_menu_navigation():
        sleep(0.5)
        print("  -> Entering submenu")
        fsm.send_message({"action": "enter_submenu"})
        sleep(0.5)
        print("  -> Entering menu item")
        fsm.send_message({"action": "enter_item"})
        sleep(0.5)
        print("  -> Going back from item")
        fsm.send_message({"action": "back"})
        sleep(0.5)
        print("  -> Going back from submenu")
        fsm.send_message({"action": "back"})
        sleep(0.5)
        print("  -> Exiting")
        fsm.send_message({"action": "exit"})

    # Start simulation
    spawn(simulate_menu_navigation)

    # Run FSM
    fsm.run(max_iterations=20)

    print(f"Final state: {fsm.current_state.name()}")
    print(f"Stack depth at end: {len(fsm.state_stack)}")


class WorkflowTransitions(TransitionName):
    START_TASK = "start_task"
    COMPLETE_TASK = "complete_task"
    INTERRUPT = "interrupt"
    RESUME = "resume"
    FINISH = "finish"


# Fixed example with explicit returns
class MainWorkflow(State):
    CONFIG = StateConfiguration(can_dwell=True)

    @classmethod
    def transitions(cls):
        return {
            # Now includes itself for return after interrupt
            WorkflowTransitions.INTERRUPT: Push(
                "enoki_example.InterruptHandler", "enoki_example.MainWorkflow"
            ),
            WorkflowTransitions.START_TASK: Push(
                "enoki_example.TaskExecution",
                "enoki_example.TaskCleanup",
                "enoki_example.MainWorkflow",
            ),
            WorkflowTransitions.FINISH: "enoki_example.ProcessingState",
            GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED,
        }

    @classmethod
    def on_state(cls, ctx: SharedContext) -> TransitionName:
        return GlobalTransitions.UNHANDLED


class TaskExecution(State):
    CONFIG = StateConfiguration(timeout=10.0)

    @classmethod
    def transitions(cls):
        return {
            WorkflowTransitions.COMPLETE_TASK: Pop,  # Pops to TaskCleanup
            # Includes itself for return after interrupt
            WorkflowTransitions.INTERRUPT: Push(
                "enoki_example.InterruptHandler", "enoki_example.TaskExecution"
            ),
            GlobalTransitions.TIMEOUT: GlobalTransitions.UNHANDLED,
            GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED,
        }

    @classmethod
    def on_state(cls, ctx: SharedContext) -> TransitionName:
        return GlobalTransitions.UNHANDLED


class TaskCleanup(State):
    CONFIG = StateConfiguration(timeout=5.0)

    @classmethod
    def transitions(cls):
        return {
            WorkflowTransitions.COMPLETE_TASK: Pop,  # Pops to MainWorkflow
            GlobalTransitions.TIMEOUT: Pop,
            GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED,
        }

    @classmethod
    def on_state(cls, ctx: SharedContext) -> TransitionName:
        return GlobalTransitions.UNHANDLED


class InterruptHandler(State):
    CONFIG = StateConfiguration(timeout=3.0)

    @classmethod
    def transitions(cls):
        return {
            # This Pop can go to either MainWorkflow or TaskExecution
            # depending on who pushed it
            WorkflowTransitions.RESUME: Pop,
            GlobalTransitions.TIMEOUT: Pop,
            GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED,
        }

    @classmethod
    def on_state(cls, ctx: SharedContext) -> TransitionName:
        return GlobalTransitions.UNHANDLED


class ProcessingState(State):
    CONFIG = StateConfiguration(terminal=True)

    @classmethod
    def transitions(cls):
        return {GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED}

    @classmethod
    def on_state(cls, ctx: SharedContext) -> TransitionName:
        return GlobalTransitions.UNHANDLED


def example_push_pop_detailed():
    """Test the improved analysis with multiple pop targets"""
    print("=" * 60)
    print("Testing Multiple Pop Target Analysis")
    print("=" * 60)

    # Create FSM
    fsm = StateMachine(initial_state=MainWorkflow, error_state=None)

    print(f"\n✓ FSM created with {len(fsm.discovered_states)} states")

    # Get push/pop relationships
    relationships = fsm.get_push_pop_relationships()

    # Analyze InterruptHandler - should have multiple pop targets
    print("\nInterruptHandler Analysis:")
    print("-" * 40)

    interrupt_info = relationships.get("enoki_example.InterruptHandler", {})

    print(f"Push sources: {len(interrupt_info.get('push_sources', []))}")
    for source in interrupt_info.get("push_sources", []):
        print(f"  - From {source['from_state']} via {source['transition']}")

    print(f"\nPop targets: {len(interrupt_info.get('pop_targets', []))}")
    for pop in interrupt_info.get("pop_targets", []):
        if pop.get("is_multi"):
            print(
                f"  - {pop['transition']}: Can pop to {pop['targets']} (multiple targets)"
            )
        elif pop.get("error"):
            print(f"  - {pop['transition']}: {pop.get('error')}")
        else:
            print(f"  - {pop['transition']}: Pops to {pop['targets'][0]}")

    print(
        f"\nPossible pop destinations: {interrupt_info.get('possible_pop_destinations', [])}"
    )
    print(f"Reachable contexts: {interrupt_info.get('reachable_contexts', 0)}")
    print(f"Has multiple pop targets: {interrupt_info.get('has_multi_pop', False)}")

    # Check other states
    print("\nOther State Pop Behaviors:")
    print("-" * 40)

    for state_name in ["TaskExecution", "TaskCleanup"]:
        full_name = f"enoki_example.{state_name}"
        state_info = relationships.get(full_name, {})
        pop_targets = state_info.get("pop_targets", [])

        if pop_targets:
            for pop in pop_targets:
                if pop.get("targets"):
                    print(f"{state_name}: {pop['transition']} -> {pop['targets']}")

    # Generate and display flowchart
    print("\nMermaid Flowchart (showing all pop targets):")
    print("-" * 40)
    flowchart = fsm.generate_mermaid_flowchart(show_all_pop_targets=True)
    print(flowchart)


class DeepTransitions(TransitionName):
    DIVE = "dive"
    SURFACE = "surface"


class Level1(State):
    CONFIG = StateConfiguration()

    @classmethod
    def transitions(cls):
        return {
            DeepTransitions.DIVE: Push(
                "enoki_example.Level2", "enoki_example.Shared", "enoki_example.Level1"
            ),
            GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED,
        }

    @classmethod
    def on_state(cls, ctx: SharedContext) -> TransitionName:
        return GlobalTransitions.UNHANDLED


class Level2(State):
    CONFIG = StateConfiguration()

    @classmethod
    def transitions(cls):
        return {
            DeepTransitions.DIVE: Push(
                "enoki_example.Level3", "enoki_example.Shared", "enoki_example.Level2"
            ),
            DeepTransitions.SURFACE: Pop,
            GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED,
        }

    @classmethod
    def on_state(cls, ctx: SharedContext) -> TransitionName:
        return GlobalTransitions.UNHANDLED


class Level3(State):
    CONFIG = StateConfiguration()

    @classmethod
    def transitions(cls):
        return {
            DeepTransitions.SURFACE: Pop,
            GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED,
        }

    @classmethod
    def on_state(cls, ctx: SharedContext) -> TransitionName:
        return GlobalTransitions.UNHANDLED


class Shared(State):
    CONFIG = StateConfiguration()

    @classmethod
    def transitions(cls):
        return {
            DeepTransitions.SURFACE: Pop,  # Can pop to Level1, Level2, or others
            GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED,
        }

    @classmethod
    def on_state(cls, ctx: SharedContext) -> TransitionName:
        return GlobalTransitions.UNHANDLED


def deep_stack_analysis():
    print("\n" + "=" * 50)
    print("Example: Deep Stack Analysis")
    print("=" * 50)
    # Create FSM with deep nesting
    fsm2 = StateMachine(initial_state=Level1)

    print(f"\n✓ Deep FSM created with {len(fsm2.discovered_states)} states")

    # Check Shared state - should have multiple contexts
    relationships2 = fsm2.get_push_pop_relationships()
    shared_info = relationships2.get("enoki_example.Shared", {})

    fsm2.print_push_pop_analysis()
    print(fsm2.generate_mermaid_flowchart())

    print(f"\nShared state analysis:")
    print(f"  Reachable contexts: {shared_info.get('reachable_contexts', 0)}")
    print(f"  Unique stack states: {shared_info.get('unique_stack_states', 0)}")
    print(
        f"  Possible pop destinations: {shared_info.get('possible_pop_destinations', [])}"
    )
    print(f"  Max stack depth: {shared_info.get('max_stack_depth', 0)}")


def example_validation_errors():
    """Example showing validation errors"""
    print("\n" + "=" * 50)
    print("Example: Validation Errors")
    print("=" * 50)

    # Example with broken state reference
    class BrokenState(State):
        CONFIG = StateConfiguration()

        @classmethod
        def transitions(cls):
            return {GlobalTransitions.UNHANDLED: "nonexistent.module.BadState"}

        @classmethod
        def on_state(cls, ctx: SharedContext) -> TransitionName:
            return GlobalTransitions.UNHANDLED

    try:
        fsm = StateMachine(initial_state=BrokenState)
        print("❌ Should have failed!")
    except Exception as e:
        print(f"✓ Correctly caught validation error: {e}")

    # Example with missing timeout handler
    class TimeoutState(State):
        CONFIG = StateConfiguration(timeout=5.0)

        @classmethod
        def transitions(cls):
            return {
                GlobalTransitions.UNHANDLED: GlobalTransitions.UNHANDLED
                # Missing TIMEOUT handler - will generate warning
            }

        @classmethod
        def on_state(cls, ctx: SharedContext) -> TransitionName:
            return GlobalTransitions.UNHANDLED

    try:
        fsm = StateMachine(initial_state=TimeoutState, error_state=ErrorState)
        print("✓ FSM created with warnings about missing timeout handler")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    # Run all examples
    example_network_fsm()
    # example_imaging_fsm()
    # example_menu_fsm()
    # example_push_pop_detailed()
    # deep_stack_analysis()
    # example_validation_errors()

    print("\n" + "=" * 50)
    print("All examples completed!")
