#!/usr/bin/env python3
"""
Enoki Usage Examples (Asyncio Version)

This file demonstrates how to use the Enoki state machine library
with various patterns including timeouts, retries, and state grouping.
"""

from dataclasses import dataclass
import asyncio
from enum import auto, Enum
from random import randint
from typing import List, Union
from mycorrhizal.enoki import (
    State,
    StateMachine,
    StateConfiguration,
    SharedContext,
    StateRef,
    Push,
    Pop,
    Retry,
    StateMachineComplete,
    LabeledTransition,
    TransitionType
)


# Example 1: Basic Network Request State Machine


class NetworkRequestState(State):
    CONFIG = StateConfiguration(timeout=5.0, retries=3)

    # Enums can be used to label transitions for improved diagramming, or to
    # take the place of a stringly-defined state (which can be used to
    # avoid circular imports in large FSMs)
    class T(Enum):
        SUCCESS = auto()
        FAILURE = auto()
        TIMEOUT = auto()

    # Note the use of cls here (rather than self). All methods in here are
    # classmethods, states are FULLY EPHEMERAL, they are never instantiated. The
    # on_enter/state/fail/etc functions are essentially just namespaced code
    # that gets references to the class type that's calling them. (They do not
    # require that the user decorate them as this is handled by the state
    # metaclass).
    def transitions(cls) -> Union[TransitionType, List[Union[TransitionType, LabeledTransition]]]:
        return [
            LabeledTransition(cls.T.SUCCESS, StateRef("enoki_example.ProcessingState")),
            LabeledTransition(cls.T.FAILURE, StateRef("enoki_example.ErrorState")),
            Retry,
        ]

    async def on_enter(cls, ctx: SharedContext):
        print("Starting network request...")
        # In real implementation, would initiate async network request here
        ctx.common.request_id = f"req_{randint(1000, 9999)}"

    async def on_state(cls, ctx: SharedContext):
        if ctx.msg and ctx.msg.get("type") == "network_response":
            if ctx.msg.get("success"):
                return cls.T.SUCCESS
            else:
                return cls.T.FAILURE

    async def on_timeout(cls, ctx: SharedContext):
        print("Network request timed out")
        return Retry

    async def on_fail(cls, ctx: SharedContext):
        print("Network request failed after all retries")
        return cls.T.FAILURE


class ProcessingState(State):
    CONFIG = StateConfiguration(terminal=True)

    async def on_state(cls, ctx: SharedContext):
        print("Processing complete!")


class ErrorState(State):
    CONFIG = StateConfiguration(terminal=True)

    async def on_state(cls, ctx: SharedContext):
        print("Error state reached")


# Example usage and testing
async def example_network_fsm():
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
    async def simulate_network_response():
        await asyncio.sleep(2)  # Simulate network delay
        fsm.send_message({"type": "network_response", "success": True})

    # Start simulation
    asyncio.create_task(simulate_network_response())

    print("---- Starting FSM ----")
    # Run FSM
    try:
        await fsm.run(timeout=0.1)
    except StateMachineComplete:
        pass

    print(f"Final state: {fsm.current_state.name}")


# Example 2: State Grouping for Imaging System


class Imaging:
    # Exterior classes can be used as namespace objects to keep clusters together

    class ImagingState(State):
        """Base class for imaging-related states"""

        pass

    class T(Enum):
        LIGHTING_SET = auto()
        PICTURE_TAKEN = auto()
        FAILED = auto()

    class Error(State):
        CONFIG = StateConfiguration(terminal=True)

        async def on_state(cls, ctx: SharedContext):
            print("Error state reached")

    class SetImageLighting(ImagingState):
        CONFIG = StateConfiguration(timeout=10.0, retries=3)

        def transitions(cls):
            return (
                LabeledTransition(Imaging.T.LIGHTING_SET, Imaging.TakePicture),
                LabeledTransition(Imaging.T.FAILED, Imaging.Error),
                Retry,
            )

        async def on_enter(cls, ctx: SharedContext):
            print("Setting up lighting...")
            # Simulate async lighting setup
            asyncio.create_task(cls._simulate_lighting_setup(ctx))

        async def _simulate_lighting_setup(cls, ctx: SharedContext):
            """Simulate async lighting setup"""
            await asyncio.sleep(2)  # Simulate lighting adjustment time
            print("Light ready")
            ctx.send_message({"type": "lighting_ready"})

        async def on_state(cls, ctx: SharedContext):
            if ctx.msg and ctx.msg.get("type") == "lighting_ready":
                return Imaging.T.LIGHTING_SET

        async def on_timeout(cls, ctx: SharedContext):
            print("Lighting setup timed out")
            return Retry

        async def on_fail(cls, ctx: SharedContext):
            print("Lighting setup failed after all retries")
            return Imaging.T.FAILED

    class TakePicture(ImagingState):
        CONFIG = StateConfiguration(timeout=15.0, retries=2)

        def transitions(cls):
            return (
                LabeledTransition(Imaging.T.PICTURE_TAKEN, ProcessingState),
                LabeledTransition(Imaging.T.FAILED, Imaging.Error),
            )

        async def on_enter(cls, ctx: SharedContext):
            print("Taking picture...")
            asyncio.create_task(cls._simulate_picture_capture(ctx))

        async def _simulate_picture_capture(cls, ctx: SharedContext):
            """Simulate async picture capture"""
            await asyncio.sleep(3)  # Simulate capture time
            ctx.send_message({"type": "picture_complete", "success": True})

        async def on_state(cls, ctx: SharedContext):
            if ctx.msg and ctx.msg.get("type") == "picture_complete":
                if ctx.msg.get("success"):
                    return Imaging.T.PICTURE_TAKEN
                else:
                    return Imaging.T.FAILED

        async def on_timeout(cls, ctx: SharedContext):
            print("Picture capture timed out")
            return Retry

        async def on_fail(cls, ctx: SharedContext):
            print("Picture capture failed after all retries")
            return Imaging.T.FAILED


async def example_imaging_fsm():
    """Example 2: Imaging system with state grouping"""
    print("\n" + "=" * 50)
    print("Example 2: Imaging System FSM")
    print("=" * 50)

    # Create the state machine
    fsm = StateMachine(
        initial_state=Imaging.SetImageLighting,
        error_state=Imaging.Error,
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
        from_state=Imaging.SetImageLighting, max_depth=2, include_groups=False
    )
    print("\nSubgraph (depth 2, no groups):")
    print(subgraph)

    # Show reachable states
    reachable = fsm.get_reachable_states(Imaging.SetImageLighting)
    print(f"\nStates reachable from SetImageLighting: {reachable}")

    # Run FSM
    try:
        await fsm.run()
    except StateMachineComplete:
        pass

    print(f"Final state: {fsm.current_state.name}")


# Example 3: Push/Pop State Stack Example


class PushPop:
    class T(Enum):
        ENTER_SUBMENU = "enter_submenu"
        BACK = "back"
        EXIT = "exit"

    class MainMenu(State):
        CONFIG = StateConfiguration(can_dwell=True)

        def transitions(cls):
            return (
                LabeledTransition(
                    PushPop.T.ENTER_SUBMENU, Push(PushPop.SubMenu, PushPop.MainMenu)
                ),
                LabeledTransition(PushPop.T.EXIT, ProcessingState),
            )

        async def on_state(cls, ctx: SharedContext):
            if ctx.msg and ctx.msg.get("action") == "enter_submenu":
                return PushPop.T.ENTER_SUBMENU
            elif ctx.msg and ctx.msg.get("action") == "exit":
                return ProcessingState

    class SubMenu(State):
        CONFIG = StateConfiguration(can_dwell=True)

        def transitions(cls):
            return (
                LabeledTransition(PushPop.T.BACK, Pop),
                LabeledTransition(
                    PushPop.T.ENTER_SUBMENU, Push(PushPop.SubMenuItem, PushPop.SubMenu)
                ),
            )

        async def on_state(cls, ctx: SharedContext):
            if ctx.msg and ctx.msg.get("action") == "back":
                return PushPop.T.BACK
            elif ctx.msg and ctx.msg.get("action") == "enter_item":
                return PushPop.T.ENTER_SUBMENU

    class SubMenuItem(State):
        CONFIG = StateConfiguration(can_dwell=True)

        def transitions(cls):
            return [LabeledTransition(PushPop.T.BACK, Pop)]

        async def on_state(cls, ctx: SharedContext):
            if ctx.msg and ctx.msg.get("action") == "back":
                return PushPop.T.BACK


async def example_menu_fsm():
    """Example 3: Menu system with push/pop"""
    print("\n" + "=" * 50)
    print("Example 3: Menu System FSM")
    print("=" * 50)

    # Create the state machine
    fsm = StateMachine(
        initial_state=PushPop.MainMenu,
        error_state=ErrorState,
        common_data={"navigation_history": []},
    )

    print(f"✓ FSM created with {len(fsm.discovered_states)} states")

    # Generate flowchart
    flowchart = fsm.generate_mermaid_flowchart()
    print("\nMermaid Flowchart:")
    print(flowchart)

    # Simulate menu navigation
    async def simulate_menu_navigation():
        await asyncio.sleep(0.5)
        print("  -> Entering submenu")
        fsm.send_message({"action": "enter_submenu"})
        await asyncio.sleep(0.5)
        print("  -> Entering menu item")
        fsm.send_message({"action": "enter_item"})
        await asyncio.sleep(0.5)
        print("  -> Going back from item")
        fsm.send_message({"action": "back"})
        await asyncio.sleep(0.5)
        print("  -> Going back from submenu")
        fsm.send_message({"action": "back"})
        await asyncio.sleep(0.5)
        print("  -> Exiting")
        fsm.send_message({"action": "exit"})

    fsm.print_push_pop_analysis()

    # Run FSM with menu navigation
    print("\nRunning menu navigation simulation...")

    # Start simulation
    asyncio.create_task(simulate_menu_navigation())

    # Run FSM
    try:
        await fsm.run()
    except StateMachineComplete:
        pass

    print(f"Final state: {fsm.current_state.name}")
    print(f"Stack depth at end: {len(fsm.state_stack)}")


class Workflow:
    class T(Enum):
        START_TASK = auto()
        COMPLETE_TASK = auto()
        INTERRUPT = auto()
        RESUME = auto()
        FINISH = auto()
        TIMEOUT = auto()

    # Fixed example with explicit returns
    class MainWorkflow(State):
        CONFIG = StateConfiguration(can_dwell=True)

        def transitions(cls):
            return (
                # Now includes itself for return after interrupt
                LabeledTransition(
                    Workflow.T.INTERRUPT,
                    Push(Workflow.InterruptHandler, Workflow.MainWorkflow),
                ),
                LabeledTransition(
                    Workflow.T.START_TASK,
                    Push(
                        Workflow.TaskExecution,
                        Workflow.TaskCleanup,
                        Workflow.MainWorkflow,
                    ),
                ),
                LabeledTransition(Workflow.T.FINISH, Workflow.ProcessingState),
            )

        async def on_state(cls, ctx: SharedContext):
            pass

    class TaskExecution(State):
        CONFIG = StateConfiguration(timeout=10.0)

        def transitions(cls):
            return (
                LabeledTransition(Workflow.T.COMPLETE_TASK, Pop),  # Pops to TaskCleanup
                # Includes itself for return after interrupt
                LabeledTransition(
                    Workflow.T.INTERRUPT,
                    Push(
                        Workflow.InterruptHandler,
                        Workflow.TaskExecution,
                    ),
                ),
            )

        async def on_state(cls, ctx: SharedContext):
            pass

        async def on_timeout(cls, ctx: SharedContext):
            return Workflow.T.COMPLETE_TASK

    class TaskCleanup(State):
        CONFIG = StateConfiguration(timeout=5.0)

        def transitions(cls):
            return (
                LabeledTransition(Workflow.T.COMPLETE_TASK, Pop),
                LabeledTransition(Workflow.T.TIMEOUT, Pop),
            )

        async def on_state(cls, ctx: SharedContext):
            pass

        async def on_timeout(cls, ctx: SharedContext):
            return Workflow.T.TIMEOUT

    class InterruptHandler(State):
        CONFIG = StateConfiguration(timeout=3.0)

        def transitions(cls):
            return (
                # This Pop can go to either MainWorkflow or TaskExecution
                # depending on who pushed it
                LabeledTransition(Workflow.T.RESUME, Pop),
                LabeledTransition(Workflow.T.TIMEOUT, Pop),
            )

        async def on_state(cls, ctx: SharedContext):
            pass

        async def on_timeout(cls, ctx: SharedContext):
            return Workflow.T.TIMEOUT

    class ProcessingState(State):
        CONFIG = StateConfiguration(terminal=True)

        async def on_state(cls, ctx: SharedContext):
            pass


async def example_push_pop_detailed():
    """Test the improved analysis with multiple pop targets"""
    print("=" * 60)
    print("Testing Multiple Pop Target Analysis")
    print("=" * 60)

    # Create FSM
    fsm = StateMachine(initial_state=Workflow.MainWorkflow, error_state=None)

    print(f"\n✓ FSM created with {len(fsm.discovered_states)} states")

    # Get push/pop relationships
    relationships = fsm.get_push_pop_relationships()

    # Analyze InterruptHandler - should have multiple pop targets
    print("\nInterruptHandler Analysis:")
    print("-" * 40)
    
    fsm.print_push_pop_analysis(Workflow.InterruptHandler)
    fsm.print_push_pop_analysis(Workflow.TaskExecution)
    fsm.print_push_pop_analysis(Workflow.TaskCleanup)
  
    # Generate and display flowchart
    print("\nMermaid Flowchart (showing all pop targets):")
    print("-" * 40)
    flowchart = fsm.generate_mermaid_flowchart(show_all_pop_targets=True)
    print(flowchart)


class Deep:
    class T(Enum):
        DIVE = "dive"
        SURFACE = "surface"

    class Level1(State):
        CONFIG = StateConfiguration()

        def transitions(cls):
            return (
                LabeledTransition(
                    Deep.T.DIVE, Push(Deep.Level2, Deep.Shared, Deep.Level1)
                ),
            )

        async def on_state(cls, ctx: SharedContext):
            pass

    class Level2(State):
        CONFIG = StateConfiguration()

        def transitions(cls):
            return (
                LabeledTransition(
                    Deep.T.DIVE, Push(Deep.Level3, Deep.Shared, Deep.Level2)
                ),
                LabeledTransition(Deep.T.SURFACE, Pop),
            )

        async def on_state(cls, ctx: SharedContext):
            pass

    class Level3(State):
        CONFIG = StateConfiguration()

        def transitions(cls):
            return [
                LabeledTransition(Deep.T.SURFACE, Pop),
            ]

        async def on_state(cls, ctx: SharedContext):
            pass

    class Shared(State):
        CONFIG = StateConfiguration()

        def transitions(cls):
            return [
                LabeledTransition(
                    Deep.T.SURFACE, Pop
                ),  # Can pop to Level1, Level2, or others
            ]

        async def on_state(cls, ctx: SharedContext):
            pass


async def deep_stack_analysis():
    print("\n" + "=" * 50)
    print("Example: Deep Stack Analysis")
    print("=" * 50)
    # Create FSM with deep nesting
    fsm2 = StateMachine(initial_state=Deep.Level1)

    print(f"\n✓ Deep FSM created with {len(fsm2.discovered_states)} states")

    # Check Shared state - should have multiple contexts
    relationships2 = fsm2.get_push_pop_relationships()
    shared_info = relationships2.get('enoki_example.Deep.Shared', {})
    fsm2.print_push_pop_analysis()
    print(fsm2.generate_mermaid_flowchart())

    print(f"\nShared state analysis for Deep.Shared:")
    print(f"  Reachable contexts: {shared_info.get('reachable_contexts', 0)}")
    print(f"  Unique stack states: {shared_info.get('unique_stack_states', 0)}")
    print(
        f"  Possible pop destinations: {shared_info.get('possible_pop_destinations', [])}"
    )
    print(f"  Max stack depth: {shared_info.get('max_stack_depth', 0)}")


async def example_validation_errors():
    """Example showing validation errors"""
    print("\n" + "=" * 50)
    print("Example: Validation Errors")
    print("=" * 50)

    class BrokenTransition(Enum):
        NADA = auto()

    # Example with broken state reference
    class BrokenState(State):
        CONFIG = StateConfiguration()

        def transitions(cls):
            return [
                LabeledTransition(
                    BrokenTransition.NADA, StateRef("nonexistent.module.BadState")
                )
            ]

        async def on_state(cls, ctx: SharedContext):
            pass

    try:
        fsm = StateMachine(initial_state=BrokenState)
        print("❌ Should have failed!")
    except Exception as e:
        print(f"✓ Correctly caught validation error: {e}")

    # Example with missing timeout handler
    class TimeoutState(State):
        CONFIG = StateConfiguration(timeout=5.0)

        def transitions(cls):
            return (
                # Missing TIMEOUT handler - will generate warning
                # Also, note the single returned state
                Retry
            )

        async def on_state(cls, ctx: SharedContext):
            pass

        # Note: Missing on_timeout implementation will generate warning

    try:
        fsm = StateMachine(initial_state=TimeoutState, error_state=ErrorState)
        print("✓ FSM created with warnings about missing timeout handler")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise e


async def main():
    """Run all examples"""
    await example_network_fsm()
    await example_imaging_fsm()
    await example_menu_fsm()
    await example_push_pop_detailed()
    await deep_stack_analysis()
    await example_validation_errors()

    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    asyncio.run(main())