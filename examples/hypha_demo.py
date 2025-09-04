#!/usr/bin/env python3
"""
Hypha Comprehensive Demo 

This example demonstrates:
- Proper interface composition with reusable interfaces
- IOInputPlace with configuration
- IOOutputPlace with error handling and termination
- Graceful shutdown mechanisms
- Token-based communication
- Mermaid diagram generation
- Fully qualified names
"""

import signal
import time
import asyncio
from asyncio import sleep, Event
from mycorrhizal.hypha import (
    Place,
    Transition,
    Interface,
    PetriNet,
    Arc,
    IOInputPlace,
    IOOutputPlace,
    PlaceName,
)
from enum import Enum, auto
from typing import Dict, List, Optional, Any
from mycorrhizal.hypha.util import Forward, Route, Fork, NewPlace

# === TOKEN DEFINITIONS ===


class TaskToken():
    """Represents a task to be processed"""

    def __init__(self, task_id: str, task_type: str):
        self.task_id = task_id
        self.task_type = task_type

    def __repr__(self):
        return f"TaskToken({self.task_id}, {self.task_type})"


class ProcessedTaskToken():
    """Represents a completed task"""

    def __init__(self, task_id: str, result: str, processing_time: float):
        self.task_id = task_id
        self.result = result
        self.processing_time = processing_time

    def __repr__(self):
        return f"ProcessedTaskToken({self.task_id}, {self.result})"


class NotificationToken():
    """Represents a notification to be sent"""

    def __init__(self, recipient: str, message: str, urgent: bool = False):
        self.recipient = recipient
        self.message = message
        self.urgent = urgent

    def __repr__(self):
        return f"NotificationToken({self.recipient}, urgent={self.urgent})"


class ErrorToken():
    """Represents an error condition"""

    def __init__(self, error_type: str, message: str, original_token: Any = None):
        self.error_type = error_type
        self.message = message
        self.original_token = original_token

    def __repr__(self):
        return f"ErrorToken({self.error_type}, {self.message})"


# === REUSABLE INTERFACES ===


class TaskGeneratorInterface(Interface):
    """Reusable interface for generating tasks with configurable parameters"""

    class TaskSourcePlace(IOInputPlace):
        """Configurable task generator"""

        def __init__(
            self, task_types: list = None, max_tasks: int = 10, interval: float = 1.0
        ):
            super().__init__()
            self.task_types = task_types or ["analysis", "computation", "validation"]
            self.max_tasks = max_tasks
            self.interval = interval
            self.tasks_generated = 0
            self.start_time = time.time()

        async def on_input(self) -> Optional[TaskToken]:
            if self.tasks_generated >= self.max_tasks:
                return None

            # Wait for interval
            await self.net.sleep_cycles(self.interval)

            # Generate task
            self.tasks_generated += 1
            task_type = self.task_types[
                (self.tasks_generated - 1) % len(self.task_types)
            ]

            task = TaskToken(
                task_id=f"task_{self.tasks_generated:03d}", task_type=task_type
            )
            print(f"Generated: {task}")
            return task

    class TaskQueuePlace(Place):
        """Holds tasks waiting to be processed"""

        MAX_CAPACITY = 20  # Prevent unbounded growth

    Forward(TaskSourcePlace, TaskQueuePlace, name="TaskGenForward")

class TaskProcessorInterface(Interface):
    """Reusable interface for processing tasks (simplified - no priority routing)"""

    class TaskInputPlace(Place):
        """Input place for tasks to be processed"""

        pass

    class ProcessingPlace(Place):
        """Currently processing tasks"""

        pass

    class CompletedPlace(Place):
        """Successfully processed tasks"""

        pass

    class FailedPlace(Place):
        """Failed tasks for retry or error handling"""

        pass

    class ProcessTasksTransition(Transition):
        """Processes tasks with occasional failures"""

        class ArcNames(PlaceName):
            FROM_INPUT = auto()
            TO_PROCESSING = auto()
            TO_COMPLETED = auto()
            TO_FAILED = auto()

        def input_arcs(self) -> Dict[PlaceName, Arc]:
            return {
                self.ArcNames.FROM_INPUT: Arc(TaskProcessorInterface.TaskInputPlace)
            }

        def output_arcs(self) -> Dict[PlaceName, Arc]:
            return {
                self.ArcNames.TO_PROCESSING: Arc(
                    TaskProcessorInterface.ProcessingPlace
                ),
                self.ArcNames.TO_COMPLETED: Arc(TaskProcessorInterface.CompletedPlace),
                self.ArcNames.TO_FAILED: Arc(TaskProcessorInterface.FailedPlace),
            }

        async def on_fire(
            self, consumed: Dict[PlaceName, List[Any]]
        ) -> Dict[PlaceName, List[Any]]:
            results = {
                self.ArcNames.TO_PROCESSING: [],
                self.ArcNames.TO_COMPLETED: [],
                self.ArcNames.TO_FAILED: [],
            }

            for task in consumed[self.ArcNames.FROM_INPUT]:
                print(f"Processing: {task}")

                # Simulate processing time
                start_time = time.time()
                await self.net.sleep_cycles(0.3)  # Processing delay
                processing_time = time.time() - start_time

                # Simulate occasional failures (task IDs ending in 6 or 9)
                if task.task_id.endswith("6") or task.task_id.endswith("9"):
                    error = ErrorToken(
                        "processing_failed",
                        f"Task {task.task_id} failed validation",
                        task,
                    )
                    results[self.ArcNames.TO_FAILED].append(error)
                    print(f"❌ Failed: {task}")
                else:
                    processed = ProcessedTaskToken(
                        task.task_id, f"processed_{task.task_type}", processing_time
                    )
                    results[self.ArcNames.TO_COMPLETED].append(processed)
                    print(f"✅ Completed: {task}")

            return results


class NotificationInterface(Interface):
    """Reusable interface for sending notifications via different channels"""
    NotificationInputPlace = NewPlace()

    class EmailOutputPlace(IOOutputPlace):
        """Sends email notifications"""

        def __init__(self):
            super().__init__()
            self.sent_emails = []

        async def on_token_added(self, token: Any) -> Optional[Any]:
            if isinstance(token, NotificationToken):
                # Simulate email sending
                await self.net.sleep_cycles(0.1)

                # Simulate occasional email failures
                if "error" in token.message.lower():
                    error = ErrorToken(
                        "email_failed", f"Failed to send email to {token.recipient}"
                    )
                    print(f"❌ EMAIL FAILED: {token.recipient}")
                    return error

                self.sent_emails.append(token)
                print(f"📧 EMAIL: {token.recipient} - {token.message}")
                return None

            return None

    class SMSOutputPlace(IOOutputPlace):
        """Sends SMS notifications for urgent messages"""

        def __init__(self):
            super().__init__()
            self.sent_sms = []

        async def on_token_added(self, token: Any) -> Optional[Any]:
            if isinstance(token, NotificationToken) and token.urgent:
                # Simulate SMS sending
                await self.net.sleep_cycles(0.05)

                self.sent_sms.append(token)
                print(f"📱 SMS: {token.recipient} - {token.message}")
                return None
            elif isinstance(token, NotificationToken):
                # Non-urgent notifications don't go to SMS
                return None

            return None

    class LogOutputPlace(IOOutputPlace):
        """Logs all notifications"""

        async def on_token_added(self, token: Any) -> Optional[Any]:
            if isinstance(token, NotificationToken):
                timestamp = time.strftime("%H:%M:%S")
                print(f"📋 LOG [{timestamp}]: {token.recipient} - {token.message}")
                return None

            return None
        
    Fork(NotificationInputPlace, [EmailOutputPlace, SMSOutputPlace, LogOutputPlace], name="NotificationFork")

class ErrorHandlingInterface(Interface):
    """Reusable interface for handling errors and retries"""

    class ErrorInputPlace(Place):
        """Input place for errors to be handled"""

        pass

    class ErrorLogPlace(IOOutputPlace):
        """Logs errors to external system"""

        async def on_token_added(self, token: Any) -> Optional[Any]:
            if isinstance(token, ErrorToken):
                timestamp = time.strftime("%H:%M:%S")
                print(f"❌ ERROR [{timestamp}]: {token.error_type} - {token.message}")
            return None


    Forward(ErrorInputPlace, ErrorLogPlace, name="ErrorForward")


# === MAIN SYSTEM ===


class TaskProcessingSystem(PetriNet):
    """Comprehensive task processing system demonstrating reusable interface composition"""

    def __init__(self):
        super().__init__()
        self.finished_event = Event()
        self.total_tasks_to_process = 0
        self.tasks_completed = 0

    # Compose all interfaces
    class TaskGen(TaskGeneratorInterface):
        pass

    class TaskProc(TaskProcessorInterface):
        pass

    class Notify(NotificationInterface):
        pass

    class ErrorHandle(ErrorHandlingInterface):
        pass

    # Completion tracking output place
    class CompletionTracker(IOOutputPlace):
        """Tracks completed tasks and signals when all are done"""

        async def on_token_added(self, token: Any) -> Optional[Any]:
            if isinstance(token, (ProcessedTaskToken, ErrorToken)):
                net = self.net
                if net:
                    net.tasks_completed += 1
                    print(
                        f"📊 Progress: {net.tasks_completed}/{net.total_tasks_to_process} tasks completed"
                    )

                    # Check if we're done
                    if net.tasks_completed >= net.total_tasks_to_process:
                        print("🎉 All tasks completed! Signaling shutdown...")
                        net.finished_event.set()

                return None
            return None

    # Top-level transitions that connect interfaces
    
    Forward(TaskGen.TaskQueuePlace, TaskProc.TaskInputPlace)

    class ConnectProcessorToNotificationTransition(Transition):
        """Connects processor output to notification system and completion tracking"""

        class ArcNames(PlaceName):
            FROM_COMPLETED = auto()
            TO_NOTIFICATION = auto()
            TO_TRACKER = auto()

        def input_arcs(self) -> Dict[PlaceName, Arc]:
            return {
                self.ArcNames.FROM_COMPLETED: Arc(
                    TaskProcessingSystem.TaskProc.CompletedPlace
                )
            }

        def output_arcs(self) -> Dict[PlaceName, Arc]:
            return {
                self.ArcNames.TO_NOTIFICATION: Arc(
                    TaskProcessingSystem.Notify.NotificationInputPlace
                ),
                self.ArcNames.TO_TRACKER: Arc(TaskProcessingSystem.CompletionTracker),
            }

        async def on_fire(
            self, consumed: Dict[PlaceName, List[Any]]
        ) -> Dict[PlaceName, List[Any]]:
            completed_tasks = consumed[self.ArcNames.FROM_COMPLETED]

            # Create notifications for completed tasks
            notifications = []
            for completed in completed_tasks:
                is_urgent = "analysis" in completed.result  # Make analysis tasks urgent
                notification = NotificationToken(
                    recipient="admin@example.com",
                    message=f"Task {completed.task_id} completed: {completed.result}",
                    urgent=is_urgent,
                )
                notifications.append(notification)

            return {
                self.ArcNames.TO_NOTIFICATION: notifications,
                self.ArcNames.TO_TRACKER: completed_tasks,
            }

    Fork(TaskProc.FailedPlace, [ErrorHandle.ErrorInputPlace, CompletionTracker], name="FailedToErrorAndLog")


async def demonstrate_comprehensive_system():
    """Demonstrate all major Hypha features with reusable interface composition"""
    print("=" * 60)
    print("Hypha Comprehensive Demo")
    print("=" * 60)

    # Create system with reusable interface composition
    system = TaskProcessingSystem()

    # Configure the task generator
    task_source = system.get_place(TaskProcessingSystem.TaskGen.TaskSourcePlace)
    task_source.task_types = ["analysis", "computation", "validation", "reporting"]
    task_source.max_tasks = 8
    task_source.interval = 0.8

    # Set the total tasks to process for completion tracking
    system.total_tasks_to_process = task_source.max_tasks

    print("=" * 60)
    print("System Configuration")
    print("=" * 60)
    print(f"Task types: {task_source.task_types}")
    print(f"Max tasks: {task_source.max_tasks}")
    print(f"Generation interval: {task_source.interval}s")
    print(f"Total places: {len(system._places_by_type)}")
    print(f"Total transitions: {len(system._transitions)}")

    # Show qualified names to demonstrate proper composition
    print(f"\nInterface Composition:")
    print(
        f"- TaskGen: {len([p for p in system._places_by_qualified_name if 'TaskGen' in p])} places"
    )
    print(
        f"- TaskProc: {len([p for p in system._places_by_qualified_name if 'TaskProc' in p])} places"
    )
    print(
        f"- Notify: {len([p for p in system._places_by_qualified_name if 'Notify' in p])} places"
    )
    print(
        f"- ErrorHandle: {len([p for p in system._places_by_qualified_name if 'ErrorHandle' in p])} places"
    )

    # Show some example qualified names
    print(f"\nExample Qualified Names:")
    for place_name in list(system._places_by_qualified_name.keys())[:5]:
        print(f"  {place_name}")

    # Generate Mermaid diagram
    print(f"\nMermaid Diagram:")
    print("-" * 40)
    system.print_mermaid_diagram()
    print("-" * 40)

    # Set up signal handling for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\n🔔 Received signal {signum}, initiating graceful shutdown...")
        system.finished_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"\n" + "=" * 60)
    print("Starting System (Ctrl+C for graceful shutdown)")
    print("=" * 60)

    # Start the system
    start_time = time.time()
    await system.start()

    try:
        print("Running system until all tasks are processed...")
        print("Watch the autonomous processing with reusable interfaces:")

        # Wait for completion or signal (with reasonable timeout)
        await system.finished_event.wait()
        await sleep(0.5)

    finally:
        print(f"\nStopping system...")
        # Give some time for final processing        
        await system.stop(timeout=1)

    elapsed = time.time() - start_time

    print(f"\n" + "=" * 60)
    print(f"System Completed in {elapsed:.4f}s")
    print("=" * 60)

    # Show final statistics
    state = system.get_state_summary()

    email_place = system.get_place(TaskProcessingSystem.Notify.EmailOutputPlace)
    sms_place = system.get_place(TaskProcessingSystem.Notify.SMSOutputPlace)

    print(f"\nFinal Statistics:")
    print(f"Tasks generated: {task_source.tasks_generated}")
    print(f"Tasks completed: {system.tasks_completed}")
    print(f"Emails sent: {len(email_place.sent_emails)}")
    print(f"SMS sent: {len(sms_place.sent_sms)}")

    print(f"\nFinal Token Distribution:")
    for place_name, place_info in state["places"].items():
        if place_info["token_count"] > 0:
            print(f"  {place_name}: {place_info['token_count']} tokens")

    print(f"\nTransition Fire Counts:")
    for transition_name, transition_info in state["transitions"].items():
        if transition_info["fire_count"] > 0:
            print(f"  {transition_name}: {transition_info['fire_count']} fires")

    print(f"\nPost Run Mermaid Diagram:")
    print("-" * 40)
    system.print_mermaid_diagram()
    print("-" * 40)

    print(f"\n" + "=" * 60)
    print("Demonstration completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demonstrate_comprehensive_system())
