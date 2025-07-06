#!/usr/bin/env python3
"""
Cordyceps Comprehensive Demo - Fixed Interface Boundaries

This example demonstrates:
- Proper interface composition without boundary violations
- IOInputPlace with configuration
- IOOutputPlace with error handling
- Graceful shutdown mechanisms
- Token-based communication
- Async execution model
- Mermaid diagram generation
- Fully qualified names
"""

import asyncio
import signal
import time
from cordyceps.core import (
    Token, Place, Transition, Interface, PetriNet, Arc, 
    IOInputPlace, IOOutputPlace, create_simple_token
)
from typing import Optional


# === TOKEN DEFINITIONS ===

class TaskToken(Token):
    """Represents a task to be processed"""
    def __init__(self, task_id: str, task_type: str, priority: int = 0):
        super().__init__()
        self.task_id = task_id
        self.task_type = task_type
        self.priority = priority
    
    def __repr__(self):
        return f"TaskToken({self.task_id}, {self.task_type}, pri={self.priority})"


class ProcessedTaskToken(Token):
    """Represents a completed task"""
    def __init__(self, task_id: str, result: str, processing_time: float):
        super().__init__()
        self.task_id = task_id
        self.result = result
        self.processing_time = processing_time
    
    def __repr__(self):
        return f"ProcessedTaskToken({self.task_id}, {self.result})"


class NotificationToken(Token):
    """Represents a notification to be sent"""
    def __init__(self, recipient: str, message: str, urgent: bool = False):
        super().__init__()
        self.recipient = recipient
        self.message = message
        self.urgent = urgent
    
    def __repr__(self):
        return f"NotificationToken({self.recipient}, urgent={self.urgent})"


class ErrorToken(Token):
    """Represents an error condition"""
    def __init__(self, error_type: str, message: str, original_token: Token = None):
        super().__init__()
        self.error_type = error_type
        self.message = message
        self.original_token = original_token
    
    def __repr__(self):
        return f"ErrorToken({self.error_type}, {self.message})"


class ShutdownToken(Token):
    """Represents a shutdown signal"""
    def __init__(self, reason: str = "shutdown"):
        super().__init__()
        self.reason = reason
    
    def __repr__(self):
        return f"ShutdownToken({self.reason})"


# === TASK GENERATION INTERFACE ===

class TaskGeneratorInterface(Interface):
    """Interface for generating tasks with configurable parameters"""
    
    class TaskSourcePlace(IOInputPlace):
        """Configurable task generator"""
        
        def __init__(self, task_types: list = None, max_tasks: int = 10, interval: float = 1.0):
            super().__init__()
            self.task_types = task_types or ["analysis", "computation", "validation"]
            self.max_tasks = max_tasks
            self.interval = interval
            self.tasks_generated = 0
            self.start_time = time.time()
        
        async def on_input_start(self):
            self.log(f"Task generator started: {self.max_tasks} tasks, interval={self.interval}s")
        
        async def on_input(self):
            if self.tasks_generated >= self.max_tasks:
                self._ensure_shutdown_event()
                await self._shutdown_event.wait()
                return
            
            # Wait for interval
            await asyncio.sleep(self.interval)
            
            # Generate task
            self.tasks_generated += 1
            task_type = self.task_types[(self.tasks_generated - 1) % len(self.task_types)]
            priority = 3 if self.tasks_generated % 3 == 0 else 1  # Every 3rd task is high priority
            
            task = TaskToken(
                task_id=f"task_{self.tasks_generated:03d}",
                task_type=task_type,
                priority=priority
            )
            
            await self.produce_token(task)
            self.log(f"Generated: {task}")
        
        async def on_input_stop(self):
            elapsed = time.time() - self.start_time
            self.log(f"Task generator stopped after {elapsed:.1f}s, generated {self.tasks_generated} tasks")
    
    class TaskQueuePlace(Place):
        """Holds tasks waiting to be processed"""
        MAX_CAPACITY = 20  # Prevent unbounded growth
    
    class MoveTasksTransition(Transition):
        """Moves tasks from generator to queue"""
        
        def input_arcs(self):
            return {"task": Arc(TaskGeneratorInterface.TaskSourcePlace)}
        
        def output_arcs(self):
            return {"queue": Arc(TaskGeneratorInterface.TaskQueuePlace)}
        
        async def on_fire(self, consumed_tokens):
            task = consumed_tokens["task"][0]
            return {"queue": [task]}


# === TASK PROCESSING INTERFACE ===

class TaskProcessorInterface(Interface):
    """Interface for processing tasks with different handling strategies"""
    
    class TaskInputPlace(Place):
        """Input place for tasks to be processed"""
        pass
    
    class HighPriorityPlace(Place):
        """High priority tasks processed first"""
        pass
    
    class LowPriorityPlace(Place):
        """Low priority tasks processed after high priority"""
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
    
    class RouteTasksTransition(Transition):
        """Routes tasks by priority"""
        
        def input_arcs(self):
            return {"task": Arc(TaskProcessorInterface.TaskInputPlace)}
        
        def output_arcs(self):
            return {
                "high_priority": Arc(TaskProcessorInterface.HighPriorityPlace),
                "low_priority": Arc(TaskProcessorInterface.LowPriorityPlace)
            }
        
        async def on_fire(self, consumed_tokens):
            task = consumed_tokens["task"][0]
            
            if task.priority >= 3:
                return {"high_priority": [task]}
            else:
                return {"low_priority": [task]}
    
    class ProcessHighPriorityTransition(Transition):
        """Processes high priority tasks"""
        PRIORITY = 10  # Higher priority transition
        def input_arcs(self):
            return {"task": Arc(TaskProcessorInterface.HighPriorityPlace)}
        def output_arcs(self):
            return {
                "processing": Arc(TaskProcessorInterface.ProcessingPlace),
                "completed": Arc(TaskProcessorInterface.CompletedPlace),
                "failed": Arc(TaskProcessorInterface.FailedPlace)
            }
        async def guard(self, pending):
            # Consume all available high priority tasks
            for token in pending["task"]:
                self.consume("task", token)
            return bool(self._to_consume.get("task"))
        async def on_fire(self, consumed):
            for task in consumed["task"]:
                # Simulate processing
                start_time = time.time()
                await asyncio.sleep(0.2)
                processing_time = time.time() - start_time
                if task.task_id.endswith("6"):
                    error = ErrorToken("processing_failed", f"Task {task.task_id} failed validation")
                    self.produce("failed", error)
                else:
                    processed = ProcessedTaskToken(
                        task.task_id,
                        f"HIGH_PRIORITY_{task.task_type.upper()}",
                        processing_time
                    )
                    self.produce("completed", processed)
    
    class ProcessLowPriorityTransition(Transition):
        """Processes low priority tasks"""
        PRIORITY = 5  # Lower priority transition
        def input_arcs(self):
            return {"task": Arc(TaskProcessorInterface.LowPriorityPlace)}
        def output_arcs(self):
            return {
                "processing": Arc(TaskProcessorInterface.ProcessingPlace),
                "completed": Arc(TaskProcessorInterface.CompletedPlace),
                "failed": Arc(TaskProcessorInterface.FailedPlace)
            }
        async def guard(self, pending):
            # Consume all available low priority tasks
            for token in pending["task"]:
                self.consume("task", token)
            return bool(self._to_consume.get("task"))
        async def on_fire(self, consumed):
            for task in consumed["task"]:
                # Simulate processing
                start_time = time.time()
                await asyncio.sleep(0.5)
                processing_time = time.time() - start_time
                if task.task_id.endswith("9"):
                    error = ErrorToken("processing_timeout", f"Task {task.task_id} timed out")
                    self.produce("failed", error)
                else:
                    processed = ProcessedTaskToken(
                        task.task_id,
                        f"standard_{task.task_type}",
                        processing_time
                    )
                    self.produce("completed", processed)


# === NOTIFICATION INTERFACE ===

class NotificationInterface(Interface):
    """Interface for sending notifications via different channels"""
    
    class NotificationInputPlace(Place):
        """Input place for notifications to be sent"""
        pass
    
    class EmailOutputPlace(IOOutputPlace):
        """Sends email notifications"""
        
        def __init__(self):
            super().__init__()
            self.sent_emails = []
        
        async def on_token(self, token: Token) -> Optional[Token]:
            if isinstance(token, NotificationToken):
                # Simulate email sending
                await asyncio.sleep(0.1)
                
                # Simulate occasional email failures
                if "error" in token.message.lower():
                    error = ErrorToken("email_failed", f"Failed to send email to {token.recipient}")
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
        
        async def on_token(self, token: Token) -> Optional[Token]:
            if isinstance(token, NotificationToken) and token.urgent:
                # Simulate SMS sending
                await asyncio.sleep(0.05)
                
                self.sent_sms.append(token)
                print(f"📱 SMS: {token.recipient} - {token.message}")
                return None
            
            return None
    
    class LogOutputPlace(IOOutputPlace):
        """Logs all notifications"""
        
        async def on_token(self, token: Token) -> Optional[Token]:
            if isinstance(token, NotificationToken):
                timestamp = time.strftime("%H:%M:%S")
                print(f"📋 LOG [{timestamp}]: {token.recipient} - {token.message}")
                return None
            
            return None
    
    class SendNotificationTransition(Transition):
        """Sends notifications to multiple channels"""
        def input_arcs(self):
            return {"notification": Arc(NotificationInterface.NotificationInputPlace)}
        def output_arcs(self):
            return {
                "email": Arc(NotificationInterface.EmailOutputPlace),
                "sms": Arc(NotificationInterface.SMSOutputPlace),
                "log": Arc(NotificationInterface.LogOutputPlace)
            }
        async def guard(self, pending):
            # Consume all available notifications
            for token in pending["notification"]:
                self.consume("notification", token)
            return bool(self._to_consume.get("notification"))
        async def on_fire(self, consumed):
            for notification in consumed["notification"]:
                self.produce("email", notification)
                self.produce("sms", notification)
                self.produce("log", notification)


# === ERROR HANDLING INTERFACE ===

class ErrorHandlingInterface(Interface):
    """Interface for handling errors and retries"""
    
    class ErrorInputPlace(Place):
        """Input place for errors to be handled"""
        pass
    
    class ErrorLogPlace(IOOutputPlace):
        """Logs errors to external system"""
        
        async def on_token(self, token: Token) -> Optional[Token]:
            if isinstance(token, ErrorToken):
                timestamp = time.strftime("%H:%M:%S")
                print(f"❌ ERROR [{timestamp}]: {token.error_type} - {token.message}")
                return None
            
            return None
    
    class HandleErrorsTransition(Transition):
        """Handles processing errors"""
        def input_arcs(self):
            return {"error": Arc(ErrorHandlingInterface.ErrorInputPlace)}
        def output_arcs(self):
            return {"error_log": Arc(ErrorHandlingInterface.ErrorLogPlace)}
        async def guard(self, pending):
            for token in pending["error"]:
                self.consume("error", token)
            return bool(self._to_consume.get("error"))
        async def on_fire(self, consumed):
            for error in consumed["error"]:
                error_log = ErrorToken(
                    error.error_type,
                    f"Processing error logged: {error.message}",
                    error
                )
                self.produce("error_log", error_log)


# === SHUTDOWN INTERFACE ===

class ShutdownInterface(Interface):
    """Interface for handling graceful shutdown"""
    
    class ShutdownPlace(Place):
        """Holds shutdown signals"""
        pass
    
    class ShutdownOutputPlace(IOOutputPlace):
        """Outputs shutdown notifications"""
        
        async def on_token(self, token: Token) -> Optional[Token]:
            if isinstance(token, ShutdownToken):
                print(f"🛑 SHUTDOWN: {token.reason}")
                return None
            
            return None
    
    class HandleShutdownTransition(Transition):
        """Handles shutdown notifications"""
        def input_arcs(self):
            return {"shutdown": Arc(ShutdownInterface.ShutdownPlace)}
        def output_arcs(self):
            return {"output": Arc(ShutdownInterface.ShutdownOutputPlace)}
        async def guard(self, pending):
            for token in pending["shutdown"]:
                self.consume("shutdown", token)
            return bool(self._to_consume.get("shutdown"))
        async def on_fire(self, consumed):
            for shutdown in consumed["shutdown"]:
                self.produce("output", shutdown)


# === MAIN SYSTEM ===

class TaskProcessingSystem(PetriNet):
    """Comprehensive task processing system demonstrating proper interface boundaries"""
    
    # Compose all interfaces
    class TaskGen(TaskGeneratorInterface): pass
    class TaskProc(TaskProcessorInterface): pass
    class Notify(NotificationInterface): pass
    class ErrorHandle(ErrorHandlingInterface): pass
    class Shutdown(ShutdownInterface): pass
    
    # Top-level transitions that connect interfaces
    class ConnectGeneratorToProcessorTransition(Transition):
        """Connects task generator to processor"""
        def input_arcs(self):
            return {"task": Arc(TaskProcessingSystem.TaskGen.TaskQueuePlace)}
        def output_arcs(self):
            return {"processor_input": Arc(TaskProcessingSystem.TaskProc.TaskInputPlace)}
        async def guard(self, pending):
            for token in pending["task"]:
                self.consume("task", token)
            return bool(self._to_consume.get("task"))
        async def on_fire(self, consumed):
            for task in consumed["task"]:
                self.produce("processor_input", task)
    
    class ConnectProcessorToNotificationTransition(Transition):
        """Connects processor output to notification system"""
        def input_arcs(self):
            return {"completed": Arc(TaskProcessingSystem.TaskProc.CompletedPlace)}
        def output_arcs(self):
            return {"notification": Arc(TaskProcessingSystem.Notify.NotificationInputPlace)}
        async def guard(self, pending):
            for token in pending["completed"]:
                self.consume("completed", token)
            return bool(self._to_consume.get("completed"))
        async def on_fire(self, consumed):
            for completed in consumed["completed"]:
                is_urgent = "HIGH_PRIORITY" in completed.result
                notification = NotificationToken(
                    recipient="admin@example.com",
                    message=f"Task {completed.task_id} completed: {completed.result}",
                    urgent=is_urgent
                )
                self.produce("notification", notification)
    
    class ConnectProcessorToErrorHandlingTransition(Transition):
        """Connects processor errors to error handling"""
        def input_arcs(self):
            return {"error": Arc(TaskProcessingSystem.TaskProc.FailedPlace)}
        def output_arcs(self):
            return {"error_input": Arc(TaskProcessingSystem.ErrorHandle.ErrorInputPlace)}
        async def guard(self, pending):
            for token in pending["error"]:
                self.consume("error", token)
            return bool(self._to_consume.get("error"))
        async def on_fire(self, consumed):
            for error in consumed["error"]:
                self.produce("error_input", error)
    
    class TriggerShutdownTransition(Transition):
        """Triggers shutdown when all tasks are processed"""
        def input_arcs(self):
            return {}  # No input arcs needed
        def output_arcs(self):
            return {"shutdown": Arc(TaskProcessingSystem.Shutdown.ShutdownPlace)}
        async def guard(self, pending):
            # Shutdown when task generator is done and system is idle
            task_source = self._net._get_place_by_type(TaskProcessingSystem.TaskGen.TaskSourcePlace)
            generator_done = task_source.tasks_generated >= task_source.max_tasks
            task_queue = self._net._get_place_by_type(TaskProcessingSystem.TaskGen.TaskQueuePlace)
            task_input = self._net._get_place_by_type(TaskProcessingSystem.TaskProc.TaskInputPlace)
            high_priority = self._net._get_place_by_type(TaskProcessingSystem.TaskProc.HighPriorityPlace)
            low_priority = self._net._get_place_by_type(TaskProcessingSystem.TaskProc.LowPriorityPlace)
            processing = self._net._get_place_by_type(TaskProcessingSystem.TaskProc.ProcessingPlace)
            system_idle = (task_queue.is_empty and task_input.is_empty and \
                          high_priority.is_empty and low_priority.is_empty and \
                          processing.is_empty)
            completed = self._net._get_place_by_type(TaskProcessingSystem.TaskProc.CompletedPlace)
            has_processed_tasks = completed.token_count > 0
            return generator_done and system_idle and has_processed_tasks
        async def on_fire(self, consumed):
            # Stop the task generator
            task_source = self._net._get_place_by_type(TaskProcessingSystem.TaskGen.TaskSourcePlace)
            await task_source.stop_input()
            self._net._is_finished = True
            shutdown = ShutdownToken("All tasks processed")
            self.produce("shutdown", shutdown)
    
    def _should_terminate(self) -> bool:
        """Custom termination logic"""
        return self._is_finished


async def demonstrate_comprehensive_system():
    """Demonstrate all major Cordyceps features with proper interface boundaries"""
    print("=== Cordyceps Fixed Interface Boundaries Demo ===")
    print("Features demonstrated:")
    print("- Proper interface composition without boundary violations")
    print("- Fully qualified names for all components")
    print("- Configurable IOInputPlace")
    print("- IOOutputPlace with error handling")
    print("- Priority-based processing")
    print("- Multiple output channels")
    print("- Error handling and logging")
    print("- Graceful shutdown")
    print("- Async execution model")
    print()
    
    # Create system with validated interface boundaries
    system = TaskProcessingSystem()
    
    # Configure the task generator after creation
    task_source = system.get_place(TaskProcessingSystem.TaskGen.TaskSourcePlace)
    task_source.task_types = ["analysis", "computation", "validation", "reporting"]
    task_source.max_tasks = 8
    task_source.interval = 0.8
    
    print("=== System Configuration ===")
    print(f"Task types: {task_source.task_types}")
    print(f"Max tasks: {task_source.max_tasks}")
    print(f"Generation interval: {task_source.interval}s")
    print(f"Total places: {len(system._places_by_type)}")
    print(f"Total transitions: {len(system._transitions)}")
    print(f"IO input places: {len(system._io_input_places)}")
    print(f"IO output places: {len(system._io_output_places)}")
    print()
    
    # Show qualified names
    print("=== Qualified Names ===")
    print("Places:")
    for place in system._places_by_type.values():
        print(f"  {place.qualified_name}")
    print("Transitions:")
    for transition in system._transitions:
        print(f"  {transition.qualified_name}")
    print()
    
    # Set up signal handling for graceful shutdown
    shutdown_requested = False
    
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        print(f"\n🔔 Received signal {signum}, initiating graceful shutdown...")
        shutdown_requested = True
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("=== Starting System (Ctrl+C for graceful shutdown) ===")
    
    # Start the system
    start_time = time.time()
    system_task = asyncio.create_task(system.start())
    
    try:
        # Monitor for shutdown
        while not shutdown_requested and not system._is_finished:
            await asyncio.sleep(0.1)
        
        if shutdown_requested:
            # Stop task generation
            await task_source.stop_input()
            system._is_finished = True
        
        # Wait for system to complete
        await asyncio.wait_for(system_task, timeout=10.0)
        
    except asyncio.TimeoutError:
        print("⚠️  System didn't complete in time, forcing shutdown")
        await system.stop()
    
    elapsed = time.time() - start_time
    
    print(f"\n=== System Completed in {elapsed:.1f}s ===")
    
    # Show final statistics
    state = system.get_state_summary()
    
    print("\n=== Final Statistics ===")
    email_place = system.get_place(TaskProcessingSystem.Notify.EmailOutputPlace)
    sms_place = system.get_place(TaskProcessingSystem.Notify.SMSOutputPlace)
    
    print(f"Tasks generated: {task_source.tasks_generated}")
    print(f"Emails sent: {len(email_place.sent_emails)}")
    print(f"SMS sent: {len(sms_place.sent_sms)}")
    print(f"Transitions fired: {len(state['firing_log'])}")
    
    print("\n=== Token Distribution ===")
    for place_name, place_info in state['places'].items():
        if place_info['token_count'] > 0:
            print(f"{place_name}: {place_info['token_count']} tokens")
    
    print("\n=== Transition Firing Log ===")
    for time_str, transition_name in state['firing_log'][-10:]:  # Show last 10
        print(f"{transition_name}")
    
    print("\n=== Generated Mermaid Diagram ===")
    print(system.generate_mermaid_diagram())


if __name__ == "__main__":
    asyncio.run(demonstrate_comprehensive_system())