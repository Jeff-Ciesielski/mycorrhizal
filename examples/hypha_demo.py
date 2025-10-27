#!/usr/bin/env python3
"""
Port of the original `hypha_demo.py` to the core2 decorator/builder API.

This example demonstrates:
- IO input place (task generator)
- Worker subnet instantiated multiple times
- Processing transition with occasional failures
- Notification outputs (email/sms/log)
- Error handling forward
- Completion tracking via IO output place
"""

import asyncio
import time
from asyncio import sleep, Event
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from mycorrhizal.hypha.core import pn, PlaceType, Runner
from mycorrhizal.hypha.core.builder import NetBuilder
from mycorrhizal.common.timebase import CycleClock, MonotonicClock
from dataclasses import dataclass



class Blackboard(BaseModel):
    model_config = {"arbitrary_types_allowed": True} # Required to enable events
    total_tasks_to_process: int = 8
    tasks_completed: int = 0
    # optional configuration for generator
    task_types: List[str] = ["analysis", "computation", "validation", "reporting"]
    max_tasks: int = 8
    interval: float = 0.8
    finished_event: Event = Event()


class TaskToken:
    def __init__(self, task_id: str, task_type: str):
        self.task_id = task_id
        self.task_type = task_type

    def __repr__(self):
        return f"TaskToken({self.task_id}, {self.task_type})"


class ProcessedTaskToken:
    def __init__(self, task_id: str, result: str, processing_time: float):
        self.task_id = task_id
        self.result = result
        self.processing_time = processing_time

    def __repr__(self):
        return f"ProcessedTaskToken({self.task_id}, {self.result})"


class NotificationToken:
    def __init__(self, recipient: str, message: str, urgent: bool = False):
        self.recipient = recipient
        self.message = message
        self.urgent = urgent

    def __repr__(self):
        return f"NotificationToken({self.recipient}, urgent={self.urgent})"


class ErrorToken:
    def __init__(self, error_type: str, message: str, original_token: Any = None):
        self.error_type = error_type
        self.message = message
        self.original_token = original_token

    def __repr__(self):
        return f"ErrorToken({self.error_type}, {self.message})"


@pn.net
def TaskGenerator(builder: NetBuilder):
    """Configurable task generator interface 

    Exposes an IO input place that yields TaskToken instances.
    """

    # configurable generator as IO input place
    @builder.io_input_place()
    async def source(bb: Blackboard, timebase):
        # bb may provide configuration; if not, use defaults
        task_types = bb.task_types
        max_tasks = bb.max_tasks
        interval = bb.interval

        for i in range(max_tasks):
            await timebase.sleep(interval)
            task_type = task_types[i % len(task_types)]
            task = TaskToken(task_id=f"task_{i}", task_type=task_type)
            print(f"Generated: {task}")
            yield task


@pn.net
def TaskProcessor(builder: NetBuilder):
    """Processor subnet that consumes tasks and emits completed or failed tokens"""
    # places
    input = builder.place("input", type=PlaceType.QUEUE)
    processing = builder.place("processing", type=PlaceType.BAG)
    completed = builder.place("completed", type=PlaceType.QUEUE)
    failed = builder.place("failed", type=PlaceType.QUEUE)

    @builder.transition()
    async def take_to_processing(consumed, bb, timebase):
        for t in consumed:
            print(f"Starting: {t}")
            yield {processing: t}

    @builder.transition()
    async def do_processing(consumed, bb, timebase):
        out_completed = []
        out_failed = []
        for t in consumed:
            print(f"Processing: {t}")
            start = time.time()
            await timebase.sleep(0.3)
            processing_time = time.time() - start

            if str(t.task_id).endswith("6") or str(t.task_id).endswith("9"):
                err = ErrorToken("processing_failed", f"Task {t.task_id} failed", t)
                out_failed.append(err)
                print(f"❌ Failed: {t}")
            else:
                processed = ProcessedTaskToken(
                    t.task_id, f"processed_{t.task_type}", processing_time
                )
                out_completed.append(processed)
                print(f"✅ Completed: {t}")

        if out_completed:
            yield {completed: out_completed}
        if out_failed:
            yield {failed: out_failed}

    # wire arcs inside the subnet
    builder.arc(input, take_to_processing).arc(processing)
    builder.arc(processing, do_processing)
    builder.arc(do_processing, completed)
    builder.arc(do_processing, failed)


@pn.net
def Notification(builder: NetBuilder):
    """Notification interface: forks notifications to email/sms/log"""

    notification_input = builder.place("input", type=PlaceType.QUEUE)

    @builder.io_output_place()
    async def email_sink(token, bb, timebase):
        if isinstance(token, NotificationToken):
            await timebase.sleep(0.1)
            if "error" in token.message.lower():
                err = ErrorToken("email_failed", f"Failed to send to {token.recipient}")
                print(f"❌ EMAIL FAILED: {token.recipient}")
                return err
            print(f"📧 EMAIL: {token.recipient} - {token.message}")
        return None

    @builder.io_output_place()
    async def sms_sink(token, bb, timebase):
        if isinstance(token, NotificationToken) and token.urgent:
            await timebase.sleep(0.05)
            print(f"📱 SMS: {token.recipient} - {token.message}")
        return None

    @builder.io_output_place()
    async def log_sink(token, bb, timebase):
        if isinstance(token, NotificationToken):
            ts = time.strftime("%H:%M:%S")
            print(f"📋 LOG [{ts}]: {token.recipient} - {token.message}")
        return None

    # wiring: fork from input to outputs
    builder.fork(
        notification_input, [email_sink, sms_sink, log_sink], name="NotificationFork"
    )


@pn.net
def ErrorHandler(builder: NetBuilder):
    err_in = builder.place("input", type=PlaceType.QUEUE)

    @builder.io_output_place()
    async def error_log(token, bb, timebase):
        if isinstance(token, ErrorToken):
            ts = time.strftime("%H:%M:%S")
            print(f"❌ ERROR [{ts}]: {token.error_type} - {token.message}")
        return None

    builder.forward(err_in, error_log, name="ErrorForward")


@pn.net
def TaskProcessingSystem(builder: NetBuilder):
    """Top-level system composing interfaces"""

    # instantiate interfaces/subnets
    gen = builder.subnet(TaskGenerator, "TaskGen")
    proc = builder.subnet(TaskProcessor, "TaskProc")
    notify = builder.subnet(Notification, "Notify")
    err = builder.subnet(ErrorHandler, "ErrorHandle")

    # completion tracker as IO output place
    @builder.io_output_place()
    async def completion_tracker(token, bb: Blackboard, timebase):
        # token may be ProcessedTaskToken or ErrorToken
        if bb is not None and hasattr(bb, "tasks_completed"):
            bb.tasks_completed += 1
            print(
                f"📊 Progress: {bb.tasks_completed}/{bb.total_tasks_to_process} tasks completed"
            )
            if bb.tasks_completed >= bb.total_tasks_to_process:
                print("🎉 All tasks completed! Signaling shutdown...")
                # If the blackboard exposes an Event, set it
                if hasattr(bb, "finished_event") and isinstance(
                    bb.finished_event, Event
                ):
                    bb.finished_event.set()
        return None

    # connect generator to processor
    builder.forward(gen.source, proc.input)

    # connect processor outputs: completed -> notification input and completion tracker
    builder.fork(proc.completed, [notify.input, completion_tracker], name="CompletionFork")

    # failed -> error handler and completion tracker
    builder.fork(proc.failed, [err.input, completion_tracker], name="FailureFork")


async def demonstrate_comprehensive_system():
    print("=" * 60)
    print("Hypha Comprehensive Demo")
    print("=" * 60)

    bb = Blackboard()
    timebase = CycleClock(stepsize=0.1)

    runner = Runner(TaskProcessingSystem, bb)

    print("\nMermaid Diagram:")
    print("-" * 40)
    print(TaskProcessingSystem.to_mermaid())
    print("-" * 40)

    await runner.start(timebase)

    try:
        print("Running system until all tasks are processed...")
        start_time = timebase.now()
        while not bb.finished_event.is_set():
            timebase.advance()
            await asyncio.sleep(0) # let other tasks run
            if timebase.now() - start_time > 10:
                print("Timeout waiting for all tasks to complete.")
                break
    finally:
        print("Stopping system...")
        await runner.stop(timeout=1)

    print(f"\nProcessed {bb.tasks_completed} tasks")


if __name__ == "__main__":
    asyncio.run(demonstrate_comprehensive_system())
