#!/usr/bin/env python3
"""
BT-in-PN with Spores Logging

This example demonstrates:
1. Using both HyphaAdapter and MyceliumAdapter together
2. Logging Petri net transitions
3. Logging BT execution within transitions
4. Object lifecycle tracking across patterns
5. Event relationships

The scenario: A job queue processor modeled as a Petri net, with
behavior trees handling individual job processing.
"""

import sys
sys.path.insert(0, "src")

import asyncio
import tempfile
from pathlib import Path
from pydantic import BaseModel
from typing import Annotated, List

from mycorrhizal.hypha.core import pn, PlaceType
from mycorrhizal.rhizomorph.core import bt, Status
from mycorrhizal.mycelium import PNRunner
from mycorrhizal.common.timebase import MonotonicClock
from mycorrhizal.spores import configure, HyphaAdapter, RhizomorphAdapter
from mycorrhizal.spores.models import EventAttr, ObjectRef, ObjectScope
from mycorrhizal.spores.transport import AsyncFileTransport


# ======================================================================================
# Domain Models
# ======================================================================================


class Job(BaseModel):
    """Job to process."""
    id: str
    type: str
    priority: int


class JobQueue(BaseModel):
    """Job queue object."""
    id: str
    pending_jobs: List[Job]
    completed_jobs: List[Job]


# ======================================================================================
# Blackboard with Spores Annotations
# ======================================================================================


class ProcessingContext(BaseModel):
    """
    Blackboard for job processing.
    """
    job_queue: Annotated[JobQueue, ObjectRef(qualifier="queue", scope=ObjectScope.GLOBAL)]
    current_job: Annotated[Job, ObjectRef(qualifier="target", scope=ObjectScope.EVENT)]
    processor_id: Annotated[str, EventAttr]
    processing_time: Annotated[float, EventAttr]


# ======================================================================================
# Behavior Tree for Job Processing
# ======================================================================================


# Create Rhizomorph adapter for BT logging
bt_adapter = RhizomorphAdapter()


@bt.tree
def JobProcessor():
    """
    Behavior tree for processing a single job.

    Logs each node execution with status and blackboard attributes.
    """

    @bt.action
    @bt_adapter.log_node(event_type="validate_job")
    async def validate_job(bb: ProcessingContext) -> Status:
        """Validate the job is ready to process."""
        if bb.current_job.type in ["data", "compute", "io"]:
            print(f"    Validated job {bb.current_job.id} (type: {bb.current_job.type})")
            return Status.SUCCESS
        print(f"    Rejected job {bb.current_job.id} (unknown type: {bb.current_job.type})")
        return Status.FAILURE

    @bt.action
    @bt_adapter.log_node(event_type="process_job")
    async def process_job(bb: ProcessingContext) -> Status:
        """Process the job."""
        # Simulate processing time
        await asyncio.sleep(0.01)

        bb.processing_time = 0.01
        print(f"    Processed job {bb.current_job.id} by {bb.processor_id}")
        return Status.SUCCESS

    @bt.action
    @bt_adapter.log_node(event_type="complete_job")
    async def complete_job(bb: ProcessingContext) -> Status:
        """Mark job as complete."""
        bb.job_queue.completed_jobs.append(bb.current_job)
        print(f"    Completed job {bb.current_job.id}")
        return Status.SUCCESS

    @bt.root
    @bt.sequence
    def main():
        """Main processing sequence."""
        yield validate_job
        yield process_job
        yield complete_job


# ======================================================================================
# Petri Net with Spores Logging
# ======================================================================================


# Create Hypha adapter for PN logging
pn_adapter = HyphaAdapter()


@pn.net
def JobQueueNet(builder):
    """
    Petri net for job queue processing.

    Places:
    - pending: Jobs waiting to be processed
    - processing: Jobs currently being processed
    - completed: Jobs that have been processed

    Transitions:
    - dispatch: Move job from pending to processing
    - process: Execute BT to process the job
    - finish: Move job from processing to completed

    Each transition is logged with spores.
    """

    @builder.place(type=PlaceType.QUEUE)
    def pending(bb: ProcessingContext):
        """Pending jobs place."""
        return [job for job in bb.job_queue.pending_jobs]

    @builder.place(type=PlaceType.QUEUE)
    def processing(bb: ProcessingContext):
        """Jobs currently being processed."""
        return []

    @builder.place(type=PlaceType.QUEUE)
    def completed(bb: ProcessingContext):
        """Completed jobs place."""
        return [job for job in bb.job_queue.completed_jobs]

    @builder.transition()
    @pn_adapter.log_transition(event_type="dispatch_job")
    async def dispatch(consumed, bb: ProcessingContext, timebase):
        """
        Dispatch a job from pending to processing.

        Logged with:
        - event type: dispatch_job
        - attributes from blackboard (EventAttr annotations)
        - object relationships (queue, current_job)
        """
        if bb.job_queue.pending_jobs:
            job = bb.job_queue.pending_jobs.pop(0)
            bb.current_job = job
            print(f"  [Dispatch] Job {job.id} -> processing")
            yield {processing: [job]}

    @builder.transition()
    @pn_adapter.log_transition(event_type="execute_processing")
    async def process(consumed, bb: ProcessingContext, timebase):
        """
        Execute the BT to process the job.

        This transition runs the JobProcessor BT tree.

        Logged with:
        - event type: execute_processing
        - BT execution results (from RhizomorphAdapter)
        - object relationships
        """
        job_tokens = consumed[0] if consumed else []
        if job_tokens:
            # Run the BT to process this job
            from mycorrhizal.rhizomorph.core import Runner as BTRunner
            bt_runner = BTRunner(JobProcessor, bb=bb, tb=timebase)
            result = await bt_runner.run(max_ticks=10)
            print(f"  [Process] BT result: {result.name}")

            # Output to completed place
            if bb.current_job in bb.job_queue.completed_jobs:
                yield {completed: [bb.current_job]}

    @builder.transition()
    @pn_adapter.log_transition(event_type="acknowledge_complete")
    async def finish(consumed, bb: ProcessingContext, timebase):
        """
        Acknowledge job completion.

        Logged with:
        - event type: acknowledge_complete
        - completion confirmation
        """
        job_tokens = consumed[0] if consumed else []
        if job_tokens:
            job = job_tokens[0]
            print(f"  [Finish] Job {job.id} acknowledged")
            # Already moved to completed by BT, just acknowledge


# ======================================================================================
# Main Execution
# ======================================================================================


async def main():
    """Run the job queue processor with Spores logging."""

    print("=" * 70)
    print("Mycelium + Spores: BT-in-PN with Event Logging")
    print("=" * 70)
    print()
    print("This demo shows:")
    print("  - Petri net (Hypha) with BT-in-transition")
    print("  - Spores logging for both PN and BT")
    print("  - HyphaAdapter for transition logging")
    print("  - RhizomorphAdapter for node logging")
    print("  - Object lifecycle tracking (queue, jobs)")
    print("  - Event-object relationships")
    print()

    # Create temporary file for spores output
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        log_file = f.name

    print(f"Logging to: {log_file}")
    print()

    # Configure spores
    configure(transport=AsyncFileTransport(log_file))

    # Create job queue with jobs
    jobs = [
        Job(id="job-1", type="data", priority=1),
        Job(id="job-2", type="compute", priority=2),
        Job(id="job-3", type="io", priority=1),
    ]

    job_queue = JobQueue(
        id="queue-1",
        pending_jobs=jobs,
        completed_jobs=[],
    )

    bb = ProcessingContext(
        job_queue=job_queue,
        current_job=Job(id="", type="", priority=0),  # Placeholder
        processor_id="processor-1",
        processing_time=0.0,
    )

    # Create PN runner
    runner = PNRunner(JobQueueNet, blackboard=bb)

    print("Initial state:")
    print(f"  Queue: {job_queue.id}")
    print(f"  Pending jobs: {len(job_queue.pending_jobs)}")
    print()

    # Start the runner
    print("Running job queue...")
    print("-" * 70)

    await runner.start(MonotonicClock())

    print("-" * 70)
    print()
    print("Final state:")
    print(f"  Jobs completed: {len(job_queue.completed_jobs)} / {len(jobs)}")
    print(f"  Completed job IDs: {[j.id for j in job_queue.completed_jobs]}")
    print()

    # Show log file contents
    print("=" * 70)
    print("Spores Event Log (OCEL format)")
    print("=" * 70)
    print()

    with open(log_file, 'r') as f:
        for i, line in enumerate(f, 1):
            print(f"Record {i}: {line.rstrip()}")

    print()
    print("=" * 70)
    print("What was logged:")
    print("  - Events:")
    print("    - dispatch_job (PN transition)")
    print("    - execute_processing (PN transition)")
    print("    - validate_job, process_job, complete_job (BT nodes)")
    print("  - Objects:")
    print("    - JobQueue (GLOBAL scope)")
    print("    - Jobs (EVENT scope)")
    print("  - Relationships:")
    print("    - event --[queue]--> queue-1")
    print("    - event --[target]--> job-N")
    print()

    # Clean up
    Path(log_file).unlink()


if __name__ == "__main__":
    asyncio.run(main())
