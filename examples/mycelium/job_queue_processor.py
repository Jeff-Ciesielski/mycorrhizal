#!/usr/bin/env python3
"""
Job Queue Processor - BT-in-PN Integration Example

Demonstrates a job queue system that uses Behavior Trees integrated with
Petri Nets to route and process background jobs.

Use case:
- Web servers submit background jobs (email sending, report generation, data processing)
- Jobs have different priorities (CRITICAL: user-facing, HIGH: async operations, LOW: cleanup)
- Jobs need validation before processing (check dependencies, resource availability)
- Failed jobs should be rejected and retried later
- System must handle job enrichment (add metadata, timestamps, etc.)

BT-in-PN integration:
1. Defer complex routing logic to Behavior Trees
2. Validate and enrich jobs during routing
3. Handle failures without losing jobs
4. Keep Petri Net structure simple and declarative

Architecture:
    Input Queue → Priority Router (BT) → [Critical|Fast|Slow] Queues
                                    → Validator (BT) → [Validated|Rejected]
                                    → Completion Logger
"""

import asyncio
import logging
from enum import Enum
from pydantic import BaseModel, ConfigDict

from mycorrhizal.mycelium import pn, PNRunner, PlaceType
from mycorrhizal.rhizomorph.core import bt, Status
from mycorrhizal.common.timebase import MonotonicClock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ==================================================================================
# Domain Models
# ==================================================================================


class JobPriority(Enum):
    """Job priority levels matching real-world queue systems."""
    LOW = 1        # Cleanup, maintenance, analytics
    MEDIUM = 2     # Normal background processing
    HIGH = 3       # User-triggered async operations
    CRITICAL = 4   # Time-sensitive, user-facing operations


class Job(BaseModel):
    """A background job to be processed."""
    job_id: str
    priority: JobPriority
    job_type: str  # "email", "report", "backup", "cleanup", etc.
    payload: str
    requires_validation: bool = False
    is_valid: bool = True
    retry_count: int = 0

    def __repr__(self):
        return f"Job({self.job_id}, {self.job_type}, prio={self.priority.name})"


class Blackboard(BaseModel):
    """Shared state for the job queue system."""
    model_config = ConfigDict(extra='allow')  # Allow pn_ctx to be injected
    jobs_processed: int = 0
    jobs_failed: int = 0
    enrichment_count: int = 0


# ==================================================================================
# Behavior Trees for Job Processing
# ==================================================================================


@bt.tree
def PriorityRouter():
    """
    BT that routes jobs to appropriate priority queues.

    Implement complex routing logic here:
    - Check system load before accepting CRITICAL jobs
    - Throttle LOW priority jobs during peak hours
    - Route specific job types to dedicated queues
    - Add dynamic priority adjustments based on SLA
    """

    @bt.action
    async def route_critical(bb):
        """Route CRITICAL jobs to dedicated high-priority queue."""
        pn_ctx = bb.pn_ctx
        job = pn_ctx.current_token

        if job.priority == JobPriority.CRITICAL:
            logger.info(f"[Router] CRITICAL job {job.job_id} ({job.job_type}) → critical pending")
            pn_ctx.route_to(pn_ctx.places.critical_pending)
            return Status.SUCCESS
        return Status.FAILURE

    @bt.action
    async def route_high(bb):
        """Route HIGH priority jobs to fast processing queue."""
        pn_ctx = bb.pn_ctx
        job = pn_ctx.current_token

        if job.priority == JobPriority.HIGH:
            logger.info(f"[Router] HIGH priority job {job.job_id} ({job.job_type}) → fast pending")
            pn_ctx.route_to(pn_ctx.places.fast_pending)
            return Status.SUCCESS
        return Status.FAILURE

    @bt.action
    async def route_normal(bb):
        """Route normal and LOW priority jobs to standard processing queue."""
        pn_ctx = bb.pn_ctx
        job = pn_ctx.current_token

        logger.info(f"[Router] Normal priority job {job.job_id} ({job.job_type}) → normal pending")
        pn_ctx.route_to(pn_ctx.places.normal_pending)
        return Status.SUCCESS

    @bt.action
    async def reject_unknown(bb):
        """Fallback for jobs that don't match expected categories."""
        pn_ctx = bb.pn_ctx
        job = pn_ctx.current_token

        logger.warning(f"[Router] Unknown job type for {job.job_id}, rejecting")
        pn_ctx.reject_token("Unknown job type")
        return Status.SUCCESS

    @bt.root
    @bt.selector
    def root():
        """Try routing strategies in order of specificity."""
        yield route_critical
        yield route_high
        yield route_normal
        yield reject_unknown


@bt.tree
def JobValidator():
    """
    BT that validates jobs before final processing.

    Validation scenarios:
    - Check if job dependencies are satisfied
    - Verify required resources are available
    - Validate job parameters and permissions
    - Check rate limits or quotas
    - Enrich job with metadata (timestamps, IDs, etc.)
    """

    @bt.condition
    def requires_validation(bb):
        """Check if this job needs validation."""
        pn_ctx = bb.pn_ctx
        job = pn_ctx.current_token
        return job.requires_validation

    @bt.action
    async def validate_job(bb):
        """Validate job and either enrich or reject it."""
        pn_ctx = bb.pn_ctx
        job = pn_ctx.current_token

        # Simulate validation logic
        if not job.is_valid:
            logger.warning(f"[Validator] Job {job.job_id} failed validation, routing to rejected queue")
            pn_ctx.route_to(pn_ctx.places.rejected_queue)
            return Status.SUCCESS

        # Enrich job with metadata
        job_enriched = job.model_copy()
        job_enriched.retry_count += 1
        # In production, add: created_at, validation_id, checksum, etc.

        logger.info(f"[Validator] Job {job.job_id} validated and enriched")
        pn_ctx.route_to(pn_ctx.places.validated_queue, token=job_enriched)
        return Status.SUCCESS

    @bt.action
    async def pass_through(bb):
        """Pass through jobs that don't require validation."""
        pn_ctx = bb.pn_ctx
        job = pn_ctx.current_token
        logger.info(f"[Validator] Job {job.job_id} doesn't require validation, passing through")
        pn_ctx.route_to(pn_ctx.places.validated_queue)
        return Status.SUCCESS

    @bt.sequence
    def validate_if_needed():
        """Sequence: check if validation needed, then validate."""
        yield requires_validation
        yield validate_job

    @bt.root
    @bt.selector
    def root():
        """Either validate if needed, or pass through."""
        yield validate_if_needed
        yield pass_through


# ==================================================================================
# Petri Net Definition
# ==================================================================================


@pn.net
def JobQueueProcessor(builder):
    """
    Job queue processing system with BT-intelligent routing.

    PN structure:
    - Jobs flow from input through router to validation and completion
    - BTs handle decision logic at each transition
    - Extend by adding new queues, validators, or processing stages
    """

    # Places
    input_queue = builder.place("input_queue", type=PlaceType.QUEUE)
    critical_pending = builder.place("critical_pending", type=PlaceType.QUEUE)
    fast_pending = builder.place("fast_pending", type=PlaceType.QUEUE)
    normal_pending = builder.place("normal_pending", type=PlaceType.QUEUE)
    validated_queue = builder.place("validated_queue", type=PlaceType.QUEUE)
    rejected_queue = builder.place("rejected_queue", type=PlaceType.QUEUE)

    # Transition 1: Priority Router (BT-in-PN, token mode)
    @builder.transition(bt=PriorityRouter, bt_mode="token", outputs=[critical_pending, fast_pending, normal_pending])
    async def route_by_priority(consumed, bb, timebase):
        """Route jobs based on priority using BT decisions."""
        # BT handles all routing logic

    # Transition 2: Critical path validator
    @builder.transition(bt=JobValidator, bt_mode="token", outputs=[validated_queue, rejected_queue])
    async def validate_critical(consumed, bb, timebase):
        """Validate critical priority jobs using BT."""
        # BT handles validation, enrichment, and routing

    # Transition 3: Fast path validator
    @builder.transition(bt=JobValidator, bt_mode="token", outputs=[validated_queue, rejected_queue])
    async def validate_fast(consumed, bb, timebase):
        """Validate fast priority jobs using BT."""
        # BT handles validation, enrichment, and routing

    # Transition 4: Normal path validator
    @builder.transition(bt=JobValidator, bt_mode="token", outputs=[validated_queue, rejected_queue])
    async def validate_normal(consumed, bb, timebase):
        """Validate normal priority jobs using BT."""
        # BT handles validation, enrichment, and routing

    # Vanilla transition (no BT) - simple completion logging
    @builder.transition()
    async def log_completion(consumed, bb, timebase):
        """Log completed jobs (final processing step)."""
        for job in consumed:
            logger.info(f"[Completed] Job {job.job_id} ({job.job_type}) processed successfully")
            bb.jobs_processed += 1

    # Wire the net
    # input → priority router
    builder.arc(input_queue, route_by_priority)
    builder.arc(route_by_priority, critical_pending)
    builder.arc(route_by_priority, fast_pending)
    builder.arc(route_by_priority, normal_pending)

    # each pending queue → its own validator
    builder.arc(critical_pending, validate_critical)
    builder.arc(validate_critical, validated_queue)
    builder.arc(validate_critical, rejected_queue)

    builder.arc(fast_pending, validate_fast)
    builder.arc(validate_fast, validated_queue)
    builder.arc(validate_fast, rejected_queue)

    builder.arc(normal_pending, validate_normal)
    builder.arc(validate_normal, validated_queue)
    builder.arc(validate_normal, rejected_queue)

    # validated → completion
    builder.arc(validated_queue, log_completion)


# ==================================================================================
# Demo Execution
# ==================================================================================


async def main():
    """Run the job queue processor demo."""

    print("=" * 80)
    print("Job Queue Processor - BT-in-PN Integration Demo")
    print("=" * 80)
    print()
    print("This demo shows a production job queue system with:")
    print("  - Priority-based routing (CRITICAL > HIGH > MEDIUM > LOW)")
    print("  - Job validation and enrichment")
    print("  - Graceful rejection of invalid jobs")
    print("  - Behavior Trees making intelligent routing decisions")
    print()

    bb = Blackboard()
    timebase = MonotonicClock()

    runner = PNRunner(JobQueueProcessor, bb)

    # Generate and display Mermaid diagram
    print("\n" + "=" * 80)
    print("Mermaid Diagram")
    print("=" * 80)
    mermaid = runner.to_mermaid()
    print(mermaid)
    print("=" * 80)
    print()
    print("BTs are embedded as subgraphs within transitions.")
    print("Subgraphs show BT structure:")
    print("  - PriorityRouter: Selector with 4 routing actions")
    print("  - JobValidator: Selector with validation sequence and pass-through")
    print()

    await runner.start(timebase)

    # Get runtime to access places
    runtime = runner.runtime

    # Find input queue
    if runtime.places is None:
        print("ERROR: Runtime places is None!")
        return

    print(f"\nAvailable places: {list(runtime.places.keys())}")
    input_place = None
    for key, place in runtime.places.items():
        if "input_queue" in key:
            input_place = place
            break

    if not input_place:
        print("ERROR: Could not find input queue!")
        return

    # Submit jobs (simulating real job submissions)
    print("\nSubmitting jobs to queue...")
    jobs = [
        Job(job_id="job-1", job_type="email", priority=JobPriority.CRITICAL,
            payload="Send welcome email to new user", requires_validation=True),
        Job(job_id="job-2", job_type="report", priority=JobPriority.HIGH,
            payload="Generate monthly sales report", requires_validation=False),
        Job(job_id="job-3", job_type="backup", priority=JobPriority.MEDIUM,
            payload="Database backup", requires_validation=True, is_valid=False),  # Will fail validation
        Job(job_id="job-4", job_type="cleanup", priority=JobPriority.LOW,
            payload="Clean up old logs", requires_validation=False),
        Job(job_id="job-5", job_type="email", priority=JobPriority.HIGH,
            payload="Send password reset email", requires_validation=False),
    ]

    for job in jobs:
        input_place.add_token(job)
        print(f"  Submitted: {job}")

    print()
    print("Processing jobs...")
    print("-" * 80)

    # Let the system process
    await asyncio.sleep(3.0)

    print("-" * 80)
    print()
    print("Final state:")

    # Print queue states
    if runtime.places is None:
        print("ERROR: Runtime places is None!")
        return

    for name_parts, place in runtime.places.items():
        place_name = name_parts[-1]
        if place.tokens:
            print(f"\n{place_name.replace('_', ' ').title()}: {len(place.tokens)} jobs")
            for job in place.tokens:
                print(f"  {job}")

    print("\nSystem statistics:")
    print(f"  Jobs processed: {bb.jobs_processed}")
    print(f"  Enrichment operations: {bb.enrichment_count}")

    print()
    print("Stopping job queue...")
    await runner.stop(timeout=1)

    print()
    print("=" * 80)
    print("Demo complete!")
    print("=" * 80)
    print()
    print("Key takeaways:")
    print("  1. Behavior Trees handle complex routing logic outside the PN structure")
    print("  2. Jobs are validated and enriched during processing")
    print("  3. Invalid jobs are rejected and can be retried later")
    print("  4. System is easy to extend: add new job types, priorities, or validators")
    print()


if __name__ == "__main__":
    asyncio.run(main())
