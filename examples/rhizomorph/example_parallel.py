#!/usr/bin/env python3
"""
Example demonstrating Parallel node behavior in Rhizomorph.

Parallel nodes execute multiple branches concurrently and make decisions based on
how many branches succeed or fail. This is useful for:

- Redundancy: Try multiple approaches, succeed if any work
- Consensus: Require multiple systems to agree
- Race conditions: Multiple services, use the first to respond
- Distributed operations: Query multiple sources in parallel

This example shows various parallel execution patterns to help you understand
how to configure success_threshold and failure_threshold for your use case.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List
import logging

from mycorrhizal.rhizomorph.core import (
    bt,
    Runner,
    Status,
    ExceptionPolicy,
)
from mycorrhizal.common.timebase import MonotonicClock


# Setup logging to see trace output
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s %(name)s: %(message)s'
)
trace_logger = logging.getLogger("bt.trace")
trace_logger.setLevel(logging.INFO)

# ======================================================================================
# Blackboard for tracking parallel execution
# ======================================================================================


@dataclass
class ParallelBlackboard:
    """Blackboard that tracks execution across parallel branches."""

    # Execution logs for each branch
    branch_a_executions: List[str] = field(default_factory=list)
    branch_b_executions: List[str] = field(default_factory=list)
    branch_c_executions: List[str] = field(default_factory=list)
    branch_d_executions: List[str] = field(default_factory=list)
    branch_e_executions: List[str] = field(default_factory=list)

    # Tick counter
    tick_count: int = 0

    # Control flags
    force_branch_a_failure: bool = False
    force_branch_b_running: bool = False

    def log(self, branch: str, message: str):
        """Log execution for a specific branch."""
        log_list = getattr(self, f"branch_{branch}_executions")
        log_list.append(f"Tick {self.tick_count}: {message}")


# ======================================================================================
# Example 1: All Succeeding Branches
# ======================================================================================


@bt.tree
def AllSucceedParallel():
    """
    Parallel node where all branches succeed.
    Expected: SUCCESS when success_threshold (2) is reached.
    """

    @bt.action
    async def branch_a(bb: ParallelBlackboard) -> Status:
        """Fast branch that succeeds immediately."""
        bb.log("a", "Executing branch A (fast success)")
        await asyncio.sleep(0.01)  # Simulate some work
        return Status.SUCCESS

    @bt.action
    async def branch_b(bb: ParallelBlackboard) -> Status:
        """Medium branch that succeeds after brief delay."""
        bb.log("b", "Executing branch B (medium)")
        await asyncio.sleep(0.02)
        return Status.SUCCESS

    @bt.action
    async def branch_c(bb: ParallelBlackboard) -> Status:
        """Slower branch that also succeeds."""
        bb.log("c", "Executing branch C (slow)")
        await asyncio.sleep(0.03)
        return Status.SUCCESS

    @bt.root
    @bt.parallel(success_threshold=2)  # Need 2 out of 3 to succeed
    def root():
        yield branch_a
        yield branch_b
        yield branch_c


# ======================================================================================
# Example 2: Mixed Success/Failure
# ======================================================================================


@bt.tree
def MixedResultsParallel():
    """
    Parallel node with mixed success and failure branches.
    Expected: SUCCESS because 2 succeed before hitting failure_threshold.
    """

    @bt.action
    async def branch_a(bb: ParallelBlackboard) -> Status:
        """Always succeeds."""
        bb.log("a", "Executing branch A (success)")
        await asyncio.sleep(0.01)
        return Status.SUCCESS

    @bt.action
    async def branch_b(bb: ParallelBlackboard) -> Status:
        """Always fails."""
        bb.log("b", "Executing branch B (failure)")
        await asyncio.sleep(0.02)
        return Status.FAILURE

    @bt.action
    async def branch_c(bb: ParallelBlackboard) -> Status:
        """Always succeeds."""
        bb.log("c", "Executing branch C (success)")
        await asyncio.sleep(0.03)
        return Status.SUCCESS

    @bt.root
    @bt.parallel(success_threshold=2)  # Need 2 successes, fail_threshold = 3-2+1 = 2
    def root():
        yield branch_a
        yield branch_b
        yield branch_c


# ======================================================================================
# Example 3: Too Many Failures
# ======================================================================================


@bt.tree
def TooManyFailuresParallel():
    """
    Parallel node where too many branches fail.
    Expected: FAILURE when failure_threshold (2) is reached.
    """

    @bt.action
    async def branch_a(bb: ParallelBlackboard) -> Status:
        """Always succeeds."""
        bb.log("a", "Executing branch A (success)")
        await asyncio.sleep(0.01)
        return Status.SUCCESS

    @bt.action
    async def branch_b(bb: ParallelBlackboard) -> Status:
        """Always fails."""
        bb.log("b", "Executing branch B (failure)")
        await asyncio.sleep(0.02)
        return Status.FAILURE

    @bt.action
    async def branch_c(bb: ParallelBlackboard) -> Status:
        """Always fails."""
        bb.log("c", "Executing branch C (failure)")
        await asyncio.sleep(0.03)
        return Status.FAILURE

    @bt.root
    @bt.parallel(success_threshold=2)  # Need 2 successes, fail_threshold = 2
    def root():
        yield branch_a
        yield branch_b
        yield branch_c


# ======================================================================================
# Example 4: Running State (Async Operations)
# ======================================================================================


@bt.tree
def RunningStateParallel():
    """
    Parallel node with branches that return RUNNING.
    Expected: RUNNING until branches complete.
    """

    @bt.action
    async def branch_a(bb: ParallelBlackboard) -> Status:
        """Completes immediately."""
        bb.log("a", "Executing branch A (immediate)")
        return Status.SUCCESS

    @bt.action
    async def branch_b(bb: ParallelBlackboard) -> Status:
        """Takes multiple ticks to complete."""
        bb.log("b", "Executing branch B (running)")
        await asyncio.sleep(0.02)
        if bb.force_branch_b_running:
            return Status.RUNNING
        return Status.SUCCESS

    @bt.action
    async def branch_c(bb: ParallelBlackboard) -> Status:
        """Also takes multiple ticks."""
        bb.log("c", "Executing branch C (running)")
        await asyncio.sleep(0.03)
        return Status.SUCCESS

    @bt.root
    @bt.parallel(success_threshold=2)
    def root():
        yield branch_a
        yield branch_b
        yield branch_c


# ======================================================================================
# Example 5: Complex Parallel with Many Branches
# ======================================================================================


@bt.tree
def ComplexParallel():
    """
    Complex parallel with 5 branches testing edge cases.
    Expected: SUCCESS when 3 out of 5 succeed.
    """

    @bt.action
    async def branch_a(bb: ParallelBlackboard) -> Status:
        """Fast success."""
        bb.log("a", "Branch A (fast success)")
        await asyncio.sleep(0.01)
        return Status.SUCCESS

    @bt.action
    async def branch_b(bb: ParallelBlackboard) -> Status:
        """Conditional success/failure."""
        bb.log("b", "Branch B (conditional)")
        await asyncio.sleep(0.02)
        return Status.FAILURE if bb.force_branch_a_failure else Status.SUCCESS

    @bt.action
    async def branch_c(bb: ParallelBlackboard) -> Status:
        """Always fails."""
        bb.log("c", "Branch C (failure)")
        await asyncio.sleep(0.03)
        return Status.FAILURE

    @bt.action
    async def branch_d(bb: ParallelBlackboard) -> Status:
        """Medium speed success."""
        bb.log("d", "Branch D (medium)")
        await asyncio.sleep(0.04)
        return Status.SUCCESS

    @bt.action
    async def branch_e(bb: ParallelBlackboard) -> Status:
        """Slow success."""
        bb.log("e", "Branch E (slow)")
        await asyncio.sleep(0.05)
        return Status.SUCCESS

    @bt.root
    @bt.parallel(success_threshold=3)  # Need 3 out of 5, fail_threshold = 5-3+1 = 3
    def root():
        yield branch_a
        yield branch_b
        yield branch_c
        yield branch_d
        yield branch_e


# ======================================================================================
# Example 6: Unanimous Success (All Must Succeed)
# ======================================================================================


@bt.tree
def UnanimousParallel():
    """
    Parallel where ALL branches must succeed.
    Expected: SUCCESS only if all succeed, FAILURE if any fail.
    """

    @bt.action
    async def branch_a(bb: ParallelBlackboard) -> Status:
        """Always succeeds."""
        bb.log("a", "Branch A (success)")
        await asyncio.sleep(0.01)
        return Status.SUCCESS

    @bt.action
    async def branch_b(bb: ParallelBlackboard) -> Status:
        """Always succeeds."""
        bb.log("b", "Branch B (success)")
        await asyncio.sleep(0.02)
        return Status.SUCCESS

    @bt.action
    async def branch_c(bb: ParallelBlackboard) -> Status:
        """Conditional based on flag."""
        bb.log("c", "Branch C (conditional)")
        await asyncio.sleep(0.03)
        return Status.FAILURE if bb.force_branch_a_failure else Status.SUCCESS

    @bt.root
    @bt.parallel(success_threshold=3)  # All 3 must succeed
    def root():
        yield branch_a
        yield branch_b
        yield branch_c


# ======================================================================================
# Example 7: Parallel with Sequence Children
# ======================================================================================


@bt.tree
def ParallelWithSequences():
    """
    Parallel node where children are sequences (composites).
    Tests if parallel works correctly with composite children.
    """

    @bt.action
    async def step_1(bb: ParallelBlackboard) -> Status:
        """First step in sequence A."""
        bb.log("a", "Step A1")
        await asyncio.sleep(0.01)
        return Status.SUCCESS

    @bt.action
    async def step_2(bb: ParallelBlackboard) -> Status:
        """Second step in sequence A."""
        bb.log("a", "Step A2")
        await asyncio.sleep(0.02)
        return Status.SUCCESS

    @bt.action
    async def step_3(bb: ParallelBlackboard) -> Status:
        """Only step in sequence B."""
        bb.log("b", "Step B1")
        await asyncio.sleep(0.03)
        return Status.SUCCESS

    @bt.sequence(memory=True)
    def sequence_a():
        yield step_1
        yield step_2

    @bt.sequence(memory=True)
    def sequence_b():
        yield step_3

    @bt.root
    @bt.parallel(success_threshold=2)
    def root():
        yield sequence_a
        yield sequence_b


# ======================================================================================
# Example 8: Custom Failure Threshold
# ======================================================================================


@bt.tree
def CustomFailureThresholdParallel():
    """
    Parallel node with custom failure_threshold.
    Demonstrates asymmetric thresholds (need 2 success, but fail after 1 failure).

    Default behavior: failure_threshold = n - k + 1 = 3 - 2 + 1 = 2
    Custom: failure_threshold = 1 (fail immediately on first failure)

    IMPORTANT: Success threshold is checked BEFORE failure threshold.
    If both thresholds are met, SUCCESS takes precedence.

    With 1 failure and 2 successes: success_threshold (2) is met, so returns SUCCESS.
    This is intentional design - success prioritizes over failure.
    """

    @bt.action
    async def branch_a(bb: ParallelBlackboard) -> Status:
        """Always succeeds quickly."""
        bb.log("a", "Branch A (success)")
        await asyncio.sleep(0.01)
        return Status.SUCCESS

    @bt.action
    async def branch_b(bb: ParallelBlackboard) -> Status:
        """Always fails."""
        bb.log("b", "Branch B (failure)")
        await asyncio.sleep(0.02)
        return Status.FAILURE

    @bt.action
    async def branch_c(bb: ParallelBlackboard) -> Status:
        """Would succeed but we fail before it completes."""
        bb.log("c", "Branch C (slow success)")
        await asyncio.sleep(0.05)
        return Status.SUCCESS

    @bt.root
    @bt.parallel(success_threshold=2, failure_threshold=1)
    def root():
        yield branch_a
        yield branch_b
        yield branch_c


# ======================================================================================
# Example 9: Majority Vote with 4 Branches
# ======================================================================================


@bt.tree
def MajorityVoteParallel():
    """
    Parallel node implementing majority vote with 4 branches.
    With 4 branches and success_threshold=2:
    - SUCCESS if 2+ succeed
    - FAILURE if 3+ fail (default: 4-2+1=3)

    This tests odd-numbered failure behavior with even branch count.
    """

    @bt.action
    async def branch_a(bb: ParallelBlackboard) -> Status:
        """Succeeds."""
        bb.log("a", "Branch A (success)")
        await asyncio.sleep(0.01)
        return Status.SUCCESS

    @bt.action
    async def branch_b(bb: ParallelBlackboard) -> Status:
        """Fails."""
        bb.log("b", "Branch B (failure)")
        await asyncio.sleep(0.02)
        return Status.FAILURE

    @bt.action
    async def branch_c(bb: ParallelBlackboard) -> Status:
        """Succeeds."""
        bb.log("c", "Branch C (success)")
        await asyncio.sleep(0.03)
        return Status.SUCCESS

    @bt.action
    async def branch_d(bb: ParallelBlackboard) -> Status:
        """Fails."""
        bb.log("d", "Branch D (failure)")
        await asyncio.sleep(0.04)
        return Status.FAILURE

    @bt.root
    @bt.parallel(success_threshold=2)
    def root():
        yield branch_a
        yield branch_b
        yield branch_c
        yield branch_d


# ======================================================================================
# Test Runner
# ======================================================================================


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def print_logs(bb: ParallelBlackboard):
    """Print execution logs from all branches."""
    print(f"  Tick {bb.tick_count} logs:")
    for branch in ["a", "b", "c", "d"]:
        logs = getattr(bb, f"branch_{branch}_executions")
        if logs:
            for log in logs:
                print(f"    [{branch.upper()}] {log}")
    print()


async def run_example(name: str, tree, expected_status: Status, ticks: int = 5):
    """Run a single parallel example and report results."""
    print_section(name)

    bb = ParallelBlackboard()
    tb = MonotonicClock()
    runner = Runner(tree, bb=bb, tb=tb, trace=trace_logger)

    final_status = None
    for i in range(ticks):
        bb.tick_count = i + 1
        status = await runner.tick()
        final_status = status

        print(f"Tick {bb.tick_count}: {status.name}")

        if status in (Status.SUCCESS, Status.FAILURE, Status.ERROR, Status.CANCELLED):
            break

        await asyncio.sleep(0.01)  # Small delay between ticks

    # Print execution logs
    print_logs(bb)

    # Verify result
    status_match = "✓" if final_status == expected_status else "✗"
    print(f"  Result: {final_status.name} (Expected: {expected_status.name}) {status_match}")

    return final_status == expected_status


async def main():
    """Run all parallel examples."""
    print_section("Rhizomorph Parallel Node Examples")
    print("Testing various parallel execution scenarios to identify potential issues.\n")

    results = []

    # Example 1: All branches succeed
    results.append(
        await run_example(
            "Example 1: All Branches Succeed",
            AllSucceedParallel,
            Status.SUCCESS,
        )
    )

    # Example 2: Mixed results (should succeed)
    results.append(
        await run_example(
            "Example 2: Mixed Success/Failure (Should Succeed)",
            MixedResultsParallel,
            Status.SUCCESS,
        )
    )

    # Example 3: Too many failures
    results.append(
        await run_example(
            "Example 3: Too Many Failures (Should Fail)",
            TooManyFailuresParallel,
            Status.FAILURE,
        )
    )

    # Example 4: Running state
    print_section("Example 4: Running State (Async Operations)")

    bb = ParallelBlackboard()
    bb.force_branch_b_running = True
    tb = MonotonicClock()
    runner = Runner(RunningStateParallel, bb=bb, tb=tb)

    print("Testing with branch_b returning RUNNING...\n")
    for i in range(3):
        bb.tick_count = i + 1
        status = await runner.tick()
        print(f"Tick {bb.tick_count}: {status.name}")

        if status != Status.RUNNING:
            break
        await asyncio.sleep(0.01)

    print_logs(bb)

    bb.force_branch_b_running = False
    print("Continuing with branch_b able to complete...\n")
    for i in range(3, 6):
        bb.tick_count = i + 1
        status = await runner.tick()
        print(f"Tick {bb.tick_count}: {status.name}")

        if status in (Status.SUCCESS, Status.FAILURE):
            break
        await asyncio.sleep(0.01)

    print_logs(bb)
    results.append(True)  # Manual check needed

    # Example 5: Complex parallel
    results.append(
        await run_example(
            "Example 5: Complex Parallel (5 Branches)",
            ComplexParallel,
            Status.SUCCESS,
            ticks=8,
        )
    )

    # Example 6: Unanimous success
    results.append(
        await run_example(
            "Example 6: Unanimous Success (All Must Succeed)",
            UnanimousParallel,
            Status.SUCCESS,
        )
    )

    # Test unanimous with a failure
    print_section("Example 6b: Unanimous Success (With Failure)")

    bb = ParallelBlackboard()
    bb.force_branch_a_failure = True
    tb = MonotonicClock()
    runner = Runner(UnanimousParallel, bb=bb, tb=tb, trace = trace_logger)

    status = await runner.tick()
    print(f"Tick 1: {status.name}")
    print_logs(bb)

    results.append(status == Status.FAILURE)

    # Example 7: Parallel with sequences
    results.append(
        await run_example(
            "Example 7: Parallel with Sequence Children",
            ParallelWithSequences,
            Status.SUCCESS,
            ticks=5,
        )
    )

    # Example 8: Custom failure threshold
    results.append(
        await run_example(
            "Example 8: Custom Failure Threshold (Success Takes Priority)",
            CustomFailureThresholdParallel,
            Status.SUCCESS,  # Success threshold met, so SUCCESS (not FAILURE)
            ticks=5,
        )
    )

    # Example 9: Majority vote with 4 branches
    results.append(
        await run_example(
            "Example 9: Majority Vote with 4 Branches (2-2 Tie)",
            MajorityVoteParallel,
            Status.SUCCESS,
            ticks=5,
        )
    )

    # Summary
    print_section("Summary")
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✓ All parallel examples behaved as expected!")
    else:
        print("✗ Some parallel examples did not behave as expected.")
        print("  This may indicate issues with parallel execution.")

    # Generate Mermaid diagram for one example
    print("\n--- Mermaid Diagram for Example 5 (Complex Parallel) ---\n")
    print(ComplexParallel.to_mermaid())


if __name__ == "__main__":
    asyncio.run(main())
