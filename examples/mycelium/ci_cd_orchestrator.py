#!/usr/bin/env python3
"""
CI/CD Deployment Orchestrator - Mycelium BT-in-FSM Demo

This example demonstrates a CI/CD deployment pipeline orchestrator that uses
FSM states to manage the overall deployment workflow, while BT subtrees make
intelligent decisions about how to handle failures, retries, and rollbacks.

The scenario:
- FSM manages deployment stages: Build, Test, Deploy, Monitor, Cleanup
- BT decides recovery strategies based on context (environment, error type, retry count)
- Shows real-world automation with intelligent error handling
- Demonstrates how BT-in-FSM integration enables complex decision-making

This is relatable for engineers who work with CI/CD pipelines, deployment automation,
and infrastructure orchestration.
"""

import sys
sys.path.insert(0, "src")

import asyncio
from enum import Enum, auto
from pydantic import BaseModel

from mycorrhizal.mycelium import state, events, on_state as mycelium_on_state, transitions as mycelium_transitions, LabeledTransition, StateConfiguration
from mycorrhizal.rhizomorph.core import bt, Status
from mycorrhizal.common.timebase import MonotonicClock


# ======================================================================================
# BT Subtrees for Decision Making
# ======================================================================================


@bt.tree
def DecideBuildRecovery():
    """
    BT for deciding how to handle build failures.

    Strategy:
    1. If build cache is stale, clear cache and retry
    2. If dependency issue, try fresh dependency install
    3. If retry count < max, retry build
    4. Otherwise, fail and alert humans
    """

    @bt.action
    async def check_cache_stale(bb):
        """Check if build cache is stale."""
        if bb.build_cache_stale:
            print("    [BT] Build cache is stale, clearing...")
            bb.build_cache_stale = False
            return Status.SUCCESS
        return Status.FAILURE

    @bt.action
    async def check_dependency_issue(bb):
        """Check if there's a dependency issue."""
        if bb.dependency_error:
            print("    [BT] Dependency error detected, reinstalling...")
            bb.dependency_error = False
            return Status.SUCCESS
        return Status.FAILURE

    @bt.action
    async def can_retry(bb):
        """Check if we can retry the build."""
        if bb.build_retries < bb.max_build_retries:
            print(f"    [BT] Retrying build (attempt {bb.build_retries + 1}/{bb.max_build_retries})")
            bb.build_retries += 1
            return Status.SUCCESS
        print(f"    [BT] Max retries ({bb.max_build_retries}) exceeded")
        return Status.FAILURE

    @bt.action
    async def alert_humans(bb):
        """Alert humans that build failed."""
        print(f"    [BT] Alerting humans: Build failed after {bb.build_retries} attempts")
        return Status.SUCCESS

    @bt.root
    @bt.selector
    def root():
        """Try recovery strategies in order."""
        yield check_cache_stale
        yield check_dependency_issue
        yield can_retry
        yield alert_humans


@bt.tree
def DecideTestRecovery():
    """
    BT for deciding how to handle test failures.

    Strategy:
    1. If tests failed due to flakiness, retry tests
    2. If tests failed due to environment issues, fix environment
    3. If tests failed with actual bugs, mark for review
    4. If in dev environment, ignore non-critical failures
    """

    @bt.action
    async def check_test_flaky(bb):
        """Check if test failure is due to flakiness."""
        if bb.test_flaky:
            print("    [BT] Flaky test detected, retrying...")
            bb.test_flaky = False
            return Status.SUCCESS
        return Status.FAILURE

    @bt.action
    async def check_environment_issue(bb):
        """Check if test failure is due to environment."""
        if bb.test_env_issue:
            print("    [BT] Environment issue detected, fixing...")
            bb.test_env_issue = False
            return Status.SUCCESS
        return Status.FAILURE

    @bt.action
    async def check_dev_environment(bb):
        """Check if we're in dev environment (more lenient)."""
        if bb.environment == "dev":
            print("    [BT] Dev environment: ignoring non-critical test failures")
            return Status.SUCCESS
        return Status.FAILURE

    @bt.action
    async def mark_for_review(bb):
        """Mark test failure for human review."""
        print("    [BT] Critical test failure: marking for human review")
        bb.requires_review = True
        return Status.SUCCESS

    @bt.root
    @bt.selector
    def root():
        """Try recovery strategies in order."""
        yield check_test_flaky
        yield check_environment_issue
        yield check_dev_environment
        yield mark_for_review


@bt.tree
def DecideDeployRecovery():
    """
    BT for deciding how to handle deployment failures.

    Strategy:
    1. If deployment failed due to health check timeout, retry with longer timeout
    2. If deployment failed due to resource constraints, scale up resources
    3. If deployment failed and we have a previous healthy version, rollback
    4. If all else fails, abort and alert on-call
    """

    @bt.action
    async def check_health_timeout(bb):
        """Check if deployment failed due to health check timeout."""
        if bb.deploy_error == "health_timeout":
            print("    [BT] Health check timeout, retrying with extended timeout...")
            bb.deploy_health_timeout_extended = True
            return Status.SUCCESS
        return Status.FAILURE

    @bt.action
    async def check_resource_constraint(bb):
        """Check if deployment failed due to resource constraints."""
        if bb.deploy_error == "resource_constraint":
            print("    [BT] Resource constraint, scaling up...")
            bb.resources_scaled = True
            return Status.SUCCESS
        return Status.FAILURE

    @bt.action
    async def has_rollback_version(bb):
        """Check if we have a previous healthy version to rollback to."""
        if bb.previous_healthy_version:
            print(f"    [BT] Rolling back to version {bb.previous_healthy_version}")
            return Status.SUCCESS
        print("    [BT] No previous healthy version available")
        return Status.FAILURE

    @bt.action
    async def alert_on_call(bb):
        """Alert on-call engineer."""
        print("    [BT] Alerting on-call: Deployment failed critically")
        bb.alerted_on_call = True
        return Status.SUCCESS

    @bt.root
    @bt.selector
    def root():
        """Try recovery strategies in order."""
        yield check_health_timeout
        yield check_resource_constraint
        yield has_rollback_version
        yield alert_on_call


@bt.tree
def DecideMonitorAction():
    """
    BT for deciding what to do during monitoring.

    Strategy:
    1. If metrics are healthy, mark deployment successful
    2. If metrics degraded but not critical, continue monitoring
    3. If metrics critical and have rollback version, rollback immediately
    4. If metrics critical and no rollback, alert humans
    """

    @bt.action
    async def check_healthy(bb):
        """Check if deployment metrics are healthy."""
        if bb.health_score >= 90:
            print(f"    [BT] Deployment healthy (score: {bb.health_score})")
            bb.deployment_healthy = True
            return Status.SUCCESS
        return Status.FAILURE

    @bt.action
    async def check_acceptable(bb):
        """Check if deployment metrics are acceptable (degraded but usable)."""
        if bb.health_score >= 70:
            print(f"    [BT] Deployment degraded but acceptable (score: {bb.health_score})")
            return Status.SUCCESS
        return Status.FAILURE

    @bt.action
    async def check_critical_with_rollback(bb):
        """Check if critical and we can rollback."""
        if bb.health_score < 50 and bb.previous_healthy_version:
            print(f"    [BT] Critical health (score: {bb.health_score}), rolling back")
            bb.needs_rollback = True
            return Status.SUCCESS
        return Status.FAILURE

    @bt.action
    async def alert_degradation(bb):
        """Alert about degraded deployment."""
        print(f"    [BT] Critical health (score: {bb.health_score}), alerting humans")
        bb.alerted_on_call = True
        return Status.SUCCESS

    @bt.root
    @bt.selector
    def root():
        """Try monitoring actions in order."""
        yield check_healthy
        yield check_acceptable
        yield check_critical_with_rollback
        yield alert_degradation


# ======================================================================================
# FSM State Definitions
# ======================================================================================


@state(bt=DecideBuildRecovery, config=StateConfiguration(can_dwell=True))
def BuildingState():
    """Building the application."""

    @events
    class Events(Enum):
        BUILD_SUCCESS = auto()
        BUILD_FAILED = auto()
        BUILD_FAILED_RETRY = auto()

    @mycelium_on_state
    async def on_state(ctx, bt_result):
        """
        Run build and use BT to decide recovery strategy if it fails.

        BT result:
        - SUCCESS: Recovery strategy found and executed
        - FAILURE: No viable recovery strategy
        """
        bb = ctx.common

        # Simulate build
        print("  [FSM] Building application...")
        await asyncio.sleep(0.1)

        # Check if build succeeds (simulated)
        if bb.build_should_fail:
            print(f"  [FSM] Build failed: {bb.build_error_msg}")
            bb.build_failed = True

            # BT decides recovery strategy
            if bt_result == Status.SUCCESS:
                # BT found a recovery strategy - retry build
                print("  [FSM] BT found recovery strategy, retrying build...")
                bb.build_should_fail = False  # Simulate successful retry
                return Events.BUILD_FAILED_RETRY  # type: ignore[attr-defined]
            else:
                # BT couldn't find recovery - fail
                return Events.BUILD_FAILED  # type: ignore[attr-defined]
        else:
            print("  [FSM] Build successful!")
            return Events.BUILD_SUCCESS  # type: ignore[attr-defined]

    @mycelium_transitions
    def transitions():
        return [
            LabeledTransition(Events.BUILD_SUCCESS, TestingState),  # type: ignore[attr-defined]
            LabeledTransition(Events.BUILD_FAILED_RETRY, BuildingState),  # type: ignore[attr-defined]
            LabeledTransition(Events.BUILD_FAILED, CleanupState),  # type: ignore[attr-defined]
        ]


@state(bt=DecideTestRecovery, config=StateConfiguration(can_dwell=True))
def TestingState():
    """Running tests."""

    @events
    class Events(Enum):
        TESTS_PASSED = auto()
        TESTS_FAILED = auto()
        TESTS_FAILED_RETRY = auto()

    @mycelium_on_state
    async def on_state(ctx, bt_result):
        """
        Run tests and use BT to decide recovery strategy.

        BT result:
        - SUCCESS: Recovery strategy found (retry, environment fix, or ignore in dev)
        - FAILURE: No viable recovery strategy (actual bug)
        """
        bb = ctx.common

        print("  [FSM] Running tests...")
        await asyncio.sleep(0.1)

        # Check if tests pass (simulated)
        if bb.tests_should_fail:
            print(f"  [FSM] Tests failed: {bb.test_error_msg}")
            bb.tests_failed = True

            # BT decides recovery strategy
            if bt_result == Status.SUCCESS:
                # BT found a recovery strategy
                print("  [FSM] BT found recovery strategy, retrying tests...")
                bb.tests_should_fail = False  # Simulate successful retry
                return Events.TESTS_FAILED_RETRY  # type: ignore[attr-defined]
            else:
                # BT couldn't recover - mark for review
                return Events.TESTS_FAILED  # type: ignore[attr-defined]
        else:
            print("  [FSM] All tests passed!")
            return Events.TESTS_PASSED  # type: ignore[attr-defined]

    @mycelium_transitions
    def transitions():
        return [
            LabeledTransition(Events.TESTS_PASSED, DeployingState),  # type: ignore[attr-defined]
            LabeledTransition(Events.TESTS_FAILED_RETRY, TestingState),  # type: ignore[attr-defined]
            LabeledTransition(Events.TESTS_FAILED, CleanupState),  # type: ignore[attr-defined]
        ]


@state(bt=DecideDeployRecovery, config=StateConfiguration(can_dwell=True))
def DeployingState():
    """Deploying to environment."""

    @events
    class Events(Enum):
        DEPLOY_SUCCESS = auto()
        DEPLOY_FAILED = auto()
        DEPLOY_FAILED_RETRY = auto()
        ROLLBACK_INITIATED = auto()

    @mycelium_on_state
    async def on_state(ctx, bt_result):
        """
        Deploy application and use BT to decide recovery strategy.

        BT result:
        - SUCCESS: Recovery strategy found (retry, scale, or rollback)
        - FAILURE: No viable recovery strategy
        """
        bb = ctx.common

        print(f"  [FSM] Deploying to {bb.environment} environment...")
        await asyncio.sleep(0.1)

        # Check if deploy succeeds (simulated)
        if bb.deploy_should_fail:
            print(f"  [FSM] Deployment failed: {bb.deploy_error}")
            bb.deploy_failed = True

            # BT decides recovery strategy
            if bt_result == Status.SUCCESS:
                # Check if BT decided to rollback
                if bb.needs_rollback:
                    print("  [FSM] BT decided to rollback")
                    return Events.ROLLBACK_INITIATED  # type: ignore[attr-defined]
                else:
                    # BT found another recovery strategy
                    print("  [FSM] BT found recovery strategy, retrying deploy...")
                    bb.deploy_should_fail = False  # Simulate successful retry
                    return Events.DEPLOY_FAILED_RETRY  # type: ignore[attr-defined]
            else:
                # BT couldn't recover
                return Events.DEPLOY_FAILED  # type: ignore[attr-defined]
        else:
            print("  [FSM] Deployment successful!")
            bb.previous_healthy_version = bb.new_version
            return Events.DEPLOY_SUCCESS  # type: ignore[attr-defined]

    @mycelium_transitions
    def transitions():
        return [
            LabeledTransition(Events.DEPLOY_SUCCESS, MonitoringState),  # type: ignore[attr-defined]
            LabeledTransition(Events.DEPLOY_FAILED_RETRY, DeployingState),  # type: ignore[attr-defined]
            LabeledTransition(Events.DEPLOY_FAILED, CleanupState),  # type: ignore[attr-defined]
            LabeledTransition(Events.ROLLBACK_INITIATED, RollingBackState),  # type: ignore[attr-defined]
        ]


@state(bt=DecideMonitorAction, config=StateConfiguration(can_dwell=True))
def MonitoringState():
    """Monitoring deployment health."""

    @events
    class Events(Enum):
        HEALTHY = auto()
        DEGRADED = auto()
        CRITICAL = auto()

    @mycelium_on_state
    async def on_state(ctx, bt_result):
        """
        Monitor deployment health and use BT to decide next action.

        BT result:
        - SUCCESS: Action determined (healthy, acceptable, rollback, or alert)
        - FAILURE: Should not happen with this BT
        """
        bb = ctx.common

        print("  [FSM] Monitoring deployment health...")
        await asyncio.sleep(0.1)

        # BT decides what to do based on health score
        if bb.deployment_healthy:
            print("  [FSM] Deployment is healthy!")
            return Events.HEALTHY  # type: ignore[attr-defined]
        elif bb.needs_rollback:
            print("  [FSM] Critical health detected, initiating rollback...")
            return Events.CRITICAL  # type: ignore[attr-defined]
        else:
            print("  [FSM] Deployment degraded but monitoring...")
            return Events.DEGRADED  # type: ignore[attr-defined]

    @mycelium_transitions
    def transitions():
        return [
            LabeledTransition(Events.HEALTHY, CompletedState),  # type: ignore[attr-defined]
            LabeledTransition(Events.DEGRADED, MonitoringState),  # type: ignore[attr-defined] - Continue monitoring
            LabeledTransition(Events.CRITICAL, RollingBackState),  # type: ignore[attr-defined]
        ]


@state()
def RollingBackState():
    """Rolling back to previous version."""

    @events
    class Events(Enum):
        ROLLBACK_COMPLETE = auto()

    @mycelium_on_state
    async def on_state(ctx):
        print(f"  [FSM] Rolling back to version {ctx.common.previous_healthy_version}...")
        await asyncio.sleep(0.1)
        print("  [FSM] Rollback complete!")
        return Events.ROLLBACK_COMPLETE  # type: ignore[attr-defined]

    @mycelium_transitions
    def transitions():
        return [
            LabeledTransition(Events.ROLLBACK_COMPLETE, CleanupState),  # type: ignore[attr-defined]
        ]


@state()
def CleanupState():
    """Cleanup and reporting."""

    @events
    class Events(Enum):
        DONE = auto()

    @mycelium_on_state
    async def on_state(ctx):
        bb = ctx.common
        print("  [FSM] Cleaning up...")
        await asyncio.sleep(0.05)
        print("  [FSM] Deployment Summary:")
        print(f"    - Version: {bb.new_version}")
        print(f"    - Environment: {bb.environment}")
        print(f"    - Build successful: {not bb.build_failed}")
        print(f"    - Tests successful: {not bb.tests_failed}")
        print(f"    - Deploy successful: {not bb.deploy_failed}")
        print(f"    - Requires review: {bb.requires_review}")
        print(f"    - Alerted on-call: {bb.alerted_on_call}")
        return Events.DONE  # type: ignore[attr-defined]

    @mycelium_transitions
    def transitions():
        return [
            LabeledTransition(Events.DONE, CompletedState),  # type: ignore[attr-defined]
        ]


@state()
def CompletedState():
    """Deployment complete."""

    @events
    class Events(Enum):
        FINISHED = auto()

    @mycelium_on_state
    async def on_state(ctx):
        bb = ctx.common
        if bb.deployment_healthy:
            print("  [FSM] Deployment completed successfully!")
        else:
            print("  [FSM] Deployment completed with issues")
        return None  # Terminal state

    @mycelium_transitions
    def transitions():
        return []  # Terminal state


# ======================================================================================
# Blackboard
# ======================================================================================


class DeploymentBlackboard(BaseModel):
    """Shared state for CI/CD deployment."""

    # Deployment info
    new_version: str = "v1.2.3"
    environment: str = "production"
    previous_healthy_version: str | None = None

    # Build state
    build_should_fail: bool = False
    build_error_msg: str = "Compilation error"
    build_cache_stale: bool = False
    dependency_error: bool = False
    build_retries: int = 0
    max_build_retries: int = 3
    build_failed: bool = False

    # Test state
    tests_should_fail: bool = False
    test_error_msg: str = "Test assertion failed"
    test_flaky: bool = False
    test_env_issue: bool = False
    tests_failed: bool = False

    # Deploy state
    deploy_should_fail: bool = False
    deploy_error: str = "health_timeout"
    deploy_health_timeout_extended: bool = False
    resources_scaled: bool = False
    deploy_failed: bool = False

    # Monitor state
    health_score: int = 95  # 0-100
    deployment_healthy: bool = False
    needs_rollback: bool = False

    # Overall state
    requires_review: bool = False
    alerted_on_call: bool = False


# ======================================================================================
# Main Execution
# ======================================================================================


async def run_deployment(scenario_name: str, bb: DeploymentBlackboard):
    """Run a deployment scenario."""

    print("=" * 80)
    print(f"Scenario: {scenario_name}")
    print("=" * 80)
    print()

    # Reset state
    bb.build_retries = 0
    bb.build_failed = False
    bb.tests_failed = False
    bb.deploy_failed = False
    bb.deployment_healthy = False
    bb.requires_review = False
    bb.alerted_on_call = False
    bb.needs_rollback = False

    # Create FSM
    from mycorrhizal.septum.core import StateMachine
    from mycorrhizal.rhizomorph.core import Runner as BTRunner

    fsm = StateMachine(initial_state=BuildingState, common_data=bb)  # type: ignore[arg-type]

    # Create BT runners for all states
    bt_runners = {
        "BuildingState": BTRunner(DecideBuildRecovery, bb=bb, tb=MonotonicClock()),
        "TestingState": BTRunner(DecideTestRecovery, bb=bb, tb=MonotonicClock()),
        "DeployingState": BTRunner(DecideDeployRecovery, bb=bb, tb=MonotonicClock()),
        "MonitoringState": BTRunner(DecideMonitorAction, bb=bb, tb=MonotonicClock()),
    }

    # Inject BT runners into FSM context based on current state
    await fsm.initialize()

    print(f"Starting deployment: {bb.new_version} to {bb.environment}")
    print(f"Initial state: {fsm.current_state.name if fsm.current_state else 'Unknown'}")
    print("-" * 80)
    print()

    # Run FSM until completion
    max_ticks = 20
    for tick in range(max_ticks):
        # Inject appropriate BT runner based on current state
        # State names include module prefix (e.g., "ci_cd_orchestrator.BuildingState")
        # so we extract just the class name
        current_state_name = fsm.current_state.name if fsm.current_state else ""
        state_class_name = current_state_name.split(".")[-1] if "." in current_state_name else current_state_name

        if state_class_name in bt_runners:
            fsm.context._mycelium_bt_runner = bt_runners[state_class_name]  # type: ignore[attr-defined]

        await fsm.tick()

        # Check if we're done
        if fsm.current_state == CompletedState:
            break

        # Small delay for readability
        await asyncio.sleep(0.05)

    print("-" * 80)
    print()
    print(f"Deployment finished: {scenario_name}")
    print(f"Final state: {fsm.current_state.name if fsm.current_state else 'Unknown'}")
    print()


async def main():
    """Run the CI/CD orchestrator demo."""

    print("=" * 80)
    print("Mycelium BT-in-FSM Demo: CI/CD Deployment Orchestrator")
    print("=" * 80)
    print()
    print("This demo shows a CI/CD pipeline using FSM for workflow management")
    print("and BT subtrees for intelligent error recovery decisions:")
    print()
    print("FSM States:")
    print("  - BuildingState: Compiles the application")
    print("  - TestingState: Runs test suite")
    print("  - DeployingState: Deploys to environment")
    print("  - MonitoringState: Monitors deployment health")
    print("  - RollingBackState: Rolls back if needed")
    print("  - CleanupState: Cleanup and reporting")
    print()
    print("BT Decision Trees:")
    print("  - DecideBuildRecovery: Handle build failures (cache, deps, retry, alert)")
    print("  - DecideTestRecovery: Handle test failures (flaky, env, ignore, review)")
    print("  - DecideDeployRecovery: Handle deploy failures (timeout, scale, rollback)")
    print("  - DecideMonitorAction: Monitor health (healthy, degraded, rollback)")
    print()
    print("Key Features:")
    print("  - FSM manages overall deployment workflow")
    print("  - BTs make intelligent recovery decisions based on context")
    print("  - Each failure scenario uses different recovery strategies")
    print("  - Demonstrates real-world automation patterns")
    print()

    # Scenario 1: Successful deployment
    bb = DeploymentBlackboard(
        new_version="v2.0.0",
        environment="production",
        build_should_fail=False,
        tests_should_fail=False,
        deploy_should_fail=False,
        health_score=95,
    )
    await run_deployment("Successful Deployment", bb)

    print()

    # Scenario 2: Build failure with cache recovery
    bb = DeploymentBlackboard(
        new_version="v2.1.0",
        environment="production",
        build_should_fail=True,
        build_cache_stale=True,
        tests_should_fail=False,
        deploy_should_fail=False,
        health_score=92,
    )
    await run_deployment("Build Failure - Cache Recovery", bb)

    print()

    # Scenario 3: Test failure in dev (ignored)
    bb = DeploymentBlackboard(
        new_version="v2.2.0",
        environment="dev",
        build_should_fail=False,
        tests_should_fail=True,
        test_flaky=True,
        deploy_should_fail=False,
        health_score=88,
    )
    await run_deployment("Test Failure - Flaky Test Retry in Dev", bb)

    print()

    # Scenario 4: Deployment failure with rollback
    bb = DeploymentBlackboard(
        new_version="v2.3.0",
        environment="production",
        build_should_fail=False,
        tests_should_fail=False,
        deploy_should_fail=True,
        deploy_error="health_timeout",
        previous_healthy_version="v2.0.0",
        health_score=40,  # Critical after failed deploy
    )
    await run_deployment("Deployment Failure - Rollback", bb)

    print()
    print("=" * 80)
    print("Demo complete!")
    print("=" * 80)
    print()
    print("The CI/CD orchestrator demonstrates:")
    print("  1. FSM provides clear workflow structure (Build -> Test -> Deploy -> Monitor)")
    print("  2. BTs enable context-aware decision making for error recovery")
    print("  3. Each state can have different BT strategies based on its needs")
    print("  4. Complex automation with minimal boilerplate code")
    print()


if __name__ == "__main__":
    asyncio.run(main())
