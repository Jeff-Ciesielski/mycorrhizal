#!/usr/bin/env python3
"""
Regression test for bt.sequence() and bt.selector() direct call form.

This test verifies that bt.sequence() and bt.selector() can be called
with node arguments directly, not just as decorators.

Issue: bt.sequence(action1, action2) should work but currently fails.
"""

import asyncio
import pytest
from dataclasses import dataclass, field
from typing import List

from mycorrhizal.rhizomorph.core import (
    bt,
    Status,
    Runner,
    NodeSpec,
)
from mycorrhizal.common.timebase import Timebase


@dataclass
class MockBlackboard:
    """Simple blackboard for testing."""
    log: List[str] = field(default_factory=list)
    value: int = 0


@dataclass
class MockTimebase(Timebase):
    """Controllable timebase for testing."""
    _time: float = 0.0

    def now(self) -> float:
        return self._time

    def advance(self, delta: float = 0.1) -> None:
        self._time += delta

    async def sleep(self, duration: float) -> None:
        self._time += duration


class TestDirectCallComposites:
    """Tests for direct call form of bt.sequence() and bt.selector()."""

    @pytest.mark.asyncio
    async def test_sequence_direct_call_with_actions(self):
        """Test bt.sequence(action1, action2) creates a sequence node."""
        @bt.tree
        def Tree():
            @bt.action
            def action1(bb: MockBlackboard) -> Status:
                bb.log.append("action1")
                return Status.SUCCESS

            @bt.action
            def action2(bb: MockBlackboard) -> Status:
                bb.log.append("action2")
                return Status.SUCCESS

            @bt.condition
            def always_true(bb: MockBlackboard) -> bool:
                return True

            # Direct call form: bt.sequence(action1, action2)
            # This should work like a constructor that creates a sequence
            @bt.root
            @bt.selector
            def root():
                # Using gate with a direct sequence call
                yield bt.gate(always_true)(
                    bt.sequence(action1, action2)
                )

        bb = MockBlackboard()
        tb = MockTimebase()
        runner = Runner(Tree, bb, tb=tb)
        status = await runner.tick_until_complete()

        assert status == Status.SUCCESS
        assert bb.log == ["action1", "action2"]

    @pytest.mark.asyncio
    async def test_selector_direct_call_with_actions(self):
        """Test bt.selector(action1, action2) creates a selector node."""
        @bt.tree
        def Tree():
            @bt.action
            def action1(bb: MockBlackboard) -> Status:
                bb.log.append("action1")
                return Status.FAILURE

            @bt.action
            def action2(bb: MockBlackboard) -> Status:
                bb.log.append("action2")
                return Status.SUCCESS

            @bt.root
            @bt.selector
            def root():
                # Using selector in direct call form
                yield bt.selector(action1, action2)

        bb = MockBlackboard()
        tb = MockTimebase()
        runner = Runner(Tree, bb, tb=tb)
        status = await runner.tick_until_complete()

        assert status == Status.SUCCESS
        # First action fails, second succeeds
        assert bb.log == ["action1", "action2"]

    @pytest.mark.asyncio
    async def test_sequence_direct_call_in_gate(self):
        """Test the original failing case: bt.gate(condition)(bt.sequence(actions))."""
        @bt.tree
        def Tree():
            @bt.action
            def send_resume_command(bb: MockBlackboard) -> Status:
                bb.log.append("resume_command")
                return Status.SUCCESS

            @bt.action
            def send_reply(bb: MockBlackboard) -> Status:
                bb.log.append("reply")
                return Status.SUCCESS

            @bt.action
            def send_error_reply(bb: MockBlackboard) -> Status:
                bb.log.append("error_reply")
                return Status.SUCCESS

            @bt.condition
            def validate_pod(bb: MockBlackboard) -> bool:
                bb.value = 1
                return True  # Pod exists

            @bt.root
            @bt.selector
            def root():
                """Resume previous profile mode with error handling."""
                # Success path: pod exists
                yield bt.gate(validate_pod)(
                    bt.sequence(send_resume_command, send_reply)
                )
                # Error path: pod not found (won't run since gate succeeds)
                yield send_error_reply

        bb = MockBlackboard()
        tb = MockTimebase()
        runner = Runner(Tree, bb, tb=tb)
        status = await runner.tick_until_complete()

        # Gate should succeed, sequence should run
        assert status == Status.SUCCESS
        assert bb.log == ["resume_command", "reply"]
        # validate_pod was called in the gate
        assert bb.value == 1

    @pytest.mark.asyncio
    async def test_selector_direct_call_in_selector(self):
        """Test bt.selector with direct call children."""
        @bt.tree
        def Tree():
            @bt.action
            def option1(bb: MockBlackboard) -> Status:
                bb.log.append("option1")
                return Status.FAILURE

            @bt.action
            def option2(bb: MockBlackboard) -> Status:
                bb.log.append("option2")
                return Status.FAILURE

            @bt.action
            def option3(bb: MockBlackboard) -> Status:
                bb.log.append("option3")
                return Status.SUCCESS

            @bt.root
            @bt.selector
            def root():
                # Nested direct call selectors
                yield bt.selector(
                    bt.selector(option1, option2),
                    option3
                )

        bb = MockBlackboard()
        tb = MockTimebase()
        runner = Runner(Tree, bb, tb=tb)
        status = await runner.tick_until_complete()

        assert status == Status.SUCCESS
        # option1 fails, option2 fails, option3 succeeds
        assert bb.log == ["option1", "option2", "option3"]

    @pytest.mark.asyncio
    async def test_sequence_with_decorator_form_still_works(self):
        """Ensure decorator form @bt.sequence still works after changes."""
        @bt.tree
        def Tree():
            @bt.action
            def action1(bb: MockBlackboard) -> Status:
                bb.log.append("action1")
                return Status.SUCCESS

            @bt.action
            def action2(bb: MockBlackboard) -> Status:
                bb.log.append("action2")
                return Status.SUCCESS

            # Traditional decorator form
            @bt.root
            @bt.sequence
            def root():
                yield action1
                yield action2

        bb = MockBlackboard()
        tb = MockTimebase()
        runner = Runner(Tree, bb, tb=tb)
        status = await runner.tick_until_complete()

        assert status == Status.SUCCESS
        assert bb.log == ["action1", "action2"]
