#!/usr/bin/env python3
"""
Example demonstrating bt.try_catch error handling pattern.

The try_catch decorator provides a clean way to handle fallback behavior:
- Try block executes first
- If try succeeds, catch is skipped
- If try fails, catch executes as fallback
- Returns SUCCESS if either block succeeds

IMPORTANT: Try and catch blocks should be pre-defined with explicit memory settings.
"""

import asyncio
from dataclasses import dataclass, field
from typing import List

from mycorrhizal.rhizomorph.core import bt, Runner, Status


@dataclass
class Blackboard:
    """Shared state for behavior tree."""
    attempts: int = 0
    log: List[str] = field(default_factory=list)
    use_fallback: bool = False


async def main():
    """Run try_catch examples."""

    print("=" * 60)
    print("Example 1: Try block succeeds")
    print("=" * 60)

    @bt.tree
    def TrySucceedsTree():
        @bt.action
        def try_primary_method(bb: Blackboard):
            bb.log.append("try: primary method")
            return Status.SUCCESS

        @bt.action
        def catch_fallback_method(bb: Blackboard):
            bb.log.append("catch: fallback method")
            return Status.SUCCESS

        @bt.root
        @bt.sequence()
        def root():
            yield bt.try_catch(try_primary_method)(catch_fallback_method)

    bb1 = Blackboard()
    runner1 = Runner(TrySucceedsTree, bb1)
    result1 = await runner1.tick_until_complete()

    print(f"Result: {result1}")
    print(f"Log: {bb1.log}")
    print(f"Notice: Catch block was NOT executed (try succeeded)\n")

    print("=" * 60)
    print("Example 2: Try block fails, catch succeeds")
    print("=" * 60)

    @bt.tree
    def TryFailsTree():
        @bt.action
        def try_primary_method(bb: Blackboard):
            bb.log.append("try: primary method")
            return Status.FAILURE  # Primary method fails

        @bt.action
        def catch_fallback_method(bb: Blackboard):
            bb.log.append("catch: fallback method")
            return Status.SUCCESS  # Fallback succeeds

        @bt.root
        @bt.sequence()
        def root():
            yield bt.try_catch(try_primary_method)(catch_fallback_method)

    bb2 = Blackboard()
    runner2 = Runner(TryFailsTree, bb2)
    result2 = await runner2.tick_until_complete()

    print(f"Result: {result2}")
    print(f"Log: {bb2.log}")
    print(f"Notice: Both try and catch executed (fallback worked)\n")

    print("=" * 60)
    print("Example 3: Complex try block (sequence with memory)")
    print("=" * 60)

    @bt.tree
    def ComplexTryTree():
        @bt.action
        def initialize(bb: Blackboard):
            bb.log.append("initialize")
            return Status.SUCCESS

        @bt.action
        def process(bb: Blackboard):
            bb.log.append("process")
            if bb.use_fallback:
                bb.log.append("process failed")
                return Status.FAILURE
            return Status.SUCCESS

        @bt.action
        def finalize(bb: Blackboard):
            bb.log.append("finalize")
            return Status.SUCCESS

        @bt.action
        def fallback_handler(bb: Blackboard):
            bb.log.append("fallback: using alternative method")
            return Status.SUCCESS

        # Pre-defined try block with explicit memory setting
        @bt.sequence(memory=True)
        def try_block():
            yield initialize
            yield process
            yield finalize

        # Pre-defined catch block with explicit memory setting
        @bt.sequence(memory=True)
        def catch_block():
            yield fallback_handler

        @bt.root
        @bt.sequence()
        def root():
            # Try a sequence of operations
            yield bt.try_catch(try_block)(catch_block)

    bb3a = Blackboard(use_fallback=False)
    runner3a = Runner(ComplexTryTree, bb3a)
    result3a = await runner3a.tick_until_complete()

    print(f"Result (normal path): {result3a}")
    print(f"Log: {bb3a.log}")
    print(f"All operations succeeded, fallback not used\n")

    bb3b = Blackboard(use_fallback=True)
    runner3b = Runner(ComplexTryTree, bb3b)
    result3b = await runner3b.tick_until_complete()

    print(f"Result (with failure): {result3b}")
    print(f"Log: {bb3b.log}")
    print(f"Process failed, fallback was activated\n")

    print("=" * 60)
    print("Example 4: Mermaid diagram")
    print("=" * 60)

    @bt.tree
    def DiagramTree():
        @bt.action
        def try_action(bb):
            return Status.SUCCESS

        @bt.action
        def catch_action(bb):
            return Status.SUCCESS

        @bt.root
        @bt.sequence()
        def root():
            yield bt.try_catch(try_action)(catch_action)

    print(DiagramTree.to_mermaid())
    print()


if __name__ == "__main__":
    asyncio.run(main())
