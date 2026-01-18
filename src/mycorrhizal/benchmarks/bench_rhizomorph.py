"""
Throughput benchmarks for Rhizomorph (Behavior tree DSL)

Measures tree ticks/executions per second across various scenarios:
- Simple action nodes
- Sequence composites
- Selector composites
- Mixed complex trees
"""

import asyncio
import pytest

from mycorrhizal.rhizomorph.core import bt, Runner, Status
from mycorrhizal.common.timebase import CycleClock


# ============================================================================
# Fixtures and Helpers
# ============================================================================

class BenchmarkBlackboard:
    """Blackboard for benchmarks"""
    def __init__(self):
        self.action_count = 0
        self.condition_count = 0
        self.value = 0


# ============================================================================
# Benchmark Trees
# ============================================================================

@bt.tree
def SimpleActionTree():
    """Tree with a single action node

    Measures baseline tick throughput.
    """

    @bt.action
    async def do_action(bb: BenchmarkBlackboard) -> Status:
        bb.action_count += 1
        return Status.SUCCESS

    @bt.root
    @bt.sequence
    def root():
        yield do_action


@bt.tree
def SequenceTree():
    """Tree with sequence of actions

    Measures tick throughput through sequential composites.
    """

    @bt.action
    async def action1(bb: BenchmarkBlackboard) -> Status:
        bb.action_count += 1
        return Status.SUCCESS

    @bt.action
    async def action2(bb: BenchmarkBlackboard) -> Status:
        bb.action_count += 1
        return Status.SUCCESS

    @bt.action
    async def action3(bb: BenchmarkBlackboard) -> Status:
        bb.action_count += 1
        return Status.SUCCESS

    @bt.root
    @bt.sequence
    def root():
        yield action1
        yield action2
        yield action3


@bt.tree
def SelectorTree():
    """Tree with selector (fallback) structure

    Measures tick throughput through selector composites.
    """

    @bt.condition
    def condition1(bb: BenchmarkBlackboard) -> bool:
        bb.condition_count += 1
        return False

    @bt.condition
    def condition2(bb: BenchmarkBlackboard) -> bool:
        bb.condition_count += 1
        return False

    @bt.action
    async def fallback(bb: BenchmarkBlackboard) -> Status:
        bb.action_count += 1
        return Status.SUCCESS

    @bt.root
    @bt.selector
    def root():
        yield condition1
        yield condition2
        yield fallback


@bt.tree
def MixedTree():
    """Tree with mixed sequence/selector structure

    Measures tick throughput through realistic tree.
    """

    @bt.condition
    def check_ready(bb: BenchmarkBlackboard) -> bool:
        bb.condition_count += 1
        return True

    @bt.action
    async def prepare(bb: BenchmarkBlackboard) -> Status:
        bb.action_count += 1
        return Status.SUCCESS

    @bt.action
    async def execute(bb: BenchmarkBlackboard) -> Status:
        bb.action_count += 1
        return Status.SUCCESS

    @bt.action
    async def cleanup(bb: BenchmarkBlackboard) -> Status:
        bb.action_count += 1
        return Status.SUCCESS

    @bt.action
    async def fallback(bb: BenchmarkBlackboard) -> Status:
        bb.action_count += 1
        return Status.SUCCESS

    @bt.sequence
    def main_sequence():
        yield check_ready
        yield prepare
        yield execute
        yield cleanup

    @bt.root
    @bt.selector
    def root():
        yield main_sequence
        yield fallback


@bt.tree
def ParallelTree():
    """Tree with parallel composite

    Measures tick throughput with parallel execution.
    """

    @bt.action
    async def task1(bb: BenchmarkBlackboard) -> Status:
        bb.action_count += 1
        return Status.SUCCESS

    @bt.action
    async def task2(bb: BenchmarkBlackboard) -> Status:
        bb.action_count += 1
        return Status.SUCCESS

    @bt.action
    async def task3(bb: BenchmarkBlackboard) -> Status:
        bb.action_count += 1
        return Status.SUCCESS

    @bt.root
    @bt.parallel(success_threshold=2)
    def root():
        yield task1
        yield task2
        yield task3


@bt.tree
def ConditionHeavyTree():
    """Tree with many condition checks

    Measures tick throughput through condition-heavy trees.
    """

    @bt.condition
    def check1(bb: BenchmarkBlackboard) -> bool:
        bb.condition_count += 1
        return True

    @bt.condition
    def check2(bb: BenchmarkBlackboard) -> bool:
        bb.condition_count += 1
        return True

    @bt.condition
    def check3(bb: BenchmarkBlackboard) -> bool:
        bb.condition_count += 1
        return True

    @bt.condition
    def check4(bb: BenchmarkBlackboard) -> bool:
        bb.condition_count += 1
        return True

    @bt.action
    async def action(bb: BenchmarkBlackboard) -> Status:
        bb.action_count += 1
        return Status.SUCCESS

    @bt.root
    @bt.sequence
    def root():
        yield check1
        yield check2
        yield check3
        yield check4
        yield action


# ============================================================================
# Benchmarks
# ============================================================================

@pytest.mark.rhizomorph
@pytest.mark.benchmark(group="rhizomorph-simple")
def test_simple_action_throughput(benchmark):
    """Benchmark simple action tree

    Measures: tree ticks per second through single action
    Scenario: 100 ticks of simple action tree
    Expected: Baseline tick throughput metric
    """
    NUM_TICKS = 100

    def run_tree():
        async def _run():
            bb = BenchmarkBlackboard()
            runner = Runner(SimpleActionTree, bb)

            for _ in range(NUM_TICKS):
                await runner.tick()

            return bb.action_count

        return asyncio.run(_run())

    result = benchmark(run_tree)
    assert result == NUM_TICKS


@pytest.mark.rhizomorph
@pytest.mark.benchmark(group="rhizomorph-composites")
def test_sequence_throughput(benchmark):
    """Benchmark sequence composite

    Measures: tree ticks per second through 3-action sequence
    Scenario: 100 ticks of sequence tree
    Expected: Each tick executes 3 actions (count = 300)
    """
    NUM_TICKS = 100

    def run_tree():
        async def _run():
            bb = BenchmarkBlackboard()
            runner = Runner(SequenceTree, bb)

            for _ in range(NUM_TICKS):
                await runner.tick()

            return bb.action_count

        return asyncio.run(_run())

    result = benchmark(run_tree)
    assert result == NUM_TICKS * 3


@pytest.mark.rhizomorph
@pytest.mark.benchmark(group="rhizomorph-composites")
def test_selector_throughput(benchmark):
    """Benchmark selector composite

    Measures: tree ticks per second through selector with fallback
    Scenario: 100 ticks of selector tree (2 failing conditions)
    Expected: Each tick checks 2 conditions + 1 action
    """
    NUM_TICKS = 100

    def run_tree():
        async def _run():
            bb = BenchmarkBlackboard()
            runner = Runner(SelectorTree, bb)

            for _ in range(NUM_TICKS):
                await runner.tick()

            return bb.condition_count + bb.action_count

        return asyncio.run(_run())

    result = benchmark(run_tree)
    assert result == NUM_TICKS * 3  # 2 conditions + 1 action per tick


@pytest.mark.rhizomorph
@pytest.mark.benchmark(group="rhizomorph-complex")
def test_mixed_tree_throughput(benchmark):
    """Benchmark mixed sequence/selector tree

    Measures: tree ticks per second through realistic tree
    Scenario: 100 ticks of mixed structure tree
    Expected: Each tick executes sequence of 4 nodes
    """
    NUM_TICKS = 100

    def run_tree():
        async def _run():
            bb = BenchmarkBlackboard()
            runner = Runner(MixedTree, bb)

            for _ in range(NUM_TICKS):
                await runner.tick()

            return bb.action_count + bb.condition_count

        return asyncio.run(_run())

    result = benchmark(run_tree)
    assert result == NUM_TICKS * 4  # 4 nodes per tick


@pytest.mark.rhizomorph
@pytest.mark.benchmark(group="rhizomorph-composites")
def test_parallel_throughput(benchmark):
    """Benchmark parallel composite

    Measures: tree ticks per second through parallel execution
    Scenario: 100 ticks of parallel tree with 3 actions
    Expected: Each tick executes 3 actions in parallel
    """
    NUM_TICKS = 100

    def run_tree():
        async def _run():
            bb = BenchmarkBlackboard()
            runner = Runner(ParallelTree, bb)

            for _ in range(NUM_TICKS):
                await runner.tick()

            return bb.action_count

        return asyncio.run(_run())

    result = benchmark(run_tree)
    assert result == NUM_TICKS * 3


@pytest.mark.rhizomorph
@pytest.mark.benchmark(group="rhizomorph-conditions")
def test_condition_heavy_throughput(benchmark):
    """Benchmark condition-heavy tree

    Measures: tree ticks per second through condition checks
    Scenario: 100 ticks of tree with 4 conditions + 1 action
    Expected: Each tick checks 4 conditions then executes action
    """
    NUM_TICKS = 100

    def run_tree():
        async def _run():
            bb = BenchmarkBlackboard()
            runner = Runner(ConditionHeavyTree, bb)

            for _ in range(NUM_TICKS):
                await runner.tick()

            return bb.condition_count + bb.action_count

        return asyncio.run(_run())

    result = benchmark(run_tree)
    assert result == NUM_TICKS * 5  # 4 conditions + 1 action


@pytest.mark.rhizomorph
@pytest.mark.benchmark(group="rhizomorph-scalability")
@pytest.mark.parametrize("num_ticks", [10, 50, 100, 500])
def test_scalability_different_ticks(benchmark, num_ticks):
    """Benchmark scalability with different tick counts

    Measures: how throughput scales with tick count
    Scenario: Varying numbers of ticks through simple tree
    Expected: Linear scaling (2x ticks = 2x time)
    """
    def run_tree():
        async def _run():
            bb = BenchmarkBlackboard()
            runner = Runner(SimpleActionTree, bb)

            for _ in range(num_ticks):
                await runner.tick()

            return bb.action_count

        return asyncio.run(_run())

    result = benchmark(run_tree)
    assert result == num_ticks
