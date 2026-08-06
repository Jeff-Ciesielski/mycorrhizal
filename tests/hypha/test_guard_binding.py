#!/usr/bin/env python3
"""
Tests for guard bindings in the Hypha runtime.

A guard receives the candidate bindings of its transition as token values and
yields the bindings it accepts. The transition consumes exactly the tokens of
the first accepted binding. Transitions that fire in the same cycle never
share a token. The runner evaluates guards again only when the marking
changes or when a transition delay ends.

Run with: pytest tests/hypha/test_guard_binding.py -v
"""

import asyncio
from collections import Counter

import pytest

from mycorrhizal.common.timebase import CycleClock
from mycorrhizal.hypha.core import Runner, pn


def place(runner, net_name, name):
    return runner.runtime.places[(net_name, name)]


def tokens(runner, net_name, name):
    return list(place(runner, net_name, name).tokens)


class TestGuardSelectsBinding:

    async def test_guard_selects_token_that_is_not_first(self):
        @pn.net
        def PickNet(b):
            src = b.place("src")
            out = b.place("out")

            def wants_b(combos, bb, tb):
                for c in combos:
                    if c[0][0] == "b":
                        yield c

            @b.transition(guard=b.guard(wants_b))
            async def take_b(consumed, bb, tb):
                yield {out: consumed[0]}

            b.arc(src, take_b).arc(out)

        runner = Runner(PickNet, None)
        await runner.start(None)
        place(runner, "PickNet", "src").add_token("a")
        place(runner, "PickNet", "src").add_token("b")
        await asyncio.sleep(0.1)

        assert tokens(runner, "PickNet", "src") == ["a"]
        assert tokens(runner, "PickNet", "out") == ["b"]
        await runner.stop()

    async def test_guard_receives_token_values(self):
        seen = []

        @pn.net
        def SeeNet(b):
            src = b.place("src")
            out = b.place("out")

            def record(combos, bb, tb):
                for c in combos:
                    seen.append(c[0][0])
                    yield c

            @b.transition(guard=b.guard(record))
            async def move(consumed, bb, tb):
                yield {out: consumed[0]}

            b.arc(src, move).arc(out)

        runner = Runner(SeeNet, None)
        await runner.start(None)
        place(runner, "SeeNet", "src").add_token({"id": "x"})
        await asyncio.sleep(0.1)

        assert seen == [{"id": "x"}]
        await runner.stop()

    async def test_weighted_arc_offers_every_combination(self):
        @pn.net
        def PairNet(b):
            src = b.place("src")
            out = b.place("out")

            def sums_to_seven(combos, bb, tb):
                for c in combos:
                    if sum(c[0]) == 7:
                        yield c

            @b.transition(guard=b.guard(sums_to_seven))
            async def pair(consumed, bb, tb):
                yield {out: tuple(sorted(consumed))}

            b.arc(src, pair, weight=2).arc(out)

        runner = Runner(PairNet, None)
        await runner.start(None)
        for value in (1, 2, 3, 4):
            place(runner, "PairNet", "src").add_token(value)
        await asyncio.sleep(0.1)

        assert tokens(runner, "PairNet", "out") == [(3, 4)]
        assert sorted(tokens(runner, "PairNet", "src")) == [1, 2]
        await runner.stop()

    async def test_binding_spans_input_places(self):
        @pn.net
        def JoinNet(b):
            left = b.place("left")
            right = b.place("right")
            out = b.place("out")

            def sums_to_22(combos, bb, tb):
                for c in combos:
                    if c[0][0] + c[1][0] == 22:
                        yield c

            @b.transition(guard=b.guard(sums_to_22))
            async def join(consumed, bb, tb):
                yield {out: tuple(consumed)}

            b.arc(left, join)
            b.arc(right, join)
            b.arc(join, out)

        runner = Runner(JoinNet, None)
        await runner.start(None)
        for value in (1, 2):
            place(runner, "JoinNet", "left").add_token(value)
        for value in (10, 20):
            place(runner, "JoinNet", "right").add_token(value)
        await asyncio.sleep(0.1)

        assert tokens(runner, "JoinNet", "out") == [(2, 20)]
        assert tokens(runner, "JoinNet", "left") == [1]
        assert tokens(runner, "JoinNet", "right") == [10]
        await runner.stop()

    async def test_async_guard_selects_binding(self):
        @pn.net
        def AsyncPickNet(b):
            src = b.place("src")
            out = b.place("out")

            async def wants_b(combos, bb, tb):
                for c in combos:
                    if c[0][0] == "b":
                        yield c

            @b.transition(guard=b.guard(wants_b))
            async def take_b(consumed, bb, tb):
                yield {out: consumed[0]}

            b.arc(src, take_b).arc(out)

        runner = Runner(AsyncPickNet, None)
        await runner.start(None)
        place(runner, "AsyncPickNet", "src").add_token("a")
        place(runner, "AsyncPickNet", "src").add_token("b")
        await asyncio.sleep(0.1)

        assert tokens(runner, "AsyncPickNet", "src") == ["a"]
        assert tokens(runner, "AsyncPickNet", "out") == ["b"]
        await runner.stop()

    def test_sync_mode_guard_selects_token_that_is_not_first(self):
        @pn.net
        def SyncPickNet(b):
            src = b.place("src")
            out = b.place("out")

            def wants_b(combos, bb, tb):
                for c in combos:
                    if c[0][0] == "b":
                        yield c

            @b.transition(guard=b.guard(wants_b))
            def take_b(consumed, bb, tb):
                yield {out: consumed[0]}

            b.arc(src, take_b).arc(out)

        runner = Runner(SyncPickNet, None)
        runner.run_sync(None, max_cycles=0)
        place(runner, "SyncPickNet", "src").add_token("a")
        place(runner, "SyncPickNet", "src").add_token("b")
        runner.runtime.run_sync()

        assert tokens(runner, "SyncPickNet", "src") == ["a"]
        assert tokens(runner, "SyncPickNet", "out") == ["b"]

    def test_guard_that_yields_a_new_tuple_is_an_error(self):
        @pn.net
        def BadGuardNet(b):
            src = b.place("src")
            out = b.place("out")

            def rebuilds(combos, bb, tb):
                for c in combos:
                    yield tuple(c)

            @b.transition(guard=b.guard(rebuilds))
            def move(consumed, bb, tb):
                yield {out: consumed[0]}

            b.arc(src, move).arc(out)

        runner = Runner(BadGuardNet, None)
        runner.run_sync(None, max_cycles=0)
        place(runner, "BadGuardNet", "src").add_token("a")

        with pytest.raises(TypeError, match="binding"):
            runner.runtime.run_sync()


class TestStepsDoNotShareTokens:

    async def test_two_guarded_transitions_split_two_tokens(self):
        @pn.net
        def SplitNet(b):
            src = b.place("src")
            out_a = b.place("out_a")
            out_b = b.place("out_b")

            def accept_all(combos, bb, tb):
                yield from combos

            @b.transition(guard=b.guard(accept_all))
            async def first(consumed, bb, tb):
                yield {out_a: consumed[0]}

            @b.transition(guard=b.guard(accept_all))
            async def second(consumed, bb, tb):
                yield {out_b: consumed[0]}

            b.arc(src, first).arc(out_a)
            b.arc(src, second).arc(out_b)

        runner = Runner(SplitNet, None)
        await runner.start(None)
        place(runner, "SplitNet", "src").add_token("x")
        place(runner, "SplitNet", "src").add_token("y")
        await asyncio.sleep(0.1)

        produced = tokens(runner, "SplitNet", "out_a") + tokens(runner, "SplitNet", "out_b")
        assert sorted(produced) == ["x", "y"]
        assert tokens(runner, "SplitNet", "src") == []
        await runner.stop()

    async def test_one_token_fires_one_of_two_transitions(self):
        @pn.net
        def RaceNet(b):
            src = b.place("src")
            out_a = b.place("out_a")
            out_b = b.place("out_b")

            def accept_all(combos, bb, tb):
                yield from combos

            @b.transition(guard=b.guard(accept_all))
            async def first(consumed, bb, tb):
                yield {out_a: consumed[0]}

            @b.transition(guard=b.guard(accept_all))
            async def second(consumed, bb, tb):
                yield {out_b: consumed[0]}

            b.arc(src, first).arc(out_a)
            b.arc(src, second).arc(out_b)

        runner = Runner(RaceNet, None)
        await runner.start(None)
        place(runner, "RaceNet", "src").add_token("only")
        await asyncio.sleep(0.1)

        produced = tokens(runner, "RaceNet", "out_a") + tokens(runner, "RaceNet", "out_b")
        assert produced == ["only"]
        await runner.stop()


class TestTokenIdentity:

    async def test_integer_tokens_are_not_confused_with_token_ids(self):
        @pn.net
        def IntNet(b):
            src = b.place("src")
            out = b.place("out")

            @b.transition()
            async def move(consumed, bb, tb):
                yield {out: consumed[0]}

            b.arc(src, move).arc(out)

        runner = Runner(IntNet, None)
        await runner.start(None)
        for value in (0, "a", 1, 1):
            place(runner, "IntNet", "src").add_token(value)
        await asyncio.sleep(0.1)

        assert Counter(tokens(runner, "IntNet", "out")) == Counter([0, "a", 1, 1])
        await runner.stop()

    async def test_consumed_tokens_leave_the_registry(self):
        @pn.net
        def DrainNet(b):
            src = b.place("src")

            @b.transition()
            async def drain(consumed, bb, tb):
                return
                yield

            b.arc(src, drain)

        runner = Runner(DrainNet, None)
        await runner.start(None)
        for i in range(10):
            place(runner, "DrainNet", "src").add_token(i)
        await asyncio.sleep(0.1)

        assert tokens(runner, "DrainNet", "src") == []
        assert len(runner.runtime.token_registry._tokens) == 0
        await runner.stop()


class TestIdleRunner:

    async def test_rejecting_guard_is_not_called_while_marking_is_unchanged(self):
        calls = 0

        @pn.net
        def IdleNet(b):
            never = b.place("never")
            sink = b.place("sink")

            def reject_all(combos, bb, tb):
                nonlocal calls
                calls += 1

            @b.transition(guard=b.guard(reject_all))
            async def never_fires(consumed, bb, tb):
                yield {sink: consumed[0]}

            b.arc(never, never_fires).arc(sink)

        runner = Runner(IdleNet, None)
        await runner.start(None)
        place(runner, "IdleNet", "never").add_token("x")
        await asyncio.sleep(0.3)

        assert calls == 1
        await runner.stop()

    async def test_new_token_wakes_the_runner(self):
        @pn.net
        def WakeNet(b):
            src = b.place("src")
            out = b.place("out")

            def wants_go(combos, bb, tb):
                for c in combos:
                    if c[0][0] == "go":
                        yield c

            @b.transition(guard=b.guard(wants_go))
            async def go(consumed, bb, tb):
                yield {out: consumed[0]}

            b.arc(src, go).arc(out)

        runner = Runner(WakeNet, None)
        await runner.start(None)
        place(runner, "WakeNet", "src").add_token("wait")
        await asyncio.sleep(0.1)
        assert tokens(runner, "WakeNet", "out") == []

        place(runner, "WakeNet", "src").add_token("go")
        await asyncio.sleep(0.1)

        assert tokens(runner, "WakeNet", "out") == ["go"]
        assert tokens(runner, "WakeNet", "src") == ["wait"]
        await runner.stop()

    async def test_delay_end_wakes_the_runner(self):
        clock = CycleClock()

        @pn.net
        def DelayNet(b):
            src = b.place("src")
            out = b.place("out")

            @b.transition(delay=2)
            async def later(consumed, bb, tb):
                yield {out: consumed[0]}

            b.arc(src, later).arc(out)

        runner = Runner(DelayNet, None)
        await runner.start(clock)
        place(runner, "DelayNet", "src").add_token("t")
        await asyncio.sleep(0.05)
        clock.advance()
        await asyncio.sleep(0.05)
        assert tokens(runner, "DelayNet", "out") == []

        clock.advance()
        await asyncio.sleep(0.05)
        assert tokens(runner, "DelayNet", "out") == ["t"]
        await runner.stop()
