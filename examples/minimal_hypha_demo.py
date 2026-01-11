#!/usr/bin/env python3
"""
Minimal Hypha Core2 Demo
- One token source (IO input)
- One queue place
- One processor transition
- Print token flow
"""

import asyncio
from mycorrhizal.hypha.core import pn, PlaceType, Runner
from mycorrhizal.hypha.core.builder import NetBuilder
from mycorrhizal.common.timebase import MonotonicClock


class SimpleToken:
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"SimpleToken({self.data})"


@pn.net
def MinimalNet(builder: NetBuilder):
    # Places
    queue = builder.place("queue", type=PlaceType.QUEUE)
    processed = builder.place("processed", type=PlaceType.QUEUE)

    # IO input source
    @builder.io_input_place()
    async def source(bb, timebase):
        for i in range(5):
            await timebase.sleep(0.2)
            token = SimpleToken(i)
            print(f"[Source] Produced: {token}")
            yield token

    # transition: move to queue (implicit guard via weight)
    @builder.transition()
    async def move_to_queue(consumed, bb, timebase):
        # consumed is list of tokens
        for token in consumed:
            print(f"[MoveToQueue] Moving {token} to queue")
            yield {queue: token}

    @builder.transition()
    async def process_tokens(consumed, bb, timebase):
        for token in consumed:
            print(f"[ProcessTokens] Processing {token}")
            await timebase.sleep(0.1)
            yield {processed: token}

    # wiring
    builder.arc(source, move_to_queue).arc(queue)
    builder.arc(queue, process_tokens).arc(processed)


async def main():
    print("Minimal Hypha Core2 Demo")

    from pydantic import BaseModel
    class DummyBB(BaseModel):
        pass

    bb = DummyBB()
    # Use a monotonic clock so the timebase advances automatically while the
    # runner is active. CycleClock is a manual-step clock and will not
    # progress here which prevented the IO input from running.
    timebase = MonotonicClock()

    from mycorrhizal.hypha.util import to_mermaid

    runner = Runner(MinimalNet, bb)
    print("Mermaid Diagram:\n", to_mermaid(MinimalNet._spec))

    await runner.start(timebase)

    # wait a short while for processing
    await asyncio.sleep(3)

    await runner.stop()


if __name__ == '__main__':
    asyncio.run(main())
