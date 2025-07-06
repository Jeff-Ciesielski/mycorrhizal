#!/usr/bin/env python3
"""
Minimal Cordyceps Petri Net Demo
- One token source
- One queue
- One processor
- Print token flow
"""
import asyncio
import time
from cordyceps.core import Token, Place, Transition, PetriNet, Arc, IOInputPlace

class SimpleToken(Token):
    def __init__(self, value):
        super().__init__()
        self.value = value
    def __repr__(self):
        return f"SimpleToken({self.value})"

class Source(IOInputPlace):
    @property
    def is_finished(self):
        return self.generated >= self.count
    def __init__(self, count=3):
        super().__init__()
        self.count = count
        self.generated = 0
    async def on_input(self):
        if self.generated >= self.count:
            self._ensure_shutdown_event()
            await self._shutdown_event.wait()
            return
        token = SimpleToken(self.generated)
        self.generated += 1
        await self.produce_token(token)
        print(f"[Source] Produced: {token}")

class Queue(Place):
    pass

class Processor(Place):
    pass

class MoveToQueue(Transition):
    def input_arcs(self):
        return {"src": Arc(MinimalNet.Src)}
    def output_arcs(self):
        return {"queue": Arc(MinimalNet.Q)}
    async def on_fire(self, consumed):
        token = consumed["src"][0]
        print(f"[MoveToQueue] Moving {token} to Queue")
        result = {"queue": [token]}
        # Print queue size after move
        queue = self._net._get_place_by_type(MinimalNet.Q)
        print(f"[Queue] Token count after move: {queue.token_count + 1}")  # +1 because not yet added
        return result

class ProcessToken(Transition):
    PRIORITY = 10  # Higher than MoveToQueue
    def input_arcs(self):
        return {"queue": Arc(MinimalNet.Q)}
    def output_arcs(self):
        return {"proc": Arc(MinimalNet.Proc)}
    async def guard(self, pending):
        # Consume all tokens in the queue
        for token in pending["queue"]:
            self.consume("queue", token)
        return bool(self._to_consume.get("queue"))
    async def on_fire(self, consumed):
        for token in consumed["queue"]:
            print(f"[ProcessToken] Processing {token}")
            await asyncio.sleep(0.1)
            self.produce("proc", token)
        # Print queue size after processing
        queue = self._net._get_place_by_type(MinimalNet.Q)
        print(f"[Queue] Token count after processing: {queue.token_count}")

class MinimalNet(PetriNet):
    class Src(Source): pass
    class Q(Queue): pass
    class Proc(Processor): pass
    class T1(MoveToQueue): pass
    class T2(ProcessToken): pass

    def _should_terminate(self):
        src = self.get_place(self.Src)
        queue = self.get_place(self.Q)
        # Stop when source is done and queue is empty
        return src.is_finished and queue.is_empty

def main():
    net = MinimalNet()
    src = net.get_place(MinimalNet.Src)
    src.count = 5
    print("Starting minimal net...")
    asyncio.run(net.start())
    print("Done.")

if __name__ == "__main__":
    main()
