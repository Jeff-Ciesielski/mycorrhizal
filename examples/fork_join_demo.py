#!/usr/bin/env python3
"""
Fork/Join Petri Net Demo - Asyncio Version
- One token source
- Forks to two branches (broadcast)
- Two queues (one per branch)
- Join merges both branches
- Final output place
- Prints token flow at each stage
"""

import asyncio
from asyncio import sleep
from mycorrhizal.hypha import (
    Token,
    Place,
    Transition,
    PetriNet,
    Arc,
    IOInputPlace,
    IOOutputPlace,
    PlaceName,
)
from mycorrhizal.hypha.util import ForkN, Merge, JoinFrom, ForwardFrom
from typing import Dict, List, Optional

print(f"PetriNet metaclass: {PetriNet.__class__}")
print(f"PetriNet MRO: {PetriNet.__class__.__mro__}")
print(f"Has DeclarativeMeta in MRO: {'DeclarativeMeta' in [cls.__name__ for cls in PetriNet.__class__.__mro__]}")
class SimpleToken(Token):
    pass


class ForkJoinNet(PetriNet):
    """Fork/Join pipeline: Source -> Forward -> Fork -> Join -> Fork -> Merge -> Output"""

    def __init__(self):
        super().__init__()
        self.finished = False

    # Places
    class TokenSource(IOInputPlace):
        def __init__(self, count=3):
            super().__init__()
            self.count = count
            self.generated = 0

        async def on_input(self) -> Optional[SimpleToken]:
            if self.generated >= self.count:
                if self.net:
                    self.net.finished = True
                return None
            token = SimpleToken(data=self.generated)
            self.generated += 1
            print(f"[Source] Produced: {token}")
            await self.net.sleep_cycles(0.5)
            return token

    class FinalOutput(Place):
        def on_token(self, token: Token):
            print(f"[FinalOutput] Received token: {token}")

    TokenBuffer = ForwardFrom(TokenSource)

    places = ForkN(2, TokenBuffer)
    Intermediate = JoinFrom(places)
    places2 = ForkN(3, Intermediate)
    Merges = Merge(places2, FinalOutput)


async def main():
    print("=" * 50)
    print("Fork/Join Hypha Demo")
    print("=" * 50)
    net = ForkJoinNet()
    source = net.get_place(ForkJoinNet.TokenSource)
    # Set token count if possible
    if hasattr(source, "count"):
        source.count = 5
        print(f"✓ Source configured to generate {source.count} tokens")
    print(f"✓ Network created")
    print(f"\nStarting network...")

    await net.start()
    try:
        print("Running until all tokens are processed...")
        while not net.finished:
            await sleep(0.1)
        print("\n" + "=" * 50)
        print("Final State")
        print("=" * 50)
        state = net.get_state_summary()
        print("Final token counts:")
        for place_name, place_info in state["places"].items():
            status = []
            if place_info["is_empty"]:
                status.append("empty")
            if place_info["is_full"]:
                status.append("FULL")
            status_str = f" ({', '.join(status)})" if status else ""
            print(f"  {place_name}: {place_info['token_count']} tokens{status_str}")
        print("\nTransition fire counts:")
        for transition_name, transition_info in state["transitions"].items():
            print(f"  {transition_name}: {transition_info['fire_count']} fires")
        processed_place = net.get_place(ForkJoinNet.FinalOutput)
        print(f"\n✓ Successfully processed {processed_place.token_count} tokens")
    finally:
        print("\nStopping network...")
        await net.stop()
        print("✓ Network stopped")
        print("\nMermaid Diagram:")
        print("-" * 40)
        net.print_mermaid_diagram()
        print("-" * 40)


if __name__ == "__main__":
    asyncio.run(main())
