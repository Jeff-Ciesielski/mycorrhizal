"""
Example: Using Fork and Join interfaces in Cordyceps
"""
import asyncio
from cordyceps.core import Token, PetriNet
from cordyceps.util import Fork, Join

class MyToken(Token):
    def __init__(self, data, dest=None):
        super().__init__(data)
        self.destination = dest
    def __repr__(self):
        return f"MyToken({self.data}, dest={self.destination})"

# Fork example: replicate to all outputs
Fork3 = Fork(3)

class ForkNet(PetriNet):
    class Forker(Fork3): pass

# Join example: merge all inputs to one output
Join2 = Join(2)

class JoinNet(PetriNet):
    class Joiner(Join2): pass

async def main():
    print("=== Fork Example ===")
    net = ForkNet()
    await net.produce_token(ForkNet.Forker.Input, MyToken("hello"))
    await net.run_until_complete()
    for i in range(1, 4):
        out = net.get_place(getattr(ForkNet.Forker, f'Output_{i}'))
        print(f"Output_{i}: {out.tokens}")

    print("\n=== Join Example ===")
    net = JoinNet()
    await net.produce_token(JoinNet.Joiner.Input_1, MyToken("a"))
    await net.produce_token(JoinNet.Joiner.Input_2, MyToken("b"))
    await net.run_until_complete()
    out = net.get_place(JoinNet.Joiner.Output)
    print(f"Output: {out.tokens}")


if __name__ == "__main__":
    asyncio.run(main())
