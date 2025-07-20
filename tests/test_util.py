"""
Unit tests for cordyceps.utils Fork and Join interfaces.
"""
# import pytest
# import asyncio
# from cordyceps.core import Token, PetriNet
# from cordyceps.util import ForkN, JoinN

# class SimpleToken(Token):
#     pass

# @pytest.mark.asyncio
# async def test_fork_3_replication():
#     Fork3 = ForkN(3)
#     class Net(PetriNet):
#         class Fork(Fork3): pass
#     net = Net()
#     input_place = net.get_place(Net.Fork.Input)
#     await net.produce_token(Net.Fork.Input, SimpleToken("foo"))
#     await net.run_until_complete()
#     # All outputs should have the token
#     for i in range(1, 4):
#         out = net.get_place(getattr(Net.Fork, f'Output_{i}'))
#         assert out.token_count == 1
#         assert out.tokens[0].data == "foo"

# @pytest.mark.asyncio
# async def test_join_2_merge():
#     Join2 = JoinN(2)
#     class Net(PetriNet):
#         class Join(Join2): pass
#     net = Net()
#     await net.produce_token(Net.Join.Input_1, SimpleToken("a"))
#     await net.produce_token(Net.Join.Input_2, SimpleToken("b"))
#     await net.run_until_complete()
#     out = net.get_place(Net.Join.Output)
#     assert out.token_count == 2
#     datas = {t.data for t in out.tokens}
#     assert datas == {"a", "b"}
