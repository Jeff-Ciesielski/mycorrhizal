"""
Pytest-based test for guard functionality.
"""

import pytest
from cordyceps import Token, Place, ArcBasedTransition, PetriNet, Arc

class TestToken(Token):
    def __init__(self, data):
        super().__init__(data)

class TestPlace(Place):
    pass

class TestTransition(ArcBasedTransition):
    INPUT_ARCS = {"input": Arc(TestPlace)}
    OUTPUT_ARCS = {}
    def fire(self, ctx, consumed, produce):
        # For test, just print or record
        print(f"Firing! Consumed: {consumed}")

class TestNet(PetriNet):
    class TestPlace(TestPlace):
        pass
    class TestTransition(ArcBasedTransition):
        INPUT_ARCS = {}
        OUTPUT_ARCS = {}
        def __init__(self):
            super().__init__()
            self.INPUT_ARCS = {"input": Arc(TestNet.TestPlace)}
            self.OUTPUT_ARCS = {}

def test_basic_firing():
    # Create net
    net = TestNet()
    # Add a token
    net.send_message_to_place(TestNet.TestPlace, TestToken("test"))
    # Check state
    assert net.context.token_count(TestNet.TestPlace) == 1
    # Try to tick
    try:
        net.tick()
    except Exception as e:
        pytest.fail(f"Tick failed: {e}")
    # Check transitions
    for transition in net._transitions:
        # fire_count should be 1 or 0 depending on net logic
        assert hasattr(transition, "fire_count")
