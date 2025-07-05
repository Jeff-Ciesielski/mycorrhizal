#!/usr/bin/env python3
"""
Minimal test to debug the guard issue
"""

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
        print(f"Firing! Consumed: {consumed}")

class TestNet(PetriNet):
    class TestPlace(TestPlace):
        pass
    
    class TestTransition(TestTransition):
        INPUT_ARCS = {}
        OUTPUT_ARCS = {}
        
        def __init__(self):
            super().__init__()
            self.INPUT_ARCS = {"input": Arc(TestNet.TestPlace)}
            self.OUTPUT_ARCS = {}

def test_basic_firing():
    print("=== Testing Basic Firing ===")
    
    # Create net
    net = TestNet()
    
    # Add a token
    net.send_message_to_place(TestNet.TestPlace, TestToken("test"))
    
    # Check state
    print(f"Tokens in TestPlace: {net.context.token_count(TestNet.TestPlace)}")
    
    # Try to tick
    try:
        net.tick()
        print("Tick completed successfully")
    except Exception as e:
        print(f"Tick failed: {e}")
    
    # Check transitions
    for transition in net._transitions:
        print(f"Transition {transition.__class__.__name__}: fired {transition.fire_count} times")

if __name__ == "__main__":
    test_basic_firing()