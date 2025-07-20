#!/usr/bin/env python3
"""
Minimal Hypha Petri Net Demo - Gevent Version
- One token source
- One queue  
- One processor
- Print token flow
"""

import time
from gevent import sleep
from mycorrhizal.hypha import (
    Token, Place, Transition, PetriNet, Arc, IOInputPlace, IOOutputPlace,
    PlaceName, DispatchPolicy
)
from enum import Enum, auto
from typing import Dict, List, Optional


class SimpleToken(Token):
    def __init__(self, value):
        super().__init__({"value": value})
        self.value = value

    def __repr__(self):
        return f"SimpleToken({self.value})"


class MinimalNet(PetriNet):
    """Simple linear pipeline: Source -> Queue -> Processor"""
    
    def __init__(self):
        super().__init__()
        self.finished = False
    
    class TokenQueue(Place):
        """Queue to hold tokens between source and processor"""
        pass
    
    class ProcessedTokens(Place):
        """Final destination for processed tokens"""
        pass
    
    class TokenSource(IOInputPlace):
        """Generates a fixed number of tokens autonomously"""
        
        def __init__(self, count=3):
            super().__init__()
            self.count = count
            self.generated = 0

        def on_input(self) -> Optional[SimpleToken]:
            if self.generated >= self.count:
                # Signal that we're done
                if self.net:
                    self.net.finished = True
                return None
            
            token = SimpleToken(self.generated)
            self.generated += 1
            print(f"[Source] Produced: {token}")
            sleep(0.5)  # Simulate generation delay
            return token
    
    class MoveToQueue(Transition):
        """Moves tokens from source to queue"""
        
        class ArcNames(PlaceName):
            FROM_SOURCE = auto()
            TO_QUEUE = auto()
        
        DISPATCH_POLICY = DispatchPolicy.ALL
        
        def input_arcs(self) -> Dict[PlaceName, Arc]:
            return {
                self.ArcNames.FROM_SOURCE: Arc(MinimalNet.TokenSource)
            }
        
        def output_arcs(self) -> Dict[PlaceName, Arc]:
            return {
                self.ArcNames.TO_QUEUE: Arc(MinimalNet.TokenQueue)
            }
        
        def on_fire(self, consumed: Dict[PlaceName, List[Token]]) -> Dict[PlaceName, List[Token]]:
            tokens = consumed[self.ArcNames.FROM_SOURCE]
            print(f"[MoveToQueue] Moving {len(tokens)} token(s) to Queue")
            
            # Show queue state after this transition fires
            for token in tokens:
                print(f"[MoveToQueue] Moving {token} to Queue")
            
            return {self.ArcNames.TO_QUEUE: tokens}
    
    class ProcessTokens(Transition):
        """Processes tokens from queue to final destination"""
        
        class ArcNames(PlaceName):
            FROM_QUEUE = auto()
            TO_PROCESSED = auto()
        
        DISPATCH_POLICY = DispatchPolicy.ALL
        
        def input_arcs(self) -> Dict[PlaceName, Arc]:
            return {
                self.ArcNames.FROM_QUEUE: Arc(MinimalNet.TokenQueue)
            }
        
        def output_arcs(self) -> Dict[PlaceName, Arc]:
            return {
                self.ArcNames.TO_PROCESSED: Arc(MinimalNet.ProcessedTokens)
            }
        
        def on_fire(self, consumed: Dict[PlaceName, List[Token]]) -> Dict[PlaceName, List[Token]]:
            tokens = consumed[self.ArcNames.FROM_QUEUE]
            print(f"[ProcessTokens] Received {len(tokens)} token(s)")
            
            processed_tokens = []
            for token in tokens:
                print(f"[ProcessTokens] Processing {token}")
                # Could modify the token here if needed
                processed_tokens.append(token)
            
            sleep(0.2)  # Simulate processing time
            return {self.ArcNames.TO_PROCESSED: processed_tokens}


def main():
    """Run the minimal demo"""
    print("="*50)
    print("Minimal Cordyceps Demo - Gevent Version")
    print("="*50)
    
    # Create network with 5 tokens
    net = MinimalNet()
    
    # Configure source
    source = net.get_place(MinimalNet.TokenSource)
    source.count = 5
    
    print(f"✓ Network created")
    print(f"✓ Source configured to generate {source.count} tokens")
    
    # Show network structure
    print("\nNetwork Structure:")
    print("-" * 30)
    for place_name, place in net._places_by_qualified_name.items():
        place_type = "INPUT" if isinstance(place, IOInputPlace) else "REGULAR"
        print(f"  {place_name} ({place_type})")
    
    print("\nTransitions:")
    for transition in net._transitions:
        policy = transition.DISPATCH_POLICY.value
        print(f"  {transition.qualified_name} (policy: {policy})")
    
    # Generate Mermaid diagram
    print("\nMermaid Diagram:")
    print("-" * 40)
    net.print_mermaid_diagram()
    print("-" * 40)
    
    print(f"\nStarting network...")
    net.start()
    
    try:
        print("Running until all tokens are processed...")
        
        # Wait for completion
        while not net.finished:
            sleep(0.1)
        
        # Give a bit more time for final processing
        sleep(1.0)
        
        # Show final state
        print("\n" + "="*50)
        print("Final State")
        print("="*50)
        
        state = net.get_state_summary()
        
        print("Final token counts:")
        for place_name, place_info in state['places'].items():
            status = []
            if place_info['is_empty']:
                status.append("empty")
            if place_info['is_full']:
                status.append("FULL")
            status_str = f" ({', '.join(status)})" if status else ""
            print(f"  {place_name}: {place_info['token_count']} tokens{status_str}")
        
        print("\nTransition fire counts:")
        for transition_name, transition_info in state['transitions'].items():
            print(f"  {transition_name}: {transition_info['fire_count']} fires")
        
        processed_place = net.get_place(MinimalNet.ProcessedTokens)
        print(f"\n✓ Successfully processed {processed_place.token_count} tokens")
        
    finally:
        print("\nStopping network...")
        net.stop()
        print("✓ Network stopped")


if __name__ == "__main__":
    main()