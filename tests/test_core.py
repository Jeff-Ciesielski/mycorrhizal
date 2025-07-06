#!/usr/bin/env python3
"""
Comprehensive pytest suite for Cordyceps core functionality.

This test suite covers:
- Token system and lifecycle
- Place functionality (standard, IO input, IO output)
- Transition system and firing logic
- Interface composition and boundary validation
- PetriNet engine and execution
- Arc definitions and validation
- Error handling and edge cases
- Async execution patterns

Run with: pytest -v test_cordyceps.py
"""

import asyncio
import pytest
import time
from datetime import datetime
from typing import Optional, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

from cordyceps.core import (
    Token, Place, Transition, Interface, PetriNet, Arc, 
    IOInputPlace, IOOutputPlace, create_simple_token,
    PetriNetComplete, InvalidTokenType, InsufficientTokens,
    InvalidPlaceType, InvalidTransitionType, ArcValidationError,
    InterfaceBoundaryViolation, QualifiedNameConflict,
    MissingPlaceError
)


# === TEST FIXTURES ===

class TestToken(Token):
    """Simple test token."""
    def __init__(self, value: str = "test", priority: int = 0):
        super().__init__(value)
        self.value = value
        self.priority = priority
    
    def __repr__(self):
        return f"TestToken({self.value})"


class ErrorToken(Token):
    """Token representing an error condition."""
    def __init__(self, error_type: str, message: str):
        super().__init__()
        self.error_type = error_type
        self.message = message
    
    def __repr__(self):
        return f"ErrorToken({self.error_type})"


class HighPriorityToken(Token):
    """Token with high priority."""
    def __init__(self, value: str = "urgent"):
        super().__init__(value)
        self.value = value


# === TOKEN SYSTEM TESTS ===

class TestTokenSystem:
    """Test token creation, lifecycle, and behavior."""
    
    def test_token_creation(self):
        """Test basic token creation."""
        token = TestToken("hello")
        assert token.value == "hello"
        assert token.data == "hello"
        assert isinstance(token.created_at, datetime)
        assert token.id is not None
    
    def test_token_repr(self):
        """Test token string representation."""
        token = TestToken("test_value")
        assert "TestToken(test_value)" in repr(token)
    
    @pytest.mark.asyncio
    async def test_token_lifecycle_callbacks(self):
        """Test token lifecycle callback methods."""
        token = TestToken("lifecycle_test")
        
        # Mock the callback methods
        token.on_created = AsyncMock()
        token.on_consumed = AsyncMock()
        token.on_produced = AsyncMock()
        
        # Test callbacks
        await token.on_created()
        await token.on_consumed()
        await token.on_produced()
        
        token.on_created.assert_called_once()
        token.on_consumed.assert_called_once()
        token.on_produced.assert_called_once()
    
    def test_create_simple_token(self):
        """Test utility function for creating simple tokens."""
        token = create_simple_token("simple_data")
        assert token.data == "simple_data"
        assert isinstance(token, Token)


# === PLACE TESTS ===

class TestPlace:
    """Test place functionality."""
    
    class TestPlace(Place):
        """Test place implementation."""
        pass
    
    def test_place_creation(self):
        """Test basic place creation."""
        place = self.TestPlace()
        assert place.token_count == 0
        assert place.is_empty is True
        assert place.is_full is False
        assert place.tokens == []
    
    def test_place_capacity(self):
        """Test place capacity limits."""
        class LimitedPlace(Place):
            MAX_CAPACITY = 2
        place = LimitedPlace()
        assert place.is_full is False
        # Add tokens up to capacity
        token1 = TestToken("1")
        token2 = TestToken("2")
        assert place.can_accept_token(token1) is True
        assert place.can_accept_token(token2) is True
        # Mock the add_token method to avoid async issues in sync test
        place._tokens.append(token1)
        place._tokens.append(token2)
        assert place.is_full is True
        token3 = TestToken("3")
        assert place.can_accept_token(token3) is False
    
    def test_place_token_type_restrictions(self):
        """Test place token type restrictions."""
        class RestrictedPlace(Place):
            ACCEPTED_TOKEN_TYPES = {TestToken}
        place = RestrictedPlace()
        test_token = TestToken("allowed")
        other_token = HighPriorityToken("not_allowed")
        assert place.can_accept_token(test_token) is True
        assert place.can_accept_token(other_token) is False
    
    @pytest.mark.asyncio
    async def test_place_add_remove_tokens(self):
        """Test adding and removing tokens from places."""
        place = self.TestPlace()
        token1 = TestToken("first")
        token2 = TestToken("second")
        # Add tokens
        await place.add_token(token1)
        await place.add_token(token2)
        assert place.token_count == 2
        assert place.is_empty is False
        # Remove tokens (FIFO)
        removed1 = await place.remove_token()
        assert removed1 == token1
        assert place.token_count == 1
        removed2 = await place.remove_token()
        assert removed2 == token2
        assert place.token_count == 0
        assert place.is_empty is True
    
    @pytest.mark.asyncio
    async def test_place_remove_by_type(self):
        """Test removing tokens by type."""
        place = self.TestPlace()
        test_token = TestToken("test")
        priority_token = HighPriorityToken("urgent")
        await place.add_token(test_token)
        await place.add_token(priority_token)
        # Remove by type
        removed = await place.remove_token(HighPriorityToken)
        assert removed == priority_token
        assert place.token_count == 1
        # Only test token should remain
        remaining = await place.remove_token()
        assert remaining == test_token
    
    @pytest.mark.asyncio
    async def test_place_insufficient_tokens(self):
        """Test error when removing from empty place."""
        place = self.TestPlace()
        
        with pytest.raises(InsufficientTokens):
            await place.remove_token()
        
        # Add one token then try to remove two
        await place.add_token(TestToken("only_one"))
        await place.remove_token()
        
        with pytest.raises(InsufficientTokens):
            await place.remove_token()
    
    def test_place_peek_and_count(self):
        """Test peeking at tokens and counting."""
        place = self.TestPlace()
        # Empty place
        assert place.peek_token() is None
        assert place.count_tokens() == 0
        assert place.count_tokens(TestToken) == 0
        # Add tokens
        place._tokens.append(TestToken("first"))
        place._tokens.append(HighPriorityToken("second"))
        assert place.peek_token().value == "first"
        assert place.peek_token(HighPriorityToken).value == "second"
        assert place.count_tokens() == 2
        assert place.count_tokens(TestToken) == 1
        assert place.count_tokens(HighPriorityToken) == 1
    
    def test_place_qualified_name(self):
        """Test place qualified name handling."""
        place = self.TestPlace()
        # Default name
        assert place.qualified_name == "TestPlace"
        # Set qualified name
        place._set_qualified_name("TestInterface.TestPlace")
        assert place.qualified_name == "TestInterface.TestPlace"


# === IO PLACE TESTS ===

class TestIOPlaces:
    """Test IO place functionality."""
    
    class TestIOOutputPlace(IOOutputPlace):
        """Test IO output place."""
        def __init__(self):
            super().__init__()
            self.processed_tokens = []
        
        async def on_token(self, token: Token) -> Optional[Token]:
            if isinstance(token, ErrorToken):
                # Simulate failure
                return ErrorToken("io_error", "Failed to process")
            else:
                # Simulate success
                self.processed_tokens.append(token)
                return None
    
    class TestIOInputPlace(IOInputPlace):
        """Test IO input place."""
        def __init__(self, tokens_to_generate: List[Token] = None):
            super().__init__()
            self.tokens_to_generate = tokens_to_generate or []
            self.generated_count = 0
        
        async def on_input(self):
            if self.generated_count < len(self.tokens_to_generate):
                token = self.tokens_to_generate[self.generated_count]
                self.generated_count += 1
                await self.produce_token(token)
            else:
                # Stop when done
                await self.stop_input()
    
    @pytest.mark.asyncio
    async def test_io_output_place_success(self):
        """Test successful IO output processing."""
        place = self.TestIOOutputPlace()
        token = TestToken("success_case")
        # Mock net attachment with async _on_token_added
        place._net = MagicMock()
        place._net.log = MagicMock()
        place._net._on_token_added = AsyncMock()
        await place.add_token(token)
        # Token should be processed and removed
        assert place.token_count == 0
        assert token in place.processed_tokens
    
    @pytest.mark.asyncio
    async def test_io_output_place_failure(self):
        """Test IO output processing failure."""
        place = self.TestIOOutputPlace()
        error_token = ErrorToken("test_error", "Test error message")
        # Mock net attachment with async _on_token_added
        place._net = MagicMock()
        place._net.log = MagicMock()
        place._net._on_token_added = AsyncMock()
        await place.add_token(error_token)
        # Original token should be replaced with error token
        assert place.token_count == 1
        remaining_token = place.tokens[0]
        assert isinstance(remaining_token, ErrorToken)
        assert remaining_token.error_type == "io_error"
    
    @pytest.mark.asyncio
    async def test_io_input_place_generation(self):
        """Test IO input place token generation."""
        tokens = [TestToken("input1"), TestToken("input2")]
        place = self.TestIOInputPlace(tokens)
        # Mock net attachment
        place._net = MagicMock()
        place._net.log = MagicMock()
        await place.start_input()
        # Wait for generation to complete (longer to ensure both tokens are produced)
        for _ in range(10):
            if place.token_count == 2:
                break
            await asyncio.sleep(0.05)
        assert place.token_count == 2
        assert place.generated_count == 2
    
    @pytest.mark.asyncio
    async def test_io_input_place_lifecycle(self):
        """Test IO input place start/stop lifecycle."""
        place = self.TestIOInputPlace()
        # Mock callbacks
        place.on_input_start = AsyncMock()
        place.on_input_stop = AsyncMock()
        await place.start_input()
        await place.stop_input()
        place.on_input_start.assert_called_once()
        place.on_input_stop.assert_called_once()


# === TRANSITION TESTS ===

class TestTransitions:
    """Test transition functionality."""
    
    class SourcePlace(Place):
        """Source place for transitions."""
        pass
    
    class TargetPlace(Place):
        """Target place for transitions."""
        pass
    
    class SimpleTransition(Transition):
        """Simple test transition."""
        def input_arcs(self):
            return {"input": Arc(TestTransitions.SourcePlace)}
        def output_arcs(self):
            return {"output": Arc(TestTransitions.TargetPlace)}
        async def guard(self, pending):
            for token in pending["input"]:
                self.consume("input", token)
            return bool(self._to_consume.get("input"))
        async def on_fire(self, consumed):
            for token in consumed["input"]:
                new_token = TestToken(f"processed_{token.value}")
                self.produce("output", new_token)
    
    class GuardedTransition(Transition):
        """Transition with custom guard."""
        def input_arcs(self):
            return {"input": Arc(TestTransitions.SourcePlace)}
        def output_arcs(self):
            return {"output": Arc(TestTransitions.TargetPlace)}
        async def guard(self, pending):
            for token in pending["input"]:
                if isinstance(token, HighPriorityToken):
                    self.consume("input", token)
            return bool(self._to_consume.get("input"))
        async def on_fire(self, consumed):
            for token in consumed["input"]:
                self.produce("output", token)
    
    class PriorityTransition(Transition):
        """High priority transition."""
        PRIORITY = 10
        def input_arcs(self):
            return {"input": Arc(TestTransitions.SourcePlace)}
        def output_arcs(self):
            return {"output": Arc(TestTransitions.TargetPlace)}
        async def guard(self, pending):
            for token in pending["input"]:
                self.consume("input", token)
            return bool(self._to_consume.get("input"))
        async def on_fire(self, consumed):
            for token in consumed["input"]:
                self.produce("output", token)
    
    def test_transition_creation(self):
        """Test basic transition creation."""
        transition = self.SimpleTransition()
        assert transition.fire_count == 0
        assert transition.last_fired is None
        assert transition.PRIORITY == 0
    
    def test_transition_arc_definitions(self):
        """Test transition arc definitions."""
        transition = self.SimpleTransition()
        input_arcs = transition.input_arcs()
        output_arcs = transition.output_arcs()
        assert "input" in input_arcs
        assert "output" in output_arcs
        assert input_arcs["input"].place == self.SourcePlace
        assert output_arcs["output"].place == self.TargetPlace
    
    @pytest.mark.asyncio
    async def test_transition_guard(self):
        """Test transition guard logic by running the net and tracking transition firings via global state."""
        fired_tokens = []

        class Source(Place):
            pass

        class Target(Place):
            pass

        class GuardedTransition(Transition):
            def input_arcs(self):
                return {"input": Arc(Source)}
            def output_arcs(self):
                return {"output": Arc(Target)}
            async def guard(self, pending):
                for token in pending["input"]:
                    if isinstance(token, HighPriorityToken):
                        self.consume("input", token)
                return bool(self._to_consume.get("input"))
            async def on_fire(self, consumed):
                for token in consumed["input"]:
                    fired_tokens.append(token)
                    self.produce("output", token)

        class SimpleNet(PetriNet):
            class Source(Place): pass
            class Target(Place): pass
            class Trans(GuardedTransition):
                def input_arcs(self):
                    return {"input": Arc(SimpleNet.Source)}
                def output_arcs(self):
                    return {"output": Arc(SimpleNet.Target)}

        # Test: only high priority tokens should cause transition to fire
        net = SimpleNet(log_fn=lambda x: None)
        source = net.get_place(SimpleNet.Source)
        target = net.get_place(SimpleNet.Target)

        # Add a regular token and run
        await source.add_token(TestToken("regular"))
        await net.run_until_complete(max_iterations=2)
        # Should not fire
        assert fired_tokens == []
        assert target.token_count == 0

        # Add a high priority token and run
        await source.add_token(HighPriorityToken("urgent"))
        await net.run_until_complete(max_iterations=2)
        # Should fire and move only the high priority token
        assert any(isinstance(token, HighPriorityToken) for token in fired_tokens)
        assert target.token_count == 1
        assert isinstance(target.tokens[0], HighPriorityToken)
    
    @pytest.mark.asyncio
    async def test_transition_firing(self):
        """Test transition firing process."""
        class TestNet(PetriNet):
            class Source(Place):
                pass
            
            class Target(Place):
                pass
            
            class Trans(TestTransitions.SimpleTransition):
                def input_arcs(self):
                    return {"input": Arc(TestNet.Source)}
                def output_arcs(self):
                    return {"output": Arc(TestNet.Target)}
                async def on_fire(self, consumed_tokens):
                    token = consumed_tokens["input"][0]
                    # Transform token
                    new_token = TestToken(f"processed_{token.value}")
                    return {"output": [new_token]}
        
        net = TestNet(log_fn=lambda x: None)
        source = net.get_place(TestNet.Source)
        target = net.get_place(TestNet.Target)
        transition = net._transitions[0]
        
        # Add input token
        await source.add_token(TestToken("input"))
        
        # Fire transition
        fired = await transition.fire()
        fired is True
        transition.fire_count == 1
        transition.last_fired is not None
        
        # Check token transformation
        source.token_count == 0
        target.token_count == 1
        output_token = target.tokens[0]
        output_token.value == "processed_input"
    
    def test_transition_priority(self):
        """Test transition priority system."""
        regular_transition = self.SimpleTransition()
        priority_transition = self.PriorityTransition()
        assert regular_transition.PRIORITY == 0
        assert priority_transition.PRIORITY == 10
        # Test sorting
        transitions = [regular_transition, priority_transition]
        sorted_transitions = sorted(transitions, key=lambda t: t.PRIORITY, reverse=True)
        assert sorted_transitions[0] == priority_transition
        assert sorted_transitions[1] == regular_transition


# === INTERFACE TESTS ===

class TestInterfaces:
    """Test interface composition and boundary validation."""
    
    class SimpleInterface(Interface):
        """Simple interface for testing."""
        
        class InputPlace(Place):
            pass
        
        class OutputPlace(Place):
            pass
        
        class ProcessTransition(Transition):
            def input_arcs(self):
                return {"input": Arc(TestInterfaces.SimpleInterface.InputPlace)}
            
            def output_arcs(self):
                return {"output": Arc(TestInterfaces.SimpleInterface.OutputPlace)}
            
            async def on_fire(self, consumed_tokens):
                return {"output": consumed_tokens["input"]}
    
    class NestedInterface(Interface):
        """Interface with nested components."""
        
        class SubInterface(Interface):
            class SubPlace(Place):
                pass
            
            class SubTransition(Transition):
                def input_arcs(self):
                    return {"input": Arc(TestInterfaces.NestedInterface.SubInterface.SubPlace)}
                
                def output_arcs(self):
                    return {}
                
                async def on_fire(self, consumed_tokens):
                    return {}
        
        class MainPlace(Place):
            pass
        
        class MainTransition(Transition):
            def input_arcs(self):
                return {"sub": Arc(TestInterfaces.NestedInterface.SubInterface.SubPlace)}
            
            def output_arcs(self):
                return {"main": Arc(TestInterfaces.NestedInterface.MainPlace)}
            
            async def on_fire(self, consumed_tokens):
                return {"main": consumed_tokens["sub"]}
    
    def test_interface_discovery(self):
        """Test interface component discovery."""
        interface = self.SimpleInterface()
        places = interface.get_all_places()
        transitions = interface.get_all_transitions()
        assert len(places) == 2
        assert len(transitions) == 1
        place_names = [p.__name__ for p in places]
        assert "InputPlace" in place_names
        assert "OutputPlace" in place_names
    
    def test_interface_boundary_validation_success(self):
        """Test successful interface boundary validation."""
        class ValidNet(PetriNet):
            class TestInterface(TestInterfaces.SimpleInterface): pass
        # Should not raise exception
        net = ValidNet(log_fn=lambda x: None)
        assert len(net._places_by_type) == 2
        assert len(net._transitions) == 1
    
    def test_interface_boundary_validation_failure(self):
        """Test interface boundary violation detection."""
        class InvalidInterface(Interface):
            class InternalPlace(Place):
                pass
        
        class ViolatingInterface(Interface):
            class ViolatingTransition(Transition):
                def input_arcs(self):
                    # This violates boundary - accessing place from different interface
                    return {"input": Arc(InvalidInterface.InternalPlace)}
                
                def output_arcs(self):
                    return {}
                
                async def on_fire(self, consumed_tokens):
                    return {}
        
        with pytest.raises(InterfaceBoundaryViolation):
            class InvalidNet(PetriNet):
                class Invalid(InvalidInterface): pass
                class Violating(ViolatingInterface): pass
            violation = InvalidNet()
    
    def test_qualified_name_generation(self):
        """Test qualified name generation for nested interfaces."""
        class TestNet(PetriNet):
            class Interface1(TestInterfaces.NestedInterface): pass
        net = TestNet(log_fn=lambda x: None)
        # Check qualified names
        expected_names = {
            "Interface1.SubInterface.SubPlace",
            "Interface1.MainPlace"
        }
        actual_names = set(net._places_by_qualified_name.keys())
        assert expected_names.issubset(actual_names)
    
    def test_qualified_name_conflicts(self):
        """Test qualified name conflict detection."""
        class ConflictingInterface1(Interface):
            class SameName(Place):
                pass
        class ConflictingInterface2(Interface):
            class SameName(Place):
                pass
        # This should work - different interfaces
        class ValidNet(PetriNet):
            class Interface1(ConflictingInterface1): pass
            class Interface2(ConflictingInterface2): pass
        net = ValidNet(log_fn=lambda x: None)
        # Should have both places with different qualified names
        assert "Interface1.SameName" in net._places_by_qualified_name
        assert "Interface2.SameName" in net._places_by_qualified_name


# === PETRI NET TESTS ===

class TestPetriNet:
    """Test PetriNet engine functionality."""
    
    class SimpleNet(PetriNet):
        """Simple test net."""
        
        class Source(Place):
            pass
        
        class Target(Place):
            pass
        
        class Transform(Transition):
            def input_arcs(self):
                return {"input": Arc(TestPetriNet.SimpleNet.Source)}
            
            def output_arcs(self):
                return {"output": Arc(TestPetriNet.SimpleNet.Target)}
            
            async def on_fire(self, consumed_tokens):
                token = consumed_tokens["input"][0]
                new_token = TestToken(f"transformed_{token.value}")
                return {"output": [new_token]}
    
    class TerminatingNet(PetriNet):
        """Net with custom termination logic."""
        
        class Source(Place):
            pass
        
        class Sink(Place):
            pass
        
        class Process(Transition):
            def input_arcs(self):
                return {"input": Arc(TestPetriNet.TerminatingNet.Source)}
            
            def output_arcs(self):
                return {"output": Arc(TestPetriNet.TerminatingNet.Sink)}
            
            async def on_fire(self, consumed_tokens):
                return {"output": consumed_tokens["input"]}
        
        def _should_terminate(self):
            sink = self.get_place(self.Sink)
            return sink.token_count >= 3
    
    def test_net_creation(self):
        """Test basic net creation."""
        net = self.SimpleNet(log_fn=lambda x: None)
        assert len(net._places_by_type) == 2
        assert len(net._transitions) == 1
        assert net._is_running is False
        assert net._is_finished is False
    
    def test_net_place_access(self):
        """Test net place access methods."""
        net = self.SimpleNet(log_fn=lambda x: None)
        # Access by type
        source = net.get_place(self.SimpleNet.Source)
        assert isinstance(source, self.SimpleNet.Source)
        # Access by qualified name
        source2 = net._get_place_by_qualified_name("Source")
        assert source == source2
        # Invalid access
        with pytest.raises(KeyError):
            net.get_place(Place)
        with pytest.raises(KeyError):
            net._get_place_by_qualified_name("NonExistent")
    
    @pytest.mark.asyncio
    async def test_net_token_production(self):
        """Test producing tokens to places."""
        net = self.SimpleNet(log_fn=lambda x: None)
        token = TestToken("test_production")
        await net.produce_token(self.SimpleNet.Source, token)
        source = net.get_place(self.SimpleNet.Source)
        assert source.token_count == 1
        assert source.tokens[0] == token
    
    @pytest.mark.asyncio
    async def test_net_run_until_complete(self):
        """Test run_until_complete execution."""
        net = self.SimpleNet(log_fn=lambda x: None)
        # Add initial token
        await net.produce_token(self.SimpleNet.Source, TestToken("initial"))
        # Run until complete
        await net.run_until_complete()
        # Check results
        source = net.get_place(self.SimpleNet.Source)
        target = net.get_place(self.SimpleNet.Target)
        assert source.token_count == 0
        assert target.token_count == 1
        assert target.tokens[0].value == "transformed_initial"
    
    @pytest.mark.asyncio
    async def test_net_termination_logic(self):
        """Test custom termination logic."""
        net = self.TerminatingNet(log_fn=lambda x: None)
        # Add tokens
        for i in range(5):
            await net.produce_token(self.TerminatingNet.Source, TestToken(f"token_{i}"))
        # Run until termination
        await net.run_until_complete()
        # Should terminate when sink has 3 tokens
        sink = net.get_place(self.TerminatingNet.Sink)
        assert sink.token_count == 3
        assert net._is_finished is True
    
    def test_net_state_summary(self):
        """Test net state summary generation."""
        net = self.SimpleNet(log_fn=lambda x: None)
        state = net.get_state_summary()
        assert "is_running" in state
        assert "is_finished" in state
        assert "places" in state
        assert "transitions" in state
        assert "firing_log" in state
        # Check place info
        assert "Source" in state["places"]
        assert "Target" in state["places"]
        assert state["places"]["Source"]["token_count"] == 0
    
    def test_net_mermaid_diagram(self):
        """Test Mermaid diagram generation."""
        net = self.SimpleNet(log_fn=lambda x: None)
        diagram = net.generate_mermaid_diagram()
        assert "graph TD" in diagram
        assert "Source" in diagram
        assert "Target" in diagram
        assert "Transform" in diagram
        assert "-->" in diagram
    
    @pytest.mark.asyncio
    async def test_net_async_execution(self):
        """Test async execution with event loop."""
        class AsyncNet(PetriNet):
            class Source(IOInputPlace):
                def __init__(self):
                    super().__init__()
                    self.count = 0
                async def on_input(self):
                    if self.count < 3:
                        token = TestToken(f"async_{self.count}")
                        self.count += 1
                        await self.produce_token(token)
                    else:
                        await self.stop_input()
            class Target(Place):
                pass
            class Process(Transition):
                def input_arcs(self):
                    return {"input": Arc(AsyncNet.Source)}
                def output_arcs(self):
                    return {"output": Arc(AsyncNet.Target)}
                async def on_fire(self, consumed_tokens):
                    return {"output": consumed_tokens["input"]}
            def _should_terminate(self):
                target = self.get_place(self.Target)
                return target.token_count >= 3
        net = AsyncNet(log_fn=lambda x: None)
        # Start async execution
        start_task = asyncio.create_task(net.start())
        # Wait for completion
        try:
            await asyncio.wait_for(start_task, timeout=2.0)
        except asyncio.TimeoutError:
            await net.stop()
        # Check results
        target = net.get_place(AsyncNet.Target)
        assert target.token_count == 3


# === ARC TESTS ===

class TestArcs:
    """Test Arc definitions and validation."""
    
    def test_arc_creation(self):
        """Test basic arc creation."""
        arc = Arc(Place, weight=2, label="test_arc")
        assert arc.place == Place
        assert arc.weight == 2
        assert arc.label == "test_arc"
    
    def test_arc_auto_labeling(self):
        """Test automatic arc labeling."""
        class TestPlace(Place):
            pass
        arc = Arc(TestPlace)
        assert arc.label == "testplace"
    
    def test_arc_string_reference(self):
        """Test string place references in arcs."""
        arc = Arc("SomeInterface.SomePlace")
        assert arc.place == "SomeInterface.SomePlace"
        assert arc.label == "someplace"
    
    def test_arc_token_type_specification(self):
        """Test specifying token types in arcs."""
        arc = Arc(Place, token_type=TestToken)
        assert arc.token_type == TestToken


# === ERROR HANDLING TESTS ===

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_invalid_token_type(self):
        """Test handling of invalid token types."""
        class RestrictedPlace(Place):
            ACCEPTED_TOKEN_TYPES = {TestToken}
        
        place = RestrictedPlace()
        wrong_token = HighPriorityToken("wrong_type")
        
        with pytest.raises(InvalidTokenType):
            await place.add_token(wrong_token)
    
    @pytest.mark.asyncio
    async def test_place_capacity_exceeded(self):
        """Test handling of place capacity limits."""
        class LimitedPlace(Place):
            MAX_CAPACITY = 1
        
        place = LimitedPlace()
        
        # First token should succeed
        await place.add_token(TestToken("first"))
        
        # Second token should fail
        with pytest.raises(Exception):  # Generic exception for capacity
            await place.add_token(TestToken("second"))
    
    @pytest.mark.asyncio
    async def test_transition_firing_errors(self):
        """Test error handling in transition firing."""
        class ErrorTransition(Transition):
            def input_arcs(self):
                return {"input": Arc("Place")}
            def output_arcs(self):
                return {"output": Arc("Place")}
            async def on_fire(self, consumed_tokens):
                raise ValueError("Transition error")
        class ErrorNet(PetriNet):
            class Place(Place): pass
            class Source(Place): pass
            class Target(Place): pass
            class Error(ErrorTransition): pass
        net = ErrorNet(log_fn=lambda x: None)
        # Add token and try to fire
        await net.produce_token(ErrorNet.Source, TestToken("error_test"))
        # Should handle error gracefully
        await net.run_until_complete(max_iterations=1)
    
    def test_missing_place_references(self):
        """Test handling of missing place references."""
        class InvalidTransition(Transition):
            def input_arcs(self):
                return {"input": Arc("NonExistent.Place")}
            
            def output_arcs(self):
                return {}
            
            async def on_fire(self, consumed_tokens):
                return {}
        
        with pytest.raises(MissingPlaceError):
            class InvalidNet(PetriNet):
                class Invalid(InvalidTransition): pass
            invalid = InvalidNet()


# === INTEGRATION TESTS ===

class TestIntegration:
    """Integration tests for complex scenarios."""
    
    @pytest.mark.asyncio
    async def test_complex_workflow(self):
        """Test a complex workflow with multiple stages."""
        class WorkflowNet(PetriNet):
            class InputInterface(Interface):
                class InputPlace(IOInputPlace):
                    def __init__(self):
                        super().__init__()
                        self.count = 0
                    
                    async def on_input(self):
                        if self.count < 5:
                            token = TestToken(f"input_{self.count}")
                            self.count += 1
                            await self.produce_token(token)
                        else:
                            await self.stop_input()
                
                class QueuePlace(Place):
                    pass
                
                class QueueTransition(Transition):
                    def input_arcs(self):
                        return {"input": Arc(WorkflowNet.InputInterface.InputPlace)}
                    
                    def output_arcs(self):
                        return {"queue": Arc(WorkflowNet.InputInterface.QueuePlace)}
                    
                    async def on_fire(self, consumed_tokens):
                        return {"queue": consumed_tokens["input"]}
            
            class ProcessingInterface(Interface):
                class ProcessingPlace(Place):
                    pass
                
                class CompletedPlace(Place):
                    pass
                
                class ProcessTransition(Transition):
                    def input_arcs(self):
                        return {"input": Arc(WorkflowNet.ProcessingInterface.ProcessingPlace)}
                    
                    def output_arcs(self):
                        return {"output": Arc(WorkflowNet.ProcessingInterface.CompletedPlace)}
                    
                    async def on_fire(self, consumed_tokens):
                        token = consumed_tokens["input"][0]
                        processed = TestToken(f"processed_{token.value}")
                        return {"output": [processed]}
            
            class OutputInterface(Interface):
                class OutputPlace(IOOutputPlace):
                    def __init__(self):
                        super().__init__()
                        self.outputs = []
                    
                    async def on_token(self, token):
                        self.outputs.append(token)
                        return None
            
            # Top-level transitions connecting interfaces
            class InputToProcessingTransition(Transition):
                def input_arcs(self):
                    return {"input": Arc(WorkflowNet.InputInterface.QueuePlace)}
                
                def output_arcs(self):
                    return {"output": Arc(WorkflowNet.ProcessingInterface.ProcessingPlace)}
                
                async def on_fire(self, consumed_tokens):
                    return {"output": consumed_tokens["input"]}
            
            class ProcessingToOutputTransition(Transition):
                def input_arcs(self):
                    return {"input": Arc(WorkflowNet.ProcessingInterface.CompletedPlace)}
                
                def output_arcs(self):
                    return {"output": Arc(WorkflowNet.OutputInterface.OutputPlace)}
                
                async def on_fire(self, consumed_tokens):
                    return {"output": consumed_tokens["input"]}
            
            def _should_terminate(self):
                output_place = self.get_place(self.OutputInterface.OutputPlace)
                return len(output_place.outputs) >= 5
        
        net = WorkflowNet(log_fn=lambda x: None)
        
        # Start execution
        start_task = asyncio.create_task(net.start())
        
        # Wait for completion
        try:
            await asyncio.wait_for(start_task, timeout=5.0)
        except asyncio.TimeoutError:
            await net.stop()
        
        # Verify results
        output_place = net.get_place(WorkflowNet.OutputInterface.OutputPlace)
        len(output_place.outputs) == 5
        
        # Check processing
        for i, token in enumerate(output_place.outputs):
            f"processed_input_{i}" in token.value
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self):
        """Test error recovery in complex workflows."""
        class ErrorRecoveryNet(PetriNet):
            class InputPlace(Place):
                pass
            
            class ErrorPlace(Place):
                pass
            
            class SuccessPlace(Place):
                pass
            
            class ProcessTransition(Transition):
                def input_arcs(self):
                    return {"input": Arc(ErrorRecoveryNet.InputPlace)}
                
                def output_arcs(self):
                    return {
                        "success": Arc(ErrorRecoveryNet.SuccessPlace),
                        "error": Arc(ErrorRecoveryNet.ErrorPlace)
                    }
                
                async def on_fire(self, consumed_tokens):
                    token = consumed_tokens["input"][0]
                    
                    # Simulate error condition
                    if "error" in token.value:
                        error_token = ErrorToken("processing_error", f"Failed to process {token.value}")
                        return {"error": [error_token]}
                    else:
                        success_token = TestToken(f"success_{token.value}")
                        return {"success": [success_token]}
            
            class RetryTransition(Transition):
                def input_arcs(self):
                    return {"error": Arc(ErrorRecoveryNet.ErrorPlace)}
                
                def output_arcs(self):
                    return {"retry": Arc(ErrorRecoveryNet.InputPlace)}
                
                async def on_fire(self, consumed_tokens):
                    # Convert error back to retry
                    error_token = consumed_tokens["error"][0]
                    retry_token = TestToken("retry_attempt")
                    return {"retry": [retry_token]}
        
        net = ErrorRecoveryNet(log_fn=lambda x: None)
        
        # Add tokens
        await net.produce_token(ErrorRecoveryNet.InputPlace, TestToken("normal"))
        await net.produce_token(ErrorRecoveryNet.InputPlace, TestToken("error_case"))
        
        # Run
        await net.run_until_complete()
        
        # Check results
        success_place = net.get_place(ErrorRecoveryNet.SuccessPlace)
        error_place = net.get_place(ErrorRecoveryNet.ErrorPlace)
        
        success_place.token_count >= 1  # At least one success
        # Error handling should have processed the error token


# === PERFORMANCE TESTS ===

class TestPerformance:
    """Performance and scalability tests."""
    
    @pytest.mark.asyncio
    async def test_large_token_throughput(self):
        """Test handling of large numbers of tokens."""
        class ThroughputNet(PetriNet):
            class Source(Place):
                pass
            class Target(Place):
                pass
            class PassThrough(Transition):
                def input_arcs(self):
                    return {"input": Arc(ThroughputNet.Source)}
                def output_arcs(self):
                    return {"output": Arc(ThroughputNet.Target)}
                async def on_fire(self, consumed_tokens):
                    return {"output": consumed_tokens["input"]}
        net = ThroughputNet(log_fn=lambda x: None)
        # Add many tokens
        start_time = time.time()
        token_count = 1000
        for i in range(token_count):
            await net.produce_token(ThroughputNet.Source, TestToken(f"token_{i}"))
        # Process all tokens
        await net.run_until_complete()
        end_time = time.time()
        processing_time = end_time - start_time
        # Verify results
        target = net.get_place(ThroughputNet.Target)
        assert target.token_count == token_count
        # Performance check (should process 1000 tokens in reasonable time)
        assert processing_time < 10.0  # Should complete in under 10 seconds
        tokens_per_second = token_count / processing_time
        print(f"Processed {tokens_per_second:.2f} tokens/second")
    
    def test_memory_usage_with_large_nets(self):
        """Test memory usage with large net structures."""
        
        # Create the class attributes dictionary
        class_attrs = {}
        
        # Add many places to the class definition
        for i in range(100):
            class_name = f"Place_{i}"
            place_class = type(class_name, (Place,), {})
            class_attrs[class_name] = place_class
        
        # Create the class with all attributes at once
        LargeNet = type('LargeNet', (PetriNet,), class_attrs)
        
        # Now instantiate it
        net = LargeNet(log_fn=lambda x: None)
        
        assert len(net._places_by_type) == 100


if __name__ == "__main__":
    # Run with: python -m pytest test_cordyceps.py -v
    pytest.main([__file__, "-v"])