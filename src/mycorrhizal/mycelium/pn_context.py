#!/usr/bin/env python3
"""
PNContext - Context object for BT-in-PN integration.

This module provides the PNContext class that gets injected into the blackboard
during BT execution for BT-in-PN transitions. It provides methods for routing
tokens, rejecting tokens, and accessing PN runtime information.
"""

from typing import Any, List, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from ..hypha.core.runtime import NetRuntime


@dataclass
class PlacesContext:
    """
    Simple namespace for accessing output places by name.

    This provides a clean API like `pn_ctx.places.critical_queue` instead of
    needing to use string keys.
    """
    _place_refs: dict

    def __getattr__(self, name: str) -> Any:
        """Get a place reference by name."""
        if name in self._place_refs:
            return self._place_refs[name]
        raise AttributeError(f"No place named '{name}' available for routing")

    def __contains__(self, name: str) -> bool:
        """Check if a place is available."""
        return name in self._place_refs


class PNContext:
    """
    Context object for BT-in-PN integration.

    This is injected into the blackboard as `pn_ctx` during BT execution
    for BT-in-PN transitions. It provides:

    - Access to current token(s) being processed
    - Access to output places for routing
    - Methods for routing, rejecting, deferring tokens
    - Access to timebase

    Example usage in BT action:
        ```python
        @bt.action
        async def route_critical(bb):
            token = bb.pn_ctx.current_token
            pn_ctx = bb.pn_ctx

            if token.priority == TaskPriority.CRITICAL:
                pn_ctx.route_to(pn_ctx.places.critical_queue)
                return Status.SUCCESS
            return Status.FAILURE
        ```
    """

    def __init__(
        self,
        net_runtime: "NetRuntime",
        output_places: List[Any],
        tokens: List[Any],
        timebase: Any,
        token_origins: Optional[dict] = None,
    ):
        """
        Initialize PNContext.

        Args:
            net_runtime: The PN net runtime instance
            output_places: List of output PlaceRef objects for this transition
            tokens: Tokens being processed (single token for token mode, all for batch mode)
            timebase: The timebase for timing operations
            token_origins: Optional dict mapping tokens to their source place tuples
        """
        self._net_runtime = net_runtime
        self._timebase = timebase

        # Store tokens
        self._tokens = tokens if tokens else []
        self._current_index = 0  # For iterating through tokens in token mode

        # Store token origins for proper rejection handling
        self._token_origins: dict = token_origins if token_origins else {}

        # Build output places mapping
        self._output_place_refs: dict[str, Any] = {}
        for place_ref in output_places:
            # Use the local name from the place reference
            place_name = place_ref.local_name
            self._output_place_refs[place_name] = place_ref

        # Create places namespace
        self.places = PlacesContext(self._output_place_refs)

        # Track routing decisions
        self._routing_decisions: List[tuple[Any, Any]] = []  # (token, place_ref) pairs
        self._rejected_tokens: List[tuple[Any, str]] = []  # (token, reason) pairs
        self._deferred_tokens: List[Any] = []  # tokens to defer

    @property
    def current_token(self) -> Any:
        """
        Get the current token being processed (token mode).

        In token mode, returns the single token for this BT execution.
        In batch mode, returns None (use `tokens` instead).
        """
        if len(self._tokens) == 1:
            return self._tokens[0]
        elif self._tokens and self._current_index < len(self._tokens):
            return self._tokens[self._current_index]
        return None

    @property
    def tokens(self) -> List[Any]:
        """
        Get all tokens being processed (batch mode).

        In batch mode, returns all tokens for this BT execution.
        In token mode, returns a list with one token.
        """
        return self._tokens

    @property
    def timebase(self) -> Any:
        """Get the timebase for timing operations."""
        return self._timebase

    def route_to(self, place: Any, token: Any = None) -> None:
        """
        Route a token to a specific output place.

        Args:
            place: The place reference (from pn_ctx.places.<name>)
            token: The token to route. If None, uses current_token

        Example:
            ```python
            pn_ctx.route_to(pn_ctx.places.critical_queue)
            pn_ctx.route_to(pn_ctx.places.validated, token=enriched_token)
            ```
        """
        if token is None:
            token = self.current_token

        if token is None:
            raise ValueError("No token to route (current_token is None)")

        # Store routing decision for later execution by PN runtime
        self._routing_decisions.append((token, place))

    def route_all_to(self, place: Any) -> None:
        """
        Route all tokens to a specific output place (batch mode).

        Args:
            place: The place reference (from pn_ctx.places.<name>)

        Example:
            ```python
            # Route all high-priority tasks to fast queue
            pn_ctx.route_all_to(pn_ctx.places.fast_queue)
            ```
        """
        for token in self._tokens:
            self._routing_decisions.append((token, place))

    def reject_token(self, reason: str = "", token: Any = None) -> None:
        """
        Reject a token (returns it to input place).

        Args:
            reason: Optional reason for rejection
            token: The token to reject. If None, uses current_token

        Example:
            ```python
            if not token.valid:
                pn_ctx.reject_token("Invalid task data")
                return Status.SUCCESS
            ```
        """
        if token is None:
            token = self.current_token

        if token is None:
            raise ValueError("No token to reject (current_token is None)")

        # Store rejection decision
        self._rejected_tokens.append((token, reason))

    def defer_token(self, token: Any = None) -> None:
        """
        Defer processing of a token (keeps it in input place for later).

        Args:
            token: The token to defer. If None, uses current_token

        Example:
            ```python
            if external_system_not_ready:
                pn_ctx.defer_token()
                return Status.SUCCESS
            ```
        """
        if token is None:
            token = self.current_token

        if token is None:
            raise ValueError("No token to defer (current_token is None)")

        # Store deferral decision
        self._deferred_tokens.append(token)

    def get_routing_decisions(self) -> List[tuple[Any, Any]]:
        """
        Get all routing decisions made by the BT.

        Returns:
            List of (token, place_ref) tuples

        This is used by the PN runtime to execute the routing after BT completes.
        """
        return self._routing_decisions

    def get_rejected_tokens(self) -> List[tuple[Any, str]]:
        """
        Get all rejected tokens with reasons.

        Returns:
            List of (token, reason) tuples

        This is used by the PN runtime to return tokens to input places.
        """
        return self._rejected_tokens

    def get_deferred_tokens(self) -> List[Any]:
        """
        Get all deferred tokens.

        Returns:
            List of tokens to defer (keep in input)

        This is used by the PN runtime to keep tokens in their input places.
        """
        return self._deferred_tokens

    def has_decisions(self) -> bool:
        """Check if any routing decisions were made."""
        return bool(self._routing_decisions or self._rejected_tokens or self._deferred_tokens)

    def get_token_origin(self, token: Any) -> Optional[tuple]:
        """
        Get the origin place tuple for a token.

        Args:
            token: The token to look up

        Returns:
            The place tuple (e.g., ('NetName', 'place_name')) or None if not found

        Note:
            token_origins maps id(token) -> (token, place_parts)
        """
        # Look up by id(token)
        token_info = self._token_origins.get(id(token))
        if token_info:
            return token_info[1]  # Return place_parts
        return None
