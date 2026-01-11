#!/usr/bin/env python3
"""
Spores DSL Adapters

DSL-specific adapters for integrating spores logging with:
- Hypha (Petri nets)
- Rhizomorph (Behavior trees)
- Enoki (State machines)

Usage:
    ```python
    from mycorrhizal.spores.dsl import HyphaAdapter, RhizomorphAdapter, EnokiAdapter

    # For Hypha (Petri nets)
    hypha_adapter = HyphaAdapter()

    @pn.transition()
    @hypha_adapter.log_transition(event_type="process")
    async def process(consumed, bb, timebase):
        yield {output: consumed[0]}

    # For Rhizomorph (Behavior trees)
    rhizo_adapter = RhizomorphAdapter()

    @bt.action
    @rhizo_adapter.log_node(event_type="check")
    async def check(bb: Blackboard) -> Status:
        return Status.SUCCESS

    # For Enoki (State machines)
    enoki_adapter = EnokiAdapter()

    @enoki.on_state
    @enoki_adapter.log_state(event_type="state_execute")
    async def on_state(ctx: SharedContext):
        return Events.DONE
    ```
"""

from .hypha import HyphaAdapter
from .rhizomorph import RhizomorphAdapter
from .enoki import EnokiAdapter

__all__ = [
    'HyphaAdapter',
    'RhizomorphAdapter',
    'EnokiAdapter',
]
