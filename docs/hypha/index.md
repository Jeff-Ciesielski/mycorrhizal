# Hypha Petri Nets

Hypha is a decorator-based Colored Petri Net DSL for modeling concurrent workflows.

## Overview

Hypha Petri Nets provide:

- **Decorator-based syntax** - Define nets, places, and transitions with decorators
- **Colored tokens** - Rich data objects as tokens, not just markers
- **Multiple place types** - Queues, sets, and boolean flags
- **Async execution** - Full asyncio support for concurrent transitions
- **Modular composition** - Subnets for hierarchical design

## Quick Example

```python
from mycorrhizal.hypha.core import pn, PlaceType, Runner as PNRunner
from pydantic import BaseModel

class WorkItem(BaseModel):
    id: int
    data: str

@pn.net
class ProcessingNet:
    @pn.place(type=PlaceType.QUEUE)
    def input_queue(bb):
        return []

    @pn.place(type=PlaceType.QUEUE)
    def processed(bb):
        return []

    @pn.transition()
    async def process_item(consumed, bb, timebase):
        item = consumed[0].value
        print(f"Processing: {item.data}")
        result = WorkItem(id=item.id, data=item.data.upper())
        return [pn.Token(value=result)]

    @pn.transition()
    async def output_results(consumed, bb, timebase):
        item = consumed[0].value
        print(f"Output: {item.data}")
        return []

# Create and run
net = ProcessingNet()
runner = PNRunner(
    net=net,
    initial_tokens={
        "input_queue": [pn.Token(value=WorkItem(id=1, data="hello"))]
    }
)
await runner.run()
```

## Key Concepts

### Nets

A net is the container for places and transitions:

```python
@pn.net
class MyNet:
    @pn.place(type=PlaceType.QUEUE)
    def my_place(bb):
        return []

    @pn.transition()
    async def my_transition(consumed, bb, timebase):
        return []
```

### Places

Places hold tokens. Three types are available:

```python
# QUEUE - Ordered collection, can have duplicates
@pn.place(type=PlaceType.QUEUE)
def work_queue(bb):
    return []

# SET - Unordered unique items
@pn.place(type=PlaceType.SET)
def unique_items(bb):
    return set()

# BOOLEAN - Simple flag (true if token present)
@pn.place(type=PlaceType.BOOLEAN)
def is_processing(bb):
    return False
```

### Transitions

Transitions consume and produce tokens:

```python
@pn.transition()
async def my_transition(consumed: List[pn.Token], bb: Blackboard, timebase: Timebase):
    """
    Process consumed tokens and return output tokens.

    Args:
        consumed: List of consumed tokens (from input places)
        bb: Shared blackboard
        timebase: Time abstraction

    Returns:
        List of output tokens
    """
    # Process consumed tokens
    input_data = [t.value for t in consumed]

    # Produce output tokens
    return [pn.Token(value=output)]
```

### Arcs

Arcs connect places to transitions:

```python
@pn.arc
def place_to_transition():
    """Connect place to transition."""
    return pn.Arc(
        place=my_place,
        transition=my_transition,
        place_type=PlaceType.QUEUE  # Optional, inferred from place
    )
```

### Tokens

Tokens carry data through the net:

```python
# Create tokens
token = pn.Token(value=my_object)

# Access token value in transitions
@pn.transition()
async def process(consumed, bb, timebase):
    data = consumed[0].value  # Access token's value
    return [pn.Token(value=processed_data)]
```

## Place Types

### QUEUE

FIFO order, allows duplicates:

```python
@pn.place(type=PlaceType.QUEUE)
def task_queue(bb):
    return []

# Tokens processed in insertion order
```

### SET

Unordered, unique items:

```python
@pn.place(type=PlaceType.SET)
def active_users(bb):
    return set()

# Only one token per unique value
```

### BOOLEAN

Simple presence/absence:

```python
@pn.place(type=PlaceType.BOOLEAN)
def system_ready(bb):
    return False

# Presence of token = True, absence = False
```

## Subnets

Compose nets hierarchically:

```python
@pn.net
class ValidationNet:
    """Reusable validation subnet."""
    @pn.place(type=PlaceType.QUEUE)
    def input(bb):
        return []

    @pn.transition()
    async def validate(consumed, bb, timebase):
        # Validation logic
        return [pn.Token(value=validated)]

@pn.net
class MainNet:
    """Main processing net using subnet."""
    validator = ValidationNet()

    @pn.transition()
    async def process(consumed, bb, timebase):
        # Process validated data
        return []
```

## Blackboard Integration

Access and modify shared state:

```python
from pydantic import BaseModel

class NetContext(BaseModel):
    processed_count: int = 0

@pn.net
class CountingNet:
    @pn.transition()
    async def count_and_process(consumed, bb, timebase):
        # Access blackboard
        bb.processed_count += 1
        print(f"Processed {bb.processed_count} items")

        # Process tokens
        return [pn.Token(value=output)]
```

## Examples

- [Petri Net Demo](../../examples/hypha_demo.py) - Basic workflow
- [Blended Demo](../../examples/blended_demo.py) - Petri net + behavior tree

## Documentation

- [API Reference](../api/hypha.md) - Complete API documentation
- [Getting Started](../getting-started/your-first-hypha.md) - Tutorial
- [Composition](../guides/composition.md) - Subnet patterns

## Mermaid Export

### Visualize Before You Run

Hypha enables **static verification of Petri net structure** through Mermaid diagram export:

```python
net = MyNet()
mermaid = net.to_mermaid()
print(mermaid)
```

Paste into [Mermaid Live Editor](https://mermaid.live/) to visualize your workflow.

**Benefits:**
- Verify token flow through places and transitions
- Identify potential deadlocks or unreachable places
- Check that all transitions can fire when tokens are available
- Validate workflow architecture before execution
- Document complex Petri nets automatically

**Catch structural issues in your workflow before running any transitions!**

## Programmatic Net Building

In addition to the decorator-based DSL, Hypha supports **programmatic net construction** using the `NetBuilder` API. This is useful for:

- Building nets from configuration files (JSON, YAML)
- Creating dynamic workflows at runtime
- Generating nets from external definitions
- Building workflow engines and visual editors

See [Programmatic Hypha Building](../guides/programmatic-hypha.md) for complete documentation.

## See Also

- [Programmatic Building](../guides/programmatic-hypha.md) - NetBuilder API guide
- [Rhizomorph](../rhizomorph/) - Behavior Trees for control logic
- [Septum](../septum/) - State Machines for stateful behavior
