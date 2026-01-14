# Spores Logging

Spores provides structured event and object logging in OCEL (Object-Centric Event Log) format.

## Overview

Spores Logging provides:

- **OCEL-compliant logs** - Standard Object-Centric Event Log format
- **Automatic relationship tracking** - E2O (Event-to-Object) relationships
- **Attribute logging** - Mark fields with `SporesAttr` for automatic extraction
- **DSL adapters** - Seamless integration with Hypha, Rhizomorph, and Enoki
- **Sync and async** - Both synchronous and asynchronous logging support

## Quick Example

```python
from mycorrhizal.spores import configure, get_spore_sync
from mycorrhizal.spores.transport import SyncFileTransport
from mycorrhizal.spores.models import SporesAttr
from pydantic import BaseModel
from typing import Annotated

# Configure logging
configure(transport=SyncFileTransport("logs/orders.jsonl"))
spore = get_spore_sync(__name__)

# Mark fields for logging
class Customer(BaseModel):
    id: str
    name: Annotated[str, SporesAttr]
    email: Annotated[str, SporesAttr]

class Order(BaseModel):
    id: str
    status: Annotated[str, SporesAttr]
    total: Annotated[float, SporesAttr]

# Log events with relationships
@spore.log_event(
    event_type="OrderCreated",
    relationships={
        "order": ("return", "Order"),           # Log returned object
        "customer": ("customer", "Customer"),   # Log parameter
    },
    attributes={
        "item_count": lambda items: len(items),
    }
)
def create_order(customer: Customer, items: list) -> Order:
    order = Order(
        id=f"ORD-{len(items)}",
        status="pending",
        total=sum(item.get("price", 0) for item in items)
    )
    return order

# Use your function - logging is automatic
customer = Customer(id="CUST-1", name="Alice", email="alice@example.com")
order = create_order(customer, [{"name": "Widget", "price": 10.0}])
# Event logged to: logs/orders.jsonl
```

## Two Usage Patterns

### Pattern 1: Standalone Logging

For regular Python code, use `get_spore_sync()` or `get_spore_async()`:

```python
from mycorrhizal.spores import configure, get_spore_sync
from mycorrhizal.spores.models import SporesAttr
from typing import Annotated

# Configure
configure(transport=SyncFileTransport("logs/events.jsonl"))
spore = get_spore_sync(__name__)

# Mark domain model fields
class Order(BaseModel):
    id: str
    status: Annotated[str, SporesAttr]  # Mark for logging
    total: Annotated[float, SporesAttr]

# Log events
@spore.log_event(
    event_type="OrderCreated",
    relationships={"order": ("return", "Order")},
)
def create_order() -> Order:
    return Order(id="1", status="pending", total=100.0)
```

### Pattern 2: DSL Adapters

For Hypha/Rhizomorph/Enoki integration:

```python
from mycorrhizal.spores import spore
from mycorrhizal.spores.models import EventAttr, ObjectRef, ObjectScope
from typing import Annotated

# Mark blackboard fields
class MissionContext(BaseModel):
    mission_id: Annotated[str, EventAttr]  # Event attribute
    robot: Annotated[Robot, ObjectRef(qualifier="actor", scope=ObjectScope.GLOBAL)]

# Mark object types for logging
@spore.object(object_type="Robot")
class Robot(BaseModel):
    id: str
    name: str

# Use with behavior tree
@bt.tree
@spore.object(object_type="MissionControl")
class MissionAI:
    # All node executions logged automatically
    @bt.action
    async def execute_mission(bb: MissionContext) -> Status:
        return Status.SUCCESS
```

## Annotation Types

### SporesAttr

Mark domain model fields for object attribute logging:

```python
from mycorrhizal.spores.models import SporesAttr
from typing import Annotated

class Order(BaseModel):
    id: str  # Not logged
    status: Annotated[str, SporesAttr]  # Logged
    total: Annotated[float, SporesAttr]  # Logged
```

Used with standalone logging (`get_spore_sync/async`).

### EventAttr

Mark blackboard fields for event attribute extraction:

```python
from mycorrhizal.spores.models import EventAttr
from typing import Annotated

class MissionContext(BaseModel):
    mission_id: Annotated[str, EventAttr]  # Included in event attributes
    timestamp: Annotated[float, EventAttr]
```

Used with DSL adapters (`spore` with `@spore.object()`).

### ObjectRef

Mark blackboard fields as object references with scope:

```python
from mycorrhizal.spores.models import ObjectRef, ObjectScope
from typing import Annotated

class MissionContext(BaseModel):
    robot: Annotated[Robot, ObjectRef(
        qualifier="actor",      # Relationship qualifier
        scope=ObjectScope.GLOBAL  # OCEL object scope
    )]
```

## Relationships

Specify event-to-object relationships:

### 2-Tuple Format (Auto-detect)

Auto-detects `SporesAttr` fields:

```python
relationships={
    "order": ("return", "Order"),      # Log returned object
    "customer": ("customer", "Customer"),  # Log parameter
}
```

### 3-Tuple Format (Explicit)

Explicit attribute list:

```python
relationships={
    "order": ("return", "Order", ["id", "status"]),
}
```

### Source Values

- `"return"` or `"ret"` - Return value of function
- `"self"` - The object instance (for methods)
- Parameter name - Any function parameter

## Transport Options

### File Transport (Sync)

```python
from mycorrhizal.spores.transport import SyncFileTransport

configure(transport=SyncFileTransport("logs/events.jsonl"))
```

### File Transport (Async)

```python
from mycorrhizal.spores.transport import AsyncFileTransport

configure(transport=AsyncFileTransport("logs/events.jsonl"))
```

### Custom Transport

Implement your own transport:

```python
from mycorrhizal.spores.transport import Transport

class CustomTransport(Transport):
    def emit(self, event: OCELEvent):
        # Send to your logging system
        my_logger.log(event)

configure(transport=CustomTransport())
```

## Async Logging

```python
from mycorrhizal.spores import get_spore_async

spore = get_spore_async(__name__)

@spore.log_event(
    event_type="AsyncEvent",
    relationships={"data": ("return", "Data")},
)
async def async_function() -> Data:
    result = await fetch_data()
    return result
```

## OCEL Format

Spores outputs OCEL (Object-Centric Event Log) JSONL format. Each line is a JSON record with both `event` and `object` fields - exactly one is null (union type).

### Event Record

```json
{
  "event": {
    "id": "evt-123",
    "type": "OrderCreated",
    "activity": null,
    "time": "2024-01-01T12:00:00.000000000Z",
    "attributes": [
      {"name": "item_count", "value": "3", "type": "integer", "time": null}
    ],
    "relationships": [
      {"objectId": "ORD-1", "qualifier": "order"},
      {"objectId": "CUST-1", "qualifier": "customer"}
    ]
  },
  "object": null
}
```

### Object Record

```json
{
  "event": null,
  "object": {
    "id": "ORD-1",
    "type": "Order",
    "attributes": [
      {"name": "status", "value": "pending", "type": "string", "time": null},
      {"name": "total", "value": "100.0", "type": "float", "time": null}
    ],
    "relationships": []
  }
}
```

### Attribute Types

Spores automatically infers types for attributes:

- `"string"` - Text values
- `"integer"` - Whole numbers
- `"float"` - Floating point numbers
- `"boolean"` - true/false values (lowercase)
- `"timestamp"` - ISO 8601 datetime strings

### Event Activity vs Type

- `type` - Event class/category (structural)
- `activity` - Semantic activity label (optional, defaults to type if null)

## Examples

- [Standalone Demo](../../examples/spores/standalone_spores_demo.py) - Standalone logging
- [Sync Demo](../../examples/spores/standalone_sync_demo.py) - Synchronous example

## Documentation

- [API Reference](../api/spores.md) - Complete API documentation
- [Getting Started](../getting-started/your-first-spores.md) - Tutorial
- [Observability](../guides/observability.md) - Monitoring and debugging

## Best Practices

1. **Use SporesAttr for domain models** - Mark fields you want logged
2. **Use EventAttr for blackboards** - Extract event attributes from DSL state
3. **Log at boundaries** - Log events at system boundaries (APIs, services)
4. **Use explicit relationships** - Make object relationships clear
5. **Avoid over-logging** - Don't log everything, focus on important events

## See Also

- [Observability Guide](../guides/observability.md) - Monitoring and debugging
- [Blackboards](../guides/blackboards.md) - State management
- [Enoki](../enoki/) | [Hypha](../hypha/) | [Rhizomorph](../rhizomorph/) - DSL integration
