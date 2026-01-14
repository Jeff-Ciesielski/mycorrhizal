# Your First Spores Logger

This tutorial will walk you through using Spores for event and object logging with OCEL (Object-Centric Event Log) format.

## What You'll Build

You'll create a simple order processing system that logs events and object relationships.

## Step 1: Import Spores

```python
from mycorrhizal.spores import configure, get_spore_sync
from mycorrhizal.spores.transport import SyncFileTransport
from mycorrhizal.spores.models import SporesAttr
from pydantic import BaseModel
from typing import Annotated
```

## Step 2: Define Your Data Models

```python
class Customer(BaseModel):
    id: str
    name: Annotated[str, SporesAttr]
    email: Annotated[str, SporesAttr]

class Order(BaseModel):
    id: str
    customer_id: str
    status: Annotated[str, SporesAttr]
    total: Annotated[float, SporesAttr]
    items: list
```

## Step 3: Configure Logging

```python
# Configure Spores to write to a file
configure(
    transport=SyncFileTransport("logs/orders.jsonl")
)

# Get a logger for this module
spore = get_spore_sync(__name__)
```

## Step 4: Log Events

```python
@spore.log_event(
    event_type="OrderCreated",
    relationships={
        "order": ("return", "Order"),      # Log returned object
        "customer": ("customer", "Customer"),  # Log parameter
    },
    attributes={
        "item_count": lambda items: len(items),
    }
)
def create_order(customer: Customer, items: list) -> Order:
    """Create a new order for a customer."""
    order = Order(
        id=f"ORD-{len(items)}",
        customer_id=customer.id,
        status="pending",
        total=sum(item.get("price", 0) for item in items),
        items=items
    )
    return order

@spore.log_event(
    event_type="OrderShipped",
    relationships={
        "order": ("order", "Order"),
    }
)
def ship_order(order: Order):
    """Ship an order."""
    order.status = "shipped"
    print(f"Shipping order {order.id}")
```

## Step 5: Use Your Logged Functions

```python
# Create a customer
customer = Customer(id="CUST-1", name="Alice", email="alice@example.com")

# Create some orders
order1 = create_order(customer, [
    {"name": "Widget", "price": 10.0},
    {"name": "Gadget", "price": 20.0},
])

order2 = create_order(customer, [
    {"name": "Doohickey", "price": 15.0},
])

# Ship the orders
ship_order(order1)
ship_order(order2)

print(f"Logged events to logs/orders.jsonl")
```

## Understanding the Output

The OCEL log file (`logs/orders.jsonl`) will contain event logs with:
- Event types and timestamps
- Object relationships (which objects participated)
- Object attributes (marked with `SporesAttr`)

## Integration with Visualization

While Spores itself is a logging system (not a control structure), it integrates seamlessly with the other DSLs that **do support Mermaid visualization**:

```python
from mycorrhizal.rhizomorph import bt
from mycorrhizal.spores import spore

# Define your behavior tree
@bt.tree
@spore.object(object_type="OrderProcessor")
class OrderProcessingTree:
    @bt.action
    async def process_order(bb):
        return Status.SUCCESS

# Export the tree structure to Mermaid
tree = OrderProcessingTree()
mermaid = tree.to_mermaid()
print(mermaid)  # Verify structure before running!
```

This means you can:
1. **Visualize your control flow** (FSMs, behavior trees, Petri nets)
2. **Run the system** with automatic Spores logging
3. **Analyze the logs** to understand what happened

Best of both worlds: **static verification + runtime observability**!

## Understanding the Output

The OCEL log file (`logs/orders.jsonl`) contains JSONL records. Each line is a JSON object with both `event` and `object` fields - exactly one is null (union type).

### Example Event Record

```json
{
  "event": {
    "id": "evt-123",
    "type": "OrderCreated",
    "activity": null,
    "time": "2024-01-01T12:00:00.000000000Z",
    "attributes": [
      {"name": "item_count", "value": "2", "type": "integer", "time": null}
    ],
    "relationships": [
      {"objectId": "ORD-2", "qualifier": "order"},
      {"objectId": "CUST-1", "qualifier": "customer"}
    ]
  },
  "object": null
}
```

### Example Object Record

```json
{
  "event": null,
  "object": {
    "id": "ORD-2",
    "type": "Order",
    "attributes": [
      {"name": "status", "value": "pending", "type": "string", "time": null},
      {"name": "total", "value": "30.0", "type": "float", "time": null}
    ],
    "relationships": []
  }
}
```

## Key Concepts

### `SporesAttr` Annotation

Mark fields with `SporesAttr` to include them in object logs:

```python
class Order(BaseModel):
    id: str  # Not logged (no annotation)
    status: Annotated[str, SporesAttr]  # Logged
    total: Annotated[float, SporesAttr]  # Logged
```

### Relationship Tuples

- **2-tuple** `(source, obj_type)`: Auto-detects `SporesAttr` fields
  - `("return", "Order")` - Log the returned object
  - `("customer", "Customer")` - Log a parameter

- **3-tuple** `(source, obj_type, attrs)`: Explicit attribute list
  - `("return", "Order", ["id", "status"])` - Only log specific fields

## Full Example

See the full example in the repository:
```bash
python examples/spores/standalone_spores_demo.py
```

## Next Steps

- Learn about [async logging](../../api/spores.md) with `get_spore_async()`
- Understand [DSL adapters](../guides/observability.md) for Hypha/Rhizomorph integration
- Explore [OCEL format](../guides/observability.md) for event logs
