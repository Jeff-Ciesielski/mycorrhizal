#!/usr/bin/env python3
"""
Standalone Spores Demo - ASYNC VERSION

This demo shows how to use Spores for OCEL-compatible logging
in async Python applications using the explicit API.
"""

import asyncio
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

# For standalone use, we import just what we need from spores
sys.path.insert(0, "src")

from mycorrhizal.spores import configure, get_spore_async
from mycorrhizal.spores.transport import AsyncFileTransport
from mycorrhizal.spores.models import SporesAttr
from pydantic import BaseModel
from typing import Annotated


# ============================================================================
# Domain Models with SporesAttr annotations
# ============================================================================

class Customer(BaseModel):
    """Customer in our system - marked fields are logged to OCEL."""
    id: str
    name: Annotated[str, SporesAttr]
    email: Annotated[str, SporesAttr]
    tier: Annotated[str, SporesAttr]


@dataclass
class Order:
    """Order in our system."""
    id: str
    customer_id: str
    items: List[Dict[str, Any]] = field(default_factory=list)
    status: Annotated[str, SporesAttr] = "pending"
    total: Annotated[float, SporesAttr] = 0.0


# ============================================================================
# Business Logic with Proper OCEL Logging
# ============================================================================

# Configure spores with async transport
configure(
    enabled=True,
    transport=AsyncFileTransport("logs/async_ocel.jsonl"),
    object_cache_size=100
)

# Get async spore instance (explicit)
spore = get_spore_async(__name__)


class OrderProcessingSystem:
    """
    Order processing system with event and object logging.

    Relationship formats:
    - 2-tuple ("source", "Type"): Auto-detects all fields marked with SporesAttr
    - 3-tuple ("source", "Type", ["attr1", "attr2"]): Explicitly specify which attributes to log

    Use 2-tuple when you want to log all tagged attributes (e.g., object creation).
    Use 3-tuple when you want specific attributes only (e.g., updates, partial logging).
    """

    @spore.log_event(
        event_type="OrderCreated",
        relationships={
            "order": ("return", "Order"),  # 2-tuple: logs all SporesAttr fields
            "customer": ("customer", "Customer"),  # 2-tuple: logs all SporesAttr fields
        },
        attributes={
            "item_count": lambda items: len(items),
        },
    )
    async def create_order(self, customer: Customer, items: List[dict]) -> Order:
        """Create a new order."""
        order = Order(
            id=f"order_{datetime.now().timestamp()}",
            customer_id=customer.id,
            items=items,
            status="created",
            total=sum(item['price'] * item['quantity'] for item in items)
        )
        return order

    @spore.log_event(
        event_type="PaymentProcessed",
        relationships={
            # 3-tuple: log only specific attributes (status changed during payment)
            "order": ("order", "Order", ["status", "total"]),
        },
        attributes={
            "amount": "amount",
            "success": lambda ret: str(ret),
        },
    )
    async def process_payment(self, order: Order, amount: float) -> bool:
        """Process payment for an order."""
        success = amount >= order.total

        if success:
            order.status = "paid"

        return success

    @spore.log_event(
        event_type="OrderShipped",
        relationships={
            # 3-tuple: log only the status attribute (changed during shipping)
            "order": ("order", "Order", ["status"]),
        },
        attributes={
            "shipping_address": "address",
        },
    )
    async def ship_order(self, order: Order, address: str) -> None:
        """Ship an order."""
        order.status = "shipped"


# ============================================================================
# Demo
# ============================================================================

async def main():
    """Run the order processing demo."""

    print("=" * 70)
    print("Standalone Spores Demo - ASYNC VERSION")
    print("=" * 70)
    print()

    # Create sample data
    customer = Customer(
        id="cust_001",
        name="Alice Smith",
        email="alice@example.com",
        tier="gold"
    )

    items = [
        {"product_id": "prod_001", "name": "Widget", "price": 29.99, "quantity": 2},
        {"product_id": "prod_002", "name": "Gadget", "price": 49.99, "quantity": 1},
    ]

    # Process order using decorated methods
    print("1. Creating order...")
    system = OrderProcessingSystem()
    order = await system.create_order(customer, items)
    print(f"   Order {order.id} created with total ${order.total:.2f}")
    print()

    print("2. Processing payment...")
    await system.process_payment(order, order.total)
    print(f"   Payment of ${order.total:.2f} processed")
    print()

    print("3. Shipping order...")
    await system.ship_order(order, "123 Main St, Anytown, USA")
    print(f"   Order {order.id} shipped")
    print()

    # Wait for async logging to complete
    await asyncio.sleep(0.1)

    print("=" * 70)
    print("Demo complete! Check 'logs/async_ocel.jsonl' for OCEL output")
    print("=" * 70)
    print()


if __name__ == "__main__":
    asyncio.run(main())
