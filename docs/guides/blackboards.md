# Blackboards & Interfaces

The blackboard pattern is central to Mycorrhizal, enabling shared state management across all DSLs. This guide covers both basic blackboard usage and advanced interface-based access control.

## Table of Contents

- [Basic Blackboards](#basic-blackboards)
- [Blackboard Interfaces](#blackboard-interfaces)
- [DSL Integration with Interfaces](#dsl-integration-with-interfaces)
- [Best Practices](#best-practices)

---

## Basic Blackboards

### What is a Blackboard?

A **blackboard** is a shared data structure that all components (states, transitions, tree nodes) can read from and write to. In Mycorrhizal, blackboards are implemented as Pydantic `BaseModel` classes.

### Defining a Blackboard

```python
from pydantic import BaseModel
from typing import Optional

class GameContext(BaseModel):
    player_health: int = 100
    enemy_count: int = 0
    current_room: str = "entrance"
    has_key: bool = False
```

### Using in Rhizomorph (Behavior Trees)

```python
from mycorrhizal.rhizomorph.core import bt, Status

@bt.tree
class GameAI:
    @bt.condition
    def is_low_health(bb: GameContext) -> bool:
        return bb.player_health < 30

    @bt.action
    async def heal(bb: GameContext) -> Status:
        bb.player_health = min(100, bb.player_health + 20)
        return Status.SUCCESS
```

### Using in Septum (State Machines)

```python
from mycorrhizal.septum.core import septum, StateMachine

@septum.state
class GameState:
    @septum.on_state
    async def on_state(ctx: GameContext):
        if ctx.player_health <= 0:
            print("Game Over")
        # ...
```

### Using in Hypha (Petri Nets)

```python
from mycorrhizal.hypha.core import pn

@pn.net
class GameNet:
    @pn.place(type=pn.PlaceType.BOOLEAN)
    def player_alive(bb: GameContext):
        return bb.player_health > 0
```

### Type Safety

Pydantic provides automatic type validation:

```python
class GameContext(BaseModel):
    player_health: int = 100  # Must be an int

# This will raise a validation error
try:
    bb = GameContext(player_health="invalid")
except ValidationError as e:
    print(f"Validation error: {e}")
```

### Default Values

Use default values to initialize your blackboard:

```python
class Config(BaseModel):
    max_retries: int = 3
    timeout: float = 5.0
    debug_mode: bool = False
```

### Nested Models

Blackboards can contain nested models for complex state:

```python
class Position(BaseModel):
    x: float
    y: float
    z: float = 0.0

class Robot(BaseModel):
    id: str
    position: Position
    battery: int = 100

class WorldContext(BaseModel):
    robots: dict[str, Robot] = {}
    obstacles: list[Position] = []
```

### Shared Blackboards Across DSLs

You can share the same blackboard between different DSLs:

```python
# Create a shared blackboard
bb = RobotContext()

# Use in a behavior tree
bt_runner = BTRunner(tree=tree, blackboard=bb)

# Use in a Petri net
pn_runner = PNRunner(net=net, blackboard=bb)
```

### Advanced: Computed Properties

Use Pydantic's `computed_field` for derived values:

```python
from pydantic import BaseModel, computed_field

class Character(BaseModel):
    health: int = 100
    max_health: int = 100

    @computed_field
    @property
    def health_percentage(self) -> float:
        return (self.health / self.max_health) * 100
```

---

## Blackboard Interfaces

While basic blackboards work well for simple systems, larger projects benefit from **interfaces** - type-safe, constrained views of blackboard state that enforce access control at both type-level and runtime.

### Why Use Interfaces?

Interfaces provide several key benefits:

1. **Access Control** - Prevent accidental modification of configuration or internal state
2. **Type Safety** - Type checkers can verify only appropriate fields are accessed
3. **Better Testing** - Mock interfaces instead of full blackboards
4. **Clearer Contracts** - Function signatures show exactly what state they need
5. **Safer Composition** - Components can only access relevant fields

### Defining Interfaces

Use the `@blackboard_interface` decorator with `Annotated` types to define interfaces:

```python
from typing import Annotated
from mycorrhizal.common.interface_builder import blackboard_interface, readonly, readwrite

@blackboard_interface
class GameInterface:
    # Readonly configuration - can read but not write
    max_health: Annotated[int, readonly]
    difficulty: Annotated[str, readonly] = "normal"

    # Readwrite state - can both read and write
    current_health: Annotated[int, readwrite]
    player_name: Annotated[str, readwrite]

    # Internal fields (not annotated) are hidden
    # _internal_debug: str  # Would be excluded from interface
```

**Access Control Markers:**

- `Annotated[type, readonly]` - Field can be read but not modified
- `Annotated[type, readwrite]` - Field can be both read and modified
- Unannotated fields - Hidden from the interface
- Private fields (starting with `_`) - Automatically excluded

### Creating Views

Once you have an interface, create a **view** - a runtime wrapper that enforces the interface constraints:

```python
from mycorrhizal.common.wrappers import create_view_from_protocol, AccessControlError

# Your blackboard (Pydantic model)
class GameBlackboard(BaseModel):
    max_health: int = 100
    current_health: int = 100
    player_name: str = "Player"
    difficulty: str = "normal"
    _internal_debug: bool = False

# Create blackboard instance
bb = GameBlackboard()

# Create a view that enforces GameInterface constraints
game_view = create_view_from_protocol(bb, GameInterface, readonly_fields={'max_health', 'difficulty'})

# Reading works for all fields
print(game_view.max_health)       # ✓ OK
print(game_view.current_health)   # ✓ OK

# Writing to readwrite fields works
game_view.current_health = 50     # ✓ OK

# Writing to readonly fields is prevented
try:
    game_view.max_health = 200    # ✗ Raises AccessControlError
except AccessControlError as e:
    print(f"Protected: {e}")

# Accessing internal fields is prevented
try:
    _ = game_view._internal_debug  # ✗ Raises AttributeError
except AttributeError:
    print("Internal field hidden")
```

### Wrapper Classes

Mycorrhizal provides several wrapper classes for different access patterns:

#### ReadOnlyView

Prevents all modifications:

```python
from mycorrhizal.common.wrappers import ReadOnlyView

# View that only allows reading configuration
config_view = ReadOnlyView(bb, {'max_health', 'difficulty'})

print(config_view.max_health)  # ✓ OK
config_view.max_health = 200   # ✗ Raises AccessControlError
```

#### ConstrainedView

Allows selective access with mixed permissions:

```python
from mycorrhizal.common.wrappers import ConstrainedView

# Some fields readonly, some readwrite
view = ConstrainedView(
    bb,
    allowed_fields={'max_health', 'current_health'},
    readonly_fields={'max_health'}
)

view.current_health = 50   # ✓ OK (readwrite)
view.max_health = 200      # ✗ Raises (readonly)
```

#### CompositeView

Combines multiple views:

```python
from mycorrhizal.common.wrappers import CompositeView

# Create multiple views
readonly_view = ReadOnlyView(bb, {'max_health'})
state_view = ConstrainedView(bb, {'current_health'}, readonly_fields=set())

# Combine into single interface
composite = CompositeView.combine([readonly_view, state_view])

print(composite.max_health)       # ✓ From readonly_view
composite.current_health = 50     # ✓ Through state_view
composite.max_health = 200        # ✗ Still readonly
```

#### View Factory

The `View()` function provides a clean, type-safe API:

```python
from mycorrhizal.common.wrappers import View

# Type-safe view creation
view: GameInterface = View(bb, GameInterface)

# Type checker knows view has max_health and current_health
print(view.max_health)
```

### Interface Metadata

Interfaces store metadata for introspection:

```python
@blackboard_interface
class TaskInterface:
    max_tasks: Annotated[int, readonly]
    completed_tasks: Annotated[int, readwrite]
    failed_tasks: Annotated[int, readwrite]

# Check interface metadata
print(TaskInterface._readonly_fields)   # {'max_tasks'}
print(TaskInterface._readwrite_fields)  # {'completed_tasks', 'failed_tasks'}

# Validate that a blackboard implements an interface
# Use isinstance() directly with runtime_checkable protocols
is_valid = isinstance(TaskBlackboard, TaskInterface)
```

---

## DSL Integration with Interfaces

All three DSLs (Hypha, Rhizomorph, Septum) support interfaces through their runners.

### Rhizomorph (Behavior Trees)

Pass a view instead of the full blackboard:

```python
from mycorrhizal.rhizomorph.core import BTRunner, bt, Status
from mycorrhizal.common.wrappers import create_view_from_protocol

# Create blackboard and constrained view
bb = GameBlackboard()
game_view = create_view_from_protocol(bb, GameInterface, {'max_health'})

# Use view in behavior tree
@bt.tree
class GameAI:
    @bt.condition
    def is_low_health(bb: GameInterface) -> bool:  # Type hint shows interface
        return bb.current_health < bb.max_health * 0.3

    @bt.action
    async def heal(bb: GameInterface) -> Status:
        bb.current_health = min(bb.max_health, bb.current_health + 20)
        return Status.SUCCESS

# Create runner with view
runner = BTRunner(tree=GameAI, blackboard=game_view)
```

### Septum (State Machines)

```python
from mycorrhizal.septum.core import StateMachine, septum

# Create view
game_view = create_view_from_protocol(bb, GameInterface, {'max_health'})

@septum.state
class PlayingState:
    @septum.on_state
    async def on_state(ctx: GameInterface):  # Interface type hint
        if ctx.current_health <= 0:
            print("Game Over")
            # ctx.max_health = 200  # ✗ Would be caught by type checker

# Create FSM with view
fsm = StateMachine(initial_state=PlayingState, blackboard=game_view)
```

### Hypha (Petri Nets)

```python
from mycorrhizal.hypha.core import Runner as PNRunner, pn

# Create view
game_view = create_view_from_protocol(bb, GameInterface, {'max_health'})

@pn.net
class GameNet:
    @pn.place(type=pn.PlaceType.BOOLEAN)
    def is_alive(bb: GameInterface):  # Interface type hint
        return bb.current_health > 0

    @pn.transition()
    async def take_damage(consumed, bb: GameInterface, timebase):
        bb.current_health = max(0, bb.current_health - 10)

# Create runner with view
runner = PNRunner(net=GameNet, blackboard=game_view)
```

### Multiple Interfaces per Blackboard

Create different interfaces for different access levels:

```python
@blackboard_interface
class ReadOnlyConfig:
    """Read-only access to configuration"""
    max_health: Annotated[int, readonly]
    difficulty: Annotated[str, readonly]

@blackboard_interface
class GameStateAccess:
    """Full access to game state"""
    max_health: Annotated[int, readonly]
    current_health: Annotated[int, readwrite]
    player_name: Annotated[str, readwrite]

@blackboard_interface
class AdminAccess:
    """Administrative access with internal fields"""
    max_health: Annotated[int, readwrite]  # Can modify config
    current_health: Annotated[int, readwrite]
    _debug_mode: Annotated[bool, readwrite]  # Can access internals

# Use appropriate interface for each component
readonly_view = create_view_from_protocol(bb, ReadOnlyConfig, {'max_health', 'difficulty'})
player_view = create_view_from_protocol(bb, GameStateAccess, {'max_health'})
admin_view = create_view_from_protocol(bb, AdminAccess, set())
```

---

## Best Practices

### General

1. **Use type hints** - Enable better IDE support and validation
2. **Document important fields** - Help other developers understand the state
3. **Avoid overly nested models** - Keep the structure relatively flat
4. **Use default values** - Make initialization easier
5. **Consider immutability** - For complex concurrent systems

### Using Interfaces

1. **Use interfaces for large systems** - Prevent accidental modification
2. **Make configuration read-only** - Use `readonly` for fields that shouldn't change
3. **Hide internal fields** - Don't annotate implementation details
4. **Create focused interfaces** - One interface per component role
5. **Validate at boundaries** - Create views when passing between components
6. **Leverage type checking** - Use interface type hints in function signatures

### Common Patterns

#### Configuration Blackboard

```python
class SystemConfig(BaseModel):
    max_workers: int = 4
    timeout: float = 30.0
    log_level: str = "INFO"

@blackboard_interface
class ConfigView:
    """Read-only configuration access"""
    max_workers: Annotated[int, readonly]
    timeout: Annotated[float, readonly]
    log_level: Annotated[str, readonly]
```

#### State Tracking Blackboard

```python
class ProcessState(BaseModel):
    current_step: str = "init"
    steps_completed: list[str] = []
    error_count: int = 0
    last_error: Optional[str] = None

@blackboard_interface
class StateAccess:
    """Access to process state"""
    current_step: Annotated[str, readwrite]
    steps_completed: Annotated[list[str], readwrite]
    error_count: Annotated[int, readwrite]
    last_error: Annotated[Optional[str], readwrite]
```

#### Resource Management Blackboard

```python
class ResourceManager(BaseModel):
    total_memory: int = 1024
    allocated_memory: int = 0
    active_connections: int = 0

@blackboard_interface
class ResourceConsumer:
    """Consumer view - can allocate but not change total"""
    total_memory: Annotated[int, readonly]
    allocated_memory: Annotated[int, readwrite]
    active_connections: Annotated[int, readwrite]

@blackboard_interface
class ResourceAdmin:
    """Admin view - can modify total resources"""
    total_memory: Annotated[int, readwrite]
    allocated_memory: Annotated[int, readwrite]
    active_connections: Annotated[int, readwrite]
```

## See Also

- [Timebases](timebases.md) - Time abstraction for blackboards
- [Composition](composition.md) - Combining systems with shared state
- [API Reference](../api/common.md) - Complete API documentation for interfaces and wrappers
