# Rhizomorph

Asyncio-friendly **Behavior Trees** for Python, designed for clarity, composability, and ergonomics in asynchronous systems.

- **Fluent decorator chains**: `bt.pipe().failer().gate(N.condition).timeout(0.5)(N.action)`
- **Owner-aware composites**: factories expand with the correct class owner, so large trees can be split across modules
- **Cross-tree composition**: embed full subtrees with `bt.subtree()` or borrow composites with `bt.bind()`
- **Mermaid diagrams**: export static structure diagrams for documentation/debugging

---

## Quick Example

```python
from mycorrhizal.rhizomorph.core import bt, Tree, Runner, Blackboard, Status

class Patrol(Tree):
    @bt.condition
    def has_waypoints(bb: Blackboard) -> bool:
        return bb.get("waypoints", 0) > 0

    @bt.action
    async def go_to_next(bb: Blackboard) -> Status:
        # pretend this takes a few ticks
        remaining = bb.get("waypoints", 0)
        if remaining <= 0:
            return Status.SUCCESS
        bb["waypoints"] = remaining - 1
        return Status.RUNNING if bb["waypoints"] > 0 else Status.SUCCESS

    @bt.sequence(memory=True)
    def Root(N):
        yield N.has_waypoints
        # left→right: timeout(1.0) applied to go_to_next
        yield bt.pipe().timeout(1.0)(N.go_to_next)

    ROOT = Root
```

```python
# runner.py
from mycorrhizal.rhizomorph.core import Runner, Blackboard, Status
from patrol import Patrol

async def main():
    bb = Blackboard()
    bb["waypoints"] = 3
    runner = Runner(Patrol, bb=bb)

    while True:
        status = await runner.tick()
        print("Status:", status.name, "remaining:", bb.get("waypoints"))
        if status in (Status.SUCCESS, Status.FAILURE):
            break
```

---

## Fluent Decorator Chains

All decorators are available via a **fluent chain** off `bt.pipe()` and apply to a child when you call the chain:

```python
yield bt.pipe()    .failer()    .gate(N.battery_ok)    .timeout(0.25)    (N.engage)
```

**Reads left→right:**
- `failer()` → on completion, force FAILURE
- `gate(N.battery_ok)` → only runs if `battery_ok` succeeds (RUNNING bubbles)
- `timeout(0.25)` → abort if `N.engage` exceeds 0.25s
- then apply to `N.engage`

**Available wrappers** (compose in any order):
- `inverter()`
- `succeeder()`
- `failer()`
- `timeout(seconds: float)`
- `retry(max_attempts: int, retry_on: Tuple[Status, ...] = (Status.FAILURE, Status.ERROR))`
- `ratelimit(hz: float | None = None, period: float | None = None)`
- `gate(condition: @bt.condition | NodeSpec)`

> Tip: `.gate(...)` accepts a condition defined on the same or another class. It’s built owner-aware under the hood.

---

## Nodes & Composites

Leaves:
- `@bt.action` — async/sync function returning `Status | bool | None`
- `@bt.condition` — synchronous/async predicate; truthy → `SUCCESS`, falsy → `FAILURE`

Composites (owner-aware, defined as **generator factories**):
- `@bt.sequence(memory: bool = True)`
- `@bt.selector(memory: bool = True, reactive: bool = False)`
- `@bt.parallel(success_threshold: int, failure_threshold: Optional[int] = None)`

Inside a composite factory, use the **owner class** parameter `N` to reference siblings, not strings:

```python
@bt.selector(reactive=True)
def Root(N):
    yield N.EngageThreat
    yield N.Patrol
    yield bt.pipe().ratelimit(hz=5.0)(N.telemetry_push)
```

---

## Cross-Tree Composition

Split large BTs across modules and recombine them in a main tree.

**Structure**

```
behavior/
├── patrol.py        # leaves + composites for patrolling
├── engage.py        # leaves + composites for threat engagement
├── telemetry.py     # telemetry leaves and rate-limited composite
└── main.py          # integrates the modules
```

**patrol.py**

```python
from mycorrhizal.rhizomorph.core import bt, Tree, Blackboard, Status

class Patrol(Tree):
    @bt.condition
    def has_waypoints(bb: Blackboard) -> bool: ...
    @bt.action
    async def go_to_next(bb: Blackboard) -> Status: ...
    @bt.sequence(memory=True)
    def Root(N):
        yield N.has_waypoints
        yield bt.pipe().timeout(1.0)(N.go_to_next)
    ROOT = Root
```

**engage.py**

```python
from mycorrhizal.rhizomorph.core import bt, Tree, Blackboard, Status

class Engage(Tree):
    @bt.condition
    def threat_detected(bb: Blackboard) -> bool: ...
    @bt.action
    async def engage(bb: Blackboard) -> Status: ...
    @bt.sequence(memory=False)
    def Root(N):
        yield N.threat_detected
        yield bt.pipe().failer().timeout(0.25)(N.engage)
    ROOT = Root
```

**telemetry.py**

```python
from mycorrhizal.rhizomorph.core import bt, Tree, Blackboard, Status

class Telemetry(Tree):
    @bt.action
    async def push(bb: Blackboard) -> Status: ...
    @bt.sequence()
    def Root(N):
        yield bt.pipe().ratelimit(hz=2.0)(N.push)
    ROOT = Root
```

**main.py**

```python
from mycorrhizal.rhizomorph.core import bt, Tree
from patrol import Patrol
from engage import Engage
from telemetry import Telemetry

class Main(Tree):
    @bt.selector(reactive=True)
    def Root(N):
        yield bt.subtree(Engage)              # mount full Engage tree
        yield bt.bind(Patrol, Patrol.Root)    # borrow Patrol composite (owner-aware)
        yield bt.bind(Telemetry, Telemetry.Root)
    ROOT = Root
```

- `bt.subtree(OtherTree)` mounts another tree’s `ROOT` as a child.
- `bt.bind(Owner, Owner.Composite)` borrows a composite from another class and expands it with that `Owner` context.

---

## Mermaid Export

Export a static structure diagram (no runtime execution):

```python
from mycorrhizal.rhizomorph.core import to_mermaid, bt
from main import Main

diagram = to_mermaid(bt._as_spec(Main.ROOT), owner_cls=Main)
print(diagram)
```

Paste the output into the [Mermaid Live Editor](https://mermaid-js.github.io/mermaid-live-editor).

---

## Design Principles

- **Async-first** — every node can be `async def`; scheduler is `asyncio`-based.
- **String-free** — nodes are referenced by `N.<member>`, not strings.
- **Composable** — trees split across modules; correct owner expansion is built-in.
- **Fluent-only API** — decorator chains read left→right.

---

## Statuses

- `SUCCESS`, `FAILURE`, `RUNNING`, `CANCELLED`, `ERROR`

`@bt.action` may return a `Status`, `bool`, or `None` (treated as `SUCCESS`).

---

## License

MIT
