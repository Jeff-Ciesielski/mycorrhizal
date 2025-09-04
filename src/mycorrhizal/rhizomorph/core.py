#!/usr/bin/env python3
"""
Rhizomorph — asyncio Behavior Tree core (owner-aware + fluent wrappers)
Multi-file composition: owner tagging + bt.subtree/bt.bind
"""
from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union


# ======================================================================================
# Status / helpers
# ======================================================================================

class Status(Enum):
    SUCCESS = 1
    FAILURE = 2
    RUNNING = 3
    CANCELLED = 4
    ERROR = 5


def _name_of(obj: Any) -> str:
    if hasattr(obj, "__name__"):
        return obj.__name__
    if hasattr(obj, "name"):
        return str(obj.name)
    return f"{obj.__class__.__name__}@{id(obj):x}"


# ======================================================================================
# Blackboard
# ======================================================================================

class Blackboard(dict):
    """Dict-like blackboard with async helpers and a virtual-clock seam."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        if key.startswith("_"):
            return super().__setattr__(key, value)
        self[key] = value

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)

    def now(self) -> float:
        return time.monotonic()


# ======================================================================================
# Node model
# ======================================================================================

class Node:
    """Abstract BT node; override `tick` and (optionally) lifecycle hooks."""

    def __init__(self, name: Optional[str] = None) -> None:
        self.name: str = name or _name_of(self)
        self.parent: Optional[Node] = None
        self._entered: bool = False
        self._last_status: Optional[Status] = None

    async def tick(self, bb: Blackboard) -> Status:
        raise NotImplementedError

    async def on_enter(self, bb: Blackboard) -> None:
        return

    async def on_exit(self, bb: Blackboard, status: Status) -> None:
        return

    def reset(self) -> None:
        self._entered = False
        self._last_status = None

    async def _ensure_entered(self, bb: Blackboard) -> None:
        if not self._entered:
            await self.on_enter(bb)
            self._entered = True

    async def _finish(self, bb: Blackboard, status: Status) -> Status:
        self._last_status = status
        if status is not Status.RUNNING:
            try:
                await self.on_exit(bb, status)
            finally:
                self._entered = False
        return status


# ======================================================================================
# Leaves
# ======================================================================================

class Action(Node):
    """Leaf wrapping a sync/async function -> Status | bool | None."""

    def __init__(self, func: Callable[..., Any]) -> None:
        super().__init__(name=_name_of(func))
        self._func = func

    async def tick(self, bb: Blackboard) -> Status:
        await self._ensure_entered(bb)
        try:
            result = await self._func(bb) if inspect.iscoroutinefunction(self._func) else self._func(bb)
        except asyncio.CancelledError:
            return await self._finish(bb, Status.CANCELLED)
        except Exception:
            return await self._finish(bb, Status.ERROR)

        if isinstance(result, Status):
            return await self._finish(bb, result)
        if isinstance(result, bool):
            return await self._finish(bb, Status.SUCCESS if result else Status.FAILURE)
        return await self._finish(bb, Status.SUCCESS)


class Condition(Action):
    """Boolean leaf: True→SUCCESS, False→FAILURE (Status accepted but discouraged)."""

    async def tick(self, bb: Blackboard) -> Status:
        await self._ensure_entered(bb)
        try:
            result = await self._func(bb) if inspect.iscoroutinefunction(self._func) else self._func(bb)
        except asyncio.CancelledError:
            return await self._finish(bb, Status.CANCELLED)
        except Exception:
            return await self._finish(bb, Status.ERROR)

        if isinstance(result, Status):
            return await self._finish(bb, result)
        return await self._finish(bb, Status.SUCCESS if bool(result) else Status.FAILURE)


# ======================================================================================
# Composites
# ======================================================================================

class Sequence(Node):
    """Sequence (AND): fail/err fast; RUNNING bubbles; all SUCCESS → SUCCESS."""

    def __init__(self, children: Sequence[Node], *, memory: bool = True, name: Optional[str] = None) -> None:
        super().__init__(name or "Sequence")
        self.children = list(children)
        for ch in self.children:
            ch.parent = self
        self.memory = memory
        self._idx = 0

    def reset(self) -> None:
        super().reset()
        self._idx = 0
        for ch in self.children:
            ch.reset()

    async def tick(self, bb: Blackboard) -> Status:
        await self._ensure_entered(bb)
        start = self._idx if self.memory else 0
        for i in range(start, len(self.children)):
            st = await self.children[i].tick(bb)
            if st is Status.RUNNING:
                if self.memory:
                    self._idx = i
                return Status.RUNNING
            if st in (Status.FAILURE, Status.ERROR, Status.CANCELLED):
                self._idx = 0
                return await self._finish(bb, st)
        self._idx = 0
        return await self._finish(bb, Status.SUCCESS)


class Selector(Node):
    """Selector (Fallback): first SUCCESS wins; RUNNING bubbles; else FAILURE."""

    def __init__(self, children: Sequence[Node], *, memory: bool = True, reactive: bool = False,
                 name: Optional[str] = None) -> None:
        super().__init__(name or "Selector")
        self.children = list(children)
        for ch in self.children:
            ch.parent = self
        self.memory = memory
        self.reactive = reactive
        self._idx = 0

    def reset(self) -> None:
        super().reset()
        self._idx = 0
        for ch in self.children:
            ch.reset()

    async def tick(self, bb: Blackboard) -> Status:
        await self._ensure_entered(bb)
        start = 0 if self.reactive else (self._idx if self.memory else 0)
        for i in range(start, len(self.children)):
            st = await self.children[i].tick(bb)
            if st is Status.SUCCESS:
                self._idx = 0
                return await self._finish(bb, Status.SUCCESS)
            if st is Status.RUNNING:
                if self.memory and not self.reactive:
                    self._idx = i
                return Status.RUNNING
        self._idx = 0
        return await self._finish(bb, Status.FAILURE)


class Parallel(Node):
    """
    Parallel: tick all children per tick concurrently.
    - success_threshold (k) SUCCESS to report SUCCESS
    - failure_threshold defaults to n-k+1 (cannot reach success anymore) to report FAILURE
    Otherwise RUNNING.
    """

    def __init__(self, children: Sequence[Node], *, success_threshold: int,
                 failure_threshold: Optional[int] = None, name: Optional[str] = None) -> None:
        super().__init__(name or "Parallel")
        self.children = list(children)
        for ch in self.children:
            ch.parent = self
        self.n = len(self.children)
        self.k = max(1, min(success_threshold, self.n))
        self.f = failure_threshold if failure_threshold is not None else (self.n - self.k + 1)

    def reset(self) -> None:
        super().reset()
        for ch in self.children:
            ch.reset()

    async def tick(self, bb: Blackboard) -> Status:
        await self._ensure_entered(bb)
        tasks = [asyncio.create_task(ch.tick(bb)) for ch in self.children]
        try:
            results = await asyncio.gather(*tasks, return_exceptions=False)
        finally:
            pass
        succ = sum(1 for s in results if s is Status.SUCCESS)
        fail = sum(1 for s in results if s in (Status.FAILURE, Status.ERROR, Status.CANCELLED))
        if succ >= self.k:
            return await self._finish(bb, Status.SUCCESS)
        if fail >= self.f:
            return await self._finish(bb, Status.FAILURE)
        return Status.RUNNING


# ======================================================================================
# Decorators (wrappers)
# ======================================================================================

class Inverter(Node):
    def __init__(self, child: Node, name: Optional[str] = None) -> None:
        super().__init__(name or f"Inverter({_name_of(child)})")
        self.child = child
        self.child.parent = self

    def reset(self) -> None:
        super().reset()
        self.child.reset()

    async def tick(self, bb: Blackboard) -> Status:
        await self._ensure_entered(bb)
        st = await self.child.tick(bb)
        if st is Status.SUCCESS:
            return await self._finish(bb, Status.FAILURE)
        if st is Status.FAILURE:
            return await self._finish(bb, Status.SUCCESS)
        return st


class Retry(Node):
    def __init__(self, child: Node, *, max_attempts: int,
                 retry_on: Tuple[Status, ...] = (Status.FAILURE, Status.ERROR),
                 name: Optional[str] = None) -> None:
        super().__init__(name or f"Retry({_name_of(child)},{max_attempts})")
        self.child = child
        self.child.parent = self
        self.max_attempts = max(1, int(max_attempts))
        self.retry_on = retry_on
        self._attempt = 0

    def reset(self) -> None:
        super().reset()
        self._attempt = 0
        self.child.reset()

    async def tick(self, bb: Blackboard) -> Status:
        await self._ensure_entered(bb)
        st = await self.child.tick(bb)
        if st is Status.RUNNING:
            return Status.RUNNING
        if st is Status.SUCCESS:
            self._attempt = 0
            return await self._finish(bb, Status.SUCCESS)
        # failure-like
        self._attempt += 1
        if st in self.retry_on and self._attempt < self.max_attempts:
            self.child.reset()
            return Status.RUNNING
        self._attempt = 0
        return await self._finish(bb, Status.FAILURE)


class Timeout(Node):
    def __init__(self, child: Node, *, seconds: float, name: Optional[str] = None) -> None:
        super().__init__(name or f"Timeout({_name_of(child)},{seconds}s)")
        self.child = child
        self.child.parent = self
        self.seconds = max(0.0, float(seconds))
        self._deadline: Optional[float] = None

    def reset(self) -> None:
        super().reset()
        self._deadline = None
        self.child.reset()

    async def on_enter(self, bb: Blackboard) -> None:
        self._deadline = bb.now() + self.seconds

    async def tick(self, bb: Blackboard) -> Status:
        await self._ensure_entered(bb)
        if self._deadline is None:
            self._deadline = bb.now() + self.seconds
        if bb.now() > self._deadline:
            self.child.reset()
            return await self._finish(bb, Status.FAILURE)
        st = await self.child.tick(bb)
        if st is Status.RUNNING:
            return Status.RUNNING
        return await self._finish(bb, st)


class Succeeder(Node):
    def __init__(self, child: Node, name: Optional[str] = None) -> None:
        super().__init__(name or f"Succeeder({_name_of(child)})")
        self.child = child
        self.child.parent = self

    def reset(self) -> None:
        super().reset()
        self.child.reset()

    async def tick(self, bb: Blackboard) -> Status:
        await self._ensure_entered(bb)
        st = await self.child.tick(bb)
        if st is Status.RUNNING:
            return Status.RUNNING
        return await self._finish(bb, Status.SUCCESS)


class Failer(Node):
    def __init__(self, child: Node, name: Optional[str] = None) -> None:
        super().__init__(name or f"Failer({_name_of(child)})")
        self.child = child
        self.child.parent = self

    def reset(self) -> None:
        super().reset()
        self.child.reset()

    async def tick(self, bb: Blackboard) -> Status:
        await self._ensure_entered(bb)
        st = await self.child.tick(bb)
        if st is Status.RUNNING:
            return Status.RUNNING
        return await self._finish(bb, Status.FAILURE)


class RateLimit(Node):
    """
    Throttle *starting* the child to at most 1 per period (or hz).
    If child is RUNNING, do not throttle (avoid starvation).
    """
    def __init__(self, child: Node, *, hz: Optional[float] = None,
                 period: Optional[float] = None, name: Optional[str] = None) -> None:
        if (hz is None) == (period is None):
            raise ValueError("RateLimit requires exactly one of (hz, period)")
        per = (1.0 / float(hz)) if hz is not None else float(period)
        super().__init__(name or f"RateLimit({_name_of(child)},{per:.6f}s)")
        self.child = child
        self.child.parent = self
        self._period = max(0.0, per)
        self._next_allowed: Optional[float] = None
        self._last: Optional[Status] = None

    def reset(self) -> None:
        super().reset()
        self._next_allowed = None
        self._last = None
        self.child.reset()

    async def tick(self, bb: Blackboard) -> Status:
        await self._ensure_entered(bb)
        if self._last is Status.RUNNING:
            st = await self.child.tick(bb)
            self._last = st
            if st is Status.RUNNING:
                return Status.RUNNING
            self._next_allowed = bb.now() + self._period
            return await self._finish(bb, st)

        now = bb.now()
        if self._next_allowed is not None and now < self._next_allowed:
            return Status.RUNNING  # throttled
        st = await self.child.tick(bb)
        self._last = st
        if st is Status.RUNNING:
            return Status.RUNNING
        self._next_allowed = now + self._period
        return await self._finish(bb, st)


class Gate(Node):
    """
    Guard a child with a condition:
      SUCCESS → tick child
      RUNNING → RUNNING
      else    → FAILURE
    """
    def __init__(self, condition: Node, child: Node, name: Optional[str] = None) -> None:
        super().__init__(name or f"Gate(cond={_name_of(condition)}, child={_name_of(child)})")
        self.condition = condition
        self.child = child
        self.condition.parent = self
        self.child.parent = self

    def reset(self) -> None:
        super().reset()
        self.condition.reset()
        self.child.reset()

    async def tick(self, bb: Blackboard) -> Status:
        await self._ensure_entered(bb)
        c = await self.condition.tick(bb)
        if c is Status.RUNNING:
            return Status.RUNNING
        if c is not Status.SUCCESS:
            return await self._finish(bb, Status.FAILURE)
        st = await self.child.tick(bb)
        if st is Status.RUNNING:
            return Status.RUNNING
        return await self._finish(bb, st)


# ======================================================================================
# Authoring DSL (NodeSpec + bt namespace)
# ======================================================================================

@dataclass
class NodeSpec:
    kind: str  # 'action'|'condition'|'sequence'|'selector'|'parallel'|'decorator'|'subtree'|'bind'
    name: str
    payload: Any = None     # leaf func / composite config (incl. factory) / decorator builder
    children: List["NodeSpec"] = field(default_factory=list)
    
    def __hash__(self):
        return hash((self.kind, self.name, id(self.payload), tuple(self.children)))

    def to_node(self, owner_cls: Optional[type] = None) -> Node:
        if self.kind == "action":
            return Action(self.payload)
        if self.kind == "condition":
            return Condition(self.payload)

        if self.kind in ("sequence", "selector", "parallel"):
            owner_effective = getattr(self, "_owner_cls", None) or owner_cls
            factory = self.payload["factory"]
            expanded = _bt_expand_children_with_owner(factory, owner_effective)
            self.children = expanded  # cache for exporter
            built = [ch.to_node(owner_effective) for ch in expanded]
            if self.kind == "sequence":
                return Sequence(built, memory=self.payload.get("memory", True), name=self.name)
            if self.kind == "selector":
                return Selector(built,
                                memory=self.payload.get("memory", True),
                                reactive=self.payload.get("reactive", False),
                                name=self.name)
            return Parallel(built,
                            success_threshold=self.payload["success_threshold"],
                            failure_threshold=self.payload.get("failure_threshold"),
                            name=self.name)

        if self.kind == "decorator":
            assert len(self.children) == 1, "Decorator must wrap exactly one child"
            child_node = self.children[0].to_node(owner_cls)
            builder = self.payload
            try:
                if len(inspect.signature(builder).parameters) == 2:
                    return builder(child_node, owner_cls)
            except (TypeError, ValueError):
                pass
            return builder(child_node)

        if self.kind == "subtree":
            tree_cls = self.payload["tree_cls"]
            return tree_cls.build()

        if self.kind == "bind":
            base = self.payload["spec"]
            owner = self.payload["owner"]
            return base.to_node(owner_cls=owner)

        raise ValueError(f"Unknown spec kind: {self.kind}")


def _bt_expand_children_with_owner(factory: Callable[..., Generator[Any, None, None]],
                                   owner_cls: Optional[type]) -> List[NodeSpec]:
    """
    Execute a composite factory *after* class construction.
    Convention: factory accepts one positional arg 'N' = owner class.
    Users should write: def Root(N): yield N.Child, yield N.leaf, ...
    """
    try:
        gen = factory(owner_cls)
    except TypeError:
        gen = factory()  # allow zero-arg factories too

    if not inspect.isgenerator(gen):
        raise TypeError(f"Composite factory {factory.__name__} must be a generator (use 'yield').")

    out: List[NodeSpec] = []
    for yielded in gen:
        if isinstance(yielded, (list, tuple)):
            for y in yielded:
                out.append(bt._as_spec(y))
            continue
        out.append(bt._as_spec(yielded))
    return out


# --------------------------------------------------------------------------------------
# Fluent wrapper chain (left-to-right)
# --------------------------------------------------------------------------------------

class _WrapperChain:
    """
    Fluent factory for decorator stacks that read left→right.

        chain = bt.pipe().failer().gate(N.battery_ok).timeout(0.12)
        yield chain(N.engage)
    """

    def __init__(self, builders: Optional[List[Callable[..., Any]]] = None,
                 labels: Optional[List[str]] = None) -> None:
        self._builders: List[Callable[..., Any]] = list(builders or [])
        self._labels: List[str] = list(labels or [])

    def _append(self, label: str, builder: Callable[..., Any]) -> "_WrapperChain":
        self._builders.append(builder)
        self._labels.append(label)
        return self

    # chainable steps (builders accept (child[, owner]))
    def failer(self) -> "_WrapperChain":
        return self._append("Failer", lambda ch, owner=None: Failer(ch))

    def succeeder(self) -> "_WrapperChain":
        return self._append("Succeeder", lambda ch, owner=None: Succeeder(ch))

    def inverter(self) -> "_WrapperChain":
        return self._append("Inverter", lambda ch, owner=None: Inverter(ch))

    def timeout(self, seconds: float) -> "_WrapperChain":
        return self._append(f"Timeout({seconds}s)",
                            lambda ch, owner=None: Timeout(ch, seconds=seconds))

    def retry(self, max_attempts: int,
              retry_on: Tuple[Status, ...] = (Status.FAILURE, Status.ERROR)) -> "_WrapperChain":
        return self._append(f"Retry({max_attempts})",
                            lambda ch, owner=None: Retry(ch, max_attempts=max_attempts, retry_on=retry_on))

    def ratelimit(self, *, hz: Optional[float] = None, period: Optional[float] = None) -> "_WrapperChain":
        label = "RateLimit(?)"
        if hz is not None:
            label = f"RateLimit({1.0/float(hz):.6f}s)"
        elif period is not None:
            label = f"RateLimit({float(period):.6f}s)"
        return self._append(label, lambda ch, owner=None: RateLimit(ch, hz=hz, period=period))

    def gate(self, condition_spec_or_fn: Union["NodeSpec", Callable[..., Any]]) -> "_WrapperChain":
        cond_spec = bt._as_spec(condition_spec_or_fn)
        return self._append(f"Gate(cond={_name_of(cond_spec)})",
                            lambda ch, owner=None: Gate(cond_spec.to_node(owner_cls=owner), ch))

    def __call__(self, inner: Union["NodeSpec", Callable[..., Any]]) -> "NodeSpec":
        """
        Apply the chain to a child spec → nested decorator NodeSpecs.
        Left→right call order becomes outermost→…→innermost when built.
        """
        spec = bt._as_spec(inner)
        result = spec
        for label, builder in reversed(list(zip(self._labels, self._builders))):
            result = NodeSpec(
                kind="decorator",
                name=f"{label}({_name_of(result)})",
                payload=builder,      # builder(child[, owner])
                children=[result],
            )
        return result

    def __rshift__(self, inner: Union["NodeSpec", Callable[..., Any]]) -> "NodeSpec":
        return self(inner)


# --------------------------------------------------------------------------------------
# User-facing decorator/constructor namespace
# --------------------------------------------------------------------------------------

class _BT:
    """User-facing decorator/constructor namespace."""

    # Leaves
    def action(self, fn: Callable[..., Any]) -> NodeSpec:
        spec = NodeSpec(kind="action", name=_name_of(fn), payload=fn)
        setattr(fn, "_node_spec", spec)
        return spec

    def condition(self, fn: Callable[..., Any]) -> NodeSpec:
        spec = NodeSpec(kind="condition", name=_name_of(fn), payload=fn)
        setattr(fn, "_node_spec", spec)
        return spec

    # Composites (store factory; expand later with owner)
    def sequence(self, *, memory: bool = True):
        def deco(factory: Callable[..., Generator[Any, None, None]]) -> NodeSpec:
            spec = NodeSpec(kind="sequence", name=_name_of(factory),
                            payload={"factory": factory, "memory": memory})
            setattr(factory, "_node_spec", spec)
            return spec
        return deco

    def selector(self, *, memory: bool = True, reactive: bool = False):
        def deco(factory: Callable[..., Generator[Any, None, None]]) -> NodeSpec:
            spec = NodeSpec(kind="selector", name=_name_of(factory),
                            payload={"factory": factory, "memory": memory, "reactive": reactive})
            setattr(factory, "_node_spec", spec)
            return spec
        return deco

    def parallel(self, *, success_threshold: int, failure_threshold: Optional[int] = None):
        def deco(factory: Callable[..., Generator[Any, None, None]]) -> NodeSpec:
            spec = NodeSpec(kind="parallel", name=_name_of(factory),
                            payload={"factory": factory, "success_threshold": success_threshold,
                                     "failure_threshold": failure_threshold})
            setattr(factory, "_node_spec", spec)
            return spec
        return deco

    def inverter(self) -> _WrapperChain:
        return _WrapperChain().inverter()

    def succeeder(self) -> _WrapperChain:
        return _WrapperChain().succeeder()

    def failer(self) -> _WrapperChain:
        return _WrapperChain().failer()

    def timeout(self, seconds: float) -> _WrapperChain:
        return _WrapperChain().timeout(seconds)

    def retry(self, max_attempts: int, retry_on: Tuple[Status, ...] = (Status.FAILURE, Status.ERROR),
              ) -> _WrapperChain:
        return _WrapperChain().retry(max_attempts, retry_on=retry_on)

    def ratelimit(self, hz: Optional[float] = None, period: Optional[float] = None) -> _WrapperChain:
        return _WrapperChain().ratelimit(hz=hz, period=period)
       
    def gate(self, condition: Union[NodeSpec, Callable[..., Any]]) -> _WrapperChain:
        cond_spec = self._as_spec(condition)
        return _WrapperChain().gate(cond_spec)

    # ---- cross-file composition helpers ----
    def subtree(self, tree_cls: type["Tree"]) -> NodeSpec:
        """
        Mount another Tree's root as a child in the current tree.
        For Mermaid, attach the other tree's ROOT spec so it can expand visually.
        """
        root_spec = self._as_spec(tree_cls.ROOT)
        spec = NodeSpec(kind="subtree", name=f"Subtree({tree_cls.__name__})",
                        payload={"tree_cls": tree_cls}, children=[root_spec])
        # Ensure the child knows its owner for expansion
        setattr(root_spec, "_owner_cls", getattr(root_spec, "_owner_cls", None) or tree_cls)
        return spec

    def bind(self, owner: type, spec_or_fn: Union[NodeSpec, Callable[..., Any]]) -> NodeSpec:
        """
        Borrow a composite/spec defined on 'owner' and use it here safely.
        For Mermaid, attach the borrowed spec as a child.
        """
        base = self._as_spec(spec_or_fn)
        # Ensure the borrowed spec carries its original owner for expansion
        setattr(base, "_owner_cls", getattr(base, "_owner_cls", None) or owner)
        return NodeSpec(kind="bind", name=f"Bind({owner.__name__}:{_name_of(base)})",
                        payload={"owner": owner, "spec": base}, children=[base])

    # Internals
    def _as_spec(self, maybe: Union[NodeSpec, Callable[..., Any]]) -> NodeSpec:
        if isinstance(maybe, NodeSpec):
            return maybe
        spec = getattr(maybe, "_node_spec", None)
        if spec is None:
            raise TypeError(f"{maybe!r} is not a BT node (missing @_node_spec).")
        return spec


bt = _BT()


# ======================================================================================
# Tree container & Runner
# ======================================================================================

class Tree:
    ROOT: Any = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Tag any NodeSpec attributes on the class with their owner for later expansion.
        for _, val in vars(cls).items():
            if isinstance(val, NodeSpec):
                setattr(val, "_owner_cls", cls)

    @classmethod
    def build(cls) -> Node:
        root_spec = bt._as_spec(cls.ROOT)
        return root_spec.to_node(owner_cls=cls)


class Runner:
    def __init__(self, tree_cls: type[Tree], bb: Optional[Blackboard] = None) -> None:
        self.tree_cls = tree_cls
        self.bb = bb or Blackboard()
        self.root: Node = tree_cls.build()

    async def tick(self) -> Status:
        return await self.root.tick(self.bb)

    async def tick_until_complete(self, timeout: Optional[float] = None) -> Status:
        start = self.bb.now()
        while True:
            st = await self.tick()
            if st in (Status.SUCCESS, Status.FAILURE, Status.CANCELLED, Status.ERROR):
                return st
            if timeout is not None and (self.bb.now() - start) > timeout:
                return Status.CANCELLED
            await asyncio.sleep(0)


# ======================================================================================
# Mermaid exporter (owner-aware)
# ======================================================================================

def to_mermaid(root_spec: NodeSpec, owner_cls: Optional[type] = None) -> str:
    """
    Render a static structure graph. Composites are expanded using the owner class.
    Decorator nesting shows as sequential nodes.
    Supports: bind/subtree expansion for cross-file composition.
    """
    lines: List[str] = ["flowchart TD"]
    node_ids: Dict[NodeSpec, str] = {}
    counter = 0

    def nid(spec: NodeSpec) -> str:
        nonlocal counter
        if spec in node_ids:
            return node_ids[spec]
        counter += 1
        node_ids[spec] = f"N{counter}"
        return node_ids[spec]

    def label(spec: NodeSpec) -> str:
        if spec.kind in ("action", "condition"):
            return f"{spec.kind.upper()}<br/>{spec.name}"
        if spec.kind in ("sequence", "selector", "parallel"):
            return f"{spec.kind.capitalize()}<br/>{spec.name}"
        if spec.kind == "decorator":
            return f"Decor<br/>{spec.name}"
        if spec.kind == "subtree":
            return f"Subtree<br/>{spec.name}"
        if spec.kind == "bind":
            return f"Bind<br/>{spec.name}"
        return spec.name

    def ensure_children(spec: NodeSpec) -> List[NodeSpec]:
        if spec.kind in ("sequence", "selector", "parallel"):
            # Prefer tagged owner on the spec; fall back to the diagram owner
            owner_for_spec = getattr(spec, "_owner_cls", None) or owner_cls
            factory = spec.payload["factory"]
            spec.children = _bt_expand_children_with_owner(factory, owner_for_spec)
        elif spec.kind == "bind":
            # Child already attached; make sure it carries the bound owner for expansion
            base = spec.payload["spec"]
            owner = spec.payload["owner"]
            setattr(base, "_owner_cls", getattr(base, "_owner_cls", None) or owner)
            spec.children = [base]
        elif spec.kind == "subtree":
            # Child is the other tree's ROOT; tag it so its composites expand
            tree_cls = spec.payload["tree_cls"]
            child = bt._as_spec(tree_cls.ROOT)
            setattr(child, "_owner_cls", getattr(child, "_owner_cls", None) or tree_cls)
            spec.children = [child]
        return spec.children

    def walk(spec: NodeSpec) -> None:
        this_id = nid(spec)
        shape = ("((%s))" if spec.kind in ("action", "condition") else "[\"%s\"]") % label(spec)
        lines.append(f"  {this_id}{shape}")
        for child in ensure_children(spec):
            child_id = nid(child)
            lines.append(f"  {this_id} --> {child_id}")
            walk(child)

    walk(root_spec)
    return "\n".join(lines)


# ======================================================================================
# Demo
# ======================================================================================

if __name__ == "__main__":
    # Demo showcasing fluent composition + owner-aware composites
    class Engage(Tree):
        @bt.action
        async def engage(bb: Blackboard) -> Status:
            steps = bb.setdefault("engage_steps", 0)
            bb["engage_steps"] = steps + 1
            await bb.sleep(0.02)
            return Status.SUCCESS if bb["engage_steps"] >= 10 else Status.RUNNING

        @bt.condition
        def threat_detected(bb: Blackboard) -> bool:
            return bb.get("counter", 0) == 3

        @bt.condition
        def battery_ok(bb: Blackboard) -> bool:
            return bb.get("battery", 100) > 20

        @bt.sequence(memory=False)
        def EngageThreat(N):
            yield N.threat_detected
            yield bt.failer().gate(N.battery_ok).timeout(0.12)(N.engage)
        ROOT = EngageThreat

    class Demo(Tree):
        # Conditions
        @bt.condition
        def has_waypoints(bb: Blackboard) -> bool:
            counter = bb.get("counter", 0)
            return (counter % 2) == 1
        # Actions
        @bt.action
        async def go_to_next(bb: Blackboard) -> Status:
            progress = bb.setdefault("goto_progress", 0)
            await bb.sleep(0.01)
            bb["goto_progress"] = progress + 1
            return Status.SUCCESS if bb["goto_progress"] >= 3 else Status.RUNNING

        @bt.action
        async def scan_area(bb: Blackboard) -> Status:
            attempts = bb.setdefault("scan_attempts", 0)
            bb["scan_attempts"] = attempts + 1
            await bb.sleep(0.005)
            return Status.SUCCESS if bb["scan_attempts"] >= 2 else Status.FAILURE

        @bt.action
        async def telemetry_push(bb: Blackboard) -> Status:
            bb["telemetry_count"] = bb.get("telemetry_count", 0) + 1
            return Status.SUCCESS

        # Composites (factories take N = owner class)
        @bt.sequence(memory=True)
        def Patrol(N):
            yield N.has_waypoints
            yield N.go_to_next
            # Fluent: succeeder → retry(3) → timeout(1.0) applied to scan_area
            yield bt.succeeder().retry(3).timeout(1.0)(N.scan_area)


        @bt.selector(memory=True, reactive=True)
        def Root(N):
            yield bt.subtree(Engage)
            yield N.Patrol
            yield bt.ratelimit(hz=5.0)(N.telemetry_push)

        ROOT = Root

    async def _demo():
        bb = Blackboard()
        bb["battery"] = 50
        runner = Runner(Demo, bb=bb)

        for i in range(1, 10):
            bb["counter"] = i
            st = await runner.tick()
            print(f"[tick {i}] status={st.name} "
                  f"(goto={bb.get('goto_progress')}, scan={bb.get('scan_attempts')}, "
                  f"eng={bb.get('engage_steps')}, tel={bb.get('telemetry_count')})")
            if st in (Status.SUCCESS, Status.FAILURE):
                break
            await asyncio.sleep(0)

        print("\n--- Mermaid ---")
        print(to_mermaid(bt._as_spec(Demo.ROOT), owner_cls=Demo))

    asyncio.run(_demo())
