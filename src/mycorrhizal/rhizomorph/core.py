#!/usr/bin/env python3
"""
Rhizomorph — asyncio Behavior Tree core (owner-aware + fluent wrappers)
Multi-file composition: owner tagging + bt.subtree/bt.bind
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    Set,
    Protocol,
)
from typing import Sequence as SequenceT
from types import SimpleNamespace

from mycorrhizal.common.timebase import *

logger = logging.getLogger(__name__)

BB = TypeVar("BB")


# ======================================================================================
# Function signature protocols
# ======================================================================================


def _supports_timebase(func: Callable) -> bool:
    """Check if a function accepts a 'tb' parameter."""
    try:
        sig = inspect.signature(func)
        return 'tb' in sig.parameters
    except (ValueError, TypeError):
        return False


async def _call_node_function(func: Callable, bb: Any, tb: Timebase) -> Any:
    """Call a node function with appropriate parameters based on its signature."""
    if _supports_timebase(func):
        if inspect.iscoroutinefunction(func):
            return await func(bb=bb, tb=tb)
        else:
            return func(bb=bb, tb=tb)
    else:
        if inspect.iscoroutinefunction(func):
            return await func(bb)
        else:
            return func(bb)


# ======================================================================================
# Status / helpers
# ======================================================================================


class Status(Enum):
    SUCCESS = 1
    FAILURE = 2
    RUNNING = 3
    CANCELLED = 4
    ERROR = 5


class ExceptionPolicy(Enum):
    """Policy for handling exceptions in action/condition nodes."""

    LOG_AND_CONTINUE = 1
    PROPAGATE = 2


def _name_of(obj: Any) -> str:
    if hasattr(obj, "__name__"):
        return obj.__name__
    if hasattr(obj, "name"):
        return str(obj.name)
    return f"{obj.__class__.__name__}@{id(obj):x}"


# ======================================================================================
# Recursion Detection
# ======================================================================================


class RecursionError(Exception):
    """Raised when recursive behavior tree structure is detected"""

    pass


# ======================================================================================
# Node model
# ======================================================================================


class Node(Generic[BB]):
    """Abstract BT node; override `tick` and (optionally) lifecycle hooks."""

    def __init__(
        self,
        name: Optional[str] = None,
        exception_policy: ExceptionPolicy = ExceptionPolicy.LOG_AND_CONTINUE,
    ) -> None:
        self.name: str = name or _name_of(self)
        self.parent: Optional[Node[BB]] = None
        self.exception_policy = exception_policy
        self._entered: bool = False
        self._last_status: Optional[Status] = None

    async def tick(self, bb: BB, tb: Timebase) -> Status:
        raise NotImplementedError

    async def on_enter(self, bb: BB, tb: Timebase) -> None:
        return

    async def on_exit(self, bb: BB, status: Status, tb: Timebase) -> None:
        return

    def reset(self) -> None:
        self._entered = False
        self._last_status = None

    async def _ensure_entered(self, bb: BB, tb: Timebase) -> None:
        if not self._entered:
            await self.on_enter(bb, tb)
            self._entered = True

    async def _finish(self, bb: BB, status: Status, tb: Timebase) -> Status:
        self._last_status = status
        if status is not Status.RUNNING:
            try:
                await self.on_exit(bb, status, tb)
            finally:
                self._entered = False
        return status


# ======================================================================================
# Leaves
# ======================================================================================


class Action(Node[BB]):
    """Leaf wrapping a sync/async function -> Status | bool | None."""

    def __init__(
        self,
        func: Callable[..., Any],
        exception_policy: ExceptionPolicy = ExceptionPolicy.LOG_AND_CONTINUE,
    ) -> None:
        super().__init__(name=_name_of(func), exception_policy=exception_policy)
        self._func = func

    async def tick(self, bb: BB, tb: Timebase) -> Status:
        await self._ensure_entered(bb, tb)
        try:
            result = await _call_node_function(self._func, bb, tb)
        except asyncio.CancelledError:
            return await self._finish(bb, Status.CANCELLED, tb)
        except Exception as e:
            logger.error(
                f"Exception in Action '{self.name}':\n{traceback.format_exc()}"
            )
            if self.exception_policy == ExceptionPolicy.PROPAGATE:
                raise
            return await self._finish(bb, Status.ERROR, tb)

        if isinstance(result, Status):
            return await self._finish(bb, result, tb)
        if isinstance(result, bool):
            return await self._finish(
                bb, Status.SUCCESS if result else Status.FAILURE, tb
            )
        return await self._finish(bb, Status.SUCCESS, tb)


class Condition(Action[BB]):
    """Boolean leaf: True→SUCCESS, False→FAILURE (Status accepted but discouraged)."""

    async def tick(self, bb: BB, tb: Timebase) -> Status:
        await self._ensure_entered(bb, tb)
        try:
            result = await _call_node_function(self._func, bb, tb)
        except asyncio.CancelledError:
            return await self._finish(bb, Status.CANCELLED, tb)
        except Exception as e:
            logger.error(
                f"Exception in Condition '{self.name}':\n{traceback.format_exc()}"
            )
            if self.exception_policy == ExceptionPolicy.PROPAGATE:
                raise
            return await self._finish(bb, Status.ERROR, tb)

        if isinstance(result, Status):
            return await self._finish(bb, result, tb)
        return await self._finish(
            bb, Status.SUCCESS if bool(result) else Status.FAILURE, tb
        )


# ======================================================================================
# Composites
# ======================================================================================


class Sequence(Node[BB]):
    """Sequence (AND): fail/err fast; RUNNING bubbles; all SUCCESS → SUCCESS."""

    def __init__(
        self,
        children: SequenceT[Node[BB]],
        *,
        memory: bool = True,
        name: Optional[str] = None,
        exception_policy: ExceptionPolicy = ExceptionPolicy.LOG_AND_CONTINUE,
    ) -> None:
        super().__init__(name or "Sequence", exception_policy=exception_policy)
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

    async def tick(self, bb: BB, tb: Timebase) -> Status:
        await self._ensure_entered(bb, tb)
        start = self._idx if self.memory else 0
        for i in range(start, len(self.children)):
            st = await self.children[i].tick(bb, tb)
            if st is Status.RUNNING:
                if self.memory:
                    self._idx = i
                return Status.RUNNING
            if st in (Status.FAILURE, Status.ERROR, Status.CANCELLED):
                self._idx = 0
                return await self._finish(bb, st, tb)
        self._idx = 0
        return await self._finish(bb, Status.SUCCESS, tb)


class Selector(Node[BB]):
    """Selector (Fallback): first SUCCESS wins; RUNNING bubbles; else FAILURE."""

    def __init__(
        self,
        children: SequenceT[Node[BB]],
        *,
        memory: bool = True,
        name: Optional[str] = None,
        exception_policy: ExceptionPolicy = ExceptionPolicy.LOG_AND_CONTINUE,
    ) -> None:
        super().__init__(name or "Selector", exception_policy=exception_policy)
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

    async def tick(self, bb: BB, tb: Timebase) -> Status:
        await self._ensure_entered(bb, tb)
        start = self._idx if self.memory else 0
        for i in range(start, len(self.children)):
            st = await self.children[i].tick(bb, tb)
            if st is Status.SUCCESS:
                self._idx = 0
                return await self._finish(bb, Status.SUCCESS, tb)
            if st is Status.RUNNING:
                if self.memory:
                    self._idx = i
                return Status.RUNNING
        self._idx = 0
        return await self._finish(bb, Status.FAILURE, tb)


class Parallel(Node[BB]):
    """
    Parallel: tick all children per tick concurrently.
    - success_threshold (k) SUCCESS to report SUCCESS
    - failure_threshold defaults to n-k+1 (cannot reach success anymore) to report FAILURE
    Otherwise RUNNING.
    """

    def __init__(
        self,
        children: SequenceT[Node[BB]],
        *,
        success_threshold: int,
        failure_threshold: Optional[int] = None,
        name: Optional[str] = None,
        exception_policy: ExceptionPolicy = ExceptionPolicy.LOG_AND_CONTINUE,
    ) -> None:
        super().__init__(name or "Parallel", exception_policy=exception_policy)
        self.children = list(children)
        for ch in self.children:
            ch.parent = self
        self.n = len(self.children)
        self.k = max(1, min(success_threshold, self.n))
        self.f = (
            failure_threshold
            if failure_threshold is not None
            else (self.n - self.k + 1)
        )

    def reset(self) -> None:
        super().reset()
        for ch in self.children:
            ch.reset()

    async def tick(self, bb: BB, tb: Timebase) -> Status:
        await self._ensure_entered(bb, tb)
        tasks = [asyncio.create_task(ch.tick(bb, tb)) for ch in self.children]
        try:
            results = await asyncio.gather(*tasks, return_exceptions=False)
        finally:
            pass
        succ = sum(1 for s in results if s is Status.SUCCESS)
        fail = sum(
            1 for s in results if s in (Status.FAILURE, Status.ERROR, Status.CANCELLED)
        )
        if succ >= self.k:
            return await self._finish(bb, Status.SUCCESS, tb)
        if fail >= self.f:
            return await self._finish(bb, Status.FAILURE, tb)
        return Status.RUNNING


# ======================================================================================
# Decorators (wrappers)
# ======================================================================================


class Inverter(Node[BB]):
    def __init__(
        self,
        child: Node[BB],
        name: Optional[str] = None,
        exception_policy: ExceptionPolicy = ExceptionPolicy.LOG_AND_CONTINUE,
    ) -> None:
        super().__init__(
            name or f"Inverter({_name_of(child)})", exception_policy=exception_policy
        )
        self.child = child
        self.child.parent = self

    def reset(self) -> None:
        super().reset()
        self.child.reset()

    async def tick(self, bb: BB, tb: Timebase) -> Status:
        await self._ensure_entered(bb, tb)
        st = await self.child.tick(bb, tb)
        if st is Status.SUCCESS:
            return await self._finish(bb, Status.FAILURE, tb)
        if st is Status.FAILURE:
            return await self._finish(bb, Status.SUCCESS, tb)
        return st


class Retry(Node[BB]):
    def __init__(
        self,
        child: Node[BB],
        *,
        max_attempts: int,
        retry_on: Tuple[Status, ...] = (Status.FAILURE, Status.ERROR),
        name: Optional[str] = None,
        exception_policy: ExceptionPolicy = ExceptionPolicy.LOG_AND_CONTINUE,
    ) -> None:
        super().__init__(
            name or f"Retry({_name_of(child)},{max_attempts})",
            exception_policy=exception_policy,
        )
        self.child = child
        self.child.parent = self
        self.max_attempts = max(1, int(max_attempts))
        self.retry_on = retry_on
        self._attempt = 0

    def reset(self) -> None:
        super().reset()
        self._attempt = 0
        self.child.reset()

    async def tick(self, bb: BB, tb: Timebase) -> Status:
        await self._ensure_entered(bb, tb)
        st = await self.child.tick(bb, tb)
        if st is Status.RUNNING:
            return Status.RUNNING
        if st is Status.SUCCESS:
            self._attempt = 0
            return await self._finish(bb, Status.SUCCESS, tb)
        self._attempt += 1
        if st in self.retry_on and self._attempt < self.max_attempts:
            self.child.reset()
            return Status.RUNNING
        self._attempt = 0
        return await self._finish(bb, Status.FAILURE, tb)


class Timeout(Node[BB]):
    def __init__(
        self,
        child: Node[BB],
        *,
        seconds: float,
        name: Optional[str] = None,
        exception_policy: ExceptionPolicy = ExceptionPolicy.LOG_AND_CONTINUE,
    ) -> None:
        super().__init__(
            name or f"Timeout({_name_of(child)},{seconds}s)",
            exception_policy=exception_policy,
        )
        self.child = child
        self.child.parent = self
        self.seconds = max(0.0, float(seconds))
        self._deadline: Optional[float] = None

    def reset(self) -> None:
        super().reset()
        self._deadline = None
        self.child.reset()

    async def on_enter(self, bb: BB, tb: Timebase) -> None:
        self._deadline = tb.now() + self.seconds

    async def tick(self, bb: BB, tb: Timebase) -> Status:
        await self._ensure_entered(bb, tb)
        if self._deadline is None:
            self._deadline = tb.now() + self.seconds
        if tb.now() > self._deadline:
            self.child.reset()
            return await self._finish(bb, Status.FAILURE, tb)
        st = await self.child.tick(bb, tb)
        if st is Status.RUNNING:
            return Status.RUNNING
        return await self._finish(bb, st, tb)


class Succeeder(Node[BB]):
    def __init__(
        self,
        child: Node[BB],
        name: Optional[str] = None,
        exception_policy: ExceptionPolicy = ExceptionPolicy.LOG_AND_CONTINUE,
    ) -> None:
        super().__init__(
            name or f"Succeeder({_name_of(child)})", exception_policy=exception_policy
        )
        self.child = child
        self.child.parent = self

    def reset(self) -> None:
        super().reset()
        self.child.reset()

    async def tick(self, bb: BB, tb: Timebase) -> Status:
        await self._ensure_entered(bb, tb)
        st = await self.child.tick(bb, tb)
        if st is Status.RUNNING:
            return Status.RUNNING
        return await self._finish(bb, Status.SUCCESS, tb)


class Failer(Node[BB]):
    def __init__(
        self,
        child: Node[BB],
        name: Optional[str] = None,
        exception_policy: ExceptionPolicy = ExceptionPolicy.LOG_AND_CONTINUE,
    ) -> None:
        super().__init__(
            name or f"Failer({_name_of(child)})", exception_policy=exception_policy
        )
        self.child = child
        self.child.parent = self

    def reset(self) -> None:
        super().reset()
        self.child.reset()

    async def tick(self, bb: BB, tb: Timebase) -> Status:
        await self._ensure_entered(bb, tb)
        st = await self.child.tick(bb, tb)
        if st is Status.RUNNING:
            return Status.RUNNING
        return await self._finish(bb, Status.FAILURE, tb)


class RateLimit(Node[BB]):
    """
    Throttle *starting* the child to at most 1 per period (or hz).
    If child is RUNNING, do not throttle (avoid starvation).
    """

    def __init__(
        self,
        child: Node[BB],
        *,
        hz: Optional[float] = None,
        period: Optional[float] = None,
        name: Optional[str] = None,
        exception_policy: ExceptionPolicy = ExceptionPolicy.LOG_AND_CONTINUE,
    ) -> None:
        if (hz is not None) and (period is not None):
            raise ValueError("RateLimit requires exactly one of (hz, period)")

        if period is not None:
            per = period
        elif hz is not None:
            per = 1.0 / float(hz)
        else:
            raise ValueError("RateLimit requires exactly one of (hz, period)")

        super().__init__(
            name or f"RateLimit({_name_of(child)},{per:.6f}s)",
            exception_policy=exception_policy,
        )
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

    async def tick(self, bb: BB, tb: Timebase) -> Status:
        await self._ensure_entered(bb, tb)
        if self._last is Status.RUNNING:
            st = await self.child.tick(bb, tb)
            self._last = st
            if st is Status.RUNNING:
                return Status.RUNNING
            self._next_allowed = tb.now() + self._period
            return await self._finish(bb, st, tb)

        now = tb.now()
        if self._next_allowed is not None and now < self._next_allowed:
            return Status.RUNNING
        st = await self.child.tick(bb, tb)
        self._last = st
        if st is Status.RUNNING:
            return Status.RUNNING
        self._next_allowed = now + self._period
        return await self._finish(bb, st, tb)


class Gate(Node[BB]):
    """
    Guard a child with a condition:
      SUCCESS → tick child
      RUNNING → RUNNING
      else    → FAILURE
    """

    def __init__(
        self,
        condition: Node[BB],
        child: Node[BB],
        name: Optional[str] = None,
        exception_policy: ExceptionPolicy = ExceptionPolicy.LOG_AND_CONTINUE,
    ) -> None:
        super().__init__(
            name or f"Gate(cond={_name_of(condition)}, child={_name_of(child)})",
            exception_policy=exception_policy,
        )
        self.condition = condition
        self.child = child
        self.condition.parent = self
        self.child.parent = self

    def reset(self) -> None:
        super().reset()
        self.condition.reset()
        self.child.reset()

    async def tick(self, bb: BB, tb: Timebase) -> Status:
        await self._ensure_entered(bb, tb)
        c = await self.condition.tick(bb, tb)
        if c is Status.RUNNING:
            return Status.RUNNING
        if c is not Status.SUCCESS:
            return await self._finish(bb, Status.FAILURE, tb)
        st = await self.child.tick(bb, tb)
        if st is Status.RUNNING:
            return Status.RUNNING
        return await self._finish(bb, st, tb)


class Match(Node[BB]):
    """
    Pattern-matching dispatch node.
    
    Evaluates a key function against the blackboard, then checks each case
    in order. The first matching case's child is executed. If the child
    completes (SUCCESS or FAILURE), that status is returned immediately.
    
    Cases can match by:
      - Type: isinstance(value, case_type)
      - Predicate: case_predicate(value) returns True
      - Value: value == case_value
      - Default: always matches (should be last)
    
    If no case matches and there's no default, returns FAILURE.
    """

    def __init__(
        self,
        key_fn: Callable[[Any], Any],
        cases: List[Tuple[Any, Node[BB]]],
        name: Optional[str] = None,
        exception_policy: ExceptionPolicy = ExceptionPolicy.LOG_AND_CONTINUE,
    ) -> None:
        super().__init__(
            name or f"Match({_name_of(key_fn)})",
            exception_policy=exception_policy,
        )
        self._key_fn = key_fn
        self._cases = cases
        for _, child in self._cases:
            child.parent = self
        self._matched_idx: Optional[int] = None

    def reset(self) -> None:
        super().reset()
        self._matched_idx = None
        for _, child in self._cases:
            child.reset()

    def _matches(self, matcher: Any, value: Any) -> bool:
        """Check if a matcher matches the given value."""
        if matcher is _DefaultCase:
            return True
        if isinstance(matcher, type):
            return isinstance(value, matcher)
        if callable(matcher):
            return bool(matcher(value))
        return value == matcher

    async def tick(self, bb: BB, tb: Timebase) -> Status:
        await self._ensure_entered(bb, tb)
        
        if self._matched_idx is not None:
            _, child = self._cases[self._matched_idx]
            st = await child.tick(bb, tb)
            if st is Status.RUNNING:
                return Status.RUNNING
            self._matched_idx = None
            return await self._finish(bb, st, tb)
        
        value = await _call_node_function(self._key_fn, bb, tb)
        
        for i, (matcher, child) in enumerate(self._cases):
            if self._matches(matcher, value):
                st = await child.tick(bb, tb)
                if st is Status.RUNNING:
                    self._matched_idx = i
                    return Status.RUNNING
                return await self._finish(bb, st, tb)
        
        return await self._finish(bb, Status.FAILURE, tb)


# ======================================================================================
# Authoring DSL (NodeSpec + bt namespace)
# ======================================================================================


class NodeSpecKind(Enum):
    ACTION = "action"
    CONDITION = "condition"
    SEQUENCE = "sequence"
    SELECTOR = "selector"
    PARALLEL = "parallel"
    DECORATOR = "decorator"
    SUBTREE = "subtree"
    MATCH = "match"


class _DefaultCase:
    """Sentinel for default case in match expressions."""
    pass


@dataclass
class CaseSpec:
    """Specification for a case in a match expression."""
    matcher: Any
    child: "NodeSpec"
    label: str = ""
    
    def __post_init__(self):
        if not self.label:
            if self.matcher is _DefaultCase:
                self.label = "default"
            elif isinstance(self.matcher, type):
                self.label = self.matcher.__name__
            elif callable(self.matcher):
                self.label = _name_of(self.matcher)
            else:
                self.label = repr(self.matcher)


@dataclass
class NodeSpec:
    kind: NodeSpecKind
    name: str
    payload: Any = None
    children: List["NodeSpec"] = field(default_factory=list)

    def __hash__(self):
        return hash((self.kind, self.name, id(self.payload), tuple(self.children)))

    def to_node(
        self,
        owner: Optional[Any] = None,
        exception_policy: ExceptionPolicy = ExceptionPolicy.LOG_AND_CONTINUE,
    ) -> Node[Any]:
        match self.kind:
            case NodeSpecKind.ACTION:
                return Action(self.payload, exception_policy=exception_policy)
            case NodeSpecKind.CONDITION:
                return Condition(self.payload, exception_policy=exception_policy)
            case NodeSpecKind.SEQUENCE | NodeSpecKind.SELECTOR | NodeSpecKind.PARALLEL:
                owner_effective = getattr(self, "owner", None) or owner
                factory = self.payload["factory"]
                expanded = _bt_expand_children(factory, owner_effective)
                self.children = expanded
                built = [
                    ch.to_node(owner_effective, exception_policy) for ch in expanded
                ]
                match self.kind:
                    case NodeSpecKind.SEQUENCE:
                        return Sequence(
                            built,
                            memory=self.payload.get("memory", True),
                            name=self.name,
                            exception_policy=exception_policy,
                        )
                    case NodeSpecKind.SELECTOR:
                        return Selector(
                            built,
                            memory=self.payload.get("memory", True),
                            name=self.name,
                            exception_policy=exception_policy,
                        )
                    case NodeSpecKind.PARALLEL:
                        return Parallel(
                            built,
                            success_threshold=self.payload["success_threshold"],
                            failure_threshold=self.payload.get("failure_threshold"),
                            name=self.name,
                            exception_policy=exception_policy,
                        )

            case NodeSpecKind.DECORATOR:
                assert len(self.children) == 1, "Decorator must wrap exactly one child"
                child_node = self.children[0].to_node(owner, exception_policy)
                builder = self.payload
                return builder(child_node)

            case NodeSpecKind.SUBTREE:
                subtree_root = self.payload["root"]
                return subtree_root.to_node(exception_policy=exception_policy)

            case NodeSpecKind.MATCH:
                key_fn = self.payload["key_fn"]
                case_specs: List[CaseSpec] = self.payload["cases"]
                cases = [
                    (cs.matcher, cs.child.to_node(owner, exception_policy))
                    for cs in case_specs
                ]
                return Match(
                    key_fn,
                    cases,
                    name=self.name,
                    exception_policy=exception_policy,
                )

            case _:
                raise ValueError(f"Unknown spec kind: {self.kind}")


def _bt_expand_children(
    factory: Callable[..., Generator[Any, None, None]],
    owner: Optional[Any],
    expansion_stack: Optional[Set[str]] = None,
) -> List[NodeSpec]:
    """
    Execute a composite factory to get child specs.

    Args:
        factory: The generator function that yields child specs
        owner: The namespace object for resolving N references
        expansion_stack: Stack of factory names to detect recursion
    """
    if expansion_stack is None:
        expansion_stack = set()

    factory_name = _name_of(factory)
    if factory_name in expansion_stack:
        chain = " -> ".join(expansion_stack) + f" -> {factory_name}"
        raise RecursionError(
            f"Recursive behavior tree structure detected: {chain}\n"
            f"Behavior trees must be acyclic. Consider using:\n"
            f"  - A Selector with memory to iterate through options\n"
            f"  - A Sequence with conditions to control flow\n"
            f"  - A Retry decorator for repeated attempts\n"
            f"  - State in the blackboard to track progress"
        )

    expansion_stack = expansion_stack.copy()
    expansion_stack.add(factory_name)

    try:
        gen = factory(owner)
    except TypeError:
        gen = factory()

    if not inspect.isgenerator(gen):
        raise TypeError(
            f"Composite factory {factory.__name__} must be a generator (use 'yield')."
        )

    out: List[NodeSpec] = []
    for yielded in gen:
        if isinstance(yielded, (list, tuple)):
            for y in yielded:
                spec = bt.as_spec(y)
                if (
                    hasattr(spec, "payload")
                    and isinstance(spec.payload, dict)
                    and "factory" in spec.payload
                ):
                    spec._expansion_stack = expansion_stack  # type: ignore
                out.append(spec)
            continue
        spec = bt.as_spec(yielded)
        if (
            hasattr(spec, "payload")
            and isinstance(spec.payload, dict)
            and "factory" in spec.payload
        ):
            spec._expansion_stack = expansion_stack  # type: ignore
        out.append(spec)

    for spec in out:
        if spec.kind in (
            NodeSpecKind.SEQUENCE,
            NodeSpecKind.SELECTOR,
            NodeSpecKind.PARALLEL,
        ) and hasattr(spec, "_expansion_stack"):
            child_factory = spec.payload["factory"]
            child_owner = getattr(spec, "owner", None) or owner
            _bt_expand_children(child_factory, child_owner, spec._expansion_stack)  # type: ignore

    return out


# --------------------------------------------------------------------------------------
# Fluent wrapper chain (left-to-right)
# --------------------------------------------------------------------------------------


class _WrapperChain:
    """
    Fluent factory for decorator stacks that read left→right.

        chain = bt.failer().gate(battery_ok).timeout(0.12)
        yield chain(engage)
    """

    def __init__(
        self,
        builders: Optional[List[Callable[..., Any]]] = None,
        labels: Optional[List[str]] = None,
    ) -> None:
        self._builders: List[Callable[..., Any]] = list(builders or [])
        self._labels: List[str] = list(labels or [])

    def _append(self, label: str, builder: Callable[..., Any]) -> "_WrapperChain":
        self._builders.append(builder)
        self._labels.append(label)
        return self

    def failer(self) -> "_WrapperChain":
        return self._append("Failer", lambda ch: Failer(ch))

    def succeeder(self) -> "_WrapperChain":
        return self._append("Succeeder", lambda ch: Succeeder(ch))

    def inverter(self) -> "_WrapperChain":
        return self._append("Inverter", lambda ch: Inverter(ch))

    def timeout(self, seconds: float) -> "_WrapperChain":
        return self._append(
            f"Timeout({seconds}s)", lambda ch: Timeout(ch, seconds=seconds)
        )

    def retry(
        self,
        max_attempts: int,
        retry_on: Tuple[Status, ...] = (Status.FAILURE, Status.ERROR),
    ) -> "_WrapperChain":
        return self._append(
            f"Retry({max_attempts})",
            lambda ch: Retry(ch, max_attempts=max_attempts, retry_on=retry_on),
        )

    def ratelimit(
        self, *, hz: Optional[float] = None, period: Optional[float] = None
    ) -> "_WrapperChain":
        label = "RateLimit(?)"
        if hz is not None:
            label = f"RateLimit({1.0/float(hz):.6f}s)"
        elif period is not None:
            label = f"RateLimit({float(period):.6f}s)"
        return self._append(label, lambda ch: RateLimit(ch, hz=hz, period=period))

    def gate(
        self, condition_spec_or_fn: Union["NodeSpec", Callable[[Any], Any]]
    ) -> "_WrapperChain":
        cond_spec = bt.as_spec(condition_spec_or_fn)
        return self._append(
            f"Gate(cond={_name_of(cond_spec)})",
            lambda ch: Gate(cond_spec.to_node(), ch),
        )

    def __call__(self, inner: Union["NodeSpec", Callable[[Any], Any]]) -> "NodeSpec":
        """
        Apply the chain to a child spec → nested decorator NodeSpecs.
        Left→right call order becomes outermost→…→innermost when built.
        """
        spec = bt.as_spec(inner)
        result = spec
        for label, builder in reversed(list(zip(self._labels, self._builders))):
            result = NodeSpec(
                kind=NodeSpecKind.DECORATOR,
                name=f"{label}({_name_of(result)})",
                payload=builder,
                children=[result],
            )
        return result

    def __rshift__(self, inner: Union["NodeSpec", Callable[[Any], Any]]) -> "NodeSpec":
        return self(inner)


# --------------------------------------------------------------------------------------
# Match expression builders
# --------------------------------------------------------------------------------------


class _CaseBuilder:
    """Builder for individual match cases."""
    
    def __init__(self, matcher: Any) -> None:
        self._matcher = matcher
    
    def __call__(self, child: Union["NodeSpec", Callable[[Any], Any]]) -> CaseSpec:
        child_spec = bt.as_spec(child)
        return CaseSpec(matcher=self._matcher, child=child_spec)


class _MatchBuilder:
    """Builder for match expressions."""
    
    def __init__(self, key_fn: Callable[[Any], Any]) -> None:
        self._key_fn = key_fn
    
    def __call__(self, *cases: CaseSpec) -> NodeSpec:
        if not cases:
            raise ValueError("bt.match() requires at least one case")
        
        for case in cases:
            if not isinstance(case, CaseSpec):
                raise TypeError(
                    f"bt.match() expects CaseSpec instances (from bt.case() or bt.defaultcase()), "
                    f"got {type(case).__name__}"
                )
        
        children = [case.child for case in cases]
        
        return NodeSpec(
            kind=NodeSpecKind.MATCH,
            name=f"Match({_name_of(self._key_fn)})",
            payload={
                "key_fn": self._key_fn,
                "cases": list(cases),
            },
            children=children,
        )


# --------------------------------------------------------------------------------------
# User-facing decorator/constructor namespace
# --------------------------------------------------------------------------------------


class _BT:
    """User-facing decorator/constructor namespace."""

    # TODO: Fix decorators to not throw a fit when passing timebase arguments to actions/conditions
    def __init__(self):
        self._tracking_stack: List[List[Tuple[str, Any]]] = []

    def action(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator to mark a function as an action node."""
        spec = NodeSpec(kind=NodeSpecKind.ACTION, name=_name_of(fn), payload=fn)
        fn.node_spec = spec  # type: ignore
        if self._tracking_stack:
            self._tracking_stack[-1].append((fn.__name__, fn))
        return fn

    def condition(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator to mark a function as a condition node."""
        spec = NodeSpec(kind=NodeSpecKind.CONDITION, name=_name_of(fn), payload=fn)
        fn.node_spec = spec  # type: ignore
        if self._tracking_stack:
            self._tracking_stack[-1].append((fn.__name__, fn))
        return fn

    def sequence(self, *, memory: bool = True):
        """Decorator to mark a generator function as a sequence composite."""

        def deco(
            factory: Callable[..., Generator[Any, None, None]],
        ) -> Callable[..., Generator[Any, None, None]]:
            spec = NodeSpec(
                kind=NodeSpecKind.SEQUENCE,
                name=_name_of(factory),
                payload={"factory": factory, "memory": memory},
            )
            factory.node_spec = spec  # type: ignore
            if self._tracking_stack:
                self._tracking_stack[-1].append((factory.__name__, factory))
            return factory

        return deco

    def selector(self, *, memory: bool = True, reactive: bool = False):
        """Decorator to mark a generator function as a selector composite."""

        def deco(
            factory: Callable[..., Generator[Any, None, None]],
        ) -> Callable[..., Generator[Any, None, None]]:
            spec = NodeSpec(
                kind=NodeSpecKind.SELECTOR,
                name=_name_of(factory),
                payload={"factory": factory, "memory": memory, "reactive": reactive},
            )
            factory.node_spec = spec  # type: ignore
            if self._tracking_stack:
                self._tracking_stack[-1].append((factory.__name__, factory))
            return factory

        return deco

    def parallel(
        self, *, success_threshold: int, failure_threshold: Optional[int] = None
    ):
        """Decorator to mark a generator function as a parallel composite."""

        def deco(
            factory: Callable[..., Generator[Any, None, None]],
        ) -> Callable[..., Generator[Any, None, None]]:
            spec = NodeSpec(
                kind=NodeSpecKind.PARALLEL,
                name=_name_of(factory),
                payload={
                    "factory": factory,
                    "success_threshold": success_threshold,
                    "failure_threshold": failure_threshold,
                },
            )
            factory.node_spec = spec  # type: ignore
            if self._tracking_stack:
                self._tracking_stack[-1].append((factory.__name__, factory))
            return factory

        return deco

    def inverter(self) -> _WrapperChain:
        return _WrapperChain().inverter()

    def succeeder(self) -> _WrapperChain:
        return _WrapperChain().succeeder()

    def failer(self) -> _WrapperChain:
        return _WrapperChain().failer()

    def timeout(self, seconds: float) -> _WrapperChain:
        return _WrapperChain().timeout(seconds)

    def retry(
        self,
        max_attempts: int,
        retry_on: Tuple[Status, ...] = (Status.FAILURE, Status.ERROR),
    ) -> _WrapperChain:
        return _WrapperChain().retry(max_attempts, retry_on=retry_on)

    def ratelimit(
        self, hz: Optional[float] = None, period: Optional[float] = None
    ) -> _WrapperChain:
        return _WrapperChain().ratelimit(hz=hz, period=period)

    def gate(self, condition: Union[NodeSpec, Callable[[Any], Any]]) -> _WrapperChain:
        cond_spec = self.as_spec(condition)
        return _WrapperChain().gate(cond_spec)

    def match(self, key_fn: Callable[[Any], Any]) -> "_MatchBuilder":
        """
        Create a pattern-matching dispatch node.
        
        Usage:
            bt.match(lambda bb: bb.current_action)(
                bt.case(ImageAction)(bt.subtree(handle_image)),
                bt.case(MoveAction)(bt.subtree(handle_move)),
                bt.case(lambda a: a.priority > 5)(bt.subtree(handle_urgent)),
                bt.defaultcase(log_unknown),
            )
        
        Args:
            key_fn: Function that extracts the value to match against from the blackboard
        
        Returns:
            A builder that accepts case specs and returns a NodeSpec
        """
        return _MatchBuilder(key_fn)

    def case(self, matcher: Any) -> "_CaseBuilder":
        """
        Define a case for a match expression.
        
        Args:
            matcher: Can be:
                - A type: matches via isinstance(value, matcher)
                - A callable: matches if matcher(value) returns True
                - Any other value: matches via value == matcher
        
        Returns:
            A builder that accepts a child node spec
        """
        return _CaseBuilder(matcher)

    def defaultcase(self, child: Union[NodeSpec, Callable[[Any], Any]]) -> CaseSpec:
        """
        Define a default case for a match expression (matches anything).
        
        Args:
            child: The node to execute if this case matches
        
        Returns:
            A CaseSpec that always matches
        """
        child_spec = self.as_spec(child)
        return CaseSpec(matcher=_DefaultCase, child=child_spec, label="default")

    def subtree(self, tree: SimpleNamespace) -> NodeSpec:
        """
        Mount another tree's root spec as a subtree.

        Args:
            tree: A tree namespace created with @bt.tree
        """
        if not hasattr(tree, "root"):
            raise ValueError(
                f"Tree namespace must have a 'root' attribute. Did you forget @bt.root on a composite?"
            )

        root_spec = tree.root
        return NodeSpec(
            kind=NodeSpecKind.SUBTREE,
            name=f"Subtree({root_spec.name})",
            payload={"root": root_spec},
            children=[root_spec],
        )

    def tree(self, fn: Callable[[], Any]) -> SimpleNamespace:
        """
        Decorator to create a behavior tree namespace.

        Usage:
            @bt.tree
            def MyTree():
                @bt.action
                def my_action(bb: BB) -> Status:
                    ...

                @bt.sequence()
                def root(N):
                    yield N.my_action
        """
        created_nodes = []
        self._tracking_stack.append(created_nodes)

        try:
            fn()
        finally:
            self._tracking_stack.pop()

        nodes = {
            name: node for name, node in created_nodes if hasattr(node, "node_spec")
        }
        namespace = SimpleNamespace(**nodes)

        for name, node in nodes.items():
            if hasattr(node, "node_spec"):
                node.node_spec.owner = namespace

        root_nodes = [v for v in nodes.values() if hasattr(v.node_spec, "is_root")]
        if root_nodes:
            namespace.root = root_nodes[0].node_spec

        namespace.to_mermaid = lambda: _generate_mermaid(namespace)

        return namespace

    def root(
        self, fn: Callable[..., Generator[Any, None, None]]
    ) -> Callable[..., Generator[Any, None, None]]:
        """Mark a composite as the root of the tree."""
        fn.node_spec.is_root = True  # type: ignore
        return fn

    def as_spec(self, maybe: Union[NodeSpec, Callable[[Any], Any]]) -> NodeSpec:
        """Convert a function or NodeSpec to a NodeSpec."""
        if isinstance(maybe, NodeSpec):
            return maybe

        spec = getattr(maybe, "node_spec", None)
        if spec is None:
            raise TypeError(
                f"{maybe!r} is not a BT node (missing node_spec attribute)."
            )
        return spec


bt = _BT()


# ======================================================================================
# Mermaid Generation
# ======================================================================================


def _generate_mermaid(tree: SimpleNamespace) -> str:
    """
    Render a static structure graph for a tree namespace.
    """
    if not hasattr(tree, "root"):
        raise ValueError("Tree namespace must have a 'root' attribute")

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
        match spec.kind:
            case NodeSpecKind.ACTION | NodeSpecKind.CONDITION:
                return f"{spec.kind.value.upper()}<br/>{spec.name}"
            case NodeSpecKind.SEQUENCE | NodeSpecKind.SELECTOR | NodeSpecKind.PARALLEL:
                return f"{spec.kind.value.capitalize()}<br/>{spec.name}"
            case NodeSpecKind.DECORATOR:
                return f"Decor<br/>{spec.name}"
            case NodeSpecKind.SUBTREE:
                return f"Subtree<br/>{spec.name}"
            case NodeSpecKind.MATCH:
                return f"Match<br/>{spec.name}"
            case _:
                return spec.name

    def ensure_children(spec: NodeSpec) -> List[NodeSpec]:
        match spec.kind:
            case NodeSpecKind.SEQUENCE | NodeSpecKind.SELECTOR | NodeSpecKind.PARALLEL:
                owner = getattr(spec, "owner", None) or tree
                factory = spec.payload["factory"]
                spec.children = _bt_expand_children(factory, owner)
            case NodeSpecKind.SUBTREE:
                subtree_root = spec.payload["root"]
                spec.children = [subtree_root]
            case NodeSpecKind.MATCH:
                case_specs: List[CaseSpec] = spec.payload["cases"]
                spec.children = [cs.child for cs in case_specs]
        return spec.children

    def walk(spec: NodeSpec) -> None:
        this_id = nid(spec)
        shape = (
            "((%s))"
            if spec.kind in (NodeSpecKind.ACTION, NodeSpecKind.CONDITION)
            else '["%s"]'
        ) % label(spec)
        lines.append(f"  {this_id}{shape}")
        
        children = ensure_children(spec)
        
        if spec.kind == NodeSpecKind.MATCH:
            case_specs: List[CaseSpec] = spec.payload["cases"]
            for case_spec, child in zip(case_specs, children):
                child_id = nid(child)
                edge_label = case_spec.label.replace('"', "'")
                lines.append(f'  {this_id} -->|"{edge_label}"| {child_id}')
                walk(child)
        else:
            for child in children:
                child_id = nid(child)
                lines.append(f"  {this_id} --> {child_id}")
                walk(child)

    walk(tree.root)
    return "\n".join(lines)


# ======================================================================================
# Runner
# ======================================================================================


class Runner(Generic[BB]):
    def __init__(
        self,
        tree: SimpleNamespace,
        bb: BB,
        tb: Optional[Timebase] = None,
        exception_policy: ExceptionPolicy = ExceptionPolicy.LOG_AND_CONTINUE,
    ) -> None:
        self.tree = tree
        self.bb: BB = bb
        self.tb = tb or MonotonicClock()
        self.exception_policy = exception_policy

        if not hasattr(tree, "root"):
            raise ValueError("Tree namespace must have a 'root' attribute")

        self.root: Node[BB] = tree.root.to_node(
            owner=tree, exception_policy=exception_policy
        )

    async def tick(self) -> Status:
        result = await self.root.tick(self.bb, self.tb)
        self.tb.advance()
        return result

    async def tick_until_complete(self, timeout: Optional[float] = None) -> Status:
        start = self.tb.now()
        while True:
            st = await self.tick()
            if st in (Status.SUCCESS, Status.FAILURE, Status.CANCELLED, Status.ERROR):
                return st
            if timeout is not None and (self.tb.now() - start) > timeout:
                return Status.CANCELLED
            await asyncio.sleep(0)