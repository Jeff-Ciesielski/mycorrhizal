#!/usr/bin/env python3
"""
Spores Object Cache

LRU cache for tracking seen objects and logging them on eviction.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Callable, Optional, TypeVar, Generic, TYPE_CHECKING
from dataclasses import dataclass

from .models import Object

if TYPE_CHECKING:
    from .core import CacheMetrics


# Generic types for cache
K = TypeVar('K')  # Key type (object ID)
V = TypeVar('V')  # Value type (Object)


@dataclass
class CacheEntry:
    """
    An entry in the object cache.

    Attributes:
        object: The OCEL Object
        sight_count: How many times we've seen this object
        first_sight_time: When we first saw the object
        attributes_hash: Hash of object attributes for change detection
    """
    object: Object
    sight_count: int = 1
    first_sight_time: Optional[float] = None
    attributes_hash: Optional[int] = None


def _compute_attributes_hash(obj: Object) -> Optional[int]:
    """
    Compute hash of object attributes for change detection.

    Args:
        obj: The OCEL Object

    Returns:
        Hash of attributes, or None if object has no attributes
    """
    if not obj.attributes:
        return None
    # Hash frozenset of (attr_name, attr_value) tuples
    # ObjectAttributeValue.value holds the actual Python value
    attrs = {k: v.value for k, v in obj.attributes.items()}
    return hash(frozenset(attrs.items()))


class ObjectLRUCache(Generic[K, V]):
    """
    LRU cache with unified callback for object logging.

    The cache fires the `needs_logged` callback when an object needs to be logged:
    - On first sight (new object added to cache)
    - On eviction (object removed from cache to make room)
    - When attributes change (detected via hash comparison)
    - Every N touches (configurable via `touch_resend_n`)

    This ensures OCEL consumers see object evolution:
    - First sight: Initial object state
    - Attribute changes: Updated state
    - Eviction: Final state before being removed from cache
    - Periodic resend: Long-lived objects are re-logged periodically

    Example:
        ```python
        def needs_logged(object_id: str, obj: Object):
            # Send object to transport
            transport.send(LogRecord(object=obj))

        cache = ObjectLRUCache(maxsize=128, needs_logged=needs_logged, touch_resend_n=100)

        # Check if object exists, add if not
        if not cache.contains_or_add(object_id, object):
            # Object not in cache, already logged by needs_logged
            pass
        ```

    Args:
        maxsize: Maximum number of objects to cache
        needs_logged: Callback when object needs logging (object_id, object) -> None
        touch_resend_n: Resend object every N touches (0 to disable periodic resend)
    """

    def __init__(
        self,
        maxsize: int = 128,
        needs_logged: Optional[Callable[[K, Object], None]] = None,
        touch_resend_n: int = 100,
        metrics: Optional["CacheMetrics"] = None
    ):
        self.maxsize = maxsize
        self.needs_logged = needs_logged
        self.touch_resend_n = touch_resend_n
        self.metrics = metrics
        self._cache: OrderedDict[K, CacheEntry] = OrderedDict()

    def _evict_if_needed(self) -> None:
        """Evict oldest entry if cache is full."""
        while len(self._cache) >= self.maxsize:
            # FIFO from OrderedDict (oldest first)
            object_id, entry = self._cache.popitem(last=False)
            if self.metrics is not None:
                self.metrics.evictions += 1
            if self.needs_logged:
                self.needs_logged(object_id, entry.object)

    def get(self, key: K) -> Optional[Object]:
        """
        Get an object from the cache without updating its position.

        Args:
            key: The object ID

        Returns:
            The Object if found, None otherwise
        """
        entry = self._cache.get(key)
        return entry.object if entry else None

    def get_and_touch(self, key: K) -> Optional[Object]:
        """
        Get an object from the cache and mark it as recently used.

        Args:
            key: The object ID

        Returns:
            The Object if found, None otherwise
        """
        entry = self._cache.get(key)
        if entry:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return entry.object
        return None

    def contains_or_add(self, key: K, obj: Object) -> bool:
        """
        Check if key exists, add if not.

        This is the primary method for object tracking:
        - If key exists: check for attribute changes, periodic resend, mark as recently used
        - If key doesn't exist: add object, potentially evict, log first sight

        The `needs_logged` callback is fired when:
        - First sight: new object added to cache
        - Attribute change: object attributes changed (hash comparison)
        - Every N touches: periodic resend for long-lived objects

        Args:
            key: The object ID
            obj: The OCEL Object to add if not present

        Returns:
            True if object was already in cache, False if it was just added
        """
        if key in self._cache:
            # Object exists - check if we need to log
            entry = self._cache[key]
            entry.sight_count += 1

            # Compute current attribute hash
            current_hash = _compute_attributes_hash(obj)

            # Check if attributes changed
            attrs_changed = (entry.attributes_hash is not None and
                            current_hash != entry.attributes_hash)

            # Update if attributes changed
            if attrs_changed:
                entry.object = obj
                entry.attributes_hash = current_hash

            # Fire callback if ANY condition met
            should_log = any([
                attrs_changed,
                self.touch_resend_n > 0 and entry.sight_count % self.touch_resend_n == 0,
            ])
            if should_log and self.needs_logged:
                self.needs_logged(key, entry.object)

            self._cache.move_to_end(key)
            return True
        else:
            # First sight - store hash, fire callback
            self._evict_if_needed()
            if self.metrics is not None:
                self.metrics.first_sights += 1
            attr_hash = _compute_attributes_hash(obj)
            entry = CacheEntry(object=obj, attributes_hash=attr_hash)
            self._cache[key] = entry
            if self.needs_logged:
                self.needs_logged(key, obj)
            return False

    def add(self, key: K, obj: Object) -> None:
        """
        Add an object to the cache (or update if exists).

        Note: This method fires needs_logged for new objects, but not for updates.
        For attribute change detection and periodic resend, use contains_or_add().

        Args:
            key: The object ID
            obj: The OCEL Object
        """
        if key in self._cache:
            # Update existing entry
            entry = self._cache[key]
            entry.object = obj
            entry.sight_count += 1
            self._cache.move_to_end(key)
        else:
            # New entry
            self._evict_if_needed()

            attr_hash = _compute_attributes_hash(obj)
            entry = CacheEntry(object=obj, attributes_hash=attr_hash)
            self._cache[key] = entry

            if self.needs_logged:
                self.needs_logged(key, obj)

    def remove(self, key: K) -> Optional[Object]:
        """
        Remove an object from the cache.

        Args:
            key: The object ID

        Returns:
            The removed Object, or None if key wasn't in cache
        """
        return self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all objects from the cache without logging."""
        self._cache.clear()

    def __len__(self) -> int:
        """Return the number of objects in the cache."""
        return len(self._cache)

    def __contains__(self, key: K) -> bool:
        """Check if key is in cache."""
        return key in self._cache

    def keys(self) -> list[K]:
        """Return all keys in the cache."""
        return list(self._cache.keys())

    def values(self) -> list[Object]:
        """Return all objects in the cache."""
        return [entry.object for entry in self._cache.values()]

    def items(self) -> list[tuple[K, Object]]:
        """Return all (key, object) pairs in the cache."""
        return [(key, entry.object) for key, entry in self._cache.items()]
