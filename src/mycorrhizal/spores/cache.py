#!/usr/bin/env python3
"""
Spores Object Cache

LRU cache for tracking seen objects and logging them on eviction.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Callable, Optional, TypeVar, Generic
from dataclasses import dataclass

from .models import Object


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
    """
    object: Object
    sight_count: int = 1
    first_sight_time: Optional[float] = None


class ObjectLRUCache(Generic[K, V]):
    """
    LRU cache with eviction callback for object tracking.

    When an object is first seen, it should be logged via on_first_sight.
    When an object is evicted from the cache, it's logged via on_evict.

    This ensures OCEL consumers see object evolution:
    - First sight: Initial object state
    - Eviction: Final state before being removed from cache

    Example:
        ```python
        def on_evict(object_id: str, obj: Object):
            # Send object to transport
            transport.send(LogRecord(object=obj))

        cache = ObjectLRUCache(maxsize=128, on_evict=on_evict)

        # Check if object exists, add if not
        if not cache.contains_or_add(object_id, object):
            # Object not in cache, already logged by on_first_sight
            pass
        ```

    Args:
        maxsize: Maximum number of objects to cache
        on_evict: Callback when object is evicted (object_id, object) -> None
        on_first_sight: Optional callback when object is first seen (object_id, object) -> None
    """

    def __init__(
        self,
        maxsize: int = 128,
        on_evict: Optional[Callable[[K, Object], None]] = None,
        on_first_sight: Optional[Callable[[K, Object], None]] = None
    ):
        self.maxsize = maxsize
        self.on_evict = on_evict
        self.on_first_sight = on_first_sight
        self._cache: OrderedDict[K, CacheEntry] = OrderedDict()

    def _evict_if_needed(self) -> None:
        """Evict oldest entry if cache is full."""
        if len(self._cache) >= self.maxsize:
            # FIFO from OrderedDict (oldest first)
            object_id, entry = self._cache.popitem(last=False)
            if self.on_evict:
                self.on_evict(object_id, entry.object)

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
        - If key exists: mark as recently used, return True
        - If key doesn't exist: add object, potentially evict, call on_first_sight, return False

        Args:
            key: The object ID
            obj: The OCEL Object to add if not present

        Returns:
            True if object was already in cache, False if it was just added
        """
        if key in self._cache:
            # Already seen, mark as recently used
            self._cache.move_to_end(key)
            entry = self._cache[key]
            entry.sight_count += 1
            return True
        else:
            # First sight
            self._evict_if_needed()

            entry = CacheEntry(object=obj)
            self._cache[key] = entry

            if self.on_first_sight:
                self.on_first_sight(key, obj)

            return False

    def add(self, key: K, obj: Object) -> None:
        """
        Add an object to the cache (or update if exists).

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

            entry = CacheEntry(object=obj)
            self._cache[key] = entry

            if self.on_first_sight:
                self.on_first_sight(key, obj)

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
