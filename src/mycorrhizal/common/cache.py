"""
Shared cache implementations for mycorrhizal runtimes.

Provides LRU cache implementations for use across Hypha, Rhizomorph, and Septum.
"""
from collections import OrderedDict
from typing import Any, Dict, Tuple, Type, Optional


class InterfaceViewCache:
    """
    LRU cache for interface views bounded by maximum size.

    This cache stores interface view objects keyed by (blackboard_id, interface_type, readonly_fields).
    Uses OrderedDict for manual LRU tracking with bounded size.

    Thread-safety: Safe for single reader/writer (async environments).
    For multi-threaded environments, external synchronization is required.

    Example:
        cache = InterfaceViewCache(maxsize=256)
        view = cache.get_or_create(bb_id, interface_type, readonly_fields, creator_func)
        stats = cache.get_stats()
    """

    def __init__(self, maxsize: int = 256):
        """
        Initialize the LRU cache.

        Args:
            maxsize: Maximum number of entries to cache. Defaults to 256.
        """
        self._maxsize = maxsize
        self._cache: OrderedDict[Tuple[int, Type, Tuple[str, ...]], Any] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get_or_create(
        self,
        bb_id: int,
        interface_type: Type,
        readonly_fields: Optional[Tuple[str, ...]],
        creator_func: callable
    ) -> Any:
        """
        Get cached view or create using provided function.

        Args:
            bb_id: ID of blackboard (from id() builtin)
            interface_type: The interface protocol class
            readonly_fields: Tuple of readonly field names
            creator_func: Function that creates the view if not cached

        Returns:
            The cached or newly created interface view
        """
        # Create cache key from hashable components
        readonly_fields_tuple = tuple(sorted(readonly_fields)) if readonly_fields else ()
        cache_key = (bb_id, interface_type, readonly_fields_tuple)

        # Check cache
        if cache_key in self._cache:
            # Cache hit - move to end (most recently used)
            self._cache.move_to_end(cache_key)
            self._hits += 1
            return self._cache[cache_key]

        # Cache miss - create view and cache it
        self._misses += 1
        view = creator_func()

        # Add to cache and move to end
        self._cache[cache_key] = view
        self._cache.move_to_end(cache_key)

        # Evict oldest entry if over limit
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

        return view

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for monitoring and debugging.

        Returns:
            Dict with current cache size, maxsize, and hit/miss counts
        """
        return {
            'cached_views': len(self._cache),
            'maxsize': self._maxsize,
            'hits': self._hits,
            'misses': self._misses,
        }

    def clear(self) -> None:
        """
        Clear all cached entries and reset statistics.

        Useful for testing and memory management.
        """
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def maxsize(self) -> int:
        """Get maximum cache size."""
        return self._maxsize

    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)
