"""Thread-safe in-memory cache with TTL expiration and LRU eviction."""

import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Hashable, Optional, Tuple


class CacheEntry:
    """A single cache entry storing value, metadata, and expiration."""

    __slots__ = ("key", "value", "created_at", "expires_at", "hit_count")

    def __init__(self, key: Hashable, value: Any, ttl: Optional[float] = None):
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.expires_at = self.created_at + ttl if ttl else None
        self.hit_count = 0

    def is_expired(self) -> bool:
        """Check if this entry has passed its TTL."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def time_remaining(self) -> Optional[float]:
        """Return seconds until expiration, or None if no TTL."""
        if self.expires_at is None:
            return None
        remaining = self.expires_at - time.time()
        return max(0.0, remaining)


class LRUCache:
    """Least-Recently-Used cache with optional TTL expiration.

    Thread-safe implementation using OrderedDict for O(1) access
    and eviction. Supports maximum size limits and per-entry TTL.

    Example:
        cache = LRUCache(max_size=100, default_ttl=300.0)
        cache.put("user:123", {"name": "Alice"})
        user = cache.get("user:123")
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[float] = None,
    ):
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._store: OrderedDict[Hashable, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: Hashable) -> Optional[Any]:
        """Retrieve a value by key.

        Moves the entry to the end (most recently used) on access.
        Returns None if the key is missing or expired.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.is_expired():
                del self._store[key]
                self._misses += 1
                return None
            self._store.move_to_end(key)
            entry.hit_count += 1
            self._hits += 1
            return entry.value

    def put(
        self, key: Hashable, value: Any, ttl: Optional[float] = None
    ) -> None:
        """Insert or update a cache entry.

        If the cache is full, evicts the least recently used entry.

        Args:
            key: The cache key.
            value: The value to store.
            ttl: Time-to-live in seconds. Uses default_ttl if None.
        """
        effective_ttl = ttl if ttl is not None else self._default_ttl
        with self._lock:
            if key in self._store:
                del self._store[key]
            elif len(self._store) >= self._max_size:
                self._store.popitem(last=False)
            self._store[key] = CacheEntry(key, value, effective_ttl)

    def delete(self, key: Hashable) -> bool:
        """Remove an entry by key. Returns True if the key existed."""
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        """Remove all entries from the cache."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    def evict_expired(self) -> int:
        """Remove all expired entries. Returns the count removed."""
        with self._lock:
            expired_keys = [
                k for k, v in self._store.items() if v.is_expired()
            ]
            for k in expired_keys:
                del self._store[k]
            return len(expired_keys)

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        """Return the cache hit rate as a fraction between 0 and 1."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def stats(self) -> dict:
        """Return cache performance statistics."""
        return {
            "size": self.size,
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }

    def get_or_compute(
        self,
        key: Hashable,
        compute_fn: Callable[[], Any],
        ttl: Optional[float] = None,
    ) -> Any:
        """Get from cache or compute and store the value.

        This is a convenience method that checks the cache first,
        and if the key is not found, calls compute_fn() to generate
        the value, stores it, and returns it.

        Args:
            key: The cache key.
            compute_fn: A callable that returns the value to cache.
            ttl: Optional TTL override.

        Returns:
            The cached or freshly computed value.
        """
        result = self.get(key)
        if result is not None:
            return result
        value = compute_fn()
        self.put(key, value, ttl)
        return value

    def keys(self):
        """Return a list of all non-expired keys."""
        with self._lock:
            self.evict_expired()
            return list(self._store.keys())

    def __contains__(self, key: Hashable) -> bool:
        with self._lock:
            entry = self._store.get(key)
            if entry is None or entry.is_expired():
                return False
            return True

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"LRUCache(size={self.size}, max_size={self._max_size})"
