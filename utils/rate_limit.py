"""Minimal in-process sliding-window rate limiter.

Deliberately dependency-free. This is per-worker state, so the effective limit
across an N-worker deployment is N * rate_limit_requests. Put a shared limiter
(nginx, Cloudflare, an API gateway) in front for a hard global cap.
"""

import threading
import time
from collections import defaultdict, deque


class SlidingWindowRateLimiter:
    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._last_sweep = 0.0

    def check(self, key: str) -> tuple[bool, float]:
        """Record a hit for `key`.

        Returns (allowed, retry_after_seconds). retry_after is 0 when allowed.
        """
        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            self._sweep(now, cutoff)
            hits = self._hits[key]
            while hits and hits[0] <= cutoff:
                hits.popleft()

            if len(hits) >= self.max_requests:
                retry_after = hits[0] + self.window_seconds - now
                return False, max(retry_after, 1.0)

            hits.append(now)
            return True, 0.0

    def _sweep(self, now: float, cutoff: float) -> None:
        """Drop idle keys so the dict cannot grow without bound."""
        if now - self._last_sweep < self.window_seconds:
            return
        self._last_sweep = now
        for key in [k for k, v in self._hits.items() if not v or v[-1] <= cutoff]:
            del self._hits[key]


def client_key(scope_client: tuple[str, int] | None, forwarded_for: str | None) -> str:
    """Best-effort client identity.

    X-Forwarded-For is only meaningful behind a proxy that overwrites it. Direct
    exposure to the internet means it is caller-controlled and effectively
    disables limiting, so keep this service behind a proxy.
    """
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if scope_client:
        return scope_client[0]
    return "unknown"
