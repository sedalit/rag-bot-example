"""In-memory per-session rate limiting."""

import time
from collections import defaultdict, deque


class RateLimiter:
    """Sliding-window rate limiter keyed by an arbitrary string (session id)."""

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        if max_requests < 1:
            raise ValueError("max_requests must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, now: float | None = None) -> bool:
        """Return True and record the request if it fits within the window."""
        now = time.monotonic() if now is None else now
        window = self._hits[key]
        while window and now - window[0] >= self._window_seconds:
            window.popleft()
        if len(window) >= self._max_requests:
            return False
        window.append(now)
        return True
