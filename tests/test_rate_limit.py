"""Unit tests for the sliding-window rate limiter."""

import pytest

from app.api.rate_limit import RateLimiter


def test_allow_under_limit():
    limiter = RateLimiter(max_requests=2, window_seconds=10)
    assert limiter.allow("s", now=0.0) is True
    assert limiter.allow("s", now=1.0) is True
    assert limiter.allow("s", now=2.0) is False


def test_window_evicts_old_requests():
    limiter = RateLimiter(max_requests=2, window_seconds=10)
    assert limiter.allow("s", now=0.0) is True
    assert limiter.allow("s", now=1.0) is True
    # Once the window has passed, old requests no longer count.
    assert limiter.allow("s", now=11.0) is True


def test_keys_are_isolated():
    limiter = RateLimiter(max_requests=1, window_seconds=10)
    assert limiter.allow("a", now=0.0) is True
    assert limiter.allow("b", now=0.0) is True
    assert limiter.allow("a", now=1.0) is False


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        RateLimiter(max_requests=0, window_seconds=10)
    with pytest.raises(ValueError):
        RateLimiter(max_requests=1, window_seconds=0)
