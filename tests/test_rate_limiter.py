"""Unit tests for RateLimiter."""
import sys
import os
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_rate_limiter_acquire_success():
    """Should acquire token when under limit."""
    with patch('redis.Redis') as mock_redis:
        mock_instance = MagicMock()
        mock_redis.return_value = mock_instance
        mock_instance.incr.return_value = 1  # First request in window
        
        from shared.messaging import RedisBus
        from shared.rate_limiter import RateLimiter
        
        bus = RedisBus(host='localhost', port=6379, db=0)
        limiter = RateLimiter(bus, max_requests=10, window_seconds=1)
        
        result = limiter.acquire(timeout=1.0)
        
        assert result is True
        mock_instance.incr.assert_called_once()
        mock_instance.expire.assert_called_once()


def test_rate_limiter_get_remaining():
    """Should return remaining tokens in window."""
    with patch('redis.Redis') as mock_redis:
        mock_instance = MagicMock()
        mock_redis.return_value = mock_instance
        mock_instance.get.return_value = '3'  # 3 requests made
        
        from shared.messaging import RedisBus
        from shared.rate_limiter import RateLimiter
        
        bus = RedisBus(host='localhost', port=6379, db=0)
        limiter = RateLimiter(bus, max_requests=10, window_seconds=1)
        
        remaining = limiter.get_remaining()
        
        assert remaining == 7  # 10 - 3


def test_rate_limiter_get_remaining_empty_window():
    """Should return max tokens when no requests in window."""
    with patch('redis.Redis') as mock_redis:
        mock_instance = MagicMock()
        mock_redis.return_value = mock_instance
        mock_instance.get.return_value = None
        
        from shared.messaging import RedisBus
        from shared.rate_limiter import RateLimiter
        
        bus = RedisBus(host='localhost', port=6379, db=0)
        limiter = RateLimiter(bus, max_requests=10, window_seconds=1)
        
        remaining = limiter.get_remaining()
        
        assert remaining == 10


def test_rate_limiter_fail_closed_on_error():
    """Should deny request on Redis error (fail closed)."""
    with patch('redis.Redis') as mock_redis:
        mock_instance = MagicMock()
        mock_redis.return_value = mock_instance
        mock_instance.incr.side_effect = RuntimeError("redis down")
        
        from shared.messaging import RedisBus
        from shared.rate_limiter import RateLimiter
        
        bus = RedisBus(host='localhost', port=6379, db=0)
        limiter = RateLimiter(bus, max_requests=10, window_seconds=1)
        
        result = limiter.acquire(timeout=0.1)
        
        # Should fail closed to prevent IP bans
        assert result is False
