"""
Redis-backed Token Bucket Rate Limiter for global API coordination.

Uses a sliding window approach with Redis INCR + EXPIRE to coordinate
rate limiting across multiple workers in the swarm.
"""
import time
import logging


class RateLimiter:
    """
    Global token bucket rate limiter using Redis.
    
    Coordinates API calls across all workers to prevent IP bans
    from exchange rate limits (typically 10-20 req/sec).
    """
    
    def __init__(self, redis_bus, max_requests=10, window_seconds=1, key_prefix='ratelimit:api'):
        """
        Initialize rate limiter.
        
        Args:
            redis_bus: RedisBus instance for Redis access
            max_requests: Maximum requests allowed per window
            window_seconds: Window duration in seconds
            key_prefix: Redis key prefix for rate limit counters
        """
        self.redis = redis_bus.r  # Direct Redis client
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.key_prefix = key_prefix
        self.logger = logging.getLogger(__name__)
    
    def _get_window_key(self):
        """Get the current window's Redis key."""
        window_id = int(time.time() / self.window_seconds)
        return f"{self.key_prefix}:{window_id}"
    
    def acquire(self, timeout=5.0) -> bool:
        """
        Acquire a rate limit token. Blocks until available or timeout.
        
        Args:
            timeout: Maximum seconds to wait for a token
            
        Returns:
            True if token acquired, False if timeout
        """
        start_time = time.time()
        
        while True:
            try:
                key = self._get_window_key()
                
                # Atomic increment
                current = self.redis.incr(key)
                
                # Set expiry on first request in window
                if current == 1:
                    self.redis.expire(key, self.window_seconds + 1)
                
                if current <= self.max_requests:
                    return True
                
                # Over limit - wait and retry
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    self.logger.warning(f"Rate limit acquire timeout after {timeout}s")
                    return False
                
                # Calculate wait time until next window
                wait_time = min(
                    self.window_seconds - (time.time() % self.window_seconds),
                    timeout - elapsed
                )
                if wait_time > 0:
                    self.logger.debug(f"Rate limited, waiting {wait_time:.2f}s for next window")
                    time.sleep(wait_time)
                    
            except Exception as e:
                self.logger.error(f"Rate limiter error: {e}")
                # On Redis error, allow the request (fail open to prevent deadlock)
                return True
    
    def get_remaining(self) -> int:
        """Get remaining tokens in current window."""
        try:
            key = self._get_window_key()
            current = self.redis.get(key)
            if current is None:
                return self.max_requests
            return max(0, self.max_requests - int(current))
        except Exception:
            return self.max_requests
