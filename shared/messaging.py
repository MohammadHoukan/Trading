import redis
import json
import logging

class RedisBus:
    def __init__(self, host='localhost', port=6379, db=0):
        self.r = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.logger = logging.getLogger(__name__)

    def publish(self, channel, message):
        """Publish a dictionary message to a channel."""
        try:
            payload = json.dumps(message)
            self.r.publish(channel, payload)
            return True
        except Exception as e:
            self.logger.error(f"Failed to publish to {channel}: {e}")
            return False

    def subscribe(self, channel):
        """Return a pubsub object subscribed to a channel."""
        pubsub = self.r.pubsub()
        pubsub.subscribe(channel)
        return pubsub

    def get_message(self, pubsub):
        """Non-blocking read of a message from pubsub."""
        try:
            message = pubsub.get_message()
        except Exception as e:
            self.logger.error(f"Failed to read pubsub message: {e}")
            raise
        if message and message['type'] == 'message':
            try:
                return json.loads(message['data'])
            except json.JSONDecodeError:
                self.logger.error("Failed to decode message")
        return None

    def hset(self, name, key, value):
        """Set a hash field to a value."""
        try:
            self.r.hset(name, key, value)
            return True
        except Exception as e:
            self.logger.error(f"Failed to hset {name}[{key}]: {e}")
            return False

    def hgetall(self, name):
        """Get all fields from a hash."""
        try:
            return self.r.hgetall(name)
        except Exception as e:
            self.logger.error(f"Failed to hgetall {name}: {e}")
            return None

    def set(self, key, value, nx=False, ex=None):
        """Set key/value with optional NX and expiry."""
        try:
            return self.r.set(key, value, nx=nx, ex=ex)
        except Exception as e:
            self.logger.error(f"Failed to set {key}: {e}")
            return False

    def expire(self, key, seconds):
        """Update key expiry."""
        try:
            return self.r.expire(key, seconds)
        except Exception as e:
            self.logger.error(f"Failed to expire {key}: {e}")
            return False
