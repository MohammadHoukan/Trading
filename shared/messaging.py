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
        except Exception as e:
            self.logger.error(f"Failed to publish to {channel}: {e}")

    def subscribe(self, channel):
        """Return a pubsub object subscribed to a channel."""
        pubsub = self.r.pubsub()
        pubsub.subscribe(channel)
        return pubsub

    def get_message(self, pubsub):
        """Non-blocking read of a message from pubsub."""
        message = pubsub.get_message()
        if message and message['type'] == 'message':
            try:
                return json.loads(message['data'])
            except json.JSONDecodeError:
                self.logger.error("Failed to decode message")
        return None
