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

    # --- Redis Streams for reliable command delivery ---
    
    def xadd(self, stream, message, maxlen=1000):
        """
        Add message to a Redis Stream.
        
        Args:
            stream: Stream name
            message: Dict of field-value pairs
            maxlen: Maximum stream length (auto-trimmed)
            
        Returns:
            Message ID or None on error
        """
        try:
            return self.r.xadd(stream, message, maxlen=maxlen)
        except Exception as e:
            self.logger.error(f"Failed to xadd to {stream}: {e}")
            return None

    def create_consumer_group(self, stream, group, start_id='0'):
        """
        Create a consumer group for a stream.
        
        Args:
            stream: Stream name
            group: Consumer group name
            start_id: Starting message ID ('0' for all, '$' for new only)
        """
        try:
            self.r.xgroup_create(stream, group, id=start_id, mkstream=True)
            return True
        except Exception as e:
            # Group already exists is OK
            if 'BUSYGROUP' in str(e):
                return True
            self.logger.error(f"Failed to create consumer group {group}: {e}")
            return False

    def xreadgroup(self, group, consumer, stream, count=1, block=1000):
        """
        Read from stream with consumer group (guaranteed delivery).
        
        Args:
            group: Consumer group name
            consumer: Consumer name (usually worker_id)
            stream: Stream name
            count: Max messages to read
            block: Block timeout in ms (0 = no block)
            
        Returns:
            List of (message_id, fields) tuples or empty list
        """
        try:
            result = self.r.xreadgroup(
                group, consumer, 
                {stream: '>'}, 
                count=count, 
                block=block
            )
            if result:
                # Result format: [[stream_name, [(id, fields), ...]]]
                return [(msg_id, fields) for msg_id, fields in result[0][1]]
            return []
        except Exception as e:
            self.logger.error(f"Failed to xreadgroup from {stream}: {e}")
            return []

    def xack(self, stream, group, message_id):
        """
        Acknowledge message processing.
        
        Args:
            stream: Stream name
            group: Consumer group name
            message_id: Message ID to acknowledge
        """
        try:
            return self.r.xack(stream, group, message_id)
        except Exception as e:
            self.logger.error(f"Failed to xack {message_id}: {e}")
            return 0

    def xautoclaim(self, stream, group, consumer, min_idle_time, start_id='0-0', count=10):
        """
        Auto-claim stale pending messages for a consumer.

        Returns:
            List of (message_id, fields) tuples
            None if xautoclaim is unsupported by Redis server/client
        """
        try:
            result = self.r.xautoclaim(
                stream,
                group,
                consumer,
                min_idle_time=min_idle_time,
                start_id=start_id,
                count=count,
            )
            # redis-py returns (next_start_id, [(id, fields), ...], [deleted_ids?]).
            # Note: Depending on version/client options, it might be a list or tuple.
            
            # Normalize list to tuple if needed for unpacking, OR inspect by index
            if isinstance(result, list):
                result = tuple(result)
            
            if not isinstance(result, tuple) or len(result) < 2:
                self.logger.error(f"Unexpected xautoclaim result format: {result}")
                return []
                
            messages = result[1]
            if not messages:
                return []
            return [(msg_id, fields) for msg_id, fields in messages]
        except AttributeError:
            return None
        except Exception as e:
            text = str(e).lower()
            if 'unknown command' in text or 'xautoclaim' in text and 'unsupported' in text:
                return None
            self.logger.error(f"Failed to xautoclaim from {stream}: {e}")
            return []

    def xpending_range(self, stream, group, min_id='-', max_id='+', count=10, consumer=None):
        """Fetch pending entries metadata for a stream consumer group."""
        try:
            return self.r.xpending_range(
                stream,
                group,
                min=min_id,
                max=max_id,
                count=count,
                consumername=consumer,
            )
        except Exception as e:
            self.logger.error(f"Failed to xpending_range from {stream}: {e}")
            return []

    def xclaim(self, stream, group, consumer, min_idle_time, message_ids):
        """Claim pending messages and return list of (message_id, fields)."""
        if not message_ids:
            return []
        try:
            claimed = self.r.xclaim(
                stream,
                group,
                consumer,
                min_idle_time=min_idle_time,
                message_ids=message_ids,
            )
            if not claimed:
                return []
            return [(msg_id, fields) for msg_id, fields in claimed]
        except Exception as e:
            self.logger.error(f"Failed to xclaim from {stream}: {e}")
            return []
