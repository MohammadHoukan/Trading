"""Unit tests for RedisBus messaging."""
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_redisbus_init():
    """RedisBus accepts only connection params (host, port, db)."""
    with patch('redis.Redis') as mock_redis:
        from shared.messaging import RedisBus
        
        bus = RedisBus(host='localhost', port=6379, db=0)
        
        mock_redis.assert_called_once_with(
            host='localhost', port=6379, db=0, decode_responses=True
        )


def test_redisbus_publish_serializes_json():
    """Publish serializes dict to JSON."""
    with patch('redis.Redis') as mock_redis:
        mock_instance = MagicMock()
        mock_redis.return_value = mock_instance
        
        from shared.messaging import RedisBus
        
        bus = RedisBus(host='localhost', port=6379, db=0)
        result = bus.publish('test_channel', {'key': 'value'})
        
        mock_instance.publish.assert_called_once_with(
            'test_channel', '{"key": "value"}'
        )
        assert result is True


def test_redisbus_publish_returns_false_on_redis_error():
    with patch('redis.Redis') as mock_redis:
        mock_instance = MagicMock()
        mock_instance.publish.side_effect = RuntimeError("redis down")
        mock_redis.return_value = mock_instance

        from shared.messaging import RedisBus

        bus = RedisBus(host='localhost', port=6379, db=0)
        result = bus.publish('test_channel', {'key': 'value'})

        assert result is False


def test_redisbus_get_message_deserializes_json():
    """get_message deserializes JSON from pubsub."""
    with patch('redis.Redis') as mock_redis:
        mock_instance = MagicMock()
        mock_redis.return_value = mock_instance
        
        from shared.messaging import RedisBus
        
        bus = RedisBus(host='localhost', port=6379, db=0)
        
        # Mock pubsub
        mock_pubsub = MagicMock()
        mock_pubsub.get_message.return_value = {
            'type': 'message',
            'data': '{"command": "STOP"}'
        }
        
        result = bus.get_message(mock_pubsub)
        
        assert result == {'command': 'STOP'}


def test_redisbus_get_message_returns_none_for_non_message():
    """get_message returns None for non-message types."""
    with patch('redis.Redis') as mock_redis:
        mock_instance = MagicMock()
        mock_redis.return_value = mock_instance
        
        from shared.messaging import RedisBus
        
        bus = RedisBus(host='localhost', port=6379, db=0)
        
        mock_pubsub = MagicMock()
        mock_pubsub.get_message.return_value = {
            'type': 'subscribe',
            'data': 1
        }
        
        result = bus.get_message(mock_pubsub)
        
        assert result is None


def test_redisbus_hset_returns_false_on_redis_error():
    with patch('redis.Redis') as mock_redis:
        mock_instance = MagicMock()
        mock_instance.hset.side_effect = RuntimeError("redis down")
        mock_redis.return_value = mock_instance

        from shared.messaging import RedisBus

        bus = RedisBus(host='localhost', port=6379, db=0)
        result = bus.hset('workers:data', 'w1', '{"x":1}')

        assert result is False


def test_redisbus_hgetall_returns_none_on_redis_error():
    with patch('redis.Redis') as mock_redis:
        mock_instance = MagicMock()
        mock_instance.hgetall.side_effect = RuntimeError("redis down")
        mock_redis.return_value = mock_instance

        from shared.messaging import RedisBus

        bus = RedisBus(host='localhost', port=6379, db=0)
        result = bus.hgetall('workers:data')

        assert result is None
