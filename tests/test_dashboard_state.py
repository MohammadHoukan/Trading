import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dashboard.state import (
    DEFAULT_STALE_AFTER_SECONDS,
    broadcast_stop,
    parse_worker_rows,
)


def test_broadcast_stop_success():
    mock_redis = MagicMock()
    mock_redis.publish.return_value = 2

    ok, message = broadcast_stop(mock_redis, 'swarm:cmd')

    assert ok is True
    assert "delivered to 2" in message


def test_broadcast_stop_no_subscribers():
    mock_redis = MagicMock()
    mock_redis.publish.return_value = 0

    ok, message = broadcast_stop(mock_redis, 'swarm:cmd')

    assert ok is False
    assert "no active command subscribers" in message


def test_broadcast_stop_redis_error():
    mock_redis = MagicMock()
    mock_redis.publish.side_effect = RuntimeError("redis down")

    ok, message = broadcast_stop(mock_redis, 'swarm:cmd')

    assert ok is False
    assert "Failed to broadcast STOP" in message


def test_parse_worker_rows_filters_stale_and_bad_records():
    now_ts = 10_000.0
    workers_raw = {
        'w_active': '{"worker_id":"w_active","status":"RUNNING","last_updated":9990.0}',
        'w_stale': '{"worker_id":"w_stale","status":"RUNNING","last_updated":9900.0}',
        'w_no_ts': '{"worker_id":"w_no_ts","status":"RUNNING"}',
        'w_bad': '{not-json}',
    }

    active, stale, malformed = parse_worker_rows(
        workers_raw,
        now_ts=now_ts,
        stale_after_seconds=DEFAULT_STALE_AFTER_SECONDS,
    )

    assert len(active) == 1
    assert active[0]['worker_id'] == 'w_active'
    assert active[0]['is_stale'] is False

    stale_ids = {row['worker_id'] for row in stale}
    assert stale_ids == {'w_stale', 'w_no_ts'}
    assert malformed == 1
