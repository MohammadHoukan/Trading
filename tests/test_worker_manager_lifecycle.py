import json
import os
import sys
import unittest
from collections import defaultdict
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from manager.orchestrator import Orchestrator
from manager.risk_engine import RiskEngine


class _FakePubSub:
    def __init__(self):
        self.queue = []


class InMemoryBus:
    """Minimal in-memory RedisBus replacement for lifecycle integration tests."""

    def __init__(self):
        self.hashes = defaultdict(dict)
        self.subscribers = defaultdict(list)
        self.streams = defaultdict(list)
        self.group_offsets = {}
        self.stream_seq = defaultdict(int)

    def publish(self, channel, message):
        for subscriber in self.subscribers[channel]:
            subscriber.queue.append(message.copy() if isinstance(message, dict) else message)
        return True

    def subscribe(self, channel):
        pubsub = _FakePubSub()
        self.subscribers[channel].append(pubsub)
        return pubsub

    def get_message(self, pubsub):
        if pubsub.queue:
            return pubsub.queue.pop(0)
        return None

    def hset(self, name, key, value):
        self.hashes[name][key] = value
        return True

    def hgetall(self, name):
        return dict(self.hashes.get(name, {}))

    def xadd(self, stream, message, maxlen=1000):
        self.stream_seq[stream] += 1
        msg_id = f"{self.stream_seq[stream]}-0"
        entries = self.streams[stream]
        entries.append((msg_id, message.copy() if isinstance(message, dict) else message))
        if maxlen and len(entries) > maxlen:
            self.streams[stream] = entries[-maxlen:]
        return msg_id

    def create_consumer_group(self, stream, group, start_id='0'):
        key = (stream, group)
        if key in self.group_offsets:
            return True
        if start_id == '$':
            self.group_offsets[key] = len(self.streams[stream]) - 1
        else:
            self.group_offsets[key] = -1
        return True

    def xreadgroup(self, group, consumer, stream, count=1, block=1000):
        _ = (consumer, block)  # Unused by this test double.
        key = (stream, group)
        offset = self.group_offsets.get(key, -1)
        entries = self.streams.get(stream, [])
        unread = entries[offset + 1: offset + 1 + count]
        if unread:
            self.group_offsets[key] = offset + len(unread)
        return unread

    def xack(self, stream, group, message_id):
        _ = (stream, group, message_id)
        return 1

    def xautoclaim(self, stream, group, consumer, min_idle_time, start_id='0-0', count=10):
        _ = (stream, group, consumer, min_idle_time, start_id, count)
        return []

    def xpending_range(self, stream, group, min_id='-', max_id='+', count=10, consumer=None):
        _ = (stream, group, min_id, max_id, count, consumer)
        return []

    def xclaim(self, stream, group, consumer, min_idle_time, message_ids):
        _ = (stream, group, consumer, min_idle_time, message_ids)
        return []


class TestWorkerManagerLifecycle(unittest.TestCase):
    @patch('workers.grid_bot.RateLimiter')
    @patch('workers.grid_bot.OrderManager')
    @patch('workers.grid_bot.Database')
    @patch('workers.grid_bot.RedisBus')
    @patch('workers.grid_bot.load_config')
    def test_stop_lifecycle_unregisters_worker_and_persists_terminal_state(
        self,
        mock_load_config,
        mock_redis_bus,
        mock_database_cls,
        mock_order_manager_cls,
        mock_rate_limiter_cls,
    ):
        shared_bus = InMemoryBus()

        config = {
            'exchange': {
                'name': 'binance',
                'mode': 'testnet',
                'api_key': 'k',
                'secret': 's',
            },
            'redis': {
                'channels': {
                    'command': 'cmd',
                    'status': 'stat',
                }
            },
            'swarm': {
                'risk_per_bot': 100.0,
                'max_global_capital': 100.0,
                'max_concurrency': 5,
            },
        }

        mock_load_config.return_value = config
        mock_redis_bus.return_value = shared_bus
        mock_database_cls.return_value = MagicMock()
        mock_rate_limiter_cls.return_value = MagicMock()

        om = MagicMock()
        om.fetch_open_orders.return_value = []
        mock_order_manager_cls.return_value = om

        from workers.grid_bot import GridBot

        with patch.object(
            GridBot,
            '_load_strategy_params',
            return_value={
                'enabled': True,
                'grid_levels': 4,
                'lower_limit': 10.0,
                'upper_limit': 30.0,
                'amount_per_grid': 1.0,
            },
        ):
            bot = GridBot('SOL/USDT', 4)

        orch = Orchestrator.__new__(Orchestrator)
        orch.config = {'redis': {'channels': {'command': 'cmd'}}}
        orch.bus = shared_bus
        orch.risk_engine = RiskEngine(config)
        orch.logger = MagicMock()
        orch.stop_broadcast_sent = False
        orch.rejected_workers = set()

        status_sub = shared_bus.subscribe('stat')

        # 1) Worker reports RUNNING status and manager registers exposure.
        bot.inventory = 10.0  # exposure = 10 * 20 = 200 > global cap 100
        bot.report_status(20.0)
        running_msg = shared_bus.get_message(status_sub)
        self.assertIsNotNone(running_msg)
        self.assertEqual(running_msg['status'], 'RUNNING')
        orch.handle_worker_update(running_msg)

        self.assertIn(bot.worker_id, orch.risk_engine.active_bots)
        self.assertEqual(orch.risk_engine.allocations[bot.worker_id], 200.0)

        # 2) Manager sends STOP (stream+pubsub) on risk breach.
        orch.perform_risk_checks()
        stream_messages = shared_bus.streams.get('swarm:commands', [])
        self.assertGreaterEqual(len(stream_messages), 1)
        self.assertEqual(stream_messages[-1][1].get('command'), 'STOP')
        self.assertEqual(stream_messages[-1][1].get('target'), 'all')

        # 3) Worker consumes STOP from stream and publishes terminal status.
        bot._check_stream_commands()
        self.assertFalse(bot.running)
        om.cancel_order.assert_not_called()

        terminal_msg = shared_bus.get_message(status_sub)
        self.assertIsNotNone(terminal_msg)
        self.assertEqual(terminal_msg['status'], 'STOPPED')

        # 4) Manager handles terminal status, unregisters worker, persists snapshot.
        orch.handle_worker_update(terminal_msg)
        self.assertNotIn(bot.worker_id, orch.risk_engine.active_bots)
        self.assertNotIn(bot.worker_id, orch.risk_engine.allocations)

        worker_data = shared_bus.hgetall('workers:data')
        self.assertIn(bot.worker_id, worker_data)
        persisted = json.loads(worker_data[bot.worker_id])
        self.assertEqual(persisted['status'], 'STOPPED')


if __name__ == '__main__':
    unittest.main()
