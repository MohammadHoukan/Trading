
import unittest
from unittest.mock import MagicMock, patch
import json
import time

# Use patch dict to mock config correctly if needed, but here we can mock the instance
from workers.grid_bot import GridBot

class TestSystemHardening(unittest.TestCase):
    def setUp(self):
        # Mock dependencies
        self.patcher_config = patch('workers.grid_bot.load_config')
        self.mock_config = self.patcher_config.start()
        self.mock_config.return_value = {
            'exchange': {'pool': [{'api_key': 'k', 'secret': 's'}], 'name': 'binance', 'mode': 'live'}, # Enable pool & set name/mode
            'redis': {'host': 'localhost', 'port': 6379, 'channels': {'status': 'status_chan'}}
        }
        
        self.patcher_redis = patch('workers.grid_bot.RedisBus')
        self.MockRedis = self.patcher_redis.start()
        self.mock_bus = self.MockRedis.return_value
        self.mock_bus.subscribe.return_value = MagicMock()
        
        # Initialize bot (it will try to connect)
        # We need to mock _get_redis_params too to avoid connection errors if connect() is called
        with patch('workers.grid_bot.get_redis_params', return_value={}):
             # Also mock os.getenv to avoid HOSTNAME issues
            with patch('os.getenv', return_value='worker-1'):
                 self.bot = GridBot('SOL/USDT', 10)

        # Mock OrderManager
        self.bot.order_manager = MagicMock()

    def tearDown(self):
        self.patcher_config.stop()
        self.patcher_redis.stop()

    def test_terminal_state_persistence(self):
        """Test that _publish_terminal_status writes to Redis Hash."""
        self.bot.bus.hset = MagicMock(return_value=True)
        self.bot.bus.publish = MagicMock(return_value=True)
        
        self.bot._publish_terminal_status('STOPPED')
        
        # Check publish called
        self.assertTrue(self.bot.bus.publish.called)
        
        # Check hset called (The Fix)
        self.assertTrue(self.bot.bus.hset.called)
        args, _ = self.bot.bus.hset.call_args
        self.assertEqual(args[0], 'workers:data')
        self.assertEqual(args[1], self.bot.worker_id)
        data = json.loads(args[2])
        self.assertEqual(data['status'], 'STOPPED')

    def test_place_order_guard_clause(self):
        """Test place_order returns False if key lock is lost."""
        self.bot.use_pool = True
        self.bot.key_lock_id = None # Simulate lost lock
        self.bot.grids = [{'price': 100.0, 'orders': []}] * 2 # Mock grids
        
        result = self.bot.place_order(1, 'buy', 1.0)
        
        self.assertFalse(result)
        self.assertFalse(self.bot.order_manager.create_limit_buy.called)
        self.assertFalse(self.bot.running) # Should force stop

    def test_place_order_success_with_lock(self):
        """Test place_order fails on lost lock but succeeds (mocked) with lock."""
        self.bot.use_pool = True
        self.bot.key_lock_id = 'lock-123'
        self.bot._has_existing_order = MagicMock(return_value=False)
        self.bot.grids = [{'price': 100.0, 'orders': []}] * 2 # Mock grids
        self.bot.active_orders = set()
        self.bot._log_grid_event = MagicMock()
        
        self.bot.place_order(1, 'buy', 1.0)
        
        self.assertTrue(self.bot.order_manager.create_limit_buy.called)

if __name__ == '__main__':
    unittest.main()
