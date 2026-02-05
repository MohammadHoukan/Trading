
import unittest
from unittest.mock import MagicMock, patch, call
import sys
import os
import json

# Path hack
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from workers.grid_bot import GridBot

class MockOrderManager:
    def __init__(self):
        self.orders = {}
        self.id_counter = 0
        self.fail_cancel_ids = set()

    def fetch_ticker(self, symbol):
        return {'last': 20.0}

    def create_limit_buy(self, symbol, amount, price):
        self.id_counter += 1
        oid = f"mock_bt_{self.id_counter}"
        self.orders[oid] = {
            'id': oid, 'symbol': symbol, 'amount': amount, 'price': price,
            'side': 'buy', 'status': 'open', 'filled': 0.0
        }
        return self.orders[oid]

    def create_limit_sell(self, symbol, amount, price):
        self.id_counter += 1
        oid = f"mock_sl_{self.id_counter}"
        self.orders[oid] = {
            'id': oid, 'symbol': symbol, 'amount': amount, 'price': price,
            'side': 'sell', 'status': 'open', 'filled': 0.0
        }
        return self.orders[oid]

    def fetch_open_orders(self, symbol):
        return [o for o in self.orders.values() if o['status'] == 'open']

    def fetch_order(self, order_id, symbol):
        return self.orders.get(order_id)

    def cancel_order(self, order_id, symbol):
        if order_id in self.fail_cancel_ids:
            raise RuntimeError(f"cancel failed for {order_id}")
        if order_id in self.orders:
            self.orders[order_id]['status'] = 'canceled'
        return self.orders.get(order_id)

    # Helper to simulate fill
    def fill_order(self, order_id):
        if order_id in self.orders:
            self.orders[order_id]['status'] = 'closed'
            self.orders[order_id]['filled'] = self.orders[order_id]['amount']

    def close_without_fill(self, order_id):
        if order_id in self.orders:
            self.orders[order_id]['status'] = 'closed'
            self.orders[order_id]['filled'] = 0.0

    def expire_order(self, order_id):
        if order_id in self.orders:
            self.orders[order_id]['status'] = 'expired'
            self.orders[order_id]['filled'] = 0.0

class TestGridLogic(unittest.TestCase):
    def setUp(self):
        # Create a dummy config for testing
        self.strategy = {
            "SOL/USDT": {
                "enabled": True,
                "grid_levels": 4, # Small number for testing
                "lower_limit": 10.0,
                "upper_limit": 30.0,
                "amount_per_grid": 1.0,
                "stop_loss": 5.0
            }
        }

        self.config = {
            'exchange': {
                'name': 'binance',
                'mode': 'testnet',
                'api_key': 'k',
                'secret': 's',
            },
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'db': 0,
                'channels': {'command': 'cmd', 'status': 'stat'},
            },
            'swarm': {},
        }
        self.config_patch = patch('workers.grid_bot.load_config', return_value=self.config)
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)

        self.db_patch = patch('workers.grid_bot.Database')
        self.mock_db_cls = self.db_patch.start()
        self.addCleanup(self.db_patch.stop)
        self.mock_db_cls.return_value = MagicMock()
        
        # Override _load_strategy_params to return our test strategy
        GridBot._load_strategy_params = MagicMock(return_value=self.strategy['SOL/USDT'])
        
        # Mock Redis
        GridBot.bus = MagicMock()
        
        self.bot = GridBot("SOL/USDT", 4)
        self.bot.order_manager = MockOrderManager()
        # Disable logging to stdout to keep test clean
        self.bot.logger = MagicMock()

    def test_grid_calculation_and_initial_placement(self):
        # 4 levels, 10 to 30. Step = (30-10)/4 = 5.
        # Lines: 10, 15, 20, 25, 30.
        
        # Current price = 20.
        # 10: Buy
        # 15: Buy
        # 20: Skip (Buffer)
        # 25: Sell
        # 30: Sell
        
        self.bot.place_initial_orders(20.0)
        
        # Check active orders in mock manager
        open_orders = self.bot.order_manager.fetch_open_orders("SOL/USDT")
        
        # We expect 4 orders (skipping 20).
        self.assertEqual(len(open_orders), 4)
        
        prices = sorted([o['price'] for o in open_orders])
        self.assertEqual(prices, [10.0, 15.0, 25.0, 30.0])
        
        # Verify sides
        buys = [o for o in open_orders if o['side'] == 'buy']
        self.assertEqual([o['price'] for o in buys], [10.0, 15.0])
        
        sells = [o for o in open_orders if o['side'] == 'sell']
        self.assertEqual([o['price'] for o in sells], [25.0, 30.0])

    def test_rebalancing_buy_to_sell(self):
        # Scenario: Price drops to 15. Buy at 15 fills.
        self.bot.place_initial_orders(20.0)
        
        # Find Buy order at 15
        buy_order = next(o for o in self.bot.order_manager.fetch_open_orders("SOL/USDT") if o['price'] == 15.0)
        
        # Determine Grid Index for 15.0
        # Grids: 0=10, 1=15, 2=20, 3=25, 4=30
        # Index is 1.
        
        # Simulate Fill
        self.bot.order_manager.fill_order(buy_order['id'])
        
        # Run Check Orders
        self.bot.check_orders()
        
        # Expectation: 
        # Buy at 15 (Index 1) Filled.
        # Logic: Place SELL at Index 1+1 = 2 (Price 20.0).
        
        open_orders = self.bot.order_manager.fetch_open_orders("SOL/USDT")
        
        # Check if we have a Sell at 20.0
        sell_20 = [o for o in open_orders if o['price'] == 20.0 and o['side'] == 'sell']
        self.assertTrue(len(sell_20) > 0, "Should have placed a Sell at 20.0")

    def test_rebalancing_sell_to_buy(self):
        # Scenario: Price rises to 25. Sell at 25 fills.
        self.bot.place_initial_orders(20.0)
        
        # Find Sell order at 25
        sell_order = next(o for o in self.bot.order_manager.fetch_open_orders("SOL/USDT") if o['price'] == 25.0)
        
        # Index for 25 is 3.
        
        # Simulate Fill
        self.bot.order_manager.fill_order(sell_order['id'])
        
        # Run Check Orders
        self.bot.check_orders()
        
        # Expectation: 
        # Sell at 25 (Index 3) Filled.
        # Logic: Place BUY at Index 3-1 = 2 (Price 20.0).
        
        open_orders = self.bot.order_manager.fetch_open_orders("SOL/USDT")
        
        # Check if we have a Buy at 20.0
        buy_20 = [o for o in open_orders if o['price'] == 20.0 and o['side'] == 'buy']
        self.assertTrue(len(buy_20) > 0, "Should have placed a Buy at 20.0")

    def test_closed_without_fill_does_not_rebalance_or_change_inventory(self):
        self.bot.place_initial_orders(20.0)
        order = next(
            o for o in self.bot.order_manager.fetch_open_orders("SOL/USDT")
            if o['price'] == 15.0 and o['side'] == 'buy'
        )

        self.bot.order_manager.close_without_fill(order['id'])
        self.bot.check_orders()

        open_orders = self.bot.order_manager.fetch_open_orders("SOL/USDT")
        sell_20 = [o for o in open_orders if o['price'] == 20.0 and o['side'] == 'sell']

        self.assertEqual(self.bot.inventory, 0.0)
        self.assertEqual(self.bot.realized_profit, 0.0)
        self.assertEqual(len(sell_20), 0)
        self.assertNotIn(order['id'], self.bot.active_orders)

    def test_expired_order_removed_from_tracking(self):
        self.bot.place_initial_orders(20.0)
        order = next(
            o for o in self.bot.order_manager.fetch_open_orders("SOL/USDT")
            if o['price'] == 15.0 and o['side'] == 'buy'
        )

        self.bot.order_manager.expire_order(order['id'])
        self.bot.check_orders()

        self.assertNotIn(order['id'], self.bot.active_orders)
        tracked_ids = [
            tracked['id']
            for grid in self.bot.grids
            for tracked in grid['orders']
        ]
        self.assertNotIn(order['id'], tracked_ids)

    def test_cancel_open_orders_keeps_failed_cancels_tracked(self):
        self.bot.place_initial_orders(20.0)
        open_orders = self.bot.order_manager.fetch_open_orders("SOL/USDT")
        keep_open = next(o for o in open_orders if o['price'] == 10.0 and o['side'] == 'buy')
        self.bot.order_manager.fail_cancel_ids.add(keep_open['id'])

        self.bot.cancel_open_orders()

        remaining_open_ids = {o['id'] for o in self.bot.order_manager.fetch_open_orders("SOL/USDT")}
        self.assertEqual(remaining_open_ids, {keep_open['id']})
        self.assertIn(keep_open['id'], self.bot.active_orders)

    def test_apply_param_update_changes_grid_levels(self):
        """Test that _apply_param_update correctly recalculates grid with new parameters."""
        # Initial setup: 4 levels, 10-30
        self.bot.place_initial_orders(20.0)

        # Update to 6 levels, 15-35
        new_params = {
            'grid_levels': 6,
            'lower_limit': 15.0,
            'upper_limit': 35.0,
        }
        self.bot._apply_param_update(new_params)

        # Verify parameters were updated
        self.assertEqual(self.bot.grid_levels, 6)
        self.assertEqual(self.bot.strategy_params['lower_limit'], 15.0)
        self.assertEqual(self.bot.strategy_params['upper_limit'], 35.0)

        # Verify new grid was calculated (6 levels = 7 price points)
        self.assertEqual(len(self.bot.grids), 7)

        # Verify old orders were canceled and new orders placed
        new_orders = self.bot.order_manager.fetch_open_orders("SOL/USDT")
        # New orders should exist (not the same as initial)
        self.assertGreater(len(new_orders), 0)

    def test_apply_param_update_handles_invalid_params(self):
        """Test that _apply_param_update ignores invalid parameter keys."""
        self.bot.place_initial_orders(20.0)
        original_lower = self.bot.strategy_params['lower_limit']

        # Try to update with invalid keys
        new_params = {
            'invalid_key': 999,
            'another_bad_key': 'foo',
        }
        self.bot._apply_param_update(new_params)

        # Original params should be unchanged
        self.assertEqual(self.bot.strategy_params['lower_limit'], original_lower)

    def test_handle_message_update_params(self):
        """Test that handle_message correctly processes UPDATE_PARAMS command."""
        self.bot.place_initial_orders(20.0)

        msg = {
            'command': 'UPDATE_PARAMS',
            'target': self.bot.worker_id,
            'params': {'grid_levels': 8}
        }
        self.bot.handle_message(msg)

        # Verify grid_levels was updated
        self.assertEqual(self.bot.grid_levels, 8)

    def test_handle_message_update_scale(self):
        self.bot.place_initial_orders(20.0)
        before_ids = {o['id'] for o in self.bot.order_manager.fetch_open_orders("SOL/USDT")}

        msg = {
            'command': 'UPDATE_SCALE',
            'target': self.bot.worker_id,
            'scale': 0.5,
        }
        self.bot.handle_message(msg)

        self.assertAlmostEqual(self.bot.exposure_scale, 0.5, places=6)
        self.assertAlmostEqual(self.bot.strategy_params['amount_per_grid'], 0.5, places=6)
        after_ids = {o['id'] for o in self.bot.order_manager.fetch_open_orders("SOL/USDT")}
        self.assertEqual(after_ids, before_ids)

    def test_handle_message_update_scale_rejects_invalid_value(self):
        self.bot.place_initial_orders(20.0)
        before_scale = self.bot.exposure_scale
        before_amount = self.bot.strategy_params['amount_per_grid']

        msg = {
            'command': 'UPDATE_SCALE',
            'target': self.bot.worker_id,
            'scale': 0,
        }
        self.bot.handle_message(msg)

        self.assertEqual(self.bot.exposure_scale, before_scale)
        self.assertEqual(self.bot.strategy_params['amount_per_grid'], before_amount)

    def test_apply_param_update_rejects_invalid_grid_levels_type_without_clearing_orders(self):
        self.bot.place_initial_orders(20.0)
        before_ids = {o['id'] for o in self.bot.order_manager.fetch_open_orders("SOL/USDT")}
        before_levels = self.bot.grid_levels

        self.bot._apply_param_update({'grid_levels': 'not-an-int'})

        after_ids = {o['id'] for o in self.bot.order_manager.fetch_open_orders("SOL/USDT")}
        self.assertEqual(after_ids, before_ids)
        self.assertEqual(self.bot.grid_levels, before_levels)

    def test_apply_param_update_rejects_inverted_range_without_clearing_orders(self):
        self.bot.place_initial_orders(20.0)
        before_ids = {o['id'] for o in self.bot.order_manager.fetch_open_orders("SOL/USDT")}
        before_limits = (
            self.bot.strategy_params['lower_limit'],
            self.bot.strategy_params['upper_limit'],
        )

        self.bot._apply_param_update({'lower_limit': 30.0, 'upper_limit': 10.0})

        after_ids = {o['id'] for o in self.bot.order_manager.fetch_open_orders("SOL/USDT")}
        self.assertEqual(after_ids, before_ids)
        self.assertEqual(
            (self.bot.strategy_params['lower_limit'], self.bot.strategy_params['upper_limit']),
            before_limits,
        )

    def test_check_stream_commands_processes_recovered_before_new_messages(self):
        self.bot.handle_message = MagicMock()
        self.bot.bus.xautoclaim = MagicMock(return_value=[
            ('1-0', {'command': 'PAUSE', 'target': self.bot.worker_id}),
        ])
        self.bot.bus.xreadgroup = MagicMock(return_value=[
            ('2-0', {'command': 'RESUME', 'target': self.bot.worker_id}),
        ])
        self.bot.bus.xack = MagicMock(return_value=1)

        self.bot._check_stream_commands()

        self.bot.handle_message.assert_has_calls([
            call({'command': 'PAUSE', 'target': self.bot.worker_id}),
            call({'command': 'RESUME', 'target': self.bot.worker_id}),
        ])
        self.assertEqual(self.bot.bus.xack.call_count, 2)

    def test_check_stream_commands_fallbacks_to_xclaim_when_xautoclaim_unsupported(self):
        self.bot.handle_message = MagicMock()
        self.bot.bus.xautoclaim = MagicMock(return_value=None)
        self.bot.bus.xpending_range = MagicMock(return_value=[
            {
                'message_id': '9-0',
                'consumer': 'dead_worker',
                'time_since_delivered': 60_000,
                'times_delivered': 1,
            }
        ])
        self.bot.bus.xclaim = MagicMock(return_value=[
            ('9-0', {'command': 'STOP', 'target': self.bot.worker_id}),
        ])
        self.bot.bus.xreadgroup = MagicMock(return_value=[])
        self.bot.bus.xack = MagicMock(return_value=1)

        self.bot._check_stream_commands()

        self.bot.bus.xclaim.assert_called_once()
        self.bot.handle_message.assert_called_once_with(
            {'command': 'STOP', 'target': self.bot.worker_id}
        )
        self.bot.bus.xack.assert_called_once_with('swarm:commands', 'workers', '9-0')

if __name__ == '__main__':
    unittest.main()
