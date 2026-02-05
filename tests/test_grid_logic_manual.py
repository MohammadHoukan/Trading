
import unittest
from unittest.mock import MagicMock
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

    # Helper to simulate fill
    def fill_order(self, order_id):
        if order_id in self.orders:
            self.orders[order_id]['status'] = 'closed'
            self.orders[order_id]['filled'] = self.orders[order_id]['amount']

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

if __name__ == '__main__':
    unittest.main()
