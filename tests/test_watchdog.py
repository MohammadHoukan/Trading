"""Unit tests for watchdog (stale data detection)."""
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestWatchdog(unittest.TestCase):
    def setUp(self):
        self.strategy = {
            "SOL/USDT": {
                "enabled": True,
                "grid_levels": 4,
                "lower_limit": 10.0,
                "upper_limit": 30.0,
                "amount_per_grid": 1.0,
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

    @patch('workers.grid_bot.load_config')
    @patch('workers.grid_bot.RedisBus')
    @patch('workers.grid_bot.Database')
    @patch('workers.grid_bot.OrderManager')
    @patch('workers.grid_bot.RateLimiter')
    def test_stale_data_not_detected_when_fresh(self, mock_limiter, mock_om, mock_db, mock_bus, mock_config):
        """Should return False when data is fresh."""
        mock_config.return_value = self.config
        mock_bus_instance = MagicMock()
        mock_bus.return_value = mock_bus_instance
        
        from workers.grid_bot import GridBot
        GridBot._load_strategy_params = MagicMock(return_value=self.strategy['SOL/USDT'])
        
        bot = GridBot("SOL/USDT", 4)
        bot.logger = MagicMock()
        bot.last_price_update = time.time()  # Just updated
        
        result = bot._check_stale_data()
        
        assert result is False
        bot.logger.critical.assert_not_called()

    @patch('workers.grid_bot.load_config')
    @patch('workers.grid_bot.RedisBus')
    @patch('workers.grid_bot.Database')
    @patch('workers.grid_bot.OrderManager')
    @patch('workers.grid_bot.RateLimiter')
    def test_stale_data_detected_when_old(self, mock_limiter, mock_om, mock_db, mock_bus, mock_config):
        """Should return True when data is stale (>15s old)."""
        mock_config.return_value = self.config
        mock_bus_instance = MagicMock()
        mock_bus.return_value = mock_bus_instance
        
        from workers.grid_bot import GridBot
        GridBot._load_strategy_params = MagicMock(return_value=self.strategy['SOL/USDT'])
        
        bot = GridBot("SOL/USDT", 4)
        bot.logger = MagicMock()
        bot.last_price_update = time.time() - 20  # 20 seconds ago
        
        result = bot._check_stale_data()
        
        assert result is True
        bot.logger.critical.assert_called()


if __name__ == '__main__':
    unittest.main()
