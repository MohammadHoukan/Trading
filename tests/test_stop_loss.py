"""Unit tests for stop-loss functionality."""
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestStopLoss(unittest.TestCase):
    def setUp(self):
        self.strategy = {
            "SOL/USDT": {
                "enabled": True,
                "grid_levels": 4,
                "lower_limit": 10.0,
                "upper_limit": 30.0,
                "amount_per_grid": 1.0,
                "stop_loss": 15.0  # Stop loss at $15
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
    def test_stop_loss_triggers_at_threshold(self, mock_limiter, mock_om, mock_db, mock_bus, mock_config):
        """Stop-loss should trigger when price <= threshold."""
        mock_config.return_value = self.config
        mock_bus_instance = MagicMock()
        mock_bus.return_value = mock_bus_instance
        
        from workers.grid_bot import GridBot
        GridBot._load_strategy_params = MagicMock(return_value=self.strategy['SOL/USDT'])
        
        bot = GridBot("SOL/USDT", 4)
        bot.logger = MagicMock()
        
        # Price at exactly stop-loss
        result = bot._check_stop_loss(15.0)
        
        assert result is True
        assert bot.stop_loss_triggered is True
        assert bot.running is False
        bot.logger.critical.assert_called()

    @patch('workers.grid_bot.load_config')
    @patch('workers.grid_bot.RedisBus')
    @patch('workers.grid_bot.Database')
    @patch('workers.grid_bot.OrderManager')
    @patch('workers.grid_bot.RateLimiter')
    def test_stop_loss_triggers_below_threshold(self, mock_limiter, mock_om, mock_db, mock_bus, mock_config):
        """Stop-loss should trigger when price < threshold."""
        mock_config.return_value = self.config
        mock_bus_instance = MagicMock()
        mock_bus.return_value = mock_bus_instance
        
        from workers.grid_bot import GridBot
        GridBot._load_strategy_params = MagicMock(return_value=self.strategy['SOL/USDT'])
        
        bot = GridBot("SOL/USDT", 4)
        bot.logger = MagicMock()
        
        # Price below stop-loss
        result = bot._check_stop_loss(14.0)
        
        assert result is True
        assert bot.stop_loss_triggered is True

    @patch('workers.grid_bot.load_config')
    @patch('workers.grid_bot.RedisBus')
    @patch('workers.grid_bot.Database')
    @patch('workers.grid_bot.OrderManager')
    @patch('workers.grid_bot.RateLimiter')
    def test_stop_loss_does_not_trigger_above_threshold(self, mock_limiter, mock_om, mock_db, mock_bus, mock_config):
        """Stop-loss should NOT trigger when price > threshold."""
        mock_config.return_value = self.config
        mock_bus_instance = MagicMock()
        mock_bus.return_value = mock_bus_instance
        
        from workers.grid_bot import GridBot
        GridBot._load_strategy_params = MagicMock(return_value=self.strategy['SOL/USDT'])
        
        bot = GridBot("SOL/USDT", 4)
        bot.logger = MagicMock()
        
        # Price above stop-loss
        result = bot._check_stop_loss(20.0)
        
        assert result is False
        assert bot.stop_loss_triggered is False
        assert bot.running is True

    @patch('workers.grid_bot.load_config')
    @patch('workers.grid_bot.RedisBus')
    @patch('workers.grid_bot.Database')
    @patch('workers.grid_bot.OrderManager')
    @patch('workers.grid_bot.RateLimiter')
    def test_stop_loss_with_no_stop_loss_configured(self, mock_limiter, mock_om, mock_db, mock_bus, mock_config):
        """Should not trigger if no stop_loss defined in strategy."""
        mock_config.return_value = self.config
        mock_bus_instance = MagicMock()
        mock_bus.return_value = mock_bus_instance
        
        strategy_no_sl = {
            "enabled": True,
            "grid_levels": 4,
            "lower_limit": 10.0,
            "upper_limit": 30.0,
            "amount_per_grid": 1.0,
            # No stop_loss key
        }
        
        from workers.grid_bot import GridBot
        GridBot._load_strategy_params = MagicMock(return_value=strategy_no_sl)
        
        bot = GridBot("SOL/USDT", 4)
        bot.logger = MagicMock()
        
        # Even at very low price, should not trigger
        result = bot._check_stop_loss(1.0)
        
        assert result is False
        assert bot.running is True


if __name__ == '__main__':
    unittest.main()
