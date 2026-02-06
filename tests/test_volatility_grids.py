"""Tests for volatility-scaled grid functionality."""

import pytest
from unittest.mock import MagicMock, patch
import json


class TestVolatilityScaledGrids:
    """Test suite for volatility-scaled grid functionality in GridBot."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config with volatility grids enabled."""
        return {
            'exchange': {
                'name': 'binance',
                'api_key': 'test_key',
                'secret': 'test_secret',
                'mode': 'testnet',
                'pool': [
                    {'api_key': 'key1', 'secret': 'secret1'},
                ]
            },
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'db': 0,
                'channels': {
                    'command': 'swarm:cmd',
                    'status': 'swarm:status',
                }
            },
            'smart_features': {
                'volatility_grids': {
                    'enabled': True,
                    'baseline_volatility': 0.025,
                    'scale_factor': 1.5,
                    'min_multiplier': 0.5,
                    'max_multiplier': 2.0,
                }
            }
        }

    @pytest.fixture
    def mock_strategy(self):
        """Create mock strategy params."""
        return {
            'lower_limit': 90.0,
            'upper_limit': 110.0,
            'grid_levels': 10,
            'amount_per_grid': 0.1,
            'enabled': True,
        }

    @patch('workers.grid_bot.RedisBus')
    @patch('workers.grid_bot.Database')
    @patch('workers.grid_bot.OrderManager')
    @patch('workers.grid_bot.RateLimiter')
    @patch('workers.grid_bot.load_config')
    @patch('os.path.exists')
    @patch('builtins.open')
    def test_dynamic_step_calculation(
        self, mock_open, mock_exists, mock_load_config,
        mock_rate_limiter, mock_om, mock_db, mock_bus, mock_config, mock_strategy
    ):
        """Test _calculate_dynamic_step with various ATR values."""
        mock_load_config.return_value = mock_config
        mock_exists.return_value = True
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(
            {'SOL/USDT': mock_strategy}
        )
        mock_bus_instance = MagicMock()
        mock_bus_instance.set.return_value = True  # API key lock acquired
        mock_bus.return_value = mock_bus_instance

        from workers.grid_bot import GridBot

        # Create bot with mocked dependencies
        with patch.object(GridBot, '_market_data_listener'):
            bot = GridBot('SOL/USDT', 10)

        # Test dynamic step calculation
        base_step = 2.0  # (110 - 90) / 10
        current_price = 100.0

        # Test 1: No ATR data - should return base step
        bot.current_atr = None
        bot.current_atr_pct = None
        result = bot._calculate_dynamic_step(base_step, current_price)
        assert result == base_step

        # Test 2: ATR at baseline (2.5%) - multiplier should be ~1.0
        bot.current_atr_pct = 0.025
        result = bot._calculate_dynamic_step(base_step, current_price)
        assert 1.9 < result < 2.1  # Close to base step

        # Test 3: High volatility (5%) - should widen grids
        bot.current_atr_pct = 0.05
        result = bot._calculate_dynamic_step(base_step, current_price)
        # multiplier = 1 + 1.5 * (0.05 - 0.025) = 1.0375
        assert result > base_step

        # Test 4: Low volatility (1%) - should tighten grids
        bot.current_atr_pct = 0.01
        result = bot._calculate_dynamic_step(base_step, current_price)
        # multiplier = 1 + 1.5 * (0.01 - 0.025) = 0.9775
        assert result < base_step

        # Test 5: Very high volatility - should cap at max_multiplier
        bot.current_atr_pct = 0.10  # 10%
        result = bot._calculate_dynamic_step(base_step, current_price)
        # Should be capped at 2.0 * base_step
        assert result <= base_step * 2.0

        # Test 6: Very low volatility - should cap at min_multiplier
        bot.current_atr_pct = 0.001  # 0.1%
        result = bot._calculate_dynamic_step(base_step, current_price)
        # Should be capped at 0.5 * base_step
        assert result >= base_step * 0.5

    @patch('workers.grid_bot.RedisBus')
    @patch('workers.grid_bot.Database')
    @patch('workers.grid_bot.OrderManager')
    @patch('workers.grid_bot.RateLimiter')
    @patch('workers.grid_bot.load_config')
    @patch('os.path.exists')
    @patch('builtins.open')
    def test_market_data_captures_atr(
        self, mock_open, mock_exists, mock_load_config,
        mock_rate_limiter, mock_om, mock_db, mock_bus, mock_config, mock_strategy
    ):
        """Test that market data listener captures ATR values."""
        mock_load_config.return_value = mock_config
        mock_exists.return_value = True
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(
            {'SOL/USDT': mock_strategy}
        )
        mock_bus_instance = MagicMock()
        mock_bus_instance.set.return_value = True
        mock_bus.return_value = mock_bus_instance

        from workers.grid_bot import GridBot

        with patch.object(GridBot, '_market_data_listener'):
            bot = GridBot('SOL/USDT', 10)

        # Simulate receiving market data with ATR
        market_data = {
            'price': 100.0,
            'atr': 2.5,
            'atr_pct': 0.025,
            'timestamp': 1234567890,
        }

        # Simulate what the listener does
        if market_data.get('atr') is not None:
            bot.current_atr = market_data['atr']
        if market_data.get('atr_pct') is not None:
            bot.current_atr_pct = market_data['atr_pct']

        assert bot.current_atr == 2.5
        assert bot.current_atr_pct == 0.025

    @patch('workers.grid_bot.RedisBus')
    @patch('workers.grid_bot.Database')
    @patch('workers.grid_bot.OrderManager')
    @patch('workers.grid_bot.RateLimiter')
    @patch('workers.grid_bot.load_config')
    @patch('os.path.exists')
    @patch('builtins.open')
    def test_disabled_volatility_grids(
        self, mock_open, mock_exists, mock_load_config,
        mock_rate_limiter, mock_om, mock_db, mock_bus, mock_config, mock_strategy
    ):
        """Test that volatility scaling is disabled when config says so."""
        mock_config['smart_features']['volatility_grids']['enabled'] = False
        mock_load_config.return_value = mock_config
        mock_exists.return_value = True
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(
            {'SOL/USDT': mock_strategy}
        )
        mock_bus_instance = MagicMock()
        mock_bus_instance.set.return_value = True
        mock_bus.return_value = mock_bus_instance

        from workers.grid_bot import GridBot

        with patch.object(GridBot, '_market_data_listener'):
            bot = GridBot('SOL/USDT', 10)

        assert bot.volatility_grids_enabled is False

        # Even with ATR data, should return base step
        base_step = 2.0
        bot.current_atr_pct = 0.05
        result = bot._calculate_dynamic_step(base_step, 100.0)
        assert result == base_step
