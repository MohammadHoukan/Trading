"""Tests for the Mean Reversion Filter."""

import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np


class TestMeanReversionFilter:
    """Test suite for MeanReversionFilter."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for testing."""
        return {
            'smart_features': {
                'mean_reversion': {
                    'enabled': True,
                    'sma_period': 20,
                    'std_period': 50,
                    'entry_z_threshold': 1.5,
                    'exit_z_threshold': 0.5,
                }
            },
            'regime': {
                'timeframe': '1h',
            },
            'exchange': {
                'name': 'binance',
                'api_key': 'test_key',
                'secret': 'test_secret',
                'mode': 'testnet',
            }
        }

    @pytest.fixture
    def sample_candles(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 100
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)

        return [
            [i * 3600000, p - 0.5, p + 0.5, p - 1, p, 1000]
            for i, p in enumerate(prices)
        ]

    @patch('manager.mean_reversion_filter.OrderManager')
    def test_init(self, mock_om_class, mock_config):
        """Test filter initialization."""
        from manager.mean_reversion_filter import MeanReversionFilter

        mrf = MeanReversionFilter(mock_config)

        assert mrf.enabled is True
        assert mrf.sma_period == 20
        assert mrf.std_period == 50
        assert mrf.entry_z_threshold == 1.5
        assert mrf.exit_z_threshold == 0.5

    @patch('manager.mean_reversion_filter.OrderManager')
    def test_analyze_extended_above(self, mock_om_class, mock_config, sample_candles):
        """Test detection of price extended above mean."""
        from manager.mean_reversion_filter import MeanReversionFilter

        # Create candles where price is significantly above SMA
        candles = sample_candles.copy()
        # Raise the last few prices significantly
        for i in range(-5, 0):
            candles[i][4] = candles[i][4] + 20  # Raise close price

        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = candles
        mock_om_class.return_value.exchange = mock_exchange

        mrf = MeanReversionFilter(mock_config)
        result = mrf.analyze('SOL/USDT')

        assert result['z_score'] > 0
        # With high z-score, should suggest entry
        if abs(result['z_score']) > 1.5:
            assert result['action'] == 'ENTER'

    @patch('manager.mean_reversion_filter.OrderManager')
    def test_analyze_near_mean(self, mock_om_class, mock_config, sample_candles):
        """Test detection of price near mean."""
        from manager.mean_reversion_filter import MeanReversionFilter

        # Use standard candles - price should be near mean
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = sample_candles
        mock_om_class.return_value.exchange = mock_exchange

        mrf = MeanReversionFilter(mock_config)
        result = mrf.analyze('SOL/USDT')

        assert 'z_score' in result
        assert 'price' in result
        assert 'sma' in result
        # With random walk, z-score likely near 0
        assert -3 < result['z_score'] < 3

    @patch('manager.mean_reversion_filter.OrderManager')
    def test_disabled_filter(self, mock_om_class, mock_config):
        """Test filter when disabled."""
        from manager.mean_reversion_filter import MeanReversionFilter

        mock_config['smart_features']['mean_reversion']['enabled'] = False
        mrf = MeanReversionFilter(mock_config)

        result = mrf.analyze('SOL/USDT')
        assert result['action'] == 'HOLD'
        assert 'disabled' in result['recommendation'].lower()

    @patch('manager.mean_reversion_filter.OrderManager')
    def test_should_enter(self, mock_om_class, mock_config, sample_candles):
        """Test should_enter helper method."""
        from manager.mean_reversion_filter import MeanReversionFilter

        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = sample_candles
        mock_om_class.return_value.exchange = mock_exchange

        mrf = MeanReversionFilter(mock_config)

        # With typical data, should return bool
        result = mrf.should_enter('SOL/USDT')
        assert isinstance(result, bool)

    @patch('manager.mean_reversion_filter.OrderManager')
    def test_get_z_score(self, mock_om_class, mock_config, sample_candles):
        """Test get_z_score helper method."""
        from manager.mean_reversion_filter import MeanReversionFilter

        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = sample_candles
        mock_om_class.return_value.exchange = mock_exchange

        mrf = MeanReversionFilter(mock_config)

        z = mrf.get_z_score('SOL/USDT')
        assert z is None or isinstance(z, float)

    @patch('manager.mean_reversion_filter.OrderManager')
    def test_insufficient_data(self, mock_om_class, mock_config):
        """Test handling of insufficient data."""
        from manager.mean_reversion_filter import MeanReversionFilter

        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = []  # No data
        mock_om_class.return_value.exchange = mock_exchange

        mrf = MeanReversionFilter(mock_config)
        result = mrf.analyze('SOL/USDT')

        assert result['action'] == 'HOLD'
        assert 'insufficient' in result['recommendation'].lower()
