
import unittest
from unittest.mock import MagicMock
from backtest.metrics import calculate_metrics, Metrics

class TestMetrics(unittest.TestCase):
    def test_profit_factor_capped_at_100(self):
        """Test that profit_factor is capped at 100.0 instead of inf when no losses."""
        # 2 winning trades, 0 losing trades
        trades = [
            MagicMock(side='sell', pnl=10.0),
            MagicMock(side='sell', pnl=20.0),
        ]
        
        metrics = calculate_metrics(
            trades=trades,
            initial_capital=1000.0,
            final_capital=1030.0,
            equity_curve=MagicMock()
        )
        
        self.assertEqual(metrics.profit_factor, 100.0)
        self.assertNotEqual(metrics.profit_factor, float('inf'))

    def test_profit_factor_normal(self):
        """Test normal profit factor calculation."""
        # 100 profit, 50 loss => 2.0
        trades = [
            MagicMock(side='sell', pnl=100.0),
            MagicMock(side='sell', pnl=-50.0),
        ]
        metrics = calculate_metrics(trades, 1000, 1050, MagicMock())
        self.assertEqual(metrics.profit_factor, 2.0)

    def test_profit_factor_zero_profit(self):
        """Test break-even profit factor."""
        # 0 profit, 0 loss (or just losses)
        trades = [
            MagicMock(side='sell', pnl=-10.0),
        ]
        metrics = calculate_metrics(trades, 1000, 990, MagicMock())
        self.assertEqual(metrics.profit_factor, 0.0)

if __name__ == '__main__':
    unittest.main()
