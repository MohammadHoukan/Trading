import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import pandas as pd

# Path hack
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from manager.regime_filter import RegimeFilter
from manager.orchestrator import Orchestrator

class TestRegimeIntegration(unittest.TestCase):
    def setUp(self):
        self.config = {
            'exchange': {'name': 'binance', 'mode': 'testnet', 'api_key': 'k', 'secret': 's'},
            'regime': {'adx_threshold': 25.0},
            'redis': {'channels': {'command': 'cmd', 'status': 'stat'}},
            'swarm': {},
        }
    def _run_regime_with_signals(self, adx_value, atr_value, close_series):
        rf = RegimeFilter(self.config)
        rf.data_source = MagicMock()
        rf.data_source.exchange.fetch_ohlcv.return_value = [[0, 1, 2, 3, 4, 5]] * len(close_series)
        rf._get_fill_rate = MagicMock(return_value=None)
        real_df = pd.DataFrame
        real_series = pd.Series

        class FakeTA:
            def __init__(self, adx, atr):
                self._adx = adx
                self._atr = atr

            def adx(self, *args, **kwargs):
                return real_df({'ADX_14': [self._adx]})

            def atr(self, *args, **kwargs):
                return real_series([self._atr])

        class FakeDF:
            def __init__(self, adx, atr, closes):
                self.ta = FakeTA(adx, atr)
                self._close = real_series(closes)
                self._high = self._close + 1
                self._low = self._close - 1

            def __getitem__(self, key):
                if key == 'close':
                    return self._close
                if key == 'high':
                    return self._high
                if key == 'low':
                    return self._low
                raise KeyError(key)

        with patch(
            'manager.regime_filter.pd.DataFrame',
            return_value=FakeDF(adx_value, atr_value, close_series),
        ):
            return rf.analyze_market()

    def test_regime_logic_trending(self):
        # Strong trend + high volatility + far from mean => trending / reduce exposure.
        close_series = [100.0] * 99 + [150.0]
        analysis = self._run_regime_with_signals(80.0, 20.0, close_series)
        self.assertEqual(analysis['regime'], 'TRENDING')
        self.assertEqual(analysis['recommendation'], 'REDUCE_EXPOSURE')
        self.assertAlmostEqual(analysis['scale'], 0.5, places=6)

    def test_regime_logic_ranging(self):
        # Weak trend + moderate volatility + near mean => ranging / run.
        close_series = [100.0] * 100
        analysis = self._run_regime_with_signals(10.0, 3.0, close_series)
        self.assertEqual(analysis['regime'], 'RANGING')
        self.assertEqual(analysis['recommendation'], 'RUN')

    def test_orchestrator_reaction(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch.config = self.config
        orch.bus = MagicMock()
        orch.risk_engine = MagicMock()
        orch.regime_filter = MagicMock()
        orch.logger = MagicMock()
        orch.regime_by_symbol = {'SOL/USDT': 'RANGING'}

        # Test 1: Market becomes TRENDING -> targeted scale reduction for symbol workers.
        orch.bus.hgetall.return_value = {
            'w_sol': '{"worker_id":"w_sol","symbol":"SOL/USDT","status":"RUNNING"}'
        }
        orch.regime_filter.analyze_market.return_value = {
            'regime': 'TRENDING',
            'score': 20,
            'recommendation': 'REDUCE_EXPOSURE',
            'scale': 0.5,
        }
        orch.perform_regime_checks()

        orch.bus.publish.assert_called_with(
            'cmd', {'command': 'UPDATE_SCALE', 'target': 'w_sol', 'scale': 0.5}
        )
        self.assertEqual(orch.regime_by_symbol['SOL/USDT'], 'TRENDING')

        # Test 2: Market stays TRENDING -> no extra action.
        orch.bus.publish.reset_mock()
        orch.perform_regime_checks()
        orch.bus.publish.assert_not_called()

        # Test 3: Market becomes RANGING -> restore full exposure scale.
        orch.bus.hgetall.return_value = {
            'w_sol': '{"worker_id":"w_sol","symbol":"SOL/USDT","status":"PAUSED"}'
        }
        orch.regime_filter.analyze_market.return_value = {
            'regime': 'RANGING',
            'score': 75,
            'recommendation': 'RUN',
            'scale': 1.0,
        }
        orch.perform_regime_checks()

        orch.bus.publish.assert_called_with(
            'cmd', {'command': 'UPDATE_SCALE', 'target': 'w_sol', 'scale': 1.0}
        )
        self.assertEqual(orch.regime_by_symbol['SOL/USDT'], 'RANGING')

if __name__ == '__main__':
    unittest.main()
