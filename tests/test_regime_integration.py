
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
        # Mock dependencies in RegimeFilter
        # We need to mock OrderManager inside it
        
    def _run_regime_with_adx(self, adx_value):
        rf = RegimeFilter(self.config)
        rf.data_source = MagicMock()
        rf.data_source.exchange.fetch_ohlcv.return_value = [[0, 1, 2, 3, 4, 5]] * 20
        real_df = pd.DataFrame

        class FakeTA:
            def __init__(self, value):
                self._value = value

            def adx(self, *args, **kwargs):
                return real_df({'ADX_14': [self._value]})

        class FakeDF:
            def __init__(self, value):
                self.ta = FakeTA(value)

            def __getitem__(self, key):
                return [1, 2, 3]

        with patch('manager.regime_filter.pd.DataFrame', return_value=FakeDF(adx_value)):
            return rf.analyze_market()

    def test_regime_logic_trending(self):
        regime = self._run_regime_with_adx(35.0)
        self.assertEqual(regime, 'TRENDING')

    def test_regime_logic_ranging(self):
        regime = self._run_regime_with_adx(10.0)
        self.assertEqual(regime, 'RANGING')

    def test_orchestrator_reaction(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch.config = self.config
        orch.bus = MagicMock()
        orch.risk_engine = MagicMock()
        orch.regime_filter = MagicMock()
        orch.logger = MagicMock()
        
        # Test 1: Market becomes TRENDING -> PAUSE
        orch.regime_filter.analyze_market.return_value = 'TRENDING'
        orch.last_regime = 'RANGING'
        
        orch.perform_regime_checks()
        
        orch.bus.publish.assert_called_with('cmd', {'command': 'PAUSE', 'target': 'all'})
        self.assertEqual(orch.last_regime, 'TRENDING')
        
        # Test 2: Market stays TRENDING -> No Action
        orch.bus.publish.reset_mock()
        orch.perform_regime_checks()
        orch.bus.publish.assert_not_called()
        
        # Test 3: Market becomes RANGING -> RESUME
        orch.regime_filter.analyze_market.return_value = 'RANGING'
        orch.perform_regime_checks()
        
        orch.bus.publish.assert_called_with('cmd', {'command': 'RESUME', 'target': 'all'})

if __name__ == '__main__':
    unittest.main()
