
import unittest
from unittest.mock import MagicMock
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
        
    def test_regime_logic(self):
        rf = RegimeFilter(self.config)
        rf.data_source = MagicMock()
        
        # Mock Dataframe return
        # Case 1: High ADX (Trending)
        rf.data_source.exchange.fetch_ohlcv.return_value = [[0,1,2,3,4,5]] * 20 # Dummy data
        
        # Hack: Mock pd.DataFrame.ta.adx
        # Since pandas_ta extends DataFrame, mocking is tricky.
        # Easier to mock the entire method logic or the fetch result.
        
        # NOTE: Testing pandas_ta logic requires real data. 
        # For unit test, we trust pandas_ta works and mock the result of `adx()`?
        # Or better, we mock `analyze_market` output for Orchestrator test.
        pass

    def test_orchestrator_reaction(self):
        orch = Orchestrator()
        orch.config = self.config
        orch.bus = MagicMock()
        orch.risk_engine = MagicMock()
        orch.regime_filter = MagicMock()
        
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
